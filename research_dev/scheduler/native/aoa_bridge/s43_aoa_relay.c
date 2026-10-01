// s43-aoa-relay: phone side of the Pixel AOA bridge (WS10).
//
// Reads S43A frames (s43a_protocol.h) from /dev/usb_accessory and relays every stream to the unchanged
// FFN worker on 127.0.0.1:<worker port>; worker bytes go back as DATA frames. The worker binary, its
// protocol and its numerics are untouched: to the worker the relay is one more TCP client per stream,
// exactly like adbd behind `adb forward`.
//
//   s43-aoa-relay --worker-port P [--accessory /dev/usb_accessory | --accessory-fd N]
//                 [--cpus HEXMASK] [--uclamp-min 0..1024] [--fifo-priority 1..99]
//                 [--qos-latency-us N --qos-window-ms W] [--echo-server PORT] [--open-timeout-s S]
//
// Keep-awake (all opt-in, all released by process exit, nothing persistent):
//   --qos-latency-us N   hold a PM QoS CPU-latency request of N us (/dev/cpu_dma_latency) while stream
//                        traffic (OPEN/DATA) arrives; released W ms after the last such frame (--qos-window-ms,
//                        0 = whole lifetime). PING/NOP/HELLO never arm it, so an idle helper does not hold it.
//   --cpus / --uclamp-min / --fifo-priority  placement, utilisation floor and RT priority of the relay.
// --echo-server PORT starts a phone-local TCP echo server (transport benchmarks through the same relay).
// --accessory-fd N / --accessory-unix PATH use an inherited descriptor or a connected UNIX SOCK_SEQPACKET
// socket instead of the accessory node (host loopback tests; each message stands for one USB transfer).
//
// stderr: one READY line ("[s43-aoa-relay] ready ...") once the accessory is open, one
// "S43AOARELAYSTATS {json}" line at exit. Exit codes: 0 clean (SIGTERM/SIGINT/SIGHUP or SHUTDOWN frame),
// 2 usage, 3 accessory open failed or lost, 4 internal error.
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/eventfd.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/un.h>

#include "s43a_protocol.h"

#define RELAY_VERSION 1
#define MAX_STREAMS 8
#define READ_CHUNK (256 * 1024)
#define RX_CAPACITY (2 * (S43A_HEADER_BYTES + S43A_MAX_PAYLOAD + 1) + READ_CHUNK)
#define SOCKET_CHUNK (256 * 1024)

struct stream {
    int used;
    int closing;     // closed by the bridge (or reset): the socket thread reaps it without a CLOSE frame
    int echo;        // relay-local echo stream, no socket
    int fd;
    uint32_t id;
    uint32_t session;  // frames of this stream carry the session it was opened in (stale ones are dropped)
    uint64_t last_rx_ns;
};

struct counters {
    uint64_t frames_in, frames_out, bytes_in, bytes_out, data_in, data_out;
    uint64_t streams_opened, streams_closed, connect_failures, stale_frames, dropped_data;
    uint64_t resyncs, discarded_bytes, pings, nops, hellos, qos_acquires, qos_releases;
};

static const char * accessory_path = "/dev/usb_accessory";
static int accessory_fd_arg = -1;
static const char * accessory_unix_path = NULL;
static int worker_port = 0;
static int echo_port = 0;
static int open_timeout_s = 30;
static long qos_latency_us = -1;
static const char * qos_device = "/dev/cpu_dma_latency";
static long qos_window_ms = 1500;
static unsigned long cpu_mask = 0;
static long uclamp_min = -1;
static int fifo_priority = 0;

static int acc_fd = -1;
static int event_fd = -1;
static volatile sig_atomic_t stop_requested = 0;
static int shutdown_frame = 0;
static const char * exit_reason = "running";

static pthread_mutex_t streams_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t write_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t qos_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t stats_lock = PTHREAD_MUTEX_INITIALIZER;
static struct stream streams[MAX_STREAMS];
static struct counters stats;
static uint32_t session_id = 0;
static int synced = 0;
static int qos_fd = -1;
static uint64_t qos_expiry_ns = 0;

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
}

static void count(uint64_t * field, uint64_t value) {
    pthread_mutex_lock(&stats_lock);
    *field += value;
    pthread_mutex_unlock(&stats_lock);
}

static void on_signal(int number) {
    (void) number;
    stop_requested = 1;
}

static void notify_socket_thread(void) {
    const uint64_t one = 1;
    ssize_t ignored = write(event_fd, &one, sizeof(one));
    (void) ignored;
}

static int write_all(int fd, const uint8_t * data, size_t size) {
    while (size > 0) {
        const ssize_t count_written = write(fd, data, size);
        if (count_written < 0 && errno == EINTR) {
            continue;
        }
        if (count_written <= 0) {
            return -1;
        }
        data += count_written;
        size -= (size_t) count_written;
    }
    return 0;
}

// Frames are written whole under one lock so the two writer threads never interleave on the link.
// `buffer` holds S43A_HEADER_BYTES of headroom before the payload and one spare byte after it.
static int send_frame_in_place(uint8_t * buffer, uint16_t type, uint32_t session, uint32_t stream,
                               uint32_t length, uint64_t aux) {
    struct s43a_header header;
    header.magic = S43A_MAGIC;
    header.version = S43A_VERSION;
    header.type = type;
    header.session = session;
    header.stream = stream;
    header.length = length;
    header.pad = s43a_pad(length);
    header.aux = aux;
    if (header.pad) {
        buffer[S43A_HEADER_BYTES + length] = 0;
    }
    pthread_mutex_lock(&write_lock);
    header.stamp_ns = now_ns();
    memcpy(buffer, &header, sizeof(header));
    const size_t total = S43A_HEADER_BYTES + (size_t) length + header.pad;
    const int result = write_all(acc_fd, buffer, total);
    pthread_mutex_unlock(&write_lock);
    if (result == 0) {
        pthread_mutex_lock(&stats_lock);
        stats.frames_out += 1;
        stats.bytes_out += total;
        if (type == S43A_DATA) {
            stats.data_out += length;
        }
        pthread_mutex_unlock(&stats_lock);
    }
    return result;
}

static int send_small(uint16_t type, uint32_t session, uint32_t stream, const void * payload, uint32_t length,
                      uint64_t aux) {
    uint8_t buffer[S43A_HEADER_BYTES + 256 + 1];
    if (length > 256) {
        return -1;
    }
    if (length) {
        memcpy(buffer + S43A_HEADER_BYTES, payload, length);
    }
    return send_frame_in_place(buffer, type, session, stream, length, aux);
}

static void send_close(uint32_t session, uint32_t stream, uint32_t reason) {
    send_small(S43A_CLOSE, session, stream, &reason, sizeof(reason), 0);
}

static void send_error(uint32_t code, const char * text) {
    uint8_t payload[128];
    const size_t text_bytes = strnlen(text, sizeof(payload) - sizeof(code));
    memcpy(payload, &code, sizeof(code));
    memcpy(payload + sizeof(code), text, text_bytes);
    send_small(S43A_ERROR, session_id, 0, payload, (uint32_t) (sizeof(code) + text_bytes), 0);
}

// --- keep-awake --------------------------------------------------------------------------------------

static void qos_touch(uint64_t now) {
    if (qos_latency_us < 0) {
        return;
    }
    pthread_mutex_lock(&qos_lock);
    if (qos_fd < 0) {
        qos_fd = open(qos_device, O_WRONLY | O_CLOEXEC);
        if (qos_fd >= 0) {
            const int32_t value = (int32_t) qos_latency_us;
            if (write(qos_fd, &value, sizeof(value)) != (ssize_t) sizeof(value)) {
                close(qos_fd);
                qos_fd = -1;
            } else {
                count(&stats.qos_acquires, 1);
                notify_socket_thread();
            }
        }
    }
    qos_expiry_ns = qos_window_ms > 0 ? now + (uint64_t) qos_window_ms * 1000000ull : UINT64_MAX;
    pthread_mutex_unlock(&qos_lock);
}

// Returns the poll timeout (ms) until the QoS window ends, -1 when nothing is held.
static int qos_expire(uint64_t now) {
    int timeout = -1;
    pthread_mutex_lock(&qos_lock);
    if (qos_fd >= 0 && qos_expiry_ns != UINT64_MAX) {
        if (now >= qos_expiry_ns) {
            close(qos_fd);  // closing the file drops the request
            qos_fd = -1;
            count(&stats.qos_releases, 1);
        } else {
            timeout = (int) ((qos_expiry_ns - now) / 1000000ull) + 1;
        }
    }
    pthread_mutex_unlock(&qos_lock);
    return timeout;
}

static int apply_uclamp(long value) {
    struct {
        uint32_t size, sched_policy;
        uint64_t sched_flags;
        int32_t sched_nice;
        uint32_t sched_priority;
        uint64_t sched_runtime, sched_deadline, sched_period;
        uint32_t sched_util_min, sched_util_max;
    } attr;
    memset(&attr, 0, sizeof(attr));
    attr.size = sizeof(attr);
    // SCHED_FLAG_KEEP_POLICY | SCHED_FLAG_KEEP_PARAMS | SCHED_FLAG_UTIL_CLAMP_MIN
    attr.sched_flags = 0x08 | 0x10 | 0x20;
    attr.sched_util_min = (uint32_t) value;
    return (int) syscall(SYS_sched_setattr, 0, &attr, 0);
}

// --- streams -----------------------------------------------------------------------------------------

static struct stream * find_stream(uint32_t id) {
    for (int i = 0; i < MAX_STREAMS; ++i) {
        if (streams[i].used && !streams[i].closing && streams[i].id == id) {
            return &streams[i];
        }
    }
    return NULL;
}

// Caller holds streams_lock. The socket thread closes the descriptor after it sees the shutdown.
static void retire_stream(struct stream * stream) {
    if (stream->echo) {
        memset(stream, 0, sizeof(*stream));
        stream->fd = -1;
        count(&stats.streams_closed, 1);
        return;
    }
    stream->closing = 1;
    shutdown(stream->fd, SHUT_RDWR);
}

static void retire_all_streams(void) {
    pthread_mutex_lock(&streams_lock);
    for (int i = 0; i < MAX_STREAMS; ++i) {
        if (streams[i].used && !streams[i].closing) {
            retire_stream(&streams[i]);
        }
    }
    pthread_mutex_unlock(&streams_lock);
    notify_socket_thread();
}

static int connect_worker(void) {
    const int fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) {
        return -1;
    }
    struct sockaddr_in address;
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons((uint16_t) worker_port);
    if (connect(fd, (struct sockaddr *) &address, sizeof(address)) != 0) {
        const int saved = errno;
        close(fd);
        errno = saved;
        return -1;
    }
    const int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    return fd;
}

static void open_stream(uint32_t id, uint32_t flags, uint64_t rx_ns) {
    pthread_mutex_lock(&streams_lock);
    struct stream * existing = find_stream(id);
    if (existing) {
        retire_stream(existing);  // a reused id restarts the stream
    }
    struct stream * slot = NULL;
    for (int i = 0; i < MAX_STREAMS && !slot; ++i) {
        if (!streams[i].used) {
            slot = &streams[i];
        }
    }
    pthread_mutex_unlock(&streams_lock);
    if (!slot) {
        send_close(session_id, id, S43A_CLOSE_TOO_MANY_STREAMS);
        return;
    }
    int fd = -1;
    const int echo = (flags & S43A_OPEN_LOCAL_ECHO) != 0;
    if (!echo) {
        fd = connect_worker();
        if (fd < 0) {
            const int32_t status = errno;
            count(&stats.connect_failures, 1);
            send_small(S43A_OPEN_ACK, session_id, id, &status, sizeof(status), rx_ns);
            send_close(session_id, id, S43A_CLOSE_CONNECT_FAILED);
            return;
        }
    }
    pthread_mutex_lock(&streams_lock);
    slot->used = 1;
    slot->closing = 0;
    slot->echo = echo;
    slot->fd = fd;
    slot->id = id;
    slot->session = session_id;
    slot->last_rx_ns = rx_ns;
    pthread_mutex_unlock(&streams_lock);
    count(&stats.streams_opened, 1);
    notify_socket_thread();
    const int32_t status = 0;
    send_small(S43A_OPEN_ACK, session_id, id, &status, sizeof(status), rx_ns);
}

// --- inbound frames (USB reader thread) --------------------------------------------------------------

static uint8_t echo_buffer[S43A_HEADER_BYTES + S43A_MAX_PAYLOAD + 1];

static int header_valid(const struct s43a_header * header) {
    return header->magic == S43A_MAGIC && header->version == S43A_VERSION && header->type >= S43A_HELLO &&
           header->type <= S43A_ERROR && header->length <= S43A_MAX_PAYLOAD &&
           header->pad == s43a_pad(header->length);
}

static void handle_frame(const struct s43a_header * header, const uint8_t * payload, uint64_t rx_ns) {
    pthread_mutex_lock(&stats_lock);
    stats.frames_in += 1;
    stats.bytes_in += S43A_HEADER_BYTES + header->length + header->pad;
    pthread_mutex_unlock(&stats_lock);
    if (header->type == S43A_DATA || header->type == S43A_OPEN) {
        qos_touch(rx_ns);  // only real traffic arms the window: liveness PINGs and keep-alive NOPs never do
    }
    switch (header->type) {
        case S43A_HELLO: {
            retire_all_streams();
            session_id = header->session;
            synced = 1;
            count(&stats.hellos, 1);
            struct s43a_hello_ack ack;
            ack.relay_version = RELAY_VERSION;
            ack.relay_pid = (uint32_t) getpid();
            ack.max_streams = MAX_STREAMS;
            ack.options = (qos_latency_us >= 0 ? 1u : 0u) | (echo_port > 0 ? 2u : 0u);
            send_small(S43A_HELLO_ACK, session_id, 0, &ack, sizeof(ack), rx_ns);
            return;
        }
        case S43A_OPEN: {
            uint32_t flags = 0;
            if (header->length >= sizeof(flags)) {
                memcpy(&flags, payload, sizeof(flags));
            }
            open_stream(header->stream, flags, rx_ns);
            return;
        }
        case S43A_DATA: {
            pthread_mutex_lock(&streams_lock);
            struct stream * stream = find_stream(header->stream);
            int fd = -1;
            int echo = 0;
            if (stream) {
                stream->last_rx_ns = rx_ns;
                echo = stream->echo;
                // a private duplicate: the socket thread may close and reuse the stream's descriptor
                // number while this thread writes
                fd = echo ? -1 : fcntl(stream->fd, F_DUPFD_CLOEXEC, 0);
            }
            pthread_mutex_unlock(&streams_lock);
            count(&stats.data_in, header->length);
            if (!stream) {
                count(&stats.dropped_data, header->length);
                return;
            }
            if (echo) {
                memcpy(echo_buffer + S43A_HEADER_BYTES, payload, header->length);
                send_frame_in_place(echo_buffer, S43A_DATA, session_id, header->stream, header->length, rx_ns);
                return;
            }
            const int failed = fd < 0 || write_all(fd, payload, header->length) != 0;
            if (fd >= 0) {
                close(fd);
            }
            if (failed) {
                pthread_mutex_lock(&streams_lock);
                stream = find_stream(header->stream);
                if (stream) {
                    retire_stream(stream);
                }
                pthread_mutex_unlock(&streams_lock);
                notify_socket_thread();
                send_close(session_id, header->stream, S43A_CLOSE_IO);
            }
            return;
        }
        case S43A_CLOSE: {
            pthread_mutex_lock(&streams_lock);
            struct stream * stream = find_stream(header->stream);
            if (stream) {
                retire_stream(stream);
            }
            pthread_mutex_unlock(&streams_lock);
            notify_socket_thread();
            return;
        }
        case S43A_NOP:
            count(&stats.nops, 1);
            return;
        case S43A_PING: {
            struct s43a_ping ping;
            memset(&ping, 0, sizeof(ping));
            memcpy(&ping, payload, header->length < sizeof(ping) ? header->length : sizeof(ping));
            uint32_t reply = ping.reply_bytes;
            if (reply < sizeof(struct s43a_pong)) {
                reply = sizeof(struct s43a_pong);
            }
            if (reply > S43A_MAX_PAYLOAD) {
                reply = S43A_MAX_PAYLOAD;
            }
            count(&stats.pings, 1);
            memset(echo_buffer + S43A_HEADER_BYTES, 0x5a, reply);
            struct s43a_pong pong;
            pong.id = ping.id;
            pong.relay_rx_ns = rx_ns;
            pong.relay_tx_ns = now_ns();
            memcpy(echo_buffer + S43A_HEADER_BYTES, &pong, sizeof(pong));
            send_frame_in_place(echo_buffer, S43A_PONG, session_id, 0, reply, ping.id);
            return;
        }
        case S43A_SHUTDOWN:
            retire_all_streams();
            shutdown_frame = 1;
            stop_requested = 1;
            return;
        default:
            send_error(S43A_ERROR_UNEXPECTED_TYPE, "unexpected frame type");
            return;
    }
}

static uint8_t rx_buffer[RX_CAPACITY];

// Consumes whole frames from rx_buffer[0, *length); keeps a trailing partial frame.
static void process_rx(size_t * length, uint64_t rx_ns) {
    size_t offset = 0;
    while (*length - offset >= S43A_HEADER_BYTES && !stop_requested) {
        struct s43a_header header;
        memcpy(&header, rx_buffer + offset, sizeof(header));
        const int valid = header_valid(&header);
        if (!synced && !(valid && header.type == S43A_HELLO)) {
            offset += 1;  // resynchronise on the next HELLO
            count(&stats.discarded_bytes, 1);
            continue;
        }
        if (!valid) {
            send_error(S43A_ERROR_PROTOCOL, "invalid frame header; resynchronising");
            count(&stats.resyncs, 1);
            synced = 0;
            offset += 1;
            count(&stats.discarded_bytes, 1);
            continue;
        }
        const size_t total = S43A_HEADER_BYTES + (size_t) header.length + header.pad;
        if (*length - offset < total) {
            break;
        }
        if (header.type != S43A_HELLO && header.session != session_id) {
            count(&stats.stale_frames, 1);  // a previous bridge's frame
        } else {
            handle_frame(&header, rx_buffer + offset + S43A_HEADER_BYTES, rx_ns);
        }
        offset += total;
    }
    if (offset) {
        memmove(rx_buffer, rx_buffer + offset, *length - offset);
        *length -= offset;
    }
}

// --- outbound (socket thread) ------------------------------------------------------------------------

static uint8_t socket_buffer[S43A_HEADER_BYTES + SOCKET_CHUNK + 1];

static void * socket_thread(void * unused) {
    (void) unused;
    for (;;) {
        struct pollfd items[MAX_STREAMS + 1];
        int indexes[MAX_STREAMS + 1];
        int count_items = 0;
        items[count_items].fd = event_fd;
        items[count_items].events = POLLIN;
        indexes[count_items++] = -1;
        pthread_mutex_lock(&streams_lock);
        for (int i = 0; i < MAX_STREAMS; ++i) {
            if (streams[i].used && !streams[i].echo) {
                items[count_items].fd = streams[i].fd;
                items[count_items].events = POLLIN;
                indexes[count_items++] = i;
            }
        }
        pthread_mutex_unlock(&streams_lock);
        const int timeout = qos_expire(now_ns());
        const int ready = poll(items, (nfds_t) count_items, timeout);
        if (stop_requested) {
            break;
        }
        if (ready < 0) {
            if (errno == EINTR) {
                continue;
            }
            exit_reason = "poll failed";
            break;
        }
        if (items[0].revents & POLLIN) {
            uint64_t value;
            ssize_t ignored = read(event_fd, &value, sizeof(value));
            (void) ignored;
        }
        for (int k = 1; k < count_items; ++k) {
            if (!items[k].revents) {
                continue;
            }
            struct stream * stream = &streams[indexes[k]];
            const ssize_t got = read(items[k].fd, socket_buffer + S43A_HEADER_BYTES, SOCKET_CHUNK);
            if (got > 0) {
                pthread_mutex_lock(&streams_lock);
                const int closing = stream->closing;
                const uint32_t id = stream->id;
                const uint32_t session = stream->session;
                const uint64_t aux = stream->last_rx_ns;
                pthread_mutex_unlock(&streams_lock);
                if (!closing) {
                    send_frame_in_place(socket_buffer, S43A_DATA, session, id, (uint32_t) got, aux);
                }
                continue;
            }
            if (got < 0 && (errno == EINTR || errno == EAGAIN)) {
                continue;
            }
            // EOF or error: the worker closed (send CLOSE) or the bridge closed (already retired)
            pthread_mutex_lock(&streams_lock);
            const int closing = stream->closing;
            const uint32_t id = stream->id;
            const uint32_t session = stream->session;
            close(stream->fd);
            memset(stream, 0, sizeof(*stream));
            stream->fd = -1;
            pthread_mutex_unlock(&streams_lock);
            count(&stats.streams_closed, 1);
            if (!closing) {
                send_close(session, id, got == 0 ? S43A_CLOSE_EOF : S43A_CLOSE_IO);
            }
        }
    }
    return NULL;
}

// --- echo server (benchmarks) ------------------------------------------------------------------------

static void * echo_thread(void * unused) {
    (void) unused;
    const int listener = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    const int one = 1;
    struct sockaddr_in address;
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons((uint16_t) echo_port);
    if (listener < 0 || setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) != 0 ||
        bind(listener, (struct sockaddr *) &address, sizeof(address)) != 0 || listen(listener, 4) != 0) {
        fprintf(stderr, "[s43-aoa-relay] echo server setup failed: %s\n", strerror(errno));
        return NULL;
    }
    static uint8_t buffer[SOCKET_CHUNK];
    while (!stop_requested) {
        const int client = accept(listener, NULL, NULL);
        if (client < 0) {
            continue;
        }
        setsockopt(client, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        for (;;) {
            const ssize_t got = read(client, buffer, sizeof(buffer));
            if (got < 0 && errno == EINTR) {
                continue;
            }
            if (got <= 0 || write_all(client, buffer, (size_t) got) != 0) {
                break;
            }
        }
        close(client);
    }
    close(listener);
    return NULL;
}

// --- setup -------------------------------------------------------------------------------------------

static void usage(void) {
    fprintf(stderr,
            "usage: s43-aoa-relay --worker-port P [--accessory PATH | --accessory-fd N | --accessory-unix PATH]\n"
            "       [--cpus HEX] [--uclamp-min 0..1024] [--fifo-priority 1..99] [--qos-latency-us N] [--qos-window-ms W]\n"
            "       [--echo-server PORT] [--open-timeout-s S]\n");
}

static int parse_long(const char * text, long low, long high, long * out) {
    char * end = NULL;
    errno = 0;
    const long value = strtol(text, &end, 10);
    if (errno || end == text || *end || value < low || value > high) {
        return -1;
    }
    *out = value;
    return 0;
}

static int parse_arguments(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) {
        const char * name = argv[i];
        if (i + 1 >= argc) {
            return -1;
        }
        const char * value = argv[++i];
        long number = 0;
        if (strcmp(name, "--accessory") == 0) {
            accessory_path = value;
        } else if (strcmp(name, "--accessory-unix") == 0) {
            accessory_unix_path = value;
        } else if (strcmp(name, "--accessory-fd") == 0) {
            if (parse_long(value, 0, 65535, &number)) return -1;
            accessory_fd_arg = (int) number;
        } else if (strcmp(name, "--worker-port") == 0) {
            if (parse_long(value, 1, 65535, &number)) return -1;
            worker_port = (int) number;
        } else if (strcmp(name, "--echo-server") == 0) {
            if (parse_long(value, 1, 65535, &number)) return -1;
            echo_port = (int) number;
        } else if (strcmp(name, "--open-timeout-s") == 0) {
            if (parse_long(value, 0, 3600, &number)) return -1;
            open_timeout_s = (int) number;
        } else if (strcmp(name, "--qos-latency-us") == 0) {
            if (parse_long(value, 0, 1000000, &qos_latency_us)) return -1;
        } else if (strcmp(name, "--qos-device") == 0) {
            qos_device = value;  // tests: a writable stand-in for /dev/cpu_dma_latency
        } else if (strcmp(name, "--qos-window-ms") == 0) {
            if (parse_long(value, 0, 3600000, &qos_window_ms)) return -1;
        } else if (strcmp(name, "--uclamp-min") == 0) {
            if (parse_long(value, 0, 1024, &uclamp_min)) return -1;
        } else if (strcmp(name, "--fifo-priority") == 0) {
            if (parse_long(value, 1, 99, &number)) return -1;
            fifo_priority = (int) number;
        } else if (strcmp(name, "--cpus") == 0) {
            char * end = NULL;
            errno = 0;
            cpu_mask = strtoul(value, &end, 16);
            if (errno || end == value || *end || cpu_mask == 0 || cpu_mask > 0xffffffffUL) return -1;
        } else {
            return -1;
        }
    }
    return worker_port > 0 ? 0 : -1;
}

static int connect_unix(const char * path) {
    struct sockaddr_un address;
    memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    if (strlen(path) >= sizeof(address.sun_path)) {
        errno = ENAMETOOLONG;
        return -1;
    }
    strcpy(address.sun_path, path);
    const int fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    if (fd < 0) {
        return -1;
    }
    if (connect(fd, (struct sockaddr *) &address, sizeof(address)) != 0) {
        const int saved = errno;
        close(fd);
        errno = saved;
        return -1;
    }
    return fd;
}

static int open_accessory(void) {
    if (accessory_fd_arg >= 0) {
        return accessory_fd_arg;
    }
    const uint64_t deadline = now_ns() + (uint64_t) open_timeout_s * 1000000000ull;
    for (;;) {
        const int fd = accessory_unix_path ? connect_unix(accessory_unix_path) : open(accessory_path, O_RDWR | O_CLOEXEC);
        if (fd >= 0) {
            return fd;
        }
        if (errno == EBUSY) {
            fprintf(stderr, "[s43-aoa-relay] %s is busy (another opener)\n", accessory_path);
            return -1;
        }
        if (stop_requested || now_ns() >= deadline) {
            fprintf(stderr, "[s43-aoa-relay] cannot open %s: %s\n", accessory_path, strerror(errno));
            return -1;
        }
        usleep(100000);
    }
}

static void print_stats(int code) {
    struct rusage usage;
    memset(&usage, 0, sizeof(usage));
    getrusage(RUSAGE_SELF, &usage);  // phone CPU cost of the relay itself (keep-alive wake-ups included)
    const double cpu_s = (double) usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1e6 +
                         (double) usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1e6;
    pthread_mutex_lock(&stats_lock);
    fprintf(stderr,
            "S43AOARELAYSTATS {\"exit_code\":%d,\"exit_reason\":\"%s\",\"frames_in\":%llu,\"frames_out\":%llu,"
            "\"bytes_in\":%llu,\"bytes_out\":%llu,\"data_in\":%llu,\"data_out\":%llu,\"streams_opened\":%llu,"
            "\"streams_closed\":%llu,\"connect_failures\":%llu,\"stale_frames\":%llu,\"dropped_data\":%llu,"
            "\"resyncs\":%llu,\"discarded_bytes\":%llu,\"pings\":%llu,\"nops\":%llu,\"hellos\":%llu,"
            "\"qos_acquires\":%llu,\"qos_releases\":%llu,\"cpu_s\":%.6f,\"voluntary_switches\":%ld,"
            "\"involuntary_switches\":%ld}\n",
            code, exit_reason, (unsigned long long) stats.frames_in, (unsigned long long) stats.frames_out,
            (unsigned long long) stats.bytes_in, (unsigned long long) stats.bytes_out,
            (unsigned long long) stats.data_in, (unsigned long long) stats.data_out,
            (unsigned long long) stats.streams_opened, (unsigned long long) stats.streams_closed,
            (unsigned long long) stats.connect_failures, (unsigned long long) stats.stale_frames,
            (unsigned long long) stats.dropped_data, (unsigned long long) stats.resyncs,
            (unsigned long long) stats.discarded_bytes, (unsigned long long) stats.pings,
            (unsigned long long) stats.nops, (unsigned long long) stats.hellos,
            (unsigned long long) stats.qos_acquires, (unsigned long long) stats.qos_releases, cpu_s,
            (long) usage.ru_nvcsw, (long) usage.ru_nivcsw);
    pthread_mutex_unlock(&stats_lock);
    fflush(stderr);
}

int main(int argc, char ** argv) {
    if (parse_arguments(argc, argv) != 0) {
        usage();
        return 2;
    }
    for (int i = 0; i < MAX_STREAMS; ++i) {
        streams[i].fd = -1;
    }
    signal(SIGPIPE, SIG_IGN);
    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_handler = on_signal;  // no SA_RESTART: a blocking accessory read returns EINTR
    sigaction(SIGTERM, &action, NULL);
    sigaction(SIGINT, &action, NULL);
    sigaction(SIGHUP, &action, NULL);

    if (cpu_mask) {
        cpu_set_t cores;
        CPU_ZERO(&cores);
        for (int i = 0; i < 32; ++i) {
            if (cpu_mask & (1ul << i)) {
                CPU_SET(i, &cores);
            }
        }
        if (sched_setaffinity(0, sizeof(cores), &cores) != 0) {
            fprintf(stderr, "[s43-aoa-relay] affinity failed: %s\n", strerror(errno));
            return 4;
        }
    }
    if (uclamp_min >= 0 && apply_uclamp(uclamp_min) != 0) {
        fprintf(stderr, "[s43-aoa-relay] uclamp failed: %s\n", strerror(errno));
        return 4;
    }
    event_fd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    if (event_fd < 0) {
        return 4;
    }
    acc_fd = open_accessory();
    if (acc_fd < 0) {
        exit_reason = "accessory open failed";
        print_stats(3);
        return 3;
    }
    if (qos_latency_us >= 0 && qos_window_ms == 0) {
        qos_touch(now_ns());  // whole-lifetime request
    }

    // Only the USB reader (this thread) takes the stop signals.
    sigset_t blocked, previous;
    sigemptyset(&blocked);
    sigaddset(&blocked, SIGTERM);
    sigaddset(&blocked, SIGINT);
    sigaddset(&blocked, SIGHUP);
    pthread_sigmask(SIG_BLOCK, &blocked, &previous);
    pthread_t socket_worker, echo_worker;
    if (pthread_create(&socket_worker, NULL, socket_thread, NULL) != 0) {
        return 4;
    }
    if (echo_port > 0 && pthread_create(&echo_worker, NULL, echo_thread, NULL) == 0) {
        pthread_detach(echo_worker);
    }
    pthread_sigmask(SIG_SETMASK, &previous, NULL);
    if (fifo_priority > 0) {
        struct sched_param parameter;
        memset(&parameter, 0, sizeof(parameter));
        parameter.sched_priority = fifo_priority;
        if (sched_setscheduler(0, SCHED_FIFO, &parameter) != 0) {
            fprintf(stderr, "[s43-aoa-relay] SCHED_FIFO failed: %s\n", strerror(errno));
        }
    }
    fprintf(stderr,
            "[s43-aoa-relay] ready version=%d pid=%d worker_port=%d qos_latency_us=%ld qos_window_ms=%ld "
            "cpus=%lx uclamp_min=%ld fifo_priority=%d echo_port=%d\n",
            RELAY_VERSION, (int) getpid(), worker_port, qos_latency_us, qos_window_ms, cpu_mask, uclamp_min,
            fifo_priority, echo_port);
    fflush(stderr);

    int code = 0;
    size_t length = 0;
    while (!stop_requested) {
        if (RX_CAPACITY - length < READ_CHUNK) {
            // a frame larger than the buffer cannot occur (header_valid bounds it); drop and resync
            length = 0;
            synced = 0;
            count(&stats.resyncs, 1);
        }
        const ssize_t got = read(acc_fd, rx_buffer + length, READ_CHUNK);
        if (got < 0) {
            if (errno == EINTR) {
                continue;
            }
            exit_reason = "accessory read failed";
            code = 3;
            break;
        }
        if (got == 0) {
            if (accessory_fd_arg >= 0 || accessory_unix_path) {  // loopback test peer closed
                exit_reason = "accessory closed";
                code = 3;
                break;
            }
            continue;  // f_accessory never returns 0; tolerate it anyway
        }
        length += (size_t) got;
        process_rx(&length, now_ns());
    }
    if (code == 0) {
        exit_reason = shutdown_frame ? "shutdown frame" : "signal";
    }
    stop_requested = 1;
    retire_all_streams();
    notify_socket_thread();
    pthread_join(socket_worker, NULL);
    pthread_mutex_lock(&streams_lock);
    for (int i = 0; i < MAX_STREAMS; ++i) {
        if (streams[i].used && streams[i].fd >= 0) {
            close(streams[i].fd);
        }
    }
    pthread_mutex_unlock(&streams_lock);
    pthread_mutex_lock(&qos_lock);
    if (qos_fd >= 0) {
        close(qos_fd);
        qos_fd = -1;
        stats.qos_releases += 1;
    }
    pthread_mutex_unlock(&qos_lock);
    print_stats(code);
    return code;
}
