#define _GNU_SOURCE

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define TRANSFER_BYTES (64U * 1024U)

struct bridge_context {
    int accessory;
    int socket;
    atomic_int stopping;
    atomic_int failure;
    uint64_t usb_to_tcp_bytes;
    uint64_t tcp_to_usb_bytes;
    uint64_t usb_reads;
    uint64_t usb_writes;
};

static uint64_t now_ns(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (uint64_t) value.tv_sec * UINT64_C(1000000000) +
        (uint64_t) value.tv_nsec;
}

static int parse_port(const char * text) {
    errno = 0;
    char * end = NULL;
    const long value = strtol(text, &end, 10);
    return errno == 0 && end != text && *end == '\0' &&
        value >= 1 && value <= 65535 ? (int) value : -1;
}

static int open_accessory(void) {
    for (unsigned int attempt = 0; attempt < 600; ++attempt) {
        const int descriptor = open("/dev/usb_accessory", O_RDWR);
        if (descriptor >= 0) {
            return descriptor;
        }
        usleep(100000);
    }
    return -1;
}

static int connect_loopback(int port) {
    for (unsigned int attempt = 0; attempt < 600; ++attempt) {
        const int descriptor = socket(AF_INET, SOCK_STREAM, 0);
        if (descriptor < 0) {
            return -1;
        }
        struct sockaddr_in address;
        memset(&address, 0, sizeof(address));
        address.sin_family = AF_INET;
        address.sin_port = htons((uint16_t) port);
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (connect(descriptor, (struct sockaddr *) &address,
                sizeof(address)) == 0) {
            const int no_delay = 1;
            setsockopt(descriptor, IPPROTO_TCP, TCP_NODELAY,
                    &no_delay, sizeof(no_delay));
            return descriptor;
        }
        close(descriptor);
        usleep(100000);
    }
    return -1;
}

static int send_all(int descriptor, const unsigned char * data, size_t size) {
    while (size > 0) {
        const ssize_t count = send(descriptor, data, size, MSG_NOSIGNAL);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            return -1;
        }
        data += count;
        size -= (size_t) count;
    }
    return 0;
}

static int write_all(int descriptor, const unsigned char * data, size_t size) {
    while (size > 0) {
        const ssize_t count = write(descriptor, data, size);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            return -1;
        }
        data += count;
        size -= (size_t) count;
    }
    return 0;
}

static void stop_bridge(struct bridge_context * context, int failure) {
    if (failure != 0) {
        int expected = 0;
        atomic_compare_exchange_strong(&context->failure, &expected, failure);
    }
    atomic_store(&context->stopping, 1);
    shutdown(context->socket, SHUT_RDWR);
}

static void * usb_to_tcp(void * opaque) {
    struct bridge_context * context = opaque;
    unsigned char * buffer = malloc(TRANSFER_BYTES);
    if (buffer == NULL) {
        stop_bridge(context, 1);
        return NULL;
    }
    while (!atomic_load(&context->stopping)) {
        const ssize_t count = read(context->accessory, buffer, TRANSFER_BYTES);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            stop_bridge(context, context->usb_to_tcp_bytes == 0 ? 1 : 0);
            break;
        }
        context->usb_to_tcp_bytes += (uint64_t) count;
        ++context->usb_reads;
        if (send_all(context->socket, buffer, (size_t) count) != 0) {
            if (!atomic_load(&context->stopping)) {
                fprintf(stderr, "phone TCP send failed: %s\n", strerror(errno));
                stop_bridge(context, 1);
            }
            break;
        }
    }
    free(buffer);
    return NULL;
}

static void * tcp_to_usb(void * opaque) {
    struct bridge_context * context = opaque;
    unsigned char * buffer = malloc(TRANSFER_BYTES);
    if (buffer == NULL) {
        stop_bridge(context, 1);
        return NULL;
    }
    while (!atomic_load(&context->stopping)) {
        const ssize_t count = recv(context->socket, buffer, TRANSFER_BYTES, 0);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count == 0) {
            stop_bridge(context, 0);
            break;
        }
        if (count < 0) {
            if (!atomic_load(&context->stopping)) {
                fprintf(stderr, "phone TCP receive failed: %s\n", strerror(errno));
                stop_bridge(context, 1);
            }
            break;
        }
        if (write_all(context->accessory, buffer, (size_t) count) != 0) {
            if (!atomic_load(&context->stopping)) {
                fprintf(stderr, "phone AOA write failed: %s\n", strerror(errno));
                stop_bridge(context, 1);
            }
            break;
        }
        context->tcp_to_usb_bytes += (uint64_t) count;
        ++context->usb_writes;
    }
    free(buffer);
    return NULL;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s PHONE_STAGE_PORT\n", argv[0]);
        return 2;
    }
    const int port = parse_port(argv[1]);
    if (port < 0) {
        fprintf(stderr, "invalid phone stage port\n");
        return 2;
    }
    signal(SIGPIPE, SIG_IGN);
    fprintf(stderr, "[aoa-phone-bridge] configured stage_port=%d\n", port);
    fflush(stderr);
    const int accessory = open_accessory();
    if (accessory < 0) {
        fprintf(stderr, "accessory open timed out: %s\n", strerror(errno));
        return 1;
    }
    fprintf(stderr, "[aoa-phone-bridge] endpoint open\n");
    fflush(stderr);
    const int stage_socket = connect_loopback(port);
    if (stage_socket < 0) {
        fprintf(stderr, "phone stage connect failed: %s\n", strerror(errno));
        close(accessory);
        return 1;
    }
    fprintf(stderr, "[aoa-phone-bridge] stage connected\n");
    fflush(stderr);

    struct bridge_context context = {
        .accessory = accessory,
        .socket = stage_socket,
    };
    atomic_init(&context.stopping, 0);
    atomic_init(&context.failure, 0);
    const uint64_t started = now_ns();
    pthread_t from_usb;
    pthread_t to_usb;
    if (pthread_create(&from_usb, NULL, usb_to_tcp, &context) != 0) {
        fprintf(stderr, "phone bridge thread creation failed\n");
        close(stage_socket);
        close(accessory);
        return 1;
    }
    if (pthread_create(&to_usb, NULL, tcp_to_usb, &context) != 0) {
        fprintf(stderr, "phone bridge thread creation failed\n");
        atomic_store(&context.stopping, 1);
        close(accessory);
        pthread_join(from_usb, NULL);
        close(stage_socket);
        return 1;
    }
    pthread_join(from_usb, NULL);
    atomic_store(&context.stopping, 1);
    shutdown(stage_socket, SHUT_RDWR);
    pthread_join(to_usb, NULL);
    const uint64_t elapsed = now_ns() - started;
    fprintf(stderr,
            "AOABRIDGE {\"role\":\"phone\",\"status\":%d,"
            "\"usb_to_tcp_bytes\":%llu,\"tcp_to_usb_bytes\":%llu,"
            "\"usb_reads\":%llu,\"usb_writes\":%llu,"
            "\"elapsed_ms\":%.3f}\n",
            atomic_load(&context.failure),
            (unsigned long long) context.usb_to_tcp_bytes,
            (unsigned long long) context.tcp_to_usb_bytes,
            (unsigned long long) context.usb_reads,
            (unsigned long long) context.usb_writes,
            elapsed / 1e6);
    fflush(stderr);
    close(stage_socket);
    close(accessory);
    return atomic_load(&context.failure) == 0 ? 0 : 1;
}
