#include "phone_pim_socket.h"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>

namespace phone_pim {
namespace {

using Clock = std::chrono::steady_clock;

int socket_timeout_ms(int fd, int option) {
    timeval value = {};
    socklen_t size = sizeof(value);
    if (getsockopt(fd, SOL_SOCKET, option, &value, &size) != 0 ||
        (value.tv_sec == 0 && value.tv_usec == 0)) {
        return 10000;
    }
    const int64_t millis = static_cast<int64_t>(value.tv_sec) * 1000 +
                           (value.tv_usec + 999) / 1000;
    return millis > INT32_MAX ? INT32_MAX : static_cast<int>(millis);
}

bool wait_until(int fd, short events, Clock::time_point deadline, std::string & error) {
    for (;;) {
        const auto now = Clock::now();
        if (now >= deadline) {
            error = "frame deadline exceeded";
            return false;
        }
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
        pollfd descriptor = { fd, events, 0 };
        int rc = poll(&descriptor, 1, static_cast<int>(std::max<int64_t>(remaining, 1)));
        if (rc < 0 && errno == EINTR) {
            continue;
        }
        if (rc <= 0) {
            error = rc == 0 ? "frame deadline exceeded" : std::string("poll failed: ") + std::strerror(errno);
            return false;
        }
        if ((descriptor.revents & (POLLERR | POLLNVAL)) != 0) {
            error = "socket poll reported an error";
            return false;
        }
        if ((descriptor.revents & events) != 0 || (descriptor.revents & POLLHUP) != 0) {
            return true;
        }
    }
}

bool send_until(
        int fd,
        const void * data,
        size_t size,
        Clock::time_point deadline,
        std::string & error) {
    const uint8_t * cursor = static_cast<const uint8_t *>(data);
    size_t sent = 0;
    while (sent < size) {
        if (!wait_until(fd, POLLOUT, deadline, error)) {
            return false;
        }
        const ssize_t count = send(fd, cursor + sent, size - sent, MSG_NOSIGNAL | MSG_DONTWAIT);
        if (count < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) {
            continue;
        }
        if (count <= 0) {
            error = std::string("send failed: ") + std::strerror(errno);
            return false;
        }
        sent += static_cast<size_t>(count);
    }
    return true;
}

ReceiveResult receive_until(
        int fd,
        void * data,
        size_t size,
        Clock::time_point deadline,
        std::string & error) {
    uint8_t * cursor = static_cast<uint8_t *>(data);
    size_t received = 0;
    while (received < size) {
        if (!wait_until(fd, POLLIN, deadline, error)) {
            return ReceiveResult::error;
        }
        const ssize_t count = recv(fd, cursor + received, size - received, MSG_DONTWAIT);
        if (count < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) {
            continue;
        }
        if (count == 0) {
            return received == 0 ? ReceiveResult::eof : ReceiveResult::error;
        }
        if (count < 0) {
            error = std::string("receive failed: ") + std::strerror(errno);
            return ReceiveResult::error;
        }
        received += static_cast<size_t>(count);
    }
    return ReceiveResult::ok;
}

} // namespace

Fd::~Fd() {
    reset();
}

Fd::Fd(Fd && other) noexcept : value_(other.release()) {}

Fd & Fd::operator=(Fd && other) noexcept {
    if (this != &other) {
        reset(other.release());
    }
    return *this;
}

int Fd::release() {
    const int value = value_;
    value_ = -1;
    return value;
}

void Fd::reset(int value) {
    if (value_ >= 0) {
        close(value_);
    }
    value_ = value;
}

bool set_socket_deadlines(int fd, int timeout_ms, std::string & error) {
    if (timeout_ms <= 0) {
        error = "timeout must be positive";
        return false;
    }
    timeval tv = {};
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) != 0 ||
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) != 0) {
        error = std::string("setsockopt timeout: ") + std::strerror(errno);
        return false;
    }
    const int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    return true;
}

Fd connect_tcp(const std::string & host, uint16_t port, int timeout_ms, std::string & error) {
    addrinfo hints = {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo * result = nullptr;
    const std::string service = std::to_string(port);
    const int gai = getaddrinfo(host.c_str(), service.c_str(), &hints, &result);
    if (gai != 0) {
        error = std::string("getaddrinfo: ") + gai_strerror(gai);
        return {};
    }

    Fd connected;
    for (addrinfo * ai = result; ai != nullptr && !connected.valid(); ai = ai->ai_next) {
        Fd candidate(socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol));
        if (!candidate.valid()) {
            continue;
        }
        const int old_flags = fcntl(candidate.get(), F_GETFL, 0);
        if (old_flags < 0 || fcntl(candidate.get(), F_SETFL, old_flags | O_NONBLOCK) != 0) {
            continue;
        }
        int rc = connect(candidate.get(), ai->ai_addr, ai->ai_addrlen);
        if (rc != 0 && errno != EINPROGRESS) {
            continue;
        }
        if (rc != 0) {
            pollfd pfd = { candidate.get(), POLLOUT, 0 };
            do {
                rc = poll(&pfd, 1, timeout_ms);
            } while (rc < 0 && errno == EINTR);
            if (rc <= 0) {
                continue;
            }
            int socket_error = 0;
            socklen_t length = sizeof(socket_error);
            if (getsockopt(candidate.get(), SOL_SOCKET, SO_ERROR, &socket_error, &length) != 0 || socket_error != 0) {
                continue;
            }
        }
        if (fcntl(candidate.get(), F_SETFL, old_flags) != 0) {
            continue;
        }
        if (!set_socket_deadlines(candidate.get(), timeout_ms, error)) {
            continue;
        }
        connected = std::move(candidate);
    }
    freeaddrinfo(result);
    if (!connected.valid() && error.empty()) {
        error = "connection failed or timed out";
    }
    return connected;
}

Fd listen_tcp(const std::string & bind_host, uint16_t port, int backlog, std::string & error) {
    addrinfo hints = {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    addrinfo * result = nullptr;
    const std::string service = std::to_string(port);
    const char * node = bind_host.empty() ? nullptr : bind_host.c_str();
    const int gai = getaddrinfo(node, service.c_str(), &hints, &result);
    if (gai != 0) {
        error = std::string("getaddrinfo: ") + gai_strerror(gai);
        return {};
    }

    Fd listener;
    for (addrinfo * ai = result; ai != nullptr && !listener.valid(); ai = ai->ai_next) {
        Fd candidate(socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol));
        if (!candidate.valid()) {
            continue;
        }
        const int one = 1;
        setsockopt(candidate.get(), SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        if (bind(candidate.get(), ai->ai_addr, ai->ai_addrlen) != 0 || listen(candidate.get(), backlog) != 0) {
            continue;
        }
        listener = std::move(candidate);
    }
    freeaddrinfo(result);
    if (!listener.valid()) {
        error = std::string("listen failed: ") + std::strerror(errno);
    }
    return listener;
}

Fd accept_tcp(int listener_fd, int timeout_ms, std::string & error) {
    pollfd pfd = { listener_fd, POLLIN, 0 };
    int rc = 0;
    do {
        rc = poll(&pfd, 1, timeout_ms);
    } while (rc < 0 && errno == EINTR);
    if (rc <= 0) {
        error = rc == 0 ? "accept timed out" : std::string("accept poll failed: ") + std::strerror(errno);
        return {};
    }
    Fd client(accept(listener_fd, nullptr, nullptr));
    if (!client.valid()) {
        error = std::string("accept failed: ") + std::strerror(errno);
        return {};
    }
    if (!set_socket_deadlines(client.get(), timeout_ms, error)) {
        return {};
    }
    return client;
}

bool send_bytes(int fd, const void * data, size_t size, std::string & error) {
    const uint8_t * cursor = static_cast<const uint8_t *>(data);
    size_t sent = 0;
    while (sent < size) {
        const ssize_t count = send(fd, cursor + sent, size - sent, MSG_NOSIGNAL);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            error = std::string("send failed: ") + std::strerror(errno);
            return false;
        }
        sent += static_cast<size_t>(count);
    }
    return true;
}

ReceiveResult receive_bytes(int fd, void * data, size_t size, std::string & error) {
    uint8_t * cursor = static_cast<uint8_t *>(data);
    size_t received = 0;
    while (received < size) {
        const ssize_t count = recv(fd, cursor + received, size - received, 0);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count == 0) {
            return received == 0 ? ReceiveResult::eof : ReceiveResult::error;
        }
        if (count < 0) {
            error = std::string("receive failed: ") + std::strerror(errno);
            return ReceiveResult::error;
        }
        received += static_cast<size_t>(count);
    }
    return ReceiveResult::ok;
}

// Measurement-only frame profiling (opt-in, default OFF). Records only the MOST RECENT
// frame's timing split; the caller attributes it by opcode. No wire/behavior change.
namespace {
bool g_recv_profile = false;
bool g_send_profile = false;
RecvProfile g_last_recv;
SendProfile g_last_send;
uint64_t prof_now_us() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
            Clock::now().time_since_epoch()).count());
}
} // namespace

void set_receive_profiling(bool on) { g_recv_profile = on; }
void set_send_profiling(bool on) { g_send_profile = on; }
void get_last_recv_profile(RecvProfile & out) { out = g_last_recv; }
void get_last_send_profile(SendProfile & out) { out = g_last_send; }

bool send_frame(int fd, Header header, const std::vector<uint8_t> & payload, std::string & error) {
    header.payload_bytes = payload.size();
    const uint64_t t_sha = g_send_profile ? prof_now_us() : 0;
    header.payload_sha256 = sha256(payload.data(), payload.size());
    if (g_send_profile) {
        g_last_send.sha_us = prof_now_us() - t_sha;
        g_last_send.payload_bytes = payload.size();
    }
    const auto encoded = encode_header(header);
    const auto deadline = Clock::now() + std::chrono::milliseconds(socket_timeout_ms(fd, SO_SNDTIMEO));
    const uint64_t t_write = g_send_profile ? prof_now_us() : 0;
    const bool ok = send_until(fd, encoded.data(), encoded.size(), deadline, error) &&
                    send_until(fd, payload.data(), payload.size(), deadline, error);
    if (g_send_profile) g_last_send.write_us = prof_now_us() - t_write;
    return ok;
}

ReceiveResult receive_frame(int fd, Frame & frame, uint64_t max_payload, std::string & error) {
    const auto deadline = Clock::now() + std::chrono::milliseconds(socket_timeout_ms(fd, SO_RCVTIMEO));
    frame.payload.clear();
    if (g_recv_profile) g_last_recv = RecvProfile{};
    std::array<uint8_t, k_header_bytes> encoded = {};
    const uint64_t t_hdr = g_recv_profile ? prof_now_us() : 0;
    const ReceiveResult header_result = receive_until(fd, encoded.data(), encoded.size(), deadline, error);
    if (g_recv_profile) g_last_recv.header_us = prof_now_us() - t_hdr;
    if (header_result != ReceiveResult::ok) {
        if (header_result == ReceiveResult::error && error.empty()) {
            error = "truncated frame header";
        }
        return header_result;
    }
    Header header;
    if (!decode_header(encoded.data(), encoded.size(), header, error)) {
        return ReceiveResult::error;
    }
    const uint64_t opcode_limit = opcode_payload_limit(header.opcode);
    if (opcode_limit == 0 || header.payload_bytes > max_payload ||
        header.payload_bytes > opcode_limit || header.payload_bytes > static_cast<uint64_t>(SIZE_MAX)) {
        error = "frame payload exceeds configured bound";
        return ReceiveResult::error;
    }
    frame.payload.resize(static_cast<size_t>(header.payload_bytes));
    const uint64_t t_pay = g_recv_profile ? prof_now_us() : 0;
    const ReceiveResult payload_result = receive_until(
            fd, frame.payload.data(), frame.payload.size(), deadline, error);
    if (g_recv_profile) g_last_recv.payload_us = prof_now_us() - t_pay;
    if (payload_result != ReceiveResult::ok) {
        if (error.empty()) {
            error = "truncated frame payload";
        }
        return ReceiveResult::error;
    }
    const uint64_t t_sha = g_recv_profile ? prof_now_us() : 0;
    const bool sha_ok = sha256(frame.payload.data(), frame.payload.size()) == header.payload_sha256;
    if (g_recv_profile) {
        g_last_recv.sha_us = prof_now_us() - t_sha;
        g_last_recv.payload_bytes = header.payload_bytes;
    }
    if (!sha_ok) {
        error = "payload SHA-256 mismatch";
        return ReceiveResult::error;
    }
    frame.header = header;
    return ReceiveResult::ok;
}

} // namespace phone_pim
