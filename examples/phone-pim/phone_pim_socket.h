#pragma once

#include "phone_pim_protocol.h"

#include <cstdint>
#include <string>

namespace phone_pim {

class Fd {
public:
    Fd() = default;
    explicit Fd(int value) : value_(value) {}
    ~Fd();

    Fd(const Fd &) = delete;
    Fd & operator=(const Fd &) = delete;
    Fd(Fd && other) noexcept;
    Fd & operator=(Fd && other) noexcept;

    int get() const { return value_; }
    bool valid() const { return value_ >= 0; }
    int release();
    void reset(int value = -1);

private:
    int value_ = -1;
};

enum class ReceiveResult {
    ok,
    eof,
    error,
};

Fd connect_tcp(const std::string & host, uint16_t port, int timeout_ms, std::string & error);
Fd listen_tcp(const std::string & bind_host, uint16_t port, int backlog, std::string & error);
Fd accept_tcp(int listener_fd, int timeout_ms, std::string & error);

bool set_socket_deadlines(int fd, int timeout_ms, std::string & error);
bool send_bytes(int fd, const void * data, size_t size, std::string & error);
ReceiveResult receive_bytes(int fd, void * data, size_t size, std::string & error);

bool send_frame(int fd, Header header, const std::vector<uint8_t> & payload, std::string & error);
ReceiveResult receive_frame(int fd, Frame & frame, uint64_t max_payload, std::string & error);

// Measurement-only frame profiling (opt-in, default OFF; no wire/behavior change).
// Values are for the MOST RECENT frame only; the caller attributes them by opcode so
// non-stage traffic (HELLO/PREPARE/EXECUTE/RELEASE/SHUTDOWN) and idle oracle gaps are
// excluded. The outer-frame SHA-256 covers the whole frame payload (opcode envelope +
// data); it is a strictly larger domain than the data-only manifest/chunk/prefix hash.
struct RecvProfile {
    uint64_t header_us = 0;   // socket recv of the 104-byte header (includes idle wait)
    uint64_t payload_us = 0;  // socket recv of the payload bytes
    uint64_t sha_us = 0;      // outer-frame SHA-256 over the whole payload (envelope + data)
    uint64_t payload_bytes = 0;
};
struct SendProfile {
    uint64_t sha_us = 0;      // outer-frame SHA-256 over the whole payload (envelope + data)
    uint64_t write_us = 0;    // socket send of header + payload
    uint64_t payload_bytes = 0;
};
void set_receive_profiling(bool on);
void set_send_profiling(bool on);
void get_last_recv_profile(RecvProfile & out); // last receive_frame call
void get_last_send_profile(SendProfile & out); // last send_frame call

} // namespace phone_pim
