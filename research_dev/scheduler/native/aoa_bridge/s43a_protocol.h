// S43A: framing between the host AOA bridge (adapters/aoa_bridge.py) and the phone relay
// (s43_aoa_relay.c) over the Android Open Accessory bulk endpoints.
//
// The link carries whole frames: a 40-byte little-endian header, `length` payload bytes, then `pad`
// zero bytes. pad is 1 exactly when (40 + length) is a multiple of 512, otherwise 0, so no frame ever
// ends on a USB max-packet boundary: every bulk transfer ends with a short packet and the phone's
// 16 KiB f_accessory read request completes at the frame end without relying on a zero-length
// packet. The Python mirror is adapters/aoa_bridge.py; keep both in step (tests check the layout).
//
// A bridge opens a session with HELLO(session = fresh nonce); the relay drops every byte until it
// sees a valid HELLO (resynchronisation after a stale or partial frame), closes every stream of the
// previous session and answers HELLO_ACK. Frames of another session are dropped as stale. Each TCP
// connection the bridge accepts is one stream: OPEN -> DATA* -> CLOSE in both directions.
#ifndef S43A_PROTOCOL_H
#define S43A_PROTOCOL_H

#include <stdint.h>

#define S43A_MAGIC UINT32_C(0x41333453)  // "S43A" little endian
#define S43A_VERSION 1
#define S43A_HEADER_BYTES 40
#define S43A_MAX_PAYLOAD (1u << 20)
#define S43A_PAD_QUANTUM 512

enum s43a_type {
    S43A_HELLO = 1,      // bridge -> relay: payload s43a_hello
    S43A_HELLO_ACK = 2,  // relay -> bridge: payload s43a_hello_ack; aux = relay receive ns of the HELLO
    S43A_OPEN = 3,       // bridge -> relay: payload s43a_open
    S43A_OPEN_ACK = 4,   // relay -> bridge: payload int32 status (0 or errno)
    S43A_DATA = 5,       // both: stream bytes; relay -> bridge aux = receive ns of the stream's last inbound frame
    S43A_CLOSE = 6,      // both: payload uint32 reason
    S43A_NOP = 7,        // bridge -> relay keep-alive, ignored
    S43A_PING = 8,       // bridge -> relay: payload s43a_ping + filler
    S43A_PONG = 9,       // relay -> bridge: payload s43a_pong + filler up to reply_bytes
    S43A_SHUTDOWN = 10,  // bridge -> relay: close every stream and exit 0
    S43A_ERROR = 11,     // relay -> bridge: payload uint32 code + text
};

enum s43a_close_reason {
    S43A_CLOSE_EOF = 0,
    S43A_CLOSE_CONNECT_FAILED = 1,
    S43A_CLOSE_IO = 2,
    S43A_CLOSE_PROTOCOL = 3,
    S43A_CLOSE_SESSION_RESET = 4,
    S43A_CLOSE_SHUTDOWN = 5,
    S43A_CLOSE_TOO_MANY_STREAMS = 6,
};

enum s43a_error_code {
    S43A_ERROR_PROTOCOL = 1,
    S43A_ERROR_UNEXPECTED_TYPE = 2,
};

#define S43A_OPEN_LOCAL_ECHO UINT32_C(1)  // the relay echoes DATA itself (no phone TCP hop): hop benchmark

struct s43a_header {
    uint32_t magic;
    uint16_t version;
    uint16_t type;
    uint32_t session;
    uint32_t stream;
    uint32_t length;
    uint32_t pad;
    uint64_t stamp_ns;  // sender CLOCK_MONOTONIC when the frame was written
    uint64_t aux;
};

struct s43a_hello {
    uint32_t bridge_pid;
    uint32_t flags;
};

struct s43a_hello_ack {
    uint32_t relay_version;
    uint32_t relay_pid;
    uint32_t max_streams;
    uint32_t options;  // bit 0: CPU-latency QoS configured, bit 1: echo server running
};

struct s43a_open {
    uint32_t flags;
};

struct s43a_ping {
    uint64_t id;
    uint32_t reply_bytes;  // total PONG payload size, >= sizeof(s43a_pong)
    uint32_t reserved;
};

struct s43a_pong {
    uint64_t id;
    uint64_t relay_rx_ns;
    uint64_t relay_tx_ns;
};

#ifdef __cplusplus
static_assert(sizeof(s43a_header) == S43A_HEADER_BYTES, "S43A header size");
#else
_Static_assert(sizeof(struct s43a_header) == S43A_HEADER_BYTES, "S43A header size");
_Static_assert(sizeof(struct s43a_hello_ack) == 16, "S43A HELLO_ACK size");
_Static_assert(sizeof(struct s43a_ping) == 16, "S43A PING size");
_Static_assert(sizeof(struct s43a_pong) == 24, "S43A PONG size");
#endif

static inline uint32_t s43a_pad(uint32_t length) {
    return ((S43A_HEADER_BYTES + length) % S43A_PAD_QUANTUM) == 0 ? 1u : 0u;
}

#endif
