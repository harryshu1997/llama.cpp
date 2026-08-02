#ifndef S41_BACKEND_DUMMY_PROTOCOL_H
#define S41_BACKEND_DUMMY_PROTOCOL_H

#include <cstdint>

static constexpr uint32_t S41_DUMMY_REQUEST_MAGIC = 0x53344451U;
static constexpr uint32_t S41_DUMMY_RESPONSE_MAGIC = 0x53344452U;
static constexpr uint16_t S41_DUMMY_PROTOCOL_VERSION = 1;

struct s41_dummy_request {
    uint32_t magic;
    uint16_t version;
    uint16_t reserved;
    uint32_t request_id;
    uint32_t elements;
    uint32_t input_bytes;
    uint32_t input_crc32;
};

struct s41_dummy_response {
    uint32_t magic;
    uint16_t version;
    uint16_t status;
    uint32_t request_id;
    uint32_t elements;
    uint32_t output_bytes;
    uint32_t output_crc32;
    uint64_t validate_ns;
    uint64_t decode_ns;
    uint64_t set_ns;
    uint64_t submit_ns;
    uint64_t sync_ns;
    uint64_t get_ns;
    uint64_t encode_hash_ns;
    uint64_t prewrite_ns;
    uint64_t previous_write_ns;
};

static_assert(sizeof(s41_dummy_request) == 24,
        "unexpected dummy request layout");
static_assert(sizeof(s41_dummy_response) == 96,
        "unexpected dummy response layout");

#endif
