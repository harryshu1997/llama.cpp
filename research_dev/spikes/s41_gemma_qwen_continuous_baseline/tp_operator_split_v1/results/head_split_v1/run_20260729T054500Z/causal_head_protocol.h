#ifndef S41_CAUSAL_HEAD_PROTOCOL_H
#define S41_CAUSAL_HEAD_PROTOCOL_H

#include "causal_ffn_protocol.h"

#include <cstddef>
#include <cstdint>

static constexpr uint32_t S41_HEAD_REQUEST_MAGIC = 0x53344851U;
static constexpr uint32_t S41_HEAD_RESPONSE_MAGIC = 0x53344852U;
static constexpr uint16_t S41_HEAD_PROTOCOL_VERSION = 1;
static constexpr uint16_t S41_HEAD_FLAG_FAST_HASH = 1;
static constexpr uint16_t S41_HEAD_FLAG_F16_INPUT = 2;
static constexpr uint16_t S41_HEAD_FLAG_I8_INPUT = 4;

struct s41_head_request {
    uint32_t magic;
    uint16_t version;
    uint16_t flags;
    uint32_t request_id;
    uint32_t type;
    uint32_t k;
    uint32_t vocab;
    uint32_t offset;
    uint32_t count;
    uint32_t input_bytes;
    uint32_t input_hash;
    uint64_t weight_hash;
};

struct s41_head_top1 {
    uint32_t token;
    float score;
};

struct s41_head_response {
    uint32_t magic;
    uint16_t version;
    uint16_t status;
    uint32_t request_id;
    uint32_t token;
    float score;
    uint32_t output_hash;
    uint64_t weight_hash;
    uint32_t set_us;
    uint32_t compute_us;
    uint32_t get_us;
    uint32_t reduce_us;
};

static_assert(sizeof(s41_head_request) == 48, "unexpected request layout");
static_assert(sizeof(s41_head_top1) == 8, "unexpected top1 layout");
static_assert(sizeof(s41_head_response) == 48, "unexpected response layout");

#endif
