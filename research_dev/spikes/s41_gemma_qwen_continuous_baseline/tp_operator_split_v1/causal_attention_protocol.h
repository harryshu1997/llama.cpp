#ifndef S41_CAUSAL_ATTENTION_PROTOCOL_H
#define S41_CAUSAL_ATTENTION_PROTOCOL_H

#include "causal_ffn_protocol.h"

#include <cstddef>
#include <cstdint>

static constexpr uint32_t S41_ATTN_REQUEST_MAGIC = 0x53344151U;
static constexpr uint32_t S41_ATTN_RESPONSE_MAGIC = 0x53344152U;
static constexpr uint16_t S41_ATTN_PROTOCOL_VERSION = 3;
static constexpr uint16_t S41_ATTN_FLAG_FAST_HASH = 1;
static constexpr uint16_t S41_ATTN_FLAG_F16_INPUT = 2;
static constexpr uint16_t S41_ATTN_FLAG_F16_OUTPUT = 4;
static constexpr uint16_t S41_ATTN_FLAG_LAST_SLOT_UPDATE = 8;
static constexpr uint16_t S41_ATTN_FLAG_SINGLE_GROUP = 16;
static constexpr uint16_t S41_ATTN_FLAG_STATE_OUTPUT = 32;
static constexpr uint16_t S41_ATTN_FLAG_RESIDUAL_OUTPUT = 64;

struct s41_attention_request {
    uint32_t magic;
    uint16_t version;
    uint16_t flags;
    uint32_t request_id;
    uint32_t type;
    uint32_t k;
    uint32_t n_kv;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t group_offset;
    uint32_t group_count;
    uint32_t input_bytes;
    uint32_t input_hash;
    uint64_t weight_hash;
};

struct s41_attention_response {
    uint32_t magic;
    uint16_t version;
    uint16_t status;
    uint32_t request_id;
    uint32_t output_elements;
    uint32_t output_bytes;
    uint32_t output_hash;
    uint64_t weight_hash;
    uint32_t set_us;
    uint32_t compute_us;
    uint32_t get_us;
    uint32_t reserved;
};

static_assert(
        sizeof(s41_attention_request) == 56,
        "unexpected attention request layout");
static_assert(
        sizeof(s41_attention_response) == 48,
        "unexpected attention response layout");

static inline float s41_kv_value(
        uint32_t matrix,
        uint32_t group,
        uint32_t token,
        uint32_t dimension) {
    uint64_t key = (uint64_t) matrix << 56;
    key ^= (uint64_t) group << 40;
    key ^= (uint64_t) token * 0xd6e8feb86659fd93ULL;
    key ^= (uint64_t) dimension * 0xa5a3564e27f8862fULL;
    const uint32_t bits = (uint32_t) (s41_mix64(key) >> 40);
    const int32_t centered = (int32_t) bits - 0x800000;
    return (float) centered * (1.0f / 33554432.0f);
}

#endif
