#ifndef S41_GEMMA_GLOBAL_ATTENTION_PROTOCOL_H
#define S41_GEMMA_GLOBAL_ATTENTION_PROTOCOL_H

#include "ggml.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

static constexpr uint32_t S41_GEMMA_ATTN_REQUEST_MAGIC = 0x47415131;
static constexpr uint32_t S41_GEMMA_ATTN_RESPONSE_MAGIC = 0x47415231;
static constexpr uint16_t S41_GEMMA_ATTN_PROTOCOL_VERSION = 1;

static constexpr uint32_t S41_GEMMA_ATTN_HEAD_DIM = 512;
static constexpr uint32_t S41_GEMMA_ATTN_HEADS = 16;
static constexpr uint32_t S41_GEMMA_ATTN_Q_ELEMENTS =
        S41_GEMMA_ATTN_HEAD_DIM * S41_GEMMA_ATTN_HEADS;
static constexpr uint32_t S41_GEMMA_ATTN_PACKED_STRIDE =
        S41_GEMMA_ATTN_HEAD_DIM + 2;
static constexpr uint32_t S41_GEMMA_ATTN_STATE_BYTES =
        S41_GEMMA_ATTN_Q_ELEMENTS * sizeof(ggml_fp16_t);
static constexpr uint32_t S41_GEMMA_ATTN_ANCHOR_BYTES =
        2 * S41_GEMMA_ATTN_HEADS * sizeof(float);

static constexpr uint64_t S41_GEMMA_HASH_OFFSET = 1469598103934665603ULL;
static constexpr uint64_t S41_GEMMA_HASH_PRIME = 1099511628211ULL;

#pragma pack(push, 1)
struct s41_gemma_attention_request {
    uint32_t magic;
    uint16_t version;
    uint16_t flags;
    uint32_t request_id;
    uint32_t total_kv;
    uint32_t segment_offset;
    uint32_t segment_tokens;
    uint32_t q_bytes;
    uint64_t q_hash;
    uint64_t kv_hash;
};

struct s41_gemma_attention_response {
    uint32_t magic;
    uint16_t version;
    uint16_t status;
    uint32_t request_id;
    uint32_t state_bytes;
    uint32_t anchor_bytes;
    uint64_t output_hash;
    uint64_t kv_hash;
    uint32_t set_us;
    uint32_t compute_us;
    uint32_t get_us;
    uint32_t encode_us;
};
#pragma pack(pop)

static_assert(sizeof(s41_gemma_attention_request) == 44,
        "unexpected Gemma attention request size");
static_assert(sizeof(s41_gemma_attention_response) == 52,
        "unexpected Gemma attention response size");

static inline uint64_t s41_gemma_hash_bytes(
        const void * data, size_t size,
        uint64_t hash = S41_GEMMA_HASH_OFFSET) {
    const uint8_t * bytes = static_cast<const uint8_t *>(data);
    for (size_t index = 0; index < size; ++index) {
        hash ^= bytes[index];
        hash *= S41_GEMMA_HASH_PRIME;
    }
    return hash;
}

static inline float s41_gemma_q_value(uint32_t head, uint32_t dimension) {
    const double x = (double) (head * S41_GEMMA_ATTN_HEAD_DIM + dimension);
    return (float) (0.045 * std::sin(0.013 * x) +
                    0.025 * std::cos(0.007 * x + 0.17 * head));
}

static inline float s41_gemma_k_value(
        uint32_t token, uint32_t dimension) {
    const double x = (double) token;
    const double d = (double) dimension;
    return (float) (0.050 * std::sin(0.00091 * x + 0.017 * d) +
                    0.020 * std::cos(0.00037 * x - 0.011 * d));
}

static inline float s41_gemma_v_value(
        uint32_t token, uint32_t dimension) {
    const double x = (double) token;
    const double d = (double) dimension;
    return (float) (0.060 * std::sin(0.019 * d + 0.1) +
                    0.080 * std::sin(0.00073 * x - 0.009 * d) +
                    0.030 * std::cos(0.00029 * x + 0.015 * d));
}

#endif
