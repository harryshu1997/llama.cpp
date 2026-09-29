#ifndef S41_MODEL_OPS_PROTOCOL_H
#define S41_MODEL_OPS_PROTOCOL_H

#include <cstddef>
#include <cstdint>

static constexpr uint32_t S41_MODEL_OP_REQUEST_MAGIC = 0x534d4f51U;
static constexpr uint32_t S41_MODEL_OP_RESPONSE_MAGIC = 0x534d4f52U;
static constexpr uint16_t S41_MODEL_OP_PROTOCOL_VERSION = 1;

static constexpr uint32_t S41_QWEN3_14B_HIDDEN = 5120;
static constexpr uint32_t S41_QWEN3_14B_INTERMEDIATE = 17408;
static constexpr uint32_t S41_QWEN3_14B_HEAD_DIM = 128;
static constexpr uint32_t S41_QWEN3_14B_GQA = 5;

enum s41_model_op : uint16_t {
    S41_MODEL_OP_RMSNORM = 1,
    S41_MODEL_OP_SWIGLU = 2,
    S41_MODEL_OP_ATTENTION = 3,
};

struct s41_model_op_request {
    uint32_t magic;
    uint16_t version;
    uint16_t op;
    uint32_t request_id;
    uint32_t input_elements;
    uint32_t output_elements;
    uint32_t input_bytes;
    uint32_t input_crc32;
};

struct s41_model_op_response {
    uint32_t magic;
    uint16_t version;
    uint16_t status;
    uint32_t request_id;
    uint32_t op;
    uint32_t output_elements;
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

static_assert(sizeof(s41_model_op_request) == 28,
        "unexpected model-op request layout");
static_assert(sizeof(s41_model_op_response) == 104,
        "unexpected model-op response layout");

static inline uint32_t s41_model_op_input_elements(s41_model_op op) {
    switch (op) {
        case S41_MODEL_OP_RMSNORM:
            return S41_QWEN3_14B_HIDDEN;
        case S41_MODEL_OP_SWIGLU:
            return 2 * S41_QWEN3_14B_INTERMEDIATE;
        case S41_MODEL_OP_ATTENTION:
            return S41_QWEN3_14B_GQA * S41_QWEN3_14B_HEAD_DIM;
    }
    return 0;
}

static inline uint32_t s41_model_op_output_elements(s41_model_op op) {
    switch (op) {
        case S41_MODEL_OP_RMSNORM:
            return S41_QWEN3_14B_HIDDEN;
        case S41_MODEL_OP_SWIGLU:
            return S41_QWEN3_14B_INTERMEDIATE;
        case S41_MODEL_OP_ATTENTION:
            return S41_QWEN3_14B_GQA * S41_QWEN3_14B_HEAD_DIM;
    }
    return 0;
}

static inline float s41_model_norm_weight(uint32_t index) {
    const int32_t centered = (int32_t) ((index * 17U + 3U) % 33U) - 16;
    return 1.0f + (float) centered * (1.0f / 2048.0f);
}

static inline float s41_model_key_value(uint32_t token, uint32_t dimension) {
    const int32_t centered =
            (int32_t) ((token * 13U + dimension * 7U + 11U) % 257U) - 128;
    return (float) centered * (1.0f / 256.0f);
}

static inline float s41_model_value_value(uint32_t token, uint32_t dimension) {
    const int32_t centered =
            (int32_t) ((token * 5U + dimension * 11U + 19U) % 257U) - 128;
    return (float) centered * (1.0f / 512.0f);
}

#endif
