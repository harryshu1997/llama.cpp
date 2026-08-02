#ifndef S41_CAUSAL_FFN_PROTOCOL_H
#define S41_CAUSAL_FFN_PROTOCOL_H

#include <cstddef>
#include <cstdint>

static constexpr uint32_t S41_FFN_REQUEST_MAGIC  = 0x53343151U;
static constexpr uint32_t S41_FFN_RESPONSE_MAGIC = 0x53343152U;
static constexpr uint16_t S41_FFN_PROTOCOL_VERSION = 2;
static constexpr uint16_t S41_FFN_FLAG_FAST_HASH = 1;
static constexpr uint16_t S41_FFN_FLAG_F16_INPUT = 2;
static constexpr uint16_t S41_FFN_FLAG_I8_INPUT = 4;
static constexpr uint16_t S41_FFN_FLAG_F16_OUTPUT = 8;
static constexpr uint64_t S41_HASH64_OFFSET = 14695981039346656037ULL;

enum s41_weight_id : uint32_t {
    S41_WEIGHT_GATE = 1,
    S41_WEIGHT_UP   = 2,
    S41_WEIGHT_DOWN = 3,
    S41_WEIGHT_Q    = 4,
    S41_WEIGHT_K    = 5,
    S41_WEIGHT_V    = 6,
    S41_WEIGHT_O    = 7,
};

struct s41_ffn_request {
    uint32_t magic;
    uint16_t version;
    uint16_t reserved;
    uint32_t request_id;
    uint32_t type;
    uint32_t k;
    uint32_t n_ff;
    uint32_t offset;
    uint32_t count;
    uint32_t input_bytes;
    uint32_t input_hash;
    uint64_t weight_hash;
};

struct s41_ffn_response {
    uint32_t magic;
    uint16_t version;
    uint16_t status;
    uint32_t request_id;
    uint32_t k;
    uint32_t output_bytes;
    uint32_t output_hash;
    uint64_t weight_hash;
};

static_assert(sizeof(s41_ffn_request) == 48, "unexpected request layout");
static_assert(sizeof(s41_ffn_response) == 32, "unexpected response layout");

static inline uint64_t s41_mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static inline float s41_activation_value(uint32_t index) {
    const uint32_t bits = (uint32_t) (s41_mix64(0x41a7c9e3ULL ^ index) >> 40);
    const int32_t centered = (int32_t) bits - 0x800000;
    return (float) centered * (1.0f / 8388608.0f);
}

static inline uint32_t s41_hash_bytes(const void * data, size_t size) {
    const uint8_t * bytes = (const uint8_t *) data;
    uint32_t hash = 2166136261U;
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 16777619U;
    }
    return hash;
}

static inline uint32_t s41_fast_hash_bytes(const void * data, size_t size) {
    const uint8_t * bytes = (const uint8_t *) data;
    uint64_t hash = s41_mix64(S41_HASH64_OFFSET ^ (uint64_t) size);
    size_t offset = 0;
    while (offset + sizeof(uint64_t) <= size) {
        uint64_t lane = 0;
        __builtin_memcpy(&lane, bytes + offset, sizeof(lane));
        hash ^= s41_mix64(lane + (uint64_t) offset);
        hash = ((hash << 27) | (hash >> 37)) *
                0x3c79ac492ba7b653ULL + 0x1c69b3f74ac4ae35ULL;
        offset += sizeof(uint64_t);
    }
    uint64_t tail = 0;
    for (size_t index = 0; offset + index < size; ++index) {
        tail |= (uint64_t) bytes[offset + index] << (8 * index);
    }
    hash ^= s41_mix64(tail ^ (uint64_t) (size - offset));
    hash = s41_mix64(hash);
    return (uint32_t) (hash ^ (hash >> 32));
}

static inline uint64_t s41_hash64_update(
        uint64_t hash, const void * data, size_t size) {
    const uint8_t * bytes = (const uint8_t *) data;
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static inline uint64_t s41_weight_hash_update(
        uint64_t hash,
        uint32_t matrix,
        uint32_t row_offset,
        uint32_t column_offset,
        uint32_t columns,
        uint32_t rows,
        const void * data,
        size_t size) {
    const uint32_t metadata[] = {
        matrix, row_offset, column_offset, columns, rows,
    };
    const uint64_t byte_count = size;
    hash = s41_hash64_update(hash, metadata, sizeof(metadata));
    hash = s41_hash64_update(hash, &byte_count, sizeof(byte_count));
    return s41_hash64_update(hash, data, size);
}

#endif
