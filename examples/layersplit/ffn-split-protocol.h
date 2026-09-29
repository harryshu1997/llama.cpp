#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace ffn_split {

static constexpr uint32_t protocol_magic = 0x46534631U;
static constexpr uint16_t protocol_version = 6;
static constexpr uint16_t flag_f16_io = 1;
static constexpr uint16_t flag_swiglu = 2;

enum class message_type : uint16_t {
    hello_request = 1,
    hello_response = 2,
    execute_request = 3,
    execute_response = 4,
};

struct hello_request {
    uint32_t magic;
    uint16_t version;
    uint16_t message;
    uint64_t layer_mask;
    uint32_t n_embd;
    uint32_t max_columns;
    uint16_t flags;
    uint16_t max_tokens;
    uint8_t artifact_sha256[32];
};

struct hello_response {
    uint32_t magic;
    uint16_t version;
    uint16_t message;
    uint16_t status;
    uint16_t flags;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t offset;
    uint32_t max_columns;
    uint32_t weight_type;
    uint32_t layer_count;
    uint64_t layer_mask;
    uint64_t weight_hash;
    uint32_t column_quantum;
    uint16_t max_tokens;
    uint16_t alternate_columns_32;
    uint8_t artifact_sha256[32];
};

struct execute_request {
    uint32_t magic;
    uint16_t version;
    uint16_t message;
    uint32_t request_id;
    int32_t layer;
    uint32_t elements;
    uint32_t payload_bytes;
    uint32_t payload_hash;
    uint32_t columns;
    uint32_t tokens;
};

struct execute_response {
    uint32_t magic;
    uint16_t version;
    uint16_t message;
    uint16_t status;
    uint16_t reserved;
    uint32_t request_id;
    int32_t layer;
    uint32_t elements;
    uint32_t payload_bytes;
    uint32_t payload_hash;
    uint32_t columns;
    uint32_t tokens;
    uint64_t compute_us;
};

static_assert(sizeof(hello_request) == 64, "unexpected FFN split hello request layout");
static_assert(sizeof(hello_response) == 96, "unexpected FFN split hello response layout");
static_assert(sizeof(execute_request) == 36, "unexpected FFN split execute request layout");
static_assert(sizeof(execute_response) == 48, "unexpected FFN split execute response layout");

static inline uint32_t hash_bytes(const void * data, size_t size) {
    const uint8_t * bytes = static_cast<const uint8_t *>(data);
    uint32_t hash = 2166136261U;
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 16777619U;
    }
    return hash;
}

static inline uint64_t hash64_update(uint64_t hash, const void * data, size_t size) {
    const uint8_t * bytes = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static inline int hex_digit(char value) {
    if (value >= '0' && value <= '9') {
        return value - '0';
    }
    if (value >= 'a' && value <= 'f') {
        return value - 'a' + 10;
    }
    return -1;
}

static inline bool parse_artifact_sha256(
        const std::string & value, uint8_t (&digest)[32]) {
    if (value.size() != 71 || value.compare(0, 7, "sha256:") != 0) {
        return false;
    }
    for (size_t index = 0; index < 32; ++index) {
        const int high = hex_digit(value[7 + 2 * index]);
        const int low = hex_digit(value[8 + 2 * index]);
        if (high < 0 || low < 0) {
            return false;
        }
        digest[index] = static_cast<uint8_t>((high << 4) | low);
    }
    return true;
}

static inline bool same_artifact_sha256(
        const uint8_t (&left)[32], const uint8_t (&right)[32]) {
    return memcmp(left, right, sizeof(left)) == 0;
}

} // namespace ffn_split
