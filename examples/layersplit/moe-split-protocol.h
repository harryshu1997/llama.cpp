#pragma once

#include <cstddef>
#include <cstdint>

namespace moe_split {

static constexpr uint32_t protocol_magic = 0x4d534631U;
static constexpr uint16_t protocol_version = 1;
static constexpr uint16_t flag_f16_io = 1;

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
    uint16_t flags;
    uint16_t reserved;
};

struct hello_response {
    uint32_t magic;
    uint16_t version;
    uint16_t message;
    uint16_t status;
    uint16_t flags;
    uint32_t n_embd;
    uint32_t n_ff_exp;
    uint32_t n_expert;
    uint32_t n_expert_used;
    uint32_t weight_type;
    uint32_t layer_count;
    uint64_t layer_mask;
    uint64_t weight_hash;
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
    uint64_t compute_us;
};

static_assert(sizeof(hello_request) == 24, "unexpected MoE split hello request layout");
static_assert(sizeof(hello_response) == 56, "unexpected MoE split hello response layout");
static_assert(sizeof(execute_request) == 28, "unexpected MoE split execute request layout");
static_assert(sizeof(execute_response) == 40, "unexpected MoE split execute response layout");

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

} // namespace moe_split
