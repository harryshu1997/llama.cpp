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
// the client asks for it in HELLO and the phone echoes it: every payload hash of the session
// is hash_payload_fast instead of hash_bytes; a phone build without it refuses the HELLO
static constexpr uint16_t flag_fast_hash = 4;

enum class message_type : uint16_t {
    hello_request = 1,
    hello_response = 2,
    execute_request = 3,
    execute_response = 4,
    // Fused attention-projection calls (WS15). They reuse the execute_request /
    // execute_response frames (header + payload), so a worker without fused
    // support rejects them as an invalid execute request and the HELLO above is
    // unchanged. A client sends fused_hello_request once after HELLO, and only
    // when its configuration owns fused layers.
    fused_hello_request = 9,
    fused_hello_response = 10,
    // owned layer l: payload rows [attention output (n_q) | residual (n_embd)],
    // response rows [delta (n_embd)] with l_out = residual + delta, where
    // delta = O(l)(attention) + FFN(l)(ffn_norm(l)(residual + O(l)(attention)))
    fused_execute_request = 11,
    // as above, response rows [delta | q (n_q) | k (n_kv) | v (n_kv)]: the raw
    // (pre q/k-norm, pre-RoPE) projections of layer l+1 from attn_norm(l+1)(l_out)
    fused_next_execute_request = 12,
    fused_execute_response = 13,
};

// payload of fused_hello_request (the client's request) and fused_hello_response
// (the worker's capability; status in the response header)
struct fused_hello_payload {
    uint64_t fused_layer_mask;
    uint64_t next_qkv_mask;
    uint32_t n_q;
    uint32_t n_kv;
    uint64_t fused_weight_hash;
    float rms_eps;
    uint32_t reserved;
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
static_assert(sizeof(fused_hello_payload) == 40, "unexpected fused hello payload layout");

// rows of a fused response: delta, then (with the next layer's projections) q, k, v
static inline uint32_t fused_response_row(uint32_t n_embd, uint32_t n_q, uint32_t n_kv, bool next_qkv) {
    return n_embd + (next_qkv ? n_q + 2 * n_kv : 0);
}

static inline uint32_t hash_bytes(const void * data, size_t size) {
    const uint8_t * bytes = static_cast<const uint8_t *>(data);
    uint32_t hash = 2166136261U;
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 16777619U;
    }
    return hash;
}

static inline uint64_t hash_rotl64(uint64_t value, int bits) {
    return (value << bits) | (value >> (64 - bits));
}

// Transport checksum of the same strength class as hash_bytes, at memory speed: four independent
// xxh64-style lanes over 8-byte little-endian words (zero-padded tail), merged with the length and
// avalanched, folded to 32 bits. hash_bytes is one multiply per byte, ~10 ms per 4 MB on a phone core.
static inline uint32_t hash_payload_fast(const void * data, size_t size) {
    constexpr uint64_t p1 = 11400714785074694791ULL;
    constexpr uint64_t p2 = 14029467366897019727ULL;
    constexpr uint64_t p3 = 1609587929392839161ULL;
    constexpr uint64_t p4 = 9650029242287828579ULL;
    const auto round = [](uint64_t acc, uint64_t word) {
        return hash_rotl64(acc + word * p2, 31) * p1;
    };
    const uint8_t * bytes = static_cast<const uint8_t *>(data);
    uint64_t lane0 = p1 + p2;
    uint64_t lane1 = p2;
    uint64_t lane2 = 0;
    uint64_t lane3 = 0 - p1;
    size_t offset = 0;
    for (; offset + 32 <= size; offset += 32) {
        uint64_t words[4];
        memcpy(words, bytes + offset, sizeof(words));
        lane0 = round(lane0, words[0]);
        lane1 = round(lane1, words[1]);
        lane2 = round(lane2, words[2]);
        lane3 = round(lane3, words[3]);
    }
    uint64_t hash = hash_rotl64(lane0, 1) + hash_rotl64(lane1, 7) +
            hash_rotl64(lane2, 12) + hash_rotl64(lane3, 18);
    hash += static_cast<uint64_t>(size);
    while (offset < size) {
        uint64_t word = 0;
        const size_t count = size - offset < 8 ? size - offset : 8;
        memcpy(&word, bytes + offset, count);
        hash = hash_rotl64(hash ^ round(0, word), 27) * p1 + p4;
        offset += count;
    }
    hash ^= hash >> 33;
    hash *= p2;
    hash ^= hash >> 29;
    hash *= p3;
    hash ^= hash >> 32;
    return static_cast<uint32_t>(hash ^ (hash >> 32));
}

// payload hash of a session whose HELLO flags are flags
static inline uint32_t payload_hash(uint16_t flags, const void * data, size_t size) {
    return (flags & flag_fast_hash) != 0 ? hash_payload_fast(data, size) : hash_bytes(data, size);
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
