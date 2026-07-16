#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace phone_pim {

constexpr uint32_t k_magic = 0x4d495050U; // "PPIM" on the wire.
constexpr uint16_t k_version = 3;
constexpr size_t k_header_bytes = 104;
constexpr uint64_t k_default_max_payload = 64ULL * 1024 * 1024;
constexpr uint64_t k_max_stage_manifest_payload = 256ULL * 1024;
constexpr uint64_t k_max_stage_chunk_bytes = 16ULL * 1024 * 1024;
constexpr uint64_t k_stage_chunk_envelope_bytes = 56;
constexpr uint32_t k_max_stage_chunks = 4096;

enum class Opcode : uint16_t {
    hello    = 1,
    prepare  = 2,
    execute  = 3,
    status   = 4,
    release  = 5,
    ping     = 6,
    close    = 7,
    shutdown = 8,
    stage_begin  = 9,
    stage_chunk  = 10,
    stage_commit = 11,
    stage_abort  = 12,
    error    = 255,
};

enum FrameFlag : uint32_t {
    flag_response = 1U << 0,
    flag_error    = 1U << 1,
};

enum class ModelSource : uint32_t {
    prestaged = 1,
    published_store = 2,
};

struct Header {
    Opcode opcode = Opcode::error;
    uint32_t flags = 0;
    uint64_t request_id = 0;
    uint64_t command_seq = 0;
    uint64_t session_epoch = 0;
    uint64_t route_epoch = 0;
    uint64_t residency_generation = 0;
    uint64_t payload_bytes = 0;
    std::array<uint8_t, 32> payload_sha256 = {};
};

struct Frame {
    Header header;
    std::vector<uint8_t> payload;
};

enum class ErrorCode : uint32_t {
    protocol = 1,
    stale_epoch = 2,
    duplicate_command = 3,
    invalid_state = 4,
    invalid_argument = 5,
    unsupported = 6,
    backend = 7,
    internal = 8,
    storage = 9,
    hash_mismatch = 10,
};

std::array<uint8_t, 32> sha256(const void * data, size_t size);
bool sha256_file(
        const std::string & path,
        std::array<uint8_t, 32> & digest,
        uint64_t & size,
        std::string & error);
bool sha256_fd_range(
        int fd,
        uint64_t offset,
        uint64_t size,
        std::array<uint8_t, 32> & digest,
        std::string & error);
std::string hex_sha256(const std::array<uint8_t, 32> & digest);
uint64_t opcode_payload_limit(Opcode opcode);

std::array<uint8_t, k_header_bytes> encode_header(const Header & header);
bool decode_header(const uint8_t * bytes, size_t size, Header & header, std::string & error);

bool validate_command(
        const Header & header,
        uint64_t expected_session_epoch,
        uint64_t expected_route_epoch,
        uint64_t expected_generation,
        uint64_t last_command_seq,
        uint64_t last_request_id,
        bool require_generation,
        std::string & error);

class Writer {
public:
    void u32(uint32_t value);
    void u64(uint64_t value);
    void string(const std::string & value);
    void bytes(const void * data, size_t size);

    const std::vector<uint8_t> & data() const { return data_; }
    std::vector<uint8_t> take() { return std::move(data_); }

private:
    std::vector<uint8_t> data_;
};

class Reader {
public:
    Reader(const uint8_t * data, size_t size) : data_(data), size_(size) {}
    explicit Reader(const std::vector<uint8_t> & data) : Reader(data.data(), data.size()) {}

    bool u32(uint32_t & value);
    bool u64(uint64_t & value);
    bool string(std::string & value, size_t max_size = 4096);
    bool bytes(void * dst, size_t size);
    bool skip(size_t size);

    const uint8_t * current() const { return data_ + offset_; }
    size_t remaining() const { return size_ - offset_; }
    bool done() const { return offset_ == size_; }

private:
    const uint8_t * data_ = nullptr;
    size_t size_ = 0;
    size_t offset_ = 0;
};

std::vector<uint8_t> make_error_payload(ErrorCode code, const std::string & message);
bool parse_error_payload(const std::vector<uint8_t> & payload, ErrorCode & code, std::string & message);

} // namespace phone_pim
