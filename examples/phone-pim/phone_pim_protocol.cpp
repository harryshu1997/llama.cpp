#include "phone_pim_protocol.h"

extern "C" {
#include "sha256.h"
}

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <sys/stat.h>
#include <unistd.h>

namespace phone_pim {
namespace {

constexpr size_t k_header_crc_offset = 96;

void put_u16(uint8_t * dst, uint16_t value) {
    dst[0] = static_cast<uint8_t>(value);
    dst[1] = static_cast<uint8_t>(value >> 8);
}

void put_u32(uint8_t * dst, uint32_t value) {
    for (size_t i = 0; i < 4; ++i) {
        dst[i] = static_cast<uint8_t>(value >> (8 * i));
    }
}

void put_u64(uint8_t * dst, uint64_t value) {
    for (size_t i = 0; i < 8; ++i) {
        dst[i] = static_cast<uint8_t>(value >> (8 * i));
    }
}

uint16_t get_u16(const uint8_t * src) {
    return static_cast<uint16_t>(src[0]) |
           static_cast<uint16_t>(src[1]) << 8;
}

uint32_t get_u32(const uint8_t * src) {
    uint32_t value = 0;
    for (size_t i = 0; i < 4; ++i) {
        value |= static_cast<uint32_t>(src[i]) << (8 * i);
    }
    return value;
}

uint64_t get_u64(const uint8_t * src) {
    uint64_t value = 0;
    for (size_t i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(src[i]) << (8 * i);
    }
    return value;
}

uint32_t crc32(const uint8_t * data, size_t size) {
    uint32_t crc = 0xffffffffU;
    for (size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit) {
            const uint32_t mask = 0U - (crc & 1U);
            crc = (crc >> 1) ^ (0xedb88320U & mask);
        }
    }
    return ~crc;
}

bool known_opcode(uint16_t value) {
    switch (static_cast<Opcode>(value)) {
        case Opcode::hello:
        case Opcode::prepare:
        case Opcode::execute:
        case Opcode::status:
        case Opcode::release:
        case Opcode::ping:
        case Opcode::close:
        case Opcode::shutdown:
        case Opcode::stage_begin:
        case Opcode::stage_chunk:
        case Opcode::stage_commit:
        case Opcode::stage_abort:
        case Opcode::error:
            return true;
    }
    return false;
}

} // namespace

std::array<uint8_t, 32> sha256(const void * data, size_t size) {
    std::array<uint8_t, 32> digest = {};
    sha256_hash(digest.data(), static_cast<const unsigned char *>(data), size);
    return digest;
}

bool sha256_file(
        const std::string & path,
        std::array<uint8_t, 32> & digest,
        uint64_t & size,
        std::string & error) {
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        error = "cannot open file for SHA-256: " + std::string(std::strerror(errno));
        return false;
    }

    struct stat st = {};
    if (fstat(fd, &st) != 0 || st.st_size < 0) {
        error = "cannot stat file for SHA-256: " + std::string(std::strerror(errno));
        close(fd);
        return false;
    }
    size = static_cast<uint64_t>(st.st_size);
    const bool ok = sha256_fd_range(fd, 0, size, digest, error);
    close(fd);
    return ok;
}

bool sha256_fd_range(
        int fd,
        uint64_t offset,
        uint64_t size,
        std::array<uint8_t, 32> & digest,
        std::string & error) {
    if (fd < 0 || offset > static_cast<uint64_t>(std::numeric_limits<off_t>::max()) ||
        size > static_cast<uint64_t>(std::numeric_limits<off_t>::max()) - offset) {
        error = "SHA-256 file range exceeds off_t";
        return false;
    }
    sha256_t context;
    sha256_init(&context);
    std::array<uint8_t, 1024 * 1024> buffer = {};
    uint64_t completed = 0;
    while (completed < size) {
        const size_t wanted = static_cast<size_t>(std::min<uint64_t>(buffer.size(), size - completed));
        const ssize_t count = pread(
                fd, buffer.data(), wanted, static_cast<off_t>(offset + completed));
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            error = count == 0 ? "short read while hashing file range"
                               : "cannot hash file range: " + std::string(std::strerror(errno));
            return false;
        }
        sha256_update(&context, buffer.data(), static_cast<size_t>(count));
        completed += static_cast<uint64_t>(count);
    }
    sha256_final(&context, digest.data());
    return true;
}

std::string hex_sha256(const std::array<uint8_t, 32> & digest) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string out(64, '0');
    for (size_t i = 0; i < digest.size(); ++i) {
        out[2 * i] = hex[digest[i] >> 4];
        out[2 * i + 1] = hex[digest[i] & 0x0f];
    }
    return out;
}

uint64_t opcode_payload_limit(Opcode opcode) {
    switch (opcode) {
        case Opcode::stage_begin:
            return k_max_stage_manifest_payload;
        case Opcode::stage_chunk:
            return k_max_stage_chunk_bytes + k_stage_chunk_envelope_bytes;
        case Opcode::execute:
            return k_default_max_payload;
        case Opcode::hello:
        case Opcode::prepare:
        case Opcode::status:
        case Opcode::release:
        case Opcode::ping:
        case Opcode::close:
        case Opcode::shutdown:
        case Opcode::stage_commit:
        case Opcode::stage_abort:
        case Opcode::error:
            return 64ULL * 1024;
    }
    return 0;
}

std::array<uint8_t, k_header_bytes> encode_header(const Header & header) {
    std::array<uint8_t, k_header_bytes> out = {};
    put_u32(out.data() + 0, k_magic);
    put_u16(out.data() + 4, k_version);
    put_u16(out.data() + 6, static_cast<uint16_t>(header.opcode));
    put_u32(out.data() + 8, header.flags);
    put_u32(out.data() + 12, static_cast<uint32_t>(k_header_bytes));
    put_u64(out.data() + 16, header.request_id);
    put_u64(out.data() + 24, header.command_seq);
    put_u64(out.data() + 32, header.session_epoch);
    put_u64(out.data() + 40, header.route_epoch);
    put_u64(out.data() + 48, header.residency_generation);
    put_u64(out.data() + 56, header.payload_bytes);
    std::copy(header.payload_sha256.begin(), header.payload_sha256.end(), out.begin() + 64);
    put_u32(out.data() + k_header_crc_offset, 0);
    put_u32(out.data() + k_header_crc_offset, crc32(out.data(), out.size()));
    return out;
}

bool decode_header(const uint8_t * bytes, size_t size, Header & header, std::string & error) {
    if (size != k_header_bytes) {
        error = "invalid header size";
        return false;
    }
    if (get_u32(bytes + 0) != k_magic) {
        error = "invalid magic";
        return false;
    }
    if (get_u16(bytes + 4) != k_version) {
        error = "unsupported version";
        return false;
    }
    const uint16_t opcode = get_u16(bytes + 6);
    if (!known_opcode(opcode)) {
        error = "unknown opcode";
        return false;
    }
    if (get_u32(bytes + 12) != k_header_bytes) {
        error = "invalid encoded header size";
        return false;
    }
    std::array<uint8_t, k_header_bytes> copy = {};
    std::copy(bytes, bytes + size, copy.begin());
    const uint32_t expected_crc = get_u32(copy.data() + k_header_crc_offset);
    put_u32(copy.data() + k_header_crc_offset, 0);
    if (crc32(copy.data(), copy.size()) != expected_crc) {
        error = "header checksum mismatch";
        return false;
    }
    header.opcode = static_cast<Opcode>(opcode);
    header.flags = get_u32(bytes + 8);
    header.request_id = get_u64(bytes + 16);
    header.command_seq = get_u64(bytes + 24);
    header.session_epoch = get_u64(bytes + 32);
    header.route_epoch = get_u64(bytes + 40);
    header.residency_generation = get_u64(bytes + 48);
    header.payload_bytes = get_u64(bytes + 56);
    std::copy(bytes + 64, bytes + 96, header.payload_sha256.begin());
    return true;
}

bool validate_command(
        const Header & header,
        uint64_t expected_session_epoch,
        uint64_t expected_route_epoch,
        uint64_t expected_generation,
        uint64_t last_command_seq,
        uint64_t last_request_id,
        bool require_generation,
        std::string & error) {
    if (header.flags != 0) {
        error = "request contains response flags";
        return false;
    }
    if (header.command_seq == 0 || header.command_seq <= last_command_seq) {
        error = "duplicate or reordered command sequence";
        return false;
    }
    if (header.request_id == 0 || header.request_id <= last_request_id) {
        error = "duplicate or reordered request id";
        return false;
    }
    if (header.session_epoch != expected_session_epoch) {
        error = "stale session epoch";
        return false;
    }
    if (header.route_epoch != expected_route_epoch) {
        error = "stale route epoch";
        return false;
    }
    if (require_generation && header.residency_generation != expected_generation) {
        error = "stale residency generation";
        return false;
    }
    return true;
}

void Writer::u32(uint32_t value) {
    const size_t old = data_.size();
    data_.resize(old + 4);
    put_u32(data_.data() + old, value);
}

void Writer::u64(uint64_t value) {
    const size_t old = data_.size();
    data_.resize(old + 8);
    put_u64(data_.data() + old, value);
}

void Writer::string(const std::string & value) {
    u32(static_cast<uint32_t>(value.size()));
    bytes(value.data(), value.size());
}

void Writer::bytes(const void * data, size_t size) {
    const size_t old = data_.size();
    data_.resize(old + size);
    if (size > 0) {
        std::memcpy(data_.data() + old, data, size);
    }
}

bool Reader::u32(uint32_t & value) {
    if (remaining() < 4) {
        return false;
    }
    value = get_u32(current());
    offset_ += 4;
    return true;
}

bool Reader::u64(uint64_t & value) {
    if (remaining() < 8) {
        return false;
    }
    value = get_u64(current());
    offset_ += 8;
    return true;
}

bool Reader::string(std::string & value, size_t max_size) {
    uint32_t length = 0;
    if (!u32(length) || length > max_size || remaining() < length) {
        return false;
    }
    value.assign(reinterpret_cast<const char *>(current()), length);
    offset_ += length;
    return true;
}

bool Reader::bytes(void * dst, size_t size) {
    if (remaining() < size) {
        return false;
    }
    if (size > 0) {
        std::memcpy(dst, current(), size);
    }
    offset_ += size;
    return true;
}

bool Reader::skip(size_t size) {
    if (remaining() < size) {
        return false;
    }
    offset_ += size;
    return true;
}

std::vector<uint8_t> make_error_payload(ErrorCode code, const std::string & message) {
    Writer writer;
    writer.u32(static_cast<uint32_t>(code));
    writer.string(message.substr(0, 4096));
    return writer.take();
}

bool parse_error_payload(const std::vector<uint8_t> & payload, ErrorCode & code, std::string & message) {
    Reader reader(payload);
    uint32_t raw = 0;
    if (!reader.u32(raw) || !reader.string(message) || !reader.done()) {
        return false;
    }
    code = static_cast<ErrorCode>(raw);
    return true;
}

} // namespace phone_pim
