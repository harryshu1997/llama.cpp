#include "phone_pim_protocol.h"
#include "phone_pim_socket.h"

#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace pp = phone_pim;

namespace {

int failures = 0;
int checks = 0;

void check(bool condition, const char * name) {
    ++checks;
    std::printf("  %s %s\n", condition ? "PASS" : "FAIL", name);
    if (!condition) {
        ++failures;
    }
}

std::array<pp::Fd, 2> socket_pair() {
    int sockets[2] = {-1, -1};
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0) {
        return {};
    }
    std::array<pp::Fd, 2> result;
    result[0].reset(sockets[0]);
    result[1].reset(sockets[1]);
    return result;
}

} // namespace

int main() {
    const char abc[] = "abc";
    check(
            pp::hex_sha256(pp::sha256(abc, 3)) ==
                    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            "SHA-256 known vector");

    {
        char path[] = "/tmp/phone_pim_sha256_XXXXXX";
        pp::Fd file(mkstemp(path));
        std::array<uint8_t, 32> digest = {};
        std::array<uint8_t, 32> range_digest = {};
        uint64_t size = 0;
        std::string file_error;
        const bool written = file.valid() && write(file.get(), abc, 3) == 3;
        const bool range_hashed = written &&
                pp::sha256_fd_range(file.get(), 1, 2, range_digest, file_error);
        file.reset();
        const bool hashed = written && pp::sha256_file(path, digest, size, file_error);
        unlink(path);
        check(hashed && size == 3 && digest == pp::sha256(abc, 3), "file SHA-256 binds bytes and size");
        check(range_hashed && range_digest == pp::sha256(abc + 1, 2),
              "file-range SHA-256 hashes the exact prefix/range bytes");
    }

    check(pp::opcode_payload_limit(pp::Opcode::stage_begin) == pp::k_max_stage_manifest_payload &&
                  pp::opcode_payload_limit(pp::Opcode::stage_chunk) ==
                          pp::k_max_stage_chunk_bytes + pp::k_stage_chunk_envelope_bytes &&
                  pp::opcode_payload_limit(pp::Opcode::ping) == 64 * 1024,
          "v3 opcodes have fixed pre-allocation payload caps");

    pp::Header source;
    source.opcode = pp::Opcode::execute;
    source.flags = 0x1234;
    source.request_id = 0x0102030405060708ULL;
    source.command_seq = 9;
    source.session_epoch = 13;
    source.route_epoch = 10;
    source.residency_generation = 11;
    source.payload_bytes = 12;
    source.payload_sha256 = pp::sha256(abc, 3);
    const auto encoded = pp::encode_header(source);
    check(encoded[0] == 'P' && encoded[1] == 'P' && encoded[2] == 'I' && encoded[3] == 'M',
          "header magic is explicit little-endian bytes");
    check(encoded[16] == 0x08 && encoded[23] == 0x01, "u64 header fields are little-endian");

    pp::Header decoded;
    std::string error;
    check(pp::decode_header(encoded.data(), encoded.size(), decoded, error) &&
                  decoded.opcode == source.opcode && decoded.flags == source.flags &&
                  decoded.request_id == source.request_id && decoded.command_seq == source.command_seq &&
                  decoded.session_epoch == source.session_epoch &&
                  decoded.route_epoch == source.route_epoch &&
                  decoded.residency_generation == source.residency_generation &&
                  decoded.payload_bytes == source.payload_bytes &&
                  decoded.payload_sha256 == source.payload_sha256,
          "header round trip");

    auto corrupt_header = encoded;
    corrupt_header[24] ^= 0x40;
    check(!pp::decode_header(corrupt_header.data(), corrupt_header.size(), decoded, error),
          "header checksum rejects corruption");

    pp::Writer writer;
    writer.u32(0x11223344U);
    writer.u64(0x0102030405060708ULL);
    writer.string("island");
    pp::Reader reader(writer.data());
    uint32_t value32 = 0;
    uint64_t value64 = 0;
    std::string text;
    check(reader.u32(value32) && reader.u64(value64) && reader.string(text) && reader.done() &&
                  value32 == 0x11223344U && value64 == 0x0102030405060708ULL && text == "island",
          "bounded payload codec round trip");
    pp::Reader truncated(writer.data().data(), writer.data().size() - 1);
    check(truncated.u32(value32) && truncated.u64(value64) && !truncated.string(text),
          "payload codec rejects truncation");

    pp::Header command;
    command.request_id = 2;
    command.command_seq = 2;
    command.session_epoch = 6;
    command.route_epoch = 7;
    command.residency_generation = 8;
    check(pp::validate_command(command, 6, 7, 8, 1, 1, true, error), "command gate accepts current epochs");
    check(!pp::validate_command(command, 5, 7, 8, 1, 1, true, error), "command gate rejects stale session");
    check(!pp::validate_command(command, 6, 7, 8, 2, 1, true, error), "command gate rejects duplicate sequence");
    check(!pp::validate_command(command, 6, 7, 8, 1, 2, true, error), "command gate rejects duplicate request id");
    check(!pp::validate_command(command, 6, 9, 8, 1, 1, true, error), "command gate rejects stale route");
    check(!pp::validate_command(command, 6, 7, 9, 1, 1, true, error), "command gate rejects stale residency");

    {
        auto sockets = socket_pair();
        pp::Header header;
        header.opcode = pp::Opcode::ping;
        header.request_id = 5;
        header.command_seq = 1;
        const std::vector<uint8_t> payload = {1, 2, 3, 4, 5};
        std::string send_error;
        std::thread sender([&] { pp::send_frame(sockets[0].get(), header, payload, send_error); });
        pp::Frame frame;
        const pp::ReceiveResult result = pp::receive_frame(sockets[1].get(), frame, 1024, error);
        sender.join();
        check(send_error.empty() && result == pp::ReceiveResult::ok && frame.payload == payload &&
                      frame.header.request_id == 5,
              "socket frame round trip");
    }

    {
        auto sockets = socket_pair();
        pp::Header header;
        header.opcode = pp::Opcode::ping;
        header.command_seq = 1;
        header.payload_bytes = 65;
        header.payload_sha256 = pp::sha256(nullptr, 0);
        const auto bytes = pp::encode_header(header);
        check(pp::send_bytes(sockets[0].get(), bytes.data(), bytes.size(), error), "oversize test header sent");
        pp::Frame frame;
        check(pp::receive_frame(sockets[1].get(), frame, 64, error) == pp::ReceiveResult::error &&
                      error.find("bound") != std::string::npos,
              "frame receiver rejects oversize before allocation");
    }

    {
        auto sockets = socket_pair();
        const std::vector<uint8_t> payload = {7, 8, 9};
        pp::Header header;
        header.opcode = pp::Opcode::ping;
        header.command_seq = 1;
        header.payload_bytes = payload.size();
        header.payload_sha256 = {};
        const auto bytes = pp::encode_header(header);
        std::thread sender([&] {
            std::string send_error;
            pp::send_bytes(sockets[0].get(), bytes.data(), bytes.size(), send_error);
            pp::send_bytes(sockets[0].get(), payload.data(), payload.size(), send_error);
        });
        pp::Frame frame;
        error.clear();
        const pp::ReceiveResult result = pp::receive_frame(sockets[1].get(), frame, 64, error);
        sender.join();
        check(result == pp::ReceiveResult::error && error.find("SHA-256") != std::string::npos,
              "frame receiver rejects payload corruption");
    }

    {
        auto sockets = socket_pair();
        const std::array<uint8_t, 2> payload = {1, 2};
        pp::Header header;
        header.opcode = pp::Opcode::ping;
        header.command_seq = 1;
        header.payload_bytes = 4;
        header.payload_sha256 = pp::sha256(payload.data(), payload.size());
        const auto bytes = pp::encode_header(header);
        std::thread sender([&] {
            std::string send_error;
            pp::send_bytes(sockets[0].get(), bytes.data(), bytes.size(), send_error);
            pp::send_bytes(sockets[0].get(), payload.data(), payload.size(), send_error);
            shutdown(sockets[0].get(), SHUT_WR);
        });
        pp::Frame frame;
        error.clear();
        const pp::ReceiveResult result = pp::receive_frame(sockets[1].get(), frame, 64, error);
        sender.join();
        check(result == pp::ReceiveResult::error && error.find("truncated") != std::string::npos,
              "frame receiver rejects truncated payload");
    }

    {
        auto sockets = socket_pair();
        std::string deadline_error;
        const bool deadline_set = pp::set_socket_deadlines(sockets[1].get(), 20, deadline_error);
        pp::Header header;
        header.opcode = pp::Opcode::ping;
        header.command_seq = 1;
        header.payload_bytes = 4;
        const std::array<uint8_t, 4> payload = {1, 2, 3, 4};
        header.payload_sha256 = pp::sha256(payload.data(), payload.size());
        const auto bytes = pp::encode_header(header);
        const bool header_sent = pp::send_bytes(sockets[0].get(), bytes.data(), bytes.size(), deadline_error);
        pp::Frame frame;
        const pp::ReceiveResult result = pp::receive_frame(sockets[1].get(), frame, 64, deadline_error);
        check(deadline_set && header_sent && result == pp::ReceiveResult::error &&
                      deadline_error.find("deadline") != std::string::npos,
              "frame receiver enforces one absolute header-plus-payload deadline");
    }

    std::printf("protocol tests: %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
