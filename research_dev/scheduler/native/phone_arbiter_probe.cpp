#include "ffn-split-protocol.h"

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

using steady_clock = std::chrono::steady_clock;

uint64_t now_us() {
    return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                    steady_clock::now().time_since_epoch()).count());
}

int parse_port(const char * text) {
    errno = 0;
    char * end = nullptr;
    const long value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        value <= 0 || value > 65535) {
        throw std::runtime_error("invalid port");
    }
    return static_cast<int>(value);
}

uint64_t parse_gap(const char * text) {
    errno = 0;
    char * end = nullptr;
    const unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value == 0) {
        throw std::runtime_error("invalid protected gap");
    }
    return static_cast<uint64_t>(value);
}

std::vector<uint32_t> parse_filler_tokens(const char * text) {
    if (text == nullptr || *text == '\0') {
        throw std::runtime_error("empty filler token list");
    }
    std::vector<uint32_t> result;
    std::string remaining(text);
    while (!remaining.empty()) {
        const size_t comma = remaining.find(',');
        const std::string item = remaining.substr(0, comma);
        errno = 0;
        char * end = nullptr;
        const unsigned long value = strtoul(item.c_str(), &end, 10);
        if (errno != 0 || end == item.c_str() || *end != '\0' ||
            value == 0 || value > 16 ||
            std::find(result.begin(), result.end(), value) != result.end()) {
            throw std::runtime_error("invalid filler token list");
        }
        result.push_back(static_cast<uint32_t>(value));
        if (comma == std::string::npos) {
            break;
        }
        remaining.erase(0, comma + 1);
    }
    return result;
}

bool send_exact(int fd, const void * data, size_t size) {
    const uint8_t * cursor = static_cast<const uint8_t *>(data);
    while (size > 0) {
        const ssize_t count = send(fd, cursor, size, MSG_NOSIGNAL);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            return false;
        }
        cursor += count;
        size -= static_cast<size_t>(count);
    }
    return true;
}

bool receive_exact(int fd, void * data, size_t size) {
    uint8_t * cursor = static_cast<uint8_t *>(data);
    while (size > 0) {
        const ssize_t count = recv(fd, cursor, size, 0);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            return false;
        }
        cursor += count;
        size -= static_cast<size_t>(count);
    }
    return true;
}

int connect_client(const char * host, int port) {
    const uint64_t deadline_us = now_us() + UINT64_C(30000000);
    for (;;) {
        const int fd = socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in address = {};
        address.sin_family = AF_INET;
        address.sin_port = htons(static_cast<uint16_t>(port));
        if (fd >= 0 &&
            inet_pton(AF_INET, host, &address.sin_addr) == 1 &&
            connect(fd, reinterpret_cast<sockaddr *>(&address),
                    sizeof(address)) == 0) {
            const int one = 1;
            setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
            return fd;
        }
        if (fd >= 0) {
            close(fd);
        }
        if (now_us() >= deadline_us) {
            throw std::runtime_error("client connect timed out");
        }
        usleep(10000);
    }
}

ffn_split::hello_response handshake(
        int fd, uint64_t layer_mask, uint32_t n_embd,
        uint32_t columns, uint16_t flags) {
    ffn_split::hello_request request = {};
    request.magic = ffn_split::protocol_magic;
    request.version = ffn_split::protocol_version;
    request.message = static_cast<uint16_t>(
            ffn_split::message_type::hello_request);
    request.layer_mask = layer_mask;
    request.n_embd = n_embd;
    request.max_columns = columns;
    request.flags = flags;
    request.max_tokens = 512;
    ffn_split::hello_response response = {};
    if (!send_exact(fd, &request, sizeof(request)) ||
        !receive_exact(fd, &response, sizeof(response)) ||
        response.magic != ffn_split::protocol_magic ||
        response.version != ffn_split::protocol_version ||
        response.message != static_cast<uint16_t>(
                ffn_split::message_type::hello_response) ||
        response.status != 0 || response.flags != flags ||
        response.n_embd != n_embd ||
        response.max_columns != columns ||
        response.layer_mask != layer_mask) {
        throw std::runtime_error("client HELLO failed");
    }
    return response;
}

uint64_t execute(
        int fd, uint32_t request_id, int layer,
        uint32_t n_embd, uint32_t columns, uint32_t tokens = 1) {
    std::vector<uint16_t> input(n_embd * tokens, 0);
    ffn_split::execute_request request = {};
    request.magic = ffn_split::protocol_magic;
    request.version = ffn_split::protocol_version;
    request.message = static_cast<uint16_t>(
            ffn_split::message_type::execute_request);
    request.request_id = request_id;
    request.layer = layer;
    request.elements = n_embd * tokens;
    request.payload_bytes = static_cast<uint32_t>(
            input.size() * sizeof(input[0]));
    request.payload_hash = ffn_split::hash_bytes(
            input.data(), request.payload_bytes);
    request.columns = columns;
    request.tokens = tokens;
    const uint64_t started_us = now_us();
    if (!send_exact(fd, &request, sizeof(request)) ||
        !send_exact(fd, input.data(), request.payload_bytes)) {
        throw std::runtime_error("execute request failed");
    }
    ffn_split::execute_response response = {};
    std::vector<uint16_t> output(n_embd * tokens);
    if (!receive_exact(fd, &response, sizeof(response)) ||
        !receive_exact(fd, output.data(), request.payload_bytes) ||
        response.magic != ffn_split::protocol_magic ||
        response.version != ffn_split::protocol_version ||
        response.message != static_cast<uint16_t>(
                ffn_split::message_type::execute_response) ||
        response.status != 0 || response.request_id != request_id ||
        response.layer != layer || response.elements != request.elements ||
        response.payload_bytes != request.payload_bytes ||
        response.columns != columns || response.tokens != tokens ||
        response.payload_hash != ffn_split::hash_bytes(
                output.data(), request.payload_bytes)) {
        throw std::runtime_error("execute response failed");
    }
    return now_us() - started_us;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 5 && argc != 6) {
        fprintf(stderr,
                "usage: %s <host> <protected-port> <filler-port> "
                "<protected-gap-us> [comma-separated-filler-tokens-1-to-16]\n",
                argv[0]);
        return 2;
    }
    try {
        const int protected_port = parse_port(argv[2]);
        const int filler_port = parse_port(argv[3]);
        const uint64_t protected_gap_us = parse_gap(argv[4]);
        const std::vector<uint32_t> filler_tokens =
                argc == 6 ? parse_filler_tokens(argv[5]) :
                            std::vector<uint32_t>{1};
        const int protected_fd = connect_client(argv[1], protected_port);
        handshake(protected_fd, UINT64_C(0xfff), 5120, 17408,
                ffn_split::flag_f16_io | ffn_split::flag_swiglu);
        const int filler_fd = connect_client(argv[1], filler_port);
        handshake(filler_fd, UINT64_C(0x7fffff), 3840, 6144,
                ffn_split::flag_f16_io);

        struct sample {
            uint32_t tokens = 0;
            uint64_t observed_gap_us = 0;
            uint64_t protected_group_rpc_us = 0;
            uint64_t filler_client_rpc_us = 0;
        };
        std::vector<sample> samples;
        uint32_t protected_request_id = 1;
        uint32_t filler_request_id = 1;
        for (uint32_t tokens : filler_tokens) {
            uint64_t filler_rpc_us = 0;
            std::string filler_error;
            std::thread filler([&]() {
                try {
                    filler_rpc_us = execute(
                            filler_fd, filler_request_id++, 0,
                            3840, 6144, tokens);
                } catch (const std::exception & error) {
                    filler_error = error.what();
                }
            });

            uint64_t protected_rpc_total_us = 0;
            for (int layer = 0; layer <= 11; ++layer) {
                protected_rpc_total_us += execute(
                        protected_fd, protected_request_id++,
                        layer, 5120, 17408);
            }
            const uint64_t idle_started_us = now_us();
            filler.join();
            if (!filler_error.empty()) {
                throw std::runtime_error(filler_error);
            }
            const uint64_t elapsed_us = now_us() - idle_started_us;
            if (elapsed_us < protected_gap_us) {
                usleep(static_cast<useconds_t>(
                        protected_gap_us - elapsed_us));
            }
            samples.push_back({
                tokens,
                now_us() - idle_started_us,
                protected_rpc_total_us,
                filler_rpc_us,
            });
        }

        uint64_t validation_group_rpc_us = 0;
        for (int layer = 0; layer <= 11; ++layer) {
            validation_group_rpc_us += execute(
                    protected_fd, protected_request_id++,
                    layer, 5120, 17408);
        }
        close(filler_fd);
        close(protected_fd);

        printf(
                "PHONEARBITERPROBE {\"status\":\"PASS\","
                "\"protected_groups\":%zu,"
                "\"protected_group_calls\":%zu,"
                "\"filler_calls\":%zu,"
                "\"configured_gap_us\":%llu,\"samples\":[",
                samples.size() + 1, (samples.size() + 1) * 12,
                samples.size(),
                static_cast<unsigned long long>(protected_gap_us));
        for (size_t index = 0; index < samples.size(); ++index) {
            const sample & value = samples[index];
            printf(
                    "%s{\"tokens\":%u,\"observed_gap_us\":%llu,"
                    "\"protected_group_rpc_us\":%llu,"
                    "\"filler_client_rpc_us\":%llu}",
                    index == 0 ? "" : ",", value.tokens,
                    static_cast<unsigned long long>(value.observed_gap_us),
                    static_cast<unsigned long long>(
                            value.protected_group_rpc_us),
                    static_cast<unsigned long long>(
                            value.filler_client_rpc_us));
        }
        printf(
                "],\"validation_group_rpc_us\":%llu}\n",
                static_cast<unsigned long long>(
                        validation_group_rpc_us));
        return 0;
    } catch (const std::exception & error) {
        fprintf(stderr, "[phone-arbiter-probe] %s\n", error.what());
        return 1;
    }
}
