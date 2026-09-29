#include "ffn-split-dmabuf.h"
#include "ffn-split-functionfs.h"
#include "ffn-split-protocol.h"

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <climits>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

namespace {

struct config {
    std::string ffs_root;
    std::string ready_file;
    std::vector<int> target_ports;
    long max_sessions = 0;
};

struct target_connection {
    int fd = -1;
    int port = 0;
    ffn_split::hello_response response = {};
};

struct target_selection {
    std::vector<target_connection> targets;
    ffn_split::hello_response response = {};
};

bool parse_long(const char * text, long minimum, long maximum, long & value) {
    if (text == nullptr || *text == '\0') {
        return false;
    }
    errno = 0;
    char * end = nullptr;
    const long parsed = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        parsed < minimum || parsed > maximum) {
        return false;
    }
    value = parsed;
    return true;
}

bool parse_config(int argc, char ** argv, config & result) {
    for (int index = 1; index < argc; ++index) {
        if (strcmp(argv[index], "--ffs-root") == 0 && index + 1 < argc) {
            result.ffs_root = argv[++index];
        } else if (strcmp(argv[index], "--ready-file") == 0 &&
                   index + 1 < argc) {
            result.ready_file = argv[++index];
        } else if (strcmp(argv[index], "--target") == 0 &&
                   index + 1 < argc) {
            long port = 0;
            if (!parse_long(argv[++index], 1, 65535, port)) {
                return false;
            }
            result.target_ports.push_back(static_cast<int>(port));
        } else if (strcmp(argv[index], "--max-sessions") == 0 &&
                   index + 1 < argc) {
            if (!parse_long(argv[++index], 0, LONG_MAX,
                            result.max_sessions)) {
                return false;
            }
        } else {
            return false;
        }
    }
    std::sort(result.target_ports.begin(), result.target_ports.end());
    return !result.ffs_root.empty() && !result.ready_file.empty() &&
            !result.target_ports.empty() &&
            std::adjacent_find(
                    result.target_ports.begin(),
                    result.target_ports.end()) == result.target_ports.end();
}

bool receive_exact(int fd, void * destination, size_t size) {
    uint8_t * cursor = static_cast<uint8_t *>(destination);
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

bool send_exact(int fd, const void * source, size_t size) {
    const uint8_t * cursor = static_cast<const uint8_t *>(source);
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

int connect_target(int port) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in address = {};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<uint16_t>(port));
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    const int one = 1;
    if (fd < 0 ||
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) != 0 ||
        connect(fd, reinterpret_cast<sockaddr *>(&address),
                sizeof(address)) != 0) {
        if (fd >= 0) {
            close(fd);
        }
        return -1;
    }
    return fd;
}

bool valid_hello_response(
        const ffn_split::hello_request & request,
        const ffn_split::hello_response & response) {
    const uint16_t supported_flags =
            ffn_split::flag_f16_io | ffn_split::flag_swiglu;
    return response.magic == ffn_split::protocol_magic &&
            response.version == ffn_split::protocol_version &&
            response.message == static_cast<uint16_t>(
                    ffn_split::message_type::hello_response) &&
            response.status == 0 && response.weight_hash != 0 &&
            (request.flags & ~supported_flags) == 0 &&
            response.flags == request.flags &&
            response.n_embd == request.n_embd &&
            response.max_columns == request.max_columns &&
            response.max_tokens == request.max_tokens &&
            response.layer_mask != 0 &&
            (response.layer_mask & ~request.layer_mask) == 0 &&
            response.layer_count == static_cast<uint32_t>(
                    __builtin_popcountll(response.layer_mask)) &&
            response.column_quantum != 0 &&
            response.offset + response.max_columns == response.n_ff;
}

bool same_target_identity(
        const ffn_split::hello_response & lhs,
        const ffn_split::hello_response & rhs) {
    return lhs.flags == rhs.flags && lhs.n_embd == rhs.n_embd &&
            lhs.n_ff == rhs.n_ff && lhs.offset == rhs.offset &&
            lhs.max_columns == rhs.max_columns &&
            lhs.weight_type == rhs.weight_type &&
            lhs.column_quantum == rhs.column_quantum &&
            lhs.max_tokens == rhs.max_tokens &&
            lhs.alternate_columns_32 == rhs.alternate_columns_32;
}

void close_selection(target_selection & selected) {
    for (target_connection & target : selected.targets) {
        if (target.fd >= 0) {
            close(target.fd);
            target.fd = -1;
        }
    }
    selected.targets.clear();
}

bool select_target(
        const config & cfg,
        const ffn_split::hello_request & hello,
        target_selection & selected) {
    uint64_t covered_layers = 0;
    uint64_t aggregate_hash = UINT64_C(14695981039346656037);
    for (int port : cfg.target_ports) {
        const int fd = connect_target(port);
        if (fd < 0) {
            fprintf(stderr,
                    "[resident-router] unavailable target_port=%d\n", port);
            continue;
        }
        ffn_split::hello_response response = {};
        const bool matched = send_exact(fd, &hello, sizeof(hello)) &&
                receive_exact(fd, &response, sizeof(response)) &&
                valid_hello_response(hello, response);
        if (!matched) {
            close(fd);
            continue;
        }
        if ((covered_layers & response.layer_mask) != 0 ||
            (!selected.targets.empty() &&
             !same_target_identity(selected.response, response))) {
            fprintf(stderr,
                    "[resident-router] incompatible target port=%d\n", port);
            close(fd);
            close_selection(selected);
            return false;
        }
        if (selected.targets.empty()) {
            selected.response = response;
        }
        const uint64_t identity[] = {
            response.layer_mask,
            response.weight_hash,
        };
        aggregate_hash = ffn_split::hash64_update(
                aggregate_hash, identity, sizeof(identity));
        covered_layers |= response.layer_mask;
        selected.targets.push_back({fd, port, response});
    }
    if (covered_layers != hello.layer_mask) {
        close_selection(selected);
        return false;
    }
    selected.response.layer_mask = covered_layers;
    selected.response.layer_count = static_cast<uint32_t>(
            __builtin_popcountll(covered_layers));
    selected.response.weight_hash = aggregate_hash;
    return !selected.targets.empty();
}

target_connection * target_for_layer(
        target_selection & selected, int layer) {
    const uint64_t bit = UINT64_C(1) << layer;
    for (target_connection & target : selected.targets) {
        if ((target.response.layer_mask & bit) != 0) {
            return &target;
        }
    }
    return nullptr;
}

ffn_split::hello_response rejected_hello(
        const ffn_split::hello_request & request) {
    ffn_split::hello_response response = {};
    response.magic = ffn_split::protocol_magic;
    response.version = ffn_split::protocol_version;
    response.message = static_cast<uint16_t>(
            ffn_split::message_type::hello_response);
    response.status = 1;
    response.flags = request.flags;
    response.n_embd = request.n_embd;
    response.max_columns = request.max_columns;
    response.layer_mask = request.layer_mask;
    response.max_tokens = request.max_tokens;
    return response;
}

bool valid_execute_request(
        const ffn_split::hello_response & hello,
        const ffn_split::execute_request & request) {
    const size_t element_size = hello.flags & ffn_split::flag_f16_io ?
            sizeof(uint16_t) : sizeof(float);
    const uint64_t expected_elements =
            static_cast<uint64_t>(hello.n_embd) * request.tokens;
    const uint64_t expected_bytes = expected_elements * element_size;
    return request.magic == ffn_split::protocol_magic &&
            request.version == ffn_split::protocol_version &&
            request.message == static_cast<uint16_t>(
                    ffn_split::message_type::execute_request) &&
            request.request_id != 0 && request.layer >= 0 &&
            request.layer < 64 &&
            (hello.layer_mask & (UINT64_C(1) << request.layer)) != 0 &&
            request.tokens > 0 && request.tokens <= hello.max_tokens &&
            request.elements == expected_elements &&
            request.payload_bytes == expected_bytes &&
            request.columns > 0 && request.columns <= hello.max_columns &&
            (request.columns == hello.max_columns ||
             request.columns ==
                     static_cast<uint32_t>(hello.alternate_columns_32) * 32 ||
             request.columns % hello.column_quantum == 0);
}

bool valid_execute_response(
        const ffn_split::execute_request & request,
        const ffn_split::execute_response & response,
        const std::vector<uint8_t> & payload) {
    return response.magic == ffn_split::protocol_magic &&
            response.version == ffn_split::protocol_version &&
            response.message == static_cast<uint16_t>(
                    ffn_split::message_type::execute_response) &&
            response.status == 0 &&
            response.request_id == request.request_id &&
            response.layer == request.layer &&
            response.elements == request.elements &&
            response.payload_bytes == request.payload_bytes &&
            response.columns == request.columns &&
            response.tokens == request.tokens &&
            response.payload_hash == ffn_split::hash_bytes(
                    payload.data(), payload.size());
}

} // namespace

int main(int argc, char ** argv) {
    config cfg;
    if (!parse_config(argc, argv, cfg)) {
        fprintf(stderr,
                "usage: %s --ffs-root PATH --ready-file PATH "
                "--target PORT [--target PORT ...] [--max-sessions N]\n",
                argv[0]);
        return 2;
    }
    signal(SIGPIPE, SIG_IGN);

    const int ep0 = ffn_split::functionfs_open_endpoint(
            cfg.ffs_root, "ep0");
    if (ep0 < 0 ||
        !ffn_split::functionfs_write_exact(
                ep0, &ffn_split::functionfs_descriptors,
                sizeof(ffn_split::functionfs_descriptors)) ||
        !ffn_split::functionfs_write_exact(
                ep0, &ffn_split::functionfs_strings,
                sizeof(ffn_split::functionfs_strings))) {
        fprintf(stderr,
                "[resident-router] descriptor setup failed: %s\n",
                strerror(errno));
        return 1;
    }
    const int device_to_host = ffn_split::functionfs_open_endpoint(
            cfg.ffs_root, "ep1");
    const int host_to_device = ffn_split::functionfs_open_endpoint(
            cfg.ffs_root, "ep2");
    if (device_to_host < 0 || host_to_device < 0 ||
        !ffn_split::functionfs_touch(
                cfg.ready_file, "resident_router_ready\n")) {
        fprintf(stderr,
                "[resident-router] endpoint setup failed: %s\n",
                strerror(errno));
        return 1;
    }

    ffn_split::functionfs_event_monitor event_monitor(ep0);
    uint64_t generation = 0;
    if (!event_monitor.wait_enabled(
                generation, ffn_split::functionfs_enable_timeout_ms)) {
        fprintf(stderr,
                "[resident-router] FunctionFS enable failed: %s\n",
                strerror(errno));
        return 1;
    }
    fprintf(stderr,
            "[resident-router] ready targets=%zu generation=%llu\n",
            cfg.target_ports.size(),
            static_cast<unsigned long long>(generation));
    fflush(stderr);

    long sessions = 0;
    long requests = 0;
    bool terminate_requested = false;
    bool wait_for_reenable = false;
    uint64_t previous_generation = generation;
    for (;;) {
        if (wait_for_reenable) {
            if (!event_monitor.wait_reenabled(
                        previous_generation, generation,
                        ffn_split::functionfs_enable_timeout_ms)) {
                fprintf(stderr,
                        "[resident-router] FunctionFS re-enable failed: %s\n",
                        strerror(errno));
                return 1;
            }
            wait_for_reenable = false;
        }

        ffn_split::hello_request hello = {};
        if (!ffn_split::functionfs_read_exact(
                    host_to_device, &hello, sizeof(hello))) {
            previous_generation = generation;
            wait_for_reenable = true;
            continue;
        }
        target_selection selected;
        if (!select_target(cfg, hello, selected)) {
            const ffn_split::hello_response response = rejected_hello(hello);
            ffn_split::functionfs_write_exact(
                    device_to_host, &response, sizeof(response));
            fprintf(stderr, "[resident-router] arm rejected\n");
            continue;
        }
        if (!ffn_split::functionfs_write_exact(
                    device_to_host, &selected.response,
                    sizeof(selected.response))) {
            close_selection(selected);
            previous_generation = generation;
            wait_for_reenable = true;
            continue;
        }
        fprintf(stderr,
                "RESIDENTARM {\"status\":\"WARM\","
                "\"target_count\":%zu,\"layer_mask\":\"%016llx\","
                "\"n_embd\":%u,"
                "\"columns\":%u,\"weight_hash\":\"%016llx\"}\n",
                selected.targets.size(),
                static_cast<unsigned long long>(
                        selected.response.layer_mask),
                selected.response.n_embd,
                selected.response.max_columns,
                static_cast<unsigned long long>(
                        selected.response.weight_hash));
        fflush(stderr);

        bool reset = false;
        bool complete_session = false;
        while (!reset && !complete_session) {
            ffn_split::execute_request request = {};
            if (!ffn_split::functionfs_read_exact(
                        host_to_device, &request, sizeof(request))) {
                reset = true;
                break;
            }
            const bool shutdown =
                    request.magic == ffn_split::protocol_magic &&
                    request.version == ffn_split::protocol_version &&
                    request.message == static_cast<uint16_t>(
                            ffn_split::message_type::execute_request) &&
                    request.request_id == 0;
            if (shutdown) {
                terminate_requested = request.layer == -1;
                complete_session = true;
                break;
            }
            if (!valid_execute_request(selected.response, request)) {
                fprintf(stderr, "[resident-router] invalid execute request\n");
                close_selection(selected);
                return 1;
            }
            target_connection * target = target_for_layer(
                    selected, request.layer);
            if (target == nullptr) {
                fprintf(stderr, "[resident-router] missing layer target\n");
                close_selection(selected);
                return 1;
            }

            std::vector<uint8_t> input(request.payload_bytes);
            const uint32_t payload_ready =
                    ffn_split::dmabuf_payload_ready_magic ^ request.request_id;
            if (!ffn_split::functionfs_write_exact(
                        device_to_host, &payload_ready,
                        sizeof(payload_ready)) ||
                !ffn_split::functionfs_read_exact(
                        host_to_device, input.data(), input.size())) {
                reset = true;
                break;
            }
            if (request.payload_hash != ffn_split::hash_bytes(
                        input.data(), input.size()) ||
                !send_exact(target->fd, &request, sizeof(request)) ||
                !send_exact(target->fd, input.data(), input.size())) {
                fprintf(stderr, "[resident-router] target request failed\n");
                close_selection(selected);
                return 1;
            }

            ffn_split::execute_response response = {};
            std::vector<uint8_t> output(request.payload_bytes);
            if (!receive_exact(target->fd, &response, sizeof(response)) ||
                !receive_exact(target->fd, output.data(), output.size()) ||
                !valid_execute_response(request, response, output)) {
                fprintf(stderr, "[resident-router] target response failed\n");
                close_selection(selected);
                return 1;
            }
            std::vector<uint8_t> wire(
                    ffn_split::dmabuf_payload_offset + output.size(), 0);
            memcpy(wire.data(), &response, sizeof(response));
            memcpy(wire.data() + ffn_split::dmabuf_payload_offset,
                    output.data(), output.size());
            if (!ffn_split::functionfs_write_exact(
                        device_to_host, wire.data(), wire.size())) {
                reset = true;
                break;
            }
            ++requests;
        }
        close_selection(selected);
        if (reset) {
            previous_generation = generation;
            wait_for_reenable = true;
            fprintf(stderr,
                    "[resident-router] USB reset requests=%ld\n", requests);
            fflush(stderr);
            continue;
        }

        ++sessions;
        fprintf(stderr,
                "[resident-router] session complete sessions=%ld requests=%ld\n",
                sessions, requests);
        fflush(stderr);
        if (cfg.max_sessions > 0 && sessions >= cfg.max_sessions) {
            break;
        }
        if (terminate_requested) {
            break;
        }
    }

    fprintf(stderr,
            "RESIDENTROUTER {\"status\":\"ok\",\"sessions\":%ld,"
            "\"requests\":%ld,\"terminate_requested\":%s}\n",
            sessions, requests, terminate_requested ? "true" : "false");
    close(host_to_device);
    close(device_to_host);
    close(ep0);
    return 0;
}
