#include "ffn-split-dmabuf.h"
#include "ffn-split-functionfs.h"
#include "ffn-split-protocol.h"
#include "ffn-split-session-manifest.h"

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <climits>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

namespace {

using steady_clock = std::chrono::steady_clock;

struct config {
    std::string ffs_root;
    std::string ready_file;
    std::string manifest;
    long max_sessions = 0;
    long call_log_period = 16;
    int queue_depth = 1;
};

struct session_totals {
    long calls = 0;
    uint64_t rows = 0;
    uint64_t h2d_bytes = 0;
    uint64_t d2h_bytes = 0;
    uint64_t receive_block_us = 0;
    uint64_t d2h_us = 0;
    uint64_t compute_us = 0;
    uint64_t rpc_us = 0;
};

struct session_proof_totals {
    ffn_split::resident_session_shard shard;
    session_totals totals;
};

struct target_connection {
    int fd = -1;
    size_t proof_index = 0;
    ffn_split::hello_response response = {};
};

struct target_selection {
    std::vector<target_connection> targets;
    ffn_split::hello_response response = {};
};

uint64_t now_us() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
            steady_clock::now().time_since_epoch()).count());
}

uint64_t epoch_us() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
}

void print_call(
        const ffn_split::resident_session_shard & shard, long calls,
        long period) {
    if (calls != 1 && calls % period != 0) {
        return;
    }
    fprintf(stderr,
            "RESIDENTCALL {\"schema\":\"s42-phone-residency-call-v1\"," 
            "\"session_id\":\"%s\",\"artifact_sha256\":\"%s\"," 
            "\"session_generation\":%llu,\"calls\":%ld," 
            "\"monotonic_us\":%llu,\"epoch_us\":%llu}\n",
            shard.session_id.c_str(), shard.artifact_sha256.c_str(),
            static_cast<unsigned long long>(shard.session_generation), calls,
            static_cast<unsigned long long>(now_us()),
            static_cast<unsigned long long>(epoch_us()));
    fflush(stderr);
}

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
        } else if (strcmp(argv[index], "--ready-file") == 0 && index + 1 < argc) {
            result.ready_file = argv[++index];
        } else if (strcmp(argv[index], "--manifest") == 0 && index + 1 < argc) {
            result.manifest = argv[++index];
        } else if (strcmp(argv[index], "--max-sessions") == 0 && index + 1 < argc) {
            if (!parse_long(argv[++index], 0, LONG_MAX, result.max_sessions)) {
                return false;
            }
        } else if (strcmp(argv[index], "--call-log-period") == 0 && index + 1 < argc) {
            if (!parse_long(argv[++index], 1, LONG_MAX, result.call_log_period)) {
                return false;
            }
        } else if (strcmp(argv[index], "--queue-depth") == 0 && index + 1 < argc) {
            long value = 0;
            if (!parse_long(argv[++index], 1, 8, value)) {
                return false;
            }
            result.queue_depth = static_cast<int>(value);
        } else {
            return false;
        }
    }
    return !result.ffs_root.empty() && !result.ready_file.empty() &&
            !result.manifest.empty();
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

ssize_t receive_packet(int fd, void * destination, size_t capacity) {
    for (;;) {
        const ssize_t count = read(fd, destination, capacity);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        return count;
    }
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
        connect(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0) {
        if (fd >= 0) {
            close(fd);
        }
        return -1;
    }
    return fd;
}

bool valid_hello_response(
        const ffn_split::hello_request & request,
        const ffn_split::resident_session_shard & shard,
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
            response.max_columns == shard.columns &&
            response.max_tokens == request.max_tokens &&
            ffn_split::same_artifact_sha256(
                    response.artifact_sha256, request.artifact_sha256) &&
            response.layer_mask == shard.layer_mask &&
            response.layer_count == static_cast<uint32_t>(
                    __builtin_popcountll(response.layer_mask)) &&
            response.column_quantum != 0 &&
            response.offset + response.max_columns == response.n_ff;
}

bool same_target_identity(
        const ffn_split::hello_response & left,
        const ffn_split::hello_response & right) {
    return left.flags == right.flags && left.n_embd == right.n_embd &&
            left.n_ff == right.n_ff && left.offset == right.offset &&
            left.max_columns == right.max_columns &&
            left.weight_type == right.weight_type &&
            left.column_quantum == right.column_quantum &&
            left.max_tokens == right.max_tokens &&
            left.alternate_columns_32 == right.alternate_columns_32 &&
            ffn_split::same_artifact_sha256(
                    left.artifact_sha256, right.artifact_sha256);
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

bool same_shard_identity(
        const ffn_split::resident_session_shard & left,
        const ffn_split::resident_session_shard & right) {
    return left.session_id == right.session_id &&
            left.artifact_sha256 == right.artifact_sha256 &&
            left.layer_mask == right.layer_mask && left.columns == right.columns &&
            left.endpoint_sha256 == right.endpoint_sha256 &&
            left.resident_geometry_sha256 == right.resident_geometry_sha256 &&
            left.operator_plan_sha256 == right.operator_plan_sha256 &&
            left.session_generation == right.session_generation;
}

size_t proof_index_for(
        const ffn_split::resident_session_shard & shard,
        std::vector<session_proof_totals> & proofs) {
    for (size_t index = 0; index < proofs.size(); ++index) {
        if (same_shard_identity(proofs[index].shard, shard)) {
            return index;
        }
    }
    proofs.push_back({shard, {}});
    return proofs.size() - 1;
}

bool select_targets(
        const std::vector<ffn_split::resident_session_shard> & shards,
        const ffn_split::hello_request & hello,
        target_selection & selected,
        std::vector<session_proof_totals> & proofs) {
    uint64_t covered_layers = 0;
    uint64_t aggregate_hash = UINT64_C(14695981039346656037);
    for (size_t index = 0; index < shards.size(); ++index) {
        const auto & shard = shards[index];
        uint8_t shard_artifact[32] = {};
        if (!ffn_split::parse_artifact_sha256(
                    shard.artifact_sha256, shard_artifact) ||
            !ffn_split::same_artifact_sha256(
                    shard_artifact, hello.artifact_sha256) ||
            (shard.layer_mask & hello.layer_mask) == 0) {
            continue;
        }
        const int fd = connect_target(shard.port);
        if (fd < 0) {
            close_selection(selected);
            return false;
        }
        ffn_split::hello_response response = {};
        if (!send_exact(fd, &hello, sizeof(hello)) ||
            !receive_exact(fd, &response, sizeof(response)) ||
            !valid_hello_response(hello, shard, response) ||
            (!selected.targets.empty() &&
             !same_target_identity(selected.response, response))) {
            close(fd);
            close_selection(selected);
            return false;
        }
        if (selected.targets.empty()) {
            selected.response = response;
        }
        const uint64_t identity[] = { response.layer_mask, response.weight_hash };
        aggregate_hash = ffn_split::hash64_update(
                aggregate_hash, identity, sizeof(identity));
        covered_layers |= response.layer_mask;
        selected.targets.push_back({fd, proof_index_for(shard, proofs), response});
    }
    if (covered_layers != hello.layer_mask) {
        close_selection(selected);
        return false;
    }
    selected.response.layer_mask = covered_layers;
    selected.response.layer_count = static_cast<uint32_t>(
            __builtin_popcountll(covered_layers));
    selected.response.weight_hash = aggregate_hash;
    return true;
}

target_connection * target_for_layer(target_selection & selected, int layer) {
    const uint64_t bit = UINT64_C(1) << layer;
    for (target_connection & target : selected.targets) {
        if ((target.response.layer_mask & bit) != 0) {
            return &target;
        }
    }
    return nullptr;
}

ffn_split::hello_response rejected_hello(const ffn_split::hello_request & request) {
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
    memcpy(
            response.artifact_sha256, request.artifact_sha256,
            sizeof(response.artifact_sha256));
    return response;
}

bool valid_execute_request(
        const ffn_split::hello_response & hello,
        const ffn_split::execute_request & request) {
    const size_t element_size = hello.flags & ffn_split::flag_f16_io ?
            sizeof(uint16_t) : sizeof(float);
    const uint64_t expected_elements =
            static_cast<uint64_t>(hello.n_embd) * request.tokens;
    return request.magic == ffn_split::protocol_magic &&
            request.version == ffn_split::protocol_version &&
            request.message == static_cast<uint16_t>(
                    ffn_split::message_type::execute_request) &&
            request.request_id != 0 && request.layer >= 0 && request.layer < 64 &&
            (hello.layer_mask & (UINT64_C(1) << request.layer)) != 0 &&
            request.tokens > 0 && request.tokens <= hello.max_tokens &&
            request.elements == expected_elements &&
            request.payload_bytes == expected_elements * element_size &&
            request.columns > 0 && request.columns <= hello.max_columns &&
            (request.columns == hello.max_columns ||
             request.columns ==
                     static_cast<uint32_t>(hello.alternate_columns_32) * 32 ||
             request.columns % hello.column_quantum == 0);
}

bool valid_execute_response(
        const ffn_split::execute_request & request,
        const ffn_split::execute_response & response,
        const uint8_t * payload, size_t payload_bytes) {
    return response.magic == ffn_split::protocol_magic &&
            response.version == ffn_split::protocol_version &&
            response.message == static_cast<uint16_t>(
                    ffn_split::message_type::execute_response) &&
            response.status == 0 && response.request_id == request.request_id &&
            response.layer == request.layer &&
            response.elements == request.elements &&
            response.payload_bytes == request.payload_bytes &&
            response.columns == request.columns &&
            response.tokens == request.tokens &&
            response.payload_hash == ffn_split::hash_bytes(payload, payload_bytes);
}

void print_terminal(
        const std::vector<session_proof_totals> & proofs,
        long sessions, long requests, int queue_depth,
        long reset_recoveries, int status) {
    fprintf(stderr,
            "MULTIPHONEFFN {\"transport\":\"session-router\","
            "\"requests\":%ld,\"queue_depth\":%d,"
            "\"maximum_pending_outputs\":0,"
            "\"phone_payload_copies\":%ld,"
            "\"d2h_completions\":%ld,\"d2h_queue_us\":0,"
            "\"d2h_queue_max_us\":0,\"recoveries\":%ld,"
            "\"status\":%d,\"host_sessions\":%ld,\"shards\":[",
            requests, queue_depth, requests * 4, requests,
            reset_recoveries, status, sessions);
    for (size_t index = 0; index < proofs.size(); ++index) {
        if (index != 0) {
            fputc(',', stderr);
        }
        const auto & shard = proofs[index].shard;
        const auto & row = proofs[index].totals;
        fprintf(stderr,
                "{\"session_id\":\"%s\",\"artifact_sha256\":\"%s\","
                "\"endpoint_sha256\":\"%s\","
                "\"resident_geometry_sha256\":\"%s\","
                "\"operator_plan_sha256\":\"%s\","
                "\"session_generation\":%llu,"
                "\"layer_mask\":%llu,\"calls\":%ld,\"rows\":%llu,"
                "\"h2d_bytes\":%llu,\"d2h_bytes\":%llu,"
                "\"h2d_us\":0,\"h2d_active_us\":0,"
                "\"idle_receive_wait_us\":%llu,"
                "\"h2d_timing_scope\":\"host-completion-required-v2\","
                "\"d2h_us\":%llu,"
                "\"compute_us\":%llu,\"rpc_us\":%llu}",
                shard.session_id.c_str(), shard.artifact_sha256.c_str(),
                shard.endpoint_sha256.c_str(),
                shard.resident_geometry_sha256.c_str(),
                shard.operator_plan_sha256.c_str(),
                static_cast<unsigned long long>(shard.session_generation),
                static_cast<unsigned long long>(shard.layer_mask), row.calls,
                static_cast<unsigned long long>(row.rows),
                static_cast<unsigned long long>(row.h2d_bytes),
                static_cast<unsigned long long>(row.d2h_bytes),
                static_cast<unsigned long long>(row.receive_block_us),
                static_cast<unsigned long long>(row.d2h_us),
                static_cast<unsigned long long>(row.compute_us),
                static_cast<unsigned long long>(row.rpc_us));
    }
    fprintf(stderr, "]}\n");
    fflush(stderr);
}

} // namespace

int main(int argc, char ** argv) {
    config cfg;
    if (!parse_config(argc, argv, cfg)) {
        fprintf(stderr,
                "usage: %s --ffs-root PATH --ready-file PATH --manifest PATH "
                "[--queue-depth N] [--max-sessions N]\n",
                argv[0]);
        return 2;
    }
    std::vector<ffn_split::resident_session_shard> shards;
    std::string error;
    if (!ffn_split::load_resident_session_manifest(cfg.manifest, shards, error)) {
        fprintf(stderr, "[resident-router] %s\n", error.c_str());
        return 2;
    }
    signal(SIGPIPE, SIG_IGN);

    const int ep0 = ffn_split::functionfs_open_endpoint(cfg.ffs_root, "ep0");
    if (ep0 < 0 ||
        !ffn_split::functionfs_write_exact(
                ep0, &ffn_split::functionfs_descriptors,
                sizeof(ffn_split::functionfs_descriptors)) ||
        !ffn_split::functionfs_write_exact(
                ep0, &ffn_split::functionfs_strings,
                sizeof(ffn_split::functionfs_strings))) {
        fprintf(stderr, "[resident-router] descriptor setup failed: %s\n",
                strerror(errno));
        return 1;
    }
    const int device_to_host = ffn_split::functionfs_open_endpoint(
            cfg.ffs_root, "ep1");
    const int host_to_device = ffn_split::functionfs_open_endpoint(
            cfg.ffs_root, "ep2");
    if (device_to_host < 0 || host_to_device < 0 ||
        !ffn_split::functionfs_touch(cfg.ready_file, "resident_router_ready\n")) {
        fprintf(stderr, "[resident-router] endpoint setup failed: %s\n",
                strerror(errno));
        return 1;
    }

    ffn_split::functionfs_event_monitor event_monitor(ep0);
    uint64_t generation = 0;
    if (!event_monitor.wait_enabled(
                generation, ffn_split::functionfs_enable_timeout_ms)) {
        fprintf(stderr, "[resident-router] FunctionFS enable failed: %s\n",
                strerror(errno));
        return 1;
    }
    fprintf(stderr,
            "[resident-router] ready targets=%zu generation=%llu queue_depth=%d\n",
            shards.size(), static_cast<unsigned long long>(generation),
            cfg.queue_depth);
    fflush(stderr);

    std::vector<session_proof_totals> proofs;
    for (const auto & shard : shards) {
        proof_index_for(shard, proofs);
    }
    uint64_t layout_generation = 1;
    long host_sessions = 0;
    long requests = 0;
    long reset_recoveries = 0;
    bool terminate_requested = false;
    bool wait_for_reenable = false;
    uint64_t previous_generation = generation;
    int run_status = 0;
    while (run_status == 0) {
        if (wait_for_reenable) {
            if (!event_monitor.wait_reenabled(
                        previous_generation, generation,
                        ffn_split::functionfs_enable_timeout_ms)) {
                run_status = 1;
                break;
            }
            wait_for_reenable = false;
        }
        ffn_split::hello_request hello = {};
        if (!ffn_split::functionfs_read_exact(
                    host_to_device, &hello, sizeof(hello))) {
            previous_generation = generation;
            wait_for_reenable = true;
            ++reset_recoveries;
            continue;
        }
        // Residency can change while waiting for the next HELLO.
        std::vector<ffn_split::resident_session_shard> observed_shards;
        if (!ffn_split::load_resident_session_manifest(
                    cfg.manifest, observed_shards, error)) {
            fprintf(stderr, "[resident-router] manifest reload failed: %s\n",
                    error.c_str());
            run_status = 1;
            break;
        }
        const bool layout_changed = observed_shards.size() != shards.size() ||
                !std::equal(
                        observed_shards.begin(), observed_shards.end(), shards.begin(),
                        same_shard_identity);
        if (layout_changed) {
            shards = std::move(observed_shards);
            ++layout_generation;
            for (const auto & shard : shards) {
                proof_index_for(shard, proofs);
            }
            fprintf(stderr,
                    "RESIDENTLAYOUT {\"status\":\"READY\","
                    "\"generation\":%llu,\"session_count\":%zu}\n",
                    static_cast<unsigned long long>(layout_generation),
                    shards.size());
            fflush(stderr);
        }
        target_selection selected;
        if (!select_targets(shards, hello, selected, proofs)) {
            const auto response = rejected_hello(hello);
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
            ++reset_recoveries;
            continue;
        }
        const size_t element_size = selected.response.flags &
                ffn_split::flag_f16_io ? sizeof(uint16_t) : sizeof(float);
        const size_t max_payload = static_cast<size_t>(selected.response.n_embd) *
                selected.response.max_tokens * element_size;
        std::vector<uint8_t> wire(ffn_split::dmabuf_payload_offset + max_payload);
        fprintf(stderr,
                "RESIDENTARM {\"status\":\"WARM\","
                "\"target_count\":%zu,\"layer_mask\":\"%016llx\"}\n",
                selected.targets.size(),
                static_cast<unsigned long long>(selected.response.layer_mask));
        fflush(stderr);

        bool reset = false;
        bool complete_session = false;
        while (!reset && !complete_session && run_status == 0) {
            const uint64_t receive_started = now_us();
            const ssize_t received = receive_packet(
                    host_to_device, wire.data(), wire.size());
            const uint64_t receive_block_us = now_us() - receive_started;
            if (received <= 0) {
                reset = true;
                break;
            }
            ffn_split::execute_request request = {};
            if (static_cast<size_t>(received) < sizeof(request)) {
                run_status = 1;
                break;
            }
            memcpy(&request, wire.data(), sizeof(request));
            const bool shutdown = request.magic == ffn_split::protocol_magic &&
                    request.version == ffn_split::protocol_version &&
                    request.message == static_cast<uint16_t>(
                            ffn_split::message_type::execute_request) &&
                    request.request_id == 0;
            if (shutdown) {
                if (static_cast<size_t>(received) != sizeof(request)) {
                    run_status = 1;
                    break;
                }
                terminate_requested = request.layer == -1;
                complete_session = true;
                break;
            }
            const size_t wire_bytes =
                    ffn_split::dmabuf_payload_offset + request.payload_bytes;
            if (!valid_execute_request(selected.response, request) ||
                request.payload_bytes > max_payload ||
                static_cast<size_t>(received) != wire_bytes) {
                run_status = 1;
                break;
            }
            target_connection * target = target_for_layer(selected, request.layer);
            if (target == nullptr) {
                run_status = 1;
                break;
            }
            session_totals & row = proofs[target->proof_index].totals;
            row.receive_block_us += receive_block_us;
            const uint8_t * input =
                    wire.data() + ffn_split::dmabuf_payload_offset;
            const uint64_t input_check_started = now_us();
            if (request.payload_hash != ffn_split::hash_bytes(
                        input, request.payload_bytes)) {
                run_status = 1;
                break;
            }
            const uint64_t rpc_started = now_us();
            if (!send_exact(target->fd, &request, sizeof(request)) ||
                !send_exact(target->fd, input, request.payload_bytes)) {
                run_status = 1;
                break;
            }
            const uint64_t worker_sent = now_us();
            ffn_split::execute_response response = {};
            if (!receive_exact(target->fd, &response, sizeof(response)) ||
                !receive_exact(
                        target->fd,
                        wire.data() + ffn_split::dmabuf_payload_offset,
                        request.payload_bytes)) {
                run_status = 1;
                break;
            }
            const uint64_t worker_received = now_us();
            if (!valid_execute_response(
                        request, response,
                        wire.data() + ffn_split::dmabuf_payload_offset,
                        request.payload_bytes)) {
                run_status = 1;
                break;
            }
            const uint64_t output_checked = now_us();
            row.rpc_us += output_checked - rpc_started;
            memset(wire.data(), 0, ffn_split::dmabuf_payload_offset);
            memcpy(wire.data(), &response, sizeof(response));
            const uint64_t d2h_started = now_us();
            if (!ffn_split::functionfs_write_exact(
                        device_to_host, wire.data(), wire_bytes)) {
                reset = true;
                break;
            }
            const uint64_t d2h_finished = now_us();
            row.d2h_us += d2h_finished - d2h_started;
            ++row.calls;
            row.rows += request.tokens;
            row.h2d_bytes += request.payload_bytes;
            row.d2h_bytes += request.payload_bytes;
            row.compute_us += response.compute_us;
            ++requests;
            print_call(
                    proofs[target->proof_index].shard, row.calls,
                    cfg.call_log_period);
            if (request.tokens > 1 || row.calls == 1 ||
                row.calls % cfg.call_log_period == 0) {
                fprintf(stderr,
                        "RESIDENTTIMING {\"schema\":\"s42-resident-router-timing-v1\","
                        "\"request_id\":%u,\"layer\":%d,\"tokens\":%u,"
                        "\"payload_bytes\":%u,\"input_check_us\":%llu,"
                        "\"worker_send_us\":%llu,\"worker_receive_us\":%llu,"
                        "\"worker_compute_us\":%llu,\"output_check_us\":%llu,"
                        "\"usb_write_us\":%llu,\"receive_including_idle_us\":%llu}\n",
                        request.request_id, request.layer, request.tokens,
                        request.payload_bytes,
                        static_cast<unsigned long long>(rpc_started - input_check_started),
                        static_cast<unsigned long long>(worker_sent - rpc_started),
                        static_cast<unsigned long long>(worker_received - worker_sent),
                        static_cast<unsigned long long>(response.compute_us),
                        static_cast<unsigned long long>(output_checked - worker_received),
                        static_cast<unsigned long long>(d2h_finished - d2h_started),
                        static_cast<unsigned long long>(receive_block_us));
            }
        }
        close_selection(selected);
        if (reset) {
            ++reset_recoveries;
            previous_generation = generation;
            wait_for_reenable = true;
            continue;
        }
        ++host_sessions;
        if (terminate_requested ||
            (cfg.max_sessions > 0 && host_sessions >= cfg.max_sessions)) {
            break;
        }
    }

    event_monitor.stop();
    close(host_to_device);
    close(device_to_host);
    close(ep0);
    print_terminal(
            proofs, host_sessions, requests, cfg.queue_depth,
            reset_recoveries, run_status);
    return run_status;
}
