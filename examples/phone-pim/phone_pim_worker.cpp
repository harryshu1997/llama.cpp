#include "phone_pim_ffn.h"
#include "phone_pim_protocol.h"
#include "phone_pim_socket.h"
#include "phone_pim_store.h"

#include <cerrno>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <sys/random.h>
#include <vector>

namespace pp = phone_pim;

namespace {

struct Config {
    std::string model_path;
    std::string store_dir;
    std::string backend = "HTP0";
    std::string bind_host = "127.0.0.1";
    uint16_t port = 9090;
    uint64_t route_epoch = 1;
    uint64_t generation = 1;
    uint64_t max_payload = pp::k_default_max_payload;
    uint64_t max_store_bytes = 16ULL * 1024 * 1024 * 1024;
    uint64_t max_model_bytes = 16ULL * 1024 * 1024 * 1024;
    uint64_t min_free_bytes = 256ULL * 1024 * 1024;
    int timeout_ms = 600000;
    bool profile_recv = false; // measurement-only; splits stage receive_frame recv vs. frame-SHA
};

// Stage-scoped receive accounting: only STAGE_* frames are summed here, so HELLO/PREPARE/
// EXECUTE/RELEASE/SHUTDOWN and idle oracle gaps are excluded. store_prefix_hash_us is the
// data-only durable-prefix advance, captured from the commit metrics (never on the wire).
struct StageRecvProfile {
    uint64_t recv_header_us = 0;   // socket header recv (includes stop-and-wait idle)
    uint64_t recv_payload_us = 0;  // socket payload recv
    uint64_t frame_sha_us = 0;     // outer-frame SHA over envelope + data
    uint64_t frames = 0;
    uint64_t payload_bytes = 0;
    uint64_t store_prefix_hash_us = 0;
};

bool parse_u64(const char * text, uint64_t & value) {
    if (text == nullptr || *text == '\0' || *text == '-') {
        return false;
    }
    char * end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        return false;
    }
    value = static_cast<uint64_t>(parsed);
    return true;
}

bool seed_session_epoch(uint64_t & value) {
    uint64_t random = 0;
    size_t received = 0;
    while (received < sizeof(random)) {
        const ssize_t count = getrandom(
                reinterpret_cast<uint8_t *>(&random) + received, sizeof(random) - received, 0);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            return false;
        }
        received += static_cast<size_t>(count);
    }
    value = random & (std::numeric_limits<uint64_t>::max() >> 1);
    if (value == 0) {
        value = 1;
    }
    return true;
}

bool parse_args(int argc, char ** argv, Config & config) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : nullptr; };
        if (arg == "--model") {
            const char * value = next();
            if (value == nullptr) return false;
            config.model_path = value;
        } else if (arg == "--store-dir") {
            const char * value = next();
            if (value == nullptr || *value == '\0') return false;
            config.store_dir = value;
        } else if (arg == "--backend") {
            const char * value = next();
            if (value == nullptr) return false;
            config.backend = value;
        } else if (arg == "--bind") {
            const char * value = next();
            if (value == nullptr || std::strcmp(value, "127.0.0.1") != 0) return false;
            config.bind_host = value;
        } else if (arg == "--port") {
            uint64_t value = 0;
            if (!parse_u64(next(), value) || value == 0 || value > 65535) return false;
            config.port = static_cast<uint16_t>(value);
        } else if (arg == "--route-epoch") {
            if (!parse_u64(next(), config.route_epoch) || config.route_epoch == 0) return false;
        } else if (arg == "--generation") {
            if (!parse_u64(next(), config.generation) || config.generation == 0 ||
                config.generation == std::numeric_limits<uint64_t>::max()) return false;
        } else if (arg == "--max-payload-mib") {
            uint64_t value = 0;
            if (!parse_u64(next(), value) || value == 0 || value > 64) return false;
            config.max_payload = value * 1024 * 1024;
        } else if (arg == "--max-store-mib") {
            uint64_t value = 0;
            if (!parse_u64(next(), value) || value == 0 || value > 65536) return false;
            config.max_store_bytes = value * 1024 * 1024;
        } else if (arg == "--max-model-mib") {
            uint64_t value = 0;
            if (!parse_u64(next(), value) || value == 0 || value > 65536) return false;
            config.max_model_bytes = value * 1024 * 1024;
        } else if (arg == "--min-free-mib") {
            uint64_t value = 0;
            if (!parse_u64(next(), value) || value > 65536) return false;
            config.min_free_bytes = value * 1024 * 1024;
        } else if (arg == "--timeout-ms") {
            uint64_t value = 0;
            if (!parse_u64(next(), value) || value == 0 || value > std::numeric_limits<int>::max()) return false;
            config.timeout_ms = static_cast<int>(value);
        } else if (arg == "--profile-recv") {
            config.profile_recv = true;
        } else {
            return false;
        }
    }
    return (!config.model_path.empty() || !config.store_dir.empty()) && !config.backend.empty() &&
           config.max_model_bytes <= config.max_store_bytes;
}

bool canonical_tensor_prefix(const std::string & prefix) {
    if (prefix.size() < 5 || prefix.compare(0, 4, "blk.") != 0 || prefix[4] == '0') {
        return false;
    }
    for (size_t i = 4; i < prefix.size(); ++i) {
        if (prefix[i] < '0' || prefix[i] > '9') {
            return false;
        }
    }
    uint64_t layer = 0;
    return parse_u64(prefix.c_str() + 4, layer) && layer <= 100000;
}

bool send_response(int fd, const pp::Frame & request, pp::Opcode opcode, std::vector<uint8_t> payload) {
    pp::Header header;
    header.opcode = opcode;
    header.flags = pp::flag_response;
    header.request_id = request.header.request_id;
    header.command_seq = request.header.command_seq;
    header.session_epoch = request.header.session_epoch;
    header.route_epoch = request.header.route_epoch;
    header.residency_generation = request.header.residency_generation;
    std::string error;
    if (!pp::send_frame(fd, header, payload, error)) {
        std::fprintf(stderr, "[phone-pim] response send failed: %s\n", error.c_str());
        return false;
    }
    return true;
}

bool send_error(int fd, const pp::Frame & request, pp::ErrorCode code, const std::string & message) {
    pp::Header header;
    header.opcode = pp::Opcode::error;
    header.flags = pp::flag_response | pp::flag_error;
    header.request_id = request.header.request_id;
    header.command_seq = request.header.command_seq;
    header.session_epoch = request.header.session_epoch;
    header.route_epoch = request.header.route_epoch;
    header.residency_generation = request.header.residency_generation;
    std::string error;
    if (!pp::send_frame(fd, header, pp::make_error_payload(code, message), error)) {
        std::fprintf(stderr, "[phone-pim] error response send failed: %s\n", error.c_str());
        return false;
    }
    return true;
}

struct ResidentState {
    pp::FfnIsland island;
    pp::FfnPrepareMetrics metrics;
    uint64_t island_id = 0;
    std::string prefix;
    uint32_t token_count = 0;
    uint64_t model_bytes = 0;
    std::array<uint8_t, 32> model_sha256 = {};
    uint64_t verify_us = 0;
    pp::ModelSource model_source = pp::ModelSource::prestaged;
    std::vector<float> input;
    std::vector<float> output;

    void clear() {
        island.reset();
        metrics = {};
        island_id = 0;
        prefix.clear();
        token_count = 0;
        model_bytes = 0;
        model_sha256 = {};
        verify_us = 0;
        model_source = pp::ModelSource::prestaged;
        input.clear();
        input.shrink_to_fit();
        output.clear();
        output.shrink_to_fit();
    }
};

std::vector<uint8_t> prepared_payload(const ResidentState & state) {
    pp::Writer writer;
    writer.u64(state.island_id);
    writer.u64(state.island.n_embd());
    writer.u64(state.island.n_ff());
    writer.u32(state.token_count);
    writer.u64(state.model_bytes);
    writer.bytes(state.model_sha256.data(), state.model_sha256.size());
    writer.u32(static_cast<uint32_t>(state.model_source));
    writer.u64(state.verify_us);
    writer.u64(state.metrics.weight_bytes);
    writer.u64(state.metrics.resident_buffer_bytes);
    writer.u64(state.metrics.load_us);
    writer.u64(state.metrics.upload_us);
    writer.u64(state.metrics.warmup_us);
    writer.string(state.island.backend_description());
    return writer.take();
}

bool status_payload(
        const ResidentState & state,
        const pp::ArtifactStore & store,
        uint64_t generation,
        std::vector<uint8_t> & payload,
        std::string & error) {
    pp::Writer writer;
    writer.u32(state.island.ready() ? 1 : 0);
    writer.u64(state.island_id);
    writer.u64(generation);
    writer.u64(state.island.n_embd());
    writer.u64(state.island.n_ff());
    writer.u32(state.token_count);
    writer.string(state.prefix);
    pp::StageProgress progress;
    const bool active = store.active_progress(progress, error);
    if (!active && !error.empty()) {
        return false;
    }
    writer.u32(store.enabled() ? 1 : 0);
    writer.u32(active ? 1 : 0);
    writer.u64(active ? progress.ticket_id : 0);
    writer.u64(active ? progress.verified_bytes : 0);
    writer.u32(active ? progress.next_chunk : 0);
    writer.bytes(progress.manifest_sha256.data(), progress.manifest_sha256.size());
    payload = writer.take();
    return true;
}

bool decode_stage_spec(pp::Reader & reader, uint64_t & ticket_id, pp::StageObjectSpec & spec) {
    uint32_t chunk_count = 0;
    if (!reader.u64(ticket_id) || !reader.u64(spec.bytes) ||
        !reader.bytes(spec.sha256.data(), spec.sha256.size()) || !reader.u32(spec.chunk_bytes) ||
        !reader.u32(chunk_count) || chunk_count == 0 || chunk_count > pp::k_max_stage_chunks ||
        !reader.bytes(spec.manifest_sha256.data(), spec.manifest_sha256.size())) {
        return false;
    }
    spec.chunks.resize(chunk_count);
    for (uint32_t i = 0; i < chunk_count; ++i) {
        pp::StageChunkSpec & chunk = spec.chunks[i];
        if (!reader.u32(chunk.index) || !reader.u64(chunk.offset) || !reader.u32(chunk.bytes) ||
            !reader.bytes(chunk.sha256.data(), chunk.sha256.size())) {
            return false;
        }
    }
    return reader.done();
}

std::vector<uint8_t> stage_progress_payload(const pp::StageProgress & progress) {
    pp::Writer writer;
    writer.u32(static_cast<uint32_t>(progress.state));
    writer.u64(progress.ticket_id);
    writer.u64(progress.residency_generation);
    writer.u64(progress.verified_bytes);
    writer.u32(progress.next_chunk);
    writer.bytes(progress.prefix_sha256.data(), progress.prefix_sha256.size());
    writer.bytes(progress.manifest_sha256.data(), progress.manifest_sha256.size());
    writer.u64(progress.resume_scan_us);
    return writer.take();
}

std::vector<uint8_t> stage_commit_payload(
        const pp::StageProgress & progress,
        const pp::StageMetrics & metrics) {
    pp::Writer writer;
    const std::vector<uint8_t> base = stage_progress_payload(progress);
    writer.bytes(base.data(), base.size());
    writer.u64(metrics.chunk_hash_us);
    writer.u64(metrics.write_us);
    writer.u64(metrics.data_sync_us);
    writer.u64(metrics.full_verify_us);
    writer.u64(metrics.file_sync_us);
    writer.u64(metrics.publish_us);
    writer.u64(metrics.directory_sync_us);
    writer.u64(metrics.accepted_chunks);
    writer.u64(metrics.duplicate_chunks);
    return writer.take();
}

bool decode_float_vector(pp::Reader & reader, uint64_t count, std::vector<float> & values) {
    if (count > SIZE_MAX / sizeof(float) || reader.remaining() != count * sizeof(float)) {
        return false;
    }
    values.resize(static_cast<size_t>(count));
    for (size_t i = 0; i < values.size(); ++i) {
        uint32_t bits = 0;
        if (!reader.u32(bits)) {
            return false;
        }
        std::memcpy(values.data() + i, &bits, sizeof(bits));
    }
    return reader.done();
}

void encode_float_vector(pp::Writer & writer, const std::vector<float> & values) {
    for (float value : values) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        writer.u32(bits);
    }
}

bool serve_connection(
        int fd,
        const Config & config,
        ResidentState & state,
        pp::ArtifactStore & store,
        uint64_t session_epoch,
        uint64_t & current_generation,
        bool & shutdown_requested,
        StageRecvProfile & stage_recv) {
    uint64_t last_command_seq = 0;
    uint64_t last_request_id = 0;
    pp::Frame request;
    for (;;) {
        std::string error;
        const pp::ReceiveResult received = pp::receive_frame(fd, request, config.max_payload, error);
        if (received == pp::ReceiveResult::eof) {
            return true;
        }
        if (received != pp::ReceiveResult::ok) {
            std::fprintf(stderr, "[phone-pim] invalid frame: %s\n", error.c_str());
            return false;
        }
        if (config.profile_recv) {
            const pp::Opcode op = request.header.opcode;
            if (op == pp::Opcode::stage_begin || op == pp::Opcode::stage_chunk ||
                op == pp::Opcode::stage_commit || op == pp::Opcode::stage_abort) {
                pp::RecvProfile rp;
                pp::get_last_recv_profile(rp);
                stage_recv.recv_header_us += rp.header_us;
                stage_recv.recv_payload_us += rp.payload_us;
                stage_recv.frame_sha_us += rp.sha_us;
                stage_recv.payload_bytes += rp.payload_bytes;
                ++stage_recv.frames;
            }
        }

        const bool generation_required = request.header.opcode != pp::Opcode::hello &&
                                         request.header.opcode != pp::Opcode::ping &&
                                         request.header.opcode != pp::Opcode::status;
        if (!pp::validate_command(
                    request.header,
                    request.header.opcode == pp::Opcode::hello ? 0 : session_epoch,
                    config.route_epoch,
                    current_generation,
                    last_command_seq,
                    last_request_id,
                    generation_required,
                    error)) {
            const pp::ErrorCode code = error.find("sequence") != std::string::npos
                    ? pp::ErrorCode::duplicate_command
                    : (error.find("epoch") != std::string::npos ||
                       error.find("generation") != std::string::npos)
                            ? pp::ErrorCode::stale_epoch : pp::ErrorCode::protocol;
            if (!send_error(fd, request, code, error)) {
                return false;
            }
            continue;
        }
        last_command_seq = request.header.command_seq;
        last_request_id = request.header.request_id;

        if (request.header.opcode == pp::Opcode::hello) {
            if (!request.payload.empty()) {
                if (!send_error(fd, request, pp::ErrorCode::invalid_argument, "HELLO payload must be empty")) return false;
                continue;
            }
            pp::Writer writer;
            writer.u32(pp::k_version);
            writer.u64(session_epoch);
            writer.u64(config.max_payload);
            writer.string(config.backend);
            const uint64_t feature_bits = (store.enabled() ? 1ULL : 0ULL) |
                                          (!config.model_path.empty() ? 2ULL : 0ULL);
            const uint64_t max_chunk = config.max_payload > pp::k_stage_chunk_envelope_bytes
                    ? std::min<uint64_t>(
                            pp::k_max_stage_chunk_bytes,
                            config.max_payload - pp::k_stage_chunk_envelope_bytes)
                    : 0;
            writer.u64(feature_bits);
            writer.u64(config.max_model_bytes);
            writer.u64(max_chunk);
            writer.u32(pp::k_max_stage_chunks);
            writer.string(store.enabled()
                    ? "sequential_dynamic_gemma4_dense_ffn_v3"
                    : "prestaged_gemma4_dense_ffn_v3");
            if (!send_response(fd, request, pp::Opcode::hello, writer.take())) return false;
            continue;
        }

        if (request.header.opcode == pp::Opcode::stage_begin) {
            if (!store.enabled()) {
                if (!send_error(fd, request, pp::ErrorCode::unsupported, "dynamic store is disabled")) return false;
                continue;
            }
            pp::Reader reader(request.payload);
            uint64_t ticket_id = 0;
            pp::StageObjectSpec spec;
            if (!decode_stage_spec(reader, ticket_id, spec)) {
                if (!send_error(fd, request, pp::ErrorCode::invalid_argument, "invalid STAGE_BEGIN payload")) return false;
                continue;
            }
            pp::StageProgress progress;
            if (!store.begin(ticket_id, current_generation, spec, progress, error)) {
                if (!send_error(fd, request, pp::ErrorCode::storage, error)) return false;
                continue;
            }
            if (!send_response(fd, request, pp::Opcode::stage_begin, stage_progress_payload(progress))) return false;
            continue;
        }

        if (request.header.opcode == pp::Opcode::stage_chunk) {
            if (!store.enabled()) {
                if (!send_error(fd, request, pp::ErrorCode::unsupported, "dynamic store is disabled")) return false;
                continue;
            }
            pp::Reader reader(request.payload);
            uint64_t ticket_id = 0;
            uint32_t chunk_index = 0;
            uint64_t offset = 0;
            uint32_t chunk_bytes = 0;
            std::array<uint8_t, 32> chunk_sha256 = {};
            if (!reader.u64(ticket_id) || !reader.u32(chunk_index) || !reader.u64(offset) ||
                !reader.u32(chunk_bytes) || !reader.bytes(chunk_sha256.data(), chunk_sha256.size()) ||
                chunk_bytes == 0 || chunk_bytes > pp::k_max_stage_chunk_bytes ||
                reader.remaining() != chunk_bytes) {
                if (!send_error(fd, request, pp::ErrorCode::invalid_argument, "invalid STAGE_CHUNK payload")) return false;
                continue;
            }
            pp::StageProgress progress;
            pp::StageMetrics metrics;
            if (!store.put_chunk(
                        ticket_id, current_generation, chunk_index, offset, reader.current(), chunk_bytes,
                        chunk_sha256, progress, metrics, error)) {
                const pp::ErrorCode code = error.find("SHA-256") != std::string::npos
                        ? pp::ErrorCode::hash_mismatch : pp::ErrorCode::storage;
                if (!send_error(fd, request, code, error)) return false;
                continue;
            }
            if (!send_response(fd, request, pp::Opcode::stage_chunk, stage_progress_payload(progress))) return false;
            continue;
        }

        if (request.header.opcode == pp::Opcode::stage_commit) {
            if (!store.enabled()) {
                if (!send_error(fd, request, pp::ErrorCode::unsupported, "dynamic store is disabled")) return false;
                continue;
            }
            pp::Reader reader(request.payload);
            uint64_t ticket_id = 0;
            std::array<uint8_t, 32> manifest_sha256 = {};
            if (!reader.u64(ticket_id) || !reader.bytes(manifest_sha256.data(), manifest_sha256.size()) ||
                !reader.done()) {
                if (!send_error(fd, request, pp::ErrorCode::invalid_argument, "invalid STAGE_COMMIT payload")) return false;
                continue;
            }
            pp::StageProgress progress;
            pp::StageMetrics metrics;
            if (!store.commit(
                        ticket_id, current_generation, manifest_sha256, progress, metrics, error)) {
                const pp::ErrorCode code = error.find("SHA-256") != std::string::npos
                        ? pp::ErrorCode::hash_mismatch : pp::ErrorCode::storage;
                if (!send_error(fd, request, code, error)) return false;
                continue;
            }
            if (config.profile_recv) {
                stage_recv.store_prefix_hash_us = metrics.prefix_hash_us;
            }
            if (!send_response(fd, request, pp::Opcode::stage_commit,
                               stage_commit_payload(progress, metrics))) return false;
            continue;
        }

        if (request.header.opcode == pp::Opcode::stage_abort) {
            if (!store.enabled()) {
                if (!send_error(fd, request, pp::ErrorCode::unsupported, "dynamic store is disabled")) return false;
                continue;
            }
            pp::Reader reader(request.payload);
            uint64_t ticket_id = 0;
            uint32_t quarantine = 0;
            if (!reader.u64(ticket_id) || !reader.u32(quarantine) || !reader.done() || quarantine > 1) {
                if (!send_error(fd, request, pp::ErrorCode::invalid_argument, "invalid STAGE_ABORT payload")) return false;
                continue;
            }
            if (!store.abort(ticket_id, current_generation, quarantine != 0, error)) {
                if (!send_error(fd, request, pp::ErrorCode::storage, error)) return false;
                continue;
            }
            if (!send_response(fd, request, pp::Opcode::stage_abort, {})) return false;
            continue;
        }

        if (request.header.opcode == pp::Opcode::prepare) {
            pp::Reader reader(request.payload);
            uint64_t island_id = 0;
            uint32_t token_count = 0;
            uint32_t source_value = 0;
            std::string prefix;
            uint64_t model_bytes = 0;
            std::array<uint8_t, 32> model_sha256 = {};
            if (!reader.u64(island_id) || !reader.u32(token_count) || !reader.u32(source_value) ||
                (source_value != static_cast<uint32_t>(pp::ModelSource::prestaged) &&
                 source_value != static_cast<uint32_t>(pp::ModelSource::published_store)) ||
                !reader.string(prefix, 256) ||
                !reader.u64(model_bytes) || !reader.bytes(model_sha256.data(), model_sha256.size()) ||
                !reader.done() || island_id == 0 || token_count == 0 || token_count > 1024 ||
                !canonical_tensor_prefix(prefix) || model_bytes == 0) {
                if (!send_error(fd, request, pp::ErrorCode::invalid_argument, "invalid PREPARE payload")) return false;
                continue;
            }
            const pp::ModelSource source = static_cast<pp::ModelSource>(source_value);
            if (state.island.ready()) {
                if (state.island_id != island_id || state.token_count != token_count || state.prefix != prefix ||
                    state.model_bytes != model_bytes || state.model_sha256 != model_sha256 ||
                    state.model_source != source) {
                    if (!send_error(fd, request, pp::ErrorCode::invalid_state, "a different island is already resident")) return false;
                    continue;
                }
                if (!send_response(fd, request, pp::Opcode::prepare, prepared_payload(state))) return false;
                continue;
            }
            pp::FfnSpec spec;
            pp::Fd published_model;
            if (source == pp::ModelSource::published_store) {
                std::string lookup_error;
                if (!store.enabled() ||
                    !store.lookup(model_bytes, model_sha256, published_model, lookup_error)) {
                    if (!send_error(fd, request, pp::ErrorCode::invalid_state,
                                    "requested dynamic object is not durably published: " + lookup_error)) {
                        return false;
                    }
                    continue;
                }
                spec.model_fd = published_model.get();
                spec.model_fd_verified = true;
            } else if (!config.model_path.empty()) {
                spec.model_path = config.model_path;
            } else {
                if (!send_error(fd, request, pp::ErrorCode::invalid_state,
                                "pre-staged model source is unavailable")) return false;
                continue;
            }
            spec.tensor_prefix = prefix;
            spec.backend_name = config.backend;
            spec.token_count = token_count;
            spec.expected_model_bytes = model_bytes;
            spec.expected_model_sha256 = model_sha256;
            pp::FfnPrepareMetrics metrics;
            if (!state.island.prepare(spec, metrics, error)) {
                state.clear();
                store.detach_active();
                if (!send_error(fd, request, pp::ErrorCode::backend, error)) return false;
                continue;
            }
            state.island_id = island_id;
            state.prefix = prefix;
            state.token_count = token_count;
            state.model_bytes = model_bytes;
            state.model_sha256 = model_sha256;
            state.model_source = source;
            state.verify_us = metrics.verify_us;
            state.metrics = metrics;
            state.input.reserve(static_cast<size_t>(state.island.input_elements()));
            state.output.reserve(static_cast<size_t>(state.island.output_elements()));
            std::fprintf(stderr,
                    "[phone-pim] READY island=%llu prefix=%s M=%u backend=%s resident=%.1f MiB\n",
                    static_cast<unsigned long long>(state.island_id), state.prefix.c_str(), state.token_count,
                    config.backend.c_str(), state.metrics.resident_buffer_bytes / 1048576.0);
            if (!send_response(fd, request, pp::Opcode::prepare, prepared_payload(state))) return false;
            continue;
        }

        if (request.header.opcode == pp::Opcode::execute) {
            if (!state.island.ready()) {
                if (!send_error(fd, request, pp::ErrorCode::invalid_state, "no READY island")) return false;
                continue;
            }
            pp::Reader reader(request.payload);
            uint64_t island_id = 0;
            uint64_t element_count = 0;
            std::vector<float> & input = state.input;
            if (!reader.u64(island_id) || !reader.u64(element_count) || island_id != state.island_id ||
                element_count != state.island.input_elements() || !decode_float_vector(reader, element_count, input) ||
                !pp::all_finite(input.data(), input.size())) {
                if (!send_error(fd, request, pp::ErrorCode::invalid_argument, "invalid EXECUTE payload")) return false;
                continue;
            }
            std::vector<float> & output = state.output;
            pp::FfnExecuteMetrics metrics;
            if (!state.island.execute(input, output, metrics, error)) {
                state.clear();
                store.detach_active();
                if (current_generation == std::numeric_limits<uint64_t>::max()) {
                    std::fprintf(stderr, "[phone-pim] backend failed and generation is exhausted\n");
                    shutdown_requested = true;
                    return false;
                }
                ++current_generation;
                if (!send_error(fd, request, pp::ErrorCode::backend,
                                error + "; residency quarantined")) return false;
                if (current_generation == std::numeric_limits<uint64_t>::max()) {
                    shutdown_requested = true;
                    return false;
                }
                continue;
            }
            pp::Writer writer;
            writer.u64(state.island_id);
            writer.u64(output.size());
            writer.u64(metrics.input_set_us);
            writer.u64(metrics.compute_us);
            writer.u64(metrics.output_get_us);
            encode_float_vector(writer, output);
            if (!send_response(fd, request, pp::Opcode::execute, writer.take())) return false;
            continue;
        }

        if (request.header.opcode == pp::Opcode::status) {
            if (!request.payload.empty()) {
                if (!send_error(fd, request, pp::ErrorCode::invalid_argument, "STATUS payload must be empty")) return false;
                continue;
            }
            std::vector<uint8_t> payload;
            if (!status_payload(state, store, current_generation, payload, error)) {
                if (!send_error(fd, request, pp::ErrorCode::storage, error)) return false;
                continue;
            }
            if (!send_response(fd, request, pp::Opcode::status, std::move(payload))) return false;
            continue;
        }

        if (request.header.opcode == pp::Opcode::release) {
            pp::Reader reader(request.payload);
            uint64_t island_id = 0;
            if (!state.island.ready() || !reader.u64(island_id) || !reader.done() || island_id != state.island_id) {
                if (!send_error(fd, request, pp::ErrorCode::invalid_argument, "invalid RELEASE payload")) return false;
                continue;
            }
            if (current_generation == std::numeric_limits<uint64_t>::max()) {
                state.clear();
                store.detach_active();
                shutdown_requested = true;
                if (!send_error(fd, request, pp::ErrorCode::internal, "residency generation exhausted")) return false;
                return true;
            }
            state.clear();
            store.detach_active();
            ++current_generation;
            pp::Writer writer;
            writer.u64(current_generation);
            if (!send_response(fd, request, pp::Opcode::release, writer.take())) return false;
            if (current_generation == std::numeric_limits<uint64_t>::max()) {
                shutdown_requested = true;
                return true;
            }
            continue;
        }

        if (request.header.opcode == pp::Opcode::ping) {
            if (request.payload.size() > 4096) {
                if (!send_error(fd, request, pp::ErrorCode::invalid_argument, "PING payload exceeds 4096 bytes")) return false;
                continue;
            }
            if (!send_response(fd, request, pp::Opcode::ping, request.payload)) return false;
            continue;
        }

        if (request.header.opcode == pp::Opcode::close) {
            if (!request.payload.empty()) {
                if (!send_error(fd, request, pp::ErrorCode::invalid_argument, "CLOSE payload must be empty")) return false;
                continue;
            }
            if (!send_response(fd, request, pp::Opcode::close, {})) return false;
            return true;
        }

        if (request.header.opcode == pp::Opcode::shutdown) {
            if (!request.payload.empty()) {
                if (!send_error(fd, request, pp::ErrorCode::invalid_argument, "SHUTDOWN payload must be empty")) return false;
                continue;
            }
            if (!send_response(fd, request, pp::Opcode::shutdown, {})) return false;
            shutdown_requested = true;
            return true;
        }

        if (!send_error(fd, request, pp::ErrorCode::unsupported, "unsupported request opcode")) return false;
    }
}

} // namespace

int main(int argc, char ** argv) {
    Config config;
    if (!parse_args(argc, argv, config)) {
        std::fprintf(stderr,
                "usage: %s [--model shard.gguf] [--store-dir DIR] [--max-store-mib 16384] "
                "[--max-model-mib 16384] [--min-free-mib 256] [--backend HTP0] "
                "[--bind 127.0.0.1] [--port 9090] [--route-epoch 1] [--generation 1] "
                "[--max-payload-mib 64] [--timeout-ms 600000] [--profile-recv]\n",
                argv[0]);
        return 1;
    }

    std::string error;
    pp::ArtifactStore store;
    if (!config.store_dir.empty()) {
        pp::StoreLimits limits;
        limits.max_store_bytes = config.max_store_bytes;
        limits.max_object_bytes = config.max_model_bytes;
        limits.min_free_bytes = config.min_free_bytes;
        if (!store.open(config.store_dir, limits, error)) {
            std::fprintf(stderr, "error: dynamic store: %s\n", error.c_str());
            return 2;
        }
    }
    pp::Fd listener = pp::listen_tcp(config.bind_host, config.port, 4, error);
    if (!listener.valid()) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 2;
    }
    std::fprintf(stderr,
            "[phone-pim] listening %s:%u backend=%s route=%llu generation=%llu store=%s\n",
            config.bind_host.c_str(), config.port, config.backend.c_str(),
            static_cast<unsigned long long>(config.route_epoch),
            static_cast<unsigned long long>(config.generation),
            store.enabled() ? config.store_dir.c_str() : "disabled");

    if (config.profile_recv) {
        pp::set_receive_profiling(true); // measurement-only; opt-in, default off
    }
    StageRecvProfile stage_recv;
    ResidentState state;
    uint64_t current_generation = config.generation;
    uint64_t next_session_epoch = 0;
    if (!seed_session_epoch(next_session_epoch)) {
        std::fprintf(stderr, "error: cannot seed session epoch: %s\n", std::strerror(errno));
        return 2;
    }
    bool shutdown_requested = false;
    while (!shutdown_requested) {
        error.clear();
        pp::Fd client = pp::accept_tcp(listener.get(), config.timeout_ms, error);
        if (!client.valid()) {
            if (error == "accept timed out") {
                continue;
            }
            std::fprintf(stderr, "[phone-pim] accept failed: %s\n", error.c_str());
            return 3;
        }
        if (next_session_epoch == std::numeric_limits<uint64_t>::max()) {
            std::fprintf(stderr, "[phone-pim] session epoch exhausted\n");
            return 4;
        }
        const uint64_t session_epoch = ++next_session_epoch;
        std::fprintf(stderr, "[phone-pim] client connected session=%llu\n",
                     static_cast<unsigned long long>(session_epoch));
        if (!serve_connection(
                    client.get(), config, state, store, session_epoch,
                    current_generation, shutdown_requested, stage_recv)) {
            std::fprintf(stderr, "[phone-pim] client failed; resident state retained\n");
        }
    }
    state.clear();
    if (config.profile_recv) {
        // Stage-scoped (STAGE_* opcodes only). recv_header includes stop-and-wait idle;
        // frame_sha covers envelope+data; store_prefix_hash is data-only.
        std::fprintf(stderr,
                "[phone-pim] stage_recv_profile {\"frames\":%llu,\"payload_bytes\":%llu,"
                "\"recv_header_ms\":%.3f,\"recv_payload_ms\":%.3f,\"frame_sha_ms\":%.3f,"
                "\"store_prefix_hash_ms\":%.3f}\n",
                static_cast<unsigned long long>(stage_recv.frames),
                static_cast<unsigned long long>(stage_recv.payload_bytes),
                stage_recv.recv_header_us / 1000.0, stage_recv.recv_payload_us / 1000.0,
                stage_recv.frame_sha_us / 1000.0, stage_recv.store_prefix_hash_us / 1000.0);
    }
    return 0;
}
