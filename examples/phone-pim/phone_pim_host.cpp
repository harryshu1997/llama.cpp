#include "phone_pim_ffn.h"
#include "phone_pim_protocol.h"
#include "phone_pim_socket.h"
#include "phone_pim_store.h"

#include "llama.h"
#include "ggml-backend.h"
#include "../../src/llama-ext.h"

extern "C" {
#include "sha256.h"
}

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace pp = phone_pim;

namespace {

uint64_t now_us() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

struct Config {
    std::string host = "127.0.0.1";
    uint16_t port = 9090;
    std::string model_path;
    std::string prefix = "blk.2";
    uint32_t token_count = 16;
    uint32_t repeat = 7;
    uint64_t route_epoch = 1;
    uint64_t generation = 1;
    uint64_t island_id = 1;
    uint64_t max_payload = pp::k_default_max_payload;
    int timeout_ms = 600000;
    bool release = false;
    bool shutdown = false;
    bool provision = false;
    bool provision_only = false;
    bool replace_active_staging = false;
    uint32_t stage_chunk_bytes = 4U * 1024 * 1024;
    uint32_t test_stop_after_chunks = 0;
    uint32_t stage_window = 1; // bounded outstanding STAGE_CHUNK requests; 1 == stop-and-wait
    bool profile_transport = false; // measurement-only: split host frame-SHA vs. socket send
};

// Total in-flight STAGE_CHUNK payload is capped regardless of window depth so a large
// window can never pin more than this much unacknowledged data (invariant from the task).
constexpr uint64_t k_max_inflight_bytes = 64ULL * 1024 * 1024;

std::string json_escape(const std::string & in) {
    std::string out;
    out.reserve(in.size() + 8);
    for (char c : in) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

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

bool parse_args(int argc, char ** argv, Config & config) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : nullptr; };
        if (arg == "--host") {
            const char * value = next();
            if (value == nullptr) return false;
            config.host = value;
        } else if (arg == "--port") {
            uint64_t value = 0;
            if (!parse_u64(next(), value) || value == 0 || value > 65535) return false;
            config.port = static_cast<uint16_t>(value);
        } else if (arg == "--model" || arg == "-m") {
            const char * value = next();
            if (value == nullptr) return false;
            config.model_path = value;
        } else if (arg == "--prefix") {
            const char * value = next();
            if (value == nullptr) return false;
            config.prefix = value;
        } else if (arg == "--M") {
            uint64_t value = 0;
            if (!parse_u64(next(), value) || value == 0 || value > 1024) return false;
            config.token_count = static_cast<uint32_t>(value);
        } else if (arg == "--repeat") {
            uint64_t value = 0;
            if (!parse_u64(next(), value) || value < 2 || value > 10000) return false;
            config.repeat = static_cast<uint32_t>(value);
        } else if (arg == "--route-epoch") {
            if (!parse_u64(next(), config.route_epoch) || config.route_epoch == 0) return false;
        } else if (arg == "--generation") {
            if (!parse_u64(next(), config.generation) || config.generation == 0) return false;
        } else if (arg == "--island-id") {
            if (!parse_u64(next(), config.island_id) || config.island_id == 0) return false;
        } else if (arg == "--max-payload-mib") {
            uint64_t value = 0;
            if (!parse_u64(next(), value) || value == 0 || value > 64) return false;
            config.max_payload = value * 1024 * 1024;
        } else if (arg == "--timeout-ms") {
            uint64_t value = 0;
            if (!parse_u64(next(), value) || value == 0 || value > std::numeric_limits<int>::max()) return false;
            config.timeout_ms = static_cast<int>(value);
        } else if (arg == "--release") {
            config.release = true;
        } else if (arg == "--shutdown") {
            config.shutdown = true;
        } else if (arg == "--provision") {
            const char * value = next();
            if (value == nullptr) return false;
            if (std::strcmp(value, "never") == 0) {
                config.provision = false;
            } else if (std::strcmp(value, "if-missing") == 0) {
                config.provision = true;
            } else {
                return false;
            }
        } else if (arg == "--chunk-mib") {
            uint64_t value = 0;
            if (!parse_u64(next(), value) || value == 0 || value > 16) return false;
            config.stage_chunk_bytes = static_cast<uint32_t>(value * 1024 * 1024);
        } else if (arg == "--provision-only") {
            config.provision_only = true;
        } else if (arg == "--replace-active-staging") {
            config.replace_active_staging = true;
        } else if (arg == "--test-stop-after-chunks") {
            uint64_t value = 0;
            if (!parse_u64(next(), value) || value == 0 || value > UINT32_MAX) return false;
            config.test_stop_after_chunks = static_cast<uint32_t>(value);
        } else if (arg == "--stage-window") {
            // Frozen tested contract: only {1,2,4,8}. Other depths are rejected unless the
            // contract is explicitly revised and re-tested.
            uint64_t value = 0;
            if (!parse_u64(next(), value) ||
                (value != 1 && value != 2 && value != 4 && value != 8)) return false;
            config.stage_window = static_cast<uint32_t>(value);
        } else if (arg == "--profile-transport") {
            config.profile_transport = true;
        } else {
            return false;
        }
    }
    return !config.model_path.empty() && !config.prefix.empty() &&
           (!config.provision_only || config.provision) &&
           (config.test_stop_after_chunks == 0 || config.provision);
}

struct OracleCapture {
    std::vector<float> ffn_input;
    std::vector<float> ffn_delta;
    int64_t n_embd = 0;
    int64_t n_tokens = 0;
};

bool oracle_capture_cb(ggml_tensor * tensor, bool ask, void * user_data) {
    auto * capture = static_cast<OracleCapture *>(user_data);
    const bool is_input = tensor->name[0] != '\0' &&
                          std::strncmp(tensor->name, "attn_out", 8) == 0;
    const bool is_delta = tensor->name[0] != '\0' &&
                          std::strncmp(tensor->name, "ffn_post_norm", 13) == 0;
    const bool wanted = is_input || is_delta;
    if (ask) {
        return wanted;
    }
    std::vector<float> & destination = is_input ? capture->ffn_input : capture->ffn_delta;
    if (!wanted || !destination.empty() || tensor->type != GGML_TYPE_F32 || tensor->buffer == nullptr ||
        tensor->ne[0] <= 0 || tensor->ne[1] <= 0) {
        return true;
    }
    capture->n_embd = tensor->ne[0];
    capture->n_tokens = tensor->ne[1];
    destination.resize(static_cast<size_t>(tensor->ne[0] * tensor->ne[1]));
    ggml_backend_tensor_get(tensor, destination.data(), 0, destination.size() * sizeof(float));
    return true;
}

bool parse_layer_prefix(const std::string & prefix, int & layer) {
    constexpr const char marker[] = "blk.";
    if (prefix.compare(0, sizeof(marker) - 1, marker) != 0) {
        return false;
    }
    const char * begin = prefix.c_str() + sizeof(marker) - 1;
    char * end = nullptr;
    errno = 0;
    const long parsed = std::strtol(begin, &end, 10);
    if (errno != 0 || end == begin || *end != '\0' || parsed <= 0 || parsed >= std::numeric_limits<int>::max()) {
        return false;
    }
    layer = static_cast<int>(parsed);
    return true;
}

bool safe_json_label(const std::string & value) {
    if (value.empty() || value.size() > 64) {
        return false;
    }
    for (char ch : value) {
        if (!((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
              (ch >= '0' && ch <= '9') || ch == '_' || ch == '-')) {
            return false;
        }
    }
    return true;
}

struct OracleCase {
    std::vector<float> input;
    std::vector<float> expected;
};

class ProductionOracle {
public:
    ~ProductionOracle() {
        if (model_ != nullptr) {
            llama_model_free(model_);
        }
    }

    bool init(const std::string & model_path, const std::string & prefix, std::string & error) {
        int layer = 0;
        if (!parse_layer_prefix(prefix, layer)) {
            error = "production oracle requires a canonical blk.N prefix";
            return false;
        }
        const std::string start = std::to_string(layer);
        const std::string end = std::to_string(layer + 1);
        if (setenv("LLAMA_LAYER_START", start.c_str(), 1) != 0 ||
            setenv("LLAMA_LAYER_END", end.c_str(), 1) != 0) {
            error = "cannot configure production one-layer oracle";
            return false;
        }
        llama_model_params params = llama_model_default_params();
        params.n_gpu_layers = 0;
        model_ = llama_model_load_from_file(model_path.c_str(), params);
        if (model_ == nullptr) {
            error = "production oracle model load failed";
            return false;
        }
        n_embd_ = llama_model_n_embd(model_);
        const int n_layer = static_cast<int>(llama_model_n_layer(model_));
        if (n_embd_ <= 0 || layer + 1 >= n_layer) {
            error = "production oracle requires a headless middle layer";
            return false;
        }
        return true;
    }

    int n_embd() const { return n_embd_; }

    bool run(uint32_t token_count, uint32_t seed, OracleCase & result, std::string & error) const {
        OracleCapture capture;
        llama_context_params params = llama_context_default_params();
        params.n_seq_max = 1;
        params.n_ctx = token_count + 64;
        params.n_batch = std::max<uint32_t>(token_count, 8);
        params.n_ubatch = params.n_batch;
        params.no_perf = true;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        params.cb_eval = oracle_capture_cb;
        params.cb_eval_user_data = &capture;
        llama_context * context = llama_init_from_model(model_, params);
        if (context == nullptr) {
            error = "production oracle context allocation failed";
            return false;
        }
        llama_set_embeddings_nextn(context, true, false);
        llama_batch batch = llama_batch_init(static_cast<int32_t>(token_count), n_embd_, 1);
        batch.n_tokens = static_cast<int32_t>(token_count);
        for (uint32_t token = 0; token < token_count; ++token) {
            float * embedding = batch.embd + static_cast<size_t>(token) * n_embd_;
            for (int i = 0; i < n_embd_; ++i) {
                embedding[i] = 0.001f * static_cast<float>(
                        (static_cast<int64_t>(i) + token * 13 + seed) % 17 - 8);
            }
            batch.pos[token] = static_cast<llama_pos>(token);
            batch.n_seq_id[token] = 1;
            batch.seq_id[token][0] = 0;
            batch.logits[token] = 1;
        }

        const int decode_status = llama_decode(context, batch);
        llama_batch_free(batch);
        llama_free(context);

        if (decode_status != 0 || capture.n_embd != n_embd_ || capture.n_tokens != token_count ||
            capture.ffn_input.empty() || capture.ffn_input.size() != capture.ffn_delta.size() ||
            !pp::all_finite(capture.ffn_input.data(), capture.ffn_input.size()) ||
            !pp::all_finite(capture.ffn_delta.data(), capture.ffn_delta.size())) {
            error = "production oracle failed to capture a finite FFN boundary";
            return false;
        }
        result.expected.resize(capture.ffn_input.size());
        for (size_t i = 0; i < result.expected.size(); ++i) {
            result.expected[i] = capture.ffn_input[i] + capture.ffn_delta[i];
        }
        if (!pp::all_finite(result.expected.data(), result.expected.size())) {
            error = "production oracle FFN residual is non-finite";
            return false;
        }
        result.input = std::move(capture.ffn_input);
        return true;
    }

private:
    llama_model * model_ = nullptr;
    int n_embd_ = 0;
};

struct Client {
    pp::Fd socket;
    Config config;
    uint64_t command_seq = 0;
    uint64_t request_id = 0;
    uint64_t session_epoch = 0;
    // Measurement-only accumulators (no wire effect): split blocked send vs. ACK-wait.
    uint64_t send_us = 0;
    uint64_t recv_us = 0;
    // Populated only when --profile-transport enabled send profiling in the socket layer:
    // outer-frame SHA (envelope+data) vs. actual socket write, split out of send_us.
    uint64_t frame_sha_us = 0;
    uint64_t socket_send_us = 0;

    // Send one request frame and return the exact header sent (for later matching).
    // No response is read here; used both by call() and by the bounded-window pipeline.
    bool send_request(pp::Opcode opcode, const std::vector<uint8_t> & payload, pp::Header & header, std::string & error) {
        if (payload.size() > config.max_payload || payload.size() > pp::opcode_payload_limit(opcode)) {
            error = "request payload exceeds negotiated maximum";
            return false;
        }
        header = pp::Header{};
        header.opcode = opcode;
        header.request_id = ++request_id;
        header.command_seq = ++command_seq;
        header.session_epoch = opcode == pp::Opcode::hello ? 0 : session_epoch;
        header.route_epoch = config.route_epoch;
        header.residency_generation = opcode == pp::Opcode::hello ? 0 : config.generation;
        const uint64_t send_start = now_us();
        const bool ok = pp::send_frame(socket.get(), header, payload, error);
        send_us += now_us() - send_start;
        if (config.profile_transport) {
            pp::SendProfile sp;
            pp::get_last_send_profile(sp);
            frame_sha_us += sp.sha_us;
            socket_send_us += sp.write_us;
        }
        return ok;
    }

    // Receive one response and validate it against a specific previously-sent header.
    // Because requests are issued in strictly increasing id/seq order and TCP is ordered,
    // the caller drains responses in the same FIFO order it sent them.
    bool recv_response(const pp::Header & sent, pp::Frame & response, std::string & error) {
        const uint64_t recv_start = now_us();
        const pp::ReceiveResult rr = pp::receive_frame(socket.get(), response, config.max_payload, error);
        recv_us += now_us() - recv_start;
        if (rr != pp::ReceiveResult::ok) {
            return false;
        }
        if ((response.header.flags & pp::flag_response) == 0) {
            error = "response flag missing";
            return false;
        }
        if (response.header.request_id != sent.request_id) {
            error = "response request id mismatch";
            return false;
        }
        if (response.header.command_seq != sent.command_seq) {
            error = "response command sequence mismatch";
            return false;
        }
        if (response.header.session_epoch != sent.session_epoch) {
            error = "response session epoch mismatch";
            return false;
        }
        if (response.header.route_epoch != sent.route_epoch) {
            error = "response route epoch mismatch";
            return false;
        }
        if (response.header.residency_generation != sent.residency_generation) {
            error = "response residency generation mismatch";
            return false;
        }
        if ((response.header.flags & pp::flag_error) != 0 || response.header.opcode == pp::Opcode::error) {
            pp::ErrorCode code;
            std::string remote_error;
            if (!pp::parse_error_payload(response.payload, code, remote_error)) {
                error = "malformed remote error";
            } else {
                error = "remote error " + std::to_string(static_cast<uint32_t>(code)) + ": " + remote_error;
            }
            return false;
        }
        if (response.header.opcode != sent.opcode || response.header.flags != pp::flag_response) {
            error = "unexpected response opcode or flags";
            return false;
        }
        return true;
    }

    // Stop-and-wait convenience: send one request and block for its single response.
    // Byte-identical to the pre-pipeline behavior; used for begin/commit/abort and window=1.
    bool call(pp::Opcode opcode, const std::vector<uint8_t> & payload, pp::Frame & response, std::string & error) {
        pp::Header sent;
        return send_request(opcode, payload, sent, error) && recv_response(sent, response, error);
    }
};

bool pread_exact(int fd, uint64_t offset, uint8_t * data, size_t size, std::string & error) {
    if (offset > static_cast<uint64_t>(std::numeric_limits<off_t>::max()) ||
        size > static_cast<uint64_t>(std::numeric_limits<off_t>::max()) - offset) {
        error = "local chunk range exceeds off_t";
        return false;
    }
    size_t completed = 0;
    while (completed < size) {
        const ssize_t count = pread(
                fd, data + completed, size - completed, static_cast<off_t>(offset + completed));
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            error = count == 0 ? "short local model read"
                               : "local model read failed: " + std::string(std::strerror(errno));
            return false;
        }
        completed += static_cast<size_t>(count);
    }
    return true;
}

bool seed_prefix_hash(
        int fd,
        uint64_t bytes,
        sha256_t & context,
        std::array<uint8_t, 32> & digest,
        std::string & error) {
    sha256_init(&context);
    std::vector<uint8_t> buffer(1024 * 1024);
    uint64_t offset = 0;
    while (offset < bytes) {
        const size_t amount = static_cast<size_t>(std::min<uint64_t>(buffer.size(), bytes - offset));
        if (!pread_exact(fd, offset, buffer.data(), amount, error)) {
            return false;
        }
        sha256_update(&context, buffer.data(), amount);
        offset += amount;
    }
    sha256_t copy = context;
    sha256_final(&copy, digest.data());
    return true;
}

bool build_stage_spec(
        int fd,
        uint64_t bytes,
        const std::array<uint8_t, 32> & digest,
        uint32_t chunk_bytes,
        pp::StageObjectSpec & spec,
        uint64_t & chunk_hash_us,
        std::string & error) {
    if (bytes == 0 || chunk_bytes == 0) {
        error = "cannot stage an empty object or zero-sized chunk";
        return false;
    }
    const uint64_t count64 = 1 + (bytes - 1) / chunk_bytes;
    if (count64 == 0 || count64 > pp::k_max_stage_chunks) {
        error = "local model needs too many stage chunks";
        return false;
    }
    spec = {};
    spec.bytes = bytes;
    spec.sha256 = digest;
    spec.chunk_bytes = chunk_bytes;
    spec.chunks.resize(static_cast<size_t>(count64));
    const uint64_t start = now_us();
    for (uint32_t i = 0; i < spec.chunks.size(); ++i) {
        pp::StageChunkSpec & chunk = spec.chunks[i];
        chunk.index = i;
        chunk.offset = static_cast<uint64_t>(i) * chunk_bytes;
        chunk.bytes = static_cast<uint32_t>(std::min<uint64_t>(chunk_bytes, bytes - chunk.offset));
        if (!pp::sha256_fd_range(fd, chunk.offset, chunk.bytes, chunk.sha256, error)) {
            return false;
        }
    }
    chunk_hash_us = now_us() - start;
    spec.manifest_sha256 = pp::stage_manifest_sha256(spec);
    pp::StoreLimits limits;
    limits.max_store_bytes = std::numeric_limits<uint64_t>::max();
    limits.max_object_bytes = std::numeric_limits<uint64_t>::max();
    limits.min_free_bytes = 0;
    return pp::validate_stage_spec(spec, limits, error);
}

uint64_t make_ticket_id(const std::array<uint8_t, 32> & digest, uint64_t session_epoch) {
    uint64_t ticket = 0;
    for (size_t i = 0; i < 8; ++i) {
        ticket |= static_cast<uint64_t>(digest[i]) << (8 * i);
    }
    ticket ^= session_epoch * 0x9e3779b97f4a7c15ULL;
    return ticket == 0 ? 1 : ticket;
}

void encode_stage_spec(pp::Writer & writer, uint64_t ticket_id, const pp::StageObjectSpec & spec) {
    writer.u64(ticket_id);
    writer.u64(spec.bytes);
    writer.bytes(spec.sha256.data(), spec.sha256.size());
    writer.u32(spec.chunk_bytes);
    writer.u32(static_cast<uint32_t>(spec.chunks.size()));
    writer.bytes(spec.manifest_sha256.data(), spec.manifest_sha256.size());
    for (const pp::StageChunkSpec & chunk : spec.chunks) {
        writer.u32(chunk.index);
        writer.u64(chunk.offset);
        writer.u32(chunk.bytes);
        writer.bytes(chunk.sha256.data(), chunk.sha256.size());
    }
}

bool decode_stage_progress(pp::Reader & reader, pp::StageProgress & progress) {
    uint32_t state = 0;
    if (!reader.u32(state) || (state != static_cast<uint32_t>(pp::StageState::receiving) &&
                               state != static_cast<uint32_t>(pp::StageState::published)) ||
        !reader.u64(progress.ticket_id) || !reader.u64(progress.residency_generation) ||
        !reader.u64(progress.verified_bytes) || !reader.u32(progress.next_chunk) ||
        !reader.bytes(progress.prefix_sha256.data(), progress.prefix_sha256.size()) ||
        !reader.bytes(progress.manifest_sha256.data(), progress.manifest_sha256.size()) ||
        !reader.u64(progress.resume_scan_us)) {
        return false;
    }
    progress.state = static_cast<pp::StageState>(state);
    return true;
}

struct ProvisionMetrics {
    bool requested = false;
    bool cache_hit = false;
    bool partial = false;
    uint64_t ticket_id = 0;
    uint64_t source_chunk_hash_us = 0;
    uint64_t resume_scan_us = 0;
    uint64_t resume_offset = 0;
    uint64_t bytes_sent = 0;         // == acked_durable_bytes (data only)
    uint32_t chunks_sent = 0;
    uint64_t e2e_us = 0;
    // Data-byte accounting for this run (data only, excludes the 56-byte envelope):
    uint64_t attempted_bytes = 0;      // chunks the host began sending
    uint64_t socket_written_bytes = 0; // chunks fully written to the socket
    uint64_t acked_durable_bytes = 0;  // chunks whose ACK confirmed a durable verified prefix
    uint64_t wasted_bytes = 0;         // written but not confirmed durable (= written - acked)
    uint64_t retried_bytes = 0;        // resent within this run (0; resend only happens across runs)
    // Measurement-only host-side decomposition of the window=1 stage wall.
    uint64_t host_read_us = 0;       // pread of source chunks inside the stage loop
    uint64_t host_hash_us = 0;       // per-chunk manifest re-hash inside the stage loop
    uint64_t host_prefix_hash_us = 0;// per-chunk rolling durable-prefix SHA (data only)
    uint64_t host_send_us = 0;       // blocked send_frame across begin+chunks+commit (SHA + socket)
    uint64_t host_frame_sha_us = 0;  // --profile-transport: outer-frame SHA (envelope+data), subset of host_send
    uint64_t host_socket_send_us = 0;// --profile-transport: actual socket write, subset of host_send
    uint64_t host_ack_wait_us = 0;   // blocked receive_frame (ACK wait) across begin+chunks+commit
    pp::StageMetrics remote;
};

bool resolve_active_transfer(
        Client & client,
        const pp::StageObjectSpec & requested,
        bool replace_active,
        std::string & error) {
    pp::Frame response;
    if (!client.call(pp::Opcode::status, {}, response, error)) {
        return false;
    }
    pp::Reader reader(response.payload);
    uint32_t ready = 0;
    uint64_t island_id = 0;
    uint64_t generation = 0;
    uint64_t n_embd = 0;
    uint64_t n_ff = 0;
    uint32_t token_count = 0;
    std::string prefix;
    uint32_t store_enabled = 0;
    uint32_t active = 0;
    uint64_t ticket_id = 0;
    uint64_t verified_bytes = 0;
    uint32_t next_chunk = 0;
    std::array<uint8_t, 32> manifest = {};
    if (!reader.u32(ready) || ready > 1 || !reader.u64(island_id) ||
        !reader.u64(generation) || !reader.u64(n_embd) || !reader.u64(n_ff) ||
        !reader.u32(token_count) || !reader.string(prefix, 256) ||
        !reader.u32(store_enabled) || store_enabled != 1 || !reader.u32(active) || active > 1 ||
        !reader.u64(ticket_id) || !reader.u64(verified_bytes) || !reader.u32(next_chunk) ||
        !reader.bytes(manifest.data(), manifest.size()) || !reader.done() ||
        generation != client.config.generation ||
        (active != 0 && (ticket_id == 0 || manifest == std::array<uint8_t, 32>{}))) {
        error = "malformed STATUS response for dynamic provisioning";
        return false;
    }
    if (active == 0 || manifest == requested.manifest_sha256) {
        return true;
    }
    if (!replace_active) {
        error = "a different staged transfer is active; explicit replacement is required";
        return false;
    }
    pp::Writer abort_writer;
    abort_writer.u64(ticket_id);
    abort_writer.u32(0);
    if (!client.call(pp::Opcode::stage_abort, abort_writer.data(), response, error) ||
        !response.payload.empty()) {
        if (error.empty()) error = "malformed STAGE_ABORT response";
        return false;
    }
    return true;
}

bool provision_model(
        Client & client,
        int model_fd,
        const pp::StageObjectSpec & spec,
        const Config & config,
        ProvisionMetrics & metrics,
        std::string & error) {
    metrics.requested = true;
    metrics.ticket_id = make_ticket_id(spec.sha256, client.session_epoch);
    const uint64_t start = now_us();
    const uint64_t send_us0 = client.send_us;
    const uint64_t recv_us0 = client.recv_us;
    const uint64_t frame_sha_us0 = client.frame_sha_us;
    const uint64_t socket_send_us0 = client.socket_send_us;
    if (!resolve_active_transfer(
                client, spec, config.replace_active_staging, error)) {
        return false;
    }
    pp::Writer begin_writer;
    encode_stage_spec(begin_writer, metrics.ticket_id, spec);
    pp::Frame response;
    if (!client.call(pp::Opcode::stage_begin, begin_writer.data(), response, error)) {
        return false;
    }
    pp::Reader begin_reader(response.payload);
    pp::StageProgress progress;
    if (!decode_stage_progress(begin_reader, progress) || !begin_reader.done() ||
        progress.ticket_id != metrics.ticket_id ||
        progress.residency_generation != client.config.generation ||
        progress.manifest_sha256 != spec.manifest_sha256 ||
        progress.next_chunk > spec.chunks.size() || progress.verified_bytes > spec.bytes) {
        error = "malformed STAGE_BEGIN response";
        return false;
    }
    metrics.resume_scan_us = progress.resume_scan_us;
    metrics.resume_offset = progress.verified_bytes;
    if (progress.state == pp::StageState::published) {
        if (progress.verified_bytes != spec.bytes || progress.next_chunk != spec.chunks.size() ||
            progress.prefix_sha256 != spec.sha256) {
            error = "published cache hit has the wrong identity";
            return false;
        }
        metrics.cache_hit = true;
        metrics.host_send_us = client.send_us - send_us0;
        metrics.host_frame_sha_us = client.frame_sha_us - frame_sha_us0;
        metrics.host_socket_send_us = client.socket_send_us - socket_send_us0;
        metrics.host_ack_wait_us = client.recv_us - recv_us0;
        metrics.e2e_us = now_us() - start;
        return true;
    }

    const uint64_t expected_offset = progress.next_chunk < spec.chunks.size()
            ? spec.chunks[progress.next_chunk].offset : spec.bytes;
    sha256_t local_prefix_context = {};
    std::array<uint8_t, 32> local_prefix = {};
    if (progress.verified_bytes != expected_offset ||
        !seed_prefix_hash(
                model_fd, progress.verified_bytes, local_prefix_context, local_prefix, error) ||
        local_prefix != progress.prefix_sha256) {
        error = "remote durable prefix differs from the local source";
        return false;
    }

    // Bounded-window pipeline. At most `window` STAGE_CHUNK requests are outstanding and
    // never more than k_max_inflight_bytes of unacknowledged payload. Chunks are still sent
    // strictly in ascending index order (TCP preserves order; the store enforces
    // chunk_index==next_chunk), so the worker processes them serially and its cumulative,
    // durable, manifest-verified prefix ACK still matches the expectation retained per
    // outstanding request. window==1 is byte-identical to the prior stop-and-wait path.
    struct Outstanding {
        pp::Header header;                    // to match the ACK back to this request
        uint32_t index;                       // chunk index sent
        uint64_t end_offset;                  // expected verified_bytes after this chunk
        std::array<uint8_t, 32> expected_prefix; // durable-prefix digest expected in the ACK
        uint32_t bytes;
    };
    std::deque<Outstanding> inflight;
    uint64_t inflight_frame_bytes = 0; // complete outstanding wire payload (envelope + data)
    const uint32_t window = std::max<uint32_t>(1, config.stage_window);
    const uint32_t total_chunks = static_cast<uint32_t>(spec.chunks.size());
    // Chunks to send this run: bounded by the test-stop knob so a window never overshoots it.
    const uint32_t send_limit = (config.test_stop_after_chunks != 0)
            ? std::min<uint32_t>(config.test_stop_after_chunks, total_chunks - progress.next_chunk)
            : (total_chunks - progress.next_chunk);
    std::vector<uint8_t> bytes(spec.chunk_bytes);
    uint32_t next_i = progress.next_chunk;
    uint32_t sent_this_run = 0;
    uint32_t acked_this_run = 0;

    while (acked_this_run < send_limit) {
        // Fill the window: keep >=1 outstanding, respect the depth, and cap the total
        // outstanding *complete request payload* (envelope + data) at k_max_inflight_bytes
        // using overflow-safe arithmetic.
        while (sent_this_run < send_limit && inflight.size() < window) {
            const pp::StageChunkSpec & chunk = spec.chunks[next_i];
            const uint64_t frame_payload =
                    static_cast<uint64_t>(chunk.bytes) + pp::k_stage_chunk_envelope_bytes;
            if (!inflight.empty() && inflight_frame_bytes > k_max_inflight_bytes - frame_payload) {
                break; // adding this frame would exceed the 64 MiB outstanding-payload bound
            }
            const uint64_t read_start = now_us();
            const bool read_ok = pread_exact(model_fd, chunk.offset, bytes.data(), chunk.bytes, error);
            const uint64_t hash_start = now_us();
            metrics.host_read_us += hash_start - read_start;
            const bool hash_ok = read_ok && pp::sha256(bytes.data(), chunk.bytes) == chunk.sha256;
            metrics.host_hash_us += now_us() - hash_start;
            if (!read_ok || !hash_ok) {
                if (error.empty()) error = "local chunk changed after manifest construction";
                return false;
            }
            pp::Writer chunk_writer;
            chunk_writer.u64(metrics.ticket_id);
            chunk_writer.u32(chunk.index);
            chunk_writer.u64(chunk.offset);
            chunk_writer.u32(chunk.bytes);
            chunk_writer.bytes(chunk.sha256.data(), chunk.sha256.size());
            chunk_writer.bytes(bytes.data(), chunk.bytes);
            pp::Header sent;
            metrics.attempted_bytes += chunk.bytes;
            if (!client.send_request(pp::Opcode::stage_chunk, chunk_writer.data(), sent, error)) {
                return false;
            }
            metrics.socket_written_bytes += chunk.bytes;
            // Advance the rolling durable-prefix digest and retain the expected value until
            // this specific chunk's ACK is validated below.
            const uint64_t prefix_hash_start = now_us();
            sha256_update(&local_prefix_context, bytes.data(), chunk.bytes);
            sha256_t expected_context = local_prefix_context;
            Outstanding oc;
            oc.header = sent;
            oc.index = next_i;
            oc.end_offset = chunk.offset + chunk.bytes;
            sha256_final(&expected_context, oc.expected_prefix.data());
            oc.bytes = chunk.bytes;
            metrics.host_prefix_hash_us += now_us() - prefix_hash_start;
            inflight.push_back(oc);
            inflight_frame_bytes += frame_payload;
            ++next_i;
            ++sent_this_run;
        }
        // Drain exactly one ACK, in FIFO order, and validate it against its expectation.
        const Outstanding oc = inflight.front();
        if (!client.recv_response(oc.header, response, error)) {
            return false;
        }
        pp::Reader chunk_reader(response.payload);
        pp::StageProgress ack;
        if (!decode_stage_progress(chunk_reader, ack) || !chunk_reader.done() ||
            ack.state != pp::StageState::receiving || ack.ticket_id != metrics.ticket_id ||
            ack.residency_generation != client.config.generation ||
            ack.manifest_sha256 != spec.manifest_sha256 || ack.next_chunk != oc.index + 1 ||
            ack.verified_bytes != oc.end_offset || ack.prefix_sha256 != oc.expected_prefix) {
            error = "malformed STAGE_CHUNK acknowledgement";
            return false;
        }
        inflight.pop_front();
        inflight_frame_bytes -= static_cast<uint64_t>(oc.bytes) + pp::k_stage_chunk_envelope_bytes;
        metrics.bytes_sent += oc.bytes;
        metrics.acked_durable_bytes += oc.bytes;
        ++metrics.chunks_sent;
        ++acked_this_run;
    }
    if (config.test_stop_after_chunks != 0 && next_i < total_chunks) {
        metrics.partial = true;
        metrics.host_send_us = client.send_us - send_us0;
        metrics.host_frame_sha_us = client.frame_sha_us - frame_sha_us0;
        metrics.host_socket_send_us = client.socket_send_us - socket_send_us0;
        metrics.host_ack_wait_us = client.recv_us - recv_us0;
        metrics.e2e_us = now_us() - start;
        return true;
    }

    pp::Writer commit_writer;
    commit_writer.u64(metrics.ticket_id);
    commit_writer.bytes(spec.manifest_sha256.data(), spec.manifest_sha256.size());
    if (!client.call(pp::Opcode::stage_commit, commit_writer.data(), response, error)) {
        return false;
    }
    pp::Reader commit_reader(response.payload);
    pp::StageProgress committed;
    if (!decode_stage_progress(commit_reader, committed) ||
        !commit_reader.u64(metrics.remote.chunk_hash_us) ||
        !commit_reader.u64(metrics.remote.write_us) ||
        !commit_reader.u64(metrics.remote.data_sync_us) ||
        !commit_reader.u64(metrics.remote.full_verify_us) ||
        !commit_reader.u64(metrics.remote.file_sync_us) ||
        !commit_reader.u64(metrics.remote.publish_us) ||
        !commit_reader.u64(metrics.remote.directory_sync_us) ||
        !commit_reader.u64(metrics.remote.accepted_chunks) ||
        !commit_reader.u64(metrics.remote.duplicate_chunks) || !commit_reader.done() ||
        committed.state != pp::StageState::published || committed.ticket_id != metrics.ticket_id ||
        committed.residency_generation != client.config.generation ||
        committed.verified_bytes != spec.bytes || committed.next_chunk != spec.chunks.size() ||
        committed.prefix_sha256 != spec.sha256 ||
        committed.manifest_sha256 != spec.manifest_sha256) {
        error = "malformed STAGE_COMMIT response";
        return false;
    }
    metrics.host_send_us = client.send_us - send_us0;
    metrics.host_frame_sha_us = client.frame_sha_us - frame_sha_us0;
    metrics.host_socket_send_us = client.socket_send_us - socket_send_us0;
    metrics.host_ack_wait_us = client.recv_us - recv_us0;
    metrics.e2e_us = now_us() - start;
    return true;
}

void encode_floats(pp::Writer & writer, const std::vector<float> & values) {
    for (float value : values) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        writer.u32(bits);
    }
}

bool decode_floats(pp::Reader & reader, uint64_t count, std::vector<float> & values) {
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

struct PreparedInfo {
    uint64_t island_id = 0;
    uint64_t n_embd = 0;
    uint64_t n_ff = 0;
    uint32_t token_count = 0;
    uint64_t model_bytes = 0;
    std::array<uint8_t, 32> model_sha256 = {};
    pp::ModelSource model_source = pp::ModelSource::prestaged;
    uint64_t verify_us = 0;
    uint64_t weight_bytes = 0;
    uint64_t resident_bytes = 0;
    uint64_t load_us = 0;
    uint64_t upload_us = 0;
    uint64_t warmup_us = 0;
    std::string backend_description;
};

bool parse_prepared(const std::vector<uint8_t> & payload, PreparedInfo & info) {
    pp::Reader reader(payload);
    uint32_t source = 0;
    const bool ok = reader.u64(info.island_id) && reader.u64(info.n_embd) && reader.u64(info.n_ff) &&
           reader.u32(info.token_count) && reader.u64(info.model_bytes) &&
           reader.bytes(info.model_sha256.data(), info.model_sha256.size()) && reader.u32(source) &&
           reader.u64(info.verify_us) &&
           reader.u64(info.weight_bytes) && reader.u64(info.resident_bytes) &&
           reader.u64(info.load_us) && reader.u64(info.upload_us) && reader.u64(info.warmup_us) &&
           reader.string(info.backend_description, 1024) && reader.done();
    if (!ok || (source != static_cast<uint32_t>(pp::ModelSource::prestaged) &&
                source != static_cast<uint32_t>(pp::ModelSource::published_store))) {
        return false;
    }
    info.model_source = static_cast<pp::ModelSource>(source);
    return true;
}

struct RemoteExecution {
    std::vector<float> output;
    uint64_t input_set_us = 0;
    uint64_t compute_us = 0;
    uint64_t output_get_us = 0;
    uint64_t e2e_us = 0;
};

bool execute_remote(
        Client & client,
        uint64_t island_id,
        const std::vector<float> & input,
        RemoteExecution & execution,
        std::string & error) {
    pp::Writer writer;
    writer.u64(island_id);
    writer.u64(input.size());
    encode_floats(writer, input);
    pp::Frame response;
    const uint64_t start = now_us();
    if (!client.call(pp::Opcode::execute, writer.data(), response, error)) {
        return false;
    }
    execution.e2e_us = now_us() - start;
    pp::Reader reader(response.payload);
    uint64_t response_island = 0;
    uint64_t element_count = 0;
    if (!reader.u64(response_island) || !reader.u64(element_count) || response_island != island_id ||
        !reader.u64(execution.input_set_us) || !reader.u64(execution.compute_us) ||
        !reader.u64(execution.output_get_us) || !decode_floats(reader, element_count, execution.output) ||
        element_count != input.size()) {
        error = "malformed EXECUTE response";
        return false;
    }
    return true;
}

bool release_remote(Client & client, uint64_t island_id, std::string & error) {
    pp::Writer writer;
    writer.u64(island_id);
    pp::Frame response;
    if (!client.call(pp::Opcode::release, writer.data(), response, error)) {
        return false;
    }
    pp::Reader reader(response.payload);
    uint64_t next_generation = 0;
    if (!reader.u64(next_generation) || !reader.done() ||
        next_generation <= client.config.generation) {
        error = "malformed RELEASE response";
        return false;
    }
    client.config.generation = next_generation;
    return true;
}

struct Stats {
    double p50 = 0.0;
    double p95 = 0.0;
    double mean = 0.0;
    double cov = 0.0;
};

Stats stats(std::vector<uint64_t> values) {
    Stats result;
    if (values.empty()) {
        return result;
    }
    std::sort(values.begin(), values.end());
    result.p50 = values[(values.size() - 1) / 2] / 1000.0;
    result.p95 = values[static_cast<size_t>(std::ceil(0.95 * values.size())) - 1] / 1000.0;
    double sum = 0.0;
    for (uint64_t value : values) sum += value;
    const double mean_us = sum / values.size();
    double squared = 0.0;
    for (uint64_t value : values) {
        const double difference = value - mean_us;
        squared += difference * difference;
    }
    result.mean = mean_us / 1000.0;
    result.cov = mean_us > 0.0 ? std::sqrt(squared / values.size()) / mean_us : 0.0;
    return result;
}

} // namespace

int main(int argc, char ** argv) {
    Config config;
    if (!parse_args(argc, argv, config)) {
        std::fprintf(stderr,
                "usage: %s -m local-shard.gguf [--host 127.0.0.1] [--port 9090] [--prefix blk.2] "
                "[--M 16] [--repeat 7] [--route-epoch 1] [--generation 1] [--island-id 1] "
                "[--provision never|if-missing] [--chunk-mib 4] [--provision-only] "
                "[--replace-active-staging] [--stage-window 1|2|4|8] [--profile-transport] "
                "[--test-stop-after-chunks N] [--release] [--shutdown]\n",
                argv[0]);
        return 1;
    }
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            std::fprintf(stderr, "%s", text);
        }
    }, nullptr);

    std::string error;
    pp::Fd local_model(open(config.model_path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
    struct stat local_stat = {};
    if (!local_model.valid() || fstat(local_model.get(), &local_stat) != 0 ||
        !S_ISREG(local_stat.st_mode) || local_stat.st_size <= 0) {
        std::fprintf(stderr, "error: cannot open a regular local model: %s\n", std::strerror(errno));
        return 2;
    }
    const uint64_t local_model_bytes = static_cast<uint64_t>(local_stat.st_size);
    std::array<uint8_t, 32> local_model_sha256 = {};
    const uint64_t source_hash_start = now_us();
    if (!pp::sha256_fd_range(
                local_model.get(), 0, local_model_bytes, local_model_sha256, error)) {
        std::fprintf(stderr, "error: local model identity: %s\n", error.c_str());
        return 2;
    }
    const uint64_t source_hash_us = now_us() - source_hash_start;

    if (config.profile_transport) {
        pp::set_send_profiling(true); // measurement-only; opt-in, default off
    }
    Client client;
    client.config = config;
    client.socket = pp::connect_tcp(config.host, config.port, config.timeout_ms, error);
    if (!client.socket.valid()) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 2;
    }

    pp::Frame response;
    if (!client.call(pp::Opcode::hello, {}, response, error)) {
        std::fprintf(stderr, "error: HELLO: %s\n", error.c_str());
        return 3;
    }
    pp::Reader hello(response.payload);
    uint32_t protocol_version = 0;
    uint64_t session_epoch = 0;
    uint64_t remote_max_payload = 0;
    std::string remote_backend;
    uint64_t remote_features = 0;
    uint64_t remote_max_model_bytes = 0;
    uint64_t remote_max_chunk_bytes = 0;
    uint32_t remote_max_chunks = 0;
    std::string capability;
    if (!hello.u32(protocol_version) || !hello.u64(session_epoch) || !hello.u64(remote_max_payload) ||
        !hello.string(remote_backend) || !hello.u64(remote_features) ||
        !hello.u64(remote_max_model_bytes) || !hello.u64(remote_max_chunk_bytes) ||
        !hello.u32(remote_max_chunks) || !hello.string(capability) || !hello.done() ||
        protocol_version != pp::k_version || session_epoch == 0 || remote_max_payload == 0 ||
        !safe_json_label(remote_backend) || !safe_json_label(capability) ||
        (capability != "prestaged_gemma4_dense_ffn_v3" &&
         capability != "sequential_dynamic_gemma4_dense_ffn_v3")) {
        std::fprintf(stderr, "error: malformed HELLO response\n");
        return 3;
    }
    client.session_epoch = session_epoch;
    client.config.max_payload = std::min(client.config.max_payload, remote_max_payload);

    ProvisionMetrics provision;
    provision.source_chunk_hash_us = 0;
    pp::StageObjectSpec stage_spec;
    if (config.provision) {
        if ((remote_features & 1) == 0 || local_model_bytes > remote_max_model_bytes ||
            config.stage_chunk_bytes > remote_max_chunk_bytes ||
            config.stage_chunk_bytes > remote_max_payload -
                    std::min<uint64_t>(remote_max_payload, pp::k_stage_chunk_envelope_bytes)) {
            std::fprintf(stderr, "error: remote dynamic-store limits reject the local model\n");
            return 4;
        }
        if (!build_stage_spec(
                    local_model.get(), local_model_bytes, local_model_sha256,
                    config.stage_chunk_bytes, stage_spec, provision.source_chunk_hash_us, error) ||
            stage_spec.chunks.size() > remote_max_chunks) {
            if (error.empty()) error = "remote chunk-count limit exceeded";
            std::fprintf(stderr, "error: stage manifest: %s\n", error.c_str());
            return 4;
        }
        if (!provision_model(client, local_model.get(), stage_spec, config, provision, error)) {
            std::fprintf(stderr, "error: dynamic provisioning: %s\n", error.c_str());
            // Failure record (CP3): even on disconnect, persist byte accounting so a
            // subsequent resumed run can be reconciled for retry/waste.
            const uint64_t wasted = provision.socket_written_bytes >= provision.acked_durable_bytes
                    ? provision.socket_written_bytes - provision.acked_durable_bytes : 0;
            std::printf(
                    "{\"record_schema_version\":1,\"verdict\":\"FAIL_PROVISION\","
                    "\"protocol_version\":%u,\"stage_window\":%u,\"error\":\"%s\","
                    "\"model_bytes\":%llu,\"model_sha256\":\"%s\",\"manifest_sha256\":\"%s\","
                    "\"chunk_bytes\":%u,\"chunk_count\":%zu,\"resume_offset\":%llu,"
                    "\"chunks_sent\":%u,\"attempted_bytes\":%llu,\"socket_written_bytes\":%llu,"
                    "\"acked_durable_bytes\":%llu,\"wasted_bytes\":%llu,\"retried_bytes\":%llu}\n",
                    pp::k_version, config.stage_window, json_escape(error).c_str(),
                    static_cast<unsigned long long>(local_model_bytes),
                    pp::hex_sha256(local_model_sha256).c_str(),
                    pp::hex_sha256(stage_spec.manifest_sha256).c_str(), config.stage_chunk_bytes,
                    stage_spec.chunks.size(), static_cast<unsigned long long>(provision.resume_offset),
                    provision.chunks_sent, static_cast<unsigned long long>(provision.attempted_bytes),
                    static_cast<unsigned long long>(provision.socket_written_bytes),
                    static_cast<unsigned long long>(provision.acked_durable_bytes),
                    static_cast<unsigned long long>(wasted),
                    static_cast<unsigned long long>(provision.retried_bytes));
            return 4;
        }
        if (provision.partial) {
            client.socket.reset();
            const double useful_goodput = provision.e2e_us == 0 ? 0.0 :
                    provision.bytes_sent * 1000000.0 / provision.e2e_us / 1048576.0;
            std::printf(
                    "{\"record_schema_version\":1,\"verdict\":\"DYNAMIC_PROVISION_PARTIAL\","
                    "\"protocol_version\":%u,"
                    "\"capability\":\"%s\",\"session_epoch\":%llu,\"route_epoch\":%llu,"
                    "\"residency_generation\":%llu,\"ticket_id\":%llu,"
                    "\"model_bytes\":%llu,\"model_sha256\":\"%s\","
                    "\"manifest_sha256\":\"%s\",\"chunk_bytes\":%u,\"chunk_count\":%zu,"
                    "\"resume_offset\":%llu,\"chunks_sent\":%u,\"bytes_sent\":%llu,"
                    "\"stage_window\":%u,\"attempted_bytes\":%llu,\"socket_written_bytes\":%llu,"
                    "\"acked_durable_bytes\":%llu,\"wasted_bytes\":%llu,\"retried_bytes\":%llu,"
                    "\"source_hash_ms\":%.3f,\"source_chunk_hash_ms\":%.3f,"
                    "\"stage_e2e_ms\":%.3f,\"useful_goodput_mib_s\":%.3f}\n",
                    pp::k_version, capability.c_str(),
                    static_cast<unsigned long long>(session_epoch),
                    static_cast<unsigned long long>(config.route_epoch),
                    static_cast<unsigned long long>(config.generation),
                    static_cast<unsigned long long>(provision.ticket_id),
                    static_cast<unsigned long long>(local_model_bytes),
                    pp::hex_sha256(local_model_sha256).c_str(),
                    pp::hex_sha256(stage_spec.manifest_sha256).c_str(), config.stage_chunk_bytes,
                    stage_spec.chunks.size(), static_cast<unsigned long long>(provision.resume_offset),
                    provision.chunks_sent, static_cast<unsigned long long>(provision.bytes_sent),
                    config.stage_window, static_cast<unsigned long long>(provision.attempted_bytes),
                    static_cast<unsigned long long>(provision.socket_written_bytes),
                    static_cast<unsigned long long>(provision.acked_durable_bytes),
                    static_cast<unsigned long long>(provision.socket_written_bytes >= provision.acked_durable_bytes
                            ? provision.socket_written_bytes - provision.acked_durable_bytes : 0),
                    static_cast<unsigned long long>(provision.retried_bytes),
                    source_hash_us / 1000.0, provision.source_chunk_hash_us / 1000.0,
                    provision.e2e_us / 1000.0, useful_goodput);
            return 0;
        }
        if (config.provision_only) {
            if (config.shutdown) {
                if (!client.call(pp::Opcode::shutdown, {}, response, error)) {
                    std::fprintf(stderr, "error: SHUTDOWN: %s\n", error.c_str());
                    return 7;
                }
            } else {
                client.call(pp::Opcode::close, {}, response, error);
            }
            const double useful_goodput = provision.e2e_us == 0 ? 0.0 :
                    provision.bytes_sent * 1000000.0 / provision.e2e_us / 1048576.0;
            std::printf(
                    "{\"record_schema_version\":1,\"verdict\":\"DYNAMIC_PROVISION_PASS\","
                    "\"protocol_version\":%u,\"capability\":\"%s\",\"route\":\"%s\","
                    "\"model_source\":\"published_store\",\"outcome\":\"%s\","
                    "\"session_epoch\":%llu,\"route_epoch\":%llu,"
                    "\"residency_generation\":%llu,\"ticket_id\":%llu,"
                    "\"model_bytes\":%llu,\"model_sha256\":\"%s\","
                    "\"manifest_sha256\":\"%s\",\"chunk_bytes\":%u,\"chunk_count\":%zu,"
                    "\"resume_offset\":%llu,\"chunks_sent\":%u,\"bytes_sent\":%llu,"
                    "\"stage_window\":%u,\"attempted_bytes\":%llu,\"socket_written_bytes\":%llu,"
                    "\"acked_durable_bytes\":%llu,\"wasted_bytes\":%llu,\"retried_bytes\":%llu,"
                    "\"source_hash_ms\":%.3f,\"source_chunk_hash_ms\":%.3f,"
                    "\"stage_e2e_ms\":%.3f,\"useful_goodput_mib_s\":%.3f,"
                    "\"remote_resume_scan_ms\":%.3f,\"remote_chunk_hash_ms\":%.3f,"
                    "\"remote_write_ms\":%.3f,\"remote_full_verify_ms\":%.3f,"
                    "\"remote_data_sync_ms\":%.3f,\"remote_file_sync_ms\":%.3f,"
                    "\"remote_publish_ms\":%.3f,\"remote_directory_sync_ms\":%.3f,"
                    "\"remote_accepted_chunks\":%llu,\"remote_duplicate_chunks\":%llu}\n",
                    pp::k_version, capability.c_str(), remote_backend.c_str(),
                    provision.cache_hit ? "cache_hit" : "uploaded",
                    static_cast<unsigned long long>(session_epoch),
                    static_cast<unsigned long long>(config.route_epoch),
                    static_cast<unsigned long long>(config.generation),
                    static_cast<unsigned long long>(provision.ticket_id),
                    static_cast<unsigned long long>(local_model_bytes),
                    pp::hex_sha256(local_model_sha256).c_str(),
                    pp::hex_sha256(stage_spec.manifest_sha256).c_str(), config.stage_chunk_bytes,
                    stage_spec.chunks.size(), static_cast<unsigned long long>(provision.resume_offset),
                    provision.chunks_sent, static_cast<unsigned long long>(provision.bytes_sent),
                    config.stage_window, static_cast<unsigned long long>(provision.attempted_bytes),
                    static_cast<unsigned long long>(provision.socket_written_bytes),
                    static_cast<unsigned long long>(provision.acked_durable_bytes),
                    static_cast<unsigned long long>(provision.socket_written_bytes >= provision.acked_durable_bytes
                            ? provision.socket_written_bytes - provision.acked_durable_bytes : 0),
                    static_cast<unsigned long long>(provision.retried_bytes),
                    source_hash_us / 1000.0, provision.source_chunk_hash_us / 1000.0,
                    provision.e2e_us / 1000.0, useful_goodput, provision.resume_scan_us / 1000.0,
                    provision.remote.chunk_hash_us / 1000.0, provision.remote.write_us / 1000.0,
                    provision.remote.full_verify_us / 1000.0,
                    provision.remote.data_sync_us / 1000.0, provision.remote.file_sync_us / 1000.0,
                    provision.remote.publish_us / 1000.0,
                    provision.remote.directory_sync_us / 1000.0,
                    static_cast<unsigned long long>(provision.remote.accepted_chunks),
                    static_cast<unsigned long long>(provision.remote.duplicate_chunks));
            return 0;
        }
    }
    if (!config.provision && (remote_features & 2) == 0) {
        std::fprintf(stderr, "error: remote pre-staged model source is unavailable\n");
        return 4;
    }

    struct stat current_local_stat = {};
    if (fstat(local_model.get(), &current_local_stat) != 0 ||
        current_local_stat.st_dev != local_stat.st_dev || current_local_stat.st_ino != local_stat.st_ino ||
        current_local_stat.st_size != local_stat.st_size ||
        current_local_stat.st_mtim.tv_sec != local_stat.st_mtim.tv_sec ||
        current_local_stat.st_mtim.tv_nsec != local_stat.st_mtim.tv_nsec) {
        std::fprintf(stderr, "error: local model changed while provisioning\n");
        return 4;
    }

    pp::Writer prepare_writer;
    prepare_writer.u64(config.island_id);
    prepare_writer.u32(config.token_count);
    const pp::ModelSource requested_source = config.provision
            ? pp::ModelSource::published_store : pp::ModelSource::prestaged;
    prepare_writer.u32(static_cast<uint32_t>(requested_source));
    prepare_writer.string(config.prefix);
    prepare_writer.u64(local_model_bytes);
    prepare_writer.bytes(local_model_sha256.data(), local_model_sha256.size());
    const uint64_t prepare_e2e_start = now_us();
    if (!client.call(pp::Opcode::prepare, prepare_writer.data(), response, error)) {
        std::fprintf(stderr, "error: PREPARE: %s\n", error.c_str());
        return 4;
    }
    const uint64_t prepare_e2e_us = now_us() - prepare_e2e_start;
    PreparedInfo prepared;
    if (!parse_prepared(response.payload, prepared) || prepared.island_id != config.island_id ||
        prepared.token_count != config.token_count || prepared.n_embd == 0 || prepared.n_ff == 0 ||
        prepared.model_bytes != local_model_bytes || prepared.model_sha256 != local_model_sha256 ||
        prepared.model_source != requested_source) {
        std::fprintf(stderr, "error: malformed PREPARED response\n");
        if (!release_remote(client, config.island_id, error)) {
            std::fprintf(stderr, "error: RELEASE after malformed PREPARED: %s\n", error.c_str());
            return 7;
        }
        return 4;
    }

    ProductionOracle oracle;
    const std::string oracle_path = "/proc/self/fd/" + std::to_string(local_model.get());
    if (!oracle.init(oracle_path, config.prefix, error)) {
        std::fprintf(stderr, "error: production oracle init: %s\n", error.c_str());
        if (!release_remote(client, config.island_id, error)) {
            std::fprintf(stderr, "error: RELEASE after oracle failure: %s\n", error.c_str());
            return 7;
        }
        return 5;
    }
    if (static_cast<uint64_t>(oracle.n_embd()) != prepared.n_embd) {
        std::fprintf(stderr, "error: remote and production graph shapes differ\n");
        if (!release_remote(client, config.island_id, error)) {
            std::fprintf(stderr, "error: RELEASE after oracle failure: %s\n", error.c_str());
            return 7;
        }
        return 5;
    }
    std::array<OracleCase, 2> oracle_cases;
    for (size_t i = 0; i < oracle_cases.size(); ++i) {
        if (!oracle.run(config.token_count, static_cast<uint32_t>(17 + i * 97), oracle_cases[i], error)) {
            std::fprintf(stderr, "error: production oracle case %zu: %s\n", i, error.c_str());
            if (!release_remote(client, config.island_id, error)) {
                std::fprintf(stderr, "error: RELEASE after oracle failure: %s\n", error.c_str());
                return 7;
            }
            return 5;
        }
    }

    std::vector<uint64_t> e2e_values;
    std::vector<uint64_t> compute_values;
    std::vector<uint64_t> transport_values;
    double worst_l2 = 0.0;
    bool correctness_ok = true;
    for (uint32_t iteration = 0; iteration < config.repeat; ++iteration) {
        const OracleCase & oracle_case = oracle_cases[iteration % oracle_cases.size()];
        RemoteExecution execution;
        if (!execute_remote(client, config.island_id, oracle_case.input, execution, error)) {
            std::fprintf(stderr, "error: EXECUTE[%u]: %s\n", iteration, error.c_str());
            return 6;
        }
        bool finite = false;
        const double l2 = pp::relative_l2(execution.output, oracle_case.expected, finite);
        worst_l2 = std::max(worst_l2, l2);
        correctness_ok = correctness_ok && finite && l2 <= 5e-3;
        e2e_values.push_back(execution.e2e_us);
        compute_values.push_back(execution.compute_us);
        const uint64_t phone_accounted = execution.input_set_us + execution.compute_us + execution.output_get_us;
        transport_values.push_back(execution.e2e_us > phone_accounted ? execution.e2e_us - phone_accounted : 0);
    }

    if (config.release || !correctness_ok) {
        if (!release_remote(client, config.island_id, error)) {
            std::fprintf(stderr, "error: RELEASE: %s\n", error.c_str());
            return 7;
        }
    }
    if (config.shutdown) {
        if (!client.call(pp::Opcode::shutdown, {}, response, error)) {
            std::fprintf(stderr, "error: SHUTDOWN: %s\n", error.c_str());
            return 7;
        }
    } else {
        client.call(pp::Opcode::close, {}, response, error);
    }

    const Stats e2e = stats(e2e_values);
    const Stats compute = stats(compute_values);
    const Stats transport = stats(transport_values);
    const char * verdict = correctness_ok
            ? (config.provision ? "DYNAMIC_FFN_PASS" : "PRESTAGED_FFN_PASS")
            : "FAIL_CORRECTNESS";
    const char * provision_outcome = !config.provision ? "not_requested"
            : (provision.cache_hit ? "cache_hit" : "uploaded");
    const std::string manifest_sha = config.provision
            ? pp::hex_sha256(stage_spec.manifest_sha256) : "";
    const double stage_goodput = provision.e2e_us == 0 ? 0.0 :
            provision.bytes_sent * 1000000.0 / provision.e2e_us / 1048576.0;
    std::printf(
            "{\"record_schema_version\":1,\"verdict\":\"%s\",\"protocol_version\":%u,"
            "\"route\":\"%s\","
            "\"capability\":\"%s\",\"session_epoch\":%llu,\"route_epoch\":%llu,"
            "\"residency_generation_start\":%llu,\"residency_generation_end\":%llu,"
            "\"prefix\":\"%s\",\"model_source\":\"%s\","
            "\"oracle\":\"production_gemma4_cb_eval\","
            "\"M\":%u,\"n_embd\":%llu,\"n_ff\":%llu,"
            "\"repetitions\":%u,\"rel_l2_max\":%.9g,\"prepare_e2e_ms\":%.3f,"
            "\"phone_verify_ms\":%.3f,\"phone_load_ms\":%.3f,\"phone_upload_ms\":%.3f,"
            "\"phone_warmup_ms\":%.3f,\"model_bytes\":%llu,\"model_sha256\":\"%s\","
            "\"source_hash_ms\":%.3f,\"provision_outcome\":\"%s\","
            "\"stage_manifest_sha256\":\"%s\",\"stage_chunk_bytes\":%u,\"stage_window\":%u,"
            "\"stage_chunk_count\":%zu,\"stage_ticket_id\":%llu,"
            "\"stage_resume_offset\":%llu,\"stage_chunks_sent\":%u,"
            "\"stage_bytes_sent\":%llu,\"stage_attempted_bytes\":%llu,"
            "\"stage_socket_written_bytes\":%llu,\"stage_acked_durable_bytes\":%llu,"
            "\"stage_wasted_bytes\":%llu,\"stage_retried_bytes\":%llu,\"stage_e2e_ms\":%.3f,"
            "\"stage_useful_goodput_mib_s\":%.3f,\"stage_resume_scan_ms\":%.3f,"
            "\"stage_source_chunk_hash_ms\":%.3f,"
            "\"stage_host_read_ms\":%.3f,\"stage_host_hash_ms\":%.3f,"
            "\"stage_host_prefix_hash_ms\":%.3f,"
            "\"stage_host_send_ms\":%.3f,\"stage_host_frame_sha_ms\":%.3f,"
            "\"stage_host_socket_send_ms\":%.3f,\"profile_transport\":%s,"
            "\"stage_host_ack_wait_ms\":%.3f,"
            "\"stage_remote_chunk_hash_ms\":%.3f,"
            "\"stage_remote_write_ms\":%.3f,\"stage_remote_data_sync_ms\":%.3f,"
            "\"stage_remote_full_verify_ms\":%.3f,\"stage_remote_file_sync_ms\":%.3f,"
            "\"stage_remote_publish_ms\":%.3f,\"stage_remote_directory_sync_ms\":%.3f,"
            "\"stage_remote_accepted_chunks\":%llu,\"stage_remote_duplicate_chunks\":%llu,"
            "\"resident_mib\":%.3f,\"warm_e2e_p50_ms\":%.3f,\"warm_e2e_p95_ms\":%.3f,"
            "\"warm_e2e_cov\":%.6f,\"phone_compute_p50_ms\":%.3f,"
            "\"transport_and_protocol_p50_ms\":%.3f}\n",
            verdict, pp::k_version, remote_backend.c_str(), capability.c_str(),
            static_cast<unsigned long long>(session_epoch),
            static_cast<unsigned long long>(config.route_epoch),
            static_cast<unsigned long long>(config.generation),
            static_cast<unsigned long long>(client.config.generation),
            config.prefix.c_str(), config.provision ? "published_store" : "prestaged",
            config.token_count,
            static_cast<unsigned long long>(prepared.n_embd), static_cast<unsigned long long>(prepared.n_ff),
            config.repeat, worst_l2, prepare_e2e_us / 1000.0, prepared.verify_us / 1000.0,
            prepared.load_us / 1000.0, prepared.upload_us / 1000.0, prepared.warmup_us / 1000.0,
            static_cast<unsigned long long>(local_model_bytes),
            pp::hex_sha256(prepared.model_sha256).c_str(), source_hash_us / 1000.0,
            provision_outcome, manifest_sha.c_str(), config.provision ? config.stage_chunk_bytes : 0,
            config.stage_window,
            config.provision ? stage_spec.chunks.size() : 0,
            static_cast<unsigned long long>(provision.ticket_id),
            static_cast<unsigned long long>(provision.resume_offset), provision.chunks_sent,
            static_cast<unsigned long long>(provision.bytes_sent),
            static_cast<unsigned long long>(provision.attempted_bytes),
            static_cast<unsigned long long>(provision.socket_written_bytes),
            static_cast<unsigned long long>(provision.acked_durable_bytes),
            static_cast<unsigned long long>(provision.socket_written_bytes >= provision.acked_durable_bytes
                    ? provision.socket_written_bytes - provision.acked_durable_bytes : 0),
            static_cast<unsigned long long>(provision.retried_bytes),
            provision.e2e_us / 1000.0,
            stage_goodput, provision.resume_scan_us / 1000.0,
            provision.source_chunk_hash_us / 1000.0,
            provision.host_read_us / 1000.0, provision.host_hash_us / 1000.0,
            provision.host_prefix_hash_us / 1000.0,
            provision.host_send_us / 1000.0, provision.host_frame_sha_us / 1000.0,
            provision.host_socket_send_us / 1000.0, config.profile_transport ? "true" : "false",
            provision.host_ack_wait_us / 1000.0,
            provision.remote.chunk_hash_us / 1000.0,
            provision.remote.write_us / 1000.0, provision.remote.data_sync_us / 1000.0,
            provision.remote.full_verify_us / 1000.0, provision.remote.file_sync_us / 1000.0,
            provision.remote.publish_us / 1000.0, provision.remote.directory_sync_us / 1000.0,
            static_cast<unsigned long long>(provision.remote.accepted_chunks),
            static_cast<unsigned long long>(provision.remote.duplicate_chunks),
            prepared.resident_bytes / 1048576.0, e2e.p50, e2e.p95, e2e.cov,
            compute.p50, transport.p50);
    return correctness_ok ? 0 : 8;
}
