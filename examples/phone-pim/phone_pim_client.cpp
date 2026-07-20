#include "phone_pim_client.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <utility>

namespace phone_pim {
namespace {

uint64_t now_us() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

bool bounded_label(const std::string & value) {
    if (value.empty() || value.size() > 128) {
        return false;
    }
    for (char ch : value) {
        if (!((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
              (ch >= '0' && ch <= '9') || ch == '_' || ch == '-' || ch == '.' || ch == ' ')) {
            return false;
        }
    }
    return true;
}

bool bounded_text(const std::string & value, size_t max_size) {
    if (value.empty() || value.size() > max_size) {
        return false;
    }
    for (unsigned char ch : value) {
        if (ch < 0x20 || ch > 0x7e) {
            return false;
        }
    }
    return true;
}

bool canonical_prefix(const std::string & value) {
    if (value.size() < 5 || value.compare(0, 4, "blk.") != 0 || value[4] == '0') {
        return false;
    }
    for (size_t i = 4; i < value.size(); ++i) {
        if (value[i] < '0' || value[i] > '9') {
            return false;
        }
    }
    return true;
}

bool finite_floats(const std::vector<float> & values) {
    for (float value : values) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        if ((bits & 0x7f800000U) == 0x7f800000U) {
            return false;
        }
    }
    return true;
}

void encode_floats(Writer & writer, const std::vector<float> & values) {
    for (float value : values) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        writer.u32(bits);
    }
}

bool decode_floats(Reader & reader, uint64_t count, std::vector<float> & values) {
    if (count > SIZE_MAX / sizeof(float) || reader.remaining() != count * sizeof(float)) {
        return false;
    }
    values.resize(static_cast<size_t>(count));
    for (float & value : values) {
        uint32_t bits = 0;
        if (!reader.u32(bits)) {
            return false;
        }
        std::memcpy(&value, &bits, sizeof(bits));
    }
    return reader.done() && finite_floats(values);
}

bool parse_hello(const std::vector<uint8_t> & payload, HelloInfo & info) {
    Reader reader(payload);
    return reader.u32(info.protocol_version) && reader.u64(info.session_epoch) &&
           reader.u64(info.max_payload) && reader.string(info.backend, 128) &&
           reader.u64(info.feature_bits) && reader.u64(info.max_model_bytes) &&
           reader.u64(info.max_chunk_bytes) && reader.u32(info.max_chunks) &&
           reader.string(info.capability, 128) && reader.done() &&
           info.protocol_version == k_version && info.session_epoch != 0 &&
           info.max_payload != 0 && info.max_payload <= k_default_max_payload &&
           bounded_label(info.backend) && bounded_label(info.capability) &&
           (info.capability == "prestaged_gemma4_dense_ffn_v3" ||
            info.capability == "sequential_dynamic_gemma4_dense_ffn_v3");
}

bool parse_status(const std::vector<uint8_t> & payload, StatusInfo & info) {
    Reader reader(payload);
    uint32_t ready = 0;
    uint32_t store_enabled = 0;
    uint32_t staging_active = 0;
    if (!reader.u32(ready) || ready > 1 || !reader.u64(info.island_id) ||
        !reader.u64(info.residency_generation) || !reader.u64(info.n_embd) ||
        !reader.u64(info.n_ff) || !reader.u32(info.token_count) ||
        !reader.string(info.prefix, 256) || !reader.u32(store_enabled) || store_enabled > 1 ||
        !reader.u32(staging_active) || staging_active > 1 ||
        !reader.u64(info.staging_ticket_id) || !reader.u64(info.staging_verified_bytes) ||
        !reader.u32(info.staging_next_chunk) ||
        !reader.bytes(info.staging_manifest_sha256.data(), info.staging_manifest_sha256.size()) ||
        !reader.done() || info.residency_generation == 0) {
        return false;
    }
    info.ready = ready != 0;
    info.store_enabled = store_enabled != 0;
    info.staging_active = staging_active != 0;
    const std::array<uint8_t, 32> zero_digest = {};
    if (info.ready) {
        if (info.island_id == 0 || info.n_embd == 0 || info.n_ff == 0 ||
            info.token_count == 0 || !canonical_prefix(info.prefix)) {
            return false;
        }
    } else if (info.island_id != 0 || info.n_embd != 0 || info.n_ff != 0 ||
               info.token_count != 0 || !info.prefix.empty()) {
        return false;
    }
    if (info.staging_active) {
        if (!info.store_enabled || info.staging_ticket_id == 0 ||
            info.staging_manifest_sha256 == zero_digest) {
            return false;
        }
    } else if (info.staging_ticket_id != 0 || info.staging_verified_bytes != 0 ||
               info.staging_next_chunk != 0 || info.staging_manifest_sha256 != zero_digest) {
        return false;
    }
    return true;
}

bool parse_prepared(const std::vector<uint8_t> & payload, PreparedInfo & info) {
    Reader reader(payload);
    uint32_t source = 0;
    const std::array<uint8_t, 32> zero_digest = {};
    if (!reader.u64(info.island_id) || !reader.u64(info.n_embd) || !reader.u64(info.n_ff) ||
        !reader.u32(info.token_count) || !reader.u64(info.model_bytes) ||
        !reader.bytes(info.model_sha256.data(), info.model_sha256.size()) || !reader.u32(source) ||
        !reader.u64(info.verify_us) || !reader.u64(info.weight_bytes) ||
        !reader.u64(info.resident_bytes) || !reader.u64(info.load_us) ||
        !reader.u64(info.upload_us) || !reader.u64(info.warmup_us) ||
        !reader.string(info.backend_description, 1024) || !reader.done() ||
        (source != static_cast<uint32_t>(ModelSource::prestaged) &&
         source != static_cast<uint32_t>(ModelSource::published_store)) ||
        info.island_id == 0 || info.n_embd == 0 || info.n_ff == 0 ||
        info.token_count == 0 || info.model_bytes == 0 || info.model_sha256 == zero_digest ||
        info.weight_bytes == 0 ||
        info.resident_bytes == 0 || !bounded_text(info.backend_description, 1024)) {
        return false;
    }
    info.model_source = static_cast<ModelSource>(source);
    return true;
}

} // namespace

ClientSession::ClientSession(ClientOptions options) : options_(std::move(options)) {}

bool ClientSession::connect(HelloInfo & hello, std::string & error) {
    if (connected()) {
        error = "client session is already connected";
        return false;
    }
    if (options_.host.empty() || options_.port == 0 || options_.route_epoch == 0 ||
        options_.residency_generation == 0 || options_.max_payload == 0 ||
        options_.max_payload > k_default_max_payload || options_.timeout_ms <= 0) {
        error = "invalid client options";
        return false;
    }
    socket_ = connect_tcp(options_.host, options_.port, options_.timeout_ms, error);
    if (!socket_.valid()) {
        return false;
    }
    Frame response;
    HelloInfo parsed;
    if (!call(Opcode::hello, {}, response, error)) {
        return false;
    }
    if (!parse_hello(response.payload, parsed)) {
        error = "malformed HELLO response";
        poison();
        return false;
    }
    hello = std::move(parsed);
    session_epoch_ = hello.session_epoch;
    options_.max_payload = std::min(options_.max_payload, hello.max_payload);
    generation_synchronized_ = false;
    return true;
}

bool ClientSession::status(StatusInfo & status_info, std::string & error) {
    Frame response;
    if (!connected() || !call(Opcode::status, {}, response, error)) {
        if (error.empty()) error = "client session is not connected";
        return false;
    }
    StatusInfo parsed;
    if (!parse_status(response.payload, parsed)) {
        error = "malformed STATUS response";
        poison();
        return false;
    }
    status_info = std::move(parsed);
    options_.residency_generation = status_info.residency_generation;
    generation_synchronized_ = true;
    prepared_island_id_ = 0;
    prepared_elements_ = 0;
    return true;
}

bool ClientSession::prepare(
        const PrepareRequest & request,
        PreparedInfo & prepared,
        std::string & error) {
    const std::array<uint8_t, 32> zero_digest = {};
    if (!connected() || !generation_synchronized_ || request.island_id == 0 || request.token_count == 0 ||
        request.token_count > 1024 || request.prefix.empty() || request.prefix.size() > 256 ||
        !canonical_prefix(request.prefix) ||
        request.model_bytes == 0 || request.model_sha256 == zero_digest ||
        (request.model_source != ModelSource::prestaged &&
         request.model_source != ModelSource::published_store)) {
        error = "invalid PREPARE request";
        return false;
    }
    Writer writer;
    writer.u64(request.island_id);
    writer.u32(request.token_count);
    writer.u32(static_cast<uint32_t>(request.model_source));
    writer.string(request.prefix);
    writer.u64(request.model_bytes);
    writer.bytes(request.model_sha256.data(), request.model_sha256.size());
    Frame response;
    PreparedInfo parsed;
    if (!call(Opcode::prepare, writer.data(), response, error)) {
        return false;
    }
    if (!parse_prepared(response.payload, parsed)) {
        error = "malformed PREPARE response";
        poison();
        return false;
    }
    if (parsed.island_id != request.island_id || parsed.token_count != request.token_count ||
        parsed.model_source != request.model_source || parsed.model_bytes != request.model_bytes ||
        parsed.model_sha256 != request.model_sha256 ||
        parsed.n_embd > std::numeric_limits<uint64_t>::max() / parsed.token_count) {
        error = "PREPARE response identity mismatch";
        poison();
        return false;
    }
    const uint64_t elements = parsed.n_embd * parsed.token_count;
    if (elements == 0 || options_.max_payload < 40 ||
        elements > (options_.max_payload - 40) / sizeof(float)) {
        error = "PREPARE response activation shape exceeds the negotiated payload";
        poison();
        return false;
    }
    prepared = std::move(parsed);
    prepared_island_id_ = request.island_id;
    prepared_elements_ = elements;
    return true;
}

bool ClientSession::execute(
        uint64_t island_id,
        const std::vector<float> & input,
        ExecutionInfo & execution,
        std::string & error) {
    if (!connected() || !generation_synchronized_ || island_id == 0 ||
        island_id != prepared_island_id_ || input.size() != prepared_elements_ || input.empty() ||
        !finite_floats(input) || options_.max_payload < 40 ||
        input.size() > (options_.max_payload - 40) / sizeof(float)) {
        error = "invalid EXECUTE request";
        return false;
    }
    const uint64_t operation_start = now_us();
    Writer writer;
    writer.u64(island_id);
    writer.u64(input.size());
    encode_floats(writer, input);
    Frame response;
    if (!call(Opcode::execute, writer.data(), response, error)) {
        return false;
    }
    ExecutionInfo parsed;
    Reader reader(response.payload);
    uint64_t response_island = 0;
    uint64_t element_count = 0;
    if (!reader.u64(response_island) || !reader.u64(element_count) ||
        response_island != island_id || element_count != input.size() ||
        !reader.u64(parsed.input_set_us) || !reader.u64(parsed.compute_us) ||
        !reader.u64(parsed.output_get_us) ||
        !decode_floats(reader, element_count, parsed.output)) {
        error = "malformed EXECUTE response";
        poison();
        return false;
    }
    uint64_t remote_total_us = parsed.input_set_us;
    if (parsed.compute_us > std::numeric_limits<uint64_t>::max() - remote_total_us) {
        error = "EXECUTE timing fields overflow";
        poison();
        return false;
    }
    remote_total_us += parsed.compute_us;
    if (parsed.output_get_us > std::numeric_limits<uint64_t>::max() - remote_total_us) {
        error = "EXECUTE timing fields overflow";
        poison();
        return false;
    }
    remote_total_us += parsed.output_get_us;
    parsed.e2e_us = now_us() - operation_start;
    if (remote_total_us > parsed.e2e_us) {
        error = "EXECUTE worker timing exceeds host E2E time";
        poison();
        return false;
    }
    execution = std::move(parsed);
    return true;
}

bool ClientSession::release(uint64_t island_id, uint64_t & next_generation, std::string & error) {
    if (!connected() || !generation_synchronized_ || island_id == 0) {
        error = "invalid RELEASE request";
        return false;
    }
    Writer writer;
    writer.u64(island_id);
    Frame response;
    if (!call(Opcode::release, writer.data(), response, error)) {
        return false;
    }
    Reader reader(response.payload);
    uint64_t parsed_generation = 0;
    if (!reader.u64(parsed_generation) || !reader.done() ||
        options_.residency_generation == std::numeric_limits<uint64_t>::max() ||
        parsed_generation != options_.residency_generation + 1) {
        error = "malformed RELEASE response";
        poison();
        return false;
    }
    next_generation = parsed_generation;
    options_.residency_generation = parsed_generation;
    prepared_island_id_ = 0;
    prepared_elements_ = 0;
    return true;
}

bool ClientSession::close(std::string & error) {
    if (!connected() || !generation_synchronized_) {
        poison();
        return true;
    }
    Frame response;
    if (!call(Opcode::close, {}, response, error)) {
        return false;
    }
    if (!response.payload.empty()) {
        error = "malformed CLOSE response";
        poison();
        return false;
    }
    poison();
    return true;
}

bool ClientSession::call(
        Opcode opcode,
        const std::vector<uint8_t> & payload,
        Frame & response,
        std::string & error) {
    if (!socket_.valid()) {
        error = "client socket is not connected";
        return false;
    }
    if (payload.size() > options_.max_payload || payload.size() > opcode_payload_limit(opcode)) {
        error = "request payload exceeds negotiated maximum";
        return false;
    }
    if (command_seq_ == std::numeric_limits<uint64_t>::max() ||
        request_id_ == std::numeric_limits<uint64_t>::max()) {
        error = "client command sequence exhausted";
        return false;
    }
    Header sent;
    sent.opcode = opcode;
    sent.request_id = ++request_id_;
    sent.command_seq = ++command_seq_;
    sent.session_epoch = opcode == Opcode::hello ? 0 : session_epoch_;
    sent.route_epoch = options_.route_epoch;
    sent.residency_generation = opcode == Opcode::hello ? 0 : options_.residency_generation;
    if (!send_frame(socket_.get(), sent, payload, error)) {
        poison();
        return false;
    }
    if (receive_frame(socket_.get(), response, options_.max_payload, error) != ReceiveResult::ok) {
        if (error.empty()) error = "connection closed before a response";
        poison();
        return false;
    }
    if ((response.header.flags & flag_response) == 0 ||
        response.header.request_id != sent.request_id ||
        response.header.command_seq != sent.command_seq ||
        response.header.session_epoch != sent.session_epoch ||
        response.header.route_epoch != sent.route_epoch ||
        response.header.residency_generation != sent.residency_generation) {
        error = "response header does not match request";
        poison();
        return false;
    }
    if ((response.header.flags & flag_error) != 0 || response.header.opcode == Opcode::error) {
        ErrorCode code = ErrorCode::internal;
        std::string remote_error;
        if (!parse_error_payload(response.payload, code, remote_error)) {
            error = "malformed remote error";
            poison();
        } else {
            error = "remote error " + std::to_string(static_cast<uint32_t>(code)) + ": " + remote_error;
            generation_synchronized_ = false;
            prepared_island_id_ = 0;
            prepared_elements_ = 0;
        }
        return false;
    }
    if (response.header.opcode != opcode || response.header.flags != flag_response) {
        error = "unexpected response opcode or flags";
        poison();
        return false;
    }
    return true;
}

void ClientSession::poison() {
    socket_.reset();
    session_epoch_ = 0;
    generation_synchronized_ = false;
    prepared_island_id_ = 0;
    prepared_elements_ = 0;
}

} // namespace phone_pim
