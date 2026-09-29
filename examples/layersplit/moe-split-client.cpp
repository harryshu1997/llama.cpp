#include "moe-split-client.h"

#include "moe-split-protocol.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

namespace moe_split {
namespace {

using steady_clock = std::chrono::steady_clock;

double now_ms() {
    return std::chrono::duration<double, std::milli>(
            steady_clock::now().time_since_epoch()).count();
}

bool send_exact(int fd, const void * data, size_t size) {
    const uint8_t * ptr = static_cast<const uint8_t *>(data);
    while (size > 0) {
        const ssize_t count = send(fd, ptr, size, MSG_NOSIGNAL);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            return false;
        }
        ptr += count;
        size -= static_cast<size_t>(count);
    }
    return true;
}

bool receive_exact(int fd, void * data, size_t size) {
    uint8_t * ptr = static_cast<uint8_t *>(data);
    while (size > 0) {
        const ssize_t count = recv(fd, ptr, size, 0);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            return false;
        }
        ptr += count;
        size -= static_cast<size_t>(count);
    }
    return true;
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    return values[static_cast<size_t>(
            fraction * static_cast<double>(values.size() - 1))];
}

} // namespace

client::client(client_config config) : config_(std::move(config)) {}

client::~client() {
    finish();
    if (fd_ >= 0) {
        close(fd_);
    }
}

bool client::connect(std::string & error) {
    if (fd_ >= 0 || config_.host.empty() || config_.port <= 0 ||
        config_.port > 65535 || config_.layer_mask == 0 ||
        config_.n_embd == 0 || config_.timeout_ms <= 0) {
        error = "invalid MoE split client configuration";
        return false;
    }

    addrinfo hints = {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo * addresses = nullptr;
    const std::string port = std::to_string(config_.port);
    if (getaddrinfo(config_.host.c_str(), port.c_str(), &hints, &addresses) != 0) {
        error = "cannot resolve MoE split worker";
        return false;
    }
    for (addrinfo * current = addresses; current != nullptr; current = current->ai_next) {
        const int candidate = socket(
                current->ai_family, current->ai_socktype, current->ai_protocol);
        if (candidate < 0) {
            continue;
        }
        if (::connect(candidate, current->ai_addr, current->ai_addrlen) == 0) {
            fd_ = candidate;
            break;
        }
        close(candidate);
    }
    freeaddrinfo(addresses);
    if (fd_ < 0) {
        error = "cannot connect to MoE split worker";
        return false;
    }

    const int one = 1;
    setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    timeval timeout = {};
    timeout.tv_sec = config_.timeout_ms / 1000;
    timeout.tv_usec = (config_.timeout_ms % 1000) * 1000;
    setsockopt(fd_, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

    hello_request request = {};
    request.magic = protocol_magic;
    request.version = protocol_version;
    request.message = static_cast<uint16_t>(message_type::hello_request);
    request.layer_mask = config_.layer_mask;
    request.n_embd = config_.n_embd;
    request.flags = config_.f16_io ? flag_f16_io : 0;
    hello_response response = {};
    if (!send_exact(fd_, &request, sizeof(request)) ||
        !receive_exact(fd_, &response, sizeof(response))) {
        error = "MoE split HELLO exchange failed";
        close(fd_);
        fd_ = -1;
        return false;
    }
    if (response.magic != protocol_magic || response.version != protocol_version ||
        response.message != static_cast<uint16_t>(message_type::hello_response) ||
        response.status != 0 || response.flags != request.flags ||
        response.layer_mask != config_.layer_mask || response.n_embd != config_.n_embd ||
        response.n_ff_exp == 0 || response.n_expert == 0 ||
        response.n_expert_used == 0 || response.n_expert_used > response.n_expert ||
        response.layer_count != static_cast<uint32_t>(__builtin_popcountll(config_.layer_mask))) {
        error = "MoE split HELLO identity mismatch";
        close(fd_);
        fd_ = -1;
        return false;
    }
    n_ff_exp_ = response.n_ff_exp;
    n_expert_ = response.n_expert;
    n_expert_used_ = response.n_expert_used;
    layer_count_ = response.layer_count;
    weight_hash_ = response.weight_hash;
    input_.resize(config_.n_embd);
    output_.resize(config_.n_embd);
    fprintf(stderr,
            "[moe-split] connected %s:%d layers=%u mask=%016llx K=%u "
            "NFF=%u experts=%u top_k=%u io=%s hash=%016llx\n",
            config_.host.c_str(), config_.port, layer_count_,
            static_cast<unsigned long long>(config_.layer_mask), config_.n_embd,
            n_ff_exp_, n_expert_, n_expert_used_, config_.f16_io ? "f16" : "f32",
            static_cast<unsigned long long>(weight_hash_));
    return true;
}

bool client::exchange(uint32_t request_id, int layer, std::string & error) {
    std::vector<ggml_fp16_t> encoded_input;
    const void * input_payload = input_.data();
    size_t payload_bytes = input_.size() * sizeof(float);
    if (config_.f16_io) {
        encoded_input.resize(input_.size());
        ggml_fp32_to_fp16_row(input_.data(), encoded_input.data(), encoded_input.size());
        input_payload = encoded_input.data();
        payload_bytes = encoded_input.size() * sizeof(ggml_fp16_t);
    }

    execute_request request = {};
    request.magic = protocol_magic;
    request.version = protocol_version;
    request.message = static_cast<uint16_t>(message_type::execute_request);
    request.request_id = request_id;
    request.layer = layer;
    request.elements = config_.n_embd;
    request.payload_bytes = static_cast<uint32_t>(payload_bytes);
    request.payload_hash = hash_bytes(input_payload, payload_bytes);
    std::vector<uint8_t> packet(sizeof(request) + payload_bytes);
    memcpy(packet.data(), &request, sizeof(request));
    memcpy(packet.data() + sizeof(request), input_payload, payload_bytes);

    const double started = now_ms();
    execute_response response = {};
    if (!send_exact(fd_, packet.data(), packet.size()) ||
        !receive_exact(fd_, &response, sizeof(response))) {
        error = "MoE split EXECUTE header exchange failed";
        return false;
    }
    if (response.magic != protocol_magic || response.version != protocol_version ||
        response.message != static_cast<uint16_t>(message_type::execute_response) ||
        response.status != 0 || response.request_id != request_id ||
        response.layer != layer || response.elements != config_.n_embd ||
        response.payload_bytes != payload_bytes) {
        error = "MoE split EXECUTE response mismatch";
        return false;
    }
    std::vector<uint8_t> payload(payload_bytes);
    if (!receive_exact(fd_, payload.data(), payload.size()) ||
        response.payload_hash != hash_bytes(payload.data(), payload.size())) {
        error = "MoE split EXECUTE payload read failed";
        return false;
    }
    rpc_ms_ = now_ms() - started;
    compute_ms_ = static_cast<double>(response.compute_us) / 1000.0;
    if (config_.f16_io) {
        ggml_fp16_to_fp32_row(
                reinterpret_cast<const ggml_fp16_t *>(payload.data()),
                output_.data(), output_.size());
    } else {
        memcpy(output_.data(), payload.data(), payload.size());
    }
    return true;
}

bool client::parse_named_layer(const char * name, const char * prefix, int & layer) const {
    const size_t prefix_size = strlen(prefix);
    if (strncmp(name, prefix, prefix_size) != 0 || name[prefix_size] != '-') {
        return false;
    }
    errno = 0;
    char * end = nullptr;
    const long parsed = strtol(name + prefix_size + 1, &end, 10);
    if (errno != 0 || end == name + prefix_size + 1 || *end != '\0' ||
        parsed < 0 || parsed >= 64 ||
        (config_.layer_mask & (UINT64_C(1) << parsed)) == 0) {
        return false;
    }
    layer = static_cast<int>(parsed);
    return true;
}

bool client::eval(ggml_tensor * tensor, bool ask) {
    if (tensor == nullptr || failed_ || fd_ < 0) {
        return false;
    }
    const bool decode_shape =
            tensor->type == GGML_TYPE_F32 && tensor->ne[0] == config_.n_embd &&
            tensor->ne[1] == 1;
    int input_layer = -1;
    int output_layer = -1;
    const bool is_input = decode_shape &&
            parse_named_layer(tensor->name, "moe_phone_input", input_layer);
    const bool is_output = decode_shape &&
            parse_named_layer(tensor->name, "moe_phone_output", output_layer);
    if (ask) {
        return is_input || is_output;
    }
    if (!is_input && !is_output) {
        return true;
    }
    if (tensor->buffer == nullptr) {
        set_error("MoE split callback tensor has no storage");
        return false;
    }

    if (is_input) {
        if (pending_) {
            set_error("MoE split input arrived with a request already pending");
            return false;
        }
        ggml_backend_tensor_get(
                tensor, input_.data(), 0, input_.size() * sizeof(float));
        const uint32_t request_id = next_request_id_++;
        thread_ok_ = false;
        thread_error_.clear();
        launch_ms_ = now_ms();
        pending_ = true;
        pending_layer_ = input_layer;
        thread_ = std::thread([this, request_id, input_layer]() {
            thread_ok_ = exchange(request_id, input_layer, thread_error_);
        });
        return true;
    }

    if (!pending_ || !thread_.joinable() || output_layer != pending_layer_) {
        set_error("MoE split output arrived without a matching request");
        return false;
    }
    const double host_ready_ms = now_ms();
    thread_.join();
    const double joined_ms = now_ms();
    pending_ = false;
    pending_layer_ = -1;
    if (!thread_ok_) {
        set_error(thread_error_.empty() ? "MoE split worker request failed" : thread_error_);
        return false;
    }
    ggml_backend_tensor_set(
            tensor, output_.data(), 0, output_.size() * sizeof(float));
    rpc_samples_.push_back(rpc_ms_);
    compute_samples_.push_back(compute_ms_);
    host_branch_samples_.push_back(host_ready_ms - launch_ms_);
    wait_samples_.push_back(joined_ms - host_ready_ms);
    overlap_samples_.push_back(joined_ms - launch_ms_);
    return true;
}

void client::finish() {
    if (thread_.joinable()) {
        thread_.join();
    }
    if (pending_) {
        pending_ = false;
        pending_layer_ = -1;
        set_error("MoE split decode ended with a pending request");
    }
}

void client::set_error(const std::string & error) {
    failed_ = true;
    if (error_.empty()) {
        error_ = error;
    }
}

bool client::failed() const { return failed_; }
const std::string & client::error() const { return error_; }

client_summary client::summary() const {
    client_summary result;
    result.calls = rpc_samples_.size();
    result.rpc_p50_ms = percentile(rpc_samples_, 0.50);
    result.rpc_p90_ms = percentile(rpc_samples_, 0.90);
    result.compute_p50_ms = percentile(compute_samples_, 0.50);
    result.host_branch_p50_ms = percentile(host_branch_samples_, 0.50);
    result.wait_p50_ms = percentile(wait_samples_, 0.50);
    result.overlap_p50_ms = percentile(overlap_samples_, 0.50);
    return result;
}

uint32_t client::n_ff_exp() const { return n_ff_exp_; }
uint32_t client::n_expert() const { return n_expert_; }
uint32_t client::n_expert_used() const { return n_expert_used_; }
uint32_t client::layer_count() const { return layer_count_; }
uint64_t client::layer_mask() const { return config_.layer_mask; }
uint64_t client::weight_hash() const { return weight_hash_; }

} // namespace moe_split
