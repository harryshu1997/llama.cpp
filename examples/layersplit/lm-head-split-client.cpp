#include "lm-head-split-client.h"

#include "lm-head-split-protocol.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

namespace lm_head_split {
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
    const size_t index = static_cast<size_t>(
            fraction * static_cast<double>(values.size() - 1));
    return values[index];
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
        config_.port > 65535 || config_.rows == 0 || config_.top_k == 0 ||
        config_.top_k > config_.rows || config_.n_embd == 0 ||
        config_.n_vocab <= config_.rows || config_.timeout_ms <= 0) {
        error = "invalid LM-head split client configuration";
        return false;
    }

    addrinfo hints = {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo * addresses = nullptr;
    const std::string port = std::to_string(config_.port);
    const int resolve_status = getaddrinfo(
            config_.host.c_str(), port.c_str(), &hints, &addresses);
    if (resolve_status != 0) {
        error = "cannot resolve LM-head split worker";
        return false;
    }
    for (addrinfo * current = addresses; current != nullptr; current = current->ai_next) {
        const int candidate_fd = socket(
                current->ai_family, current->ai_socktype, current->ai_protocol);
        if (candidate_fd < 0) {
            continue;
        }
        if (::connect(candidate_fd, current->ai_addr, current->ai_addrlen) == 0) {
            fd_ = candidate_fd;
            break;
        }
        close(candidate_fd);
    }
    freeaddrinfo(addresses);
    if (fd_ < 0) {
        error = "cannot connect to LM-head split worker";
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
    request.n_embd = config_.n_embd;
    request.rows = config_.rows;
    request.top_k = config_.top_k;
    request.flags = config_.f16_io ? flag_f16_io : 0;
    hello_response response = {};
    if (!send_exact(fd_, &request, sizeof(request)) ||
        !receive_exact(fd_, &response, sizeof(response))) {
        error = "LM-head HELLO exchange failed";
        close(fd_);
        fd_ = -1;
        return false;
    }
    if (response.magic != protocol_magic || response.version != protocol_version ||
        response.message != static_cast<uint16_t>(message_type::hello_response) ||
        response.status != 0 || response.flags != request.flags ||
        response.n_embd != config_.n_embd || response.n_vocab != config_.n_vocab ||
        response.rows != config_.rows || response.top_k != config_.top_k ||
        response.offset + response.rows != response.n_vocab) {
        error = "LM-head HELLO identity mismatch";
        close(fd_);
        fd_ = -1;
        return false;
    }
    offset_ = response.offset;
    weight_hash_ = response.weight_hash;
    input_.resize(config_.n_embd);
    candidate_ids_.resize(config_.top_k);
    approximate_scores_.resize(config_.top_k);
    exact_scores_.resize(config_.top_k);
    fprintf(stderr,
            "[lm-head-split] connected %s:%d rows=[%u,%u) K=%u top_k=%u io=%s hash=%016llx\n",
            config_.host.c_str(), config_.port, offset_, offset_ + config_.rows,
            config_.n_embd, config_.top_k, config_.f16_io ? "f16" : "f32",
            static_cast<unsigned long long>(weight_hash_));
    return true;
}

bool client::exchange(uint32_t request_id, std::string & error) {
    std::vector<ggml_fp16_t> encoded_input;
    const void * input_payload = input_.data();
    size_t input_bytes = input_.size() * sizeof(float);
    if (config_.f16_io) {
        encoded_input.resize(input_.size());
        ggml_fp32_to_fp16_row(input_.data(), encoded_input.data(), encoded_input.size());
        input_payload = encoded_input.data();
        input_bytes = encoded_input.size() * sizeof(ggml_fp16_t);
    }
    execute_request request = {};
    request.magic = protocol_magic;
    request.version = protocol_version;
    request.message = static_cast<uint16_t>(message_type::execute_request);
    request.request_id = request_id;
    request.elements = config_.n_embd;
    request.payload_bytes = static_cast<uint32_t>(input_bytes);
    request.payload_hash = hash_bytes(input_payload, input_bytes);
    std::vector<uint8_t> packet(sizeof(request) + input_bytes);
    memcpy(packet.data(), &request, sizeof(request));
    memcpy(packet.data() + sizeof(request), input_payload, input_bytes);

    const double started = now_ms();
    execute_response response = {};
    if (!send_exact(fd_, packet.data(), packet.size()) ||
        !receive_exact(fd_, &response, sizeof(response))) {
        error = "LM-head EXECUTE header exchange failed";
        return false;
    }
    const size_t result_bytes = config_.top_k * sizeof(candidate);
    if (response.magic != protocol_magic || response.version != protocol_version ||
        response.message != static_cast<uint16_t>(message_type::execute_response) ||
        response.status != 0 || response.request_id != request_id ||
        response.count != config_.top_k || response.payload_bytes != result_bytes) {
        error = "LM-head EXECUTE response mismatch";
        return false;
    }
    std::vector<candidate> candidates(config_.top_k);
    if (!receive_exact(fd_, candidates.data(), result_bytes) ||
        response.payload_hash != hash_bytes(candidates.data(), result_bytes)) {
        error = "LM-head EXECUTE candidate read failed";
        return false;
    }
    rpc_ms_ = now_ms() - started;
    compute_ms_ = static_cast<double>(response.compute_us) / 1000.0;
    reduce_ms_ = static_cast<double>(response.reduce_us) / 1000.0;
    std::vector<uint32_t> sorted_ids;
    sorted_ids.reserve(candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (candidates[i].token_id < offset_ ||
            candidates[i].token_id >= offset_ + config_.rows ||
            !std::isfinite(candidates[i].score)) {
            error = "LM-head candidate is out of range or non-finite";
            return false;
        }
        candidate_ids_[i] = candidates[i].token_id;
        approximate_scores_[i] = candidates[i].score;
        sorted_ids.push_back(candidates[i].token_id);
    }
    std::sort(sorted_ids.begin(), sorted_ids.end());
    if (std::adjacent_find(sorted_ids.begin(), sorted_ids.end()) != sorted_ids.end()) {
        error = "LM-head candidate IDs are not unique";
        return false;
    }
    return true;
}

bool client::eval(ggml_tensor * tensor, bool ask) {
    if (tensor == nullptr || failed_ || fd_ < 0) {
        return false;
    }
    const bool is_input = strcmp(tensor->name, "lm_head_phone_input") == 0 &&
            tensor->type == GGML_TYPE_F32 && tensor->ne[0] == config_.n_embd &&
            tensor->ne[1] == 1;
    const bool is_ids = strcmp(tensor->name, "lm_head_phone_ids") == 0 &&
            tensor->type == GGML_TYPE_I32 && tensor->ne[0] == config_.top_k &&
            tensor->ne[1] == 1;
    const bool is_rescore = strcmp(tensor->name, "lm_head_phone_rescore") == 0 &&
            tensor->type == GGML_TYPE_F32 && tensor->ne[0] == config_.top_k &&
            tensor->ne[1] == 1;
    const bool is_publish = strcmp(tensor->name, "lm_head_split_output") == 0 &&
            tensor->type == GGML_TYPE_F32 && tensor->ne[0] == config_.n_vocab &&
            tensor->ne[1] == 1;
    if (ask) {
        return is_input || is_ids || is_rescore || is_publish;
    }
    if (!is_input && !is_ids && !is_rescore && !is_publish) {
        return true;
    }
    if (tensor->buffer == nullptr) {
        set_error("LM-head callback tensor has no storage");
        return false;
    }

    if (is_input) {
        if (request_active_) {
            set_error("LM-head input arrived with a request already active");
            return false;
        }
        ggml_backend_tensor_get(
                tensor, input_.data(), 0, input_.size() * sizeof(float));
        const uint32_t request_id = next_request_id_++;
        thread_ok_ = false;
        thread_error_.clear();
        launch_ms_ = now_ms();
        request_active_ = true;
        rpc_pending_ = true;
        scores_ready_ = false;
        thread_ = std::thread([this, request_id]() {
            thread_ok_ = exchange(request_id, thread_error_);
        });
        return true;
    }

    if (is_ids) {
        if (!request_active_ || !rpc_pending_ || !thread_.joinable()) {
            set_error("LM-head candidate IDs arrived without a pending request");
            return false;
        }
        const double host_ready_ms = now_ms();
        thread_.join();
        joined_ms_ = now_ms();
        rpc_pending_ = false;
        if (!thread_ok_) {
            set_error(thread_error_.empty() ?
                    "LM-head worker request failed" : thread_error_);
            return false;
        }
        ggml_backend_tensor_set(
                tensor, candidate_ids_.data(), 0,
                candidate_ids_.size() * sizeof(candidate_ids_[0]));
        std::vector<uint32_t> observed_ids(candidate_ids_.size());
        ggml_backend_tensor_get(
                tensor, observed_ids.data(), 0,
                observed_ids.size() * sizeof(observed_ids[0]));
        if (observed_ids != candidate_ids_) {
            set_error("LM-head candidate ID publication failed");
            return false;
        }
        rpc_samples_.push_back(rpc_ms_);
        compute_samples_.push_back(compute_ms_);
        reduce_samples_.push_back(reduce_ms_);
        host_branch_samples_.push_back(host_ready_ms - launch_ms_);
        wait_samples_.push_back(joined_ms_ - host_ready_ms);
        return true;
    }

    if (is_rescore) {
        if (!request_active_ || rpc_pending_) {
            set_error("LM-head rescore arrived before candidates");
            return false;
        }
        ggml_backend_tensor_get(
                tensor, exact_scores_.data(), 0,
                exact_scores_.size() * sizeof(exact_scores_[0]));
        for (size_t i = 0; i < exact_scores_.size(); ++i) {
            if (!std::isfinite(exact_scores_[i])) {
                set_error("LM-head exact rescore is non-finite");
                return false;
            }
            score_error_max_ = std::max(
                    score_error_max_,
                    std::abs(static_cast<double>(exact_scores_[i]) -
                             static_cast<double>(approximate_scores_[i])));
        }
        scores_ready_ = true;
        return true;
    }

    if (!request_active_ || rpc_pending_ || !scores_ready_) {
        set_error("LM-head publication arrived before exact rescoring");
        return false;
    }
    std::vector<float> suffix(config_.rows, -std::numeric_limits<float>::infinity());
    for (size_t i = 0; i < candidate_ids_.size(); ++i) {
        suffix[candidate_ids_[i] - offset_] = exact_scores_[i];
    }
    ggml_backend_tensor_set(
            tensor, suffix.data(), static_cast<size_t>(offset_) * sizeof(float),
            suffix.size() * sizeof(float));
    rescore_samples_.push_back(now_ms() - joined_ms_);
    reset_request();
    return true;
}

void client::reset_request() {
    request_active_ = false;
    rpc_pending_ = false;
    scores_ready_ = false;
}

void client::finish() {
    if (thread_.joinable()) {
        thread_.join();
    }
    if (request_active_) {
        reset_request();
        set_error("LM-head decode ended with an incomplete request");
    }
}

void client::set_error(const std::string & error) {
    failed_ = true;
    if (error_.empty()) {
        error_ = error;
    }
}

bool client::failed() const {
    return failed_;
}

const std::string & client::error() const {
    return error_;
}

client_summary client::summary() const {
    client_summary result;
    result.calls = rpc_samples_.size();
    result.rpc_p50_ms = percentile(rpc_samples_, 0.50);
    result.rpc_p90_ms = percentile(rpc_samples_, 0.90);
    result.compute_p50_ms = percentile(compute_samples_, 0.50);
    result.reduce_p50_ms = percentile(reduce_samples_, 0.50);
    result.host_branch_p50_ms = percentile(host_branch_samples_, 0.50);
    result.wait_p50_ms = percentile(wait_samples_, 0.50);
    result.rescore_p50_ms = percentile(rescore_samples_, 0.50);
    result.score_error_max = score_error_max_;
    return result;
}

uint32_t client::offset() const {
    return offset_;
}

uint64_t client::weight_hash() const {
    return weight_hash_;
}

} // namespace lm_head_split
