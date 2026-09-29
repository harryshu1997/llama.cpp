#include "ffn-split-client.h"

#include "ffn-split-protocol.h"

#if defined(FFN_SPLIT_USB_TRANSPORT)
#include "ffn-split-dmabuf.h"
#include "ffn-split-usb-client.h"
#endif

#include "ggml.h"
#include "ggml-backend.h"

extern "C" {
#include "../gguf-hash/deps/sha256/sha256.h"
}

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <map>
#include <numeric>
#include <system_error>
#include <tuple>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

namespace ffn_split {
namespace {

using steady_clock = std::chrono::steady_clock;

double now_ms() {
    return std::chrono::duration<double, std::milli>(
            steady_clock::now().time_since_epoch()).count();
}

int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            steady_clock::now().time_since_epoch()).count();
}

std::string hex_encode(const std::string & value) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(value.size() * 2);
    for (const unsigned char byte : value) {
        result.push_back(digits[byte >> 4]);
        result.push_back(digits[byte & 0x0f]);
    }
    return result;
}

std::string runtime_context_log(
        const std::vector<client_runtime_context_entry> & entries) {
    std::string result;
    for (const auto & entry : entries) {
        if (!result.empty()) {
            result.push_back(',');
        }
        result += hex_encode(entry.request_id) + ":" +
                std::to_string(entry.slot_id) + ":" +
                std::to_string(entry.rows) + ":" +
                std::to_string(entry.plan_generation);
    }
    return result;
}

static constexpr uint32_t tail_fence_magic = UINT32_C(0x53343250);
static constexpr uint16_t tail_fence_version = 1;
static constexpr uint16_t tail_fence_begin = 1;
static constexpr uint16_t tail_fence_done = 2;

struct tail_fence_request {
    uint32_t magic;
    uint16_t version;
    uint16_t message;
    uint64_t sequence;
    uint32_t request_id;
    uint32_t layer;
    uint32_t tokens;
    uint32_t reserved;
    int64_t begin_ns;
};

struct tail_fence_response {
    uint32_t magic;
    uint16_t version;
    uint16_t message;
    uint64_t sequence;
    uint32_t request_id;
    uint32_t status;
    uint64_t copied_bytes;
    uint32_t copied_chunks;
    uint32_t reserved;
    int64_t work_started_ns;
    int64_t work_completed_ns;
};

static_assert(sizeof(tail_fence_request) == 40);
static_assert(sizeof(tail_fence_response) == 56);

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

double mean(const std::vector<double> & values) {
    if (values.empty()) {
        return 0.0;
    }
    return std::accumulate(values.begin(), values.end(), 0.0) /
            static_cast<double>(values.size());
}

} // namespace

client::client(client_config config) : config_(std::move(config)) {
    next_request_id_ = config_.first_request_id == 0 ? 1 : config_.first_request_id;
}

client::~client() {
    finish();
    if (tail_fence_fd_ >= 0) {
        close(tail_fence_fd_);
    }
    if (fd_ >= 0) {
        close(fd_);
    }
#if defined(FFN_SPLIT_USB_TRANSPORT)
    if (usb_ != nullptr) {
        usb_->close();
    }
#endif
}

bool client::connected() const {
    if (config_.transport == client_transport::tcp) {
        return fd_ >= 0;
    }
#if defined(FFN_SPLIT_USB_TRANSPORT)
    return usb_ != nullptr && usb_->connected();
#else
    return false;
#endif
}

bool client::connect(std::string & error, uint64_t layer_mask) {
    const bool tcp = config_.transport == client_transport::tcp;
    const bool usb = config_.transport == client_transport::functionfs_usb;
    const uint64_t requested_layer_mask =
            (layer_mask == 0 ? config_.layer_mask : layer_mask) |
            config_.remote_resident_layer_mask;
    uint8_t artifact_sha256[32] = {};
    if (connected() || (!tcp && !usb) ||
        (config_.remote_resident_layer_mask & ~config_.layer_mask) != 0 ||
        (tcp && (config_.host.empty() || config_.port <= 0 ||
                 config_.port > 65535)) || requested_layer_mask == 0 ||
        (requested_layer_mask & ~config_.layer_mask) != 0 ||
        config_.max_columns == 0 || config_.n_embd == 0 ||
        config_.max_tokens == 0 || config_.timeout_ms <= 0 ||
        (config_.row_diagnostic_steps != 0 && ((config_.row_diagnostic_steps != 5 && config_.row_diagnostic_steps != 64) ||
         !config_.runtime_control || config_.remote_resident_layer_mask != 0)) ||
        (config_.tail_fence_socket.empty() !=
                (config_.tail_fence_layer < 0)) ||
        config_.tail_fence_layer >= 64 ||
        config_.tail_fence_join_layer >= 64 ||
        (config_.tail_fence_join_layer >= 0 &&
         config_.tail_fence_join_layer <= config_.tail_fence_layer) ||
        (config_.tail_fence_socket.empty() &&
                config_.tail_fence_join_layer >= 0) ||
        !parse_artifact_sha256(config_.artifact_sha256, artifact_sha256) ||
        (config_.tail_fence_layer >= 0 &&
         (config_.layer_mask &
                 (UINT64_C(1) << config_.tail_fence_layer)) == 0)) {
        error = "invalid FFN split client configuration";
        return false;
    }
#if !defined(FFN_SPLIT_USB_TRANSPORT)
    if (usb) {
        error = "FunctionFS USB support is not built";
        return false;
    }
#endif
    if (!config_.tail_fence_socket.empty()) {
        sockaddr_un address = {};
        if (config_.tail_fence_socket[0] != '/' ||
            config_.tail_fence_socket.size() >= sizeof(address.sun_path)) {
            error = "invalid FFN split tail fence socket";
            return false;
        }
    }

    const auto disconnect = [this]() {
        connected_layer_mask_ = 0;
        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }
#if defined(FFN_SPLIT_USB_TRANSPORT)
        if (usb_ != nullptr) {
            usb_->close();
            usb_.reset();
        }
#endif
    };

    if (tcp) {
        addrinfo hints = {};
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        addrinfo * addresses = nullptr;
        const std::string port = std::to_string(config_.port);
        const int resolve_status = getaddrinfo(
                config_.host.c_str(), port.c_str(), &hints, &addresses);
        if (resolve_status != 0) {
            error = "cannot resolve FFN split worker";
            return false;
        }

        timeval timeout = {};
        timeout.tv_sec = config_.timeout_ms / 1000;
        timeout.tv_usec = (config_.timeout_ms % 1000) * 1000;
        for (addrinfo * current = addresses; current != nullptr;
             current = current->ai_next) {
            const int candidate = socket(
                    current->ai_family, current->ai_socktype,
                    current->ai_protocol);
            if (candidate < 0) {
                continue;
            }
            // Linux bounds a blocking connect() by SO_SNDTIMEO: an unreachable worker fails after the FFN timeout
            setsockopt(candidate, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
            setsockopt(candidate, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
            if (::connect(candidate, current->ai_addr,
                        current->ai_addrlen) == 0) {
                fd_ = candidate;
                break;
            }
            close(candidate);
        }
        freeaddrinfo(addresses);
        if (fd_ < 0) {
            error = "cannot connect to FFN split worker";
            return false;
        }

        const int one = 1;
        setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    }

    hello_request request = {};
    request.magic = protocol_magic;
    request.version = protocol_version;
    request.message = static_cast<uint16_t>(message_type::hello_request);
    request.layer_mask = requested_layer_mask;
    request.max_columns = config_.max_columns;
    request.n_embd = config_.n_embd;
    request.flags = (config_.f16_io ? flag_f16_io : 0) |
            (config_.swiglu ? flag_swiglu : 0);
    request.max_tokens = config_.max_tokens;
    memcpy(request.artifact_sha256, artifact_sha256, sizeof(artifact_sha256));
    hello_response response = {};
    if (tcp) {
        if (!send_exact(fd_, &request, sizeof(request)) ||
            !receive_exact(fd_, &response, sizeof(response))) {
            error = "FFN split HELLO exchange failed";
            disconnect();
            return false;
        }
    } else {
#if defined(FFN_SPLIT_USB_TRANSPORT)
        const size_t element_bytes = config_.f16_io ?
                sizeof(ggml_fp16_t) : sizeof(float);
        if (config_.n_embd >
                (SIZE_MAX - dmabuf_payload_offset) /
                        config_.max_tokens / element_bytes) {
            error = "FFN split USB payload capacity overflows";
            return false;
        }
        const size_t required_payload =
                static_cast<size_t>(config_.n_embd) * config_.max_tokens *
                element_bytes;
        if (config_.usb_max_payload_bytes == 0) {
            config_.usb_max_payload_bytes = required_payload;
        }
        if (config_.usb_max_payload_bytes < required_payload ||
            config_.usb_max_payload_bytes >
                    SIZE_MAX - dmabuf_payload_offset ||
            config_.usb_queue_depth == 0 ||
            config_.usb_transport_generation.empty() ||
            (config_.usb_batch_plan != "coalesced-batch" &&
             config_.usb_batch_plan != "split-row") ||
            config_.usb_vendor_id == 0 || config_.usb_product_id == 0) {
            error = "invalid FFN split USB transport contract";
            return false;
        }
        usb_host_allocator allocator;
        if (!parse_usb_host_allocator(config_.usb_allocator, allocator)) {
            error = "invalid FFN split USB allocator";
            return false;
        }
        usb_client_config usb_config;
        usb_config.vendor_id = config_.usb_vendor_id;
        usb_config.product_id = config_.usb_product_id;
        usb_config.host_to_device_endpoint = dmabuf_out_endpoint;
        usb_config.device_to_host_endpoint = dmabuf_in_endpoint;
        if (config_.f16_io && config_.usb_max_payload_bytes >
                (SIZE_MAX - dmabuf_payload_offset) / 3) {
            error = "FFN split USB staging size overflows";
            disconnect();
            return false;
        }
        usb_config.host_to_device_slot_bytes = dmabuf_payload_offset +
                config_.usb_max_payload_bytes +
                (config_.f16_io ? 2 * config_.usb_max_payload_bytes : 0);
        usb_config.device_to_host_slot_bytes = dmabuf_payload_offset +
                config_.usb_max_payload_bytes;
        usb_config.usbfs_available_bytes = config_.usbfs_available_bytes;
        usb_config.slot_safety_bytes = config_.usb_slot_safety_bytes;
        usb_config.configured_max_queue_depth = config_.usb_queue_depth;
        usb_config.timeout_ms = static_cast<unsigned int>(config_.timeout_ms);
        usb_config.allocator = allocator;
        usb_config.transport_generation = config_.usb_transport_generation;
        usb_ = std::make_unique<usb_client>(std::move(usb_config));
        if (!usb_->connect(error)) {
            disconnect();
            return false;
        }
        usb_transfer_record transfer;
        if (!usb_->exchange(
                    &request, sizeof(request), &response, sizeof(response),
                    {}, transfer, error)) {
            error = "FFN split USB HELLO exchange failed: " + error;
            disconnect();
            return false;
        }
#endif
    }
    const bool response_identity_ok =
            response.magic == protocol_magic &&
            response.version == protocol_version &&
            response.message == static_cast<uint16_t>(
                    message_type::hello_response) &&
            response.status == 0 &&
            response.layer_mask == requested_layer_mask &&
            response.n_embd == config_.n_embd &&
            response.max_columns == config_.max_columns &&
            response.max_tokens == config_.max_tokens &&
            response.flags == request.flags &&
            same_artifact_sha256(
                    response.artifact_sha256, request.artifact_sha256);
    const bool response_geometry_ok =
            response.offset + response.max_columns == response.n_ff &&
            response.column_quantum != 0 &&
            response.max_columns % 32 == 0 &&
            response.column_quantum % 32 == 0 &&
            (response.alternate_columns_32 == 0 ||
             static_cast<uint32_t>(response.alternate_columns_32) * 32 <
                     response.max_columns) &&
            response.layer_count == static_cast<uint32_t>(
                    __builtin_popcountll(requested_layer_mask));
    if (!response_identity_ok || !response_geometry_ok) {
        char detail[512];
        std::snprintf(
                detail, sizeof(detail),
                "FFN split HELLO identity mismatch "
                "magic=%u/%u version=%u/%u message=%u/%u status=%u "
                "mask=%llu/%llu n_embd=%u/%u columns=%u/%u "
                "tokens=%u/%u flags=%u/%u n_ff=%u offset=%u "
                "quantum=%u alternate32=%u layers=%u/%u",
                response.magic, protocol_magic,
                response.version, protocol_version,
                response.message,
                static_cast<unsigned>(message_type::hello_response),
                response.status,
                static_cast<unsigned long long>(response.layer_mask),
                static_cast<unsigned long long>(requested_layer_mask),
                response.n_embd, config_.n_embd,
                response.max_columns, config_.max_columns,
                response.max_tokens, config_.max_tokens,
                response.flags, request.flags,
                response.n_ff, response.offset,
                response.column_quantum,
                response.alternate_columns_32,
                response.layer_count,
                static_cast<unsigned>(__builtin_popcountll(
                        requested_layer_mask)));
        error = detail;
        disconnect();
        return false;
    }
    if (config_.row_diagnostic_steps && (response.offset != 0 || response.n_ff != config_.max_columns)) {
        error = "FFN row diagnostic requires full-width resident host and worker weights";
        return false;
    }
    if (expected_weight_hash_ != 0 && response.weight_hash != expected_weight_hash_) {
        error = "FFN split worker weights differ from the session it replaces";
        disconnect();
        return false;
    }
    n_ff_ = response.n_ff;
    offset_ = response.offset;
    layer_count_ = response.layer_count;
    column_quantum_ = response.column_quantum;
    alternate_columns_ =
            static_cast<uint32_t>(response.alternate_columns_32) * 32;
    weight_hash_ = response.weight_hash;
    if (config_.remote_resident_layer_mask != 0 &&
        (offset_ != 0 || n_ff_ != config_.max_columns)) {
        // a remote-resident layer has no host prefix: the phone must hold every column
        error = "remote-resident FFN layers require complete phone shards (offset=" +
                std::to_string(offset_) + " columns=" +
                std::to_string(config_.max_columns) + " n_ff=" +
                std::to_string(n_ff_) + ")";
        disconnect();
        return false;
    }
    connected_layer_mask_ = requested_layer_mask;
    if (const char * policy_text = getenv("LLAMA_FFN_SPLIT_POLICY")) {
        if (!llama_ffn_split_policy::parse(
                    policy_text, config_.max_tokens, config_.max_columns,
                    policy_, error)) {
            disconnect();
            return false;
        }
        for (const auto & value : policy_) {
            if (value.columns != 0 && value.columns != config_.max_columns &&
                value.columns != alternate_columns_ &&
                value.columns % column_quantum_ != 0) {
                error = "FFN split policy uses an unsupported suffix width";
                disconnect();
                return false;
            }
        }
    }
    try {
        thread_ = std::thread(&client::worker_loop, this);
    } catch (const std::system_error &) {
        error = "cannot start FFN split I/O thread";
        disconnect();
        return false;
    }
    fprintf(stderr,
            "[ffn-split] connected transport=%s endpoint=%s:%d "
            "layers=%u mask=%016llx remote_resident=%016llx "
            "slice=[%u,%u) K=%u NFF=%u io=%s activation=%s quantum=%u alternate=%u "
            "hash=%016llx\n",
            tcp ? "tcp" : "functionfs-usb",
            tcp ? config_.host.c_str() : "usb", tcp ? config_.port : 0,
            layer_count_,
            static_cast<unsigned long long>(connected_layer_mask_),
            static_cast<unsigned long long>(config_.remote_resident_layer_mask),
            offset_, offset_ + config_.max_columns, config_.n_embd, n_ff_,
            config_.f16_io ? "f16" : "f32",
            config_.swiglu ? "swiglu" : "geglu", column_quantum_,
            alternate_columns_,
            static_cast<unsigned long long>(weight_hash_));
    return true;
}

bool client::ready() const {
    return connected();
}

#if defined(FFN_SPLIT_USB_TRANSPORT)
void client::release_usb_exchange() {
    if (usb_parts_.empty() || usb_ == nullptr) {
        return;
    }
    std::string ignored;
    for (const usb_pending_part & part : usb_parts_) {
        usb_->release(part.buffers, ignored);
    }
    usb_parts_.clear();
}

bool client::start_usb_exchange(
        uint32_t request_id, int layer, uint32_t elements,
        uint32_t columns, uint32_t tokens, ggml_tensor * tensor,
        unsigned int & part_count, std::string & error) {
    part_count = 0;
    if (usb_ == nullptr || !usb_->connected() || !usb_parts_.empty() ||
        tensor == nullptr || tensor->buffer == nullptr) {
        error = "FFN split USB input tensor is unavailable";
        return false;
    }
    const unsigned int available_depth =
            config_.usb_split_h2d || config_.usb_batch_plan == "coalesced-batch" ?
            1 : std::min<unsigned int>(usb_->queue_depth(), tokens);
    if (available_depth == 0 || elements != config_.n_embd * tokens) {
        error = "FFN split USB batch shape is invalid";
        return false;
    }
    usb_parts_.resize(available_depth);
    for (usb_pending_part & part : usb_parts_) {
        if (!usb_->acquire(part.buffers, error)) {
            release_usb_exchange();
            return false;
        }
    }

    const size_t input_offset = config_.usb_split_h2d ?
            0 : dmabuf_payload_offset;
    const bool host_accessible =
            ggml_backend_buffer_is_host(tensor->buffer) &&
            tensor->data != nullptr;
    const uint32_t base_tokens = tokens / available_depth;
    const uint32_t extra_tokens = tokens % available_depth;
    uint32_t token_offset = 0;
    for (unsigned int index = 0; index < available_depth; ++index) {
        usb_pending_part & part = usb_parts_[index];
        part.request_id = request_id + index;
        part.tokens = base_tokens + (index < extra_tokens ? 1 : 0);
        part.token_offset = token_offset;
        part.elements = config_.n_embd * part.tokens;
        part.payload_bytes = static_cast<size_t>(part.elements) *
                (config_.f16_io ? sizeof(ggml_fp16_t) : sizeof(float));
        const size_t wire_bytes = dmabuf_payload_offset + part.payload_bytes;
        part.host_to_device_bytes = config_.usb_split_h2d ?
                part.payload_bytes : wire_bytes;
        part.device_to_host_bytes = wire_bytes;
        const size_t staging_bytes = config_.f16_io ?
                static_cast<size_t>(part.elements) * sizeof(float) : 0;
        if (part.buffers.host_to_device_capacity <
                    part.host_to_device_bytes + staging_bytes ||
            part.buffers.device_to_host_capacity < wire_bytes) {
            error = "FFN split USB slot is smaller than the batch part";
            release_usb_exchange();
            return false;
        }

        unsigned char * input_wire = part.buffers.host_to_device;
        const size_t tensor_byte_offset =
                static_cast<size_t>(token_offset) * config_.n_embd *
                sizeof(float);
        if (config_.f16_io) {
            const float * input = nullptr;
            if (host_accessible) {
                input = static_cast<const float *>(tensor->data) +
                        static_cast<size_t>(token_offset) * config_.n_embd;
            } else {
                auto * staging = reinterpret_cast<float *>(
                        input_wire + part.host_to_device_bytes);
                ggml_backend_tensor_get(
                        tensor, staging, tensor_byte_offset,
                        static_cast<size_t>(part.elements) * sizeof(float));
                input = staging;
            }
            ggml_fp32_to_fp16_row(
                    input,
                    reinterpret_cast<ggml_fp16_t *>(
                            input_wire + input_offset),
                    part.elements);
        } else if (host_accessible) {
            memcpy(
                    input_wire + input_offset,
                    static_cast<const uint8_t *>(tensor->data) +
                            tensor_byte_offset,
                    part.payload_bytes);
        } else {
            ggml_backend_tensor_get(
                    tensor, input_wire + input_offset, tensor_byte_offset,
                    part.payload_bytes);
        }

        execute_request request = {};
        request.magic = protocol_magic;
        request.version = protocol_version;
        request.message = static_cast<uint16_t>(
                message_type::execute_request);
        request.request_id = part.request_id;
        request.layer = layer;
        request.elements = part.elements;
        request.payload_bytes = static_cast<uint32_t>(part.payload_bytes);
        request.payload_hash = hash_bytes(
                input_wire + input_offset, part.payload_bytes);
        request.columns = columns;
        request.tokens = part.tokens;
        const usb_transfer_identity identity = {
            part.request_id,
            weight_hash_,
            static_cast<uint64_t>(layer),
        };
        if (config_.usb_split_h2d) {
            uint32_t payload_ready = 0;
            usb_transfer_record header_transfer;
            if (!usb_->exchange(
                        &request, sizeof(request), &payload_ready,
                        sizeof(payload_ready), identity, header_transfer,
                        error) ||
                payload_ready !=
                        (dmabuf_payload_ready_magic ^ request.request_id)) {
                if (error.empty()) {
                    error = "FFN split USB payload-ready mismatch";
                }
                release_usb_exchange();
                return false;
            }
        } else {
            memset(input_wire, 0, dmabuf_payload_offset);
            memcpy(input_wire, &request, sizeof(request));
        }
        token_offset += part.tokens;
    }

    for (usb_pending_part & part : usb_parts_) {
        const usb_transfer_identity identity = {
            part.request_id,
            weight_hash_,
            static_cast<uint64_t>(layer),
        };
        if (!usb_->submit_acquired(
                    part.buffers, part.host_to_device_bytes,
                    part.device_to_host_bytes, identity, error)) {
            usb_->close();
            usb_parts_.clear();
            return false;
        }
    }
    part_count = available_depth;
    return true;
}

bool client::publish_usb_output(
        ggml_tensor * tensor, std::string & error) {
    if (usb_parts_.empty() || tensor == nullptr ||
        tensor->buffer == nullptr ||
        ggml_nelements(tensor) != pending_elements_) {
        error = "FFN split USB output tensor is unavailable";
        release_usb_exchange();
        return false;
    }
    const bool host_accessible =
            ggml_backend_buffer_is_host(tensor->buffer) &&
            tensor->data != nullptr;
    for (usb_pending_part & part : usb_parts_) {
        const unsigned char * response_payload =
                part.buffers.device_to_host + dmabuf_payload_offset;
        const size_t tensor_element_offset =
                static_cast<size_t>(part.token_offset) * config_.n_embd;
        const size_t tensor_byte_offset =
                tensor_element_offset * sizeof(float);
        if (config_.f16_io) {
            float * output = host_accessible ?
                    static_cast<float *>(tensor->data) +
                            tensor_element_offset :
                    reinterpret_cast<float *>(
                            part.buffers.host_to_device);
            ggml_fp16_to_fp32_row(
                    reinterpret_cast<const ggml_fp16_t *>(response_payload),
                    output, part.elements);
            if (!host_accessible) {
                ggml_backend_tensor_set(
                        tensor, output, tensor_byte_offset,
                        static_cast<size_t>(part.elements) * sizeof(float));
            }
        } else if (host_accessible) {
            memcpy(
                    static_cast<uint8_t *>(tensor->data) + tensor_byte_offset,
                    response_payload, part.payload_bytes);
        } else {
            ggml_backend_tensor_set(
                    tensor, response_payload, tensor_byte_offset,
                    part.payload_bytes);
        }
    }
    release_usb_exchange();
    return true;
}
#endif

bool client::exchange(
        uint32_t request_id, int layer, uint32_t elements,
        uint32_t columns, uint32_t tokens, std::string & error) {
    const bool tcp = config_.transport == client_transport::tcp;
    if (tcp && (input_.size() != elements || output_.size() != elements)) {
        error = "FFN split request vector size mismatch";
        return false;
    }
    size_t payload_bytes = static_cast<size_t>(elements) * sizeof(float);
    const void * input_payload = tcp ? input_.data() : nullptr;
    if (config_.f16_io) {
        payload_bytes = static_cast<size_t>(elements) * sizeof(ggml_fp16_t);
    }
    if (config_.f16_io && tcp) {
        encoded_input_.resize(elements);
        ggml_fp32_to_fp16_row(
                input_.data(),
                reinterpret_cast<ggml_fp16_t *>(encoded_input_.data()),
                encoded_input_.size());
        input_payload = encoded_input_.data();
        payload_bytes = encoded_input_.size() * sizeof(ggml_fp16_t);
    }

    execute_request request = {};
    request.magic = protocol_magic;
    request.version = protocol_version;
    request.message = static_cast<uint16_t>(message_type::execute_request);
    request.request_id = request_id;
    request.layer = layer;
    request.elements = elements;
    request.payload_bytes = static_cast<uint32_t>(payload_bytes);
    request.columns = columns;
    request.tokens = tokens;

    const double started = now_ms();
    execute_response response = {};
    const unsigned char * response_payload = nullptr;
    if (tcp) {
        request.payload_hash = hash_bytes(input_payload, payload_bytes);
        request_packet_.resize(sizeof(execute_request) + payload_bytes);
        response_payload_.resize(payload_bytes);
        memcpy(request_packet_.data(), &request, sizeof(request));
        memcpy(
                request_packet_.data() + sizeof(request),
                input_payload, payload_bytes);
        if (!send_exact(fd_, request_packet_.data(), request_packet_.size()) ||
            !receive_exact(fd_, &response, sizeof(response))) {
            error = "FFN split EXECUTE header exchange failed";
            return false;
        }
        if (!receive_exact(
                    fd_, response_payload_.data(),
                    response_payload_.size())) {
            error = "FFN split EXECUTE payload read failed";
            return false;
        }
        response_payload = response_payload_.data();
    } else {
#if defined(FFN_SPLIT_USB_TRANSPORT)
        if (usb_parts_.empty() || usb_ == nullptr || !usb_->connected()) {
            error = "FFN split USB transfer was not submitted";
            return false;
        }
        uint64_t first_started_ns = UINT64_MAX;
        uint64_t last_h2d_ns = 0;
        uint64_t last_d2h_ns = 0;
        compute_ms_ = 0.0;
        size_t completed_elements = 0;
        size_t completed_tokens = 0;
        for (usb_pending_part & part : usb_parts_) {
            usb_transfer_record transfer;
            if (!usb_->wait_acquired(part.buffers, transfer, error)) {
                usb_->close();
                usb_parts_.clear();
                return false;
            }
            memcpy(&response, part.buffers.device_to_host, sizeof(response));
            response_payload = part.buffers.device_to_host +
                    dmabuf_payload_offset;
            if (response.magic != protocol_magic ||
                response.version != protocol_version ||
                response.message != static_cast<uint16_t>(
                        message_type::execute_response) ||
                response.status != 0 ||
                response.request_id != part.request_id ||
                response.layer != layer ||
                response.elements != part.elements ||
                response.payload_bytes != part.payload_bytes ||
                response.columns != columns ||
                response.tokens != part.tokens ||
                response.payload_hash != hash_bytes(
                        response_payload, part.payload_bytes)) {
                error = "FFN split USB batch response mismatch";
                usb_->close();
                usb_parts_.clear();
                return false;
            }
            first_started_ns = std::min(
                    first_started_ns, transfer.started_ns);
            last_h2d_ns = std::max(
                    last_h2d_ns, transfer.host_to_device_completed_ns);
            last_d2h_ns = std::max(
                    last_d2h_ns, transfer.device_to_host_completed_ns);
            compute_ms_ += static_cast<double>(response.compute_us) / 1000.0;
            part.transfer = transfer;
            part.compute_us = response.compute_us;
            std::fprintf(stderr,
                    "S41SERVERFFNUSB request=%u layer=%d tokens=%u "
                    "columns=%u slot=%u h2d_bytes=%zu d2h_bytes=%zu "
                    "started_ns=%llu h2d_completed_ns=%llu "
                    "d2h_completed_ns=%llu compute_us=%llu\n",
                    part.request_id, layer, part.tokens, columns,
                    transfer.slot_index, transfer.host_to_device_bytes,
                    transfer.device_to_host_bytes,
                    static_cast<unsigned long long>(transfer.started_ns),
                    static_cast<unsigned long long>(
                            transfer.host_to_device_completed_ns),
                    static_cast<unsigned long long>(
                            transfer.device_to_host_completed_ns),
                    static_cast<unsigned long long>(response.compute_us));
            completed_elements += part.elements;
            completed_tokens += part.tokens;
        }
        if (first_started_ns == UINT64_MAX ||
            completed_elements != elements || completed_tokens != tokens ||
            last_h2d_ns < first_started_ns || last_d2h_ns < first_started_ns) {
            error = "FFN split USB batch completion is incomplete";
            release_usb_exchange();
            return false;
        }
        const double h2d_ms = static_cast<double>(
                last_h2d_ns - first_started_ns) / 1e6;
        rpc_ms_ = static_cast<double>(
                last_d2h_ns - first_started_ns) / 1e6;
        h2d_samples_.push_back(h2d_ms);
        d2h_exposed_samples_.push_back(std::max(
                0.0, rpc_ms_ - h2d_ms - compute_ms_));
        return true;
#else
        error = "FunctionFS USB support is not built";
        return false;
#endif
    }
    if (response.magic != protocol_magic ||
        response.version != protocol_version ||
        response.message != static_cast<uint16_t>(message_type::execute_response) ||
        response.status != 0 || response.request_id != request_id ||
        response.layer != layer || response.elements != elements ||
        response.payload_bytes != payload_bytes ||
        response.columns != columns || response.tokens != tokens) {
        error = "FFN split EXECUTE response mismatch";
        return false;
    }
    rpc_ms_ = now_ms() - started;
    compute_ms_ = static_cast<double>(response.compute_us) / 1000.0;
    if (response.payload_hash !=
        hash_bytes(response_payload, payload_bytes)) {
        error = "FFN split EXECUTE payload hash mismatch";
        return false;
    }
    if (config_.f16_io) {
        ggml_fp16_to_fp32_row(
                reinterpret_cast<const ggml_fp16_t *>(response_payload),
                output_.data(), output_.size());
    } else {
        memcpy(output_.data(), response_payload, payload_bytes);
    }
    return true;
}

bool client::exchange_tail_fence(
        uint64_t sequence, uint32_t request_id, int layer, uint32_t tokens,
        std::string & error) {
    if (tail_fence_fd_ < 0) {
        sockaddr_un address = {};
        address.sun_family = AF_UNIX;
        memcpy(address.sun_path, config_.tail_fence_socket.c_str(),
                config_.tail_fence_socket.size() + 1);
        tail_fence_fd_ = socket(AF_UNIX, SOCK_SEQPACKET, 0);
        if (tail_fence_fd_ < 0 ||
            ::connect(tail_fence_fd_,
                    reinterpret_cast<sockaddr *>(&address),
                    sizeof(address)) != 0) {
            error = std::string("FFN split tail fence connect failed: ") +
                    strerror(errno);
            if (tail_fence_fd_ >= 0) {
                close(tail_fence_fd_);
                tail_fence_fd_ = -1;
            }
            return false;
        }
    }

    tail_fence_request request = {};
    request.magic = tail_fence_magic;
    request.version = tail_fence_version;
    request.message = tail_fence_begin;
    request.sequence = sequence;
    request.request_id = request_id;
    request.layer = static_cast<uint32_t>(layer);
    request.tokens = tokens;
    request.begin_ns = now_ns();
    tail_fence_response response = {};
    const ssize_t sent = send(
            tail_fence_fd_, &request, sizeof(request), MSG_NOSIGNAL);
    if (sent != static_cast<ssize_t>(sizeof(request))) {
        error = std::string("FFN split tail fence send failed: ") +
                strerror(errno);
        return false;
    }
    ssize_t received;
    do {
        received = recv(tail_fence_fd_, &response, sizeof(response), 0);
    } while (received < 0 && errno == EINTR);
    if (received != static_cast<ssize_t>(sizeof(response)) ||
        response.magic != tail_fence_magic ||
        response.version != tail_fence_version ||
        response.message != tail_fence_done ||
        response.sequence != sequence ||
        response.request_id != request_id || response.status != 0 ||
        response.work_completed_ns < response.work_started_ns) {
        error = "invalid FFN split tail fence response";
        return false;
    }
    return true;
}

bool client::start_async_tail_fence(
        uint32_t request_id, int layer, uint32_t tokens,
        std::string & error) {
    if (tail_fence_pending_ || tail_fence_thread_.joinable()) {
        error = "FFN split macro tail fence is already pending";
        return false;
    }
    const uint64_t sequence = next_tail_fence_sequence_++;
    {
        std::lock_guard<std::mutex> lock(tail_fence_mutex_);
        tail_fence_done_ = false;
        tail_fence_ok_ = false;
        tail_fence_started_ns_ = now_ns();
        tail_fence_done_ns_ = 0;
        tail_fence_error_.clear();
    }
    tail_fence_pending_ = true;
    try {
        tail_fence_thread_ = std::thread(
                [this, sequence, request_id, layer, tokens]() {
            std::string exchange_error;
            const bool ok = exchange_tail_fence(
                    sequence, request_id, layer, tokens, exchange_error);
            {
                std::lock_guard<std::mutex> lock(tail_fence_mutex_);
                tail_fence_ok_ = ok;
                tail_fence_error_ = std::move(exchange_error);
                tail_fence_done_ns_ = now_ns();
                tail_fence_done_ = true;
            }
            tail_fence_done_cv_.notify_one();
        });
    } catch (const std::system_error &) {
        tail_fence_pending_ = false;
        error = "cannot start FFN split macro tail fence thread";
        return false;
    }
    return true;
}

bool client::finish_async_tail_fence(
        int64_t join_ready_ns, std::string & error) {
    if (!tail_fence_pending_ || !tail_fence_thread_.joinable()) {
        error = "FFN split macro tail fence join is not pending";
        return false;
    }
    bool ok = false;
    int64_t started_ns = 0;
    int64_t done_ns = 0;
    {
        std::unique_lock<std::mutex> lock(tail_fence_mutex_);
        tail_fence_done_cv_.wait(lock, [this]() { return tail_fence_done_; });
        ok = tail_fence_ok_;
        error = tail_fence_error_;
        started_ns = tail_fence_started_ns_;
        done_ns = tail_fence_done_ns_;
    }
    tail_fence_thread_.join();
    tail_fence_pending_ = false;
    if (!ok || started_ns <= 0 || done_ns < started_ns ||
        join_ready_ns < started_ns) {
        if (error.empty()) {
            error = "FFN split macro tail fence timing is invalid";
        }
        return false;
    }
    const double service_ms = static_cast<double>(done_ns - started_ns) / 1e6;
    const double window_ms =
            static_cast<double>(join_ready_ns - started_ns) / 1e6;
    const double join_wait_ms = static_cast<double>(
            std::max<int64_t>(0, done_ns - join_ready_ns)) / 1e6;
    tail_fence_samples_.push_back(service_ms);
    tail_fence_window_samples_.push_back(window_ms);
    tail_fence_join_wait_samples_.push_back(join_wait_ms);
    tail_fence_overlap_samples_.push_back(std::min(service_ms, window_ms));
    tail_fence_overrun_samples_.push_back(join_wait_ms);
    return true;
}

void client::worker_loop() {
    for (;;) {
        uint32_t request_id = 0;
        int layer = -1;
        uint32_t elements = 0;
        uint32_t columns = 0;
        uint32_t tokens = 0;
        {
            std::unique_lock<std::mutex> lock(thread_mutex_);
            thread_job_cv_.wait(lock, [this]() {
                return thread_stop_ || thread_has_job_;
            });
            if (thread_stop_ && !thread_has_job_) {
                return;
            }
            request_id = thread_request_id_;
            layer = thread_layer_;
            elements = thread_elements_;
            columns = thread_columns_;
            tokens = thread_tokens_;
            thread_has_job_ = false;
        }

        std::string exchange_error;
        const bool ok = exchange(
                request_id, layer, elements, columns, tokens, exchange_error);
        {
            std::lock_guard<std::mutex> lock(thread_mutex_);
            thread_ok_ = ok;
            thread_error_ = std::move(exchange_error);
            thread_done_ns_ = now_ns();
            thread_done_ = true;
        }
        thread_done_cv_.notify_one();
    }
}

bool client::runtime_columns(
        uint32_t tokens, uint64_t & layer_mask, uint32_t & columns,
        std::string & error) const {
    if (config_.runtime_control) {
        std::lock_guard<std::mutex> lock(runtime_policy_mutex_);
        layer_mask = runtime_layer_mask_.load(std::memory_order_relaxed);
        columns = runtime_columns_.load(std::memory_order_acquire);
        return true;
    }
    layer_mask = config_.layer_mask;
    const char * text = getenv("LLAMA_FFN_SPLIT_COLUMNS");
    if (text == nullptr || *text == '\0') {
        error = "LLAMA_FFN_SPLIT_COLUMNS is not set";
        return false;
    }
    errno = 0;
    char * end = nullptr;
    const unsigned long long parsed = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        parsed > config_.max_columns) {
        error = "invalid runtime FFN split width";
        return false;
    }
    columns = static_cast<uint32_t>(parsed);

    if (!policy_.empty()) {
        columns = llama_ffn_split_policy::select(policy_, tokens);
        return true;
    }

    const char * m1_text = getenv("LLAMA_FFN_SPLIT_M1_COLUMNS");
    const char * small_text = getenv("LLAMA_FFN_SPLIT_SMALL_M_COLUMNS");
    const char * large_text = getenv("LLAMA_FFN_SPLIT_LARGE_M_COLUMNS");
    const char * small_max_text = getenv("LLAMA_FFN_SPLIT_SMALL_M_MAX");
    if (m1_text != nullptr && small_text != nullptr &&
        large_text != nullptr && small_max_text != nullptr) {
        auto parse_policy_value = [&error](
                const char * value_text, uint32_t & value) {
            errno = 0;
            char * value_end = nullptr;
            const unsigned long long parsed_value =
                    strtoull(value_text, &value_end, 10);
            if (errno != 0 || value_end == value_text ||
                *value_end != '\0' || parsed_value > UINT32_MAX) {
                error = "invalid runtime FFN split shape policy";
                return false;
            }
            value = static_cast<uint32_t>(parsed_value);
            return true;
        };
        uint32_t m1_columns = 0;
        uint32_t small_columns = 0;
        uint32_t large_columns = 0;
        uint32_t small_max = 0;
        if (!parse_policy_value(m1_text, m1_columns) ||
            !parse_policy_value(small_text, small_columns) ||
            !parse_policy_value(large_text, large_columns) ||
            !parse_policy_value(small_max_text, small_max) ||
            small_max == 0) {
            return false;
        }
        columns = tokens == 1 ? m1_columns :
                (tokens <= small_max ? small_columns : large_columns);
    }

    if (columns != 0 && columns != config_.max_columns &&
        columns != alternate_columns_ &&
        (column_quantum_ == 0 || columns % column_quantum_ != 0)) {
        error = "runtime FFN split width is not a supported suffix";
        return false;
    }
    return true;
}

bool client::set_runtime_policy(
        uint64_t layer_mask, uint32_t columns, std::string & error) {
    if (!config_.runtime_control) {
        error = "FFN split runtime control is disabled";
        return false;
    }
    if (config_.row_diagnostic_steps && columns != 0 && columns != n_ff_) {
        error = "FFN row diagnostic requires a full-width split";
        return false;
    }
    std::lock_guard<std::mutex> lock(runtime_policy_mutex_);
    if ((layer_mask & config_.remote_resident_layer_mask) != 0) {
        error = "FFN split runtime policy targets remote-resident layers";
        return false;
    }
    if (connected() && (layer_mask & ~connected_layer_mask_) != 0 &&
        !rearm_runtime_connection(
                layer_mask | config_.remote_resident_layer_mask, error)) {
        return false;
    }
    const bool supported_columns =
            columns == 0 || columns == config_.max_columns ||
            columns == alternate_columns_ ||
            (column_quantum_ != 0 && columns % column_quantum_ == 0);
    if ((layer_mask & ~config_.layer_mask) != 0 ||
        (layer_mask == 0) != (columns == 0) || !supported_columns) {
        error = "FFN split runtime policy exceeds the resident slice";
        return false;
    }
    runtime_layer_mask_.store(layer_mask, std::memory_order_relaxed);
    runtime_columns_.store(columns, std::memory_order_release);
    return true;
}

bool client::rearm_runtime_connection(
        uint64_t layer_mask, std::string & error) {
    if (!connected() || layer_mask == 0 ||
        (layer_mask & ~config_.layer_mask) != 0 || pending_ ||
        tail_fence_pending_ || tail_fence_thread_.joinable()) {
        error = "FFN split runtime connection cannot be rearmed";
        return false;
    }
    const uint64_t previous_mask = connected_layer_mask_;
    if (thread_.joinable()) {
        {
            std::lock_guard<std::mutex> lock(thread_mutex_);
            thread_stop_ = true;
        }
        thread_job_cv_.notify_one();
        thread_.join();
    }
    bool shutdown_ok = true;
#if defined(FFN_SPLIT_USB_TRANSPORT)
    if (config_.transport == client_transport::functionfs_usb &&
        usb_ != nullptr && usb_->connected()) {
        execute_request shutdown = {};
        shutdown.magic = protocol_magic;
        shutdown.version = protocol_version;
        shutdown.message = static_cast<uint16_t>(
                message_type::execute_request);
        shutdown.layer = 0;
        usb_transfer_record transfer;
        shutdown_ok = usb_->exchange(
                &shutdown, sizeof(shutdown), nullptr, 0, {}, transfer,
                error);
        usb_->close();
        usb_.reset();
        if (!shutdown_ok) {
            error = "FFN split USB rearm shutdown failed: " + error;
        }
    }
#endif
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
    connected_layer_mask_ = 0;
    thread_ok_ = false;
    thread_stop_ = false;
    thread_has_job_ = false;
    thread_done_ = false;
    thread_done_ns_ = 0;
    thread_error_.clear();
    usb_shutdown_sent_ = false;
    policy_.clear();
    if (!shutdown_ok) {
        return false;
    }
    if (!connect(error, layer_mask)) {
        return false;
    }
    fprintf(stderr,
            "[ffn-split] rearmed old_mask=%016llx new_mask=%016llx\n",
            static_cast<unsigned long long>(previous_mask),
            static_cast<unsigned long long>(layer_mask));
    return true;
}

std::string client::runtime_context_key(
        const std::vector<std::string> & request_ids) {
    std::vector<std::string> sorted = request_ids;
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
    std::string result;
    for (const std::string & request_id : sorted) {
        result += std::to_string(request_id.size()) + ":" + request_id;
    }
    return result;
}

void client::log_diagnostic_rows(const char * stage, ggml_tensor * tensor, uint32_t call, int layer) {
    if (config_.row_diagnostic_steps == 0) {
        return;
    }
    std::vector<float> row(config_.n_embd);
    std::vector<ggml_fp16_t> encoded(config_.n_embd);
    auto digest = [](const void * data, size_t size) {
        unsigned char hash[SHA256_DIGEST_SIZE];
        sha256_hash(hash, static_cast<const unsigned char *>(data), size);
        return hex_encode(std::string(reinterpret_cast<const char *>(hash), sizeof(hash)));
    };
    for (size_t member = 0; member < pending_runtime_context_.size(); ++member) {
        const auto & entry = pending_runtime_context_[member];
        const int32_t step = entry.decoded_token_index - entry.applied_token_index;
        if (entry.applied_token_index < 0 || step >= int32_t(config_.row_diagnostic_steps)) {
            continue;
        }
        for (size_t i = 0; i < entry.ubatch_rows.size(); ++i) {
            const uint32_t index = entry.ubatch_rows[i];
            ggml_backend_tensor_get(tensor, row.data(), index * row.size() * sizeof(float), row.size() * sizeof(float));
            ggml_fp32_to_fp16_row(row.data(), encoded.data(), row.size());
            const std::string f32 = digest(row.data(), row.size() * sizeof(float));
            const std::string wire = config_.f16_io ? digest(encoded.data(), encoded.size() * sizeof(ggml_fp16_t)) : f32;
            // Hashes cannot localize a bad row: the phone's f16 result never equals the host's f32 shadow
            // bit for bit. Keep the local shadow row and report the numeric distance of the returned row
            // to it (relative L2 and max abs), so one wrong or swapped row stands out from the ~1e-3
            // background of every correct row.
            double rel_l2 = -1.0;
            double max_abs = -1.0;
            double local_l2 = -1.0;
            if (std::strcmp(stage, "local") == 0) {
                diagnostic_local_rows_[index] = row;
            } else if (std::strcmp(stage, "returned") == 0) {
                auto it = diagnostic_local_rows_.find(index);
                if (it != diagnostic_local_rows_.end()) {
                    double diff2 = 0.0, ref2 = 0.0;
                    max_abs = 0.0;
                    for (size_t k = 0; k < row.size(); ++k) {
                        const double d = double(row[k]) - double(it->second[k]);
                        diff2 += d * d;
                        ref2 += double(it->second[k]) * double(it->second[k]);
                        max_abs = std::max(max_abs, std::fabs(d));
                    }
                    local_l2 = std::sqrt(ref2);
                    rel_l2 = ref2 > 0.0 ? std::sqrt(diff2 / ref2) : (diff2 > 0.0 ? INFINITY : 0.0);
                    diagnostic_local_rows_.erase(it);
                }
            }
            std::fprintf(stderr, "S41SERVERFFNROW stage=%s call=%u layer=%d ubatch_row=%u member=%zu "
                    "request_id=%s slot_id=%d payload_row=%u position=%d decoded=%d applied=%d step=%d "
                    "f32_sha256=%s wire_type=%s wire_sha256=%s local_rel_l2=%.3e local_max_abs=%.3e local_l2=%.3e\n",
                    stage, call, layer, index, member, hex_encode(entry.request_id).c_str(), entry.slot_id,
                    index, entry.positions[i], entry.decoded_token_index, entry.applied_token_index, step + 1,
                    f32.c_str(), config_.f16_io ? "f16" : "f32", wire.c_str(), rel_l2, max_abs, local_l2);
        }
    }
}

bool client::set_runtime_context(
        const std::vector<client_runtime_context_entry> & entries,
        std::string & error) {
    if (!config_.runtime_control || entries.empty()) {
        error = "FFN split runtime context is unavailable";
        return false;
    }
    std::vector<client_runtime_context_entry> normalized = entries;
    std::sort(normalized.begin(), normalized.end(), [](const auto & first,
                                                       const auto & second) {
        return std::tie(first.request_id, first.slot_id) <
                std::tie(second.request_id, second.slot_id);
    });
    const uint32_t total_rows = std::accumulate(normalized.begin(), normalized.end(), uint32_t{0},
            [](uint32_t total, const auto & entry) { return total + entry.rows; });
    std::vector<bool> diagnostic_rows(config_.row_diagnostic_steps ? total_rows : 0, false);
    for (size_t index = 0; index < normalized.size(); ++index) {
        const auto & entry = normalized[index];
        if (config_.row_diagnostic_steps) {
            for (uint32_t row : entry.ubatch_rows) {
                if (row >= total_rows || diagnostic_rows[row]) {
                    error = "FFN row diagnostic indices overlap or exceed the microbatch";
                    return false;
                }
                diagnostic_rows[row] = true;
            }
        }
        const bool ascii = !entry.request_id.empty() && std::all_of(
                entry.request_id.begin(), entry.request_id.end(),
                [](unsigned char value) {
                    return value >= 0x20 && value <= 0x7e;
                });
        if (config_.row_diagnostic_steps && (entry.ubatch_rows.size() != entry.rows ||
                entry.positions.size() != entry.rows)) {
            error = "FFN row diagnostic requires physical microbatch indices";
            return false;
        }
        if (!ascii || entry.slot_id < 0 || entry.rows == 0 ||
            (index > 0 && normalized[index - 1].request_id == entry.request_id)) {
            error = "FFN split runtime context is invalid";
            return false;
        }
    }
    std::lock_guard<std::mutex> lock(runtime_context_mutex_);
    runtime_context_ = std::move(normalized);
    return true;
}

void client::record_runtime_summary(
        const std::vector<client_runtime_context_entry> & entries,
        uint32_t tokens, size_t payload_bytes,
        uint32_t transfer_subrequests, double h2d_ms,
        double d2h_exposed_ms, double rpc_ms, double compute_ms,
        double host_branch_ms, double wait_ms,
        double useful_overlap_ms) {
    if (entries.empty() || tokens == 0) {
        return;
    }
    std::vector<std::string> all_request_ids;
    std::map<std::string, uint32_t> rows_by_request;
    uint64_t context_rows = 0;
    for (const auto & entry : entries) {
        all_request_ids.push_back(entry.request_id);
        rows_by_request[entry.request_id] += entry.rows;
        context_rows += entry.rows;
    }
    if (context_rows != tokens) {
        set_error("FFN split runtime context rows differ from the call");
        return;
    }
    std::map<std::string, uint32_t> rows_by_key;
    rows_by_key[runtime_context_key(all_request_ids)] = tokens;
    for (const auto & entry : rows_by_request) {
        rows_by_key[runtime_context_key({entry.first})] = entry.second;
    }
    std::lock_guard<std::mutex> lock(runtime_context_mutex_);
    for (const auto & entry : rows_by_key) {
        const uint32_t selected_rows = entry.second;
        auto & summary = runtime_summary_by_context_[entry.first];
        ++summary.calls;
        summary.batched_calls += transfer_subrequests > 1 ? 1 : 0;
        summary.transfer_subrequests += transfer_subrequests;
        summary.input_rows += selected_rows;
        summary.maximum_tokens = std::max(
                summary.maximum_tokens, selected_rows);
        const uint64_t selected_bytes = static_cast<uint64_t>(
                payload_bytes) * selected_rows / tokens;
        summary.upload_bytes += selected_bytes;
        summary.download_bytes += selected_bytes;
        summary.h2d_us += static_cast<uint64_t>(
                std::llround(std::max(0.0, h2d_ms) * 1000.0));
        summary.d2h_exposed_us += static_cast<uint64_t>(
                std::llround(std::max(0.0, d2h_exposed_ms) * 1000.0));
        summary.rpc_total_ms += rpc_ms;
        summary.compute_total_ms += compute_ms;
        summary.host_branch_total_ms += host_branch_ms;
        summary.wait_total_ms += wait_ms;
        summary.useful_overlap_total_ms += useful_overlap_ms;
    }
}

bool client::parse_named_layer_any(
        const char * name, const char * prefix, int & layer) const {
    const size_t prefix_size = strlen(prefix);
    if (strncmp(name, prefix, prefix_size) != 0 || name[prefix_size] != '-') {
        return false;
    }
    errno = 0;
    char * end = nullptr;
    const long parsed = strtol(name + prefix_size + 1, &end, 10);
    if (errno != 0 || end == name + prefix_size + 1 || *end != '\0' ||
        parsed < 0 || parsed >= 64) {
        return false;
    }
    layer = static_cast<int>(parsed);
    return true;
}

bool client::parse_named_layer(const char * name, const char * prefix, int & layer) const {
    return parse_named_layer_any(name, prefix, layer) &&
            (config_.layer_mask & (UINT64_C(1) << layer)) != 0;
}

bool client::eval(ggml_tensor * tensor, bool ask) {
    if (tensor == nullptr) {
        return false;
    }
    if (!connected()) {
        // Without a phone session an assisted-copy layer simply runs on the host, but a
        // remote-resident layer has no host weights: abort the graph instead of emitting zeros.
        int layer = -1;
        const bool remote_marker = config_.remote_resident_layer_mask != 0 &&
                (parse_named_layer_any(tensor->name, "ffn_norm", layer) ||
                 parse_named_layer_any(tensor->name, "ffn_phone_partial", layer)) &&
                remote_resident_layer(layer);
        if (!remote_marker) {
            return false;
        }
        if (ask) {
            return true;
        }
        set_error("remote-resident FFN layer " + std::to_string(layer) +
                " has no connected phone session");
        return false;
    }
    if (failed_) {
        // a failed session whose runtime policy owns no layer is idle: the host computes every FFN
        const bool idle = config_.runtime_control && config_.remote_resident_layer_mask == 0 &&
                !pending_ && !tail_fence_pending_ &&
                runtime_layer_mask_.load(std::memory_order_relaxed) == 0;
        return idle ? !ask : ask;
    }
    int diagnostic_layer = -1;
    if (config_.row_diagnostic_steps && parse_named_layer(tensor->name, "ffn_diag_local", diagnostic_layer)) {
        if (ask) {
            return true;
        }
        if (!pending_ || diagnostic_layer != pending_layer_ || tensor->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(tensor) || ggml_nelements(tensor) != pending_elements_) {
            set_error("FFN local row diagnostic differs from the pending call");
            return false;
        }
        log_diagnostic_rows("local", tensor, pending_request_id_, diagnostic_layer);
        return true;
    }
    int input_layer = -1;
    int publish_layer = -1;
    int join_layer = -1;
    const bool named_input =
            parse_named_layer(tensor->name, "ffn_norm", input_layer);
    const bool named_publish =
            parse_named_layer(tensor->name, "ffn_phone_partial", publish_layer);
    const bool named_join = config_.tail_fence_join_layer >= 0 &&
            parse_named_layer_any(tensor->name, "l_out", join_layer) &&
            join_layer == config_.tail_fence_join_layer;
    if (named_join) {
        const bool active_join = tensor->ne[1] == 1 &&
                tensor->ne[2] == 1 && tensor->ne[3] == 1 &&
                tail_fence_pending_;
        if (ask) {
            return active_join;
        }
        if (!active_join) {
            set_error("FFN split macro tail fence join lost its pending request");
            return false;
        }
        std::string join_error;
        if (!finish_async_tail_fence(now_ns(), join_error)) {
            set_error(join_error);
            return false;
        }
        return true;
    }
    if (!named_input && !named_publish) {
        return !ask;
    }
    const bool valid_tokens = tensor->ne[1] > 0 &&
            tensor->ne[1] <= config_.max_tokens;
    const uint32_t tokens = valid_tokens ?
            static_cast<uint32_t>(tensor->ne[1]) : 0;
    uint64_t layer_mask = 0;
    uint32_t columns = 0;
    std::string width_error;
    if (!runtime_columns(tokens, layer_mask, columns, width_error)) {
        set_error(width_error);
        return false;
    }
    const int policy_layer = named_input ? input_layer : publish_layer;
    const bool remote_layer = remote_resident_layer(policy_layer);
    const bool active_layer = policy_layer >= 0 &&
            (layer_mask & (UINT64_C(1) << policy_layer)) != 0;
    const bool active = remote_layer || (columns != 0 && active_layer);
    if (!active) {
        return !ask;
    }
    if (remote_layer) {
        // no host prefix exists for this layer: the phone computes every column
        columns = n_ff_;
    }
    const bool supported_shape = active && tensor->ne[0] == config_.n_embd &&
            valid_tokens &&
            tensor->ne[2] == 1 && tensor->ne[3] == 1;
    const bool is_input = supported_shape && named_input;
    const bool is_publish = supported_shape && named_publish;
    if ((named_input || named_publish) && !supported_shape) {
        if (ask) {
            return true;
        }
        set_error(
                "FFN split graph tensor is unsupported: name=" +
                std::string(tensor->name) +
                " shape=[" + std::to_string(tensor->ne[0]) + "," +
                std::to_string(tensor->ne[1]) + "," +
                std::to_string(tensor->ne[2]) + "," +
                std::to_string(tensor->ne[3]) + "] columns=" +
                std::to_string(columns) +
                " max_tokens=" + std::to_string(config_.max_tokens));
        return false;
    }
    if (ask) {
        return is_input || is_publish;
    }
    if (!is_input && !is_publish) {
        return true;
    }

    if (is_input) {
        if (pending_) {
            set_error(
                    "FFN split input layer " +
                    std::to_string(input_layer) +
                    " arrived while layer " +
                    std::to_string(pending_layer_) +
                    " request " +
                    std::to_string(pending_request_id_) +
                    " was pending");
            return false;
        }
        if (tensor->type != GGML_TYPE_F32 || tensor->buffer == nullptr ||
            !ggml_is_contiguous(tensor)) {
            set_error("FFN split input tensor shape or storage mismatch");
            return false;
        }
        const uint32_t tokens = static_cast<uint32_t>(tensor->ne[1]);
        const uint32_t elements = config_.n_embd * tokens;
        const uint32_t request_id = next_request_id_;
        if (config_.runtime_control) {
            std::lock_guard<std::mutex> lock(runtime_context_mutex_);
            const uint64_t context_rows = std::accumulate(
                    runtime_context_.begin(), runtime_context_.end(),
                    uint64_t { 0 }, [](uint64_t total, const auto & entry) {
                        return total + entry.rows;
                    });
            if (runtime_context_.empty() && remote_layer) {
                // warm-up and ownership validation run before any request context exists;
                // the call is still accounted, just not attributed to a request
                pending_runtime_context_.clear();
            } else if (runtime_context_.empty() || context_rows != tokens) {
                set_error(
                        "FFN split runtime context differs from tensor rows");
                return false;
            } else {
                pending_runtime_context_ = runtime_context_;
            }
        } else {
            pending_runtime_context_.clear();
        }
        log_diagnostic_rows("input", tensor, request_id, input_layer);
#if defined(FFN_SPLIT_USB_TRANSPORT)
        if (config_.transport == client_transport::functionfs_usb) {
            std::string submit_error;
            unsigned int part_count = 0;
            if (!start_usb_exchange(
                        request_id, input_layer, elements, columns,
                        tokens, tensor, part_count, submit_error)) {
                set_error(submit_error);
                return false;
            }
            next_request_id_ += part_count;
        } else
#endif
        {
            ++next_request_id_;
            input_.resize(elements);
            output_.resize(elements);
            ggml_backend_tensor_get(
                    tensor, input_.data(), 0,
                    input_.size() * sizeof(float));
        }
        launch_ms_ = now_ms();
        pending_ = true;
        pending_request_id_ = request_id;
        pending_layer_ = input_layer;
        pending_elements_ = elements;
        pending_columns_ = columns;
        pending_tokens_ = tokens;
        {
            std::lock_guard<std::mutex> lock(thread_mutex_);
            thread_ok_ = false;
            thread_error_.clear();
            thread_done_ = false;
            thread_done_ns_ = 0;
            thread_request_id_ = request_id;
            thread_layer_ = input_layer;
            thread_elements_ = elements;
            thread_columns_ = columns;
            thread_tokens_ = tokens;
            thread_has_job_ = true;
        }
        thread_job_cv_.notify_one();
        return true;
    }

    if (!pending_ || !thread_.joinable() || publish_layer != pending_layer_ ||
        columns != pending_columns_ ||
        static_cast<uint32_t>(tensor->ne[1]) != pending_tokens_) {
        set_error(
                "FFN split publication layer " +
                std::to_string(publish_layer) +
                " differs from pending layer " +
                std::to_string(pending_layer_) +
                " request " +
                std::to_string(pending_request_id_));
        return false;
    }
    if (tensor->type != GGML_TYPE_F32 || tensor->buffer == nullptr ||
        !ggml_is_contiguous(tensor) ||
        ggml_nelements(tensor) != pending_elements_) {
        set_error("FFN split publication tensor shape or storage mismatch");
        std::unique_lock<std::mutex> lock(thread_mutex_);
        thread_done_cv_.wait(lock, [this]() { return thread_done_; });
        pending_ = false;
        pending_layer_ = -1;
#if defined(FFN_SPLIT_USB_TRANSPORT)
        release_usb_exchange();
#endif
        return false;
    }

    const int64_t host_ready_ns = now_ns();
    const double host_ready_ms = static_cast<double>(host_ready_ns) / 1e6;
    double tail_fence_ms = 0.0;
    bool tail_fence_called = false;
    bool tail_fence_ok = true;
    std::string tail_fence_error;
    if (!config_.tail_fence_socket.empty() &&
        publish_layer == config_.tail_fence_layer && pending_tokens_ == 1) {
        ++tail_fence_opportunities_;
        if (config_.tail_fence_join_layer >= 0) {
            tail_fence_ok = start_async_tail_fence(
                    pending_request_id_, publish_layer, pending_tokens_,
                    tail_fence_error);
        } else {
            bool phone_complete = false;
            {
                std::lock_guard<std::mutex> lock(thread_mutex_);
                phone_complete = thread_done_;
            }
            if (phone_complete) {
                ++tail_fence_skipped_complete_;
            } else {
                const int64_t fence_started_ns = now_ns();
                tail_fence_called = true;
                tail_fence_ok = exchange_tail_fence(
                        next_tail_fence_sequence_++, pending_request_id_,
                        publish_layer, pending_tokens_, tail_fence_error);
                tail_fence_ms = static_cast<double>(
                        now_ns() - fence_started_ns) / 1e6;
            }
        }
    }
    int64_t phone_done_ns = 0;
    {
        std::unique_lock<std::mutex> lock(thread_mutex_);
        thread_done_cv_.wait(lock, [this]() { return thread_done_; });
        phone_done_ns = thread_done_ns_;
    }
    const double joined_ms = now_ms();
    pending_ = false;
    pending_layer_ = -1;
    if (!tail_fence_ok) {
#if defined(FFN_SPLIT_USB_TRANSPORT)
        release_usb_exchange();
#endif
        set_error(tail_fence_error);
        return false;
    }
    if (!thread_ok_) {
#if defined(FFN_SPLIT_USB_TRANSPORT)
        release_usb_exchange();
#endif
        set_error(thread_error_.empty() ? "FFN split worker request failed" : thread_error_);
        return false;
    }
    uint32_t completed_transfer_subrequests = 1;
#if defined(FFN_SPLIT_USB_TRANSPORT)
    if (config_.transport == client_transport::functionfs_usb) {
        const uint32_t transfer_parts = static_cast<uint32_t>(
                usb_parts_.size());
        std::string publish_error;
        if (!publish_usb_output(tensor, publish_error)) {
            set_error(publish_error);
            return false;
        }
        transfer_parts_samples_.push_back(transfer_parts);
        completed_transfer_subrequests = transfer_parts;
    } else
#endif
    {
        transfer_parts_samples_.push_back(1);
        ggml_backend_tensor_set(
                tensor, output_.data(), 0,
                output_.size() * sizeof(float));
    }
    log_diagnostic_rows("returned", tensor, pending_request_id_, publish_layer);
    const size_t payload_bytes = static_cast<size_t>(pending_elements_) *
            (config_.f16_io ? sizeof(ggml_fp16_t) : sizeof(float));
    if (pending_runtime_context_.empty()) {
        std::fprintf(stderr,
                "S41SERVERFFNCALL request=%u layer=%d tokens=%u columns=%u payload_bytes=%zu\n",
                pending_request_id_, publish_layer, pending_tokens_,
                pending_columns_, payload_bytes);
    } else {
        const std::string context = runtime_context_log(
                pending_runtime_context_);
        std::fprintf(stderr,
                "S41SERVERFFNCALL context=%s request=%u layer=%d tokens=%u columns=%u payload_bytes=%zu\n",
                context.c_str(),
                pending_request_id_, publish_layer, pending_tokens_,
                pending_columns_, payload_bytes);
    }
    rpc_samples_.push_back(rpc_ms_);
    compute_samples_.push_back(compute_ms_);
    const double host_branch_ms = host_ready_ms - launch_ms_;
    const double phone_branch_ms = std::max(
            0.0, static_cast<double>(phone_done_ns) / 1e6 - launch_ms_);
    host_branch_samples_.push_back(host_branch_ms);
    const double wait_ms = joined_ms - host_ready_ms;
    wait_samples_.push_back(wait_ms);
    overlap_samples_.push_back(joined_ms - launch_ms_);
    const double useful_overlap_ms =
            std::max(0.0, std::min(host_branch_ms, phone_branch_ms));
    useful_overlap_samples_.push_back(useful_overlap_ms);
    const double phone_tail_ms = static_cast<double>(std::max<int64_t>(
            0, phone_done_ns - host_ready_ns)) / 1e6;
    phone_tail_samples_.push_back(phone_tail_ms);
    if (tail_fence_called) {
        tail_fence_samples_.push_back(tail_fence_ms);
        tail_fence_overlap_samples_.push_back(
                std::min(tail_fence_ms, phone_tail_ms));
        tail_fence_overrun_samples_.push_back(
                std::max(0.0, tail_fence_ms - phone_tail_ms));
    }
    columns_samples_.push_back(pending_columns_);
    tokens_samples_.push_back(pending_tokens_);
    // a decode pass gives every slot one row; coalesced multi-slot decode calls are not prefill
    decode_samples_.push_back(pending_runtime_context_.empty() ? pending_tokens_ == 1 :
            std::all_of(pending_runtime_context_.begin(), pending_runtime_context_.end(),
                    [](const auto & entry) { return entry.rows == 1; }));
    payload_samples_.push_back(static_cast<uint32_t>(
            pending_elements_ * (config_.f16_io ? sizeof(ggml_fp16_t) : sizeof(float))));
    const double h2d_ms = h2d_samples_.empty() ? 0.0 : h2d_samples_.back();
    const double d2h_exposed_ms = d2h_exposed_samples_.empty() ?
            0.0 : d2h_exposed_samples_.back();
    record_runtime_summary(
            pending_runtime_context_, pending_tokens_, payload_bytes,
            completed_transfer_subrequests, h2d_ms, d2h_exposed_ms,
            rpc_ms_, compute_ms_, host_branch_ms, wait_ms,
            useful_overlap_ms);
    pending_elements_ = 0;
    pending_request_id_ = 0;
    pending_columns_ = 0;
    pending_tokens_ = 0;
    pending_runtime_context_.clear();
    diagnostic_local_rows_.clear();
    return true;
}

void client::finish() {
    if (tail_fence_pending_) {
        std::string fence_error;
        finish_async_tail_fence(now_ns(), fence_error);
        set_error(fence_error.empty() ?
                "FFN split decode ended before the macro tail fence join" :
                fence_error);
    }
    if (pending_) {
        std::unique_lock<std::mutex> lock(thread_mutex_);
        thread_done_cv_.wait(lock, [this]() { return thread_done_; });
        pending_ = false;
        pending_layer_ = -1;
#if defined(FFN_SPLIT_USB_TRANSPORT)
        release_usb_exchange();
#endif
        set_error("FFN split decode ended with a pending request");
    }
    if (thread_.joinable()) {
        {
            std::lock_guard<std::mutex> lock(thread_mutex_);
            thread_stop_ = true;
        }
        thread_job_cv_.notify_one();
        thread_.join();
    }
#if defined(FFN_SPLIT_USB_TRANSPORT)
    if (config_.transport == client_transport::functionfs_usb &&
        usb_ != nullptr && usb_->connected() && !usb_shutdown_sent_) {
        execute_request shutdown = {};
        shutdown.magic = protocol_magic;
        shutdown.version = protocol_version;
        shutdown.message = static_cast<uint16_t>(
                message_type::execute_request);
        shutdown.layer = 0;
        usb_transfer_record transfer;
        std::string shutdown_error;
        if (!usb_->exchange(
                    &shutdown, sizeof(shutdown), nullptr, 0, {}, transfer,
                    shutdown_error)) {
            set_error("FFN split USB shutdown failed: " + shutdown_error);
        }
        usb_shutdown_sent_ = true;
    }
#endif
}

void client::reset_session() {
    if (tail_fence_pending_ && tail_fence_thread_.joinable()) {
        std::string ignored;
        finish_async_tail_fence(now_ns(), ignored);
    }
    if (tail_fence_thread_.joinable()) {
        tail_fence_thread_.join();
    }
    tail_fence_pending_ = false;
    if (pending_) {
        std::unique_lock<std::mutex> lock(thread_mutex_);
        thread_done_cv_.wait(lock, [this]() { return thread_done_; });
        pending_ = false;
        pending_layer_ = -1;
    }
    if (thread_.joinable()) {
        {
            std::lock_guard<std::mutex> lock(thread_mutex_);
            thread_stop_ = true;
        }
        thread_job_cv_.notify_one();
        thread_.join();
    }
#if defined(FFN_SPLIT_USB_TRANSPORT)
    release_usb_exchange();
    if (usb_ != nullptr) {
        if (usb_->connected() && !usb_shutdown_sent_) {
            // best effort: a live worker ends its session instead of reading the next HELLO as a request
            std::string ignored;
            send_usb_shutdown(ignored);
        }
        usb_->close();
        usb_.reset();
    }
#endif
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
    if (weight_hash_ != 0) {
        expected_weight_hash_ = weight_hash_;
    }
    last_reset_error_ = failed_ ? error_ : "worker closed the idle session";
    ++reset_count_;
    connected_layer_mask_ = 0;
    thread_ok_ = false;
    thread_stop_ = false;
    thread_has_job_ = false;
    thread_done_ = false;
    thread_done_ns_ = 0;
    thread_error_.clear();
    usb_shutdown_sent_ = false;
    policy_.clear();
    {
        std::lock_guard<std::mutex> lock(runtime_policy_mutex_);
        runtime_layer_mask_.store(0, std::memory_order_relaxed);
        runtime_columns_.store(0, std::memory_order_release);
    }
    failed_ = false;
    error_.clear();
}

void client::latch_error(const std::string & error) {
    set_error(error);
}

size_t client::reset_count() const {
    return reset_count_;
}

const std::string & client::last_reset_error() const {
    return last_reset_error_;
}

bool client::send_usb_shutdown(std::string & error) {
#if defined(FFN_SPLIT_USB_TRANSPORT)
    execute_request shutdown = {};
    shutdown.magic = protocol_magic;
    shutdown.version = protocol_version;
    shutdown.message = static_cast<uint16_t>(message_type::execute_request);
    shutdown.layer = 0;
    usb_transfer_record transfer;
    usb_shutdown_sent_ = true;
    return usb_->exchange(&shutdown, sizeof(shutdown), nullptr, 0, {}, transfer, error);
#else
    error = "FunctionFS USB support is not built";
    return false;
#endif
}

bool client::peer_closed() const {
    if (config_.transport != client_transport::tcp || fd_ < 0 || pending_) {
        return false;
    }
    char byte = 0;
    const ssize_t received = recv(fd_, &byte, 1, MSG_PEEK | MSG_DONTWAIT);
    return received == 0 ||
            (received < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR);
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
    result.transport = config_.transport == client_transport::tcp ?
            "tcp" : "functionfs-usb";
    if (config_.transport == client_transport::functionfs_usb) {
        result.allocator = config_.usb_allocator;
        result.transport_generation = config_.usb_transport_generation;
        result.batch_plan = config_.usb_batch_plan;
        result.maximum_payload_bytes = config_.usb_max_payload_bytes;
        result.full_duplex = config_.usb_full_duplex;
#if defined(FFN_SPLIT_USB_TRANSPORT)
        result.queue_depth = usb_ == nullptr ? 0 : usb_->queue_depth();
        result.maximum_active_slots = usb_ == nullptr ? 0 :
                usb_->maximum_active_slots();
        result.maximum_outstanding_transfers = usb_ == nullptr ? 0 :
                usb_->maximum_outstanding_transfers();
#endif
    } else {
        result.allocator = "socket";
        result.transport_generation = "tcp";
        result.batch_plan = "single";
        result.queue_depth = 1;
    }
    result.calls = rpc_samples_.size();
    result.transfer_subrequests = std::accumulate(
            transfer_parts_samples_.begin(), transfer_parts_samples_.end(),
            size_t { 0 });
    result.batched_calls = static_cast<size_t>(std::count_if(
            transfer_parts_samples_.begin(), transfer_parts_samples_.end(),
            [](uint32_t count) { return count > 1; }));
    result.rpc_mean_ms = mean(rpc_samples_);
    result.rpc_p50_ms = percentile(rpc_samples_, 0.50);
    result.rpc_p90_ms = percentile(rpc_samples_, 0.90);
    result.compute_mean_ms = mean(compute_samples_);
    result.compute_p50_ms = percentile(compute_samples_, 0.50);
    result.host_branch_mean_ms = mean(host_branch_samples_);
    result.host_branch_p50_ms = percentile(host_branch_samples_, 0.50);
    result.wait_mean_ms = mean(wait_samples_);
    result.wait_min_ms = wait_samples_.empty() ? 0.0 :
            *std::min_element(wait_samples_.begin(), wait_samples_.end());
    result.wait_p10_ms = percentile(wait_samples_, 0.10);
    result.wait_p50_ms = percentile(wait_samples_, 0.50);
    result.wait_p90_ms = percentile(wait_samples_, 0.90);
    result.wait_max_ms = wait_samples_.empty() ? 0.0 :
            *std::max_element(wait_samples_.begin(), wait_samples_.end());
    result.overlap_mean_ms = mean(overlap_samples_);
    result.overlap_p50_ms = percentile(overlap_samples_, 0.50);
    result.useful_overlap_mean_ms = mean(useful_overlap_samples_);
    result.phone_tail_mean_ms = mean(phone_tail_samples_);
    result.phone_tail_min_ms = phone_tail_samples_.empty() ? 0.0 :
            *std::min_element(
                    phone_tail_samples_.begin(), phone_tail_samples_.end());
    result.phone_tail_p10_ms = percentile(phone_tail_samples_, 0.10);
    result.phone_tail_p50_ms = percentile(phone_tail_samples_, 0.50);
    result.phone_tail_p90_ms = percentile(phone_tail_samples_, 0.90);
    result.phone_tail_max_ms = phone_tail_samples_.empty() ? 0.0 :
            *std::max_element(
                    phone_tail_samples_.begin(), phone_tail_samples_.end());
    result.tail_fence_opportunities = tail_fence_opportunities_;
    result.tail_fence_calls = tail_fence_samples_.size();
    result.tail_fence_skipped_complete = tail_fence_skipped_complete_;
    result.tail_fence_mean_ms = mean(tail_fence_samples_);
    result.tail_fence_p50_ms = percentile(tail_fence_samples_, 0.50);
    result.tail_fence_p90_ms = percentile(tail_fence_samples_, 0.90);
    result.tail_fence_max_ms = tail_fence_samples_.empty() ? 0.0 :
            *std::max_element(
                    tail_fence_samples_.begin(), tail_fence_samples_.end());
    result.tail_fence_overlap_mean_ms = mean(tail_fence_overlap_samples_);
    result.tail_fence_overlap_p50_ms = percentile(
            tail_fence_overlap_samples_, 0.50);
    result.tail_fence_overrun_mean_ms = mean(tail_fence_overrun_samples_);
    result.tail_fence_overrun_max_ms = tail_fence_overrun_samples_.empty() ?
            0.0 : *std::max_element(
                    tail_fence_overrun_samples_.begin(),
                    tail_fence_overrun_samples_.end());
    result.tail_fence_macro_windows = tail_fence_window_samples_.size();
    result.tail_fence_window_mean_ms = mean(tail_fence_window_samples_);
    result.tail_fence_window_min_ms = tail_fence_window_samples_.empty() ?
            0.0 : *std::min_element(
                    tail_fence_window_samples_.begin(),
                    tail_fence_window_samples_.end());
    result.tail_fence_window_p10_ms = percentile(
            tail_fence_window_samples_, 0.10);
    result.tail_fence_window_p50_ms = percentile(
            tail_fence_window_samples_, 0.50);
    result.tail_fence_window_p90_ms = percentile(
            tail_fence_window_samples_, 0.90);
    result.tail_fence_window_max_ms = tail_fence_window_samples_.empty() ?
            0.0 : *std::max_element(
                    tail_fence_window_samples_.begin(),
                    tail_fence_window_samples_.end());
    result.tail_fence_join_wait_mean_ms = mean(
            tail_fence_join_wait_samples_);
    result.tail_fence_join_wait_p50_ms = percentile(
            tail_fence_join_wait_samples_, 0.50);
    result.tail_fence_join_wait_p90_ms = percentile(
            tail_fence_join_wait_samples_, 0.90);
    result.tail_fence_join_wait_max_ms =
            tail_fence_join_wait_samples_.empty() ? 0.0 :
            *std::max_element(
                    tail_fence_join_wait_samples_.begin(),
                    tail_fence_join_wait_samples_.end());
    for (size_t i = 0; i < tokens_samples_.size(); ++i) {
        result.input_rows += tokens_samples_[i];
        result.maximum_tokens = std::max(
                result.maximum_tokens, tokens_samples_[i]);
        result.upload_bytes += payload_samples_[i];
        result.download_bytes += payload_samples_[i];
        if (decode_samples_[i]) {
            ++result.decode_calls;
        } else {
            ++result.prefill_calls;
        }
    }
    const double h2d_ms = std::accumulate(
            h2d_samples_.begin(), h2d_samples_.end(), 0.0);
    const double d2h_ms = std::accumulate(
            d2h_exposed_samples_.begin(), d2h_exposed_samples_.end(), 0.0);
    result.h2d_us = static_cast<uint64_t>(std::llround(h2d_ms * 1000.0));
    result.d2h_exposed_us = static_cast<uint64_t>(
            std::llround(d2h_ms * 1000.0));
    if (h2d_ms > 0.0) {
        result.h2d_payload_MBps =
                static_cast<double>(result.upload_bytes) / h2d_ms / 1000.0;
    }
    if (d2h_ms > 0.0) {
        result.d2h_exposed_payload_MBps =
                static_cast<double>(result.download_bytes) / d2h_ms / 1000.0;
    }
    auto phase_percentile = [this](
            const std::vector<double> & samples, bool prefill) {
        std::vector<double> selected;
        for (size_t i = 0; i < samples.size(); ++i) {
            if ((decode_samples_[i] == 0) == prefill) {
                selected.push_back(samples[i]);
            }
        }
        return percentile(std::move(selected), 0.50);
    };
    result.decode_rpc_p50_ms = phase_percentile(rpc_samples_, false);
    result.decode_compute_p50_ms = phase_percentile(compute_samples_, false);
    result.decode_overlap_p50_ms = phase_percentile(overlap_samples_, false);
    result.prefill_rpc_p50_ms = phase_percentile(rpc_samples_, true);
    result.prefill_compute_p50_ms = phase_percentile(compute_samples_, true);
    result.prefill_overlap_p50_ms = phase_percentile(overlap_samples_, true);

    std::map<std::pair<uint32_t, uint32_t>, std::vector<size_t>> shape_indices;
    for (size_t i = 0; i < tokens_samples_.size(); ++i) {
        shape_indices[{tokens_samples_[i], columns_samples_[i]}].push_back(i);
    }
    for (const auto & entry : shape_indices) {
        std::vector<double> rpc;
        std::vector<double> compute;
        std::vector<double> host_branch;
        std::vector<double> wait;
        std::vector<double> overlap;
        std::vector<double> useful_overlap;
        for (size_t index : entry.second) {
            rpc.push_back(rpc_samples_[index]);
            compute.push_back(compute_samples_[index]);
            host_branch.push_back(host_branch_samples_[index]);
            wait.push_back(wait_samples_[index]);
            overlap.push_back(overlap_samples_[index]);
            useful_overlap.push_back(useful_overlap_samples_[index]);
        }
        client_shape_summary shape;
        shape.tokens = entry.first.first;
        shape.columns = entry.first.second;
        shape.calls = entry.second.size();
        shape.rpc_mean_ms = mean(rpc);
        shape.rpc_p50_ms = percentile(std::move(rpc), 0.50);
        shape.compute_mean_ms = mean(compute);
        shape.compute_p50_ms = percentile(std::move(compute), 0.50);
        shape.host_branch_mean_ms = mean(host_branch);
        shape.wait_mean_ms = mean(wait);
        shape.overlap_mean_ms = mean(overlap);
        shape.overlap_p50_ms = percentile(std::move(overlap), 0.50);
        shape.useful_overlap_mean_ms = mean(useful_overlap);
        result.shapes.push_back(shape);
    }
    return result;
}

client_summary client::summary(
        const std::vector<std::string> & request_ids) const {
    client_summary result;
    result.transport = config_.transport == client_transport::tcp ?
            "tcp" : "functionfs-usb";
    if (config_.transport == client_transport::functionfs_usb) {
        result.allocator = config_.usb_allocator;
        result.transport_generation = config_.usb_transport_generation;
        result.batch_plan = config_.usb_batch_plan;
        result.maximum_payload_bytes = config_.usb_max_payload_bytes;
        result.full_duplex = config_.usb_full_duplex;
#if defined(FFN_SPLIT_USB_TRANSPORT)
        result.queue_depth = usb_ == nullptr ? 0 : usb_->queue_depth();
        result.maximum_active_slots = usb_ == nullptr ? 0 :
                usb_->maximum_active_slots();
        result.maximum_outstanding_transfers = usb_ == nullptr ? 0 :
                usb_->maximum_outstanding_transfers();
#endif
    } else {
        result.allocator = "socket";
        result.transport_generation = "tcp";
        result.batch_plan = "single";
        result.queue_depth = 1;
    }
    runtime_summary_accumulator value;
    {
        std::lock_guard<std::mutex> lock(runtime_context_mutex_);
        const auto found = runtime_summary_by_context_.find(
                runtime_context_key(request_ids));
        if (found != runtime_summary_by_context_.end()) {
            value = found->second;
        }
    }
    result.calls = value.calls;
    result.batched_calls = value.batched_calls;
    result.transfer_subrequests = value.transfer_subrequests;
    result.input_rows = value.input_rows;
    result.maximum_tokens = value.maximum_tokens;
    result.upload_bytes = value.upload_bytes;
    result.download_bytes = value.download_bytes;
    result.h2d_us = value.h2d_us;
    result.d2h_exposed_us = value.d2h_exposed_us;
    if (value.calls > 0) {
        const double count = static_cast<double>(value.calls);
        result.rpc_mean_ms = value.rpc_total_ms / count;
        result.compute_mean_ms = value.compute_total_ms / count;
        result.host_branch_mean_ms =
                value.host_branch_total_ms / count;
        result.wait_mean_ms = value.wait_total_ms / count;
        result.useful_overlap_mean_ms =
                value.useful_overlap_total_ms / count;
    }
    if (value.h2d_us > 0) {
        result.h2d_payload_MBps = static_cast<double>(
                value.upload_bytes) / static_cast<double>(value.h2d_us);
    }
    if (value.d2h_exposed_us > 0) {
        result.d2h_exposed_payload_MBps = static_cast<double>(
                value.download_bytes) /
                static_cast<double>(value.d2h_exposed_us);
    }
    return result;
}

uint32_t client::n_ff() const {
    return n_ff_;
}

uint32_t client::offset() const {
    return offset_;
}

uint32_t client::layer_count() const {
    return layer_count_;
}

uint32_t client::max_columns() const {
    return config_.max_columns;
}

uint32_t client::column_quantum() const {
    return column_quantum_;
}

uint32_t client::alternate_columns() const {
    return alternate_columns_;
}

uint16_t client::max_tokens() const {
    return config_.max_tokens;
}

uint64_t client::remote_resident_layer_mask() const {
    return config_.remote_resident_layer_mask;
}

bool client::remote_resident_layer(int layer) const {
    return layer >= 0 && layer < 64 &&
            (config_.remote_resident_layer_mask & (UINT64_C(1) << layer)) != 0;
}

uint64_t client::layer_mask() const {
    return config_.layer_mask;
}

uint64_t client::weight_hash() const {
    return weight_hash_;
}

} // namespace ffn_split
