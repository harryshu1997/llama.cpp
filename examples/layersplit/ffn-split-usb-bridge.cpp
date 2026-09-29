#include "ffn-split-dmabuf.h"
#include "ffn-split-usb-client.h"

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <vector>

namespace {

static constexpr unsigned int transfer_timeout_ms = 10000;
static constexpr unsigned int recovery_timeout_ms = 30000;
static constexpr unsigned int maximum_reset_recoveries = 8;

using steady_clock = std::chrono::steady_clock;

static constexpr uint32_t prefetch_fence_magic = UINT32_C(0x53343250);
static constexpr uint16_t prefetch_fence_version = 1;
static constexpr uint16_t prefetch_fence_begin = 1;
static constexpr uint16_t prefetch_fence_done = 2;

struct prefetch_fence_request {
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

struct prefetch_fence_response {
    uint32_t magic;
    uint16_t version;
    uint16_t message;
    uint64_t sequence;
    uint32_t request_id;
    uint32_t status;
    uint64_t copied_bytes;
    uint32_t copied_chunks;
    uint32_t reserved;
    int64_t copy_started_ns;
    int64_t copy_completed_ns;
};

static_assert(sizeof(prefetch_fence_request) == 40);
static_assert(sizeof(prefetch_fence_response) == 56);

double now_ms() {
    return std::chrono::duration<double, std::milli>(
            steady_clock::now().time_since_epoch()).count();
}

int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            steady_clock::now().time_since_epoch()).count();
}

double percentile(std::vector<double> values, double fraction);

class prefetch_fence_client {
public:
    prefetch_fence_client() {
        const char * path = getenv("S42_FFN_PREFETCH_FENCE_SOCKET");
        if (path == nullptr || *path == '\0') {
            return;
        }
        sockaddr_un address = {};
        if (strlen(path) >= sizeof(address.sun_path)) {
            throw std::runtime_error("prefetch fence socket path is too long");
        }
        socket_path_ = path;
        const char * first = getenv(
                "S42_FFN_PREFETCH_FENCE_GROUP_FIRST_LAYER");
        const char * last = getenv(
                "S42_FFN_PREFETCH_FENCE_GROUP_LAST_LAYER");
        if ((first == nullptr) != (last == nullptr)) {
            throw std::runtime_error(
                    "prefetch fence group requires both layer bounds");
        }
        if (first != nullptr) {
            group_first_layer_ = parse_layer(first);
            group_last_layer_ = parse_layer(last);
            if (group_first_layer_ >= group_last_layer_) {
                throw std::runtime_error(
                    "prefetch fence group layer bounds are invalid");
            }
        }
        const char * expected = getenv(
                "S42_FFN_PREFETCH_FENCE_TOTAL_BYTES");
        if (expected != nullptr) {
            expected_total_bytes_ = parse_bytes(expected);
        }
        configured_ = true;
    }

    ~prefetch_fence_client() {
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    bool enabled() const { return configured_ && !completed_; }
    bool configured() const { return configured_; }
    bool completed() const { return completed_; }
    uint64_t expected_total_bytes() const { return expected_total_bytes_; }
    int group_first_layer() const { return group_first_layer_; }
    int group_last_layer() const { return group_last_layer_; }

    void begin(const ffn_split::execute_request & request) {
        if (!enabled()) {
            return;
        }
        ensure_connected();
        if (group_first_layer_ >= 0) {
            if (!awaiting_response_) {
                if (request.layer != group_first_layer_) {
                    throw std::runtime_error(
                            "prefetch fence group did not start at first layer");
                }
                send_begin(request);
                expected_group_layer_ = group_first_layer_ + 1;
                return;
            }
            if (request.layer != expected_group_layer_ ||
                request.layer > group_last_layer_) {
                throw std::runtime_error(
                        "prefetch fence group layer order mismatch");
            }
            ++expected_group_layer_;
            return;
        }
        if (awaiting_response_) {
            throw std::runtime_error("prefetch fence response is pending");
        }
        send_begin(request);
    }

    void finish(
            const ffn_split::execute_request & request,
            double protected_ready_ms) {
        if (!enabled()) {
            return;
        }
        if (group_first_layer_ >= 0 && request.layer != group_last_layer_) {
            return;
        }
        if (!awaiting_response_ ||
            (group_first_layer_ >= 0 &&
             expected_group_layer_ != group_last_layer_ + 1)) {
            throw std::runtime_error("prefetch fence completion is unpaired");
        }
        receive_done(protected_ready_ms);
        awaiting_response_ = false;
        expected_group_layer_ = -1;
    }

    void require_complete() const {
        if (awaiting_response_) {
            throw std::runtime_error("prefetch fence group is incomplete");
        }
    }

    uint64_t calls() const { return calls_; }
    uint64_t copied_bytes() const { return copied_bytes_; }
    uint64_t copied_chunks() const { return copied_chunks_; }
    double copy_p50_ms() const { return percentile(copy_samples_, 0.50); }
    double copy_p90_ms() const { return percentile(copy_samples_, 0.90); }
    double window_min_ms() const {
        return window_samples_.empty() ? 0.0 :
                *std::min_element(
                        window_samples_.begin(), window_samples_.end());
    }
    double window_p10_ms() const { return percentile(window_samples_, 0.10); }
    double window_p50_ms() const { return percentile(window_samples_, 0.50); }
    double window_p90_ms() const { return percentile(window_samples_, 0.90); }
    double decode_window_min_ms() const {
        return decode_window_samples_.empty() ? 0.0 :
                *std::min_element(
                        decode_window_samples_.begin(),
                        decode_window_samples_.end());
    }
    double decode_window_p10_ms() const {
        return percentile(decode_window_samples_, 0.10);
    }
    double decode_window_p50_ms() const {
        return percentile(decode_window_samples_, 0.50);
    }
    double prefill_window_min_ms() const {
        return prefill_window_samples_.empty() ? 0.0 :
                *std::min_element(
                        prefill_window_samples_.begin(),
                        prefill_window_samples_.end());
    }
    double prefill_window_p10_ms() const {
        return percentile(prefill_window_samples_, 0.10);
    }
    double prefill_window_p50_ms() const {
        return percentile(prefill_window_samples_, 0.50);
    }
    double phone_idle_min_ms() const {
        return phone_idle_samples_.empty() ? 0.0 :
                *std::min_element(
                        phone_idle_samples_.begin(),
                        phone_idle_samples_.end());
    }
    double phone_idle_p10_ms() const {
        return percentile(phone_idle_samples_, 0.10);
    }
    double phone_idle_p50_ms() const {
        return percentile(phone_idle_samples_, 0.50);
    }
    double phone_idle_p90_ms() const {
        return percentile(phone_idle_samples_, 0.90);
    }
    double overrun_p90_ms() const {
        return percentile(overrun_samples_, 0.90);
    }
    double overrun_max_ms() const {
        return overrun_samples_.empty() ? 0.0 :
                *std::max_element(
                        overrun_samples_.begin(), overrun_samples_.end());
    }
    double copied_window_overrun_p90_ms() const {
        return percentile(copy_overrun_samples_, 0.90);
    }
    double copied_window_overrun_max_ms() const {
        return copy_overrun_samples_.empty() ? 0.0 :
                *std::max_element(
                        copy_overrun_samples_.begin(),
                        copy_overrun_samples_.end());
    }

private:
    void ensure_connected() {
        if (fd_ >= 0) {
            return;
        }
        sockaddr_un address = {};
        address.sun_family = AF_UNIX;
        strcpy(address.sun_path, socket_path_.c_str());
        fd_ = socket(AF_UNIX, SOCK_SEQPACKET, 0);
        if (fd_ < 0) {
            throw std::runtime_error(std::string(
                    "prefetch fence socket failed: ") + strerror(errno));
        }
        if (connect(fd_, reinterpret_cast<sockaddr *>(&address),
                    sizeof(address)) != 0) {
            const std::string message = std::string(
                    "prefetch fence connect failed: ") + strerror(errno);
            close(fd_);
            fd_ = -1;
            throw std::runtime_error(message);
        }
    }

    static int parse_layer(const char * text) {
        errno = 0;
        char * end = nullptr;
        const long value = strtol(text, &end, 10);
        if (errno != 0 || end == text || *end != '\0' ||
            value < 0 || value >= 64) {
            throw std::runtime_error("invalid prefetch fence group layer");
        }
        return static_cast<int>(value);
    }

    static uint64_t parse_bytes(const char * text) {
        errno = 0;
        char * end = nullptr;
        const unsigned long long value = strtoull(text, &end, 10);
        if (errno != 0 || end == text || *end != '\0' || value == 0) {
            throw std::runtime_error(
                    "invalid prefetch fence total bytes");
        }
        return static_cast<uint64_t>(value);
    }

    void send_begin(const ffn_split::execute_request & request) {
        current_sequence_ = ++sequence_;
        current_request_id_ = request.request_id;
        prefetch_fence_request message = {};
        message.magic = prefetch_fence_magic;
        message.version = prefetch_fence_version;
        message.message = prefetch_fence_begin;
        message.sequence = current_sequence_;
        message.request_id = request.request_id;
        message.layer = request.layer;
        message.tokens = request.tokens;
        message.begin_ns = now_ns();
        if (last_protected_ready_ns_ != 0 &&
            message.begin_ns >= last_protected_ready_ns_) {
            phone_idle_samples_.push_back(static_cast<double>(
                    message.begin_ns - last_protected_ready_ns_) / 1e6);
        }
        const ssize_t count = send(fd_, &message, sizeof(message), MSG_NOSIGNAL);
        if (count != static_cast<ssize_t>(sizeof(message))) {
            throw std::runtime_error(std::string(
                    "prefetch fence send failed: ") + strerror(errno));
        }
        current_begin_ns_ = message.begin_ns;
        current_tokens_ = request.tokens;
        awaiting_response_ = true;
    }

    void receive_done(double protected_ready_ms) {
        prefetch_fence_response response = {};
        ssize_t count;
        do {
            count = recv(fd_, &response, sizeof(response), 0);
        } while (count < 0 && errno == EINTR);
        if (count != static_cast<ssize_t>(sizeof(response)) ||
            response.magic != prefetch_fence_magic ||
            response.version != prefetch_fence_version ||
            response.message != prefetch_fence_done ||
            response.sequence != current_sequence_ ||
            response.request_id != current_request_id_ ||
            response.status != 0 ||
            response.copy_completed_ns < response.copy_started_ns) {
            throw std::runtime_error("invalid prefetch fence response");
        }
        const double copy_ms = static_cast<double>(
                response.copy_completed_ns - response.copy_started_ns) / 1e6;
        const int64_t protected_ready_ns = static_cast<int64_t>(
                protected_ready_ms * 1e6);
        const double window_ms = static_cast<double>(std::max<int64_t>(
                0, protected_ready_ns - current_begin_ns_)) / 1e6;
        window_samples_.push_back(window_ms);
        if (current_tokens_ == 1) {
            decode_window_samples_.push_back(window_ms);
        } else {
            prefill_window_samples_.push_back(window_ms);
        }
        const double overrun_ms = std::max(
                0.0,
                static_cast<double>(response.copy_completed_ns) / 1e6 -
                        protected_ready_ms);
        copy_samples_.push_back(copy_ms);
        overrun_samples_.push_back(overrun_ms);
        if (response.copied_bytes != 0) {
            copy_overrun_samples_.push_back(overrun_ms);
        }
        if (expected_total_bytes_ != 0 &&
            response.copied_bytes > expected_total_bytes_ - copied_bytes_) {
            throw std::runtime_error(
                    "prefetch fence copied beyond expected bytes");
        }
        copied_bytes_ += response.copied_bytes;
        copied_chunks_ += response.copied_chunks;
        ++calls_;
        if (expected_total_bytes_ != 0 &&
            copied_bytes_ == expected_total_bytes_) {
            close(fd_);
            fd_ = -1;
            completed_ = true;
        }
        last_protected_ready_ns_ = protected_ready_ns;
    }
    int fd_ = -1;
    std::string socket_path_;
    int group_first_layer_ = -1;
    int group_last_layer_ = -1;
    int expected_group_layer_ = -1;
    bool awaiting_response_ = false;
    bool configured_ = false;
    bool completed_ = false;
    uint64_t sequence_ = 0;
    uint64_t current_sequence_ = 0;
    uint32_t current_request_id_ = 0;
    uint64_t calls_ = 0;
    uint64_t copied_bytes_ = 0;
    uint64_t copied_chunks_ = 0;
    uint64_t expected_total_bytes_ = 0;
    int64_t current_begin_ns_ = 0;
    int64_t last_protected_ready_ns_ = 0;
    uint32_t current_tokens_ = 0;
    std::vector<double> copy_samples_;
    std::vector<double> window_samples_;
    std::vector<double> decode_window_samples_;
    std::vector<double> prefill_window_samples_;
    std::vector<double> phone_idle_samples_;
    std::vector<double> overrun_samples_;
    std::vector<double> copy_overrun_samples_;
};

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

bool valid_hello_response(
        const ffn_split::hello_request & request,
        const ffn_split::hello_response & response) {
    const uint16_t supported_flags =
            ffn_split::flag_f16_io | ffn_split::flag_swiglu;
    return response.status == 0 && response.n_embd != 0 &&
            (request.flags & ~supported_flags) == 0 &&
            response.flags == request.flags &&
            response.n_embd == request.n_embd &&
            response.max_columns == request.max_columns &&
            response.max_tokens == request.max_tokens &&
            ffn_split::same_artifact_sha256(
                    response.artifact_sha256, request.artifact_sha256) &&
            response.max_tokens != 0 && response.column_quantum != 0 &&
            (response.alternate_columns_32 == 0 ||
             static_cast<uint32_t>(response.alternate_columns_32) * 32 <
                     response.max_columns) &&
            response.offset + response.max_columns == response.n_ff;
}

[[maybe_unused]] bool same_hello_request(
        const ffn_split::hello_request & first,
        const ffn_split::hello_request & second) {
    return first.magic == second.magic &&
            first.version == second.version &&
            first.message == second.message &&
            first.layer_mask == second.layer_mask &&
            first.n_embd == second.n_embd &&
            first.max_columns == second.max_columns &&
            first.flags == second.flags &&
            first.max_tokens == second.max_tokens &&
            ffn_split::same_artifact_sha256(
                    first.artifact_sha256, second.artifact_sha256);
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

[[maybe_unused]] double payload_bytes_per_second(
        uint64_t bytes, const std::vector<double> & milliseconds) {
    double total_ms = 0.0;
    for (const double value : milliseconds) {
        total_ms += value;
    }
    if (bytes == 0 || total_ms <= 0.0) {
        return 0.0;
    }
    return static_cast<double>(bytes) * 1000.0 / total_ms;
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

[[maybe_unused]] unsigned int parse_queue_depth(const char * text) {
    errno = 0;
    char * end = nullptr;
    const unsigned long value = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        value == 0 || value > 64) {
        throw std::runtime_error("invalid USB queue depth");
    }
    return static_cast<unsigned int>(value);
}

[[maybe_unused]] size_t parse_byte_count(const char * text) {
    errno = 0;
    char * end = nullptr;
    const unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value == 0 ||
        value > static_cast<unsigned long long>(SIZE_MAX)) {
        throw std::runtime_error("invalid USB byte count");
    }
    return static_cast<size_t>(value);
}

int open_listener(const char * bind_address, int port) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    const int one = 1;
    sockaddr_in address = {};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<uint16_t>(port));
    if (fd < 0 ||
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) != 0 ||
        inet_pton(AF_INET, bind_address, &address.sin_addr) != 1 ||
        bind(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0 ||
        listen(fd, 1) != 0) {
        if (fd >= 0) {
            close(fd);
        }
        throw std::runtime_error(std::string("TCP listen setup failed: ") +
                strerror(errno));
    }
    return fd;
}

} // namespace

#if !defined(FFN_DMABUF_BRIDGE_NO_MAIN)
int main(int argc, char ** argv) {
    if (argc < 4 || argc > 7) {
        fprintf(stderr,
                "usage: %s <bind-address> <port> "
                "<malloc|devmem|malloc-split|devmem-split> "
                "[max-queue-depth] [usbfs-available-bytes] "
                "[transport-generation]\n",
                argv[0]);
        return 2;
    }

    try {
        const int port = parse_port(argv[2]);
        const std::string allocator = argv[3];
        const bool split_h2d = allocator == "malloc-split" ||
                allocator == "devmem-split";
        if (allocator != "malloc" && allocator != "devmem" &&
            !split_h2d) {
            throw std::runtime_error("invalid host allocator");
        }
        const unsigned int configured_queue_depth = argc >= 5 ?
                parse_queue_depth(argv[4]) : 1;
        const size_t available_usbfs_bytes = argc >= 6 ?
                parse_byte_count(argv[5]) : ffn_split::usbfs_memory_bytes();
        const std::string transport_generation = argc >= 7 ? argv[6] :
                "functionfs-dmabuf-async-v1";
        if (transport_generation.empty()) {
            throw std::runtime_error("invalid transport generation");
        }
        signal(SIGPIPE, SIG_IGN);
        prefetch_fence_client prefetch_fence;

        const int listen_fd = open_listener(argv[1], port);
        int client_fd = accept(listen_fd, nullptr, nullptr);
        if (client_fd < 0) {
            throw std::runtime_error("TCP accept failed");
        }
        const int one = 1;
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

        ffn_split::hello_request hello = {};
        ffn_split::hello_response hello_response = {};
        if (!receive_exact(client_fd, &hello, sizeof(hello))) {
            throw std::runtime_error("local HELLO read failed");
        }
        const size_t wire_element_bytes =
                hello.flags & ffn_split::flag_f16_io ?
                        sizeof(uint16_t) : sizeof(float);
        if (hello.n_embd == 0 || hello.max_tokens == 0 ||
            hello.n_embd > (SIZE_MAX - ffn_split::dmabuf_payload_offset) /
                    hello.max_tokens / wire_element_bytes) {
            throw std::runtime_error("local HELLO payload capacity is invalid");
        }
        const size_t max_payload_bytes =
                static_cast<size_t>(hello.n_embd) * hello.max_tokens *
                wire_element_bytes;
        const size_t max_wire_bytes =
                ffn_split::dmabuf_payload_offset + max_payload_bytes;
        ffn_split::usb_host_allocator host_allocator;
        const std::string base_allocator =
                allocator.rfind("devmem", 0) == 0 ? "devmem" : "malloc";
        if (!ffn_split::parse_usb_host_allocator(
                    base_allocator, host_allocator)) {
            throw std::runtime_error("invalid USB host allocator");
        }
        ffn_split::usb_client_config usb_config;
        usb_config.vendor_id = ffn_split::dmabuf_vendor_id;
        usb_config.product_id = ffn_split::dmabuf_product_id;
        usb_config.host_to_device_endpoint =
                ffn_split::dmabuf_out_endpoint;
        usb_config.device_to_host_endpoint =
                ffn_split::dmabuf_in_endpoint;
        usb_config.host_to_device_slot_bytes = max_wire_bytes;
        usb_config.device_to_host_slot_bytes = max_wire_bytes;
        usb_config.usbfs_available_bytes = available_usbfs_bytes;
        usb_config.slot_safety_bytes = 64U * 1024U;
        usb_config.configured_max_queue_depth = configured_queue_depth;
        usb_config.timeout_ms = transfer_timeout_ms;
        usb_config.allocator = host_allocator;
        usb_config.transport_generation = transport_generation;
        ffn_split::usb_client usb(std::move(usb_config));
        std::string usb_error;
        if (!usb.connect(usb_error)) {
            throw std::runtime_error(usb_error);
        }
        ffn_split::usb_transfer_record hello_transfer;
        if (!usb.exchange(
                    &hello, sizeof(hello), &hello_response,
                    sizeof(hello_response), {}, hello_transfer, usb_error)) {
            throw std::runtime_error(usb_error);
        }
        if (!send_exact(client_fd, &hello_response, sizeof(hello_response))) {
            throw std::runtime_error("local HELLO write failed");
        }
        if (!valid_hello_response(hello, hello_response)) {
            fprintf(stderr,
                    "[ffn-dmabuf-bridge] HELLO mismatch "
                    "request={magic=%08x version=%u message=%u mask=%016llx "
                    "n_embd=%u columns=%u flags=%u max_tokens=%u} "
                    "response={magic=%08x version=%u message=%u status=%u "
                    "flags=%u n_embd=%u n_ff=%u offset=%u columns=%u "
                    "layers=%u mask=%016llx quantum=%u max_tokens=%u "
                    "alternate=%u}\n",
                    hello.magic, hello.version, hello.message,
                    static_cast<unsigned long long>(hello.layer_mask),
                    hello.n_embd, hello.max_columns, hello.flags,
                    hello.max_tokens, hello_response.magic,
                    hello_response.version, hello_response.message,
                    hello_response.status, hello_response.flags,
                    hello_response.n_embd, hello_response.n_ff,
                    hello_response.offset, hello_response.max_columns,
                    hello_response.layer_count,
                    static_cast<unsigned long long>(
                            hello_response.layer_mask),
                    hello_response.column_quantum,
                    hello_response.max_tokens,
                    static_cast<unsigned>(
                            hello_response.alternate_columns_32) * 32);
            fflush(stderr);
            throw std::runtime_error("phone rejected FFN HELLO");
        }

        fprintf(stderr,
                "[ffn-dmabuf-bridge] ready bind=%s:%d allocator=%s "
                "queue_depth=%u max_wire_bytes=%zu generation=%s\n",
                argv[1], port, allocator.c_str(), usb.queue_depth(),
                max_wire_bytes, transport_generation.c_str());
        fflush(stderr);
        std::vector<unsigned char> request_buffer(max_wire_bytes);
        std::vector<unsigned char> response_buffer(max_wire_bytes);
        std::vector<double> rpc_samples;
        std::vector<double> out_samples;
        std::vector<double> in_samples;
        std::vector<double> compute_samples;
        std::vector<double> d2h_exposed_samples;
        std::vector<double> decode_rpc_samples;
        std::vector<double> prefill_rpc_samples;
        uint64_t calls = 0;
        uint64_t decode_calls = 0;
        uint64_t prefill_calls = 0;
        uint64_t upload_bytes = 0;
        uint64_t download_bytes = 0;
        uint64_t reset_recoveries = 0;
        bool clean_disconnect = false;
        bool terminate_phone_session = false;
        const char * qualification_clients_text = getenv(
                "S42_FFN_BRIDGE_QUALIFICATION_CLIENTS");
        bool qualification_client_pending =
                qualification_clients_text != nullptr &&
                strcmp(qualification_clients_text, "1") == 0;
        const char * shutdown_clients_text = getenv(
                "S42_FFN_BRIDGE_SHUTDOWN_CLIENTS");
        bool shutdown_client_pending =
                shutdown_clients_text != nullptr &&
                strcmp(shutdown_clients_text, "1") == 0;

        const auto reconnect = [&]() {
            usb.close();
            const double deadline = now_ms() + recovery_timeout_ms;
            std::string last_error = "recovery timeout";
            while (now_ms() < deadline) {
                std::string reconnect_error;
                if (!usb.connect(reconnect_error)) {
                    last_error = reconnect_error;
                    usleep(100000);
                    continue;
                }
                ffn_split::hello_response recovered = {};
                ffn_split::usb_transfer_record transfer;
                if (!usb.exchange(
                            &hello, sizeof(hello), &recovered,
                            sizeof(recovered), {}, transfer,
                            reconnect_error)) {
                    last_error = reconnect_error;
                    usb.close();
                    usleep(100000);
                    continue;
                }
                if (!valid_hello_response(hello, recovered) ||
                    memcmp(&recovered, &hello_response,
                            sizeof(recovered)) != 0) {
                    throw std::runtime_error(
                            "phone identity changed during USB recovery");
                }
                return;
            }
            throw std::runtime_error(
                    "USB reset recovery failed: " + last_error);
        };

        for (;;) {
            ffn_split::execute_request request = {};
            if (!receive_exact(client_fd, &request, sizeof(request))) {
                if (qualification_client_pending) {
                    fprintf(stderr,
                            "FFNDMABUFQUAL {\"status\":\"ok\","
                            "\"calls\":%llu,\"allocator\":\"%s\","
                            "\"host_allocator\":\"%s\","
                            "\"queue_depth\":%u,"
                            "\"configured_queue_depth\":%u,"
                            "\"transport_generation\":\"%s\","
                            "\"async_libusb\":true,"
                            "\"reset_recoveries\":%llu,"
                            "\"upload_bytes\":%llu,"
                            "\"download_bytes\":%llu,"
                            "\"h2d_payload_bytes_per_s\":%.3f,"
                            "\"d2h_conservative_payload_bytes_per_s\":%.3f,"
                            "\"d2h_exposed_payload_bytes_per_s\":%.3f,"
                            "\"full_duplex_payload_bytes_per_s\":0.0,"
                            "\"full_duplex_measured\":false,"
                            "\"usb_p50_ms\":%.6f,"
                            "\"usb_p90_ms\":%.6f,"
                            "\"h2d_p50_ms\":%.6f,"
                            "\"d2h_plus_compute_p50_ms\":%.6f,"
                            "\"d2h_exposed_p50_ms\":%.6f,"
                            "\"phone_compute_p50_ms\":%.6f,"
                            "\"decode_usb_p50_ms\":%.6f,"
                            "\"prefill_usb_p50_ms\":%.6f}\n",
                            static_cast<unsigned long long>(calls),
                            allocator.c_str(),
                            ffn_split::usb_host_allocator_name(
                                    host_allocator),
                            usb.queue_depth(), configured_queue_depth,
                            transport_generation.c_str(),
                            static_cast<unsigned long long>(
                                    reset_recoveries),
                            static_cast<unsigned long long>(upload_bytes),
                            static_cast<unsigned long long>(download_bytes),
                            payload_bytes_per_second(
                                    upload_bytes, out_samples),
                            payload_bytes_per_second(
                                    download_bytes, in_samples),
                            payload_bytes_per_second(
                                    download_bytes, d2h_exposed_samples),
                            percentile(rpc_samples, 0.50),
                            percentile(rpc_samples, 0.90),
                            percentile(out_samples, 0.50),
                            percentile(in_samples, 0.50),
                            percentile(d2h_exposed_samples, 0.50),
                            percentile(compute_samples, 0.50),
                            percentile(decode_rpc_samples, 0.50),
                            percentile(prefill_rpc_samples, 0.50));
                    fflush(stderr);
                    close(client_fd);
                    client_fd = accept(listen_fd, nullptr, nullptr);
                    if (client_fd < 0) {
                        throw std::runtime_error(
                                "second TCP accept failed");
                    }
                    setsockopt(
                            client_fd, IPPROTO_TCP, TCP_NODELAY,
                            &one, sizeof(one));
                    ffn_split::hello_request next_hello = {};
                    if (!receive_exact(
                                client_fd, &next_hello,
                                sizeof(next_hello)) ||
                        !same_hello_request(next_hello, hello) ||
                        !send_exact(
                                client_fd, &hello_response,
                                sizeof(hello_response))) {
                        throw std::runtime_error(
                                "second local HELLO differs");
                    }
                    qualification_client_pending = false;
                    continue;
                }
                if (shutdown_client_pending) {
                    close(client_fd);
                    client_fd = accept(listen_fd, nullptr, nullptr);
                    if (client_fd < 0) {
                        throw std::runtime_error(
                                "shutdown TCP accept failed");
                    }
                    setsockopt(
                            client_fd, IPPROTO_TCP, TCP_NODELAY,
                            &one, sizeof(one));
                    ffn_split::hello_request next_hello = {};
                    if (!receive_exact(
                                client_fd, &next_hello,
                                sizeof(next_hello)) ||
                        !same_hello_request(next_hello, hello) ||
                        !send_exact(
                                client_fd, &hello_response,
                                sizeof(hello_response))) {
                        throw std::runtime_error(
                                "shutdown local HELLO differs");
                    }
                    shutdown_client_pending = false;
                    continue;
                }
                clean_disconnect = true;
                break;
            }
            const bool local_shutdown =
                    request.magic == ffn_split::protocol_magic &&
                    request.version == ffn_split::protocol_version &&
                    request.message == static_cast<uint16_t>(
                            ffn_split::message_type::execute_request) &&
                    request.request_id == 0 && request.layer == -1 &&
                    request.elements == 0 && request.payload_bytes == 0 &&
                    request.payload_hash == 0 && request.columns == 0 &&
                    request.tokens == 0;
            if (local_shutdown) {
                clean_disconnect = true;
                terminate_phone_session = true;
                break;
            }
            if (request.magic != ffn_split::protocol_magic ||
                request.version != ffn_split::protocol_version ||
                request.message != static_cast<uint16_t>(
                        ffn_split::message_type::execute_request) ||
                request.request_id == 0 ||
                request.tokens == 0 ||
                request.tokens > hello_response.max_tokens ||
                request.elements !=
                        static_cast<uint64_t>(hello_response.n_embd) *
                                request.tokens ||
                request.payload_bytes !=
                        static_cast<uint64_t>(request.elements) *
                                (hello.flags & ffn_split::flag_f16_io ?
                                         sizeof(uint16_t) : sizeof(float)) ||
                request.payload_bytes > max_payload_bytes ||
                request.columns == 0 ||
                request.columns > hello_response.max_columns ||
                (request.columns != hello_response.max_columns &&
                 request.columns != static_cast<uint32_t>(
                         hello_response.alternate_columns_32) * 32 &&
                 request.columns % hello_response.column_quantum != 0)) {
                throw std::runtime_error("invalid local execute request");
            }
            const size_t payload_bytes = request.payload_bytes;
            const size_t wire_bytes =
                    ffn_split::dmabuf_payload_offset + payload_bytes;
            memset(request_buffer.data(), 0,
                    ffn_split::dmabuf_payload_offset);
            memcpy(request_buffer.data(), &request, sizeof(request));
            if (!receive_exact(
                        client_fd,
                        request_buffer.data() +
                                ffn_split::dmabuf_payload_offset,
                        payload_bytes)) {
                throw std::runtime_error("local execute payload read failed");
            }

            const double started = now_ms();
            prefetch_fence.begin(request);
            ffn_split::execute_response response = {};
            double out_done = 0.0;
            double completed = 0.0;
            std::vector<unsigned char> replay;
            for (;;) {
                try {
                    const ffn_split::usb_transfer_identity identity = {
                        request.request_id,
                        hello_response.weight_hash,
                        static_cast<uint64_t>(request.layer),
                    };
                    ffn_split::usb_transfer_record payload_transfer;
                    std::string exchange_error;
                    if (split_h2d) {
                        uint32_t payload_ready = 0;
                        ffn_split::usb_transfer_record header_transfer;
                        if (!usb.exchange(
                                    &request, sizeof(request),
                                    &payload_ready, sizeof(payload_ready),
                                    identity, header_transfer,
                                    exchange_error)) {
                            throw std::runtime_error(exchange_error);
                        }
                        if (calls == 0) {
                            fprintf(stderr,
                                    "[ffn-dmabuf-bridge] first header sent\n");
                            fflush(stderr);
                        }
                        if (payload_ready !=
                                (ffn_split::dmabuf_payload_ready_magic ^
                                 request.request_id)) {
                            throw std::runtime_error(
                                    "invalid phone DMA payload-ready response");
                        }
                        if (!usb.exchange(
                                    request_buffer.data() +
                                            ffn_split::dmabuf_payload_offset,
                                    payload_bytes, response_buffer.data(),
                                    wire_bytes, identity, payload_transfer,
                                    exchange_error)) {
                            throw std::runtime_error(exchange_error);
                        }
                        if (calls == 0) {
                            fprintf(stderr,
                                    "[ffn-dmabuf-bridge] first payload sent\n");
                            fflush(stderr);
                        }
                    } else {
                        if (!usb.exchange(
                                    request_buffer.data(), wire_bytes,
                                    response_buffer.data(), wire_bytes,
                                    identity, payload_transfer,
                                    exchange_error)) {
                            throw std::runtime_error(exchange_error);
                        }
                    }
                    out_done = static_cast<double>(
                            payload_transfer.host_to_device_completed_ns) /
                            1e6;
                    if (calls == 0) {
                        fprintf(stderr,
                                "[ffn-dmabuf-bridge] waiting for first response\n");
                        fflush(stderr);
                    }
                    completed = static_cast<double>(
                            payload_transfer.device_to_host_completed_ns) /
                            1e6;
                    memcpy(&response, response_buffer.data(),
                            sizeof(response));
                    if (response.magic != ffn_split::protocol_magic ||
                        response.version != ffn_split::protocol_version ||
                        response.message != static_cast<uint16_t>(
                                ffn_split::message_type::execute_response) ||
                        response.status != 0 ||
                        response.request_id != request.request_id ||
                        response.layer != request.layer ||
                        response.elements != request.elements ||
                        response.payload_bytes != payload_bytes ||
                        response.columns != request.columns ||
                        response.tokens != request.tokens) {
                        throw std::runtime_error(
                                "invalid phone execute response");
                    }
                    break;
                } catch (const std::exception & error) {
                    if (reset_recoveries >= maximum_reset_recoveries) {
                        throw;
                    }
                    if (replay.empty()) {
                        replay.assign(
                                request_buffer.data(),
                                request_buffer.data() + wire_bytes);
                    }
                    ++reset_recoveries;
                    fprintf(stderr,
                            "[ffn-dmabuf-bridge] USB reset recovery=%llu "
                            "request_id=%u error=%s\n",
                            static_cast<unsigned long long>(reset_recoveries),
                            request.request_id, error.what());
                    fflush(stderr);
                    reconnect();
                    memcpy(request_buffer.data(), replay.data(), wire_bytes);
                }
            }
            prefetch_fence.finish(request, completed);
            if (!send_exact(client_fd, &response, sizeof(response)) ||
                !send_exact(
                        client_fd,
                        response_buffer.data() +
                                ffn_split::dmabuf_payload_offset,
                        payload_bytes)) {
                throw std::runtime_error("local execute response write failed");
            }

            rpc_samples.push_back(completed - started);
            out_samples.push_back(out_done - started);
            in_samples.push_back(completed - out_done);
            const double compute_ms =
                    static_cast<double>(response.compute_us) / 1000.0;
            compute_samples.push_back(compute_ms);
            d2h_exposed_samples.push_back(std::max(
                    0.0, completed - out_done - compute_ms));
            if (request.tokens == 1) {
                decode_rpc_samples.push_back(completed - started);
                ++decode_calls;
            } else {
                prefill_rpc_samples.push_back(completed - started);
                ++prefill_calls;
            }
            upload_bytes += payload_bytes;
            download_bytes += payload_bytes;
            ++calls;
            if (calls % 32 == 0) {
                fprintf(stderr,
                        "[ffn-dmabuf-bridge] calls=%llu usb_p50_ms=%.3f\n",
                        static_cast<unsigned long long>(calls),
                        percentile(rpc_samples, 0.50));
                fflush(stderr);
            }
        }

        prefetch_fence.require_complete();
        if (clean_disconnect) {
            ffn_split::execute_request shutdown = {};
            shutdown.magic = ffn_split::protocol_magic;
            shutdown.version = ffn_split::protocol_version;
            shutdown.message = static_cast<uint16_t>(
                    ffn_split::message_type::execute_request);
            shutdown.layer = terminate_phone_session ? -1 : 0;
            bool sent = false;
            while (!sent) {
                try {
                    memset(request_buffer.data(), 0, request_buffer.size());
                    memcpy(request_buffer.data(), &shutdown, sizeof(shutdown));
                    ffn_split::usb_transfer_record shutdown_transfer;
                    std::string shutdown_error;
                    if (!usb.exchange(
                                request_buffer.data(),
                                split_h2d ? sizeof(shutdown) :
                                        ffn_split::dmabuf_payload_offset,
                                nullptr, 0, {}, shutdown_transfer,
                                shutdown_error)) {
                        throw std::runtime_error(shutdown_error);
                    }
                    sent = true;
                } catch (const std::exception & error) {
                    if (reset_recoveries >= maximum_reset_recoveries) {
                        throw;
                    }
                    ++reset_recoveries;
                    fprintf(stderr,
                            "[ffn-dmabuf-bridge] USB reset during shutdown "
                            "recovery=%llu error=%s\n",
                            static_cast<unsigned long long>(reset_recoveries),
                            error.what());
                    reconnect();
                }
            }
        }

        fprintf(stderr,
                "FFNDMABUF {\"status\":\"ok\",\"calls\":%llu,"
                "\"allocator\":\"%s\",\"host_allocator\":\"%s\","
                "\"queue_depth\":%u,"
                "\"configured_queue_depth\":%u,"
                "\"transport_generation\":\"%s\","
                "\"async_libusb\":true,"
                "\"max_wire_bytes\":%zu,"
                "\"decode_calls\":%llu,\"prefill_calls\":%llu,"
                "\"reset_recoveries\":%llu,"
                "\"upload_bytes\":%llu,\"download_bytes\":%llu,"
                "\"usb_p50_ms\":%.6f,\"usb_p90_ms\":%.6f,"
                "\"out_p50_ms\":%.6f,\"out_p90_ms\":%.6f,"
                "\"in_p50_ms\":%.6f,\"in_p90_ms\":%.6f,"
                "\"compute_p50_ms\":%.6f,\"compute_p90_ms\":%.6f,"
                "\"decode_usb_p50_ms\":%.6f,"
                "\"decode_usb_p90_ms\":%.6f,"
                "\"prefill_usb_p50_ms\":%.6f,"
                "\"prefill_usb_p90_ms\":%.6f,"
                "\"prefetch_fence_enabled\":%s,"
                "\"prefetch_fence_completed\":%s,"
                "\"prefetch_expected_total_bytes\":%llu,"
                "\"prefetch_group_first_layer\":%d,"
                "\"prefetch_group_last_layer\":%d,"
                "\"prefetch_fence_calls\":%llu,"
                "\"prefetch_copied_bytes\":%llu,"
                "\"prefetch_copied_chunks\":%llu,"
                "\"prefetch_copy_p50_ms\":%.6f,"
                "\"prefetch_copy_p90_ms\":%.6f,"
                "\"prefetch_window_min_ms\":%.6f,"
                "\"prefetch_window_p10_ms\":%.6f,"
                "\"prefetch_window_p50_ms\":%.6f,"
                "\"prefetch_window_p90_ms\":%.6f,"
                "\"prefetch_decode_window_min_ms\":%.6f,"
                "\"prefetch_decode_window_p10_ms\":%.6f,"
                "\"prefetch_decode_window_p50_ms\":%.6f,"
                "\"prefetch_prefill_window_min_ms\":%.6f,"
                "\"prefetch_prefill_window_p10_ms\":%.6f,"
                "\"prefetch_prefill_window_p50_ms\":%.6f,"
                "\"phone_idle_min_ms\":%.6f,"
                "\"phone_idle_p10_ms\":%.6f,"
                "\"phone_idle_p50_ms\":%.6f,"
                "\"phone_idle_p90_ms\":%.6f,"
                "\"prefetch_overrun_p90_ms\":%.6f,"
                "\"prefetch_overrun_max_ms\":%.6f,"
                "\"prefetch_copied_window_overrun_p90_ms\":%.6f,"
                "\"prefetch_copied_window_overrun_max_ms\":%.6f}\n",
                static_cast<unsigned long long>(calls), allocator.c_str(),
                ffn_split::usb_host_allocator_name(host_allocator),
                usb.queue_depth(), configured_queue_depth,
                transport_generation.c_str(), max_wire_bytes,
                static_cast<unsigned long long>(decode_calls),
                static_cast<unsigned long long>(prefill_calls),
                static_cast<unsigned long long>(reset_recoveries),
                static_cast<unsigned long long>(upload_bytes),
                static_cast<unsigned long long>(download_bytes),
                percentile(rpc_samples, 0.50),
                percentile(rpc_samples, 0.90),
                percentile(out_samples, 0.50),
                percentile(out_samples, 0.90),
                percentile(in_samples, 0.50),
                percentile(in_samples, 0.90),
                percentile(compute_samples, 0.50),
                percentile(compute_samples, 0.90),
                percentile(decode_rpc_samples, 0.50),
                percentile(decode_rpc_samples, 0.90),
                percentile(prefill_rpc_samples, 0.50),
                percentile(prefill_rpc_samples, 0.90),
                prefetch_fence.configured() ? "true" : "false",
                prefetch_fence.completed() ? "true" : "false",
                static_cast<unsigned long long>(
                        prefetch_fence.expected_total_bytes()),
                prefetch_fence.group_first_layer(),
                prefetch_fence.group_last_layer(),
                static_cast<unsigned long long>(prefetch_fence.calls()),
                static_cast<unsigned long long>(
                        prefetch_fence.copied_bytes()),
                static_cast<unsigned long long>(
                        prefetch_fence.copied_chunks()),
                prefetch_fence.copy_p50_ms(),
                prefetch_fence.copy_p90_ms(),
                prefetch_fence.window_min_ms(),
                prefetch_fence.window_p10_ms(),
                prefetch_fence.window_p50_ms(),
                prefetch_fence.window_p90_ms(),
                prefetch_fence.decode_window_min_ms(),
                prefetch_fence.decode_window_p10_ms(),
                prefetch_fence.decode_window_p50_ms(),
                prefetch_fence.prefill_window_min_ms(),
                prefetch_fence.prefill_window_p10_ms(),
                prefetch_fence.prefill_window_p50_ms(),
                prefetch_fence.phone_idle_min_ms(),
                prefetch_fence.phone_idle_p10_ms(),
                prefetch_fence.phone_idle_p50_ms(),
                prefetch_fence.phone_idle_p90_ms(),
                prefetch_fence.overrun_p90_ms(),
                prefetch_fence.overrun_max_ms(),
                prefetch_fence.copied_window_overrun_p90_ms(),
                prefetch_fence.copied_window_overrun_max_ms());

        close(client_fd);
        close(listen_fd);
        usb.close();
        return 0;
    } catch (const std::exception & error) {
        fprintf(stderr, "[ffn-dmabuf-bridge] %s\n", error.what());
        return 1;
    }
}
#endif
