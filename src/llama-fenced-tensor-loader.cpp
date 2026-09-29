#include "llama-fenced-tensor-loader.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#if defined(__linux__)
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#endif

namespace llama_fenced_tensor {
namespace {

using steady_clock = std::chrono::steady_clock;

constexpr uint32_t fence_magic = UINT32_C(0x53343250);
constexpr uint16_t fence_version = 1;
constexpr uint16_t fence_begin = 1;
constexpr uint16_t fence_done = 2;

struct fence_request {
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

struct fence_response {
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

static_assert(sizeof(fence_request) == 40);
static_assert(sizeof(fence_response) == 56);

struct config {
    bool enabled = false;
    std::string socket_path;
    std::string arm_path;
    std::string tensor_name;
    size_t expected_offset = 0;
    size_t expected_bytes = 0;
    size_t chunk_bytes = 0;
    uint32_t chunks_per_window = 0;
    size_t gpu_reserve_bytes = 0;
};

constexpr std::array<const char *, 8> config_keys = {
    "S42_FENCED_TENSOR_SOCKET",
    "S42_FENCED_TENSOR_ARM_FILE",
    "S42_FENCED_TENSOR_NAME",
    "S42_FENCED_TENSOR_EXPECTED_OFFSET",
    "S42_FENCED_TENSOR_EXPECTED_BYTES",
    "S42_FENCED_TENSOR_CHUNK_BYTES",
    "S42_FENCED_TENSOR_CHUNKS_PER_WINDOW",
    "S42_FENCED_TENSOR_GPU_RESERVE_BYTES",
};

uint64_t parse_u64(const char * text, const char * name) {
    errno = 0;
    char * end = nullptr;
    const unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        throw std::runtime_error(std::string("invalid ") + name);
    }
    return static_cast<uint64_t>(value);
}

config read_config() {
    config result;
    bool any = false;
    for (const char * key : config_keys) {
        const char * value = getenv(key);
        any = any || value != nullptr;
    }
    if (!any) {
        return result;
    }
    for (const char * key : config_keys) {
        const char * value = getenv(key);
        if (value == nullptr || *value == '\0') {
            throw std::runtime_error(std::string(
                    "incomplete fenced tensor configuration: ") + key);
        }
    }
#if !defined(__linux__)
    throw std::runtime_error(
            "fenced tensor loading is supported only on Linux");
#else
    result.enabled = true;
    result.socket_path = getenv(config_keys[0]);
    result.arm_path = getenv(config_keys[1]);
    result.tensor_name = getenv(config_keys[2]);
    if (result.socket_path.front() != '/' || result.arm_path.front() != '/' ||
        result.socket_path.size() >= sizeof(sockaddr_un{}.sun_path)) {
        throw std::runtime_error("invalid fenced tensor path");
    }
    const uint64_t offset = parse_u64(
            getenv(config_keys[3]), "fenced tensor offset");
    const uint64_t bytes = parse_u64(
            getenv(config_keys[4]), "fenced tensor bytes");
    const uint64_t chunk = parse_u64(
            getenv(config_keys[5]), "fenced tensor chunk bytes");
    const uint64_t chunks = parse_u64(
            getenv(config_keys[6]), "fenced tensor chunks per window");
    const uint64_t reserve = parse_u64(
            getenv(config_keys[7]), "fenced tensor GPU reserve");
    if (offset > std::numeric_limits<size_t>::max() || bytes == 0 ||
        bytes > std::numeric_limits<size_t>::max() || chunk == 0 ||
        chunk > UINT64_C(64) * 1024 * 1024 ||
        chunk > std::numeric_limits<size_t>::max() || chunks == 0 ||
        chunks > UINT32_MAX || reserve == 0 ||
        reserve > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("invalid fenced tensor geometry");
    }
    result.expected_offset = static_cast<size_t>(offset);
    result.expected_bytes = static_cast<size_t>(bytes);
    result.chunk_bytes = static_cast<size_t>(chunk);
    result.chunks_per_window = static_cast<uint32_t>(chunks);
    result.gpu_reserve_bytes = static_cast<size_t>(reserve);
    return result;
#endif
}

int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            steady_clock::now().time_since_epoch()).count();
}

#if defined(__linux__)
void send_response(int fd, const fence_response & response) {
    ssize_t count;
    do {
        count = send(fd, &response, sizeof(response), MSG_NOSIGNAL);
    } while (count < 0 && errno == EINTR);
    if (count != static_cast<ssize_t>(sizeof(response))) {
        throw std::runtime_error("fenced tensor response send failed");
    }
}
#endif

} // namespace

bool configured() {
    return read_config().enabled;
}

bool matches(const char * tensor_name) {
    const config cfg = read_config();
    return cfg.enabled && tensor_name != nullptr &&
            cfg.tensor_name == tensor_name;
}

bool eligible(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->view_src != nullptr ||
        tensor->buffer == nullptr || tensor->data == nullptr ||
        ggml_backend_buffer_is_host(tensor->buffer)) {
        return false;
    }
    ggml_backend_buffer_type_t buft =
            ggml_backend_buffer_get_type(tensor->buffer);
    ggml_backend_dev_t device = ggml_backend_buft_get_device(buft);
    return device != nullptr &&
            ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU;
}

struct stage::impl {
    config cfg;
    ggml_tensor * tensor = nullptr;
    ggml_backend_dev_t device = nullptr;
    ggml_backend_buffer_t source_buffer = nullptr;
    ggml_backend_buffer_t compare_buffer = nullptr;
    unsigned char * source = nullptr;
    unsigned char * compare = nullptr;
    size_t source_offset = 0;
    size_t minimum_free_bytes = std::numeric_limits<size_t>::max();
    size_t copied_bytes = 0;
    uint64_t copied_chunks = 0;
    uint64_t copy_windows = 0;
    uint64_t fence_calls = 0;
    uint64_t armed_calls = 0;
#if defined(__linux__)
    int listener = -1;
    int connection = -1;
    bool socket_bound = false;
#endif

    impl(
            ggml_tensor * target,
            const void * input,
            size_t offset,
            size_t size) :
            cfg(read_config()), tensor(target), source_offset(offset) {
        if (!cfg.enabled || tensor == nullptr || input == nullptr ||
            cfg.tensor_name != ggml_get_name(tensor) ||
            cfg.expected_offset != offset || cfg.expected_bytes != size ||
            ggml_nbytes(tensor) != size || !eligible(tensor)) {
            throw std::runtime_error("fenced tensor identity mismatch");
        }
        ggml_backend_buffer_type_t buft =
                ggml_backend_buffer_get_type(tensor->buffer);
        device = ggml_backend_buft_get_device(buft);
        if (device == nullptr ||
            ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            throw std::runtime_error("fenced tensor is not on a GPU");
        }
        ggml_backend_buffer_type_t host_buft =
                ggml_backend_dev_host_buffer_type(device);
        if (host_buft == nullptr) {
            throw std::runtime_error(
                    "fenced tensor GPU has no pinned host buffer");
        }
        source_buffer = ggml_backend_buft_alloc_buffer(host_buft, size);
        compare_buffer = ggml_backend_buft_alloc_buffer(
                host_buft, cfg.chunk_bytes);
        if (source_buffer == nullptr || compare_buffer == nullptr ||
            ggml_backend_buffer_get_type(source_buffer) != host_buft ||
            ggml_backend_buffer_get_type(compare_buffer) != host_buft) {
            throw std::runtime_error(
                    "fenced tensor pinned host allocation failed");
        }
        source = static_cast<unsigned char *>(
                ggml_backend_buffer_get_base(source_buffer));
        compare = static_cast<unsigned char *>(
                ggml_backend_buffer_get_base(compare_buffer));
        memcpy(source, input, size);

        const size_t warm_bytes = std::min(cfg.chunk_bytes, size);
        ggml_backend_tensor_set(tensor, source, 0, warm_bytes);
        ggml_backend_tensor_get(tensor, compare, 0, warm_bytes);
        if (memcmp(source, compare, warm_bytes) != 0) {
            throw std::runtime_error("fenced tensor warmup verification failed");
        }
        size_t total = 0;
        ggml_backend_dev_memory(device, &minimum_free_bytes, &total);
        if (minimum_free_bytes < cfg.gpu_reserve_bytes) {
            throw std::runtime_error(
                    "fenced tensor reserve failed before READY");
        }
    }

    ~impl() {
#if defined(__linux__)
        if (connection >= 0) {
            close(connection);
        }
        if (listener >= 0) {
            close(listener);
        }
        if (socket_bound) {
            unlink(cfg.socket_path.c_str());
        }
#endif
        ggml_backend_buffer_free(compare_buffer);
        ggml_backend_buffer_free(source_buffer);
    }

#if defined(__linux__)
    void open_socket() {
        struct stat socket_stat = {};
        if (lstat(cfg.socket_path.c_str(), &socket_stat) == 0 ||
            errno != ENOENT) {
            throw std::runtime_error(
                    "fenced tensor socket path already exists");
        }
        if (access(cfg.arm_path.c_str(), F_OK) == 0 || errno != ENOENT) {
            throw std::runtime_error(
                    "fenced tensor arm path must not exist before READY");
        }
        sockaddr_un address = {};
        address.sun_family = AF_UNIX;
        strcpy(address.sun_path, cfg.socket_path.c_str());
        listener = socket(AF_UNIX, SOCK_SEQPACKET, 0);
        if (listener < 0) {
            throw std::runtime_error(std::string(
                    "fenced tensor listen failed: ") + strerror(errno));
        }
        if (bind(listener, reinterpret_cast<sockaddr *>(&address),
                    sizeof(address)) != 0) {
            throw std::runtime_error(std::string(
                    "fenced tensor bind failed: ") + strerror(errno));
        }
        socket_bound = true;
        if (chmod(cfg.socket_path.c_str(), 0600) != 0 ||
            listen(listener, 1) != 0) {
            throw std::runtime_error(std::string(
                    "fenced tensor listen failed: ") + strerror(errno));
        }
        fprintf(stderr,
                "S42_FENCED_TENSOR_READY tensor=%s offset=%zu bytes=%zu "
                "chunk_bytes=%zu chunks_per_window=%u source_pinned=true "
                "warmup_verified=true free_bytes=%zu reserve_bytes=%zu\n",
                cfg.tensor_name.c_str(), source_offset, cfg.expected_bytes,
                cfg.chunk_bytes, cfg.chunks_per_window, minimum_free_bytes,
                cfg.gpu_reserve_bytes);
        fflush(stderr);
        do {
            connection = accept(listener, nullptr, nullptr);
        } while (connection < 0 && errno == EINTR);
        if (connection < 0) {
            throw std::runtime_error("fenced tensor accept failed");
        }
        close(listener);
        listener = -1;
    }

    void run() {
        open_socket();
        uint64_t last_sequence = 0;
        bool armed = false;
        while (copied_bytes < cfg.expected_bytes) {
            fence_request request = {};
            ssize_t count;
            do {
                count = recv(connection, &request, sizeof(request), 0);
            } while (count < 0 && errno == EINTR);
            if (count == 0) {
                throw std::runtime_error(
                        "fenced tensor client closed before READY");
            }
            if (count != static_cast<ssize_t>(sizeof(request)) ||
                request.magic != fence_magic ||
                request.version != fence_version ||
                request.message != fence_begin ||
                request.sequence != last_sequence + 1 ||
                request.request_id == 0 || request.tokens == 0 ||
                request.begin_ns <= 0) {
                throw std::runtime_error("invalid fenced tensor request");
            }
            last_sequence = request.sequence;
            ++fence_calls;
            if (!armed && access(cfg.arm_path.c_str(), F_OK) == 0) {
                armed = true;
            }
            if (armed) {
                ++armed_calls;
            }

            fence_response response = {};
            response.magic = fence_magic;
            response.version = fence_version;
            response.message = fence_done;
            response.sequence = request.sequence;
            response.request_id = request.request_id;
            response.copy_started_ns = now_ns();

            size_t free_now = 0;
            size_t total = 0;
            ggml_backend_dev_memory(device, &free_now, &total);
            minimum_free_bytes = std::min(minimum_free_bytes, free_now);
            if (free_now < cfg.gpu_reserve_bytes) {
                response.status = 1;
            } else if (armed) {
                for (uint32_t chunk = 0;
                     chunk < cfg.chunks_per_window &&
                             copied_bytes < cfg.expected_bytes;
                     ++chunk) {
                    const size_t bytes = std::min(
                            cfg.chunk_bytes,
                            cfg.expected_bytes - copied_bytes);
                    ggml_backend_tensor_set(
                            tensor, source + copied_bytes,
                            copied_bytes, bytes);
                    ggml_backend_tensor_get(
                            tensor, compare, copied_bytes, bytes);
                    if (memcmp(source + copied_bytes, compare, bytes) != 0) {
                        response.status = 2;
                        break;
                    }
                    copied_bytes += bytes;
                    ++copied_chunks;
                    response.copied_bytes += bytes;
                    ++response.copied_chunks;
                }
            }
            response.copy_completed_ns = now_ns();
            if (response.copied_bytes != 0) {
                ++copy_windows;
            }
            send_response(connection, response);
            fprintf(stderr,
                    "S42_FENCED_TENSOR_WINDOW sequence=%llu request_id=%u "
                    "armed=%s copied_bytes=%llu copied_chunks=%u "
                    "copy_verify_ms=%.6f free_bytes=%zu status=%u\n",
                    static_cast<unsigned long long>(request.sequence),
                    request.request_id, armed ? "true" : "false",
                    static_cast<unsigned long long>(response.copied_bytes),
                    response.copied_chunks,
                    static_cast<double>(response.copy_completed_ns -
                            response.copy_started_ns) / 1e6,
                    free_now, response.status);
            fflush(stderr);
            if (response.status != 0) {
                throw std::runtime_error(
                        "fenced tensor copy or verification failed");
            }
        }
        fprintf(stderr,
                "S42_FENCED_TENSOR_RESULT status=PASS tensor=%s "
                "source_offset=%zu bytes=%zu chunk_bytes=%zu "
                "chunks_per_window=%u fence_calls=%llu armed_calls=%llu "
                "copy_windows=%llu copied_chunks=%llu verified=true "
                "source_pinned=true adoptable=true gpu_free_min_bytes=%zu "
                "gpu_reserve_bytes=%zu\n",
                cfg.tensor_name.c_str(), source_offset, cfg.expected_bytes,
                cfg.chunk_bytes, cfg.chunks_per_window,
                static_cast<unsigned long long>(fence_calls),
                static_cast<unsigned long long>(armed_calls),
                static_cast<unsigned long long>(copy_windows),
                static_cast<unsigned long long>(copied_chunks),
                minimum_free_bytes, cfg.gpu_reserve_bytes);
        fflush(stderr);
    }
#endif
};

stage::stage(
        ggml_tensor * tensor,
        const void * source,
        size_t source_offset,
        size_t size) :
        impl_(std::make_unique<impl>(
                tensor, source, source_offset, size)) {
}

stage::~stage() = default;

void stage::execute() {
#if defined(__linux__)
    impl_->run();
#else
    throw std::runtime_error(
            "fenced tensor loading is supported only on Linux");
#endif
}

size_t stage::size() const {
    return impl_->cfg.expected_bytes;
}

} // namespace llama_fenced_tensor
