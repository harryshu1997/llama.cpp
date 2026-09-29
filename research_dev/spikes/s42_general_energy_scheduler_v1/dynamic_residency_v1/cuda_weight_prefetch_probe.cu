#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#include <vector>

namespace {

using steady_clock = std::chrono::steady_clock;

constexpr uint32_t fence_magic = UINT32_C(0x53343250);
constexpr uint16_t fence_version = 1;
constexpr uint16_t fence_begin = 1;
constexpr uint16_t fence_done = 2;
constexpr uint64_t fnv_offset = UINT64_C(14695981039346656037);
constexpr uint64_t fnv_prime = UINT64_C(1099511628211);

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

int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            steady_clock::now().time_since_epoch()).count();
}

uint64_t parse_u64(const char * text, const char * name) {
    errno = 0;
    char * end = nullptr;
    const unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        throw std::runtime_error(std::string("invalid ") + name);
    }
    return static_cast<uint64_t>(value);
}

void require_cuda(cudaError_t status, const char * operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                cudaGetErrorString(status));
    }
}

void pread_exact(int fd, void * data, size_t size, uint64_t offset) {
    auto * cursor = static_cast<unsigned char *>(data);
    size_t remaining = size;
    while (remaining > 0) {
        const ssize_t count = pread(
                fd, cursor, remaining, static_cast<off_t>(offset));
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            throw std::runtime_error("model range read failed");
        }
        cursor += count;
        remaining -= static_cast<size_t>(count);
        offset += static_cast<uint64_t>(count);
    }
}

uint64_t fnv1a(uint64_t value, const unsigned char * data, size_t size) {
    for (size_t index = 0; index < size; ++index) {
        value ^= data[index];
        value *= fnv_prime;
    }
    return value;
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

int open_listener(const char * path) {
    sockaddr_un address = {};
    if (path[0] != '/' || strlen(path) >= sizeof(address.sun_path)) {
        throw std::runtime_error("invalid fence socket path");
    }
    const int fd = socket(AF_UNIX, SOCK_SEQPACKET, 0);
    if (fd < 0) {
        throw std::runtime_error(std::string("socket failed: ") +
                strerror(errno));
    }
    address.sun_family = AF_UNIX;
    strcpy(address.sun_path, path);
    if (bind(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0 ||
        listen(fd, 1) != 0) {
        const std::string message = std::string("listen failed: ") +
                strerror(errno);
        close(fd);
        throw std::runtime_error(message);
    }
    return fd;
}

void send_response(int fd, const fence_response & response) {
    ssize_t count;
    do {
        count = send(fd, &response, sizeof(response), MSG_NOSIGNAL);
    } while (count < 0 && errno == EINTR);
    if (count != static_cast<ssize_t>(sizeof(response))) {
        throw std::runtime_error("fence response send failed");
    }
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 10) {
        fprintf(stderr,
                "usage: %s <observe|prefetch> <socket> <arm-file> <model> "
                "<offset> <bytes> <chunk-bytes> <chunks-per-window> "
                "<gpu-reserve-bytes>\n",
                argv[0]);
        return 2;
    }

    const std::string mode = argv[1];
    const char * socket_path = argv[2];
    const char * arm_path = argv[3];
    const char * model_path = argv[4];
    int listener = -1;
    int connection = -1;
    int model_fd = -1;
    cudaStream_t stream = nullptr;
    unsigned char * source = nullptr;
    unsigned char * compare = nullptr;
    unsigned char * device = nullptr;
    unsigned char * scratch = nullptr;
    bool socket_bound = false;
    int result = 1;
    try {
        if (mode != "observe" && mode != "prefetch") {
            throw std::runtime_error("invalid mode");
        }
        if (arm_path[0] != '/' || model_path[0] != '/') {
            throw std::runtime_error("arm and model paths must be absolute");
        }
        const uint64_t source_offset = parse_u64(argv[5], "offset");
        const uint64_t stage_bytes = parse_u64(argv[6], "bytes");
        const uint64_t chunk_bytes_value = parse_u64(argv[7], "chunk bytes");
        const uint64_t chunks_per_window = parse_u64(
                argv[8], "chunks per window");
        const uint64_t gpu_reserve_bytes = parse_u64(
                argv[9], "GPU reserve bytes");
        if (stage_bytes == 0 || chunk_bytes_value == 0 ||
            chunk_bytes_value > UINT64_C(64) * 1024 * 1024 ||
            gpu_reserve_bytes == 0 ||
            (mode == "observe" && chunks_per_window != 0) ||
            (mode == "prefetch" && chunks_per_window == 0) ||
            chunks_per_window > UINT32_MAX) {
            throw std::runtime_error("invalid transfer geometry");
        }
        if (stage_bytes > static_cast<uint64_t>(SIZE_MAX) ||
            chunk_bytes_value > static_cast<uint64_t>(SIZE_MAX) ||
            source_offset > static_cast<uint64_t>(INT64_MAX) - stage_bytes) {
            throw std::runtime_error("transfer geometry exceeds host limits");
        }
        const size_t chunk_bytes = static_cast<size_t>(chunk_bytes_value);

        model_fd = open(model_path, O_RDONLY | O_CLOEXEC);
        if (model_fd < 0) {
            throw std::runtime_error(std::string("model open failed: ") +
                    strerror(errno));
        }
        struct stat model_stat = {};
        if (fstat(model_fd, &model_stat) != 0 || model_stat.st_size < 0 ||
            source_offset + stage_bytes >
                    static_cast<uint64_t>(model_stat.st_size)) {
            throw std::runtime_error("model range is outside the file");
        }

        require_cuda(cudaSetDevice(0), "cudaSetDevice");
        int least_priority = 0;
        int greatest_priority = 0;
        require_cuda(cudaDeviceGetStreamPriorityRange(
                &least_priority, &greatest_priority),
                "cudaDeviceGetStreamPriorityRange");
        require_cuda(cudaStreamCreateWithPriority(
                &stream, cudaStreamNonBlocking, greatest_priority),
                "cudaStreamCreateWithPriority");
        require_cuda(cudaHostAlloc(
                reinterpret_cast<void **>(&source),
                static_cast<size_t>(stage_bytes),
                cudaHostAllocDefault), "cudaHostAlloc source");

        size_t free_before = 0;
        size_t total_bytes = 0;
        require_cuda(cudaMemGetInfo(&free_before, &total_bytes),
                "cudaMemGetInfo before allocation");
        require_cuda(cudaMalloc(
                reinterpret_cast<void **>(&device),
                static_cast<size_t>(stage_bytes)), "cudaMalloc destination");
        require_cuda(cudaMalloc(
                reinterpret_cast<void **>(&scratch), chunk_bytes),
                "cudaMalloc scratch");
        require_cuda(cudaMemsetAsync(
                device, 0, static_cast<size_t>(stage_bytes), stream),
                "cudaMemsetAsync destination");
        require_cuda(cudaMemsetAsync(
                scratch, 0, chunk_bytes, stream),
                "cudaMemsetAsync scratch");
        require_cuda(cudaStreamSynchronize(stream),
                "cudaStreamSynchronize initialization");
        size_t free_after_allocation = 0;
        require_cuda(cudaMemGetInfo(&free_after_allocation, &total_bytes),
                "cudaMemGetInfo after allocation");
        if (free_after_allocation < gpu_reserve_bytes) {
            throw std::runtime_error("GPU reserve violated at allocation");
        }

        uint64_t source_hash = fnv_offset;
        for (uint64_t completed = 0; completed < stage_bytes;) {
            const size_t bytes = static_cast<size_t>(std::min<uint64_t>(
                    chunk_bytes_value, stage_bytes - completed));
            pread_exact(
                    model_fd, source + completed, bytes,
                    source_offset + completed);
            source_hash = fnv1a(source_hash, source + completed, bytes);
            completed += bytes;
        }

        listener = open_listener(socket_path);
        socket_bound = true;
        fprintf(stderr,
                "PREFETCH_READY mode=%s bytes=%llu chunk_bytes=%zu "
                "free_after_allocation=%zu\n",
                mode.c_str(), static_cast<unsigned long long>(stage_bytes),
                chunk_bytes, free_after_allocation);
        fflush(stderr);
        connection = accept(listener, nullptr, nullptr);
        if (connection < 0) {
            throw std::runtime_error("fence accept failed");
        }
        close(listener);
        listener = -1;

        uint64_t last_sequence = 0;
        uint64_t copied_bytes = 0;
        uint64_t copied_chunks = 0;
        uint64_t fence_calls = 0;
        uint64_t armed_calls = 0;
        uint64_t copy_windows = 0;
        uint64_t warmup_copy_calls = 0;
        uint64_t warmup_copy_bytes = 0;
        size_t minimum_free_bytes = free_after_allocation;
        std::vector<double> copy_samples;
        bool arm_seen = false;
        bool protocol_failed = false;
        for (;;) {
            fence_request request = {};
            ssize_t count;
            do {
                count = recv(connection, &request, sizeof(request), 0);
            } while (count < 0 && errno == EINTR);
            if (count == 0) {
                break;
            }
            if (count != static_cast<ssize_t>(sizeof(request)) ||
                request.magic != fence_magic ||
                request.version != fence_version ||
                request.message != fence_begin ||
                request.sequence != last_sequence + 1 ||
                request.request_id == 0 || request.tokens == 0 ||
                request.begin_ns <= 0) {
                throw std::runtime_error("invalid fence request");
            }
            last_sequence = request.sequence;
            ++fence_calls;
            if (!arm_seen && access(arm_path, F_OK) == 0) {
                arm_seen = true;
            }
            if (arm_seen) {
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
            require_cuda(cudaMemGetInfo(&free_now, &total_bytes),
                    "cudaMemGetInfo at fence");
            minimum_free_bytes = std::min(minimum_free_bytes, free_now);
            if (free_now < gpu_reserve_bytes) {
                response.status = 1;
                protocol_failed = true;
            } else if (!arm_seen) {
                const size_t bytes = static_cast<size_t>(std::min<uint64_t>(
                        chunk_bytes_value, stage_bytes));
                require_cuda(cudaMemcpyAsync(
                        scratch, source, bytes, cudaMemcpyHostToDevice, stream),
                        "cudaMemcpyAsync warmup H2D");
                require_cuda(cudaStreamSynchronize(stream),
                        "cudaStreamSynchronize warmup H2D");
                ++warmup_copy_calls;
                warmup_copy_bytes += bytes;
            } else if (mode == "prefetch") {
                for (uint64_t chunk = 0;
                     chunk < chunks_per_window && copied_bytes < stage_bytes;
                     ++chunk) {
                    const size_t bytes = static_cast<size_t>(
                            std::min<uint64_t>(
                                    chunk_bytes_value,
                                    stage_bytes - copied_bytes));
                    require_cuda(cudaMemcpyAsync(
                            device + copied_bytes, source + copied_bytes, bytes,
                            cudaMemcpyHostToDevice, stream),
                            "cudaMemcpyAsync staged H2D");
                    require_cuda(cudaStreamSynchronize(stream),
                            "cudaStreamSynchronize staged H2D");
                    copied_bytes += bytes;
                    ++copied_chunks;
                    response.copied_bytes += bytes;
                    ++response.copied_chunks;
                }
            }
            response.copy_completed_ns = now_ns();
            if (response.copied_bytes != 0) {
                ++copy_windows;
                copy_samples.push_back(static_cast<double>(
                        response.copy_completed_ns -
                                response.copy_started_ns) / 1e6);
            }
            send_response(connection, response);
            printf(
                    "PREFETCH_WINDOW sequence=%llu request_id=%u layer=%u "
                    "tokens=%u armed=%u copied_bytes=%llu copied_chunks=%u "
                    "copy_ms=%.6f free_bytes=%zu status=%u\n",
                    static_cast<unsigned long long>(request.sequence),
                    request.request_id, request.layer, request.tokens,
                    arm_seen ? 1U : 0U,
                    static_cast<unsigned long long>(response.copied_bytes),
                    response.copied_chunks,
                    static_cast<double>(response.copy_completed_ns -
                            response.copy_started_ns) / 1e6,
                    free_now, response.status);
            fflush(stdout);
            if (protocol_failed) {
                break;
            }
        }

        bool verified = false;
        uint64_t destination_hash = fnv_offset;
        if (!protocol_failed && mode == "prefetch" &&
            copied_bytes == stage_bytes) {
            require_cuda(cudaHostAlloc(
                    reinterpret_cast<void **>(&compare), chunk_bytes,
                    cudaHostAllocDefault), "cudaHostAlloc compare");
            verified = true;
            for (uint64_t completed = 0; completed < stage_bytes;) {
                const size_t bytes = static_cast<size_t>(std::min<uint64_t>(
                        chunk_bytes_value, stage_bytes - completed));
                require_cuda(cudaMemcpyAsync(
                        compare, device + completed, bytes,
                        cudaMemcpyDeviceToHost, stream),
                        "cudaMemcpyAsync verification D2H");
                require_cuda(cudaStreamSynchronize(stream),
                        "cudaStreamSynchronize verification D2H");
                if (memcmp(source + completed, compare, bytes) != 0) {
                    verified = false;
                    break;
                }
                destination_hash = fnv1a(destination_hash, compare, bytes);
                completed += bytes;
            }
            verified = verified && destination_hash == source_hash;
        }
        const bool passed = !protocol_failed && arm_seen &&
                (mode == "observe" ||
                 (copied_bytes == stage_bytes && verified));
        printf(
                "PREFETCH_RESULT status=%s mode=%s source_offset=%llu "
                "stage_bytes=%llu chunk_bytes=%zu chunks_per_window=%llu "
                "fence_calls=%llu armed_calls=%llu copy_windows=%llu "
                "warmup_copy_calls=%llu warmup_copy_bytes=%llu "
                "copied_bytes=%llu copied_chunks=%llu copy_p50_ms=%.6f "
                "copy_p90_ms=%.6f gpu_total_bytes=%zu "
                "gpu_free_before_bytes=%zu gpu_free_after_allocation_bytes=%zu "
                "gpu_free_min_bytes=%zu gpu_reserve_bytes=%llu "
                "source_resident_bytes=%llu source_pinned=true "
                "stream_priority=%d "
                "source_fnv64=%016llx destination_fnv64=%016llx "
                "verified=%s adoptable=false\n",
                passed ? "PASS" : "FAIL", mode.c_str(),
                static_cast<unsigned long long>(source_offset),
                static_cast<unsigned long long>(stage_bytes), chunk_bytes,
                static_cast<unsigned long long>(chunks_per_window),
                static_cast<unsigned long long>(fence_calls),
                static_cast<unsigned long long>(armed_calls),
                static_cast<unsigned long long>(copy_windows),
                static_cast<unsigned long long>(warmup_copy_calls),
                static_cast<unsigned long long>(warmup_copy_bytes),
                static_cast<unsigned long long>(copied_bytes),
                static_cast<unsigned long long>(copied_chunks),
                percentile(copy_samples, 0.50),
                percentile(copy_samples, 0.90), total_bytes, free_before,
                free_after_allocation, minimum_free_bytes,
                static_cast<unsigned long long>(gpu_reserve_bytes),
                static_cast<unsigned long long>(stage_bytes),
                greatest_priority,
                static_cast<unsigned long long>(source_hash),
                static_cast<unsigned long long>(destination_hash),
                verified ? "true" : "false");
        fflush(stdout);
        result = passed ? 0 : 3;
    } catch (const std::exception & error) {
        fprintf(stderr, "PREFETCH_ERROR %s\n", error.what());
    }

    if (connection >= 0) {
        close(connection);
    }
    if (listener >= 0) {
        close(listener);
    }
    if (socket_bound) {
        unlink(socket_path);
    }
    if (compare != nullptr) {
        cudaFreeHost(compare);
    }
    if (device != nullptr) {
        cudaFree(device);
    }
    if (scratch != nullptr) {
        cudaFree(scratch);
    }
    if (source != nullptr) {
        cudaFreeHost(source);
    }
    if (stream != nullptr) {
        cudaStreamDestroy(stream);
    }
    if (model_fd >= 0) {
        close(model_fd);
    }
    return result;
}
