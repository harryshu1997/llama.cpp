#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using steady_clock = std::chrono::steady_clock;

static void require_cuda(cudaError_t status, const char * operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(1);
    }
}

static int64_t unix_time_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
}

static double elapsed_ms(steady_clock::time_point started) {
    return std::chrono::duration<double, std::milli>(
            steady_clock::now() - started).count();
}

static uint64_t parse_u64(const char * text, const char * name) {
    char * end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0') {
        std::fprintf(stderr, "invalid %s\n", name);
        std::exit(2);
    }
    return static_cast<uint64_t>(value);
}

static double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const size_t index = std::min(
            values.size() - 1,
            static_cast<size_t>(fraction * values.size()));
    return values[index];
}

int main(int argc, char ** argv) {
    if (argc != 5) {
        std::fprintf(stderr,
                "usage: %s <h2d|d2h|duplex> <bytes> <warmup> <iterations>\n",
                argv[0]);
        return 2;
    }
    const std::string mode = argv[1];
    const uint64_t bytes_value = parse_u64(argv[2], "bytes");
    const uint64_t warmup = parse_u64(argv[3], "warmup");
    const uint64_t iterations = parse_u64(argv[4], "iterations");
    if ((mode != "h2d" && mode != "d2h" && mode != "duplex") ||
            bytes_value < 64 || bytes_value > UINT64_C(1) << 30 ||
            iterations == 0 || warmup > 1000000 || iterations > 10000000) {
        std::fprintf(stderr, "invalid arguments\n");
        return 2;
    }
    const size_t bytes = static_cast<size_t>(bytes_value);
    require_cuda(cudaSetDevice(0), "cudaSetDevice");
    unsigned char * host_a = nullptr;
    unsigned char * host_b = nullptr;
    unsigned char * device_a = nullptr;
    unsigned char * device_b = nullptr;
    require_cuda(cudaHostAlloc(&host_a, bytes, cudaHostAllocDefault),
            "cudaHostAlloc A");
    require_cuda(cudaMalloc(&device_a, bytes), "cudaMalloc A");
    if (mode == "duplex") {
        require_cuda(cudaHostAlloc(&host_b, bytes, cudaHostAllocDefault),
                "cudaHostAlloc B");
        require_cuda(cudaMalloc(&device_b, bytes), "cudaMalloc B");
    }
    std::memset(host_a, 0xa5, bytes);
    if (host_b != nullptr) {
        std::memset(host_b, 0x5a, bytes);
    }
    require_cuda(cudaMemset(device_a, 0x3c, bytes), "cudaMemset A");
    if (device_b != nullptr) {
        require_cuda(cudaMemset(device_b, 0xc3, bytes), "cudaMemset B");
    }

    cudaStream_t stream_a = nullptr;
    cudaStream_t stream_b = nullptr;
    require_cuda(cudaStreamCreate(&stream_a), "cudaStreamCreate A");
    if (mode == "duplex") {
        require_cuda(cudaStreamCreate(&stream_b), "cudaStreamCreate B");
    }
    auto transfer = [&]() {
        if (mode == "h2d") {
            require_cuda(cudaMemcpyAsync(
                    device_a, host_a, bytes, cudaMemcpyHostToDevice, stream_a),
                    "cudaMemcpyAsync H2D");
            require_cuda(cudaStreamSynchronize(stream_a), "cudaSync H2D");
        } else if (mode == "d2h") {
            require_cuda(cudaMemcpyAsync(
                    host_a, device_a, bytes, cudaMemcpyDeviceToHost, stream_a),
                    "cudaMemcpyAsync D2H");
            require_cuda(cudaStreamSynchronize(stream_a), "cudaSync D2H");
        } else {
            require_cuda(cudaMemcpyAsync(
                    device_a, host_a, bytes, cudaMemcpyHostToDevice, stream_a),
                    "cudaMemcpyAsync duplex H2D");
            require_cuda(cudaMemcpyAsync(
                    host_b, device_b, bytes, cudaMemcpyDeviceToHost, stream_b),
                    "cudaMemcpyAsync duplex D2H");
            require_cuda(cudaStreamSynchronize(stream_a), "cudaSync duplex H2D");
            require_cuda(cudaStreamSynchronize(stream_b), "cudaSync duplex D2H");
        }
    };

    for (uint64_t i = 0; i < warmup; ++i) {
        transfer();
    }
    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(iterations));
    const auto paid_started = steady_clock::now();
    std::printf("ENERGY_WINDOW_START unix_ns=%lld mode=%s n=%llu\n",
            static_cast<long long>(unix_time_ns()), mode.c_str(),
            static_cast<unsigned long long>(iterations));
    std::fflush(stdout);
    for (uint64_t i = 0; i < iterations; ++i) {
        const auto started = steady_clock::now();
        transfer();
        samples.push_back(elapsed_ms(started));
    }
    const double paid_s = elapsed_ms(paid_started) / 1000.0;
    std::printf("ENERGY_WINDOW_END unix_ns=%lld mode=%s n=%llu\n",
            static_cast<long long>(unix_time_ns()), mode.c_str(),
            static_cast<unsigned long long>(iterations));
    const uint64_t directions = mode == "duplex" ? 2 : 1;
    const double rate_gbps = bytes * static_cast<double>(iterations) *
            directions / paid_s / 1e9;
    std::printf(
            "PCIE_RESULT status=PASS mode=%s bytes=%zu iterations=%llu "
            "p50_ms=%.6f p90_ms=%.6f aggregate_GBps=%.6f\n",
            mode.c_str(), bytes, static_cast<unsigned long long>(iterations),
            percentile(samples, 0.50), percentile(samples, 0.90), rate_gbps);

    if (stream_b != nullptr) {
        cudaStreamDestroy(stream_b);
    }
    cudaStreamDestroy(stream_a);
    if (device_b != nullptr) {
        cudaFree(device_b);
    }
    cudaFree(device_a);
    if (host_b != nullptr) {
        cudaFreeHost(host_b);
    }
    cudaFreeHost(host_a);
    return 0;
}
