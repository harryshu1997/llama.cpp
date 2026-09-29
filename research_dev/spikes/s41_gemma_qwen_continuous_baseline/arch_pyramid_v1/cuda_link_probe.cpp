#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>

static void check(cudaError_t status, const char * operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(1);
    }
}

static double measure(
        void * dst,
        const void * src,
        size_t bytes,
        cudaMemcpyKind kind,
        int repetitions) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check(cudaEventCreate(&start), "cudaEventCreate(start)");
    check(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    for (int i = 0; i < 5; ++i) {
        check(cudaMemcpy(dst, src, bytes, kind), "warmup cudaMemcpy");
    }
    check(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int i = 0; i < repetitions; ++i) {
        check(cudaMemcpyAsync(dst, src, bytes, kind), "cudaMemcpyAsync");
    }
    check(cudaEventRecord(stop), "cudaEventRecord(stop)");
    check(cudaEventSynchronize(stop), "cudaEventSynchronize");

    float elapsed_ms = 0.0f;
    check(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
    check(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
    check(cudaEventDestroy(start), "cudaEventDestroy(start)");
    return static_cast<double>(bytes) * repetitions / (elapsed_ms * 1.0e6);
}

int main() {
    constexpr size_t bytes = 256ULL * 1024 * 1024;
    constexpr int repetitions = 20;
    void * host = nullptr;
    void * device_a = nullptr;
    void * device_b = nullptr;

    check(cudaSetDevice(0), "cudaSetDevice");
    check(cudaMallocHost(&host, bytes), "cudaMallocHost");
    check(cudaMalloc(&device_a, bytes), "cudaMalloc(device_a)");
    check(cudaMalloc(&device_b, bytes), "cudaMalloc(device_b)");
    check(cudaMemset(device_a, 0x5a, bytes), "cudaMemset");

    const double h2d = measure(
            device_a, host, bytes, cudaMemcpyHostToDevice, repetitions);
    const double d2h = measure(
            host, device_a, bytes, cudaMemcpyDeviceToHost, repetitions);
    const double d2d = measure(
            device_b, device_a, bytes, cudaMemcpyDeviceToDevice, repetitions);

    std::printf("bytes=%zu repetitions=%d h2d_GBps=%.3f d2h_GBps=%.3f "
                "d2d_copy_GBps=%.3f d2d_memory_traffic_GBps=%.3f\n",
                bytes, repetitions, h2d, d2h, d2d, 2.0 * d2d);

    check(cudaFree(device_b), "cudaFree(device_b)");
    check(cudaFree(device_a), "cudaFree(device_a)");
    check(cudaFreeHost(host), "cudaFreeHost");
    return 0;
}
