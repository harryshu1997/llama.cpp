// cpustream: NEON xor-read of a buffer with T threads pinned to cores 2..7 for S seconds; prints GB/s. usage: cpustream MiB threads seconds
#include <arm_neon.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main(int argc, char ** argv) {
    size_t bytes = size_t(std::atoi(argv[1])) << 20; int T = std::atoi(argv[2]); double secs = std::atof(argv[3]);
    auto * data = (uint32_t *) mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    for (size_t i = 0; i < bytes / 4; ++i) data[i] = uint32_t(i * 2654435761u);
    std::atomic<size_t> chunks{0}; const size_t chunk = 4u << 20, nchunks = bytes / chunk; std::atomic<size_t> done{0}; std::atomic<bool> stop{false};
    std::vector<std::thread> th; double t0 = now(); uint32_t sink = 0; std::atomic<uint32_t> asink{0};
    for (int t = 0; t < T; ++t) th.emplace_back([&, t]() {
        cpu_set_t set; CPU_ZERO(&set); CPU_SET(7 - t, &set); sched_setaffinity(0, sizeof set, &set);
        uint32x4_t a = vdupq_n_u32(0);
        while (!stop.load(std::memory_order_relaxed)) {
            size_t c = chunks.fetch_add(1, std::memory_order_relaxed) % nchunks; const uint32_t * p = data + c * chunk / 4;
            for (size_t i = 0; i < chunk / 4; i += 16) { a = veorq_u32(a, vld1q_u32(p + i)); a = veorq_u32(a, vld1q_u32(p + i + 4)); a = veorq_u32(a, vld1q_u32(p + i + 8)); a = veorq_u32(a, vld1q_u32(p + i + 12)); }
            done.fetch_add(1, std::memory_order_relaxed);
        }
        asink.fetch_xor(vgetq_lane_u32(a, 0) ^ vgetq_lane_u32(a, 3));
    });
    while (now() - t0 < secs) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    stop = true; for (auto & x : th) x.join(); double el = now() - t0; sink = asink.load();
    std::printf("{\"cpu_stream_GBps\":%.2f,\"threads\":%d,\"seconds\":%.2f,\"sink\":%u}\n", done.load() * double(chunk) / el / 1e9, T, el, sink);
    return 0;
}
