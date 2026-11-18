// chase: random pointer chase over a buffer (latency-bound, ~0.3 GB/s) to raise the CPU memory-latency vote without consuming bandwidth.
#include <sched.h>
#include <sys/mman.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main(int argc, char ** argv) {
    size_t bytes = size_t(std::atoi(argv[1])) << 20; double secs = std::atof(argv[2]); int core = argc > 3 ? std::atoi(argv[3]) : 6;
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(core, &set); sched_setaffinity(0, sizeof set, &set);
    size_t n = bytes / 64; std::vector<uint32_t> perm(n); for (size_t i = 0; i < n; ++i) perm[i] = uint32_t(i);
    uint64_t s = 88172645463325252ull; for (size_t i = n - 1; i > 0; --i) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; size_t j = s % (i + 1); std::swap(perm[i], perm[j]); }
    auto * buf = (uint32_t *) mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    for (size_t i = 0; i < n; ++i) buf[size_t(perm[i]) * 16] = perm[(i + 1) % n];   // one pointer per 64 B line, random cycle
    double t0 = now(); uint64_t hops = 0; uint32_t p = 0;
    while (now() - t0 < secs) { for (int i = 0; i < 100000; ++i) p = buf[size_t(p) * 16]; hops += 100000; }
    double el = now() - t0;
    std::printf("{\"chase_ns_per_hop\":%.1f,\"hops\":%llu,\"bytes_MiB\":%zu,\"p\":%u}\n", el / hops * 1e9, (unsigned long long) hops, bytes >> 20, p);
    return 0;
}
