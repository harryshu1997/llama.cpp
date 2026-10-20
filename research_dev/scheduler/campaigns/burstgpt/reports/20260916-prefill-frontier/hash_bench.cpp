#include "ffn-split-protocol.h"

#include <chrono>
#include <cstdio>
#include <vector>

__attribute__((noinline)) uint32_t reference(const void * data, size_t size) {
    return ffn_split::hash_bytes(data, size);
}

__attribute__((noinline)) uint32_t unrolled(const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    uint32_t hash = 2166136261U;
    while (size >= 8) {
        hash = (hash ^ bytes[0]) * 16777619U;
        hash = (hash ^ bytes[1]) * 16777619U;
        hash = (hash ^ bytes[2]) * 16777619U;
        hash = (hash ^ bytes[3]) * 16777619U;
        hash = (hash ^ bytes[4]) * 16777619U;
        hash = (hash ^ bytes[5]) * 16777619U;
        hash = (hash ^ bytes[6]) * 16777619U;
        hash = (hash ^ bytes[7]) * 16777619U;
        bytes += 8;
        size -= 8;
    }
    while (size-- > 0) {
        hash = (hash ^ *bytes++) * 16777619U;
    }
    return hash;
}

int main() {
    std::vector<uint8_t> data(3932160 + 16);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>((i * 31) ^ (i >> 8));
    }
    for (size_t offset = 0; offset < 8; ++offset) {
        for (size_t size = 0; size < 1024; ++size) {
            if (reference(data.data() + offset, size) != unrolled(data.data() + offset, size)) {
                return 1;
            }
        }
    }
    volatile uint32_t sink = 0;
    for (int round = 0; round < 6; ++round) {
        for (int arm = 0; arm < 2; ++arm) {
            const bool optimized = (arm + round) % 2 == 1;
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < 16; ++i) {
                data[0] = static_cast<uint8_t>(i);
                sink = optimized ? unrolled(data.data(), 3932160) : reference(data.data(), 3932160);
            }
            const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - start).count();
            printf("{\"round\":%d,\"unrolled\":%s,\"per_hash_us\":%.3f,\"hash\":%u}\n",
                    round, optimized ? "true" : "false", us / 16.0, sink);
        }
    }
}
