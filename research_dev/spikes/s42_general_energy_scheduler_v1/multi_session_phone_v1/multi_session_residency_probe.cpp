#include "ggml-backend.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

namespace {

using steady_clock = std::chrono::steady_clock;

double now_ms() {
    return std::chrono::duration<double, std::milli>(
            steady_clock::now().time_since_epoch()).count();
}

bool parse_size(const char * text, size_t & value) {
    if (text == nullptr || *text == '\0') {
        return false;
    }
    char * end = nullptr;
    const unsigned long long parsed = strtoull(text, &end, 10);
    if (end == text || *end != '\0' || parsed == 0 ||
        parsed > std::numeric_limits<size_t>::max()) {
        return false;
    }
    value = static_cast<size_t>(parsed);
    return true;
}

size_t mem_available_kib() {
    std::ifstream input("/proc/meminfo");
    std::string key;
    size_t value = 0;
    std::string unit;
    while (input >> key >> value >> unit) {
        if (key == "MemAvailable:") {
            return value;
        }
    }
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s SESSION_MIB [SESSION_MIB ...]\n", argv[0]);
        return 2;
    }

    std::vector<size_t> requested_mib;
    for (int index = 1; index < argc; ++index) {
        size_t value = 0;
        if (!parse_size(argv[index], value)) {
            fprintf(stderr, "invalid session MiB: %s\n", argv[index]);
            return 2;
        }
        requested_mib.push_back(value);
    }

    size_t minimum_available_mib = 1536;
    if (const char * value = getenv("PROBE_MIN_AVAILABLE_MIB")) {
        if (!parse_size(value, minimum_available_mib)) {
            fprintf(stderr, "invalid PROBE_MIN_AVAILABLE_MIB\n");
            return 2;
        }
    }

    size_t hold_seconds = 0;
    if (const char * value = getenv("PROBE_HOLD_SECONDS")) {
        if (!parse_size(value, hold_seconds)) {
            fprintf(stderr, "invalid PROBE_HOLD_SECONDS\n");
            return 2;
        }
    }

    ggml_backend_load_all();

    std::vector<ggml_backend_t> backends;
    std::vector<ggml_backend_buffer_t> buffers;
    backends.reserve(requested_mib.size());
    buffers.reserve(requested_mib.size());

    bool passed = true;
    for (size_t index = 0; index < requested_mib.size(); ++index) {
        const std::string name = "HTP" + std::to_string(index);
        ggml_backend_dev_t device = ggml_backend_dev_by_name(name.c_str());
        if (device == nullptr) {
            fprintf(stderr, "missing backend: %s\n", name.c_str());
            passed = false;
            break;
        }

        ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
        if (backend == nullptr) {
            fprintf(stderr, "backend initialization failed: %s\n", name.c_str());
            passed = false;
            break;
        }
        backends.push_back(backend);

        const size_t available_before_kib = mem_available_kib();
        const size_t request_kib = requested_mib[index] * 1024;
        if (available_before_kib <= request_kib + minimum_available_mib * 1024) {
            fprintf(stderr,
                    "memory safety floor: %s request=%zu MiB available=%zu KiB\n",
                    name.c_str(), requested_mib[index], available_before_kib);
            passed = false;
            break;
        }

        const size_t bytes = requested_mib[index] * 1024 * 1024;
        const double allocate_start = now_ms();
        ggml_backend_buffer_t buffer = ggml_backend_alloc_buffer(backend, bytes);
        const double allocate_ms = now_ms() - allocate_start;
        if (buffer == nullptr) {
            fprintf(stderr, "allocation failed: %s %zu MiB\n",
                    name.c_str(), requested_mib[index]);
            passed = false;
            break;
        }
        buffers.push_back(buffer);

        const double touch_start = now_ms();
        ggml_backend_buffer_clear(buffer, static_cast<uint8_t>(index + 1));
        ggml_backend_synchronize(backend);
        const double touch_ms = now_ms() - touch_start;

        printf(
                "{\"event\":\"resident\",\"backend\":\"%s\","
                "\"requested_mib\":%zu,\"allocated_bytes\":%zu,"
                "\"allocate_ms\":%.3f,\"touch_ms\":%.3f,"
                "\"mem_available_kib\":%zu}\n",
                name.c_str(), requested_mib[index],
                ggml_backend_buffer_get_size(buffer), allocate_ms, touch_ms,
                mem_available_kib());
        fflush(stdout);
    }

    if (passed && hold_seconds > 0) {
        printf("{\"event\":\"hold\",\"seconds\":%zu}\n", hold_seconds);
        fflush(stdout);
        std::this_thread::sleep_for(std::chrono::seconds(hold_seconds));
    }

    for (auto iterator = buffers.rbegin(); iterator != buffers.rend(); ++iterator) {
        ggml_backend_buffer_free(*iterator);
    }
    for (auto iterator = backends.rbegin(); iterator != backends.rend(); ++iterator) {
        ggml_backend_free(*iterator);
    }

    printf("{\"event\":\"complete\",\"status\":\"%s\","
           "\"resident_sessions\":%zu,\"mem_available_kib\":%zu}\n",
           passed ? "PASS" : "FAIL", buffers.size(), mem_available_kib());
    return passed ? 0 : 1;
}
