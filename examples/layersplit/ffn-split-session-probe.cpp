#include "ggml-backend.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

using steady_clock = std::chrono::steady_clock;

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

double now_ms() {
    return std::chrono::duration<double, std::milli>(
            steady_clock::now().time_since_epoch()).count();
}

} // namespace

int main(int argc, char ** argv) {
    size_t session_mib = 0;
    size_t minimum_available_mib = 1536;
    size_t maximum_sessions = 16;
    for (int index = 1; index < argc; ++index) {
        if (index + 1 >= argc) {
            return 2;
        }
        if (strcmp(argv[index], "--session-mib") == 0) {
            if (!parse_size(argv[++index], session_mib)) {
                return 2;
            }
        } else if (strcmp(argv[index], "--minimum-available-mib") == 0) {
            if (!parse_size(argv[++index], minimum_available_mib)) {
                return 2;
            }
        } else if (strcmp(argv[index], "--maximum-sessions") == 0) {
            if (!parse_size(argv[++index], maximum_sessions) ||
                maximum_sessions > 64) {
                return 2;
            }
        } else {
            return 2;
        }
    }
    if (session_mib == 0) {
        fprintf(stderr,
                "usage: %s --session-mib MIB "
                "[--minimum-available-mib MIB] "
                "[--maximum-sessions N]\n",
                argv[0]);
        return 2;
    }

    ggml_backend_load_all();
    std::vector<ggml_backend_t> backends;
    std::vector<ggml_backend_buffer_t> buffers;
    const size_t requested_bytes = session_mib * 1024 * 1024;
    for (size_t index = 0; index < maximum_sessions; ++index) {
        const std::string name = "HTP" + std::to_string(index);
        ggml_backend_dev_t device = ggml_backend_dev_by_name(name.c_str());
        if (device == nullptr) {
            printf(
                    "{\"event\":\"unavailable\",\"backend\":\"%s\","
                    "\"session_index\":%zu,\"requested_bytes\":%zu,"
                    "\"reason\":\"device_absent\"}\n",
                    name.c_str(), index, requested_bytes);
            fflush(stdout);
            break;
        }
        const size_t available_before_kib = mem_available_kib();
        if (available_before_kib <= session_mib * 1024 +
                minimum_available_mib * 1024) {
            printf(
                    "{\"event\":\"unavailable\",\"backend\":\"%s\","
                    "\"session_index\":%zu,\"requested_bytes\":%zu,"
                    "\"reason\":\"memory_safety_floor\","
                    "\"mem_available_kib\":%zu}\n",
                    name.c_str(), index, requested_bytes,
                    available_before_kib);
            fflush(stdout);
            break;
        }
        ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
        if (backend == nullptr) {
            printf(
                    "{\"event\":\"unavailable\",\"backend\":\"%s\","
                    "\"session_index\":%zu,\"requested_bytes\":%zu,"
                    "\"reason\":\"backend_init_failed\"}\n",
                    name.c_str(), index, requested_bytes);
            fflush(stdout);
            break;
        }
        const double allocate_start = now_ms();
        ggml_backend_buffer_t buffer = ggml_backend_alloc_buffer(
                backend, requested_bytes);
        const double allocate_ms = now_ms() - allocate_start;
        if (buffer == nullptr) {
            printf(
                    "{\"event\":\"unavailable\",\"backend\":\"%s\","
                    "\"session_index\":%zu,\"requested_bytes\":%zu,"
                    "\"reason\":\"allocation_failed\"}\n",
                    name.c_str(), index, requested_bytes);
            fflush(stdout);
            ggml_backend_free(backend);
            break;
        }
        const double touch_start = now_ms();
        ggml_backend_buffer_clear(buffer, static_cast<uint8_t>(index + 1));
        ggml_backend_synchronize(backend);
        const double touch_ms = now_ms() - touch_start;
        backends.push_back(backend);
        buffers.push_back(buffer);
        printf(
                "{\"event\":\"resident\",\"backend\":\"%s\","
                "\"session_index\":%zu,\"requested_mib\":%zu,"
                "\"allocated_bytes\":%zu,\"allocate_ms\":%.3f,"
                "\"touch_ms\":%.3f,\"mem_available_kib\":%zu}\n",
                name.c_str(), index, session_mib,
                ggml_backend_buffer_get_size(buffer), allocate_ms, touch_ms,
                mem_available_kib());
        fflush(stdout);
    }

    const size_t resident_sessions = buffers.size();
    for (auto iterator = buffers.rbegin(); iterator != buffers.rend(); ++iterator) {
        ggml_backend_buffer_free(*iterator);
    }
    for (auto iterator = backends.rbegin(); iterator != backends.rend(); ++iterator) {
        ggml_backend_free(*iterator);
    }
    printf(
            "{\"event\":\"complete\",\"status\":\"%s\","
            "\"resident_sessions\":%zu,\"mem_available_kib\":%zu}\n",
            resident_sessions > 0 ? "PASS" : "FAIL",
            resident_sessions, mem_available_kib());
    return resident_sessions > 0 ? 0 : 1;
}
