#include "phone_pim_client.h"
#include "phone_pim_ffn.h"
#include "phone_pim_json.h"
#include "phone_pim_protocol.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace pp = phone_pim;

namespace {

constexpr double k_rel_l2_limit = 5e-3;
constexpr uint64_t k_max_oracle_job_bytes = 512ULL * 1024 * 1024;

struct DeviceConfig {
    std::string id;
    std::string host;
    uint16_t port = 0;
    std::string physical_id;
};

struct Config {
    std::vector<DeviceConfig> devices;
    std::string model_path;
    std::string prefix = "blk.2";
    uint32_t token_count = 16;
    uint32_t jobs = 8;
    uint64_t route_epoch = 1;
    uint64_t generation_hint = 1;
    uint64_t island_id = 1;
    int timeout_ms = 600000;
    pp::ModelSource model_source = pp::ModelSource::prestaged;
    bool release = false;
};

struct Job {
    std::vector<float> input;
    std::vector<float> expected;
};

struct JobResult {
    std::string device_id;
    uint64_t e2e_us = 0;
    uint64_t compute_us = 0;
    double rel_l2 = 1.0;
    bool complete = false;
};

struct DeviceRuntime {
    DeviceConfig config;
    std::unique_ptr<pp::ClientSession> client;
    pp::HelloInfo hello;
    pp::StatusInfo initial_status;
    pp::PreparedInfo prepared;
    std::vector<uint64_t> e2e_us;
    std::vector<uint64_t> compute_us;
    uint64_t connect_status_us = 0;
    uint64_t prepare_e2e_us = 0;
    uint32_t completed = 0;
    double max_rel_l2 = 0.0;
    uint64_t final_generation = 0;
    bool prepared_ok = false;
    bool prepare_attempted = false;
    bool created_residency = false;
    bool quarantine = false;
    std::string error;
};

class StartGate {
public:
    explicit StartGate(size_t participants) : participants_(participants) {}

    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (++arrived_ == participants_) {
            open_ = true;
            cv_.notify_all();
            return;
        }
        cv_.wait(lock, [&] { return open_; });
    }

private:
    const size_t participants_;
    size_t arrived_ = 0;
    bool open_ = false;
    std::mutex mutex_;
    std::condition_variable cv_;
};

uint64_t now_us() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

bool parse_u64(const char * text, uint64_t & value) {
    if (text == nullptr || *text == '\0' || *text == '-') {
        return false;
    }
    char * end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        return false;
    }
    value = static_cast<uint64_t>(parsed);
    return true;
}

bool safe_id(const std::string & value) {
    if (value.empty() || value.size() > 64) {
        return false;
    }
    for (char ch : value) {
        if (!((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
              (ch >= '0' && ch <= '9') || ch == '_' || ch == '-')) {
            return false;
        }
    }
    return true;
}

bool canonical_prefix(const std::string & value) {
    if (value.size() < 5 || value.compare(0, 4, "blk.") != 0 || value[4] == '0') {
        return false;
    }
    for (size_t i = 4; i < value.size(); ++i) {
        if (value[i] < '0' || value[i] > '9') return false;
    }
    return true;
}

bool parse_device(const char * text, DeviceConfig & device) {
    if (text == nullptr) {
        return false;
    }
    const std::string value(text);
    const size_t first = value.find(',');
    const size_t second = first == std::string::npos ? std::string::npos : value.find(',', first + 1);
    const size_t third = second == std::string::npos ? std::string::npos : value.find(',', second + 1);
    if (first == std::string::npos || second == std::string::npos || third == std::string::npos ||
        value.find(',', third + 1) != std::string::npos) {
        return false;
    }
    device.id = value.substr(0, first);
    device.host = value.substr(first + 1, second - first - 1);
    device.physical_id = value.substr(third + 1);
    uint64_t port = 0;
    if (!safe_id(device.id) || device.host.empty() || device.host.size() > 255 ||
        !safe_id(device.physical_id) ||
        !parse_u64(value.substr(second + 1, third - second - 1).c_str(), port) ||
        port == 0 || port > 65535) {
        return false;
    }
    device.port = static_cast<uint16_t>(port);
    return true;
}

bool parse_args(int argc, char ** argv, Config & config, std::string & error) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : nullptr; };
        if (arg == "--device") {
            DeviceConfig device;
            if (!parse_device(next(), device)) return false;
            config.devices.push_back(std::move(device));
        } else if (arg == "--model" || arg == "-m") {
            const char * value = next();
            if (value == nullptr || *value == '\0') return false;
            config.model_path = value;
        } else if (arg == "--prefix") {
            const char * value = next();
            if (value == nullptr || *value == '\0') return false;
            config.prefix = value;
        } else if (arg == "--M") {
            uint64_t value = 0;
            if (!parse_u64(next(), value) || value == 0 || value > 1024) return false;
            config.token_count = static_cast<uint32_t>(value);
        } else if (arg == "--jobs") {
            uint64_t value = 0;
            if (!parse_u64(next(), value) || value == 0 || value > 256) return false;
            config.jobs = static_cast<uint32_t>(value);
        } else if (arg == "--route-epoch") {
            if (!parse_u64(next(), config.route_epoch) || config.route_epoch == 0) return false;
        } else if (arg == "--generation-hint") {
            if (!parse_u64(next(), config.generation_hint) || config.generation_hint == 0) return false;
        } else if (arg == "--island-id") {
            if (!parse_u64(next(), config.island_id) || config.island_id == 0) return false;
        } else if (arg == "--timeout-ms") {
            uint64_t value = 0;
            if (!parse_u64(next(), value) || value == 0 || value > std::numeric_limits<int>::max()) return false;
            config.timeout_ms = static_cast<int>(value);
        } else if (arg == "--model-source") {
            const char * value = next();
            if (value == nullptr) return false;
            if (std::strcmp(value, "prestaged") == 0) {
                config.model_source = pp::ModelSource::prestaged;
            } else if (std::strcmp(value, "published-store") == 0) {
                config.model_source = pp::ModelSource::published_store;
            } else {
                return false;
            }
        } else if (arg == "--release") {
            config.release = true;
        } else {
            return false;
        }
    }
    if (config.devices.empty() || config.devices.size() > 16 || config.model_path.empty() ||
        !canonical_prefix(config.prefix) ||
        config.jobs < config.devices.size()) {
        return false;
    }
    std::sort(config.devices.begin(), config.devices.end(), [](const DeviceConfig & a, const DeviceConfig & b) {
        return a.id < b.id;
    });
    for (size_t i = 0; i < config.devices.size(); ++i) {
        if (i != 0 && config.devices[i - 1].id == config.devices[i].id) {
            error = "duplicate logical device ID";
            return false;
        }
        for (size_t j = 0; j < i; ++j) {
            if (config.devices[i].host == config.devices[j].host &&
                config.devices[i].port == config.devices[j].port) {
                error = "duplicate device endpoint";
                return false;
            }
            if (config.devices[i].physical_id == config.devices[j].physical_id) {
                error = "duplicate physical device ID";
                return false;
            }
        }
    }
    return true;
}

double percentile_ms(std::vector<uint64_t> values, double percentile) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>(std::ceil(percentile * values.size())) - 1;
    return values[std::min(index, values.size() - 1)] / 1000.0;
}

std::string json_escape(const std::string & input) {
    std::string output;
    for (char ch : input) {
        switch (ch) {
            case '"': output += "\\\""; break;
            case '\\': output += "\\\\"; break;
            case '\n': output += "\\n"; break;
            case '\r': output += "\\r"; break;
            case '\t': output += "\\t"; break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20) {
                    char escaped[8] = {};
                    std::snprintf(escaped, sizeof(escaped), "\\u%04x", static_cast<unsigned char>(ch));
                    output += escaped;
                } else {
                    output += ch;
                }
                break;
        }
    }
    return output;
}

void close_sessions(std::vector<DeviceRuntime> & devices) {
    for (DeviceRuntime & device : devices) {
        if (device.client != nullptr) {
            std::string ignored;
            device.client->close(ignored);
        }
    }
}

bool release_residencies(
        std::vector<DeviceRuntime> & devices,
        const Config & config,
        uint64_t model_bytes,
        const std::array<uint8_t, 32> & model_sha256,
        bool rollback_new,
        bool release_all) {
    bool ok = true;
    for (DeviceRuntime & device : devices) {
        const bool possible_new_residency = rollback_new && device.prepare_attempted &&
                device.created_residency;
        const bool selected = possible_new_residency ||
                (device.prepared_ok && (release_all || device.quarantine));
        if (!selected) {
            if (device.client != nullptr && device.client->connected()) {
                device.final_generation = device.client->residency_generation();
            }
            continue;
        }
        std::string cleanup_error;
        if (device.client == nullptr || !device.client->connected()) {
            pp::ClientOptions options;
            options.host = device.config.host;
            options.port = device.config.port;
            options.route_epoch = config.route_epoch;
            options.residency_generation = device.initial_status.residency_generation != 0
                    ? device.initial_status.residency_generation : config.generation_hint;
            options.timeout_ms = config.timeout_ms;
            device.client = std::make_unique<pp::ClientSession>(std::move(options));
            pp::HelloInfo hello;
            if (!device.client->connect(hello, cleanup_error)) {
                std::fprintf(stderr, "error: device %s cleanup reconnect: %s\n",
                        device.config.id.c_str(), cleanup_error.c_str());
                ok = false;
                continue;
            }
        }
        pp::StatusInfo status;
        if (!device.client->status(status, cleanup_error)) {
            std::fprintf(stderr, "error: device %s cleanup STATUS: %s\n",
                    device.config.id.c_str(), cleanup_error.c_str());
            ok = false;
            continue;
        }
        device.final_generation = status.residency_generation;
        if (!status.ready) {
            continue;
        }
        pp::PrepareRequest request;
        request.island_id = config.island_id;
        request.token_count = config.token_count;
        request.model_source = config.model_source;
        request.prefix = config.prefix;
        request.model_bytes = model_bytes;
        request.model_sha256 = model_sha256;
        pp::PreparedInfo rebound;
        if (!device.client->prepare(request, rebound, cleanup_error)) {
            std::fprintf(stderr, "error: device %s cleanup identity rebind: %s\n",
                    device.config.id.c_str(), cleanup_error.c_str());
            ok = false;
            continue;
        }
        uint64_t next_generation = 0;
        if (!device.client->release(config.island_id, next_generation, cleanup_error)) {
            std::fprintf(stderr, "error: device %s cleanup RELEASE: %s\n",
                    device.config.id.c_str(), cleanup_error.c_str());
            ok = false;
            continue;
        }
        device.final_generation = next_generation;
        device.prepared_ok = false;
        device.prepare_attempted = false;
        device.created_residency = false;
    }
    return ok;
}

} // namespace

int main(int argc, char ** argv) {
    Config config;
    std::string parse_error;
    if (!parse_args(argc, argv, config, parse_error)) {
        if (!parse_error.empty()) {
            std::fprintf(stderr, "error: %s\n", parse_error.c_str());
        }
        std::fprintf(stderr,
                "usage: %s --device ID,HOST,PORT,PHYSICAL_ID [--device ...] -m shard.gguf "
                "[--prefix blk.2] [--M 16] [--jobs 8] [--route-epoch 1] "
                "[--generation-hint 1] [--island-id 1] "
                "[--model-source prestaged|published-store] [--release]\n",
                argv[0]);
        return 2;
    }

    std::string error;
    std::array<uint8_t, 32> model_sha256 = {};
    uint64_t model_bytes = 0;
    const uint64_t source_hash_start = now_us();
    if (!pp::sha256_file(config.model_path, model_sha256, model_bytes, error) || model_bytes == 0) {
        std::fprintf(stderr, "error: local model identity: %s\n", error.c_str());
        return 2;
    }
    const uint64_t source_hash_us = now_us() - source_hash_start;

    pp::FfnSpec oracle_spec;
    oracle_spec.model_path = config.model_path;
    oracle_spec.tensor_prefix = config.prefix;
    oracle_spec.backend_name = "CPU";
    oracle_spec.token_count = config.token_count;
    oracle_spec.expected_model_bytes = model_bytes;
    oracle_spec.expected_model_sha256 = model_sha256;
    pp::FfnIsland oracle;
    pp::FfnPrepareMetrics oracle_prepare;
    const uint64_t oracle_prepare_start = now_us();
    if (!oracle.prepare(oracle_spec, oracle_prepare, error)) {
        std::fprintf(stderr, "error: local FFN oracle: %s\n", error.c_str());
        return 3;
    }
    const uint64_t oracle_prepare_us = now_us() - oracle_prepare_start;

    const uint64_t oracle_elements = oracle.input_elements();
    if (oracle_elements == 0 || oracle_elements > std::numeric_limits<uint64_t>::max() / (2 * sizeof(float)) ||
        oracle_elements * (2 * sizeof(float)) > k_max_oracle_job_bytes / config.jobs) {
        std::fprintf(stderr, "error: requested oracle job set exceeds the 512 MiB memory bound\n");
        return 3;
    }
    std::vector<Job> jobs(config.jobs);
    const uint64_t oracle_jobs_start = now_us();
    try {
        for (uint32_t i = 0; i < config.jobs; ++i) {
            jobs[i].input = oracle.make_test_input(0x71756575U + i * 0x9e3779b9U);
            pp::FfnExecuteMetrics metrics;
            if (jobs[i].input.empty() || !oracle.execute(jobs[i].input, jobs[i].expected, metrics, error)) {
                std::fprintf(stderr, "error: local oracle job %u: %s\n", i, error.c_str());
                return 3;
            }
        }
    } catch (const std::bad_alloc &) {
        std::fprintf(stderr, "error: local oracle job allocation failed within the configured bound\n");
        return 3;
    }
    const uint64_t oracle_jobs_us = now_us() - oracle_jobs_start;

    std::vector<DeviceRuntime> devices(config.devices.size());
    std::vector<std::thread> setup_threads;
    const uint64_t fleet_setup_start = now_us();
    for (size_t i = 0; i < devices.size(); ++i) {
        devices[i].config = config.devices[i];
        setup_threads.emplace_back([&, i] {
            DeviceRuntime & device = devices[i];
            pp::ClientOptions options;
            options.host = device.config.host;
            options.port = device.config.port;
            options.route_epoch = config.route_epoch;
            options.residency_generation = config.generation_hint;
            options.timeout_ms = config.timeout_ms;
            device.client = std::make_unique<pp::ClientSession>(std::move(options));
            const uint64_t connect_start = now_us();
            if (!device.client->connect(device.hello, device.error) ||
                !device.client->status(device.initial_status, device.error)) {
                return;
            }
            device.connect_status_us = now_us() - connect_start;
            pp::PrepareRequest request;
            request.island_id = config.island_id;
            request.token_count = config.token_count;
            request.model_source = config.model_source;
            request.prefix = config.prefix;
            request.model_bytes = model_bytes;
            request.model_sha256 = model_sha256;
            const uint64_t prepare_start = now_us();
            device.prepare_attempted = true;
            device.created_residency = !device.initial_status.ready;
            if (!device.client->prepare(request, device.prepared, device.error)) {
                return;
            }
            device.prepare_e2e_us = now_us() - prepare_start;
            device.prepared_ok = true;
            device.final_generation = device.client->residency_generation();
            if (device.prepared.n_embd != oracle.n_embd() || device.prepared.n_ff != oracle.n_ff() ||
                device.prepared.token_count != oracle.token_count()) {
                device.error = "remote PREPARE shape differs from local oracle";
                device.quarantine = true;
            }
        });
    }
    for (std::thread & thread : setup_threads) thread.join();
    const uint64_t fleet_setup_us = now_us() - fleet_setup_start;
    for (const DeviceRuntime & device : devices) {
        if (!device.error.empty()) {
            std::fprintf(stderr, "error: device %s setup: %s\n",
                    device.config.id.c_str(), device.error.c_str());
            release_residencies(devices, config, model_bytes, model_sha256, true, false);
            close_sessions(devices);
            return 4;
        }
    }

    std::vector<JobResult> results(config.jobs);
    std::atomic<uint32_t> next_job(static_cast<uint32_t>(devices.size()));
    std::atomic<bool> failed(false);
    std::atomic<uint64_t> last_rpc_completion_us(0);
    uint64_t dispatch_start = 0;
    StartGate gate(devices.size() + 1);
    std::vector<std::thread> workers;
    for (size_t i = 0; i < devices.size(); ++i) {
        workers.emplace_back([&, i] {
            DeviceRuntime & device = devices[i];
            gate.arrive_and_wait();
            uint32_t job_id = static_cast<uint32_t>(i);
            for (;;) {
                if (failed.load(std::memory_order_acquire)) return;
                pp::ExecutionInfo execution;
                if (!device.client->execute(
                            config.island_id, jobs[job_id].input, execution, device.error)) {
                    device.quarantine = true;
                    failed.store(true, std::memory_order_release);
                    return;
                }
                const uint64_t rpc_complete = now_us();
                uint64_t prior = last_rpc_completion_us.load(std::memory_order_relaxed);
                while (prior < rpc_complete &&
                       !last_rpc_completion_us.compare_exchange_weak(
                               prior, rpc_complete, std::memory_order_release, std::memory_order_relaxed)) {}
                bool finite = false;
                const double rel_l2 = pp::relative_l2(execution.output, jobs[job_id].expected, finite);
                if (!finite || rel_l2 > k_rel_l2_limit) {
                    device.error = "remote result failed the CPU oracle";
                    device.quarantine = true;
                    failed.store(true, std::memory_order_release);
                    return;
                }
                JobResult & result = results[job_id];
                result.device_id = device.config.id;
                result.e2e_us = execution.e2e_us;
                result.compute_us = execution.compute_us;
                result.rel_l2 = rel_l2;
                result.complete = true;
                device.e2e_us.push_back(execution.e2e_us);
                device.compute_us.push_back(execution.compute_us);
                device.max_rel_l2 = std::max(device.max_rel_l2, rel_l2);
                ++device.completed;
                job_id = next_job.fetch_add(1, std::memory_order_acq_rel);
                if (job_id >= config.jobs) return;
            }
        });
    }
    dispatch_start = now_us();
    gate.arrive_and_wait();
    for (std::thread & thread : workers) thread.join();
    const uint64_t dispatch_us = now_us() - dispatch_start;
    const uint64_t last_rpc_us = last_rpc_completion_us.load(std::memory_order_acquire) >= dispatch_start
            ? last_rpc_completion_us.load(std::memory_order_relaxed) - dispatch_start : 0;

    bool all_complete = !failed.load(std::memory_order_acquire);
    for (const JobResult & result : results) all_complete = all_complete && result.complete;
    if (!all_complete) {
        for (const DeviceRuntime & device : devices) {
            if (!device.error.empty()) {
                std::fprintf(stderr, "error: device %s dispatch: %s\n",
                        device.config.id.c_str(), device.error.c_str());
            }
        }
        release_residencies(devices, config, model_bytes, model_sha256, true, false);
        close_sessions(devices);
        return 5;
    }

    if (config.release) {
        if (!release_residencies(devices, config, model_bytes, model_sha256, false, true)) {
            close_sessions(devices);
            return 6;
        }
    }
    for (DeviceRuntime & device : devices) {
        if (device.final_generation == 0) {
            device.final_generation = device.client->residency_generation();
        }
        if (!device.client->close(device.error)) {
            std::fprintf(stderr, "error: device %s close: %s\n",
                    device.config.id.c_str(), device.error.c_str());
            release_residencies(devices, config, model_bytes, model_sha256, true, false);
            close_sessions(devices);
            return 6;
        }
    }

    double max_rel_l2 = 0.0;
    for (const JobResult & result : results) max_rel_l2 = std::max(max_rel_l2, result.rel_l2);
    const char * model_source = config.model_source == pp::ModelSource::prestaged
            ? "prestaged" : "published_store";
    const char * verdict = devices.size() >= 2 ? "REAL_FLEET_FFN_PASS" : "REAL_DEVICE_FFN_PASS";
    const std::string route_epoch_json = pp::json_uint64(config.route_epoch);
    const std::string island_id_json = pp::json_uint64(config.island_id);
    std::printf("{\"record_schema_version\":2,\"verdict\":\"%s\","
                "\"scheduler\":\"persistent_session_work_stealing_v1\","
                "\"protocol_version\":%u,\"route_epoch\":%s,\"island_id\":%s,"
                "\"model_source\":\"%s\",\"device_count\":%zu,\"jobs\":%u,"
                "\"M\":%u,\"prefix\":\"%s\",\"model_bytes\":%llu,"
                "\"model_sha256\":\"%s\",\"oracle_backend\":\"%s\","
                "\"rel_l2_limit\":%.6g,\"rel_l2_max\":%.12g,"
                "\"source_hash_ms\":%.3f,\"oracle_prepare_ms\":%.3f,"
                "\"oracle_jobs_ms\":%.3f,\"remote_setup_ms\":%.3f,"
                "\"last_rpc_complete_ms\":%.3f,\"validated_dispatch_makespan_ms\":%.3f,"
                "\"devices\":[",
            verdict, pp::k_version, route_epoch_json.c_str(), island_id_json.c_str(), model_source,
            devices.size(), config.jobs, config.token_count,
            json_escape(config.prefix).c_str(), static_cast<unsigned long long>(model_bytes),
            pp::hex_sha256(model_sha256).c_str(), "CPU",
            k_rel_l2_limit, max_rel_l2, source_hash_us / 1000.0,
            oracle_prepare_us / 1000.0, oracle_jobs_us / 1000.0,
            fleet_setup_us / 1000.0, last_rpc_us / 1000.0, dispatch_us / 1000.0);
    for (size_t i = 0; i < devices.size(); ++i) {
        const DeviceRuntime & device = devices[i];
        const std::string session_epoch_json = pp::json_uint64(device.hello.session_epoch);
        const std::string initial_generation_json =
                pp::json_uint64(device.initial_status.residency_generation);
        const std::string final_generation_json = pp::json_uint64(device.final_generation);
        if (i != 0) std::printf(",");
        std::printf("{\"id\":\"%s\",\"physical_id\":\"%s\","
                    "\"physical_id_source\":\"caller_asserted\","
                    "\"host\":\"%s\",\"port\":%u,\"session_epoch\":%s,"
                    "\"backend\":\"%s\",\"capability\":\"%s\","
                    "\"initial_generation\":%s,"
                    "\"initial_ready\":%s,\"connect_status_ms\":%.3f,"
                    "\"prepare_e2e_ms\":%.3f,\"jobs\":%u,"
                    "\"client_operation_p50_ms\":%.3f,"
                    "\"client_operation_p95_ms\":%.3f,\"worker_compute_p50_ms\":%.3f,"
                    "\"rel_l2_max\":%.12g,\"final_generation\":%s}",
                json_escape(device.config.id).c_str(), json_escape(device.config.physical_id).c_str(),
                json_escape(device.config.host).c_str(), static_cast<unsigned>(device.config.port),
                session_epoch_json.c_str(),
                json_escape(device.hello.backend).c_str(), json_escape(device.hello.capability).c_str(),
                initial_generation_json.c_str(),
                device.initial_status.ready ? "true" : "false",
                device.connect_status_us / 1000.0, device.prepare_e2e_us / 1000.0, device.completed,
                percentile_ms(device.e2e_us, 0.50), percentile_ms(device.e2e_us, 0.95),
                percentile_ms(device.compute_us, 0.50), device.max_rel_l2,
                final_generation_json.c_str());
    }
    std::printf("],\"assignments\":[");
    for (size_t i = 0; i < results.size(); ++i) {
        if (i != 0) std::printf(",");
        std::printf("{\"job\":%zu,\"device\":\"%s\",\"client_operation_ms\":%.3f,"
                    "\"worker_compute_ms\":%.3f,\"rel_l2\":%.12g}",
                i, json_escape(results[i].device_id).c_str(), results[i].e2e_us / 1000.0,
                results[i].compute_us / 1000.0, results[i].rel_l2);
    }
    std::printf("]}\n");
    return 0;
}
