// Local (no socket) Google Tensor TPU latency probe for one compiled FFN layer.
// Derived from the M3 litert_ffn_probe (metrics variant): NPU-only accelerator, every graph op must be
// a dispatch (custom) op, persistent managed AHWB tensor buffers, synchronous LiteRtRunCompiledModel.
// Adds: file I/O for large batches, separate load/first-invoke timing, memory snapshots, timed loop
// without metrics followed by an instrumented loop (vendor hardware_execution_time).
//
// usage: q32_tpu_probe MODEL DISPATCH_DIR INPUTS.f32 CASES OUT_PREFIX WARMUP TIMED METRIC_CALLS HOLD_S
//   INPUTS.f32 holds CASES consecutive input tensors (FP32, model input geometry).
//   Writes OUT_PREFIX.case<c>.f32 = output of the last timed call for case c.
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_metrics.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include <dirent.h>
#include <unistd.h>
#include <chrono>
#include <ctime>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static void check(LiteRtStatus status, const char * call) {
    if (status != kLiteRtStatusOk) {
        std::fprintf(stderr, "%s: status=%d\n", call, status);
        throw std::runtime_error(call);
    }
}
#define CHECK(call) check((call), #call)

static double boottime_ms() {
    timespec ts;
    clock_gettime(CLOCK_BOOTTIME, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

static std::string grep_lines(const char * path, const std::vector<std::string> & keys) {
    std::ifstream in(path);
    std::string line, out;
    while (std::getline(in, line)) {
        for (const auto & k : keys) {
            if (line.compare(0, k.size(), k) == 0) {
                std::string v = line.substr(k.size());
                size_t a = v.find_first_not_of(" \t:");
                out += k + "=" + (a == std::string::npos ? v : v.substr(a)) + " ";
            }
        }
    }
    return out;
}

// Sum sizes of dma-buf fds held by this process (readlink /proc/self/fd/* -> /dmabuf:..., fdinfo size:).
static std::string dmabuf_summary() {
    DIR * dir = opendir("/proc/self/fd");
    if (!dir) { return "dmabuf=unreadable"; }
    size_t count = 0, total = 0;
    std::string names;
    while (dirent * e = readdir(dir)) {
        if (e->d_name[0] == '.') { continue; }
        char link[512] = {};
        std::string p = std::string("/proc/self/fd/") + e->d_name;
        ssize_t n = readlink(p.c_str(), link, sizeof(link) - 1);
        if (n <= 0 || std::strstr(link, "dmabuf") == nullptr) { continue; }
        std::ifstream info(std::string("/proc/self/fdinfo/") + e->d_name);
        std::string line;
        size_t size = 0;
        std::string exp;
        while (std::getline(info, line)) {
            if (line.rfind("size:", 0) == 0) { size = std::strtoull(line.c_str() + 5, nullptr, 10); }
            if (line.rfind("exp_name:", 0) == 0) { exp = line.substr(9); }
        }
        ++count;
        total += size;
        if (size >= (1u << 20)) { names += "[" + std::to_string(size >> 20) + "MiB" + exp + "]"; }
    }
    closedir(dir);
    return "dmabuf_fds=" + std::to_string(count) + " dmabuf_bytes=" + std::to_string(total) + " big=" + names;
}

// Optional ODPM energy counters (needs root): Q32_ODPM=/sys/bus/iio/devices/iio:device0/energy_value
static void odpm(const char * tag) {
    const char * path = std::getenv("Q32_ODPM");
    if (!path) { return; }
    std::ifstream in(path);
    std::stringstream ss;
    ss << in.rdbuf();
    std::printf("odpm_begin tag=%s mono_ns=%llu\n%sodpm_end\n", tag, (unsigned long long) now_ns(), ss.str().c_str());
}

static long mem_available_kb() {
    std::ifstream in("/proc/meminfo");
    std::string key;
    long value = -1;
    std::string unit;
    while (in >> key >> value >> unit) {
        if (key == "MemAvailable:") { return value; }
    }
    return -1;
}

static void snapshot(const char * tag) {
    std::printf("mem tag=%s %s%s%s\n", tag,
                grep_lines("/proc/self/status", {"VmRSS", "VmHWM", "RssAnon", "RssFile", "VmSize"}).c_str(),
                grep_lines("/proc/meminfo", {"MemAvailable", "MemFree", "Cached"}).c_str(),
                dmabuf_summary().c_str());
}

struct runtime {
    LiteRtEnvironment env = nullptr;
    LiteRtModel model = nullptr;
    LiteRtOptions options = nullptr;
    LiteRtCompiledModel compiled = nullptr;
    LiteRtTensorBuffer input = nullptr;
    LiteRtTensorBuffer output = nullptr;
    size_t in_bytes = 0, out_bytes = 0;

    ~runtime() {
        if (input) { LiteRtDestroyTensorBuffer(input); }
        if (output) { LiteRtDestroyTensorBuffer(output); }
        if (compiled) { LiteRtDestroyCompiledModel(compiled); }
        if (options) { LiteRtDestroyOptions(options); }
        if (model) { LiteRtDestroyModel(model); }
        if (env) { LiteRtDestroyEnvironment(env); }
    }

    void initialize(const char * path, const char * library_dir) {
        LiteRtEnvOption option = {};
        option.tag = kLiteRtEnvOptionTagDispatchLibraryDir;
        option.value.type = kLiteRtAnyTypeString;
        option.value.str_value = library_dir;
        uint64_t t0 = now_ns();
        CHECK(LiteRtCreateEnvironment(1, &option, &env));
        uint64_t t1 = now_ns();
        CHECK(LiteRtCreateModelFromFile(env, path, &model));
        uint64_t t2 = now_ns();
        snapshot("after_model_file");
        LiteRtSignature signature = nullptr;
        CHECK(LiteRtGetModelSignature(model, 0, &signature));
        LiteRtParamIndex input_count = 0, output_count = 0;
        CHECK(LiteRtGetNumSignatureInputs(signature, &input_count));
        CHECK(LiteRtGetNumSignatureOutputs(signature, &output_count));
        if (input_count != 1 || output_count != 1) { throw std::runtime_error("expected one input and one output"); }
        LiteRtSubgraph graph = nullptr;
        LiteRtParamIndex op_count = 0;
        CHECK(LiteRtGetSignatureSubgraph(signature, &graph));
        CHECK(LiteRtGetNumSubgraphOps(graph, &op_count));
        for (LiteRtParamIndex i = 0; i < op_count; ++i) {
            LiteRtOp op = nullptr;
            LiteRtOpCode code;
            CHECK(LiteRtGetSubgraphOp(graph, i, &op));
            CHECK(LiteRtGetOpCode(op, &code));
            std::printf("model_op index=%zu code=%d\n", static_cast<size_t>(i), code);
            if (code != kLiteRtOpCodeTflCustom) { throw std::runtime_error("model contains non-dispatch operations"); }
        }
        CHECK(LiteRtCreateOptions(&options));
        CHECK(LiteRtSetOptionsHardwareAccelerators(options, kLiteRtHwAcceleratorNpu));
        // Optional runtime performance mode (Google Tensor opaque options, TOML payload "performance_mode = N"):
        // 0 extreme power saver, 1 power saver, 2 balanced (default), 3 high performance, 4 sustained, 5 burst.
        if (const char * mode = std::getenv("Q32_PERF_MODE")) {
            std::string toml = std::string("performance_mode = ") + mode + "\n";
            char * payload = strdup(toml.c_str());
            LiteRtOpaqueOptions opaque = nullptr;
            CHECK(LiteRtCreateOpaqueOptions("google_tensor", payload, [](void * p) { std::free(p); }, &opaque));
            CHECK(LiteRtAddOpaqueOptions(options, opaque));
            std::printf("runtime_option google_tensor performance_mode=%s\n", mode);
        }
        uint64_t t3 = now_ns();
        CHECK(LiteRtCreateCompiledModel(env, model, options, &compiled));
        uint64_t t4 = now_ns();
        for (int i = 0; i < 2; ++i) {
            LiteRtTensor tensor = nullptr;
            LiteRtRankedTensorType type;
            LiteRtTensorBufferRequirements requirements = nullptr;
            if (i == 0) {
                CHECK(LiteRtGetSignatureInputTensorByIndex(signature, 0, &tensor));
                CHECK(LiteRtGetCompiledModelInputBufferRequirements(compiled, 0, 0, &requirements));
            } else {
                CHECK(LiteRtGetSignatureOutputTensorByIndex(signature, 0, &tensor));
                CHECK(LiteRtGetCompiledModelOutputBufferRequirements(compiled, 0, 0, &requirements));
            }
            CHECK(LiteRtGetRankedTensorType(tensor, &type));
            if (type.element_type != kLiteRtElementTypeFloat32) { throw std::runtime_error("expected float32 I/O"); }
            LiteRtTensorBuffer * buffer = i == 0 ? &input : &output;
            CHECK(LiteRtCreateManagedTensorBufferFromRequirements(env, &type, requirements, buffer));
            size_t size = 0;
            CHECK(LiteRtGetTensorBufferPackedSize(*buffer, &size));
            (i == 0 ? in_bytes : out_bytes) = size;
            LiteRtTensorBufferType buffer_type;
            CHECK(LiteRtGetTensorBufferType(*buffer, &buffer_type));
            std::printf("tensor io=%s bytes=%zu buffer_type=%d shape=", i == 0 ? "in" : "out", size, buffer_type);
            for (unsigned j = 0; j < type.layout.rank; ++j) { std::printf("%d,", type.layout.dimensions[j]); }
            std::printf("\n");
        }
        uint64_t t5 = now_ns();
        std::printf("load env_ns=%llu model_file_ns=%llu compiled_model_ns=%llu buffers_ns=%llu\n",
                    (unsigned long long) (t1 - t0), (unsigned long long) (t2 - t1),
                    (unsigned long long) (t4 - t3), (unsigned long long) (t5 - t4));
    }

    void run(const float * values, float * result, uint64_t & invoke_ns, uint64_t & worker_ns) {
        const uint64_t start = now_ns();
        void * address = nullptr;
        CHECK(LiteRtLockTensorBuffer(input, &address, kLiteRtTensorBufferLockModeWrite));
        std::memcpy(address, values, in_bytes);
        CHECK(LiteRtUnlockTensorBuffer(input));
        const uint64_t before_invoke = now_ns();
        CHECK(LiteRtRunCompiledModel(compiled, 0, 1, &input, 1, &output));
        invoke_ns = now_ns() - before_invoke;
        CHECK(LiteRtLockTensorBuffer(output, &address, kLiteRtTensorBufferLockModeRead));
        std::memcpy(result, address, out_bytes);
        CHECK(LiteRtUnlockTensorBuffer(output));
        worker_ns = now_ns() - start;
    }
};

int main(int argc, char ** argv) {
    if (argc != 10) {
        std::fprintf(stderr, "usage: %s MODEL DISPATCH_DIR INPUTS.f32 CASES OUT_PREFIX WARMUP TIMED METRIC_CALLS HOLD_S\n", argv[0]);
        return 2;
    }
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int cases = std::atoi(argv[4]), warmup = std::atoi(argv[6]), timed = std::atoi(argv[7]);
    const int metric_calls = std::atoi(argv[8]), hold_s = std::atoi(argv[9]);
    if (cases < 1 || warmup < 0 || timed < 1 || metric_calls < 0 || hold_s < 0) { return 2; }
    try {
        std::printf("pid=%d\n", static_cast<int>(getpid()));
        snapshot("start");
        runtime r;
        const uint64_t load_start = now_ns();
        r.initialize(argv[1], argv[2]);
        std::printf("load_ns=%llu accelerator=npu_only\n", (unsigned long long) (now_ns() - load_start));
        snapshot("after_compiled_model");
        std::vector<float> inputs(cases * r.in_bytes / sizeof(float));
        {
            std::ifstream in(argv[3], std::ios::binary);
            in.read(reinterpret_cast<char *>(inputs.data()), inputs.size() * sizeof(float));
            if (static_cast<size_t>(in.gcount()) != inputs.size() * sizeof(float)) { throw std::runtime_error("short input file"); }
        }
        const size_t in_floats = r.in_bytes / sizeof(float), out_floats = r.out_bytes / sizeof(float);
        std::vector<float> output(out_floats);
        std::vector<std::vector<float>> first(cases), last(cases);
        int call = 0;
        uint64_t invoke_ns = 0, worker_ns = 0;
        auto one = [&](const char * phase) {
            const int c = call % cases;
            r.run(inputs.data() + c * in_floats, output.data(), invoke_ns, worker_ns);
            bool finite = true;
            for (float v : output) { if (!std::isfinite(v)) { finite = false; break; } }
            if (first[c].empty()) { first[c] = output; }
            const bool repeat = std::memcmp(first[c].data(), output.data(), r.out_bytes) == 0;
            last[c] = output;
            std::printf("call=%d phase=%s case=%d invoke_ns=%llu worker_ns=%llu finite=%d repeat_exact=%d\n", call, phase, c,
                        (unsigned long long) invoke_ns, (unsigned long long) worker_ns, finite, repeat);
            ++call;
        };
        one("first");
        snapshot("after_first_invoke");
        for (int i = 1; i < warmup; ++i) { one("warmup"); }
        odpm("timed_begin");
        const double boot_begin = boottime_ms();
        const uint64_t timed_start = now_ns();
        for (int i = 0; i < timed; ++i) { one("timed"); }
        const uint64_t timed_end = now_ns();
        const double boot_end = boottime_ms();
        odpm("timed_end");
        std::printf("timed_boottime_ms begin=%.3f end=%.3f\n", boot_begin, boot_end);
        std::printf("timed_window_ns=%llu calls=%d\n", (unsigned long long) (timed_end - timed_start), timed);
        if (metric_calls > 0) {
            const LiteRtStatus status = LiteRtCompiledModelStartMetricsCollection(r.compiled, 1);
            std::printf("metrics_start status=%d detail=1\n", status);
            for (int i = 0; i < metric_calls; ++i) { one("metrics"); }
            if (status == kLiteRtStatusOk) {
                LiteRtMetrics metrics = nullptr;
                CHECK(LiteRtCreateMetrics(&metrics));
                const LiteRtStatus stop = LiteRtCompiledModelStopMetricsCollection(r.compiled, metrics);
                std::printf("metrics_stop status=%d\n", stop);
                if (stop == kLiteRtStatusOk) {
                    int size = 0;
                    CHECK(LiteRtGetNumMetrics(metrics, &size));
                    for (int i = 0; i < size; ++i) {
                        LiteRtMetric value;
                        CHECK(LiteRtGetMetric(metrics, i, &value));
                        std::printf("metric name=%s type=%d value=", value.name, value.value.type);
                        switch (value.value.type) {
                            case kLiteRtAnyTypeInt: std::printf("%lld", static_cast<long long>(value.value.int_value)); break;
                            case kLiteRtAnyTypeReal: std::printf("%.12g", value.value.real_value); break;
                            case kLiteRtAnyTypeBool: std::printf("%d", value.value.bool_value); break;
                            case kLiteRtAnyTypeString: std::printf("%s", value.value.str_value); break;
                            default: std::printf("unsupported"); break;
                        }
                        std::printf("\n");
                    }
                }
                LiteRtDestroyMetrics(metrics);
            }
        }
        snapshot("after_loop");
        for (int c = 0; c < cases; ++c) {
            std::string path = std::string(argv[5]) + ".case" + std::to_string(c) + ".f32";
            std::ofstream out(path, std::ios::binary);
            out.write(reinterpret_cast<const char *>(last[c].data()), r.out_bytes);
        }
        // Optional residency test: Q32_NCOPIES=N loads N-1 more independent copies of the same compiled layer
        // (own environment, dispatch context, weight dma-buf and I/O buffers), stops early if MemAvailable
        // would fall below Q32_MIN_AVAIL_MB (default 3000), then invokes all copies round-robin.
        if (const char * nc = std::getenv("Q32_NCOPIES")) {
            const int n = std::atoi(nc);
            const char * mm = std::getenv("Q32_MIN_AVAIL_MB");
            const long min_avail_kb = (mm ? std::atol(mm) : 3000) * 1024;
            std::vector<std::unique_ptr<runtime>> extra;
            for (int i = 1; i < n; ++i) {
                const long avail = mem_available_kb();
                if (avail - static_cast<long>(r.out_bytes / 1024) - 900 * 1024 < min_avail_kb) {
                    std::printf("residency stop_before_copy=%d mem_available_kb=%ld\n", i, avail);
                    break;
                }
                extra.emplace_back(new runtime());
                const uint64_t t0 = now_ns();
                extra.back()->initialize(argv[1], argv[2]);
                const uint64_t t1 = now_ns();
                uint64_t inv = 0, wk = 0;
                extra.back()->run(inputs.data(), output.data(), inv, wk);
                const bool same = std::memcmp(first[0].data(), output.data(), r.out_bytes) == 0;
                std::printf("residency copy=%d load_ms=%.1f first_invoke_ms=%.1f same_output=%d\n", i, (t1 - t0) / 1e6, inv / 1e6, same);
                snapshot((std::string("resident_") + std::to_string(i + 1)).c_str());
            }
            std::vector<runtime *> all = {&r};
            for (auto & e : extra) { all.push_back(e.get()); }
            for (int round = 0; round < 3; ++round) {
                for (size_t i = 0; i < all.size(); ++i) {
                    uint64_t inv = 0, wk = 0;
                    all[i]->run(inputs.data(), output.data(), inv, wk);
                    const bool same = std::memcmp(first[0].data(), output.data(), r.out_bytes) == 0;
                    std::printf("residency round=%d copy=%zu invoke_ns=%llu same_output=%d\n", round, i, (unsigned long long) inv, same);
                }
            }
            std::printf("residency resident_copies=%zu\n", all.size());
            snapshot("resident_all");
        }
        if (hold_s > 0) {
            std::printf("hold_begin s=%d\n", hold_s);
            std::this_thread::sleep_for(std::chrono::seconds(hold_s));
            std::printf("hold_end\n");
        }
        std::printf("PASS calls=%d normal_exit=1\n", call);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
