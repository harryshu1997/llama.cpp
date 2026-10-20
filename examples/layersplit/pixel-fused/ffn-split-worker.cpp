// WS15: the NEON packed kernels exist only on aarch64; a host (x86) build serves the same protocol with the
// portable pair dot (S42_PIXEL_CPU_PAIR_DOT / ROW_CHUNK / NEON layouts must stay unset there).
#if defined(__aarch64__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#include <arm_neon.h>
#endif
#include <atomic>
#include "ffn-split-protocol.h"
#include "ffn-split-dmabuf.h"
#include "ffn-split-worker-entry.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include <pthread.h>
#include <sched.h>
#include "ggml-impl.h"
#include "gguf.h"

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <string>
#include <strings.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
#include <vector>

#if defined(__ANDROID__)
#include "ffn-split-functionfs.h"

#include <linux/dma-heap.h>
#include <sys/mman.h>

#endif

namespace {

using steady_clock = std::chrono::steady_clock;

struct config {
    std::string model;
    std::string artifact_sha256;
    std::string backend = "CPU";
    std::string bind = "127.0.0.1";
    std::string ffs_root;
    std::string ready_file;
    int port = 0;
    uint64_t layer_mask = 0;
    int64_t columns = 0;
    int64_t column_quantum = 512;
    int64_t alternate_columns = 0;
    int64_t max_tokens = 1;
    int64_t queue_depth = 1;
    bool f16_io = false;
    bool staged_dmabuf = false;
    long max_requests = 0;
};

// Opt-in NPU+GPU split: the primary backend computes the leading columns of
// every weight block, the secondary backend the trailing ones, concurrently.
struct secondary_config {
    std::string backend;
    double fraction = 0.0;
    unsigned block_mask = 0;
    int64_t cpu_half_columns = 0;
    // the secondary owns the trailing columns of the served suffix as whole blocks
    int64_t trailing_columns = 0;
    int64_t align = 64;
    long log_period = 1;
    // requests with more tokens run NPU-only on a primary copy of the
    // secondary columns (0 = always split, no copy)
    int64_t max_tokens = 0;

    bool enabled() const {
        return !backend.empty() && (fraction > 0.0 || block_mask != 0 || trailing_columns > 0);
    }
};

bool parse_secondary_config(secondary_config & result, std::string & error) {
    const char * backend = getenv("S43_FFN_SECONDARY_BACKEND");
    const char * fraction = getenv("S43_FFN_SECONDARY_FRACTION");
    const char * align = getenv("S43_FFN_SECONDARY_ALIGN");
    const char * log_period = getenv("S43_FFN_DUAL_LOG_PERIOD");
    const char * max_tokens = getenv("S43_FFN_SECONDARY_MAX_TOKENS");
    if (backend == nullptr || *backend == '\0') {
        return true;
    }

    result.backend = backend;
    if (const char * mask = getenv("S42_PIXEL_GPU_BLOCK_MASK")) {
        char * end = nullptr;
        errno = 0;
        const unsigned long value = strtoul(mask, &end, 16);
        if (errno || end == mask || *end || value == 0 || value >= 15) {
            error = "S42_PIXEL_GPU_BLOCK_MASK must select some of four blocks";
            return false;
        }
        result.block_mask = static_cast<unsigned>(value);
    }
    if (const char * columns = getenv("S42_PIXEL_CPU_HALF_COLUMNS")) {
        char * end = nullptr;
        errno = 0;
        const long value = strtol(columns, &end, 10);
        if (errno || end == columns || *end || value < 64 || value > 8640 ||
                value % 64 != 0 || result.block_mask != 6) {
            error = "S42_PIXEL_CPU_HALF_COLUMNS requires mask6 and a multiple of64 in [64,8640]";
            return false;
        }
        result.cpu_half_columns = value;
    }
    if (const char * columns = getenv("S43_PIXEL_GPU_TRAILING_COLUMNS")) {
        char * end = nullptr;
        errno = 0;
        const long value = strtol(columns, &end, 10);
        if (errno || end == columns || *end || value < 256 || value % 256 != 0 ||
                result.block_mask != 0 || result.cpu_half_columns != 0) {
            error = "S43_PIXEL_GPU_TRAILING_COLUMNS must be a positive multiple of 256 without block masks";
            return false;
        }
        result.trailing_columns = value;
    }
    if (const char * mask = getenv("S42_PIXEL_GPU_HOST_MASK")) {
        char * end = nullptr;
        errno = 0;
        const unsigned long value = strtoul(mask, &end, 16);
        if (errno || end == mask || *end || value == 0 || value > 255) {
            error = "S42_PIXEL_GPU_HOST_MASK is invalid";
            return false;
        }
    }
    if (fraction != nullptr && *fraction != '\0') {
        char * end = nullptr;
        errno = 0;
        result.fraction = strtod(fraction, &end);
        if (errno != 0 || end == fraction || *end != '\0' ||
            !(result.fraction >= 0.0 && result.fraction <= 0.5)) {
            error = "S43_FFN_SECONDARY_FRACTION must be in [0, 0.5]";
            return false;
        }
    }
    if (align != nullptr && *align != '\0') {
        int64_t value = 0;
        char * end = nullptr;
        errno = 0;
        value = strtoll(align, &end, 10);
        if (errno != 0 || end == align || *end != '\0' || value <= 0 ||
            value > 65536) {
            error = "S43_FFN_SECONDARY_ALIGN is invalid";
            return false;
        }
        result.align = value;
    }
    if (log_period != nullptr && *log_period != '\0') {
        char * end = nullptr;
        errno = 0;
        const long value = strtol(log_period, &end, 10);
        if (errno != 0 || end == log_period || *end != '\0' || value < 0) {
            error = "S43_FFN_DUAL_LOG_PERIOD is invalid";
            return false;
        }
        result.log_period = value;
    }
    if (max_tokens != nullptr && *max_tokens != '\0') {
        char * end = nullptr;
        errno = 0;
        const long long value = strtoll(max_tokens, &end, 10);
        if (errno != 0 || end == max_tokens || *end != '\0' || value < 0) {
            error = "S43_FFN_SECONDARY_MAX_TOKENS is invalid";
            return false;
        }
        result.max_tokens = value;
    }
    return true;
}

// Optional per-process utilization floor so bursty calls do not run at idle
// clocks; threads created later inherit it.
bool apply_uclamp_min() {
    const char * value = getenv("S43_PIXEL_UCLAMP_MIN");
    if (value == nullptr || *value == '\0') {
        return true;
    }
    char * end = nullptr;
    errno = 0;
    const long util = strtol(value, &end, 10);
    if (errno || end == value || *end || util < 0 || util > 1024) {
        fprintf(stderr, "[ffn-worker] S43_PIXEL_UCLAMP_MIN must be in [0,1024]\n");
        return false;
    }
    struct {
        uint32_t size, sched_policy;
        uint64_t sched_flags;
        int32_t sched_nice;
        uint32_t sched_priority;
        uint64_t sched_runtime, sched_deadline, sched_period;
        uint32_t sched_util_min, sched_util_max;
    } attr = {};
    attr.size = sizeof(attr);
    // SCHED_FLAG_KEEP_POLICY | SCHED_FLAG_KEEP_PARAMS | SCHED_FLAG_UTIL_CLAMP_MIN
    attr.sched_flags = 0x08 | 0x10 | 0x20;
    attr.sched_util_min = static_cast<uint32_t>(util);
    if (syscall(SYS_sched_setattr, 0, &attr, 0) != 0) {
        fprintf(stderr, "[ffn-worker] sched_setattr uclamp failed: %s\n", strerror(errno));
        return false;
    }
    fprintf(stderr, "S43PIXELUCLAMP min=%ld\n", util);
    return true;
}

// Runs one job at a time on a persistent thread so the secondary backend
// overlaps the (blocking) primary graph compute.
class helper_thread {
public:
    helper_thread() : thread_([this] { loop(); }) {}

    ~helper_thread() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        thread_.join();
    }

    void start(std::function<void()> job) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            job_ = std::move(job);
            done_ = false;
        }
        cv_.notify_all();
    }

    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return done_; });
    }

private:

    void loop() {
        pthread_setname_np(pthread_self(), "pixel-gpu-host");
        if (const char * value = getenv("S42_PIXEL_GPU_HOST_MASK")) {
            const unsigned long mask = strtoul(value, nullptr, 16);
            cpu_set_t cores;
            CPU_ZERO(&cores);
            for (int i = 0; i < 8; ++i) if (mask & (1u << i)) CPU_SET(i, &cores);
            if (sched_setaffinity(0, sizeof(cores), &cores) != 0) {
                perror("private GPU host affinity");
                std::abort();
            }
        }
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            cv_.wait(lock, [this] { return stop_ || job_ != nullptr; });
            if (job_ == nullptr) {
                return;
            }
            std::function<void()> job = std::move(job_);
            job_ = nullptr;
            lock.unlock();
            job();
            lock.lock();
            done_ = true;
            cv_.notify_all();
        }
    }

    std::mutex mutex_;
    std::condition_variable cv_;
    std::function<void()> job_;
    bool done_ = true;
    bool stop_ = false;
    std::thread thread_;
};

struct host_tensor {
    std::vector<uint8_t> bytes;
    int64_t ne0 = 0;
    int64_t ne1 = 0;
    ggml_type type = GGML_TYPE_COUNT;
};

struct gguf_reader {
    int fd = -1;
    ggml_context * metadata = nullptr;
    gguf_context * gguf = nullptr;

    ~gguf_reader() {
        if (gguf != nullptr) {
            gguf_free(gguf);
        }
        if (metadata != nullptr) {
            ggml_free(metadata);
        }
        if (fd >= 0) {
            close(fd);
        }
    }
};

// Offline FFN shard GGUF (research_dev/scheduler/native/ffn_shard_gguf.py):
// holds only the ffn_gate/ffn_up/ffn_down suffix slices of selected layers.
struct ffn_shard_info {
    bool present = false;
    std::string parent_sha256;
    int64_t n_embd = 0;
    int64_t n_ff = 0;
    int64_t column_offset = 0;
    int64_t columns = 0;
    uint64_t layer_mask = 0;
    std::string weight_type;
};

// WS15 fused attention projections: s43.fused.* metadata of a fused shard
// (research_dev/scheduler/native/pixel_packed_shard.py --fused-layers).
struct fused_shard_info {
    bool present = false;
    uint64_t layer_mask = 0;
    uint64_t next_mask = 0;
    int64_t n_q = 0;
    int64_t n_kv = 0;
    float rms_eps = 0.0f;
};

uint64_t now_us() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
            steady_clock::now().time_since_epoch()).count());
}

uint64_t epoch_us() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
}

bool parse_i64(const char * text, int64_t & value) {
    if (text == nullptr || *text == '\0') {
        return false;
    }
    errno = 0;
    char * end = nullptr;
    const long long parsed = strtoll(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        return false;
    }
    value = parsed;
    return true;
}

void print_residency_phase(const config & cfg, const char * phase) {
    const char * session_id = getenv("S42_RESIDENCY_SESSION_ID");
    const char * generation_text = getenv(
            "S42_RESIDENCY_SESSION_GENERATION");
    int64_t session_generation = 0;
    if (session_id == nullptr || *session_id == '\0' ||
        generation_text == nullptr ||
        !parse_i64(generation_text, session_generation) ||
        session_generation < 1) {
        return;
    }
    for (const char * cursor = session_id; *cursor != '\0'; ++cursor) {
        const char value = *cursor;
        if (!((value >= 'A' && value <= 'Z') ||
              (value >= 'a' && value <= 'z') ||
              (value >= '0' && value <= '9') || value == '.' ||
              value == '_' || value == '-')) {
            return;
        }
    }
    fprintf(stderr,
            "RESIDENTPHASE {\"schema\":\"s42-phone-residency-phase-v1\"," 
            "\"component\":\"ffn-worker\"," 
            "\"phase\":\"%s\",\"session_id\":\"%s\"," 
            "\"artifact_sha256\":\"%s\",\"session_generation\":%lld," 
            "\"monotonic_us\":%llu,\"epoch_us\":%llu}\n",
            phase, session_id, cfg.artifact_sha256.c_str(),
            static_cast<long long>(session_generation),
            static_cast<unsigned long long>(now_us()),
            static_cast<unsigned long long>(epoch_us()));
    fflush(stderr);
}

bool parse_layer_spec(const char * text, uint64_t & mask) {
    if (text == nullptr || *text == '\0') {
        return false;
    }
    uint64_t parsed_mask = 0;
    std::string remaining(text);
    while (!remaining.empty()) {
        const size_t comma = remaining.find(',');
        const std::string item = remaining.substr(0, comma);
        if (item.empty()) {
            return false;
        }
        const size_t dash = item.find('-');
        int64_t first = 0;
        int64_t last = 0;
        if (dash == std::string::npos) {
            if (!parse_i64(item.c_str(), first)) {
                return false;
            }
            last = first;
        } else {
            if (item.find('-', dash + 1) != std::string::npos ||
                !parse_i64(item.substr(0, dash).c_str(), first) ||
                !parse_i64(item.substr(dash + 1).c_str(), last)) {
                return false;
            }
        }
        if (first < 0 || last < first || last >= 64) {
            return false;
        }
        for (int64_t layer = first; layer <= last; ++layer) {
            parsed_mask |= UINT64_C(1) << layer;
        }
        if (comma == std::string::npos) {
            break;
        }
        remaining.erase(0, comma + 1);
    }
    mask = parsed_mask;
    return mask != 0;
}

bool parse_config(int argc, char ** argv, config & result) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            result.model = argv[++i];
        } else if (strcmp(argv[i], "--artifact-sha256") == 0 && i + 1 < argc) {
            result.artifact_sha256 = argv[++i];
        } else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            result.backend = argv[++i];
        } else if (strcmp(argv[i], "--bind") == 0 && i + 1 < argc) {
            result.bind = argv[++i];
        } else if (strcmp(argv[i], "--ffs-root") == 0 && i + 1 < argc) {
            result.ffs_root = argv[++i];
        } else if (strcmp(argv[i], "--ready-file") == 0 && i + 1 < argc) {
            result.ready_file = argv[++i];
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            int64_t value = 0;
            if (!parse_i64(argv[++i], value) || value <= 0 || value > 65535) {
                return false;
            }
            result.port = static_cast<int>(value);
        } else if (strcmp(argv[i], "--layer") == 0 && i + 1 < argc) {
            int64_t value = 0;
            if (result.layer_mask != 0 || !parse_i64(argv[++i], value) ||
                value < 0 || value >= 64) {
                return false;
            }
            result.layer_mask = UINT64_C(1) << value;
        } else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
            if (result.layer_mask != 0 ||
                !parse_layer_spec(argv[++i], result.layer_mask)) {
                return false;
            }
        } else if (strcmp(argv[i], "--columns") == 0 && i + 1 < argc) {
            if (!parse_i64(argv[++i], result.columns) || result.columns <= 0) {
                return false;
            }
        } else if (strcmp(argv[i], "--column-quantum") == 0 && i + 1 < argc) {
            if (!parse_i64(argv[++i], result.column_quantum) ||
                result.column_quantum <= 0) {
                return false;
            }
        } else if (strcmp(argv[i], "--alternate-columns") == 0 && i + 1 < argc) {
            if (!parse_i64(argv[++i], result.alternate_columns) ||
                result.alternate_columns <= 0) {
                return false;
            }
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            if (!parse_i64(argv[++i], result.max_tokens) ||
                result.max_tokens <= 0 || result.max_tokens > UINT16_MAX) {
                return false;
            }
        } else if (strcmp(argv[i], "--queue-depth") == 0 && i + 1 < argc) {
            if (!parse_i64(argv[++i], result.queue_depth) ||
                result.queue_depth <= 0 || result.queue_depth > 8) {
                return false;
            }
        } else if (strcmp(argv[i], "--f16-io") == 0) {
            result.f16_io = true;
        } else if (strcmp(argv[i], "--staged-dmabuf") == 0) {
            result.staged_dmabuf = true;
        } else if (strcmp(argv[i], "--max-requests") == 0 && i + 1 < argc) {
            int64_t value = 0;
            if (!parse_i64(argv[++i], value) || value < 0 || value > LONG_MAX) {
                return false;
            }
            result.max_requests = static_cast<long>(value);
        } else {
            return false;
        }
    }
    const bool tcp = result.port > 0 && result.ffs_root.empty() &&
            result.ready_file.empty();
#if defined(__ANDROID__)
    const bool dmabuf = result.port == 0 && !result.ffs_root.empty() &&
            !result.ready_file.empty();
#else
    const bool dmabuf = false;
#endif
    uint8_t artifact_digest[32] = {};
    return !result.model.empty() && !result.backend.empty() &&
           ffn_split::parse_artifact_sha256(
                   result.artifact_sha256, artifact_digest) &&
           !result.bind.empty() && result.layer_mask != 0 &&
           result.columns > 0 && result.column_quantum > 0 &&
           (!result.staged_dmabuf || dmabuf) &&
           (!result.staged_dmabuf || result.queue_depth == 1) &&
           (result.alternate_columns == 0 ||
            (result.alternate_columns < result.columns &&
             result.alternate_columns % 32 == 0 &&
             result.alternate_columns / 32 <= UINT16_MAX)) &&
           result.max_tokens > 0 && (tcp != dmabuf);
}

bool read_exact(int fd, void * data, size_t size) {
    uint8_t * ptr = static_cast<uint8_t *>(data);
    while (size > 0) {
        const ssize_t count = read(fd, ptr, size);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            return false;
        }
        ptr += count;
        size -= static_cast<size_t>(count);
    }
    return true;
}

bool write_exact(int fd, const void * data, size_t size) {
    const uint8_t * ptr = static_cast<const uint8_t *>(data);
    while (size > 0) {
        const ssize_t count = write(fd, ptr, size);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            return false;
        }
        ptr += count;
        size -= static_cast<size_t>(count);
    }
    return true;
}

bool read_u64_key(
        const gguf_context * gguf, const char * name, uint64_t & value) {
    const int64_t key = gguf_find_key(gguf, name);
    if (key < 0) {
        return false;
    }
    switch (gguf_get_kv_type(gguf, key)) {
        case GGUF_TYPE_UINT64:
            value = gguf_get_val_u64(gguf, key);
            return true;
        case GGUF_TYPE_UINT32:
            value = gguf_get_val_u32(gguf, key);
            return true;
        default:
            return false;
    }
}

bool read_string_key(
        const gguf_context * gguf, const char * name, std::string & value) {
    const int64_t key = gguf_find_key(gguf, name);
    if (key < 0 || gguf_get_kv_type(gguf, key) != GGUF_TYPE_STRING) {
        return false;
    }
    value = gguf_get_val_str(gguf, key);
    return true;
}

bool read_fused_shard_info(
        const gguf_context * gguf, fused_shard_info & fused, std::string & error) {
    uint64_t version = 0;
    if (!read_u64_key(gguf, "s43.fused.version", version)) {
        fused.present = false;
        return true;
    }
    uint64_t n_q = 0;
    uint64_t n_kv = 0;
    const int64_t eps_key = gguf_find_key(gguf, "s43.fused.rms_eps");
    if (version != 1 ||
        !read_u64_key(gguf, "s43.fused.layer_mask", fused.layer_mask) ||
        !read_u64_key(gguf, "s43.fused.next_mask", fused.next_mask) ||
        !read_u64_key(gguf, "s43.fused.n_q", n_q) ||
        !read_u64_key(gguf, "s43.fused.n_kv", n_kv) ||
        eps_key < 0 || gguf_get_kv_type(gguf, eps_key) != GGUF_TYPE_FLOAT32 ||
        fused.layer_mask == 0 || (fused.next_mask & ~fused.layer_mask) != 0 ||
        n_q == 0 || n_kv == 0 || n_q > INT32_MAX || n_kv > INT32_MAX) {
        error = "fused shard metadata is invalid";
        return false;
    }
    fused.rms_eps = gguf_get_val_f32(gguf, eps_key);
    if (!(fused.rms_eps > 0.0f) || !std::isfinite(fused.rms_eps)) {
        error = "fused shard rms epsilon is invalid";
        return false;
    }
    fused.present = true;
    fused.n_q = static_cast<int64_t>(n_q);
    fused.n_kv = static_cast<int64_t>(n_kv);
    return true;
}

bool read_ffn_shard_info(
        const gguf_context * gguf, ffn_shard_info & shard, std::string & error) {
    uint64_t version = 0;
    if (!read_u64_key(gguf, "s42.ffn_shard.version", version)) {
        shard.present = false;
        return true;
    }
    uint64_t n_embd = 0;
    uint64_t n_ff = 0;
    uint64_t column_offset = 0;
    uint64_t columns = 0;
    uint8_t digest[32] = {};
    if (version != 1 ||
        !read_string_key(gguf, "s42.ffn_shard.parent_sha256", shard.parent_sha256) ||
        !ffn_split::parse_artifact_sha256(shard.parent_sha256, digest) ||
        !read_u64_key(gguf, "s42.ffn_shard.n_embd", n_embd) ||
        !read_u64_key(gguf, "s42.ffn_shard.n_ff", n_ff) ||
        !read_u64_key(gguf, "s42.ffn_shard.column_offset", column_offset) ||
        !read_u64_key(gguf, "s42.ffn_shard.columns", columns) ||
        !read_u64_key(gguf, "s42.ffn_shard.layer_mask", shard.layer_mask) ||
        !read_string_key(gguf, "s42.ffn_shard.weight_type", shard.weight_type) ||
        n_embd == 0 || n_ff == 0 || columns == 0 || shard.layer_mask == 0 ||
        n_embd > INT64_MAX || n_ff > INT64_MAX ||
        column_offset + columns != n_ff) {
        error = "FFN shard metadata is invalid";
        return false;
    }
    shard.present = true;
    shard.n_embd = static_cast<int64_t>(n_embd);
    shard.n_ff = static_cast<int64_t>(n_ff);
    shard.column_offset = static_cast<int64_t>(column_offset);
    shard.columns = static_cast<int64_t>(columns);
    return true;
}

bool open_gguf(
        const std::string & path, gguf_reader & reader,
        bool & swiglu, ffn_shard_info & shard, std::string & error) {
    reader.fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    struct stat file_stat = {};
    if (reader.fd < 0 || fstat(reader.fd, &file_stat) != 0 || !S_ISREG(file_stat.st_mode)) {
        error = "cannot open model GGUF";
        return false;
    }
    const std::string fd_path = "/proc/self/fd/" + std::to_string(reader.fd);
    gguf_init_params params = { true, &reader.metadata };
    reader.gguf = gguf_init_from_file(fd_path.c_str(), params);
    if (reader.gguf == nullptr || reader.metadata == nullptr) {
        error = "cannot parse model GGUF";
        return false;
    }
    const int64_t architecture_key = gguf_find_key(reader.gguf, "general.architecture");
    if (architecture_key < 0 ||
        gguf_get_kv_type(reader.gguf, architecture_key) != GGUF_TYPE_STRING) {
        error = "FFN split worker requires a model architecture";
        return false;
    }
    const char * architecture = gguf_get_val_str(reader.gguf, architecture_key);
    if (strcmp(architecture, "gemma4") == 0) {
        swiglu = false;
    } else if (strcmp(architecture, "qwen3") == 0 ||
               strcmp(architecture, "llama") == 0) {
        swiglu = true;
    } else {
        error = "FFN split worker requires a Gemma4, Qwen3, or Llama GGUF";
        return false;
    }
    return read_ffn_shard_info(reader.gguf, shard, error);
}

bool read_tensor(
        gguf_reader & reader,
        const std::string & name,
        host_tensor & result,
        std::string & error) {
    const int64_t tensor_id = gguf_find_tensor(reader.gguf, name.c_str());
    ggml_tensor * tensor = tensor_id >= 0 ?
            ggml_get_tensor(reader.metadata, name.c_str()) : nullptr;
    if (tensor_id < 0 || tensor == nullptr || tensor->ne[2] != 1 || tensor->ne[3] != 1) {
        error = "missing or non-matrix tensor: " + name;
        return false;
    }
    result.ne0 = tensor->ne[0];
    result.ne1 = tensor->ne[1];
    result.type = tensor->type;
    result.bytes.resize(ggml_nbytes(tensor));
    const size_t data_offset = gguf_get_data_offset(reader.gguf);
    const size_t tensor_offset = gguf_get_tensor_offset(reader.gguf, tensor_id);
    if (tensor_offset > std::numeric_limits<size_t>::max() - data_offset) {
        error = "tensor offset overflow: " + name;
        return false;
    }
    const off_t offset = static_cast<off_t>(data_offset + tensor_offset);
    size_t completed = 0;
    while (completed < result.bytes.size()) {
        const ssize_t count = pread(
                reader.fd, result.bytes.data() + completed,
                result.bytes.size() - completed, offset + completed);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            break;
        }
        completed += static_cast<size_t>(count);
    }
    if (completed != result.bytes.size()) {
        error = "short tensor read: " + name;
        return false;
    }
    return true;
}

bool slice_rows(
        const host_tensor & source,
        int64_t row_offset,
        int64_t row_count,
        host_tensor & result) {
    if (row_offset < 0 || row_count <= 0 || row_offset + row_count > source.ne1) {
        return false;
    }
    const size_t row_bytes = ggml_row_size(source.type, source.ne0);
    if (row_bytes == 0 || source.bytes.size() != row_bytes * static_cast<size_t>(source.ne1)) {
        return false;
    }
    result.ne0 = source.ne0;
    result.ne1 = row_count;
    result.type = source.type;
    result.bytes.resize(row_bytes * static_cast<size_t>(row_count));
    memcpy(
            result.bytes.data(),
            source.bytes.data() + row_bytes * static_cast<size_t>(row_offset),
            result.bytes.size());
    return true;
}

bool slice_columns(
        const host_tensor & source,
        int64_t column_offset,
        int64_t column_count,
        host_tensor & result) {
    const int64_t block = ggml_blck_size(source.type);
    if (column_offset < 0 || column_count <= 0 ||
        column_offset + column_count > source.ne0 ||
        column_offset % block != 0 || column_count % block != 0) {
        return false;
    }
    const size_t source_row_bytes = ggml_row_size(source.type, source.ne0);
    const size_t prefix_bytes = ggml_row_size(source.type, column_offset);
    const size_t result_row_bytes = ggml_row_size(source.type, column_count);
    if (source.bytes.size() != source_row_bytes * static_cast<size_t>(source.ne1) ||
        prefix_bytes + result_row_bytes > source_row_bytes) {
        return false;
    }
    result.ne0 = column_count;
    result.ne1 = source.ne1;
    result.type = source.type;
    result.bytes.resize(result_row_bytes * static_cast<size_t>(source.ne1));
    for (int64_t row = 0; row < source.ne1; ++row) {
        memcpy(
                result.bytes.data() + result_row_bytes * static_cast<size_t>(row),
                source.bytes.data() + source_row_bytes * static_cast<size_t>(row) + prefix_bytes,
                result_row_bytes);
    }
    return true;
}

bool all_finite(const float * values, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        uint32_t bits = 0;
        memcpy(&bits, values + i, sizeof(bits));
        if ((bits & 0x7f800000U) == 0x7f800000U) {
            return false;
        }
    }
    return true;
}

bool all_finite(const std::vector<float> & values) {
    return all_finite(values.data(), values.size());
}

bool all_finite(const ggml_fp16_t * values, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        uint16_t bits = 0;
        memcpy(&bits, values + i, sizeof(bits));
        if ((bits & 0x7c00U) == 0x7c00U) {
            return false;
        }
    }
    return true;
}

#if defined(__ANDROID__)
struct mapped_dmabuf {
    ~mapped_dmabuf() {
        release();
    }

    void release() {
        if (data != nullptr) {
            munmap(data, size);
        }
        if (fd >= 0) {
            close(fd);
        }
        fd = -1;
        data = nullptr;
        size = 0;
    }

    bool allocate(size_t bytes) {
        release();
        const int heap = open(
                "/dev/dma_heap/qcom,system", O_RDONLY | O_CLOEXEC);
        if (heap < 0) {
            return false;
        }
        dma_heap_allocation_data allocation = {};
        allocation.len = bytes;
        allocation.fd_flags = O_RDWR | O_CLOEXEC;
        const bool allocated =
                ioctl(heap, DMA_HEAP_IOCTL_ALLOC, &allocation) == 0;
        const int saved_errno = errno;
        close(heap);
        errno = saved_errno;
        if (!allocated) {
            return false;
        }
        void * mapping = mmap(
                nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED,
                static_cast<int>(allocation.fd), 0);
        if (mapping == MAP_FAILED) {
            const int map_errno = errno;
            close(static_cast<int>(allocation.fd));
            errno = map_errno;
            return false;
        }
        fd = static_cast<int>(allocation.fd);
        data = static_cast<uint8_t *>(mapping);
        size = bytes;
        return true;
    }

    int fd = -1;
    uint8_t * data = nullptr;
    size_t size = 0;
};
#endif



void pixel_q8_residual(ggml_tensor * dst, const ggml_tensor * input, int ith, int nth, void *) {
    GGML_ASSERT(input->type == GGML_TYPE_F32 && ggml_is_contiguous(input));
    GGML_ASSERT(ggml_nelements(input) % 256 == 0);
    const auto quantize = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float;
    GGML_ASSERT(quantize != nullptr);
    alignas(64) uint8_t packed[512];
    alignas(64) float decoded[256];
    GGML_ASSERT(ggml_type_size(GGML_TYPE_Q8_K) <= sizeof(packed));
    const float * source = static_cast<const float *>(input->data);
    float * output = static_cast<float *>(dst->data);
    for (int64_t block = ith; block < ggml_nelements(input) / 256; block += nth) {
        const float * row = source + block * 256;
        quantize(row, packed, 256);
        dequantize_row_q8_K(reinterpret_cast<const block_q8_K *>(packed), decoded, 256);
        for (int i = 0; i < 256; ++i) output[block * 256 + i] = row[i] - decoded[i];
    }
}

ggml_tensor * pixel_residual_matmul(ggml_context * ctx, ggml_tensor * weights,
                                   ggml_tensor * input, bool correction) {
    ggml_tensor * result = ggml_mul_mat(ctx, weights, input);
    if (!correction) return result;
    GGML_ASSERT(weights->type == GGML_TYPE_Q4_K || weights->type == GGML_TYPE_Q6_K);
    ggml_tensor * residual = ggml_map_custom1(ctx, input, pixel_q8_residual, GGML_N_TASKS_MAX, nullptr);
    return ggml_add(ctx, result, ggml_mul_mat(ctx, weights, residual));
}

#include "pixel_f16_layout.h"

host_tensor pixel_join(const host_tensor & a, const host_tensor & b, bool rows) {
    if (a.type != b.type || (rows ? a.ne0 != b.ne0 : a.ne1 != b.ne1)) std::abort();
    host_tensor result;
    result.type = a.type;
    result.ne0 = rows ? a.ne0 : a.ne0 + b.ne0;
    result.ne1 = rows ? a.ne1 + b.ne1 : a.ne1;
    result.bytes.resize(a.bytes.size() + b.bytes.size());
    if (rows) {
        memcpy(result.bytes.data(), a.bytes.data(), a.bytes.size());
        memcpy(result.bytes.data() + a.bytes.size(), b.bytes.data(), b.bytes.size());
    } else {
        const size_t na = ggml_row_size(a.type, a.ne0);
        const size_t nb = ggml_row_size(b.type, b.ne0);
        for (int64_t row = 0; row < a.ne1; ++row) {
            memcpy(result.bytes.data() + row * (na + nb), a.bytes.data() + row * na, na);
            memcpy(result.bytes.data() + row * (na + nb) + na, b.bytes.data() + row * nb, nb);
        }
    }
    return result;
}

struct layer_state {
    int layer = -1;
    struct block {
        int64_t offset = 0;
        int64_t columns = 0;
        host_tensor gate;
        host_tensor up;
        host_tensor down;
        ggml_tensor * gate_weight = nullptr;
        ggml_tensor * up_weight = nullptr;
        ggml_tensor * down_weight = nullptr;
        // trailing columns of the block that run on the secondary backend
        int64_t secondary_columns = 0;
        host_tensor secondary_gate;
        host_tensor secondary_up;
        host_tensor secondary_down;
        ggml_tensor * secondary_gate_weight = nullptr;
        ggml_tensor * secondary_up_weight = nullptr;
        ggml_tensor * secondary_down_weight = nullptr;
        ggml_tensor * copy_gate_weight = nullptr;
        ggml_tensor * copy_up_weight = nullptr;
        ggml_tensor * copy_down_weight = nullptr;

        int64_t primary_columns() const {
            return columns - secondary_columns;
        }
    };

    std::vector<block> blocks;
    bool coalesced = false;
    // WS15 fused call of this layer: O(l), ffn_norm(l), and with next_qkv the
    // attn_norm and raw Q/K/V projections of layer l+1
    bool fused = false;
    bool next_qkv = false;
    host_tensor wo;
    host_tensor ffn_norm;
    host_tensor attn_norm_next;
    host_tensor wq_next;
    host_tensor wk_next;
    host_tensor wv_next;
    ggml_tensor * wo_weight = nullptr;
    ggml_tensor * ffn_norm_weight = nullptr;
    ggml_tensor * attn_norm_next_weight = nullptr;
    ggml_tensor * wq_next_weight = nullptr;
    ggml_tensor * wk_next_weight = nullptr;
    ggml_tensor * wv_next_weight = nullptr;
    std::pair<size_t, size_t> selection(int64_t columns) const {
        if (coalesced && columns == 17408) return {0, 1};
        size_t first = blocks.size();
        int64_t selected = 0;
        while (first > 0 && selected < columns) selected += blocks[--first].columns;
        return selected == columns ? std::make_pair(first, blocks.size()) : std::make_pair(size_t(0), size_t(0));
    }
};

} // namespace

int ffn_split_worker_main(int argc, char ** argv) {
    config cfg;
    if (!parse_config(argc, argv, cfg)) {
        fprintf(stderr,
                "usage: %s -m MODEL --artifact-sha256 SHA256 "
                "(--layer N | --layers SPEC) --columns N "
                "--backend DEVICE (--port P [--bind ADDRESS] | "
                "--ffs-root PATH --ready-file PATH) "
                "[--f16-io] [--staged-dmabuf] [--max-tokens N] "
                "[--queue-depth N] "
                "[--column-quantum N] "
                "[--alternate-columns N] [--max-requests N]\n",
                argv[0]);
        return 2;
    }
    uint8_t artifact_sha256[32] = {};
    if (!ffn_split::parse_artifact_sha256(
                cfg.artifact_sha256, artifact_sha256)) {
        return 2;
    }
    signal(SIGPIPE, SIG_IGN);
    if (!apply_uclamp_min()) {
        return 2;
    }

    gguf_reader reader;
    std::string error;
    secondary_config secondary;
    if (!parse_secondary_config(secondary, error)) {
        fprintf(stderr, "[ffn-worker] %s\n", error.c_str());
        return 2;
    }


    const bool dual = secondary.enabled();
    const bool pixel_coalesce = getenv("S42_PIXEL_COALESCE_FULL") != nullptr;

    const bool pixel_packed = getenv("S42_PIXEL_PACKED_WEIGHTS") != nullptr;
    const bool pixel_correction = getenv("S42_PIXEL_CPU_QUANT_RESIDUAL") != nullptr;
    if (pixel_correction && (!pixel_packed || cfg.backend != "CPU")) {
        fprintf(stderr, "[ffn-worker] residual correction requires packed CPU weights\n");
        return 2;
    }
    if (pixel_correction) fprintf(stderr, "S42PIXELRESIDUAL activation=q8_K correction=second_dot\n");
    if (pixel_coalesce && (!dual || !secondary.cpu_half_columns)) {
        fprintf(stderr, "[ffn-worker] coalescing requires a fine CPU/GPU split\n");
        return 2;
    }
    if (pixel_packed && secondary.cpu_half_columns % 256 != 0) {
        fprintf(stderr, "[ffn-worker] packed ratios require 256-column alignment\n");
        return 2;
    }
    if (secondary.block_mask && (cfg.backend != "CPU" || secondary.backend != "Vulkan0" ||
            cfg.columns != 17408 || cfg.column_quantum != 4352 || cfg.alternate_columns != 0 ||
            !cfg.ffs_root.empty() || secondary.fraction != 0 || secondary.max_tokens != 0)) {
        fprintf(stderr, "[ffn-worker] incompatible private whole-block configuration\n");
        return 2;
    }
    if (secondary.trailing_columns && (secondary.fraction != 0 || secondary.max_tokens != 0 ||
            pixel_coalesce || cfg.alternate_columns != 0 || secondary.trailing_columns >= cfg.columns)) {
        fprintf(stderr, "[ffn-worker] incompatible trailing secondary configuration\n");
        return 2;
    }
    bool swiglu = false;
    ffn_shard_info shard;
    print_residency_phase(cfg, "WEIGHT_READ_BEGIN");
    if (!open_gguf(cfg.model, reader, swiglu, shard, error)) {
        fprintf(stderr, "[ffn-worker] %s\n", error.c_str());
        return 1;
    }
    fused_shard_info fused_info;
    if (!read_fused_shard_info(reader.gguf, fused_info, error)) {
        fprintf(stderr, "[ffn-worker] %s\n", error.c_str());
        return 1;
    }
    // fused capability: the shard's fused layers this worker serves
    const uint64_t fused_layer_mask = fused_info.present ? (fused_info.layer_mask & cfg.layer_mask) : 0;
    const uint64_t fused_next_mask = fused_info.present ? (fused_info.next_mask & fused_layer_mask) : 0;
    if (shard.present) {
        // The shard must derive from the requested artifact and must cover
        // the requested layers and column suffix; the served suffix may be a
        // smaller tail of the stored slice (fraction changes never reload).
        if (shard.parent_sha256 != cfg.artifact_sha256) {
            fprintf(stderr,
                    "[ffn-worker] FFN shard parent %s differs from artifact %s\n",
                    shard.parent_sha256.c_str(), cfg.artifact_sha256.c_str());
            return 1;
        }
        if ((cfg.layer_mask & ~shard.layer_mask) != 0) {
            fprintf(stderr,
                    "[ffn-worker] FFN shard layers %016llx do not cover %016llx\n",
                    static_cast<unsigned long long>(shard.layer_mask),
                    static_cast<unsigned long long>(cfg.layer_mask));
            return 1;
        }
        if (cfg.columns > shard.columns) {
            fprintf(stderr,
                    "[ffn-worker] FFN shard stores %lld columns, %lld requested\n",
                    static_cast<long long>(shard.columns),
                    static_cast<long long>(cfg.columns));
            return 1;
        }
        fprintf(stderr,
                "[ffn-worker] FFN shard parent=%s layers=%016llx "
                "slice=[%lld,%lld) NFF=%lld type=%s\n",
                shard.parent_sha256.c_str(),
                static_cast<unsigned long long>(shard.layer_mask),
                static_cast<long long>(shard.column_offset),
                static_cast<long long>(shard.n_ff),
                static_cast<long long>(shard.n_ff),
                shard.weight_type.c_str());
    }
    std::vector<layer_state> states;
    int64_t n_embd = 0;
    int64_t n_ff = 0;
    int64_t offset = 0;
    ggml_type weight_type = GGML_TYPE_COUNT;
    uint64_t weight_hash = 14695981039346656037ULL;
    std::vector<int64_t> runtime_columns;
    for (int64_t columns = cfg.column_quantum;
         columns < cfg.columns;
         columns += cfg.column_quantum) {
        runtime_columns.push_back(columns);
    }
    if (cfg.alternate_columns > 0) {
        runtime_columns.push_back(cfg.alternate_columns);
    }
    runtime_columns.push_back(cfg.columns);
    std::sort(runtime_columns.begin(), runtime_columns.end());
    runtime_columns.erase(
            std::unique(runtime_columns.begin(), runtime_columns.end()),
            runtime_columns.end());
    std::vector<int64_t> partition_columns;
    int64_t previous_columns = 0;
    for (int64_t columns : runtime_columns) {
        partition_columns.push_back(columns - previous_columns);
        previous_columns = columns;
    }

    std::reverse(partition_columns.begin(), partition_columns.end());
    if (secondary.trailing_columns) {
        // every runtime width stays a sum of trailing blocks
        const int64_t boundary = cfg.columns - secondary.trailing_columns;
        std::vector<int64_t> split;
        int64_t start = 0;
        for (int64_t block_columns : partition_columns) {
            if (start < boundary && boundary < start + block_columns) {
                split.push_back(boundary - start);
                split.push_back(start + block_columns - boundary);
            } else {
                split.push_back(block_columns);
            }
            start += block_columns;
        }
        partition_columns = split;
        fprintf(stderr, "S43PIXELTRAILING secondary_columns=%lld primary_columns=%lld blocks=%zu\n",
                static_cast<long long>(secondary.trailing_columns), static_cast<long long>(boundary),
                partition_columns.size());
    }
    if (secondary.cpu_half_columns) {
        const int64_t cpu = secondary.cpu_half_columns;
        const int64_t gpu = 8704 - cpu;
        runtime_columns = {8704, 17408};
        partition_columns = {cpu, gpu, gpu, cpu};
        fprintf(stderr, "S42PIXELRATIO cpu_per_half=%lld gpu_per_half=%lld grain=64\n",
                static_cast<long long>(cpu), static_cast<long long>(gpu));
    }

    if (getenv("S42_PIXEL_CPU_PAIR_BLOCKS")) {
        if (dual || cfg.backend != "CPU" || !pixel_packed || cfg.columns != 17408 || cfg.column_quantum != 4352) {
            fprintf(stderr, "[ffn-worker] CPU block pairing requires packed CPU-only mode\n");
            return 2;
        }
        runtime_columns = {8704, 17408};
        partition_columns = {8704, 8704};
        fprintf(stderr, "S42PIXELCPUPAIR full_projections=6 half_projections=3\n");
    }
    for (int layer = 0; layer < 64; ++layer) {
        if ((cfg.layer_mask & (UINT64_C(1) << layer)) == 0) {
            continue;
        }
        const std::string prefix = "blk." + std::to_string(layer);
        host_tensor gate_full;
        host_tensor up_full;
        host_tensor down_full;
        if (!read_tensor(reader, prefix + ".ffn_gate.weight", gate_full, error) ||
            !read_tensor(reader, prefix + ".ffn_up.weight", up_full, error) ||
            !read_tensor(reader, prefix + ".ffn_down.weight", down_full, error)) {
            fprintf(stderr, "[ffn-worker] %s\n", error.c_str());
            return 1;
        }
        if (gate_full.ne0 <= 0 || gate_full.ne1 < cfg.columns ||
            gate_full.ne0 != up_full.ne0 || gate_full.ne1 != up_full.ne1 ||
            down_full.ne0 != gate_full.ne1 || down_full.ne1 != gate_full.ne0 ||
            (!pixel_packed && (gate_full.type != up_full.type || gate_full.type != down_full.type))) {
            fprintf(stderr, "[ffn-worker] layer %d tensor type or shape mismatch\n", layer);
            return 1;
        }
        if (shard.present &&
            (gate_full.ne0 != shard.n_embd || gate_full.ne1 != shard.columns ||
             (!pixel_packed && strcasecmp(ggml_type_name(gate_full.type), shard.weight_type.c_str()) != 0))) {
            fprintf(stderr,
                    "[ffn-worker] layer %d FFN shard tensors do not match shard metadata\n",
                    layer);
            return 1;
        }

        if (pixel_packed) {
            for (const host_tensor * tensor : {&gate_full, &up_full, &down_full}) {
                if (tensor->type != GGML_TYPE_Q4_K && tensor->type != GGML_TYPE_Q6_K) {
                    fprintf(stderr, "[ffn-worker] packed experiment requires original Q4_K/Q6_K weights\n");
                    return 1;
                }
            }
            fprintf(stderr, "S42PIXELPACKED layer=%d gate=%s up=%s down=%s\n", layer,
                    ggml_type_name(gate_full.type), ggml_type_name(up_full.type), ggml_type_name(down_full.type));
        }
        if (states.empty()) {
            n_embd = gate_full.ne0;
            // In shard mode the stored tensors are already the n_ff suffix, so
            // the parent geometry comes from the shard metadata; every hash
            // input below stays identical to the full-GGUF path.
            n_ff = shard.present ? shard.n_ff : gate_full.ne1;
            offset = n_ff - cfg.columns;
            weight_type = gate_full.type;
            const int64_t block = ggml_blck_size(weight_type);
            if (offset % block != 0 || cfg.columns % block != 0 ||
                cfg.column_quantum % block != 0 ||
                cfg.alternate_columns % block != 0) {
                fprintf(stderr,
                        "[ffn-worker] split and quantum must align to weight block=%lld\n",
                        static_cast<long long>(block));
                return 1;
            }
        } else if (gate_full.ne0 != n_embd ||
                   gate_full.ne1 != (shard.present ? shard.columns : n_ff) ||
                   gate_full.type != weight_type) {
            fprintf(stderr, "[ffn-worker] selected layers do not share one FFN shape\n");
            return 1;
        }
        // Global column index -> index inside the stored tensor.
        const int64_t slice_base = shard.present ? shard.column_offset : 0;

        layer_state state;
        state.layer = layer;
        host_tensor gate_suffix;
        host_tensor up_suffix;
        host_tensor down_suffix;
        if (!slice_rows(gate_full, offset - slice_base, cfg.columns, gate_suffix) ||
            !slice_rows(up_full, offset - slice_base, cfg.columns, up_suffix) ||
            !slice_columns(down_full, offset - slice_base, cfg.columns, down_suffix)) {
            fprintf(stderr, "[ffn-worker] cannot extract layer %d FFN suffix\n", layer);
            return 1;
        }
        const uint64_t metadata[] = {
            static_cast<uint64_t>(layer), static_cast<uint64_t>(n_embd),
            static_cast<uint64_t>(n_ff), static_cast<uint64_t>(offset),
            static_cast<uint64_t>(cfg.columns), static_cast<uint64_t>(weight_type),
        };
        weight_hash = ffn_split::hash64_update(
                weight_hash, metadata, sizeof(metadata));
        weight_hash = ffn_split::hash64_update(
                weight_hash, gate_suffix.bytes.data(), gate_suffix.bytes.size());
        weight_hash = ffn_split::hash64_update(
                weight_hash, up_suffix.bytes.data(), up_suffix.bytes.size());
        weight_hash = ffn_split::hash64_update(
                weight_hash, down_suffix.bytes.data(), down_suffix.bytes.size());

        int64_t block_offset = offset;
        for (int64_t block_columns : partition_columns) {
            layer_state::block weight_block;
            weight_block.offset = block_offset;
            weight_block.columns = block_columns;
            if (!slice_rows(
                        gate_full, block_offset - slice_base, block_columns,
                        weight_block.gate) ||
                !slice_rows(
                        up_full, block_offset - slice_base, block_columns,
                        weight_block.up) ||
                !slice_columns(
                        down_full, block_offset - slice_base, block_columns,
                        weight_block.down)) {
                fprintf(stderr,
                        "[ffn-worker] cannot partition layer %d FFN suffix\n",
                        layer);
                return 1;
            }

            if (dual && secondary.trailing_columns) {
                if (block_offset - offset >= cfg.columns - secondary.trailing_columns) {
                    weight_block.secondary_columns = block_columns;
                    weight_block.secondary_gate = std::move(weight_block.gate);
                    weight_block.secondary_up = std::move(weight_block.up);
                    weight_block.secondary_down = std::move(weight_block.down);
                }
            } else if (dual && secondary.block_mask) {
                if (secondary.block_mask & (1u << state.blocks.size())) {
                    weight_block.secondary_columns = block_columns;
                    weight_block.secondary_gate = std::move(weight_block.gate);
                    weight_block.secondary_up = std::move(weight_block.up);
                    weight_block.secondary_down = std::move(weight_block.down);
                }
            } else if (dual) {
                const int64_t align = secondary.align;
                int64_t secondary_columns = static_cast<int64_t>(std::llround(
                        secondary.fraction * static_cast<double>(block_columns) /
                        static_cast<double>(align))) * align;
                secondary_columns = std::min(
                        secondary_columns, (block_columns - 1) / align * align);
                const int64_t primary_columns = block_columns - secondary_columns;
                host_tensor gate;
                host_tensor up;
                host_tensor down;
                if (secondary_columns > 0 &&
                    (primary_columns % ggml_blck_size(weight_type) != 0 ||
                     !slice_rows(weight_block.gate, primary_columns,
                                 secondary_columns, weight_block.secondary_gate) ||
                     !slice_rows(weight_block.up, primary_columns,
                                 secondary_columns, weight_block.secondary_up) ||
                     !slice_columns(weight_block.down, primary_columns,
                                    secondary_columns, weight_block.secondary_down) ||
                     !slice_rows(weight_block.gate, 0, primary_columns, gate) ||
                     !slice_rows(weight_block.up, 0, primary_columns, up) ||
                     !slice_columns(weight_block.down, 0, primary_columns, down))) {
                    fprintf(stderr,
                            "[ffn-worker] cannot split layer %d block for the secondary backend\n",
                            layer);
                    return 1;
                }
                if (secondary_columns > 0) {
                    weight_block.secondary_columns = secondary_columns;
                    weight_block.gate = std::move(gate);
                    weight_block.up = std::move(up);
                    weight_block.down = std::move(down);
                }
            }
            state.blocks.push_back(std::move(weight_block));
            block_offset += block_columns;
        }

        if (pixel_coalesce) {
            layer_state::block full;
            full.offset = offset;
            full.columns = cfg.columns;
            full.secondary_columns = 2 * (8704 - secondary.cpu_half_columns);
            full.gate = pixel_join(state.blocks[0].gate, state.blocks[3].gate, true);
            full.up = pixel_join(state.blocks[0].up, state.blocks[3].up, true);
            full.down = pixel_join(state.blocks[0].down, state.blocks[3].down, false);
            full.secondary_gate = pixel_join(state.blocks[1].secondary_gate, state.blocks[2].secondary_gate, true);
            full.secondary_up = pixel_join(state.blocks[1].secondary_up, state.blocks[2].secondary_up, true);
            full.secondary_down = pixel_join(state.blocks[1].secondary_down, state.blocks[2].secondary_down, false);
            std::vector<layer_state::block> packed;
            packed.push_back(std::move(full));
            packed.push_back(std::move(state.blocks[2]));
            packed.push_back(std::move(state.blocks[3]));
            state.blocks = std::move(packed);
            state.coalesced = true;
        }
        states.push_back(std::move(state));
    }
    if (states.empty()) {
        fprintf(stderr, "[ffn-worker] no layers selected\n");
        return 1;
    }
    uint64_t fused_weight_hash = 14695981039346656037ULL;
    for (layer_state & state : states) {
        const uint64_t bit = UINT64_C(1) << state.layer;
        if ((fused_layer_mask & bit) == 0) {
            continue;
        }
        state.fused = true;
        state.next_qkv = (fused_next_mask & bit) != 0;
        const std::string prefix = "blk." + std::to_string(state.layer);
        const std::string next_prefix = "blk." + std::to_string(state.layer + 1);
        if (!read_tensor(reader, prefix + ".attn_output.weight", state.wo, error) ||
            !read_tensor(reader, prefix + ".ffn_norm.weight", state.ffn_norm, error) ||
            (state.next_qkv &&
             (!read_tensor(reader, next_prefix + ".attn_norm.weight", state.attn_norm_next, error) ||
              !read_tensor(reader, next_prefix + ".attn_q.weight", state.wq_next, error) ||
              !read_tensor(reader, next_prefix + ".attn_k.weight", state.wk_next, error) ||
              !read_tensor(reader, next_prefix + ".attn_v.weight", state.wv_next, error)))) {
            fprintf(stderr, "[ffn-worker] fused layer %d: %s\n", state.layer, error.c_str());
            return 1;
        }
        auto projection_ok = [&](const host_tensor & tensor, int64_t ne0, int64_t ne1) {
            return tensor.ne0 == ne0 && tensor.ne1 == ne1 &&
                    (!pixel_packed || tensor.type == GGML_TYPE_Q4_K || tensor.type == GGML_TYPE_Q6_K);
        };
        auto norm_ok = [&](const host_tensor & tensor) {
            return tensor.ne0 == n_embd && tensor.ne1 == 1 && tensor.type == GGML_TYPE_F32;
        };
        if (!projection_ok(state.wo, fused_info.n_q, n_embd) || !norm_ok(state.ffn_norm) ||
            (state.next_qkv &&
             (!norm_ok(state.attn_norm_next) ||
              !projection_ok(state.wq_next, n_embd, fused_info.n_q) ||
              !projection_ok(state.wk_next, n_embd, fused_info.n_kv) ||
              !projection_ok(state.wv_next, n_embd, fused_info.n_kv)))) {
            fprintf(stderr, "[ffn-worker] fused layer %d tensor type or shape mismatch\n", state.layer);
            return 1;
        }
        int kind = 0;
        for (const host_tensor * tensor : {&state.wo, &state.ffn_norm, &state.attn_norm_next,
                                           &state.wq_next, &state.wk_next, &state.wv_next}) {
            const uint64_t metadata[] = {
                static_cast<uint64_t>(state.layer), static_cast<uint64_t>(kind++),
                static_cast<uint64_t>(tensor->ne0), static_cast<uint64_t>(tensor->ne1),
                static_cast<uint64_t>(tensor->type),
            };
            fused_weight_hash = ffn_split::hash64_update(fused_weight_hash, metadata, sizeof(metadata));
            fused_weight_hash = ffn_split::hash64_update(
                    fused_weight_hash, tensor->bytes.data(), tensor->bytes.size());
        }
    }
    if (fused_layer_mask != 0 && (cfg.columns != n_ff || offset != 0)) {
        fprintf(stderr, "[ffn-worker] fused calls need the full FFN width (columns=%lld n_ff=%lld)\n",
                static_cast<long long>(cfg.columns), static_cast<long long>(n_ff));
        return 1;
    }
    print_residency_phase(cfg, "WEIGHT_READ_READY");

    print_residency_phase(cfg, "HTP_INIT_BEGIN");
    ggml_backend_load_all();
    ggml_backend_dev_t device = ggml_backend_dev_by_name(cfg.backend.c_str());
    if (device == nullptr) {
        fprintf(stderr, "[ffn-worker] backend not found: %s\n", cfg.backend.c_str());
        return 1;
    }
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    if (backend == nullptr) {
        fprintf(stderr, "[ffn-worker] backend initialization failed\n");
        return 1;
    }

    std::unique_ptr<ggml_threadpool, decltype(&ggml_threadpool_free)> cpu_pool(nullptr, ggml_threadpool_free);
    if (cfg.backend == "CPU" && getenv("S42_PIXEL_CPU_THREADS")) {
        auto setting = [](const char * name, int fallback, int base) -> long {
            const char * value = getenv(name);
            if (!value) return fallback;
            char * end = nullptr;
            errno = 0;
            long result = strtol(value, &end, base);
            return errno || end == value || *end ? -1 : result;
        };
        long threads = setting("S42_PIXEL_CPU_THREADS", 4, 10);
        long mask = setting("S42_PIXEL_CPU_MASK", 0, 16);
        long persistent = setting("S42_PIXEL_CPU_POOL", 0, 10);
        if (threads < 1 || threads > 8 || mask < 0 || mask > 255 || persistent < 0 || persistent > 1 ||
            (mask && (!persistent || __builtin_popcount(unsigned(mask)) < threads))) {
            fprintf(stderr, "[ffn-worker] invalid private CPU tuning parameters\n");
            ggml_backend_free(backend);
            return 1;
        }
        ggml_backend_cpu_set_n_threads(backend, int(threads));
        if (persistent) {
            ggml_threadpool_params params = ggml_threadpool_params_default(int(threads));
            params.paused = true;
            params.poll = 0;
            if (const char * poll = getenv("S43_PIXEL_CPU_POLL")) {
                params.poll = static_cast<uint32_t>(std::clamp(atoi(poll), 0, 100));
                fprintf(stderr, "S43PIXELCPUPOLL poll=%u\n", params.poll);
            }
            params.strict_cpu = mask != 0;
            for (int i = 0; i < 8; ++i) params.cpumask[i] = (mask & (1 << i)) != 0;
            cpu_pool.reset(ggml_threadpool_new(&params));
            if (!cpu_pool) { ggml_backend_free(backend); return 1; }
            ggml_backend_cpu_set_threadpool(backend, cpu_pool.get());
        }
        fprintf(stderr, "S42PIXELCPU threads=%ld mask=%lx persistent=%ld poll=0\n", threads, mask, persistent);
    }

    print_residency_phase(cfg, "HTP_INIT_READY");

    const size_t block_count = states.front().blocks.size();
    if (block_count == 0 || std::any_of(
                states.begin(), states.end(), [block_count](const layer_state & state) {
                    return state.blocks.size() != block_count;
                })) {
        fprintf(stderr, "[ffn-worker] inconsistent FFN block partition\n");
        return 1;
    }
    ggml_backend_buffer_type_t weight_buft =
            ggml_backend_get_default_buffer_type(backend);
    if (ggml_is_quantized(weight_type) && !pixel_packed) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
        auto get_extra_bufts = reinterpret_cast<
                ggml_backend_dev_get_extra_bufts_t>(
                ggml_backend_reg_get_proc_address(
                    reg, "ggml_backend_dev_get_extra_bufts"));
        ggml_backend_buffer_type_t * extra_bufts =
                get_extra_bufts != nullptr ? get_extra_bufts(device) : nullptr;
        if (extra_bufts != nullptr && extra_bufts[0] != nullptr) {
            weight_buft = extra_bufts[0];
        }
    }
    std::vector<ggml_context *> weight_contexts(block_count, nullptr);
    std::vector<ggml_backend_buffer_t> weight_buffers(block_count, nullptr);
    size_t weight_buffer_bytes = 0;
#if defined(S41_FFN_SINGLE_WEIGHT_BUFFER)
    const size_t weight_allocation_count = 1;
#else
    const size_t weight_allocation_count = block_count;
#endif
    weight_contexts.resize(weight_allocation_count, nullptr);
    weight_buffers.resize(weight_allocation_count, nullptr);

    pixel_cpu_layout.rows = pixel_layout_integer("S42_PIXEL_NEON_ROWS", 0);
    pixel_cpu_layout.chunk = pixel_layout_integer("S42_PIXEL_NEON_CHUNK", 0);
    pixel_cpu_layout.unroll = pixel_layout_integer("S42_PIXEL_NEON_UNROLL", 1);
    pixel_cpu_layout.prefetch = pixel_layout_integer("S42_PIXEL_NEON_PREFETCH", 0);
    pixel_cpu_layout.fmlal = pixel_layout_integer("S42_PIXEL_NEON_FMLAL", 0);
    GGML_ASSERT(pixel_cpu_layout.fmlal <= 1);
    pixel_fused_residual = pixel_layout_integer("S42_PIXEL_FUSED_RESIDUAL", 0) == 1;
    if (pixel_fused_residual) {
        GGML_ASSERT(pixel_packed && pixel_correction && cfg.backend == "CPU" && !pixel_cpu_layout.rows);
        fprintf(stderr, "S42PIXELFUSEDRESIDUAL quantization=q8_K_pair weights=shared_per_row\n");
    }

    pixel_cpu_row_chunk = pixel_layout_integer("S42_PIXEL_CPU_ROW_CHUNK", 0);
    pixel_cpu_pair_mode = pixel_layout_integer("S42_PIXEL_CPU_PAIR_DOT", 0);
    pixel_cpu_row_profile = pixel_layout_integer("S42_PIXEL_CPU_ROW_PROFILE", 0) == 1;
    GGML_ASSERT(pixel_cpu_pair_mode <= 2);
    GGML_ASSERT(pixel_cpu_row_chunk == 0 || pixel_cpu_row_chunk == 16 || pixel_cpu_row_chunk == 32 ||
                pixel_cpu_row_chunk == 64 || pixel_cpu_row_chunk == 128 || pixel_cpu_row_chunk == 256);
    if (pixel_cpu_row_chunk || pixel_cpu_pair_mode || pixel_cpu_row_profile) {
        GGML_ASSERT(pixel_fused_residual && cfg.backend == "CPU" && (!secondary.enabled() || secondary.trailing_columns));
#if defined(__aarch64__)
        if (pixel_cpu_pair_mode) GGML_ASSERT(getauxval(AT_HWCAP) & HWCAP_ASIMDDP);
#else
        GGML_ASSERT(!pixel_cpu_pair_mode && !pixel_cpu_row_chunk && "NEON packed dots need aarch64");
#endif
        fprintf(stderr, "S42PIXELCPUPACK chunk=%d mode=%d profile=%d\n",
                pixel_cpu_row_chunk, pixel_cpu_pair_mode, int(pixel_cpu_row_profile));
    }
    pixel_cpu_batch_pair = pixel_layout_integer("S43_PIXEL_CPU_BATCH_PAIR", 0) == 1;
    if (pixel_cpu_batch_pair) {
        GGML_ASSERT(pixel_cpu_pair_mode == 1 && cfg.max_tokens <= 4);
        fprintf(stderr, "S43PIXELBATCHPAIR rows=shared max_tokens=%lld\n", static_cast<long long>(cfg.max_tokens));
    }
    pixel_gpu_layout.rows = pixel_layout_integer("S42_PIXEL_GPU_LAYOUT_ROWS", 0);
    pixel_gpu_layout.chunk = pixel_layout_integer("S42_PIXEL_GPU_LAYOUT_CHUNK", 0);
    const bool pixel_gpu_f16 = pixel_layout_integer("S42_PIXEL_GPU_EXPAND_F16", 0) == 1;
    if (pixel_gpu_f16) {
        GGML_ASSERT(pixel_packed && !secondary.max_tokens && !pixel_coalesce);
        GGML_ASSERT(cfg.backend == "Vulkan0" || secondary.backend == "Vulkan0");
        fprintf(stderr, "S42PIXELDEVICEWEIGHTS cpu=original_packed gpu=decoded_f16 requantized=0\n");
    }
    if (pixel_cpu_layout.rows) {
        GGML_ASSERT(cfg.backend == "CPU" && !pixel_packed && !pixel_coalesce && !secondary.max_tokens);
        GGML_ASSERT(pixel_cpu_layout.rows == 2 || pixel_cpu_layout.rows == 4 || pixel_cpu_layout.rows == 8);
        GGML_ASSERT(pixel_cpu_layout.chunk == 0 || pixel_cpu_layout.chunk == 8 || pixel_cpu_layout.chunk == 32);
        GGML_ASSERT(pixel_cpu_layout.unroll == 1 || (pixel_cpu_layout.unroll == 2 && pixel_cpu_layout.rows <= 4));
        fprintf(stderr, "S42PIXELNEON rows=%d chunk=%d unroll=%d prefetch=%d precision=f32\n",
                pixel_cpu_layout.rows, pixel_cpu_layout.chunk, pixel_cpu_layout.unroll, pixel_cpu_layout.prefetch);
        fprintf(stderr, "S42PIXELNEONMATH activation=%s accumulate=f32\n", pixel_cpu_layout.fmlal ? "f16" : "f32");
    } else {
        GGML_ASSERT(pixel_cpu_layout.chunk == 0);
    }
    if (pixel_gpu_layout.rows) {
        GGML_ASSERT((!pixel_packed || pixel_gpu_f16) && !pixel_coalesce && !secondary.max_tokens);
        GGML_ASSERT(pixel_gpu_layout.rows == 8 && (pixel_gpu_layout.chunk == 4 || pixel_gpu_layout.chunk == 16));
        GGML_ASSERT(cfg.backend == "Vulkan0" || secondary.backend == "Vulkan0");
        const std::string shader = "layout8k" + std::to_string(pixel_gpu_layout.chunk);
        GGML_ASSERT(getenv("S42_PIXEL_F16_SHADER") && shader == getenv("S42_PIXEL_F16_SHADER"));
        GGML_ASSERT(pixel_layout_integer("S42_PIXEL_F16_ROWS", 0) == 8);
        GGML_ASSERT(pixel_layout_integer("S42_PIXEL_F16_WG", 0) == 128);
        fprintf(stderr, "S42PIXELGPULAYOUT rows=8 chunk=%d precision=f32\n", pixel_gpu_layout.chunk);
    } else {
        const char * shader = getenv("S42_PIXEL_F16_SHADER");
        GGML_ASSERT(!shader || std::string(shader).find("layout") != 0);
    }
    for (layer_state & state : states) {
        for (layer_state::block & block : state.blocks) {
            const auto & primary = cfg.backend == "CPU" ? pixel_cpu_layout : pixel_gpu_layout;
            for (host_tensor * tensor : {&block.gate, &block.up, &block.down}) {
                if (pixel_gpu_f16 && cfg.backend == "Vulkan0") pixel_expand_gpu_weights(*tensor);
                pixel_pack_f16(*tensor, primary);
            }
            for (host_tensor * tensor : {&block.secondary_gate, &block.secondary_up, &block.secondary_down}) {
                if (pixel_gpu_f16) pixel_expand_gpu_weights(*tensor);
                pixel_pack_f16(*tensor, pixel_gpu_layout);
            }
        }
    }
    fprintf(stderr, "S42PIXELPACKLAYOUT bytes=%llu startup_us=%llu extra_resident_bytes=0\n",
            static_cast<unsigned long long>(pixel_layout_bytes), static_cast<unsigned long long>(pixel_layout_us));
    print_residency_phase(cfg, "WEIGHT_UPLOAD_BEGIN");
    for (size_t allocation_index = 0;
         allocation_index < weight_allocation_count; ++allocation_index) {
#if defined(S41_FFN_SINGLE_WEIGHT_BUFFER)
        const size_t first_block = 0;
        const size_t last_block = block_count;
#else
        const size_t first_block = allocation_index;
        const size_t last_block = allocation_index + 1;
#endif

        bool has_primary = false;
        for (size_t i = first_block; i < last_block; ++i) {
            has_primary |= states.front().blocks[i].primary_columns() > 0;
        }
        if (!has_primary) {
            continue;
        }
        ggml_init_params weight_params = {
            ggml_tensor_overhead() *
                    (states.size() * (last_block - first_block) * 6 + 8),
            nullptr, true,
        };
        ggml_context * weight_ctx = ggml_init(weight_params);
        weight_contexts[allocation_index] = weight_ctx;
        for (layer_state & state : states) {
            for (size_t block_index = first_block;
                 block_index < last_block; ++block_index) {
                layer_state::block & block = state.blocks[block_index];
                if (block.primary_columns() == 0) continue;
                block.gate_weight = ggml_new_tensor_2d(
                        weight_ctx, block.gate.type, n_embd, block.primary_columns());
                block.up_weight = ggml_new_tensor_2d(
                        weight_ctx, block.up.type, n_embd, block.primary_columns());
                block.down_weight = ggml_new_tensor_2d(
                        weight_ctx, block.down.type, block.primary_columns(), n_embd);
                if (block.secondary_columns > 0 && secondary.max_tokens > 0) {
                    block.copy_gate_weight = ggml_new_tensor_2d(
                            weight_ctx, block.secondary_gate.type, n_embd, block.secondary_columns);
                    block.copy_up_weight = ggml_new_tensor_2d(
                            weight_ctx, block.secondary_up.type, n_embd, block.secondary_columns);
                    block.copy_down_weight = ggml_new_tensor_2d(
                            weight_ctx, block.secondary_down.type, block.secondary_columns, n_embd);
                }
            }
        }
        ggml_backend_buffer_t weight_buffer =
                ggml_backend_alloc_ctx_tensors_from_buft(
                        weight_ctx, weight_buft);
        if (weight_buffer == nullptr) {
            fprintf(stderr, "[ffn-worker] backend weight allocation failed\n");
            return 1;
        }
        weight_buffers[allocation_index] = weight_buffer;
        weight_buffer_bytes += ggml_backend_buffer_get_size(weight_buffer);
        for (layer_state & state : states) {
            for (size_t block_index = first_block;
                 block_index < last_block; ++block_index) {
                layer_state::block & block = state.blocks[block_index];
                if (block.primary_columns() == 0) continue;
                if (ggml_nbytes(block.gate_weight) != block.gate.bytes.size() ||
                    ggml_nbytes(block.up_weight) != block.up.bytes.size() ||
                    ggml_nbytes(block.down_weight) != block.down.bytes.size()) {
                    fprintf(stderr,
                            "[ffn-worker] layer %d weight allocation mismatch\n",
                            state.layer);
                    return 1;
                }
                ggml_backend_tensor_set(
                        block.gate_weight, block.gate.bytes.data(), 0,
                        block.gate.bytes.size());
                ggml_backend_tensor_set(
                        block.up_weight, block.up.bytes.data(), 0,
                        block.up.bytes.size());
                ggml_backend_tensor_set(
                        block.down_weight, block.down.bytes.data(), 0,
                        block.down.bytes.size());
                if (block.copy_gate_weight != nullptr) {
                    ggml_backend_tensor_set(
                            block.copy_gate_weight, block.secondary_gate.bytes.data(), 0,
                            block.secondary_gate.bytes.size());
                    ggml_backend_tensor_set(
                            block.copy_up_weight, block.secondary_up.bytes.data(), 0,
                            block.secondary_up.bytes.size());
                    ggml_backend_tensor_set(
                            block.copy_down_weight, block.secondary_down.bytes.data(), 0,
                            block.secondary_down.bytes.size());
                }
                std::vector<uint8_t>().swap(block.gate.bytes);
                std::vector<uint8_t>().swap(block.up.bytes);
                std::vector<uint8_t>().swap(block.down.bytes);
            }
        }
        ggml_backend_synchronize(backend);
    }
    ggml_backend_t secondary_backend = nullptr;
    ggml_context * secondary_weight_ctx = nullptr;
    ggml_backend_buffer_t secondary_weight_buffer = nullptr;
    int64_t secondary_block_columns = 0;
    if (dual) {
        ggml_backend_dev_t secondary_device =
                ggml_backend_dev_by_name(secondary.backend.c_str());
        secondary_backend = secondary_device != nullptr ?
                ggml_backend_dev_init(secondary_device, nullptr) : nullptr;
        if (secondary_backend == nullptr) {
            fprintf(stderr, "[ffn-worker] secondary backend unavailable: %s\n",
                    secondary.backend.c_str());
            return 1;
        }
        ggml_init_params params = {
            ggml_tensor_overhead() * (states.size() * block_count * 3 + 8),
            nullptr, true,
        };
        secondary_weight_ctx = ggml_init(params);
        for (layer_state & state : states) {
            for (layer_state::block & block : state.blocks) {
                if (block.secondary_columns == 0) {
                    continue;
                }
                block.secondary_gate_weight = ggml_new_tensor_2d(
                        secondary_weight_ctx, block.secondary_gate.type, n_embd, block.secondary_columns);
                block.secondary_up_weight = ggml_new_tensor_2d(
                        secondary_weight_ctx, block.secondary_up.type, n_embd, block.secondary_columns);
                block.secondary_down_weight = ggml_new_tensor_2d(
                        secondary_weight_ctx, block.secondary_down.type, block.secondary_columns, n_embd);
            }
        }
        const auto full_range = states.front().selection(cfg.columns);
        for (size_t i = full_range.first; i < full_range.second; ++i) {
            secondary_block_columns += states.front().blocks[i].secondary_columns;
        }
        if (secondary_block_columns == 0) {
            fprintf(stderr,
                    "[ffn-worker] secondary fraction %.3f leaves no aligned columns\n",
                    secondary.fraction);
            return 1;
        }
        secondary_weight_buffer = ggml_backend_alloc_ctx_tensors(
                secondary_weight_ctx, secondary_backend);
        if (secondary_weight_buffer == nullptr) {
            fprintf(stderr, "[ffn-worker] secondary weight allocation failed\n");
            return 1;
        }
        for (layer_state & state : states) {
            for (layer_state::block & block : state.blocks) {
                if (block.secondary_columns == 0) {
                    continue;
                }
                ggml_backend_tensor_set(
                        block.secondary_gate_weight, block.secondary_gate.bytes.data(), 0,
                        block.secondary_gate.bytes.size());
                ggml_backend_tensor_set(
                        block.secondary_up_weight, block.secondary_up.bytes.data(), 0,
                        block.secondary_up.bytes.size());
                ggml_backend_tensor_set(
                        block.secondary_down_weight, block.secondary_down.bytes.data(), 0,
                        block.secondary_down.bytes.size());
                std::vector<uint8_t>().swap(block.secondary_gate.bytes);
                std::vector<uint8_t>().swap(block.secondary_up.bytes);
                std::vector<uint8_t>().swap(block.secondary_down.bytes);
            }
        }
        ggml_backend_synchronize(secondary_backend);
        fprintf(stderr,
                "[ffn-worker] dual secondary=%s fraction=%.3f align=%lld "
                "secondary_columns=%lld/%lld max_tokens=%lld weights=%.2f MiB\n",
                ggml_backend_name(secondary_backend), secondary.fraction,
                static_cast<long long>(secondary.align),
                static_cast<long long>(secondary_block_columns),
                static_cast<long long>(cfg.columns),
                static_cast<long long>(secondary.max_tokens),
                static_cast<double>(ggml_backend_buffer_get_size(secondary_weight_buffer)) /
                        (1024.0 * 1024.0));
    }

    if (pixel_coalesce) fprintf(stderr, "S42PIXELCOALESCE full_gemv=6 half_gemv=6 resident_factor=1.5\n");
    if (fused_layer_mask != 0) {
        // projections share the FFN weight buffer type (packed CPU on the Pixel); the F32 norm
        // vectors live in a plain CPU buffer, which every CPU extra buffer type can read from
        const size_t fused_layers = static_cast<size_t>(__builtin_popcountll(fused_layer_mask));
        ggml_init_params fused_params = { ggml_tensor_overhead() * (fused_layers * 4 + 8), nullptr, true };
        ggml_context * projection_ctx = ggml_init(fused_params);
        ggml_context * norm_ctx = ggml_init(fused_params);
        auto new_weight = [](ggml_context * ctx, const host_tensor & tensor) {
            return ggml_new_tensor_2d(ctx, tensor.type, tensor.ne0, tensor.ne1);
        };
        for (layer_state & state : states) {
            if (!state.fused) continue;
            state.wo_weight = new_weight(projection_ctx, state.wo);
            state.ffn_norm_weight = ggml_new_tensor_1d(norm_ctx, GGML_TYPE_F32, n_embd);
            if (state.next_qkv) {
                state.attn_norm_next_weight = ggml_new_tensor_1d(norm_ctx, GGML_TYPE_F32, n_embd);
                state.wq_next_weight = new_weight(projection_ctx, state.wq_next);
                state.wk_next_weight = new_weight(projection_ctx, state.wk_next);
                state.wv_next_weight = new_weight(projection_ctx, state.wv_next);
            }
        }
        ggml_backend_buffer_t projection_buffer = ggml_backend_alloc_ctx_tensors_from_buft(projection_ctx, weight_buft);
        ggml_backend_buffer_t norm_buffer = ggml_backend_alloc_ctx_tensors_from_buft(
                norm_ctx, ggml_backend_get_default_buffer_type(backend));
        if (projection_buffer == nullptr || norm_buffer == nullptr) {
            fprintf(stderr, "[ffn-worker] fused weight allocation failed\n");
            return 1;
        }
        weight_contexts.push_back(projection_ctx);
        weight_contexts.push_back(norm_ctx);
        weight_buffers.push_back(projection_buffer);
        weight_buffers.push_back(norm_buffer);
        weight_buffer_bytes += ggml_backend_buffer_get_size(projection_buffer) +
                ggml_backend_buffer_get_size(norm_buffer);
        for (layer_state & state : states) {
            if (!state.fused) continue;
            const std::pair<ggml_tensor *, host_tensor *> uploads[] = {
                {state.wo_weight, &state.wo}, {state.ffn_norm_weight, &state.ffn_norm},
                {state.attn_norm_next_weight, &state.attn_norm_next}, {state.wq_next_weight, &state.wq_next},
                {state.wk_next_weight, &state.wk_next}, {state.wv_next_weight, &state.wv_next},
            };
            for (const auto & upload : uploads) {
                if (upload.first == nullptr) continue;
                if (ggml_nbytes(upload.first) != upload.second->bytes.size()) {
                    fprintf(stderr, "[ffn-worker] fused layer %d weight allocation mismatch\n", state.layer);
                    return 1;
                }
                ggml_backend_tensor_set(upload.first, upload.second->bytes.data(), 0, upload.second->bytes.size());
                std::vector<uint8_t>().swap(upload.second->bytes);
            }
        }
        ggml_backend_synchronize(backend);
        fprintf(stderr, "S43FUSEDWEIGHTS layers=%016llx next=%016llx n_q=%lld n_kv=%lld rms_eps=%g "
                "bytes=%zu hash=%016llx\n",
                static_cast<unsigned long long>(fused_layer_mask), static_cast<unsigned long long>(fused_next_mask),
                static_cast<long long>(fused_info.n_q), static_cast<long long>(fused_info.n_kv),
                static_cast<double>(fused_info.rms_eps),
                ggml_backend_buffer_get_size(projection_buffer) + ggml_backend_buffer_get_size(norm_buffer),
                static_cast<unsigned long long>(fused_weight_hash));
    }
    print_residency_phase(cfg, "WEIGHT_UPLOAD_READY");
    fprintf(stderr, "[ffn-worker] weight buffers=%s count=%zu\n",
            ggml_backend_buft_name(weight_buft), weight_buffers.size());

    const bool dmabuf_transport = !cfg.ffs_root.empty();
    const bool direct_dmabuf_transport =
            dmabuf_transport && !cfg.staged_dmabuf;
    ggml_backend_buffer_type_t graph_buft =
            ggml_backend_get_default_buffer_type(backend);
    const size_t wire_element_size = cfg.f16_io ?
            sizeof(ggml_fp16_t) : sizeof(float);
    const size_t max_elements = static_cast<size_t>(n_embd) *
            static_cast<size_t>(cfg.max_tokens);
    const size_t max_payload_bytes = max_elements * wire_element_size;
    const size_t graph_arena_count = 1;
    const int64_t max_tokens_per_arena = cfg.max_tokens;
    struct graph_arena {
        ggml_backend_buffer_t input_buffer = nullptr;
        ggml_backend_buffer_t output_buffer = nullptr;
        ggml_context * context = nullptr;
        ggml_gallocr_t allocator = nullptr;
    };
    std::vector<graph_arena> graph_arenas(graph_arena_count);
    if (direct_dmabuf_transport) {
        const size_t alignment = ggml_backend_buft_get_alignment(graph_buft);
        if (alignment > ffn_split::dmabuf_payload_offset ||
            ffn_split::dmabuf_payload_offset % alignment != 0) {
            fprintf(stderr,
                    "[ffn-worker] DMA-BUF wire prefix does not meet backend alignment=%zu\n",
                    alignment);
            return 1;
        }
        for (graph_arena & arena : graph_arenas) {
            arena.input_buffer = ggml_backend_buft_alloc_buffer(
                    graph_buft,
                    ffn_split::dmabuf_payload_offset + max_payload_bytes);
            arena.output_buffer = ggml_backend_buft_alloc_buffer(
                    graph_buft,
                    ffn_split::dmabuf_payload_offset + max_payload_bytes);
            if (arena.input_buffer == nullptr ||
                arena.output_buffer == nullptr) {
                fprintf(stderr,
                        "[ffn-worker] DMA-BUF I/O allocation failed\n");
                return 1;
            }
            ggml_backend_buffer_set_usage(
                    arena.input_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            ggml_backend_buffer_set_usage(
                    arena.output_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        }
    }

    const size_t graph_tensor_capacity = block_count * 16 + 128;
    ggml_init_params graph_params = {
        ggml_tensor_overhead() * graph_tensor_capacity + ggml_graph_overhead(),
        nullptr, true,
    };
    for (graph_arena & arena : graph_arenas) {
        arena.context = ggml_init(graph_params);
        arena.allocator = ggml_gallocr_new(graph_buft);
        if (arena.context == nullptr || arena.allocator == nullptr) {
            fprintf(stderr, "[ffn-worker] graph arena allocation failed\n");
            return 1;
        }
    }
    struct graph_instance {
        std::vector<std::unique_ptr<pixel_packed_task>> packed_tasks;
        ggml_tensor * input = nullptr;
        ggml_tensor * output = nullptr;
        ggml_cgraph * graph = nullptr;
    };
    auto supported_columns = [&](int64_t columns) {
        return std::binary_search(
                runtime_columns.begin(), runtime_columns.end(), columns);
    };
    auto build_graph = [&](
            layer_state & state, int64_t columns, int64_t tokens,
            size_t arena_index, graph_instance & instance,
            bool with_copy = false) {
        if (!supported_columns(columns) || tokens <= 0 ||
            (pixel_gpu_layout.rows && tokens != 1 && tokens != 2 && tokens != 4 && tokens != 8) ||
            tokens > max_tokens_per_arena ||
            arena_index >= graph_arenas.size()) {
            return false;
        }
        graph_arena & arena = graph_arenas[arena_index];
        ggml_context * graph_ctx = arena.context;
        instance.packed_tasks.clear();
        ggml_reset(graph_ctx);
        const ggml_type input_type = cfg.f16_io ?
                GGML_TYPE_F16 : GGML_TYPE_F32;
        ggml_tensor * input_wire = ggml_new_tensor_2d(
                graph_ctx, input_type, n_embd, tokens);
        if (direct_dmabuf_transport &&
            ggml_backend_tensor_alloc(
                    arena.input_buffer, input_wire,
                    static_cast<uint8_t *>(
                            ggml_backend_buffer_get_base(
                                    arena.input_buffer)) +
                            ffn_split::dmabuf_payload_offset) !=
                    GGML_STATUS_SUCCESS) {
            return false;
        }
        ggml_set_input(input_wire);
        ggml_tensor * input = input_type == GGML_TYPE_F16 ?
                ggml_cast(graph_ctx, input_wire, GGML_TYPE_F32) : input_wire;

        const auto range = state.selection(columns);
        const size_t first_block = range.first;
        if (range.first == range.second) return false;

        ggml_tensor * sum = nullptr;
        for (size_t index = first_block; index < range.second; ++index) {
            layer_state::block & block = state.blocks[index];
            if (block.primary_columns() == 0) continue;
            ggml_tensor * gate = pixel_neon_projection(
                    graph_ctx, block.gate_weight, input, pixel_correction, instance.packed_tasks);
            ggml_tensor * up = pixel_neon_projection(
                    graph_ctx, block.up_weight, input, pixel_correction, instance.packed_tasks);
            ggml_tensor * activation = swiglu ?
                    ggml_swiglu_split(graph_ctx, gate, up) :
                    ggml_geglu_split(graph_ctx, gate, up);
            ggml_tensor * partial = pixel_neon_projection(
                    graph_ctx, block.down_weight, activation, pixel_correction, instance.packed_tasks);
            sum = sum == nullptr ? partial : ggml_add(graph_ctx, sum, partial);
            if (with_copy && block.copy_gate_weight != nullptr) {
                ggml_tensor * copy_gate = ggml_mul_mat(
                        graph_ctx, block.copy_gate_weight, input);
                ggml_tensor * copy_up = ggml_mul_mat(
                        graph_ctx, block.copy_up_weight, input);
                ggml_tensor * copy_activation = swiglu ?
                        ggml_swiglu_split(graph_ctx, copy_gate, copy_up) :
                        ggml_geglu_split(graph_ctx, copy_gate, copy_up);
                sum = ggml_add(graph_ctx, sum, ggml_mul_mat(
                        graph_ctx, block.copy_down_weight, copy_activation));
            }
        }
        if (sum == nullptr) {
            instance = graph_instance();
            return dual;
        }
        // dual mode reads the F32 partial and merges it on the CPU
        ggml_tensor * output_wire = cfg.f16_io && !dual ?
                ggml_cast(graph_ctx, sum, GGML_TYPE_F16) : sum;
        if (direct_dmabuf_transport && !dual &&
            ggml_backend_tensor_alloc(
                    arena.output_buffer, output_wire,
                    static_cast<uint8_t *>(
                            ggml_backend_buffer_get_base(
                                    arena.output_buffer)) +
                            ffn_split::dmabuf_payload_offset) !=
                    GGML_STATUS_SUCCESS) {
            return false;
        }
        ggml_set_output(output_wire);
        ggml_cgraph * graph = ggml_new_graph(graph_ctx);
        ggml_build_forward_expand(graph, output_wire);
        graph->uid = 0;
        for (int node_index = 0;
             node_index < ggml_graph_n_nodes(graph); ++node_index) {
            if (!ggml_backend_supports_op(
                        backend, ggml_graph_node(graph, node_index))) {
                return false;
            }
        }
        if (!ggml_gallocr_alloc_graph(arena.allocator, graph)) {
            return false;
        }
        instance.input = input_wire;
        instance.output = output_wire;
        instance.graph = graph;
        return true;
    };

    // WS15 fused call of an owned layer: rows [attention (n_q) | residual (n_embd)] ->
    // [delta (n_embd) (| q (n_q) | k (n_kv) | v (n_kv))], same projection kernels as the FFN
    const int64_t fused_n_q = fused_info.n_q;
    const int64_t fused_n_kv = fused_info.n_kv;
    auto build_fused_graph = [&](
            layer_state & state, int64_t tokens, bool next_qkv, graph_instance & instance) {
        if (!state.fused || (next_qkv && !state.next_qkv) || tokens <= 0 ||
            tokens > max_tokens_per_arena || dual) {
            return false;
        }
        graph_arena & arena = graph_arenas[0];
        ggml_context * graph_ctx = arena.context;
        instance.packed_tasks.clear();
        ggml_reset(graph_ctx);
        const ggml_type input_type = cfg.f16_io ? GGML_TYPE_F16 : GGML_TYPE_F32;
        ggml_tensor * input_wire = ggml_new_tensor_2d(graph_ctx, input_type, fused_n_q + n_embd, tokens);
        ggml_set_input(input_wire);
        ggml_tensor * input = input_type == GGML_TYPE_F16 ?
                ggml_cast(graph_ctx, input_wire, GGML_TYPE_F32) : input_wire;
        ggml_tensor * attention = ggml_cont(graph_ctx, ggml_view_2d(
                graph_ctx, input, fused_n_q, tokens, input->nb[1], 0));
        ggml_tensor * residual = ggml_cont(graph_ctx, ggml_view_2d(
                graph_ctx, input, n_embd, tokens, input->nb[1], fused_n_q * ggml_element_size(input)));
        auto project = [&](ggml_tensor * weights, ggml_tensor * x) {
            return pixel_neon_projection(graph_ctx, weights, x, pixel_correction, instance.packed_tasks);
        };
        ggml_tensor * attention_out = project(state.wo_weight, attention);
        ggml_tensor * ffn_input = ggml_add(graph_ctx, attention_out, residual);
        ggml_tensor * normed = ggml_mul(graph_ctx,
                ggml_rms_norm(graph_ctx, ffn_input, fused_info.rms_eps), state.ffn_norm_weight);
        const auto range = state.selection(cfg.columns);
        if (range.first == range.second) return false;
        ggml_tensor * ffn_out = nullptr;
        for (size_t index = range.first; index < range.second; ++index) {
            layer_state::block & block = state.blocks[index];
            if (block.primary_columns() == 0) continue;
            ggml_tensor * gate = project(block.gate_weight, normed);
            ggml_tensor * up = project(block.up_weight, normed);
            ggml_tensor * activation = swiglu ?
                    ggml_swiglu_split(graph_ctx, gate, up) :
                    ggml_geglu_split(graph_ctx, gate, up);
            ggml_tensor * partial = project(block.down_weight, activation);
            ffn_out = ffn_out == nullptr ? partial : ggml_add(graph_ctx, ffn_out, partial);
        }
        if (ffn_out == nullptr) return false;
        ggml_tensor * delta = ggml_add(graph_ctx, attention_out, ffn_out);
        ggml_tensor * out = delta;
        if (next_qkv) {
            ggml_tensor * layer_out = ggml_add(graph_ctx, residual, delta);
            ggml_tensor * next_input = ggml_mul(graph_ctx,
                    ggml_rms_norm(graph_ctx, layer_out, fused_info.rms_eps), state.attn_norm_next_weight);
            ggml_tensor * q = project(state.wq_next_weight, next_input);
            ggml_tensor * k = project(state.wk_next_weight, next_input);
            ggml_tensor * v = project(state.wv_next_weight, next_input);
            out = ggml_concat(graph_ctx, ggml_concat(graph_ctx, ggml_concat(graph_ctx, delta, q, 0), k, 0), v, 0);
        }
        ggml_tensor * output_wire = cfg.f16_io ? ggml_cast(graph_ctx, out, GGML_TYPE_F16) : out;
        ggml_set_output(output_wire);
        ggml_cgraph * graph = ggml_new_graph(graph_ctx);
        ggml_build_forward_expand(graph, output_wire);
        graph->uid = 0;
        for (int node_index = 0; node_index < ggml_graph_n_nodes(graph); ++node_index) {
            if (!ggml_backend_supports_op(backend, ggml_graph_node(graph, node_index))) {
                return false;
            }
        }
        if (!ggml_gallocr_alloc_graph(arena.allocator, graph)) {
            return false;
        }
        instance.input = input_wire;
        instance.output = output_wire;
        instance.graph = graph;
        return true;
    };
    if (fused_layer_mask != 0 && dual) {
        fprintf(stderr, "[ffn-worker] fused calls do not support a secondary backend\n");
        return 1;
    }

    for (size_t arena_index = 0;
         arena_index < graph_arenas.size(); ++arena_index) {
        graph_instance reserve_graph;
        if (!build_graph(
                    states.front(), cfg.columns, max_tokens_per_arena,
                    arena_index, reserve_graph)) {
            fprintf(stderr,
                    "[ffn-worker] cannot reserve maximum FFN graph\n");
            return 1;
        }
    }
    std::vector<float> input_data(static_cast<size_t>(n_embd), 0.0f);
    std::vector<float> output_data(static_cast<size_t>(n_embd));
    std::vector<ggml_fp16_t> input_data_f16(
            cfg.f16_io ? static_cast<size_t>(n_embd) : 0,
            static_cast<ggml_fp16_t>(0));
    std::vector<ggml_fp16_t> output_data_f16(
            cfg.f16_io ? static_cast<size_t>(n_embd) : 0);
    for (size_t arena_index = 0;
         arena_index < graph_arenas.size(); ++arena_index) {
        for (layer_state & state : states) {
            graph_instance warmup;
            if (!build_graph(
                        state, cfg.columns, 1, arena_index, warmup)) {
                fprintf(stderr,
                        "[ffn-worker] layer %d arena %zu warmup graph failed\n",
                        state.layer, arena_index);
                return 1;
            }
            if (cfg.f16_io) {
                ggml_backend_tensor_set(
                        warmup.input, input_data_f16.data(), 0,
                        input_data_f16.size() * sizeof(ggml_fp16_t));
            } else {
                ggml_backend_tensor_set(
                        warmup.input, input_data.data(), 0,
                        input_data.size() * sizeof(float));
            }
            if (ggml_backend_graph_compute(backend, warmup.graph) !=
                    GGML_STATUS_SUCCESS) {
                fprintf(stderr,
                        "[ffn-worker] layer %d arena %zu warmup failed\n",
                        state.layer, arena_index);
                return 1;
            }
            if (!dmabuf_transport) {
                if (cfg.f16_io) {
                    ggml_backend_tensor_get(
                            warmup.output, output_data_f16.data(), 0,
                            output_data_f16.size() * sizeof(ggml_fp16_t));
                } else {
                    ggml_backend_tensor_get(
                            warmup.output, output_data.data(), 0,
                            output_data.size() * sizeof(float));
                }
            }
        }
    }

    ggml_context * secondary_graph_ctx = nullptr;
    ggml_gallocr_t secondary_allocator = nullptr;
    std::unique_ptr<helper_thread> secondary_thread;
    std::vector<uint8_t> dual_input;
    std::vector<float> dual_secondary_input;
    std::vector<float> dual_primary_output;
    std::vector<float> dual_secondary_output;
    std::vector<uint8_t> dual_output;
    std::vector<double> dual_primary_samples;
    std::vector<double> dual_secondary_samples;
    std::vector<double> dual_total_samples;
    const auto free_secondary = [&]() {
        secondary_thread.reset();
        if (secondary_allocator != nullptr) {
            ggml_gallocr_free(secondary_allocator);
        }
        if (secondary_graph_ctx != nullptr) {
            ggml_free(secondary_graph_ctx);
        }
        if (secondary_weight_buffer != nullptr) {
            ggml_backend_buffer_free(secondary_weight_buffer);
        }
        if (secondary_weight_ctx != nullptr) {
            ggml_free(secondary_weight_ctx);
        }
        if (secondary_backend != nullptr) {
            ggml_backend_free(secondary_backend);
        }
    };
    // graph == nullptr when the selected blocks have no secondary columns
    const auto build_secondary_graph = [&](
            layer_state & state, int64_t columns, int64_t tokens,
            graph_instance & instance) {
        instance = graph_instance();
        if (!supported_columns(columns) || tokens <= 0 ||
            (pixel_gpu_layout.rows && tokens != 1 && tokens != 2 && tokens != 4 && tokens != 8) ||
            tokens > max_tokens_per_arena) {
            return false;
        }
        ggml_reset(secondary_graph_ctx);
        ggml_tensor * input = ggml_new_tensor_2d(
                secondary_graph_ctx, GGML_TYPE_F32, n_embd, tokens);
        ggml_set_input(input);
        const auto range = state.selection(columns);
        const size_t first_block = range.first;
        if (range.first == range.second) return false;
        ggml_tensor * sum = nullptr;
        for (size_t index = first_block; index < range.second; ++index) {
            layer_state::block & block = state.blocks[index];
            if (block.secondary_columns == 0) {
                continue;
            }
            ggml_tensor * gate = ggml_mul_mat(
                    secondary_graph_ctx, block.secondary_gate_weight, input);
            ggml_tensor * up = ggml_mul_mat(
                    secondary_graph_ctx, block.secondary_up_weight, input);
            ggml_tensor * activation = swiglu ?
                    ggml_swiglu_split(secondary_graph_ctx, gate, up) :
                    ggml_geglu_split(secondary_graph_ctx, gate, up);
            ggml_tensor * partial = ggml_mul_mat(
                    secondary_graph_ctx, block.secondary_down_weight, activation);
            sum = sum == nullptr ? partial : ggml_add(secondary_graph_ctx, sum, partial);
        }
        if (sum == nullptr) {
            return true;
        }
        ggml_set_output(sum);
        ggml_cgraph * graph = ggml_new_graph(secondary_graph_ctx);
        ggml_build_forward_expand(graph, sum);
        for (int node_index = 0; node_index < ggml_graph_n_nodes(graph); ++node_index) {
            if (!ggml_backend_supports_op(
                        secondary_backend, ggml_graph_node(graph, node_index))) {
                return false;
            }
        }
        if (!ggml_gallocr_alloc_graph(secondary_allocator, graph)) {
            return false;
        }
        instance.input = input;
        instance.output = sum;
        instance.graph = graph;
        return true;
    };
    const auto run_secondary = [&](
            layer_state & state, int64_t columns, int64_t tokens) {
        const size_t elements = static_cast<size_t>(n_embd * tokens);
        graph_instance instance;
        if (!build_secondary_graph(state, columns, tokens, instance)) {
            return false;
        }
        dual_secondary_output.assign(elements, 0.0f);
        if (instance.graph == nullptr) {
            return true;
        }
        dual_secondary_input.resize(elements);
        if (cfg.f16_io) {
            ggml_fp16_to_fp32_row(
                    reinterpret_cast<const ggml_fp16_t *>(dual_input.data()),
                    dual_secondary_input.data(), static_cast<int64_t>(elements));
        } else {
            memcpy(dual_secondary_input.data(), dual_input.data(),
                   elements * sizeof(float));
        }
        ggml_backend_tensor_set(
                instance.input, dual_secondary_input.data(), 0,
                elements * sizeof(float));
        if (ggml_backend_graph_compute(secondary_backend, instance.graph) !=
                GGML_STATUS_SUCCESS) {
            return false;
        }
        ggml_backend_tensor_get(
                instance.output, dual_secondary_output.data(), 0,
                elements * sizeof(float));
        return true;
    };
    // Computes one request on both backends and writes the merged result in
    // wire format to dual_output.
    const auto run_dual = [&](
            layer_state & state, uint32_t request_id, int64_t columns,
            int64_t tokens, const void * input_payload, size_t payload_bytes,
            uint64_t & compute_us) {
        const uint64_t started = now_us();
        const size_t elements = static_cast<size_t>(n_embd * tokens);
        const bool split = secondary.max_tokens == 0 || tokens <= secondary.max_tokens;
        dual_input.assign(
                static_cast<const uint8_t *>(input_payload),
                static_cast<const uint8_t *>(input_payload) + payload_bytes);
        bool secondary_ok = false;
        uint64_t secondary_us = 0;
        uint64_t secondary_started = 0;
        uint64_t secondary_done = 0;
        if (split) {
            secondary_thread->start([&]() {
                secondary_started = now_us();
                secondary_ok = run_secondary(state, columns, tokens);
                secondary_done = now_us();
                secondary_us = secondary_done - secondary_started;
            });
        } else {
            secondary_ok = true;
            dual_secondary_output.assign(elements, 0.0f);
        }
        const uint64_t primary_started = now_us();
        graph_instance execution;
        bool primary_ok = build_graph(state, columns, tokens, 0, execution, !split);
        if (primary_ok && execution.graph != nullptr) {
            ggml_backend_tensor_set(
                    execution.input, dual_input.data(), 0, payload_bytes);
            primary_ok = ggml_backend_graph_compute(backend, execution.graph) ==
                    GGML_STATUS_SUCCESS;
        }
        dual_primary_output.assign(elements, 0.0f);
        if (primary_ok && execution.graph != nullptr) {
            ggml_backend_tensor_get(
                    execution.output, dual_primary_output.data(), 0,
                    elements * sizeof(float));
        }
        const uint64_t primary_done = now_us();
        if (split) {
            secondary_thread->wait();
        }
        const uint64_t joined = now_us();
        if (!primary_ok || !secondary_ok) {
            fprintf(stderr, "[ffn-worker] dual execution failed primary=%d secondary=%d\n",
                    primary_ok ? 1 : 0, secondary_ok ? 1 : 0);
            return false;
        }
        for (size_t i = 0; i < elements; ++i) {
            dual_primary_output[i] += dual_secondary_output[i];
        }
        dual_output.resize(payload_bytes);
        if (cfg.f16_io) {
            ggml_fp32_to_fp16_row(
                    dual_primary_output.data(),
                    reinterpret_cast<ggml_fp16_t *>(dual_output.data()),
                    static_cast<int64_t>(elements));
        } else {
            memcpy(dual_output.data(), dual_primary_output.data(), payload_bytes);
        }
        const uint64_t finished = now_us();
        compute_us = finished - started;
        int64_t primary_columns = 0;
        int64_t secondary_columns = 0;
        const auto range = state.selection(columns);
        for (size_t index = range.first; index < range.second; ++index) {
            const layer_state::block & block = state.blocks[index];
            primary_columns += split ? block.primary_columns() : block.columns;
            secondary_columns += split ? block.secondary_columns : 0;
        }
        dual_primary_samples.push_back(static_cast<double>(primary_done - started));
        dual_secondary_samples.push_back(static_cast<double>(secondary_us));
        dual_total_samples.push_back(static_cast<double>(compute_us));
        const size_t calls = dual_total_samples.size();
        if (secondary.log_period > 0 &&
            calls % static_cast<size_t>(secondary.log_period) == 0) {
            fprintf(stderr,
                    "S43DUALFFN request=%u layer=%d tokens=%lld columns=%lld "
                    "primary_columns=%lld secondary_columns=%lld "
                    "primary_us=%llu secondary_us=%llu wait_us=%llu "
                    "merge_us=%llu total_us=%llu primary_start_us=%llu "
                    "secondary_start_us=%llu secondary_end_us=%llu overlap_us=%llu\n",
                    request_id, state.layer, static_cast<long long>(tokens),
                    static_cast<long long>(columns),
                    static_cast<long long>(primary_columns),
                    static_cast<long long>(secondary_columns),
                    static_cast<unsigned long long>(primary_done - started),
                    static_cast<unsigned long long>(secondary_us),
                    static_cast<unsigned long long>(joined - primary_done),
                    static_cast<unsigned long long>(finished - joined),

                    static_cast<unsigned long long>(compute_us),
                    static_cast<unsigned long long>(primary_started - started),
                    static_cast<unsigned long long>(secondary_started ? secondary_started - started : 0),
                    static_cast<unsigned long long>(secondary_done ? secondary_done - started : 0),
                    static_cast<unsigned long long>(secondary_done > primary_started && primary_done > secondary_started ?
                        std::min(primary_done, secondary_done) - std::max(primary_started, secondary_started) : 0));
        }
        if (calls % 32 == 0) {
            const auto p50 = [](std::vector<double> samples) {
                std::sort(samples.begin(), samples.end());
                return samples[samples.size() / 2];
            };
            fprintf(stderr,
                    "[ffn-worker] dual requests=%zu primary_p50_us=%.0f "
                    "secondary_p50_us=%.0f total_p50_us=%.0f\n",
                    calls, p50(dual_primary_samples), p50(dual_secondary_samples),
                    p50(dual_total_samples));
            fflush(stderr);
        }
        return true;
    };
    if (dual) {
        secondary_graph_ctx = ggml_init(graph_params);
        secondary_allocator = ggml_gallocr_new(
                ggml_backend_get_default_buffer_type(secondary_backend));
        if (secondary_graph_ctx == nullptr || secondary_allocator == nullptr) {
            fprintf(stderr, "[ffn-worker] secondary graph arena allocation failed\n");
            return 1;
        }
        graph_instance reserve_graph;
        if (secondary.max_tokens > 0 &&
            !build_graph(states.front(), cfg.columns, max_tokens_per_arena, 0,
                         reserve_graph, true)) {
            fprintf(stderr, "[ffn-worker] cannot reserve NPU-only fallback FFN graph\n");
            return 1;
        }
        if (!build_secondary_graph(
                    states.front(), cfg.columns, max_tokens_per_arena, reserve_graph)) {
            fprintf(stderr, "[ffn-worker] cannot reserve maximum secondary FFN graph\n");
            return 1;
        }
        dual_input.assign(max_payload_bytes, 0);
        for (layer_state & state : states) {
            if (!run_secondary(state, cfg.columns, 1)) {
                fprintf(stderr, "[ffn-worker] layer %d secondary warmup failed\n",
                        state.layer);
                return 1;
            }
        }
        secondary_thread = std::make_unique<helper_thread>();
    }

#if defined(__ANDROID__)
    if (dmabuf_transport) {
        const size_t max_wire_bytes =
                ffn_split::dmabuf_payload_offset + max_payload_bytes;
        mapped_dmabuf staged_input;
        mapped_dmabuf staged_output;
        int input_fd = -1;
        int output_fd = -1;
        uint8_t * input_base = nullptr;
        uint8_t * output_base = nullptr;
        struct direct_io_slot {
            std::unique_ptr<mapped_dmabuf> input_buffer;
            std::unique_ptr<mapped_dmabuf> output_buffer;
            int input_fd = -1;
            int output_fd = -1;
            uint8_t * input_base = nullptr;
            uint8_t * output_base = nullptr;
            bool output_pending = false;
            uint32_t output_request_id = 0;
            int32_t output_layer = -1;
            uint32_t output_tokens = 0;
            uint32_t output_columns = 0;
            size_t output_wire_bytes = 0;
            uint64_t output_queued_us = 0;
        };
        std::vector<direct_io_slot> direct_slots;
        if (cfg.staged_dmabuf) {
            input_fd = -1;
            output_fd = -1;
        } else {
            direct_slots.resize(static_cast<size_t>(cfg.queue_depth));
            for (size_t index = 0; index < direct_slots.size(); ++index) {
                direct_io_slot & slot = direct_slots[index];
                slot.input_buffer = std::make_unique<mapped_dmabuf>();
                slot.output_buffer = std::make_unique<mapped_dmabuf>();
                if (!slot.input_buffer->allocate(max_wire_bytes) ||
                    !slot.output_buffer->allocate(max_wire_bytes)) {
                    fprintf(stderr,
                            "[ffn-worker] persistent DMA-BUF ring allocation failed\n");
                    return 1;
                }
                slot.input_fd = slot.input_buffer->fd;
                slot.output_fd = slot.output_buffer->fd;
                slot.input_base = slot.input_buffer->data;
                slot.output_base = slot.output_buffer->data;
            }
            input_fd = direct_slots.front().input_fd;
            output_fd = direct_slots.front().output_fd;
            input_base = direct_slots.front().input_base;
            output_base = direct_slots.front().output_base;
        }

        const int ep0 = ffn_split::functionfs_open_endpoint(cfg.ffs_root, "ep0");
        if (ep0 < 0 ||
            !ffn_split::functionfs_write_exact(
                    ep0, &ffn_split::functionfs_descriptors,
                    sizeof(ffn_split::functionfs_descriptors)) ||
            !ffn_split::functionfs_write_exact(
                    ep0, &ffn_split::functionfs_strings,
                    sizeof(ffn_split::functionfs_strings))) {
            fprintf(stderr, "[ffn-worker] FunctionFS descriptor setup failed: %s\n",
                    strerror(errno));
            return 1;
        }
        const int device_to_host =
                ffn_split::functionfs_open_endpoint(cfg.ffs_root, "ep1");
        const int host_to_device =
                ffn_split::functionfs_open_endpoint(cfg.ffs_root, "ep2");
        if (device_to_host < 0 || host_to_device < 0 ||
            !ffn_split::functionfs_touch(
                    cfg.ready_file, "descriptors_ready\n")) {
            fprintf(stderr, "[ffn-worker] FunctionFS endpoint setup failed: %s\n",
                    strerror(errno));
            return 1;
        }
        fprintf(stderr,
                "[ffn-worker] descriptors_ready backend=%s layers=%zu "
                "mask=%016llx K=%lld NFF=%lld slice=[%lld,%lld) type=%s "
                "io=%s activation=%s max_tokens=%lld quantum=%lld alternate=%lld blocks=%zu "
                "transport=%s weights=%.2f MiB hash=%016llx "
                "queue_depth=%lld input_fd=%d output_fd=%d\n",
                ggml_backend_dev_description(device), states.size(),
                static_cast<unsigned long long>(cfg.layer_mask),
                static_cast<long long>(n_embd), static_cast<long long>(n_ff),
                static_cast<long long>(offset),
                static_cast<long long>(offset + cfg.columns),
                ggml_type_name(weight_type),
                cfg.f16_io ? "f16" : "f32",
                swiglu ? "swiglu" : "geglu",
                static_cast<long long>(cfg.max_tokens),
                static_cast<long long>(cfg.column_quantum),
                static_cast<long long>(cfg.alternate_columns), block_count,
                cfg.staged_dmabuf ? "staged" : "direct",
                static_cast<double>(weight_buffer_bytes) / (1024.0 * 1024.0),
                static_cast<unsigned long long>(weight_hash),
                static_cast<long long>(cfg.queue_depth), input_fd, output_fd);
        fflush(stderr);
        ffn_split::functionfs_event_monitor event_monitor(ep0);
        uint64_t session_generation = 0;
        if (!event_monitor.wait_enabled(
                    session_generation,
                    ffn_split::functionfs_enable_timeout_ms)) {
            fprintf(stderr, "[ffn-worker] FunctionFS enable failed: %s\n",
                    strerror(errno));
            return 1;
        }

        bool input_attached = false;
        bool output_attached = false;
        if (!cfg.staged_dmabuf) {
            size_t attached = 0;
            for (; attached < direct_slots.size(); ++attached) {
                const direct_io_slot & slot = direct_slots[attached];
                if (!ffn_split::functionfs_dmabuf_attach(
                            host_to_device, slot.input_fd) ||
                    !ffn_split::functionfs_dmabuf_attach(
                            device_to_host, slot.output_fd)) {
                    break;
                }
            }
            if (attached != direct_slots.size()) {
                fprintf(stderr,
                        "[ffn-worker] FunctionFS DMA-BUF ring attach failed: %s\n",
                        strerror(errno));
                for (size_t index = 0; index <= attached &&
                     index < direct_slots.size(); ++index) {
                    ffn_split::functionfs_dmabuf_detach(
                            host_to_device, direct_slots[index].input_fd);
                    ffn_split::functionfs_dmabuf_detach(
                            device_to_host, direct_slots[index].output_fd);
                }
                return 1;
            }
            input_attached = true;
            output_attached = true;
        }

        int run_status = 0;
        long served = 0;
        long reset_recoveries = 0;
        size_t maximum_pending_outputs = 0;
        size_t d2h_completions = 0;
        uint64_t d2h_queue_us = 0;
        uint64_t d2h_queue_max_us = 0;
        std::vector<double> compute_samples;
        std::vector<uint8_t> staged_input_payload(
                cfg.staged_dmabuf ? max_payload_bytes : 0);
        bool complete = false;
        bool first_session = true;
        uint64_t previous_generation = session_generation;
        while (run_status == 0 && !complete) {
            if (!first_session && !event_monitor.wait_reenabled(
                        previous_generation, session_generation,
                        ffn_split::functionfs_enable_timeout_ms)) {
                fprintf(stderr,
                        "[ffn-worker] FunctionFS re-enable failed: %s\n",
                        strerror(errno));
                run_status = 1;
                break;
            }
            first_session = false;

            ffn_split::hello_request hello = {};
            if (!ffn_split::functionfs_read_exact(
                        host_to_device, &hello, sizeof(hello))) {
                const int hello_error = errno;
                previous_generation = session_generation;
                ++reset_recoveries;
                fprintf(stderr,
                        "[ffn-worker] USB reset during HELLO read recovery=%ld "
                        "error=%s\n",
                        reset_recoveries, strerror(hello_error));
                if (reset_recoveries > 8) {
                    run_status = 1;
                }
                continue;
            }
            const bool hello_ok =
                    hello.magic == ffn_split::protocol_magic &&
                    hello.version == ffn_split::protocol_version &&
                    hello.message == static_cast<uint16_t>(
                            ffn_split::message_type::hello_request) &&
                    hello.layer_mask == cfg.layer_mask &&
                    hello.max_columns == cfg.columns &&
                    hello.n_embd == n_embd &&
                    hello.max_tokens == cfg.max_tokens &&
                    ffn_split::same_artifact_sha256(
                            hello.artifact_sha256, artifact_sha256) &&
                    hello.flags ==
                            ((cfg.f16_io ? ffn_split::flag_f16_io : 0) |
                             (swiglu ? ffn_split::flag_swiglu : 0));
            ffn_split::hello_response hello_response = {};
            hello_response.magic = ffn_split::protocol_magic;
            hello_response.version = ffn_split::protocol_version;
            hello_response.message = static_cast<uint16_t>(
                    ffn_split::message_type::hello_response);
            hello_response.status = hello_ok ? 0 : 1;
            hello_response.flags =
                    (cfg.f16_io ? ffn_split::flag_f16_io : 0) |
                    (swiglu ? ffn_split::flag_swiglu : 0);
            hello_response.n_embd = static_cast<uint32_t>(n_embd);
            hello_response.n_ff = static_cast<uint32_t>(n_ff);
            hello_response.offset = static_cast<uint32_t>(offset);
            hello_response.max_columns = static_cast<uint32_t>(cfg.columns);
            hello_response.weight_type = static_cast<uint32_t>(weight_type);
            hello_response.layer_count = static_cast<uint32_t>(states.size());
            hello_response.layer_mask = cfg.layer_mask;
            hello_response.weight_hash = weight_hash;
            hello_response.column_quantum =
                    static_cast<uint32_t>(cfg.column_quantum);
            hello_response.max_tokens = static_cast<uint16_t>(cfg.max_tokens);
            hello_response.alternate_columns_32 = static_cast<uint16_t>(
                    cfg.alternate_columns / 32);
            memcpy(
                    hello_response.artifact_sha256, artifact_sha256,
                    sizeof(artifact_sha256));
            if (!ffn_split::functionfs_write_exact(
                        device_to_host, &hello_response,
                        sizeof(hello_response))) {
                const int hello_error = errno;
                previous_generation = session_generation;
                ++reset_recoveries;
                fprintf(stderr,
                        "[ffn-worker] USB reset during HELLO write recovery=%ld "
                        "error=%s\n",
                        reset_recoveries, strerror(hello_error));
                if (reset_recoveries > 8) {
                    run_status = 1;
                }
                continue;
            }
            if (!hello_ok) {
                fprintf(stderr,
                        "[ffn-worker] DMA-BUF HELLO identity mismatch\n");
                run_status = 1;
                break;
            }

            fprintf(stderr,
                    "[ffn-worker] DMA-BUF host connected generation=%llu "
                    "max_wire_bytes=%zu\n",
                    static_cast<unsigned long long>(session_generation),
                    max_wire_bytes);
            fflush(stderr);

            bool recover_session = false;
            const auto trace_first_request = [&](const char * stage) {
                if (served == 0) {
                    fprintf(stderr, "[ffn-worker] first_request=%s\n", stage);
                    fflush(stderr);
                    fsync(STDERR_FILENO);
                }
            };
            const auto recover = [&](const char * stage) {
                previous_generation = session_generation;
                ++reset_recoveries;
                recover_session = true;
                fprintf(stderr,
                        "[ffn-worker] USB reset during %s recovery=%ld\n",
                        stage, reset_recoveries);
                fflush(stderr);
                if (reset_recoveries > 8) {
                    run_status = 1;
                }
            };
            if (!cfg.staged_dmabuf) {
                for (direct_io_slot & slot : direct_slots) {
                    slot.output_pending = false;
                    if (!event_monitor.current(session_generation) ||
                        !ffn_split::functionfs_dmabuf_queue(
                                host_to_device, slot.input_fd,
                                max_wire_bytes)) {
                        recover("input ring queue");
                        break;
                    }
                }
            }
            while (run_status == 0 && !recover_session && !complete) {
                ffn_split::execute_request request = {};
                bool input_sync_started = false;
                size_t direct_slot_index = 0;
                if (cfg.staged_dmabuf) {
                    if (!ffn_split::functionfs_read_exact(
                                host_to_device, &request, sizeof(request))) {
                        recover("input header");
                        break;
                    }
                    trace_first_request("header_received");
                } else {
                    direct_slot_index = static_cast<size_t>(served) %
                            direct_slots.size();
                    direct_io_slot & slot = direct_slots[direct_slot_index];
                    input_fd = slot.input_fd;
                    output_fd = slot.output_fd;
                    input_base = slot.input_base;
                    output_base = slot.output_base;
                    if (slot.output_pending) {
                        if (!ffn_split::functionfs_dmabuf_wait(
                                    slot.output_fd)) {
                            recover("output ring reuse");
                            break;
                        }
                        const uint64_t completed_us = now_us();
                        const uint64_t queued_us =
                                completed_us - slot.output_queued_us;
                        ++d2h_completions;
                        d2h_queue_us += queued_us;
                        d2h_queue_max_us = std::max(
                                d2h_queue_max_us, queued_us);
                        fprintf(stderr,
                                "S41PHONEFFNUSB request=%u layer=%d "
                                "tokens=%u columns=%u d2h_bytes=%zu "
                                "d2h_queued_us=%llu "
                                "d2h_completed_us=%llu d2h_queue_us=%llu\n",
                                slot.output_request_id, slot.output_layer,
                                slot.output_tokens, slot.output_columns,
                                slot.output_wire_bytes,
                                static_cast<unsigned long long>(
                                        slot.output_queued_us),
                                static_cast<unsigned long long>(completed_us),
                                static_cast<unsigned long long>(queued_us));
                        fflush(stderr);
                        slot.output_pending = false;
                    }
                    if (!ffn_split::functionfs_dmabuf_cpu_start(input_fd)) {
                        recover("input wait");
                        break;
                    }
                    input_sync_started = true;
                    memcpy(&request, input_base, sizeof(request));
                }
                const bool shutdown =
                        request.magic == ffn_split::protocol_magic &&
                        request.version == ffn_split::protocol_version &&
                        request.message == static_cast<uint16_t>(
                                ffn_split::message_type::execute_request) &&
                        request.request_id == 0;
                const size_t payload_bytes = request.payload_bytes;
                const size_t wire_bytes =
                        ffn_split::dmabuf_payload_offset + payload_bytes;
                const bool request_header_ok =
                        request.magic == ffn_split::protocol_magic &&
                        request.version == ffn_split::protocol_version &&
                        request.message == static_cast<uint16_t>(
                                ffn_split::message_type::execute_request) &&
                        request.request_id != 0 && request.layer >= 0 &&
                        request.layer < 64 &&
                        (cfg.layer_mask &
                         (UINT64_C(1) << request.layer)) != 0 &&
                        request.tokens > 0 &&
                        request.tokens <= max_tokens_per_arena &&
                        request.elements ==
                                static_cast<uint64_t>(n_embd) *
                                        request.tokens &&
                        request.payload_bytes ==
                                static_cast<uint64_t>(request.elements) *
                                        wire_element_size &&
                        wire_bytes <= max_wire_bytes &&
                        supported_columns(request.columns);
                if (cfg.staged_dmabuf && !shutdown) {
                    if (!request_header_ok) {
                        fprintf(stderr,
                                "[ffn-worker] invalid DMA-BUF execute header\n");
                        run_status = 1;
                        break;
                    }
                    if (output_attached) {
                        if (!ffn_split::functionfs_dmabuf_wait(output_fd) ||
                            !ffn_split::functionfs_dmabuf_detach(
                                    device_to_host, output_fd)) {
                            recover("previous output");
                            break;
                        }
                        output_attached = false;
                    }
                    const auto prepare_staged_buffer = [](
                            mapped_dmabuf & buffer, size_t bytes) {
                        if (buffer.size == bytes) {
                            return true;
                        }
                        if (!buffer.allocate(bytes) ||
                            !ffn_split::functionfs_dmabuf_cpu_start(
                                    buffer.fd)) {
                            return false;
                        }
                        memset(buffer.data, 0, buffer.size);
                        return ffn_split::functionfs_dmabuf_cpu_end(buffer.fd);
                    };
                    if (!prepare_staged_buffer(
                                staged_input, payload_bytes) ||
                        !prepare_staged_buffer(staged_output, wire_bytes)) {
                        fprintf(stderr,
                                "[ffn-worker] staged DMA-BUF resize failed: %s\n",
                                strerror(errno));
                        run_status = 1;
                        break;
                    }
                    input_fd = staged_input.fd;
                    output_fd = staged_output.fd;
                    input_base = staged_input.data;
                    output_base = staged_output.data;
                    trace_first_request("buffers_ready");
                    input_attached = ffn_split::functionfs_dmabuf_attach(
                            host_to_device, input_fd);
                    if (!input_attached) {
                        recover("input attach");
                        break;
                    }
                    trace_first_request("input_attached");
                    if (!event_monitor.current(session_generation) ||
                        !ffn_split::functionfs_dmabuf_queue(
                                host_to_device, input_fd, payload_bytes)) {
                        recover("input payload");
                        break;
                    }
                    trace_first_request("payload_queued");
                    const uint32_t payload_ready =
                            ffn_split::dmabuf_payload_ready_magic ^
                            request.request_id;
                    if (!ffn_split::functionfs_write_exact(
                                device_to_host, &payload_ready,
                                sizeof(payload_ready))) {
                        recover("input ready");
                        break;
                    }
                    trace_first_request("payload_ready_sent");
                    if (!ffn_split::functionfs_dmabuf_cpu_start(input_fd)) {
                        recover("input payload wait");
                        break;
                    }
                    trace_first_request("payload_ready");
                    input_sync_started = true;
                }
                const void * input_payload = input_base +
                        (cfg.staged_dmabuf ? 0 :
                         ffn_split::dmabuf_payload_offset);
                const bool request_ok = shutdown ||
                        (request_header_ok &&
                         request.payload_hash == ffn_split::hash_bytes(
                                 input_payload, payload_bytes) &&
                         (cfg.f16_io ?
                                  all_finite(
                                          static_cast<const ggml_fp16_t *>(
                                                  input_payload),
                                          request.elements) :
                                  all_finite(
                                          static_cast<const float *>(
                                                  input_payload),
                                          request.elements)));
                if (request_ok && cfg.staged_dmabuf) {
                    memcpy(
                            staged_input_payload.data(), input_payload,
                            payload_bytes);
                    trace_first_request("payload_copied");
                }
                if (input_sync_started &&
                    !ffn_split::functionfs_dmabuf_cpu_end(input_fd)) {
                    recover("input sync");
                    break;
                }
                trace_first_request("input_sync_done");
                if (cfg.staged_dmabuf && input_attached) {
                    if (!ffn_split::functionfs_dmabuf_detach(
                                host_to_device, input_fd)) {
                        recover("input detach");
                        break;
                    }
                    input_attached = false;
                    trace_first_request("input_detached");
                }
                if (!event_monitor.current(session_generation)) {
                    recover("input completion");
                    break;
                }
                if (shutdown) {
                    fprintf(stderr, "[ffn-worker] DMA-BUF host disconnected\n");
                    complete = true;
                    break;
                }
                if (!request_ok) {
                    fprintf(stderr,
                            "[ffn-worker] invalid DMA-BUF execute request\n");
                    run_status = 1;
                    break;
                }

                const auto state_it = std::lower_bound(
                        states.begin(), states.end(), request.layer,
                        [](const layer_state & state, int layer) {
                            return state.layer < layer;
                        });
                if (state_it == states.end() ||
                    state_it->layer != request.layer) {
                    fprintf(stderr, "[ffn-worker] layer lookup failed\n");
                    run_status = 1;
                    break;
                }
                layer_state & state = *state_it;
                graph_instance execution;
                ggml_status status = GGML_STATUS_SUCCESS;
                uint64_t compute_us = 0;
                if (dual) {
                    if (!run_dual(
                                state, request.request_id, request.columns,
                                request.tokens, input_payload, payload_bytes,
                                compute_us)) {
                        status = GGML_STATUS_FAILED;
                    }
                    trace_first_request("compute_done");
                } else {
                    if (!build_graph(
                                state, request.columns, request.tokens, 0,
                                execution)) {
                        fprintf(stderr,
                                "[ffn-worker] dynamic FFN graph build failed\n");
                        run_status = 1;
                        break;
                    }
                    trace_first_request("graph_built");
                    if (dmabuf_transport) {
                        ggml_backend_tensor_set(
                                execution.input, input_payload, 0,
                                payload_bytes);
                        trace_first_request("input_staged");
                    }
                    const uint64_t started = now_us();
                    status = ggml_backend_graph_compute_async(backend, execution.graph);
                    ggml_backend_synchronize(backend);
                    trace_first_request("compute_done");
                    compute_us = now_us() - started;
                }
                if (status != GGML_STATUS_SUCCESS) {
                    fprintf(stderr, "[ffn-worker] graph execution failed\n");
                    run_status = 1;
                    break;
                }
                if (!event_monitor.current(session_generation)) {
                    recover("graph execution");
                    break;
                }
                if (!ffn_split::functionfs_dmabuf_cpu_start(output_fd)) {
                    recover("output sync start");
                    break;
                }

                void * output_payload =
                        output_base + ffn_split::dmabuf_payload_offset;
                if (dual) {
                    memcpy(output_payload, dual_output.data(), payload_bytes);
                    trace_first_request("output_staged");
                } else if (dmabuf_transport) {
                    ggml_backend_tensor_get(
                            execution.output, output_payload, 0,
                            payload_bytes);
                    trace_first_request("output_staged");
                }

                const bool output_ok = cfg.f16_io ?
                        all_finite(
                                static_cast<const ggml_fp16_t *>(
                                        output_payload),
                                request.elements) :
                        all_finite(
                                static_cast<const float *>(
                                        output_payload),
                                request.elements);
                ffn_split::execute_response response = {};
                response.magic = ffn_split::protocol_magic;
                response.version = ffn_split::protocol_version;
                response.message = static_cast<uint16_t>(
                        ffn_split::message_type::execute_response);
                response.status = output_ok ? 0 : 1;
                response.request_id = request.request_id;
                response.layer = request.layer;
                response.elements = request.elements;
                response.payload_bytes = static_cast<uint32_t>(payload_bytes);
                response.payload_hash = ffn_split::hash_bytes(
                        output_payload, payload_bytes);
                response.columns = request.columns;
                response.tokens = request.tokens;
                response.compute_us = compute_us;
                memset(output_base, 0, ffn_split::dmabuf_payload_offset);
                memcpy(output_base, &response, sizeof(response));
                if (!ffn_split::functionfs_dmabuf_cpu_end(output_fd)) {
                    recover("output sync end");
                    break;
                }
                trace_first_request("output_sync_done");
                if (!output_ok) {
                    fprintf(stderr, "[ffn-worker] non-finite output\n");
                    run_status = 1;
                    break;
                }
                if (cfg.staged_dmabuf) {
                    output_attached = output_attached ||
                            ffn_split::functionfs_dmabuf_attach(
                                    device_to_host, output_fd);
                    if (!output_attached) {
                        recover("output attach");
                        break;
                    }
                    trace_first_request("output_attached");
                }
                if (!event_monitor.current(session_generation) ||
                    !ffn_split::functionfs_dmabuf_queue(
                            device_to_host, output_fd, wire_bytes) ||
                    !event_monitor.current(session_generation)) {
                    recover("output transfer");
                    break;
                }
                if (!cfg.staged_dmabuf) {
                    direct_io_slot & output_slot =
                            direct_slots[direct_slot_index];
                    output_slot.output_pending = true;
                    output_slot.output_request_id = request.request_id;
                    output_slot.output_layer = request.layer;
                    output_slot.output_tokens = request.tokens;
                    output_slot.output_columns = request.columns;
                    output_slot.output_wire_bytes = wire_bytes;
                    output_slot.output_queued_us = now_us();
                    const size_t pending_outputs = static_cast<size_t>(
                            std::count_if(
                                    direct_slots.begin(), direct_slots.end(),
                                    [](const direct_io_slot & slot) {
                                        return slot.output_pending;
                                    }));
                    maximum_pending_outputs = std::max(
                            maximum_pending_outputs, pending_outputs);
                }
                trace_first_request("output_queued");
                compute_samples.push_back(static_cast<double>(compute_us));
                ++served;
                if (served % 32 == 0) {
                    std::vector<double> sorted = compute_samples;
                    std::sort(sorted.begin(), sorted.end());
                    fprintf(stderr,
                            "[ffn-worker] requests=%ld compute_p50_us=%.0f\n",
                            served, sorted[sorted.size() / 2]);
                    fflush(stderr);
                }
                if (cfg.max_requests > 0 && served >= cfg.max_requests) {
                    complete = true;
                    break;
                }
                if (!cfg.staged_dmabuf &&
                    !ffn_split::functionfs_dmabuf_queue(
                            host_to_device,
                            direct_slots[direct_slot_index].input_fd,
                            max_wire_bytes)) {
                    recover("input requeue");
                    break;
                }
            }
            if (recover_session) {
                if (cfg.staged_dmabuf) {
                    if (input_attached) {
                        ffn_split::functionfs_dmabuf_detach(
                                host_to_device, input_fd);
                        input_attached = false;
                    }
                    if (output_attached) {
                        ffn_split::functionfs_dmabuf_detach(
                                device_to_host, output_fd);
                        output_attached = false;
                    }
                }
                continue;
            }
        }

        if (!cfg.staged_dmabuf) {
            for (direct_io_slot & slot : direct_slots) {
                if (slot.output_pending &&
                    !ffn_split::functionfs_dmabuf_wait(slot.output_fd)) {
                    fprintf(stderr,
                            "[ffn-worker] output ring drain failed: %s\n",
                            strerror(errno));
                    run_status = 1;
                } else if (slot.output_pending) {
                    const uint64_t completed_us = now_us();
                    const uint64_t queued_us =
                            completed_us - slot.output_queued_us;
                    ++d2h_completions;
                    d2h_queue_us += queued_us;
                    d2h_queue_max_us = std::max(
                            d2h_queue_max_us, queued_us);
                    fprintf(stderr,
                            "S41PHONEFFNUSB request=%u layer=%d tokens=%u "
                            "columns=%u d2h_bytes=%zu d2h_queued_us=%llu "
                            "d2h_completed_us=%llu d2h_queue_us=%llu\n",
                            slot.output_request_id, slot.output_layer,
                            slot.output_tokens, slot.output_columns,
                            slot.output_wire_bytes,
                            static_cast<unsigned long long>(
                                    slot.output_queued_us),
                            static_cast<unsigned long long>(completed_us),
                            static_cast<unsigned long long>(queued_us));
                }
                slot.output_pending = false;
            }
        }
        event_monitor.stop();
        if (!cfg.staged_dmabuf) {
            for (const direct_io_slot & slot : direct_slots) {
                ffn_split::functionfs_dmabuf_detach(
                        host_to_device, slot.input_fd);
                ffn_split::functionfs_dmabuf_detach(
                        device_to_host, slot.output_fd);
            }
        } else {
            if (input_attached) {
                ffn_split::functionfs_dmabuf_detach(
                        host_to_device, input_fd);
            }
            if (output_attached) {
                ffn_split::functionfs_dmabuf_detach(
                        device_to_host, output_fd);
            }
        }
        close(host_to_device);
        close(device_to_host);
        close(ep0);
        fprintf(stderr,
                "[ffn-worker] DMA-BUF complete transport=%s requests=%ld "
                "queue_depth=%lld maximum_pending_outputs=%zu "
                "phone_payload_copies=%ld "
                "d2h_completions=%zu d2h_queue_us=%llu "
                "d2h_queue_max_us=%llu "
                "recoveries=%ld status=%d\n",
                cfg.staged_dmabuf ? "staged" : "direct", served,
                static_cast<long long>(cfg.queue_depth),
                maximum_pending_outputs, served * 2, d2h_completions,
                static_cast<unsigned long long>(d2h_queue_us),
                static_cast<unsigned long long>(d2h_queue_max_us),
                reset_recoveries, run_status);
        for (graph_arena & arena : graph_arenas) {
            ggml_gallocr_free(arena.allocator);
            if (arena.output_buffer != nullptr) {
                ggml_backend_buffer_free(arena.output_buffer);
            }
            if (arena.input_buffer != nullptr) {
                ggml_backend_buffer_free(arena.input_buffer);
            }
            ggml_free(arena.context);
        }
        for (ggml_backend_buffer_t weight_buffer : weight_buffers) {
            ggml_backend_buffer_free(weight_buffer);
        }
        for (ggml_context * weight_ctx : weight_contexts) {
            ggml_free(weight_ctx);
        }
        free_secondary();
        ggml_backend_free(backend);
        return run_status;
    }
#endif

    print_residency_phase(cfg, "ENDPOINT_SETUP_BEGIN");
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    const int one = 1;
    sockaddr_in address = {};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<uint16_t>(cfg.port));
    if (listen_fd < 0 ||
        setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) != 0 ||
        inet_pton(AF_INET, cfg.bind.c_str(), &address.sin_addr) != 1 ||
        bind(listen_fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0 ||
        listen(listen_fd, 4) != 0) {
        fprintf(stderr, "[ffn-worker] listen setup failed: %s\n", strerror(errno));
        return 1;
    }
    print_residency_phase(cfg, "ENDPOINT_READY");
    if (fused_layer_mask != 0) {
        fprintf(stderr, "S43FUSEDREADY layers=%016llx next=%016llx n_q=%lld n_kv=%lld hash=%016llx\n",
                static_cast<unsigned long long>(fused_layer_mask), static_cast<unsigned long long>(fused_next_mask),
                static_cast<long long>(fused_n_q), static_cast<long long>(fused_n_kv),
                static_cast<unsigned long long>(fused_weight_hash));
    }
    fprintf(stderr,
            "[ffn-worker] ready backend=%s layers=%zu mask=%016llx K=%lld NFF=%lld "
            "slice=[%lld,%lld) type=%s io=%s activation=%s max_tokens=%lld quantum=%lld "
            "alternate=%lld blocks=%zu weights=%.2f MiB hash=%016llx\n",
            ggml_backend_dev_description(device), states.size(),
            static_cast<unsigned long long>(cfg.layer_mask),
            static_cast<long long>(n_embd), static_cast<long long>(n_ff),
            static_cast<long long>(offset),
            static_cast<long long>(offset + cfg.columns),
            ggml_type_name(weight_type), cfg.f16_io ? "f16" : "f32",
            swiglu ? "swiglu" : "geglu",
            static_cast<long long>(cfg.max_tokens),
            static_cast<long long>(cfg.column_quantum),
            static_cast<long long>(cfg.alternate_columns), block_count,
            static_cast<double>(weight_buffer_bytes) / (1024.0 * 1024.0),
            static_cast<unsigned long long>(weight_hash));
    fflush(stderr);

    std::vector<uint8_t> request_payload;
    std::vector<uint8_t> response_payload;
    std::vector<uint8_t> response_frame;
    std::vector<ggml_fp16_t> input_f16;
    std::vector<ggml_fp16_t> output_f16;
    long served = 0;
    std::vector<double> compute_samples;
    long fused_served = 0;

    // releases everything and prints the kernel counters once the request budget is spent
    auto release_at_budget = [&](int client_fd) {
            close(client_fd);
            close(listen_fd);
            for (graph_arena & arena : graph_arenas) {
                ggml_gallocr_free(arena.allocator);
                ggml_free(arena.context);
            }
            for (ggml_backend_buffer_t weight_buffer : weight_buffers) {
                ggml_backend_buffer_free(weight_buffer);
            }
            for (ggml_context * weight_ctx : weight_contexts) {
                ggml_free(weight_ctx);
            }

            free_secondary();
            ggml_backend_free(backend);

            fprintf(stderr, "S42PIXELCPUPACKCALLS q4=%llu q6=%llu\n",
                    static_cast<unsigned long long>(pixel_cpu_q4_calls.load()),
                    static_cast<unsigned long long>(pixel_cpu_q6_calls.load()));
            if (pixel_cpu_row_profile) {
                for (int i = 0; i < 8; ++i) {
                    const auto & sample = pixel_row_totals[i];
                    if (!sample.calls) continue;
                    fprintf(stderr, "S42PIXELROWPROFILE thread=%d calls=%llu rows=%llu work_us=%llu slack_us=%llu\n", i,
                            static_cast<unsigned long long>(sample.calls), static_cast<unsigned long long>(sample.rows),
                            static_cast<unsigned long long>(sample.work_us), static_cast<unsigned long long>(sample.slack_us));
                }
            }
            fprintf(stderr, "S42PIXELNEONCALLS projections=%llu\n",
                    static_cast<unsigned long long>(pixel_neon_calls.load()));
            fprintf(stderr, "S42PIXELRESIDUALCALLS projections=%llu\n",
                    static_cast<unsigned long long>(pixel_residual_calls.load()));
            if (fused_layer_mask != 0) {
                fprintf(stderr, "S43FUSEDCALLS calls=%ld\n", fused_served);
            }
    };
    for (;;) {
        const int client_fd = accept(listen_fd, nullptr, nullptr);
        if (client_fd < 0) {
            continue;
        }
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

        ffn_split::hello_request hello = {};
        if (!read_exact(client_fd, &hello, sizeof(hello))) {
            close(client_fd);
            continue;
        }
        const uint16_t expected_flags =
                (cfg.f16_io ? ffn_split::flag_f16_io : 0) |
                (swiglu ? ffn_split::flag_swiglu : 0);
        const bool hello_ok =
                hello.magic == ffn_split::protocol_magic &&
                hello.version == ffn_split::protocol_version &&
                hello.message == static_cast<uint16_t>(ffn_split::message_type::hello_request) &&
                (hello.layer_mask & cfg.layer_mask) == cfg.layer_mask &&
                hello.max_columns == cfg.columns &&
                hello.max_tokens == cfg.max_tokens &&
                hello.n_embd == n_embd && hello.flags == expected_flags &&
                ffn_split::same_artifact_sha256(
                        hello.artifact_sha256, artifact_sha256);
        ffn_split::hello_response hello_response = {};
        hello_response.magic = ffn_split::protocol_magic;
        hello_response.version = ffn_split::protocol_version;
        hello_response.message = static_cast<uint16_t>(ffn_split::message_type::hello_response);
        hello_response.status = hello_ok ? 0 : 1;
        hello_response.flags = expected_flags;
        hello_response.n_embd = static_cast<uint32_t>(n_embd);
        hello_response.n_ff = static_cast<uint32_t>(n_ff);
        hello_response.offset = static_cast<uint32_t>(offset);
        hello_response.max_columns = static_cast<uint32_t>(cfg.columns);
        hello_response.weight_type = static_cast<uint32_t>(weight_type);
        hello_response.layer_count = static_cast<uint32_t>(states.size());
        hello_response.layer_mask = cfg.layer_mask;
        hello_response.weight_hash = weight_hash;
        hello_response.column_quantum =
                static_cast<uint32_t>(cfg.column_quantum);
        hello_response.max_tokens = static_cast<uint16_t>(cfg.max_tokens);
        hello_response.alternate_columns_32 = static_cast<uint16_t>(
                cfg.alternate_columns / 32);
        memcpy(
                hello_response.artifact_sha256, artifact_sha256,
                sizeof(artifact_sha256));
        if (!write_exact(client_fd, &hello_response, sizeof(hello_response)) || !hello_ok) {
            close(client_fd);
            continue;
        }
        fprintf(stderr, "[ffn-worker] client connected\n");

        for (;;) {
            ffn_split::execute_request request = {};
            if (!read_exact(client_fd, &request, sizeof(request))) {
                break;
            }
            const bool protocol_frame = request.magic == ffn_split::protocol_magic &&
                    request.version == ffn_split::protocol_version;
            if (protocol_frame &&
                request.message == static_cast<uint16_t>(ffn_split::message_type::fused_hello_request)) {
                ffn_split::fused_hello_payload asked = {};
                if (request.payload_bytes != sizeof(asked) ||
                    !read_exact(client_fd, &asked, sizeof(asked)) ||
                    request.payload_hash != ffn_split::hash_bytes(&asked, sizeof(asked))) {
                    fprintf(stderr, "[ffn-worker] invalid fused hello\n");
                    break;
                }
                const bool fused_ok = fused_layer_mask != 0 &&
                        (asked.fused_layer_mask & ~fused_layer_mask) == 0 &&
                        (asked.next_qkv_mask & ~fused_next_mask) == 0 &&
                        (asked.next_qkv_mask & ~asked.fused_layer_mask) == 0 &&
                        (asked.n_q == 0 || asked.n_q == static_cast<uint32_t>(fused_n_q)) &&
                        (asked.n_kv == 0 || asked.n_kv == static_cast<uint32_t>(fused_n_kv));
                ffn_split::fused_hello_payload offered = {};
                offered.fused_layer_mask = fused_layer_mask;
                offered.next_qkv_mask = fused_next_mask;
                offered.n_q = static_cast<uint32_t>(fused_n_q);
                offered.n_kv = static_cast<uint32_t>(fused_n_kv);
                offered.fused_weight_hash = fused_weight_hash;
                offered.rms_eps = fused_info.rms_eps;
                ffn_split::execute_response reply = {};
                reply.magic = ffn_split::protocol_magic;
                reply.version = ffn_split::protocol_version;
                reply.message = static_cast<uint16_t>(ffn_split::message_type::fused_hello_response);
                reply.status = fused_ok ? 0 : 1;
                reply.layer = -1;
                reply.payload_bytes = sizeof(offered);
                reply.payload_hash = ffn_split::hash_bytes(&offered, sizeof(offered));
                response_frame.resize(sizeof(reply) + sizeof(offered));
                memcpy(response_frame.data(), &reply, sizeof(reply));
                memcpy(response_frame.data() + sizeof(reply), &offered, sizeof(offered));
                if (!write_exact(client_fd, response_frame.data(), response_frame.size()) || !fused_ok) {
                    fprintf(stderr, "[ffn-worker] fused hello %s\n", fused_ok ? "write failed" : "refused");
                    break;
                }
                fprintf(stderr, "[ffn-worker] fused session layers=%016llx next=%016llx\n",
                        static_cast<unsigned long long>(asked.fused_layer_mask),
                        static_cast<unsigned long long>(asked.next_qkv_mask));
                continue;
            }
            const bool fused_plain = protocol_frame &&
                    request.message == static_cast<uint16_t>(ffn_split::message_type::fused_execute_request);
            const bool fused_next = protocol_frame &&
                    request.message == static_cast<uint16_t>(ffn_split::message_type::fused_next_execute_request);
            if (fused_plain || fused_next) {
                const uint64_t bit = request.layer >= 0 && request.layer < 64 ?
                        UINT64_C(1) << request.layer : 0;
                const uint64_t in_row = static_cast<uint64_t>(fused_n_q + n_embd);
                const uint64_t out_row = ffn_split::fused_response_row(
                        static_cast<uint32_t>(n_embd), static_cast<uint32_t>(fused_n_q),
                        static_cast<uint32_t>(fused_n_kv), fused_next);
                const bool fused_request_ok = bit != 0 && (fused_layer_mask & bit) != 0 &&
                        (!fused_next || (fused_next_mask & bit) != 0) &&
                        request.request_id != 0 && request.tokens > 0 && request.tokens <= cfg.max_tokens &&
                        request.columns == cfg.columns &&
                        request.elements == in_row * request.tokens &&
                        request.payload_bytes == static_cast<uint64_t>(request.elements) * wire_element_size;
                if (!fused_request_ok) {
                    fprintf(stderr, "[ffn-worker] invalid fused execute request\n");
                    break;
                }
                request_payload.resize(request.payload_bytes);
                if (!read_exact(client_fd, request_payload.data(), request_payload.size()) ||
                    request.payload_hash != ffn_split::hash_bytes(request_payload.data(), request_payload.size())) {
                    fprintf(stderr, "[ffn-worker] invalid fused execute payload\n");
                    break;
                }
                const bool input_finite = cfg.f16_io ?
                        all_finite(reinterpret_cast<const ggml_fp16_t *>(request_payload.data()), request.elements) :
                        all_finite(reinterpret_cast<const float *>(request_payload.data()), request.elements);
                if (!input_finite) {
                    fprintf(stderr, "[ffn-worker] non-finite fused input\n");
                    break;
                }
                const uint64_t started = now_us();
                const auto state_it = std::lower_bound(
                        states.begin(), states.end(), request.layer,
                        [](const layer_state & state, int layer) { return state.layer < layer; });
                graph_instance execution;
                if (state_it == states.end() || state_it->layer != request.layer ||
                    !build_fused_graph(*state_it, request.tokens, fused_next, execution)) {
                    fprintf(stderr, "[ffn-worker] fused graph build failed\n");
                    break;
                }
                ggml_backend_tensor_set(execution.input, request_payload.data(), 0, request_payload.size());
                const ggml_status status = ggml_backend_graph_compute(backend, execution.graph);
                const size_t out_elements = static_cast<size_t>(out_row * request.tokens);
                response_payload.resize(out_elements * wire_element_size);
                if (status == GGML_STATUS_SUCCESS) {
                    ggml_backend_tensor_get(execution.output, response_payload.data(), 0, response_payload.size());
                }
                const uint64_t compute_us = now_us() - started;
                const bool output_finite = cfg.f16_io ?
                        all_finite(reinterpret_cast<const ggml_fp16_t *>(response_payload.data()), out_elements) :
                        all_finite(reinterpret_cast<const float *>(response_payload.data()), out_elements);
                if (status != GGML_STATUS_SUCCESS || !output_finite) {
                    fprintf(stderr, "[ffn-worker] fused graph execution failed\n");
                    break;
                }
                ffn_split::execute_response response = {};
                response.magic = ffn_split::protocol_magic;
                response.version = ffn_split::protocol_version;
                response.message = static_cast<uint16_t>(ffn_split::message_type::fused_execute_response);
                response.status = 0;
                response.request_id = request.request_id;
                response.layer = request.layer;
                response.elements = static_cast<uint32_t>(out_elements);
                response.payload_bytes = static_cast<uint32_t>(response_payload.size());
                response.payload_hash = ffn_split::hash_bytes(response_payload.data(), response_payload.size());
                response.columns = request.columns;
                response.tokens = request.tokens;
                response.compute_us = compute_us;
                response_frame.resize(sizeof(response) + response_payload.size());
                memcpy(response_frame.data(), &response, sizeof(response));
                memcpy(response_frame.data() + sizeof(response), response_payload.data(), response_payload.size());
                if (!write_exact(client_fd, response_frame.data(), response_frame.size())) {
                    break;
                }
                ++fused_served;
                compute_samples.push_back(static_cast<double>(compute_us));
                ++served;
                if (cfg.max_requests > 0 && served >= cfg.max_requests) {
                    release_at_budget(client_fd);
                    return 0;
                }
                continue;
            }
            const size_t payload_bytes = request.payload_bytes;
            const bool request_ok =
                    request.magic == ffn_split::protocol_magic &&
                    request.version == ffn_split::protocol_version &&
                    request.message == static_cast<uint16_t>(ffn_split::message_type::execute_request) &&
                    request.request_id != 0 && request.layer >= 0 && request.layer < 64 &&
                    (cfg.layer_mask & (UINT64_C(1) << request.layer)) != 0 &&
                    request.tokens > 0 && request.tokens <= cfg.max_tokens &&
                    request.elements ==
                            static_cast<uint64_t>(n_embd) * request.tokens &&
                    payload_bytes == static_cast<uint64_t>(request.elements) *
                            wire_element_size &&
                    payload_bytes <= max_payload_bytes &&
                    supported_columns(request.columns);
            if (!request_ok) {
                fprintf(stderr, "[ffn-worker] invalid execute request\n");
                break;
            }
            request_payload.resize(payload_bytes);
            if (!read_exact(
                        client_fd, request_payload.data(),
                        request_payload.size()) ||
                request.payload_hash != ffn_split::hash_bytes(
                        request_payload.data(), request_payload.size())) {
                fprintf(stderr, "[ffn-worker] invalid execute payload\n");
                break;
            }
            if (cfg.f16_io) {
                input_f16.resize(request.elements);
                output_f16.resize(request.elements);
                memcpy(input_f16.data(), request_payload.data(), payload_bytes);
                if (!all_finite(input_f16.data(), input_f16.size())) {
                    fprintf(stderr, "[ffn-worker] non-finite input\n");
                    break;
                }
            } else {
                input_data.resize(request.elements);
                output_data.resize(request.elements);
                memcpy(input_data.data(), request_payload.data(), payload_bytes);
                if (!all_finite(input_data)) {
                    fprintf(stderr, "[ffn-worker] non-finite input\n");
                    break;
                }
            }

            const uint64_t started = now_us();
            const auto state_it = std::lower_bound(
                    states.begin(), states.end(), request.layer,
                    [](const layer_state & state, int layer) {
                        return state.layer < layer;
                    });
            if (state_it == states.end() || state_it->layer != request.layer) {
                fprintf(stderr, "[ffn-worker] layer lookup failed\n");
                break;
            }
            layer_state & state = *state_it;
            graph_instance execution;
            ggml_status status = GGML_STATUS_SUCCESS;
            uint64_t compute_us = 0;
            if (dual) {
                uint64_t dual_us = 0;
                if (!run_dual(
                            state, request.request_id, request.columns,
                            request.tokens, request_payload.data(),
                            payload_bytes, dual_us)) {
                    status = GGML_STATUS_FAILED;
                } else if (cfg.f16_io) {
                    memcpy(output_f16.data(), dual_output.data(), payload_bytes);
                } else {
                    memcpy(output_data.data(), dual_output.data(), payload_bytes);
                }
                compute_us = now_us() - started;
            } else {
                if (!build_graph(
                            state, request.columns, request.tokens, 0,
                            execution)) {
                    fprintf(stderr, "[ffn-worker] dynamic FFN graph build failed\n");
                    break;
                }
                if (cfg.f16_io) {
                    ggml_backend_tensor_set(
                            execution.input, input_f16.data(), 0,
                            input_f16.size() * sizeof(ggml_fp16_t));
                } else {
                    ggml_backend_tensor_set(
                            execution.input, input_data.data(), 0,
                            input_data.size() * sizeof(float));
                }
                status = ggml_backend_graph_compute(backend, execution.graph);
                if (cfg.f16_io) {
                    ggml_backend_tensor_get(
                            execution.output, output_f16.data(), 0,
                            output_f16.size() * sizeof(ggml_fp16_t));
                } else {
                    ggml_backend_tensor_get(
                            execution.output, output_data.data(), 0,
                            output_data.size() * sizeof(float));
                }
                compute_us = now_us() - started;
            }
            const bool output_ok = cfg.f16_io ?
                    all_finite(output_f16.data(), output_f16.size()) :
                    all_finite(output_data);
            if (status != GGML_STATUS_SUCCESS || !output_ok) {
                fprintf(stderr, "[ffn-worker] graph execution failed\n");
                break;
            }

            response_payload.resize(payload_bytes);
            if (cfg.f16_io) {
                memcpy(response_payload.data(), output_f16.data(), payload_bytes);
            } else {
                memcpy(response_payload.data(), output_data.data(), payload_bytes);
            }
            ffn_split::execute_response response = {};
            response.magic = ffn_split::protocol_magic;
            response.version = ffn_split::protocol_version;
            response.message = static_cast<uint16_t>(ffn_split::message_type::execute_response);
            response.status = 0;
            response.request_id = request.request_id;
            response.layer = request.layer;
            response.elements = request.elements;
            response.payload_bytes = static_cast<uint32_t>(payload_bytes);
            response.payload_hash = ffn_split::hash_bytes(
                    response_payload.data(), response_payload.size());
            response.columns = request.columns;
            response.tokens = request.tokens;
            response.compute_us = compute_us;
            response_frame.resize(sizeof(response) + response_payload.size());
            memcpy(response_frame.data(), &response, sizeof(response));
            memcpy(response_frame.data() + sizeof(response),
                    response_payload.data(), response_payload.size());
            if (!write_exact(client_fd, response_frame.data(), response_frame.size())) {
                break;
            }
            compute_samples.push_back(static_cast<double>(compute_us));
            ++served;
            if (served % 32 == 0) {
                std::vector<double> sorted = compute_samples;
                std::sort(sorted.begin(), sorted.end());
                fprintf(stderr,
                        "[ffn-worker] requests=%ld compute_p50_us=%.0f\n",
                        served, sorted[sorted.size() / 2]);
                fflush(stderr);
            }
            if (cfg.max_requests > 0 && served >= cfg.max_requests) {
                release_at_budget(client_fd);
                return 0;
            }
        }
        close(client_fd);
        fprintf(stderr, "[ffn-worker] client disconnected\n");
    }
}

#if !defined(FFN_SPLIT_WORKER_NO_MAIN)
int main(int argc, char ** argv) {
    return ffn_split_worker_main(argc, argv);
}
#endif
