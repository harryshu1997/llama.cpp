// Remote-resident FFN probe (S42 phase concurrency, first milestone).
//
// Loads a GGUF either completely ("full" mode) or with the dense FFN weights of the layers in
// --remote-mask omitted from the desktop ("remote" mode, LLAMA_FFN_REMOTE_RESIDENT_LAYER_MASK).
// In remote mode the omitted layers are executed by an FFN split worker over TCP through the
// same client the server uses. The probe writes one JSON record with
//   * the loader's allocation proof (omitted / unmapped / mapped bytes, buffer sizes),
//   * an independent kernel-level proof from /proc/self/smaps: no VMA of the model file covers
//     a page that lies completely inside an omitted tensor,
//   * the greedy decode trace (argmax per step) and a raw f32 logits file for numerical
//     comparison between the two modes,
//   * the fail-closed checks (context without an owner, control targeting a remote layer).
//
// usage: llama-ffn-remote-resident-probe --model PATH --out JSON [--logits PATH]
//        [--remote-mask MASK --worker-host H --worker-port P --artifact-sha256 sha256:HEX
//         --max-tokens N] [--tokens 1,2,3] [--decode N] [--expect-no-owner]

#include "llama.h"
#include "../../src/llama-ext.h"
#include "gguf.h"
#include "ggml.h"

#include "ffn-split-client.h"

#include <cerrno>
#include <climits>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

struct options {
    std::string model;
    std::string out;
    std::string logits;
    std::string worker_host = "127.0.0.1";
    std::string artifact_sha256;
    int worker_port = 0;
    uint64_t remote_mask = 0;
    uint64_t assisted_mask = 0;
    uint32_t row_diagnostic_steps = 0;
    uint32_t max_tokens = 16;
    uint32_t batch_size = 0;
    uint32_t ubatch_size = 0;
    uint32_t sequence_count = 1;
    std::string runtime_context;
    std::vector<llama_token> tokens = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<llama_token> decode_tokens;
    int decode_steps = 4;
    bool expect_no_owner = false;
    int n_threads = 2;
    int gpu_layers = 0;
    int context_size = 128;
    llama_flash_attn_type flash_attn = LLAMA_FLASH_ATTN_TYPE_AUTO;
    std::vector<int32_t> kv_cpu_layers;
    std::vector<llama_kv_device_cells> kv_device_cells;
    // dormant host share: release the FFN column suffix [host_columns, n_ff) of the masked layers
    // after a local decode, decode again from a fresh context, then populate the pages back
    uint64_t dormant_mask = 0;
    int64_t dormant_host_columns = -1;
    // consume the released bytes with touched anonymous memory before the dormant restore
    bool dormant_consume = false;
    // clear the KV cache (data) after the first decode and decode the same prompt again on the same context
    bool clear_reuse = false;
    bool state_reuse = false;
    // touch the pages of the first N KV cells (after the dormant release when one is requested)
    int64_t kv_touch_tokens = 0;
    // dormant: skip the local re-decode (measure memory only)
    bool dormant_no_decode = false;
    bool dormant_drop_cache = true;
    bool dormant_populate = true;
    bool dormant_restore_before_decode = false;
};

bool parse_u64(const char * text, uint64_t & value) {
    errno = 0;
    char * end = nullptr;
    const unsigned long long parsed = strtoull(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0') {
        return false;
    }
    value = parsed;
    return true;
}

bool parse_options(int argc, char ** argv, options & result) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](std::string & target) {
            if (i + 1 >= argc) {
                return false;
            }
            target = argv[++i];
            return true;
        };
        std::string value;
        uint64_t number = 0;
        if (arg == "--model") {
            if (!next(result.model)) return false;
        } else if (arg == "--out") {
            if (!next(result.out)) return false;
        } else if (arg == "--logits") {
            if (!next(result.logits)) return false;
        } else if (arg == "--worker-host") {
            if (!next(result.worker_host)) return false;
        } else if (arg == "--artifact-sha256") {
            if (!next(result.artifact_sha256)) return false;
        } else if (arg == "--worker-port") {
            if (!next(value) || !parse_u64(value.c_str(), number) || number == 0 || number > 65535) return false;
            result.worker_port = (int) number;
        } else if (arg == "--remote-mask") {
            if (!next(value) || !parse_u64(value.c_str(), number)) return false;
            result.remote_mask = number;
        } else if (arg == "--assisted-mask") {
            if (!next(value) || !parse_u64(value.c_str(), number) || number == 0) return false;
            result.assisted_mask = number;
        } else if (arg == "--max-tokens") {
            if (!next(value) || !parse_u64(value.c_str(), number) || number == 0 || number > UINT16_MAX) return false;
            result.max_tokens = (uint32_t) number;
        } else if (arg == "--batch-size") {
            if (!next(value) || !parse_u64(value.c_str(), number) || number == 0 || number > UINT16_MAX) return false;
            result.batch_size = (uint32_t) number;
        } else if (arg == "--ubatch-size") {
            if (!next(value) || !parse_u64(value.c_str(), number) || number == 0 || number > UINT16_MAX) return false;
            result.ubatch_size = (uint32_t) number;
        } else if (arg == "--sequence-count") {
            if (!next(value) || !parse_u64(value.c_str(), number) || number == 0 || number > 8) return false;
            result.sequence_count = (uint32_t) number;
        } else if (arg == "--runtime-context") {
            if (!next(result.runtime_context) || (result.runtime_context != "ubatch" &&
                result.runtime_context != "logical" && result.runtime_context != "reject")) return false;
        } else if (arg == "--gpu-layers") {
            if (!next(value) || !parse_u64(value.c_str(), number) || number > 1024) return false;
            result.gpu_layers = (int) number;
        } else if (arg == "--threads") {
            if (!next(value) || !parse_u64(value.c_str(), number) || number == 0 || number > 256) return false;
            result.n_threads = (int) number;
        } else if (arg == "--ctx-size") {
            if (!next(value) || !parse_u64(value.c_str(), number) || number == 0 || number > INT32_MAX) return false;
            result.context_size = (int) number;
        } else if (arg == "--flash-attn") {
            if (!next(value) || (value != "on" && value != "off" && value != "auto")) return false;
            result.flash_attn = value == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED :
                value == "off" ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_AUTO;
        } else if (arg == "--kv-cpu-layers") {
            if (!next(value) || value.empty() || value.back() == ',') return false;
            std::stringstream stream(value);
            std::string item;
            while (std::getline(stream, item, ',')) {
                if (item.empty() || item.find_first_not_of("0123456789") != std::string::npos ||
                        !parse_u64(item.c_str(), number) || number > INT32_MAX) return false;
                result.kv_cpu_layers.push_back((int32_t) number);
            }
        } else if (arg == "--kv-device-cells") {
            if (!next(value) || value.empty() || value.back() == ',') return false;
            std::stringstream stream(value);
            std::string item;
            while (std::getline(stream, item, ',')) {
                const size_t colon = item.find(':');
                uint64_t layer, cells;
                if (colon == std::string::npos || !parse_u64(item.substr(0, colon).c_str(), layer) ||
                    !parse_u64(item.substr(colon + 1).c_str(), cells) || layer > INT32_MAX || cells > UINT32_MAX) return false;
                result.kv_device_cells.push_back({int32_t(layer), uint32_t(cells)});
            }
        } else if (arg == "--decode") {
            if (!next(value) || !parse_u64(value.c_str(), number) || number > 4096) return false;
            result.decode_steps = (int) number;
        } else if (arg == "--tokens" || arg == "--decode-tokens") {
            if (!next(value)) return false;
            auto & tokens = arg == "--tokens" ? result.tokens : result.decode_tokens;
            tokens.clear();
            std::stringstream stream(value);
            std::string item;
            while (std::getline(stream, item, ',')) {
                if (item.empty() || !parse_u64(item.c_str(), number)) return false;
                tokens.push_back((llama_token) number);
            }
            if (tokens.empty()) return false;
        } else if (arg == "--expect-no-owner") {
            result.expect_no_owner = true;
        } else if (arg == "--dormant-mask") {
            if (!next(value) || !parse_u64(value.c_str(), number) || number == 0) return false;
            result.dormant_mask = number;
        } else if (arg == "--dormant-host-columns") {
            if (!next(value) || !parse_u64(value.c_str(), number) || number > INT32_MAX) return false;
            result.dormant_host_columns = (int64_t) number;
        } else if (arg == "--dormant-consume") {
            result.dormant_consume = true;
        } else if (arg == "--clear-reuse") {
            result.clear_reuse = true;
        } else if (arg == "--state-reuse") {
            result.state_reuse = true;
        } else if (arg == "--kv-touch-tokens") {
            if (!next(value) || !parse_u64(value.c_str(), number) || number > INT32_MAX) return false;
            result.kv_touch_tokens = (int64_t) number;
        } else if (arg == "--dormant-no-decode") {
            result.dormant_no_decode = true;
        } else if (arg == "--dormant-drop-cache") {
            if (!next(value) || !parse_u64(value.c_str(), number) || number > 1) return false;
            result.dormant_drop_cache = number != 0;
        } else if (arg == "--dormant-populate") {
            if (!next(value) || !parse_u64(value.c_str(), number) || number > 1) return false;
            result.dormant_populate = number != 0;
        } else if (arg == "--dormant-restore-before-decode") {
            result.dormant_restore_before_decode = true;
        } else {
            return false;
        }
    }
    if ((result.dormant_mask != 0) != (result.dormant_host_columns >= 0)) return false;
    if (result.dormant_consume && result.dormant_mask == 0) return false;
    if ((!result.dormant_drop_cache || !result.dormant_populate || result.dormant_restore_before_decode) && result.dormant_mask == 0) return false;
    if (result.dormant_mask != 0 && result.remote_mask != 0) return false;
    if (const char * diagnostic = std::getenv("S41_SERVER_FFN_ROW_DIAGNOSTIC_STEPS")) {
        uint64_t steps = 0;
        if (!parse_u64(diagnostic, steps) || (steps != 0 && steps != 5 && steps != 64)) return false;
        result.row_diagnostic_steps = uint32_t(steps);
    }
    if (result.assisted_mask && (result.remote_mask || result.dormant_mask || result.runtime_context != "ubatch")) return false;
    if (result.row_diagnostic_steps && !result.assisted_mask) return false;
    return !result.model.empty() && !result.out.empty();
}

// ---- log capture -------------------------------------------------------------------------

struct log_capture {
    std::vector<std::string> lines;
    std::string partial;
};

void log_callback(ggml_log_level level, const char * text, void * user_data) {
    auto * capture = static_cast<log_capture *>(user_data);
    capture->partial += text;
    size_t newline;
    while ((newline = capture->partial.find('\n')) != std::string::npos) {
        std::string line = capture->partial.substr(0, newline);
        capture->partial.erase(0, newline + 1);
        if (line.find("REMOTE_RESIDENT_FFN") != std::string::npos ||
            line.find("KV_PLACEMENT") != std::string::npos ||
            line.find("KV_SPLIT") != std::string::npos ||
            line.find("KV buffer size") != std::string::npos ||
            line.find("KV buffer zeroed lazily") != std::string::npos ||
            line.find("compute buffer size") != std::string::npos ||
            line.find("graph splits") != std::string::npos ||
            line.find("model buffer size") != std::string::npos ||
            line.find("remote-resident") != std::string::npos ||
            line.find("released") != std::string::npos ||
            level >= GGML_LOG_LEVEL_WARN) {
            capture->lines.push_back(line);
        }
    }
    if (level >= GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

std::string json_escape(const std::string & text) {
    std::string out;
    for (char c : text) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\t': out += "\\t"; break;
            default:
                if ((unsigned char) c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", (unsigned) (unsigned char) c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

// ---- omitted tensor ranges from the GGUF metadata --------------------------------------

struct tensor_range {
    std::string name;
    size_t offs = 0;
    size_t nbytes = 0;
    size_t page_first = 0;   // inward page-aligned range [page_first, page_last)
    size_t page_last = 0;
    size_t vma_overlap_bytes = 0;  // bytes of that range still covered by a VMA of the file
    size_t rss_overlap_bytes = 0;  // resident bytes of overlapping VMAs (upper bound)
};

struct ffn_geometry {
    uint32_t n_embd = 0;
    uint32_t n_ff = 0;
    std::string arch;
};

bool omitted_ranges(const std::string & model, uint64_t remote_mask, std::vector<tensor_range> & ranges,
                    ffn_geometry & geometry, std::string & error) {
    ggml_context * meta = nullptr;
    gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ &meta };
    gguf_context * gguf = gguf_init_from_file(model.c_str(), params);
    if (gguf == nullptr) {
        error = "cannot read GGUF metadata";
        return false;
    }
    const int64_t arch_key = gguf_find_key(gguf, "general.architecture");
    if (arch_key < 0) {
        error = "GGUF lacks general.architecture";
        gguf_free(gguf);
        ggml_free(meta);
        return false;
    }
    const std::string arch = gguf_get_val_str(gguf, arch_key);
    const int64_t embd_key = gguf_find_key(gguf, (arch + ".embedding_length").c_str());
    const int64_t ff_key = gguf_find_key(gguf, (arch + ".feed_forward_length").c_str());
    if (embd_key < 0 || ff_key < 0) {
        error = "GGUF lacks embedding_length/feed_forward_length";
        gguf_free(gguf);
        ggml_free(meta);
        return false;
    }
    geometry.n_embd = gguf_get_val_u32(gguf, embd_key);
    geometry.n_ff = gguf_get_val_u32(gguf, ff_key);
    geometry.arch = arch;
    const size_t data_offset = gguf_get_data_offset(gguf);
    const long page = sysconf(_SC_PAGESIZE);
    for (int il = 0; il < 64; ++il) {
        if ((remote_mask & (UINT64_C(1) << il)) == 0) {
            continue;
        }
        for (const char * kind : {"ffn_gate", "ffn_up", "ffn_down"}) {
            const std::string name = "blk." + std::to_string(il) + "." + kind + ".weight";
            const int64_t tid = gguf_find_tensor(gguf, name.c_str());
            if (tid < 0) {
                error = "tensor missing from GGUF: " + name;
                gguf_free(gguf);
                ggml_free(meta);
                return false;
            }
            tensor_range range;
            range.name = name;
            range.offs = data_offset + gguf_get_tensor_offset(gguf, tid);
            range.nbytes = gguf_get_tensor_size(gguf, tid);
            const size_t first = (range.offs + page - 1) & ~((size_t) page - 1);
            const size_t last = (range.offs + range.nbytes) & ~((size_t) page - 1);
            range.page_first = first;
            range.page_last = last > first ? last : first;
            ranges.push_back(range);
        }
    }
    gguf_free(gguf);
    ggml_free(meta);
    return true;
}

// ---- /proc/self/smaps scan --------------------------------------------------------------

struct vma {
    uintptr_t start = 0;
    uintptr_t end = 0;
    size_t file_offset = 0;
    size_t rss_bytes = 0;
};

std::vector<vma> model_vmas(const std::string & real_model) {
    std::vector<vma> result;
    std::ifstream smaps("/proc/self/smaps");
    std::string line;
    vma current;
    bool in_model = false;
    while (std::getline(smaps, line)) {
        unsigned long long start = 0, end = 0, offset = 0;
        char perms[8] = {};
        char dev[32] = {};
        unsigned long long inode = 0;
        int consumed = 0;
        if (sscanf(line.c_str(), "%llx-%llx %7s %llx %31s %llu %n", &start, &end, perms, &offset, dev, &inode, &consumed) == 6) {
            if (in_model) {
                result.push_back(current);
            }
            std::string path = consumed < (int) line.size() ? line.substr(consumed) : "";
            while (!path.empty() && path[0] == ' ') {
                path.erase(0, 1);
            }
            in_model = path == real_model;
            current = vma{ (uintptr_t) start, (uintptr_t) end, (size_t) offset, 0 };
            continue;
        }
        if (in_model && line.rfind("Rss:", 0) == 0) {
            unsigned long long kb = 0;
            if (sscanf(line.c_str(), "Rss: %llu kB", &kb) == 1) {
                current.rss_bytes = (size_t) kb * 1024;
            }
        }
    }
    if (in_model) {
        result.push_back(current);
    }
    return result;
}

void annotate_ranges(const std::vector<vma> & vmas, std::vector<tensor_range> & ranges) {
    for (auto & range : ranges) {
        for (const auto & area : vmas) {
            const size_t area_first = area.file_offset;
            const size_t area_last = area.file_offset + (area.end - area.start);
            const size_t lo = std::max(area_first, range.page_first);
            const size_t hi = std::min(area_last, range.page_last);
            if (lo < hi) {
                range.vma_overlap_bytes += hi - lo;
                range.rss_overlap_bytes += std::min(area.rss_bytes, hi - lo);
            }
        }
    }
}

// ---- eval callback ------------------------------------------------------------------------

bool client_eval_cb(ggml_tensor * tensor, bool ask, void * user_data) {
    auto * client = static_cast<ffn_split::client *>(user_data);
    return client != nullptr && client->eval(tensor, ask);
}

struct runtime_attribution {
    ffn_split::client * client = nullptr;
    bool reject = false;
    int32_t decoded_token_index = 0;
    std::vector<std::vector<ffn_split::client_runtime_context_entry>> batches;
};

bool runtime_context_cb(const llama_batch & batch, void * user_data) {
    auto & state = *static_cast<runtime_attribution *>(user_data);
    if (state.reject) {
        return false;
    }
    std::map<int32_t, std::vector<uint32_t>> rows;
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (!batch.n_seq_id || !batch.seq_id || batch.n_seq_id[i] != 1 || !batch.seq_id[i]) {
            return false;
        }
        rows[batch.seq_id[i][0]].push_back(i);
    }
    std::vector<ffn_split::client_runtime_context_entry> entries;
    for (const auto & row : rows) {
        std::vector<int32_t> positions;
        for (uint32_t index : row.second) {
            positions.push_back(batch.pos ? batch.pos[index] : -1);
        }
        entries.push_back({"probe-" + std::to_string(row.first), row.first, uint32_t(row.second.size()),
                           11 + static_cast<uint64_t>(row.first), row.second, positions,
                           state.decoded_token_index, 0});
    }
    state.batches.push_back(entries);
    std::string error;
    return !state.client || state.client->set_runtime_context(entries, error);
}

} // namespace

int main(int argc, char ** argv) {
    options opts;
    if (!parse_options(argc, argv, opts)) {
        fprintf(stderr, "usage: %s --model PATH --out JSON [--logits PATH] [--remote-mask MASK --worker-host H "
                        "--worker-port P --artifact-sha256 sha256:HEX --max-tokens N] [--tokens 1,2,3] [--decode N] "
                        "[--expect-no-owner] [--threads N] [--gpu-layers N] "
                        "[--dormant-mask MASK --dormant-host-columns N]\n", argv[0]);
        return 2;
    }
    const bool remote = opts.remote_mask != 0;
    if (remote) {
        setenv("LLAMA_FFN_REMOTE_RESIDENT_LAYER_MASK", std::to_string(opts.remote_mask).c_str(), 1);
    } else {
        unsetenv("LLAMA_FFN_REMOTE_RESIDENT_LAYER_MASK");
    }
    setenv("LLAMA_FFN_SPLIT_LAYER_MASK", "0", 1);
    setenv("LLAMA_FFN_SPLIT_COLUMNS", "0", 1);

    log_capture capture;
    llama_log_set(log_callback, &capture);
    llama_backend_init();

    char real_model_buf[PATH_MAX];
    const std::string real_model = realpath(opts.model.c_str(), real_model_buf) ? real_model_buf : opts.model;

    std::vector<tensor_range> ranges;
    ffn_geometry geometry;
    std::string error;
    if ((remote || opts.assisted_mask) && !omitted_ranges(opts.model, opts.remote_mask | opts.assisted_mask, ranges, geometry, error)) {
        fprintf(stderr, "probe: %s\n", error.c_str());
        return 1;
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = opts.gpu_layers;
    mparams.use_mmap = true;
    llama_model * model = llama_model_load_from_file(opts.model.c_str(), mparams);
    if (model == nullptr) {
        fprintf(stderr, "probe: model load failed\n");
        return 1;
    }
    if (opts.dormant_mask != 0 && !llama_model_ffn_host_share_configure(model, opts.dormant_drop_cache, opts.dormant_populate)) {
        fprintf(stderr, "probe: dormant host share policy is unsupported\n");
        llama_model_free(model);
        return 1;
    }

    // ---- kernel-level proof right after loading (before any context exists) ----
    std::vector<vma> vmas = model_vmas(real_model);
    size_t file_mapped_bytes = 0;
    size_t file_rss_bytes = 0;
    for (const auto & area : vmas) {
        file_mapped_bytes += area.end - area.start;
        file_rss_bytes += area.rss_bytes;
    }
    annotate_ranges(vmas, ranges);

    const uint64_t loaded_mask = llama_model_remote_resident_ffn_layer_mask(model);
    const uint64_t omitted_bytes = llama_model_remote_resident_ffn_bytes(model);
    const uint64_t unmapped_bytes = llama_model_remote_resident_ffn_unmapped_bytes(model);

    // ---- context ----
    std::unique_ptr<ffn_split::client> client;
    std::string client_error;
    bool client_connected = false;
    std::string policy_rejection;
    if ((remote || opts.assisted_mask) && !opts.expect_no_owner) {
        ffn_split::client_config config;
        config.transport = ffn_split::client_transport::tcp;
        config.artifact_sha256 = opts.artifact_sha256;
        config.host = opts.worker_host;
        config.port = opts.worker_port;
        config.layer_mask = opts.remote_mask | opts.assisted_mask;
        config.remote_resident_layer_mask = opts.remote_mask;
        config.max_columns = geometry.n_ff;
        config.n_embd = geometry.n_embd;
        config.max_tokens = (uint16_t) opts.max_tokens;
        config.f16_io = false;
        // the worker derives the activation from the shard architecture: gemma4 is GEGLU
        config.swiglu = geometry.arch != "gemma4";
        config.runtime_control = true;
        config.row_diagnostic_steps = opts.row_diagnostic_steps;
        config.timeout_ms = 20000;
        client = std::make_unique<ffn_split::client>(std::move(config));
        client_connected = client->connect(client_error);
        if (!client_connected) {
            fprintf(stderr, "probe: FFN worker connect failed: %s\n", client_error.c_str());
        } else if (remote) {
            // a control that targets a remote-resident layer must be refused
            std::string rejection;
            const int first_layer = __builtin_ctzll(opts.remote_mask);
            if (client->set_runtime_policy(UINT64_C(1) << first_layer, geometry.n_ff, rejection)) {
                policy_rejection = "ACCEPTED";
            } else {
                policy_rejection = rejection;
            }
        }
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = opts.context_size;
    cparams.flash_attn_type = opts.flash_attn;
    cparams.kv_cpu_layers = opts.kv_cpu_layers.data();
    cparams.n_kv_cpu_layers = opts.kv_cpu_layers.size();
    cparams.kv_device_cells = opts.kv_device_cells.data();
    cparams.n_kv_device_cells = opts.kv_device_cells.size();
    cparams.n_batch = opts.batch_size ? opts.batch_size : opts.max_tokens;
    cparams.n_ubatch = opts.ubatch_size ? opts.ubatch_size : opts.max_tokens;
    cparams.n_seq_max = opts.sequence_count;
    cparams.n_threads = opts.n_threads;
    cparams.n_threads_batch = opts.n_threads;
    cparams.no_perf = true;
    if (client && client_connected) {
        cparams.cb_eval = client_eval_cb;
        cparams.cb_eval_user_data = client.get();
    }
    llama_context * ctx = ((remote || opts.assisted_mask) && !opts.expect_no_owner && !client_connected) ? nullptr : llama_init_from_model(model, cparams);
    const bool context_created = ctx != nullptr;
    auto major_faults = []() {
        std::ifstream stat("/proc/self/stat");
        std::string line;
        std::getline(stat, line);
        // fields after the parenthesised comm: state ppid pgrp session tty tpgid flags minflt cminflt majflt
        const size_t close = line.rfind(')');
        std::istringstream rest(line.substr(close + 2));
        std::string state; long long v = 0, majflt = 0;
        rest >> state;
        for (int i = 0; i < 9 && (rest >> v); ++i) { if (i == 8) majflt = v; }
        return (size_t) majflt;
    };
    auto rss_anon_bytes = []() {
        std::ifstream status("/proc/self/status");
        std::string line;
        while (std::getline(status, line)) {
            if (line.rfind("RssAnon:", 0) == 0) {
                return (size_t) std::stoull(line.substr(8)) * 1024;
            }
        }
        return (size_t) 0;
    };
    // anonymous RSS right after context creation: with page-granular KV zeroing the host KV
    // allocation is not resident yet, with LLAMA_KV_CACHE_EAGER_CLEAR=1 it is
    size_t context_rss_anon_bytes = 0;
    {
        std::ifstream status("/proc/self/status");
        std::string line;
        while (std::getline(status, line)) {
            if (line.rfind("RssAnon:", 0) == 0) {
                context_rss_anon_bytes = (size_t) std::stoull(line.substr(8)) * 1024;
                break;
            }
        }
    }
    runtime_attribution attribution;
    attribution.client = client.get();
    attribution.reject = opts.runtime_context == "reject";
    if (ctx && (opts.runtime_context == "ubatch" || attribution.reject)) {
        llama_set_ffn_split_ubatch_callback(ctx, runtime_context_cb, &attribution);
    }

    // ---- decode ----
    std::vector<int> argmax_trace;
    std::vector<float> all_logits;
    int decode_status = -1;
    std::string decode_error;
    int n_vocab = 0;
    // prompt + greedy steps on one context; reused for the dormant re-execution
    auto decode_once = [&](llama_context * target, std::vector<float> & logits_out, std::vector<int> & argmax_out) {
        const llama_vocab * vocab = llama_model_get_vocab(model);
        n_vocab = llama_vocab_n_tokens(vocab);
        std::vector<llama_token> tokens = opts.tokens;
        llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
        batch.n_tokens = (int32_t) tokens.size();
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            batch.token[i] = tokens[i];
            batch.pos[i] = i / opts.sequence_count;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = i % opts.sequence_count;
            batch.logits[i] = i == batch.n_tokens - 1;
        }
        if (opts.runtime_context == "logical") {
            runtime_context_cb(batch, &attribution);
        }
        if (opts.assisted_mask) {
            llama_set_ffn_split_policy(target, true, 0, 0);
        }
        int status = llama_decode(target, batch);
        llama_batch_free(batch);
        for (int step = 0; status == 0 && step <= opts.decode_steps; ++step) {
            const float * logits = llama_get_logits_ith(target, -1);
            int best = 0;
            for (int t = 1; t < n_vocab; ++t) {
                if (logits[t] > logits[best]) {
                    best = t;
                }
            }
            argmax_out.push_back(best);
            logits_out.insert(logits_out.end(), logits, logits + n_vocab);
            if (step == opts.decode_steps) {
                break;
            }
            llama_token next = opts.decode_tokens.empty() ? (llama_token) best : opts.decode_tokens.at(step);
            batch = llama_batch_get_one(&next, 1);
            if (opts.assisted_mask) {
                attribution.decoded_token_index = step;
                if (!client->set_runtime_policy(opts.assisted_mask, geometry.n_ff, client_error)) {
                    return -1;
                }
                llama_set_ffn_split_policy(target, true, opts.assisted_mask, geometry.n_ff,
                        step < int(opts.row_diagnostic_steps));
            }
            status = llama_decode(target, batch);
        }
        return status;
    };
    if (!opts.decode_tokens.empty() && opts.decode_tokens.size() < size_t(opts.decode_steps)) {
        fprintf(stderr, "--decode-tokens must cover all decode steps\n");
        return 1;
    }
    if (ctx != nullptr) {
        decode_status = decode_once(ctx, all_logits, argmax_trace);
        if (decode_status != 0) {
            decode_error = "llama_decode returned " + std::to_string(decode_status);
        }
        if (client && client->failed()) {
            decode_error += (decode_error.empty() ? "" : "; ") + std::string("client: ") + client->error();
        }
    }

    struct state_reuse_result {
        bool ran = false;
        size_t bytes = 0, restored_bytes = 0;
        int decode_status = -1;
        bool logits_identical = false;
    } state_reuse;
    if (opts.state_reuse && ctx != nullptr && decode_status == 0) {
        state_reuse.ran = true;
        std::vector<uint8_t> saved(llama_state_get_size(ctx));
        state_reuse.bytes = llama_state_get_data(ctx, saved.data(), saved.size());
        llama_token token = 7;
        llama_batch batch = llama_batch_get_one(&token, 1);
        const int first_status = llama_decode(ctx, batch);
        std::vector<float> expected;
        if (first_status == 0) {
            const float * logits = llama_get_logits_ith(ctx, -1);
            expected.assign(logits, logits + n_vocab);
        }
        llama_memory_clear(llama_get_memory(ctx), true);
        state_reuse.restored_bytes = llama_state_set_data(ctx, saved.data(), state_reuse.bytes);
        state_reuse.decode_status = llama_decode(ctx, batch);
        state_reuse.logits_identical = first_status == 0 && state_reuse.decode_status == 0 &&
            state_reuse.bytes == saved.size() && state_reuse.restored_bytes == state_reuse.bytes &&
            std::equal(expected.begin(), expected.end(), llama_get_logits_ith(ctx, -1));
    }

    // ---- KV touch: occupy the pages a cache fill of N tokens would use (memory-only, no compute) ----
    struct kv_touch_result {
        bool ran = false;
        int64_t tokens = 0;
        size_t bytes = 0, rss_anon_before = 0, rss_anon_after = 0, file_rss_before = 0, file_rss_after = 0;
        size_t majflt_before = 0, majflt_after = 0, vma_count = 0;
        long long elapsed_us = 0;
    } kv_touch;
    auto file_rss_now = [&](size_t & vma_count) {
        size_t total = 0;
        std::vector<vma> areas = model_vmas(real_model);
        vma_count = areas.size();
        for (const auto & area : areas) total += area.rss_bytes;
        return total;
    };
    auto do_kv_touch = [&](llama_context * target) {
        kv_touch.ran = true;
        kv_touch.tokens = opts.kv_touch_tokens;
        kv_touch.rss_anon_before = rss_anon_bytes();
        kv_touch.file_rss_before = file_rss_now(kv_touch.vma_count);
        kv_touch.majflt_before = major_faults();
        auto t0 = std::chrono::steady_clock::now();
        kv_touch.bytes = llama_kv_touch_cells(target, (uint32_t) opts.kv_touch_tokens);
        kv_touch.elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
        kv_touch.rss_anon_after = rss_anon_bytes();
        kv_touch.file_rss_after = file_rss_now(kv_touch.vma_count);
        kv_touch.majflt_after = major_faults();
    };
    if (opts.kv_touch_tokens > 0 && opts.dormant_mask == 0 && ctx != nullptr && decode_status == 0) {
        do_kv_touch(ctx);
    }

    // ---- clear(true) + reuse: the lazily zeroed KV must give its pages back and decode identically ----
    struct clear_reuse_result {
        bool ran = false;
        size_t rss_anon_after_decode = 0, rss_anon_after_clear = 0, rss_anon_after_reuse = 0;
        int decode_status = -1;
        bool logits_identical = false;
        float max_abs_diff = 0.0f;
        long long clear_us = 0;
    } clear_reuse;
    if (opts.clear_reuse && ctx != nullptr && decode_status == 0) {
        clear_reuse.ran = true;
        clear_reuse.rss_anon_after_decode = rss_anon_bytes();
        auto t0 = std::chrono::steady_clock::now();
        llama_memory_clear(llama_get_memory(ctx), true);
        clear_reuse.clear_us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
        clear_reuse.rss_anon_after_clear = rss_anon_bytes();
        std::vector<float> logits_again;
        std::vector<int> argmax_again;
        clear_reuse.decode_status = decode_once(ctx, logits_again, argmax_again);
        clear_reuse.rss_anon_after_reuse = rss_anon_bytes();
        clear_reuse.logits_identical = clear_reuse.decode_status == 0 && logits_again.size() == all_logits.size() &&
                std::equal(logits_again.begin(), logits_again.end(), all_logits.begin());
        for (size_t i = 0; i < std::min(logits_again.size(), all_logits.size()); ++i) {
            clear_reuse.max_abs_diff = std::max(clear_reuse.max_abs_diff, std::fabs(logits_again[i] - all_logits[i]));
        }
    }

    // ---- dormant host share: release the phone share, execute locally again, populate ----
    struct dormant_result {
        bool ran = false;
        size_t released_bytes = 0, restored_bytes = 0, range_count = 0;
        size_t rss_before = 0, rss_after = 0, rss_restored = 0;
        size_t vma_count_before = 0, vma_count_after = 0;
        int decode_status = -1;
        bool logits_identical = false;
        bool argmax_identical = false;
        float max_abs_diff = 0.0f;
        long long release_us = 0, decode_us = 0, restore_us = 0;
        size_t consumed_bytes = 0, rss_anon_before_consume = 0, rss_anon_after_consume = 0;
    } dormant;
    std::vector<uint8_t> consumer;
    if (opts.dormant_mask != 0 && ctx != nullptr && decode_status == 0) {
        auto file_rss = [&](size_t & vma_count) {
            size_t total = 0;
            std::vector<vma> areas = model_vmas(real_model);
            vma_count = areas.size();
            for (const auto & area : areas) {
                total += area.rss_bytes;
            }
            return total;
        };
        dormant.ran = true;
        dormant.rss_before = file_rss(dormant.vma_count_before);
        auto t0 = std::chrono::steady_clock::now();
        dormant.released_bytes = llama_model_ffn_host_share_release(model, opts.dormant_mask, opts.dormant_host_columns);
        dormant.range_count = llama_model_ffn_host_share_range_count(model);
        dormant.release_us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
        dormant.rss_after = file_rss(dormant.vma_count_after);
        if (opts.dormant_consume && dormant.released_bytes > 0) {
            // occupy the released room with touched anonymous memory (a stand-in for KV growth) and
            // keep it through the second decode and the restore
            dormant.rss_anon_before_consume = rss_anon_bytes();
            consumer.assign(dormant.released_bytes, 0);
            for (size_t i = 0; i < consumer.size(); i += 4096) {
                consumer[i] = 1;
            }
            dormant.consumed_bytes = consumer.size();
            dormant.rss_anon_after_consume = rss_anon_bytes();
        }
        if (opts.kv_touch_tokens > 0) {
            // the KV of the existing context grows into the released room: touch its pages now
            do_kv_touch(ctx);
        }
        auto restore_share = [&]() {
            const auto start = std::chrono::steady_clock::now();
            dormant.restored_bytes = llama_model_ffn_host_share_restore(model);
            dormant.restore_us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
            size_t ignored = 0;
            dormant.rss_restored = file_rss(ignored);
        };
        if (opts.dormant_restore_before_decode) {
            restore_share();
        }
        // the released pages fault back in from the file as the local FFN touches them
        llama_context * again = opts.dormant_no_decode ? nullptr : llama_init_from_model(model, cparams);
        std::vector<float> logits_again;
        std::vector<int> argmax_again;
        t0 = std::chrono::steady_clock::now();
        dormant.decode_status = opts.dormant_no_decode ? 0 : (again == nullptr ? -1 : decode_once(again, logits_again, argmax_again));
        dormant.decode_us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
        if (again != nullptr) {
            llama_free(again);
        }
        dormant.logits_identical = !opts.dormant_no_decode && dormant.decode_status == 0 && logits_again.size() == all_logits.size() &&
                std::equal(logits_again.begin(), logits_again.end(), all_logits.begin());
        dormant.argmax_identical = !opts.dormant_no_decode && dormant.decode_status == 0 && argmax_again == argmax_trace;
        for (size_t i = 0; i < std::min(logits_again.size(), all_logits.size()); ++i) {
            dormant.max_abs_diff = std::max(dormant.max_abs_diff, std::fabs(logits_again[i] - all_logits[i]));
        }
        if (!opts.dormant_restore_before_decode) {
            restore_share();
        }
    }

    if (!opts.logits.empty() && !all_logits.empty()) {
        FILE * f = fopen(opts.logits.c_str(), "wb");
        if (f != nullptr) {
            fwrite(all_logits.data(), sizeof(float), all_logits.size(), f);
            fclose(f);
        }
    }

    // ---- JSON ----
    std::ostringstream json;
    json << "{\n";
    json << "  \"schema\": \"s42-remote-resident-probe-v1\",\n";
    json << "  \"mode\": \"" << (remote ? "remote" : "full") << "\",\n";
    json << "  \"model\": \"" << json_escape(real_model) << "\",\n";
    json << "  \"remote_mask\": " << opts.remote_mask << ",\n";
    json << "  \"gpu_layers\": " << opts.gpu_layers << ",\n";
    json << "  \"runtime_batches\": [";
    for (size_t i = 0; i < attribution.batches.size(); ++i) {
        json << (i ? "," : "") << "[";
        const auto & entries = attribution.batches[i];
        for (size_t j = 0; j < entries.size(); ++j) {
            const auto & entry = entries[j];
            json << (j ? "," : "") << "{\"request_id\":\"" << entry.request_id
                 << "\",\"slot_id\":" << entry.slot_id << ",\"rows\":" << entry.rows
                 << ",\"plan_generation\":" << entry.plan_generation << "}";
        }
        json << "]";
    }
    json << "],\n  \"request_rows\": [";
    for (uint32_t i = 0; i < opts.sequence_count; ++i) {
        json << (i ? "," : "") << (client ? client->summary({"probe-" + std::to_string(i)}).input_rows : 0);
    }
    json << "],\n";
    json << "  \"loader\": {\"remote_mask\": " << loaded_mask << ", \"omitted_bytes\": " << omitted_bytes
         << ", \"unmapped_bytes\": " << unmapped_bytes << ", \"model_size_bytes\": " << llama_model_size(model) << "},\n";
    json << "  \"file_mapping\": {\"vma_count\": " << vmas.size() << ", \"mapped_bytes\": " << file_mapped_bytes
         << ", \"rss_bytes\": " << file_rss_bytes << "},\n";
    json << "  \"omitted_tensors\": [";
    for (size_t i = 0; i < ranges.size(); ++i) {
        const auto & r = ranges[i];
        json << (i ? ",\n    " : "\n    ") << "{\"name\": \"" << r.name << "\", \"offs\": " << r.offs << ", \"nbytes\": " << r.nbytes
             << ", \"page_first\": " << r.page_first << ", \"page_last\": " << r.page_last
             << ", \"vma_overlap_bytes\": " << r.vma_overlap_bytes << ", \"rss_overlap_bytes\": " << r.rss_overlap_bytes << "}";
    }
    json << (ranges.empty() ? "" : "\n  ") << "],\n";
    json << "  \"context_created\": " << (context_created ? "true" : "false") << ",\n";
    json << "  \"context_rss_anon_bytes\": " << context_rss_anon_bytes << ",\n";
    json << "  \"expect_no_owner\": " << (opts.expect_no_owner ? "true" : "false") << ",\n";
    json << "  \"client\": {\"connected\": " << (client_connected ? "true" : "false") << ", \"error\": \""
         << json_escape(client_error) << "\", \"remote_policy_rejection\": \"" << json_escape(policy_rejection) << "\"},\n";
    json << "  \"dormant\": {\"ran\": " << (dormant.ran ? "true" : "false") << ", \"layer_mask\": " << opts.dormant_mask
         << ", \"drop_cache\": " << (opts.dormant_drop_cache ? "true" : "false")
         << ", \"populate\": " << (opts.dormant_populate ? "true" : "false")
         << ", \"restore_before_decode\": " << (opts.dormant_restore_before_decode ? "true" : "false")
         << ", \"host_columns\": " << opts.dormant_host_columns << ", \"released_bytes\": " << dormant.released_bytes
         << ", \"restored_bytes\": " << dormant.restored_bytes << ", \"range_count\": " << dormant.range_count
         << ", \"rss_before\": " << dormant.rss_before << ", \"rss_after\": " << dormant.rss_after
         << ", \"rss_restored\": " << dormant.rss_restored << ", \"vma_count_before\": " << dormant.vma_count_before
         << ", \"vma_count_after\": " << dormant.vma_count_after << ", \"decode_status\": " << dormant.decode_status
         << ", \"logits_identical\": " << (dormant.logits_identical ? "true" : "false")
         << ", \"argmax_identical\": " << (dormant.argmax_identical ? "true" : "false")
         << ", \"max_abs_diff\": " << dormant.max_abs_diff << ", \"release_us\": " << dormant.release_us
         << ", \"decode_us\": " << dormant.decode_us << ", \"restore_us\": " << dormant.restore_us
         << ", \"consumed_bytes\": " << dormant.consumed_bytes << ", \"rss_anon_before_consume\": " << dormant.rss_anon_before_consume
         << ", \"rss_anon_after_consume\": " << dormant.rss_anon_after_consume << "},\n";
    json << "  \"kv_touch\": {\"ran\": " << (kv_touch.ran ? "true" : "false") << ", \"tokens\": " << kv_touch.tokens
         << ", \"bytes\": " << kv_touch.bytes << ", \"rss_anon_before\": " << kv_touch.rss_anon_before
         << ", \"rss_anon_after\": " << kv_touch.rss_anon_after << ", \"file_rss_before\": " << kv_touch.file_rss_before
         << ", \"file_rss_after\": " << kv_touch.file_rss_after << ", \"majflt_before\": " << kv_touch.majflt_before
         << ", \"majflt_after\": " << kv_touch.majflt_after << ", \"elapsed_us\": " << kv_touch.elapsed_us << "},\n";
    json << "  \"clear_reuse\": {\"ran\": " << (clear_reuse.ran ? "true" : "false")
         << ", \"rss_anon_after_decode\": " << clear_reuse.rss_anon_after_decode
         << ", \"rss_anon_after_clear\": " << clear_reuse.rss_anon_after_clear
         << ", \"rss_anon_after_reuse\": " << clear_reuse.rss_anon_after_reuse
         << ", \"decode_status\": " << clear_reuse.decode_status
         << ", \"logits_identical\": " << (clear_reuse.logits_identical ? "true" : "false")
         << ", \"max_abs_diff\": " << clear_reuse.max_abs_diff << ", \"clear_us\": " << clear_reuse.clear_us << "},\n";
    json << "  \"state_reuse\": {\"ran\": " << (state_reuse.ran ? "true" : "false")
         << ", \"bytes\": " << state_reuse.bytes << ", \"restored_bytes\": " << state_reuse.restored_bytes
         << ", \"decode_status\": " << state_reuse.decode_status
         << ", \"logits_identical\": " << (state_reuse.logits_identical ? "true" : "false") << "},\n";
    json << "  \"decode\": {\"status\": " << decode_status << ", \"error\": \"" << json_escape(decode_error)
         << "\", \"n_vocab\": " << n_vocab << ", \"prompt_tokens\": " << opts.tokens.size() << ", \"steps\": " << argmax_trace.size()
         << ", \"argmax\": [";
    for (size_t i = 0; i < argmax_trace.size(); ++i) {
        json << (i ? ", " : "") << argmax_trace[i];
    }
    json << "]},\n";
    json << "  \"log\": [";
    for (size_t i = 0; i < capture.lines.size(); ++i) {
        json << (i ? ",\n    " : "\n    ") << "\"" << json_escape(capture.lines[i]) << "\"";
    }
    json << (capture.lines.empty() ? "" : "\n  ") << "]\n";
    json << "}\n";

    std::ofstream out(opts.out);
    out << json.str();
    out.close();

    if (ctx != nullptr) {
        llama_free(ctx);
    }
    if (client) {
        client->finish();
    }
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
