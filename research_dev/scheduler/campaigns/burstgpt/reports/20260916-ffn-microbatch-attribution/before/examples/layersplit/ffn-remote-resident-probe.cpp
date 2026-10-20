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
#include "gguf.h"
#include "ggml.h"

#include "ffn-split-client.h"

#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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
    uint32_t max_tokens = 16;
    std::vector<llama_token> tokens = {1, 2, 3, 4, 5, 6, 7, 8};
    int decode_steps = 4;
    bool expect_no_owner = false;
    int n_threads = 2;
    int gpu_layers = 0;
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
        } else if (arg == "--max-tokens") {
            if (!next(value) || !parse_u64(value.c_str(), number) || number == 0 || number > UINT16_MAX) return false;
            result.max_tokens = (uint32_t) number;
        } else if (arg == "--gpu-layers") {
            if (!next(value) || !parse_u64(value.c_str(), number) || number > 1024) return false;
            result.gpu_layers = (int) number;
        } else if (arg == "--threads") {
            if (!next(value) || !parse_u64(value.c_str(), number) || number == 0 || number > 256) return false;
            result.n_threads = (int) number;
        } else if (arg == "--decode") {
            if (!next(value) || !parse_u64(value.c_str(), number) || number > 4096) return false;
            result.decode_steps = (int) number;
        } else if (arg == "--tokens") {
            if (!next(value)) return false;
            result.tokens.clear();
            std::stringstream stream(value);
            std::string item;
            while (std::getline(stream, item, ',')) {
                if (item.empty() || !parse_u64(item.c_str(), number)) return false;
                result.tokens.push_back((llama_token) number);
            }
            if (result.tokens.empty()) return false;
        } else if (arg == "--expect-no-owner") {
            result.expect_no_owner = true;
        } else {
            return false;
        }
    }
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

} // namespace

int main(int argc, char ** argv) {
    options opts;
    if (!parse_options(argc, argv, opts)) {
        fprintf(stderr, "usage: %s --model PATH --out JSON [--logits PATH] [--remote-mask MASK --worker-host H "
                        "--worker-port P --artifact-sha256 sha256:HEX --max-tokens N] [--tokens 1,2,3] [--decode N] "
                        "[--expect-no-owner] [--threads N] [--gpu-layers N]\n", argv[0]);
        return 2;
    }
    const bool remote = opts.remote_mask != 0;
    if (remote) {
        setenv("LLAMA_FFN_REMOTE_RESIDENT_LAYER_MASK", std::to_string(opts.remote_mask).c_str(), 1);
    } else {
        unsetenv("LLAMA_FFN_REMOTE_RESIDENT_LAYER_MASK");
    }
    // no assisted-copy split in this probe: the remote layers are the only phone work
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
    if (remote && !omitted_ranges(opts.model, opts.remote_mask, ranges, geometry, error)) {
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
    if (remote && !opts.expect_no_owner) {
        ffn_split::client_config config;
        config.transport = ffn_split::client_transport::tcp;
        config.artifact_sha256 = opts.artifact_sha256;
        config.host = opts.worker_host;
        config.port = opts.worker_port;
        config.layer_mask = opts.remote_mask;
        config.remote_resident_layer_mask = opts.remote_mask;
        config.max_columns = geometry.n_ff;
        config.n_embd = geometry.n_embd;
        config.max_tokens = (uint16_t) opts.max_tokens;
        config.f16_io = false;
        // the worker derives the activation from the shard architecture: gemma4 is GEGLU
        config.swiglu = geometry.arch != "gemma4";
        config.runtime_control = true;
        config.timeout_ms = 20000;
        client = std::make_unique<ffn_split::client>(std::move(config));
        client_connected = client->connect(client_error);
        if (!client_connected) {
            fprintf(stderr, "probe: FFN worker connect failed: %s\n", client_error.c_str());
        } else {
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
    cparams.n_ctx = 128;
    cparams.n_batch = opts.max_tokens;
    cparams.n_ubatch = opts.max_tokens;
    cparams.n_seq_max = 1;
    cparams.n_threads = opts.n_threads;
    cparams.n_threads_batch = opts.n_threads;
    cparams.no_perf = true;
    if (client && client_connected) {
        cparams.cb_eval = client_eval_cb;
        cparams.cb_eval_user_data = client.get();
    }
    llama_context * ctx = (remote && !opts.expect_no_owner && !client_connected) ? nullptr : llama_init_from_model(model, cparams);
    const bool context_created = ctx != nullptr;

    // ---- decode ----
    std::vector<int> argmax_trace;
    std::vector<float> all_logits;
    int decode_status = -1;
    std::string decode_error;
    int n_vocab = 0;
    if (ctx != nullptr) {
        const llama_vocab * vocab = llama_model_get_vocab(model);
        n_vocab = llama_vocab_n_tokens(vocab);
        std::vector<llama_token> tokens = opts.tokens;
        llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t) tokens.size());
        decode_status = llama_decode(ctx, batch);
        for (int step = 0; decode_status == 0 && step <= opts.decode_steps; ++step) {
            const float * logits = llama_get_logits_ith(ctx, -1);
            int best = 0;
            for (int t = 1; t < n_vocab; ++t) {
                if (logits[t] > logits[best]) {
                    best = t;
                }
            }
            argmax_trace.push_back(best);
            all_logits.insert(all_logits.end(), logits, logits + n_vocab);
            if (step == opts.decode_steps) {
                break;
            }
            llama_token next = (llama_token) best;
            batch = llama_batch_get_one(&next, 1);
            decode_status = llama_decode(ctx, batch);
        }
        if (decode_status != 0) {
            decode_error = "llama_decode returned " + std::to_string(decode_status);
        }
        if (client && client->failed()) {
            decode_error += (decode_error.empty() ? "" : "; ") + std::string("client: ") + client->error();
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
    json << "  \"expect_no_owner\": " << (opts.expect_no_owner ? "true" : "false") << ",\n";
    json << "  \"client\": {\"connected\": " << (client_connected ? "true" : "false") << ", \"error\": \""
         << json_escape(client_error) << "\", \"remote_policy_rejection\": \"" << json_escape(policy_rejection) << "\"},\n";
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
