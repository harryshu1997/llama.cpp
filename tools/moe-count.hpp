// Research instrumentation: counts per-layer / per-expert routing selections
// during MoE inference by riding on the existing named-tensor eval callback
// ("ffn_moe_topk-<il>") produced by build_moe_ffn in src/llama-graph.cpp.
//
// Enable with env MOE_COUNT=1. Install with moe_count_maybe_enable(params)
// before common_init_from_params(). Print at shutdown with moe_count_report().

#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "common.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <vector>

struct moe_usage_counter {
    std::map<int, std::vector<uint64_t>> counts;
    // Per-MoE-layer count of tokens that ran through the always-on shared dense MLP.
    // In Gemma 4 (and some other archs) each MoE layer has a parallel dense FFN that
    // processes every token — we record n_tokens each time ffn_moe_topk fires for that layer.
    std::map<int, uint64_t> shared_counts;
    std::mutex mtx;
    uint64_t total_routed_tokens = 0;

    static int parse_layer_from_name(const char * name) {
        const char * dash = std::strrchr(name, '-');
        if (!dash) return -1;
        char * end = nullptr;
        long il = std::strtol(dash + 1, &end, 10);
        if (end == dash + 1) return -1;
        return (int) il;
    }

    void record(const int32_t * ids, int64_t n_expert_used, int64_t n_tokens, int il) {
        std::lock_guard<std::mutex> lk(mtx);
        auto & row = counts[il];
        for (int64_t t = 0; t < n_tokens; ++t) {
            for (int64_t k = 0; k < n_expert_used; ++k) {
                int32_t e = ids[t * n_expert_used + k];
                if (e < 0) continue;
                if ((size_t) e >= row.size()) row.resize((size_t) e + 1, 0);
                row[(size_t) e]++;
                total_routed_tokens++;
            }
        }
        // same layer ran its shared dense MLP for every token in this call
        shared_counts[il] += (uint64_t) n_tokens;
    }

    void print(FILE * f) const {
        if (counts.empty()) {
            fprintf(f, "[moe-count] no MoE routing observed — model may not be MoE, or tensor name did not match 'ffn_moe_topk'.\n");
            return;
        }

        // Resolve output CSV path: $MOE_COUNT_CSV overrides, else research_dev/log/moe_count_<ts>.csv
        const char * env_path = std::getenv("MOE_COUNT_CSV");
        std::string out_path;
        if (env_path && env_path[0]) {
            out_path = env_path;
        } else {
            mkdir("research_dev", 0755);
            mkdir("research_dev/log", 0755);
            char ts[32];
            std::time_t t = std::time(nullptr);
            std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", std::localtime(&t));
            out_path = std::string("research_dev/log/moe_count_") + ts + ".csv";
        }

        FILE * g = std::fopen(out_path.c_str(), "w");
        if (!g) {
            fprintf(f, "[moe-count] failed to open %s for writing\n", out_path.c_str());
            return;
        }
        fprintf(g, "layer,expert,count\n");
        for (const auto & kv : counts) {
            for (size_t e = 0; e < kv.second.size(); ++e) {
                fprintf(g, "%d,%zu,%" PRIu64 "\n", kv.first, e, kv.second[e]);
            }
        }
        // shared dense MLP rows: expert = -1
        for (const auto & kv : shared_counts) {
            fprintf(g, "%d,-1,%" PRIu64 "\n", kv.first, kv.second);
        }
        std::fclose(g);

        fprintf(f, "[moe-count] %" PRIu64 " routed assignments across %zu layers -> %s\n",
                total_routed_tokens, counts.size(), out_path.c_str());
    }
};

inline moe_usage_counter & moe_count_singleton() {
    static moe_usage_counter c;
    return c;
}

inline bool & moe_count_enabled_flag() {
    static bool b = false;
    return b;
}

inline bool moe_count_cb_eval(struct ggml_tensor * t, bool ask, void * /*user_data*/) {
    if (ask) {
        return t->name[0] != '\0' && std::strncmp(t->name, "ffn_moe_topk", 12) == 0;
    }
    if (t->type != GGML_TYPE_I32) return true;
    int il = moe_usage_counter::parse_layer_from_name(t->name);
    if (il < 0) return true;

    const int64_t n_expert_used = t->ne[0];
    const int64_t n_tokens      = t->ne[1];
    const size_t  nbytes        = ggml_nbytes(t);

    std::vector<int32_t> buf(nbytes / sizeof(int32_t));
    ggml_backend_tensor_get(t, buf.data(), 0, nbytes);
    moe_count_singleton().record(buf.data(), n_expert_used, n_tokens, il);
    return true;
}

inline void moe_count_maybe_enable(common_params & params) {
    const char * env = std::getenv("MOE_COUNT");
    if (env && env[0] && env[0] != '0') {
        moe_count_enabled_flag() = true;
        params.cb_eval = moe_count_cb_eval;
        params.cb_eval_user_data = nullptr;
        fprintf(stderr, "[moe-count] enabled — will report expert usage at shutdown\n");
    }
}

inline void moe_count_report(FILE * f = stderr) {
    if (moe_count_enabled_flag()) {
        moe_count_singleton().print(f);
    }
}
