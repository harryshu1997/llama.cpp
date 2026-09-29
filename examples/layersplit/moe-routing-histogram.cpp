// Expert-routing histogram for MoE models.
//
// Runs a text file through the model in independent chunks and counts, per layer, how often each
// expert is selected (the "ffn_moe_topk-<il>" tensor observed through cb_eval). The output JSON is
// the input for a coverage curve: how many layer-token expert selections a resident-expert tier of a
// given byte size could serve if it held the most frequently selected experts.
//
// Usage: llama-moe-routing-histogram -m model.gguf -f text.txt [-c 2048 -b 2048 -ub 512 -ngl 99
//        -ot exps=CPU -t 16] --out histogram.json [--max-tokens N]

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

struct routing_tally {
    std::map<int, std::vector<long long>> counts;     // layer -> per-expert selection count
    std::map<int, std::vector<long long>> positions;  // layer -> count of selections at rank r (0 = top-1)
    long long observed_tensors = 0;
    long long observed_rows    = 0;
    int n_expert      = 0;
    int n_expert_used = 0;
};

static bool routing_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    routing_tally * tally = (routing_tally *) user_data;
    const char * prefix = "ffn_moe_topk-";
    const bool match = strncmp(t->name, prefix, strlen(prefix)) == 0;
    if (ask) {
        return match;
    }
    if (!match) {
        return true;
    }
    if (t->type != GGML_TYPE_I32 || t->ne[0] <= 0 || t->ne[1] <= 0) {
        LOG_ERR("%s: unexpected tensor %s type %s\n", __func__, t->name, ggml_type_name(t->type));
        return true;
    }
    const int il = atoi(t->name + strlen(prefix));
    const int n_used   = (int) t->ne[0];
    const int n_tokens = (int) t->ne[1];
    std::vector<int32_t> ids((size_t) n_used * n_tokens);
    // The tensor may be non-contiguous (a view); read row by row.
    for (int j = 0; j < n_tokens; ++j) {
        ggml_backend_tensor_get(t, ids.data() + (size_t) j * n_used, (size_t) j * t->nb[1], (size_t) n_used * sizeof(int32_t));
    }
    auto & counts = tally->counts[il];
    auto & positions = tally->positions[il];
    if (positions.size() < (size_t) n_used) {
        positions.resize(n_used, 0);
    }
    for (int j = 0; j < n_tokens; ++j) {
        for (int r = 0; r < n_used; ++r) {
            const int32_t e = ids[(size_t) j * n_used + r];
            if (e < 0) {
                continue;
            }
            if ((size_t) e >= counts.size()) {
                counts.resize((size_t) e + 1, 0);
            }
            counts[e]++;
            positions[r]++;
        }
    }
    tally->n_expert_used = std::max(tally->n_expert_used, n_used);
    tally->observed_tensors++;
    tally->observed_rows += n_tokens;
    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    std::string out_path = "moe_routing_histogram.json";
    long long max_tokens = 0;

    // Pull our own flags out before common parsing.
    std::vector<char *> args;
    for (int i = 0; i < argc; ++i) {
        if (i + 1 < argc && strcmp(argv[i], "--out") == 0) {
            out_path = argv[++i];
            continue;
        }
        if (i + 1 < argc && strcmp(argv[i], "--max-tokens") == 0) {
            max_tokens = atoll(argv[++i]);
            continue;
        }
        args.push_back(argv[i]);
    }
    if (!common_params_parse((int) args.size(), args.data(), params, LLAMA_EXAMPLE_PERPLEXITY)) {
        return 1;
    }
    if (params.prompt_file.empty()) {
        LOG_ERR("a text file is required (-f)\n");
        return 1;
    }

    routing_tally tally;
    params.cb_eval = routing_cb_eval;
    params.cb_eval_user_data = &tally;
    params.warmup = false;

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    llama_context * ctx = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("failed to load model\n");
        return 1;
    }
    std::ifstream in(params.prompt_file, std::ios::binary);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::vector<llama_token> tokens = common_tokenize(ctx, text, true, true);
    if (max_tokens > 0 && (long long) tokens.size() > max_tokens) {
        tokens.resize((size_t) max_tokens);
    }
    LOG_INF("tokens: %zu\n", tokens.size());

    const int n_ctx   = (int) llama_n_ctx(ctx);
    const int n_batch = params.n_batch;
    const int chunk   = std::min(n_ctx, n_batch);
    const auto t_start = std::chrono::steady_clock::now();
    size_t processed = 0;
    for (size_t off = 0; off < tokens.size(); off += (size_t) chunk) {
        const int n = (int) std::min<size_t>((size_t) chunk, tokens.size() - off);
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_batch batch = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; ++i) {
            // Request logits for every token: otherwise only the output rows reach the last layer and its
            // routing would be observed for one token per chunk.
            common_batch_add(batch, tokens[off + i], i, { 0 }, true);
        }
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("llama_decode failed at offset %zu\n", off);
            llama_batch_free(batch);
            return 1;
        }
        llama_batch_free(batch);
        processed += (size_t) n;
        if ((off / chunk) % 8 == 0) {
            const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
            LOG_INF("processed %zu / %zu tokens (%.1f tok/s)\n", processed, tokens.size(), processed / std::max(el, 1e-9));
        }
    }
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();

    const int n_layer = llama_model_n_layer(model);
    int n_expert = 0;
    for (auto & kv : tally.counts) {
        n_expert = std::max(n_expert, (int) kv.second.size());
    }
    {
        char buf[64];
        if (llama_model_meta_val_str(model, "qwen3moe.expert_count", buf, sizeof(buf)) > 0) {
            n_expert = std::max(n_expert, atoi(buf));
        }
    }
    tally.n_expert = n_expert;

    FILE * f = fopen(out_path.c_str(), "w");
    if (!f) {
        LOG_ERR("cannot open %s\n", out_path.c_str());
        return 1;
    }
    fprintf(f, "{\n");
    fprintf(f, "  \"model\": \"%s\",\n", params.model.path.c_str());
    fprintf(f, "  \"text\": \"%s\",\n", params.prompt_file.c_str());
    fprintf(f, "  \"tokens\": %zu,\n", tokens.size());
    fprintf(f, "  \"chunk_tokens\": %d,\n", chunk);
    fprintf(f, "  \"elapsed_s\": %.3f,\n", elapsed);
    fprintf(f, "  \"n_layer\": %d,\n", n_layer);
    fprintf(f, "  \"n_expert\": %d,\n", n_expert);
    fprintf(f, "  \"n_expert_used\": %d,\n", tally.n_expert_used);
    fprintf(f, "  \"observed_tensors\": %lld,\n", tally.observed_tensors);
    fprintf(f, "  \"observed_rows\": %lld,\n", tally.observed_rows);
    fprintf(f, "  \"counts\": {");
    bool first_layer = true;
    for (auto & kv : tally.counts) {
        fprintf(f, "%s\n    \"%d\": [", first_layer ? "" : ",", kv.first);
        first_layer = false;
        std::vector<long long> c = kv.second;
        c.resize(n_expert, 0);
        for (int e = 0; e < n_expert; ++e) {
            fprintf(f, "%s%lld", e ? "," : "", c[e]);
        }
        fprintf(f, "]");
    }
    fprintf(f, "\n  }\n}\n");
    fclose(f);
    LOG_INF("wrote %s: %lld tensors, %lld rows, %d experts, %d used\n", out_path.c_str(), tally.observed_tensors, tally.observed_rows, n_expert, tally.n_expert_used);

    llama_backend_free();
    return 0;
}
