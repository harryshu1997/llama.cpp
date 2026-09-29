#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

static ggml_tensor * output_view(ggml_context * ctx, ggml_tensor * partial, int64_t width) {
    return ggml_view_4d(ctx, partial, width, partial->ne[1], partial->ne[2], partial->ne[3],
                        partial->nb[1], partial->nb[2], partial->nb[3], 0);
}

static ggml_tensor * lse_view(ggml_context * ctx, ggml_tensor * partial, int64_t width) {
    auto * view = ggml_view_4d(ctx, partial, 1, partial->ne[1], partial->ne[2], partial->ne[3],
                             partial->nb[1], partial->nb[2], partial->nb[3], width * sizeof(float));
    return ggml_clamp(ctx, ggml_cont(ctx, view), -1e30f, 1e30f);
}

static bool run_case(ggml_backend_t backend, int64_t width, int64_t heads, int64_t ratio,
                     int64_t cells, int64_t tokens, bool sinks, float softcap, bool all_masked = false) {
    ggml_init_params params = {ggml_tensor_overhead() * 128 + ggml_graph_overhead_custom(128, false), nullptr, true};
    auto * ctx = ggml_init(params);
    auto * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, tokens, heads * ratio);
    auto * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, width, cells, heads);
    auto * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, width, cells, heads);
    auto * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, cells, tokens);
    auto * s = sinks ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, heads * ratio) : nullptr;
    const float scale = 1.0f / std::sqrt(float(width));
    auto * full = ggml_flash_attn_ext(ctx, q, k, v, m, scale, 0, softcap);
    auto * flagged = ggml_flash_attn_ext_with_lse(ctx, q, k, v, m, scale, 0, softcap);
    ggml_flash_attn_ext_add_sinks(full, s);
    ggml_flash_attn_ext_add_sinks(flagged, s);
    auto * plain_flagged = ggml_cont(ctx, output_view(ctx, flagged, width));
    const int64_t cut = cells / 2;
    ggml_tensor * partial[2];
    for (int i = 0; i < 2; ++i) {
        const int64_t begin = i ? cut : 0;
        const int64_t count = i ? cells - cut : cut;
        auto * ks = ggml_view_3d(ctx, k, width, count, heads, k->nb[1], k->nb[2], begin * k->nb[1]);
        auto * vs = ggml_view_3d(ctx, v, width, count, heads, v->nb[1], v->nb[2], begin * v->nb[1]);
        auto * ms = ggml_cont(ctx, ggml_view_2d(ctx, m, count, tokens, m->nb[1], begin * m->nb[0]));
        partial[i] = ggml_flash_attn_ext_with_lse(ctx, q, ks, vs, ms, scale, 0, softcap);
        if (i == 0) {
            ggml_flash_attn_ext_add_sinks(partial[i], s);
        }
    }
    auto * l0 = lse_view(ctx, partial[0], width);
    auto * l1 = lse_view(ctx, partial[1], width);
    auto * w0 = ggml_sigmoid(ctx, ggml_sub(ctx, l0, l1));
    auto * w1 = ggml_sigmoid(ctx, ggml_sub(ctx, l1, l0));
    auto * merged = ggml_add(ctx, ggml_mul(ctx, output_view(ctx, partial[0], width), w0),
                                 ggml_mul(ctx, output_view(ctx, partial[1], width), w1));
    auto * graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(graph, full);
    ggml_build_forward_expand(graph, plain_flagged);
    ggml_build_forward_expand(graph, merged);
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        if (!ggml_backend_supports_op(backend, ggml_graph_node(graph, i))) {
            std::printf("unsupported node %s\n", ggml_op_desc(ggml_graph_node(graph, i)));
            ggml_free(ctx);
            return false;
        }
    }
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer);
    std::vector<float> queries(ggml_nelements(q));
    std::vector<ggml_fp16_t> keys(ggml_nelements(k)), values(ggml_nelements(v)), mask(ggml_nelements(m));
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> sample(-1.0f, 1.0f);
    for (size_t i = 0; i < queries.size(); ++i) queries[i] = float(int(i % 31) - 15) / 32.0f;
    for (size_t i = 0; i < keys.size(); ++i) {
        keys[i] = ggml_fp32_to_fp16(sample(rng));
        values[i] = ggml_fp32_to_fp16(sample(rng));
    }
    for (int64_t t = 0; t < tokens; ++t) {
        const int64_t end = tokens == 1 ? cells : std::max<int64_t>(1, (t + 1) * cells / tokens);
        for (int64_t c = 0; c < cells; ++c) {
            mask[t * cells + c] = ggml_fp32_to_fp16(!all_masked && c < end ? 0.0f : -INFINITY);
        }
    }
    ggml_backend_tensor_set(q, queries.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, keys.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(v, values.data(), 0, ggml_nbytes(v));
    ggml_backend_tensor_set(m, mask.data(), 0, ggml_nbytes(m));
    if (s) {
        std::vector<float> sink_values(heads * ratio, 0.25f);
        ggml_backend_tensor_set(s, sink_values.data(), 0, ggml_nbytes(s));
    }
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> reference(ggml_nelements(full)), actual(reference.size()), same(reference.size());
    ggml_backend_tensor_get(full, reference.data(), 0, ggml_nbytes(full));
    ggml_backend_tensor_get(plain_flagged, same.data(), 0, ggml_nbytes(plain_flagged));
    ggml_backend_tensor_get(merged, actual.data(), 0, ggml_nbytes(merged));
    if (all_masked) {
        // Empty partials have zero output and -inf LSE, including when both sides are empty.
        std::fill(reference.begin(), reference.end(), 0.0f);
    }
    double error = 0, flag_error = 0, magnitude = 0, max_abs = 0;
    bool ok = true;
    for (size_t i = 0; i < reference.size(); ++i) {
        ok &= std::isfinite(actual[i]) && std::isfinite(same[i]);
        error += std::pow(double(actual[i]) - reference[i], 2);
        flag_error += std::pow(double(same[i]) - reference[i], 2);
        magnitude += std::pow(double(reference[i]), 2);
        max_abs = std::max(max_abs, std::abs(double(actual[i]) - reference[i]));
    }
    double sum = sinks ? std::exp(0.25) : 0;
    const int64_t end = tokens == 1 ? cells : std::max<int64_t>(1, cells / tokens);
    for (int64_t c = 0; !all_masked && c < end; ++c) {
        double dot = 0;
        for (int64_t d = 0; d < width; ++d) dot += queries[d] * ggml_fp16_to_fp32(keys[c * width + d]);
        const double logit = softcap ? softcap * std::tanh(dot * scale / softcap) : dot * scale;
        sum += std::exp(logit);
    }
    float lse;
    ggml_backend_tensor_get(flagged, &lse, width * sizeof(float), sizeof(float));
    const double lse_error = sum == 0 && lse == -INFINITY ? 0 : std::abs(double(lse) - std::log(sum));
    const double nmse = error / std::max(magnitude, 1e-30);
    const double flag_nmse = flag_error / std::max(magnitude, 1e-30);
    ok &= nmse <= 5e-4 && flag_nmse <= 5e-4 && lse_error < 0.01;
    std::printf("%s D=%lld H=%lld/%lld K=%lld Q=%lld sinks=%d softcap=%.1f empty=%d nmse=%.9g flag_nmse=%.9g max_abs=%.9g lse_error=%.9g %s\n",
                ggml_backend_name(backend), (long long) width, (long long) (heads * ratio), (long long) heads,
                (long long) cells, (long long) tokens, sinks, softcap, all_masked, nmse, flag_nmse, max_abs, lse_error, ok ? "PASS" : "FAIL");
    std::fflush(stdout);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    const char * name = argc > 1 ? argv[1] : "CPU";
    const bool full = argc > 2 && std::strcmp(argv[2], "--full") == 0;
    auto * backend = ggml_backend_init_by_name(name, nullptr);
    if (!backend) return 77;
    if (ggml_backend_is_cpu(backend)) ggml_backend_cpu_set_n_threads(backend, 8);
    bool ok = true;
    for (int64_t width : {64, 128}) {
        for (int64_t cells : {256, 4096, 24576}) {
            if (!full && cells == 24576) continue;
            for (int64_t tokens : {1, 8, 512}) {
                if (!full && tokens == 512) continue;
                ok &= run_case(backend, width, width == 128 ? 8 : 2, width == 128 ? 5 : 2, cells, tokens, false, 0);
            }
        }
    }
    ok &= run_case(backend, 128, 8, 5, 256, 8, true, 4);
    ok &= run_case(backend, 64, 2, 2, 256, 1, false, 0, true);
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
