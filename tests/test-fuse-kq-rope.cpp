// Unit test for GGML_OP_FUSE_KQ_ROPE (TierKV).
//
// Strategy: compute an independent C++ reference of the expected output
// (using only nested loops + ggml_rope_ext for RoPE, to avoid re-deriving YaRN math)
// and compare it to the output of ggml_fuse_kq_rope on the same random inputs.
//
// Three modes are exercised:
//   (a) Tier-0 only  : zsk = w_uk = w_uv = NULL
//   (b) Tier-1 only  : k_exact = v_exact = NULL
//   (c) Combined     : all paths active
//
// The reference is single-threaded and exercises the same dtype contract
// as the production CPU kernel (Q in F32; K/V/ZSK/W_uk/W_uv/mask in F16).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

struct dims {
    int64_t d_head;
    int64_t n_q;
    int64_t n_q_heads;
    int64_t n_kv_heads;
    int64_t n_t0;       // Tier-0 cached tokens
    int64_t n_t1;       // Tier-1 latent tokens
    int64_t rank;       // SVD rank
};

float frand_uniform(std::mt19937 & gen, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    return dist(gen);
}

void fill_random_f32(float * data, int64_t n, std::mt19937 & gen) {
    for (int64_t i = 0; i < n; ++i) {
        data[i] = frand_uniform(gen, -1.0f, 1.0f);
    }
}

void fill_random_f16(ggml_fp16_t * data, int64_t n, std::mt19937 & gen) {
    for (int64_t i = 0; i < n; ++i) {
        data[i] = ggml_fp32_to_fp16(frand_uniform(gen, -1.0f, 1.0f));
    }
}

// Apply ggml_rope_ext to a single [d_head] vector at the given position;
// returns the rotated vector (length d_head, only first n_rot dims rotate).
// Uses a tiny throwaway ggml graph so we never re-derive YaRN math here.
std::vector<float> apply_rope_via_ggml(
        const std::vector<float> & k_in,
        int32_t pos,
        int n_rot,
        int rope_mode,
        int n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        const float * rope_freqs,
        int64_t rope_freqs_n) {

    const int64_t d_head = (int64_t) k_in.size();

    ggml_init_params p = { /* mem_size = */ 1024*1024,
                            /* mem_buffer = */ NULL,
                            /* no_alloc = */ false };
    ggml_context * ctx = ggml_init(p);

    ggml_tensor * src = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_head, /*n_tok=*/1, /*n_head=*/1);
    memcpy(src->data, k_in.data(), d_head * sizeof(float));

    ggml_tensor * pos_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ((int32_t *) pos_t->data)[0] = pos;

    ggml_tensor * freqs_t = NULL;
    if (rope_freqs) {
        freqs_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, rope_freqs_n);
        memcpy(freqs_t->data, rope_freqs, rope_freqs_n * sizeof(float));
    }

    ggml_tensor * out = ggml_rope_ext(ctx, src, pos_t, freqs_t,
        n_rot, rope_mode, n_ctx_orig,
        freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_cplan plan = ggml_graph_plan(gf, 1, NULL);
    std::vector<uint8_t> work; work.resize(plan.work_size);
    plan.work_data = work.data();
    ggml_graph_compute(gf, &plan);

    std::vector<float> result(d_head);
    memcpy(result.data(), out->data, d_head * sizeof(float));
    ggml_free(ctx);
    return result;
}

struct ref_inputs {
    dims D;
    std::vector<float>        Q;        // [d_head, n_q, n_q_heads]
    std::vector<ggml_fp16_t>  K_exact;  // [d_head, n_t0, n_kv_heads]   (post norm + RoPE)
    std::vector<ggml_fp16_t>  V_exact;  // [d_head, n_t0, n_kv_heads]   (post norm)
    std::vector<ggml_fp16_t>  ZSK;      // [rank,   n_t1]
    std::vector<ggml_fp16_t>  W_uk;     // [d_head*n_kv_heads, rank]    (col-major: ne[0]=d_head*n_kv_heads, ne[1]=rank)
    std::vector<ggml_fp16_t>  W_uv;     // same
    std::vector<float>        K_norm_w; // [d_head]                     (NULL → ignored)
    int                       pos_t1_offset; // Tier-1 K position[i] = pos_t1_offset + i
    std::vector<ggml_fp16_t>  mask;     // [n_t0+n_t1, n_q]

    int   n_rot;
    int   rope_mode;
    int   n_ctx_orig;
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    float scale;
    float logit_softcap;
    float rms_norm_eps;
    bool  enable_t0;
    bool  enable_t1;
    bool  enable_k_norm_w;
    bool  enable_mask;
};

// Independent C++ reference. Output shape [d_head, n_q_heads, n_q] (one batch).
std::vector<float> ref_compute(const ref_inputs & X) {
    const dims & D = X.D;
    const int64_t group = D.n_q_heads / D.n_kv_heads;
    const int64_t n_t0  = X.enable_t0 ? D.n_t0 : 0;
    const int64_t n_t1  = X.enable_t1 ? D.n_t1 : 0;
    const int64_t n_keys = n_t0 + n_t1;
    const float * rope_freqs_data = NULL; // we don't pass freq_factors in this test

    std::vector<float> out_dst(D.d_head * D.n_q_heads * D.n_q, 0.0f);

    for (int64_t i_q = 0; i_q < D.n_q; ++i_q) {
        for (int64_t h_q = 0; h_q < D.n_q_heads; ++h_q) {
            const int64_t h_kv = h_q / group;
            const float * q_vec = X.Q.data() + h_q * (D.n_q * D.d_head) + i_q * D.d_head;

            std::vector<float> scores(n_keys, 0.0f);

            // Tier-0 scores
            for (int64_t j = 0; j < n_t0; ++j) {
                const ggml_fp16_t * k_vec = X.K_exact.data()
                    + h_kv * (D.n_t0 * D.d_head) + j * D.d_head;
                float s = 0.0f;
                for (int64_t d = 0; d < D.d_head; ++d) {
                    s += q_vec[d] * ggml_fp16_to_fp32(k_vec[d]);
                }
                s *= X.scale;
                if (X.logit_softcap > 0.0f) s = X.logit_softcap * std::tanh(s / X.logit_softcap);
                if (X.enable_mask)         s += ggml_fp16_to_fp32(X.mask[i_q * (D.n_t0 + D.n_t1) + j]);
                scores[j] = s;
            }

            // Tier-1 scores
            for (int64_t j_t1 = 0; j_t1 < n_t1; ++j_t1) {
                std::vector<float> K_recon(D.d_head, 0.0f);
                const ggml_fp16_t * z_col = X.ZSK.data() + j_t1 * D.rank;
                for (int64_t r = 0; r < D.rank; ++r) {
                    const float z = ggml_fp16_to_fp32(z_col[r]);
                    // W_uk col r: ne[0] = d_head*n_kv_heads stride; address [(h_kv*d_head + d), r]
                    const ggml_fp16_t * uk_col = X.W_uk.data() + r * (D.d_head * D.n_kv_heads) + h_kv * D.d_head;
                    for (int64_t d = 0; d < D.d_head; ++d) {
                        K_recon[d] += z * ggml_fp16_to_fp32(uk_col[d]);
                    }
                }
                // RMSNorm
                float ss = 0.0f;
                for (float v : K_recon) ss += v*v;
                const float inv_rms = 1.0f / std::sqrt(ss / (float) D.d_head + X.rms_norm_eps);
                if (X.enable_k_norm_w) {
                    for (int64_t d = 0; d < D.d_head; ++d) K_recon[d] = K_recon[d] * inv_rms * X.K_norm_w[d];
                } else {
                    for (int64_t d = 0; d < D.d_head; ++d) K_recon[d] *= inv_rms;
                }
                // RoPE via ggml (so we don't re-derive YaRN here)
                std::vector<float> K_rope = apply_rope_via_ggml(
                    K_recon, X.pos_t1_offset + (int32_t) j_t1, X.n_rot, X.rope_mode, X.n_ctx_orig,
                    X.freq_base, X.freq_scale, X.ext_factor, X.attn_factor,
                    X.beta_fast, X.beta_slow, rope_freqs_data, 0);

                float s = 0.0f;
                for (int64_t d = 0; d < D.d_head; ++d) s += q_vec[d] * K_rope[d];
                s *= X.scale;
                if (X.logit_softcap > 0.0f) s = X.logit_softcap * std::tanh(s / X.logit_softcap);
                if (X.enable_mask)         s += ggml_fp16_to_fp32(X.mask[i_q * (D.n_t0 + D.n_t1) + n_t0 + j_t1]);
                scores[n_t0 + j_t1] = s;
            }

            // Softmax
            float max_s = -INFINITY;
            for (float s : scores) max_s = std::max(max_s, s);
            float sum_exp = 0.0f;
            for (auto & s : scores) { s = std::exp(s - max_s); sum_exp += s; }
            const float inv_sum = sum_exp > 0.0f ? 1.0f / sum_exp : 0.0f;
            for (auto & s : scores) s *= inv_sum;

            // Accumulate V
            std::vector<float> out(D.d_head, 0.0f);
            for (int64_t j = 0; j < n_t0; ++j) {
                const ggml_fp16_t * v_vec = X.V_exact.data()
                    + h_kv * (D.n_t0 * D.d_head) + j * D.d_head;
                const float a = scores[j];
                for (int64_t d = 0; d < D.d_head; ++d) out[d] += a * ggml_fp16_to_fp32(v_vec[d]);
            }
            for (int64_t j_t1 = 0; j_t1 < n_t1; ++j_t1) {
                std::vector<float> V_recon(D.d_head, 0.0f);
                const ggml_fp16_t * z_col = X.ZSK.data() + j_t1 * D.rank;
                for (int64_t r = 0; r < D.rank; ++r) {
                    const float z = ggml_fp16_to_fp32(z_col[r]);
                    const ggml_fp16_t * uv_col = X.W_uv.data() + r * (D.d_head * D.n_kv_heads) + h_kv * D.d_head;
                    for (int64_t d = 0; d < D.d_head; ++d) V_recon[d] += z * ggml_fp16_to_fp32(uv_col[d]);
                }
                float ss = 0.0f;
                for (float v : V_recon) ss += v*v;
                const float inv_rms = 1.0f / std::sqrt(ss / (float) D.d_head + X.rms_norm_eps);
                const float a = scores[n_t0 + j_t1];
                for (int64_t d = 0; d < D.d_head; ++d) out[d] += a * V_recon[d] * inv_rms;
            }

            // Reference output layout: [d_head, n_q_heads, n_q] same as fused op
            float * out_ptr = out_dst.data() + i_q * (D.n_q_heads * D.d_head) + h_q * D.d_head;
            for (int64_t d = 0; d < D.d_head; ++d) out_ptr[d] = out[d];
        }
    }
    return out_dst;
}

// Run the fused op via ggml and return its output as a flat vector.
std::vector<float> run_fused(const ref_inputs & X) {
    const dims & D = X.D;
    const int64_t n_t0 = X.enable_t0 ? D.n_t0 : 0;
    const int64_t n_t1 = X.enable_t1 ? D.n_t1 : 0;

    ggml_init_params p = { /* mem_size = */ 64*1024*1024,
                            /* mem_buffer = */ NULL,
                            /* no_alloc = */ false };
    ggml_context * ctx = ggml_init(p);

    ggml_tensor * Q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D.d_head, D.n_q, D.n_q_heads);
    memcpy(Q->data, X.Q.data(), X.Q.size() * sizeof(float));

    ggml_tensor * K_exact = NULL, *V_exact = NULL;
    if (X.enable_t0) {
        K_exact = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D.d_head, D.n_t0, D.n_kv_heads);
        V_exact = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D.d_head, D.n_t0, D.n_kv_heads);
        memcpy(K_exact->data, X.K_exact.data(), X.K_exact.size() * sizeof(ggml_fp16_t));
        memcpy(V_exact->data, X.V_exact.data(), X.V_exact.size() * sizeof(ggml_fp16_t));
    }

    ggml_tensor * ZSK = NULL, *W_uk = NULL, *W_uv = NULL, *K_norm_w = NULL;
    if (X.enable_t1) {
        ZSK = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, D.rank, D.n_t1);
        W_uk = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, D.d_head * D.n_kv_heads, D.rank);
        W_uv = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, D.d_head * D.n_kv_heads, D.rank);
        memcpy(ZSK->data,  X.ZSK.data(),  X.ZSK.size()  * sizeof(ggml_fp16_t));
        memcpy(W_uk->data, X.W_uk.data(), X.W_uk.size() * sizeof(ggml_fp16_t));
        memcpy(W_uv->data, X.W_uv.data(), X.W_uv.size() * sizeof(ggml_fp16_t));
        if (X.enable_k_norm_w) {
            K_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D.d_head);
            memcpy(K_norm_w->data, X.K_norm_w.data(), X.K_norm_w.size() * sizeof(float));
        }
    }

    ggml_tensor * mask_t = NULL;
    if (X.enable_mask) {
        mask_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, D.n_t0 + D.n_t1, D.n_q);
        memcpy(mask_t->data, X.mask.data(), X.mask.size() * sizeof(ggml_fp16_t));
    }

    ggml_tensor * out = ggml_fuse_kq_rope(ctx, Q, K_exact, V_exact, ZSK, W_uk, W_uv,
        K_norm_w, /*rope_freqs=*/NULL, mask_t,
        X.n_rot, X.rope_mode, X.n_ctx_orig, X.pos_t1_offset,
        X.freq_base, X.freq_scale, X.ext_factor, X.attn_factor, X.beta_fast, X.beta_slow,
        X.scale, X.logit_softcap, X.rms_norm_eps);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_cplan plan = ggml_graph_plan(gf, 1, NULL);
    std::vector<uint8_t> work; work.resize(plan.work_size);
    plan.work_data = work.data();
    ggml_graph_compute(gf, &plan);

    // Output shape [d_head, n_q_heads, n_q]; copy out
    std::vector<float> result(D.d_head * D.n_q_heads * D.n_q);
    // Output is contiguous per ggml_new_tensor — copy directly
    memcpy(result.data(), out->data, result.size() * sizeof(float));

    (void) n_t0; (void) n_t1;
    ggml_free(ctx);
    return result;
}

void compute_errors(const std::vector<float> & a, const std::vector<float> & b,
                    float & max_abs, float & rel_l2) {
    max_abs = 0.0f;
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const float diff = a[i] - b[i];
        max_abs = std::max(max_abs, std::abs(diff));
        num += (double) diff * diff;
        den += (double) a[i] * a[i];
    }
    rel_l2 = den > 0.0 ? (float) std::sqrt(num / den) : (float) std::sqrt(num);
}

ref_inputs make_inputs(const dims & D, std::mt19937 & gen,
                       int rope_mode, bool enable_t0, bool enable_t1,
                       bool enable_k_norm_w, bool enable_mask) {
    ref_inputs X;
    X.D = D;
    X.n_rot = (int) D.d_head; // full rotary
    X.rope_mode = rope_mode;
    X.n_ctx_orig = 4096;
    X.freq_base = 10000.0f;
    X.freq_scale = 1.0f;
    X.ext_factor = 0.0f;
    X.attn_factor = 1.0f;
    X.beta_fast = 32.0f;
    X.beta_slow = 1.0f;
    X.scale = 1.0f / std::sqrt((float) D.d_head);
    X.logit_softcap = 0.0f;
    X.rms_norm_eps = 1e-6f;
    X.enable_t0 = enable_t0;
    X.enable_t1 = enable_t1;
    X.enable_k_norm_w = enable_k_norm_w;
    X.enable_mask = enable_mask;

    X.Q.resize(D.d_head * D.n_q * D.n_q_heads);
    fill_random_f32(X.Q.data(), X.Q.size(), gen);

    if (enable_t0) {
        X.K_exact.resize(D.d_head * D.n_t0 * D.n_kv_heads);
        X.V_exact.resize(D.d_head * D.n_t0 * D.n_kv_heads);
        fill_random_f16(X.K_exact.data(), X.K_exact.size(), gen);
        fill_random_f16(X.V_exact.data(), X.V_exact.size(), gen);
    }

    if (enable_t1) {
        X.ZSK.resize(D.rank * D.n_t1);
        X.W_uk.resize(D.d_head * D.n_kv_heads * D.rank);
        X.W_uv.resize(D.d_head * D.n_kv_heads * D.rank);
        fill_random_f16(X.ZSK.data(),  X.ZSK.size(),  gen);
        fill_random_f16(X.W_uk.data(), X.W_uk.size(), gen);
        fill_random_f16(X.W_uv.data(), X.W_uv.size(), gen);
        if (enable_k_norm_w) {
            X.K_norm_w.resize(D.d_head);
            fill_random_f32(X.K_norm_w.data(), X.K_norm_w.size(), gen);
        }
        X.pos_t1_offset = (int32_t) D.n_t0;
    }

    if (enable_mask) {
        X.mask.resize((size_t) (D.n_t0 + D.n_t1) * D.n_q);
        fill_random_f16(X.mask.data(), X.mask.size(), gen);
    }

    return X;
}

bool run_one_case(const char * label, const dims & D, std::mt19937 & gen,
                  int rope_mode, bool t0, bool t1, bool kw, bool mask, float tol) {
    ref_inputs X = make_inputs(D, gen, rope_mode, t0, t1, kw, mask);
    std::vector<float> exp = ref_compute(X);
    std::vector<float> got = run_fused(X);
    float max_abs, rel_l2;
    compute_errors(exp, got, max_abs, rel_l2);
    printf("[%s] dims=(d=%lld nq=%lld nqH=%lld nkvH=%lld nT0=%lld nT1=%lld r=%lld) "
           "rope=%s t0=%d t1=%d knorm=%d mask=%d -> max_abs=%.3e rel_L2=%.3e (tol=%.3e) %s\n",
           label,
           (long long)D.d_head, (long long)D.n_q, (long long)D.n_q_heads, (long long)D.n_kv_heads,
           (long long)D.n_t0, (long long)D.n_t1, (long long)D.rank,
           rope_mode == GGML_ROPE_TYPE_NORMAL ? "NORMAL" : "NEOX",
           t0, t1, kw, mask, max_abs, rel_l2, tol,
           rel_l2 <= tol ? "OK" : "FAIL");
    return rel_l2 <= tol;
}

// ---------------------------------------------------------------------------
// OpenCL parity path: build the same op via ggml-backend, run on `backend`,
// pull the result back. Returns empty vector on alloc/compute failure.
// ---------------------------------------------------------------------------
std::vector<float> run_fused_via_backend(const ref_inputs & X, ggml_backend_t backend) {
    const dims & D = X.D;

    ggml_init_params p = { /*mem_size=*/ 64*1024*1024, /*mem_buffer=*/ NULL, /*no_alloc=*/ true };
    ggml_context * ctx = ggml_init(p);

    ggml_tensor * Q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D.d_head, D.n_q, D.n_q_heads);
    ggml_tensor * K_exact = NULL;
    ggml_tensor * V_exact = NULL;
    if (X.enable_t0) {
        K_exact = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D.d_head, D.n_t0, D.n_kv_heads);
        V_exact = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D.d_head, D.n_t0, D.n_kv_heads);
    }
    ggml_tensor * ZSK = NULL, *W_uk = NULL, *W_uv = NULL, *K_norm_w = NULL;
    if (X.enable_t1) {
        ZSK  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, D.rank, D.n_t1);
        W_uk = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, D.d_head * D.n_kv_heads, D.rank);
        W_uv = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, D.d_head * D.n_kv_heads, D.rank);
        if (X.enable_k_norm_w) {
            K_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D.d_head);
        }
    }
    ggml_tensor * mask = X.enable_mask
        ? ggml_new_tensor_2d(ctx, GGML_TYPE_F16, D.n_t0 + D.n_t1, D.n_q) : NULL;

    ggml_tensor * out = ggml_fuse_kq_rope(ctx, Q, K_exact, V_exact, ZSK, W_uk, W_uv,
        K_norm_w, /*rope_freqs=*/ NULL, mask,
        X.n_rot, X.rope_mode, X.n_ctx_orig, X.pos_t1_offset,
        X.freq_base, X.freq_scale, X.ext_factor, X.attn_factor, X.beta_fast, X.beta_slow,
        X.scale, X.logit_softcap, X.rms_norm_eps);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        fprintf(stderr, "  [backend] failed to allocate tensors\n");
        ggml_free(ctx);
        return {};
    }

    // Upload inputs.
    ggml_backend_tensor_set(Q, X.Q.data(), 0, X.Q.size() * sizeof(float));
    if (X.enable_t0) {
        ggml_backend_tensor_set(K_exact, X.K_exact.data(), 0, X.K_exact.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(V_exact, X.V_exact.data(), 0, X.V_exact.size() * sizeof(ggml_fp16_t));
    }
    if (X.enable_t1) {
        ggml_backend_tensor_set(ZSK,  X.ZSK.data(),  0, X.ZSK.size()  * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(W_uk, X.W_uk.data(), 0, X.W_uk.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(W_uv, X.W_uv.data(), 0, X.W_uv.size() * sizeof(ggml_fp16_t));
        if (K_norm_w) {
            ggml_backend_tensor_set(K_norm_w, X.K_norm_w.data(), 0, X.K_norm_w.size() * sizeof(float));
        }
    }
    if (mask) {
        ggml_backend_tensor_set(mask, X.mask.data(), 0, X.mask.size() * sizeof(ggml_fp16_t));
    }

    ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "  [backend] graph_compute failed: %s\n", ggml_status_to_string(status));
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return {};
    }

    std::vector<float> result(D.d_head * D.n_q_heads * D.n_q);
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return result;
}

bool run_one_case_backend(const char * label, const dims & D, std::mt19937 & gen,
                          int rope_mode, bool t0, bool t1, bool kw, bool mask,
                          float tol, ggml_backend_t backend, const char * backend_label) {
    ref_inputs X = make_inputs(D, gen, rope_mode, t0, t1, kw, mask);
    std::vector<float> exp = ref_compute(X);
    std::vector<float> got = run_fused_via_backend(X, backend);
    if (got.empty()) {
        printf("[%s/%s] dims=(d=%lld nq=%lld nqH=%lld nkvH=%lld nT0=%lld nT1=%lld r=%lld) "
               "rope=%s t0=%d t1=%d knorm=%d mask=%d -> SKIPPED (backend op failed)\n",
               label, backend_label,
               (long long)D.d_head, (long long)D.n_q, (long long)D.n_q_heads, (long long)D.n_kv_heads,
               (long long)D.n_t0, (long long)D.n_t1, (long long)D.rank,
               rope_mode == GGML_ROPE_TYPE_NORMAL ? "NORMAL" : "NEOX",
               t0, t1, kw, mask);
        return false;
    }
    float max_abs, rel_l2;
    compute_errors(exp, got, max_abs, rel_l2);
    printf("[%s/%s] dims=(d=%lld nq=%lld nqH=%lld nkvH=%lld nT0=%lld nT1=%lld r=%lld) "
           "rope=%s t0=%d t1=%d knorm=%d mask=%d -> max_abs=%.3e rel_L2=%.3e (tol=%.3e) %s\n",
           label, backend_label,
           (long long)D.d_head, (long long)D.n_q, (long long)D.n_q_heads, (long long)D.n_kv_heads,
           (long long)D.n_t0, (long long)D.n_t1, (long long)D.rank,
           rope_mode == GGML_ROPE_TYPE_NORMAL ? "NORMAL" : "NEOX",
           t0, t1, kw, mask, max_abs, rel_l2, tol,
           rel_l2 <= tol ? "OK" : "FAIL");
    return rel_l2 <= tol;
}

} // namespace

int main(int /*argc*/, char ** /*argv*/) {
    std::mt19937 gen(42);
    bool ok = true;

    // Small dimensions for fast iteration. d_head=64 to keep RoPE math simple.
    const dims D_small = {/*d_head*/64, /*n_q*/4, /*n_q_heads*/4, /*n_kv_heads*/2,
                          /*n_t0*/8, /*n_t1*/4, /*rank*/16};
    // Mid dimensions to mimic Gemma 4 sliding head_dim.
    const dims D_mid   = {/*d_head*/256, /*n_q*/2, /*n_q_heads*/8, /*n_kv_heads*/1,
                          /*n_t0*/16, /*n_t1*/8, /*rank*/64};

    const float TOL = 1e-3f;  // F16 storage rounds Q/K/V; ref also does FP16 round-trip → loose tol.

    // --- (a) Tier-0 only: ZSK path disabled, op should agree with reference (which is also Tier-0 only)
    ok &= run_one_case("Tier-0 NORMAL",  D_small, gen, GGML_ROPE_TYPE_NORMAL, true, false, false, false, TOL);
    ok &= run_one_case("Tier-0 +mask",   D_small, gen, GGML_ROPE_TYPE_NORMAL, true, false, false, true,  TOL);

    // --- (b) Tier-1 only: K/V are reconstructed from ZSK
    ok &= run_one_case("Tier-1 NORMAL",  D_small, gen, GGML_ROPE_TYPE_NORMAL, false, true, false, false, TOL);
    ok &= run_one_case("Tier-1 +knorm",  D_small, gen, GGML_ROPE_TYPE_NORMAL, false, true, true,  false, TOL);
    ok &= run_one_case("Tier-1 NEOX",    D_small, gen, GGML_ROPE_TYPE_NEOX,   false, true, false, false, TOL);

    // --- (c) Combined Tier-0 + Tier-1 (the production path)
    ok &= run_one_case("Combined NORMAL", D_small, gen, GGML_ROPE_TYPE_NORMAL, true, true, true, true, TOL);
    ok &= run_one_case("Combined NEOX",   D_small, gen, GGML_ROPE_TYPE_NEOX,   true, true, true, true, TOL);
    ok &= run_one_case("Combined Gemma-ish", D_mid, gen, GGML_ROPE_TYPE_NEOX,  true, true, true, true, TOL);

    // ----- (d) Backend parity: same cases, but driven through ggml-backend so any registered
    // accelerator backend (e.g. OpenCL / Adreno on phone) computes the op. Loose tol accounts
    // for FP16 reductions in the OpenCL kernel vs FP32 in the C++ reference.
    const float TOL_BACKEND = 5e-3f;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) continue;

        const char * dev_name = ggml_backend_dev_name(dev);
        ggml_backend_t backend = ggml_backend_dev_init(dev, NULL);
        if (!backend) {
            printf("[backend %s] init failed, skipping\n", dev_name);
            continue;
        }
        printf("\n=== Parity vs backend: %s ===\n", dev_name);
        ok &= run_one_case_backend("Tier-0 NORMAL",   D_small, gen, GGML_ROPE_TYPE_NORMAL, true, false, false, false, TOL_BACKEND, backend, dev_name);
        ok &= run_one_case_backend("Tier-0 +mask",    D_small, gen, GGML_ROPE_TYPE_NORMAL, true, false, false, true,  TOL_BACKEND, backend, dev_name);
        ok &= run_one_case_backend("Combined NORMAL", D_small, gen, GGML_ROPE_TYPE_NORMAL, true, true,  true,  true,  TOL_BACKEND, backend, dev_name);
        ok &= run_one_case_backend("Combined NEOX",   D_small, gen, GGML_ROPE_TYPE_NEOX,   true, true,  true,  true,  TOL_BACKEND, backend, dev_name);
        ok &= run_one_case_backend("Combined Gemma-ish", D_mid, gen, GGML_ROPE_TYPE_NEOX,  true, true,  true,  true,  TOL_BACKEND, backend, dev_name);
        ggml_backend_free(backend);
    }

    printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return ok ? 0 : 1;
}
