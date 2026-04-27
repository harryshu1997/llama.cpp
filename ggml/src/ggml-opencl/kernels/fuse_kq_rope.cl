// FlashAttention-style fused split-path attention kernel for TierKV (Gemma 4).
//
// Mirrors the CPU reference in ggml/src/ggml-cpu/ops.cpp:ggml_compute_forward_fuse_kq_rope_f32:
//   1. Tier-0 (exact): read pre-norm/pre-RoPE'd K, V from cache, FA-style score+accumulate.
//   2. Tier-1 (SVD): reconstruct K = wuk @ ZSK on-chip, RMSNorm-then-RoPE (per design D3),
//      score, reconstruct V = wuv @ ZSK on-chip, RMSNorm (Gemma's unweighted V-norm), accumulate.
//
// Gemma 4 carries a per-token V RMSNorm (gemma4-iswa.cpp:95). That norm is non-linear in the
// per-token V vector, so the paper's latent-space V accumulation (eq 11-12) does NOT apply
// here — we materialize V_recon per token. (For non-V-norm models we could specialize later.)
//
// One workgroup per (q_token, q_head, batch). Workgroup size = subgroup size = 64 for Adreno.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
#elif defined(cl_intel_required_subgroup_size)
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define REQD_SUBGROUP_SIZE_64 __attribute__((intel_reqd_sub_group_size(16)))
#else
#define REQD_SUBGROUP_SIZE_64
#endif

// Maximum d_head this kernel supports. Gemma 4 E2B has d_head=512 for full layers,
// 256 for SWA layers. Bumping this only costs LDS, so size for the largest case.
#define FUSE_KQ_MAX_D_HEAD 512
#define FUSE_KQ_MAX_RANK   512


// ---- RoPE-YaRN math (mirrors ggml_rope_cache_init in ggml/src/ggml.c) ----
static inline float rope_yarn_ramp(float low, float high, int i0) {
    const float y = (i0 / 2 - low) / fmax(0.001f, high - low);
    return 1.0f - fmin(1.0f, fmax(0.0f, y));
}

static inline float2 rope_yarn(
        float theta_extrap, float freq_scale, float2 corr_dims,
        int i0, float ext_factor, float mscale) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims.s0, corr_dims.s1, i0) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);
    }
    return (float2)(cos(theta) * mscale, sin(theta) * mscale);
}

static inline float rope_corr_factor(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * log((float)n_ctx_orig / (n_rot * 2.0f * M_PI_F)) / (2.0f * log(base));
}

static inline float2 rope_corr_dims(
        int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow) {
    return (float2)(
        fmax(0.0f, floor(rope_corr_factor(n_dims, n_ctx_orig, beta_fast, freq_base))),
        fmin((float)(n_dims - 1), ceil(rope_corr_factor(n_dims, n_ctx_orig, beta_slow, freq_base)))
    );
}

// ---- Cooperative block helpers (subgroup of 64) ----
static inline float wg_reduce_sum(__local float * tmp, float v) {
    // Subgroup reduction; assumes WG_SIZE == subgroup size == 64.
    return sub_group_reduce_add(v);
}

static inline float wg_reduce_max(__local float * tmp, float v) {
    return sub_group_reduce_max(v);
}

#define ROPE_MODE_NORMAL 0
#define ROPE_MODE_NEOX   2

REQD_SUBGROUP_SIZE_64
__kernel void kernel_fuse_kq_rope_fa(
        // Q: f32, ne=[d_head, n_q, n_q_heads, ns], permuted by graph build to [d_head, n_q, n_q_heads, ns].
        global void * q_data,           ulong q_offset,
        // K_exact: f16, ne=[d_head, n_t0, n_kv_heads, ns] (post graph permute). NULL if Tier-0 absent.
        global void * k_data,           ulong k_offset,
        // V_exact: f16, same ne as K_exact. d-stride may differ from sizeof(fp16) when v_trans=true,
        // so always use vnb0/vnb1/vnb2 instead of pointer index arithmetic.
        global void * v_data,           ulong v_offset,
        // ZSK: f16, ne=[rank, n_t1, ns]. NULL if Tier-1 absent.
        global void * zsk_data,         ulong zsk_offset,
        // W_uk / W_uv: f16, ne=[d_head*n_kv_heads, rank]. Bound as image1d_buffer to hit the
        // Adreno texture cache (much larger / lower latency than the buffer cache that previously
        // dominated this kernel's runtime). Element-indexed, so we pass the byte offset divided
        // by sizeof(half) plus the row stride in halves.
        read_only image1d_buffer_t wuk_img, int wuk_off_h, int wuk_stride_h,
        read_only image1d_buffer_t wuv_img, int wuv_off_h, int wuv_stride_h,
        // K-RMSNorm weight: f32, ne=[d_head]. NULL → unweighted RMS.
        global void * knorm_data,       ulong knorm_offset, int knorm_present,
        // RoPE proportional freqs: f32, ne=[n_rot/2]. NULL → all 1.0.
        global void * ropef_data,       ulong ropef_offset, int ropef_present,
        // Mask: f16, ne=[n_t0+n_t1, n_q]. NULL → no mask.
        global void * mask_data,        ulong mask_offset, int mask_present,
        // Output: f32, ne=[d_head, n_q_heads, n_q, ns].
        global float * dst_data,        ulong dst_offset,

        // Strides for irregular layouts.
        ulong qnb1, ulong qnb2, ulong qnb3,
        ulong knb0, ulong knb1, ulong knb2, ulong knb3,
        ulong vnb0, ulong vnb1, ulong vnb2, ulong vnb3,
        ulong zsknb1, ulong zsknb2,
        ulong masknb1,
        ulong dstnb1, ulong dstnb2, ulong dstnb3,

        // Dimensions.
        int d_head, int n_q_heads, int n_kv_heads,
        int n_t0, int n_t1, int rank,
        int n_rot, int rope_mode, int n_ctx_orig, int pos_t1_offset,
        float freq_base, float freq_scale, float ext_factor, float attn_factor,
        float beta_fast, float beta_slow,
        float scale, float logit_softcap, float rms_norm_eps
) {
    const int t      = get_group_id(0);   // q_token
    const int h_q    = get_group_id(1);   // q_head
    const int s      = get_group_id(2);   // batch / stream
    const int lane   = get_local_id(0);   // [0, 64)
    const int lsz    = get_local_size(0); // 64

    const int group  = n_q_heads / n_kv_heads;
    const int h_kv   = h_q / group;

    // Apply base offsets once.
    global char * q_base    = (global char *) q_data    + q_offset;
    global char * k_base    = (global char *) k_data    + k_offset;
    global char * v_base    = (global char *) v_data    + v_offset;
    global char * zsk_base  = (global char *) zsk_data  + zsk_offset;
    global char * mask_base = (global char *) mask_data + mask_offset;
    // wuk / wuv element-index helpers
    const int wuk_h_kv_off = h_kv * d_head;
    const int wuv_h_kv_off = h_kv * d_head;

    // -- Load Q for (s, h_q, t) into local memory --
    // Q layout (post graph permute): ne=[d_head, n_q, n_q_heads, ns],
    //   addr = q_base + s*qnb3 + h_q*qnb2 + t*qnb1 + d*sizeof(float)
    __local float Q_loc[FUSE_KQ_MAX_D_HEAD];
    {
        global float * q_row = (global float *)(q_base + s*qnb3 + h_q*qnb2 + t*qnb1);
        for (int d = lane; d < d_head; d += lsz) {
            Q_loc[d] = q_row[d];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // -- FA running state. `__local` declarations must live at kernel function scope
    //    (OpenCL 1.2+ spec; some compilers like Adreno's reject them inside if/loop blocks).
    __local float O[FUSE_KQ_MAX_D_HEAD];     // running output, scaled by softmax denom at the end
    __local float K_recon[FUSE_KQ_MAX_D_HEAD];   // Tier-1 reconstructed K (only used if n_t1 > 0)
    __local float V_recon[FUSE_KQ_MAX_D_HEAD];   // Tier-1 reconstructed V
    for (int d = lane; d < d_head; d += lsz) O[d] = 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    float m_i = -INFINITY;
    float l_i = 0.0f;

    // ============= Phase 1: Tier-0 (exact K/V from cache) =============
    // K_exact / V_exact both ne=[d_head, n_t0, n_kv_heads, ns].
    // For each tier-0 token j, read K[d, j, h_kv, s] and compute Q · K, then update FA state.
    for (int j = 0; j < n_t0; ++j) {
        // Score s_j = Q · K[:, j, h_kv, s] * scale
        float partial = 0.0f;
        global char * k_col = k_base + s*knb3 + h_kv*knb2 + j*knb1;
        for (int d = lane; d < d_head; d += lsz) {
            half k_d = *(global half *)(k_col + d * knb0);
            partial += Q_loc[d] * (float) k_d;
        }
        float s_j = wg_reduce_sum(0, partial) * scale;

        if (logit_softcap > 0.0f) {
            s_j = logit_softcap * tanh(s_j / logit_softcap);
        }
        if (mask_present != 0) {
            half m_v = *(global half *)(mask_base + t*masknb1 + j*sizeof(half));
            s_j += (float) m_v;
        }

        // FA softmax merge.
        const float m_new = fmax(m_i, s_j);
        const float alpha = exp(m_i - m_new);
        const float p_j   = exp(s_j  - m_new);
        l_i = alpha * l_i + p_j;

        // Update O: scale prior + add p_j * V[:, j, h_kv, s].
        global char * v_col = v_base + s*vnb3 + h_kv*vnb2 + j*vnb1;
        for (int d = lane; d < d_head; d += lsz) {
            half v_d = *(global half *)(v_col + d * vnb0);
            O[d] = alpha * O[d] + p_j * (float) v_d;
        }
        m_i = m_new;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // ============= Phase 2: Tier-1 (SVD reconstruction on-chip) =============
    if (n_t1 > 0) {
        const float2 corr_dims = rope_corr_dims(n_rot, n_ctx_orig, freq_base, beta_fast, beta_slow);
        const float  inv_n_rot = 1.0f / (float) n_rot;

        for (int j_t1 = 0; j_t1 < n_t1; ++j_t1) {
            // ---- Reconstruct K_recon[d] = sum_r ZSK[r, j_t1] * W_uk[h_kv*d_head + d, r] ----
            // wuk image1d_buffer is CL_RGBA half4 — each lane handles 4 contiguous d-values
            // per texel, all 4 channels used. d_head, h_kv*d_head, wuk_stride_h are all
            // multiples of 4; wuk_off_h is buffer-aligned. d4 stride is lsz*4 (= 256) covering
            // d_head ∈ {256, 512} in 1 or 2 iterations per lane.
            global half * zsk_col = (global half *)(zsk_base + s*zsknb2 + j_t1 * zsknb1);
            const int wuk_stride_t = wuk_stride_h >> 2;          // texels per row
            const int wuk_h_kv_t   = wuk_h_kv_off >> 2;          // texels offset for h_kv
            const int wuk_off_t    = wuk_off_h    >> 2;          // texels offset for view
            for (int d4 = lane; d4 * 4 < d_head; d4 += lsz) {
                float4 acc = (float4)(0.0f);
                const int texel_base = wuk_off_t + wuk_h_kv_t + d4;
                for (int r = 0; r < rank; ++r) {
                    half4 t = read_imageh(wuk_img, texel_base + r * wuk_stride_t);
                    acc += (float) zsk_col[r] * convert_float4(t);
                }
                vstore4(acc, d4, K_recon);
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            // ---- RMSNorm K_recon (per-head over d_head) with knorm_w ----
            float ss = 0.0f;
            for (int d = lane; d < d_head; d += lsz) ss += K_recon[d] * K_recon[d];
            ss = wg_reduce_sum(0, ss);
            const float inv_rms_k = rsqrt(ss / (float) d_head + rms_norm_eps);
            if (knorm_present != 0) {
                global float * w = (global float *)((global char *) knorm_data + knorm_offset);
                for (int d = lane; d < d_head; d += lsz) K_recon[d] = K_recon[d] * inv_rms_k * w[d];
            } else {
                for (int d = lane; d < d_head; d += lsz) K_recon[d] *= inv_rms_k;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            // ---- RoPE K_recon at pos = pos_t1_offset + j_t1 (NORMAL or NEOX) ----
            const int pos = pos_t1_offset + j_t1;
            global float * rope_factors = (ropef_present != 0)
                ? (global float *)((global char *) ropef_data + ropef_offset) : (global float *) 0;

            if (rope_mode == ROPE_MODE_NORMAL) {
                for (int i0 = 2*lane; i0 < n_rot; i0 += 2*lsz) {
                    const int ic = i0 / 2;
                    const float ff = (rope_factors != (global float *) 0) ? rope_factors[ic] : 1.0f;
                    float theta = (float) pos * pow(freq_base, -(float)i0 * inv_n_rot);
                    float2 cs;
                    if (ff == 0.0f) {
                        cs = (float2)(1.0f, 0.0f);
                    } else {
                        cs = rope_yarn(theta / ff, freq_scale, corr_dims, i0, ext_factor, attn_factor);
                    }
                    const float x0 = K_recon[i0];
                    const float x1 = K_recon[i0 + 1];
                    K_recon[i0]     = x0 * cs.s0 - x1 * cs.s1;
                    K_recon[i0 + 1] = x0 * cs.s1 + x1 * cs.s0;
                }
            } else {
                const int half_rot = n_rot / 2;
                for (int i = lane; i < half_rot; i += lsz) {
                    const int i0 = 2 * i;
                    const float ff = (rope_factors != (global float *) 0) ? rope_factors[i] : 1.0f;
                    float theta = (float) pos * pow(freq_base, -(float)i0 * inv_n_rot);
                    float2 cs;
                    if (ff == 0.0f) {
                        cs = (float2)(1.0f, 0.0f);
                    } else {
                        cs = rope_yarn(theta / ff, freq_scale, corr_dims, i0, ext_factor, attn_factor);
                    }
                    const float x0 = K_recon[i];
                    const float x1 = K_recon[i + half_rot];
                    K_recon[i]            = x0 * cs.s0 - x1 * cs.s1;
                    K_recon[i + half_rot] = x0 * cs.s1 + x1 * cs.s0;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            // ---- Score s_j = Q · K_recon * scale + mask ----
            float partial = 0.0f;
            for (int d = lane; d < d_head; d += lsz) partial += Q_loc[d] * K_recon[d];
            float s_j = wg_reduce_sum(0, partial) * scale;
            if (logit_softcap > 0.0f) {
                s_j = logit_softcap * tanh(s_j / logit_softcap);
            }
            if (mask_present != 0) {
                half m_v = *(global half *)(mask_base + t*masknb1 + (n_t0 + j_t1) * sizeof(half));
                s_j += (float) m_v;
            }

            // ---- Reconstruct V_recon and unweighted-RMSNorm it (Gemma's V-norm) ----
            // Same RGBA texel-stride pattern as K_recon.
            const int wuv_stride_t = wuv_stride_h >> 2;
            const int wuv_h_kv_t   = wuv_h_kv_off >> 2;
            const int wuv_off_t    = wuv_off_h    >> 2;
            for (int d4 = lane; d4 * 4 < d_head; d4 += lsz) {
                float4 acc = (float4)(0.0f);
                const int texel_base = wuv_off_t + wuv_h_kv_t + d4;
                for (int r = 0; r < rank; ++r) {
                    half4 t = read_imageh(wuv_img, texel_base + r * wuv_stride_t);
                    acc += (float) zsk_col[r] * convert_float4(t);
                }
                vstore4(acc, d4, V_recon);
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            float ss_v = 0.0f;
            for (int d = lane; d < d_head; d += lsz) ss_v += V_recon[d] * V_recon[d];
            ss_v = wg_reduce_sum(0, ss_v);
            const float inv_rms_v = rsqrt(ss_v / (float) d_head + rms_norm_eps);

            // ---- FA softmax merge with V_recon * inv_rms_v ----
            const float m_new = fmax(m_i, s_j);
            const float alpha = exp(m_i - m_new);
            const float p_j   = exp(s_j  - m_new);
            l_i = alpha * l_i + p_j;
            for (int d = lane; d < d_head; d += lsz) {
                O[d] = alpha * O[d] + p_j * V_recon[d] * inv_rms_v;
            }
            m_i = m_new;
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // ============= Final normalization and store =============
    // dst layout: ne=[d_head, n_q_heads, n_q, ns],
    //   addr = dst_data + dst_offset + s*dstnb3 + t*dstnb2 + h_q*dstnb1 + d*sizeof(float)
    global float * dst_row = (global float *)((global char *) dst_data + dst_offset
                                              + s*dstnb3 + t*dstnb2 + h_q*dstnb1);
    const float l_inv = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
    for (int d = lane; d < d_head; d += lsz) {
        dst_row[d] = O[d] * l_inv;
    }
}
