#include "models.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"

// get 2D slice view from a 3D tensor, the idx corresponds to the 3rd dim
static ggml_tensor * ggml_view_2d_slice(ggml_context * ctx0, ggml_tensor * x, int idx) {
    GGML_ASSERT(idx < (int) x->ne[2]);
    return ggml_view_2d(ctx0, x, x->ne[0], x->ne[1], ggml_row_size(x->type, x->ne[0]),
                        idx * x->ne[0] * x->ne[1] * ggml_element_size(x));
}

llm_build_gemma4_iswa::llm_build_gemma4_iswa(const llama_model & model, const llm_graph_params & params) :
        llm_graph_context(params),
        model(model),
        n_embd_per_layer(model.hparams.n_embd_per_layer) {
    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // important: do not normalize weights for raw embeddings input (i.e. encoded image emdeddings)
    inpL = ggml_scale(ctx0, inpL, ubatch.token ? sqrtf(n_embd) : 1.0f);
    cb(inpL, "inp_scaled", -1);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    // TODO: is causal == true correct? might need some changes
    auto * inp_attn = build_attn_inp_kv_iswa();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * inp_per_layer = nullptr;
    if (model.per_layer_tok_embd) {
        inp_per_layer = build_inp_per_layer();
        ggml_build_forward_expand(gf, inp_per_layer);

        // inp_per_layer shape: [n_embd_per_layer, n_tokens, n_layer]
        inp_per_layer = project_per_layer_inputs(inpL, inp_per_layer);
    }

    for (int il = 0; il < n_layer; ++il) {
        const int64_t n_embd_head = hparams.n_embd_head_k(il);
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_v(il));

        const int64_t n_head    = hparams.n_head(il);
        const int64_t n_head_kv = hparams.n_head_kv(il);

        const float freq_base_l  = model.get_rope_freq_base(cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);
        const int   n_rot_l      = hparams.n_rot(il);

        // norm
        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_tensor * freq_factors = nullptr;
        if (!hparams.is_swa(il)) {
            // full_attention layers use rope_freqs for proportional rope
            freq_factors = model.layers[il].rope_freqs;
        }

        // Q projection (shared for both non-KV and KV layers)
        // this is to mirror Gemma4Attention in pytorch code
        ggml_tensor * Qcur;
        {
            Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur_pos", il);
        }

        // self-attention
        if (hparams.has_kv(il)) {
            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = model.layers[il].wv
                                    ? build_lora_mm(model.layers[il].wv, cur)
                                    : Kcur; // if v_proj is not present, use Kcur as Vcur
            cb(Vcur, "Vcur", il);

            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
            Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps);

            cb(Kcur, "Kcur_normed", il);
            cb(Vcur, "Vcur_normed", il);

            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                 ext_factor, attn_factor, beta_fast, beta_slow);

            cb(Kcur, "Kcur_pos", il);

            // ----- TierKV (Path B) -----
            // Use the SVD-compressed Tier-1 path when (a) this layer carries SVD bases,
            // (b) the user enabled it via env vars, and (c) the cache already holds at least
            // kv_svd_size tokens so a non-empty Tier-1 region exists. Otherwise fall through to
            // the stock Path A below.
            //
            // Note: K/V projections are always computed (even for ubatches whose tokens land
            // entirely in Tier-1) because skipping them and not referencing v_idxs would leave
            // graph inputs un-allocated and crash llm_graph_input_attn_kv_iswa::set_input.
            // Skipping the K/V projection requires deeper graph-input refactoring (Phase 4 v3).
            const auto * mctx_iswa = inp_attn->mctx;
            const auto * mctx_cur  = hparams.is_swa(il) ? mctx_iswa->get_swa() : mctx_iswa->get_base();
            const uint32_t kv_svd_size = mctx_cur->get_kv_svd_size();
            const uint32_t n_kv_cur    = mctx_cur->get_n_kv();
            const bool tierkv_active   =
                model.layers[il].wuk != nullptr &&
                model.layers[il].wvs != nullptr &&
                kv_svd_size > 0 &&
                kv_svd_size < n_kv_cur;

            if (tierkv_active) {
                // 1) Compute ZSK for the current ubatch from the post-attn-norm input `cur`.
                //    cur shape:  [d_in, n_tokens]
                //    wvs shape:  [d_in, rank]   (col-major in ggml: ne[0]=d_in, ne[1]=rank)
                //    ZSK_cur:    [rank, n_tokens]
                ggml_tensor * ZSK_cur = ggml_mul_mat(ctx0, model.layers[il].wvs, cur);
                cb(ZSK_cur, "ZSK_cur", il);

                // 2) Make K/V/ZSK part of the graph and write them to caches.
                ggml_build_forward_expand(gf, Qcur);
                ggml_build_forward_expand(gf, Kcur);
                ggml_build_forward_expand(gf, Vcur);
                ggml_build_forward_expand(gf, ZSK_cur);

                const auto & k_idxs = hparams.is_swa(il) ? inp_attn->get_k_idxs_swa() : inp_attn->get_k_idxs();
                const auto & v_idxs = hparams.is_swa(il) ? inp_attn->get_v_idxs_swa() : inp_attn->get_v_idxs();

                ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, Kcur,    k_idxs, il));
                ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, Vcur,    v_idxs, il));
                // ZSK uses K's idxs; Tier-0 region of the ZSK buffer is overwritten harmlessly.
                ggml_build_forward_expand(gf, mctx_cur->cpy_zsk(ctx0, ZSK_cur, k_idxs, il));

                // 3) Read attention sources.
                const uint32_t n_t0 = std::min<uint32_t>(n_kv_cur, kv_svd_size);
                const uint32_t n_t1 = n_kv_cur - n_t0;
                ggml_tensor * K_exact = mctx_cur->get_k_n(ctx0, il, n_t0);   // [d_head, n_head_kv, n_t0, ns]
                ggml_tensor * V_exact = mctx_cur->get_v_n(ctx0, il, n_t0);   // [d_head, n_head_kv, n_t0, ns] (or transposed)
                ggml_tensor * ZSK_in  = mctx_cur->get_zsk(ctx0, il, n_t1);   // [rank, n_t1, ns]
                ggml_tensor * mask    = hparams.is_swa(il) ? inp_attn->get_kq_mask_swa() : inp_attn->get_kq_mask();

                // 4) Permute Q / K_exact / V_exact to flash-attn layout: [d_head, n_seq, n_head, ns].
                const int64_t n_stream_q = K_exact->ne[3]; // matches K cache stream count
                ggml_tensor * Qp = ggml_view_4d(ctx0, Qcur, Qcur->ne[0], Qcur->ne[1], Qcur->ne[2]/n_stream_q, n_stream_q,
                                                Qcur->nb[1], Qcur->nb[2], Qcur->nb[3]/n_stream_q, 0);
                Qp = ggml_permute(ctx0, Qp, 0, 2, 1, 3);
                ggml_tensor * Kp = ggml_permute(ctx0, K_exact, 0, 2, 1, 3);
                ggml_tensor * Vp = ggml_permute(ctx0, V_exact, 0, 2, 1, 3);

                // 5) Build the fused split-path attention.
                // pos_t1_offset = kv_svd_size assumes Tier-1 tokens are filled in monotonic order
                // starting from absolute position kv_svd_size (single-sequence linear-fill case).
                ggml_tensor * attn_out = ggml_fuse_kq_rope(
                    ctx0, Qp, Kp, Vp, ZSK_in,
                    model.layers[il].wuk, model.layers[il].wuv,
                    model.layers[il].attn_k_norm, freq_factors, mask,
                    n_rot_l, rope_type, n_ctx_orig, (int) kv_svd_size,
                    freq_base_l, freq_scale_l,
                    ext_factor, attn_factor, beta_fast, beta_slow,
                    hparams.f_attention_scale, /*logit_softcap=*/ 0.0f, hparams.f_norm_rms_eps);
                cb(attn_out, "kqv_out_tierkv", il);

                // 6) Reshape [d_head, n_q_heads, n_q, ns] -> [n_embd, n_tokens] and project.
                attn_out = ggml_reshape_2d(ctx0, attn_out,
                    attn_out->ne[0]*attn_out->ne[1], attn_out->ne[2]*attn_out->ne[3]);
                cur = build_lora_mm(model.layers[il].wo, attn_out);
            } else {
                cur = build_attn(inp_attn, model.layers[il].wo,
                        nullptr, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                        hparams.f_attention_scale, il);
            }
        } else {
            // reuse KV cache of earlier layers
            cur = build_attn(inp_attn,
                    model.layers[il].wo, nullptr,
                    Qcur, nullptr, nullptr, nullptr, nullptr, nullptr, hparams.f_attention_scale, il);
        }

        // TODO @ngxson : strip unused token right after the last KV layer to speed up prompt processing
        if (il == n_layer - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }
        cur = build_norm(cur,
                model.layers[il].attn_post_norm, nullptr,
                LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        ggml_tensor * attn_out = ggml_add(ctx0, cur, inpL);
        cb(attn_out, "attn_out", il);

        // feed-forward network
        const bool is_moe_layer = model.layers[il].ffn_gate_inp != nullptr;
        if (is_moe_layer) {
            // MLP (shared exp)
            ggml_tensor * cur_mlp = build_norm(attn_out,
                    model.layers[il].ffn_norm, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur_mlp, "ffn_norm_1", il);

            cur_mlp = build_ffn(cur_mlp,
                    model.layers[il].ffn_up,   nullptr, nullptr,
                    model.layers[il].ffn_gate, nullptr, nullptr,
                    model.layers[il].ffn_down, nullptr, nullptr,
                    nullptr,
                    LLM_FFN_GELU, LLM_FFN_PAR, il);
            cur_mlp = build_norm(cur_mlp,
                    model.layers[il].ffn_post_norm_1, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur_mlp, "ffn_mlp", il);

            // Expert FFN
            ggml_tensor * cur_moe = build_norm(attn_out,
                    model.layers[il].ffn_pre_norm_2, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur_moe, "ffn_norm_2", il);

            // custom MoE logits calculation (router operates on attn_out, not cur)
            ggml_tensor * tmp = ggml_rms_norm(ctx0, attn_out, hparams.f_norm_rms_eps);
            tmp = ggml_scale(ctx0, tmp, 1.0f / sqrtf((float) n_embd));
            tmp = ggml_mul(ctx0, tmp, model.layers[il].ffn_gate_inp_s);
            ggml_tensor * logits = build_lora_mm(model.layers[il].ffn_gate_inp, tmp); // [n_expert, n_tokens]
            cb(logits, "ffn_moe_logits", il);

            cur_moe = build_moe_ffn(cur_moe,
                    nullptr, // gate_inp
                    nullptr, // up_exps
                    nullptr, // gate_exps
                    model.layers[il].ffn_down_exps,
                    nullptr, // exp_probs_b (not used for gemma4)
                    n_expert, n_expert_used,
                    LLM_FFN_GELU, true,
                    1.0f,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il, logits,
                    model.layers[il].ffn_gate_up_exps,
                    nullptr, // up_exps_s
                    nullptr, // gate_exps_s
                    model.layers[il].ffn_down_exps_s);
            cur_moe = build_norm(cur_moe,
                    model.layers[il].ffn_post_norm_2, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur_moe, "ffn_moe", il);

            cur = ggml_add(ctx0, cur_mlp, cur_moe);
            cb(cur, "ffn_moe_combined", il);
        } else {
            cur = build_norm(attn_out,
                    model.layers[il].ffn_norm, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   nullptr, nullptr,
                    model.layers[il].ffn_gate, nullptr, nullptr,
                    model.layers[il].ffn_down, nullptr, nullptr,
                    nullptr,
                    LLM_FFN_GELU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        }
        cur = build_norm(cur,
                model.layers[il].ffn_post_norm, nullptr,
                LLM_NORM_RMS, -1);
        cb(cur, "ffn_post_norm", il);

        // residual connection
        cur = ggml_add(ctx0, cur, attn_out);

        // per-layer embedding
        if (inp_per_layer) {
            ggml_tensor * pe_in = cur;
            cb(cur, "pe_in", il);

            cur = build_lora_mm(model.layers[il].per_layer_inp_gate, cur); // [n_embd_per_layer, n_tokens]
            cur = ggml_gelu(ctx0, cur);

            ggml_tensor * inp_this_layer = ggml_view_2d_slice(ctx0, inp_per_layer, il); // [n_embd_per_layer, n_tokens]

            // TODO @ngxson : improve this
            if (il == n_layer - 1 && inp_out_ids) {
                inp_this_layer = ggml_get_rows(ctx0, inp_this_layer, inp_out_ids);
            }

            cur = ggml_mul(ctx0, cur, inp_this_layer);
            cur = build_lora_mm(model.layers[il].per_layer_proj, cur); // [n_embd, n_tokens]
            cur = build_norm(cur, model.layers[il].per_layer_post_norm, nullptr, LLM_NORM_RMS, il);
            cb(cur, "per_layer_embd_out", il);

            // residual connection
            cur = ggml_add(ctx0, pe_in, cur);
        }

        // layer_scalar
        if (model.layers[il].out_scale) {
            cur = ggml_mul(ctx0, cur, model.layers[il].out_scale);
            cb(cur, "out_scaled", il);
        }

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, nullptr,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);

    if (hparams.f_final_logit_softcapping) {
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
        cur = ggml_tanh(ctx0, cur);
        cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// equivalent to get_per_layer_inputs() in python code
// output shape: [n_embd_per_layer, n_layer, n_tokens]
ggml_tensor * llm_build_gemma4_iswa::build_inp_per_layer() {
    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);

    ggml_tensor * inp_per_layer;
    float tok_embd_scale = sqrtf((float) n_embd_per_layer);
    if (ubatch.token) {
        inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
        ggml_set_input(inp->tokens);
        res->t_inp_tokens = inp->tokens;

        inp_per_layer = ggml_get_rows  (ctx0, model.per_layer_tok_embd, inp->tokens);
        inp_per_layer = ggml_reshape_3d(ctx0, inp_per_layer, n_embd_per_layer, n_layer, n_tokens);
        inp_per_layer = ggml_scale     (ctx0, inp_per_layer, tok_embd_scale);
        cb(inp_per_layer, "inp_per_layer_selected", -1);

        res->add_input(std::move(inp));
    } else {
        // Multimodal embedding path: use padding token (ID=0) embedding
        // TODO: verify if this is the correct behavior in transformers implementation
        const int64_t embd_size = model.per_layer_tok_embd->ne[0];  // n_embd_per_layer * n_layer

        // Extract and dequantize padding token embedding (row 0)
        ggml_tensor * padding = ggml_view_1d(ctx0, model.per_layer_tok_embd, embd_size, 0);
        inp_per_layer = ggml_cast (ctx0, padding, GGML_TYPE_F32);
        inp_per_layer = ggml_scale(ctx0, inp_per_layer, tok_embd_scale);

        // Reshape to [n_embd_per_layer, n_layer, 1]
        inp_per_layer = ggml_reshape_3d(ctx0, inp_per_layer, n_embd_per_layer, n_layer, 1);
        cb(inp_per_layer, "inp_per_layer_multimodal", -1);
    }
    return inp_per_layer;
}

// equivalent to project_per_layer_inputs() in python code
// this calculates the per-layer inputs, so the final tensor shape will have n_layer as the last dim
// inp_batch     shape: [n_embd, n_tokens]
// inp_per_layer shape: [n_embd_per_layer, n_layer, n_tokens] (from build_inp_per_layer)
// output shape: [n_embd_per_layer, n_tokens, n_layer]
ggml_tensor * llm_build_gemma4_iswa::project_per_layer_inputs(ggml_tensor * inp_batch, ggml_tensor * inp_per_layer) {
    const float per_layer_projection_scale = 1.0f / sqrtf((float) n_embd);
    const float per_layer_input_scale      = 1.0f / sqrtf(2.0f);

    // note: this matrix multiplication will be performed in the input layer (i.e. on the CPU)
    ggml_tensor * per_layer_proj;
    per_layer_proj = ggml_mul_mat   (ctx0, model.per_layer_model_proj, inp_batch);
    per_layer_proj = ggml_scale     (ctx0, per_layer_proj, per_layer_projection_scale);
    per_layer_proj = ggml_reshape_3d(ctx0, per_layer_proj, n_embd_per_layer, n_layer, n_tokens);

    per_layer_proj = build_norm(per_layer_proj, model.per_layer_proj_norm, nullptr, LLM_NORM_RMS, -1);
    cb(per_layer_proj, "per_layer_proj", -1);

    inp_per_layer = ggml_add  (ctx0, per_layer_proj, inp_per_layer);
    inp_per_layer = ggml_scale(ctx0, inp_per_layer, per_layer_input_scale);
    cb(inp_per_layer, "inp_per_layer", -1);

    // permute to shape: [n_embd_per_layer, n_tokens, n_layer]
    inp_per_layer = ggml_cont(ctx0, ggml_permute(ctx0, inp_per_layer, 0, 2, 1, 3));
    return inp_per_layer;
}
