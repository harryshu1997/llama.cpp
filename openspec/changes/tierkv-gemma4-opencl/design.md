## Context

Gemma 4 E2B has 35 transformer layers and is the target deployment model on a OnePlus 12 (Snapdragon 8 Gen 3 + Adreno 750, 11.2 GB RAM, ~6.8 GB application budget). Its graph builder (`llm_build_gemma4_iswa`) uses:

- **GQA** — `n_head_kv < n_head`; K/V stored at fewer heads than Q.
- **ISWA** (Interleaved Sliding Window Attention) — each layer is either *sliding* (attends to the last W tokens) or *global* (attends to full context). The pattern is fixed per model config.
- **QK-Norm** — an RMSNorm is applied to `Q` and `K` **after** projection and **before** RoPE.
- **Shared KV layers** — a suffix of layers reuses the K/V tensors of an earlier "source" layer.

A reference TierKV implementation exists in a private fork (`llm.cpp @ harry-tierKV`) but:
1. Is wired only into `llm_build_llama`, not any Gemma builder.
2. Produces garbage output today (user-reported) — the branch's own commit message flags the SVD recompute path as unverified, and a shape comment (`// kv_approx is (512, N)`) suggests an off-by-one in the reconstruction.
3. Predates QK-Norm and ISWA concerns — it always assumes global dense attention with RoPE-after-projection and no per-Q/per-K norm.

Constraints from the user:
- No PMCO solver, no length predictor, no auto-tuner — tier boundaries come from env vars `KV_SVD_SIZE` and `KV_OFFLOAD_SIZE`.
- Success criterion: coherent text from Gemma 4 E2B under Tier-0+1 and Tier-0+1+2 configs, not perplexity-tight parity with FP16.
- Implementation will proceed step by step with the user driving; we must be able to verify correctness at each phase independently.

## Goals / Non-Goals

**Goals:**
- End-to-end inference of Gemma 4 E2B through `llama-cli` on x86 Linux with OpenCL, with KV cache split across three tiers based on env-var thresholds.
- CPU reference implementation of the fused split-path attention op that is bit-identical (within fp16 rounding) to an unfused reference `matmul → norm → rope → attn → matmul` sequence, so the SVD math can be validated **before** any OpenCL work.
- A minimal regression: a fixed short prompt produces coherent (non-garbage) output under three configs — Tier-0 only (baseline), Tier-0+1, Tier-0+1+2.
- An OpenCL kernel for `GGML_OP_FUSE_KQ_ROPE` whose output matches CPU reference within a set L2 tolerance.

**Non-Goals:**
- Reproducing the paper's exact throughput or memory numbers.
- PMCO solver, entropy predictor, kernel auto-tuner.
- On-device Android builds, `adb` deployment scripts, thermal-aware scheduling. (A separate change can cover cross-compilation once x86 works.)
- Supporting LLM_ARCH other than `LLM_ARCH_GEMMA4` in this change. Llama / Qwen are follow-ups.
- INT8 / INT4 quantization integration — FP16 only here.

## Decisions

### D1. Start with a CPU reference before any OpenCL
**Decision.** Implement `GGML_OP_FUSE_KQ_ROPE` first on the CPU backend as a clear reference, validated against a plain unfused sequence of existing ops on a tiny model. Only then port to OpenCL.

**Why.** The reference implementation's accuracy bug proves this is hard to debug directly in a kernel. A CPU reference gives us a golden output to diff against and lets us isolate whether garbage output is from the SVD math, the QK-Norm-before-RoPE ordering, the tier routing, or the kernel itself.

**Alternatives considered.** Start directly in OpenCL (faster but uncertain — we would have no known-good baseline). Use the reference implementation's kernel as-is (rejected — it has the bug and was never wired to Gemma).

### D2. Per-layer tier policy, not global
**Decision.** The tier decision `{Tier-0 only | Tier-0+1 | Tier-0+1+2}` is computed per layer at graph build time, not once per forward pass. For sliding-window layers where `W ≤ KV_SVD_SIZE`, skip SVD entirely (all reachable tokens already fit in Tier-0). Shared-KV layers reuse the source layer's ZSK buffer.

**Why.** Applying SVD to a sliding-window layer's "older" tokens is pointless because that layer cannot attend to them. Forcing SVD everywhere would add compute + memory with zero accuracy or memory-budget benefit.

**Alternatives considered.** Single global policy (simpler but wasteful and potentially harmful on SWA layers). Tier policy driven by the PMCO solver (out of scope for this change).

### D3. QK-Norm is applied inside the fused kernel, *before* RoPE
**Decision.** The fused op takes `w_q_norm`, `w_k_norm`, `rope_freqs`, `W_uk`, and `ZSK` as inputs. Computation order is:
```
K_recon = ZSK · W_uk           # reconstruct full-rank K from latent
K_norm  = RMSNorm(K_recon; w_k_norm)
K_rope  = RoPE(K_norm; rope_freqs)
Q_norm  = RMSNorm(Q; w_q_norm)
Q_rope  = RoPE(Q_norm; rope_freqs)
scores  = Q_rope · K_ropeᵀ · scale
```

**Why.** Gemma 4's `llm_build_gemma4_iswa` does exactly `q = RMSNorm(q) → RoPE; k = RMSNorm(k) → RoPE`. If we apply the norm to pre-projection Q/K but reconstruct K from SVD without re-normalizing, the resulting K has a different statistical distribution than the K path saw during training — this is a likely root cause of the garbage output bug in the reference implementation, which only supported LLaMA (no QK-Norm).

**Alternatives considered.** Normalize in the caller, pass normalized K to the kernel (rejected — forces materializing the full-rank K tensor outside the kernel, defeating the purpose of the fused reconstruction). Skip QK-Norm in the SVD tier (rejected — would break training-time invariants and guarantees garbage output).

### D4. GGUF tensor layout and weight-conversion ordering
**Decision.** For each Gemma 4 layer `l`, export three new tensors:
- `blk.l.attn_uk.weight` : fp16, shape `[n_embd_k_gqa, rank_l]` — left factor `U_k`.
- `blk.l.attn_uv.weight` : fp16, shape `[n_embd_v_gqa, rank_l]` — left factor `U_v`.
- `blk.l.attn_vs.weight` : fp16, shape `[rank_l, max_svd_tokens]` — singular-value-scaled right factor cache basis. Stored column-major to match llama.cpp convention (outer dim = first dim in GGUF).

Decomposition: joint SVD of `stack([K_weight; V_weight], axis=0) ∈ ℝ^[(nk+nv) × d_in]`, truncated to `rank_l`, then split `U` into `U_k` and `U_v`. `ZSK` is computed online per forward pass as `ZSK = K · Σ^{-1} · ... ` — i.e., no persistent disk artifact for the dynamic latent codes; only the projection bases `U_k, U_v` are persisted in GGUF.

**Why col-major.** llama.cpp ggml tensors are stored with `ne[0]` as the contiguous (fastest) dimension, which corresponds to column-major in standard math notation. The reference implementation's conversion script had commented-out fusion tensors suggesting unfinished layout work — a likely secondary source of "garbage output" if the user re-uses that converter. We fix the layout contract explicitly and write a round-trip unit test.

**Alternatives considered.** Separate SVD for K and V (rejected — paper §4.2 and the reference implementation both use joint SVD to reduce FLOP for V accumulation in latent space). Store full `V_u = Σ · Vᵀ` and skip latent-space accumulation (simpler but larger and slower per the paper's Algorithm 2).

### D5. Disk offload path comes last and is bypassed by default
**Decision.** The Tier-2 (disk offload) ops are implemented and wired, but `KV_OFFLOAD_SIZE` defaults to a value larger than any test prompt's context so the offload path is not exercised in the first milestones. First milestone target: Tier-0 only (Gemma 4 baseline still works after our changes). Second: Tier-0+1. Third: add Tier-2.

**Why.** The offload path adds async I/O, filesystem lifecycle, and prefetch overlap complexity that are independent of the SVD correctness problem. Decoupling them lets us sign off on each piece.

**Alternatives considered.** Build all three tiers at once (rejected — too many simultaneous failure modes).

### D6. OpenCL kernel structure mirrors the reference's `fuse_kq_rope.cl`
**Decision.** A single `.cl` source file with multiple kernel entry points:
- `kernel_fuse_kq_rope_reconstruct` — baseline version, no tiling tricks.
- (Optional, only if baseline is correct and too slow) `kernel_fuse_kq_rope_reconstruct_l2` / `_l4` — loop-unrolled variants.

We do **not** port the `_hist` variant (historical K concat) from the reference; that is the path Gemma's graph handles via its existing KV cache views, which we keep.

**Why.** Kernel variants for unrolling are performance tuning. Correctness comes from the baseline kernel. Auto-tuning is out of scope, so we do not need parameterized variants.

### D7. Weight conversion is gated by a CLI flag and does not change default behavior
**Decision.** `convert_hf_to_gguf.py` gets `--svd-rank INT` (default 0 = off). When 0, conversion is identical to today. When > 0, per-layer SVD runs and the three new tensors are emitted alongside the original `attn_k` / `attn_v` (we keep the full-precision K/V projection weights because the forward pass still needs them — Tier-0 uses standard K,V paths).

**Why.** We must not break existing GGUF regeneration workflows. Also, keeping the original K/V projections means Tier-0 forward math is unchanged — the SVD path only affects how *cached* K/V from prior tokens are stored, not how *newly projected* K/V from the current token enter the cache.

## Risks / Trade-offs

- **[Risk] SVD rank too low → quality drop.** The paper reports large accuracy swings at rank ≤ 25%. → Mitigation: default `--svd-rank` in conversion to `d_head * n_head_kv / 2` (~50%). Let the user lower it after baseline parity is confirmed.

- **[Risk] Per-layer tier policy interacts badly with shared KV layers.** If layer `L` shares KV with source layer `S < L` and `S` was chosen to compress to Tier-1 but `L` would not, we have a conflict. → Mitigation: tier policy for shared-KV layers is locked to the source layer's policy; enforced at graph-build time with an assertion.

- **[Risk] RoPE scaling / position-ID offsets for reconstructed K tokens.** The original K tensors in the cache were RoPE'd at their original insertion positions. When we reconstruct K from ZSK in a later step, we must apply RoPE with the original position, not the current one. → Mitigation: ZSK store includes position IDs; the fused kernel takes `pos[]` as input, same contract as llama.cpp's existing `ggml_rope_ext`.

- **[Risk] OpenCL backend on x86 dev machine may not expose the exact extensions that Adreno 750 does.** Kernel that compiles on Mesa rusticl may fail on Qualcomm's OpenCL. → Mitigation: keep the kernel in OpenCL 2.0 core features only (no subgroups, no fp16 extension yet). If Adreno is faster with fp16 math, that's a follow-up optimization.

- **[Trade-off] We keep the full-precision `attn_k_weight` and `attn_v_weight` alongside the new SVD tensors.** This inflates the GGUF by ~(rank × d_in × 2 × n_layers × 2B) bytes. For Gemma 4 E2B at rank 512, this is a few tens of MB — acceptable. → The alternative (only storing SVD factors) would require re-deriving projections on every forward pass even for tokens that fit in Tier-0, which is a larger perf cost.

- **[Trade-off] No INT8/INT4 quant in this change.** FP16 only simplifies kernel and memory accounting but forgoes some memory savings. → Documented non-goal; a follow-up change can add Q8_0 KV support.

## Migration Plan

1. Branch from current master (clean). No migration of existing GGUF files is required — existing files continue to work because the new code paths are only taken when the new tensors are present *and* env vars enable them.
2. Conversion of an existing Gemma 4 E2B HF checkpoint with `--svd-rank 512` produces a new GGUF suffixed `-tierkv`. Users keep both files.
3. Rollback: unset `KV_SVD_SIZE` and `KV_OFFLOAD_SIZE` → the graph falls back to Tier-0-only, i.e., stock Gemma 4 behavior. If that still misbehaves, use the un-suffixed GGUF.

## Confirmed Facts (resolved Open Questions)

Confirmed by inspecting `/home/myid/zs89458/Documents/models/gemma-4-E2B/config.json`:

- **F1. Architecture is MQA, not GQA.** `num_key_value_heads = 1`. Reference TierKV implementation assumed GQA — its MUL_MAT broadcast paths must be re-derived for `n_kv_heads = 1`.
- **F2. Per-layer-type head_dim.** Sliding layers `head_dim = 256`; full-attention layers `global_head_dim = 512`. SVD rank in conversion script must accept per-layer-type override.
- **F3. ISWA pattern.** Layers `[4, 9, 14, 19, 24, 29, 34]` are `full_attention`; the remaining 28 are `sliding_attention`. Sliding window `W = 512`. With the default `KV_SVD_SIZE = 512`, sliding layers always fit in Tier-0 — only the 7 global layers traverse Tier-1/2.
- **F4. Shared KV layers.** `num_kv_shared_layers = 20`. The first 15 layers own their KV; layers 15-34 reuse KV from earlier layers. Mapping logic lives in `src/models/gemma4-iswa.cpp` (`hparams.has_kv(il)` branch). Tier policy for layer `l` follows the policy chosen for layer `l`'s actual KV owner.
- **F5. Partial RoPE on full layers.** Full layers use `partial_rotary_factor = 0.25` with `rope_type = "proportional"` and `rope_theta = 1e6`. llama.cpp implements this with `freq_factors = [1, 1, ..., inf, inf]` so only the first 25% of dims rotate. The fused op MUST accept `freq_factors` and pass it to its internal RoPE step. Sliding layers use `rope_theta = 10000`, no partial rotation, no freq_factors.
- **F6. V is also normalized.** `Vcur = ggml_rms_norm(V)` with no learned weights ([src/models/gemma4-iswa.cpp:92](../../../src/models/gemma4-iswa.cpp)). The fused op must apply unweighted RMSNorm to reconstructed V from latent space before accumulation. The spec for `fused-split-path-attention` is updated accordingly.
- **F7. Multimodal model.** The HF checkpoint is `Gemma4ForConditionalGeneration` with vision (16-layer ViT) and audio (12-layer) encoders. **TierKV touches only the text decoder.** Vision/audio encoders are out of scope.
- **F8. Target device confirmed.** OnePlus 12, Qualcomm Snapdragon 8 Gen 3, Adreno 750. Develop-time OpenCL backend is NVIDIA RTX A6000 (`libnvidia-opencl.so.1`, OpenCL 3.0 capable). Kernels MUST be restricted to OpenCL 1.2 / 2.0 core features and avoid NVIDIA-specific extensions to minimize Adreno portability risk. Final acceptance must run on the actual OnePlus 12.

## Remaining Open Questions

- **Q4.** The current OpenCL backend op-registration pattern in `ggml/src/ggml-opencl/ggml-opencl.cpp` will be re-examined during Phase 6; mostly mechanical.
- **Q5.** Whether the `attn_post_norm` weight needs adjustment for the SVD-attention path. Current Gemma 4 graph applies `attn_post_norm` to the attention output; if our fused op already produces the post-attention sum in the same numeric distribution as the unfused path, no change is needed. To verify in Phase 4.6.
