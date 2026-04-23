## ADDED Requirements

### Requirement: Gemma 4 graph routes through three-tier attention
`llm_build_gemma4_iswa` SHALL select one of three attention paths per layer at graph-build time based on the current `n_kv` and tier boundaries:
- **Path A (Tier-0 only)**: `actual_tokens ≤ KV_SVD_SIZE` OR the layer is a sliding-window layer with `W ≤ KV_SVD_SIZE`. Uses the existing pre-TierKV attention path — no behavior change.
- **Path B (Tier-0 + Tier-1)**: `KV_SVD_SIZE < actual_tokens ≤ KV_OFFLOAD_SIZE` AND the layer is global OR has `W > KV_SVD_SIZE`. Calls the new `build_attn_svd_combined` with `ZSK`, `W_uk`, `W_uv`.
- **Path C (Tier-0 + Tier-1 + Tier-2)**: `actual_tokens > KV_OFFLOAD_SIZE` AND the layer is global. Calls `build_attn_svd_offload_combined` which additionally inserts `KV_PREFETCH_START` / `KV_PREFETCH_WAIT` around the disk-load path.

#### Scenario: Short prompt uses Path A only
- **WHEN** inference runs on a 100-token prompt with `KV_SVD_SIZE=512`
- **THEN** no layer invokes `build_attn_svd_combined` or `build_attn_svd_offload_combined`
- **AND** output tokens are bit-identical to a run on the same GGUF without TierKV env vars set

#### Scenario: Long prompt uses Path B on global layers, Path A on SWA layers
- **WHEN** inference runs a 1500-token prompt with `KV_SVD_SIZE=512, KV_OFFLOAD_SIZE=2048`
- **AND** the model's sliding window `W = 512`
- **THEN** global layers run Path B and sliding-window layers run Path A

#### Scenario: Very long prompt uses Path C
- **WHEN** inference runs a 3000-token prompt with `KV_SVD_SIZE=512, KV_OFFLOAD_SIZE=2048`
- **THEN** at least one global layer invokes `build_attn_svd_offload_combined` in the current decoding step's graph

### Requirement: QK-Norm applied before RoPE in SVD path
The SVD-path attention builders SHALL pass `w_q_norm` and `w_k_norm` tensors into the fused op so that normalization happens **before** RoPE, consistent with Gemma 4's non-SVD path. The builders SHALL NOT apply RMSNorm to the reconstructed K tensor post-RoPE.

#### Scenario: Norm ordering explicit in graph
- **WHEN** the graph is dumped via `ggml_graph_print` for a Path B layer
- **THEN** the `FUSE_KQ_ROPE` node's inputs include both `attn_q_norm.weight` and `attn_k_norm.weight` tensors
- **AND** no `RMS_NORM` node appears downstream of the `FUSE_KQ_ROPE` output feeding back into attention

### Requirement: Shared-KV layers inherit source layer's tier path
For any layer `l` with `gemma4.svd.shared_kv_source[l] = s`, the builder SHALL use the tier path chosen for layer `s` and reuse `s`'s `ZSK`, `W_uk`, `W_uv` tensors. The builder SHALL NOT compute a new SVD projection at runtime for the shared layer.

#### Scenario: Shared layers do not duplicate SVD work
- **WHEN** layer 32 shares KV with layer 27 and both reach Path B
- **THEN** the compute graph for layer 32 contains no `MUL_MAT` node producing a second `ZSK` tensor
- **AND** layer 32's `FUSE_KQ_ROPE` references the same `ZSK` buffer as layer 27's

### Requirement: Graceful fallback when TierKV tensors absent
When the loaded GGUF does not contain `attn_uk` / `attn_uv` / `attn_vs` tensors, `llm_build_gemma4_iswa` SHALL always take Path A regardless of env var settings, with a single warning log per session.

#### Scenario: Non-TierKV GGUF + env vars set
- **WHEN** a stock Gemma 4 E2B GGUF is loaded with `KV_SVD_SIZE=512 KV_OFFLOAD_SIZE=2048`
- **THEN** the log line `TierKV disabled: GGUF does not contain SVD tensors, falling back to Tier-0 only` appears exactly once
- **AND** inference proceeds unchanged
