## ADDED Requirements

### Requirement: Per-layer joint SVD export for Gemma 4
The conversion tool SHALL support exporting per-layer joint SVD factors of the K and V projection weights for `LLM_ARCH_GEMMA4` models, gated behind a new `--svd-rank <int>` CLI flag. The flag default is `0` (disabled). When enabled, for each transformer layer the tool SHALL write three new tensors into the output GGUF and preserve the original `attn_k.weight` and `attn_v.weight` tensors.

#### Scenario: Flag disabled preserves baseline output
- **WHEN** the user runs `convert_hf_to_gguf.py` against a Gemma 4 E2B checkpoint without `--svd-rank`
- **THEN** the produced GGUF is byte-for-byte identical in tensor set to the current master's output (no `attn_uk`, `attn_uv`, `attn_vs` tensors present)

#### Scenario: Flag enabled adds three tensors per layer
- **WHEN** the user runs the conversion with `--svd-rank 512` against `google/gemma-4-E2B`
- **THEN** every layer block `blk.l` in the resulting GGUF contains `attn_uk.weight`, `attn_uv.weight`, `attn_vs.weight`
- **AND** the original `attn_k.weight` and `attn_v.weight` tensors are also present
- **AND** a new GGUF metadata key `gemma4.svd.rank` equals `512`

#### Scenario: Invalid rank rejected
- **WHEN** `--svd-rank` is negative or exceeds `min(n_embd_k_gqa, n_embd_v_gqa, d_in)` for any layer
- **THEN** the converter exits with a non-zero status and prints an error naming the first offending layer

### Requirement: Joint SVD decomposition semantics
The converter SHALL compute, for each layer, a joint SVD of `stack([W_k; W_v])` (vertical concatenation along the output dimension) and split the left singular vectors `U` into `U_k` (rows `0 : n_embd_k_gqa`) and `U_v` (rows `n_embd_k_gqa : n_embd_k_gqa + n_embd_v_gqa`). The stored `attn_vs.weight` tensor SHALL equal `Σ · Vᵀ` truncated to the first `rank` rows, in fp16.

#### Scenario: Round-trip reconstruction error bounded
- **WHEN** a unit test multiplies `U_k @ (Σ · Vᵀ)` from the exported tensors against an input embedding `x`
- **AND** compares to the original `W_k @ x` on the same input
- **THEN** the relative L2 error is below `0.05` for `rank = 512` on `google/gemma-4-E2B`

### Requirement: Column-major tensor layout
All three new tensors SHALL be stored with ne[0] as the contiguous innermost dimension matching llama.cpp's existing convention:
- `attn_uk.weight`: ne = `[n_embd_k_gqa, rank]`
- `attn_uv.weight`: ne = `[n_embd_v_gqa, rank]`
- `attn_vs.weight`: ne = `[rank, d_in]`

#### Scenario: Tensor shapes verified on load
- **WHEN** `llama.cpp` loads the converted GGUF
- **THEN** each `attn_uk` tensor reports `ne[0] == n_embd_k_gqa` and `ne[1] == rank` via `ggml_n_dims` and `tensor->ne[]`
- **AND** a mismatched shape triggers a `GGML_ASSERT` with the layer index in the message

### Requirement: Shared KV layer metadata
The converter SHALL emit a GGUF metadata array `gemma4.svd.shared_kv_source` of length `n_layer` where element `l` is either `l` itself (layer has its own SVD) or an earlier layer index `s < l` that `l` shares with. The converter SHALL NOT recompute SVD for a shared layer; it SHALL instead copy the source layer's tensors under the shared layer's name so runtime lookup is uniform.

#### Scenario: Shared layer points to source
- **WHEN** Gemma 4 E2B's config specifies that layers 32-34 share KV with layer 27
- **THEN** `gemma4.svd.shared_kv_source[32] == gemma4.svd.shared_kv_source[33] == gemma4.svd.shared_kv_source[34] == 27`
- **AND** `blk.32.attn_uk.weight` has the same data bytes as `blk.27.attn_uk.weight`
