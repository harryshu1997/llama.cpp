## ADDED Requirements

### Requirement: Three-tier KV cache data layout
The `llama_kv_cache` data structure SHALL maintain three distinct storage regions per layer:
- Tier-0: full-precision fp16 `K` and `V` tensors, identical to today's behavior, sized to hold up to `KV_SVD_SIZE` tokens.
- Tier-1: an fp16 `ZSK` tensor of shape `[rank_l, KV_OFFLOAD_SIZE - KV_SVD_SIZE]` storing SVD latent codes for tokens in the `[KV_SVD_SIZE, KV_OFFLOAD_SIZE)` index range.
- Tier-2: a per-layer disk-backed store holding tokens with index `≥ KV_OFFLOAD_SIZE`, structured as fixed-size blocks aligned to `n_pad`.

#### Scenario: Cache allocation respects env vars
- **WHEN** the process starts with `KV_SVD_SIZE=512 KV_OFFLOAD_SIZE=2048`
- **THEN** the log line `KV cache config: kv_svd_size=512, kv_offload_size=2048, svd_tokens=1536` is emitted
- **AND** per-layer Tier-0 tensor has `ne[1] >= 512`, Tier-1 ZSK has `ne[1] >= 1536`

#### Scenario: Env var alignment to n_pad
- **WHEN** the user sets `KV_SVD_SIZE=500` and `n_pad = 32`
- **THEN** the effective `kv_svd_size` is rounded down to `480` and logged

#### Scenario: Degenerate single-tier mode
- **WHEN** `KV_SVD_SIZE` is at least the model's max context
- **THEN** no Tier-1 or Tier-2 allocations occur, and forward passes are numerically identical to stock llama.cpp on the same GGUF

### Requirement: Tier boundaries are env-var driven
Tier boundaries SHALL be read from environment variables `KV_SVD_SIZE` and `KV_OFFLOAD_SIZE` at `llama_kv_cache` construction time. If unset, `KV_SVD_SIZE` defaults to the full cache size (effectively disabling Tier-1 and Tier-2). If set, values MUST satisfy `0 < KV_SVD_SIZE ≤ KV_OFFLOAD_SIZE` and be multiples of `n_pad` (or be rounded down). Values not meeting these constraints SHALL cause a non-fatal warning log and a rounded-down replacement.

#### Scenario: Missing env vars fall back to single tier
- **WHEN** neither `KV_SVD_SIZE` nor `KV_OFFLOAD_SIZE` is set
- **THEN** the cache allocates only Tier-0 and its size equals the existing default behavior

#### Scenario: Inconsistent env vars rejected
- **WHEN** `KV_SVD_SIZE=2048` and `KV_OFFLOAD_SIZE=1024`
- **THEN** the construction logs a warning and clamps `KV_OFFLOAD_SIZE` to `KV_SVD_SIZE`

### Requirement: ZSK read/write accessors
`llama_kv_cache` SHALL expose two accessors usable from graph-build code:
- `get_zsk(ctx, layer_id, n_tokens, sinfo) → ggml_tensor*` — returns a view over the ZSK region covering the requested token range.
- `cpy_zsk(ctx, zsk_new, zsk_idxs, layer_id, sinfo)` — stores newly computed latent codes into the cache at the requested token indices.

#### Scenario: Read-after-write consistency
- **WHEN** `cpy_zsk` writes ZSK values `[a, b, c]` at positions `[10, 11, 12]` during decode step N
- **AND** `get_zsk` is called at step N+1 for the range `[10, 13)`
- **THEN** the returned view contains `[a, b, c]` in order, bit-for-bit

### Requirement: Shared KV layers bind to source layer's tier state
When layer `l` has `gemma4.svd.shared_kv_source[l] = s` with `s < l`, the accessors `get_zsk(ctx, l, …)` and reads of Tier-0 / Tier-2 for layer `l` SHALL internally redirect to layer `s`. Writes via `cpy_zsk(ctx, …, l, …)` for shared layers SHALL be rejected (assertion) — only the source layer writes.

#### Scenario: Shared layer reads from source
- **WHEN** layer 32 shares KV with layer 27, and a graph runs `get_zsk(..., layer_id=32, ...)`
- **THEN** the returned tensor aliases the same memory as `get_zsk(..., layer_id=27, ...)`
