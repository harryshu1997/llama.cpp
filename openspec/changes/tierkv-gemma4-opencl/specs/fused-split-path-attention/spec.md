## ADDED Requirements

### Requirement: GGML_OP_FUSE_KQ_ROPE public op
A new ggml operation `GGML_OP_FUSE_KQ_ROPE` SHALL be added to `ggml/include/ggml.h` with a constructor `ggml_fuse_kq_rope(...)` that takes as inputs: `Q` (post-projection, pre-norm), `K_exact` (Tier-0 K tensor or NULL), `ZSK` (Tier-1 latent codes or NULL), `W_uk` (K reconstruction basis), `W_uv` (V reconstruction basis, may be NULL if V-latent-accum disabled), `w_q_norm`, `w_k_norm` (RMSNorm weights), `rope_freqs` (RoPE factor table), `pos` (int32 position IDs, one per token across tiers), `mask` (attention mask), and scaling / mode parameters matching `ggml_rope_ext`.

#### Scenario: Op discoverable at runtime
- **WHEN** code calls `ggml_op_name(GGML_OP_FUSE_KQ_ROPE)`
- **THEN** it returns the string `"FUSE_KQ_ROPE"` (or similar stable name)

#### Scenario: Constructor validates shapes
- **WHEN** `ggml_fuse_kq_rope` is called with `W_uk` whose `ne[1]` disagrees with `ZSK->ne[0]`
- **THEN** a `GGML_ASSERT` fires naming both tensors and their shapes

### Requirement: CPU reference semantics
The CPU backend implementation of `GGML_OP_FUSE_KQ_ROPE` SHALL compute, for each output head and token-pair `(i, j)` where `i` is a Q token and `j` spans Tier-0 keys then Tier-1 keys:

1. `Q_norm[h, i, d]  = RMSNorm(Q[h, i, :], w_q_norm)[d]`
2. `Q_rot[h, i, :]   = RoPE(Q_norm[h, i, :], pos[i], rope_freqs)`
3. For Tier-0 key index `j` in `[0, N_exact)`: `K_used[h, j, :] = RoPE(RMSNorm(K_exact[h, j, :], w_k_norm), pos[j], rope_freqs)`. If `K_exact` is already stored post-RoPE-post-norm, this step is a no-op view (caller signals via `k_exact_already_rotated` flag).
4. For Tier-1 key index `j` in `[N_exact, N_exact + N_svd)`: `K_recon[h, j, :] = (ZSK[:, j - N_exact]ᵀ · W_uk)[h, :]`; `K_used[h, j, :] = RoPE(RMSNorm(K_recon[h, j, :], w_k_norm), pos[j], rope_freqs)`.
5. `score[h, i, j]    = Q_rot[h, i, :] · K_used[h, j, :] · scale` plus `mask[i, j]`.
6. `attn[h, i, j]     = softmax_j(score[h, i, :])[j]`
7. For V accumulation: `O[h, i, :] = Σ_j (attn[h, i, j] · V_used[h, j, :])` where `V_used` follows the same split: Tier-0 reads `V_exact[h, j, :]` directly; Tier-1 uses `O_latent[h, i, :] = Σ_j (attn[h, i, j] · ZSV[:, j - N_exact])` then `O_tier1 = O_latent · W_uvᵀ`, summed into `O`.

The order **RMSNorm → RoPE** is mandatory for both Q and K.

#### Scenario: CPU ref matches unfused sequence
- **WHEN** a test builds two graphs — (A) `ggml_fuse_kq_rope(...)` and (B) an unfused `matmul + rms_norm + rope_ext + matmul + soft_max + matmul` sequence over the same inputs — on random fp32 data of small dimension (e.g., d=64, n_head=4, n_tokens=16)
- **THEN** the outputs match within relative L2 error `1e-4`

#### Scenario: Tier-0-only call matches stock attention
- **WHEN** `ggml_fuse_kq_rope` is invoked with `ZSK=NULL, W_uk=NULL, W_uv=NULL`
- **THEN** its output equals the result of the existing `ggml_flash_attn_ext` or equivalent sequence within fp16 rounding

#### Scenario: Tier-1-only call ignores Tier-0
- **WHEN** `K_exact=NULL` is passed and only `ZSK, W_uk` are provided
- **THEN** the op computes attention using only the reconstructed SVD keys and produces output of the correct shape

### Requirement: V accumulation in latent space when W_uv provided
When `W_uv` is non-NULL, V accumulation for Tier-1 tokens SHALL be performed in the latent space first (accumulating into a `[rank, d_head]` buffer per head per query token) and projected to full dimension **exactly once** per query token, matching Algorithm 2 of the TierKV paper.

#### Scenario: Latent-space accumulation saves FLOPs
- **WHEN** a benchmark sets `rank = d_head / 2` and measures FLOP count through `ggml_graph_compute_flops` or equivalent instrumentation
- **THEN** the op's FLOP count for Tier-1 V is within 60% of a naive `V_recon · attn` implementation, demonstrating the latent-space path is active

### Requirement: OpenCL kernel parity with CPU reference
An OpenCL kernel implementation of `GGML_OP_FUSE_KQ_ROPE` SHALL be added and registered with the existing `ggml-opencl` backend. For any input the backend accepts, its output SHALL match the CPU reference within relative L2 error `5e-3` (allowing fp16 intermediate rounding).

#### Scenario: Kernel parity test
- **WHEN** a test runs the same inputs through CPU and OpenCL backends and compares outputs
- **THEN** relative L2 error is below `5e-3` on at least 10 random seeds with `(n_tokens, n_head, d_head) ∈ {(16, 4, 64), (64, 8, 128), (256, 16, 128)}`
