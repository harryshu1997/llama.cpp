## Why

Running Gemma 4 E2B on a OnePlus 12 (11.2 GB physical RAM, ~6.8 GB app budget) hits the memory wall long before the model's supported context length — the FP16 KV cache alone scales linearly and dominates the application budget at modest context sizes. TierKV (ICS 2026 submission #732) resolves this by splitting the KV cache across three tiers — FP16 exact (Tier-0), SVD-compressed (Tier-1), and flash-offloaded (Tier-2) — preserving full context without eviction. A reference implementation exists in a private fork (`llm.cpp @ harry-tierKV`), but it targets Llama architectures only, produces garbage output in its current state due to a known shape mismatch in the SVD recompute path, and has not been wired into Gemma 4's distinctive graph (ISWA + QK-Norm + shared KV layers). This change re-implements TierKV cleanly in current `llama.cpp` master for Gemma 4 E2B on the OpenCL backend.

## What Changes

- Export per-layer SVD factors (`attn_uk`, `attn_uv`, `attn_vs`) during HuggingFace → GGUF conversion for Gemma 4 E2B. Gated by a CLI flag; default off preserves existing baseline conversion.
- Add three-tier KV cache layout to `llama_kv_cache` with tier boundaries configurable via `KV_SVD_SIZE` and `KV_OFFLOAD_SIZE` environment variables (no PMCO solver, no length predictor).
- Add new `ggml` ops for the disk-offload path: `GGML_OP_KV_OFFLOAD`, `GGML_OP_KV_LOAD`, `GGML_OP_KV_CLEAN`, `GGML_OP_KV_PREFETCH_START`, `GGML_OP_KV_PREFETCH_WAIT`. CPU backend only at first.
- Add new `ggml` op `GGML_OP_FUSE_KQ_ROPE` implementing fused split-path attention (reconstruct K on-chip from `ZSK · W_uk`, apply QK-Norm **before** RoPE, run attention, accumulate V in latent space, project once). CPU reference + OpenCL kernel.
- Wire the three-tier attention path into `llm_build_gemma4_iswa`, handling Gemma-4-specific wrinkles that the reference implementation never faced: **BREAKING** the reference's assumption that tier selection is global — tier strategy is now per-layer, skipping Tier-1/Tier-2 entirely for sliding-window layers when the window fits in Tier-0. Shared-KV layers reuse a single ZSK store per shared group.
- Add a CPU reference implementation of the fused split-path attention before touching OpenCL, so the SVD math can be validated against FP16 baseline in isolation.
- **NOT in scope**: PMCO optimization solver, entropy-guided length predictor, OpenCL kernel auto-tuner, on-device deployment scripts for Android. These can be added in follow-up changes.

## Capabilities

### New Capabilities
- `svd-weight-export`: Offline SVD decomposition of K/V projection weights and export as new GGUF tensors (`attn_uk`, `attn_uv`, `attn_vs`).
- `three-tier-kv-cache`: KV cache data structure spanning Tier-0 (FP16 exact), Tier-1 (SVD-compressed ZSK), Tier-2 (flash offload), with tier boundaries controlled by env vars.
- `fused-split-path-attention`: New ggml op and CPU + OpenCL implementations that fuse K reconstruction, QK-Norm, RoPE, attention score, and latent-space V accumulation in a single pass.
- `kv-disk-offload-ops`: New ggml ops for async disk I/O of KV blocks (`KV_OFFLOAD`, `KV_LOAD`, `KV_PREFETCH_START`, `KV_PREFETCH_WAIT`, `KV_CLEAN`).
- `gemma4-tierkv-graph`: Gemma 4-specific graph wiring that routes per-layer attention through the three-tier path, honoring ISWA, QK-Norm ordering, and shared-KV layer constraints.

### Modified Capabilities
<!-- No existing openspec specs to modify — this is the first TierKV-related proposal in this repo. -->

## Impact

- **Affected code**:
  - `convert_hf_to_gguf.py` — new per-model SVD export flag for Gemma 4.
  - `gguf-py/gguf/constants.py` and `tensor_mapping.py` — new tensor names.
  - `ggml/include/ggml.h` — new public op enums and constructors.
  - `ggml/src/ggml.c` + `ggml/src/ggml-cpu/ops.{cpp,h}` — CPU reference implementations.
  - `ggml/src/ggml-opencl/ggml-opencl.cpp` and new `kernels/fuse_kq_rope.cl` — OpenCL kernels.
  - `src/llama-kv-cache.{h,cpp}` — three-tier buffer layout, env-var parsing, ZSK accessors.
  - `src/llama-graph.{h,cpp}` — `build_attn_svd_combined` / `build_attn_svd_offload_combined` functions and Gemma 4 callers.
  - `src/llama-model.cpp` — `llm_build_gemma4_iswa` extended with tier-aware routing.
- **APIs**: No public API removed. New public ggml ops and KV-cache env vars are purely additive.
- **Dependencies**: No new external dependencies. Weight-conversion step adds a PyTorch SVD call (torch already required).
- **Backwards compatibility**: GGUF files produced without the `--svd-rank` flag continue to load and run as today. Models produced *with* the flag still run in baseline mode when `KV_SVD_SIZE` ≥ context length (degenerate case — all tokens stay in Tier-0).
- **Test surface**: New unit tests for (a) CPU SVD reconstruction round-trip, (b) fused-kernel parity against unfused reference, (c) end-to-end garbage-output regression: a fixed short prompt must produce coherent completion under Tier-0-only, Tier-0+1, and Tier-0+1+2 configurations.
