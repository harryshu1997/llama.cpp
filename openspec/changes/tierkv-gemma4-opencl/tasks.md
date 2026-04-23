## 1. Phase 0 — Establish baseline (no code changes yet)

- [ ] 1.1 User provides or downloads `google/gemma-4-E2B` HuggingFace checkpoint and shares its `config.json` so we can read `num_kv_shared_layers`, `sliding_window`, ISWA pattern, RoPE base/scaling.
- [ ] 1.2 Run stock `convert_hf_to_gguf.py` on the HF checkpoint to produce `gemma-4-E2B-fp16.gguf`. Save the file path.
- [ ] 1.3 Build current llama.cpp master with OpenCL backend on x86 Linux dev box; verify via `ggml-opencl: selecting device <name>` log.
- [ ] 1.4 Run `llama-cli -m gemma-4-E2B-fp16.gguf -p "The capital of France is" -n 32 --no-cnv` on CPU then with `-ngl 99` for OpenCL; capture both outputs as the baseline regression target. Confirm both produce coherent text.

## 2. Phase 1 — Offline SVD weight export (`svd-weight-export` capability)

- [ ] 2.1 In `gguf-py/gguf/constants.py`, add `MODEL_TENSOR.ATTN_UK`, `ATTN_UV`, `ATTN_VS` and metadata key `gemma4.svd.rank`, `gemma4.svd.shared_kv_source`.
- [ ] 2.2 In `gguf-py/gguf/tensor_mapping.py`, register the three new tensor names under their `MODEL_TENSOR` entries with template `blk.{bid}.attn_{uk,uv,vs}.weight`.
- [ ] 2.3 In `convert_hf_to_gguf.py` `Model` base class, add `--svd-rank INT` argparse flag (default 0).
- [ ] 2.4 In the `Gemma4Model` class (or whatever the existing Gemma 4 converter class is named), implement `_compute_joint_svd(W_k, W_v, rank)` returning `(U_k, U_v, VS)` tensors. Use `torch.linalg.svd(full_matrices=False)`.
- [ ] 2.5 Override `modify_tensors` (or equivalent hook) in the Gemma 4 converter to emit `attn_uk`, `attn_uv`, `attn_vs` per layer when `--svd-rank > 0`. Handle shared-KV layers by aliasing the source layer's tensors.
- [ ] 2.6 Write a Python unit test under `gguf-py/tests/test_gemma4_svd.py` that constructs synthetic K/V projection matrices, runs the converter helper, and verifies relative L2 reconstruction error < 0.05 at rank 512.
- [ ] 2.7 Run `convert_hf_to_gguf.py --svd-rank 512 ...` end-to-end on the real Gemma 4 E2B; confirm new GGUF loads via `gguf-py/scripts/gguf-dump.py` and shows the new tensors at expected shapes.

## 3. Phase 2 — Three-tier KV cache scaffolding (`three-tier-kv-cache` capability)

- [ ] 3.1 In `src/llama-kv-cache.h`, extend the per-layer cache struct with `ggml_tensor * zsk = nullptr` and disk-offload metadata fields. Add `kv_svd_size`, `kv_offload_size` member variables on the cache.
- [ ] 3.2 In `src/llama-kv-cache.cpp` constructor, parse `KV_SVD_SIZE` and `KV_OFFLOAD_SIZE` env vars, validate, align to `n_pad`, log effective values. Default to single-tier behavior when unset.
- [ ] 3.3 Allocate the per-layer `ZSK` tensor sized `[rank_l, kv_offload_size - kv_svd_size]` only when GGUF contains the SVD tensors and tier-1 capacity > 0.
- [ ] 3.4 Add `get_zsk(...)` and `cpy_zsk(...)` accessor methods on the cache. For shared-KV layers, internally redirect to the source layer.
- [ ] 3.5 Add a unit test under `tests/test-kv-cache-tiers.cpp` that allocates a cache with synthetic params, writes ZSK at known indices, reads back, verifies bit equality and that env-var alignment rounds correctly.

## 4. Phase 3 — Fused attention CPU reference (`fused-split-path-attention` capability, CPU only)

- [ ] 4.1 In `ggml/include/ggml.h`, add `GGML_OP_FUSE_KQ_ROPE` enum entry and the `ggml_fuse_kq_rope` constructor declaration matching the spec's parameter list.
- [ ] 4.2 In `ggml/src/ggml.c`, implement the constructor, set up output shape, register node sources.
- [ ] 4.3 In `ggml/src/ggml-cpu/ops.h` and `ops.cpp`, add the CPU compute function `ggml_compute_forward_fuse_kq_rope_f32` and dispatch entry. Implement in three sub-cases: (a) Tier-0-only (NULL ZSK), (b) Tier-1-only (NULL K_exact), (c) combined.
- [ ] 4.4 In `ggml/src/ggml-cpu/ggml-cpu.c`, route `GGML_OP_FUSE_KQ_ROPE` to the new compute function in the dispatch switch.
- [ ] 4.5 Write `tests/test-fuse-kq-rope.cpp` with three checks:
  - 4.5.1 Tier-0-only output equals plain `ggml_rope_ext + matmul + soft_max + matmul` reference within rel L2 1e-4.
  - 4.5.2 Tier-1-only output equals `matmul(ZSK, W_uk) + rms_norm + rope + matmul + soft_max + matmul` reference within rel L2 1e-4.
  - 4.5.3 Combined output matches sum of the two paths after softmax-merge within rel L2 1e-4.
- [ ] 4.6 Verify `tests/test-fuse-kq-rope` passes locally on CPU.

## 5. Phase 4 — Wire fused op into Gemma 4 graph for Tier-0 + Tier-1 (`gemma4-tierkv-graph` capability, partial)

- [ ] 5.1 In `src/llama-graph.h`, declare `build_attn_svd_combined(...)` taking the inputs spelled out in the design document.
- [ ] 5.2 In `src/llama-graph.cpp`, implement `build_attn_svd_combined(...)`. It calls the new `ggml_fuse_kq_rope` op and wires its output through the existing post-attention projection path.
- [ ] 5.3 In `src/llama-model.cpp` `llm_build_gemma4_iswa`, after computing per-layer Q/K/V, branch on the tier policy:
  - If GGUF lacks SVD tensors → existing path (Path A) + emit one-time warning if env vars set.
  - Else compute `actual_tokens` and per-layer `is_global` from Gemma 4 hparams.
  - Path A if `actual_tokens ≤ KV_SVD_SIZE` or (SWA layer and `W ≤ KV_SVD_SIZE`).
  - Path B otherwise (call `build_attn_svd_combined`).
- [ ] 5.4 Implement shared-KV layer dispatch: when `gemma4.svd.shared_kv_source[l] != l`, look up `s = source[l]` and reuse `s`'s `ZSK/W_uk/W_uv` tensors; share the tier path decision.
- [ ] 5.5 Smoke test with prompt of length < `KV_SVD_SIZE` — must produce the same output as Phase 0 baseline (Path A unchanged).
- [ ] 5.6 Smoke test with prompt of length between `KV_SVD_SIZE` and `KV_OFFLOAD_SIZE` — must produce **coherent** output (the success criterion). If garbage, debug per design D3 (norm-before-rope), D4 (column-major shapes), and Phase 4.5.2's reference.

## 6. Phase 5 — Disk offload ops + Path C (`kv-disk-offload-ops` and `gemma4-tierkv-graph` final)

- [ ] 6.1 In `ggml/include/ggml.h`, add the five new disk ops and their constructors per the spec.
- [ ] 6.2 In `ggml/src/ggml.c` and `ggml/src/ggml-cpu/ops.{cpp,h}`, implement the CPU compute functions. Use a per-layer `std::ofstream / std::ifstream` keyed by `(session_dir, layer_id)`. Prefetch uses a `std::thread` pool.
- [ ] 6.3 Implement session-dir lifecycle: create on first offload, env-var override `LLAMA_KV_OFFLOAD_DIR`, cleanup on `KV_CLEAN` or `llama_free`.
- [ ] 6.4 Add `tests/test-kv-offload.cpp`: `KV_OFFLOAD → KV_LOAD` round-trip, `KV_PREFETCH_START → compute → KV_PREFETCH_WAIT` overlap, `KV_CLEAN` removes files.
- [ ] 6.5 In `src/llama-graph.cpp`, implement `build_attn_svd_offload_combined(...)` that prefetches next layer's offloaded block while computing current layer's attention.
- [ ] 6.6 In `llm_build_gemma4_iswa`, add Path C selection and call.
- [ ] 6.7 Smoke test with prompt > `KV_OFFLOAD_SIZE` — output coherent, log shows offload/load activity.

## 7. Phase 6 — OpenCL kernel (`fused-split-path-attention` OpenCL part)

- [ ] 7.1 Create `ggml/src/ggml-opencl/kernels/fuse_kq_rope.cl`. Start with a single un-tiled kernel `kernel_fuse_kq_rope_reconstruct` that mirrors the CPU reference math step-for-step. Pass `W_uk, ZSK, w_q_norm, w_k_norm, rope_freqs, pos, mask, scale` as buffers.
- [ ] 7.2 In `ggml/src/ggml-opencl/ggml-opencl.cpp`, register the kernel: load source string, build at backend init, add a case in the op-dispatch switch for `GGML_OP_FUSE_KQ_ROPE` that sets buffer args and enqueues NDRange.
- [ ] 7.3 Add `tests/test-fuse-kq-rope-opencl.cpp` (or extend the existing test) to run the same inputs through CPU and OpenCL backends and assert rel L2 < 5e-3 across the dimension matrix in the spec scenario.
- [ ] 7.4 End-to-end test: re-run the Phase 5.6 prompts with `-ngl 99` (OpenCL backend) — output must remain coherent.

## 8. Phase 7 — Acceptance and cleanup

- [ ] 8.1 Run all new unit tests on x86 Linux (CPU + OpenCL) — record pass.
- [ ] 8.2 Run end-to-end coherent-output regression on three configurations: (Tier-0 only), (Tier-0+1), (Tier-0+1+2). Capture `llama-cli` outputs and store under `research_dev/tierKV/regression/` for future comparison.
- [ ] 8.3 Document the new env vars and `--svd-rank` flag in `docs/build.md` or a new `docs/tierkv.md`. Reference the original ICS 2026 paper.
- [ ] 8.4 Run `openspec validate tierkv-gemma4-opencl --strict` and resolve any issues.
- [ ] 8.5 Inform user that Android/Adreno cross-compile, INT8 KV, PMCO solver, and auto-tuner are deliberately out of scope and recommend follow-up changes.
