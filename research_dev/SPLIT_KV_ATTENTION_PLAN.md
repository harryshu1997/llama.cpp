# Split-KV attention for GPU-resident layers: handoff plan

Written 2026-09-18 for the agent taking this over. Status: `STEPS 1-4 IMPLEMENTED / BOUNDED VALIDATION PASSED`. Nothing here is committed;
follow the standing rules at the end before touching anything.

Implementation notes (2026-09-18): LSE/merge tests pass on CPU, A6000 and RTX 4060 Ti.
The unchanged CPU path reproduces its saved logits hash. Unsplit cache extremes are bit-exact;
split-cache boundary, reuse and accounting checks pass locally. The CPU tiny-model boundary
case has one near-tied argmax difference under identical token inputs; CUDA has none. Real
Qwen validation passed with identical 64-token output across the three KV modes. Scratch rows are per-ubatch-token, rather than one shared
trash row, to avoid concurrent scatter-write races, and are charged to both pools. The existing
CUDA MMA kernel is stream-K-only; flagged operations use the vector/tile kernels, so the
prefill performance gate must be measured rather than assumed. Context shifting, shared/SWA
caches, quantized KV and wavefront copying are not enabled for the initial split implementation.

Final evidence (single runs, 2026-09-18):

| Acceptance | Result |
| --- | --- |
| FA and merge | 20 cases on CPU, A6000 and RTX 4060 Ti; unflagged CPU hash unchanged |
| Native KV | Unsplit extremes, straddling writes, clear/reuse, two-stream state restore and launch rejection pass |
| Real request | 9,737 prompt + 64 output tokens; split 762.7 vs whole-host 819.5 ms/token, exact output |
| Prefill | Split 203.8 vs all-device 201.0 s, +1.38%, below 5% limit |
| GPU staging | One/four split layers reserve 20.50/20.76 MiB, not a new 12 MiB copy per layer |
| 18 GiB memory scope | 24,576-cell fill evicts 1.043 GiB of model pages without release, zero after release; no OOM |
| Deferred | Optional overlap, integrated phone-plus-split-KV serving and automated scheduler qualification |

The native GPU-layer count in this checkout includes the output layer: Qwen `-ngl 16`
places decoder layers 25-39 on GPU. Initial fixtures including layer 24 were corrected;
their failed measurements remain preserved. Full results, caveats and changed-file list:
`scheduler/campaigns/burstgpt/reports/20260918-split-kv-attention/README.md`.

## Goal

For a layer whose weights live on the GPU, keep KV cells `[0, C_dev)` in VRAM and the overflow cells
`[C_dev, C)` in host RAM (the room the decode-only FFN relocation frees, see
`scheduler/campaigns/burstgpt/reports/20260917-decode-relocation-kv-headroom/README.md`, section
"KV growth into the freed memory, directly"). Compute attention over each slice where the slice lives and
merge the two partial results. Today the per-layer host KV path (`--kv-cpu-layers`) is all-or-nothing:
when a layer's KV is on the host, the whole attention of that layer runs on the CPU
(`src/llama-graph.cpp:2503`, the `cpu_attention` branch of `build_attn_mha`). The split lets the GPU
keep the fast part and hands the CPU only the overflow.

Decision the user has already made: prefill numerics and placement do not change for cells that fit VRAM;
the host slice exists so that longer contexts fit under a fixed host budget after the FFN share is released.

## What exists (read these first)

| Piece | Where |
| --- | --- |
| Attention builder, `cpu_attention` branch, FA vs. non-FA paths | `src/llama-graph.cpp:2492-2625` |
| Per-layer KV placement (`kv_cpu_layers`) | `include/llama.h:370`, `src/llama-context.cpp:73-93`, `src/llama-model.cpp:2198`, `common/arg.cpp:2157` |
| Auto-FA device check per layer | `src/llama-context.cpp:525-556` |
| KV cache allocation loop (one K and one V per layer, one buffer per buft) | `src/llama-kv-cache.cpp:133-262` |
| KV reads (`get_k`/`get_v` views), `n_kv` padding to 256 | `src/llama-kv-cache.cpp:1547-1600` |
| KV writes via `ggml_set_rows` | `src/llama-kv-cache.cpp:1615-1703` |
| KQ mask fill | `src/llama-kv-cache.cpp:1836-2090` |
| Lazy page-granular KV zeroing, `touch_cells` test hook | `src/llama-kv-cache.cpp` (`llama_kv_cache_clear_buffer`, `touch_cells`), `include/llama.h:638` |
| FA op definition, op_params = `{scale, max_bias, logit_softcap}` + prec | `ggml/src/ggml.c:5360-5430` |
| CPU FA kernels: `S` (sum) and `M` (max) accumulators | `ggml/src/ggml-cpu/ops.cpp:8347` (one_chunk, `S`/`M` at 8440), tiled `8585`, dispatcher `8945` |
| CUDA FA launcher: `parallel_blocks`, stream-k, `dst_tmp_meta`, combine kernel | `ggml/src/ggml-cuda/fattn-common.cuh:913` (combine), `972` (launch), `1095-1175` (block planning) |
| Backend scheduler input copies (full-stream synchronize before host copies) | `ggml/src/ggml-backend.cpp:1541-1700` |
| FA op tests | `tests/test-backend-ops.cpp:6771` (`test_flash_attn_ext`) |
| Probe used for memory gates (`--kv-touch-tokens`, `--dormant-consume`, RSS/majflt JSON) | `examples/layersplit/ffn-remote-resident-probe.cpp` |
| Python tests on the tiny model | `research_dev/scheduler/tests/test_kv_lazy_backing.py` |

## Design (agreed)

1. **Cache layout.** Per GPU-resident layer, two K/V tensor pairs: device slice (VRAM buft) and host
   slice (CPU buft), sharing one logical cell index space. New per-layer parameter `kv_device_cells[il]`
   (`C_dev`); `C_dev == 0` reproduces today's `kv_cpu_layers`, `C_dev == C` is today's default. Keep
   `C_dev` a multiple of 256 (the `n_kv` padding, `src/llama-kv-cache.cpp:1552`).
2. **Writes.** Each slice tensor gets one extra trash row. `cpy_k`/`cpy_v` issue two `ggml_set_rows`,
   each with an index tensor that maps cells outside its slice to the trash row. A ubatch that straddles the
   boundary then needs no `find_slot` constraint.
3. **Reads.** `get_k`/`get_v` return two views: device `[0, min(n_kv, C_dev))`, host
   `[C_dev, n_kv)` (absent when `n_kv <= C_dev`, which changes graph shape the same way `n_kv` already
   does). Two mask inputs, filled by the same `set_input_kq_mask_impl` over the two column ranges.
4. **Partial attention with LSE.** Add an op flag to `GGML_OP_FLASH_ATTN_EXT` (a fourth op_param, keep
   the existing three floats and the prec slot) meaning "append the per-(head, token) log-sum-exp to the
   output row": dst `ne[0] = DV + 1` (pad to `DV + 4` if alignment demands), output stays normalized.
   - CPU: `one_chunk` already has `S` and `M` (`ops.cpp:8440`); write `M + log(S)`. Route flagged ops
     to `one_chunk`, not the tiled kernel, unless the tiled path is extended too.
   - CUDA: for flagged ops disable stream-k and force `parallel_blocks >= 2` so every kernel family
     already writes `(m, l)` meta; extend `flash_attn_combine_results` to also emit the LSE. Do not touch
     the vec/tile/mma/wmma kernels.
   - Other backends: `supports_op` returns false for the flag so the scheduler falls back to CPU.
5. **Merge** (composed from existing ops, no new kernel, overflow-safe):
   `O = O_dev * sigmoid(LSE_dev - LSE_host) + O_host * sigmoid(LSE_host - LSE_dev)`.
   Take the `[0, DV)` view of each partial; `ggml_sub`, `ggml_sigmoid`, broadcast `ggml_mul`, `ggml_add`.
6. **Phase-dependent placement of the host-slice attention.**
   - Decode (small `n_tokens`): run it on the CPU. Cost is host bandwidth: K+V is 4 KiB per cell per layer
     on Qwen3-14B; 24,576 host cells x 8 GPU layers is about 0.77 GiB per token, roughly 15 ms.
   - Prefill (`n_tokens` = ubatch): CPU compute over a long host slice is too slow. Stream the host slice
     to a one-layer VRAM scratch instead (about 96 MiB per layer per 512-token ubatch over PCIe). Use
     `ggml_backend_sched_set_tensor_backend` on the host-slice FA node to force GPU placement; the
     scheduler then copies the views. Measure both placements; pick per phase by a threshold on `n_tokens`.
7. **Concurrency is a separate step.** The scheduler serializes the two slices: before copying Q into the
   CPU split it calls `ggml_backend_synchronize` on the CUDA backend (`ggml-backend.cpp:1573` and the
   fallback near `1665`), which waits for every kernel already launched, including the device-slice FA.
   Step 1-3 below are still a win over whole-layer CPU attention. True overlap requires recording an event
   per split and waiting on that event instead of the whole stream. Do it last, measure separately.

## Steps and acceptance

### Step 1: flagged FA with LSE (ggml only)

- Implement the flag in `ggml.c` (constructor variant or setter), CPU kernel, CUDA combine path, and
  `supports_op` gates.
- Add `test-backend-ops` cases: flagged FA against unflagged FA on the `[0, DV)` view (identical), and the
  two-slice merge against single-slice FA for `n_kv` in {256, 4096, 24576}, `n_tokens` in {1, 8, 512},
  GQA head ratios of Qwen3-14B (40/8, head dim 128) and the tiny model. Tolerance: the existing FA
  test's NMSE; report max abs diff.
- Acceptance: all FA tests pass on CPU and CUDA (desktop RTX 4060 Ti, build with the CUDA paths listed
  under "Rig" below); no change to unflagged outputs (hash the existing test outputs before and after).

### Step 2: split-slice KV cache, decode placement

- `llama_context_params`: add `kv_device_cells` (per-layer int array, same shape convention as
  `kv_cpu_layers`); CLI `--kv-device-cells N0:C0,N1:C1,...` in `common/arg.cpp` next to `--kv-cpu-layers`.
  Plumb through `llama-cparams.h`, `llama-context.cpp`, and the server's runtime manifest (the scheduler
  pins the KV plan by `kv_plan_sha256`; extend `research_dev/scheduler/_internal/kv_placement.py` and the
  adapter contracts so the plan hash covers `C_dev`).
- KV cache: two tensors per layer for split layers, trash rows, two `set_rows`, two views, two masks.
  Keep `llama_kv_cache_clear_buffer` lazy zeroing and `touch_cells` working over both slices.
- Graph: in `build_attn_mha`, when the layer is split and FA is on, build device FA (flagged), host FA
  (flagged), merge. Non-FA path: not supported for split layers, assert with a clear message.
- Auto-FA device check (`llama-context.cpp:525`): the host-slice FA node is expected on the CPU, the
  device-slice node on the layer's device; teach the check both expectations.
- Tests (tiny model, `tests/test_kv_lazy_backing.py` style, through the probe):
  - `C_dev = 0` reproduces `--kv-cpu-layers` outputs bit-exactly.
  - `C_dev = C` reproduces the default bit-exactly.
  - `C_dev` strictly inside: decode argmax over 64 tokens matches the default within the FA tolerance
    used in Step 1; a prompt that straddles the boundary and one that ends exactly on it.
  - `touch_cells` across the boundary lands the expected bytes in each slice (anon RSS on host, none on host
    for the device slice).
- Acceptance: tests pass; Qwen3-14B on the desktop with 16 GPU layers, `C = 32768`, `C_dev = 8192` on
  the GPU layers, a 9,737-token prompt (the pair-v1 request) plus 64 generated tokens gives the same
  argmax sequence as the current all-host-KV configuration and a lower decode ms/token. Record both.

### Step 3: prefill placement through PCIe streaming

- Force GPU placement for the host-slice FA when `n_tokens >= threshold` (start at 32), copy through the
  scheduler. Verify VRAM scratch stays one layer's slice (watch `ggml_backend_sched` buffer sizes in the
  log).
- Acceptance: prefill time of the 9,737-token request within 5% of the all-VRAM configuration at the same
  `C`; decode unchanged from Step 2.

### Step 4: memory gate under a budget (the claim the user cares about)

- Re-run `kv_headroom_probe.sh`-style arms with the split cache: `MemoryMax` 18 GiB, release the FFN share,
  fill 24,576 cells; expect the host slice's anon RSS growth to land in the freed room with weights
  resident (as in `physical/kv-headroom-v1/`), and the device slice to stay at `C_dev`.
- Then one real long request that exceeds `C_dev` on the GPU layers, control = all-VRAM KV at the same
  total `C` if it fits, otherwise `--kv-cpu-layers` for those layers. Report prefill, decode, RSS, argmax
  equality, host energy (CPU package via the existing samplers; GPU board via NVML). Single runs are
  fine, label them as such.

### Step 5 (optional): scheduler overlap

- In `ggml_backend_sched_compute_splits`, record an event after each CUDA split and wait on that event
  when a later split needs one of its outputs, instead of `ggml_backend_synchronize`. Measure decode
  ms/token with and without. Do not start this before Steps 1-4 pass.

## Non-goals

- No change to phone-side code, transport, or the S42 scheduler routes beyond the KV-plan hash.
- No re-ordering of cells (the device slice is the low cell range, not a "recent window").
- No claim about maximum usable context beyond what Step 4 measures.

## Pitfalls already known

- Prefer the existing `n_kv` padding logic; a `C_dev` that is not a multiple of 256 breaks the mask width.
- `ggml_set_rows` cannot skip rows: use the trash row, not per-slice slot constraints.
- `ggml_soft_max_ext` / non-FA path exposes no LSE; do not try to merge through it.
- The scheduler's fallback copy path synchronizes the whole CUDA stream (Step 7 above); do not expect
  overlap from graph ordering alone.
- Tiny test model has no vocab; server-based tests abort. Use the probe binary.
- After any rebuild of `libllama`/server on the desktop, the trace bundle's transport identity
  (`/home/zhihao/s42-dormant-trace-20260917-v1-inputs/TRANSPORT_QUALIFICATION_IDENTITY.json`) is stale;
  re-materialize with `python3 -m research_dev.scheduler.adapters.materialize_transport_qualification`
  before any trace launch.

## Rig

- Desktop `zhihao@172.20.74.85` (zsh: quote `==`). cmake:
  `/mnt/storage/s21_deps/cmake-4.2.3-linux-x86_64/bin/cmake`. Build and link shells need
  `export LD_LIBRARY_PATH=/mnt/storage/s21_deps/cuda-13.2.1/lib:/mnt/storage/s21_deps/cuda-13.2.1/lib64`
  or the server link fails with undefined `cuda*` references.
- Current deploy root `/mnt/storage/s42-kv-decode-relocation-20260917-v1-eedc22` (`native-source/`,
  `cuda-build/`); make a new deploy dir for this work rather than editing that one.
- Memory budgets: `systemd-run --user --scope -p MemoryMax=... -p MemorySwapMax=0`; drop the model's
  page cache (`posix_fadvise DONTNEED`) before each arm.
- Do not touch other campaigns' processes; rig lock
  `/home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock` (flock, nonblocking).
- OP15 (serial `3C15AU002CL00000`, adb 5037 on the desktop) is not needed for Steps 1-4. Never kill an
  in-flight phone worker; a mid-transfer kill crashed the phone kernel on 2026-09-13.

## Standing rules

- `AGENTS.md`: do not commit or push without explicit human approval. No AI-written commit messages or
  PR text. Leave the tree uncommitted and report the diff.
- Keep `research_dev/talks.md` updated, newest first, timestamped, tables over prose.
- Report measurements as they came out; label single runs and unmatched bundles as such.
