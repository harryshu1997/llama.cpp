# Split-KV attention: implementation and bounded hardware validation

Status: Steps 1-4 implemented and their bounded validation passed. Optional CPU/GPU
overlap is not implemented. No scheduler trace or phone execution.

## Implementation

- `--kv-device-cells layer:count,...` fixes a GPU prefix and lazy-backed host overflow.
  Zero reproduces whole-layer host KV; the full context size reproduces default placement.
- Each partial attention returns normalized output and log-sum-exp. Existing GGML ops
  merge the partials. Attention sinks are counted only once; empty partials produce zero.
- Decode uses CPU attention for the overflow. At 32 or more query tokens, the host view
  is copied to the GPU by the existing backend scheduler. Execution remains serialized.
- Per-ubatch-token scratch rows avoid duplicate scatter writes. Both pools' scratch
  allocations enter the scheduler memory plan and exact launch identity.
- Clear/reuse, logical state serialization and independent streams are covered. A shared
  pre-existing restore bug was also repaired: clear once before restoring streams, not
  once per stream, which erased previously restored data.

Initial scope is F16 KV, dense non-shared/non-SWA attention. Context shifting, wavefront
copying, tensor-parallel split caches and quantized KV are rejected, not silently emulated.

The CUDA MMA implementation is stream-K-only. The new flagged op uses the existing
vector/tile kernels so it can export the combine statistics without rewriting the MMA
kernel. There is no CPU/GPU partial-attention overlap in this implementation.

## Software validation

- 20 partial-attention/merge cases on CPU, A6000 and RTX 4060 Ti, including 24,576 KV cells, 512
  queries, Qwen GQA geometry, sinks, softcap and both partials empty.
- 21 local KV/ledger/lazy-backing tests; five native cache tests also pass on CUDA.
- Unsplit extremes reproduce logits bit-exactly. Split logits meet NMSE <= 5e-4.
  One CPU synthetic near-tie changes argmax; the CUDA synthetic cases do not.
- Original unflagged CPU 64-step logits SHA-256 remains
  `ba257f0f2902f9a3278214fc8d16cdd407371ebb7fbf374c1a2f996258a8f89d`.
- Six CUDA backend LSE comparisons pass; 68 adapter, KV-plan, release-accounting and
  replay tests pass, including both unchanged replay goldens.
- The full historical scheduler harness has not been rerun.

## Real Qwen request

RTX 4060 Ti; Qwen3-14B dequantized F16 artifact, `--n-gpu-layers 16`, 32,768 cells,
9,737 prompt tokens plus 64 generated tokens, batch 512, ubatch 128, eight CPU threads.
This checkout includes the output layer in the GPU-layer count: decoder layers 25-39
are on GPU, and 0-24 are on CPU. No phone, no weight relocation, no memory cap in these runs.

The corrected fixture targets the actual GPU decoder layers, 25-39. The CPU layers keep
their existing host KV in all arms. "All device" below refers to KV of the GPU-resident
layers, not the KV of every model layer.

| KV of GPU-resident layers | Prefill | Decode ms/token | Request CPU+GPU energy |
| --- | ---: | ---: | ---: |
| All device (device-r1) | 201.028 s | 740.302 | 21.056 kJ |
| Split at 8,192 cells (split-r3) | 203.808 s | 762.659 | 21.591 kJ |
| All host (host-r3) | 243.249 s | 819.495 | 29.555 kJ |

All three arms have identical 64-token outputs. Against whole-host KV, split prefill
time is 16.21% lower, decode time/token is 6.94% lower and request CPU+GPU energy is
26.95% lower. Against all-device KV, split prefill takes 1.38% longer, decode 3.02%
longer and energy is 2.54% higher. The <=5% prefill gate passes. Split is a capacity
tradeoff, not an energy improvement over a feasible all-device cache.

These are single runs, not a statistically qualified speed or energy claim. Loading
is excluded from the request energy window and differed with page-cache state:
39.776 s (device), 31.278 s (split), 5.184 s (host). RAPL CPU-package and NVML GPU-board
energy are measured; neither whole-system wall power nor phone energy is included.

Remote bundle: `/mnt/storage/s42-split-kv-20260918-WwiIPY/`.
Commands and measurements are preserved under `physical/device-r1/`, `physical/split-r3/`
and `physical/host-r3/`. Both R3 arms record the loaded runtime-library hashes, which match.
The server hash is `c741f6dc03ed7a08d6d2ba47e32bf23f9bdf5e548a751df39c803341bbb70d05`.
The native source and runtime used by this comparison are frozen as `native-source-r1/`
and `runtime-r1/` before deploying the two-stream restore correction and extra probe
diagnostics. Those later changes do not execute in these single-request arms.
Each result says `scheduler_qualified=false`; request completion does not qualify an
automated scheduler route.

The earlier `split-r1` and `host-r1` fixture mistakenly included CPU-only layer 24.
They are preserved unchanged: split prefill 218.411 s, decode 778.861 ms/token, energy
24.346 kJ; host prefill 247.560 s, decode 819.619 ms/token, energy 30.311 kJ. That split
failed the prefill gate. Correcting the fixture, not a CUDA kernel rewrite, removed the
extraneous split attention on a CPU layer. Do not mix those arms into the corrected table.

## Bounded GPU staging

`staging_probe.py` reserves the native worst-case context graph for a four-layer tiny
model at 32,768 cells, prefix 8,192, ubatch 128. On both A6000 and RTX 4060 Ti, one split layer reserves
20.50 MiB of CUDA compute memory; four reserve 20.76 MiB. A single host K/V slice is
12 MiB. Extra layers therefore do not retain a full additional GPU copy of each slice.
This checks allocator reservation, not a served 32k request or asynchronous overlap.

The caller must still reserve the measured graph workspace in its phase peak, separately
from the KV bytes and per-slice scatter scratch reported by `plan_layer_kv`.

## Memory gate

`physical/headroom-r3/`: same split cache in both arms, an 18 GiB host-memory scope,
swap disabled, model-file cache advised away before each arm. After seven prompt tokens
and two decode steps, touch 24,576 logical KV cells. The released arm first drops the
existing FFN share for layers 0-17, without executing any phone work.

| Measurement | Retain FFN pages | Release FFN pages before KV fill |
| --- | ---: | ---: |
| FFN pages released | 0 | 9,625,706,496 B (8.965 GiB) |
| Release time | N/A | 259.8 ms |
| KV written, both tiers | 3.750 GiB | 3.750 GiB |
| Host / device share of written KV | 3.281 / 0.469 GiB | 3.281 / 0.469 GiB |
| Anonymous RSS before / after fill | 0.219 / 3.499 GiB | 0.219 / 3.499 GiB |
| Model-file RSS before / after fill | 15.382 / 14.340 GiB | 6.417 / 6.417 GiB |
| Model pages evicted during fill | 1.043 GiB | 0 |
| New major faults during fill | 6 | 0 |
| OOM / OOM kills | 0 / 0 | 0 / 0 |

The released space absorbed the host KV growth without evicting the retained weights.
This is an occupancy probe, not a served 24,576-token prompt. The KV fill is synthetic;
the shared argmax `[4180, 13, 2160]` was computed before the fill and release. No attention
is evaluated using the synthetic KV. A usable decode with those FFN pages absent still
requires the phone owner; this gate does not prove that integrated workload.

The device KV allocation was 487.50 MiB including scatter scratch, versus 1,920 MiB for
the same GPU layers' unsplit 32k cache. The host allocation was 4,647.50 MiB, lazily backed.
Qwen's graph reserved 313.50 MiB of CUDA compute memory and 21.01 MiB of host compute
memory. These are allocations/reservations, not all resident host pages.

Restoring the released share took 34.098 s under the cap. Although all 9,625,706,496 bytes
were repopulated, total model-file RSS ended at 14.404 GiB because the budget cannot retain
all weights plus the touched KV. Release credit must not fund the next prompt unless KV
is relinquished or restoration headroom is separately reserved. The raw dormant
`logits_identical=false` is not a failed comparison: `--dormant-no-decode` deliberately
did not run a second decode. Clear/reuse and state/logit equality are separate native tests.

The earlier `headroom-r2/` record is retained, with the same extraneous CPU-layer split
as the initial timing fixture. It is not the source of the corrected table.

Run scripts are `campaigns/burstgpt/split_kv_gate.py` and
`campaigns/burstgpt/split_kv_headroom_gate.sh` under the canonical scheduler directory.
The rig lock is mandatory. Only test-owned desktop processes are stopped.

## Scope and next integration

- Initial supported path: F16 KV, dense attention, serialized partials. Other cache
  types/architectures require separate implementation and validation.
- No automatic context-limit expansion: configured cells remain 32,768 and the model
  limit remains 40,960. VRAM, workspace and next-prompt restoration remain constraints.
- KV-plan hashes include split prefixes. Cache bytes and scratch enter the existing
  ledger, but automated scheduler selection and integrated phone-plus-split-KV serving
  have not been qualified by this work.
- Optional overlap is a separate optimization; the current comparison includes its absence.
- The full scheduler harness and repeated statistical performance trials were not run.
- Before a scheduler trace, re-materialize transport qualification for the rebuilt runtime.
  No full trace, commit or push was performed; existing unrelated edits were preserved.

Changed paths are listed in `CHANGES.json`. Raw artifacts and validation logs are kept
in `physical/` and `software/`; SHA-256 identities are in `ARTIFACT_SHA256.txt`.
