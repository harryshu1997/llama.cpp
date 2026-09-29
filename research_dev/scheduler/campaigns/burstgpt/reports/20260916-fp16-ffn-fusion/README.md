# FP16 HMX gate/up/GLU fusion

Status: implemented and physically verified on the OP15. Both matched bounded
document runs passed. Real FFN latency improved; overall prefill improved only
1.25%, and this pair does not demonstrate an energy saving from fusion.
No trace runs.

Scope: reuse existing HMX tile conversion/matmul and HVX activation math for
FP16 weights, FP32 activations and two-dimensional prefill. Preserve the
small-row decode path, down projection, shard files and all scheduler contracts.
The full FFN already uses one RPC; this optimization cannot remove USB trips.

Upstream review: [PR 28202](https://github.com/ggml-org/llama.cpp/pull/28202),
merged as `960dffab0583556351d829308a0d8f533624dc0a`. Its HMX projection fusion
still writes separate outputs. Review its implementation without updating this
dirty tree. Store the patch and before-images here; preserve all prior results.

Baseline references: pre-affinity prefill 46.435 s versus desktop 22.710 s;
the newer process-affinity experiment is 36.008 s versus 22.802 s. Do not
substitute one for the other. The actual Gemma worker uses 1280-column blocks.

Validation order: deployed per-op profile, existing backend fusion tests,
disabled/enabled real-weight worker comparison, then the bounded document
request only if the fused operation is beneficial. TCP/ADB probe latency is
diagnostic and is not FunctionFS latency. Profiling overhead is reported
separately from unprofiled timings. Phone power remains assumed.

## Implementation

The deployed profile confirmed HMX tiled FP16 projections for 137/512 rows and
HVX for 1/4 rows. Worker blocks are 1280 columns, even when the requested FFN
width is 15360. The new matcher recognizes only two consecutive ordinary 2D
FP16-weight/F32-input projections and their GELU or SiLU GLU, sharing exactly
the same input. It respects operand order, default precision, alignment,
contiguity and single-consumer/output constraints. Rows <= 4 and unsupported
patterns use the original graph.

The DSP operation reuses existing input/weight tile conversion, HMX job queue,
FP16-to-F32 conversion, and HVX activation math. One raw weight tile, two packed
weight tiles and three projection tiles enable pipelining. The next projection
runs on HMX while HVX finishes the prior gate/up pair. Only the GLU result is
written to DDR. Down projection is unchanged. There is no second resident or
packed copy of the weights, and no change to an RPC or scheduler contract.

The shared host/DSP scratch-layout calculation includes every buffer and
alignment gap. The DSP recomputes and verifies the descriptor and available
VTCM before execution. Profiled scratch is 7,534,592 bytes for Gemma's 512-row
fused block, versus 8,372,480 for its individual unfused projection. This is
operation scratch, not whole-worker peak RSS: the down projection still needs
its scratch, and the original GGML graph arena allocations remain. Eliminating
intermediate DDR traffic does not imply those arena reservations disappeared.

An initial correct implementation was slower because of scalar output copies
and insufficient overlap. Its failure is retained in `worker-enabled-v1` and
`worker-trace-v1`. HVX output copying and projection/GLU pipelining corrected it.
Initial diagnostic profiles used eight HVX threads; all final matched results
use the canonical four-thread setting. Do not compare the two configurations.

## Focused validation

- Existing `test-backend-ops`: 28/28 on physical HTP0 with fusion enabled and
  28/28 disabled. Both Gemma GELU (K=3840) and Qwen SiLU (K=5120), rows
  1/4/137/512, widths 512/1280, reordered GLU inputs, extra consumer,
  explicit precision, strided weights, bias and batched fallbacks.
- Ten eligible graph cases log `hmx-ffn-glu`; fallback cases remain unfused.
- A separate final real-weight diagnostic logs all 12 fused FP16 gate/up/GLU
  operations in the 15360-column FFN. Its profiled latency is not mixed into
  unprofiled measurements (`physical-v1/worker-profiled-v4/`).
- Existing CPU-reference NMSE tolerance remains 0.005. Scratch tests check
  alignment, buffer extents, fitting/oversized layouts and invalid dimensions.
- Real Gemma shard: 48 calls per mode, 16 row/fraction combinations repeated
  three times. Every F16 wire output is bit-identical; maximum absolute
  difference is zero. No claim of exact internal F32 equality is needed.
- Android worker, backend and v81 DSP library built and deployed together.
  `git diff --check` passes. No unrelated scheduler suite or replay was run.

Unprofiled whole-FFN worker compute medians at full width:

| Rows | Fusion off | Fusion on | Reduction |
| --- | ---: | ---: | ---: |
| 1 | 6.183 ms | 6.262 ms | unchanged kernel; run variation |
| 4 | 8.226 ms | 8.209 ms | unchanged kernel; run variation |
| 137 | 19.672 ms | 17.973 ms | 8.6% |
| 512 | 32.693 ms | 29.470 ms | 9.9% |

Across 25/50/75/100% widths, prefill-sized compute reductions range from 7.4%
to 9.9%. Full results, raw output checks, VTCM profile lines and input/output
hashes are in [WORKER_COMPARISON.json](WORKER_COMPARISON.json). TCP/ADB RPC
latency is preserved there but has large forwarding jitter; it is not used as
production transport evidence.

## Bounded document comparison

Both arms use the new identical binary bundle and the existing gate. Explicit
session wrappers set only `GGML_HEXAGON_OPFUSION=0` or `1` before invoking the
unchanged session launcher. Default CUDA graphs, c0 phone process affinity,
four HVX threads, 23 GPU layers, context 8192, batch 2048, ubatch 512, prompt
5261 tokens, output 64, seed 42, temperature 0, and full-width CPU FFNs on
layers 0-23 are fixed. The original transport qualification receipts are
rebound through the existing strict identity mechanism; transport code,
allocator, router, host binary and shard hashes have not changed.

The gate runs its desktop control and one relocated request per switch mode.
Compare the relocated request boundary across modes, not the gate's desktop
request energy directly against relocated request energy: the latter includes
desktop launch while the former reports that load separately. Phone preload
energy is separate, and no overlapping interval is subtracted to manufacture
a steady-state number.

The first disabled preflight stopped before inference because the newly
deployed wrapper lacked executable permission. Corrected permission only;
the failed preflight is preserved. The run's own full preflight then passed.

| Metric | Fusion off | Fusion on | Change |
| --- | ---: | ---: | ---: |
| 512-row worker compute, actual request mean | 32.045 ms | 29.663 ms | 7.4% lower |
| 512-row complete FunctionFS RPC, mean | 89.426 ms | 87.063 ms | 2.6% lower |
| 137-row worker compute, actual request mean | 20.107 ms | 18.923 ms | 5.9% lower |
| Prefill, 5261 input tokens | 36.156 s | 35.704 s | 1.25% lower |
| Decode, 64 output tokens | 25.501 s | 25.768 s | 1.05% higher |
| Prefill plus decode | 61.656 s | 61.472 s | 0.30% lower |
| Request plus desktop launch, host energy | 3.532 kJ | 3.535 kJ | 0.09% higher |
| Phone preload time | 43.725 s | 42.130 s | reported separately |
| Host energy during phone preload | 614.184 J | 614.114 J | reported separately |

The two fusion-mode requests produced identical 64-token sequences, including
the correct archive key ORCHID-731. The unchanged semantic-sanity gate passed.
CPU versus phone output still diverges at token 40, as recorded by the gate;
fusion does not introduce that divergence. Each request has 1800 proven calls,
600 per session, with exactly one load per shard and generations 1/1/1. Exact
artifact, geometry, operator-plan, generation, parent, binary and request
comparisons passed. Both terminal receipts have status 0 and zero reset
recoveries; USB restoration and same-boot postflight passed. No fallback.

The result is a useful backend improvement, not a fix for the prefill gap.
The matched desktop controls took 22.770/23.127 s for prefill and
29.563/29.524 s for decode. The fused phone run is still substantially slower
in prefill. At 512 rows, 87.06 ms of complete RPC contains only 29.66 ms of
worker compute. Router input/output checking alone averages about 16.0 ms;
transport, staging and other host/phone work remain. Some stage intervals
overlap, so their durations must not be added as if independent. Reducing
checksumming/copying/transport overhead is the next measured target, not
another fraction or scheduling change. Keep all integrity checks.

This is one matched pair, not a statistical estimate. Decode uses the same
unfused kernel; its 1% movement illustrates run variation. Measured host energy
is effectively unchanged. No claim of new fleet-energy savings is made.
[DOCUMENT_COMPARISON.json](DOCUMENT_COMPARISON.json) contains compatibility
checks, output comparison, all timings, preparation costs and terminal hashes.

## Reproduction and preserved evidence

Remote root: `/mnt/storage/s42-fp16-ffn-fusion-20260916-v1-yNEEYR/`.
Phone bundle: `/data/local/tmp/s42-fp16-fusion-20260916-v4/`.
Local mirror: `physical-v1/`. Prior physical results and deployment paths are
untouched. Build iterations and failed experiments are retained.

`SOURCE_MANIFEST.json` in the physical root records the paired binaries, exact
experimental wrappers and prior frozen scheduler source. The local
`BUILD_SOURCE_MANIFEST.json` inventories all 1421 native build inputs and the
toolchain image. [CHANGES.json](CHANGES.json) and
[IMPLEMENTATION.patch](IMPLEMENTATION.patch) isolate this task from the dirty
tree, using preserved before-images rather than the old Git HEAD.

Production changes are confined to `ggml/src/ggml-hexagon/`: matcher and profile
formatting; `htp/ffn-fused-ops.c/.h`; small matmul/activation helper exports;
internal opcode, dispatcher and CMake registration. Tests extend
`tests/test-backend-ops.cpp`. No worker protocol, shard format, fraction,
session-generation, lease or scheduler policy change.

On the rig, `profile_worker.py`, `run_backend_tests.py` and each document arm's
`execute_attempt.py` use the existing exclusive rig lock, idle preflight and
same-boot postflight. `collect_attempt.py` preserves router/worker logs and
joins native call timing to terminal proof bounds. Reproduce worker analysis:

```sh
python3 analyze_worker.py physical-v1 --output WORKER_COMPARISON-new.json
python3 analyze_document.py physical-v1 --output DOCUMENT_COMPARISON-new.json
```

Key SHA-256 values (full inventories in the manifests):

- ARM64 worker: `4375835a16cb2e0b5b1941a486c72ac5a0e33b9882c02c615df8db2b3426a3d4`
- Android Hexagon backend: `e6d5523a7416d5a12923e8709e621ced37670aeddaf138d6b100e050a5a1867e`
- v81 DSP library: `e6b629d205d8b8ffb6e1b9f83ac54cb4a28e9c234d1116120744ac8b3c085976`
- Fusion-off gate: `a9fb38620f369851c33fb12aa8902e26adf689c1824a935629223191a80b0620`
- Fusion-on gate: `8c5e29d3088eda65135df0d0d4e4e394b94b90a9f05a06e6df516ae3b6846e64`

The Android/Hexagon toolchain build retains the existing LTO warning about
`core_dot_chunk_fp16` activation/weight pairing. It did not prevent build or
the focused numerical gates. Builds and their earlier failed iterations are
preserved in `BUILD-v*.log`.

No kernel change, global affinity/governor change, long trace, commit or push.
