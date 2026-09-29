# Pixel FFN: packed weights, batch kernels, CPU/GPU ratios and block coalescing

Status: COMPLETE, 2026-09-23. All four proposed directions were implemented
and physically tested. Selected single-row and batched configurations pass
accuracy and matched latency checks; rejected variants remain documented.
Production defaults are unchanged.

## Confirmed single-row result

The best confirmed single-row candidate uses the original Q4_K/Q6_K weights on
six CPU threads pinned to cores 2-7, with a second packed dot to correct activation
quantization error. These are mean times for one layer's FFN, averaged equally
over layers 18-23. Half width is 8,704 channels; full width is 17,408 channels.

| FFN width | Matched old CPU/GPU controls | Corrected packed CPU | Latency reduction | Candidate p99 | Control p99 |
|---|---:|---:|---:|---:|---:|
| Half | 7.442 ms | 5.485 ms | 26.30% | 6.435 ms | 11.605 ms |
| Full | 13.695 ms | 10.792 ms | 21.20% | 12.579 ms | 22.174 ms |

PASS: all 2,160 confirmation calls meet the existing relative L2 limit of 0.01.
The candidate's maximum relative L2 is 0.000519396 (0.05194%); its 240 outputs
are byte-identical between repeats. Full-width repeat means are 10.363 and
11.221 ms; half-width means are 5.269 and 5.702 ms. Actual CPU affinity was
checked. Against the previously archived best 13.058/7.077 ms, the full/half
reductions are 17.35%/22.49%; the table uses controls from the same confirmation.

Evidence: [candidate and hashes](PIXEL_PACKED_CPU_CANDIDATE.json),
[confirmation](physical/pixel10pro-ffn-residual-confirm-1/run1/SUITE_RESULT.json).

## What changed and what failed

| Direction | Implementation and test | Measured outcome |
|---|---|---|
| Original packed weights | Per-tensor Q4_K/Q6_K storage; native packed CPU and Vulkan dot kernels | Native CPU full 5.844/5.900 ms fails accuracy (maximum relative L2 1.8877%). Packed GPU passes accuracy but takes 19.926/20.607 ms. Corrected packed CPU is the single-row winner above. |
| Multi-row GPU kernels | F16 weight reuse across 2/4/8 independent input vectors; register tiles with subgroup 32/64/128; an additional shared-memory tile | 14,496-call sweep passes accuracy and dispatch checks. Smaller subgroups regress. Final steady-batch confirmation is reported below. |
| Joint CPU count and channel ratio | CPU 1/2/3/4/6 threads, 15 ratio points, with CPU and GPU host affinity checked | 5,040 calls pass. The sweep does not establish an improvement over the existing four-thread CPU 39.706% / GPU 60.294% single-row split. Batched ratios are tested separately. |
| Coalescing full-width blocks | Join each backend's two blocks, reducing full FFNs from 12 to 6 matrix projections | Accuracy passes. Full 12.944 vs 13.064 ms over all controls, a 0.92% gain; half 7.376 vs 7.047 ms, 4.68% slower. Resident weights increase 50%. Not selected. |

Coalescing's immediately bracketing controls give approximately the same
conclusion (0.90% full gain and 4.53% half regression). The all-control figures
above preserve the earlier milestone's comparison. Exact per-arm matched
comparisons are in [the analysis](PIXEL_FFN_INTERIM_COMPARISONS.json).

Native CPU packed dots quantize activations to Q8_K. Keeping the original
weights alone therefore does not preserve the accuracy of the F16 reference.
The correction computes a quantization residual and adds a second native dot:

`W * Q(x) + W * Q(x - Q(x))`, where `Q` includes Q8_K round-trip conversion.

This is still approximate. It reduces the measured maximum FFN relative L2
from 0.018877 to 0.0005194 without expanding the weights to F16. The packed
artifact contains the original tensor bytes, not a new weight quantization.
Its 18 tensors occupy about 0.948 GB rather than 3.209 GB in F16. All copied
payloads were hashed; 32 complete rows per tensor were also checked against
the F16 proxy's rounding. That proxy check was sampled, not exhaustive.

Eight CPU threads and larger paired CPU blocks were also tested. They fail
the speed/stability goal: unbound CPU8 full means reached 69.518 ms, and paired
CPU8 reached 247-272 ms with severe tails. Those samples remain in the results.
The cause of the extreme tails was not established. Six pinned threads with
the original 4,352-column blocks are selected instead.

The shared-memory shader uses 128 threads, eight output rows, 16 reduction
lanes per output row, a 256-element K tile, and 12 KiB of shared memory for
eight input vectors. Its initial smoke accuracy and dispatch checks pass
without a speed win. At the later CPU6/GPU23.529% split, its one full eight-row
arm reaches 39.576 ms, an exploratory improvement that has not been repeated.
GPU-only execution remains slow. Register and shared-memory variants are
retained for review; neither is enabled in the production backend.

## Steady-batch confirmation

Numerical PASS: all 13,536 calls, maximum per-row relative L2 0.000562194.
The initial batch sweep alternated sizes 1/2/4/8 and produced single-row
controls around 29-32 ms. Those times must not replace the steady single-row
baseline. This confirmation groups repetitions by batch size and discards
ten warmup repetitions for each size before measuring.

The following are full-width mean batch times in ms. Both repeated arms are
shown, including the unstable unpinned CPU endpoint. These are not per-row
times. All listed arms pass the numerical threshold.

| Implementation | 1 row | 2 rows | 4 rows | 8 rows |
|---|---:|---:|---:|---:|
| Old F16 mixed controls, range of arm means | 13.588-16.416 | 26.932-28.351 | 46.477-47.668 | 77.194-89.842 |
| F16 CPU6 unpinned, first | 18.382 | 18.064 | 19.816 | 29.941 |
| F16 CPU6 unpinned, second | 121.389 | 301.126 | 30.194 | 49.497 |
| Corrected packed CPU6 pinned, first | 11.025 | 18.802 | 40.039 | 95.930 |
| Corrected packed CPU6 pinned, second | 13.164 | 23.572 | 47.919 | 113.393 |
| F16 CPU6 76.471% / GPU register tile2, first | 19.625 | 26.865 | 28.955 | 49.549 |
| F16 CPU6 76.471% / GPU register tile2, second | 19.529 | 27.617 | 28.369 | 50.742 |
| F16 CPU6 82.353% / GPU register tile2, one arm | 18.480 | 26.166 | 25.431 | 42.308 |
| F16 CPU6 76.471% / GPU shared-memory tile8, one arm | 15.653 | 22.787 | 36.711 | 39.576 |
| Shared-memory GPU-only, one arm | 20.794 | 51.644 | 72.975 | 114.975 |

The repeated register-tiled mixed configuration has full-width means
28.662/50.146 ms at batches 4/8, versus matched controls 47.044/82.182 ms:
39.07%/38.98% lower latency. Its half-width means are 16.478/30.183 ms,
37.48%/42.41% below matched controls. Both repeats improve full-width mean
and p99 at these batch sizes. The two 960-call outputs match byte-for-byte.

Corrected packed CPU is useful at batches 1/2 (full 12.094/21.187 ms, matched
gains 20.24%/24.20%). It loses at batch 8: full 104.662 ms, 27.16% slower than
matched controls. Its batch4 full result is not a repeatable win. The two
960-call outputs match byte-for-byte, so the latency regressions do not imply
an accuracy failure. This larger suite's batch1 latency is higher than the
dedicated single-row confirmation, but both still show about a 20% matched
full-width gain. These are separate measurement contexts.

CPU-only F16 has the lowest first-pass batch4/8 means, but the second unpinned
run is unstable, including very large batch1/2 stalls. That first pass alone
does not establish a stable winner. A focused reversed confirmation pinned
the F16 CPU6 endpoint to cores 2-7 and repeated the mixed candidate, below.

Dispatch coverage and partition/timing consistency PASS. Strict positive
overlap on every mixed call FAIL: 8,639/8,640 calls overlap. Arm
`04-reg6-c6656`, measured request385 (layer18, batch2, half width), finished
its CPU interval before the GPU interval began. The record is retained.
All batch4/8 calls of the repeated mixed candidate overlap.

Evidence: [batch audit](physical/pixel10pro-ffn-batch-confirm-1/run1/SUITE_RESULT.json),
[per-arm comparisons](PIXEL_FFN_BATCH_CONFIRM_COMPARISONS.json),
[repeated aggregates](PIXEL_FFN_BATCH_CONFIRM_AGGREGATES.json).

## Pinned CPU endpoint follow-up

Numerical and affinity PASS: 6,816 calls, maximum relative L2 0.000417678.
All 4,800 mixed calls overlap. Both CPU candidate repeats produce identical
output bytes for all 960 corresponding calls. The CPU-only endpoint beats the
tested register-tiled mixed path at batches 2/4/8 in both reversed comparisons.

Full-width times below are mean ms per batch. The mixed candidate uses six
pinned CPU threads, CPU76.471% / GPU23.529%, register tile2 and subgroup128.
The CPU-only candidate also uses six pinned threads, with F16 weights.

| Rows | Matched old controls | Tuned mixed CPU/GPU | Pinned F16 CPU-only | CPU reduction vs old | CPU ms/row | CPU p99 |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 28.771 | 22.927 | 19.829 | 31.08% | 9.915 | 22.876 |
| 4 | 46.546 | 30.760 | 21.697 | 53.39% | 5.424 | 26.564 |
| 8 | 81.873 | 48.166 | 35.423 | 56.73% | 4.428 | 44.989 |

Against the tuned mixed path, CPU-only improves these means by
13.51%/29.46%/26.46%. For half-width FFNs, CPU-only batch2/4/8 means are
9.907/10.873/18.106 ms, versus old controls 16.237/26.304/49.562 ms;
reductions are 38.99%/58.67%/63.47%. Both candidate repeats improve mean
and empirical p99 for both widths at batches 2/4/8.

The extreme unpinned stalls do not recur in this confirmation, but absolute
latency still varies: full batch8 means are 27.809 and 43.037 ms. Both beat
their nearby mixed arms (48.733 and 47.599 ms). Pinning is useful in this
measurement; the cause of all earlier stalls and long-run thermal stability
are not established. These results support the CPU-only batched candidate,
not a promise of a fixed 35.423 ms latency.

F16 CPU-only is slower for one row (20.235 ms aggregate), so the corrected
packed CPU candidate remains the single-row selection. Pinned F16 CPU versus
corrected packed CPU at two rows was tested in different suites; their close
ranking has not had a direct matched comparison. Neither the one-arm faster
shared-memory mixed tile nor the one-arm CPU82.353% ratio is promoted over
the repeated CPU endpoint without further evidence.

Evidence: [batch CPU configuration and hashes](PIXEL_FFN_BATCH_CPU_CANDIDATE.json),
[follow-up audit](physical/pixel10pro-ffn-pinned-endpoint-confirm-1/run1/SUITE_RESULT.json),
[per-arm comparisons](PIXEL_FFN_PINNED_ENDPOINT_COMPARISONS.json),
[repeated aggregates](PIXEL_FFN_PINNED_ENDPOINT_AGGREGATES.json).

## Test inventory

Ten completed suites contain 50,976 calls. The original packed CPU/mixed
variants fail the accuracy threshold on 920 calls; all 960 calls belonging
to those four rejected arms remain recorded. Selected configurations pass
every numerical check. Initial startup/header failures below are separate
and excluded from timing claims.

| Completed suite | Calls | Numerical result |
|---|---:|---|
| [Joint CPU/thread/ratio sweep](physical/pixel10pro-cpu-gpu-joint-1/run1/RESULT.json) | 5,040 | PASS |
| [Coalescing smoke](physical/pixel10pro-ffn-coalesce-smoke-1/run1/SUITE_RESULT.json) | 144 | PASS |
| [Native packed/coalescing comparison](physical/pixel10pro-ffn-packed-coalesce-1/run1/SUITE_RESULT.json) | 3,120 | FAIL for four packed CPU/mixed arms |
| [Corrected batch harness smoke](physical/pixel10pro-ffn-batch-smoke-2/run1/SUITE_RESULT.json) | 672 | PASS |
| [Multi-row sweep](physical/pixel10pro-ffn-batch-sweep-1/run1/SUITE_RESULT.json) | 14,496 | PASS |
| [CPU residual-correction sweep](physical/pixel10pro-ffn-residual-2/run1/SUITE_RESULT.json) | 4,320 | PASS |
| [Shared-memory tile smoke](physical/pixel10pro-ffn-shared-smoke-1/run1/SUITE_RESULT.json) | 672 | PASS |
| [Packed CPU single-row confirmation](physical/pixel10pro-ffn-residual-confirm-1/run1/SUITE_RESULT.json) | 2,160 | PASS |
| [Steady-batch confirmation](physical/pixel10pro-ffn-batch-confirm-1/run1/SUITE_RESULT.json) | 13,536 | PASS |
| [Pinned endpoint confirmation](physical/pixel10pro-ffn-pinned-endpoint-confirm-1/run1/SUITE_RESULT.json) | 6,816 | PASS |

Numerical PASS does not imply every arm improved latency or every interval
overlapped; those outcomes are stated separately above. Worker v8 builds with
`-Werror`; final ten Python files pass pyflakes and parsing; all 27 shader
modules in each batch-library build validate for Vulkan1.2. Final cleanup
PASS: no Pixel FFN worker, free experiment lock, no ADB forwarding and the
same boot ID. No artifact was deleted or worker force-killed.

Evidence: [machine-readable consolidated results](PIXEL_FFN_OPTIMIZATIONS_RESULTS.json),
[cleanup](PIXEL_FFN_OPTIMIZATIONS_CLEANUP.json).

## Measurement and correctness contract

- Pixel 10 Pro `5A040DLCH004ES`; unchanged boot
  `79b37d59-32ea-455a-9a24-0754c8e3c30d`; phone-local TCP loopback requests.
- One layer per call, Qwen layers 18-23, embedding width 5,120, suffix widths
  8,704 and 17,408. Each main performance group has ten measured repetitions
  of six layers after ten warmup repetitions. Short smoke runs are used for
  correctness only. No timing outliers are removed.
- `worker_us` includes graph preparation, backend input/output operations,
  compute/synchronization and, for mixed execution, the merge. It excludes
  USB/network round trip, server work and model startup. It is not a pure
  kernel timestamp or a full six-layer latency.
- Batched vectors are distinct deterministic transformations of archived
  inputs. Every row is checked against an independent phone CPU single-row
  result. Row zero is also checked against archived desktop CPU outputs;
  the maximum cross-architecture relative L2 is about 0.0000988841.
- A call passes numerical qualification only when every row is finite and
  relative L2 is at most 0.01. Packet shape/order, payload hash, artifact
  identity, requested partition, actual CPU affinity, actual tiled GPU
  dispatch counts and mixed-backend timing intervals are checked separately.
- Matched comparisons pool the nearest preceding and following old-selection
  controls with identical layer, width, row count and warmup composition.
  A control shared by two candidate repeats participates in both comparisons.
  Empirical p99 values have only 60 samples per arm/width/batch; they are not
  long-run tail guarantees. The sweep is sequential on shared hardware with
  uncontrolled clocks; control drift and repeat differences remain visible.

## Preserved failures and fixes

The first packed CPU smoke failed at startup because the optional repacking
buffer could not initialize mixed Q4_K/Q6_K tensors. The private variant now
uses the standard CPU buffer for packed weights. The first batch smoke sent
an incorrect element count for multiple rows. Worker validation rejected it;
the idle finite worker was drained with valid requests, without signals or a
force-kill. That run is excluded from performance claims. Corrected packet
extents are checked before deployment. Fresh retries passed.

Worker variants v6 and v7 were not run: source review caught the missing
generic Q8_K decoder callback before deployment. Variant v8 uses the exported
Q8_K decoder. Failed and superseded artifacts remain available.

Evidence: [packed startup failure](physical/pixel10pro-ffn-packed-smoke-1/run1/FAILURE.json),
[batch harness failure](physical/pixel10pro-ffn-batch-smoke-1/run1/FAILURE.json),
[fresh batch retry](physical/pixel10pro-ffn-batch-smoke-2/run1/SUITE_RESULT.json).

## Scope and reproducibility

Private generated workers, Vulkan libraries, input manifests and raw outputs
are preserved under this report. Canonical worker/backend files and production
defaults were not changed by these experiments. No commit or push was made.
The phone lock serialized Pixel tests; there was no host model inference.
The packed artifact is a private benchmark artifact, not a scheduler-qualified
transport identity. Do not copy its identity into production manifests.
Configurations are measured in separate workers; automatic switching between
packed CPU, F16 CPU and mixed execution, and its residency cost, is not implemented.

The single-row candidate uses `S42_PIXEL_PACKED_WEIGHTS=1`,
`S42_PIXEL_CPU_QUANT_RESIDUAL=1`, `S42_PIXEL_CPU_THREADS=6`,
`S42_PIXEL_CPU_POOL=1`, `S42_PIXEL_CPU_MASK=fc`, CPU backend, and 4,352-column
blocks. It uses neither a secondary GPU backend nor paired blocks.
The CPU library includes the previously qualified Android affinity fix.

For reference, the dedicated full-width packed CPU result represents about
49.55 useful matrix GFLOP/s, counting the original gate/up/down operations.
This excludes the extra residual-correction dot work and is not a measured
CPU arithmetic peak. The full eight-row F16 CPU candidate reaches 120.77
useful matrix GFLOP/s averaged across the two repeats. The 70.44% smaller
packed weights likewise do not establish a physical DRAM bandwidth measurement.

Sources: [packed and coalescing transform](pixel_ffn_experiments.py),
[residual correction](pixel_quant_residual.py),
[register tile](pixel_dense_batch.glsl),
[shared-memory tile](pixel_dense_batch_shared.glsl),
[finite suite and audit](pixel_experiment_suite.py),
[matched comparisons](analyze_pixel_ffn_optimizations.py).
Builds and static checks: [local checks](PIXEL_FFN_OPTIMIZATIONS_LOCAL_CHECKS.json),
[worker v8](software/pixel10pro-cpu-gpu-v8/BUILD_PROVENANCE.json),
[GPU library v3](software/pixel10pro-dense-batch-v3/BUILD_PROVENANCE.json),
[packed artifact](PIXEL_PACKED_WEIGHT_MANIFEST.json).

NOT VERIFIED: full-model token identity, server overlap and end-to-end latency,
phone or host energy, scheduler integration, realistic multi-request activation
streams, sustained thermal performance, physical DRAM traffic or sustained
compute peak. Useful matrix FLOP/s and nominal weight bytes per second must
not be presented as measured physical hardware utilization.
