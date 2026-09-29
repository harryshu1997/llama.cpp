# Pixel NEON, GPU and device-specific weight layout tuning

Status: COMPLETE. Numerical/dispatch PASS, CPU latency improvement PASS.
Mixed mean latency improvement PASS; mixed p99 improvement over fused CPU FAIL.
These are private research candidates. Production defaults are unchanged.

## Result

The main gain comes from fusing the two CPU residual-correction matvec passes,
keeping the existing native NEON dot routines and original packed weight bytes.
The focused endpoint run measured three arms per configuration, 20 repetitions
per arm, discarding the first 10 as warmup. Each table cell pools 180 warm calls,
equally weighted over Qwen layers 18-23. No timing outliers were removed.

All times below are phone-local worker milliseconds for one input row and one
FFN layer, including gate, up, activation, down and partial-result summation.
Full width is 17408 channels; half width is 8704. These are not token, six-layer,
USB round-trip or server times.

| Configuration | Half mean | Full mean | Full observed p99 | Full maximum | Full mean reduction vs old CPU |
|---|---:|---:|---:|---:|---:|
| Previous corrected packed CPU | 5.189 | 10.179 | 11.949 | 12.113 | baseline |
| Fused corrected packed CPU, 6 threads | 4.135 | 8.122 | 8.873 | 9.378 | 20.21% |
| Fused CPU 94.118% + tiled GPU 5.882% | 4.121 | 7.928 | 9.712 | 20.993 | 22.11% |

The mixed path improves mean full-width latency by another 2.39% over fused CPU,
and half-width latency by only 0.35%. Its p99 is 9.46% higher than fused CPU and
its largest observed call is 20.993 ms. Select fused CPU for the private B1
candidate; retain the mixed path as an experimental mean-latency option.
The three CPU means are 8.121/8.129/8.116 ms; mixed means are
7.797/7.780/8.207 ms. All three mixed arms beat their bracketing old CPU means,
but the last mixed arm has p99 14.934 ms. These sample p99 values are descriptive,
not a production tail-latency guarantee.

Useful matrix throughput, counting one mathematical FFN as 6 * 5120 * 17408
operations, is 52.54 GFLOP/s old CPU, 65.84 fused CPU, and 67.45 mixed. This
excludes extra correction arithmetic and does not measure hardware peak usage.

Evidence: [endpoint comparison](PIXEL_DEVICE_LAYOUT_ENDPOINT_COMPARISON.json),
[aggregate measurements](PIXEL_DEVICE_LAYOUT_RESULTS.json),
[CPU candidate](PIXEL_FUSED_CPU_CANDIDATE.json),
[mixed candidate](PIXEL_DEVICE_MIXED_CANDIDATE.json).

## What changed

1. Added private F16 NEON kernels with 2/4/8 output rows, 1/2-way unrolling,
   row-major or K=8/32 interleaving, and optional prefetch. F32 FMA and FMLAL
   variants accumulate in F32. FMLAL converts intermediate activations to F16,
   consistent with the existing F16 CPU dot path. Stored F16 values are unchanged.
2. Added GPU layouts with 8 output rows interleaved over K=4 or K=16. The new
   shader uses 128 threads, 16 reduction lanes per output row, vec4 loads and
   shared-memory reduction. The previous shaders remain available.
3. Added device-specific storage: CPU retains the original Q4_K/Q6_K tensor
   bytes with activation residual correction; only GPU-owned tensors expand
   to F16 and optionally use the 8x4 layout. No new weight quantization is used.
4. Fused CPU activation quantization and residual creation. Both native packed
   dot calls now run consecutively for each weight row, instead of two complete
   matvec traversals. The formula remains W*Q(x) + W*Q(x-Q(x)); the correction
   is still applied to gate, up and down projections. This is the winning change.
5. Reused the existing concurrent worker branches and FP32 partial-output sum.
   Channels are disjoint; the input is shared. Ratios were swept, not fixed 50/50.

The new CPU fusion intends to improve weight-cache reuse and reduce graph
scheduling overhead. Physical DRAM traffic was not counted, so the speed gain
cannot be assigned precisely between cache behavior and graph overhead.

Sources: [CPU kernels and packing](pixel_f16_layout.h),
[worker transformation](pixel_layout_transform.py),
[GPU shader](pixel_dense_layout.glsl). Existing private builders and the finite
sweep/audit harness were extended; the canonical worker/backend were not changed
by this investigation.

## Tuning process and rejected alternatives

| Stage | Arms / calls | Measured outcome | Decision |
|---|---:|---|---|
| F32 NEON and lossless F16 layouts | 13 / 936 | CPU 19.314 vs best new 19.167 ms; GPU 20.768 vs 8x4 20.398, 8x16 22.125 ms | Accuracy PASS; small gains exploratory; reject wider GPU tile |
| FMLAL, unrolling, prefetch | 12 / 1152 | F16 CPU controls 18.251/18.533 ms; best 8x8 FMLAL 17.946, 4x8/unroll2 17.977 ms | Accuracy PASS; prefetch loses; repeat required |
| CPU packed / GPU expanded-F16 ratios | 18 / 3456 | Best mixed 82.353% CPU 10.830 ms vs packed CPU controls 10.298/10.227 ms | Accuracy/overlap PASS; combined speed FAIL |
| CPU residual-pass fusion | 9 / 864 | CPU6 9.772 ms vs controls 11.654/11.663; CPU2/4 23.254/12.235 ms | Accuracy PASS, CPU exact, fusion promising |
| Reversed broad confirmation | 19 / 4560 | Fused CPU 8.095/10.217 ms; mixed 94.118% CPU 7.844/8.098 ms | Accuracy PASS; CPU variation requires endpoint check |
| Distinct-row shape check | 6 / 816 | B1/2/4 max row relative L2 0.000536187; fused CPU 144/144 exact to previous CPU | Correctness PASS; not a batch-throughput measurement |
| Focused CPU/mixed endpoint comparison | 9 / 2160 | CPU 8.122 ms; mixed 7.928 ms vs old CPU 10.179 ms | Mean PASS; mixed tail vs fused CPU FAIL |

Total: 86 arms, 13944 completed phone calls. All 5304 mixed calls show overlapping
backend intervals. All 1632 fused CPU payloads compared with the previous
corrected packed CPU are byte-identical, including the multi-row shape check.
Maximum row relative L2 across these suites is 0.000536187 (0.05362%), below the
existing 0.01 acceptance threshold. Inputs are synthetic qualification vectors
with actual model weights; this is not full-model token-equivalence proof.

The broad confirmation retained an anomalous slow fused-CPU repeat at 10.217 ms.
The focused three-repeat run resolves which candidate to retain, but does not
establish the cause of that earlier slowdown. Frequency snapshots were taken
before/after each arm, not continuously during execution. No thermal or DVFS
root cause is claimed.

The broad confirmation's pooled F16-only results are much smaller gains:

| F16 path | Old full mean | New full mean | New full p99 | Interpretation |
|---|---:|---:|---:|---|
| CPU native vs FMLAL 8x8 | 18.265 | 18.190 | 22.684 | 0.41% pooled gain; first repeat slower, tail worse; not selected |
| GPU old row-major vs 8x4 | 20.345 | 20.129 | 24.338 | 1.06% pooled gain; half width slightly slower; small improvement only |

Per-arm bracketing comparisons differ slightly from these pooled numbers:
CPU FMLAL -1.32%/+2.16% improvement; GPU +1.07%/+2.15%.
[All broad comparisons](PIXEL_DEVICE_LAYOUT_CONFIRM_COMPARISON.json) retain both
repeats and the immediately surrounding controls.

The coarse device-format sweep tested CPU4/6 and CPU shares from 50% through
82.353%. Some 50% CPU6 mixed arms exceed 22 ms even though both branches overlap.
At the same 82.353% share, GPU row-major gives 14.183 ms versus tiled GPU
10.830 ms. The subsequent fused sweep covered CPU shares 82.353%, 88.235%,
94.118% and 97.059%; increasing GPU work did not monotonically improve speed.
These are sampled optima, not a proof of the global best split.

## Concurrent path and weight storage

For the selected experimental mixed ratio, full-width ownership is CPU 16384
channels and GPU 1024 channels; half-width ownership is CPU 8192 and GPU 512.
CPU uses 6 persistent threads pinned to cores 2-7; the GPU host thread is pinned
to cores 0-1. The full-width warm mean branch intervals are:

| Interval | ms |
|---|---:|
| CPU branch | 7.885 |
| GPU branch including backend synchronization | 4.763 |
| Overlapping portion | 4.743 |
| Wait after CPU branch | 0.021 |
| Merge | 0.006 |
| Combined compute region | 7.912 |
| Complete phone worker | 7.928 |

Branch times overlap and must not be added. These are elapsed software intervals,
not GPU device-cycle counters. The small merge time shows that the sum itself
is not the main remaining cost. CPU remains the longer branch in this sample;
its performance can change when the GPU uses shared memory resources.

Logical resident weight payloads for all six layers, excluding runtime/graph
allocations and GGUF headers:

| Storage | Bytes | MiB |
|---|---:|---:|
| All F16 | 3208642560 | 3060.00 |
| Original packed CPU | 948387840 | 904.45 |
| Mixed CPU packed portion | 892600320 | 851.25 |
| Mixed GPU F16 portion | 188743680 | 180.00 |
| Mixed total | 1081344000 | 1031.25 |

The mixed representation uses 14.02% more weight storage than all-packed CPU.
This matches the worker allocation receipts; it is not a physical RSS/DRAM
measurement. Lossless tiling alone preserves resident bytes and uses one temporary
tensor-sized packing buffer, freed before steady execution. In the mixed run,
GPU tiling itself takes about 104 ms once at initialization; expansion, model
reading and upload are additional startup work. Startup is excluded from the
warm latency table. The stored GGUF artifacts remain unchanged; device layouts
are generated once while loading the private worker.

## Reproduction and validation

Exact run configs, request packets, input/output hashes, worker logs, affinity,
battery/frequency snapshots and normal exit receipts live under:

- [F16 layout smoke](physical/pixel10pro-f16-layout-smoke-1/run1/SUITE_RESULT.json)
- [FMLAL smoke](physical/pixel10pro-neon-fmlal-smoke-1/run1/SUITE_RESULT.json)
- [Device-format sweep](physical/pixel10pro-device-format-sweep-1/run1/SUITE_RESULT.json)
- [Fusion smoke](physical/pixel10pro-fused-residual-smoke-1/run1/SUITE_RESULT.json)
- [Broad confirmation](physical/pixel10pro-device-layout-confirm-1/run1/SUITE_RESULT.json)
- [Batch shape check](physical/pixel10pro-device-layout-batch-check-1/run1/SUITE_RESULT.json)
- [Endpoint confirmation](physical/pixel10pro-device-layout-endpoint-1/run1/SUITE_RESULT.json)

The batch suite builds 8 distinct rotated/scaled input vectors per layer and
independent single-row CPU references. It tests batches 1/2/4, checking each row;
row0 references are cross-checked against the archived desktop CPU results.
New GPU layouts accept only batch sizes 1/2/4/8; batch8 was not physically
qualified in this investigation and is not selected. Larger batches are rejected
before graph dispatch. All tuning candidates here are selected only for B1.

Build/static PASS: worker builds with -Wall -Wextra -Werror; disassembly contains
NEON FMA/FMLAL; 33 shader modules validate for Vulkan 1.2; 5 changed Python files
pass pyflakes/parse; 7 source files pass ASCII/whitespace checks; all 5 preserved
worker build snapshots match their provenance hashes.
[Final build checks](PIXEL_DEVICE_LAYOUT_FINAL_CHECKS.json).

Qualified worker v5 SHA256:
8604d14703ba64b8f62f63b901e8809ffb80e2a65cc758ef756cfc00cca5612d.
Qualified GPU library SHA256:
ee56097074684e03346c4a89f0d408d12fd0fe3b78e0cd06e572da78e0c4cd90.
Candidate JSON files record the exact environment, model identity and command.

Preparatory failures retained: first shader compile rejected reserved identifier
input; initial external validator path was absent; first smoke preparation lacked
its parent directory; two smoke configs lacked an audit-only control-list field;
endpoint preparation requested 40 repeats but only 20 archived reference repeats
exist. These were corrected before the affected physical run or, for audit-only
fields, without changing the executed arm. No failed physical arm was discarded.
Original files and failure records are preserved.

Cleanup PASS: same phone boot, no remaining FFN worker, no Pixel ADB forward,
Pixel lock acquired/released, no queued test. Only ADB port 5037 was used.
The desktop host model and other phones were not used by these finite probes.
[Cleanup receipt](PIXEL_DEVICE_LAYOUT_CLEANUP.json).

Not verified: physical DRAM/compute peak usage, energy savings, current USB/server
latency, server overlap, full-model token identity, sustained thermal performance,
or production integration. No commit, push or PR was made.
