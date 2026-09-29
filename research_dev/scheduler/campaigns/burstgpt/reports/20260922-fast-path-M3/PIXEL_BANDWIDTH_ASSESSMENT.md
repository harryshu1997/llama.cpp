# Pixel FFN bandwidth and concurrent CPU/GPU assessment

Followup: the requested streaming measurements and CPU FFN tuning are now
[complete](PIXEL_BANDWIDTH_RESULTS.md). This document retains the earlier
unmeasured assessment and estimates for provenance.

2026-09-23. Arithmetic/source review PASS. Physical DRAM bandwidth ceiling,
limiting hardware counters and concurrent CPU/GPU FFN speedup NOT VERIFIED.
No new hardware experiment, kernel change or integration in this assessment.

## What the measurements establish

The full one-token FFN has three F16 matrices containing
`3 * 5120 * 17408 * 2 = 534773760` bytes, or510MiB. Dividing these logical
weight bytes by the current matched worker times gives:

| Backend | Full worker mean | Logical weight bytes per worker second |
| --- | ---: | ---: |
| Pixel CPU,4 configured threads | 45.848635ms | 11.663897GB/s |
| Pixel GPU,vec4_u1 | 19.929104ms | 26.833808GB/s |

These are effective application rates, not DRAM performance counters. The
denominator includes graph execution, conversions, reductions, activation,
synchronization and worker overhead. The numerator excludes input/output and
intermediate traffic and does not account for cache effects or extra memory
transactions. They do not establish percent utilization of a hardware peak.
Neither sustained streaming-read bandwidth nor the phone's active memory
clock/bus configuration has been measured. Google's public specification
lists Tensor G5 and16GB RAM but does not establish a numerical memory-bandwidth
ceiling for this comparison.

The one-token matrix FLOPs equal the logical F16 weight-byte count, yielding
about1 matrix FLOP/weight byte. That makes data movement important, but does
not prove that the DRAM controller is saturated. A kernel can wait for memory
while issuing too few concurrent requests to fill the memory interface.

## Known costs and unverified limiting factors

- CPU: existing generic AArch64 build, four configured threads, no explicit
  core affinity or CPU-specific tuning. The worker supplies no persistent CPU
  threadpool, so the backend creates/frees one for each graph. Dot products,
  conversions and synchronization are real work beyond streaming reads.
- GPU: the shader loads F16 vectors, converts them to F32, calculates dot
  products, keeps eight output accumulators and reduces across a subgroup.
  The graph has12 GEMVs per full FFN. Good contiguous loads do not by themselves
  establish sufficient in-flight requests or occupancy to reach peak bandwidth.
- Occupancy, register allocation, instruction issue, memory-controller traffic
  and clocks were not measured. They remain hypotheses, not diagnosed causes.
  The fusion experiment's3.19% slowdown shows fewer launches alone did not
  improve this workload; it does not identify why bandwidth falls short.

PowerVR documents the importance of vector/contiguous access and avoiding
extra copies in its [bandwidth guidance](https://docs.imgtec.com/performance-guides/compute-recommendations/html/topics/bandwidth/minimise-usage.html).
Its [occupancy guidance](https://docs.imgtec.com/performance-guides/compute-recommendations/html/topics/utilisation/maximise-utilisation.html)
explains how resident tasks hide fetch latency and how private/shared storage
limits occupancy. These are relevant mechanisms, not Pixel-specific counter
measurements.

## Using CPU and GPU together

The detected GPU reports UMA. CPU and GPU share system memory; their physical
memory-bandwidth ceilings cannot be added as if they were independent RAM
systems. See the [Khronos UMA explanation](https://docs.vulkan.org/guide/latest/memory_allocation.html).
Concurrency can help if one engine leaves shared memory capacity unused. It
can also lose to contention, synchronization or changes in available clocks.
The current worker selects one backend; it does not run this combined path.

Split FFN hidden channels, with matching gate/up rows and down columns. Both
engines read the same input and calculate disjoint contributions:

`y = down_G * swiglu(gate_G*x, up_G*x) + down_C * swiglu(gate_C*x, up_C*x)`.

Keep disjoint weights resident, separate FP32 partial outputs, synchronize
explicitly and merge in a fixed order before converting to F16. Preserve
per-block contributions if matching the previous summation order matters.
CPU/GPU arithmetic already differs in low bits, so token equality remains a
separate requirement. Merely moving dependent layers between backends does
not create simultaneous work for one decode token.

An idealized linear model using the isolated rates is
`T(f) = max((1-f)*19.929104, f*45.848635)`ms, where f is the CPU share.
Ignoring shared-memory interference, fixed overhead and merge cost, it balances
at30.2976% CPU and69.7024% GPU:13.891055ms,1.4347x the GPU-only throughput.
Its38.497705GB/s effective aggregate is a model output, not measured bandwidth.
This is not an unconditional bound or a promised speedup.

The existing4352-channel blocks allow a simple25% CPU/75% GPU experiment.
Proportional scaling predicts14.946828ms. An affine interpolation from the
measured half/full widths predicts15.707987ms before any contention or merge
cost. The affine fit is based on only two widths and is not a validated model.
A50/50 split would already take about23.975ms at the isolated half-width CPU
rate, worse than the measured19.929ms GPU-only full FFN.

## Measurements that would resolve the question

1. Measure sustained streaming reads on CPU alone, GPU alone and both
   concurrently, with working sets larger than cache, observable checksums,
   consistent read-byte accounting, and recorded clocks/temperature where
   available. Keep read-only and read-plus-write copy bandwidth distinct.
2. Compare streaming reads with GEMV using the same weight layout/size. A large
   gap points toward kernel issue/occupancy/conversion/reduction overhead;
   similar rates point toward a memory-system limit for that engine.
3. Run GPU-only FFN versus explicit25/75 and other supported CPU/GPU splits,
   with disjoint weights and no per-request weight copy. Measure concurrent
   slowdown of each engine, total critical-path time, merge cost and output
   error using interleaved GPU-only controls.

Raw bandwidth tests and combined execution have not been run. All numerical
estimates above derive from the [existing CPU/GPU comparison](PIXEL_CPU_GPU_RESULTS.json).
