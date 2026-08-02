# Operator-split test: one matmul across A6000 + OP15 + OP12 over USB

Date: 2026-07-27; causal and desktop-proxy revalidation: 2026-07-29.

Current status:
`HTP_CACHED_INPUT_REPLAY_BUG; UID_ZERO_WORKAROUND_DYNAMIC_GREEDY_HEAD_PASS; DYNAMIC_TOP8_FAIL; COMPLETE_CPU_LAYER_RELATIVE_PASS_ARGMAX_FAIL; COMPLETE_CPU_LAYER_LATENCY_FAIL_ON_5995WX; I9_PLUS_PHONE_COMPOSED_ONLY; FLEET_PROJECTED_ONLY; ENERGY_NOT_MEASURED; FULL_MODEL_NOT_RUN`.

## 2026-07-31 CPU plus phone dynamic revalidation

This section supersedes any earlier HTP changing-input claim that did not
disable the graph cache or independently check every input. The cached HTP
graph can replay its first externally written input. DMA buffer synchronization,
uncached rpcmem, f32 transport, and disabling HOSTBUF, NHMX, or operation
fusion did not fix it. The bounded research workers set the graph UID to zero
under `S41_DISABLE_GRAPH_CACHE=1`; the Hexagon core defect remains open.

With that bypass, a real OP15 plus 5995WX sharded Gemma vocabulary head is
17.14% faster at median and 17.05% faster at p90 across three alternating
100-iteration pairs. Changing-input merged greedy top-1 is 30/30, but exact
global top-8 is only 27/30, so this is a greedy-only pass.

A complete Gemma-like CPU layer with a 1,792-column phone FFN suffix passes a
0.001812 maximum relative-L2 bound across 30 distinct inputs but changes one
hidden-state argmax. It is also 106.44% slower than the globally tuned 5995WX
CPU control. Physical i9 component measurements predict an 8.68% one-phone
complete-layer reduction, but the phone is not cabled to that host and the
result is explicitly composed.

The cache-disabled HTP FFN leg is 0.743 ms versus 2.810 ms on the valid
Adreno OpenCL fallback. HTP remains the selected phone backend. Full details,
fleet ceilings, source hashes, and claim limits are in
`RESULTS_CPU_PHONE_OPERATOR_OFFLOAD_V1.md`.

## 2026-07-30 Gemma global-attention context shard

A new bounded path partitions the sequence dimension of one Gemma-4-12B
global-attention core between CUDA and a real OP15 HTP v81. It uses 16 query
heads, one KV head, head dimension 512, f16 resident K/V, direct AOA USB, and
an exact online-softmax merge derived from one anchor logit/probability pair
per head.

The A6000 was calibrated for this exact operator. At 990 MHz graphics and
5001 MHz memory it took 2.354921 ms for 262K context, within 0.315% of the
physical 4060 Ti's 2.362369 ms. The earlier 240/5001 setting took 6.724929 ms
and is not a valid 4060 proxy for this operator.

| schedule | phone KV | CUDA | CUDA + OP15 | median change | p90 change |
| --- | ---: | ---: | ---: | ---: | ---: |
| continuous | 8,192 | 2.353188 ms | 2.300496 ms | -2.239% | -1.705% |
| gapped | 8,192 | 2.356254 ms | 2.431027 ms | +3.173% | +10.471% |
| gapped | 4,096 | 2.355132 ms | 2.338630 ms | -0.701% | +10.588% |

All paths passed correctness. Final relative L2 was 0.0001602 for the 8K
suffix and 0.0000702 for the 4K suffix. This is only the projected-Q,
resident-KV attention core. It is not a complete layer, full-model, capacity,
or energy claim. The next gate is eliminating the gapped USB wake tail before
any runtime integration.

Full report:
`RESULTS_GEMMA_GLOBAL_ATTENTION_V1.md`.

## 2026-07-29 dynamic-input and operator-route v2

This section supersedes the earlier one-input operator timings where the
result differs. The hardware path was an RTX A6000 locked at 240 MHz graphics
and 5001 MHz memory plus a rooted OP15 HTP v81 over direct AOA USB. The A6000
setting is a controlled low-frequency proxy, not an exact RTX 4060 Ti.

Evidence root:
`results/operator_routes_v2/run_20260729T165637Z/`.

### Correctness fixes

An alternating-input control found that the HTP worker returned the first
activation again after a different request. The single static allocation had
buffer usage `ANY`, so the HTP input descriptor was not marked as temporal
compute data and its CPU-written contents were not flushed before reuse.

The fix has two layers:

- each worker calls `ggml_set_input()` for its externally updated tensor;
- the deployed workers allocate that input in a separate buffer with
  `GGML_BACKEND_BUFFER_USAGE_COMPUTE`, keeping large immutable weights out of
  the per-request cache flush;
- the Hexagon backend source also treats `GGML_TENSOR_FLAG_INPUT` as compute
  data when building an HTP descriptor.

The backend-source change was not rebuilt into the phone library in this run:
the surviving Android CMake cache points at the unavailable `/workspace` and
`/opt/hexagon` build environment. All physical results instead exercise the
separate compute-buffer fix, which works with the already deployed backend.
The backend-source line remains a source-only follow-up until that SDK build
environment is restored.

Before the fix, a layer-A request followed by a different FFN-B request had
0.787 final relative L2 error and the wrong argmax. After the fix, the same
A-then-B sequence had 0.00467 relative L2 error and exact argmax. The HTP
worker reported a 4 us input set and about 0.54 ms graph compute for Qwen.

The attention residual path also had a host response-allocation bug: it
allocated the 1,280-byte state payload even in the 10,240-byte residual mode.
The first real run failed with `LIBUSB_ERROR_OVERFLOW`; response storage now
selects the exact mode-dependent size.

### Final isolated operator results

All rows use q8_0 weights on CUDA and HTP, f16 input and output on the AOA
wire, 300 paid repetitions unless noted, exact request and weight
certificates, finite-output checks, and argmax gates.

| route | model/context | CUDA-only | CUDA + OP15 | median change | correctness |
| --- | --- | ---: | ---: | ---: | --- |
| complete FFN residual, host merge | Qwen3-14B | 1.0806 ms | 0.9951 ms | -7.91% | rel L2 0.00467, argmax exact |
| complete FFN residual, CUDA merge | Qwen3-14B | 1.0805 ms | 1.0032 ms | -7.16% | rel L2 0.00467, argmax exact |
| complete FFN residual, host merge | Gemma-4-12B | 0.7694 ms | 0.6960 ms | -9.53% | rel L2 0.00499, argmax exact |
| complete FFN residual, CUDA merge | Gemma-4-12B | 0.7690 ms | 0.7357 ms | -4.33% | rel L2 0.00499, argmax exact |
| sharded vocabulary, global top-8 | Qwen3-14B | 3.2062 ms | 2.7733 ms | -13.50% | global top-8 exact |
| sharded vocabulary, global top-8 | Gemma-4-12B | 4.3744 ms | 3.6516 ms | -16.52% | global top-8 exact |
| one tail GQA group, residual return | Qwen, KV=8,192 | 0.8240 ms | 0.8408 ms | +2.05% | rel L2 0.00272, argmax exact |
| one tail GQA group, state return | Qwen, KV=16,384 | 1.2497 ms | 1.1905 ms | -4.74% | rel L2 0.00190, argmax exact |

The top-8 protocol returns eight sorted phone candidates in a 64-byte
response and merges them with the CUDA shard. The merged global top-8 is
exact for both models. Qwen's phone-local positions agree at 6/8 because of
backend numerical differences, but this does not change the exact global
result.

The attention policy is therefore context-dependent. Residual return is not
eligible at 8K. State return becomes eligible at 16K: the phone returns only
640 f16 attended-state values, then CUDA applies the phone output-projection
slice and residual add. Its 1,280-byte response and 0.090 ms CUDA continuation
fit under the seven-group CUDA path. The 32K attempt was not timed because
the frozen local-CUDA partition gate failed at 0.000310 relative L2 against a
0.0001 limit, even though treatment argmax and the phone error passed.

Residual return at 16K is also ineligible: it returned a zero phone residual
and failed correctness. State return is the only validated long-context
attention route.

### Real one-layer results

`causal_layer_host.cpp` executes a complete synthetic decode layer: RMS
normalization, Q/K/V projections, GQA attention over a real KV tensor, output
projection, attention residual, FFN normalization, and complete FFN
residual. CUDA finishes the attention dependency first; CUDA and OP15 then
run complementary FFN column partitions. The phone result is uploaded and
summed on CUDA, so the measured result remains device-resident.

The treatment now launches the CUDA FFN partition asynchronously and serves
AOA on the caller thread. This removes per-token `std::thread` creation and
reduced Qwen split latency from 1.529 ms to 1.506 ms. Control and treatment
run in separate steady processes because alternating unrelated CUDA graphs
inside one process caused repeated graph warmup resets and a false 2.799 ms
treatment result.

At KV=136, three process-level 300-iteration pairs give:

| model | CUDA-only median | CUDA + OP15 median | change | CUDA + OP15 p90 change | device-ready |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-14B | 1.5588 ms | 1.5059 ms | -3.39% | -1.96% | 1.4952 ms |
| Gemma-4-12B | 1.1707 ms | 1.1440 ms | -2.29% | +6.28% | 1.1338 ms |

All six treatments passed the final-output and phone-slice gates. Qwen is a
repeatable median and p90 win. Gemma is only a median win: one of three
treatment medians was 1.1931 ms, and the run-level median p90 is worse than
CUDA-only. Gemma should not yet be scheduler-eligible under a tail-latency
SLO.

At KV=8,192, the complete layer does not win:

| model | CUDA-only | CUDA + OP15 | change |
| --- | ---: | ---: | ---: |
| Qwen3-14B | 2.1727 ms | 2.3661 ms | +8.90% |
| Gemma-4-12B | 1.6252 ms | 2.0986-2.1262 ms | +29.1% to +30.8% |

For Qwen at 8K, the treatment breakdown is 1.108 ms attention phase, 1.179 ms
overlap span, 1.154 ms phone path, and 0.029 ms late merge. For Gemma, the
late CUDA add itself grows to 0.253-0.294 ms after the larger attention graph,
despite taking about 0.015 ms at KV=136. This dependency and copy scheduling
cost is now the main integration target. The isolated FFN win cannot be
multiplied by layer count without accounting for this boundary.

### Honest verdict

The mechanism is now valid for changing token activations, not just repeated
synthetic input. Three bounded routes beat the low-frequency CUDA proxy:
complete FFN partitioning, exact top-8 vocabulary sharding, and a 16K
single-GQA-group state route. A real complete layer also has a small,
repeatable Qwen win at short context.

This is not a full-model or BurstGPT result. Large-context complete-layer
offload still loses, Gemma tail latency is unstable, only one phone is in the
treatment, and no fleet or total-energy claim follows from these timings.
The next narrow implementation step is to fold the late phone residual into
the next CUDA layer graph (or use pinned staging) and then run a multi-layer
decode loop before changing the main model runtime.

## 2026-07-28 causal Q8 revalidation

The later `full_layer_host.cpp` result was invalid. It started the phone from a
constant vector before CUDA had produced the attention-dependent FFN input,
initialized unrelated CUDA and phone weights, and had no output correctness
gate. Its reported 1.65x layer speedup, composed 4060 Ti result, energy result,
and N greater than 1 projection are superseded. The earlier raw matmul tests
below remain historical mechanism diagnostics.

### Corrected implementation

The replacement is a bounded synthetic-layer probe, not a full llama decoder:

- `causal_layer_host.cpp`: CUDA completes attention first and materializes the
  exact FFN input. CUDA and OP15 then run complementary FFN column slices.
- `causal_ffn_worker.cpp`: OP15 runs gate, up, SiLU, multiply, and down as one
  HTP graph and returns its partial residual contribution over direct AOA USB.
- `causal_quantized_weights.h`: host and phone consume deterministic q4_0 or
  q8_0 bytes directly. They no longer quantize float generators independently.
- Protocol v2 binds request ID, dimensions, type, input hash, output hash, and
  an exact 64-bit quantized-weight fingerprint.
- The gate compares monolithic CUDA, path-matched two-phase CUDA, a local CUDA
  partition oracle, the phone slice, and the final CUDA plus phone output.
- `split` and `monolithic` modes isolate performance and power windows so CUDA
  graph switching does not contaminate the steady-state result.

Two live bugs were found and fixed:

1. Android's accessory driver sized its gadget request to `read()`. Reading
   only the 40-byte header caused a larger host bulk transfer to be dropped.
   The worker now reads the complete fixed request packet at once.
2. Independent x86 and ARM quantization produced different q4_0 bytes for some
   slice sizes. The new weight certificate caught this. Direct deterministic
   block construction now gives byte-identical weights on both devices.

### Hardware and controls

- Actual desktop control: RTX 4060 Ti
  `GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08`.
- Treatment host: RTX A6000
  `GPU-45b611d6-7c5d-9e48-f260-4fbe5f8ef69f`, locked at 405 MHz graphics and
  5001 MHz memory.
- Phone: real OP15, HTP v81, direct AOA USB, 256 FFN columns.
- Work: one M=1 synthetic decode layer, 512-entry KV cache, q8_0 weights on
  CUDA and HTP.
- Repetitions: three long isolated runs per path with alternating order.

Clock matching is kernel-dependent. At 405/5001, q4_0 A6000 latency was within
about 3.4% of the real 4060 Ti, but q8_0 remained 18.6% to 24.5% faster because
the A6000 memory system differs substantially. A single frequency setting is
therefore not a faithful universal 4060 Ti proxy. The real 4060 Ti controls are
reported directly. Since the A6000 treatment GPU is faster than the real 4060
Ti for both q8_0 shapes, the cross-host treatment comparison below is
conservative: slowing its GPU cannot turn the measured loss into a win.

### Correctness

| shape | exact weight hash | phone-slice rel L2 | final rel L2 | argmax |
| --- | --- | ---: | ---: | --- |
| Qwen3-14B | `e3ef03dd51249350` | 0.014827 | 0.000935 | 284 / 284 |
| Gemma-4-12B | `2e263bfde29814ac` | 0.013408 | 0.000531 | 284 / 284 |

All local partition controls were below `6.1e-8` relative L2. All treatment
runs had zero non-finite values and passed the 0.03 phone-slice and 0.005 final
relative-L2 gates.

### Q8 latency

| shape | real 4060 Ti only | A6000 only | A6000 + OP15 | vs A6000 | vs real 4060 Ti |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-14B | 1.367 ms | 1.031 ms | 1.637 ms | 1.587x slower | 1.197x slower |
| Gemma-4-12B | 0.950 ms | 0.773 ms | 1.311 ms | 1.696x slower | 1.380x slower |

The 256-column slice was best in a bounded q8_0 sweep:

| shape | 256 columns | 512 columns | 1024 columns |
| --- | ---: | ---: | ---: |
| Qwen3-14B | 1.656 ms | 1.704 ms | 1.962 ms |
| Gemma-4-12B | 1.322 ms | 1.352 ms | 1.435 ms |

The phone round trip at the selected cut was about 1.119 ms for Qwen and
0.875 ms for Gemma. Attention must finish before that exchange begins. This
dependency leaves no overlap schedule that beats either monolithic control.

### A6000 board energy

Power was arrival-stamped from a continuous 100 ms `nvidia-smi` stream and
integrated with zero-order hold only inside host-emitted paid-window markers.
Every sample in all six runs per shape reported exactly 405/5001 MHz.
An independent trapezoidal re-integration of the raw samples differed from the
reported zero-order-hold values by at most 0.55%.

| shape | control power | split power | control J/layer | split J/layer | change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-14B | 124.68 W | 97.32 W | 0.12864 | 0.16786 | +30.5% |
| Gemma-4-12B | 123.25 W | 94.61 W | 0.09532 | 0.13166 | +38.1% |

Offload lowers instantaneous server-board power by 21.9% to 23.2%, but the
longer critical path raises server energy per layer. Phone energy is not added:
the USB rail was pinned at 494 mA against a 500 mA input limit and the low
battery was charging, so no defensible phone-energy integral is available.
Since the server board alone already consumes more energy per completed layer,
adding any positive phone energy can only worsen total fleet energy.

### Verdict and scope

The corrected mechanism works on real CUDA plus OP15 HTP with exact shared
q8_0 weights and causal dataflow. It does not improve latency, server energy,
or server capacity at M=1 for either model shape. Do not extrapolate an N-phone
optimum from this result.

This test does not establish full-model decode, continuous batching, BurstGPT,
or fleet energy. The useful system direction remains request/session-level
phone service during model loading and model switching. Within-layer phone
offload should remain a negative microbenchmark unless a future transport or
device reduces the phone critical path below the post-attention slack.

Evidence root:
`results/causal_v1/run_20260728T194325Z/`. The final derived record is
`ANALYSIS_FINAL.json`; both energy directories contain raw host logs, raw power
samples, source hashes, and independently checkable integrals.

## 2026-07-28 activation-return critical-path reduction

The full-output result above made the phone compute the suffix down projection
and return a full residual. A bounded successor instead returns the smaller
gated activation and leaves the suffix down projection on CUDA. It also applies
the following measured reductions:

- scaled int8 input and f16 output on the AOA wire;
- one fused gate/up HTP matmul;
- vectorizable host input quantization;
- one resident root worker on OP15 CPUs 6-7 under `SCHED_FIFO` priority 20;
- a two-node late CUDA graph that treats the completed prefix as an external
  input and fuses the suffix down projection with the prefix bias; and
- steady treatment cadence after separately measuring the CUDA controls.

The cadence distinction is material. Qwen phone-only service was 0.475 ms
back-to-back but 0.802 ms after a 1.5 ms gap, without any CUDA work. The
alternating-control loop therefore measured Android wakeup and USB service
phase as well as the phone operation. It was not a valid estimate of a
continuously occupied split route.

Three fresh worker processes per shape, 500 samples per process, produced:

| shape | CUDA prefix deadline | phone median | phone p90 | median hidden | late CUDA | CUDA full | split total | change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-14B | 0.651 ms | 0.315 ms | 0.359 ms | 99.0% | 0.042 ms | 0.670 ms | 0.693 ms | +3.5% |
| Gemma-4-12B | 0.461 | 0.304 | 0.348 | 98.0% | 0.042 | 0.476 | 0.503 | +5.8% |

These are medians across the three independent process medians. Phone p90 is
below the CUDA deadline for both shapes. Rare outliers remain, so the
all-sample no-wait gate still fails.

The numerical gates pass with the same deterministic q8_0 weights:

| shape | phone activation rel L2 | final rel L2 | argmax |
| --- | ---: | ---: | --- |
| Qwen3-14B | 0.007498 | 0.001627 | exact |
| Gemma-4-12B | 0.008366 | 0.001802 | exact |

The phone is no longer the median or p90 critical path. M=1 still loses because
the 0.042 ms late CUDA continuation is larger than the CUDA work removed by the
cut: about 0.019 ms for Qwen and 0.015 ms for Gemma. Eliminating phone wait is
necessary but not sufficient.

This successor did not run a power acquisition. Do not reuse the full-output
energy result for it. The next bounded test is B=2, 4, and 8 independent
requests, where transport cadence and the late continuation can be amortized.
Stop this branch unless matched total latency or useful throughput beats CUDA.

Evidence root:
`results/activation_return_v1/run_20260728T210501Z/`. The derived record is
`ANALYSIS.json`.

## 2026-07-28 Qwen gate/up split balancing

The 0.315 ms phone result above applies only to a 256-column suffix. It is not
a fixed transport cost. A larger suffix makes the CUDA prefix faster, but also
increases HTP work, the f16 activation return, and the serial CUDA suffix-down
continuation. The correct M=1 objective is therefore:

```text
minimize max(cuda_prefix, phone_total) + late_cuda
```

A live Qwen3-14B sweep varied the fused gate/up phone suffix from 256 through
2,048 columns. Every point used the same q8_0 weights, scaled-int8 request,
f16 return, direct AOA path, locked A6000 clocks, and real OP15 HTP worker.
All correctness gates passed. The main crossover was:

| phone columns | CUDA prefix | phone median | phone p90 | late CUDA | split total | vs CUDA |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 0.651 ms | 0.318 ms | 0.363 ms | 0.042 ms | 0.693 ms | +3.54% |
| 1,024 | 0.624 | 0.437 | 0.486 | 0.061 | 0.686 | +2.51% |
| 1,536 | 0.606 | 0.525 | 0.575 | 0.069 | 0.676 | +0.99% |
| 1,792 | 0.597 | 0.555 | 0.603 | 0.072 | 0.670 | +0.17% |
| 2,048 | 0.588 | 0.605 | 0.644 | 0.075 | 0.685 | +2.31% |

The 1,664-column point was slower than both neighboring aligned shapes because
its late CUDA down projection rose to about 0.075 ms. Slice selection must use
measured end-to-end shape performance, not a smooth throughput model.

Three fresh 500-sample worker processes then repeated the useful candidates:

| phone columns | CUDA prefix | phone median | phone p90 | exposed wait p90 | late CUDA | split total | vs CUDA | hidden |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,536 | 0.606 ms | 0.528 ms | 0.572 ms | 0.000 ms | 0.069 ms | 0.676 ms | +0.95% | 94.6% |
| 1,664 | 0.603 | 0.549 | 0.599 | 0.000 | 0.075 | 0.680 | +1.64% | 90.2% |
| 1,792 | 0.597 | 0.562 | 0.610 | 0.016 | 0.072 | 0.671 | +0.21% | 75.6% |

Values are medians across the three process-level statistics. At 1,792
columns, the median phone leg stays 0.036 ms ahead of CUDA and the complete
split is only 0.0012 ms behind CUDA-only. This is median parity, not a speedup.
The phone p90 crosses the CUDA deadline, and split p90 remains about 2.6% above
CUDA p90. The 1,536-column cut is the conservative p90-hidden choice; the
1,792-column cut is the selected median-latency choice.

The 1,792-column result passes exact argmax and has final relative L2
0.004320 against the frozen 0.005 gate. At 2,048 columns it rises to 0.004656,
so larger scaled-int8 slices also consume most of the numerical margin.

This is still one synthetic M=1 FFN layer. It does not establish full-model,
continuous-batch, RTX 4060 Ti treatment, or energy improvement. The result does
show that measured shape-aware balancing removes almost all of the earlier
3.5% Qwen penalty without changing the phone or transport implementation.
Gemma requires its own sweep because its dimensions and CUDA kernel boundaries
are different.

Evidence root:
`results/activation_balance_v1/run_20260728T222515Z/`. The derived record is
`ANALYSIS.json`; all coarse, fine, and repeated raw host/phone logs are retained.

## 2026-07-28 contiguous late boundary and continuous cadence

The one-phone audit found that the returned activation was already logically
contiguous and already fed one suffix-down graph. The avoidable boundary was
more specific: the host converted the f16 wire payload to f32, synchronously
uploaded 7 KiB to CUDA, waited for that upload, and only then launched the
late graph.

The successor retains the numerically passing f32 CUDA input but enqueues the
contiguous upload asynchronously on the same CUDA stream as the suffix-down
graph. Stream order preserves the dependency without the separate
synchronization. Four selectable paths keep the old control reproducible.
Direct f16 was also tested rather than assumed.

Three fresh 500-sample paired runs produced:

| late path | upload submit | late CUDA | CUDA full | split total | change |
| --- | ---: | ---: | ---: | ---: | ---: |
| f32 synchronous | 0.0090 ms | 0.0719 ms | 0.6694 ms | 0.6708 ms | +0.18% |
| f32 asynchronous | 0.0040 | 0.0666 | 0.6695 | 0.6655 | -0.59% |

Values are medians across the three process medians. All three asynchronous
runs beat their own CUDA-only median. The final relative L2 remains 0.004320
against the 0.005 gate, with exact argmax and no non-finite values.

Direct f16 input is rejected. It passes correctness, but the CUDA q8-by-f16
suffix-down kernel raises the late continuation to 0.155-0.160 ms and makes
the layer 12.8-13.4% slower. Halving the upload bytes is not useful when it
selects a much slower compute kernel.

### Continuous-cadence control

Three paired phone-only runs compared back-to-back requests with the same
route after a 1.5 ms gap:

| cadence | phone total | USB OUT | USB IN plus compute |
| --- | ---: | ---: | ---: |
| continuous | 0.587 ms | 0.047 ms | 0.522 ms |
| 1.5 ms gap | 0.773 ms | 0.238 ms | 0.518 ms |

Continuous cadence is 24.1% faster. The HTP portion does not become slower
after the gap. Almost the entire penalty is Android accessory-service wakeup
before USB OUT, so the runtime should keep an admitted phone route occupied
instead of spacing individual operator calls.

### Phone compute profile

The native Hexagon profiler resolves the fused phone graph:

| HTP operation | native time |
| --- | ---: |
| stacked q8 gate/up matmul | 271-275 us |
| SiLU | 3 us |
| multiply | 2 us |

The matmul accounts for about 98% of native operator time. Internal trace
events show per-tile weight DMA intervals roughly three to four times longer
than HVX compute intervals. This M=1 route is weight-bandwidth-bound, not
SiLU-bound or multiply-bound.

Bounded controls reject the obvious alternatives:

- forcing HMX at M=1 raises matmul time to about 411-412 us;
- forcing flat HVX returns an incorrect zero activation and fails closed;
- completion polling provides no benefit;
- toggling graph op fusion has no material effect because gate and up are
  already stacked into one physical matmul; and
- four HVX threads reduce the repeated worker compute median from 375 to
  371 us and improve phone tails relative to the default eight threads.

With asynchronous upload plus four HVX threads, three fresh 500-sample co-runs
give:

| metric | median across process statistics |
| --- | ---: |
| CUDA-only | 0.6695 ms |
| phone median / p90 | 0.5480 / 0.5927 ms |
| CUDA prefix | 0.5975 ms |
| late CUDA | 0.0651 ms |
| split total | 0.6634 ms |
| median change | -0.86% |
| phone hidden fraction | 90.8% |

Phone p90 is now below the median CUDA deadline in the repeated aggregate.
The split p90 is still 0.6733 ms versus 0.6710 ms CUDA-only, so no p90 speedup
is claimed. A final run from the exact final source and its default
`f32_async` path confirms a 0.6654 ms split and 0.6695 ms control.

This is the first corrected causal M=1 median speedup, but remains one
synthetic FFN layer on the clock-reduced A6000 transport host. It does not
establish a full-model, RTX 4060 Ti, multi-phone, capacity, or energy result.
The next phone-compute optimization should batch independent activation rows:
that reuses the dominant q8 weight traffic and makes HMX naturally eligible.
Further M=1 compute-clock tuning is unlikely to fix a weight-DMA-bound kernel.

Evidence root:
`results/activation_boundary_v2/run_20260728T230145Z/`. The derived record is
`ANALYSIS.json`; raw paired logs, profiler output, trace events, negative
controls, source hashes, and the final-binary confirmation are retained.

## 2026-07-28 HTP queue-arena overhead

The Hexagon backend defaulted to an operation-batch capacity of 1,024 even
though `HTP_OP_MAX_REQS` is 256. Capacity controls the size of every pinned
queue block, not only how many operations are submitted. At queue depth 16,
the unused descriptor capacity reserves 9,967,616 bytes. Using the existing
256-operation limit reserves 2,496,512 bytes, a 75.0% reduction.

Three alternating 500-request phone-only repetitions show:

| batch capacity | phone median | worker graph compute |
| --- | ---: | ---: |
| 1,024 | 0.5808 ms | 398 us |
| 256 | 0.4836 ms | 307 us |
| exact graph size, 3 | 0.4809 ms | 302 us |

The general 256-operation setting recovers nearly all of the graph-specific
setting: phone latency falls 16.7% and worker graph time falls 22.9%.
Profiling with the exact three-operation setting leaves native HTP work
unchanged at 262-263 us while the non-operation batch envelope falls from
98 us to 22 us. The gain is queue and descriptor handling, not a faster
matmul kernel.

The backend default now uses `HTP_OP_MAX_REQS`. An Android build from the
changed source was deployed separately and reproduced the result without an
environment override. All correctness gates pass.

Matched 500-sample co-runs give the following medians across three fresh
processes:

| metric | old default 1,024 | compiled default 256 |
| --- | ---: | ---: |
| phone median / p90 | 0.5514 / 0.6145 ms | 0.5351 / 0.5861 ms |
| exposed wait p90 | 0.0206 ms | 0 |
| split median | 0.6654 ms | 0.6654 ms |
| split p90 | 0.6864 ms | 0.6704 ms |
| CUDA-only median / p90 | 0.6695 / 0.6710 ms | 0.6696 / 0.6711 ms |

The CUDA-hidden median was already insensitive to phone time. The useful
effect is tail headroom: the old backend makes the split 2.30% slower than
CUDA-only at p90, while the new aggregate is approximately parity at 0.11%
faster. That 0.11% is too small to claim a robust p90 speedup, but the
previous p90 penalty and phone wait are removed in this acquisition.

Reducing queue depth does not help. A 10 us DSP callback-poll build is also
rejected: it increases wake activity, does not improve the tail, and slows
profiled native HTP work. The source experiment was reverted. M=1 kernel
work remains weight-DMA-bound, so the next kernel experiment is multi-row
batching for weight reuse rather than more polling or clock tuning.

Evidence root:
`results/htp_overhead_v1/run_20260728T232510Z/`. `ANALYSIS.json` contains
the derived values and explicit claim limits. Raw phone and host logs retain
all repetitions, sweeps, profiler controls, and rejected variants.
The evidence manifest SHA256 is
`cab99509b7d6c4152bd3f2e54fc3a8d354dc1bba31c61e31564ad2a212bc33b6`.
Both the Android backend and current `test-backend-ops` build pass. A broad
device test completed 535 supported cases before reaching the already-known
noncontiguous `NORM` `dspqueue_read 0x2e` abort; it is retained as a partial
regression, not mislabeled as a full-suite pass.

## 2026-07-29 RTX 4060 Ti latency proxy and one-phone retune

This successor targets the desktop 4060 Ti rather than the faster default
A6000. The same exact CUDA binary measured 1.0616 ms on the physical 4060 Ti.
Locking A6000 GPU 0 to 240 MHz graphics and 5001 MHz memory produced three
CUDA-only medians of 1.080892, 1.080992, and 1.080902 ms. The proxy is 1.81%
slower than the physical desktop for this exact Qwen q8_0 FFN layer. This
calibration is kernel-specific and does not make A6000 board power equivalent
to 4060 Ti board power.

### Overlapped implementation

The phone computes a suffix of the fused gate/up projection and returns the
gated f16 activation. A second CUDA backend immediately starts the matching
suffix down projection while the primary CUDA stream finishes the prefix.
Pinned host buffers receive the two residual contributions, which are added on
the host. This removes the false dependency that previously serialized the
suffix down projection behind the CUDA prefix.

The selected cut assigns 3,328 of 17,408 gate/up columns, or 19.12%, to OP15.
Both devices use identical deterministic q8_0 weights; AOA carries f16 input
and f16 output. The final relative L2 is 0.004946 against the frozen 0.005
gate, with exact argmax and zero non-finite values. The cut is therefore near
the numerical limit. A larger 3,392-column cut still passes at 0.004971 but is
slower.

Three fresh 500-request phone workers reproduced the selected cut:

| run | CUDA median | split median | median change | CUDA p90 | split p90 | p90 change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.081653 ms | 0.975659 ms | -9.80% | 1.083947 ms | 0.980377 ms | -9.55% |
| 2 | 1.081012 | 0.975969 | -9.72% | 1.082996 | 0.982622 | -9.27% |
| 3 | 1.081552 | 0.976550 | -9.71% | 1.084799 | 0.998492 | -7.96% |

Phone p90 was 0.818, 0.816, and 0.871 ms against a CUDA deadline near
0.884 ms. Every run had zero exposed wait at p90 and beat CUDA-only at p90.
Rare phone stalls remain, so no all-sample no-wait claim is made.

### Server-board energy

The host now has diagnostic-only `cuda_energy` and `split_energy` modes. They
emit nanosecond paid-window markers after initialization, correctness, and
warmup. The existing power runner was extended to invoke these paths. Three
12,000-layer pairs used the alternating order
`CUDA, split, split, CUDA, CUDA, split` with a continuous 50 ms
`nvidia-smi` stream.

| path | run J/layer | median J/layer | median power | paid-window mean time |
| --- | --- | ---: | ---: | ---: |
| A6000 CUDA-only proxy | 0.135233, 0.138340, 0.138829 | 0.138340 | 127.95 W | 1.0817 ms |
| A6000 plus OP15 proxy | 0.123334, 0.124350, 0.127607 | 0.124350 | 124.18 W | 1.0069 ms |

The measured A6000 server-board reduction is 10.11% per completed layer.
Median board power falls only 2.95%; most of the energy gain comes from the
shorter paid window. The paid-window mean improves 6.91%, less than the 9.72%
median-latency result because rare stalls contribute to energy and throughput.

All six windows stayed exactly at 240/5001 MHz. Each window has 236-256 power
samples and a maximum sample gap below 52.1 ms. Independent trapezoidal
integration differs from the recorded zero-order-hold values by 0.12-0.13%.
The phone worker remained stable at a 554 us median graph compute time through
36,000 paid requests.

Phone energy was not synchronized in this acquisition. As a sensitivity
calculation only, adding 1.74 W marginal phone power gives 0.12610 J/layer and
an 8.85% proxy saving; adding 3.99 W total phone power gives 0.12837 J/layer
and a 7.21% proxy saving. These are not fleet-energy measurements.

The physical RTX 4060 Ti all-server control was also measured directly for
three 12,000-layer runs:

| run | mean time | board power | J/layer |
| ---: | ---: | ---: | ---: |
| 1 | 1.062067 ms | 106.10 W | 0.112685 |
| 2 | 1.062040 | 108.51 W | 0.115245 |
| 3 | 1.061980 | 108.79 W | 0.115537 |

Its median is 0.115245 J/layer. The A6000 split value must not be compared as a
physical 4060 treatment: it is 7.9% higher in raw J/layer because it is a
different board. A real desktop energy claim requires attaching the phone
transport to the 4060 Ti host, for example by moving the cable or using a
validated USB-over-IP path, and repeating the same paired acquisition.

Evidence:

- A6000 latency proxy and energy:
  `results/activation_4060_proxy_v1/run_20260729T034704Z/`
- Physical 4060 Ti control:
  `results/physical_4060_control_v1/physical_4060_control_20260729T035025Z/`
- Host binary SHA256:
  `5ed68836bbbf244d85304d1f02e2d7de8bdc301b9a830f107b462bd86b67c87f`
- A6000 energy record SHA256:
  `bcd5dc9b4ae2e6ac08e6b37cc05e75f0aae9359c1a8a9616504bb68c364587d2`

This remains one synthetic causal Qwen FFN layer at M=1. It does not establish
full-model decode, continuous batching, BurstGPT, multi-phone scaling, or
physical 4060 Ti treatment energy.

## 2026-07-29 Batched FFN and short-prefill crossover

The activation-return path now accepts a row count `M` without changing the
48-byte request or 32-byte response layouts. The high byte of the existing
request flag field carries `M`; `M=1` still encodes zero in that byte and is
wire-compatible with the prior worker. CUDA tensors use `[width,M]`, all rows
contain distinct deterministic inputs, and correctness now requires an exact
argmax independently for every row.

The first implementation used strided 2D views over the fused gate/up output.
The real HTP backend returned `NO-SUPPORT` and zero output even at `M=1`.
The batch-1 regression caught this before measurement. The corrected worker
keeps the proven stacked gate/up matmul and 1D views for `M=1`; for `M>1` it
uses separate contiguous gate and up matmuls in the same graph. No zero-output
run is included below.

All weights remain deterministic q8_0 on CUDA and HTP. `M=1` retains f16
activation input and output. `M>1` uses i8 input transport and f16 output to
avoid doubling the AOA input payload per row. This changes activation
transport precision, not weight precision. Every selected point passes the
frozen final relative-L2 limit of 0.005, has exact row-wise argmax, and has no
non-finite values.

### Live A6000 plus OP15

These are three fresh 500-request repetitions per point. A6000 GPU 0 remained
locked at 240 MHz graphics and 5001 MHz memory. OP15 used HTP over direct AOA.
Each row is one independent FFN item, so `M=2` and `M=4` model continuous
decode batches; `M=8` and `M=16` are short-prefill kernel probes.

| M | phone columns | input | CUDA-only median | split median | median change | CUDA p90 | split p90 | p90 change |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3,328 | f16 | 1.080961 ms | 0.975158 ms | -9.79% | 1.082485 | 0.978424 | -9.61% |
| 2 | 2,304 | i8 | 1.131178 | 1.047116 | -7.43% | 1.139654 | 1.051034 | -7.78% |
| 4 | 1,536 | i8 | 1.705654 | 1.616212 | -5.24% | 1.712677 | 1.625459 | -5.09% |
| 8 | 768 | i8 | 2.748923 | 2.732180 | -0.61% | 2.763742 | 2.754784 | -0.32% |
| 16 | 2,816 | i8 | 2.111154 | 2.970560 | +40.71% | 2.127497 | 3.081523 | +44.84% |

The median phone times are 0.755, 0.676, 0.912, 1.114, and 2.688 ms,
respectively. At `M<=4`, the phone branch is hidden in at least 98.8% of
requests and both median and p90 improve. `M=8` is only parity and is not a
useful win. At `M=16`, CUDA switches to a much more efficient matrix path
while AOA payload and phone work continue to grow, so the split fails badly.

### Physical 4060 Ti control and composed treatment

The 240 MHz A6000 calibration is only valid for the batch-1 kernel. The exact
same CUDA graph on the physical desktop shows the proxy error growing with
`M`:

| M | A6000 CUDA / physical 4060 CUDA | A6000 latency error |
| ---: | ---: | ---: |
| 1 | 1.018x | +1.83% |
| 2 | 1.056x | +5.59% |
| 4 | 1.571x | +57.08% |
| 8 | 2.449x | +144.92% |
| 16 | 1.777x | +77.74% |

Therefore the live A6000 split cannot be relabeled as a physical 4060
treatment for batching. To bound the real desktop case, the physical 4060 ran
the exact prefix and suffix graphs on two CUDA contexts. Each run injected the
median phone-ready delay measured in the matching live A6000 plus OP15 run,
then uploaded the activation, launched the suffix, synchronized both CUDA
contexts, downloaded both residuals, and merged them. Three 500-iteration
repetitions were used. This is a composed timing experiment: CUDA is physical
and the delay is measured, but USB traffic is not live on the 4060 host.

| M | physical 4060 CUDA | delayed dual-path | modeled change |
| ---: | ---: | ---: | ---: |
| 1 | 1.061558 ms | 0.931111 ms | -12.29% |
| 2 | 1.071253 | 0.983358 | -8.20% |
| 4 | 1.085832 | 1.031239 | -5.03% |
| 8 | 1.122377 | 1.198862 | +6.81% |
| 16 | 1.187785 | 2.881097 | +142.56% |

The composed 4060 result agrees on the policy boundary, not on the exact
speedup: use the phone for `M<=4`, and keep `M>=8` entirely on CUDA. The
batch-1 composed result is optimistic relative to the live 9.79% result,
which quantifies the remaining cost omitted by delay injection.

This does not establish full-model prefill. It covers one synthetic Qwen3-14B
FFN layer and excludes attention, KV-cache traffic, layer-to-layer scheduling,
and phone energy. The next bounded test is direct OP15 AOA on the physical
4060 Ti for `M=2` and `M=4`, with paired latency and energy. BurstGPT or a
full-prefill campaign should wait for that physical treatment.

Evidence root:
`results/causal_batch_v1/run_20260729T043608Z/`. `ANALYSIS.json` contains
the independent aggregation and explicit claim limits. The manifest contains
106 files and has SHA256
`9ce2e0b0f5699458df5004d975dae33df2ea4cd4d8652649c859a920eed6613b`.

## 2026-07-29 complete FFN residual return

The next bounded route removes the late CUDA down projection. OP15 holds the
gate, up, and down weights for one FFN column slice and executes the complete
`gate -> up -> SiLU -> multiply -> down` contribution in one HTP graph. It
returns one K-wide f16 residual. CUDA concurrently executes the complementary
columns, downloads its residual, and the host adds the two K-wide vectors.
The input and result are fixed-size with respect to the number of offloaded
columns.

Both sides use byte-identical deterministic q8_0 weights. The request binds
the residual-output mode, dimensions, input hash, and phone weight hash. The
gate checks the local CUDA partition, phone contribution, final sum, finite
values, and exact argmax.

Three fresh workers and 300 iterations per worker give:

| shape | phone columns | CUDA-only | phone total | split total | change | CUDA/split p90 | final rel L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-14B | 1,792 / 17,408 | 1.080500 ms | 0.711600 ms | 0.999154 ms | -7.541% | 1.082154 / 1.003231 ms | 0.004665 |
| Gemma-4-12B | 1,792 / 15,360 | 0.769121 | 0.615505 | 0.695068 | -9.640% | 0.770183 / 0.704646 | 0.004992 |

Values are medians across the three process-level medians. Every repetition
beats CUDA-only at both median and p90. The phone is below the CUDA prefix
deadline in 99.0% of Qwen requests and 93.7% of Gemma requests at the median
across repetitions.

The numerical margin limits the split. Gemma is only about `7.5e-6` below the
frozen 0.005 final relative-L2 limit, so a larger cut is not eligible without
a stronger precision scheme. An attempted zero-input HTP prewarm is excluded:
the backend captured stale first-input data and returned invalid output. The
final worker does no such prewarm and the first real request has a configurable
AOA timeout for graph preparation.

This supersedes neither the older activation-return energy result nor its
batch sweep. No energy was acquired for the complete-residual graph.

Evidence root:
`results/complete_residual_v1/run_20260729T053106Z/`.
`ANALYSIS_FINAL.json` names the included repetitions and excluded diagnostic.
The final manifest SHA256 is
`f0817af57fadf77a2849459c663f5226518df9ae89b519af8a3c173c1fad0361`.

## 2026-07-29 sharded vocabulary head

The second route targets the output head, where a small hidden vector drives a
large vocabulary matmul but the useful result can be reduced before transfer.
CUDA holds the vocabulary prefix while OP15 HTP holds a disjoint suffix.
CUDA performs device-side argmax and score gather. The phone computes its
suffix logits, performs a local ARM top-1 scan, and returns only the global
token ID and score, eight payload bytes. The host compares the two candidates.
It never transfers the phone logit vector.

Three fresh 300-iteration workers per model give:

| shape | phone vocabulary rows | CUDA-only | phone total | split total | change | CUDA/split p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-14B | 25,600 / 151,936 | 3.205491 ms | 2.649331 ms | 2.689107 ms | -16.104% | 3.208247 / 2.828496 ms |
| Gemma-4-12B | 46,080 / 262,144 | 4.374291 | 3.558601 | 3.625239 | -17.124% | 4.377007 / 3.691367 |

Every repetition beats CUDA-only at median and p90. The phone-local token
matches the CUDA suffix oracle and the merged token matches monolithic CUDA.
Phone score relative error is 0.000649 for Qwen and 0.000966 for Gemma.

This is the strongest isolated operator result because arithmetic grows with
the vocabulary slice while the phone response remains eight bytes. It
implements greedy top-1 only. Sampling, top-k candidate sets, tied logits,
and full decoder integration remain untested.

Evidence root:
`results/head_split_v1/run_20260729T054500Z/`.
`ANALYSIS_FINAL.json` contains the independent process aggregates. The final
manifest SHA256 is
`a97413a79c04ab7c0d95c087a22ef8852696dd9633c1ae4ad62078bb98b993de`.

## 2026-07-29 one Qwen GQA-group offload

The third route assigns one complete Qwen GQA group, five query heads plus one
KV head, to OP15. CUDA runs the other seven groups. Each side executes q8_0
Q/K/V projections, writes the projected K and V into the last f16 KV-cache
slot, runs flash attention over its resident cache slice, and applies its
output-projection columns. The phone returns one K-wide f16 partial residual
for a host sum.

The protocol requires the `LAST_SLOT_UPDATE` flag, preventing an older
static-cache worker from being paired with this host. Correctness compares
monolithic CUDA, a local 7+1 CUDA partition, the phone group, and the final
sum. All runs have exact final argmax, zero non-finite values, local partition
relative L2 near `1.1e-7`, and final relative L2 from 0.00247 to 0.00272.

| KV length | repetitions | CUDA-only | phone total | split total | median change | CUDA/split p90 | interpretation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 512 | 1 | 0.406633 ms | 0.451790 ms | 0.480715 ms | +18.218% | 0.408567 / 0.725006 ms | reject |
| 2,048 | 1 | 0.491757 | 0.469033 | 0.493540 | +0.363% | 0.493991 / 0.593202 | reject |
| 4,096 | 3 | 0.610145 | 0.553706 | 0.589945 | -3.296% | 0.614313 / 0.679709 | median diagnostic only |
| 8,192 | 3 | 0.824137 | 0.639180 | 0.745125 | -9.567% | 0.829457 / 0.768259 | prototype eligible |

At 8,192 entries, all three repetitions improve median and p90, and the phone
leg is hidden behind CUDA in 92.0% to 97.7% of iterations. At 4,096 entries,
all medians improve but every p90 regresses. The scheduler boundary for this
prototype is therefore 8,192 entries, not 4,096.

This remains an attention operator probe. It does not yet implement Q/K
normalization, RoPE, a causal mask, rolling cache positions, or a full decoder
layer. The A6000 at 240/5001 MHz has not been calibrated against the physical
4060 Ti for the head or attention kernels. These results therefore establish
mechanism and a proxy-host crossover, not a physical desktop speedup.

Evidence root:
`results/attention_group_stateful_v1/run_20260729T061110Z/`.
`ANALYSIS.json` is reproduced by `analyze_attention_group_v1.py`. The
evidence-manifest SHA256 is
`fbc734e96beb5d2fdef8752c8b074268074cc34487493a53ff01f2ed836a4f64`.

## Combined verdict

All three bounded mechanisms now execute concurrently on real CUDA and OP15
HTP over direct AOA and pass their path-matched correctness gates. The
vocabulary head is the best first core-integration target because it has one
graph boundary per generated token, a constant eight-byte result, and the
largest repeated latency margin. Complete FFN residual return is second. GQA
offload should remain disabled unless the real context is at least 8,192 and
the missing production attention semantics pass.

Do not add the isolated percentages together. No run executes these routes in
one full model, and no new route has synchronized phone or server energy.
Physical RTX 4060 Ti treatment, full-model continuous batching, BurstGPT,
multi-phone scheduling, and fleet energy are still open.

Question: split ONE matmul so the server takes most of it and the phones take
a share matched to their capacity, all three finishing together. How fast?

## Method

Real execution on all three devices - no simulation, no sleeps.

- `tp_worker.cpp` (Android, NDK r27c, ggml OpenCL): holds a real `[K,N_slice]`
  q8_0 weight slice resident on the phone GPU (quantized via
  `ggml_quantize_chunk`), serves `recv K f32 -> ggml_mul_mat -> send N_slice
  f32` over an adb-forwarded USB socket, TCP_NODELAY both ends.
- `tp_host.cpp` (ggml CUDA on the A6000 the phones are physically attached
  to): issues both phone sends first so their transfer overlaps GPU compute,
  computes its own slice, joins. Median of 40-60 iterations after warmup.
- Split solves `n_gpu/r_gpu = n_i/r_i + W_i` so all three land together.
- Constraint found: the Adreno 750 q8_0 kernel asserts `M % 4 == 0`, so all
  row counts are multiples of 4.

## Measured device rates (K=3840, q8_0, M=1)

| device | raw compute | **end-to-end incl. both transfers** | vs A6000 |
| --- | ---: | ---: | ---: |
| RTX A6000 | 163.9 rows/us | 163.9 | 1x |
| OP15 Adreno 840 | 16.25 | **7.91** | 21x slower |
| OP12 Adreno 750 | 13.89 | **4.88** | 34x slower |

The gap between raw and end-to-end is the finding that drives everything
else: the RETURN payload scales with assigned rows (n x 4 bytes), so the
wire is NOT a fixed cost. Giving a phone more work makes its transfer more
expensive, which halves OP15's effective rate (16.25 -> 7.91) and cuts
OP12's by 65%.

## Result 1 - Gemma-4-12B Q8 at its real matmul size: NO VALID SPLIT

`gate`/`up` is the largest matmul in the model: N=15360, K=3840, 59 MB.

```
GPU-only     :    0.117 ms   (130.8 rows/us)
NO VALID SPLIT: balance needs n_gpu=26690 > N=15360
```

The A6000 finishes the entire matmul in 117 us. A phone round trip is
4-5 ms. Phones would arrive ~40x late holding ZERO rows. No split exists at
any share.

## Result 2 - size sweep, pinned phone slices (real runs)

| N rows | matmul size | GPU alone | 3-device wall | phone :18600 | phone :18601 | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100,000 | 0.39 GB | 0.634 ms | 6.960 ms | 4.900 | 6.832 | 0.091x |
| 163,840 | 0.63 GB | 1.025 | 6.713 | 4.911 | 6.486 | 0.153x |
| 245,760 | 0.94 GB | 1.524 | 6.695 | 4.789 | 6.551 | 0.228x |
| 327,680 | 1.26 GB | 2.023 | 6.722 | 4.880 | 6.609 | 0.301x |
| 458,752 | 1.76 GB | 2.799 | 6.905 | 4.887 | 6.778 | 0.405x |
| 655,360 | 2.52 GB | 4.001 | 6.775 | 4.929 | 6.663 | 0.591x |

## Result 3 - rebalanced with MEASURED effective rates

| N rows | GPU alone | predicted | 3-device wall | speedup |
| ---: | ---: | ---: | ---: | ---: |
| 655,360 | 3.996 ms | 3.743 | 4.425 | 0.903x |
| 983,040 (first balance) | 5.976 | 5.723 | 5.746 | **1.040x** |
| 983,040 (refined) | 6.023 | 5.625 | **5.630** | **1.070x** |

Final point: GPU 918,008 rows (93.4%), OP15 40,220 (4.1%), OP12 24,812
(2.5%); phones returned at 4.919 / 5.367 ms against a 5.630 ms wall - all
three finish together as designed. Predicted 5.625 vs measured 5.630 ms:
the balance model is accurate to 0.1% once the effective rates are used.

## Verdict

- Crossover is at **N ~ 900,000 rows (~3.5 GB matmul)**. Gemma-12B Q8's
  largest matmul is 15,360 rows / 59 MB - **60x too small**.
- Best measured speedup **1.070x**; analytic ceiling `1 + (r15+r12)/r_gpu`
  = **1.078x**. Two phones can contribute at most ~7% to an A6000 matmul,
  and only for matmuls no real 12B model contains.
- A faster server makes this WORSE: the same phones against the 4060 Ti
  (65 rows/us) would break even sooner but still need ~350k rows.
- The phones do the arithmetic fine (raw rates are 10-12% of the A6000 per
  device). What kills operator-level collaboration is that the answer has
  to travel back, and the return grows with the work assigned.

## Per-call cost decomposition and optimization attempts

The 1,221 us fixed cost was decomposed by serving a 4-row slice (compute
~0, so everything measured IS the fixed cost), timing each stage inside the
worker plus the host-observed round trip:

| stage | 4 rows | 25,672 rows | nature |
| --- | ---: | ---: | --- |
| `ggml_backend_tensor_set` | 6 us | 7 us | fixed |
| `graph_compute` -> enqueue | 69-91 us | 126 us | 110 us + 26.5 us/kernel |
| `graph_compute` -> **wait** | **505-510 us** | 724 us | **constant** |
| `ggml_backend_tensor_get` | 3 us | 13 us | ~fixed |
| socket send | 42 us | 178 us | variable |
| host round trip total | **1,221 us** | 3,790 us | |
| => adb/USB portion | 584 us | 1,385 us | matches standalone echo (0.58 ms) |

Graph-size sweep (4-row matmul duplicated N times in ONE graph) proves the
wait is dispatch latency, not execution:

| kernels/graph | enqueue | wait | total | per kernel |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 91 | 510 | 601 | 601 us |
| 2 | 118 | 511 | 629 | 315 |
| 4 | 208 | 532 | 740 | 185 |
| 16 | 661 | 699 | 1,360 | 85 |
| 64 | 1,776 | 526 | 2,302 | **36** |

`wait` is ~515 us whether the graph holds 1 or 64 kernels: it is the queue
flush -> KGSL -> GPU schedule -> fence round trip, NOT compute.

### Spin-poll synchronize: implemented, NO measurable gain

Added `GGML_OPENCL_SPIN_SYNC=1` to `ggml_backend_opencl_synchronize`
(ggml-opencl.cpp): `clFlush` + busy-poll `clGetEventInfo` instead of
blocking in `clWaitForEvents`.

| config | wait | host RTT |
| --- | ---: | ---: |
| blocking, 1 kernel | 504 us | 1.223 ms |
| **spin-poll, 1 kernel** | **501 us** | 1.234 ms |
| blocking, 2 kernels | 514 us | 1.269 ms |
| **spin-poll, 2 kernels** | **507 us** | 1.254 ms |

0.6% - inside noise. HYPOTHESIS DISPROVED: the 515 us is not the host
fence-wakeup path, so removing the block recovers nothing. The cost is on
the GPU side of the driver boundary.

### What DID work: batching kernels per graph

2 kernels in one call cost 1.269 ms vs 2 x 1.223 = 2.446 ms as separate
calls - **1.93x**. This is the only optimization that moved the number, and
it is bounded by how many ops are genuinely independent: only `{q,k,v}`
(8,192 rows) and `{gate,up}` (30,720 rows) per Gemma layer.

### Decisive test: `{gate,up}` with everything applied

A6000, N=30,720, spin-sync on, 2 kernels in one graph, persistent socket:

```
GPU-only     :   0.210 ms
NO VALID SPLIT: balance needs n_gpu=40999 > N=30720
forced 2048-row phone slice -> wall 1.660 ms, SPEEDUP 0.127x
```

The A6000 computes the entire largest batchable group in 210 us; the phone
needs 1,581 us to return 2,048 of its 30,720 rows. Optimized operator-level
collaboration is still **7.9x slower** than the GPU working alone.

## AOA transport: WORKS, 3.8x faster than adb

`/dev/usb_accessory` is root:usb 0660 and shell is not in the `usb` group -
but OP15 has root via Magisk, so a native daemon (`aoa_daemon.c`) opens the
node directly and NO APK is needed. Host side is libusb via ctypes
(`aoa_bench.py`), no -dev headers; the existing udev rule already grants
plugdev MODE=0666 on VID 18d1. OP15 re-enumerated as **18d1:2d01
(accessory + ADB)** so adb survived throughout - no replug needed.

| exchange | adb forward | AOA bulk | speedup |
| --- | ---: | ---: | ---: |
| q8_0 act 4080B->1024B | 0.525 ms | **0.137 ms** | 3.83x |
| f16 act 7680B->1024B | 0.530 | **0.149** | 3.56x |
| f32 act 15360B->2048B | 0.560 | **0.171** | 3.27x |
| 64 KB bulk | 1.398 | **0.609** | 2.30x |
| 1 MiB (bandwidth) | ~99 MB/s | 119 MB/s | 1.2x |

Latency improves 3.3-3.8x; bandwidth barely moves - confirming again the adb
cost was per-transaction, not per-byte. Endpoints: interface 0, EP 0x01 OUT /
0x81 IN, wMaxPacketSize 1024.

W0 with AOA = 0.171 + 0.515 (GPU) + 0.05 = **0.736 ms** (was 1.090). Operator
split still fails: A6000 does `{gate,up}` in 0.210 ms => 3.5x short (4060 Ti
0.472 ms => 1.6x short). The GPU-dispatch half is now 70% of W0.

## GPU power state: the real bottleneck for SPARSE traffic

Clock pinning DOES NOT WORK on Adreno 840. `min_pwrlevel`/`max_pwrlevel`
accept writes and read back as set, but the GPU still idles at 160-222 MHz;
`force_clk_on`/`force_rail_on`/`force_no_nap` refuse to take at all. The GMU
firmware owns DVFS on 8xx-class parts and these legacy KGSL sysfs knobs are
vestigial. Measured effect of pinning: none.

| condition | host RTT | q->submit | submit->start | exec |
| --- | ---: | ---: | ---: | ---: |
| default power | 1.184 ms | 106.1 us | 337.8 us | 14.0 |
| "pinned" pwrlevel=0 | 1.186 ms | 101.3 us | 343.4 us | 14.0 |

What DOES matter is **power collapse**, governed by `idle_timer` (default
80 ms). Latency vs inter-request gap:

| gap | median RTT |
| ---: | ---: |
| 0 ms | 0.970 ms |
| 5 ms | 1.430 |
| 20 ms | 1.661 |
| 40 ms | 3.356 |
| 60 ms | 3.126 |
| **80 ms** | **23.525** |
| 100 ms | 17.199 |
| 500 ms | 18.726 |

A 17.7x cliff exactly at the `idle_timer` boundary. Two fixes, both measured:

| fix | gap=100ms | gap=500ms | cost |
| --- | ---: | ---: | --- |
| none | 17.199 ms | 18.726 ms | - |
| app keep-alive (ping every 20 ms) | 1.743 ms (9.9x) | - | wasted GPU work + traffic |
| **`echo 3000 > idle_timer` (root)** | **2.021 ms (8.5x)** | **4.118 ms (4.5x)** | one sysfs write, no waste |

This is THE actionable finding for phone offload: any deployment where phone
requests arrive more than ~80 ms apart pays 16-23 ms of GPU wake per request.
Raising `idle_timer` is a single root write and removes almost all of it.
It does NOT help the operator-split benchmark (which runs back-to-back and is
already warm), but it directly targets realistic sparse serving traffic.

Devices restored: `idle_timer=80`, pwrlevels default, AOA mode exited, no
stray daemons, no forwards. OP15 `sys.usb.config` reads `adb` (was `ptp,adb`;
`ptp` is media-transfer only and resets on reboot).

## Caveats

- Two phones only; scaling to N phones raises the ceiling to
  `1 + sum(r_i)/r_gpu` but every phone adds return traffic on the same
  host, and the 2-bus fan-in penalty (measured earlier: 0.58 -> 2.32 ms at
  8 phones) is not in this model.
- M=1 (decode GEMV) only. Batched M would grow both payload directions.
- Single session; phone DVFS uncontrolled.
- The host's f32 staging buffer limits N; N=983,040 was the largest run.

## Cleanup

Workers killed, adb forwards removed, both A6000s idle (111/57 MiB, 0%).
Binaries left at `/data/local/tmp/tp_slice/tp_worker` on both phones.
