# Pixel TPU execution path and latency audit

2026-09-24 UTC. Source and compiled-graph audit PASS. Vendor metrics collection
and numerical checks PASS, 48 additional FFN calls / 120 rows. Exact memory
stall, array utilization, internal instruction, and per-matmul attribution
remain NOT VERIFIED.

## What LiteRT executes

The actual full-width B1 and B4 compiled files each contain one subgraph and
one `DISPATCH_OP`, with only FP32 input/output tensors visible at the LiteRT
level. The original six operations are compiled into that dispatch payload.
One dispatch partition does not imply one hardware instruction or one fused
matrix kernel. Inspection is retained in
[COMPILED_GRAPH.json](software/pixel10pro-tpu-sdk/runtime_audit/COMPILED_GRAPH.json).

```text
Offline: real Qwen weights + six-op FFN -> Tensor SDK -> TPU executable
Startup: LiteRT creates compiled model, vendor executable and I/O buffers
Per call: input copy -> LiteRT -> Google dispatch -> SouthBound runtime
          -> submit TPU graph -> wait for completion -> output copy
```

The probe selects NPU only. Its buffers are persistent AHardwareBuffers, and
the same compiled model is reused for all requests. The AOT compilation time
and model-loading interval are outside the warm invocation timer. The process
loads `libLiteRtDispatch_GoogleTensor.so`, which resolves the vendor runtime
from `libedgetpu_litert.so`. There is no observed LiteRT CPU fallback.

The pinned v2.2.0 dispatch implementation creates an NPU node when loading
an ML executable. Its synchronous `Invoke()` calls prepare, invoke-once and
wait on the existing invocation context. Executable loading happens during
context creation, not inside that invocation method. The runtime also offers
an asynchronous API with output fences. Using it can overlap independent
work; it does not remove the time required to produce the result.

The shared-memory interface does not make this probe end-to-end zero-copy:
the probe explicitly locks/copies/unlocks each input and output buffer.
AHardwareBuffer locking may also involve synchronization/cache maintenance.
No measurement here separates those costs from the probe's memcpy calls.

Primary sources: [dispatch overview](https://developers.google.com/edge/litert/next/dispatch),
[invocation context](https://github.com/google-ai-edge/LiteRT/blob/v2.2.0/litert/vendors/google_tensor/dispatch/litert_dispatch_invocation_context.cc),
[device context](https://github.com/google-ai-edge/LiteRT/blob/v2.2.0/litert/vendors/google_tensor/dispatch/litert_dispatch_device_context.cc),
[dispatch delegate](https://github.com/google-ai-edge/LiteRT/blob/v2.2.0/litert/runtime/dispatch/dispatch_delegate_kernel.cc).
Downloaded source hashes and URLs are retained in
[DOWNLOADS.json](software/pixel10pro-tpu-sdk/runtime_audit/DOWNLOADS.json).

## New physical counters

An isolated derivative of the existing probe starts
`LiteRtCompiledModelStartMetricsCollection(..., 1)` after eight warmup calls
and stops it after sixteen measured calls. No compiler, precision or runtime
performance hint changed. Both arms pass all 24 numerical checks and repeat
exactly for identical inputs. Maximum relative L2 is 0.000547222.

The vendor returns `hardware_execution_time_us` and
`number_of_graph_executions`. Counts equal the sixteen measured calls.
No DRAM bytes, active MAC cycles or individual gate/up/down timings were
returned at this detail level.

All entries below are means per invocation, in milliseconds. These are two
short instrumented samples; retain the original longer uninstrumented runs
for general latency comparisons.

| Component | B1 | B4 |
| --- | ---: | ---: |
| Vendor-reported hardware execution | 27.337 | 27.419 |
| Invocation time outside the hardware counter | 5.680 | 5.625 |
| Probe buffer handling | 1.221 | 3.873 |
| Outside the phone worker: USB/ADB, sockets and host scheduling | 6.566 | 12.862 |
| Total round trip | 40.805 | 49.779 |

B1 hardware total is 437395 us; B4 is 438707 us. Dividing each by sixteen
produces the first row. The second row subtracts the hardware average from
the wall-clock average of `LiteRtRunCompiledModel`. It includes any runtime,
driver, queue and synchronization costs outside the vendor counter; it is
not a measurement of LiteRT framework overhead alone. The hardware counter
is vendor-defined and is not independently calibrated into pure compute
versus data movement or stalls.

Evidence: [B1 raw run](physical/pixel10pro-tpu-runtime-audit-1/run-metrics-1/RESULT.json),
[B4 raw run](physical/pixel10pro-tpu-runtime-audit-1/run-metrics-b4-1/RESULT.json),
[derived breakdown](PIXEL_TPU_RUNTIME_AUDIT.json).

## Why these shapes are difficult

The three dense matrices contain `3 * 5120 * 17408 = 267386880` parameters,
or 534773760 bytes in FP16. Their useful arithmetic is 534773760 FLOPs per
row, counting multiply-add as two FLOPs and excluding the small activation
operations. If each weight is read once, arithmetic intensity is about
one FLOP/byte at B1 and four FLOPs/byte at B4. Having many parameters does
not give a one-row matrix-vector operation much reuse of each parameter.

The almost unchanged hardware time for four times the arithmetic is
consistent with weight movement and/or fixed tile work dominating. A
partially filled matrix tile can also cost the same at B1 and B4. These
measurements cannot uniquely distinguish those causes. They do establish
that USB and the host-side LiteRT wrapper are not the only major costs.

Nominal one-pass weight rates from the hardware counter are 19.56 GB/s at
B1 and 19.50 GB/s at B4; useful matrix rates are 19.56 and 78.01 GFLOP/s.
Those are ratios of known model work to elapsed time, not measured DRAM
traffic, hardware utilization or peak throughput. Earlier F16 CPU results
were about 18.3 ms, but this audit contains no fresh matched CPU arm.

A TPU is specialized for matrix multiply-accumulate. Tiling and local
operand reuse let many arithmetic circuits work concurrently. The published
Cloud TPU architecture illustrates this with systolic arrays and vector
units. Its array sizes, HBM specifications and TOPS must not be assigned to
Pixel Tensor G5. The current SDK does not expose the G5 array geometry,
on-chip memory capacity or generated instruction schedule needed to derive
utilization for these particular shapes.
[Conceptual TPU architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm).

## Remaining tuning controls

- The tested compiler setting is `google_tensor_sharding_intensity=minimal`.
  Google documents moderate/extensive/maximum alternatives for more
  aggressive parallel processing. Their benefit for this FFN is untested.
- The probe sets no Google Tensor runtime performance mode. The selected
  vendor default and actual clocks are unknown. The v2.2.0 API exposes
  performance hints, but applying the annotation requires a supported
  SouthBound runtime (the source checks version >=0.18). No maximum-clock
  or high-performance claim is supported by the current runs.
- FP16 and BF16 were measured previously. BF16 provided no speed advantage
  and increased numerical error. Precision reduction alone did not fix this
  graph's latency.
- F16 wire I/O, direct buffer use and asynchronous execution could reduce
  overhead or enable server overlap. They cannot by themselves remove the
  observed 27 ms vendor hardware interval.

Sources: [compiler flags](https://developers.google.com/edge/tensor-sdk/compilation-flags),
[runtime performance annotation](https://github.com/google-ai-edge/LiteRT/blob/v2.2.0/litert/vendors/google_tensor/dispatch/litert_dispatch_graph.cc).
These are next experiments, not measured speedups.

Cleanup PASS: unchanged boot, no worker or forward, Pixel lock free, no
pending job. No phone reboot, forced kill, clock change or shared server
mutation. Build uses the existing NDK command with `-Wall -Wextra -Werror`.
Full-model tokens, energy and production integration remain unverified.
