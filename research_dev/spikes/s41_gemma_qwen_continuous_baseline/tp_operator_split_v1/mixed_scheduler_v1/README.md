# Mixed workload scheduler v1

This spike treats placement as three decisions with different lifetimes:

1. Task placement selects a resident server route for a whole request. A
   request already admitted to a server is not migrated, so its KV state stays
   local to that server.
2. Layer placement selects a contiguous GPU suffix when a server route is
   created. It is used for capacity, not changed once per token.
3. Operator placement selects a calibrated split for each graph shape inside
   a CPU route. The integrated phone route covers qualified Gemma Q4_0 and F16
   gated-FFN slices.

The implementation is a research scheduler around llama-server. It does not
change upstream llama.cpp placement or claim transparent KV migration.

## Policy order

| Gate | Decision |
| --- | --- |
| Feasibility | Reject a route if weights, KV, compute buffers, or reserve do not fit. |
| Priority | Keep a protected, deadline-sensitive model on its resident GPU route. |
| Admission | Do not enqueue cold work behind a server if another route will become ready first. |
| Promotion | Switch GPU models only when load, warm-up, switch-back, and guard time are amortized by the queued work. |
| Layers | Use the fastest qualified layer count within remaining VRAM and the protected-workload slowdown budget. |
| Operators | Use only exact-shape, resident-weight phone policies with measured p50 and tail behavior. |
| Feedback | Replace predicted route times with measured completion times; disable a route after correctness, reset, memory, or QoS failures. |

The default objective is lexicographic: protected deadline misses, weighted
tardiness, makespan, weighted completion time, then energy. Throughput-only
experiments can select makespan first.

## Current 4060 Ti plus OP15 decision

Qwen3 14B occupies 13,729,005,568 bytes of the 17,175,674,880-byte GPU during
the source BurstGPT trace. Only 2,993,684,480 bytes remain. Fully resident
Gemma4 12B reaches 11,512,315,904 bytes, so both full routes cannot coexist.

While Qwen has protected work, the scheduler uses:

| Scope | Current route | Reason |
| --- | --- | --- |
| Qwen requests | full CUDA | It is the protected hot workload. |
| Gemma tasks predicted to finish early | CPU plus OP15 | Independent requests can overlap without moving KV state. |
| Other Gemma tasks | scheduler hold queue | Avoid burying them in the CPU server before GPU promotion. |
| Gemma gated FFN on CPU route | dynamic OP15 HTP split | Qualified Q4_0 DMA-BUF route. |
| Other Gemma operators | CPU | No integrated winning phone route yet. |
| Any phone split from a CUDA route | CUDA | CUDA-to-phone staging is not a qualified win. |
| Gemma GPU layers while Qwen is active | disabled | Host KV fails QoS; Q8 and FP16 GPU KV fit only slow 24/49- and 14/49-layer Gemma routes. |

After the Qwen queue drains and the idle-window guard passes, the scheduler
unloads Qwen, loads full Gemma on CUDA, warms the route, and releases held
Gemma requests. Whole-request routing avoids KV migration.

The qualified Gemma FFN table is:

```text
M <= 1:    9664 HTP columns
M <= 2:    8192 HTP columns
M <= 4:    6144 HTP columns
M <= 128:  8192 HTP columns
M <= 512: 11136 HTP columns
```

## FP16 oversized-model result

The FP16-storage route uses Qwen 18/41 CUDA loads and Gemma 25/49 CUDA loads.
After Qwen drains, Gemma layers 0 through 22 remain on the CPU, layers 23
through 47 run on CUDA, and OP15 executes 6,144 of 15,360 FFN columns in
parallel with each eligible CPU layer for `M <= 16`.

| Cold Gemma route | BurstGPT makespan | Output rate |
| --- | ---: | ---: |
| CUDA + CPU + OP15 | 712.861 s | 9.706 tok/s |
| CUDA + CPU control mean, 3 runs | 779.834 s | 8.872 tok/s |

OP15 reduces steady cold execution by 8.59% and raises throughput by 9.39%.
The phone-compatible CPU weight layout adds about 29 seconds to model load and
warm-up, so the reconstructed full-trace gain is 1.42%. The measured break-even
is approximately 338 seconds of no-phone cold work.

These measurements use F16-storage performance proxies dequantized from the
existing Q4 values. They qualify placement and runtime, not original-checkpoint
FP16 quality. See `../RESULTS_FP16_MIXED_SCHEDULER_OP15_V1.md` for the complete
method, transport breakdown, and raw evidence paths.

## Files

- `hierarchical_scheduler.py`: task assignment, promotion guard, layer capacity
  planner, operator gate, and lower-bound helpers.
- `run_gpu_cold_trace.py`: CUDA or CUDA+CPU Gemma backlog control profiler.
- `run_hierarchical_trace.py`: physical Qwen-to-Gemma task switch, optionally
  with selected requests on CPU plus OP15 or the qualified FP16 switch route.
- `run_dual_cuda_trace.py`: dual-residency capacity, compressed-KV, and QoS
  probe with independent hot and cold KV policies.
- `analyze_burstgpt.py`: measured profile reconstruction and assignment oracle.
- `current_profile.json`: current machine, route, and operator measurements.
- `test_hierarchical_scheduler.py`: deterministic policy tests.

## Tests

```sh
cd research_dev/spikes/s41_gemma_qwen_continuous_baseline/tp_operator_split_v1/mixed_scheduler_v1
python3 -m unittest -v test_hierarchical_scheduler.py
```

Physical runners require their explicit confirmation strings and an absolute,
new output directory. The OP15 mode also requires the existing FunctionFS
phone session to be active before the host bridge starts.
