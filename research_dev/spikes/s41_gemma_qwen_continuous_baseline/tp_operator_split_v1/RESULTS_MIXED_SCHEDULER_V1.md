# Mixed workload scheduler v1

## Result

The hierarchical task-switch policy is the best physically qualified route
for this BurstGPT slice. It keeps Qwen3 14B on the RTX 4060 Ti until the hot
queue drains, holds cold Gemma4 12B requests outside llama-server, then changes
GPU residency once and releases the Gemma backlog.

All 74 requests and all 11,605 requested output tokens completed.

| Route | Makespan | Total output rate | Hot Qwen end | Server device energy |
| --- | ---: | ---: | ---: | ---: |
| Qwen CUDA + Gemma CPU | 733.730 s | 15.816 tok/s | 101.299 s | 136.186 kJ |
| Qwen CUDA + Gemma CPU+OP15 | 651.964 s | 17.800 tok/s | 101.149 s | 101.421 kJ |
| Hierarchical GPU switch | **183.226 s** | **63.337 tok/s** | **100.784 s** | **41.567 kJ** |

Against the prior CPU+OP15 route, the scheduler is 3.558x faster, reduces
makespan by 71.90%, raises total output throughput by 255.82%, and reduces
measured server CPU-package plus GPU-board energy by 59.02%. The phone energy
is not included in the baseline, so this is not whole-system wall-plug energy.

Against CPU without the phone, it is 4.004x faster and reduces measured server
device energy by 69.48%.

The hot route did not regress. Qwen ended 0.366 s earlier than in the static
CPU+OP15 run. Both routes met 55 of 74 trace SLOs; all 17 cold requests miss
the 30 s trace SLO in both cases.

## Critical path

| Phase | Time | Share of makespan | Critical resource |
| --- | ---: | ---: | --- |
| Protected Qwen CUDA | 100.784 s | 55.0% | RTX 4060 Ti compute |
| Unload + load + warm | 7.151 s | 3.9% | CPU/file-cache load and CUDA allocation |
| Promoted Gemma CUDA backlog | 75.292 s | 41.1% | RTX 4060 Ti compute and batching |

The switch consists of 0.316 s Qwen unload, 6.522 s Gemma load, and 0.309 s
Gemma warm-up. During the full run, GPU utilization is 95% at p50 and 100% at
p95. This removes the old 550-second-plus interval where the GPU was idle and
Gemma remained on CPU+OP15.

Cold completion p50 falls from 423.468 s to 121.041 s. Cold TTFT p50 falls
from 110.205 s to 96.093 s. The remaining TTFT floor is mostly intentional
admission hold while Qwen is protected, not USB or phone compute.

## Scheduler decisions

| Level | Decision in this trace | Runtime rule |
| --- | --- | --- |
| Task | Qwen requests stay on full CUDA. Gemma requests are held, then promoted as whole requests. | Minimize protected deadline misses first; promote only when load, warm, switch-back, and guard costs are amortized. |
| Layer | No Gemma GPU layers run while Qwen is active. Full Gemma CUDA is selected after Qwen drains. | Allocate a contiguous GPU suffix only after weights, KV, scratch, reserve, and measured protected slowdown pass. |
| Operator | On a CPU Gemma route, only the gated FFN uses the qualified OP15 HTP split. CUDA routes stay on CUDA. | Select a split only when `max(host branch, transport + phone branch) + merge` beats the unsplit route at p50 and p90. |

The qualified Gemma CPU+OP15 FFN table remains:

| Continuous-batch M | HTP intermediate columns |
| ---: | ---: |
| 1 | 9,664 |
| 2 | 8,192 |
| 3-4 | 6,144 |
| 5-128 | 8,192 |
| 129-512 | 11,136 |

Attention projections, Gemma lm_head, norms, RoPE, softmax, and CUDA-to-phone
splits remain local because no integrated route currently beats their parent
CPU or CUDA path. The phone GPU remains rejected for this FFN family; its
measured complete FFN route is slower than HTP.

## Alternatives tested

### Queue reordering

The fixed-slot model predicted that longest-processing-time-first GPU dispatch
could reduce the Gemma backlog from 71.11 s to 62.07 s. Physical testing did
the opposite: FIFO took 71.838 s and LPT took 72.688 s. Continuous batching
changes graph shapes and service rates, so queue order is treated as a
measured route property. FIFO remains selected.

### Dual CUDA weight residency

Moving both KV caches to CPU leaves 6,519 MiB after Qwen loads. Full Gemma
still cannot allocate its 6,637.69 MiB model buffer. A 48/49 placement has a
6,517.39 MiB model buffer but cannot allocate its 198.5 MiB compute buffer.
A 46/49 placement loads, but only a tiny reserve remains and Qwen fails at
runtime with CUDA out-of-memory.

A 42/49 Gemma placement is stable and leaves about 492 MiB free, but host KV
is unacceptable for the protected workload. After the Qwen server had run for
301 s, only 47 of 57 Qwen requests had completed and active decode slots were
only 0.88-1.34 tok/s. This already exceeded the complete 183.226 s switch
route before Gemma began, so the losing probe was stopped.

### Model prefetch

The Gemma GGUF was already 100% resident in the Linux page cache. Trace
validation also hashes all 6,975,878,176 bytes before admission. Consequently,
an asynchronous file read cannot reduce the measured switch. The remaining
6.522 s load interval is CUDA allocation, tensor upload, and model
initialization rather than storage I/O.

### Compressed-KV dual residency

Independent hot and cold KV policies and trace-safe 16,384-token context
budgets were added to the dual-residency runner. Qwen used GPU Q8 KV. The
maximum placements were stepped down to target roughly 512 MiB of reserve,
then two Gemma placements were measured:

| Gemma placement | Minimum free VRAM | Observed cold work | Result |
| --- | ---: | ---: | --- |
| 31/49 CUDA layers, host KV | 445 MiB | 1,017 tokens / 222.324 s | 4.574 tok/s, reject |
| 24/49 CUDA layers, GPU Q8 KV | 612 MiB | 1,959 tokens / 106.074 s | 18.468 tok/s, reject |

The maximum warm probes were also too tight: 34 host-KV layers left 216 MB,
and 28 GPU-Q8-KV layers left only 34 MB. The safer 24-layer GPU-KV route had
already reached 209.677 s total when stopped, longer than the complete
183.226 s switch route. Its measured rate projects to approximately 478 s.
Even perfect overlap cannot make an 18.468 tok/s cold route beat the switch.

Q8 KV kept 55/57 protected Qwen SLOs but increased Qwen makespan from
100.784 s to 103.603-103.782 s, a 2.80-2.97% slowdown. Across the two runs,
42 and 45 of 57 request token sequences exactly matched the F16-KV control;
sampled divergent responses remained coherent. The scheduler rejects this
route on cold throughput before applying any stricter quality requirement.

### FP16 GPU KV for both models

Both models were then tested with GPU-resident FP16 KV and the same
trace-safe 16,384-token contexts. A 16/49 Gemma placement warmed but left only
294 MiB free. The full trace therefore used 14/49 layers and reached a
547 MiB minimum reserve.

Protected Qwen completed in 100.654 s and retained 55/57 SLOs, matching the
100.784 s control. Gemma produced 1,913 tokens in 160.556 s, or 11.915 tok/s.
The trace was stopped at 261.210 s because it had already exceeded the
complete 183.226 s switch result with most cold output remaining. Its measured
rate projects to approximately 681 s.

FP16 avoids the Q8 cache conversion cost, but its larger KV allocation reduces
Gemma from 24 to 14 safe CUDA layers. The additional CPU layers dominate, so
FP16 is slower than the 18.468 tok/s Q8 dual-resident route. The scheduler
rejects both.

### CPU+OP15 task overlap

The measured-profile assignment solver selects cold requests 42, 46, and 50
for CPU+OP15 and holds the other 14 for CUDA. It improves two modeled 30 s
deadlines but does not reduce modeled makespan: the CUDA branch remains
critical. The full phone-assisted run was not used as the primary result
because the shared phone campaign occupied the FunctionFS endpoint during the
attempt. The existing full static CPU+OP15 result is still the calibration for
this route.

## Optimality

This is not a proof of global optimality. It is the best qualified physical
route tested.

Using the service times from the same physical run, an optimistic eight-slot
packing lower bound is 168.531 s. The measured 183.226 s result is 14.696 s,
or 8.72%, above that bound. The bound assumes ideal batch packing and cannot
necessarily be attained by llama-server.

The remaining measured gap consists of the 7.151 s residency switch plus
batch-shape and tail imbalance. Naive LPT failed physically, page prefetch is
already satisfied, host KV failed QoS, and compressed GPU KV failed cold
throughput. None closes the gap.

Unqualified candidates that could still change the answer are:

- in-process partial-to-full CUDA layer promotion that preserves already
  resident Gemma tensors after Qwen is evicted;
- a preemptible server route with safe request cancellation or KV migration;
- a batch scheduler optimized with measured shape-dependent service curves.

For an online service rather than this finite trace, promotion also needs a
forecast or idle guard. If another protected Qwen request can arrive before
Gemma load plus minimum residency plus Qwen reload, the scheduler must keep
Qwen resident or accept the switch-back penalty.

## Implementation and validation

The implementation is isolated from upstream llama.cpp placement code under
`mixed_scheduler_v1`. It includes task assignment, deadline-aware scoring,
promotion hysteresis, VRAM-aware layer planning, exact operator gates, the
physical trace runners, and the measured profile.

The deterministic policy suite passes 12 of 12 tests. The primary physical
result records all request timings, GPU utilization, process memory, RAPL CPU
package energy, and NVML GPU board energy. No request token-count mismatch,
process swap, or request loss occurred.
