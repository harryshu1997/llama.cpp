# S41 Gemma-Qwen server-only baseline results

Status: `SERVER_ONLY_BASELINE_ACQUIRED_AND_GRAPHED; REPLICATION_INCOMPLETE`

## Scope

These are all-server measurements. No phone served a request or contributed
compute. The exact model pair is Gemma 4 12B IT Q8_0 and Qwen3 14B Q4_K_M.
The server is the target RTX 4060 Ti desktop. Every route uses eight slots,
continuous batching, context 4096, batch 2048, ubatch 512, F16 KV, and eight
output tokens per request. C1 has one eight-slot executor at a time; C2 has
two simultaneous eight-slot executors, or 16 aggregate slots. C2 is therefore
a dual-ready deployment control, not a slot-matched control.

The versioned trace contains 74 requests: 57 Gemma requests and 17 Qwen
requests, nine target changes, and a 30 second synthetic SLO. The request
manifest is
`8a3d2b59f1de9f4d9a4b64b27cf044977a1a9ddeed87bc848d992da01e575df0`.
This trace is `SYNTHETIC_GEOMETRY_ONLY_NOT_TASK_QUALITY`: prompts are 16 to
128 tokens, 60/74 source input lengths were transformed, and every source
output was replaced by a forced eight-token `ignore_eos` continuation. It
cannot support task-quality or broad service-capacity claims.

## Qualification

Both artifacts passed independent full-CUDA B8 service and reported at least
512 MiB free VRAM at readiness. This was a readiness sample, not continuous
in-cohort VRAM measurement:

- Gemma: 49/49 layers on CUDA, 1.073 seconds for the B8 cohort, and about
  1.91 GiB free VRAM.
- Qwen: 41/41 layers on CUDA, 0.988 seconds for the B8 cohort, and about
  6.40 GiB free VRAM.

Physical simultaneous-load attempts failed in both orders while the first
model remained healthy. The two full models therefore cannot be full-CUDA
co-resident on this 16 GiB GPU under the frozen serving envelope.

## Trace results

Values are medians where more than one valid repetition exists. Throughput is
Gemma plus Qwen output tokens per second.

| Server-only mode | Valid n | Completed | SLO goodput | Throughput | P95 TTFT, completed only | P95 completion, completed only | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| C1 GPU switch, warm host cache | 2 | 74/74 | 1.227 req/s | 9.818 tok/s | 5.097 s | 5.360 s | PASS |
| C1 GPU switch, cold NVMe | 3 | 17/74 | 0.000 req/s | 1.380 tok/s | 94.454 s | 95.243 s | FAIL: 57 stranded |
| C2 Gemma GPU, Qwen CPU/RAM | 0 | 74/74 | 1.204 req/s | 9.637 tok/s | 7.427 s | 16.698 s | INVALID: swap grew |
| C2 Qwen GPU, Gemma CPU/RAM | 1 | 74/74 | 0.054 req/s | 1.898 tok/s | 254.575 s | 258.536 s | PASS |

The warm C1 runs split throughput into 7.563 Gemma and 2.255 Qwen output
tokens/s. Their maximum model-publication gap was 3.608 seconds. A third warm
execution completed all requests but was excluded because system swap grew by
8 KiB.

All three cold C1 runs stranded the same 57 Gemma requests. This is a failed
server baseline, not a lower-throughput success. Its finite P95 values are
conditioned on the 17 completed Qwen requests and censor the 57 stranded
requests.

The Gemma-GPU dual-ready placement completed the trace with throughput close
to warm C1, but its completion P95 was 3.12x worse and system swap grew by
95,559,680 bytes. It remains plotted with a red `swap gate failed` label and
is not a valid control. The reverse placement passed the no-additional-swap
gate but made Gemma CPU-bound and missed the SLO for 57/74 requests. That gate
means zero system-swap growth plus zero executor `VmSwap`, not an empty host
swap device: the reverse run began with 114,495,488 bytes of system swap
already used. Both C2 acquisitions fixed the CPU route at 16 threads but did
not bind CPU affinity or NUMA placement, so they remain provisional even apart
from the missing repetitions.

Selected-GPU energy for equal completed work was 6.129 J/output token for
warm C1, 6.002 J/output token for the resource-invalid Gemma-GPU dual route,
and 8.618 J/output token for the passing Qwen-GPU dual route. This is selected
GPU-board energy only. It is not server-wall, CPU, phone, or total-system
energy.

Energy covers only the paid replay interval. It excludes preflight, host-cache
preparation, and initial executor/model startup; C1 replacement loads that
occur inside the trace are included.

The resource figure reports the sampler's selected process RSS. For C2 it
follows the GPU executor and is not the sum of the GPU and CPU server
processes. CPU utilization was not present in these raw samples. Do not use
that figure as a total host-memory or CPU-cost comparison.

Continuous batching was enabled, but no physical batch-size distribution was
captured. These runs demonstrate concurrent request service, not the planned
physical batching characterization.

## Graphs and evidence

- `results/server_baseline_v1_20260726T073202Z/graphs/01_slo_goodput_throughput.{svg,png}`
- `results/server_baseline_v1_20260726T073202Z/graphs/02_tail_latency_publication.{svg,png}`
- `results/server_baseline_v1_20260726T073202Z/graphs/03_timeline_*.{svg,png}`
- `results/server_baseline_v1_20260726T073202Z/graphs/04_selected_gpu_board_energy.{svg,png}`
- `results/server_baseline_v1_20260726T073202Z/graphs/05_server_resources.{svg,png}`
- `results/server_baseline_v1_20260726T073202Z/NORMALIZED_SERVER_BASELINES.json`
- `results/server_gpu_switch_20260726T073202Z/`
- `results/server_dual_ready_20260726T073202Z/`

Every normalized point was independently re-derived from manifest-verified raw
request, lifecycle, resource, and selected-GPU power records. The reducer
reproduced `NORMALIZED_SERVER_BASELINES.json` byte for byte.
The normalized summary SHA-256 is
`60ff1561889d3a9d91057c5825945c0707b24bd07837e5735235d4402437d0cb`;
the 16 graph artifacts are bound by the snapshot's `graphs/SHA256SUMS.txt` with
SHA-256
`9f7ce3e3963c7ef318bf34c74a081f29d809f46e98621f7e300970ddccd1f748`.
The complete v1 snapshot manifest has SHA-256
`a7c57d1e47f165b2b3c6943bac22c35a36f5a3ceb4eb1739085c8a006bead6a5`.

## Claim boundary and next step

This immutable v1 snapshot is a usable pilot/control input for parallel phone
engineering. It is not a paper-final quantitative baseline: warm C1 has two
valid repetitions, Gemma-GPU C2 has zero valid repetitions, and reverse C2 has
one. Before a final quantitative claim, acquire one additional clean warm C1
run and three valid repetitions per C2 placement in a versioned v2 root, with
CPU affinity and NUMA placement bound. Do not overwrite or silently extend
v1.

The next system gate remains bounded: qualify Gemma Q8 on the real OP15 plus
OP12 GPU route without swap, then run the phone warm-tier treatment against
these frozen server-only inputs and metrics. Before that paid phone attempt,
freeze a new S41 Gemma/Qwen authority; the S39 Qwen/Qwen authority cannot
authorize it. The native-router Gemma smoke also exposed a shutdown
stack-smashing failure that must be fixed and requalified before a controller
treatment, although it does not invalidate these direct server-runner traces.

The later real-device T2 no-promotion prototype is reported separately in
`prototype_t2_phone_trace_v1/RESULTS.md`. It completed 74/74 with Gemma on
CUDA and Qwen on both phones, but the phone route met 0/17 SLOs. It must not
be folded into the immutable server-only v1 normalization.
