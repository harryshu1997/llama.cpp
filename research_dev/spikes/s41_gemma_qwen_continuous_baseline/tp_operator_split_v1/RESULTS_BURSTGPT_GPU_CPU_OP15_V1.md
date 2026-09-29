# BurstGPT full-model GPU plus CPU/OP15 trace

This is the historical 128-prompt, 8-output, decode-only result. The later
multi-token dynamic operator result is in
`RESULTS_BURSTGPT_DYNAMIC_FFN_OP15_V2.md`.

Date: 2026-08-03 EDT.

Verdict: `FULL_TRACE_PASS; OP15_NO_SPEEDUP; CPU_REPACK_DISABLED_CAUSES_PREFILL_LOSS; TRACE_THROUGHPUT_MINUS_7.69_PERCENT; COLD_SERVICE_PLUS_9.00_PERCENT; COLD_RSS_MINUS_45.94_PERCENT; DECODE_MINUS_35.88_PERCENT; STOCK_RESTORED`.

## Result

The physical RTX 4060 Ti served the hot Qwen3-14B Q4_K_M route while a
persistent Gemma4-12B Q4_0 process served the cold route. The control ran the
cold model on the i9-12900K. The treatment split every decode FFN by columns
between the CPU and OP15 HTP, overlapped both branches, and used direct
FunctionFS DMA-BUF for the activation exchange.

This workload does not improve with OP15. Across three valid runs per mode,
median total throughput fell from 7.126 to 6.578 output token/s, a 7.69%
regression. Cold median service time rose from 5.154 to 5.618 seconds, a
9.00% regression. The useful benefit is capacity: cold-process peak RSS fell
from 12.425 to 6.717 GiB, saving 5.707 GiB (45.94%), with no process swap.

Each cell below is the median of three fresh full-trace runs. Brackets show
the run range.

| metric | CPU cold route | CPU plus OP15 cold route | change |
| --- | ---: | ---: | ---: |
| trace duration | 83.075 s [82.550, 83.117] | 89.994 s [88.947, 90.936] | +8.33% |
| trace throughput | 7.126 token/s [7.122, 7.171] | 6.578 token/s [6.510, 6.656] | -7.69% |
| cold service p50 | 5.154 s [5.152, 5.166] | 5.618 s [5.533, 5.777] | +9.00% |
| cold queue p50 | 18.005 s [17.702, 18.171] | 23.974 s [23.401, 25.088] | +33.15% |
| cold completion p50 | 22.455 s [22.152, 22.621] | 29.261 s [28.113, 30.467] | +30.31% |
| all-request SLO | 73/74 | 67/74 [65, 67] | -6 requests |
| cold-request SLO | 16/17 | 10/17 [8, 10] | -6 requests |
| hot service p50 | 0.440 s [0.437, 0.441] | 0.448 s [0.442, 0.462] | +1.68% |
| cold peak RSS | 12.425 GiB | 6.717 GiB | -45.94% |

The hot route remained close to unchanged. The cold route is serialized in
both modes, so its extra service time accumulates as queueing delay. This is
why a 9.00% service regression becomes a 30.31% completion-time regression.

## Workload

The frozen BurstGPT-derived trace has 74 requests, a final scheduled arrival
at 57.7 seconds, 592 requested output tokens, eight output tokens per request,
and a 30-second SLO. It contains 57 frequent and 17 infrequent model-role
rows. This controlled slice caps physical prompts at 128 tokens: 60/74 total
prompts and 13/17 cold prompts reach that cap. The other cold prompts contain
23, 47, 53, and 76 tokens.

This is not the raw BurstGPT length distribution. The retained source fields
range from 16 to 8,685 input tokens and 3 to 1,234 output tokens; their median
output length is 42. The eight-token normalization makes this run useful for
controlled scheduling comparisons, but it biases the workload toward prefill
and against an optimization that only accelerates decode.

The original role labels are preserved, but the physical model assignment is
remapped because the exact phone-offload implementation supports Gemma4
Q4_0, not Qwen Q4_K_M:

- 57 source Gemma role rows -> hot Qwen3-14B Q4_K_M on the RTX 4060 Ti;
- 17 source Qwen role rows -> cold Gemma4-12B Q4_0 on CPU or CPU plus OP15.

Within this frozen normalized file, the role remap does not change arrival
times, prompt lengths, prompt token arrays, requested output lengths, or SLOs.
Every prompt token is valid in both physical model vocabularies. This is a
scheduling and performance trace, not a semantic quality trace.

The hot server uses eight slots and continuous batching. The cold process is
persistent but executes a serialized batch of one in both control and
treatment. The comparison therefore isolates OP15 for the implemented cold
route, but it does not claim cold-route continuous-batching performance.

## Operator placement

This is operator-level overlap, not layer-level offload. Prefill remains on
the desktop. For each single-token decode graph, all 48 Gemma dense FFNs are
split as follows:

    desktop CPU: columns [0, 5696)
    OP15 HTP:    columns [5696, 15360)

The desktop branch and phone branch start from the same normalized activation
and execute concurrently. The partial outputs are summed before the layer
continues. Both devices load the byte-identical Q4_0 model. Only transient
activations are packed to FP16 on the wire.

Each treatment run completed 6,048 verified phone calls: 336 warmup calls and
5,712 paid calls, equal to 17 requests x 7 decode evaluations x 48 layers.
There are no phone calls during the multi-token prefill.

Across the three primary treatment runs, median transport and compute were:

| component | median |
| --- | ---: |
| HTP compute | 1.207 ms |
| desktop FFN branch | 1.480 ms |
| phone RPC | 1.480 ms |
| direct-DMA USB round trip | 1.441 ms |
| join wait | 0.028 ms |
| overlapped FFN | 1.522 ms |

The branch balance is good and the DMA path is stable. The full-trace loss is
not caused by an unbalanced phone decode branch.

## Prefill/decode diagnosis

A timing-only result extension recorded prefill and decode separately in one
additional full-trace pair. It adds counters to the persistent reply without
changing model execution.

Across all 17 matched cold requests, median decode time for the seven decode
evaluations fell from 1.696 to 1.088 seconds, a 35.88% improvement. Median
prefill rose from 3.480 to 4.606 seconds, a 32.39% regression. The prefill
penalty is larger than the decode saving.

Thirteen of the 17 cold prompts contain 128 tokens. For those rows:

| component | CPU | CPU plus OP15 | change |
| --- | ---: | ---: | ---: |
| prefill | 3.534 s | 4.788 s | +35.47% |
| seven decode evaluations | 1.697 s | 1.094 s | -35.52% |
| request wall | 5.253 s | 5.867 s | +11.68% |

Short prompts can still win. In the diagnostic pair, the 23-, 47-, and
53-token cold rows improved by 15.99%, 1.76%, and 2.38%, respectively. The
76-token row regressed by 0.73%.

The measured decode saving is about 86 ms per additional generated-token
evaluation for a 128-token prompt. A trace-specific eligibility estimate is:

    use OP15 when
      prefill_penalty(prompt_tokens)
        < (requested_output_tokens - 1) * 86 ms

With the measured 1.201-second median prefill penalty at 128 prompt tokens,
the estimated break-even is about 15 requested output tokens. This trace asks
for only eight. The threshold is an empirical estimate for this hardware and
load, not a model-independent constant.

## Why prefill is slower

The cause is the CPU weight layout, not USB. The control enables llama.cpp's
shape-dependent CPU repacking. Its model log contains a 5,847.19 MiB
`CPU_REPACK` buffer and Q4_0 8x8 repack records. When any FFN split is active,
the driver explicitly sets `model_params.use_extra_bufts = false` because the
current FFN submatrix views are incompatible with the shape-dependent packed
layout. The treatment log therefore has only the 6,637.63 MiB mapped model
buffer and executes generic Q4_0 CPU kernels during prefill.

This also accounts for the memory result. The missing 5.71-GiB repack buffer
almost exactly matches the measured 5.707-GiB RSS reduction. The capacity
gain is primarily the removal of the desktop repack buffer, not free speed
from moving weights to the phone.

No phone RPC is issued during prefill. The 6,048 treatment calls equal exactly
18 requests x 7 decode evaluations x 48 layers, including one warmup request.
Changing the phone column count cannot remove the prefill penalty while any
nonzero split continues to disable CPU repacking globally.

## Latency bottleneck

For the component-timed median cold request:

| stage | CPU | CPU plus OP15 | treatment share |
| --- | ---: | ---: | ---: |
| prefill | 3.480 s | 4.606 s | 80.9% |
| seven decode evaluations | 1.696 s | 1.088 s | 19.1% |
| route wall | 5.176 s | 5.691 s | 100% |

Within the 1.088-second treatment decode, the 336 split FFNs account for
about 0.511 seconds using the 1.522 ms per-call overlap median. The remaining
about 0.577 seconds is attention, normalization, residual operations, the
vocabulary head, graph overhead, and sampling.

Within one split FFN, the 1.480 ms desktop branch and 1.480 ms phone RPC are
already balanced. HTP compute is 1.207 ms; the remaining roughly 0.273 ms on
the phone path covers packing, protocol, USB submission/completion, and bridge
overhead. The measured 1.441 ms USB interval includes waiting for HTP output,
so it must not be added to HTP compute. The per-call join wait is only 0.028
ms. Neither USB bandwidth nor split width is the dominant full-request
bottleneck; unrepacked CPU prefill is.

At the trace level, serialized cold queueing is the largest user-visible
latency term: 23.974 seconds at p50 in treatment versus 5.618 seconds of cold
service. Repeated prefill losses accumulate behind the cold queue.

## Correctness and safety

- All six primary runs completed 74/74 requests and 592/592 requested output
  tokens.
- CPU tokens were identical across all three CPU runs. OP15 tokens were
  identical across all three OP15 runs.
- OP15 versus CPU matched 10/17 complete cold outputs and 92/136 cold token
  positions. FP16 activation transport is therefore not byte-exact for this
  trace.
- The prompts are deterministic synthetic token sequences, so this run cannot
  establish that divergent text is semantically acceptable. It is a
  performance-shape validation only.
- Every treatment reported `FFN_OVERLAP_OK`; each bridge reported `status=ok`;
  all 18,144 primary treatment calls passed protocol and hash checks.
- No hot or cold process swapped.
- The final kernel scan found no DMA-BUF, SMMU, IOMMU, panic, BUG, or call-trace
  fault.
- A normal reboot restored stock kernel
  `6.12.23-android16-5-gb3b66ace21e0-ab14672634-4k`, USB `ptp,adb`, device
  `22d9:2772`, and no worker process.

## Recommendation

For this exact eight-output-token trace, keep the cold route on the desktop
CPU when throughput or latency is the objective. Enable OP15 only when the
5.707-GiB cold-process memory saving is more important than the measured
latency loss.

A dynamic route should gate on prompt length and expected output length. With
the current two layouts, the planner must treat optimized CPU execution and
split execution as different modes:

    T_cpu(P, O) = prefill_repacked(P) + (O - 1) * decode_repacked

    T_split(P, O, W) = prefill_generic(P)
                     + (O - 1) * decode_split(W)

    choose the mode and W with the lowest predicted service time

For a split mode, choose the phone width `W` that minimizes the maximum of the
desktop branch and phone RPC. The current 9,664-column point already satisfies
that condition at about 1.480 ms on both branches. The more important dynamic
choice for this trace is therefore zero offload with CPU repacking versus the
9,664-column phone route. The measured eight-token choices are phone for the
23-, 47-, and 53-token prompts, and repacked CPU for the 76- and 128-token
prompts.

Changing `W` inside the current process is not enough: even `W=0` would still
use the generic layout after model load. A complete implementation needs one
of these designs:

- keep separate resident repacked-CPU and split-model processes and route each
  whole request before prefill;
- preserve full repacked tensors for prefill and add shape-correct repacked
  host-prefix tensors for split decode;
- shard the FFN weights into independently repacked column blocks, use all
  blocks locally for prefill, and select a CPU/phone boundary for decode.

The second or third design removes the prefill regression without duplicating
an entire model process. After that change, the width sweep must be repeated;
a faster repacked desktop prefix will likely move the optimum to fewer phone
columns. Until then, short-prompt or long-generation requests are the useful
OP15 region; 128-token, eight-output-token requests are not.

## Reproducibility

Runner:

    burstgpt_gpu_cpu_op15_v1/run_trace.py

Frozen input:

    ../REQUESTS.jsonl

Input SHA-256:

    94c36fe3ac43281dc0c83a29a1519c7e72ac445ed9ded98d474b2231041c1735

Model SHA-256 values:

    Qwen3-14B-Q4_K_M: 500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0
    Gemma4-12B-Q4_0:  494518c2262a26e2a607af0e40bca11c4de5a0b108e7c21308906dbbfb1c6f8c

Primary raw evidence remains on the physical desktop at:

    /home/zhihao/s41-burstgpt-cpu-op15-v1/raw/

Primary directories are `cpu-r1` through `cpu-r3` and `op15-r2` through
`op15-r4`. Timing diagnostics are `diag-cpu-r4` and `diag-op15-r5`.

The primary driver binary SHA-256 is
`7aeecee2dd936660e56b9653d4ff31f1990164457d16f0534cb6766d1757177c`.
The timing-only driver SHA-256 is
`0bcba32d74e6c7efb5fe90d6071735d9a604e2dc0c8303471c6e2eeb24241821`.
