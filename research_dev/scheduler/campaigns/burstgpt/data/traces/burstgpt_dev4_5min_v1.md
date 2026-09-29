# Four-request development replay

Use `burstgpt_dev4_5min_v1.json` for short development checks, not for a
representative energy-saving claim. The existing 24-request replay is unchanged.

## Workload

This is an ordered subset of `burstgpt_sparse_locality24_v1.json`, with manually
compressed replay arrivals. Request contents, prompt tokens, requested output
lengths, source timestamps, and model assignments are not changed. These are
BurstGPT-derived requests from the existing semantic-source/overlay workload,
not original BurstGPT prompt text or original interarrival times.

| Combined index | Model | Arrival (s) | Input tokens | Output tokens | Purpose |
| --- | --- | ---: | ---: | ---: | --- |
| 34 | Qwen | 1 | 147 | 36 | Initial request and helper availability |
| 40 | Qwen | 31 | 178 | 43 | Same-model reuse with a fresh request binding |
| 56 | Llama | 71 | 625 | 12 | Small-model arrival and short-tail handling |
| 57 | Gemma | 91 | 271 | 41 | Changed demand and compatible helper attachment |

Totals: 4 requests, 1,221 input tokens, 132 output tokens, 90-second arrival span.
The runner submits each arrival at its replay time; the scheduler must not receive
future rows as live demand. No route, fraction, session assignment, or transition
time is prescribed by this trace. Keep the same large-model F16 artifacts and
Llama Q4 artifact as the comparison configuration.

The trace can exercise reuse and a Qwen-to-Gemma demand change. It does not prove
reverse replacement, a complete fraction sweep, sustained batching, or whole-phone
Llama execution. A short request may correctly finish without phone assistance.

## Five-minute budget

The target is at most 300 seconds per physical arm, not 300 seconds for an A/B
pair. Prepare binaries, qualification, catalogs, shard files, and preflight before
the timed development run. Do not repeat profiling or calibration on each launch.
Initial device loading remains inside the timed run unless explicitly reporting
a separate warm-start experiment. Phone preparation should overlap desktop loading.

Reference measurements from the matched 2026-09-07 v6 run:

| Index | Desktop inference (s) | Adaptive inference (s) |
| --- | ---: | ---: |
| 34 | 32.764154 | 30.748230 |
| 40 | 36.125723 | 58.572193 |
| 56 | 0.287942 | 0.193230 |
| 57 | 28.131950 | 27.511661 |

Taking the slower observation per request gives 119.756239 seconds of serial
inference. Observed desktop loads in that adaptive run reached 65.744803 seconds
for Qwen, 5.561847 for Llama, and 73.333517 for Gemma. With one load per model,
serial replay of the four arrivals finishes at about 265.4 seconds. This leaves
about 34.6 seconds for additional overhead, provided phone loading overlaps.
Warm initial Qwen/phone residency would provide more headroom.

These are a sizing estimate, not upper bounds or a measured dev4 result. Disk
cache, live VRAM, extra reloads, queue policy, and runtime faults can exceed it.
There has been no physical run of this trace yet.

## Launch and cutoff

Use the existing resolved physical runner command with:

```sh
--replay-schedule research_dev/scheduler/campaigns/burstgpt/data/traces/burstgpt_dev4_5min_v1.json
```

For a campaign manifest, set `trace.replay_schedule_path` to this file, clear
`trace.request_indices`, and leave `trace.arrival_scale` null. Remove conflicting
`--request-indices` or `--arrival-scale` options from a direct runner command.
Keep the live-VRAM desktop parents, explicit CUDA graph mode, shard-index flags,
and physical model artifacts unchanged.

The replay JSON itself does not impose a wall-clock timeout. Prefix the resolved
physical runner command with `timeout --signal=INT 300s` for the development
cutoff. SIGINT enters the runner's existing failure-artifact and cleanup path.
Exit 124 or fewer than four completed requests is a timeout/failure, never PASS.
Do not add a forced-kill timeout that skips USB/session cleanup. Cleanup may take
additional time; five minutes is the execution cutoff, not a guarantee that all
cleanup finishes by that instant. If the run overruns, diagnose the delay rather
than silently shortening requests or allowing an hours-long drain.

Report physical start/end, first phone call, fraction-weighted coverage, loads,
queue delay, and elapsed time. Compare energy only against a control that executes
these same four requests and arrivals with matching runtime identities and startup
conditions; the old 24-request denominator is not usable.

## Provenance

- Parent replay SHA-256:
  `78b7582ebfe063cdbb14added28b060ee955362d907cdfc1b5b7c6aa14192ce9`
- Large source: `research_dev/spikes/s41_gemma_qwen_continuous_baseline/tp_operator_split_v1/burstgpt_gpu_cpu_op15_v1/REQUESTS_SEMANTIC_SOURCE.jsonl`
  SHA-256: `b20a9ba66ee3558d835a0e19ed3cfa4c31a4a9e8b4f9c085b29a14f80250a0ff`
- Overlay source: `research_dev/spikes/s42_general_energy_scheduler_v1/full_fp16_burstgpt_v1/small_model_overlay_v1/REQUESTS_LLAMA1B_10.jsonl`
  SHA-256: `a39ed66211e490d6b5e95dae29ad76a4ad847e1a1d1e5f8c1004e4d7c16abaaf`
- Timing audit: `../../reports/20260907-mixed-host-stall/V6_MATCHED_EXACT_AUDIT.json`
- Load timing source: `/home/zhihao/s42-mixed-matched24-20260907-v6-adaptive/run/RESULT.json`
  SHA-256: `990718518f65172f548e26bae20edb4a8436c9529b10634be7eb6049481cc7d1`
