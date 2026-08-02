# S20 server-only real-trace resource timeline

Status: `SERVER_TRACE_TIMELINE_PASS` on 2026-07-20.

## Question

When a production-shaped generation trace is replayed through Gemma-4-12B on
one A6000, when is execution prefill/compute-dominant, decode/memory-dominant,
mixed, or idle?

## Frozen scope

- Workload: the 32-request observed-arrival BurstGPT cohort in
  `research_dev/spikes/s15_burst_cohort/cohort.json`.
- Observed fields: request identity, relative arrival, input-token count, and
  output-token count.
- Synthetic field: prompt token values, because BurstGPT publishes lengths but
  not prompt text.
- Model: `/home/myid/zs89458/Documents/models/gemma-4-12B-it-f16.gguf`.
- Device: physical A6000 index 0 only; GPU 1 must remain idle.
- Runtime: `llama-server`, continuous batching, 32 slots, unified F16 KV.
- Hardware sampling: Nsight Systems GA10x metrics at 1 kHz, reduced into 100 ms
  bins. The load-bearing lines are DRAM read/write throughput, SM issue, and
  tensor activity.
- Runtime sampling: `/slots` and `/metrics` every 100 ms.

## Classification contract

The phase label is derived from executed streaming response boundaries, not
from the source trace and not from an arbitrary GPU-utilization threshold:

- request start through its first returned token: prefill/TTFT interval;
- first returned token through exact completion: decode interval;
- prefill intervals only: `COMPUTE_DOMINANT`;
- decode intervals only: `MEMORY_DOMINANT`;
- both: `MIXED`;
- neither: `IDLE_OR_TRANSITION`.

All 32 slots are provisioned before the 32-request cohort begins, so this test
does not intentionally add an admission queue to the prefill/TTFT interval.
The `/slots` samples are retained as a cross-check, but they are not the
load-bearing phase clock because the endpoint can respond only between model
evaluations.

The names describe the expected Gemma execution regime at the measured shapes.
The Nsight lines are independent corroborating hardware pressure. This is not a
formal per-kernel roofline proof, and the result must not be labeled as one.

## Gates

- [x] Source cohort and model identity recorded.
- [x] Exactly one GPU visible to `llama-server`; GPU 1 idle throughout.
- [x] All 32 requests complete with exact observed prompt and output counts.
- [x] Continuous batching enabled and at least two requests overlap.
- [x] Nsight replay range and all required metrics present.
- [x] Every request has ordered start, first-token, and exact-completion events.
- [x] SVG and HTML graphs generated only from persisted raw evidence.

See `RESULTS.md` and `results/server_trace.html`.
