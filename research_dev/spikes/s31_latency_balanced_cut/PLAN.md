# S31 Latency-Balanced Phone Cut

Status: `S31_BALANCED_FULL_TRACE_PASS_HISTORICAL`; future execution requires
`SAME_Q8_IDENTITY`.

## Same-quantization repair

The original S31 measurement used F16 phone shards and a Q8_0 CUDA model, so
it remains mechanics evidence only. The executable path now requires every
StageNet HELLO to carry the same model identity: GGUF file type 7 (`Q8_0`) and
one exact SHA-256 across CUDA, OP12, and OP15. The launcher independently
hashes each file before publishing that identity, and the measurement binder
rechecks all three hashes. A missing identity, F16/Q8 mix, or different Q8 file
is rejected before any batcher starts.

The future campaign uses the full Q8_0 file on both phones with partial layer
loading. This changes model residency, not the selected layer ranges. It does
not override the S33 numerical-quality failure.

## Question

S29 assigned `[0,6)` to OP12 and `[6,8)` to OP15. At B32, the measured
OP12 stage took about 1.15-1.26 seconds per decode step while OP15 took about
0.13-0.25 seconds. The fixed cut therefore leaves OP12 as the pipeline
bottleneck.

S31 asks whether a measured layer cut can reduce that bottleneck while keeping
the same route output boundary at layer 8:

- R0: CUDA `[0,6)` -> CUDA `[6,8)` -> CUDA `[8,48)`.
- R2(k): OP12 `[0,k)` -> OP15 `[k,8)` -> CUDA `[8,48)`.

Candidate cuts are finite and declared before measurement. The initial set is
`k in {2,3,4,5,6}`. This includes the S29 baseline and moves the nonuniform
layer costs, including the global-attention layer, through the boundary. The
initial sweep selected its lower boundary, `k=2`; one explicit boundary
refinement therefore adds `k=1`, the only remaining cut that keeps both phone
stages nonempty. No further post-result search is allowed in this checkpoint.

## Selection contract

Each candidate must be measured at B32 with the exact resident workers and at
least two repetitions. The controlled calibration uses one uniform 50 ms
gather window so Python submission jitter cannot fragment the intended B32;
the production runtime keeps its separately profiled 5 ms policy. A candidate
is eligible only when:

1. Worker HELLO records match `[0,k)`, `[k,8)`, and `[8,48)` exactly.
2. Every active worker emits four successful physical B32 operations.
3. All 32 requests complete, return four tokens, drain KV, and release slots.
4. Phone and CUDA launch metadata bind one exact Q8_0 GGUF hash and the
   direct-WiFi activation path.
5. The measured route p95 is within the declared background SLO.

The selected-route calibration runs B32 before smaller shapes. A real aborted
run showed that growing one OP15 HTP session from B1/B4/B24 to B32 can stall,
while fresh-session B32 repeatedly completes. The production trace launches a
single B32 phone cohort and does not exercise that synthetic growth sequence.

The deterministic objective is lexicographic:

1. minimize `max(OP12 stage p95, OP15 stage p95)`;
2. minimize route p95;
3. minimize absolute phone-stage imbalance;
4. choose the smaller cut index.

No unmeasured cut or interpolated per-layer latency is dispatch eligible.

## Checkpoints

- [x] Freeze candidate domain and selection objective.
- [x] Implement the dynamic-cut topology, measurement record, and selector.
- [x] Provision the OP15 `[1,8)` superset shard.
- [x] Measure the initial candidates on both real phones at B32.
- [x] Measure the declared `k=1` boundary refinement and select the final cut.
- [x] Recalibrate R0 and selected R2 at B1/B4/B24/B32.
- [x] Rerun the frozen 60-request mixed trace.
- [x] Validate placement, conservation, SLOs, batch sizes, and improvement over
      the S29 `[0,6)|[6,8)` baseline.

## Claim boundary

This checkpoint can establish a better measured static partition and a
scheduler mechanism that selects it. The historical result does not establish
online migration, phone energy, network energy, total-system energy, or
numeric equivalence. Future execution is same-Q8 by construction but still
requires a new quality-eligible HTP kernel result.
