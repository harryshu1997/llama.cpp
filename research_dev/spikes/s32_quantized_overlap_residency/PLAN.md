# S32 Quantized Overlap Residency

## Status

`CAPACITY_GAIN_PASS_NUMERIC_ROUTE_FAIL`

Q8_0 expands the clean tested B32 windows to OP12 `[0,4)` and OP15
`[4,16)`, but neither Q8_0 nor Q4_0 passes the frozen relative-L2 gate. A
same-Q8 end-to-end route with 32 distinct inputs confirms that the difference
is observable at the output: only 18 of 32 four-token request sequences match
the one-GPU CUDA reference. Quantized phone routes therefore remain excluded
from the exact scheduler. See `RESULTS.md`.

## Question

Can Q8_0 or Q4_0 weights widen the layer intervals resident on OP12 and OP15
without losing the HTP placement, B32 latency, or numerical quality needed by
the elastic-boundary scheduler?

The current S31 phone workers use F16 shards. The RTX 4060 Ti control uses a
Q8_0 model. S32 does not claim that smaller weights imply a faster route.

## Fixed gates

A phone range is usable only when all of the following hold in one run:

1. The local and remote GGUF SHA-256 values match.
2. StageNet reports the requested layer interval and at least 32 sequence slots.
3. Boundary activations are deterministic on the phone.
4. HTP versus a same-GGUF CPU reference has relative L2 at most 0.005 and
   cosine similarity at least 0.999 for every tested position.
5. The isolated B32 placement certificate is `SCHEDULED_PLACEMENT_OK`, has no
   missing-buffer compute nodes, and places every operation except `GET_ROWS`
   on HTP0. B1 correctness is a separate run because quantized `ffn_down`
   currently falls back at B1; the scheduler must not dispatch that phone shape.
6. Seven measured B32 cohorts complete after two discarded warmups, with no
   non-finite output and no retained sequence state.
7. The phone process remains live with zero swap after the measured cohort.

Failure at a range excludes that range and all larger ranges with the same
start, quantization, context, batch capacity, and device memory configuration.
Numerical failure may continue into a `PERF_ONLY` capacity screen, but such a
row remains ineligible for scheduler dispatch and cannot be called a pass.

## Experiment order

- [x] Screen `[0,2)` Q8_0 and Q4_0 numerics on OP15.
- [x] Sweep OP12 Q8_0 head ranges `[0,e)` upward.
- [x] Sweep OP15 Q8_0 middle ranges `[4,e)` upward.
- [x] Record the largest zero-swap range and per-layer B32 latency.
- [x] Run a same-Q8 B32, four-step, three-device token comparison.
- [x] Repeat the token comparison with 32 distinct inputs.
- [ ] Select a phone quantization for scheduler use. Blocked by the numerical
  gate; do not silently substitute a capacity-only row.

Q8_0 is preferred if both formats pass because it matches the intended server
model and has lower quality risk. Q4_0 is selected only if its extra residency
materially enlarges the feasible boundary set without a latency or quality
failure.

The numerical and performance certificates are deliberately separate. A
performance-only run may use a middle range with deterministic finite hidden
inputs and no CPU reference. It cannot establish numerical eligibility.

## Scope

This checkpoint measures resident capacity and route eligibility. It does not
yet implement runtime boundary selection, claim workload throughput, or claim
phone or total-system energy savings.

The next exact-path prototype should retain F16 phone shards and add a finite
set of premeasured dynamic entry points within overlapping resident windows.
An alternative quantized path requires a separately frozen task-quality gate;
it must not weaken this checkpoint's predeclared exact numerical gate.
