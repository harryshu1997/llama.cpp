# S42 Gemma4 and Qwen3 energy model V1

Date: 2026-08-06 EDT.

## Goal

Measure model-resident inference energy for the first two scheduler workloads:

- Qwen3-14B Q4_K_M on the RTX 4060 Ti CUDA route;
- Gemma4-12B Q4_0 on the i9-12900K CPU route; and
- Gemma4-12B Q4_0 on the CPU plus OP15 HTP FFN route.

The result is an operator-composed route model with cohort-level physical
validation. It does not divide shared continuous-batch energy among concurrent
requests. An unseen model may reuse only kernel and shape buckets already
measured on the exact device and runtime.

## Frozen scope

Model loading is outside the paid interval. Each server loads once, completes
a warmup, and stays resident across the case grid. Loading and switching
energy will be a separate profile because the scheduler amortizes those costs
over a different horizon.

Every inference case records:

- model, runtime, GPU, CPU-affinity, case-plan, and executable hashes;
- cohort size, cohorts completed, requests, input rows, output rows, decode
  steps, and estimated prefill microbatches;
- paid start and end on the host monotonic clock;
- CPU-package and GPU-board joules over that interval;
- request latency and server timing summaries; and
- swap, placement, phone work, transfer, and cleanup state where applicable.

The first pass measures only the server boundary. For an OP15 route, the
prototype accounting adds 5 W over every paid interval for which the phone is
reserved. This produces an estimated accounted-energy component while CPU
package, GPU board, and their sum remain measured. A later physical pass may
replace that term with whole-phone USB input plus simultaneous battery
discharge using clock-aligned samples.

## Case grid

`CASES_V1.jsonl` contains one resident-idle case, prefill-heavy cohorts,
decode-heavy cohorts, and a mixed held-out case. Concurrency is limited to
1, 4, and 8. A case runs complete cohorts until its minimum paid duration is
met or its frozen cohort ceiling is reached. No case stops mid-cohort.

The estimator uses operator invocation count, primitive compute operations,
physical memory bytes, and measured kernel/device coefficients. The existing
cohort fit publishes the largest positive residual and all held-out errors as
an independent validation. A profile is usable only within observed kernel
shape ranges and exact runtime/device identity.

## Pass gates

- every planned case completes with exact request and token counts;
- model and runtime hashes are identical across repetitions of one route;
- no model-process swap, server failure, bridge reset, or leaked worker;
- energy samples cover every paid boundary and reconcile arithmetically;
- at least three fresh repetitions per final route;
- held-out absolute energy error at most 10% for a scheduler candidate;
- a treatment route must still pass its independent quality gate; and
- unknown models, devices, kernels, or shapes remain server-fallback only.

The held-out gate applies to measured server components. An assumed 5 W phone
term can support prototype ranking and sensitivity analysis, but cannot by
itself authorize a measured fleet-energy result.

## Ordered execution

1. Validate the operator estimator, runner, and cohort validator offline.
2. Measure idle power and the bounded CUDA and CPU kernel/shape set.
3. Measure OP15 HTP, direct-DMA transfer, and host-merge coefficients, using
   the explicit 5 W phone estimate for the first prototype.
4. Materialize Qwen and Gemma operator graphs from exact runtime work counts.
5. Run held-out whole-route cases and require at most 10% error.
6. Acquire three fresh route repetitions only for profiles that pass the
   pilot gate.
7. Install only measured, boundary-matched profiles in S42 enforcement.
