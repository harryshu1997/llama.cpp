# S18 two-phone independent R1 mixed-workload gate

Status: full real-device acquisition complete. Verdict:
`S18_R1_FLEET_MECHANICS_PASS_RELIEF_INSUFFICIENT`.

## Question

Can two independently certified phone-prefix routes execute low-priority Gemma
work concurrently while one selected A6000 preserves high-priority BGE service?

This gate tests R1 only. The numerically ineligible S17 hierarchical R2 route
is not loaded or dispatched.

## Routes

~~~text
high priority: BGE B16 ------------------------------> A6000

low lane 0: OP15 Gemma [0,8) B32 -> A6000 [8,48)
low lane 1: OP12 Gemma [0,6) B32 -> A6000 [6,48)

tight SLO or unsupported work ----------------------> A6000 R0
~~~

The phone lanes own disjoint requests. They are not a serial phone chain and
do not share KV state. Each phone and CUDA tail keeps one process and context
resident for the row.

## Frozen work

- Exactly one selected A6000; the second A6000 must remain idle.
- High priority: BGE-small-en-v1.5, sequence target 32, measured B16 knee.
- Low priority: repeated frozen BurstGPT prompt, Gemma-4-12B, 8 generated
  tokens per request.
- Six low rounds per full row. Each round contains 64 requests.
- P0: two sequential full-model A6000 B32 executions per round.
- P4: one OP15 B32 route serves the 5 s class every round. OP12 concurrently
  serves the 12 s class for at most two exchanges, which is the only persisted
  repeatability envelope its existing certificate proves. After that lease is
  exhausted, OP15 serves the 12 s class after the 5 s class in the same round.
- Every B32 context uses a 16-token per-request envelope. The frozen prompt is
  5 tokens, the declared prefill bound is 8, and the run generates at most 8.
  The driver invariant `max_prefill + n_gen <= context` therefore holds. This
  avoids reserving unused 512-token KV capacity while preserving the complete
  request lifetime.
- Three rotated P0/P4 pairs: P0,P4 / P4,P0 / P0,P4.
- Preparation and model loading occur outside the paid BGE window.

The screen uses two rounds and one P0/P4 pair. Priorities, SLOs, and repeated
payloads are synthetic. The source arrival cohort is observed BurstGPT data.

The initially frozen B64 P0 control was run twice, with 512 and then 1,024
tokens of per-request KV capacity. Both produced the stable sequence
`[107,45518,107,101,1509,7412,611,659]`, which differs from the certified B32
sequence `[45518,107,100,45518,107,101,1509,7412]`. Both capacities exceed the
13-token request lifetime, so capacity is not the cause. B64 is excluded rather
than accepting different generated work.
R1b therefore compares the largest exact-token batch geometry established on
all routes. It is not a general optimum-throughput server baseline.

## Gates

1. Every row completes identical BGE work and six rounds x 64 requests x 8
   Gemma tokens.
2. Every Gemma request matches the frozen CUDA token oracle.
3. P0 compute is CUDA0 except declared CUDA_Host GET_ROWS. P4 phone compute is
   HTP0 except declared CPU GET_ROWS; both tails compute on CUDA0.
4. Each process identity remains stable within a row. DETACH resets request KV;
   only the final exchange uses STOP.
5. Median BGE p95(P4) / p95(P0) is at most 1.05.
6. Every OP15 exchange is at most 5 s and every OP12 exchange is at most 12 s.
   Including queueing, the first group in each round completes within 5 s and
   the second within 12 s.
7. The first two paired phone exchanges overlap in wall time. All low work
   completes inside the paid BGE interval.
8. A claim-bearing selected-GPU result requires at least 10 percent median raw
   reduction and a positive lower bound after the Ampere 5 W uncertainty floor.

Gate 8 is only a selected-GPU board diagnostic. Phone, USB, host-wall, and
total-system energy remain UNKNOWN.

## Known first-run limitation

The current host driver accepts one phone stream per CUDA tail process and the
certified cuts differ. P4 therefore holds two overlapping CUDA tail weight
images. This is valid R1 execution but not the intended final memory layout.
The report must include peak selected-GPU memory. A later multi-ingress tail
must remove the duplicate residency before making an HBM-capacity claim.

P0 also uses sequential B32 because B64 failed exact generated-work identity.
Energy results are diagnostic for this exact-output operating point only.

The first P4 readiness screen at 512 tokens per request also remains negative
evidence: the OP15 tail allocated 4,224 MiB of KV and the OP12 tail requested
another 4,480 MiB, causing CUDA OOM. The 16-token envelope is frozen before the
next acquisition and must still pass token, placement, and SLO gates.

The first six-exchange OP12 acquisition stopped at exchange 4 after the 30 s
watchdog. It is retained as negative evidence. The scheduler may not extrapolate
the prior two-exchange certificate; the revised route has an explicit OP12
credit of two and redirects later loose-SLO cohorts to OP15.

## Stop rules

- Stop before the full acquisition if the two tails do not fit or any screen
  correctness, placement, persistence, overlap, or SLO gate fails.
- Do not substitute R2 after a failure.
- Do not claim total-system energy from NVML board power.
