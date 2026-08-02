# CP0-R1 V2.1 phased evidence correction

Verdict:

`V2_1_PHASED_EVIDENCE_READY_ACQUISITION_NOT_RUN`

V2.1 is an additive successor to the frozen V2 evidence contract. V1 and V2
are unchanged. No Qwen3 14B qualification, Qwen3 8B acquisition, model-switch
cycle, trace replay, controller integration, or energy acquisition ran.

## Phase protocol

V2 required A, B, and pair evidence in one bundle. V2.1 separates the work:

1. `A_ONLY` may emit only `MODEL_A_QUALIFICATION_PASS`.
2. `B_ONLY` must reopen and re-evaluate A's raw bundle before it may emit
   `MODEL_B_QUALIFICATION_PASS`.
3. `PAIR` must reopen and re-evaluate both model bundles before it may emit
   `TWO_ROUTE_ELIGIBILITY_PASS`.

Each bundle has one phase ID, open time, acquisition start, close time, and
`HOST_MONOTONIC_RAW` clock. Every JSONL row carries that phase and one event
time inside the interval. Route and corpus inputs precede the phase lock; the
lock precedes readiness; readiness precedes acquisition. Prior result digests
are part of the B and pair locks, but the evaluator does not trust those
digests alone: the CLI re-evaluates the referenced raw predecessor bundles.

## Closed evidence gaps

- Quality: the phase lock binds the exact corpus artifact. Each CUDA and phone
  output binds both the corpus digest and exact corpus-item digest before the
  per-item noninferiority result is derived.
- Memory: every CUDA sample requires exact `used + free = total`, one sampler
  identity, target-process attribution, B8 completion, model/KV allocation,
  headroom, and zero swap growth.
- Oracle: all three paths cover exactly eight requests. Every physical call has
  `n_seqs=8`; prefill/decode row totals must equal the recorded request rows;
  CUDA route and independent oracle shapes match; continuation lengths are
  equal before the phone-vs-CUDA diagnostic iterates them.
- Bridge: each publication binds the complete phone mechanics request row and
  exact continuation. CUDA readiness binds the exact CUDA ready-memory row and
  timestamp.
- Transfer: every OP15-to-OP12 call carries exactly
  `rows * hidden_size * sizeof(F32)` bytes. Qwen3 14B's measured hidden size is
  5120; Qwen3 8B is locked at 4096.
- Reprepare: each phone must report and counter-account exactly the complete
  target shard bytes from local UFS, with zero USB or network weight growth,
  released request state, a new readiness generation, and the frozen dwell
  bound.

## Full readiness

`cp0_r1_phase_preflight_v21.py` derives fixed argv from the phase lock. It
checks the target RTX 4060 Ti identity plus the complete desktop model, both
ADB servers, both phone identities, and each complete on-phone shard. The raw
argv, stdout, stderr, start time, and completion time become phase evidence.
The evaluator reconstructs and exact-checks every argv and output and requires
preflight completion no more than five seconds before acquisition starts.

This phase preflight was deliberately not run during contract construction.
It must be repeated immediately before the later 14B acquisition; reusing the
historical identity-only V2 preflight cannot satisfy V2.1.

## Verification

The focused V2.1 suite passes 19/19. The full S39 suite passes 310/310. Positive
cases cover A alone, B chained to
A, and a complete A-to-B-to-pair chain. Adversarial cases cover:

- an event outside its phase and a late phase lock;
- stale readiness and altered command argv;
- fabricated corpus-item linkage;
- non-exact CUDA memory and understated process memory;
- short continuations hidden by the V2 `zip()` diagnostic and non-B8 calls;
- unlinked phone publications and CUDA readiness;
- one wrong activation byte count;
- partial full-shard verification and a short UFS counter delta;
- a pair lock missing one predecessor result.

The contract-only CLI emits
`V2_1_PHASED_EVIDENCE_READY_ACQUISITION_NOT_RUN`, never an eligibility pass.
The next and only authorized acquisition is Qwen3 14B `A_ONLY`.
