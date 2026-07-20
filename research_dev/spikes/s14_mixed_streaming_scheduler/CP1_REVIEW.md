# S14 CP1 physical-validation review

Status: IN_PROGRESS 2026-07-18. The mixed-system proof has not passed.

## Checkpoints

| Checkpoint | Current state | Meaning |
|---|---|---|
| CP1-A server BGE atlas | MEASURED_TIMING | Seven processes, 140 samples per row, CUDA-vs-CPU cosine 0.999999 or better. The repaired analytic model refuses a roofline regime claim; measured batch knee is the scheduler input. |
| CP1-B phone BGE atlas | INELIGIBLE | The matched 7-process rerun passes placement, correctness, exact-shape, and thermal checks, but 10/18 rows exceed CoV 0.05. |
| CP1-C three-device route | MECHANICS_UNRELIABLE | Single acquisitions are token-correct at B={1,4,8,32}; B16/B64 did not complete, and a repeated B32 mixed run failed before readiness. |
| CP1-D priority energy | REPLAYABLE_DIAGNOSTIC | Raw replay gives 11.854 percent selected-A6000 relief, but no phone runs in its paid window and its service legs are sequential. |
| CP1-E live priority run | FAIL_MEASUREMENT | One P0/P2 pair completed; P2 used 1.293x GPU-board energy and had 13.83x Gemma p95 latency. P2 repeat 1 then failed before readiness. |
| CP1-F independent OP15 run | SERVER_RELIEF_PASS_SLO_FAIL | Seven-process B1 route passes. The live run saves 5.43 percent selected-GPU energy and preserves high-priority BGE p95, but low-priority Gemma p95 is 2.639x and fails the frozen 2.0x gate. |
| CP1-G independent OP12 route | ROUTE_PROFILE_PASS | `[0,6)` passes 7/7 processes, 56/56 exact requests, placement, thermal, raw-log replay, and CoV 0.0258. `[0,8)` is rejected after a predeclared cohort timed out on process 3. Two-phone live dispatch is not run yet. |
| CP1-H parallel heads/shared tail | POINT_PASS_REPEATABILITY_FAIL | OP15 and OP12 `[0,6)` heads run concurrently into one A6000 tail context. Tail B2 repeatedly fails exact tokens. Tail B1 passes exact tokens and all placement gates, but a later worker reload times out. Persistent sessions are required before energy. |
| CP1-I persistent session protocol | MECHANICS_PASS | Opt-in versioned DETACH/STOP protocol (layersplit.cpp only). One resident OP15 v81 + one resident OP12 v75 `[0,6)` worker serve 7 sequential shared-tail B1 sessions (6 DETACH + 1 STOP, Tail B2 not retried). Per-stream tokens identical across all 7 sessions, one resident pid/nonce per phone throughout (worker reload eliminated), every session cert SCHEDULED_PLACEMENT_OK with missing_buffer=0 and HTP0-only + declared GET_ROWS. A wire client also proves bit-identical head bytes across the 7 sessions on both phones. `PERSISTENT_SESSION_MECHANICS_PASS_PHYSICAL_ENERGY_NOT_RUN`; energy NOT measured. See `persistence/RESULTS.md`. |

CP1-D is selected-A6000 GPU-board relief only. It does not measure phone, USB,
host-wall, or total-system energy. CP1-E is the first live concurrent attempt and
does not reproduce CP1-D's saving.

## Repairs applied after review

- `bge_corpus.py` is now the single corpus generator for server and phone BGE
  measurements.
- `cp_b_phone_bge.py` defaults to seven processes and fails closed on a server
  exact-token mismatch, CoV above 0.05, or invalid thermal samples.
- `validate_cp_b_result.py` independently rejects the matched CP1-B artifact
  because its frozen variability gate fails.
- `cp_d_priority.py` v3 persists integer-us/integer-mW raw samples, exact paid
  windows, and executable/source hashes.
- `validate_cp_d_result.py` recomputes equal work and summary energy for v2, and
  replays every raw power integral for v3.

## Gate to finish CP1

1. [DONE] Certify repeatability of the independent OP15 `[0,8)` route at a predeclared
   native batch, with bounded readiness and execution deadlines. The first
   screen leaves only B1 eligible; B4/B8/B32 pass placement but fail exact
   same-batch tokens. B1 passes 7 processes and 56/56 measured requests.
2. [DONE] Run that route concurrently with high-priority BGE on one selected
   A6000, using returned activations to drive the measured suffix. Server-board
   relief passes; the low-priority p95 gate fails.
3. [DONE] Give OP12 an independent READY island or independent request stream. Do not
   serialize OP12 behind OP15 for the same request unless that route first beats
   independent placement. The accepted route is OP12 `[0,6)` B1.
4. Report p50/p95/p99, SLO attainment, useful batch fraction, late/wasted work,
   server throughput, and selected-GPU board energy. Keep phone/USB/total energy
   unknown until instrumented.

Only item 4 tests the intended mixed-workload scheduler. CP2 and CP3 remain
blocked until this CP1 gate is decided.

The next driver must share one A6000 tail model across the OP15 and OP12 lanes.
Two concurrent tail processes are not an acceptable shortcut because their
duplicated weights would consume the memory relief being measured.

That shared-tail driver now exists. Its next gate is seven sessions against one
persistent resident worker per phone. Reloading the 4.5 GiB head shard between
sessions is both operationally wrong for the residency design and empirically
unreliable. Tail B2 remains ineligible until a separate correctness mechanism
explains and closes its token mismatch.

## Runtime repair

`PriorityBatchRuntime` now rejects bare latency-only `BatchPoint` records. A
route must provide a `CertifiedBatchPoint` with non-empty correctness and
placement certificate IDs in `sha256:<64-hex>#claim` form for that exact batch. The Stage-B harness also
returns nonzero for token mismatch, incomplete requests, host failure, zero HTP
work, or undeclared CPU work.
