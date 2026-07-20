# S14 checkpoints A-E: corrected physical status

Status: LIVE_SERVER_RELIEF_PASS_LOW_PRIORITY_SLO_FAIL on 2026-07-18. Measurements use one
selected RTX A6000, OP15/Hexagon-v81, and OP12/Hexagon-v75. Only selected-GPU
board energy is measured. Phone, USB, host-wall, and total-system energy remain
unknown.

## CP-A: server BGE atlas - PASS

The repaired run contains 7 processes x 20 repetitions for every shape, raw
timing samples, one selected CUDA device, an idle second GPU at the acquisition
endpoints, and same-run placement certificates. CUDA-vs-CPU cosine is at least
0.999999. The measured 95 percent throughput knees are:

| exact tokens | knee batch | peak throughput |
|---:|---:|---:|
| 31 | 16 | 4242.7 enc/s |
| 132 | 2 | 864.2 enc/s |
| 499 | 1 | 258.0 enc/s |

These are measured scheduling inputs. The weight-only arithmetic-intensity
estimate remains an upper bound and does not classify the executed graph.

Artifact: `bge_server_result_repaired.json`.

## CP-B: phone BGE atlas - INELIGIBLE

The matched rerun uses the exact server token counts, 7 processes, both phones,
valid thermal endpoints, current binary identities, and same-run placement and
cosine checks. Placement and correctness pass: all rows report HTP execution,
zero missing-buffer nodes, only the declared GET_ROWS CPU exception, and cosine
0.9959-0.9978.

The latency gate fails. Ten of 18 rows have CoV above the frozen 0.05 limit.
Therefore the phone BGE rows are not scheduler-eligible and do not fill the
catalog. `validate_cp_b_result.py` rejects the artifact as intended.

Artifact: `cp_b_phone_bge_result_v2.json`.

## CP-C: serial three-device Gemma route - mechanics only

The route OP15 `[0,8)` -> OP12 `[8,12)` -> A6000 `[12,48)` is token-correct in
single acquisitions at B={1,4,8,32}. Its measured p50 request walls are 1.53,
2.29, 3.98, and 8.42 seconds. B16 and B64 did not complete and were manually
interrupted. This screen did not predeclare a per-case timeout, so it is
diagnostic evidence rather than a repeatability certificate.

A subsequent mixed B32 acquisition completed one P2 cohort but failed during
P2 repeat 1. The completed cohort already shows the critical-path problem:

| metric | P0 full server | P2 serial phones | treatment/control |
|---|---:|---:|---:|
| selected-GPU cohort energy | 1271.48 J | 1643.85 J | 1.293x |
| Gemma p95 latency | 0.610 s | 8.433 s | 13.83x |
| concurrent overlap | 0.623 s | 5.048 s | n/a |

Repeat 1 then failed before `DRIVER_READY`; OP15 reported an unexpected EOF
after the host was stopped and OP12 did not finish. No energy or scheduler claim
is authorized from this incomplete acquisition.

Artifacts: `cp_e_batch_cert_result.json` and
`cp_e_live_priority_b32_result.json`.

## CP-D: selected-GPU counterfactual - diagnostic only

The repaired raw replay compares full Gemma execution with an imported
phone-head counterfactual. It contains equal declared work, raw NVML samples,
artifact identities, and an idle second GPU. The independent validator
recomputes an 11.854 percent selected-GPU paid-window reduction.

This is not an end-to-end offload result: the phones do not execute in the paid
run, the BGE and Gemma legs are sequential, and phone/transport cost and
critical-path delay are absent. It is a ceiling that motivates a live test, not
evidence that the system saves energy.

Artifact: `cp_d_result_v3_repaired.json`.

## Decision

The current evidence validates individual mechanisms, not the mixed scheduler:

- server BGE profiling and policy knee selection work;
- phone BGE correctness and placement work, but timing is unstable;
- the serial two-phone Gemma chain is too slow and not repeatable;
- the 11.854 percent GPU-board number is a counterfactual ceiling only.

The next live route follows the design contract: test the independent OP15
`[0,8)` island first. If repeatable, run it concurrently with selected-GPU BGE.
OP12 must execute a separate READY island or independent request stream; it must
not sit on the same request's critical path until a serial route beats
independent placement. The server never waits for an unready phone result.

### Independent OP15 batch screen

The current-source OP15 `[0,8)` -> A6000 `[8,48)` route was screened against a
full-model reference at the same batch and prompt:

| batch | placement | exact token stream | p50 wall | decision |
|---:|---|---|---:|---|
| 1 | PASS | PASS | 1.281 s | eligible point |
| 4 | PASS | FAIL | 1.343 s | reject |
| 8 | PASS | FAIL | 1.900 s | reject |
| 32 | PASS | FAIL | 2.840 s | reject |

The failed rows produce coherent but different greedy paths; no activation,
logit, or task-quality certificate currently authorizes treating those paths as
equivalent. Exactness is therefore retained as the fail-closed gate. The
`stageb_headcert.py` harness previously returned exit 0 on a token mismatch; it
now emits `HEAD_SWEEP_FAIL`, returns exit 2, requires the complete request set,
and rejects undeclared CPU placement.

The live runtime now accepts `CertifiedBatchPoint`, not a bare latency point.
Each dispatchable batch carries explicit correctness and placement certificate
IDs. This prevents a B1 certificate from authorizing B4/B8/B32.

## CP-F: live independent OP15 priority run - FAIL_GATE

The B1 point was rerun in 7 independent processes with 8 measured requests per
process. All 56 requests match the full-model token stream, all placement gates
pass, HMX endpoints stay within 27.5-30.6 C, median p50 wall is 1.254 s, and
process CoV is 0.0135. The content-bound adapter exposes only B1 to the runtime.

The live mixed experiment then used one selected A6000, live OP15 `[0,8)`, and
three rotated P0/P2 repeats. Each cohort completed the same 50,928 BGE encodes
and 64 Gemma tokens. BGE kept the selected GPU busy while the low-priority
Gemma stream ran concurrently.

| metric | P0 server-only | P2 live OP15 | treatment/control |
|---|---:|---:|---:|
| selected-GPU board energy, median | 3931.55 J | 3718.18 J | 0.946x |
| high-priority BGE p95, median | 14.375 ms | 14.379 ms | 1.0003x |
| low-priority Gemma p95, median | 0.441 s | 1.163 s | 2.639x |
| minimum overlap | 3.517 s | 8.978 s | n/a |

Selected-GPU energy relief is 5.43 percent and the high-priority 1.05x SLO gate
passes. The predeclared low-priority 2.0x p95 gate fails at 2.639x, so the
overall verdict is `LIVE_OP15_FAIL_GATE`. This is the first live evidence that
the mechanism can save server-board energy under useful concurrent work, but it
does not validate the scheduler objective at the current SLO and cannot establish
total-system energy saving.

`validate_cp_f_result.py` independently replays raw NVML integration, all six
cohorts, matched work, latency percentiles, overlap, placement, thermals,
artifact hashes, the seven-process profile binding, and the final verdict.

The result is historical evidence, not a currently replayable live-path
bundle. CP-F recorded the host binary by its mutable `build-cuda/bin` path.
The later shared-tail build replaced that binary, and the original CP-F bytes
were not snapshotted. The validator now correctly rejects the changed path;
the measured result was not re-pinned. New acquisitions must copy every
executed binary into an immutable artifact directory before recording hashes.

## CP-G: independent OP12 route profile - PASS

OP12/v75 was screened as a separate request lane, not behind OP15. A single
`[0,8)` point passed, but its predeclared seven-process cohort failed when the
third process connected and then timed out in first compute. It is rejected;
later successes were not selected to hide the failure.

The smaller `[0,6)` B1 point passes all seven independent processes and all 56
measured requests. The adapter reopens and hashes the raw phone and host logs,
recomputes every token comparison and timing median, checks exact HTP0 placement
with only GET_ROWS on CPU, verifies non-overlapping process identities, and
requires one matched configuration and binary set.

| metric | OP12 `[0,6)` |
|---|---:|
| route-wall p50, median of processes | 1.151677 s |
| phone-stage p50, median of processes | 0.874626 s |
| route-wall process CoV | 0.02582 |
| placement per process | 13,392 HTP0 + 72 CPU GET_ROWS |
| end thermal range | 38.4-39.9 C |
| exact requests | 56/56 |

Artifact: `op12_k6_b1_7proc_profile.json`. This authorizes an independent B1
route only. Phone energy and total-system energy remain unknown, and no
two-phone live mixed result is claimed yet.

## CP-H: parallel phone heads with one shared server tail - POINT PASS

`layersplit.cpp` now has an opt-in `--parallel-heads` path. OP15 and OP12 each
execute a B1 `[0,6)` head on independent sockets and threads. Their returned
activations enter one CUDA context, so the A6000 keeps one `[6,48)` weight copy.
The legacy serial `--port2` route is unchanged.

Two tail modes were screened on both phones:

| phone heads | shared tail | wall for 2 requests | exact full-model oracle | decision |
|---|---:|---:|---|---|
| 2 x B1 concurrent | B2 | 1.13-1.20 s | FAIL on both streams | reject |
| 2 x B1 concurrent | B1 serial in one context | 1.34-1.35 s | PASS on both streams | mechanics point only |

All completed runs pass OP15 HTP0, OP12 HTP0, CUDA0 tail, missing-buffer, and
thermal gates. The B2 mismatch is repeatable and is not treated as approximate
correctness. The exact B1-tail mode proves the desired one-copy topology and
real concurrent phone execution.

It is not scheduler-eligible yet. After several model unload/reload cycles, a
later B1 audit connected both stages and then timed out in first compute. The
pre-fix timeout path also failed to persist a complete JSON result; the harness
now converts future host timeouts into `PARALLEL_HEADS_FAIL_MEASUREMENT`.
Persistent stagenet sessions that detach clients without unloading weights are
the next gate. No CP-H energy measurement was run.

Artifacts: `parallel_heads_final_b2.json` plus the passing
`parallel_heads_b1_audit.json` and their per-run logs. Phone and total-system
energy remain unknown.

## CP-H persistent-session gate (2026-07-18): PASS (mechanics only)

The named next gate -- persistent stagenet sessions that detach clients without
unloading weights -- now passes on real devices.
`PERSISTENT_SESSION_MECHANICS_PASS_PHYSICAL_ENERGY_NOT_RUN`. Full write-up in
`../persistence/RESULTS.md`; report `../persistence/results/gate_report.json`;
immutable artifacts + `SHA256SUMS.txt` in `../persistence/artifacts/`.

Added an opt-in, versioned session protocol to `examples/layersplit/layersplit.cpp`
only (no llama-graph / ggml_backend_sched / kernel edits): `STAGE_DETACH=-7`
resets request-local KV, emits a per-session `SESSIONCERT` (v2), acks, closes only
the client, re-`accept()`s, and keeps weights/backends resident; `STAGE_STOP`
drains + terminates (byte-unchanged); host `--session-end detach|stop`. The HELLO
wire response is byte-identical (v1) so legacy and serial/parallel routes are
unaffected.

Real-device gate: two resident `stagenet [0,6)` workers (OP15 v81 + OP12 v75,
HTP0, `12b-f16-head-0-6.gguf`) + host parallel-head shared tail `[6,48)` on the
selected A6000; 7 sequential B1 sessions, 6 DETACH + 1 STOP (Tail B2 NOT retried).
`certified:true`: one resident worker pid/nonce per phone across all seven sessions
(terminates only on STOP, `exit after 182 steps`); per-stream token ids identical
across all seven sessions; every session cert SCHEDULED_PLACEMENT_OK,
missing_buffer=0, HTP0-only + declared GET_ROWS on CPU. A separate wire client
proves the head hidden-state bytes are bit-identical across the seven sessions on
both phones (tail-independent reset-exactness). Deployed to
`/data/local/tmp/ls-s14-persistent/` (ls-s14-cpe untouched). Phone and
total-system energy still unknown and NOT measured -- energy is downstream of
this mechanics gate.
