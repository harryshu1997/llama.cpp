# Corrected persistent-session real-device gate

Verdict: `PERSISTENT_SESSION_REAL_GATE_PASS`

Date: 2026-07-18. Scope: correctness, scheduled placement, worker persistence,
thermal validity, and route-latency repeatability on one A6000, OP15, and OP12.
Phone energy, USB energy, and total-system energy are unknown.

## Why this rerun was required

Review of `../persistence/` found four evidence problems and two certificate
semantics defects:

- `compute_by_op_and_buffer` accumulated across sessions instead of describing
  one session.
- the terminating STOP certificate claimed that a KV reset had occurred.
- the host executable and shared libraries were used from a mutable build tree.
- the acquisition script was copied into the artifact directory after the run.
- routed output was compared only across sessions, not against a full-model CUDA
  reference for the same prompt and generation length.
- the old gate had no thermal or latency-variation bound.

The historical bytes and verdict are unchanged. They remain narrower mechanics
evidence and are not the evidence for this verdict.

## Repairs

`examples/layersplit/layersplit.cpp` now clears the placement tally only after a
successful DETACH acknowledgement. The following session therefore starts with
an empty tally. The certificate has an explicit `reset_applied` field: DETACH is
`true`; STOP and error termination are `false`.

The corrected acquisition uses only the read-only `artifacts/` bundle. The host
executable, its shared libraries, Android executable, Android libraries, and HTP
skels were copied and hashed before the run. The harness and artifact-manifest
digests are embedded in `results/gate_report.json` and rechecked after the run.

## Physical run

- Devices: selected A6000 GPU, OP15 HTP0, and OP12 HTP0.
- Route: two parallel phone heads `[0,6)` and one shared CUDA tail `[6,48)`.
- Sessions: seven sequential B1 route sessions, with DETACH for sessions 1-6
  and STOP for session 7.
- Output: 14 of 14 routed streams exactly match the 16-token A6000 monolithic
  reference.
- Persistence: each phone retains one worker PID and boot nonce for all seven
  sessions; both workers terminate after STOP.
- Reset/accounting: session IDs are 1-7; every session has 26 steps; DETACH
  certificates report reset, while STOP certificates do not.
- Placement: every session is `SCHEDULED_PLACEMENT_OK`, has zero missing-buffer
  compute nodes, uses HTP0 for phone compute, and uses CPU only for the declared
  `GET_ROWS` operation.
- Repeatability: route-wall times are 2.654-2.800 seconds, with CoV 1.77% against
  a frozen 5% gate.
- Thermals: OP15 was 28.7 C at start and 31.0 C at end; OP12 was 30.6 C at start
  and 32.1 C at end.

Independent replay:

```sh
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 validate_report.py
```

Expected result:

```text
VALID_PERSISTENCE_V2 sessions=7 streams=14 cov=0.017723
```

## Claim boundary and next integration

This gate proves that the persistent physical route works and produces exact
tokens without reloading phone workers between requests. It does not prove that
the route saves energy or improves latency; its measured route wall is about
2.67 seconds per two-stream session.

The C++ worker emits `ls-stagenet-session-v2`. S15's typed physical executor
currently consumes `s15-physical-session-v1`, so the live launcher/adapter that
binds these session certificates to S15 route, lease, residency, and boot epochs
is still required before mixed-trace physical dispatch can be claimed.
