# S15 persistent runtime results

Verdict: `PERSISTENT_TRANSPORT_MECHANICS_PASS_PHYSICAL_B32_NOT_RUN`

Date: 2026-07-18. Offline only. No device command, model graph, backend,
scheduler, kernel, C++, frozen physical artifact, talks log, memory file, commit,
or push was produced by this checkpoint.

## Implemented result

The persistent transport and certificate adapter are integrated at the existing
typed S15 boundary. An offline launcher fixture executed two requests through
one subprocess. Each request was closed by a distinct, contiguous DETACH
certificate, adapted to `s15-physical-session-v1`, and accepted by the unchanged
`PhysicalExecutor`. The adapter did not release reuse until reset, ACK,
placement, worker identity, and completion boundaries all passed.

New suite: 29/29 pass. The transport now requires an exact launch-bound stderr
exchange-end marker, assigns stderr bytes only through that marker, and rejects
missing, wrong, duplicate, or trailing markers. Launch ids must be contiguous,
not merely increasing.

Covered failures include duplicate/stale/gapped certificates, changed PID,
nonce, or boot id, wrong range/backend, missing buffers, undeclared CPU fallback,
reset false on DETACH, missing/incorrect ACK, missing cert, ERROR/EOF/wrong end,
boundary failure, absent or foreign persistent capability, duplicate/late/
partial/missing replies, request/reply cross-talk, timeout, noncanonical input,
missing/wrong/duplicate stderr boundaries, skipped launch ids, finalize during
an active exchange, and reuse after poison or STOP.

## Existing-suite status

- S15 live launcher: 26/26 pass.
- S15 persistent runtime: 29/29 pass.
- S15 runtime dispatch: 79/79 pass against the stabilized epoch-14 tree.

## Claim boundary

No physical B32 request was launched through this new transport/adapter. The
separately frozen epoch-14 resident-worker gate was produced outside this
checkpoint and was not edited or relabeled. No integrated second-session
latency, weight-residency benefit, energy, throughput, thermal, or total-system
claim is authorized. The old one-shot physical result and all frozen physical
artifacts are unchanged. The claim here is limited to framing, certificate,
capability, reset, and typed-adapter mechanics passing offline adversarial tests.
