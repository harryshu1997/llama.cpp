# S15 persistent live-launcher plan

Status: `PERSISTENT_LIVE_ADAPTER_MECHANICS_PASS_PHYSICAL_EXECUTION_NOT_RUN`.

## Purpose

Connect the strict persistent transport and LayerSplit session adapter to the
unchanged S15 `PhysicalExecutor`. The bridge translates a rich, evidence-bound
outer request into the six-field persistent LayerSplit child command. It does
not use ADB, sockets, a device, or a recorded-success executor.

This checkpoint is mechanics-only. `deadline_us` is a relative per-launch
timeout. It is not an absolute trace deadline or a measured service-level
objective.

## Contract

The outer `s15-persistent-execute-v1` record binds:

- launch id, prompt, generation length, exact batch and request ids;
- DETACH or STOP session end and the relative launch timeout;
- route, profile, evidence, device, boot, worker binary, and layer range;
- route, residency, lease, device-boot, and registry epochs;
- cohort and input-manifest digests.

The bridge sends only the exact `layersplit-persistent-command-v1` command to a
single child process. It admits a result only after receiving one exact child
result, one strict phone `ls-stagenet-session-v2` certificate, one strict host
tail `layersplit-scheduled-placement-v2` certificate, and the exact launch-bound
child stderr exchange-end marker. DETACH requires the child to remain alive;
STOP requires a clean child exit.

The current C++ driver readiness is the prefixed
`layersplit-persistent-driver-v1` record and its result includes both
`elapsed_us` and `route_wall_us`; both are required and bound. Its prompt limit
is 16 KiB, so the outer contract uses the same limit.
The bridge config carries an integer `child_ready_timeout_s` in `[1,600]` so
physical model and shard loading can use a measured setup budget without a
hardcoded 30-second failure. The enclosing transport timeout must be at least
that value.

The host placement certificate must cover the exact complementary tail range,
report positive compute exclusively on CUDA0, have zero missing buffers, and
agree with the ready host PID. The phone session certificate remains a
noncanonical prefixed record and retains its existing duplicate-key, exact-key,
type, placement, reset, and identity checks.

The outer `s15-persistent-result-v1` binds the terminal state, timing, child
PID, token digest, route identity, and hashes for eight exact artifacts. The
host adapter reopens each artifact once through a regular-file descriptor,
rejects symlinks and path escape, requires the launch-specific path, and
recomputes every digest before constructing completion boundaries.

The resident worker lease becomes reusable only after both the existing
`StageNetSessionAdapter` and unchanged `PhysicalExecutor` accept the same
evidence. A failure at either layer poisons the process and cannot complete a
request.

## Gates

- [x] Audit and repair the lower persistent transport stderr boundary.
- [x] Freeze exact rich outer and minimal child JSONL v1 contracts.
- [x] Implement a generic persistent-child bridge without device I/O.
- [x] Reuse `StageNetSessionAdapter`; do not duplicate its certificate policy.
- [x] Bind command, result, stdout, stderr, phone certificate, host placement,
      token, and phone placement artifacts by exact launch-specific path and
      SHA-256.
- [x] Pin both child host PID and resident worker PID/nonce across sessions.
- [x] Commit DETACH reuse only after unchanged `PhysicalExecutor` acceptance.
- [x] Cover malformed, duplicate, partial, cross-talk, timeout, identity,
      epoch, placement, token, artifact, symlink, and lifecycle failures.
- [x] Pass the new suite under ten `PYTHONHASHSEED` values.
- [x] Pass the lower persistent, S15 runtime, and one-shot live suites.
- [x] Stop before device execution.

## Physical next gate

The host C++ driver does not receive the phone worker's `SESSIONCERT`; that
record is emitted by the separately launched ADB StageNet process. Therefore a
direct `child_command=[llama-layersplit ...]` is not physically executable with
this bridge. An adjacent physical mux now implements ownership of both the ADB
worker and host driver and forwards the phone certificate plus host placement,
result, and marker. That mux was not executed by this checkpoint; wiring and
physical validation remain separate gates.

Next configure the bridge with that mux and the measured epoch-14 OP15 B32
route, then run consecutive DETACH sessions followed by STOP on the real phone. The run must use
exact expected token digests and retain all bridge, transport, placement, and
certificate artifacts. It must report setup, per-session wall time, route wall
time, batch size, and failure counts. It must not claim energy because neither
phone nor total-system energy is measured.
