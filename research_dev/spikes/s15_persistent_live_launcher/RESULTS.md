# S15 persistent live-launcher results

Verdict: `PERSISTENT_LIVE_ADAPTER_MECHANICS_PASS_PHYSICAL_EXECUTION_NOT_RUN`

Date: 2026-07-18. Offline only. No device command, model execution, C++ edit,
talks update, memory update, commit, or push was made by this checkpoint.

## Result

The bridge and typed host adapter completed DETACH then STOP through one generic
persistent child process. The first accepted session reused the process only
after the existing certificate adapter and unchanged `PhysicalExecutor` both
accepted it. The second accepted session observed the same child PID and worker
PID/nonce, a contiguous session id and step total, and a clean STOP.

The bridge preserves the exact minimal LayerSplit child command and result
contracts while the outer record carries the scheduler identities and evidence
digests. It parses and binds the current C++ prefixed readiness record and its
required `route_wall_us`. Eight artifacts are launch-specific and digest-bound.
The adapter checks the exact child command, result, token rows, child stdout,
child stderr, phone session certificate, host tail placement certificate, phone
placement projection, timing, terminal state, and every artifact digest before
producing any completion boundary.

## Verification

- Persistent live-launcher: 19/19 pass.
- Persistent live-launcher repeated with `PYTHONHASHSEED=0..9`: all pass.
- Lower persistent transport/session adapter: 29/29 pass.
- S15 runtime dispatch: 79/79 pass.
- Existing one-shot S15 live launcher: 26/26 pass.

Adversarial coverage includes unknown and changed launch/request identities,
deadline mutation, changed child PID, changed worker PID, gapped session ids,
reset failure, token mismatch, command mutation, encoded C++ command overflow,
cross-launch artifact reuse, symlink artifacts, nested integer-to-float
mutation, missing/wrong/duplicate/
trailing stderr markers, malformed or partial results, duplicate stdout,
undeclared CPU fallback, missing or duplicate host placement, host CPU/CUDA1
placement, host tail range mismatch, a digest-consistent placement artifact
mutation, noncanonical phone certificate acceptance, duplicate phone
certificate keys, boolean/zero/overflow child readiness timeouts, and failure
after the certificate adapter but before
`PhysicalExecutor` acceptance.

## Audit repairs

The lower transport previously used stdout arrival plus unconstrained stderr
buffer snapshots. It now requires a launch-bound stderr exchange-end marker and
rejects missing or cross-talk boundaries. It also rejects skipped launch ids.

The initial live adapter did not pin the child PID, did not require each
artifact to live in its launch directory, and followed artifact symlinks before
reading. These are now load-bearing checks with regressions. Artifact bytes are
read once from a regular-file descriptor and all digests are recomputed.

## Claim boundary

This is not a real-device result. The generic fixture proves process reuse,
contract translation, evidence binding, and fail-closed lifecycle mechanics.
It does not prove that the current Android child emits the exact contract, that
B32 completes through this new path, that latency improves, or that weights are
resident on a phone. The separately frozen seven-session worker result does not
exercise this bridge and is not relabeled. The bridge configuration is a trusted
setup input; the physical gate must reuse the existing deployed-binary, device
boot, and thermal preflight instead of treating this offline config as device
evidence. The current host driver and ADB worker are two processes, and only the
ADB worker emits `SESSIONCERT`. An adjacent fail-closed mux now joins those
streams, but this checkpoint did not execute or validate that mux with this
bridge. Physical wiring and execution remain the explicit blockers. No energy,
throughput, latency, SLO, thermal, or total-system claim is authorized.
