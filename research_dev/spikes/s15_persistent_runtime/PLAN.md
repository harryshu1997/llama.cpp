# S15 persistent runtime mechanics

Status: `PERSISTENT_TRANSPORT_MECHANICS_PASS_PHYSICAL_B32_NOT_RUN`.

## Purpose

Replace one-process-per-launch mechanics with a typed, fail-closed persistent
launcher boundary while leaving the frozen one-shot S15 path unchanged. A
resident phone worker may be reused only after its LayerSplit session ends with
a valid `ls-stagenet-session-v2` DETACH certificate, a zero ACK, a proven reset,
and an unchanged worker identity.

This is an offline mechanics checkpoint. It does not authorize a device run or
any latency, energy, throughput, or residency-benefit claim.

## Components

### `persistent_transport.py`

- One prepared launcher subprocess, kept open across multiple exchanges.
- One bounded canonical JSON request line and exactly one correlated canonical
  JSON reply line per exchange.
- Contiguous `launch_id`; a late, duplicate, or skipped prior reply cannot be
  consumed by the next request.
- One exact `LAUNCHER_EXCHANGE_END` stderr marker closes each exchange. The
  marker binds the active launch id and makes per-exchange stderr attribution
  explicit instead of relying on scheduling between stdout and stderr reader
  threads.
- Exact per-exchange request/reply/stdout/stderr/metadata artifacts and whole
  process stdout/stderr/final-state artifacts.
- Explicit `STARTING`, `READY`, `ACTIVE`, `DRAINING`, `STOPPED`, `POISONED`, and
  `FINALIZED` states.
- Timeout, partial reply, duplicate reply, unsolicited output, EOF, stream
  overflow, or cross-talk poisons the transport. No exchange follows poison or
  STOP. Finalization refuses an active exchange.
- Errors derive from the S15 `ExecutorError` hierarchy so the existing
  coordinator can fail the launch and release its route credit.

### `session_adapter.py`

- Strict parser for one prefixed `SESSIONCERT` record with the exact v2 key set.
- Exact protocol, contiguous session id, expected end type, device boot id,
  layer range, total layer count, backend, placement status, and reset checks.
- Pins worker PID and boot nonce on the first accepted session and requires the
  same PID, nonce, and device boot id thereafter.
- Requires HTP0 compute, zero missing buffers, and only explicitly allowed CPU
  metadata operations.
- DETACH requires a zero ACK, `reset_applied=true`, and a still-running worker.
  Reuse remains blocked until `release_after_detach()` is called.
- STOP requires observed worker termination, permits `reset_applied=false`, and
  permanently stops the adapter.
- ERROR, EOF, missing/duplicate certificate, missing ACK, completion-boundary
  failure, or identity mutation poisons the adapter and never permits reuse.
- Persistent capability is bound out of band to an exact worker binary SHA-256;
  STAGE_HELLO v1 is not treated as a capability advertisement.
- A valid worker certificate is not itself an S15 completion. The adapter also
  requires exact request-bound correctness/D2H boundaries, then emits the
  existing `s15-physical-session-v1` record for the unchanged
  `PhysicalExecutor` to validate again.

## Gates

- [x] Frozen one-shot transport behavior left unchanged.
- [x] Persistent bounded framing and process-lifetime state machine implemented.
- [x] Strict v2 certificate/capability/reset adapter implemented.
- [x] Two adapted DETACH sessions pass through one persistent subprocess and the
      unchanged `PhysicalExecutor` with contiguous session ids.
- [x] Required adversarial cases implemented.
- [x] Lower persistent suite passes 29/29.
- [x] Existing S15 live-launcher suite passes.
- [x] Existing S15 runtime suite passes against the stabilized epoch-14 tree.
- [x] Stop for review before a physical B32 run through this transport/adapter.

## Physical next gate

Wire the real persistent launcher to this framed transport and certificate
adapter, then run two consecutive B32 sessions through the typed coordinator
path. The separately frozen epoch-14 worker gate proves seven resident-worker
sessions, but it does not exercise this new launcher transport/adapter boundary
and is not relabeled as this checkpoint's physical result.
