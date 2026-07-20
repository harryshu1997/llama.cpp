# S15 persistent-session protocol: real-device results

Verdict: `PERSISTENT_SESSION_MECHANICS_PASS_PHYSICAL_ENERGY_NOT_RUN`

Date 2026-07-18. No commit, no push. ASCII only. Scope: the LayerSplit persistent
phone-worker protocol, its Android deployment, and its real-device repeatability
evidence. Energy is NOT measured. Only `examples/layersplit/layersplit.cpp` was
edited; `llama-graph`, `ggml_backend_sched`, and every kernel file are unchanged.

## 1. What was added to the protocol (opt-in, versioned, additive)

`examples/layersplit/layersplit.cpp` only.

- New opcode `STAGE_DETACH = -7`. On DETACH the resident `stagenet` worker resets
  request-local state (`llama_memory_clear`), emits a session-scoped placement
  certificate (`SESSIONCERT`), sends the int32 reset ack, closes only the client
  socket, re-`accept()`s on the same listening socket, and keeps all weights and
  backend contexts resident. `STAGE_STOP = -1` drains and terminates (unchanged).
- A per-session certificate `SESSIONCERT {schema:"ls-stagenet-session-v2",
  proto_version:2, session_id, session_end, expected_backend, worker_pid,
  worker_boot_nonce, device_boot_id, layer_start/end, n_layer, steps_session,
  placement_status, missing_buffer_compute_nodes, compute_by_op_and_buffer}` is
  emitted on every DETACH and on the terminating STOP/EOF. It travels over stderr
  like the existing `PLACEMENTCERT`.
- Host driver: `run_parallel_head_driver` gains a `--session-end detach|stop`
  option (default `stop`). `detach` ends each session with DETACH + ack; `stop`
  is the byte-unchanged legacy path.

Protocol version: the wire `STAGE_HELLO` response is left byte-identical
(magic `LST2`, version 1) so legacy clients and the current serial/parallel
routes are unaffected; the persistence version (2) is advertised only inside the
session certificate. DETACH is a pure superset opcode a legacy client never
sends.

## 2. Build and deployment (frozen, immutable)

- Host binary rebuilt from the current working tree: `build-cuda/bin/llama-layersplit`
  (clean compile, only pre-existing unused-parameter warnings).
- Android arm64 binary rebuilt from the current working tree via the snapdragon
  docker toolchain (`npu-harness/scripts/build_npu_op12.sh --force`). The arm64
  binary is identical for both phones; only the DSP skel differs (v75 op12,
  v81 op15).
- Every executed binary, skel, lib, and script is frozen into
  `persistence/artifacts/` and hashed in `persistence/artifacts/SHA256SUMS.txt`
  BEFORE use. Evidence is bound to that immutable directory, never to a mutable
  `build-*/bin` path. Frozen worker binary
  `sha256:d26075bcf64e90ee04e51c2f86188d709e4c2906add252d209014ad1e046646c`.
- Deployed to `/data/local/tmp/ls-s14-persistent/` on both phones; the on-device
  binary and skel sha256 equal the frozen sha256. `/data/local/tmp/ls-s14-cpe/`
  was not touched.

## 3. Real-device gate

Two resident `stagenet [0,6)` workers (OP15 Hexagon v81 + OP12 Hexagon v75, HTP0,
`12b-f16-head-0-6.gguf`) plus a host `pipedriver --parallel-heads
--parallel-tail-batch 1` shared tail `[6,48)` on the selected A6000. Seven
sequential B1 host sessions; the first six end with DETACH, the seventh with STOP.
Tail B2 was NOT run (frozen negative). `run_persistence_gate.py`, report in
`results/gate_report.json`.

Result `certified: true`, `problems: []`:

- **Protocol compatibility**: all seven host sessions completed (`rc=0`, 16 tokens
  per stream). Session 7 is the legacy HELLO/RESET/PREFILL/DECODE/STOP sequence
  and terminates the workers exactly as before; the six DETACH sessions keep them
  resident. The same host driver drives both ends.
- **Persistence**: one resident worker served all seven sessions on each phone --
  OP15 pid 23633 nonce b7e707b1, OP12 pid 26896 nonce d8e331c7, constant across
  all seven session certs. Each worker terminated only on the final STOP
  (`[stagenet] exit after 182 steps`); no reload between sessions.
- **Reset exactness**: per-stream generated token ids are identical across all
  seven sessions (7 sessions -> 1 unique 16-token sequence per stream, both
  streams). OP15 and OP12 produced the same token stream for the same prompt,
  cross-validating the two heads.
- **Placement**: every session cert is `SCHEDULED_PLACEMENT_OK` with
  `missing_buffer_compute_nodes = 0`, layer range `[0,6)`, backend HTP0, and only
  the declared `GET_ROWS` (f16 token_embd) on CPU.
- Session ids contiguous 1..7; session-end sequence DETACH x6 then STOP; 7
  SESSIONCERT lines per worker log.

## 4. Isolated worker reset-exactness (tail-independent)

To separate worker reset exactness from any A6000 tail non-determinism, each
worker was also driven directly by a wire client (`artifacts/wire_reset_check.py`)
through 6 DETACH + 1 STOP sessions with a fixed decode sequence. The raw head
hidden-state bytes are bit-identical across all seven sessions on both phones
(OP15 v81 digest `bada7465...`, OP12 v75 digest `0d5bf42e...`), same worker pid
and nonce throughout. This is the definitive reset-exactness proof; the end-to-end
token identity in section 3 corroborates it.

## 5. Claim boundary

MECHANICS only. This proves protocol compatibility, weight-resident persistence
across sessions, and exact per-session reset on real devices. It does NOT measure
phone energy, server energy, latency, throughput, or any total-system benefit,
and it does not certify the shared-tail route's numerical correctness against a
mono reference (that is a separate CP1 gate; Tail B2 remains a frozen negative and
was not retried). Energy is not run because that is downstream of this gate.
