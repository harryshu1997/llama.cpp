# Transport Contract (S9-V0)

Status: frozen historical S9-V0/v1 transport contract. The v1 frame descriptor
is `schemas/transport_frame.schema.json`; current R1 validation uses the versioned
v3 bundle. This document specifies a research wire contract and does not claim a
production transport implementation or override `TWO_LEVEL_SCHEDULER.md`.

Do NOT reuse LayerSplit or ggml-rpc framing UNCHANGED. Both are REJECT for a
control/data plane (SUBSTRATE_AUDIT D4-D6; the S8 audit B2-B6 for LayerSplit):
LayerSplit is host-endian, lockstep, unversioned, unbounded, with heap-overflow
and OOM read paths and fail-by-process-exit; ggml-rpc is host-endian pack(1)
memcpy with no frame magic/version/checksum, an uncapped `recv_msg` (allocation
DoS), 1 GiB blocking chunks with no timeout, no request ids, and `GGML_ABORT` on
any fault. The socket/length-prefix SHAPE is reference material; the wire contract
below is new. The physical endpoint ceiling is USB 3.2 Gen 1 = 5 Gbps raw on both
OP12 and OP15 regardless of a 20 Gbps cable, so goodput is the parameter, not the
label (see the goodput sweep in SIMULATOR_SPEC.md).

## 1. Two channels

Two logical channels over the link, so a long weight transfer never blocks
control/liveness (fixes ggml-rpc's shared single opcode stream, SUBSTRATE_AUDIT D5):

- CONTROL channel: small, versioned, reliable, ordered. Carries HELLO, CACHE_QUERY,
  FETCH, VERIFY, LOAD, WARM, READY, LEASE, RELEASE, ALLOC_STATE, FREE_STATE,
  RESET_STATE, EXECUTE, RESULT, CREDIT, HEARTBEAT, DRAIN, ERROR, CLOSE. Frame
  payload capped at 65536 bytes.
- BULK channel: weight-chunk transport only (`msg_type==BULK_CHUNK`). Frame payload
  capped at 16 MiB. Every bulk frame carries a `payload_sha256`.

Both are `transport_frame` instances; the schema's `channel` conditional enforces
the caps and the control/bulk separation.

## 2. Framing (bounded, versioned, checksummed, little-endian)

Every frame is length-delimited and self-describing:

```
magic "S9WR" | protocol_version | channel | msg_type | priority_class
header_len (<= 65536) | payload_bytes (<= 16 MiB; control <= 65536)
header_crc32 (CRC-32 of the header) | payload_sha256 (bulk: required; control: null-able)
request_id | batch_id | seq | idempotency_key
boot_epoch | residency_epoch | route_epoch | state_epoch
deadline_us | cancellable
```

Rules:
1. Little-endian on the wire, explicitly (never host-endian memcpy of packed
   structs). Numbers are fixed-width LE.
2. `header_crc32` covers the header; a bad CRC is a framing error -> the frame is
   dropped and an ERROR is raised on CONTROL. `payload_sha256` covers the bulk
   payload; a mismatch fails the chunk closed (no partial apply).
3. `payload_bytes` has a HARD schema maximum (16 MiB); a frame claiming more is
   rejected before any allocation (fixes the ggml-rpc uncapped `recv_msg` DoS).
   Receive buffers are bounded to the channel cap; there is no unbounded resize.
4. A frame that fails to parse is rejected fail-closed; it never aborts the process
   (fixes ggml-rpc `GGML_ABORT`, SUBSTRATE_AUDIT D6).

## 3. Chunk hashes + resumable verified ranges

- A weight segment is transferred as the ordered `chunks[]` of its WeightSegment
  (`weight_segment.schema.json`); each BULK_CHUNK frame carries the chunk bytes and
  the chunk's expected `payload_sha256`. A chunk whose bytes do not hash to it is
  discarded and re-requested (idempotent).
- Resume is VERIFIED, not offset-trusting (fixes SUBSTRATE_AUDIT C1): a TransferTicket
  carries `resumable_from_verified_offset` and `resume_partial_sha256`
  (SHA-256 of bytes `[0, offset)`). On reconnect the phone re-hashes its on-disk
  prefix and matches it against `resume_partial_sha256` before appending; a mismatch
  truncates and restarts (no full-GET deadlock).
- VERIFYING completes only when every segment's full bytes hash to
  `WeightSegment.sha256` and the WeightSet's `set_digest` recomputes.

## 4. Staging + durability + atomic publication

RECEIVING writes into a staging path; VERIFIED_ON_DISK is durable (fixes the total
absence of fsync, SUBSTRATE_AUDIT C2/C6):

```
write chunk -> fflush -> fsync/fdatasync(staging fd) -> checked close(staging)
-> [VERIFYING passes] -> rename(staging, published) -> open(parent dir) -> fsync(dir fd)
```

- The content-hash gate (VERIFYING) runs BEFORE the rename; only verified bytes ever
  become durable under the published name.
- The prior good image is retained until the new one reaches VERIFIED_ON_DISK
  (stage-verify-swap), then the old generation is reclaimed. Never delete a good
  image before its replacement is verified (fixes the ETag delete-before-redownload
  data-loss hazard, SUBSTRATE_AUDIT C4).
- A VERIFYING failure moves the payload to QUARANTINE, never re-appended into the
  same name.

## 5. Flow control (credits + bounded buffers)

- Credit-based flow control: the receiver advertises byte/slot credits via CREDIT
  frames; a sender transmits only against a credit. This bounds every queue (fixes
  the unbounded server deque and missing admission, S8 audit A7).
- A TransferTicket carries `credits_bytes`; bulk transmission consumes credit and
  stalls when starved. Credits return only after output ownership passes back to the
  transport (MIXED_WORKLOAD_DESIGN section 7).
- Every lane (CONTROL rx, BULK rx, HTP work, GPU work, state slots) has an explicit
  bound; overflow sheds (drop/defer) rather than growing unbounded.

## 6. Priority: activation/result preempts background weight prefetch

Preemption is encoded in `priority_class`, ordered high-to-low:
`control >= activation >= result > bulk_weight_background`.

- A BULK_CHUNK frame is always `bulk_weight_background` (schema const). An EXECUTE or
  RESULT frame is `activation` or `result` (schema `if/then`). Therefore live
  activation/result traffic ALWAYS outranks background weight prefetch on the wire:
  a burst of activations preempts an in-progress weight transfer.
- The scheduler admits a bulk chunk only when no higher-priority frame is pending on
  that link; a large MATERIALIZING transfer never head-of-line-blocks a decode
  activation. This is a required simulator behavior and a required test (activation
  traffic preempts bulk prefetch).

## 7. Epochs + idempotency + mutation sequence (fail-closed)

Every frame carries the epoch stack (`boot_epoch`, `residency_epoch`, `route_epoch`,
`state_epoch`), a per-(connection, channel) monotone `seq`, and an `idempotency_key`.
The receiver rejects, fail-closed:

- DUPLICATE: a frame whose `seq` was already processed, or whose `idempotency_key`
  was already applied to a mutating op -> no-op (a retried FETCH/ALLOC_STATE is not
  double-applied).
- REORDERED: a `seq` gap on a channel -> reject + request resync (TCP does not
  reorder within a connection, so a gap means a dropped/split connection).
- STALE-EPOCH: any epoch older than the receiver's current value -> reject with
  `reason_code: stale_epoch`. A rebooted phone (`boot_epoch` bumped) or a reloaded
  residency (`residency_generation` bumped) invalidates every in-flight frame bound
  to the old epoch.
- Mutation sequence: at most ONE mutation per state object may be in flight
  (`state_lease.in_flight_mutation_seq`); a second concurrent mutation is rejected.

## 8. Deadlines, cancellation, reconnect

- Bounded send/recv timeouts on both channels (fixes the ggml-rpc no-timeout hang,
  SUBSTRATE_AUDIT D6). A frame carries an optional `deadline_us`; past it, the op is
  cancelled.
- `cancellable` frames may be aborted (e.g. a background prefetch superseded by an
  eviction); an in-flight kernel is not force-interrupted but its lane is drained.
- HEARTBEAT detects a silently dead peer; a missed-heartbeat window marks the phone
  degraded and ineligible. Reconnect resumes verified transfers idempotently (section
  3); an interrupted RECEIVING resumes, it is not lost.

## 9. Drain-before-eviction + backend-fence completion

- A residency lease transition to a new generation, or an eviction, goes through
  DRAIN: the phone stops accepting new dispatch for that residency, finishes
  in-flight work at a replayable boundary, and issues a backend FENCE (HTP/GPU
  queue drained; SUBSTRATE_AUDIT E5 shows the current clFinish-only coherency is not
  a fence). EVICTING begins only after the DRAIN + fence complete AND no StateLease
  depends on the old generation (WEIGHT_RESIDENCY_CONTRACT section 5).
- Normal lease changes happen after drain at a replayable boundary; hot KV migration
  is not an initial mechanism.

## 10. Control operations (verbs)

`HELLO` (exchange + PIN protocol version, backend build, SoC, arch, layout version,
boot epoch -- extending the ggml-rpc HELLO, SUBSTRATE_AUDIT D3), `CACHE_QUERY`
(what is already VERIFIED_ON_DISK / READY), `FETCH` (issue a TransferTicket),
`VERIFY`, `LOAD` (MATERIALIZE), `WARM`, `READY` (return a ReadyCertificate),
`LEASE`/`RELEASE` (residency), `ALLOC_STATE`/`FREE_STATE`/`RESET_STATE` (request
state), `EXECUTE`/`RESULT` (activation path), `CREDIT`, `HEARTBEAT`, `DRAIN`,
`ERROR` (recoverable, never abort), `CLOSE`. Every verb is idempotent under its
`idempotency_key`; every response carries the epoch stack for stale rejection.

## 11. What this contract does NOT do

- No weight bytes on the activation path: weights move on BULK before READY; the
  EXECUTE/RESULT path carries only boundary tensors (MIXED_WORKLOAD_DESIGN section
  10; DESIGN section 2.0).
- No cross-endian guarantee is assumed from a raw memcpy; the wire is explicitly LE.
- No unbounded allocation, no process abort on a peer fault, no lockstep positional
  framing. Those are exactly the REJECT properties of the two prior transports.
