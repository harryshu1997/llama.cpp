# Substrate Audit (S8-V0a-R2)

V0a-R downgrades (per review finding 8): three classifications the first pass
marked REUSE/ADAPT are unsafe as written and are lowered here -- layersplit
`send_all`/`recv_all`/`set_nodelay` REUSE -> ADAPT (no SIGPIPE/deadline/typed
error, B1); dualengine `OverlapLeg`/`OverlapBarrier` ADAPT -> REFERENCE_ONLY
(ignored failure reports done, diagnostic data race, C1a/C2); route2 VQ ring
REUSE -> ADAPT-concept/REFERENCE_ONLY-code (publish-before-write +
overwrite-live-slot, F1). Everything reusable here is a pattern to re-implement,
not code to lift.

Status: file/line-cited audit of the CURRENT-tree and HISTORICAL mechanisms a
mixed-workload host orchestrator could reuse. Read-only; nothing was edited. Each
mechanism is classified REUSE / ADAPT / REFERENCE_ONLY / REJECT. This is
provenance and planning only: no runtime code is ported in V0a, and no model
graph, KV internal, backend scheduler, or kernel is edited.

Central distinction enforced throughout (per PLAN.md and MIXED_WORKLOAD_DESIGN.md
section 6): a process-LOCAL virtual-queue occupancy signal is NOT distributed
model-readiness / lease / cache state. Occupancy is continuous, loss-tolerant,
and staleness-decayable ("how busy is this queue now"). Readiness is discrete,
edge-triggered, and needs reliable ordered delivery ("is model X loaded and
leased to me on device D at what version"). Conflating the two is the specific
error this audit exists to prevent.

Classification key:
- REUSE: adopt largely as-is.
- ADAPT: the shape/pattern is right; specific hardening required.
- REFERENCE_ONLY: study the idea; do not adopt the code.
- REJECT: not suitable; do not adopt.

Repos: current worktree `/home/myid/zs89458/Documents/llama.cpp-release`;
historical VQ fork `/home/myid/zs89458/Documents/llama.cpp` (branch
`route2-b9531`); Unifer telemetry
`/home/myid/zs89458/Documents/Unifer/research_dev/services/lazyvlm/`.

## A. Current llama-server request/queue/slot substrate

Files: `tools/server/server-queue.{h,cpp}`, `server-task.{h,cpp}`,
`server-context.cpp`, `server-http.{h,cpp}`.

### A1. Request ID assignment -- ADAPT
- Per-process monotonic `int id` minted under `mutex_tasks`:
  `server-queue.h:15`, `server-queue.cpp:71-75` (`get_new_id`), also
  `server-queue.cpp:44-46`. Default sentinel `server-task.h:134`.
- GAP: single-process `int`; no host-global correlation id, no route/lease epoch,
  no deadline/credit metadata on `server_task`. A distributed correlation id and
  epochs must be added.

### A2. Task queue -- REFERENCE_ONLY (single-host FIFO)
- Two `std::deque<server_task>` (`queue_tasks`, `queue_tasks_deferred`) with one
  mutex + one condvar: `server-queue.h:22-26`; post `server-queue.cpp:22-61`;
  single-consumer drain `server-queue.cpp:139-209`; consumers registered
  `server-context.cpp:1458-1462`.
- Ordering is arrival-order with a binary front/back priority only. One global
  queue; no per-accelerator sub-queues; no cost-based scheduling.
- GAP: unbounded deque; no per-tenant fairness; no remote-worker capacity/credit
  accounting. This is the host virtual queue's ancestor but lacks the credit and
  per-lane structure the design requires (MIXED_WORKLOAD_DESIGN.md section 7).

### A3. Deferred / cancel -- REFERENCE_ONLY (closest analog to lease revocation)
- Cancel type `server-task.h:21` + `id_target` `server-task.h:140`; emit
  `server-queue.cpp:441-460` (posted front); erase-before-start
  `server-queue.cpp:211-222`; release-if-running `server-context.cpp:2449-2458`.
- Disconnect trigger `server-http.cpp:583/630/647` polled at
  `server-queue.cpp:389-396`. Deferral + re-queue `server-context.cpp:2399-2419`,
  `server-queue.cpp:77-100`.
- A dispatched task CAN be cancelled by releasing its slot between decode
  iterations, but cancellation is cooperative/edge-triggered and does NOT
  interrupt an executing `llama_decode`.
- GAP: no lease/epoch to invalidate work already handed to a phone; no forced
  abort of remote in-flight work. `t_max_predict_ms` is soft and newline-gated
  (`server-context.cpp:1972-1978`); `t_max_prompt_ms` is a declared-unimplemented
  TODO (`server-task.h:65`). REFERENCE_ONLY for a distributed lease/epoch model.

### A4. Slot lifecycle -- ADAPT (KV/state ownership pattern)
- N slots from `n_parallel` (`server-context.cpp:1332-1334`); state enum
  `:58-65`; slot struct `:161-853`; init `:1355-1374`; selection (id / LCP / LRU)
  `:1586-1694`. seq_id == slot.id, used directly in `common_batch_add`
  (`:136`, `:454`, `:471-473`); KV cleared via `common_context_seq_rm`
  (`:262-265`); release `:483-503`.
- The one-owner-per-sequence, hold-KV-until-release pattern is exactly the
  state-affinity model the design needs.
- GAP: seq_id is a fixed local index, not a lease; no route/lease epoch, no
  remote-accelerator identity, no phone-state ownership handle, no renewal/expiry.
  Slot selection has no notion of which DEVICE holds the state.

### A5. Continuous batching -- REFERENCE_ONLY
- Driver `update_slots` `server-context.cpp:2766-2854`; batch formation
  `pre_decode` `:2856-3582`; co-batch predicate (same type + equal lora)
  `:385-389`; single `llama_decode` `:3583-3705`, gated by `cont_batching`
  `:3059-3063`.
- GAP: single-host, single-context, single `llama_decode`, partitioned only by
  (task type, lora). No concept of splitting a batch into islands across
  accelerators, and no credit-aware batch sizing tied to remote capacity.

### A6. Result routing -- REFERENCE_ONLY
- `server_response` with `waiting_task_ids` + `queue_results`
  (`server-queue.h:119-131`); send `:319-332`; recv (linear scan)
  `:266-312`; streaming build `server-context.cpp:2084-2124`; wire streaming
  `server-http.cpp:532-565`; router fan-out `broadcast` `server-queue.cpp:334-343`.
- GAP: correlation is a local `int`; no network correlation id; no reassembly of
  partial results from multiple remote islands; `recv`/`recv_with_timeout` call
  `std::terminate()` on shutdown (`server-queue.cpp:272`, `:304`) -- not
  fault-tolerant to a dropping peer.

### A7. Admission / backpressure -- REJECT (must build)
- No queue cap; `post`/`defer` always push (`server-queue.cpp:22-69`); searched
  `max_queue|capacity|429|reject|throttle|admission` -> not found. Only reaction
  is unbounded `defer()` (`server-context.cpp:2399-2419`). Empty-batch kill
  switch `:3605-3615`.
- GAP: no admission control, no bounded remote credits, no per-worker inflight
  limit, no 429. A credit-based admission layer and per-lane inflight accounting
  are entirely missing and must be built (not ported).

Summary A: the server is single-process, single shared `llama_context`,
single-consumer over one queue. Slot KV-ownership (A4) is the reusable state
model; cancel/defer (A3) is the closest lease/admission analog but only
REFERENCE_ONLY. The host orchestrator's virtual queue, credits, and per-lane
sub-queues are NET-NEW, sitting ABOVE this server, not inside it.

## B. LayerSplit raw TCP data plane

File: `examples/layersplit/layersplit.cpp` (single TU; both peers little-endian,
silently assumed).

### B1. Byte-exact IO loops -- ADAPT (downgraded from REUSE per review)
- `send_all` `:428-440`, `recv_all` `:444-457`, `set_nodelay` `:459-462`,
  `connect_to` `:700-714`. Correct partial-read/EINTR handling.
- NOT reusable as-is: no `MSG_NOSIGNAL` / `SIGPIPE` suppression (a peer
  disconnect mid-`send` raises SIGPIPE and kills the process), no
  `SO_RCVTIMEO`/`SO_SNDTIMEO` or poll deadline (unbounded block, B5), and errors
  propagate only as a bool with no typed error/partial-progress reporting. These
  must be added before the loops are safe in a persistent agent, so ADAPT, not
  REUSE.

### B2. Framing -- REJECT (as a wire format)
- No magic, no version, no checksum. Boundaries are implicit fixed-size binary
  reads, host-endian, no `hton*` on payload. Examples: tailnet
  `:507-525`/`:549`; stagenet `:741-759`; pipedriver `:808-814`; kv int64-length
  path `:1027-1061`. Only the stream mode carries an opcode enum
  (`OP_DECODE/RESET/SHUTDOWN`) `:874`.
- REJECT the format (no forward-compat, no resync, no multiplexing).
  REFERENCE_ONLY for the opcode idea `:874`.

### B3. IDs on the wire -- REJECT
- Only `pos` (KV position, doubles as a `<0` liveness sentinel) and `tok`;
  kv adds `first`+`w64`. No request-id, batch-id, sequence, epoch, or
  correlation id anywhere. Ordering is purely positional lock-step.

### B4. Bounds checking -- REJECT (has real bugs)
- tailnet/pipedriver validate `ne == n_embd` before recv (`:519-522`, `:812`) --
  safe. But stagenet reads attacker-controlled `nh` and `recv_all(..., nh*4)`
  into an `n_embd`-sized buffer with only `nh > 0` (`:744-745`) -> heap overflow
  when `nh > n_embd`. kvclient does `buf.resize((size_t)w64)` straight from the
  wire with no bound (`:1058-1059`) -> OOM/abort. Any wire length must be
  range-checked against a fixed max before allocation.

### B5. Timeouts / heartbeat / reconnect / idempotency -- REJECT (all absent)
- No `SO_RCVTIMEO/SNDTIMEO`, no non-blocking, no poll deadline, no heartbeat,
  no reconnect, no retry/idempotency. `recv_all` blocks unboundedly
  (representative head-of-frame reads: `:508`, `:742`, `:812`, `:897`, `:1056`).
  `listen(...,1)` + single `accept` => one client per process lifetime, no
  reconnect path structurally (`:471-496`, `:722-733`).

### B6. Failure behavior -- REJECT (fail-by-process-exit)
- Clean EOF/malformed frame -> teardown and whole-process exit (e.g. `:508-510`,
  `:519-522`, `:835/847`). Positional length-implicit framing means one desync
  permanently corrupts the stream; only "recovery" is full teardown; no
  idempotent replay, so an interrupted request is lost.

Summary B: the raw relay is a lab oracle over adb-forwarded USB. IO loops (B1)
are ADAPT (they need SIGPIPE/deadline/typed-error handling before reuse);
everything about framing, IDs, bounds, timeouts, and failure is REJECT for a
control/data plane. The design's length-delimited, versioned, checksummed,
bounded frame (MIXED_WORKLOAD_DESIGN.md section 10) is net-new.

## C. Dualengine dual-backend (HTP + GPU) workers

File: `examples/layersplit/layersplit.cpp`, `run_dualengine` `:1236-1766`.

### C1. Persistent worker + context skeleton -- REFERENCE_ONLY (downgraded from ADAPT per review)
- Two models/contexts loaded once (`:1252-1268`, `:1300-1309`); `OverlapLeg`
  work units `:1173-1221`; thread started once via `start()` `:1213`, `:1502`;
  teardown `stop()`/join `:1220`, `:1708`; cleanup `:1759-1765`. The
  persistent-thread + preallocated-context SHAPE is instructive, but this code is
  a benchmark harness, not an adaptable lane: see C1a and C4/C5.

### C1a. Why REFERENCE_ONLY, not ADAPT (race + ignored-failure)
- Ignored failure: a failed worker still reports `done_gen==gen` (loop `break`s
  `:1205` then unconditionally stamps done `:1210`), so the driver sees "done"
  while the lane is broken; failure is only recovered by a post-hoc `check()`
  the driver may not run. A lane primitive must latch a sticky error and refuse
  work, which is a redesign, not an adaptation.
- Diagnostic data race: `dump_abort` reads leg fields (`gen`, `done_gen`,
  `completed`, `failed_round`) WITHOUT the lock `:1510-1515` while the worker
  mutates them -- benign only because it precedes `std::abort()`, but not a
  pattern to carry into a running agent.
- No request queue / synthetic input (C5): the "worker" runs a fixed `rounds`
  count of identical fixed-shape synthetic batches, so its lifecycle has never
  been exercised against real streamed, variable-shape traffic.

### C2. Owned-generation handshake + barrier -- REFERENCE_ONLY (downgraded from ADAPT)
- `gen` vs `done_gen` (uint64, "no caller-stack pointers") `:1180`; worker copies
  under lock, runs `rounds`, stamps done `:1197-1210`; `OverlapBarrier`
  arm/arrive/wait_release/go `:1156-1164`, driver orchestration `:1516-1524`;
  compute-completion stamped before reset `:1206`. The arm/arrive/release/done
  IDEA is sound, but it is one-shot-per-phase and inseparable from the
  ignored-failure path (C1a) and the abort-on-stall watchdog (C3). Study it;
  re-implement the lane handshake with error latching rather than adapting this
  code.

### C3. Timed rendezvous + watchdog -- REFERENCE_ONLY
- Cross-thread waits are timed (`wait_all_arrived/wait_release/wait_done` via
  `cv.wait_for`, `:1161-1162`, `:1219`; deadlines `OL_BARRIER_MS=30000`
  `:1146`, `OL_PHASE_MS=180000` `:1147`). Watchdog `dump_abort` prints leg state
  and calls `std::abort()` `:1510-1515`. The timed-rendezvous discipline is
  right; the abort-the-whole-process response is unusable for a persistent agent.

### C4. Known validity issues stated by the code -- must repair before MW2
- `std::abort()` on any stall `:1514` (code concedes a hung DSP decode is
  "cannot join out of" `:1506`) -- kills the process, not the lane.
- Three measurement regimes run because none alone is trusted: SATURATED `:1604`,
  FIXEDPAIR ("NOT request latency") `:1642-1645`, SERVICE microtrace `:1672-1675`;
  per-leg CoV `:1723`. (Matches S6_EVIDENCE_AUDIT: saturated PROVISIONAL, service
  FAIL.)
- Reset fragility: hard failure latches `reset_failed` permanently
  `:1459-1461`, `:1495-1497`.
- Cross-backend correctness can be `blocked` not `pass` (`xcorr` `:1633-1635`;
  CPU ref may fail to load on RAM-tight devices `:1267`, `:1326`).
- Weight sharing not landed: requirement-3 single-copy path env-gated on an
  unmerged dmabuf buffer-type `:1088-1089`, `:1247-1250` -> default is 2x shard
  RAM.

### C5. What must be repaired for two bounded phone lanes -- ADAPT skeleton, build the rest
- Fault isolation: replace `std::abort()` `:1514` with per-lane cancel + context
  recreate; a hung decode must not take the agent down.
- Error latching: a failed worker still reports `done_gen==gen` (loop `break`s
  `:1205` then stamps done `:1210`) -- the lane looks "done" while broken. Needs a
  sticky per-lane error state that quiesces the lane and refuses work until reset.
- Bounded queue: there is NO request queue -- `launch(rounds,...)` runs a fixed
  count of identical fixed-shape synthetic batches (`:1201-1209`, `:1331-1336`).
  Each lane needs a bounded MPSC request queue with backpressure, per-request
  payload + correlation id, and completion delivery.
- Admission: contention is measured but never capped (`svc_slowdown_D/P`
  `:1702-1703`).

Summary C: `OverlapLeg`+`OverlapBarrier` (`:1156-1221`) is REFERENCE_ONLY -- the
arm/arrive/release/done IDEA is instructive, but the code has an ignored-failure
path (a broken worker reports done, C1a) and a diagnostic data race, so a lane
primitive must be RE-IMPLEMENTED with error latching, not adapted from this code.
Its benchmark scaffolding and abort-on-stall model are REFERENCE_ONLY / REJECT.
This is the "repair persistent dualengine worker lifetime" prerequisite in
NEXT_PLAN MW2.

## D. ggml-rpc data plane

Files: `ggml/src/ggml-rpc/ggml-rpc.cpp`, `transport.{h,cpp}`.

### D1. Framing + command protocol -- REFERENCE_ONLY
- Length-prefixed messages (8-byte `uint64` + payload) `ggml-rpc.cpp:244-274`;
  command frame `| cmd(1) | req_size(8) | req |` -> `| rsp_size(8) | rsp |`
  `:290-323`; 1 GiB chunked send/recv `transport.cpp:462-504`. Command enum
  `:56-75` (buffer/tensor/graph/device/HELLO). HELLO version handshake + conn
  caps `:330-347`, `:1455-1486`. SET_TENSOR_HASH content-dedup (FNV-1a, 10 MiB
  threshold) `:80`, `:233-242`, `:1131-1175` -- an interesting one-copy idea.
- Byte order: raw host-endian `memcpy`, `#pragma pack(1)`, no `htonl`
  (`:33-51`); 32-bit strides on the wire (`rpc_tensor`, `:35-51`).

### D2. Bounds / robustness -- REFERENCE_ONLY (better than layersplit, still not a control plane)
- Fixed-size responses reject size mismatch `:251-260`, `:316-318`; variable
  input `resize(size)` guarded by try/catch but with NO upper cap `:262-274`;
  graph deserialization does incremental bounds checks `:1343-1370`; tensor
  regions sanitized against buffer base/size `:1079-1089`, `:1155-1167`,
  `:1227-1239`, `:1026-1033`.

### D3. Timeouts / reconnect / idempotency / failure -- REJECT
- Fully blocking synchronous; pairs by socket ordering; no request IDs, no
  multiplexing. Searched `timeout|SO_RCVTIMEO|reconnect|heartbeat|retry` ->
  ABSENT on the TCP path (only RDMA QP params `transport.cpp:369-375`). Client
  failure -> `GGML_ABORT` `:30`. `listen(...,1)`, IPv4-only `transport.cpp:600-649`.

Summary D: REFERENCE_ONLY. Study the framing, the bounds discipline, and the
SET_TENSOR_HASH dedup; do not adopt the transport (host-endian, IPv4-only,
abort-on-error, no timeout/heartbeat/reconnect/idempotency).

## E. Weight provisioning and readiness

### E1. GGUF sharding + partial load -- ADAPT
- `research_dev/shard_gguf.py`: slices to `[start,end)`, PRESERVES absolute block
  indices (`blk.17.*` stays `blk.17.*`) and copies ALL metadata verbatim
  (`block_count` stays full n_layer): `want_tensor` `:30-39`, metadata loop
  `:66-73`, range validation `:60-61`, writer `:88-93`. No shard hash/manifest;
  caller names the output.
- Load-time counterpart (provenance only; do NOT edit): `LLAMA_LAYER_START/END`
  skip -- `src/models/gemma4.cpp:52-58`, `:86-92`; partial-load tolerance
  `src/llama-model.cpp:1477-1481`.
- ADAPT: correct offline transform + load skip. Add a shard manifest/hash and an
  output-name convention; validate the env range against the shard's real range.

### E2. Download / cache helpers -- ADAPT (verify + fsync + rollback are ABSENT)
- Files `common/download.{h,cpp}`, `common/hf-cache.{h,cpp}`.
- PRESENT: Range-resume (`Range: bytes=`, requires 206) `download.cpp:222-235`,
  conditional on `Accept-Ranges` `:390-401`; temp-file staging (`.downloadInProgress`)
  `:373`, `:216`; atomic publish via `std::rename` `:407-411`, hf-cache blob/
  snapshot `hf-cache.cpp:455-497`; retry/backoff `:290-291`, `:380-388`; ETag
  conditional cache `:83-101`, `:331-364`.
- ABSENT (must add before MW2): content SHA-256 verification of the downloaded
  bytes against the manifest `oid`/digest -- searched `sha256|verify|checksum`,
  only OCI-digest FORMAT regex exists `:826-841`, bytes are never hashed;
  `fsync/fdatasync` -- searched, no matches (rename is not durably flushed);
  transactional rollback -- only partial temp cleanup, no post-publish integrity
  gate. Range-resume silently degrades to full re-download when the server omits
  `Accept-Ranges` `:393-398`.
- ADAPT: reuse temp + atomic-rename + Range-resume + hf-cache layout; ADD verify,
  fsync-before-rename, and rollback (the readiness contract in DECISION_CONTRACT
  section 2 depends on these).

### E3. Per-tensor Hexagon publish / OpenCL import sharing -- EXISTS in the dirty tree; works single-load; REFERENCE_ONLY (unsafe across reloads)
- This mechanism EXISTS and works TODAY in the dirty worktree for the single-load
  case (one model loaded once): the name-keyed rpcmem-fd publish + QCOM
  ext-host-ptr import gives one physical weight copy shared by HTP and OpenCL, and
  it was validated at S2 for a single load. It is NOT hypothetical. What is unsafe
  is REUSE ACROSS RELOADS (see hazards below); the classification is REFERENCE_ONLY
  because a readiness pipeline that unloads/reloads models cannot use it as-is.
- Publish (Hexagon): rpcmem dmabuf fd `ggml-hexagon.cpp:300-309`; per-tensor
  publish of native-linear `.weight` `:376-388`; registry
  `unordered_map<string, entry>` keyed by BARE TENSOR NAME, first-insert-wins,
  `:1016-1024`; C-ABI resolver `:1027-1040`. Import (OpenCL): dlsym `:6332-6344`;
  by-name resolve + QCOM ext-host-ptr import `ggml-opencl.cpp:6349-6373`,
  `:6405-6445`; fd imported once `:6326-6328`.
- Hazards (provenance only): NO model-version/generation/model-id qualifier
  (key is bare name); first-insert-wins + no unpublish/erase -> a second model
  reusing tensor names resolves to the STALE fd of the first (stale-alias);
  imports are intentionally leaked and process-global `:6346-6348` ->
  repeated unload/reload accumulates leaks and stale bindings; single `clFinish`
  coherency valid only under the sequential "decode loaded before prefill"
  assumption `:6366-6368`.
- REFERENCE_ONLY: the name-keyed one-copy dmabuf-sharing pattern is elegant but
  as built it is name-only, first-insert-wins, never-unpublished, leaked. A
  readiness pipeline that reloads models MUST add a model-version/generation
  qualifier and a teardown/unpublish path before adopting it (this is the
  MIXED_WORKLOAD_DESIGN section 5 "version/generation qualified" requirement).

## F. Historical VQ / CONWIP / HEFT (route2-b9531)

Repo `/home/myid/zs89458/Documents/llama.cpp` on branch `route2-b9531`
(confirmed). Working-tree diff vs HEAD across the three VQ files: 214 insertions,
5 deletions. THIS IS PROCESS-LOCAL OCCUPANCY, NOT DISTRIBUTED READINESS.

### F1. VQ occupancy ring -- ADAPT / REFERENCE_ONLY (downgraded from REUSE per review)
- `struct vq_ring { uint64_t slot[2048]; atomic head,tail; }`, 2048 slots/backend,
  remove-on-complete: `ggml-backend.cpp:126-135`. Packed word (`vq_pack`
  `:110-117`): backend_id 3b, model tag 4b, op_type 7b, size_class 6b,
  seq 44b. Depth `:345-349`, busy mask `:370-378`, analytic bandwidth (L1 bytes
  `:351-354`, L2 windowed monitor thread `:242-273`, `:361-368`).
- NOT reusable as-is (two concurrency defects):
  1. Publish-before-write: enqueue does `tail.fetch_add` THEN writes
     `slot[tl & MASK]` `:306-307`. A concurrent reader computing depth/scanning
     can observe the advanced `tail` before the slot payload is written, i.e. it
     can read a slot that is claimed-but-not-yet-populated. A correct MPSC ring
     must publish the slot payload with release ordering and only then advance a
     separately-observed tail.
  2. Overwrite-live-slot: past depth 2048 the same `slot[tl & MASK]` silently
     overwrites a live entry `:306-307` with no backpressure/drop signal (no
     overflow contract, F3).
- ADAPT the depth/busy-mask CONCEPT as a local occupancy signal, but only after
  fixing publish ordering and adding an overflow contract; treat the current ring
  code as REFERENCE_ONLY. REJECT reading it as distributed/device state.

### F2. CONWIP admission gate -- ADAPT
- `vq_admit_impl(backend, k, ceil_gbps)` spins until `depth < k`
  `ggml-backend.cpp:422-447`; ~2-second busy-yield deadline `:426-436`; optional
  L2 bandwidth valve `:430-433`.
- ADAPT: the K-cap pull-control idea is sound; the 2s busy-yield `yield()` loop is
  a harness expedient -- a runtime wants a condvar/futex wait and a real
  backpressure contract.

### F3. What the VQ word CANNOT carry -- (defines the net-new readiness plane)
- Absent from `vq_pack`: request id (only a per-op `seq`), deadline/slack/SLO,
  lease/ownership, route epoch, DEVICE IDENTITY (backend is a 4-way op-class;
  REMOTE is one opaque bucket, `:90`), cache/model readiness, and an overflow
  contract (`tail.fetch_add` then `slot[tl & MASK]` silently overwrites a live
  slot past depth 2048, `:306-307` -- no backpressure/drop signal). The 4-bit
  model tag (`& 0xF`, `:113`, `:381`) caps at 16 tags and is NOT a distributed
  model identity.

### F4. mtmd-sched.cpp HEFT/EMA/CONWIP -- REFERENCE_ONLY (belongs in the simulator)
- Handcrafted per-boot cost priors + online EMA (`cost[2][BK_N]` seeds
  `tools/mtmd/mtmd-sched.cpp:143-146`, EMA `:164`); ALL-at-t0 burst (no arrival
  process) `:131-136`; only 2 backend buckets `:66`, with vision@NPU a MODELED
  `sleep_for` `:156-162`; one deque VQ `:132` + dumb per-backend worker queues
  `:71-79`; HEFT min-EFT arbitration `:203-229`, K-slot admission `:212`, `:225`.
- REFERENCE_ONLY: an offline calibration harness. The HEFT-EFT arbitration and
  K-slot CONWIP ideas belong in the MW3 simulator/baseline suite; the cost
  seeding, sleep-modeled route, t0 burst, and CSV dump belong in a simulator, not
  a runtime.

### F5. mtmd-coexec{,-diff}.cpp -- REFERENCE_ONLY
- 3-phase solo/solo/coexec overlap probes using the VQ only as an observability
  oracle (`ggml_vq_busy_mask` `mtmd-coexec.cpp:181-188`); `-diff` uses a separate
  vision model to retire same-gguf weight-share inflation. Evidence that the
  overlap is real; no scheduling policy to reuse.

## G. Unifer remote telemetry + roster

Path `/home/myid/zs89458/Documents/Unifer/research_dev/services/lazyvlm/`. THIS
IS ADVISORY PER-DEVICE OCCUPANCY, NOT RELIABLE READINESS.

### G1. 16-byte telemetry word -- ADAPT (occupancy only)
- Two LE uint64 + crc8; layout `REMOTE_TELEMETRY_DESIGN.md:8-32`,
  `vq_remote.h:6-17`, codec `:58-105`. Carries rapidly-changing occupancy/thermal:
  `npu_depth`/`gpu_depth` (each peer's own `ggml_vq_depth`), `busy_mask`,
  `gpu_busy_pct`+window, `therm_c`/`therm_slope`, `mem_avail`, plus `epoch_seq`
  (reboot detector), `schema_ver` (reject-on-mismatch), and readiness FLAGS.
  Stated limits: fire-and-forget UDP, latest-wins, loss-tolerant because each
  datagram is a COMPLETE absolute snapshot `:38`; 20 Hz floor `:43-52`.
- ADAPT the compact wire format + CRC/schema/epoch discipline for advisory
  telemetry. It is explicitly advisory local-occupancy-per-device, not
  authoritative state.

### G2. telemetry_roster.py -- ADAPT (shape) / REFERENCE_ONLY (code)
- Per-device latest-wins slot `roster[dev_id]=(dec,t)` `:90`; RECEIVER-clock
  staleness (`time.monotonic()` at receipt `:82`, `eligible` STALE > 0.5s
  `:55-63`); sender `send_mono_ms` used only for jitter, never cross-device
  compared. Hard eligibility gate (stale / mem<512MB / therm>=95 / reboot / not
  accepting) `:55-63`.
- KEY COUPLING: the actual per-backend queue depths (`npu_depth`, `gpu_depth`,
  `busy_mask`) are HOST-BLIND -- sysfs cannot see Hexagon, so they must come from
  each peer's OWN local VQ ring (`:26-27`, `:42-44`, `:94`), and are stubbed 0 in
  this host-pull stand-in. ADAPT the roster shape (latest-wins + receiver-clock +
  hard gate); REFERENCE_ONLY as code.

### G3. Readiness/cache/manifest transport -- REJECT the 16-byte word for this
- The docs piggyback readiness onto the SAME lossy advisory word (warm/accepting/
  draining flags `DESIGN.md:30`; residency only a future "pinned-resident-clip
  nibble" in reserved `:32`; reboot invalidation inferred from `epoch_seq`
  `:68`). No dedicated reliable control channel is specified -- a gap to flag.
- The design's own rationale argues why this is wrong: UDP loss-tolerance depends
  on every datagram being a complete self-healing snapshot, which holds for
  continuous occupancy but FAILS for discrete readiness edges (a dropped "model X
  evicted" / "lease revoked" is not self-healing; staleness decay cannot
  reconstruct a missed transition; acting on a stale readiness bit causes a WRONG
  placement, not a conservative one).
- REJECT using the 16-byte word / its UDP transport for model-readiness / cache /
  lease / manifest state. Those need a SEPARATE reliable, ordered, acked control
  channel (versioned readiness/lease records, at-least-once delivery). The
  prototypes provide the occupancy half and NOTHING for the reliable-readiness
  half.

## H. Local-occupancy vs distributed-readiness -- explicit verdict

- Process-local virtual-queue occupancy (route2 `g_ring`/depth/busy-mask
  `ggml-backend.cpp:126-378`; and per-peer `npu_depth/gpu_depth/busy_mask` from
  each device's own ring per `telemetry_roster.py:42-44`): ADAPT the CONCEPT as a
  fast, loss-tolerant load signal; the ring CODE is REFERENCE_ONLY until its
  publish ordering and overflow contract are fixed (F1). Answers "how busy is
  this queue now."
- Distributed model-readiness / lease / cache-manifest state: NOT present as a
  first-class reliable mechanism in either repo. It is absent from the VQ ring
  (no device id, no readiness, no lease, no overflow contract -- F3) or unsafely
  folded into advisory UDP flags/epoch/reserved bits (G3). This is the net-new
  plane the S8 design must build: versioned READY/lease records over a reliable
  ordered channel, distinct from the occupancy signal.

## I. Classification summary

| Mechanism | File(s) | Class |
|---|---|---|
| server request id | server-queue.cpp:71-75 | ADAPT |
| server task queue (FIFO) | server-queue.{h,cpp} | REFERENCE_ONLY |
| server cancel/defer | server-queue.cpp:441-460; server-context.cpp:2449-2458 | REFERENCE_ONLY |
| server slot KV ownership | server-context.cpp:161-853,1355-1374 | ADAPT |
| server continuous batching | server-context.cpp:2766-3705 | REFERENCE_ONLY |
| server result routing | server-queue.cpp:266-343 | REFERENCE_ONLY |
| server admission/backpressure | (absent) | REJECT (build new) |
| layersplit send_all/recv_all/nodelay | layersplit.cpp:428-462 | ADAPT (add SIGPIPE/deadline/typed-error) |
| layersplit framing/IDs/bounds/timeouts | layersplit.cpp (B2-B6) | REJECT |
| dualengine OverlapLeg/OverlapBarrier | layersplit.cpp:1156-1221 | REFERENCE_ONLY (races + ignored failure) |
| dualengine abort-on-stall + fixed-rounds | layersplit.cpp:1201-1209,1510-1515 | REJECT |
| ggml-rpc framing + SET_TENSOR_HASH | ggml-rpc.cpp:244-323,1131-1175 | REFERENCE_ONLY |
| ggml-rpc transport | ggml-rpc.cpp; transport.cpp | REJECT |
| shard_gguf.py + LAYER_START/END | shard_gguf.py; gemma4.cpp:52-92 | ADAPT |
| download/cache (resume+atomic) | common/download.cpp; hf-cache.cpp | ADAPT (add verify/fsync/rollback) |
| per-tensor HTP/OpenCL sharing | ggml-hexagon.cpp:376-1040; ggml-opencl.cpp:6349-6445 | REFERENCE_ONLY (REJECT across reloads) |
| VQ occupancy ring/depth/busy | route2 ggml-backend.cpp:126-378 | ADAPT concept / REFERENCE_ONLY code (publish-before-write + overwrite-live-slot) |
| VQ CONWIP admission | route2 ggml-backend.cpp:422-447 | ADAPT |
| mtmd-sched HEFT/EMA | route2 mtmd-sched.cpp | REFERENCE_ONLY (simulator) |
| mtmd-coexec probes | route2 mtmd-coexec*.cpp | REFERENCE_ONLY |
| Unifer 16-byte telemetry word | vq_remote.h; REMOTE_TELEMETRY_DESIGN.md | ADAPT (occupancy only) |
| Unifer roster | telemetry_roster.py | ADAPT (shape) / REFERENCE_ONLY (code) |
| Unifer word for readiness | REMOTE_TELEMETRY_DESIGN.md:30-68 | REJECT (needs reliable channel) |

## J. Net-new components implied by this audit (nothing ported in V0a)

1. Host virtual queue with reorderable work + per-lane bounded credits (above the
   server queue A2/A7).
2. A distributed correlation-id + route/lease/seq epoch on every request
   (missing everywhere: A1, B3, D3, F3).
3. A reliable ordered control channel for READY/lease/manifest, separate from
   occupancy telemetry (G3, H).
4. A length-delimited, versioned, checksummed, bounded frame with heartbeat and
   reconnect (B, D are both REJECT for this).
5. A phone weight-readiness pipeline adding SHA-256 verify + fsync + rollback on
   top of the download helpers (E2), and a model-version/generation-qualified
   share registry with teardown (E3).
6. A repaired persistent dual-lane agent with per-lane fault isolation, sticky
   error latching, and bounded request queues (C5).

None of these are built in V0a. This audit only classifies what exists and marks
the boundary between reusable substrate and net-new work for MW1+.
