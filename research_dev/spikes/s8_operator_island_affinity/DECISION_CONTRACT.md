# What / How / When Decision Contract (S8-V0a-R2)

Status: **DRAFT-BLOCKED-BEFORE-V0c.** This contract and its schemas
(`route_action`, `route_decision`, `model_residency_lease`,
`request_state_lease` -- each carries `status: DRAFT_BLOCKED_BEFORE_V0C`) are a
design draft, NOT frozen. They are blocked from implementation until reviewed at
the start of V0c. Gate A (V0b, normalization) does not depend on any of them and
proceeds independently. Specification only; no scheduler is implemented.

Unresolved items that MUST be settled before V0c implementation (this checkpoint
does NOT resolve them and does not expand into an oracle design):

1. Request-level SLO objective: how a per-request SLO (TTFT / TPOT / completion /
   deadline) maps into the lexicographic objective for a DAG that spans several
   islands (section 4 defines a per-action score, not a per-request rollup).
2. Compound-action timing: the exact time semantics of a `chain_a0` /
   `corun_pair` (when each stage's clock starts, how overlap is credited).
3. Authoritative lane-credit ledger: the single source of truth for lane credits,
   its concurrency model, and how reservation/release stay consistent under
   failure (the schema names credits; the runtime ledger is undefined).
4. A0 layer-slice / state representation: how a chain stage's layer slice and its
   stage-local KV are represented and owned across the chain.
5. Mutation sequence / idempotency: the exact stale-completion rejection and
   at-least-once/exactly-once semantics for state-mutating operations.
6. Total compound-action tie-break: section 4.2 breaks ties for a single island;
   a total order over compound multi-lane actions is not yet defined.

This revision drafted findings 4 (define "what"; compound atomic actions), 5
(deterministic objective + p95 formula + tie-break), and 10 (finite admission
credits + explicit outcomes); they remain DRAFT pending the six items above.

Derived from `MIXED_WORKLOAD_DESIGN.md` sections 8-10 and `NEXT_PLAN.md` section
3. Schemas: `schemas/dag_cover.schema.json`, `schemas/route_action.schema.json`,
`schemas/route_decision.schema.json`.

## 0. Two loops, two horizons

- SLOW loop (residency/placement): profile-derived seconds-to-minutes horizon.
  Decides which model versions and partial shards are resident on which phone,
  for how long, and which certified plan envelope becomes available. Off the
  request critical path.
- FAST loop (dispatch): every arrival and completion. Decides, among READY
  routes, the WHAT (island cover) and the compound action for a request now.

Hard invariant: the fast loop may only choose a precompiled cover/action in the
current execution envelope whose complete route is already READY. It never
triggers download/load/warmup or invents a graph cut on the critical path.

## 1. WHAT: the island cover (review finding 4)

Before choosing a route, the fast loop selects WHAT the islands are from a
precompiled non-overlapping cover in the current execution envelope. The cover
PARTITIONS the request's service DAG
(`schemas/dag_cover.schema.json`). Rules:

1. Every DAG node belongs to exactly one island. The tool asserts
   union(member_node_ids) == all nodes AND pairwise-disjoint; otherwise
   `partition_valid=false` and the cover is rejected.
2. Each island is a CONTIGUOUS replayable subgraph (no gather of non-adjacent
   nodes). Its boundary is exactly the cut edges (`boundary_in_edges` /
   `boundary_out_edges`).
3. The candidate covers are enumerated deterministically from a COMMITTED cover
   catalog per service (e.g. for RAG: {whole-request}, {embed | retrieve+rerank |
   prefill+decode}, {embed | ... | prefill | decode}). The oracle does not invent
   covers at runtime; it selects among the catalog by the objective (section 4).
4. CPU-only glue nodes (`phone_candidate=false`) stay in the cover as
   server-pinned islands; they still gate dependencies and server idle windows.

The cover is the "what leaves the server" decision; the route is the "where each
island runs" decision (section 3).

## 2. SLOW loop -- weight placement and residency

Inputs: model/shard manifests, phone capability + RAM inventory, the atlas
(which islands have PASSING routes), predicted reuse, current model-residency
leases.

Decisions (unchanged from V0a in intent; now typed by
`schemas/model_residency_lease.schema.json`):

1. Weight placement: knapsack over predicted reuse value bounded by usable RAM
   (OP15 ~10 GB, OP12 ~6 GB) and disk. Not "fill every phone."
2. Download: manifest-driven, to STAGING only; runtime traffic never downloads.
3. Verification: candidate for READY only after full manifest + per-file SHA-256
   match (currently ABSENT in the download path; SUBSTRATE_AUDIT.md E2).
4. Warmup: load kernels, map weights, run warmup + correctness sentinel.
5. READY: only after verify + load + warmup + correctness sentinel + RAM headroom
   for state/scratch beyond weights. Publishes a `model_residency_lease`
   (`ready_state=READY`, a fresh `residency_epoch`, and a
   `share_registry_generation` that qualifies the per-tensor share registry so a
   reload cannot stale-alias a prior model's fds; SUBSTRATE_AUDIT.md E3).
6. RAM admission: grant only if reserved weight+state+scratch fit under the
   ceiling with headroom; insufficient RAM rejects the lease (no implicit
   eviction of a live lease).
7. Lease horizon: minimum horizon + hysteresis; no per-request residency thrash.
8. Eviction: only at a drain boundary, after the minimum horizon, only when
   incoming placement value exceeds evicted predicted-reuse value + hysteresis
   margin; a set with a live request-state owner drains first; no hot KV
   migration.

Lifecycle: `ABSENT -> DOWNLOADING -> VERIFYING -> ON_DISK -> LOADING -> WARMING ->
READY`; `READY -> LEASED -> DRAINING -> READY`; any `-> ERROR | STALE`.

## 3. FAST loop -- per-request dispatch

Inputs: the ready request + its chosen cover, READY routes per island, live
per-lane telemetry (`schemas/telemetry.schema.json`), the pairwise interference
matrix, current server pressure, SLO, state affinity.

Ordered procedure:

1. Admission (section 6): admit into the host virtual queue if credits allow;
   else an EXPLICIT outcome, never an unbounded wait.
2. Enumerate candidate compound actions (`schemas/route_action.schema.json`) for
   the cover: single_island, chain_a0, corun_pair, merged_batch. Reject any whose
   hard gate fails, recording the reason code.
3. State affinity: a sticky island whose state owner is a live request-state
   lease has that owner as its ONLY compute route. No hot KV migration.
4. Batching wait: for same-model-version islands, optionally hold a bounded wait
   to form a merged_batch, only if the predicted finish (section 5) still meets
   the SLO. Different model or version never batch.
5. Action selection: pick the eligible action using the proposed draft objective
   (section 4) under current server pressure. If no action beats keeping the work
   on the server, keep it there (`server_cheaper`).
6. Atomic reservation (section 3.1): reserve all lanes of the chosen action
   all-or-nothing.
7. Fallback: every dispatched island has a server replay boundary; on failure/
   timeout/stale/expiry the request replays once (token history for decode, stage
   input for stateless) with no duplicate completion.

### 3.1 Compound actions reserve all lanes atomically (review finding 4)

A `route_action` names an `assignments` set (island -> route) and a
`reservations` set (the exact lane credits it needs). It is committed only if
EVERY reservation succeeds; if any fails, the whole action is abandoned and all
its partial reservations are released (`atomic_reservation_ok=false`). This
represents:

- `chain_a0`: N stages across N `(device, backend)` lanes (e.g. OP15-HTP ->
  OP12-HTP -> A6000-CUDA), reserved together so a chain never half-commits.
- `corun_pair`: two lanes on ONE phone (HTP + GPU), eligible only if
  `corun_pair_certified=true` (the pairwise interference row is a PASS).
- `merged_batch`: one lane shared by N `request_ids` of the SAME `model_version`.

Reservation uses a fixed lane-lock order (by `device_backend_id` ascending) to
avoid deadlock, and epoch-checked idempotent release.

## 4. WHEN/WHICH: proposed capacity objective (review finding 5)

Capacity mode is the ONLY active objective (energy is DEFERRED; the `energy_gain`
reason code is reserved and never emitted in S8). The objective is a FIXED
LEXICOGRAPHIC order, evaluated per candidate action; smaller is better in each
level, compared in order:

```text
L1  slo_violations        : count of islands in this action predicted to miss SLO
L2  -relief_value          : negative weighted server relief (so more relief ranks better)
L3   risk_penalty          : transfer + cache-churn + thermal + failure risk
```

`relief_value` uses EXPLICIT FIXED shadow prices (committed constants, swept in
sensitivity, never learned at runtime):

```text
relief_value = price_gpu_ms   * gpu_ms_freed
             + price_hbm_byte  * hbm_bytes_freed
             + price_hbm_bw    * hbm_bw_freed
             + price_admission * admissions_enabled
```

`risk_penalty` uses committed prices too:

```text
risk_penalty = price_transfer * transfer_us
             + price_churn    * cache_churn_bytes
             + price_thermal  * thermal_margin_deficit
             + price_failure  * failure_prob_estimate
```

`gpu_ms_freed` / `hbm_*_freed` come ONLY from a `profile_row.server_relief` that
is measured (non-null); an action whose relief is unknown scores L2=0 (no
credit), never an estimate. All prices live in one committed config block and are
reported with every result.

### 4.1 Predicted p95 finish-time formula (review finding 5)

For an island on a route, the predicted finish is:

```text
predicted_p95_finish_us =
      queue_wait_us(lane)                     # sum of ahead-of-me p95 service on the lane
    + transfer_in_us                          # boundary_in_bytes / measured link goodput
    + service_p95_us * interference_mult      # profile_row.p95_us * matrix multiplier
    + transfer_out_us                         # boundary_out_bytes / measured link goodput
```

- `interference_mult` is 1.0 for a solo lane; for a `corun_pair` it is the
  MEASURED `corun_slowdown` from the interference matrix (never an assumed 1.0).
- A `chain_a0` finish is the sum of stage finishes along the chain (later stage's
  `queue_wait` includes earlier stages' output arrival).
- Any input to this formula that is null makes the action ineligible
  (`no_slack`/`transfer_dominated` reason), never estimated.

### 4.2 Deterministic tie-break (review finding 5)

When two actions tie on all three lexicographic levels, break ties by the ordered
key (recorded in `route_decision.tie_break_key`):

```text
(predicted_p95_finish_us ASC, island_id ASC, device_backend_id ASC)
```

`island_id` and `device_backend_id` are compared as byte strings. This key is
total, so the selection is reproducible across implementations.

## 5. HTP and GPU as separate bounded lanes

Each `(device, backend)` lane has independent bounded credits: max queued
islands, max queued activation bytes, max resident state/KV slots, max
admitted-but-not-completed work. Concurrent HTP+GPU use is eligible ONLY for a
`corun_pair` whose measured interference passes p95 and correctness. Current
evidence: no co-run pair is certified (S6_EVIDENCE_AUDIT.md). Maximum phone use =
maximum USEFUL parallelism under the objective, not forced 100% utilization; a
lane stays idle when using it raises tail latency or interference.

## 6. Admission credits and explicit outcomes (review finding 10)

The host virtual queue has a FINITE credit budget `C_host` (committed). Real
backend lanes have finite per-lane credits (section 5). When a request arrives:

1. If host credits available: admit to the virtual queue, reserve one host
   credit (released on completion/failure).
2. If host credits exhausted, the outcome is one of these EXPLICIT results
   (committed policy per service class; never an unbounded wait):
   - `server_fallback_immediate`: run on the server now (reason
     `server_cheaper`), if server admission allows;
   - `reject_backpressure`: return a 429-equivalent to the caller (external
     backpressure), for a bounded overflow queue that is full;
   - `shed_to_overflow`: place in a bounded overflow queue with a max age; on age
     expiry it becomes `reject_backpressure`.
3. A lane whose credits are exhausted is simply not a candidate route this tick
   (reason `credits_exhausted`); the request either takes another route or the
   host-level outcome in step 2 applies.

There is no state in which a request "just waits" with no bound. Every terminal
outcome is one of {completed, server_fallback_immediate, reject_backpressure,
failed-then-replayed-once}.

## 7. Reason-code enum

Authoritative (`schemas/_defs.schema.json#/$defs/reason_code`). Rejection/not-
selected: `unsupported`, `not_resident`, `state_pinned`, `no_slack`,
`transfer_dominated`, `thermal`, `stale`, `insufficient_ram`, `interference`,
`credits_exhausted`, `server_cheaper`. Selection: `capacity_relief`.
`energy_gain` is RESERVED and MUST NOT be emitted before MW5 certification.

## 8. Invariants (oracle and any live scheduler)

1. Fast loop routes only to READY routes; never downloads/loads/warms on the
   critical path.
2. A null/UNKNOWN atlas measurement makes a route ineligible; no silent
   estimation into a PASS; unknown relief scores zero, not a guess.
3. One state owner, one mutation in flight per request; no hot KV migration.
4. Stale epochs and duplicate completions rejected (request-state lease epochs).
5. Timeouts fail closed; never a successful record.
6. Every dispatched island has a server replay fallback.
7. Capacity and energy never share an unlabeled score; energy off until MW5.
8. The objective, shadow prices, p95 formula, and tie-break are proposed config,
   reported with every decision, and reproducible.
9. Cross-workload slack pooling is the systems hypothesis; scattering operators
   while the server stays in the same power state is NOT avoided server energy.
