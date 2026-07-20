# S15: host-side mixed-workload dispatch and control plane

Status: `HOST_PHYSICAL_EXECUTOR_CONTRACT_PASS_LIVE_OP15_FOLLOW_ON_PASS`.
The host contract passes, and `../s15_live_launcher/` has now exercised it once
against a real OP15/A6000 B32 route. See both result files for claim boundaries.

## 0. Purpose and claim boundary

S15 builds the host-side dispatch/control-plane mechanics that a mixed-workload
Q-PIM scheduler needs before any physical run. It consumes already-measured S14
route profiles and produces deterministic, executable dispatch decisions. It is
not a latency simulator and it makes no latency, energy, throughput, physical
execution, scheduler-success, or total-system claim. Durations from a measured
profile are used only for the policy's SLO latest-start feasibility gate, which
is the same gate `power_frontier_policy.choose_batch()` already applies.

The host now has a strict subprocess transport and physical session-record
validator. The configured launcher will own ADB or socket setup, but no launcher
implementing the pending persistent worker protocol has run. No physical
completion is fabricated.

## 1. What is reused, unchanged

- `power_frontier_policy.py` (S14): `WorkItem`, `CertifiedBatchPoint`,
  `BoundaryCertificate`, `choose_batch`, `throughput_knee`. API unchanged.
- `priority_batch_runtime.py` (S14): `PriorityBatchRuntime`, `RouteConfig`,
  `Launch`, `Completion`. API unchanged; wrapped, not modified.
- `live_profile_adapter.load_op15_head_route` and
  `op12_profile_adapter.load_op12_head_route` (S14): the only path that turns
  measured phone artifacts into an eligible B1 `RouteConfig`.

S15 adds only host control-plane code in this directory. No model, backend,
scheduler, kernel, layersplit, or S14 energy source is touched.

## 2. Physical evidence this plane is pinned to

1. OP15 `[0,8)` B1: 7-process placement/correctness profile passes; the
   `load_op15_head_route` adapter binds it to a certified B1 route.
2. OP12 `[0,6)` B1: 7/7 processes and 56/56 requests pass; profile id
   `sha256:4de8d347...`; `load_op12_head_route` binds it to a certified B1 route.
3. OP12 `[0,8)` is ineligible: its repeatability cohort timed out, so the adapter
   cannot build a seven-process route and the registry cannot admit it.
4. A two-phone parallel-head / shared-B2-tail mechanism failed exact-token
   correctness. It is representable as a compound route only to prove the plane
   refuses to dispatch it.

## 3. Components

### `route_registry.py` -- ReadyRouteRegistry

- Immutable `RouteSnapshot`: wraps a `RouteConfig` plus device id, layer range,
  batch envelope, profile id, `READY/DRAINING/UNAVAILABLE` state, four epochs
  (route, residency, lease, device-boot), thermal ceiling/observed, finite
  execution credits, compound kind, and a correctness-certified flag.
- Admission recomputes a route content digest over the `RouteConfig` and rejects
  the snapshot if a caller-declared digest does not match. Any mutation of a
  profile id, evidence certificate digest, or batch size changes the digest and
  fails closed.
- A `READY` snapshot must be `SINGLE` and correctness-certified. Compound
  (serial-chain or shared-tail) and uncertified routes can never be `READY`.
- `install()` atomically replaces the whole route table and bumps a generation.
- `acquire_lease()` fails closed unless the route is `READY`, valid, thermally
  in-envelope, and has an unused execution credit. A `Lease` pins the four
  epochs, the device id, the profile id, and the generation.
- `validate_lease()` returns false once any pinned epoch changes, the route
  drains, or the lease was released. Stale snapshots and stale leases fail closed.

### `executor_contract.py` -- typed executor boundary

- `ExecutionRequest`: launch id, route/profile/device identity, all four route
  epochs, observed registry generation, exact request ownership, cohort and
  input-manifest file digests, timeout, and the expected boundary schema.
- `ExecutionResult`: outcome, finish time, boundary schema, and exactly one
  `BoundaryCertificate` per owned request on completion.
- `Executor` is an abstract interface. `RecordedExecutor` remains the
  deterministic replay implementation used by the frozen decision log.

### `physical_executor.py` -- host physical launch boundary

- `SubprocessSessionTransport` invokes a configured launcher without a shell
  and persists exact request, stdout, stderr, and transport metadata bytes.
- `PhysicalExecutor` requires a strict versioned session record bound to route,
  profile, device, route/residency/lease/boot epochs, registry generation,
  worker binary/generation, boot identity, layer range, and contiguous session
  id.
- Completed phone records require expected-backend compute, zero missing-buffer
  nodes, only declared CPU operations, and exactly one boundary certificate per
  request. Late replies become timeouts. Contract errors release the registry
  credit and require server fallback.

### `runtime_dispatch.py` -- MixedDispatchCoordinator

- Wraps one `PriorityBatchRuntime`; callers never choose a batch.
- Four isolated lanes: selected-A6000 BGE, OP15 Gemma head, OP12 Gemma head, and
  server fallback. Service/model/island/compatibility isolation is strict.
- Dispatches only against a `READY`, valid, credited registry lease. A cold,
  draining, or unready phone route is never waited on: the request takes server
  fallback immediately when no phone route can meet its latest start.
- OP15 and OP12 lanes may be in flight independently. No OP15 -> OP12 serial
  chain and no shared-tail compound action is ever formed.
- Every request reaches exactly one terminal state: `completed_phone`,
  `completed_server`, `tardy_result`, `fallback_required`,
  `rejected_backpressure`, or `timed_out`.

### `trace_adapter.py` -- frozen trace + synthetic SLO sidecar

- Ingests the frozen S14 `mix_v1.trace.jsonl` and an explicitly synthetic
  priority/SLO sidecar. BurstGPT and RAGPulse fields are never read as real
  deadlines or priorities; the trace's `deadline_us`/`priority_class` are null
  and stay null. The sidecar assigns synthetic priority classes and relative
  deadlines by service, tagged `s15-synthetic-sidecar`.
- Preserves deterministic `(t_us, event_id)` ordering and request conservation.
- Emits a canonical JSONL decision log that is byte-identical across processes
  and hash seeds (sorted keys, no set/dict iteration in output, no wall clock).

## 4. Fail-closed matrix (never dispatch)

unknown profile, uncertified batch point, digest mismatch, stale snapshot,
stale lease, wrong device, thermal violation, exhausted credit, queue
backpressure, duplicate ownership, mismatched completion certificate set,
invalid boundary certificate, compound/serial/shared-tail route.

## 5. Tests (see `tests/`)

Registry, replay executor, physical executor, coordinator, trace, conservation,
and cross-process determinism. The required proofs are enumerated in
`RESULTS.md`.

## 6. Boundary

The physical contract has one real OP15 B32 completion in the follow-on live
checkpoint. Arrival-faithful SLO remains blocked because the frozen binary saw
the prompt during preflight. OP12 live dispatch, concurrent phone lanes, energy,
and total-system benefit remain unmeasured.
