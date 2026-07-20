# S15 results: host-side mixed-workload dispatch mechanics

Verdict: `HOST_PHYSICAL_EXECUTOR_CONTRACT_PASS_LIVE_OP15_FOLLOW_ON_PASS`

Date: 2026-07-18. HEAD unchanged, no commit. Sources ASCII-only. New files only;
no S12/S13/S14 source, `layersplit.cpp`, `talks.md`, model, backend, scheduler,
or kernel file was edited.

## 1. What was built

A host control plane that turns measured S14 route profiles into deterministic,
executable dispatch decisions, plus a fail-closed physical launch boundary. Six
modules plus tests:

| File | Role |
|---|---|
| `route_registry.py` | `ReadyRouteRegistry`: immutable snapshots, content-digest pin, four-epoch + credit + thermal leasing |
| `executor_contract.py` | typed `ExecutionRequest`/`ExecutionResult` + `RecordedExecutor` (replay only, no ADB, no fabrication) |
| `physical_executor.py` | strict physical session parser + no-shell subprocess launcher with exact byte capture |
| `runtime_dispatch.py` | `MixedDispatchCoordinator`: wraps `PriorityBatchRuntime`, four isolated lanes, six terminal states |
| `route_fixtures.py` | builds READY snapshots from the measured S14 adapters + the ineligible compound routes |
| `trace_adapter.py` | frozen `mix_v1` trace + synthetic SLO sidecar -> canonical JSONL decision log |

Reused unchanged (imported, not edited): `power_frontier_policy.py`,
`priority_batch_runtime.py`, `live_profile_adapter.load_op15_head_route`,
`op12_profile_adapter.load_op12_head_route`.

## 2. Tests

S15 (`tests/run_all.py`): 78 tests, all pass after adding exact cohort and input
manifest identity binding through the executor request, outcome, lane, and
physical route binding.

| Test file | Tests | Proves |
|---|---|---|
| `test_route_registry.py` | 22 | adapters load eligible B1 routes and explicit B32 variants; OP12 `[0,8)` cannot be built/admitted; digest pin rejects profile/evidence/batch mutations; compound and uncertified routes cannot be READY; wrong device and thermal violation rejected; independent device leases; credit cannot overcommit; stale generation/route/residency/lease/boot epochs and drained routes fail closed |
| `test_executor_contract.py` | 12 | one certificate per request; full lease and workload-artifact identity; wrong schema / missing / extra / spurious certificate rejected; unknown launch times out; recorded-identity mismatch rejected |
| `test_physical_executor.py` | 12 | strict identity/session/placement/boundary gates; late-reply timeout; subprocess byte capture; measured OP12 route through the coordinator |
| `test_runtime_dispatch.py` | 17 | both phone lanes complete independently; measured B32 policy; priority and compatibility isolation; fallback/backpressure; executor or certificate failure releases its credit; four-lane isolation |
| `test_trace_adapter.py` | 11 | deterministic ordering; null trace deadline/priority preserved; synthetic sidecar; request conservation (177); policy-consistent terminal counts; byte-identical log across processes and hash seeds |
| `test_conservation.py` | 4 | all six terminal states reachable and distinct; conservation holds and detects a missing terminal; `tardy_result` reached |

Existing S14 scheduler tests (unchanged, still pass): `test_power_frontier_policy`
(15), `test_priority_batch_runtime` (17), `test_live_profile_adapter` (4),
`test_op12_profile_adapter` (9), `test_cp1` (21) = 66 pass.
The CP0 catalog suite also passes 22/22 with `/usr/bin/python3`, where the
required `jsonschema` module is installed. The active virtual environment does
not provide that module, so catalog validation must use the documented system
Python invocation.

## 3. Measured evidence this plane is pinned to

- OP15 `[0,8)` B1 route, 7-process profile, via `load_op15_head_route`
  (`stageb_op15_k8_b1_r0..r6.json`). Median route wall 1,253,550 us.
- OP12 `[0,6)` B1 route, 7-process profile, via `load_op12_head_route`
  (`op12_k6_b1_v2_r0..r6.json`), profile id `sha256:4de8d347...`. Median route
  wall 1,151,677 us.
- OP12 `[0,8)` is ineligible: the k8 repeatability cohort is incomplete/truncated,
  so `load_op12_head_route` raises and no `[0,8)` route can enter the registry.
- The two-phone parallel-head / shared-B2-tail and the OP15->OP12 serial chain are
  representable only as `SHARED_TAIL_PARALLEL_HEAD` / `SERIAL_CHAIN` compound
  routes; the registry refuses to make them READY and the coordinator refuses to
  dispatch them.

## 4. Decision log

`decision_log.jsonl` (frozen, `PYTHONHASHSEED=0`):
`sha256:24fe4cd0c48c68db0e25eefac49766aa2f8092ae7b3d1601de1e2b490469d755`,
177 lines, byte-identical under `PYTHONHASHSEED` 0/1/12345.

Terminal counts under the synthetic sidecar (interactive =priority 0, tight
300 ms deadline; batch api_generation =priority 1, 5 s deadline):

- `completed_phone` 152 (all low-priority api_generation, split 75 OP15 / 77 OP12)
- `completed_server` 25 (13 conversation_generation + 12 rag_qa; their tight
  synthetic deadline is not phone-feasible at a ~1.2 s B1 route, so they take
  immediate server fallback)

This is a dispatch-decision reproduction. Measured durations are used only for the
policy's latest-start feasibility gate. It is not a timeline, throughput, or
energy result.

## 5. Fail-closed matrix (all enforced and tested)

unknown profile / uncertified batch point / content-digest mismatch (profile id,
evidence digest, or batch size) / stale registry generation / stale route,
residency, lease, or device-boot epoch / drained route / wrong device / thermal
violation / exhausted execution credit / queue backpressure / duplicate admission
/ mismatched or spurious completion certificate / compound serial-chain or
shared-tail route -> never dispatches. Every request ends in exactly one of
`completed_phone`, `completed_server`, `tardy_result`, `fallback_required`,
`rejected_backpressure`, `timed_out`.

## 6. Confirmed limitations

1. The frozen decision log still uses `RecordedExecutor`. A separate follow-on,
   `../s15_live_launcher/`, has now exercised `PhysicalExecutor` once through the
   real coordinator against OP15/A6000 at B32. It is a physical-mechanics result,
   not an arrival-faithful SLO or energy result.
2. The selected-A6000 BGE lane uses a structural mechanics route
   (`route_fixtures.synthetic_bge_route`), NOT a measured BGE latency/energy
   profile. Only OP15 `[0,8)` and OP12 `[0,6)` are measured routes. The BGE route
   exists to exercise four-lane isolation, and its numbers are placeholders.
3. The trace driver has no occupancy/queueing-delay model and claims no latency.
   The phone/server split reflects placement policy plus the single-item
   latest-start feasibility gate, not a timed schedule.
4. Server fallback is modeled as an always-ready correct sink; no server latency
   or energy is measured, claimed, or computed. `completed_server` means the
   request was assigned to and accepted by the server lane, not a timing result.
5. The priority/SLO sidecar is explicitly synthetic (`s15-synthetic-sidecar`).
   BurstGPT/RAGPulse deadline and priority fields are null and stay null.
6. `fallback_required` is terminal here (the phone leg failed or its lease went
   stale mid-flight). The subsequent server re-execution is out of scope, so
   those requests are counted once and never as `completed`.
7. The single-pass driver admits and drains at the same clock; a fully
   asynchronous time-advancing event loop is not built. A runtime `NO_FEASIBLE`
   after feasible admission is a defensive fail (raises) rather than a silent
   mis-route.
8. The registry content digest pins the `RouteConfig` fields (route/profile id,
   epochs, per-batch evidence certificate ids, batch sizes/durations). It does
   not re-hash the phone weight or binary artifacts; that binding stays in the
   reused S14 adapters, which perform their own artifact-digest and raw-log
   checks.

## 7. Evidence or contract defects discovered

None in the reused S14 modules. The host integration audit found and repaired
three S15 defects: `ExecutionRequest` omitted `lease_epoch` and registry
generation; an executor contract error leaked its route credit; and a successful
reply observed after its timeout could be admitted. A fourth path left a request
inflight when runtime certificate validation rejected an invalid boolean.
Regression tests make all four checks load-bearing.

One API-shape observation drove the design:
`PriorityBatchRuntime` has no eviction path for pending items that become
infeasible, so the coordinator applies the latest-start feasibility gate at
admission (before enqueue) and treats a later runtime `NO_FEASIBLE` as a
fail-closed error. No change was made to the S14 API.

## 8. Claim boundary

This checkpoint delivers the host dispatch plane and physical executor
contract. The follow-on live checkpoint establishes one real coordinator-driven
OP15/A6000 B32 completion. It does not establish arrival-faithful SLO, repeated
latency, energy, throughput benefit, OP12 dispatch, or total-system benefit.
