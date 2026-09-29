# S42 runtime and semantic gates V1

## Admission transaction

A certified route is no longer admitted from historical readiness flags. The
route must carry an exact runtime contract and the executor must provide a
fresh immutable snapshot. Immediately before lease commit, the scheduler
checks the contract again. A rejected treatment falls back before any phone or
USB lease is installed.

The dynamic snapshot covers:

- static epoch key and snapshot generation;
- readiness, worker heartbeat, circuit state, failures, and USB reset
  generation;
- thermal and contention buckets plus a conservative slowdown ceiling;
- exact model, worker, bridge, and transport residency IDs; and
- a bounded validity interval.

Every declared route resource must have a requirement. Missing telemetry does
not mean healthy: it becomes `unqualified`, `null`, or an open circuit and the
route is rejected.

## Request semantics

The request declares KV ownership and whether it needs KV migration, context
shifting, full logits, grammar, a sampler location, speculative decoding, or
cooperative cancellation. These are compared with route capabilities before
queue prediction.

The I3 CPU plus OP15 route currently guarantees:

- KV remains owned by `llama_context` on the desktop;
- no KV migration;
- full logits and the sampler remain on the desktop; and
- cancellation is safe before dispatch.

Context shifting, grammar qualification, speculative decoding, KV migration,
and cooperative mid-flight cancellation are not part of the measured I3
certificate. Requests requiring them stay on a separately qualified baseline.
The treatment also remains approximate quality.

## Failure and cancellation

`ResourceTimeline.revoke_resource` closes admission, truncates future leases,
and returns every affected owner. It does not silently replay partially
executed work. The server adapter must fail or explicitly recover those owners
according to its request transaction.

`cancel_owner` releases all current and future resource phases at the observed
cancellation time. A request already marked cancelled is rejected before it
acquires any lease. Snapshot generations must increase, preventing stale state
from replacing a newer failure observation.

## I3 contract

`runtime_routes_v1/I3_RUNTIME_GATE_CONTRACTS_4060TI_OP15_V1.json` binds both
compiled cohort routes to epoch:

```text
sha256:27d004d2958af9df39e4543eda7853cca68f9293df7aba3515bed3585775a1e5
```

The treatment requires the exact Gemma and Qwen model residency, exact bridge
and worker artifacts, Android thermal status `NONE`, no NVIDIA or CPU thermal
throttle observation, the measured I3 process topology, a fresh bridge and
worker heartbeat, no failure, and a verified reset generation of zero.

The Stage 6 physical sidecar samples the bound FunctionFS gadget and exact
bridge/server topology nominally once per second. Its measured maximum gap was
1.051465 seconds, so the compiled OP15 heartbeat bound is 1.1 seconds. RPC
progress is a work counter, not an idle heartbeat: the 7.476-second maximum
gap between RPC bursts is retained as a diagnostic and does not indicate a
dead worker while the bound gadget and bridge topology remain healthy.

## Read-only idle audit

The physical idle probe on 2026-08-06 observed:

| Signal | Observation |
| --- | --- |
| RTX 4060 Ti | 43 C, no NVIDIA thermal slowdown |
| Desktop CPU | package throttle counter unchanged during probe |
| OP15 | Android thermal status 0, maximum current HAL sensor 32.1 C |
| USB | 5,000 Mb/s SuperSpeed |
| Artifacts | expected bridge and worker files hash correctly |
| Running topology | no hot server, cold server, bridge, or worker |

Both routes correctly remain ineligible because the models, bridge, and worker
are not running and no heartbeat or reset-generation receipt exists. This is
not a failed test; it proves an idle machine cannot be mistaken for a ready
execution epoch.

Receipts:

- `RUNTIME_GATE_IDLE_SNAPSHOT_4060TI_OP15_V1.json`, file SHA-256
  `0ccc73c384537cfb3082e3fea8ebd78affaf03e6b9d9a91a7aa867b672ce23de`;
- `RUNTIME_GATE_IDLE_AUDIT_4060TI_OP15_V1.json`, file SHA-256
  `9e284c54a57ab355943f96d297385d5b6607e829be2a68b190a36398d4fceb7f`.

After Stage 6 measured the 1.051465-second topology-sampling gap, the OP15
heartbeat allowance changed from 1.0 to 1.1 seconds. The immutable V1 audit is
retained; `RUNTIME_GATE_IDLE_AUDIT_4060TI_OP15_V2.json` re-audits the same idle
snapshot against contract file SHA-256
`1057516c4d306ab98194721d22b6059093d8d17139a34045c5e2898837f7e5a4`.
Its file SHA-256 is
`27dc2d459d6ea78a85f16583d475c37ed42730f0602365982b762ff78701178f`;
both routes remain correctly rejected while idle.

Stage 6 must launch and hash the complete executor topology, establish live
heartbeats, verify reset generation, then publish a short-lived successor
snapshot before the physical A/B run.

## Stage 6 physical validation

The monitored successor pair completed on 2026-08-07. Both arms preserved the
exact model, server, bridge, worker, trace, and dynamic-policy epoch. All paid
topology, CPU, GPU, Android thermal, 5 Gb/s USB, and reset gates passed. The
treatment completed all 52,320 RPCs with zero recovery. A first receipt
bounded RPC conservation at `paid_end` and saw 52,288 calls; it is preserved
as a failed boundary diagnostic. V2 uses paid topology for liveness and the
final bridge summary for exact work conservation, yielding 52,320 calls and a
passing receipt without changing the physical result.

This is physical post-run validation of the executor invariants. A production
adapter must still publish the same snapshot before atomic lease commit and
revoke the route online if its heartbeat becomes stale.
