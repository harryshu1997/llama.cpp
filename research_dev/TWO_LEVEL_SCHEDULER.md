# Two-level power-frontier scheduler

Status: deferred broad scheduler design as of 2026-07-19; not paper-critical.

The focused Q-PIM Funnel work fixes phone weight residency before measurement
and implements one online route/admission/release loop. General DAG lookahead,
dynamic residency, power bundles, solver optimality, and DVFS remain future
work. `MIXED_WORKLOAD_DESIGN.md` and the S19 plan are authoritative for the
current research system.

MIXED_WORKLOAD_DESIGN.md defines Q-PIM. This file defines its slow residency
planner, fast dependency-aware scheduler, exact oracle, and decision invariants.
NEXT_PLAN.md defines when any implementation is authorized.

## 1. Scheduler thesis

The scheduler does not maximize phone utilization or skipped GPU-us. It changes
the topological order of independent multi-model islands so that:

1. phone-resident descendants become READY early;
2. the remaining A6000 work forms denser batches or contiguous active bursts;
3. the A6000 can use a measured lower cap/state for a break-even interval; and
4. total wall energy decreases without violating an end-to-end SLO.

The first implementation controls one selected A6000. Any additional installed
GPU is excluded from the candidate-device set and cannot absorb overflow or
contribute hidden work. OP12 and OP15 are the only additional execution devices.

The two-level invariant is:

~~~text
slow loop publishes certified resident capabilities and ownership modes;
fast loop reorders and dispatches only concrete READY islands within them.
~~~

Weight prediction never authorizes execution. Future arrivals never appear as
concrete work. An ANNOUNCED island may trigger background prefetch; only a READY
island may execute.

## 2. Why two levels remain necessary

| Level | Horizon | Decisions | Forbidden |
|---|---|---|---|
| Slow residency/power planner | seconds to minutes | weights, prepared images, leases, mirrored/exclusive ownership, route and power envelopes | assigning a concrete request or claiming planned readiness |
| Fast power-frontier scheduler | every arrival/completion | topological order, unlockers, bundles, batches, latest claims, route, phone pace, atomic dispatch | downloading, warming, eviction, arbitrary graph cuts, or uncertified routes |

S9 transfer/prepare operations are seconds-scale. A request-time route therefore
never waits for a cold weight.

## 3. Work and dependency model

The host stores a bounded union of admitted request DAGs:

~~~text
V = concrete operator-island invocations
E = data, state, side-effect, and ordering dependencies
~~~

Each invocation records:

~~~text
work/request/service/model/version/island IDs
predecessor and successor IDs
ANNOUNCED/READY/ASSIGNED/RUNNING/terminal state
input/output handles and maximum bytes
state owner, lease, mutation sequence, replay boundary
arrival and TTFT/TBT/completion SLO milestones
batch/active-burst compatibility keys
candidate plan/profile/correctness digests
latest server claim under each certified batch/power state
~~~

The queue may reorder only nodes whose predecessors are complete and whose
side-effect class permits it. Physical backend queues remain shallow and bounded.

## 4. Certified plan catalog

Every island has a finite catalog of complete plans:

- server_only at a measured A6000 batch/power point;
- whole stateless island on one READY phone backend;
- contiguous layer/stage route with explicit state ownership;
- independent HTP/GPU co-run certified by an interference profile;
- same-version native batch;
- fixed multi-stage route such as A0; and
- optional grouped-HMX island only after its dedicated gate.

Each server plan binds the selected GPU identity, roofline class, batch size,
power cap or clock point, latency distribution, throughput, energy boundary,
and uncertainty. A phone route cannot inherit a server operating point measured
on another GPU or under a different co-run condition.

Per-GEMM network row/column splitting is excluded from the core catalog.
Any future cooperative plan must still be one complete island cover with one
input boundary, one output/merge boundary, exact state ownership, and its own
measured certificate. The runtime never invents graph cuts or takes a cross
product of independently measured components.

Stateful layer plans are selected from a finite catalog of independently
certified contiguous ranges, initially `[0,k)`. Each candidate binds its exact
graph, memory, boundary, KV, latency, thermal, batch, and correctness evidence.
The selected cut, device, residency generation, and KV owner remain fixed for a
request lifetime. Dynamic selection means choosing a catalog entry for new
requests, not changing `k` during decode.

## 5. Slow residency and power-envelope planner

### Inputs

~~~text
causal demand summary from ANNOUNCED and active DAGs
island/route/power/thermal/interference atlas
server HBM and A6000 power-state inventory
phone UFS/LPDDR/prepared-image/state ledgers
directional USB profiles and contention domains
current weight, state, ownership, and thermal leases
transfer, verify, prepare, warm, drain, and reload costs
useful/unused/retried prefetch history and compute/transfer interference
~~~

### Decisions

The slow loop decides:

1. which model, certified island/range, and residency slot belongs on each phone;
2. whether OP12/OP15 replicate a hot island or hold diverse model islands;
3. which HTP/GPU prepared images exist and alias canonical bytes;
4. mirrored versus exclusive ownership;
5. active/staging slot generations, transfer order, and bounded byte windows;
6. lease/hold horizons, hysteresis, drain, reload, and eviction;
7. certified route, batch, interference, phone-pace, and server-power envelopes;
8. bounded prefetch using visible queue reuse; and
9. a fixed exploration-byte budget for uncertain future demand.

Prefetch value is based first on actual announced demand:

~~~text
visible_reuse_relief >
  transfer + verify + prepare + eviction + memory_opportunity_cost
  + measured_interference + expected_unused_prefetch
~~~

Forecast-only placement is allowed under the exploration budget, but it cannot
earn a current decision benefit until later used.

### Publication

Planning emits idempotent intents. Only observed transfer, verification, atomic
publish, preparation, warmup, correctness, memory, and thermal completion may
publish a READY execution envelope.

Each phone may expose one bounded STAGING slot beside an incumbent READY/ACTIVE
slot only when canonical bytes, derived images, scratch, activation, state, and
both slot charges fit simultaneously. Slots have independent generations.
Publishing the staging slot cannot revoke incumbent leases; replacement drains
the incumbent after the new envelope is usable. The current phone-PIM worker's
single global generation is therefore insufficient for overlapped replacement.

~~~text
ExecutionEnvelope {
  envelope/policy/snapshot epochs and expiry
  exact plan and profile digests
  model/weight/allocation/prepared-image/correctness chain
  device/backend/boot/generation identities
  shape, batch, boundary, state, thermal, and power bounds
  mirrored/exclusive ownership and drain/recovery rule
}
~~~

If a solve or publication fails, the previous valid envelope remains until
expiry; otherwise the route is server-only or explicitly unavailable.

## 6. Fast power-frontier scheduling

At each event, the scheduler performs bounded receding-horizon planning:

1. Materialize newly READY nodes after dependency completion.
2. Build an H-hop lookahead over admitted DAGs only.
3. Revalidate every candidate against current envelope, ownership, lease, state,
   memory, link, lane, thermal, and correctness epochs.
4. Enumerate READY unlockers whose completion exposes resident phone descendants.
5. Generate phone power-trigger bundles and compatible A6000 batches/bursts.
6. Compute latest SLO-safe A6000 claim and wake points.
7. Evaluate complete counterfactual wall-energy schedules.
8. Choose by the fixed lexicographic objective and total tie key.
9. Atomically reserve every lane, link, activation, state, pin, ownership, and
   fallback resource used by the first compound action.
10. Commit the first action, then replan after the next event.

The search never delays a request beyond its latest safe claim to create an
artificial energy win. It never waits for STAGING weights: if no READY phone
envelope exists, the current SLO-safe server route or declared terminal outcome
wins.

## 7. Frontier expansion and unlockers

The current phone frontier is:

~~~text
F_phone(t) = {
  v | READY(v),
      route and weights READY,
      state/thermal/link valid,
      finish before the relevant SLO
}
~~~

For a READY candidate u, the H-hop descendant set D_H(u) defines what u can
unlock. Unlocking value is non-additive. A candidate set U is evaluated through
the schedule it enables:

~~~text
frontier_value(U) =
    E_star(current optimized baseline)
  - E_star(schedule after executing U)
  - added live-activation cost
  - transfer/churn/waste cost
~~~

The evaluator credits only descendants that become concretely executable within
the horizon. It does not count support-only or forecast-only nodes.

Autoregressive future tokens are absent until sampling. Current-token islands
from different streams are independent candidates when their actual inputs are
READY.

## 8. Power-trigger bundle generation

Bundle kinds:

~~~text
SLEEP_BUNDLE
  offload every island that would interrupt one break-even idle interval

BATCH_SHAPING_BUNDLE
  move urgent/rare/incompatible nodes so remaining server nodes batch densely

POWER_CAP_BUNDLE
  remove enough critical work to make a lower A6000 cap SLO-feasible

MEMORY_BUNDLE
  promote a complete route to exclusive phone ownership and release HBM
~~~

For candidate B:

~~~text
credit(B) =
    E_star(server-only with best DAG order, batching, and power)
  - E_star(joint schedule with B)
~~~

The comparison uses the same closed request cohort, terminal accounting, and
power boundary. A per-island sum is invalid when only the bundle changes the
power state.

## 9. Lazy claims and batch densification

Batch size is an online decision, not a route constant. B32 and B64 in the
physical spikes are measured profile points, not production launch rules. For
each device d, island i, context class c, thermal state t, and power state p,
the atlas publishes a finite set of certified batch candidates:

~~~text
B(d,i,c,t,p) = measured supported batch sizes with
               latency, throughput, memory, boundary, interference,
               correctness, and downstream-service envelopes
~~~

The fast loop does not interpolate an unmeasured batch size. A candidate b is
feasible only when:

~~~text
resident_and_correct(d,i,b)
memory(d,i,b,c) <= currently reservable memory
predicted_finish(d,i,b,c,t,p) <= earliest affected latest_start
boundary_bytes(i,b) fit reserved link and activation credits
every later stage has a compatible reserved batch/queue envelope
~~~

For each feasible batch b at power state p:

~~~text
latest_start(b,p) =
  min over i in b (
    milestone_deadline(i)
    - conservative_remaining_path(i,b,p)
  )
~~~

Each phone or server lane launches when:

- its measured throughput/energy sweet region is reached;
- a lower-state break-even interval would otherwise be lost;
- the earliest latest_start is reached; or
- a fault/thermal change invalidates the phone plan.

For memory-bound decode, the fast loop maintains compatibility queues keyed by
model version, island cut, KV/state owner, attention class, and shape envelope.
It chooses the largest useful measured batch that can finish before the earliest
member's latest start and that does not overload a later island. A batch larger
than the measured knee is useful only when its profile shows additional
throughput or energy benefit. If a queue exceeds the selected size, the lane
launches multiple measured microbatches rather than inventing one oversized
batch. Low-priority work may wait inside its bounded slack; urgent work bypasses
the wait or takes its certified server route.

For compute-bound encode, prefill, or stateless islands, the fast loop chooses
the measured point with the best SLO-feasible throughput/energy tradeoff. This
is often the smallest batch in the saturation region, but memory pressure,
concurrent lanes, downstream batching, or a different power state may move the
point. Batch 1 or 2 is not a normal phone operating point. It is used only when
an explicit profile and an imminent SLO make it preferable to waiting or taking
the server fallback.

### Continuous batching

All stateful decode executors, on phones and the server, use continuous
batching. A persistent executor owns a bounded KV-slot table. At each token
boundary it:

1. retires completed or cancelled sequences and releases their exact slots;
2. admits compatible READY sequences whose route, state owner, and epochs are
   already certified;
3. chooses the next measured batch size from the current active set and SLOs;
4. executes one decode step for that ragged active set; and
5. publishes request IDs, sequence positions, mutation epochs, and the exact
   boundary manifest consumed by the next stage.

Continuous batching changes membership between decode steps; it does not move
a live request to a different layer cut or KV owner. Prefill may be chunked and
inserted at certified chunk boundaries. Stateless embedding/reranking queues
may form new microbatches at every completion event.

For a multi-stage route, upstream fullness is not optimized independently. The
scheduler selects a compatible batch vector `(b_phone, b_middle, b_tail)` and
reserves the downstream credits before launching the upstream work. A slow
producer is therefore optional capacity, never a barrier. If its result will
miss the next useful downstream batch or latest start, it is not launched and
the work remains on a direct phone-to-server or server-only route.

When the selected A6000 is compute-bound, a compute-pressure bundle may move
READY low-priority islands or compatible decode batches to OP12/OP15. Returned
phone boundaries become eligible for a server suffix batch only after D2H and
epoch validation. The selected GPU changes cap or clock only when the measured
joint plan has positive energy credit and remains feasible for every affected
SLO. Otherwise the scheduler retains the normal operating point.

The decision epoch evaluates at least these four counterfactuals over identical
ready work:

~~~text
normal GPU state, server-only batching
lower GPU state, server-only batching
normal GPU state, phone relief plus server batching
lower GPU state, phone relief plus server batching
~~~

This separates savings caused by phone execution, batch densification, and the
GPU operating-point change.

Mirrored mode permits an exact phone/server race. The phone wins only after D2H,
verification, and epoch validation. At latest_start, an unfinished phone result
loses and server execution starts. Wasted phone work is counted.

Exclusive mode has no immediate server HBM fallback. Its conservative phone
completion and lifetime lease are part of the critical path. Demotion requires
drain and completed server reload before ownership changes.

## 10. Phone pacing and local lanes

For eligible route r and operating point f:

~~~text
choose f minimizing measured phone joules(r,f,thermal)
subject to phone_finish(r,f) <= bundle_deadline
~~~

If direct DVFS is unavailable, the controllable operating point is backend,
performance mode, concurrency, batch size, and duty cycle. HTP and GPU may run
independent islands concurrently only with a measured pair profile. Bulk
provisioning yields to activation/result traffic at a bounded chunk boundary.
WiFi H2P commands and USB P2H results are modeled separately, while USB H2P
weights share the phone USB contention domain with results until a measured
duplex profile proves otherwise.

A thermal bucket change revokes future use of its old performance profile. A
stateful exclusive lease either retains a conservative route, drains safely, or
takes its declared terminal outcome.

## 11. Hard constraints

Every plan enforces:

- one plan or explicit terminal outcome per admitted island/request;
- complete DAG precedence and side-effect order;
- READY-before-use and exact model/weight/prepared-image identity;
- finite server HBM, phone UFS/LPDDR, activation, state, scratch, link, and lanes;
- batch compatibility and explicit different-model burst semantics;
- exact state owner and one mutation in flight;
- lease, use-pin, epoch, and DRAINING rules;
- independent active/staging generations and simultaneous memory accounting;
- no dispatch from partial, on-disk-only, or STAGING content;
- result-before-bulk priority and useful/unused/retried/evicted byte accounting;
- D2H completion before credit/resource release;
- mirrored versus exclusive fallback accounting;
- TTFT, TBT, completion, and bounded-wait constraints;
- server and phone power-state transition timing;
- thermal duty and cooldown; and
- one terminal result for all work at the horizon.

No correctness, ownership, or SLO constraint is traded for energy.

## 12. Objective and exact oracle

The fixed lexicographic objective is:

~~~text
L1 minimize rejected, timed-out, and SLO-missed work by priority
L2 minimize maximum and total lateness
L3 maximize completed useful work
L4 minimize synchronized total wall joules
L5 minimize peak server HBM and HBM byte-us
L6 minimize makespan, transfer/churn, and wasted phone work
~~~

For iso-power experiments, L4 is replaced by a hard wall-power/energy budget and
L3 remains the optimized level.

The first oracle supports tiny bounded instances and has three independent
parts:

1. standard-library exhaustive enumeration/branch-and-bound;
2. standalone certificate checker sharing no solver/evaluator code; and
3. later, a pinned one-worker CP-SAT model that must exactly agree on fixtures.

All units are bounded integers or checked rationals. Durations and costs round
against the claimed benefit. Lexicographic levels solve sequentially. Only a
proved optimum is an exact-oracle result.

## 13. Deployable bounded policy

The online policy does not solve the full NP-hard model. It:

- limits lookahead by nodes, edges, and wall time;
- retains the K best topological frontier states with deterministic beam search;
- enumerates only catalog bundles and measured power states;
- uses hysteresis for weight/ownership changes;
- commits one action per event; and
- falls back to the best SLO-safe incumbent on timeout.

The offline oracle reports the opportunity and online gap. A clairvoyant oracle
cannot authorize implementation when the causal bounded policy fails.

## 14. Required decision record

Each decision records:

~~~text
snapshot/envelope/objective versions
ready frontier and H-hop dependency closure
candidate unlocker and trigger bundles
all hard-gate failures
baseline and joint batch/power schedules
predicted total-wall energy and latest claims
selected route, ownership, pace, reservations, and reason
actual start/finish/bytes/power/thermal/fallback/waste
terminal outcome and prediction/oracle error
~~~

Primary reason codes include unsupported, not_ready, not_resident, dependency,
state_pinned, transfer_dominated, thermal, stale, no_power_trigger,
server_energy_better, batch_shaping, power_cap, sleep_gap, and memory_relief.

## 15. Authorization gate

S14 authorizes bounded research-harness work in this order:

1. deterministic mixed composition plus two executable service/model classes;
2. measured profile adapter and static READY mixed replay using the existing
   S12-V2 residency reducer;
3. exact tiny placement oracle plus a causal bounded policy;
4. live two-phone static placement through the S13 session substrate; and
5. per-slot-generation weight streaming only after static mechanics pass.

This does not authorize `llama-server` integration. A production server path
requires causal real-trace mechanics against optimized server-only, measured
transfer/compute interference, bounded 30-minute behavior, and a physical batch,
power, or memory mechanism that survives its claim boundary. Energy acquisition
starts only after those mechanics pass. If a gate fails, preserve S8/S9/S12/S13
as trace, residency, scheduler-mechanics, and runtime evidence rather than
silently widening the claim.
