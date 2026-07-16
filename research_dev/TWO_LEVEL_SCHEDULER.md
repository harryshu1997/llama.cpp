# Two-level power-frontier scheduler

Status: authoritative scheduler design as of 2026-07-15.

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

Per-GEMM network row/column splitting is excluded from the core catalog.
Any future cooperative plan must still be one complete island cover with one
input boundary, one output/merge boundary, exact state ownership, and its own
measured certificate. The runtime never invents graph cuts or takes a cross
product of independently measured components.

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
~~~

### Decisions

The slow loop decides:

1. which complete island weight sets reside on each phone;
2. which HTP/GPU prepared images exist and alias canonical bytes;
3. mirrored versus exclusive ownership;
4. lease/hold horizons, hysteresis, drain, reload, and eviction;
5. certified route, batch, interference, phone-pace, and server-power envelopes;
6. bounded prefetch using visible queue reuse; and
7. a fixed exploration-byte budget for uncertain future demand.

Prefetch value is based first on actual announced demand:

~~~text
visible_reuse_relief >
  transfer + verify + prepare + eviction + memory_opportunity_cost
~~~

Forecast-only placement is allowed under the exploration budget, but it cannot
earn a current decision benefit until later used.

### Publication

Planning emits idempotent intents. Only observed transfer, verification, atomic
publish, preparation, warmup, correctness, memory, and thermal completion may
publish a READY execution envelope.

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
artificial energy win.

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

For each candidate server batch b at power state p:

~~~text
latest_start(b,p) =
  min over i in b (
    milestone_deadline(i)
    - conservative_remaining_path(i,b,p)
  )
~~~

The server launches when:

- the energy-optimal compatible batch size is reached;
- a lower-state break-even interval would otherwise be lost;
- the earliest latest_start is reached; or
- a fault/thermal change invalidates the phone plan.

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
provisioning yields to activation/result traffic.

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

Before any live server scheduler:

1. S10-V0 must show a perfect-future opportunity against an optimized
   server-only batching/DVFS baseline.
2. A causal bounded policy must retain the required benefit.
3. The physical or conservatively validated power model must explain the gain
   through a real batch/power-state change.
4. The small real-device test must reproduce the mechanism without hidden state,
   transfer, fallback, or energy.

If any gate fails, preserve S9 as transport/residency evidence and stop Q-PIM
runtime work.
