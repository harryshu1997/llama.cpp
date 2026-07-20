# Q-PIM Funnel: heterogeneous continuous activation batching

Status: authoritative focused paper scope as of 2026-07-19.

Q-PIM Funnel uses two server-managed phones as resident prefix accelerators for
one busy A6000. The phones hold fixed Gemma-4-12B prefix weights and prefix KV.
They return boundary activations to a small cut-normalization stage and one
shared continuously batched CUDA tail. This is PIM-style function shipping,
not coherent memory mapping.

The paper asks one question:

> Can heterogeneous phone prefixes feed one continuously batched GPU suffix,
> through a small cut-normalization stage, while preserving a high-priority
> service SLO and reducing one-GPU HBM or selected-GPU energy per completed
> mixed workload?

The exact first topology is:

~~~text
high-priority BGE ------------------------------------> A6000

low-priority Gemma on OP12 [0,6) -> CUDA bridge [6,8) --+
low-priority Gemma on OP15 [0,8) -----------------------+-> one CUDA tail [8,48)

tight-SLO Gemma -> CUDA head [0,8) ---------------------+
~~~

The bridge and tail have disjoint weights. A server fallback uses a CUDA
`[0,8)` head and the same tail rather than a second full model. The initial
prototype may duplicate the small `[6,8)` range between that head and the
bridge, which must be measured explicitly; it may not duplicate `[8,48)`. All
requests enter the tail at layer 8, so server, OP12, and OP15 rows can share one
native tail batch and one tail weight image. Phone weights are provisioned
before the measured run. The second A6000 is excluded.

## Frozen contributions

The work claims at most three system contributions:

1. **Heterogeneous-cut normalization.** Device-specific prefix depths are
   converted to one canonical merge cut without duplicating the common GPU
   suffix. This allows a slower phone to offload fewer layers while a faster
   phone offloads more.
2. **Pipeline-wide continuous batching.** A request retains one sequence and KV
   owner in every stage. Requests may enter and leave at token boundaries while
   other requests continue. Ready normalized activations from both phones form
   one changing CUDA-tail batch.
3. **Deadline-aware merge release.** Each phone and the shared tail release at a
   measured useful batch candidate or the earliest latest-start time. The
   scheduler chooses only server-only, OP12-prefix, or OP15-prefix routes and
   never waits unconditionally for B32.

These are one mechanism: heterogeneous prefixes create ready activations,
continuous batching coalesces them at a canonical cut, and the release rule
prevents batching from violating the SLO.

Authority order:

1. this focused section defines the system and claim boundary;
2. spikes/s19_dynamic_batch_runtime/PLAN.md defines the active implementation;
3. NEXT_PLAN.md defines the executable gate order;
4. versioned schemas define record bytes after their policy is frozen; and
5. spike RESULTS files define measured evidence.

Historical Design A remains route A0 and reusable pipeline/KV/sharding
substrate. S8 provides trace contracts. S9 provides weight-residency,
phone-runtime, and transport substrate. S10's general solver and the broad
TWO_LEVEL_SCHEDULER design are future-work substrate, not paper-critical gates.
None is the new claim by itself.

## 1. Research question and objective

The evaluated workload has exactly two services:

1. high-priority BGE-small-en-v1.5 embedding on the selected A6000 at its
   measured batch knee; and
2. low-priority Gemma-4-12B generation routed through server-only, OP12-prefix,
   or OP15-prefix execution under TTFT/TBT/completion limits.

The scheduler decides route, admission, and batch release. Weight placement is
fixed before the run. It does not solve a general DAG, change a live request's
cut, or stream weights on the request critical path.

The long-term objective remains total system energy:

~~~text
minimize integral(
    server wall power
  + host relay power
  + USB/charger power
  + phone wall power
) dt
~~~

The current measurable objective is narrower: reduce peak selected-GPU HBM and
selected-GPU board J/completed equal-work request while preserving both SLO
classes. SLO-valid goodput at a fixed selected-GPU power boundary is a second
primary metric.

Capacity and HBM relief remain reportable secondary results. They are never
presented as energy savings without a measured physical power change.

While phone energy is physically unavailable, an interim experiment may report
server-side relief at an explicit boundary:

~~~text
delta_E_server = E_server_control - E_server_qpim
phone_plus_external_break_even_budget = delta_E_server
~~~

`GPU_BOARD` supports only GPU-board relief; `SERVER_WALL` supports only server
relief. A positive delta says that excluded phone, USB, charger, and relay energy
may consume at most that break-even budget. It is not total-system energy saving.
Only a synchronized `TOTAL_WALL` boundary can support the primary energy claim.
Unknown phone energy is never represented as zero.

## 2. Scope and non-goals

Paper-critical service classes:

- Gemma-4-12B stateful autoregressive generation; and
- BGE-small-en-v1.5 stateless embedding as concurrent foreground pressure.

The system does not initially provide:

- coherent server-phone virtual memory;
- dynamic weight replacement or arbitrary-model admission;
- a general multi-model DAG scheduler or NP solver;
- phone-to-phone R2 execution;
- GPU DVFS or power-state control;
- arbitrary network micro-operator scheduling;
- arbitrary per-GEMM server/phone row or column splitting;
- future-token execution before autoregressive dependencies exist;
- token-prefix KV ownership across all model layers;
- unsupported-kernel fallback hidden inside a phone route;
- an energy estimate inferred from latency, GPU-us, or utilization;
- a total-system energy claim without a valid phone/USB power boundary; or
- production llama-server integration before the S19 physical gates pass.

The broader dependency-aware frontier, dynamic residency, weight streaming,
phone-to-phone middle stages, arbitrary service catalog, and total-wall solver
described later in this file are retained as future work. They are not required
to validate or publish the focused system above.

## 3. Core abstraction: certified operator island

An island is a complete replayable subgraph with a small explicit boundary. An
island descriptor binds:

~~~text
island_id, service_class, model/version, graph_hash
predecessor/successor IDs and side-effect class
input/output tensor schema and maximum bytes
weight_set and prepared-image identities
state mode and state/KV owner
supported device/backend routes
shape/batch envelope and kernel path
correctness, latency, energy, interference, and thermal profile IDs
fallback/replay boundary and SLO milestone
~~~

A route is eligible only when support, correctness, boundary, residency, state,
memory, link, thermal, and SLO gates are known and pass. UNKNOWN is ineligible.
An arbitrary operator is not promoted to an island merely because a backend can
execute it.

Stateful transformer candidates come from a finite pre-certified cut catalog.
For the first implementation they are contiguous head ranges `[0,k)`, with a
distinct graph hash, memory/thermal profile, boundary, and correctness record for
each `k`. The slow loop may select a cut for a new request, but it cannot change
that request's cut, phone, generation, or KV owner during its lifetime. Larger
ranges are new experiments, not a post-hoc rescue of the failed S11-E0 `[0,2)`
energy route.

## 4. Architecture

~~~text
 real traces + live request DAGs          measured route/power atlas
                |                                   |
                v                                   v
        multi-model DAG catalog             device/link inventory
                |                                   |
                +---------------+-------------------+
                                v
                global virtual operator queue
                    ANNOUNCED         READY
                        |               |
                        v               v
              SLOW residency       FAST power-frontier
              and envelope plan    receding-horizon plan
                        |               |
                        +-------+-------+
                                v
                 versioned dispatch/lease authority
                      |                     |
                      v                     v
            A6000 executor/power      phone agents
            batch, cap, idle state    HTP + GPU + state
                      |                     |
                      +----------+----------+
                                 v
                  completion, wall-power, thermal,
                  queue, link, and failure feedback
~~~

The host owns request, route, ownership, and mutation epochs. Physical A6000,
HTP, and GPU queues stay shallow. The host virtual queue holds reorderable work
and lookahead state.

## 5. Multi-model DAG and virtual queue

Every admitted request is a DAG G=(V,E), where a node is a certified island and
an edge is a data, state, or ordering dependency. Queue states are:

~~~text
ANNOUNCED -> READY -> ASSIGNED -> RUNNING -> COMPLETE
                     |                       |
                     +-> CANCELED/FAILED ----+
~~~

- ANNOUNCED means future weight demand is known; weights may be prefetched.
- READY means every predecessor completed and concrete inputs exist.
- ASSIGNED names one committed route or one explicitly mirrored race.
- COMPLETE publishes verified output and any state mutation exactly once.

Q-PIM may reorder independent nodes across requests and models. It cannot
execute an ANNOUNCED node, rewrite a dependency, or reorder an externally
visible side effect.

## 6. Dependency-aware frontier shaping

For device d at time t:

~~~text
F_d(t) = {
  v | predecessors(v) complete,
      weights(v) READY on d,
      state route valid,
      boundary and finish remain SLO-feasible
}
~~~

Q-PIM does more than choose from the current frontier. It may prioritize a READY
unlocker whose completion exposes valuable phone-resident descendants within a
bounded H-hop lookahead.

~~~text
A0(server) -> A1(phone) -> A2(server)
B0(server) -> B1(phone) -> B2(server)

reordered:
server: A0, B0 | low-power or other work | A2, B2
phones:     A1, B1
~~~

Compatible same-model suffixes form a real batch. Different-model suffixes may
still form one contiguous A6000 active burst that avoids repeated state
transitions. Reordering cannot increase backend support; it increases the
fraction of eligible islands that become READY with enough useful slack.

The scheduler charges additional live activation bytes, weight churn, USB
traffic, and critical-path delay caused by reordering.

## 7. Power-trigger bundles and lazy claiming

Skipped GPU-us receives no energy credit by itself. Q-PIM selects coordinated
sets whose removal changes the optimized server schedule:

- SLEEP_BUNDLE removes every island interrupting a candidate low-power interval.
- BATCH_SHAPING_BUNDLE moves urgent, rare-model, or incompatible batch spoilers.
- POWER_CAP_BUNDLE removes enough critical work to meet SLOs at a lower cap.
- MEMORY_BUNDLE promotes a complete weight island to exclusive phone ownership.

For bundle B:

~~~text
credit(B) =
    optimized_server_only_wall_energy
  - optimized_joint_wall_energy_with_B
~~~

Both counterfactuals use their best valid batching, DAG order, and power policy.
Bundle B is useful only if credit(B)>0 and all SLO and safety constraints pass.

For server island i:

~~~text
latest_start(i, batch, power_state) =
    deadline(i) - conservative_execution_time(i, batch, power_state)
~~~

The server claims work at the latest SLO-safe point that permits a useful batch
or power interval. It wakes or claims earlier if no valid bundle exists. A
low-power interval is credited only when:

~~~text
gap >= wake_latency
     + transition_energy / (idle_power - lower_state_power)
~~~

The actual A6000 state machine and transition cost must be measured. The design
does not assume that a deep sleep state is available.

### Compute-pressure relief mode

The first live power-frontier experiment uses one selected A6000. The second
installed GPU is excluded from scheduling, model placement, control work, and
energy accounting. This prevents hidden capacity from turning phone offload
into an artificial multi-GPU comparison.

When the selected A6000 is measured to be compute-bound, the scheduler may form
a `COMPUTE_PRESSURE_BUNDLE` from READY work with sufficient SLO slack. Candidate
actions are complete phone-resident islands, contiguous certified layer routes,
or compatible low-priority decode requests that can form a native phone batch.
The phone result must expose a useful server suffix, remove a batch spoiler, or
remove enough server work to change the selected GPU schedule.

Roofline class selects the batching policy, not the route by itself:

~~~text
memory-bound decode:
  continuously admit/retire sequences at token boundaries
  wait only within SLO slack and launch the largest useful measured batch

compute-bound encode/prefill/island:
  choose the measured saturation point that minimizes SLO-feasible J/work
~~~

The sweet point is conditional on device, backend, island, context/sequence
class, memory headroom, thermal/power state, co-running lanes, and downstream
capacity. It is not a global B32 or B64 constant. The scheduler jointly selects
the compatible batch vector across phone producer, optional phone middle, and
server suffix stages. It reserves later-stage memory, link, activation, and
lane credits before starting upstream work. Queues above a device's selected
point are divided into multiple measured microbatches; queues below it wait only
inside available SLO slack. Batch 1 or 2 is an urgent/fallback exception, not a
target operating point.

Decode uses continuous batching on both phone and server executors. Requests may
enter or leave at token boundaries while retaining a fixed route, layer cut,
KV owner, and mutation epoch for their lifetime. A batch manifest carries the
exact request IDs and sequence positions across every activation boundary.

The selected A6000 may lower its power cap or clock only if a measured operating
point remains SLO-feasible after accounting for phone completion, boundary
transfer, batching delay, and uncertainty. If the lower state does not reduce
energy per completed SLO-valid work, the normal state remains active. Merely
observing high GPU utilization or skipped GPU time earns no energy credit.

The phone and server execute concurrently. Compatible server suffixes become
batch-eligible only after their phone results pass identity, epoch, correctness,
and D2H completion checks. At the latest SLO-safe claim time, unfinished mirrored
phone work loses to the server fallback and is counted as wasted phone work.

## 8. Weight ownership and execution modes

Weight lifecycle:

~~~text
ABSENT -> PREFETCHING -> VERIFIED -> PREPARED
       -> MIRRORED_READY -> EXCLUSIVE_ACTIVE -> DRAINING -> ABSENT
~~~

MIRRORED_READY retains an A6000-compatible copy. A phone may race the latest
server claim; a late result is discarded by epoch. This mode can save compute or
enable a power transition, but receives no HBM-relief credit.

EXCLUSIVE_ACTIVE removes the server HBM copy. It can claim HBM relief, but the
phone route is committed and capacity/thermal leases must cover the request.
Returning ownership requires drain plus a completed reload before the phone copy
is released. Immediate zero-cost server fallback and exclusive HBM relief cannot
be claimed simultaneously.

Dynamic replacement uses two separately accounted residency slots:

~~~text
G0: READY/ACTIVE, with execution and state leases
G1: STAGING -> VERIFIED -> PUBLISHED -> PREPARED -> READY
~~~

G0 remains usable while G1 is transferred and prepared. G1 has its own weight,
prepared-image, slot-generation, memory, and transfer identity and cannot
dispatch before complete atomic publication. Publishing G1 must not invalidate
G0 leases. After activation, G0 enters DRAINING and is reclaimed only after all
compute, result, state, and alias pins release. The current single-generation
phone worker does not implement this; it is an S14 runtime gate.

## 9. Mutable state and KV

State modes:

- STATELESS moves only explicit input/output boundaries.
- STATE_HANDBACK accepts completion only after generated state and output return.
- PINNED_LAYER_ISLAND keeps one contiguous layer range and its KV on one phone
  for the request lifetime.
- CHECKPOINTED/MIRRORED is a recovery mechanism and receives no primary memory
  or energy credit.

State handback includes all bytes:

~~~text
KV_bytes =
  2 * tokens * owned_layers * n_kv_heads * head_dim * element_size
~~~

Future decode tokens cannot execute before sampling. Decode frontier shaping
operates across current-token islands from concurrent streams. A stateful route
has one owner, one mutation in flight, and generation-qualified request, route,
lease, and mutation epochs.

## 10. Two scheduling levels

The slow loop runs over seconds to minutes and decides:

- model, phone, and finite certified contiguous layer/island placement;
- canonical and partial weight residency, replica count, and diverse placement;
- HTP/GPU prepared images and alias ownership;
- mirrored versus exclusive ownership;
- staging-slot generation, stream order, bounded prefetch, minimum hold time,
  drain, and eviction;
- route, boundary, batch, power, and thermal envelopes; and
- bounded speculative residency using visible queue demand plus a fixed
  exploration-byte budget.

The fast loop runs at arrivals and completions and decides:

- which READY unlockers to execute;
- topologically valid order within a bounded horizon;
- phone power-trigger bundles;
- A6000 batch and active/low-cap/idle schedule;
- HTP/GPU phone routes and operating points;
- latest SLO-safe server claims; and
- atomic lane, link, memory, state, and ownership reservations.

It executes the first action of a receding-horizon plan and replans. It never
downloads or invents an uncertified route. A cold or partially staged weight is
not a fast-path candidate; the server takes the current SLO-safe fallback rather
than waiting for prefetch.

## 11. Phone-local execution

Each phone owns bounded HTP and GPU lanes. The local policy chooses only
certified routes:

- HTP for measured dense/attention/FFN islands;
- GPU for measured GPU-efficient prefill or independent service islands;
- HTP and GPU concurrency only for an interference-certified pair;
- same-version native batching where supported; and
- cross-model ragged HMX grouping only after its independent kernel gate.

The phone uses the minimum-energy operating point that still meets its bundle
deadline. Profiles are conditioned on device, backend, island, shape, batch,
co-run, temperature, throttle state, and sustained duty cycle. An unknown or
changed thermal bucket stops new leases or falls back to a conservative route.

## 12. Transport and readiness substrate

S9 provides the initial explicit-command substrate:

- content-addressed staged weights and atomic publish;
- verified durable-prefix resume;
- generation-qualified PREPARE and bounded EXECUTE;
- one resident Gemma dense-FFN path on both phones; and
- bounded windowed provisioning with separate USB contention domains.

S9-V1A-R is the current transport evidence. The full-shard window gate passes:
OP12 2.61x median/1.89x conservative and OP15 2.28x/1.27x. The result is partly
DVFS-sensitive; OP15 reached 95 C. Contract completeness, capacity, and energy
remain unproven.

Bulk weight traffic is lower priority than activation/result traffic. On each
phone, USB P2H result/state traffic preempts USB H2P bulk weights at a bounded
chunk boundary; WiFi H2P commands use a distinct logical path. Completion occurs
only after D2H result/state return. Transfer and backend preparation may overlap
compute only when their measured interference profile passes. The design keeps
all transfer, verification, preparation, host-relay, unused-prefetch, retry, and
eviction costs explicit. The measured 216/262 MiB/s ADB staging controls are not
READY-weight rates; the repaired complete protocol reached about 36-37 MiB/s at
its best tested full-shard windows and OP15 reached 95 C.

## 13. Optimization and oracle

The finite problem contains precedence-constrained scheduling on unrelated
machines, batching, sequence-dependent setup, weight-placement knapsack,
power-state transitions, thermal limits, and state leases. It is NP-hard.

The offline oracle uses bounded integer/rational units and solves:

~~~text
L1 preserve correctness and minimize terminal/SLO failures
L2 maximize SLO-valid completed work
L3 minimize total wall joules
L4 minimize peak HBM and HBM byte-us
L5 minimize makespan, transfer/churn, and wasted phone work
~~~

The S10 foundation has a bounded exact temporal enumerator and structurally
independent reference for its frozen domain. S14-CP2 extends that foundation only
after the mixed trace and measured island catalog exist, adding discrete
placement, replication, residency generations, and transfer timing. The
deployable policy uses bounded lookahead/beam search or MPC; solver choice is not
the novelty.

## 14. Failure and accounting invariants

- Every offered request reaches one terminal outcome.
- Every dispatch binds exact model, island, weight, prepared-image, correctness,
  profile, device, backend, boot, ownership, and state epochs.
- DRAINING rejects new work; reclaim waits for compute, D2H, and pin release.
- A stale completion never mutates state or earns relief.
- Mirrored failure takes the SLO-safe server claim.
- Exclusive failure drains/reloads, replays from a committed boundary, or takes
  an explicit bounded rejection; hidden fallback capacity is not credited.
- Server HBM relief is credited only for bytes absent from HBM.
- Total-system energy relief is credited only from synchronized total-wall
  measurements. GPU-board and server-wall relief remain separately labeled.
- Phone work that finishes too late is reported as wasted work and energy.

## 15. Evaluation and gates

Required baselines:

1. eager server-only;
2. optimized server-only DAG order, lazy batching, and DVFS/power caps;
3. phones with fixed placement but no frontier shaping;
4. frontier shaping without power-trigger credit;
5. full Q-PIM;
6. a perfect-future oracle.

Primary experiments:

~~~text
iso-SLO energy:
  same closed request cohort and SLO
  compare total wall J/completed work

iso-power goodput:
  same total wall-power budget
  compare SLO-valid completed work
~~~

The first controlled experiment uses exactly one A6000 plus OP12 and OP15. It
contains concurrent high-priority compute-bound BGE work and low-priority
memory-bound Gemma decode. The optimized server-only control uses the same one
GPU, arrivals, priorities, SLOs, native batching, and available power/clock
points. Report results separately for normal-state batching, lower-state
batching, phone offload without a GPU-state change, and the complete joint
policy.

A system claim requires at least 10 percent lower total wall energy at equal
work/SLO or 10 percent more SLO-valid work at equal wall power against the best
valid optimized server-only control. Report p50/p95/p99, makespan, timely
offload, batch density, A6000 gap distribution, power transitions, HBM,
activation memory, link bytes, phone waste, thermal behavior, and oracle gap.

Run sustained thermal trials for at least 30 minutes. If instrumentation cannot
distinguish a 10 percent effect, energy remains BLOCKED.

## 16. Evidence, novelty, and stop conditions

Generic DAG scheduling, batching, DVFS, caching, and phone offload are not the
claim. The proposed missing mechanism is their energy-causal coupling:

> Dependency-aware frontier shaping unlocks weight-resident phone work early;
> power-trigger bundles are accepted only when they causally move the A6000 to a
> lower-energy batch/power schedule at the complete wall boundary.

Stop before building a live scheduler when any of these holds:

- the perfect-future S10 oracle cannot beat optimized server-only by 10 percent;
- a causal bounded policy cannot retain most of the oracle opportunity;
- no real power-state or batch-density change explains the predicted gain;
- hidden host/USB/phone energy removes the benefit;
- gains require invalid KV geometry or per-GEMM network splitting;
- steady-state thermal derating removes the effect; or
- the result collapses to one unsupported or non-repeatable model path.

The executable plan is in NEXT_PLAN.md and
spikes/s14_mixed_streaming_scheduler/PLAN.md. Historical S10 and S11 files remain
evidence and are not rewritten as S14 results.
