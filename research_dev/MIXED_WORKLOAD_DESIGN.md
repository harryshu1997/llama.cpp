# Q-PIM: dependency-aware power-frontier scheduling

Status: authoritative system design as of 2026-07-15.

Q-PIM uses server-managed phones as active far-memory accelerators. Immutable
weights and backend-prepared images reside beside phone HTP/GPU compute. The
server sends explicit commands and boundary tensors; the phones return explicit
results. This is PIM-style function shipping, not coherent memory mapping.

The new research target is not generic offload or a capacity-only scheduler. It
is:

> Topologically reorder concurrent multi-model DAGs to expose phone-resident
> operator islands early, then select active-memory power-trigger bundles that
> compress the remaining A6000 work into dense batches and measured low-power
> intervals under end-to-end SLOs.

Authority order:

1. this file defines the system and claim boundary;
2. TWO_LEVEL_SCHEDULER.md defines the optimization and online policy;
3. NEXT_PLAN.md defines the executable gate order;
4. spikes/s10_power_frontier_repair/PLAN.md defines the active foundation
   repair and first valid falsification test;
5. versioned schemas define record bytes after their policy is frozen; and
6. spike RESULTS files define measured evidence.

Historical Design A remains route A0 and reusable pipeline/KV/sharding
substrate. S8 provides trace contracts. S9 provides weight-residency,
phone-runtime, and transport substrate. None is the new claim by itself.

## 1. Research question and objective

The server concurrently serves generation, embedding/reranking, vision/audio,
and other tensor-inference DAGs. Q-PIM decides:

1. which topologically valid islands to execute first to unlock later work;
2. which complete islands run on the A6000, OP12, OP15, or later phones;
3. which weights and prepared images reside on each phone;
4. when the A6000 should claim work, form a batch, change power cap, or idle;
5. how each phone should pace HTP/GPU execution under thermal limits; and
6. whether the resulting schedule reduces total wall energy or improves
   SLO-valid goodput under a fixed wall-power budget.

The primary objective is total system energy, not skipped GPU time:

~~~text
minimize integral(
    server wall power
  + host relay power
  + USB/charger power
  + phone wall power
) dt
~~~

subject to correctness, TTFT/TBT/deadlines, bounded memory, links, backend
lanes, state ownership, and thermal constraints. A separate iso-power objective
maximizes SLO-valid completed work under a fixed total wall-power budget.

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

Initial service classes:

- stateful autoregressive generation, including prefill and current-token decode;
- embeddings and reranking;
- vision, OCR, and non-streaming audio encoders;
- background or deadline-flexible inference; and
- multi-stage RAG/agent DAGs composed from those services.

The system does not initially provide:

- coherent server-phone virtual memory;
- arbitrary network micro-operator scheduling;
- arbitrary per-GEMM server/phone row or column splitting;
- future-token execution before autoregressive dependencies exist;
- token-prefix KV ownership across all model layers;
- unsupported-kernel fallback hidden inside a phone route;
- an energy estimate inferred from latency, GPU-us, or utilization; or
- a production scheduler before the S10 opportunity screen passes.

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

- canonical and partial weight residency;
- HTP/GPU prepared images and alias ownership;
- mirrored versus exclusive ownership;
- minimum lease/hold time, drain, eviction, and prefetch;
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
downloads or invents an uncertified route.

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

Bulk weight traffic is lower priority than activation/result traffic. Completion
occurs only after D2H result/state return. The design keeps all transfer,
verification, preparation, and host-relay costs explicit. SHA de-duplication and
protocol v4 remain deferred unless S10 shows transport on the critical path.

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

The active S10-V0-R code has a bounded exact temporal enumerator and a structurally
independent optimum reference. They agree on the frozen intentional-delay and
activation counterexamples and on the generated in-domain corpus. Typed evidence
binding is the next gate; C0-C5 and CP-SAT remain blocked. The deployable policy
uses bounded lookahead/beam search or MPC; solver choice is not the novelty.

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

The executable plan and active repair are in NEXT_PLAN.md and
spikes/s10_power_frontier_repair/PLAN.md.
