# S10-V0-R repair plan

Status: TEMPORAL_FOUNDATION_PASS, then
TYPED_EVIDENCE_INTEGRITY_PASS_PHYSICAL_CLAIMS_BLOCKED.

The two exactness defects (transition-aware and activation-constrained intentional
delay) are closed and every optimum claim is independently verified. The typed
evidence contract, deterministic binder, and fail-closed validator are now in
place and adversarially tested (EVIDENCE_CONTRACT.md, evidence/). The live path
runs the checked-in schemas, hashes actual artifact files, binds exact token/KV
shapes, pins each route to one boundary record, validates merged-batch boundary
geometry, and refuses aggregate SERVER_WALL/TOTAL_WALL inputs. These are repairs
to the worker's first green E1 report; the
v2 temporal solver and checker remain unchanged.

The measured atlas is EMPTY: no route, power, or boundary row on disk meets the
frozen gate, and phone energy is physically unmeasurable (EVIDENCE_MATRIX.md).
Moreover, E1 now rejects every `MEASURED` instance because its additive solver
cannot represent aggregate wall timelines. The only valid bundle is fully
synthetic and `MECHANICS_ONLY`, which authorizes no physical claim.

C0-C5, PF1, and every energy verdict remain unauthorized. No system verdict is
authorized.

This directory replaces the invalid S10-V0 scheduling model without rewriting
its historical evidence. The old `../s10_power_frontier/` directory remains the
record of the invalid screen.

## Scope

This repair first establishes trustworthy offline mechanics:

1. a strict small general-DAG instance contract;
2. exact enumeration of route choices, arbitrary compatible batch partitions,
   and per-device action orders;
3. timeline energy that distinguishes schedules with equal busy time but
   different P0/P8 gaps;
4. a standalone checker that rederives feasibility and accounting;
5. a causal snapshot helper that cannot observe unarrived requests; and
6. regression tests for every load-bearing V0 review finding.

It does not yet implement the complete C0-C5 experiment, real-trace replay,
new measurements, or a physical energy result.

## Protected scope

- Do not edit llama-server, model graphs, KV internals, backend scheduling, or
  kernels.
- Do not change the phone-PIM protocol.
- Do not commit or push.
- Keep the historical V0 files and artifacts unchanged.
- Use ASCII and deterministic checked integer units.

## Foundation gate

Status of each condition (see RESULTS.md for evidence):

- [x] the partial-partition counterexample has zero misses;
- [x] the exact solver agrees with separate slow references on generated
      zero-transition, zero-activation cases, including compatible batches;
- [x] transition-aware and activation-constrained temporal placement agrees with
      a structurally independent reference on generated cases (1187 compared, 0
      mismatches, across four processes and multiple PYTHONHASHSEED values);
- [x] a standalone verifier rejects feasible but suboptimal certificates instead
      of trusting solver-reported search counters;
- [x] causal decisions are prefix invariant;
- [x] all signed-field, HBM, activation-lifetime, and transition mutations fail;
- [x] two equal-busy schedules with different gaps have different energy;
- [x] schemas and semantic validators reject unknown or inconsistent fields; and
- [x] typed evidence binds every route and power input to eligible artifacts.
      The CONTRACT and BINDING pass: every evidence-derived field is enumerated
      from the instance, must equal a PASS record pinned by digest, and anything
      missing, stale, mismatched, UNKNOWN, or ineligible fails closed
      (EVIDENCE_CONTRACT.md, evidence/validator.py, evidence/binder.py).
      The measured ATLAS is empty (EVIDENCE_MATRIX.md): zero eligible rows, and
      phone energy is UNKNOWN and physically unmeasurable. In addition, E1's
      additive model refuses all physical wall-energy inputs. C0-C5 remain
      blocked for demonstrated contract and evidence reasons.

## Exact temporal domain (frozen contract)

All arithmetic is integer: time in us, power in mW, energy in nJ (= mW * us). The
solver has exactly two exact modes, chosen by the instance and never by
convenience. Anything outside the declared domain raises; it never emits an
approximate result and never sets `complete=true`.

### EARLIEST mode

Selected only when `wake_us == idle_entry_us == transition_nj == 0` and every
`output_bytes == 0`. The solver enumerates route assignments, every compatible
server batch partition, and every per-device action order, then constructs the
earliest schedule for each order.

This is exact, and the argument is frozen here:

- With zero wake/idle-entry the server P0 window of an action is exactly the
  action. One lane per device forbids overlap, so the merged windows of a fixed
  order always total `sum(server durations)` regardless of where the actions sit,
  and the merged-window count is multiplied by `transition_nj == 0`. Server energy
  is therefore identical for every start-time placement of a given order.
- Phone energy is `active_mw * duration + extra_energy_nj`; it never depends on
  start times.
- With every `output_bytes == 0` the activation peak is 0 for every placement.
- The remaining objective terms (misses, lateness, -met) are non-decreasing in
  every finish time, and earliest-start simultaneously minimises every finish time
  of a fixed order.

So for each order the earliest schedule is lexicographically optimal, and taking
the best over all orders/partitions/routes is exact.

### TEMPORAL mode

Selected when any of `wake_us`, `idle_entry_us`, `transition_nj` is nonzero, or any
`output_bytes` is nonzero. Earliest-start is provably NOT exact there: intentional
delay can merge two P0 windows (frozen counterexample:
`fixtures/transition_delay_counterexample.json`, 162000000 -> 147250000 nJ at equal
zero-miss/zero-lateness outcomes) or move an activation lifetime out of another's
way (`fixtures/activation_delay_counterexample.json`, earliest peak 200 > bound 150
while a delayed placement reaches 100).

The solver then enumerates, in addition to routes, compatible batch partitions and
per-device action orders, **every legal integer start time** of every action.

Enumerated dimensions: release times, DAG precedence, one lane per device and every
per-device action order, intentional delay (all legal idle before and between
actions), route assignment, compatible batch partitions, P8/P0 wake, idle-entry,
merged active windows and transition energy, conservative activation allocation
from producer action start through the last direct consumer action finish
(including terminal outputs), deadlines and the horizon.

Feasibility windows come from releases, DAG and lane precedence, the wake ramp, and
the **horizon only**. Deadlines are deliberately excluded from the bounds because
TARDY is a legal terminal outcome; a deadline-derived bound would silently discard
feasible schedules.

### Declared state bound

- `TEMPORAL_MAX_NODES = 6`, `TEMPORAL_MAX_ACTIONS = 6`.
- `TEMPORAL_MAX_WINDOW_PRODUCT = 8000000`: an instance is inside the domain only if
  the total UNPRUNED start-time product over every layout and device order is at
  most this bound. The cumulative product is checked immediately before recursing
  into each device order. An out-of-domain instance may enumerate earlier layouts,
  but it raises before searching the order that crosses the bound and emits no
  certificate.
- `max_states` (default 2000000) caps entered `(layout, device-order)` states plus
  complete temporal leaves. It is an internal fail-closed work limit, not a signed
  proof counter. Exhausting it raises and never yields a best-so-far claim.

### Pruning rules (branch and bound)

Both rules are lexicographic LOWER bounds, so they can never remove an optimum.
`solve(..., prune=False)` disables them and is used by a differential test that
asserts pruned and unpruned searches agree.

- R1 lateness/miss bound: placed terminal actions contribute their exact lateness;
  unplaced terminals contribute the lateness of their earliest still-legal finish.
  Delay never reduces lateness, so this bounds the objective below.
- R2 energy bound: `p8*horizon + (p0-p8)*sum(server durations) + transition_nj *
  (1 if any server action else 0)` plus the layout's fixed phone energy. Merged P0
  windows always cover at least the actions themselves and number at least one.

Ties are broken by the canonical bytes of the schedule, giving a deterministic total
order for equal objectives. Pruning is applied only when the bound is strictly worse
than the incumbent, so the tie-break is preserved.

### Independent optimum verification

`checker/reference.py` is a checker-owned, structurally independent exhaustive
recomputation with its own declared tiny bounds. It shares no oracle candidate,
pruning, partition, or objective code and performs no branch-and-bound. Default
checker mode accepts only an objective equal to the reference optimum, rejects a
feasible-but-suboptimal certificate, and fails closed when the instance is outside
the reference domain. The signed certificate contains only `search.complete`; the
checker treats that marker as an assertion and independently proves the optimum.

## Deliberate limits

- Foundation fixtures must contain at most eight nodes unless an explicit
  enumeration bound is raised.
- Each device has one lane. Shared-link contention is deferred until a measured
  link profile is added.
- All nodes must complete by the horizon. Rejection and admission are part of
  the later policy layer, not silently simulated here.
- Phone energy is an explicitly labeled profile input. It is not physical wall
  energy.
- Power-cap actions remain blocked until measured with valid permissions.
- The standalone checker rederives feasibility and accounting. Its default
  exact-certificate mode additionally proves optimality against the independent
  reference, and fails closed outside that reference's declared tiny domain.

Passing this gate authorizes only the next typed-evidence binding checkpoint. It
does not authorize C0-C5, measurements, PF1, or production integration.

The solver no longer refuses nonzero `wake_us`, `idle_entry_us`, `transition_nj`,
or nonzero `output_bytes`. Those instances take TEMPORAL mode and enumerate every
legal integer start time, because intentional delay can merge P0 windows without
changing SLO outcomes and can move an activation lifetime out of another one's
way. Both frozen counterexamples under `fixtures/` are now solved rather than
refused, and a structurally independent reference confirms each optimum.

What remains fail closed:

- instances outside the declared temporal domain (`TEMPORAL_MAX_NODES`,
  `TEMPORAL_MAX_ACTIONS`, `TEMPORAL_MAX_WINDOW_PRODUCT`) and any search that
  exhausts `max_states`: both raise and never emit `complete=true`; and
- default-mode optimality for any instance outside the independent reference's
  declared domain (for example `fixtures/partial_partition.json`, whose horizon
  needs about 19.6M start-time combinations against a 8M bound). The checker says
  so explicitly and certifies nothing; it never falls back to trusting the
  certificate's own completeness marker.

The causal helper is a prefix snapshot test. It does not track completed or
in-flight actions and is not a rolling C5 policy.
