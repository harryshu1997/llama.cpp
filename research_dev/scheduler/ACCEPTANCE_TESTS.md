# Unified scheduler acceptance tests

This document defines the tests required before a scheduler phase may be
called complete. Passing the existing regression suite is necessary but is
not sufficient. Each phase must prove its behavior, failure handling, and
ownership boundary.

The baseline at the start of this plan is 259 scheduler tests. New tests must
use synthetic model and device identities unless they validate a specific
physical adapter. Scheduler policy tests must not depend on Qwen, Gemma,
Llama, OP15, or RTX-specific branches.

## System invariants

Every phase must preserve these invariants:

1. `UnifiedScheduler` is the only online policy and resource-control entry
   point.
2. A physical runner reports observations and executes the returned binding.
   It does not choose, filter, replace, or remap a route.
3. Every controlled request receives a scheduler decision record before
   physical dispatch and exactly one terminal record.
4. Every candidate visited by the recorded bounded search is logged with its
   estimated cost or fail-closed reason. Structurally possible placements that
   are outside that request's deterministic search budget are not evaluated.
5. Unknown energy, stale observations, unavailable executors, unqualified
   shapes, and unreserved memory cannot displace the qualified experimental
   control or the explicit recovery fallback, as applicable.
6. Multi-resource changes are atomic. A failed operation leaves the calendar,
   memory ledger, ticket, and decision log unchanged.
7. Request identity, model artifact identity, profile identity, and selected
   executor identity remain bound through queueing, replanning, and fallback.
8. Latency follows the mapped critical path. Energy is accumulated across all
   participating CPU, GPU, phone, memory, and transfer domains.
9. Identical causal inputs produce byte-identical decisions and logs without
   depending on wall-clock time, hash seed, or dictionary iteration order.
10. A new model or device enters through GGUF data, observations, and profiles;
    it must not require a scheduler or runner policy edit.

## Per-request decision log

The scheduler must own an append-only log with schema
`research-scheduler-decision-log-v1`. Runners may persist a scheduler snapshot,
but they must not construct or modify decision records.

Each record contains:

- a monotonic `sequence_index`, `record_sha256`, and
  `previous_record_sha256`;
- scheduler event time in the scheduler's integer time domain;
- event kind: `DECISION`, `ACQUIRED`, `REPLAN`, `FALLBACK`, `COMPLETED`,
  `FAILED`, or `CANCELLED`;
- decision kind: task route, runtime route, operator split, phone offload,
  phone arbitration, residency, GPU backfill, GPU wavefront, or matmul;
- sorted request IDs and the attempt index for request-scoped work;
- immutable request, model, profile, runtime snapshot, and causal-input hashes;
- every candidate with latency, energy, memory, readiness, admission, and
  rejection reason;
- the selected route, executor binding, operator plan, and resource leases;
- the decision reason and resulting lifecycle state.

Shared decisions, such as a residency transition or a cohort, list all affected
request IDs. A decision with no current request records an explicit transition
or work ID.

Required decision-log tests belong in `tests/test_decision_log.py`:

- one decision record is emitted for every scheduling attempt;
- all evaluated candidates and rejection reasons are present;
- selected route, executor, resources, and cost match the returned ticket;
- replan and fallback records reference the previous attempt;
- every request has exactly one terminal record;
- duplicate, missing, or post-terminal records are rejected;
- a failed scheduler transaction appends no record;
- mutation of any record breaks the hash chain;
- serialization is byte-identical across multiple `PYTHONHASHSEED` values;
- concurrent arrivals receive a deterministic total log order;
- the runner can only serialize the scheduler-owned log.

## Phase 0: ownership and architecture

Primary tests: `test_scheduler.py`, `test_architecture.py`, and
`test_decision_log.py`.

- Only `UnifiedScheduler` is exported as a scheduler class.
- Internal imports are acyclic and do not import the public facade.
- Spike runners do not contain route selection, candidate filtering, energy
  comparison, queue policy, quarantine policy, fallback policy, or endpoint
  remapping.
- Raw device probes can remain outside the scheduler; conversion of facts into
  eligibility is scheduler-owned.
- Public contracts have one canonical class or function identity.
- Source-string checks alone are not sufficient. Ownership tests inspect AST
  calls and behavior through a fake physical adapter.

Exit gate: the fake adapter can report raw observations and execute a binding,
but cannot influence the selected route.

## Phase 1: GGUF and route-cost estimation

Primary tests: `test_runtime_cost.py`, `test_operator_energy.py`, and
`test_gguf_cost.py`.

- Physical GGUF quantized tensor bytes are used instead of expanded F16 bytes.
- Dense matmul work uses `2MKN`; attention work uses the correct head, KV-head,
  context, and sliding-window geometry.
- Weight, activation, output, workspace, and KV traffic are counted once.
- Operator time uses launch cost plus the roofline maximum of compute and
  bandwidth time.
- Cross-device edges include latency, payload bandwidth, and transfer energy.
- Parallel branches use critical-path latency while device energy remains
  additive.
- Measured operator-shape profiles override pyramid priors.
- Pyramid-only or extrapolated routes remain shadow-only in enforce mode.
- Queue delay, switching, cold loading, restore, and causal-tail energy are
  included when applicable.
- Stale capacity, missing memory domains, insufficient capacity, or additional
  unreserved memory fail closed.
- Lower and upper bounds remain ordered and use the same whole-fleet energy
  boundary for every compared route.

Exit gate: a hand-computed synthetic DAG matches the estimator exactly, and
every estimate appears in the request's decision record.

## Phase 2: online route scheduling

Primary tests: `test_policy.py`, `test_online_placement.py`,
`test_runtime_placement.py`, and `test_scheduler.py`.

- The desktop-only baseline and every plan visited by the bounded online
  refiner are evaluated at each request arrival.
- The experimental desktop control is distinct from the emergency recovery
  fallback. A large-model desktop control is a frozen, physically qualified
  GPU or GPU+CPU placement and contains a desktop GPU executor. CPU-only may
  remain qualified for recovery, but cannot become the experimental control.
- Desktop-baseline mode excludes phone resources, queues for the frozen
  desktop placement while its GPU is busy, and does not silently substitute
  CPU-only execution. A physical control failure is recorded before a new
  recovery decision is made, outside control-arm accounting.
- The rough placement frontier is cached by artifact, device-capability
  generation, and request-shape bucket. The online budget is deterministic and
  never exceeds 32 evaluated plans.
- In energy-aware mode, every nonbaseline route must be qualified, ready,
  memory-safe, interference-safe, and have known whole-fleet energy bounds.
- In energy-aware mode, every selected nonbaseline route satisfies the
  configured conservative energy-saving margin relative to the qualified
  desktop baseline.
- A phone-assisted route references one qualified unassisted parent. The
  artifact, desktop executor, operator placement, and placement hash are
  identical between parent and child. The assisted route must satisfy the
  conservative energy margin against both that parent and the best qualified
  desktop control; CPU-only recovery evidence cannot authorize it.
- If the desktop baseline meets its deadline, an energy-aware alternative must
  also meet it. If the baseline is tardy, an alternative must finish no later
  than the baseline's conservative finish bound.
- Energy-aware mode retains the desktop baseline when no nonbaseline route
  passes every gate. Baseline tardiness does not bypass safety, energy, or
  latency-regression gates.
- The separately named deadline-first mode minimizes conservative tardiness
  before energy; energy-aware mode never switches objectives implicitly.
- Healthy-but-busy resources are queued by the scheduler rather than reported
  as absent by the runner.
- Route readiness is derived from raw observations, profiles, transitions, and
  the shared resource calendar.
- Selected route and exact executor binding are returned together.
- Quarantine and later decisions use scheduler state, not runner-local state.
- Decisions use only the observed request prefix; future trace rows cannot
  affect the result.
- Synthetic unseen model and device IDs work without source changes.

Exit gate: a fake runner executes only the selected binding for a mixed stream,
and the decision log proves that every visited alternative was recorded. The
synthetic cached-refinement mean is below 10 ms.

## Phase 3: runtime ticket, queue, and resource lifecycle

Primary tests: `test_runtime_controller.py`, `test_runtime_queue.py`, and
`test_capacity.py`.

- Admission creates one ticket binding request, estimates, decision, executor,
  and leases.
- Not-ready executors fail closed unless the scheduler owns a qualified future
  readiness event and transition.
- Multi-resource lease acquisition, extension, release, and cancellation are
  all-or-nothing.
- Injected failure at every resource mutation leaves all earlier resources and
  ticket fields unchanged.
- A renewal failure reaches the execution coordinator immediately.
- `COMPLETED`, `FAILED`, and `CANCELLED` are immutable terminal states.
- Immutable request fields and model artifact identity cannot change on replan
  or fallback. Live observations are supplied separately.
- Physical failure before execution may create one new fallback decision;
  unsafe retry evidence cannot.
- Prediction violations and lease coverage are assessed independently.
- Completion and cancellation release every owned resource exactly once.
- Concurrent requests cannot overcommit the same memory or execution slot.

Exit gate: fault-injection and concurrent lifecycle tests pass without a
partial calendar, ticket, or log update.

## Phase 4: operator splitting and virtual queues

Primary tests: `test_operator_split.py`, `test_matmul.py`, and
`test_operator_energy.py`.

- A split conserves matrix rows or columns, FLOPs, outputs, and tensor identity.
- Candidate cuts respect quantization blocks, device kernels, memory capacity,
  resident weights, and qualified shape buckets.
- Host and helper branches begin from the same dependency and join exactly
  once.
- Critical-path latency includes exposed join wait; energy includes both
  branches and transfers.
- Split fraction is selected from measured or calibrated costs at runtime, not
  from a model-name table.
- The virtual queue includes device compute, USB, workspace, and output-merge
  resources.
- A slower helper, excessive transfer, missing kernel, stale queue, or
  unmeasured cut falls back to local execution.
- The returned operator plan is hash-bound to the physical execution request.
- Multiple requests may overlap only when their resource leases do not
  conflict.
- A decode cohort contains two to four compatible requests. A prospective
  one-member cohort is dissolved before dispatch and its unchanged leases are
  atomically returned to request ownership.
- Cohort compatibility is based on artifact, resident endpoint, desktop
  placement, FFN geometry, transport geometry, and common policy. Exact output
  length and request-specific plan hashes do not partition compatible work.
- Completed members retire without blocking longer members. Shared adaptation
  continues through 4-to-3-to-2 membership changes. At one remaining member,
  shared leases and adaptive control transfer atomically to that request and
  continue on its actual slot without restarting either endpoint.
- One cohort energy boundary is recorded once and normalized by total cohort
  work. Diagnostic energy may update latency only; it cannot select or qualify
  a helper policy and is never copied into individual request receipts.
- Cohort completion aggregation is event-driven and has no fixed service wait.

Exit gate: synthetic CPU+phone and GPU+phone cases select different cuts from
live state, while a no-benefit case selects the unsplit route.

## Phase 5: phone execution and assistance

Primary tests: `test_phone_residency.py`, `test_phone_arbiter.py`,
`test_runtime_cost.py`, and `test_phone_offload.py`.

- A complete phone route is considered when weights, KV cache, and workspace
  fit and the executor is qualified.
- CPU+phone and GPU+phone routes charge host work, phone work, both transfers,
  join wait, and whole-fleet energy.
- Phone compute, memory, USB, FunctionFS/NCM mode, and shared HTP/Adreno
  constraints are represented as scheduler resources.
- Thermal, battery, memory, session generation, transport ownership, and stale
  health observations fail closed.
- Protected phone work wins arbitration; gap filling cannot delay protected
  work or violate its restore guard.
- Independent phone sessions do not imply independent shared HTP compute.
- Phone failure releases its leases, records the failure, quarantines only the
  affected qualified route, and obtains a scheduler-selected fallback.
- If phone assistance increases critical-path time or fleet energy, it is not
  selected.

Exit gate: the runner reports phone facts without a phone-route policy helper,
and every phone use is traceable to a scheduler decision record.

## Phase 6: hot, warm, and cold residency

Primary tests: `test_dynamic_residency.py`, `test_residency.py`,
`test_placement.py`, and `test_profiles.py`.

- Promotion is admitted only when conservative expected reuse amortizes load,
  eviction, fallback, restore, and interference costs.
- Promotion runs asynchronously while requests use a valid warm fallback.
- Demotion respects active leases, memory pressure, hysteresis, and reuse.
- GPU and phone capacity account for weights, KV cache, workspace, reserves,
  and concurrently resident models.
- Transition, publication, rollback, and restore are atomic and generation
  bound.
- A stale or failed transition cannot publish a new resident route.
- Residency decisions and affected requests are recorded in the same journal.

Exit gate: a deterministic hot/warm/cold trace performs the expected promotion
and demotion without holding an arrived request outside the scheduler.

## Phase 7: GPU backfill and overlap

Primary tests: `test_gpu_backfill.py` and `test_gpu_wavefront.py`.

- Only READY resident work may fill a GPU bubble.
- Backfill fits before the protected deadline including restore and guard time.
- Protected GPU work never waits for helper or filler work.
- Stale residency, missing input, insufficient workspace, or unmeasured energy
  rejects the filler.
- CPU and phone work may overlap the GPU only when dependencies and resource
  leases permit it.
- Savings use critical-path extension and whole-fleet energy, not device-only
  power.
- Ordered pipeline receipts prevent duplicate or out-of-order chunks.

Exit gate: adversarial near-boundary timings never delay protected GPU work,
and accepted overlap is energy-positive under upper/lower bounds.

## Phase 8: end-to-end traces and physical adapters

Primary tests remain in the scheduler suite for the fake adapter and in the
S42 harness for physical adapters.

- Every trace arrival enters one `UnifiedScheduler` instance immediately.
- Every request has at least one decision record and exactly one terminal
  record; counts, tokens, and request IDs are conserved.
- No `held_rows`, preselected endpoint, runner-local virtual queue, or
  runner-local fallback chooses execution order or placement.
- The endpoint that executes each request equals the scheduler-selected
  binding.
- Logs are causal and deterministic for replay; physical timestamps and energy
  receipts are attached separately and do not alter decisions.
- A mixed trace exercises CPU, GPU, phone, and at least one qualified composite
  route when those routes are physically available.
- Unknown or unavailable routes visibly fail closed instead of disappearing
  from the candidate log.
- A semantic-quality campaign preserves raw output and records exact hashes as
  diagnostics. Its hard output gate rejects malformed or clearly degenerate
  text without claiming exact-token equivalence.

Exit gate: the complete fake trace passes first. A physical run may then make a
savings claim only from a matched same-work control/treatment comparison with
the same accounting boundary, SLO rule, model hashes, and trace. The historical
25.537 percent result is a comparison baseline, not a unit-test constant.

## Phase 9: remote-resident FFN weights

Invariants (all fail closed):

- The desktop loader never allocates, loads or prefetches a masked dense FFN
  weight; the omitted GGUF ranges are released page-exactly and the loader's
  proof line reports omitted, unmapped and mapped bytes.
- A context whose model omits weights cannot be created without an FFN eval
  callback owner; graph builders route the masked layers to the phone at full
  width regardless of the adaptive split policy.
- The FFN client requires complete shards (offset 0, columns == n_ff),
  dispatches remote layers for every batch including prefill, rejects runtime
  policies that target them and aborts instead of emitting zeros when the
  owner is unreachable; the server rejects such controls too.
- `RuntimeRemoteResidentFfn` covers its layer mask exactly with complete
  gate/up/down groups, one dtype, and sessions with disjoint layers;
  generations bind execution but not the placement identity.
- Route generation rejects a declaration that disagrees with the manifest or
  the phone session capabilities, and rejects candidates while any owner is
  not READY with the declared geometry and a generation of at least one.
- A physical command carries bound generations, a phone layout generation, a
  resident layer mask covering the remote layers, no overlapping assisted-copy
  helper and no transition that evicts an owner; the launch requires a matching
  validated omission proof, and reduced and full parents never share a server.
- Accounting credits reclaimed bytes only after the proof matches the plan;
  recovery feasibility is stated per pool; reclaimed memory is split between
  fallback reserve and KV gain, never counted twice.

Tests: `test_remote_resident_contract.py`, `test_remote_resident_routes.py`,
`test_remote_resident_launch.py`, `test_remote_resident_accounting.py`,
`test_remote_resident_native.py` (needs the CPU build). Physical gates A-D:
`campaigns/burstgpt/remote_resident_gate.py`; its record identifies the runtime
image by the launcher digest and by `runtime_libraries_sha256`, the digests of
every shared library the launcher loads from the build tree (the code lives in
`libllama.so`, not in the 18 KB `llama-server` executable;
`test_desktop_parent_calibration.py`).

## Phase completion procedure

An agent may report a phase complete only when:

1. all new phase tests and every earlier phase test pass;
2. required negative and fault-injection tests are present;
3. no test relies only on finding or removing a symbol name;
4. scheduler decisions and terminal outcomes have complete journal coverage;
5. the full S42 harness passes without changing expected negative fixtures;
6. the diff passes ASCII, compile, import-cycle, and public-identity checks;
7. remaining policy outside `research_dev/scheduler` is listed explicitly; and
8. an independent review confirms the implementation, not only the test count.

If any required item is deferred, the phase status is `PARTIAL`.
