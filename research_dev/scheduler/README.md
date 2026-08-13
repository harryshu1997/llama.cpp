# Unified research scheduler

`research_dev/scheduler` is the only policy and resource-control package for
heterogeneous inference experiments in this fork. Experiment directories own
measured artifacts, model-specific graph adapters, and physical runners. They
must not contain a second route-selection policy.

## Runtime entry point

Use `UnifiedScheduler` from `research_dev.scheduler` for online decisions. It
owns one `ResourceTimeline` shared by every registered task profile and the
optional matmul virtual queue. Its public runtime operations cover:

- task, layer, and certified operator route selection;
- whole-workload placement selection from an unexpired capacity snapshot and
  exact measured trace evidence;
- per-request cost materialization from exact model identity, current token
  shape, discovered resident executors, and live device-memory capacity;
- causal request-level placement receipts over the fixed route families CPU,
  GPU, phone, GPU+CPU, GPU+phone, and CPU+phone, with an append-only observed
  prefix hash, a hash-bound profile and complete resource-calendar state, and
  no future-request input;
- lifecycle-profile selection, including CUDA cold/open versus reused epochs;
- external reservations for model loading, switching, or another protected
  owner of a physical resource;
- strict atomic or fallback-backed residency transitions with generation,
  memory, hysteresis, reuse, recovery, and fleet-energy gates;
- deadline-bounded GPU fillers that use only already-READY weight slices;
- protected-first phone arbitration across multiple resident HTP sessions;
- event-driven virtual dispatch for healthy-but-busy executors, with active
  leases retained through physical completion;
- background bounded-age executor and phone snapshots for the request path;
- marginal system accounting for CPU interference, causal tails, idle GPU
  energy, and critical-path extension;
- resource readiness changes, cancellation, and actual-completion release;
- live matmul queue admission and placement; and
- one resource snapshot for the complete scheduling session.

Callers load traces and profiles, report runtime state, ask for a decision,
execute the returned route, and release or cancel its leases. A physical
adapter may translate a route into process arguments, but it must not change
the selected device, model placement, split, or resource set.

For GPU model switching, register the post-switch GPU route as the baseline
and the measured CPU plus phone route as an alternative. Reserve the current
GPU residency and switch window on the shared timeline, then use adaptive
mode with `offload_requires_baseline_queue`. The alternative is admitted only
while the GPU route is queued and only when it both meets the measured energy
margin and finishes earlier by `offload_min_finish_saving_us`. Once the GPU is
available, new requests return to the GPU baseline.

Set `finish_before_feature` on an overflow-helper route and provide that
absolute timestamp in each request's features when helper work must drain
before a protected GPU switch. The upper latency bound, including helper
resource queues, must fit inside the window. A missing or exceeded window
fails closed to the baseline route.

## Package layout

- `scheduler.py`: the only online scheduler. `UnifiedScheduler` owns lifecycle
  state, the shared physical resource calendar, task/layer/operator admission,
  and the matmul queue.
- `trace.py`: strict schema detection and ingestion for the two-model BurstGPT
  trace and the six-model mixed trace.
- `__main__.py` and `plan_cli.py`: the public `matmul` and execution-plan
  validation support commands.
- `_internal/`: route policy, matmul, placement, residency, profile, runtime
  gate, runtime cost, causal online placement, event-driven runtime queue,
  phone, capacity, and execution-plan implementation
  modules. They are helpers used by `UnifiedScheduler` or offline compilers,
  not independent online schedulers.
- `tests/`: canonical unit and cross-trace integration tests.

The package-level `research_dev.scheduler` import is the supported API. Trace
and experiment code must not import `_internal` modules. Keeping implementation
modules private preserves reviewable contracts without presenting each planner
as another scheduler.

## Measured hardware data

The hardware pyramid is useful topology context, but peak TFLOP/s and memory
bandwidth are not scheduling costs. Decisions use measured kernel, transfer,
load, lifecycle-tail, and whole-route profiles with explicit applicability and
energy boundaries. Unsupported shapes or missing runtime evidence fail closed.

The matmul materializer includes all measured CPU, CUDA, HTP, and Adreno rows.
HTP and Adreno currently share one logical phone resource and phone-memory
capacity. This is conservative: the scheduler will not assume concurrent use
of the two phone accelerators until a concurrent overlap and power profile is
measured. Pass the task profile's `ResourceProfile` objects to
`materialize_matmul_profile` when its calendar must be shared with task
routes; matching IDs without matching capacity and identity is rejected.

Multiple HTP sessions are residency arenas, not extra compute lanes. HTP0,
HTP1, and later sessions may keep different verified weight slices warm, but
every arm signal names the same shared HTP resource. The request path may arm
an already-warm session after physical graph size is known; it may not load,
repack, or hash weights. Resident-session idle energy must be included in the
same fleet boundary and accounting scope as the baseline before a route can
pass the energy gate.

The shared phone arbiter has two explicit window states. While protected work
is active, filler upper latency must fit below the measured next-protected
lower bound minus its guard. After a timestamped protected-completion receipt,
the old lower bound is invalidated, protected work is rejected, and filler may
use the rest of the current queue-validity interval. An absent, partial, stale,
or future-dated completion receipt fails closed. The resource calendar still
leases the one HTP engine, FunctionFS transport, and desktop USB root once for
each selected call.

The native adapter is `native/phone_arbiter_bridge.cpp`. It gives Qwen priority
over Gemma, switches only between already-WARM HTP sessions, validates model
identity after every arm, retains the GPU-wavefront fence, retries at most one
USB reset, and emits one terminal `PHONEARBITER` receipt. Its current physical
mechanics screen observed a 353.134 ms minimum Qwen phone-idle interval. All
920 Gemma calls completed with a 12.166 ms maximum switch-execute-switch
sandwich under a 50 ms upper bound and 20 ms guard, with zero pending-Qwen,
deadline, reset, or router-accounting violations. This qualifies mechanics
only; matched repeated fleet-energy admission remains separate.

`PhoneOffloadCandidate.execution_mode` separates partial parallel splits from
full operator replacement. `parallel_split` applies the exposed join-wait gate
to the phone and host branches. `full_replacement` has no branch join and is
instead bounded by the candidate's end-to-end latency, deadline, and fleet
energy gates. Composite candidates may arm one model across multiple resident
sessions, while the scheduler leases shared HTP, FunctionFS, and the desktop
USB root once for the complete phone path.

Dynamic weight placements bind their memory pool, consuming execution
resources, and runtime identities. For example, a phone slice names OP15 DRAM,
the shared HTP lease, its HTP session, and its backend identity. Every executor
path acquires the READY placements from `UnifiedScheduler` before dispatch and
releases them after completion. Phone offload and GPU filling do this
automatically. A GPU filler must find all required hashes READY in the same
residency epoch and on the same GPU memory pool. The placement lease counts
prevent the slow loop from evicting weights that may still be read. A residency
transition and a GPU filler cannot begin concurrently.

`ATOMIC_STAGE_BEFORE_EVICT` remains the default residency mode. The separate
`DRAIN_OR_EVICT_BEFORE_STAGE_WITH_READY_FALLBACK` mode is eligible only when
the final placement fits but peak staging does not. Its contract must bind a
READY CPU/phone route to every affected model hash, name retained placement
and execution resources, use the same energy boundary and accounting scope,
and remain valid through the measured recovery upper bound. The shared
timeline leases the fallback and transition resources for that entire bound.
The conservative energy gate charges fallback service and restore energy. A
successful receipt proves the fallback was held until publication; a failed
receipt is accepted only after the exact source residency has been restored.

Every dynamic candidate may also declare transient workspace bytes by memory
resource. These bytes cover pinned source buffers, verification buffers, and
other transition-only allocations that disappear after publication. The
selector checks them against the same live capacity and reserve snapshot used
for resident weights. A missing memory resource fails with
`TRANSITION_MEMORY_RESOURCE_MISSING`; insufficient peak headroom fails with
`TRANSITION_WORKSPACE_MEMORY` before the executor starts the copy.

Raw campaigns stay beside the experiment that produced them. Moving policy to
this package does not duplicate or relabel historical evidence.

The current two-model F16 BurstGPT physical validation is recorded in
[`full_fp16_burstgpt_v1/results`](../spikes/s42_general_energy_scheduler_v1/full_fp16_burstgpt_v1/results/README.md).
On the RTX 4060 Ti 16 GiB host plus OP15 phone, its paired A-B-B-A result
reduced mean fleet energy by 25.54% and makespan by 7.24% while completing the
same 74 requests and 11,605 output tokens.

The legacy experiment calls `UnifiedScheduler.select_runtime_placement` after
capturing live GPU, host, and phone memory. Its schema-v2 receipt binds the
selected layer counts, phone shapes, candidate evidence, snapshot, and
rejections into the physical result. One exact-work retest selected the
qualified Qwen-18/Gemma-25 CPU plus OP15 placement and beat the GPU plus CPU
baseline mean by 21.92% fleet energy and 4.05% makespan. This remains the
historical trace-level binding path.

The `runtime-auto` path no longer calls that trace-level planner. It first
builds a model/device bootstrap from the live memory snapshot, exact model
artifacts, resident phone sessions, and measured operator routes. The
bootstrap CLI has no request-trace argument. During replay, every Qwen request
is placed when it arrives. Gemma requests are retained by the replay adapter
and placed when the Gemma resident endpoint becomes physically available.
Every decision records all six route families, the live snapshot, the prior
prefix hash, the profile and complete pre-decision resource-calendar state,
and the selected physical endpoint. A future trace suffix is not an input to
this API. The resident endpoint's request cost is fitted from one prior
physical repeat and checked on a distinct same-work repeat. Its causal inputs
are the current request shape and the unfinished same-model prefix; the target
includes the llama-server ingress queue. The 128-slot scheduler resource is
an ingress-admission capacity, not 128 physical compute lanes. The profile and
calendar hashes in each receipt are revalidated against the archived profile.
This same-work fit does not establish unseen-trace generalization. The new path
requires a fresh physical qualification;
the historical savings above are evidence for the resident placement, not a
result produced by the new request-level controller.

The focused 84-request BurstGPT plus Llama 1B trace exercises
`UnifiedScheduler.estimate_runtime_costs` during physical execution. The
runner discovers the resident CPU and OP15 executors, captures host/GPU/phone
memory at each eligible arrival, binds the exact GGUF hash, materializes costs
from the current prefill/decode shape, and keeps CPU as the automatic fallback.
In the first matched pair, all ten small-model requests selected OP15 while
Qwen occupied CUDA. Fleet compute energy fell 6.10%, makespan fell 2.11%, and
SLOs rose from 61/84 to 65/84. This is a single resident-task matched pair; it
does not replace repeated qualification or include initial model-load energy.

The 2026-08-13 natural-runtime V3 retest isolates the small-model policy under
the same current controller, runtime profile, `cpu-overflow` large-model arm,
trace, and model hashes. The profile was fitted from disjoint natural physical
repeats and passed 20 held-out route observations with zero upper-bound
violations. Runtime scheduling physically selected the queued OP15 route for
all ten Llama requests; the matched static arm used all four CPU lanes. Fleet
energy fell from 320.429 kJ to 311.752 kJ (-2.708%), makespan fell from
2914.275 s to 2875.218 s (-1.340%), and SLOs improved from 2/84 to 10/84.
Mean 1B completion fell from 60.724 s to 5.760 s. All physical route, live
snapshot, endpoint-log, residency-release, and lease-bound gates passed. The
result is one matched pair, not a repeated confidence-interval energy claim;
its hash-bound artifacts are in the experiment results directory.

The first matched early-adoption screen does not improve that baseline. For
the same seven BurstGPT-derived requests and exact outputs, whole-request
Qwen/Gemma overlap increased mean fleet energy by 15.33% and duration by
24.16% relative to delayed adoption. The unified scheduler consumes this
result as `ENERGY_SCREEN_FAIL_FULL_TRACE_BLOCKED`, admits no such backfill, and
retains the measured static policy. Future dynamic admission requires a
bounded micro-filler with measured shared-CPU/CUDA contention and restore time.

A second matched screen tests shape-aware Gemma FFN suffix widths through the
same public scheduler API. It reduces mean makespan by 1.97%, but increases
mean fleet energy by 0.79%; one of two paired repeats also regresses both
energy and makespan. The narrower phone slices nearly eliminate exposed join
wait while increasing host-branch work and mean CPU package power. The result
is `RETAIN_FIXED_POLICY`, so the latency-balanced table remains a shadow
candidate and the qualified `1:6144,16:6144,512:0` table stays active. See
[`shape_balance_v1`](../spikes/s42_general_energy_scheduler_v1/full_fp16_burstgpt_v1/shape_balance_v1/README.md).

## Required workflow

1. Load a hash-checked profile or materialize a shadow profile.
2. Load either trace schema with `research_dev.scheduler.load_trace`.
3. Create one `UnifiedScheduler` for the whole run.
4. Report lifecycle receipts, runtime snapshots, readiness, and protected
   external reservations.
5. Schedule every controlled request through that same instance.
6. Execute only the returned route or hash-bound execution plan.
7. Release actual completion, or cancel on failure.
8. Bind decisions, releases, observed work, and energy receipts to the result.

Physical adapters under S42 may load artifact-specific metadata and call this
workflow. They must import `research_dev.scheduler`; the deleted S42 scheduler
wrappers are not part of the runtime path. Tests for scheduler behavior belong
in `scheduler/tests` and use the same package API.

## Current enforcement boundaries

- The shared online calendar is effective only when task and matmul profiles
  use the same physical resource IDs.
- Matmul device memory has a live ledger. General route memory and offline
  placement memory are validated by their own contracts but do not yet share
  one online memory ledger.
- Offline placement and residency outputs are not executable until converted
  to a measured or otherwise explicitly qualified route.
- The legacy whole-workload placement API remains available for historical
  experiments. The F16 `runtime-auto` wrapper instead uses the trace-free
  model/device bootstrap plus causal request-level placement. Explicit
  `control` and `op15` arms remain experimental overrides for matched
  comparisons.
- A loaded F16 server currently fixes its FFN callback and GPU layer placement
  at process start. Therefore the large-model receipt exposes all six route
  families but admits only the physically resident endpoint; other families
  fail with `EXECUTOR_ABSENT`. Competitive per-request switching between those
  families requires either multiple resident servers or a server-side runtime
  route-control contract. The scheduler must not claim such a switch until the
  selected endpoint physically executes it.
- The F16 plus Llama 1B `runtime-auto` profile covers all eight measured
  combinations of large-model arm and Qwen, switching, Gemma, or idle phase.
  It combines that phase-conditioned CPU evidence with the independently held
  out whole-phone service model. While a large-model FunctionFS phase is
  active, the controller also leases Adreno because simultaneous HTP and
  Adreno execution has not passed a physical overlap and energy campaign.
  The route becomes available immediately after the observed phase release.
- Deadline and latency gates use the calendar-adjusted upper completion time,
  including virtual-queue delay. If the baseline is already tardy, an
  alternative may be selected only if it does not increase that tardiness.
  This prevents an energy-efficient but deeply queued phone route from
  displacing an earlier CPU fallback.
- Fallback-backed drain/reload contracts are implemented. A disabled physical
  adapter now qualifies pinned-source GPU transfer inside protected OP15
  windows, same-process Gemma ownership, full readback verification, publish,
  and exact post-publication execution. A matched whole-fleet A-B-B-A then
  rejected complete-request overlap because it increased energy and duration.
  Dynamic energy admission remains off until a bounded filler passes the
  repeated conservative screen with fallback acquisition, load, transition,
  restore, synchronized phone energy, and equal work in the same boundary.
- The current six-model live runner sends all 114 requests through one online
  scheduler session. Only Llama 1B has more than one qualified task route.
  The other models receive queue and lease decisions but cannot change device
  until another measured route is added.
- The large-model GPU order is produced by the canonical residency planner.
  Its current qualified transition graph contains only the measured
  Qwen14 -> Qwen8 -> Gemma12 path, so a different order must fail closed until
  its missing load transitions are measured.
- The resident FFN router supports bounded reconnectable USB epochs and
  preserves HTP0, HTP1, and HTP2 weight mappings across desktop clients.
  Two connected desktop clients are now mechanics-qualified behind the one
  protected-first native arbiter. Direct concurrent access remains forbidden,
  and the dual-client route has no fleet-energy admission yet.
- HTP0, HTP1, and HTP2 have independent FastRPC mapping arenas but share one
  physical HTP compute lease. The router serializes execution and rejects
  overlapping, incomplete, or model-mismatched shard sets.
- The three-session OP15 receipt qualifies Gemma M=1 through M=16 and Qwen
  full-FFN replacement at M=1 through M=4. Larger physical M values fail
  closed to local execution until separately measured.
- The measured three-session layout leaves only about 2.21 to 2.38 GiB
  available during the repeated Qwen screen. A 2 GiB live reserve is therefore
  mandatory; another resident slice requires a new memory receipt.
- The Llama 1B HTP3 dense-FFN adapter can physically execute an exact
  GGUF-derived resident slice. Its whole-request route remains absent from an
  enforce-mode profile until repeated natural and held-out physical runs pass
  latency, overlap, direct-energy, and whole-fleet gates. A compiled matmul
  split alone is shadow evidence, not runtime admission.

These boundaries must remain visible in reports. In particular, a shadow
matmul plan or fixed mixed-trace assignment is not evidence that online
task-layer-operator scheduling was executed.

## Tests

Run the canonical scheduler suite from the repository root:

```sh
python3 -m unittest discover -s research_dev/scheduler/tests -p 'test_*.py'
```

Run the scheduler suite followed by every S42 physical-adapter and artifact
test with:

```sh
python3 research_dev/spikes/s42_general_energy_scheduler_v1/run_tests.py
```
