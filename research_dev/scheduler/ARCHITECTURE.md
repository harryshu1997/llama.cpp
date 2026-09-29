# Scheduler architecture and code map

The public entry point is `UnifiedScheduler` in [scheduler.py](scheduler.py),
imported through `research_dev.scheduler`. All reusable scheduling policy and
physical control stays under this package. Campaigns supply inputs and persist
results; they do not select routes.

## Planning model: three decisions, three time scales

| Decision | Scope | Existing mechanism |
| --- | --- | --- |
| Residency | Place weights and KV state before use; account for release and restoration at phase boundaries | Placement/session controllers, decode-only relocation, dormant host share, memory ledger |
| Execution plan | Choose a supported split for the measured request shape and objective | Shape profiles, decode-split atlas selector, adaptive controller |
| Dispatch | Run dependency-ready work against current leases and device availability | Runtime controller, rig coordinator, native GGML execution |

These are responsibilities of the existing system, not three new controllers.
Residency changes pay transfer, verification and restoration costs. Execution
changes must stay within the verified resident geometry. Dispatch cannot create
overlap between dependent operations merely by assigning them to different
devices. Profile and exploit independent-request overlap separately from
intra-request splitting; credit concurrency only where the physical path
demonstrates it. HTP sessions add residency capacity, not independent HTP compute.

A split can change without reloading only if every participating device already
owns its newly assigned weight slices. Replication buys flexibility but consumes
memory; exclusive ownership saves memory but needs explicit restoration before
work returns to the desktop. Decode-phase release never funds prefill admission.

### Operator scope and evidence limits

The planner can represent work across CPU, GPU and memory tiers, but USB phone
operator offload remains **FFN-only** until a full-path profile justifies another
family. This is an execution-policy boundary, not removal of existing experimental
code, capability records or separately qualified whole-model endpoints.

- [S5's operator screen](../spikes/s5_operator_affinity/RESULTS.md) found that
  isolated phone operator timings and USB costs did not justify general
  whole-operator offload against its GPU reference. Tiny operators and projections
  are not a new phone-kernel roadmap.
- [Split-KV attention](campaigns/burstgpt/reports/20260918-split-kv-attention/README.md)
  is a CPU/GPU placement fallback: the measured split was about 3% slower in decode
  than all-device KV, but cheaper than whole-host KV. It is not evidence for
  profitable phone attention, nor for overlap of the partial attention branches.
- The [MoE screen](campaigns/burstgpt/reports/20260919-moe-edge-energy/README.md)
  measured host/paging/VRAM arms and estimated a phone tier; it did not execute
  phone experts. Its bounds do not justify building a phone-expert executor for
  this rig while the measured RAM/VRAM alternatives are available.
- The [Qwen decode atlas](campaigns/burstgpt/reports/20260917-decode-relocation-kv-headroom/README.md)
  has different measured optima for latency (50%) and energy/memory (100%).
  These are exact-environment operating points, not universal fractions.

Existing contracts are narrower than arbitrary operator partitioning:
`RuntimeOperatorAssignment` carries one split axis and one fraction per operator;
`balance_parallel_split` balances measured FFN column splits. Arbitrary geometry
would require new contract surface, a native executor and correctness validation.
The next milestone below changes none of these contracts.

### Cost the path, including transfers and residency

Profile fused groups and their boundaries, not just isolated kernels. A device
with the fastest kernel need not yield the fastest or cheapest request.
For demonstrated parallel branches, completion is dispatch overhead plus the
slowest branch (queue wait, transfers and compute) plus merge overhead. Serialize
dependent stages according to the actual execution path; do not replace their
sum with an assumed maximum.

Account for repeated weight copies, activation/KV copies, synchronization,
graph/launch overhead, page faults and release/restore separately. Charge shared
idle energy once over the measured interval, not once per concurrent operator.
Keep measured host energy separate from assumed phone power. A smaller desktop
RSS alone does not prove that page-cache memory was freed or that restoration
will be cheap. Keeping clean backing pages reclaimable is a candidate to measure,
not permission to ignore them in capacity accounting.

## Next milestone: transfer inventory, before contract expansion

Start at `ggml_backend_sched_compute_splits` in
[ggml-backend.cpp](../../ggml/src/ggml-backend.cpp), reusing the existing graph split
and copy information. This is a measurement milestone, not new placement policy
or a new phone operator.

1. Capture one representative prefill ubatch and one decode step. Record request/
   graph invocation, phase, token count, split id, backend/device, operator group,
   tensor shape/dtype and source/destination residency.
2. Record actual copy calls and bytes in/out, classified as weights, activations,
   KV or output. Count each transfer once; for used-expert copies, record the
   selected ranges rather than the full expert bank. Track repeated copies of
   the same weights across ubatches.
3. Separate host enqueue/blocking time, device completion time and synchronization
   waits. Async submission duration is not CUDA compute duration. Reuse normal
   completion boundaries or device timing evidence; do not insert a synchronization
   at every split and then treat that perturbed schedule as the normal runtime.
   Preserve CUDA graph mode and measure instrumentation overhead.
4. Aggregate the real ubatch sequence, including the remainder and changing
   attention context, to predict full prompt prefill. Keep model loading, warmup,
   first-run graph capture and restore intervals explicit and consistently bounded.

Acceptance: on the same dense 14B model, placement, prompt, context, binaries,
threads, graph mode and declared cache policy, predict measured prefill duration
at both `ubatch=128` and `ubatch=512` with
`abs(predicted - measured) / measured <= 0.10` for each. Freeze the component
calibration and prediction before reading held-out request totals; fitting each
total to itself is not validation. Report transfer-byte totals, timing components,
prediction error, repetitions/variation and profiler overhead. The first one-step
inventory is diagnostic; full-request aggregation is required for acceptance.

After that gate, rank the already identified host-side opportunities:

1. Reduce repeated host-weight streaming during prefill; measure larger ubatches
   subject to workspace and live memory admission. Do not assume a fixed speedup.
2. Sweep decode thread count/affinity and compare phone assistance against the
   tuned desktop parent, retaining the original baseline artifacts.
3. Measure release/restore with reclaimable page-cache backing under both slack
   and real memory pressure, including subsequent prompts and restoration costs.

No new phone attention, projection or expert kernel is part of this milestone.

## Reading the execution path

1. [campaigns/burstgpt/runner.py](campaigns/burstgpt/runner.py) loads a campaign
   and submits arrivals through the canonical arrival coordinator. Argument
   parsing and trace loading are in `arguments.py` and `trace_inputs.py`.
2. `_unified/automated_requests.py` exposes submission, replan, and completion.
   `automated_requests_ops/` contains the corresponding transactions.
3. `automated_candidates_ops/` materializes candidates and applies evidence.
   `automated_selection_ops/` selects a route, including desktop and helper
   alternatives. The underlying GGUF compiler remains in
   `_internal/route_generation/`.
4. `_internal/runtime_controller.py` owns request tickets and the dispatch
   queue. `runtime_controller_ops/` performs admission, cohort changes,
   reservation compaction, lease renewal, completion, and recovery on that
   same controller.
5. [adapters/runtime.py](adapters/runtime.py) executes the ticket through the
   HTTP backend and physical rig. `adapters/phone_session.py` owns phone
   process/session state; `phone_session_ops/` handles transport, weight
   sources, launch, replacement, verification, and close.

## Stable owners, smaller implementation files

| Concern | State/API owner | Implementation |
| --- | --- | --- |
| Scheduler transactions and registration | `scheduler.py` | Existing `_unified/` mixins |
| Model placement epochs | `UnifiedScheduler` | `_unified/placement_epochs_ops/` |
| Phone layout selection and offline preload | `UnifiedScheduler` | `_unified/phone_residency_ops/` |
| Helper identity, reuse, and attachment | `UnifiedScheduler` | `_unified/helper_envelopes_ops/` |
| Helper preparation and recovery | `UnifiedScheduler` | `_unified/helper_preparation_ops/` |
| Request tickets and dispatch | `_internal/runtime_controller.py` | `_internal/runtime_controller_ops/` |
| Layout generations and per-session transitions | `_internal/model_placement_controller.py` | `_internal/model_placement_ops/` |
| Decode probes, evidence, and sustained assistance | `_internal/adaptive_decode.py` | `_internal/adaptive_decode_ops/` |
| Phone processes, loaded weights, and proofs | `adapters/phone_session.py` | `adapters/phone_session_ops/` |

An `*_ops` function receives the existing owner explicitly as `controller`.
It must not construct a replacement controller or keep a parallel copy of its
state. Pure helpers receive only their inputs. A `controller_type` or
`_controller_class` parameter preserves the original class-method dispatch.

Facade methods keep their signatures, decorators, and locking boundaries.
Constructors, checkpoints, and authoritative state remain with their original
owners. This split introduces no additional scheduler, queue, memory ledger,
layout controller, or adaptive state machine.

Useful implementation files for current development:

- `adaptive_decode_ops/budgeting.py`, `sequencing.py`, and `promotion.py`:
  probe/verification budgets, retry continuity, and winner selection.
- `adaptive_decode_ops/helpers.py` and `windows.py`: helper changes, control
  acknowledgements, and measurement-window accounting.
- `helper_envelopes_ops/templates.py`, `ready_plan.py`, and
  `late_attachment.py`: reusable READY identities and request attachment.
- `helper_preparation_ops/authorization.py`, `completion.py`, and
  `recovery.py`: preparation authorization, publication, and rollback.
- `model_placement_ops/sessions.py`, `transitions.py`, and `rebind.py`:
  generation-keyed session state and progressive replacement.
- `phone_session_ops/replacement.py` and `completion.py`: physical
  reconfiguration, historical proof validation, and terminal cleanup.

The paths above are relative to the matching `*_ops/` directory shown in the
table, not separate public APIs.

## Shared phone memory cap

`UnifiedScheduler.set_phone_htp_memory_cap(phone_device_id, cap_bytes,
workspace_bytes=..., observed_at_us=...)` sets an aggregate HTP residency
ceiling after capability registration. The ceiling includes HTP weights and
the declared HTP runtime workspace. `None` removes the explicit cap. Updates
are transactional and idempotent; fixed-residency experiments reject changes.
Existing runs do not enable resident-shard resizing unless a cap is configured.

Campaigns can pass this same API through `phone_htp_memory_caps`, an array of
`{phone_device_id, cap_bytes, workspace_bytes}` records. The launch contract
serializes it as `--phone-htp-memory-caps-json`; the runner only forwards the
validated configuration to the scheduler. Duplicate devices and combination
with fixed residency are rejected. It does not choose sessions or fractions.
The capped static bound retains the larger catalog/live reserve even when
Android reports more RAM than the declared phone pool.

Whole-model endpoints may declare `whole_model_peak_memory_bytes` in their
existing adapter parameters. This must cover weights, configured KV/concurrency,
base workspace, and the measured OpenCL prepack high-water allocation. It is
not just the model file size. This field is configuration, not qualification:
existing route, energy, health, transport, and physical identity checks remain
required. Capped co-residency fails closed without the complete peak declaration.
An unqualified cold endpoint can pass memory admission only when both actual
retained allocations plus its complete peak and fresh live free memory fit.
Energy qualification is separate: missing energy evidence is not fabricated
and no proposed shrink receives memory credit.
Whole-model admission with HTP sessions enforces the declared shared pool even
without an explicit resizing cap. It includes READY HTP bytes/workspace, the
whole-service peak and the larger of the catalog/runtime reserve. Android's
larger physical MemTotal is not permission to increase the declared budget.
Unreconciled physical HTP residency defers admission instead of receiving an
assumed zero workspace. Configuring a cap enables resizing; merely proposing
a smaller layout never counts as released memory.

The existing residency planner computes the HTP weight budget as:

```text
min(configured HTP cap, phone pool after reserve - whole-service peak)
    - HTP workspace
```

Per-session limits and live free-memory observations can reduce it further.
Peak reservations include already-resident persistent services; the live-memory
calculation credits only their observed footprint, avoiding duplicate reservation.
Verified READY HTP workspace is already included in live occupied memory, so
the live-memory bound reserves only additional workspace growth. The static
pool/cap bound still subtracts the full workspace. Proposed workspace receives
no resident credit. Cached selection, replan, and epoch rematerialization pass
their current snapshot and observation time through the same admission check.
The same whole-service peak is a floor on the existing runtime memory demands
and is checked again against live physical memory before loading.
For a persistent whole-phone endpoint, the non-weight portion of that peak is
an endpoint-owned resident demand, not another per-request allocation. Hot
reuse credits only the matching observed footprint beyond weights, with an
exact executor, positive residency generation, full tensor coverage and a
READY healthy endpoint. Launch identity excludes only the enumerated generated
request-transport and six phone-power accounting fields. The power annotations
remain in route costs and execution records. All launch and memory configuration,
including unknown options, must still match exactly. This normalization does
not change request transport admission or physical execution proof checks.
Missing or partial footprint observations leave the unobserved bytes reserved;
request growth beyond the declared peak remains request-scoped. Exact-template
reuse also tracks the observed whole-service launch and tensor identities.

For a managed Android whole-model endpoint, the existing background monitor
samples `dumpsys meminfo` total PSS. It binds the observation to the verified
launch artifact/configuration, endpoint generation, PID, boot ID and process
start ticks, checking the process identity before and after the probe. The
snapshot exports this identity, sample start/end times, byte count, raw-output
hash, age and validity in `telemetry_observations[phone].resident_allocations`.
Snapshot construction never waits for this probe. Missing, malformed, stale,
timed-out or mismatched samples receive no runtime-memory credit and request
background refresh. Credited snapshots expire with their supporting sample.
The sampled PSS is a resident process footprint, not a measured allocation
peak or proof of all configured KV/prepack capacity. Unobserved peak bytes
remain reserved; no configured peak is substituted for measurement.
Android's [dumpsys documentation](https://developer.android.com/tools/dumpsys)
describes the PSS metric used here. Physical peak qualification remains required.

For bounded memory calibration, the Android launcher also exposes a separate
`probe_process_memory_peak` observation over the same control connection. It
checks process identity on both sides of `/proc/PID/status` and per-process
KGSL counters. CPU RSS and GPU allocation high-water values are summed without
subtracting possible overlap. Swapped or imported allocations mark accounting
incomplete, and missing/malformed counters fail closed. These historical peaks
are diagnostic, not current residency credit or automatic qualification for
untested request shapes. The configured peak is never silently lowered.

Whole-model Android execution uses the existing generic execution contract, not
an HTP helper contract. Physical admission and activity accounting resolve its
phone device from the exact Android executor, endpoint and launch parameters.
Live memory, battery and thermal checks run before load mutation, and both
preparation and whole-model execution contribute phone-active intervals even
when there are no HTP calls. This does not bind whole-model tickets to HTP
session generations or modify the native terminal-proof format.

When HTP sessions are present, the whole-model endpoint has a separate
`residency:whole:<executor>` ownership resource. HTP ownership stays with the
session compute group. This does not remove shared compute/transport leases or
create another RAM pool. Evictions resolve the observed executor's ownership;
queued residency projection preserves other owners on that device and charges
the whole-model allocation without crediting retained HTP bytes. Even an
unqualified but physically resident whole-model service reserves its declared
peak. Qualification controls selection, not whether allocated memory exists.

Exact automated route/transition evidence recovery is independent of adaptive
FFN contracts. Eligible saved rows pass the existing artifact, launch, resource,
placement and transition identity checks before reuse, and recovered costs are
materialized before selection. Empty stores, diagnostic energy receipts and
changed transport do not become qualified evidence through this recovery path.

Whole-model control defaults to `adb-usb`. A rig may explicitly configure
`whole_control_transport=adb-ncm` and `whole_ncm_adb_endpoint=IP:PORT` on the
existing FunctionFS NCM interface. The equivalent runner arguments are
`--phone-whole-control-transport adb-ncm --phone-whole-ncm-adb-endpoint IP:PORT`.
Before the first FFN load, the existing Android launcher stages and verifies a
bounded bootstrap script. It requires authenticated ADB and an already configured
TCP port. After Android init stops adbd during the FunctionFS changeover, it
starts that TCP service without changing USB modes, authentication settings or
HTP workers. The network connection must match the bootstrap phone serial and
boot ID. Process control, allocation probes, native logs and inference forwarding
all use this one explicit connection; they never silently fall back to USB ADB.
The normal health/power fallback uses the verified connection when HTTP telemetry
is missing. Missing or expired observations still prevent admission.

NCM mode, endpoint and bootstrap-script hash are launch-contract identity fields.
Its request links have a distinct transport generation and estimated costs;
they do not inherit USB-ADB measurement qualification. Whole-phone consumers stop
before their FunctionFS/NCM transport. A failed remote stop retains the physical
endpoint record and connection for recovery. Stop checks the original process
identity and confirms exit; it does not signal a different PID lifetime.

The [bounded NCM gate](campaigns/burstgpt/reports/20260910-whole-phone-coexistence/README.md)
proved a complete whole-phone request, fresh allocation samples and clean shutdown
with one unchanged READY FFN shard. It did not exercise simultaneous HTP calls.
The current whole-model capability still reserves the shared HTP resource, and
PSS does not prove the full OpenCL allocation/prepack peak. Concurrent compute,
independent NCM lifetime without an FFN transport owner, and peak qualification
remain separate work. No new HTTP command service or wire format was introduced.

When the current layout exceeds the new cap, the existing mixed generator can
produce smaller nonempty layer subsets of the resident shards. Artifact, session,
column width, and stored-shard coverage remain fixed. Selection uses existing
benefit/cost estimates and stages the result through the existing one-session
replacement transaction. It does not evict an entire session merely to reserve
one smaller whole-model service. Runtime fraction changes alone release no RAM.
Unsupported partial-width byte accounting or an infeasible nonempty-session
target defers rather than inventing capacity.

When capacity returns, the same opt-in resizing path offers one-session growth
for arrived demand. It keeps the existing layers, column width and all other
session shards, adding only uncovered CPU-parent FFNs present in that session's
registered storage coverage. Growth must pass the normal transition economics,
confirmation and residency checks; it is not forced capacity maintenance.
Fresh observations reconsider an increased budget without requiring a new
arrival, coalescing repeated evaluations of the same budget and READY geometry.

Only verified READY residency counts as released memory. A proposed smaller
layout cannot authorize the whole-model load. Cap events log the budget,
workspace, persistent peaks, and deferral reason without changing retained
session generations or leases. Capacity-driven maintenance bypasses the
profitability veto, but not draining, acknowledgement, physical validation,
or rollback.

Current integration limit: online preparation still needs a live or arriving
request context for the artifact being resized. An idle-only artifact therefore
logs `PHONE_RESIDENCY_MEMORY_CAP_CONTEXT_UNAVAILABLE` and defers without publishing
an unexecutable target. Completing model-owned idle preparation is separate work.
This memory policy does not establish concurrent Adreno and FunctionFS operation;
that transport path and the peak memory declaration still need physical validation.

## Contracts and compatibility

| Existing import module | Canonical definitions |
| --- | --- |
| `_internal/runtime_plan.py` | `_internal/plan_contracts/` |
| `_internal/runtime_capabilities.py` | `_internal/capability_contracts/` |
| `_internal/runtime_controller.py` ticket/receipt exports | `_internal/request_contracts/` |
| `_internal/model_placement_controller.py` record exports | `_internal/model_placement_contracts/` |
| `adapters/phone_session.py` configuration/event/receipt exports | `adapters/phone_session_contracts/` |
| `config.py` manifest exports | `configuration/` |
| `_internal/runtime_plan.py` remote-resident exports | `_internal/plan_contracts/remote_resident.py` |

The original modules re-export the same canonical class/function objects;
there are no duplicate dataclasses. Existing callers can retain their imports.
New internal code should import definitions from their owning contract module.
External campaigns should continue to use the public package API.

Configuration files separate rig, model, evidence, and campaign concerns.
Validation and serialization moved with their records; hashes and schemas
were not redesigned. The adaptive request record is in
`_internal/adaptive_decode_state.py`, and shared preparation validation is in
`_unified/helper_preparation_checks.py`.

## Split owners (2026-09-13)

The four largest implementation files were split without changing behavior.
Each owner keeps its public name and re-exports the moved definitions, so
existing imports keep working; new code should import from the new modules.

| Owner | Moved to | Contents |
| --- | --- | --- |
| `adapters/heterogeneous_rig.py` (rig class, snapshot, execution, time-sensitive transition steps) | `adapters/heterogeneous_rig_ops/common.py` | `_LiveExecutorResidency`, `_PersistentPhoneResidency`, `_HelperReconfiguration`, `_TransitionExecutionState` |
| | `adapters/heterogeneous_rig_ops/residency.py` (`RigResidencyMixin`) | residency resources, persistent phone residency state, residency views |
| | `adapters/heterogeneous_rig_ops/transitions.py` (`RigTransitionMixin`) | transition preparation, publication, cleanup, helper rollback |
| | `adapters/heterogeneous_rig_ops/lifecycle.py` (`RigLifecycleMixin`) | bridge start and qualification, session and executor stops, `close` |
| | `adapters/heterogeneous_rig_ops/observations.py` (`RigObservationMixin`) | executor samples, protected-work and allocation observations, execution success, `backend` |
| `adapters/llama_server.py` (`ManagedLlamaServer`, process launcher) | `adapters/llama_server_contracts.py` | launch contracts, phone FFN contracts, call parsing, execution proof records |
| | `adapters/llama_server_ops/proofs.py` (`ManagedServerProofMixin`) | execution call verification and proof construction |
| `_internal/route_generation/costing.py` (`RouteCostingMixin`: candidate inputs, preparation, refinement) | `costing_demands.py`, `costing_parameters.py`, `costing_estimates.py`, `costing_rough.py` | demand and transition rewriting; profiles, resource context, adapter parameters and plans; service, schedule, energy and marginal cost estimates; rough feasibility and visit generation |
| `_internal/policy.py` (profiles, `RoutePolicy`) | `_internal/policy_common.py`, `_internal/resource_timeline.py` | shared primitives and constants; `ResourceTimeline` with its lease records |
| `tests/test_automated_runtime.py` (fixtures, `AutomatedRuntimeTests` base) | `tests/test_automated_runtime_{residency,routes,runtime,admission,phone}.py` | the 185 runtime tests by theme, inheriting the fixture base |

Methods that tests control through `heterogeneous_rig.time` or the pinned
transition symbols stayed in the owner; patch a dependency in the module that
now uses it (for example `heterogeneous_rig_ops.residency.phone_ffn_resident_contract`).

## Remote-resident FFN weights

A desktop parent may omit the dense gate/up/down weights of layers that a
verified phone session owns (`RuntimeRemoteResidentFfn`, distinct from an
assisted copy where the desktop keeps every weight). The pieces and their
owners:

| Concern | Owner |
| --- | --- |
| Contract (artifact, tensor ids, dtype, shard hashes, sessions with generations, backing paths) | `_internal/plan_contracts/remote_resident.py`; `RuntimeExecutionContract.remote_resident_ffn` (desktop parents only) |
| Placement identity and qualification | `desktop_control_placement_payload(..., remote_resident_ffn)`, `RuntimeDesktopControlProfile.remote_resident_ffn`; the catalog requires the control profile and the coordinator declaration `remote_resident_ffn_v1` to agree |
| Route generation | `_internal/route_generation/remote_resident.py` (`RouteRemoteResidentMixin`): manifest and session cross-checks, owner generations bound from the READY layout, `REMOTE_RESIDENT_OWNER_NOT_READY:<session>`, omitted tensors removed from desktop weight demands, owners pinned as `session_residency_constraint` demands |
| Physical command and launch | `adapters/ticket.py` (`_validate_remote_resident_execution_command`), `adapters/llama_server.py` (`_remote_resident_launch_environment`, `ManagedLlamaServer.remote_resident_proof`, launch fails closed without a validated proof) |
| Accounting and recovery | `_internal/runtime_resources.py` (`RuntimeRemoteResidentOmissionProof`, `remote_resident_accounting`) |
| Bounded physical gates | `campaigns/burstgpt/remote_resident_gate.py` |

The C++ side is gated by `LLAMA_FFN_REMOTE_RESIDENT_LAYER_MASK` (server:
`S41_SERVER_FFN_REMOTE_RESIDENT_LAYER_MASK`): loader omission with page-exact
unmapping, graph builders forcing the full-phone FFN path, a context that
refuses to exist without an eval-callback owner, and a client that pins the
remote layers and refuses controls targeting them. Everything is omitted from
hashes and JSON when absent, so plans and goldens without the feature are
unchanged.

### Decode-only relocation (dormant host share)

The prefill of a remote-resident layer runs on the phone and is two times
slower than the desktop's PCIe-streamed prefill, so a second mode keeps every
weight on the desktop and relocates memory only for decode. It composes the
existing decode-phase assisted split (`ffn_assistance_phase=decode`,
`decode-boundary-v1`, phone share `split_fraction_ppm`) with page residency
control on the desktop:

| Concern | Owner |
| --- | --- |
| Page release/populate of still-mapped ranges | `src/llama-mmap.cpp` (`release_fragments`: `MADV_DONTNEED` on inward page-aligned pieces, plus `POSIX_FADV_DONTNEED` over the gate/up tensors whose share is contiguous; the row-interleaved `down` share keeps its pages in the cache, reclaimable; `populate_fragments`: readahead then `MADV_POPULATE_READ`) |
| Share geometry (gate/up suffix rows, `down` per-row suffix) and state | `src/llama-model.cpp` (`ffn_host_share_release/restore`, `llama_model_ffn_host_share_*` API); the loader records every dense FFN weight's file range (`dense_ffn_weights`) |
| Phase decision | `tools/server/server-context.cpp` (`apply_dormant_host_share`): release when every processing slot is generating under a split policy, populate before any prompt processing; `S41_SERVER_FFN_DORMANT_HOST_SHARE=1` (requires runtime control, excludes remote-resident layers) |
| Launch parameter | adapter parameter `ffn_host_share_release=1` -> `S41_SERVER_FFN_DORMANT_HOST_SHARE=1` (`adapters/llama_server_contracts.py`) |
| Proof and planning | `_internal/runtime_resources.py` (`RuntimeHostShareReleaseProof` from `S41SERVERFFN dormant_host_share phase=decode|local ...`; `host_share_release_lower_bound_bytes` for admission before a proof exists) |
| Native proof | `examples/layersplit/ffn-remote-resident-probe.cpp --dormant-mask --dormant-host-columns`; `tests/test_remote_resident_native.py::DormantHostShareNativeTests` |
| Bounded physical gate | `campaigns/burstgpt/reports/20260916-decode-only-relocation/dormant_gate.py` (Gemma, GPU parent); `campaigns/burstgpt/reports/20260917-decode-relocation-kv-headroom/kv_decode_relocation_gate.py` (Qwen3-14B, CPU parent, per-layer host KV, ratio sweep + matched pairs + budget cap) |
| Split selection and decode-phase credit | `_internal/decode_split_selection.py`: `select_decode_split` picks the phone share from the digest-pinned measured atlas (`campaigns/burstgpt/data/QWEN_DECODE_SPLIT_ATLAS.json`, built by the report's `build_decode_split_atlas.py`) by objective `latency` / `energy` / `memory`, optionally with a required decode-phase release and a latency bound, fail-closed outside the measured artifact/regime; `DecodeReleaseAccountant` keeps the phone share of one server endpoint as its own ledger owner (booked with a `ShareBinding` before the first prompt, credited once per release generation by a proof bound to that endpoint's layers/columns, shortfall kept, re-reserved before the next prompt in one transaction; a consumed room holds the prompt and leaves the ledger unchanged), so decode-phase credit never funds prefill; rows apply only to an exactly matching environment and a validated request shape; `tests/test_decode_split_selection.py`; physical two-request flow: the gate's `admission` arm |
| Page-granular host KV backing | `src/llama-kv-cache.cpp` (`llama_kv_cache_clear_buffer`): plain CPU KV buffers are zeroed with `MADV_DONTNEED` instead of `memset`, so the KV footprint tracks written cells and released weight pages can back KV growth; `LLAMA_KV_CACHE_EAGER_CLEAR=1` restores eager zeroing; `tests/test_kv_lazy_backing.py` |

The credit is phase-conditional: released bytes are host memory only while
the server decodes, and the next prefill pays the populate cost. It never
frees VRAM, and it cannot raise `n_ctx` by itself because llama.cpp allocates
the KV cache at launch while the weights must be resident for prefill.

## Working on the code

- Change behavior in its implementation module, not by adding a second
  implementation to a facade or a campaign.
- Keep session residency/model identity separate from request leases, decode
  progress, and attachment state.
- Keep load/drain/publication ordering and checkpoint rollback under the
  existing owner. Moving a method does not authorize changing that ordering.
- Patch a dependency in the module that uses it in tests. Patching a legacy
  re-export no longer intercepts calls inside a moved implementation.
- Add focused behavior tests for policy changes. Do not regenerate replay
  goldens merely because a refactor changes an output.

The cleanup does not remove historical experiments, results, native GGML/FFN
code, or saved baseline references. Other large files, including the physical
rig, HTTP backend, route costing, legacy policy, and large integration-test
fixtures, remain separate follow-up refactors; their internals were not
rewritten in this controller cleanup.

## Software verification

From the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_dev/scheduler/tests:. \
  python3 -m unittest test_module_boundaries test_architecture

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_dev/scheduler/tests:. \
  python3 -m unittest test_replay_determinism

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_dev/scheduler/tests:. \
  python3 research_dev/scheduler/tests/run_all.py
```

The complete harness isolates each canonical and historical S42 test module
in a separate process. These checks do not launch a physical inference
campaign. A structural cleanup alone establishes no new energy-saving or
hardware-performance result.
