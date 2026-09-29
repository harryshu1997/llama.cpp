# S42 scheduler experiments and evidence

S42 is the immutable experiment, trace, and historical evidence area for the
canonical scheduler in `../../scheduler/`. Online policy, trace parsing,
lifecycle selection, resource calendars, placement, residency, matmul
planning, reusable adapters, and active campaign entry points belong to that
package. Files here must not implement scheduling or physical runtime policy.

Energy profiles now also support `operator_sum_v1`. This model adds CPU, GPU,
phone, and transport operator energy using effective kernel throughput,
effective bandwidth, launch time, idle power, and active power. It retains the
older affine route model for compatibility.

## Files

- `../../scheduler/`: the one canonical `UnifiedScheduler`, trace adapters,
  and its private contracts and planners. Evaluated alternatives
  bind an exact model, shape, scheduling unit, work set, quality requirement,
  resource set, backend configuration, and fleet-energy boundary.
- `RESOURCE_LEASES_V1.md`: phase leases, interval-calendar queue prediction,
  upper-bound reservations, and actual-completion release.
- `runtime_snapshot_probe.py`: read-only fail-closed RTX 4060 Ti plus OP15
  snapshot acquisition.
- `runtime_gate_audit.py`: immutable contract-versus-snapshot audit.
- `RUNTIME_GATES_V1.md`: runtime transaction, current I3 capabilities, and the
  physical idle audit.
- `RUNTIME_GATE_IDLE_AUDIT_4060TI_OP15_V2.json`: immutable idle fail-closed
  audit against the Stage 6 calibrated 1.1-second heartbeat contract.
- `MATMUL_VQ_SCHEDULER_V1.md`: op contract, equations, queue behavior, memory
  rules, current shadow-only boundary, usage, and limitations.
- `FULL_BURSTGPT_TWO_MODEL_4060TI_OP15_V1.md`: matched full source-length
  two-model GPU-switch control and CPU/OP15-overlap physical result.
- `materialize_matmul_vq_profile.py`: physical-evidence adapter for the
  canonical profile materializer.
- `MATMUL_VQ_EXAMPLE_WORKLOAD_V1.json`: two-matmul model-program example with
  a shard-safe GELU placement follower.
- `PLACEMENT_PLANNER_V1.md`: objective, current four-engine coverage, and the
  remaining runtime and physical-calibration work.
- `../../scheduler/profiles/MEASURED_4060TI_OP15_KERNEL_PROFILE_V1.json`:
  hash-bound three-repeat CPU, CUDA, HTP, Adreno, PCIe, USB, preparation,
  load, and switch measurements.
- `KERNEL_ENERGY_PROFILE_4060TI_OP15_V1.md`: compact physical campaign report
  and qualification boundary.
- `kernel_energy_v1/`: repeatable physical acquisition and profile
  materialization tools.
- `graph_adapter_v1/`: validated real `llama_decode` physical-ubatch graph to
  placement-candidate adapter, with Gemma and Qwen RTX 4060 Ti receipts.
- `runtime_routes_v1/`: strict certificate compiler and two direct I3 cohort
  routes bound to the exact model, binary, policy, workload, and device epoch.
- `physical_ab_v1/`: unified-scheduler cohort planner, thin monitored physical
  adapter, runtime-gate receipts, strict I3 reuse, and the Stage 6 RTX 4060 Ti
  plus OP15 result.
- `whole_task_phone_v1/`: byte-identical resident desktop CPU, desktop CUDA,
  and OP15 Adreno whole-task routes, including measured CUDA lifecycle-tail
  accounting and cold-versus-reused CUDA epoch profiles.
- `small_model_phone_v1/`: measured resident BGE batch-32 CUDA and OP15 Adreno
  profiles, conservative cohort crossover, live residency gates, and
  fail-closed CUDA epoch-tail receipt selection.
- `multi_session_phone_v1/`: reconnectable HTP0/HTP1/HTP2 resident FFN
  workers, composite model arming, hash-bound residency plan, and the repeated
  Qwen full-FFN replacement energy screen.
- `dynamic_residency_v1/`: strict two-model residency shadow, exact GPU tensor
  manifests, capacity and service screens, and disabled OP15-fenced GPU
  transfer plus same-process Gemma adoption qualification. Its matched
  whole-request overlap screen regressed fleet energy 15.33%, so the dynamic
  full trace is blocked and the measured static fallback remains active.
- `OPERATOR_ENERGY_MODEL_V1.md`: operator equations, boundaries, and physical
  calibration scope.
- `trace_adapter.py`: mixed-trace manifest checks around the canonical
  `research_dev.scheduler.load_trace` API.
- `calibrate_profiles.py`: physical-result importer and nonnegative cost fit.
- `replay.py`: deterministic multi-mode trace replay.
- `analyze_replay.py`: physical-versus-predicted comparison.
- `live_probe.py`: read-only RTX 4060 Ti and OP15 identity/readiness probe.
- `physical_4060ti_op15_profile.json`: hash-bound physical profile.
- `BURSTGPT_REPLAY_V1.json`: control, enforce, shadow, and capacity replay.
- `BURSTGPT_REPLAY_EXACT_V1.json`: exact-quality fail-closed replay.
- `LIVE_PROBE_4060TI_OP15_V1.json`: current read-only hardware receipt.
- `BURSTGPT_LIVE_READINESS_DRY_RUN_V1.json`: current-readiness dry replay.
- `ANALYSIS_V1.json`: physical comparison and energy verdict.
- `PHYSICAL_LOOP_CONTRACT.md`: fixed energy, overlap, and reporting rules.
- `physical_iteration.py`: strict A/B physical iteration reporter.
- `render_physical_iteration.py`: fixed human-readable physical table.
- `PHYSICAL_ITERATION_I0.json`: current real-device latency/work result with
  missing accounted device energy left explicitly null.
- `PHYSICAL_ITERATION_I0.md`: rendered real-device headline and handoff.
- `model_energy_v1/`: bounded Qwen3-14B and Gemma4-12B cohort-energy
  validation, server-only RAPL/NVML acquisition, explicit 5 W prototype phone
  estimate, synchronized phone attachment, and held-out fitter.
- `../s41_gemma_qwen_continuous_baseline/tp_operator_split_v1/RESULTS_LLAMA_SERVER_I3_ENERGY_V1.md`:
  current three-pair real-device latency, quality, overlap, and fleet-energy
  authority for the fixed source-length BurstGPT workload mix.
- `physical_ab_v1/STAGE6_PHYSICAL_AB_4060TI_OP15_V1.md`: current one-pair
  monitored successor validation of that certified epoch.
- `physical_ab_v1/UNIFIED_SCHEDULER_BURSTGPT_4060TI_OP15_R5_V1.md`: fresh
  scheduler-owned physical pair with explicit control and treatment plan
  bindings.
- `../s41_gemma_qwen_continuous_baseline/tp_operator_split_v1/burstgpt_gpu_cpu_op15_v1/analyze_i3_campaign.py`:
  strict successor-campaign aggregate validator.
- `../s41_gemma_qwen_continuous_baseline/tp_operator_split_v1/burstgpt_gpu_cpu_op15_v1/render_i3_campaign.py`:
  fixed human-readable I3 campaign renderer.

## Test

From the repository root:

```sh
python3 research_dev/scheduler/tests/run_all.py
```

## Replay

```sh
python3 research_dev/spikes/s42_general_energy_scheduler_v1/replay.py \
  --profile research_dev/spikes/s42_general_energy_scheduler_v1/physical_4060ti_op15_profile.json \
  --trace research_dev/spikes/s41_gemma_qwen_continuous_baseline/tp_operator_split_v1/burstgpt_gpu_cpu_op15_v1/REQUESTS_SEMANTIC_LONG.jsonl
```

The default modes are `control,enforce,shadow,capacity`. Use
`--quality-requirement exact` to reject the approximate CPU-plus-OP15 route.
Use `--runtime-state LIVE_PROBE_4060TI_OP15_V1.json` to overlay current
resource readiness on the historical physical profile.

## Live probe

```sh
python3 research_dev/spikes/s42_general_energy_scheduler_v1/live_probe.py
```

The probe discovers only already-running ADB servers. It does not start or
stop ADB, launch a model, change USB mode, load a worker, or alter clocks.

## Profile extension

An affine cost may use `input_tokens` and `output_tokens`, or arbitrary
nonnegative request features through `affine_features_v1`. Examples include
batch, context length, transferred bytes, FLOPs, image tiles, active experts,
or KV-cache bytes. A new device or workload therefore requires a new measured
route row, not a scheduler code change.

An operator cost can instead express invocation count, primitive compute ops,
and physical memory bytes as constants or affine request features. The
scheduler publishes the resulting per-operator and per-domain breakdown with
each new decision. The placement compiler reuses the same kernel equation for
each candidate and adds directed transfer and memory-pool costs before choosing
a device.

## Unified scheduler migration

The canonical contracts, policy engines, runtime gates, placement compiler,
operator split, phone DMA/USB transport, offload composition, hash-bound
executor plan, matmul virtual queue, and residency planner live in
`../../scheduler/`. Old S42 module paths remain only as compatibility imports.
The test runner executes scheduler-owned tests first, followed by S42 adapter,
artifact, and acquisition tests.

The BurstGPT adapter no longer chooses the I3 policy itself. It asks the
unified cohort scheduler to select control or energy enforcement, saves the
hash-bound plan, configures the server, bridge, worker, split table, and USB
path from that plan, and rejects a mismatch before the paid trace. The C++
bridge and phone worker remain execution backends.

The 2026-08-08 scheduler-owned physical pair selected
`i3-cold-cpu-op15-ffn-v1` for `VERIFIED_COHORT_ENERGY_SAVING`. Both arms
completed the exact 74-request cohort, and the strict pair record reports
-16.00% accounted fleet energy and -14.11% makespan with unchanged SLO and
pinned MMLU64 results. The remaining physical integration boundary is an
atomic pre-dispatch phone thermal snapshot while FunctionFS owns the USB link;
it is no longer an execution-plan ownership gap.

The successor scheduler adapts a new model at model-registration time. It may
generate predicted task, layer, and matmul alternatives from model structure
and calibrated device models, then promote them through measured and stable
maturity levels. The hot request path selects materialized alternatives and
updates queue, residency, thermal, bandwidth, and power estimates; it does not
repeat an expensive tensor-cut search for every token.

Token-exact output is not required for semantic routes. Such routes still
require finite, non-empty output and a declared validation identity. Wider
hardware timing or numeric variation is represented by conservative metric
bounds rather than by rejecting the model family.

External model acquisition ending with a durable disk copy is outside runtime
accounting. Disk-to-RAM loading, device transfer, repacking, model switching,
and residency eviction remain runtime costs because they consume capacity and
can delay requests.
