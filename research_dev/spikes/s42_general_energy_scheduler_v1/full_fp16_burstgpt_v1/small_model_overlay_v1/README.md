# F16 BurstGPT plus resident Llama 1B overlay

This experiment keeps the exact 74-request F16 BurstGPT replay and adds the
same ten Llama 3.2 1B requests used by the focused Q4 screen. The combined
work is 84 requests, 38,948 input tokens, and 13,132 output tokens.

The large-model artifacts are unchanged F16 dequantized proxies:

| Model | Requests | Artifact bytes |
| --- | ---: | ---: |
| Qwen3-14B F16 proxy | 57 | 29,543,423,360 |
| Gemma4-12B F16 proxy | 17 | 23,832,065,056 |
| Llama 3.2 1B Q4_0 | 10 | 770,928,288 |

`build_fp16_small_overlay.py` derives the ten-request overlay from the
previous hash-bound trace. It does not retokenize, shorten, or alter the
original 74 BurstGPT requests. The base and overlay hashes remain separate in
`TRACE_MANIFEST.json` so the F16 runner can continue enforcing its immutable
source trace contract.

`build_phase_calibration_overlay.py` derives a separate 40-request trace from
the same ten natural Llama shapes. It anchors one copy of each shape to the
observed Qwen, switching, Gemma, and idle phase events. Seven shapes per phase
are training samples and three are held out in each repeat. The fitter requires
two repeats, giving 14 training and six held-out observations per phase and
large-model policy. This is a designed, phase-stratified calibration workload,
not a claim about the natural arrival distribution.

`build_natural_validation_overlay.py` derives a separate 40-request validation
trace by repeating the ten exact Llama shapes over a fixed base-2 van der
Corput schedule from 30 to 3,100 seconds. The schedule does not use phase
labels or observed phase boundaries. Two physical repeats per large-model
policy must naturally cover every Qwen, switching, Gemma, and idle contention
class before the fitted profile can pass.

## Independent experiment factors

The physical wrapper takes two independent policy arguments:

| Factor | Control | Treatment |
| --- | --- | --- |
| Large-model policy | `cpu-overflow` | `op15-assistance` |
| Small-model policy | `static-cpu` | `runtime-scheduler` |

This creates four experiment cells. A small-model scheduler comparison is
valid only between `static-cpu` and `runtime-scheduler` results that use the
same large-model policy. All four cells retain the resident phone state and
include whole-phone energy in the paid boundary.

OP15 exposes a composite FunctionFS plus NCM gadget. FunctionFS carries the
resident Qwen and Gemma FFN RPCs. NCM carries HTTP token traffic for the
resident Llama Adreno server. The combined residency contract accounts for
the three HTP slices, the Llama artifact, and a 768 MiB live memory reserve.
The initial 2026-08-12 run conservatively treated FunctionFS and NCM as one
exclusive transport for the whole trace. The current implementation models
FunctionFS and NCM as separate composite functions plus a two-lane desktop USB
root. FunctionFS holds one root lane only for an OP15-assisted large-model
phase; a phone task must acquire the NCM function, Adreno, and the other root
lane for its complete execution. The isolated qualification below validates
the NCM and Adreno route while the large models use CPU overflow. Concurrent
FunctionFS plus NCM execution under `op15-assistance` remains a separate 2x2
experiment cell and is not established by this result.

The current physical-integration branch can also start an HTP3 worker for an
exact GGUF-derived Llama FFN layer set. The general arm wrapper defaults to one
layer, matching the current physical shape calibration. The formal FFN
qualification campaign uses the same `S42_LLAMA_FFN_RESIDENT_LAYERS` value for
direct energy, route calibration, and held-out runtime execution. A physical
calibration is accepted only when its exact manifest hash matches that layer
set. In every mode, the live combined-residency gate must admit the requested
layers while preserving the complete Llama Adreno endpoint and the 768 MiB
reserve. Before admission, the controller physically prewarms the largest
phone-bearing token bucket with a bounded timeout. Process health alone cannot
mark the route ready. The matmul
virtual queue compiles a token-bucket split policy for exactly those resident
weights. That route remains shadow-only until repeated physical CPU/split
calibration passes; it is not silently selected from extrapolated kernel data.
When the active profile has no qualified split route and the experiment does
not explicitly force split execution, the wrapper omits HTP3 and its Llama FFN
weights entirely. An energy-rejected shadow route therefore cannot consume
phone memory or change the latency epoch of the resident Adreno route.
`materialize_ffn_split_route.py` is the admission boundary. It requires two
phase-calibration split runs, two natural split runs, two matched CPU natural
runs, two OP15 idle-only split runs, physical callback receipts, zero held-out
latency-bound violations, a bounded exposed join wait, a separately measured
isolated incremental-energy profile, and a positive paired whole-fleet energy
interval. Only then does it add the route
and an exact manifest/policy binding to a runtime profile. The controller
rejects a profiled split route if that binding does not match the live GGUF
manifest and compiled token-bucket policy.

The large-model phase manager leases FunctionFS, one composite USB lane, and
the shared HTP compute resource while Qwen or Gemma is active. In the
`op15-assistance` arm the Llama split route is eligible only after the observed
idle transition, with a measured contention-class-8 profile. It cannot overlap
active Qwen or Gemma HTP work.

## Scheduling cost

The runtime controller records six timings for every Llama decision:

| Timing | Included work |
| --- | --- |
| `probe_ns` | Reads of background executor and phone-health snapshots |
| `snapshot_ns` | `/proc/meminfo` plus NVML snapshot construction |
| `estimate_ns` | shape, identity, residency, capacity, latency, and energy estimates |
| `policy_ns` | route queueing, energy gate, and lease commit |
| `total_core_ns` | estimator plus policy commit |
| `total_controller_ns` | probe, snapshot, estimator, and policy commit |

The result reports mean, p50, p95, maximum, and sum in microseconds. Endpoint
I/O runs in the background monitor; request scheduling reads its bounded-age
cache and completion events request immediate refreshes. Probe and snapshot
cost are reported separately because they are controller work, not the core
scheduler algorithm itself.

A healthy endpoint with no immediately free slot remains an admissible route.
The scheduler reserves its predicted future interval, and the unified runtime
dispatch queue sleeps on that calendar or a predecessor completion event. It
does not poll the endpoint every 500 ms. An active lease remains occupied
until physical completion; its guard renews only near the predicted upper
bound and wakes conflicting queued requests for a fresh decision.

The CPU contention feature contract uses input/output shape, actual CPU prompt
batch size, active CPU slots (not queued request count), and one memory-pressure
proxy. The target desktop exposes uncore IMC devices but sets
`perf_event_paranoid=4`, so the result explicitly marks hardware memory
bandwidth unavailable and identifies Linux memory PSI as a proxy. It does not
mislabel that proxy as a measured bandwidth counter. Each request also records
phase-level RAPL package plus NVML board power; measured live power replaces
the pooled phase-power value in marginal-system accounting when available.
Contention fitting uses endpoint-reported prompt plus generation service time,
while retaining controller wall time and endpoint queue time as separate audit
fields. This prevents concurrent requests waiting behind a one-slot executor
from being learned as if the model itself were slower.

The 30 GiB desktop cannot safely keep the 1B CPU executor resident during the
peak F16 Qwen load. The controller therefore pre-warms the phone, waits for
the observed paid-start event after Qwen loading, and then loads and warms the
CPU executor before dispatching overlay work. This CPU load is inside the paid
energy boundary and is recorded as `cpu_executor_ready`; it is not hidden as
setup energy. Both matched small-model policies use the same ordering.

At the end of Gemma, the large runner terminates the large CPU/GPU executors,
closes the second large-model FFN session, emits the observed idle phase, and
leaves the three-session phone router plus the composite gadget alive. The
controller can then use the genuinely free VRAM, CPU, or resident phone for
idle arrivals. After all overlay work completes, it captures a final live
phone snapshot and writes a hash-bound `OVERLAY_COMPLETE` release receipt. The
large runner then opens and closes the third control session, which lets the
router tear down FunctionFS/NCM and restore ADB. An abort receipt fails the arm
but still permits deterministic teardown.

Phone thermal admission uses Android `thermalservice` status together with
selected battery, shell, CPU, NPU, GPU, and DDR sensors. The raw maximum over
every thermal zone is not a gate because unrelated zones on this handset can
report 95 C while Android reports no thermal throttling. A missing Android
thermal status remains unqualified.

## Qualification gates

`qualify_fp16_small_overlay.py` fails a physical result unless the selected
route equals the endpoint used, the endpoint task ID occurs in that server's
log, all active leases remain occupied until completion, phase leases are
released, live phone snapshots are present, and the trace, output, model, and
energy identities match. It also verifies that the large runner observed the
same release receipt after the final phone snapshot. A runtime-scheduler
result must physically execute at least one non-CPU route before it can
support a route-selection claim.

Output-work identity means the exact trace digest and exact per-request input
and requested/actual output-token counts. Generated-token content digests are
retained as execution outcomes but are not required to match across arms.
Even with fixed seeds and greedy sampling, continuous batching and different
physical backends can change floating-point ordering and therefore generated
tokens. Treating those content digests as workload identity would invalidate
otherwise identical repeated arms.

An upper-bound violation marks the release late. A non-baseline route is then
disabled for the rest of that run; a CPU violation marks the profile for
recalibration. Neither case can pass qualification as an unexplained lease
overrun.

## Isolated runtime-scheduler qualification

The following immutable result is from the previous whole-task-route revision.
It remains valid evidence for that revision, but it predates the background
health cache, the stricter two-repeat contention fit, live phase-power input,
and the physical FFN split route described above.

The corrected physical experiment held the large-model policy fixed at
`cpu-overflow` and changed only the small-model policy. Before the natural
trace, two 40-request calibration overlays measured all eight combinations of
large-model policy and Qwen, switching, Gemma, or idle phase. Each class used
seven training shapes and three held-out shapes. The fitted profile passed all
24 held-out upper-bound checks with zero violations.

The natural 84-request workload then ran in A-B-B-A order. All four arms
completed exactly 38,948 input tokens and 13,132 requested and actual output
tokens. Every arm passed trace, model, energy-boundary, endpoint-log, live
snapshot, active-lease, phase-release, and final resident-release gates.

| Run order | Small-model policy | Physical routes | Fleet energy | Duration | Small mean | Small SLO | Mean GPU util. |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| A1 | `static-cpu` | CPU 10 | 312.378 kJ | 2831.949 s | 44.576 s | 4/10 | 40.299% |
| B1 | `runtime-scheduler` | OP15 8, CPU 2 | 294.428 kJ | 2699.157 s | 4.693 s | 10/10 | 42.187% |
| B2 | `runtime-scheduler` | OP15 7, CPU 3 | 299.451 kJ | 2727.160 s | 5.275 s | 10/10 | 41.816% |
| A2 | `static-cpu` | CPU 10 | 312.206 kJ | 2819.418 s | 44.703 s | 4/10 | 38.973% |

| Matched pair | Fleet saving | Duration saving | CPU saving | GPU saving | Phone energy change | Small SLO gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 vs. B1 | 5.746% | 4.689% | 12.883 kJ | 4.922 kJ | -0.145 kJ | +6 |
| A2 vs. B2 | 4.086% | 3.272% | 11.317 kJ | 1.721 kJ | +0.281 kJ | +6 |

Both physical pairs improved fleet energy, duration, throughput, and SLO
attainment. Across the two repeats, static CPU completion averaged 44.640 s
and runtime scheduling averaged 4.984 s, an 88.8% reduction. The scheduler
selected OP15 only when its live endpoint and leases were available and its
measured route was energy-efficient. It selected CPU for two requests in B1
and three in B2 when the phone was still occupied or not ready. CUDA was not a
feasible 1B route while the active F16 model owned its capacity.

The server saving is larger than the added phone work. Moving seven or eight
requests away from the contended CPU shortened the paid trace by 92-133 s. In
each pair this reduced CPU package energy by 11.3-12.9 kJ and GPU board energy
by 1.7-4.9 kJ; the whole-phone change ranged from -0.145 to +0.281 kJ.

Core estimator plus policy time averaged 0.263 ms per request in B1 and 0.304
ms in B2. Including live snapshots and endpoint probes, controller time
averaged 172.632 ms and 182.267 ms per request. The occasional one-second
phone probe is included in the paid execution and explains why a nearby
arrival can observe the single phone slot as still occupied.

This is positive integration evidence, but it is not yet a qualified energy
claim. With only two matched samples, the mean fleet saving is 4.916% and its
paired 95% confidence interval is -5.634% to 15.466%. The lower bound is not
above zero, so `energy_claim_eligible` is false. The duration saving has the
same limitation: mean 3.981%, 95% CI -5.021% to 12.982%. More physical pairs
are needed to narrow both intervals.

The result also does not support a 25% scheduler-saving claim. The isolated
treatment changes placement for ten 1B requests while keeping the 74-request
F16 large-model policy fixed. The older 20.31% result changed the large-model
policy at the same time and therefore measured a different, confounded effect.

The ABBA report, fitted calibration profile and audit, and each arm's result,
phone-energy receipt, and endpoint qualification are preserved under
`results/4060ti_op15_20260812/isolated_cpu_overflow_abba_v1/`.

## Historical confounded integration result

The pair run on the RTX 4060 Ti desktop and OP15 on 2026-08-12 changed both
factors at once: `cpu-overflow + static-cpu` was compared with
`op15-assistance + runtime-scheduler`.
Both arms completed the identical 84 requests and 13,132 output tokens. The
base trace SHA-256 is
`b20a9ba66ee3558d835a0e19ed3cfa4c31a4a9e8b4f9c085b29a14f80250a0ff` and
the overlay SHA-256 is
`a39ed66211e490d6b5e95dae29ad76a4ad847e1a1d1e5f8c1004e4d7c16abaaf`.

| Metric | Server baseline | Runtime scheduler | Change |
| --- | ---: | ---: | ---: |
| Fleet energy | 306.670 kJ | 244.390 kJ | -20.31% |
| Server compute energy | 301.740 kJ | 237.550 kJ | -21.27% |
| CPU package energy | 207.005 kJ | 143.706 kJ | -63.299 kJ |
| GPU board energy | 94.734 kJ | 93.843 kJ | -0.891 kJ |
| Whole-phone energy | 4.931 kJ | 6.841 kJ | +1.910 kJ |
| Makespan | 2761.491 s | 2701.518 s | -2.17% |
| Mean GPU utilization | 39.71% | 42.26% | +2.55 points |
| Small-model mean completion | 46.494 s | 41.829 s | -10.04% |
| Small-model SLOs met | 3/10 | 4/10 | +1 |

The 20.31% reduction is a combined large-model OP15-policy result. It is not a
per-request scheduler saving. All ten Llama requests were scheduled
independently at arrival time, but each
selected `desktop-cpu`: the large-model placement owned CUDA, and the OP15
route was rejected as `SLO_INFEASIBLE` while the shared USB lease was held.
There were no transport recoveries or fallback executions.

OP15 performed 20,232 Qwen FFN bridge calls and 29,118 Gemma FFN bridge calls,
with zero reset recoveries. These calls are the physical phone work behind the
large-model energy reduction; none of the ten Llama requests executed on OP15.

In that historical run, the CPU latency model was not qualified under
simultaneous F16 CPU contention. All ten CPU leases completed after
their predicted upper bound: actual small-model completion averaged 41.829 s
against a 7.972 s predicted upper-bound latency. This did not make the selected
route unsafe in this run because CUDA and OP15 were already unavailable, but a
multi-route run needed contention-aware CPU calibration before relying on
those lease deadlines. The phase-conditioned calibration and held-out audit in
the isolated qualification above address that requirement for this model and
machine.

| Scheduler timing | Mean | p50 | p95/max | Sum for 10 decisions |
| --- | ---: | ---: | ---: | ---: |
| Runtime estimator | 109.540 us | 97.532 us | 225.560 us | 1.095 ms |
| Policy and lease commit | 209.054 us | 197.784 us | 387.128 us | 2.091 ms |
| Core scheduler | 318.594 us | 295.316 us | 612.688 us | 3.186 ms |
| Live snapshot | 32.149 ms | 30.114 ms | 37.533 ms | 321.485 ms |
| Executor probe | 3.644 ms | 1.465 ms | 21.078 ms | 36.439 ms |
| Full controller | 36.163 ms | 34.720 ms | 48.749 ms | 361.625 ms |

The core scheduler consumed 0.0032 s total. Including system snapshots and
health I/O, scheduling consumed 0.3616 s, or about 0.0134% of the 2701.518 s
makespan. The immutable result files are under
`results/4060ti_op15_20260812/`.

## All-request scheduling audit

The runtime-scheduler result directory contains three views of the same
84-request scheduling audit:

- `ALL_REQUEST_SCHEDULING_LOG.md` is the readable chronological table.
- `ALL_REQUEST_SCHEDULING_LOG.csv` is the compact analysis table.
- `ALL_REQUEST_SCHEDULING_LOG.jsonl` retains snapshots, route costs, leases,
  releases, and exact nanosecond overhead fields.

The audit intentionally distinguishes two scheduling scopes. The 74 F16
requests have `runtime_placement_inherited`: the unified scheduler selected the
large-model placement from a live startup snapshot, but it did not invoke the
cost estimator again at each request arrival. The ten Llama requests have
`per_request_runtime_cost_and_lease`: each arrival produced a new live snapshot,
three route estimates, a policy decision, and a resource lease. Thus this log
is evidence for runtime placement of all work and per-request runtime
scheduling of the Llama overlay, not yet per-request runtime cost scheduling of
all 84 requests.

For an autonomous whole-trace run, pass `runtime-auto` as the large-model
policy. The wrapper still captures the live placement snapshot and compiles a
hash-bound plan, but the physical runner now derives `control` or `op15` from
that plan instead of accepting an external arm choice. The measured baseline
is an executable fallback when no non-baseline placement passes all gates.
Use explicit `cpu-overflow` and `op15-assistance` only for isolated matched
experiments.

`SCHEDULER_PROFILE_RUNTIME_AUTO_V1.json` is the enforce-mode profile for this
path. `build_runtime_auto_profile.py` reproducibly combines the eight measured
CPU contention classes with the disjoint natural-arrival phone latency and
lease model. Its audit requires every phase class to pass its holdout bounds,
the phone route to have disjoint training and holdout inputs, and zero upper
bound violations. The checked-in profile SHA-256 is
`a9093312e9d48ece2716259ecb059b85d38005d53962a8ee714a91ffbd24c68d`.

During an OP15-assisted Qwen or Gemma phase, the phase manager leases HTP,
FunctionFS, one composite USB slot, and Adreno. This conservatively prevents
unqualified simultaneous HTP and Adreno execution. Executor health and route
availability are separate: a healthy phone remains unavailable while that
open-ended phase lease is active. The controller does not enqueue work behind
a rolling release estimate. In the OP15-assisted arm, the Adreno and split
routes reopen after the observed idle transition. A future controller-to-large
runner start handshake is required before safely admitting phone work in a
switching interval.

The physical wrappers fail before changing phone state when the reported
battery level is below `S42_MIN_PHONE_BATTERY_LEVEL`, which defaults to 20.
This floor is a run-integrity check because an incomplete phone capture cannot
support a fleet-energy result. A deliberate experiment may set a different
floor explicitly after verifying its power setup.

The audit can be regenerated from the immutable run artifacts with:

```sh
python3 build_all_request_scheduling_log.py \
  --run-dir results/4060ti_op15_20260812/runtime_scheduler
```

## Reproduce

```sh
python3 build_fp16_small_overlay.py
python3 build_phase_calibration_overlay.py
python3 build_combined_residency_plan.py

python3 build_runtime_auto_profile.py \
  --phase-profile \
    results/4060ti_op15_20260812/isolated_cpu_overflow_abba_v1/calibration/PHASE_PROFILE_V1.json \
  --phase-audit \
    results/4060ti_op15_20260812/isolated_cpu_overflow_abba_v1/calibration/PHASE_PROFILE_AUDIT_V1.json \
  --natural-profile \
    results/4060ti_op15_20260813/natural_runtime_v3_matched_v1/RUNTIME_PROFILE.json \
  --natural-audit \
    results/4060ti_op15_20260813/natural_runtime_v3_matched_v1/RUNTIME_PROFILE_AUDIT.json \
  --output-profile /new/absolute/path/RUNTIME_AUTO_PROFILE.json \
  --output-audit /new/absolute/path/RUNTIME_AUTO_PROFILE_AUDIT.json

bash run_fp16_small_overlay_arm.sh \
  /absolute/output/auto-runtime runtime-auto runtime-scheduler 5037

bash run_fp16_small_overlay_arm.sh \
  /absolute/output/cpu-static cpu-overflow static-cpu 5037
bash run_fp16_small_overlay_arm.sh \
  /absolute/output/cpu-runtime cpu-overflow runtime-scheduler 5037
bash run_fp16_small_overlay_arm.sh \
  /absolute/output/op15-static op15-assistance static-cpu 5037
bash run_fp16_small_overlay_arm.sh \
  /absolute/output/op15-runtime op15-assistance runtime-scheduler 5037

python3 compare_fp16_small_overlay.py \
  --static-result /absolute/output/cpu-static/combined/RESULT.json \
  --static-phone /absolute/output/cpu-static/capture/PHONE_ENERGY.json \
  --static-qualification /absolute/output/cpu-static/capture/QUALIFICATION.json \
  --runtime-result /absolute/output/cpu-runtime/combined/RESULT.json \
  --runtime-phone /absolute/output/cpu-runtime/capture/PHONE_ENERGY.json \
  --runtime-qualification /absolute/output/cpu-runtime/capture/QUALIFICATION.json \
  --output /absolute/output/CPU_OVERFLOW_SCHEDULER_COMPARISON.json
```

Run the same comparison separately for `op15-static` versus `op15-runtime`.
The comparison rejects a large-model-policy mismatch. One matched pair is an
integration result, not an A-B-B-A qualification; a stable energy claim still
requires repeated paired runs and confidence intervals.

Run the complete symmetric 2x2 experiment with:

```sh
bash run_fp16_small_overlay_2x2_abba.sh \
  /absolute/output/factorial-2x2 5037 \
  /absolute/input/PHASE_PROFILE_V2.json \
  /absolute/input/MARGINAL_SYSTEM_PROFILE.json
```

The physical order is `A B C D D C B A`: CPU-overflow static/runtime,
OP15-assistance static/runtime, then the reverse. The script produces one ABBA
report per fixed large-model policy and `FACTORIAL_2X2_ABBA.json`. Scheduler
energy eligibility is evaluated separately in each policy; the report does
not attribute the large-model policy effect to per-request scheduling. It also
uses the same explicitly supplied marginal profile in all eight cells and the
phase-agnostic 40-request natural-validation overlay. It also fits
`NEXT_MARGINAL_SYSTEM_PROFILE.json` from the paired CPU-service and
critical-path deltas, idle-GPU samples, CPU package power, and whole-phone
power. Validate that newly fitted profile in a second, held-out invocation of
the script. The first 2x2 cannot itself prove the effect of a profile derived
after those runs.

```sh
bash run_fp16_small_overlay_2x2_abba.sh \
  /absolute/output/factorial-2x2-heldout 5037 \
  /absolute/input/PHASE_PROFILE_V2.json \
  /absolute/output/factorial-2x2/NEXT_MARGINAL_SYSTEM_PROFILE.json
```

To wait for a running phase-calibration service and execute the seed 2x2,
held-out 2x2, and physical FFN-split qualification in dependency order, use:

```sh
bash run_full_scheduler_qualification_pipeline.sh \
  /absolute/output/full-qualification 5037 \
  /absolute/output/phase-contention \
  phase-contention.service \
  /absolute/input/LLAMA_FFN_PHYSICAL_CALIBRATION.json \
  /absolute/input/MARGINAL_SYSTEM_PROFILE.json \
  /absolute/input/LLAMA_FFN_DIRECT_ENERGY.json \
  /absolute/input/LLAMA_FFN_COMPILED_POLICY.json
```

The pipeline verifies the byte hash of the qualified phase profile before
starting. The seed result fits the marginal profile used by the held-out 2x2;
only that held-out comparison can validate the new marginal policy. The split
qualification then uses the next independently fitted marginal profile and
stays shadow-only unless every physical latency and energy gate passes.

For an isolated acquisition, run the phase calibration and qualification in
one service instead of polling a running service:

```sh
bash run_full_scheduler_campaign.sh \
  /absolute/output/full-campaign 5037 \
  /absolute/input/LLAMA_FFN_PHYSICAL_CALIBRATION.json \
  /absolute/input/MARGINAL_SYSTEM_PROFILE.json
```

The full campaign first generates `LLAMA_FFN_DIRECT_ENERGY.json` itself. It
keeps the normal four-session OP15 placement resident, runs two isolated ABBA
cycles through the desktop CPU and physical CPU/HTP3 split, samples CPU
package, GPU board, and whole-phone energy, and subtracts bracketing resident
idle measurements. The last ABBA cycle remains held out from route-energy
fitting. The exact compiled policy is then reused by all split qualification
runs, so direct energy, physical execution, and runtime admission have one
hash-bound placement identity.

`S42_FFN_PYTHON` must name a Python interpreter that can import NumPy for the
repository GGUF reader. The wrapper checks this before changing phone state.

First turn a representative physical shadow run into a conservative shape
guard. The guard compares measured CPU/phone overlap against the estimated
CPU-only bound with uncertainty margins. It disables phone execution for the
small-batch prefix and preserves only a repeatedly faster full-phone prefill
suffix.

```sh
python3 calibrate_llama_ffn_split_policy.py \
  --manifest /absolute/shadow-run/capture/LLAMA_FFN_MANIFEST.json \
  --estimated-policy \
    /absolute/shadow-run/capture/LLAMA_FFN_COMPILED_POLICY.json \
  --physical-log \
    /absolute/shadow-run/combined/llama1-cpu-phone-ffn.stderr \
  --source-label full-fp16-burstgpt-qwen-contention \
  --output /absolute/output/LLAMA_FFN_PHYSICAL_CALIBRATION.json
```

After producing the representative phase-conditioned CPU profile, acquire and
admit the physical HTP3 split for the exact configured layer set with:

```sh
bash run_ffn_split_qualification_abba.sh \
  /absolute/output/ffn-split-qualification 5037 \
  /absolute/input/PHASE_PROFILE_V2.json \
  /absolute/input/LLAMA_FFN_PHYSICAL_CALIBRATION.json \
  /absolute/input/MARGINAL_SYSTEM_PROFILE.json \
  /absolute/input/LLAMA_FFN_DIRECT_ENERGY.json \
  /absolute/input/LLAMA_FFN_COMPILED_POLICY.json
```

The natural-validation runs use CPU-split-split-CPU order with identical phone
residency. The same configured Llama FFN layer set is resident in both arms;
the default is one layer, matching the current physical calibration. The
calibrated table keeps decode and other unsafe small batches on CPU. The
wrapper and compiler stop if that placement cannot preserve the 768 MiB live
phone reserve.
Two additional phase-anchored split runs provide training and held-out shapes.
Two idle-only OP15 split runs qualify the only safe OP15 sharing class. The
direct-energy input must be a hash-bound
`s42-llama1b-ffn-direct-energy-v1` record measured with sequential,
non-overlapping requests and resident-idle subtraction. Mixed-trace average
power is not accepted as per-request energy. If
every admission gate passes, the script emits
`SCHEDULER_PROFILE_WITH_FFN_SPLIT.json` and runs one final natural trace through
the unified runtime scheduler with a required physical split selection. A
failed gate leaves the split shadow-only and stops before that final run.

Append `phase-calibration` to a wrapper command to select the 40-request
phase-anchored trace. Use `natural-validation` for the phase-agnostic
40-request holdout and `idle-split-calibration` for the 40-request OP15 idle
qualification trace. `direct-energy-calibration` runs the isolated ABBA
energy acquisition without replaying the large-model trace. The default fifth
argument is `qualification`, which keeps the original ten target-workload
arrivals.

Fit the eight CPU contention classes from two successful static phase runs and
two phase-agnostic natural-validation runs per large-model policy:

```sh
bash run_phase_contention_calibration_abba.sh \
  /absolute/output/phase-contention-calibration 5037
```

The command above runs the eight required acquisitions in symmetric order and
emits `PHASE_PROFILE_V2.json` plus its audit. The equivalent individual
commands and fitter invocation are:

```sh
bash run_fp16_small_overlay_arm.sh \
  /absolute/output/cal-cpu-r1 cpu-overflow static-cpu 5037 phase-calibration
bash run_fp16_small_overlay_arm.sh \
  /absolute/output/cal-cpu-r2 cpu-overflow static-cpu 5037 phase-calibration
bash run_fp16_small_overlay_arm.sh \
  /absolute/output/cal-op15-r1 op15-assistance static-cpu 5037 phase-calibration
bash run_fp16_small_overlay_arm.sh \
  /absolute/output/cal-op15-r2 op15-assistance static-cpu 5037 phase-calibration
bash run_fp16_small_overlay_arm.sh \
  /absolute/output/natural-cpu-r1 cpu-overflow static-cpu 5037 natural-validation
bash run_fp16_small_overlay_arm.sh \
  /absolute/output/natural-cpu-r2 cpu-overflow static-cpu 5037 natural-validation
bash run_fp16_small_overlay_arm.sh \
  /absolute/output/natural-op15-r1 op15-assistance static-cpu 5037 natural-validation
bash run_fp16_small_overlay_arm.sh \
  /absolute/output/natural-op15-r2 op15-assistance static-cpu 5037 natural-validation

python3 fit_phase_contention_profile.py \
  --cpu-overflow-result /absolute/output/cal-cpu-r1/combined/RESULT.json \
  --cpu-overflow-result /absolute/output/cal-cpu-r2/combined/RESULT.json \
  --op15-result /absolute/output/cal-op15-r1/combined/RESULT.json \
  --op15-result /absolute/output/cal-op15-r2/combined/RESULT.json \
  --cpu-overflow-natural-result \
    /absolute/output/natural-cpu-r1/combined/RESULT.json \
  --cpu-overflow-natural-result \
    /absolute/output/natural-cpu-r2/combined/RESULT.json \
  --op15-natural-result \
    /absolute/output/natural-op15-r1/combined/RESULT.json \
  --op15-natural-result \
    /absolute/output/natural-op15-r2/combined/RESULT.json \
  --base-profile SCHEDULER_PROFILE_COMPOSITE_USB_V2.json \
  --output-profile /absolute/output/PHASE_PROFILE.json \
  --output-audit /absolute/output/PHASE_PROFILE_AUDIT.json
```

The fitter exits nonzero and marks its audit `FAIL` if any phase-stratified or
covered natural-arrival upper bound is violated. Use the profile only when the
audit is `PASS` and
`all_variants_measured` is true. Then run the natural-arrival qualification
trace with `S42_RUNTIME_PROFILE=/absolute/output/PHASE_PROFILE.json`, compare
static and runtime arms at the same large-model policy, and repeat in A-B-B-A
order. `compare_fp16_small_overlay_abba.py` reports the paired 95% confidence
interval and sets `energy_claim_eligible` only when its lower bound is above
zero.

For a held-out retest of one already-observed natural contention class, fit
both whole-task routes from disjoint repeated traces with:

```sh
python3 fit_runtime_route_profile.py \
  --train-result /absolute/calibration/static-r1/RESULT.json \
  --train-result /absolute/calibration/runtime-r1/RESULT.json \
  --train-result /absolute/calibration/recent-runtime/RESULT.json \
  --holdout-result /absolute/calibration/static-r2/RESULT.json \
  --holdout-result /absolute/calibration/runtime-r2/RESULT.json \
  --base-profile SCHEDULER_PROFILE_COMPOSITE_USB_V2.json \
  --large-model-policy cpu-overflow \
  --output-profile /absolute/calibration/RUNTIME_PROFILE.json \
  --output-audit /absolute/calibration/RUNTIME_PROFILE_AUDIT.json
```

This fitter accepts both legacy and current runtime contention receipts, but
emits only classes represented by enough training and holdout samples. It
fits desktop CPU and phone service independently, adds a 25 percent training
guard, requires zero held-out upper-bound violations, and updates all
whole-phone leases with the same service model and uncertainty. Training and
holdout file hashes must be disjoint, and every input must carry identical
trace and model identities. A class absent from the resulting profile fails
closed; use the eight-class phase campaign above when a policy must cover all
large-model phases and both large-model arms.
