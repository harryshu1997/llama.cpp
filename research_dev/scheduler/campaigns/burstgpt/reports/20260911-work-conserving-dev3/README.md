# Work-conserving CPU parent and phone FFN registration

Status: the unchanged three-request physical experiment completed 3/3 with valid
terminal proofs and semantic-sanity output checks. Registration and safety
changes pass focused validation, but the start-early objective is NOT yet
demonstrated: Llama still waited and used the GPU. Live request migration is not
implemented. See the blockers below and `SUMMARY-final.json`.

## Scope and ownership

The canonical catalog materializer can add an independently measured CPU parent
and its CPU/NPU helper family alongside the existing GPU desktop control. The
normal scheduler still chooses the execution route from current demand, costs,
resource availability, memory, and evidence. No runner forces a route, fraction,
session, or future request. Existing per-session lifecycle, helper leases,
replacement, and terminal-proof implementations are unchanged.

The experiment registers the existing Llama FFN shard index through the same
verified storage path used by the large models. A model can have stored shards
for a subset of discovered sessions. Only those sessions can receive newly
generated assignments for that model; unknown session IDs fail registration.
No shard format, generator, phone worker, or native transport was changed here.

An all-CPU server is launched with `--device none`, `--no-kv-offload`, and an
empty `CUDA_VISIBLE_DEVICES` in that subprocess only. The CPU parent has its own
measured profile and launch parameters; it does not inherit GPU qualification.
GPU server environments and graph mode remain unchanged.

Memory admission retains model weights, KV, workspace, and safety reserves.
Available RAM is not treated as permission to omit those allocations. The
current three HTP sessions still accommodate at most three independently
resident assignments, even if some phone RAM is free.

Live CPU-to-GPU migration is not implemented: the HTTP execution path has no
validated KV/sampler-state handoff. A running request finishes on its exact
parent. Subsequent requests can select the GPU when it is available. Restarting
or replaying an active request is not reported as migration.

## Frozen experiment

Remote root: `/mnt/storage/s42-work-conserving-dev3-20260911-v1-NrWH4c`

| Request | Model | Arrival | Output tokens |
| --- | --- | ---: | ---: |
| 36 | Gemma | 1 s | 292 |
| 37 | Llama | 61 s | 292 |
| 50 | Qwen | 91 s | 71 |

The source workload is unchanged `burstgpt_dev3_long_v1.json`. Normal energy-aware
scheduling and the existing adaptive controller remain enabled. Initial
automated/adaptive observations are the frozen reference stores. Runtime
preparation and cleanup remain in the paid interval. No 24- or 84-request trace,
baseline rerun, commit, push, or PR is authorized by this experiment.

Actual configuration: `inputs-v3/SPEC.json` and the other files in `inputs-v3/`
under the remote root. Earlier partial `inputs/` and `inputs-v2/` are preserved
but not used. Versioned experiment scripts preserve the preparation attempts.

The fresh CUDA-capable executable includes the previously tested packed-prefix
fix; this turn did not rewrite those native paths. Graph support is built in and
graph mode is `default`. The build passed 296 native numerical cases and two
invalid-view checks. Executable SHA-256:

`c1c1613611e34ac6c96c5552bf5f0f3ee07e01be4e22ff96b921f1c212e2b6cb`

The transport identity is rebound to this executable and its libraries using
the existing qualification mechanism. Its transport-client source is identical
to the measured source; the original receipts and phone-worker identity are
retained. This is not a new throughput qualification or a same-binary baseline.
Any comparison with an earlier run is historical/diagnostic, not matched A/B.

## CPU parent calibration

Both calibration requests use the unchanged Llama request payload: 915 input
tokens, 292 output tokens, seed 42. Context 4096, batch 1024, ubatch 256,
parallel 1, threads 4, batch threads 8, GPU layers 0. CPU affinity/poll settings
are the canonical defaults, not the earlier standalone benchmark overrides.

| Attempt | Cold request | Hot request | Load | GPU isolation |
| --- | ---: | ---: | ---: | --- |
| cpu-parent-v1 | 10.835 s | 10.857 s | 0.917 s | Failed: 120 MiB process allocation |
| cpu-parent-v2 | 10.850 s | 10.805 s | 0.617 s | PID absent from NVML process list |

Both attempts completed and passed semantic validation. The v1 `status=PASS`
means request completion, not GPU isolation; it is preserved as failed evidence
for that requirement. The v2 native launch also has no visible CUDA devices.
An absent NVML process entry is not presented as a measured numeric zero.

V2 server energy (CPU RAPL package plus GPU board): cold 1008.12 J, hot 1017.29 J;
load 25.70 J. Peak observed CPU RSS high-water: 1,733,230,592 bytes. These are
isolated CPU-parent measurements, not a qualification of concurrent interference
with Gemma. The registered shape profile uses 25% heuristic bounds, not
statistical confidence intervals. No CPU/NPU energy qualification is synthesized.

Artifacts are copied locally under `physical/cpu-parent-v1/` and
`physical/cpu-parent-v2/`.

## Validation and preserved failures

- 297 focused helper/adaptive/runtime/catalog/launch tests passed, including both
  replay goldens. Log: `FOCUSED-final.log`.
- 17 targeted preflight/shard/config tests passed after preflight was made to
  register the same verified shard-storage metadata as runtime.
- First physical preflight passed before that storage-registration correction.
- `preflight-v2` failed closed: storage registration incorrectly required every
  model to have a file for every discovered session. The Llama index correctly
  contains only one shard. Registration now permits a subset while the existing
  candidate generator excludes all unbacked sessions.
- The storage/offline/replay group passed 24 tests in 123.144 s, including the new
  subset regression and both unchanged replay goldens: `STORAGE-and-replay.log`.
- Final `preflight-v3` passed 103 checks (101 PASS, two pre-existing cost-evidence
  WARNs for the large desktop parents). Its `phone_assistance_ready=false` is
  preserved: normal bounded LEARNING remains necessary; this is not proof that
  all phone routes were qualified. No gate was bypassed.

Replay goldens are not changed:

- Session COW v3:
  `ea5b30c9e4a3f705c194188d6e5788c64990953aad73ff3c4b7573bc69eca27d`
- Sparse locality v8:
  `965f218bb5f81a57dbb51167624798f02eb79354b6cc042683b2b1ed9b1a868d`

`source-before/` preserves the prior dirty contents of files edited in this
turn. Historical physical results, native changes, and unrelated user edits
were not reverted. Build/configuration failures and preflight attempts have
separate artifacts rather than being overwritten.

## Physical result

| Request | Service | Arrival to completion | Phone calls | Assisted tokens | Fraction-weighted all-token coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma 36 | 134.803 s | 178.712 s | 6,720 | 280/292 (95.89%) | 90.24% |
| Llama 37 | 1.547 s | 131.349 s | 0 | 0/292 | 0% |
| Qwen 50 | 54.771 s | 214.879 s | 354 | 59/71 (83.10%) | 83.10% |

Gemma used 100% for 247 tokens and 25/50/75% for 11 tokens each; 12 tokens
were unassisted. The executed phone layer mask was 0-23, across three sessions.
Qwen used 100% for 59 tokens on layers 12-17 in HTP2; 12 tokens were
unassisted. These fractions apply to the resident FFN slices, not the entire
model. Llama used its unchanged GPU parent and performed no phone work.

All requests completed their full output lengths. There were no execution
recoveries/fallbacks, stale execution failures, or cleanup failures. Six planning
attempts for three requests are not six executions or request restarts. The
journal's `QUALIFIED_FALLBACK_NO_SAFE_ALTERNATIVE` is normal baseline selection,
not an execution failure. There are 557 direct helper lifecycle events.

Native logs retain `USE_GRAPHS=1`, CUDA graph warmup records, and graph-reuse
counters. The generic `graphs reused` counts are NOT CUDA recapture counts;
this run did not collect a new Nsight capture/replay count.

### Preparation and independent generations

| Publication | PREPARING | READY | Scheduler preparation | Phone load-authorized to READY |
| --- | ---: | ---: | ---: | ---: |
| Gemma HTP0/gen1 | 3.879 s | 23.623 s | 19.744 s | 10.277 s |
| Gemma HTP1/gen1 | 23.708 s | 39.669 s | 15.961 s | 15.414 s |
| Gemma HTP2/gen1 | 41.494 s | 54.122 s | 12.628 s | 11.000 s |
| Qwen HTP2/gen2 | 181.821 s | 196.982 s | 15.161 s | 14.789 s |

Times in the first three numeric columns are from the scheduler paid epoch;
the final column is a duration from the phone's own monotonic timestamps.
The first proposal was at 3.094 s; preparation began 0.784 s later and
overlapped desktop loading. First-session readiness was 23.623 s and all three
Gemma sessions were READY at 54.122 s, before the first generated token at
62.741 s. Desktop execution was dispatched before all three sessions were ready.

Physical loads were 1/1/2 for HTP0/HTP1/HTP2. The only replacement was Gemma to
Qwen on HTP2. Retained sessions kept their artifact and generation 1. Every
physical load used `weight_source=ffn_shard`; fractions caused no reloads.
Llama's index was verified but its shard was not loaded. Replacement happened
after Gemma completed, so this run does not newly prove retained *active* calls
during replacement or injected rollback; the focused regressions preserve those
previously working paths.

### Whole-fleet energy

The 311.667160 s paid interval includes runtime preparation and dynamic endpoint
cleanup. CPU package energy was 8.477833 kJ, GPU board energy 9.055545 kJ,
both physically sampled. Phone active time was 115.525990 s and idle time
196.141170 s; idle power is the unchanged assumed 0.875 W.

| Assumed phone active power | Fleet energy | Difference from previous adaptive run |
| --- | ---: | ---: |
| 3 W | 18.052 kJ | 18.84% lower |
| 4.5 W | 18.225 kJ | 18.65% lower |
| 6 W | 18.398 kJ | 18.46% lower |

The previous adaptive run was 316.882832 s and 22.402 kJ at 4.5 W. These are
historical diagnostic differences, NOT matched savings or proof of the new
start-early policy: that CPU route did not execute. The catalog, source manifest,
server binaries and libraries differ. Workload, artifacts, actual desktop
placements, initial evidence and paid-boundary definitions match in the decoded
comparison. The unchanged strict validator rejects a matched comparison with
the frozen desktop control: `A/B identity differs: catalog_sha256`. No baseline
was rerun. `SUMMARY-final.json` preserves the complete decoded differences.

## Remaining blockers, not fixed in this bounded pass

1. At Llama arrival (61.032 s), the independently measured CPU candidate was
   admitted with 1,877,336,573 host bytes and no GPU memory demand, but selection
   returned `BASELINE_ENERGY_EVIDENCE_NOT_QUALIFIED`. Admission is not permission
   to claim an energy improvement against an unqualified comparison. Subsequent
   epoch snapshots mark the non-selected CPU parent `MODEL_EPOCH_AUDIT_ONLY`.
   The exact desktop comparison/load evidence must be qualified, not bypassed.
2. Gemma's cold transition requires four CPU lanes and merges them into the
   execution plan's four-lane claim. All four remain held after the transition
   receipt (44.354 s), until request completion (179.712 s). At Llama arrival,
   both CPU and GPU candidates forecast a start at 484.780 s and a 423.780 s
   queue delay. This is a reservation forecast, not the observed wait. A robust
   start-early path needs separate preparation/execution resource lifetimes and
   fresh alternative costs when lanes are released; it must not simply lower
   resource claims while loading or ignore leases. Concurrent CPU interference
   also remains unmeasured by the isolated calibration.
3. Llama CPU/NPU candidates fail closed with `TRANSPORT_PROFILE_INCOMPLETE` for
   the current activation shape, in addition to layout/learning break-even
   conditions. The earlier standalone phone proof does not automatically
   provide this exact scheduler transport qualification. No larger-payload
   profile or GPU route qualification was copied to make it pass.
4. Persistent multi-model residency and stateful CPU-to-GPU handoff remain
   future work. Free memory alone is insufficient: session slots, exact runtime
   contracts, KV state, leases, and measured shared-resource contention matter.

## Changed files and artifacts

Production changes (relative to `research_dev/scheduler/`):

- `scheduler.py`: register a verified storage subset, reject unknown sessions.
- `adapters/catalog_materialization.py`, `adapters/__init__.py`: explicit measured
  CPU-parent capability and its existing helper families.
- `adapters/llama_server.py`: subprocess-scoped GPU isolation for CPU-only launch.
- `configuration/models.py`: paired overlay FFN index/directory configuration.
- `campaigns/burstgpt/arguments.py`, `launch.py`, `runner.py`, `preflight.py`:
  configuration, index verification/registration, and physical transport wiring.

Tests: new `tests/test_work_conserving_start.py`; updated
`test_campaign_inputs.py`, `test_cuda_reference.py`, `test_ffn_shards.py`,
`test_llama_server_adapter.py`, `test_offline_phone_residency.py`, and
`test_adaptive_runtime.py` (the latter repairs an outdated snapshot/time mock).
Report-only scripts are `experiment.py`, `calibrate_cpu.py`, `analyze.py`, and
`audit.py`; `research_dev/talks.md` records this result. No active adaptive
controller, phone session transaction, native source or worker was edited.

Post-run audit passed: all 381 deployed source-manifest files match, no experiment
server/phone worker remains, USB is restored to `ptp,adb` at 5 Gbps, and GDM
PID 6871 remains present with 3,178 MiB GPU usage and 12,770 MiB free.

| Artifact | SHA-256 |
| --- | --- |
| `physical/run/RESULT.json` | `7f08fc4fdd7e53ed20627c3672fee404b7bdffcc20ed6502f6e988d9f20696ec` |
| `physical/run/SCHEDULER_DECISION_LOG.json` | `3b4fe4f045c829fa2b9e7bd35a1a3641f357674392e0251d37089ba604d5bea7` |
| `physical/inputs-v3/SOURCE_MANIFEST.json` | `ca0ae9db48fc526714b2c94afb1ba3e9adda31071e6a90ae503ccd10befcf5bb` |
| `physical/inputs-v3/CATALOG.json` | `fddcb4126b183e1839c7d1737fa93e9250964996bdbda4a92bfd56b2024fa032` |
| `physical/preflight-v3/PHYSICAL_PREFLIGHT.json` | `248da87ad7ce7d1698d212d04744b9650de2d6fbe467a9ecabe59ad9e2fb1295` |

`SUMMARY.json` is a preserved first analysis attempt that tested an adaptive
reference as arm A. Use `SUMMARY-final.json`, which invokes the unchanged
validator against the actual frozen desktop baseline and retains the separate
historical adaptive comparison.
