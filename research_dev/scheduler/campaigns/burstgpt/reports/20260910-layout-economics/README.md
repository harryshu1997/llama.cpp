# Layout-wide replacement economics: dev3 result

Execution PASS (3/3). Performance FAIL_CONFOUNDED. The Gemma unassisted tail is
gone, but this run does not establish an energy improvement. No longer trace or
baseline rerun was started.

## Changes and validation

The existing placement objective now considers the requests affected by the
whole target contract, including a running helper on retained shards. A read-only
preview distinguishes exact-compatible evidence reuse from a real layer-contract
change. A changed contract must afford a complete verification pair after the
estimated load interval, while leaving exploitation opportunity. The estimate
uses observed work and cadence, not knowledge of future arrivals or a required
request completion time.

Removed-session incumbent benefit remains in the existing per-session objective.
Only retained-session assistance lost during verification and incremental
verification overhead are added, once per affected request. Rough compute units
are not treated as joules: a replacement with nonzero measured revalidation cost
and an unknown energy benefit is deferred. An already-issued maintenance drain
keeps its acknowledgement transaction. Physical degradation still takes priority
over economic preference.

The current demand calculation already includes running remaining tokens and
already-arrived queued work. A regression covers running demand with an empty
waiting queue. Waiting for demand to drain is not an unconditional policy rule:
a higher measured net benefit can still authorize an affordable replacement.

The fraction controller now checks whether its leading candidate qualifies before
starting refinement. An unresolved promising candidate can reserve enough normal,
comparable windows to reach the next useful integer-square uncertainty step. The
tests include four baseline and four candidate windows, not a subdivision of one
measurement. If the required block and useful exploitation do not fit, the result
is INCONCLUSIVE. Missing/unfinished measurements remain INCOMPLETE; valid negative
evidence still rejects the candidate. The existing heuristic uncertainty formula,
latency limits, request-wide exploration cap, and identity checks are unchanged.

Validation command:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_dev/scheduler/tests:. python3 -m unittest test_layout_revalidation_economics test_sustained_assistance test_adaptive_decode test_model_placement_controller test_adaptive_runtime test_late_helper_energy_policy test_session_cow_transaction test_replay_determinism -q
```

303 tests passed in 87.849 s. This was the focused set, not the complete harness.
Both replay goldens and repeated replay bytes are unchanged:

```text
v3 ea5b30c9e4a3f705c194188d6e5788c64990953aad73ff3c4b7573bc69eca27d
v8 965f218bb5f81a57dbb51167624798f02eb79354b6cc042683b2b1ed9b1a868d
```

Additional checks covered the previous Gemma near-completion case, aggregate
request costs without double-counting, exact-compatible reuse, affordable
replacement, unknown energy units, safety override, and maintenance preservation.
No new subsystem, native change, model-specific assignment, forced fraction,
hysteresis change, or physical transaction rebuild was introduced.

## Experiment and comparison conditions

One adaptive `burstgpt_dev3_long_v1.json` run used the unchanged requests:
Gemma 36 at 1 s with 292 output tokens, Llama 37 at 61 s with 292 tokens,
and Qwen 50 at 91 s with 71 tokens. Normal scheduling selected the assignments
and fractions; there were no future-demand hints. Runtime startup, preparation,
execution and cleanup remain inside the paid interval.

All six references are the final `references-source-v7` results from
`baselines/cuda_graph_v1/COMPARISON.json`. None was rerun or overwritten.
The matched-runtime artifacts, native binaries, exact desktop parents, initial
evidence, graph mode, inputs and accounting configuration are unchanged.
The new Python source manifest differs. The existing strict A/B validator still
rejects this as `A/B source or binary identity differs`; it was not relaxed.
Decoded comparison differences are in [COMPARISON.json](COMPARISON.json).

The clean upstream desktop additionally has the expected upstream binary,
catalog and parent-qualification identity differences. It is not the modified
runtime, and the modified runtime is not labeled default llama.cpp. These are
historical-reference comparisons, not fresh matched A/B experiments.

Preflight passed once using unchanged native qualification. Final focused fixes
were synchronized before execution; the original preflight manifest and commands
were preserved alongside separate `SOURCE_MANIFEST_EXECUTION.json` and execution
command files. No production code changed during the physical run.

CUDA graphs actually executed: 90 captures, 66 instantiations, 90 executable
updates, and 2,374 launches, all with successful API return values. The established
recapture metric is 24 updates beyond initial instantiations. No graph-disabling
environment variable or new native binary was used.

## Physical outcome and coverage

| Metric | Previous adaptive | This adaptive |
|---|---:|---:|
| Requests completed | 3/3 | 3/3 |
| Duration including preparation/cleanup | 391.702 s | 345.347 s |
| Gemma tokens with any assistance | 254/292 (86.99%) | 280/292 (95.89%) |
| Gemma fraction-weighted all-token coverage | 81.34% | 90.24% |
| Qwen tokens with any assistance | 11/71 (15.49%) | 18/71 (25.35%) |
| Qwen fraction-weighted all-token coverage | 15.49% | 25.35% |
| Gemma phone calls | 5,720 | 6,720 |
| Qwen phone calls | 66 | 108 |
| Raw fleet energy at 4.5 W phone power | 27.292 kJ | 42.900 kJ |

The previous result is `20260910-context-continuity`; it is preserved unchanged.
Coverage uses physical token-position evidence, not the count of control messages.
Weighted coverage is the sum of the executed split fractions divided by all
decode tokens. Against the eligible-token denominator it is 90.55% for Gemma
(291 tokens) and 25.71% for Qwen (70 tokens).

Gemma executed 247 tokens at 100%, 11 each at 75%, 50% and 25%, and 12 at 0%.
Every assisted token used the same CPU-resident layers 0-23 under the exact
desktop parent. Each HTP session produced 2,240 generation-1 calls. Full-width
assistance continued to completion; the previous 26-token unassisted tail did
not recur. Its initial 12 unassisted tokens are baseline/control work, not a new
replacement tail.

Qwen executed 18 tokens at 100% of its resident shard width and 53 at 0%.
Its shard covers only layers 12-17, width 17,408, so this is not full-model FFN
coverage. All 108 calls use HTP2 generation 2. Llama stayed on its unchanged
desktop-only route with no supported FFN helper.

| Request | Queue wait | Desktop execution interval | Service latency | Arrival-to-completion |
|---|---:|---:|---:|---:|
| Gemma 36 | 0.790 s | 50.987-185.074 s | 134.088 s | 184.074 s |
| Llama 37 | 94.148 s | 189.254-191.360 s | 2.106 s | 130.360 s |
| Qwen 50 | 100.403 s | 255.825-337.918 s | 82.093 s | 246.918 s |

Queue wait is the sum of dispatch waiting for that request, not all time from
arrival to first token. Submit/replan decision measurements total 1.527 s; this
does not include every decode-boundary controller operation.

## Residency and replacement

Times below are scheduler timestamps relative to the paid boundary. They include
host-side transaction work and are distinct from the phone's native read/init
timestamps in `session_physical_phases` in COMPARISON.json.

| Layout view | Changed session | Generation | PROPOSED | PREPARING | READY |
|---|---|---:|---:|---:|---:|
| G | HTP0 | 1 | 4.727 s | 5.518 s | 25.831 s |
| GG | HTP1 | 1 | 25.831 s | 25.975 s | 42.920 s |
| GGG | HTP2 | 1 | 42.920 s | 43.220 s | 54.304 s |
| GGQ | HTP2 | 2 | 186.378 s | 186.378 s | 203.646 s |

HTP2 was selected dynamically, not forced. All three initial sessions publish
individually. Gemma attaches to the first two READY sessions at 50.504 s,
starts desktop execution at 50.987 s, and refreshes to all three at 54.648 s.
The run therefore exercises subset attachment before all sessions are READY;
it does not exercise actual FFN calls during that last initial load because
prefill and the first baseline window still precede phone execution.

The layout remained GGG until Gemma completed at 185.074 s. This directly
exercises the new running-demand/retained-revalidation protection. The subsequent
GGQ replacement takes 17.268 s from scheduler PREPARING to READY. HTP0/HTP1
remain Gemma generation 1; only HTP2 changes to Qwen generation 2.
The Qwen shard is READY before Qwen desktop execution begins.

There are four successful per-session loads: three initial loads and one
replacement. Load counts are HTP0=1, HTP1=1, HTP2=2. There are zero failed or
unprepared layout proposals. The 173 `REVALIDATION_UNAFFORDABLE` and 62
`REVALIDATION_ENERGY_UNKNOWN` evaluations are economic deferrals, not attempted
physical transitions. Issued maintenance transactions are not canceled.

The final resident weight bytes are 2,831,155,200 each for Gemma HTP0/HTP1 and
3,208,642,560 for Qwen HTP2, totaling 8,870,952,960 before workspace. Memory
admission remained strict. Actual FFN shard files, paths, parent hashes, geometry,
operator plans and per-generation terminal proofs are persisted in RESULT.json.
Weight source is `ffn_shard`; no full-GGUF fallback was used.

All requests have passing terminal and semantic sanity checks. No fallback, USB
reset, stale-generation failure, or request recovery is recorded. Cleanup
restored USB successfully; the before/after GPU reading was 3,179 MiB used and
12,770 MiB free, with GDM untouched. Final resident state in RESULT.json is
captured before cleanup and must not be read as a leaked post-cleanup service.

Because Gemma finished before replacement, this run does not supply a new
retained-session inter-call-gap test during replacement. Reverse replacement and
rollback were not injected in this workload. Their existing focused regressions
and earlier physical artifacts remain valid; they are not claimed as newly run.

## Loading diagnostics

Read-only sampling of the campaign's own server PIDs records `/proc` I/O counters
at 100 ms intervals, with process discovery within 500 ms. Existing native logs
and CUDA copy events separate initialization, tensor loading, copies and READY
work. No cache flush, prefetch, altered preload policy or interference with other
processes was performed. [LOADING_DIAGNOSTICS.json](LOADING_DIAGNOSTICS.json)
contains the marker line numbers and exact durations.

| Desktop phase | Gemma | Llama | Qwen |
|---|---:|---:|---:|
| Metadata/model initialization | 0.897 s | 0.629 s | 0.540 s |
| Tensor read/repack/transfer span | 39.745 s | 0.476 s | 47.140 s |
| Context initialization | 0.132 s | 0.007 s | 0.192 s |
| Native warmup through server READY | 1.079 s | 0.073 s | 8.292 s |
| Other spawn/post-READY host work | 1.138 s | 1.318 s | 3.403 s |
| Total desktop transition | 43.390 s | 2.999 s | 60.070 s |
| GPU H2D active interval union, overlapping above | 7.384 s | 0.445 s | 8.020 s |
| Sampled physical storage reads | 8.621 GB | 0.002 GB | 35.238 GB |

Storage activity bins span 33.828 s for Gemma and 48.701 s for Qwen; these are
not measurements of blocked I/O time. Storage, CPU repacking and GPU copies
overlap. The physical byte counters can include mapped pages, rereads and
dependencies, not only model weights. The native warmup and host remainder do
not isolate every proof/health-check duration individually; no unmeasured
subphase is invented.

The total desktop transition time is 106.459 s, versus 167.094 s in the previous
adaptive run. That reduction does not by itself attribute the earlier 70.411 s
gap against fixed GGG to a scheduler bug. Page-cache and background conditions
were not controlled, and overlapping preparation is not free preparation.

## Raw energy and historical references

CPU package and GPU board energy are physically measured; phone power remains
assumed. Startup/preparation/cleanup and all exploration remain included. No
overlapping interval is subtracted to manufacture steady-state energy.

| Arm | Duration | Fleet energy at 4.5 W | New adaptive saving versus arm |
|---|---:|---:|---:|
| Clean upstream desktop-default-cuda | 371.198 s | 31.866 kJ | -34.63% |
| Modified desktop-matched-cuda | 355.145 s | 31.699 kJ | -35.33% |
| Fixed GGG | 312.197 s | 22.492 kJ | -90.74% |
| Fixed GGQ | 315.719 s | 25.337 kJ | -69.32% |
| Fixed GQQ | 331.500 s | 31.963 kJ | -34.22% |
| Fixed QQQ | 384.840 s | 33.325 kJ | -28.73% |
| This adaptive | 345.347 s | 42.900 kJ | n/a |

| Assumed active phone power | New fleet energy | Saving versus matched desktop | Saving versus fixed GGG |
|---|---:|---:|---:|
| 3 W | 42.733 kJ | -34.81% | -91.18% |
| 4.5 W | 42.900 kJ | -35.33% | -90.74% |
| 6 W | 43.067 kJ | -35.86% | -90.30% |

The same 0.875 W phone-idle treatment is retained. Active phone time is
111.578 s and idle time 233.769 s. Nominal domain totals are CPU 32.318 kJ,
GPU 9.875 kJ, phone 0.707 kJ. Previously they were 16.549, 10.035 and 0.709 kJ.

The CUDA capture contains unrelated compiler activity: 255 `cc1plus`, 128
`nvcc`, 125 `cicc`, six `cmake` and 18 `gmake` processes. The earlier adaptive
and fixed-GGG captures contain none of these. A concurrent external build was
also visible on the desktop after this gate; this experiment only used prebuilt
binaries. No unrelated process was stopped or modified.

RAPL package energy includes other CPU work, so this is a confounded performance
comparison. The exact compiler joules and its latency impact are not isolated;
the entire increase is not attributed to that build. Raw measured totals remain
unchanged. There is no energy win, no 25% claim, and no basis to promote adaptive
over the best historical fixed GGG result. The narrow correctness fixes are not
blindly rolled back from an unisolated whole-machine measurement.

## Remaining problems

Qwen still does not acquire a qualified incumbent. Its first valid candidate
pair has about 7.84% lower mean energy than baseline, but the uncertainty bound
does not admit it. The existing coarse-probe budget stop occurs before the new
multi-window resolution stage is reached. A later bounded retry at token 47
has 4.327 s of allowance; acknowledgement and warmup finish after its deadline,
so it is INCOMPLETE, not energy-negative.

Qwen also records membership changes with numerical batch 1 -> 1, clearing its
current-context measurements. The exported zero-policy reason counts are six
CONTEXT_CHANGED, four INSUFFICIENT_OPPORTUNITY, two PROBE_INCOMPLETE and one
INITIAL_BASELINE decisions. These are event counts, not token or duration
denominators. Request-wide probing remains bounded; no winner is fabricated.

The new comparable-measurement block is covered by focused tests, but the Qwen
physical case above does not validate it end-to-end. The next narrow diagnosis
is that earlier coarse-budget exit and the live membership identity updates,
plus a non-interfering quiet-host check before any future energy measurement.
This report does not claim that every assistance or performance issue is fixed.

## Files and integrity

Production files changed in this task, all under `research_dev/scheduler`:

```text
_internal/model_placement_contracts/layout.py
_internal/model_placement_controller.py
_internal/model_placement_ops/economics.py
_internal/adaptive_decode.py
_internal/adaptive_decode_state.py
_internal/adaptive_decode_ops/budgeting.py
_internal/adaptive_decode_ops/promotion.py
_internal/adaptive_decode_ops/reporting.py
_internal/adaptive_decode_ops/sequencing.py
_unified/phone_residency.py
_unified/phone_residency_ops/common.py
_unified/phone_residency_ops/economics.py
_unified/phone_residency_ops/portfolio.py
_unified/helper_preparation_ops/start.py
```

Tests changed: `test_adaptive_decode.py`, `test_sustained_assistance.py`,
`test_session_cow_transaction.py`, and new
`test_layout_revalidation_economics.py`. [CODE_CHANGES.json](CODE_CHANGES.json)
records before/after hashes relative to the current task's backup, not the
much larger pre-existing dirty worktree. The focused stale-authorization fixture
bypasses the independent new economic precheck so it still tests stale physical
identity, without weakening that assertion. No adapter, wire format, native
generator or `route_generation.py` changes were made.

Report additions: this README, `experiment.py` (configuration/launch and read-only
measurement only), `loading.py`, `finalize_report.py`, `COMPARISON.json`,
`LOADING_DIAGNOSTICS.json`, `GATE_DECISION.json`, `ARTIFACTS.json`, and
`CODE_CHANGES.json`. The project log is updated in `research_dev/talks.md`.
Existing comparison code was reused, not altered.

Physical artifacts: `/mnt/storage/s42-layout-economics-20260910-v1/` on the
desktop, with the full 761-file copy under [physical](physical).
Deployment: `/mnt/storage/s42-layout-economics-20260910-v1-deploy/`.
Pre-change task backup: `/tmp/s42-layout-economics.BEN85g/`.
No archived artifact was overwritten, and nothing was committed or pushed.

Selected SHA-256 values (bare hex; full physical index in
[ARTIFACTS.json](ARTIFACTS.json)):

```text
run/RESULT.json
de15b1a7507997f4d7563066565a35a794954a9d1b3abb6be0e383a4b86db324
inputs/SOURCE_MANIFEST_EXECUTION.json
2a007a2883b9caa48e5c86a043ec6f6fa79b68070353cf8424ce489c1635940e
preflight/PHYSICAL_PREFLIGHT.json
2fa5994b29f815d27988904a5a1c530ec1b86619eba4af32b7110099a50d7208
CUDA.sqlite
a3b67a63ffc8f5e9d3d729bbb2cf12deb86aa272b418c4a0e1e2436a6492e8e5
LOADING_PROCESS_SAMPLES.jsonl
78d8d5e17c2ecdab0145a4bc00c6a8df9b3eb7ef9c0debf81fff8a862753a9a7
COMPARISON.json
6a61f914c8f9945cf50a03da50559d74e5fcaf0c233e242c99ebc3e7c9c8d6dd
LOADING_DIAGNOSTICS.json
e93da828e4f5452709305fe8cd4fc83062373950dfb72f1053821aed4c607425
```
