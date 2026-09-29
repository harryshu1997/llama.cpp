# Telemetry recovery and bounded mixed gate

Status: telemetry recovery implemented and physically exercised; mixed gate
still FAIL. No 24-request comparison or savings claim is authorized by this
result. No commit, push, GDM stop, unrelated process termination, or recovery
USB reset was performed.

## Current physical result: V4

Artifact root: `/home/zhihao/s42-telemetry-recovery-20260906-v4-gate/run`.
Inputs and frozen source are in the adjacent `-v4-inputs` and `-v4-deploy`
directories. Failed V1-V3 artifacts remain unchanged.

| Measurement | V4 result |
| --- | ---: |
| Cold Qwen phone calls | 4,836 |
| Online Qwen phone calls | 3,720 |
| Online Qwen fraction-weighted coverage | 75.0% (255 / 340 eligible tokens) |
| Gemma execution, reverse, fault rollback, retry | Not reached in V4 |
| Retained HTP0 / HTP1 calls during forward load | 108 / 108 |
| Session generations after forward load | HTP0=1, HTP1=1, HTP2=2 |
| Physical loads, including initial preload | HTP0=1, HTP1=1, HTP2=2 |
| Phone terminal | status=0, calls=8,556, reset recoveries=0 |

The selected session was HTP2, derived by the scheduler. Neither retained
session was reloaded. There was no phone restart during replacement; normal
gate failure teardown subsequently closed the gate-owned service. Both Qwen
requests completed. The full multi-phase mixed gate did not complete.

| Timing | Measured duration |
| --- | ---: |
| First session usable, native load authorization to READY | 11.310 s |
| All sessions usable, first native authorization to last READY | 57.964 s |
| Per-session native cold load: HTP0 / HTP1 / HTP2 | 11.310 / 14.912 / 17.529 s |
| Controller offline preload interval | 59.488 s |
| Forward native load authorization to READY | 10.660 s |
| Drain request to safe boundary | 0.031465 s |
| Safe boundary to reduced-mask control issued | 0.756435 s |
| Control issued to applied acknowledgement/quiescence | 0.526053 s |
| Total drain request to quiescence | 1.313953 s |
| Quiescence to replacement loading | 0.703476 s |
| Host replacement loading to READY publication | 11.321783 s |

Absolute timestamps and per-phase load/verification/publication records are
in [V4_AUDIT.json](V4_AUDIT.json) and the physical `DRAIN_TIMELINE.json`.
Preparation-associated energy is 5.450 kJ: CPU package 3.646 kJ, GPU board
1.565 kJ, phone 0.239 kJ at the existing assumed 4.5 W. This attribution
includes overlapping desktop work; it is not an isolated marginal loading
cost and must not be added again to a full execution energy interval.
There is no new matched desktop baseline, so no energy saving is claimed.

## Telemetry proof

The old FunctionFS snapshot producer timestamped before a roughly 2.6-second
serial sysfs temperature scan. ADB disappears in FunctionFS mode, and the
old monitor could serialize refresh behind other endpoint probes. Missing
values became conservative admission sentinels that looked like measured
exhaustion or overheating.

Probe attempts now retain source, sample timestamp, checked timestamp, age,
validity and exact failure. Missing, stale, timed-out and malformed values
are explicit. Background probes refresh independently. Planning defers and
requests refresh without replacing the authoritative READY map. Physical
loading rechecks the source map, fresh health, actual free memory and exact
changed-session memory requirement before mutation.

The FunctionFS producer uses the existing supported Android HAL current
physical-temperature readings and sensor throttling status. Sysfs remains
fallback. The timestamp stays at the beginning of observation; the five-second
age limit, thermal rules and memory limits are unchanged. No worker/router
binary or FFN shard format changed.

V4 has 3,477 valid planning snapshots and 81 unavailable observations, with
no stale snapshots. All four physical admissions used valid samples aged
0.637, 2.907, 0.925 and 1.280 seconds. See
[V4_TELEMETRY_AUDIT.json](V4_TELEMETRY_AUDIT.json).

A separate read-only capture during execution recorded 207 valid readings
and eight HTTP timeouts. Two serving-time outages recovered through
FunctionFS HTTP in 3.012 and 3.380 seconds, measured from the first failed
poll to the next successful poll at a nominal one-second cadence. Maximum
valid sample age in that capture was 2.096 seconds. The last four timeouts
occurred during gate failure teardown and are not counted as serving
outages. Recovery used no USB reset or worker restart.

## Remaining physical blocker

The unchanged equivalent-call interruption metric observed an HTP1 gap of
260.311 ms between layers 7 and 8 at 100% fraction and retained mask 4095.
Its matched reference median was 16.785 ms, so the 2x bound was 33.570 ms.
The reference has 30 intervals (two before and 28 after loading under the
existing matched-class metric). This is an observed violation, not missing
reference evidence. HTP0 passed.

Native desktop timestamps isolate 243.966 ms before submission of USB
request 381. That request's RPC was 16.503 ms, including 14.944 ms reported
HTP compute. Stderr observation lag at the two neighboring requests was
approximately 0.07 and 0.11 ms above its normal clock offset. CUDA capture
count was zero. Thus neither an expired telemetry sample, a CUDA recapture,
nor a 260 ms measured HTP RPC explains this stall. It occurs in the desktop
pre-submission path; CPU execution, scheduling or locking remains unresolved.
Its overlap with final weight upload does not establish causality.

Unprivileged host profiling is disabled (`perf_event_paranoid=4`; ptrace
scope is 1). The next investigation needs scoped host profiling permission
or explicitly authorized minimal native timing instrumentation. The session
transaction, working graph-mode implementation and interruption threshold
were not changed speculatively. An unchanged rerun could pass by avoiding the
outlier but would not fix this blocker.

## Other gate-discovered causes fixed

- V1: measurement queried adaptive state after ATTACHED-at-zero but before
  registration. It now waits for the existing FRACTION_APPLIED event.
- V2: diagnostic historical baselines could trigger exploration but were
  filtered out when evaluating its paired results. The same existing
  operational eligibility rule now applies to current and historical
  records; qualification remains strict. V2 physically executed Gemma and
  restored the selected session to generation 4 after an injected reverse
  load failure, but did not complete the clean reverse retry.
- V3: during telemetry loss, the selected CPU desktop lacked dormant FFN
  startup metadata and its own baseline contract. Exact-parent zero-assistance
  metadata is now independent of phone admission. Fresh helper authorization
  remains strict. An unchanged missing parent contract is rejected once per
  configuration, not 62 times. This CPU-parent fix passes focused software
  tests; V4 stopped before exercising Gemma, so its physical proof is pending.

## Tests and replay

The final focused adaptive/helper/recovery/probe/monitor/replay set passed
108 tests in 84.459 seconds. An added static-overcapacity subcase passed
separately. Earlier transition/rollback/offline/replay coverage passed 122
tests; later adaptive-history tests passed 57 and campaign measurement tests
passed 24. These sets overlap and must not be summed as unique tests. No
whole-scheduler rerun was used after each edit. Details: [TESTS.json](TESTS.json).

Both replay goldens remain unchanged:

- v3: `f78d2b2c37a3880a523eba4f5315ada0207678c841d633229782bfa3a05c1829`
- v8: `965f218bb5f81a57dbb51167624798f02eb79354b6cc042683b2b1ed9b1a868d`

## Files changed for this task

Production paths under `research_dev/scheduler/`:

- `scheduler.py`
- `_internal/lifecycle.py`, `_internal/runtime_queue.py`, `_internal/runtime_capabilities.py`
- `_internal/adaptive_decode.py`, `_internal/route_generation/feasibility.py`
- `_unified/phone_residency.py`, `_unified/helper_preparation.py`
- `_unified/helper_envelopes.py`, `_unified/automated_selection.py`
- `adapters/probes.py`, `adapters/heterogeneous_rig.py`, `adapters/offline_phone_residency.py`
- `adapters/native/direct_phone_ffn_session.sh`

Measurement and tests:

- `campaigns/burstgpt/offline_residency_gate.py`
- `tests/test_runtime_queue.py`, `tests/test_telemetry_recovery.py`
- `tests/test_adaptive_decode.py`, `tests/test_adaptive_runtime.py`, `tests/test_burstgpt_replay.py`
- `research_dev/talks.md` and this report directory's JSON audits/documentation

Existing unrelated dirty-worktree changes, physical results and deployment
artifacts were preserved. The FFN shard generator, format and worker were not
modified. New per-version launch/deployment/measurement scripts are retained
with their inputs, outside the production scheduling path; they choose no
route, session, fraction or transition order.

## Exact V4 hashes

- FAILURE: `93ec3effb2422fb55cfbb5d6ea0bdc80ad0e0142b4e3c75bb7a69ff39665622e`
- Phone terminal receipts: `f13f023b7710cf30f88f2ed2ccae291dda34cdcacdc6188cfb73d0d0da39c16f`
- Continuous telemetry capture: `8d8778ad39ecff9101f1516d7bf70d4c370b2e7fc6782077b8a82112ca009e20`
- Preflight: `205f82ad3075c69ba9016e70f0f0197db7a97aba413b3b50e4116fed3856428b`
- Frozen execution source manifest: `612df889919962101264c176ce66f3ac9cf45fd4b1bb23e24f07301a47b65f47`
- Phone diagnostic script: `1118eb90cf341805c2d253566ce16a5abc044d6b984ff027124471a3c73d10e6`

Additional result hashes are in [V4_AUDIT.json](V4_AUDIT.json). The previous
failed runs are documented in `research_dev/talks.md` and the V1-V3 audits.
