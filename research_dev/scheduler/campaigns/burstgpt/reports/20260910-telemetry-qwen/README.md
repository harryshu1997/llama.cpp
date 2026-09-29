# Telemetry readiness and Qwen probe recovery

This attempt fixes the assumed-power startup prerequisite and the Qwen
measurement control path. It runs only the unchanged three-request
`burstgpt_dev3_long_v1.json` adaptive workload. No baseline or longer trace is
authorized here. Historical reference comparisons are not a fresh matched A/B.

## Root causes and scoped corrections

- Assumed phone power is calculated from activity intervals, but meter startup
  also required measured phone samples. Only assumed-power mode skips that
  prerequisite. Fresh measured CPU/GPU energy and independent phone health
  remain required; measured-phone mode retains its sample prerequisite.
- The restored phone clock is about ten seconds ahead of the desktop. The old
  cross-host timestamp subtraction rejected fresh FunctionFS snapshots. The
  existing phone HTTP server supplies a Date header from the same phone clock
  as the snapshot. Age now uses that difference, plus whole-second quantization,
  the complete request duration, and any HTTP Age. The five-second maximum is
  unchanged. Stale or malformed response clocks fail closed. No phone clock,
  kernel, native worker, or USB reset is changed by this fix.
- An unaffordable next coarse fraction exited before the measured leader's
  comparable-window verification. It now skips that unaffordable candidate and
  considers the leader using the existing qualification/uncertainty rules.
- Complete-pair admission now includes observed ACK latency immediately, and
  observed warmup latency separately from qualifying token measurements. An ACK
  leaving insufficient measurement time produces INCOMPLETE, not rejection or
  an extended reservation. Request-wide exploration caps remain intact.
- A failed live-slot query was reported as a membership change. Unknown context
  is now explicit, does not erase compatible evidence or attempt identities,
  and cannot qualify its windows. Fresh membership identities distinguish a
  real batch-1-to-1 task change from recovery of the same task. Temporary desktop
  execution and bounded retry use the existing controller and ACK path.
- Direct adaptive timing events are persisted in RESULT.json and a separate
  file, including on failure, to distinguish these cases in physical evidence.

No changes to shard format, binary, placement, fractions, qualification margin,
session fencing, lease validation, or terminal proof checks.

## Validation

Focused helper/adaptive, HTTP adapter, phone-power, telemetry, and replay tests:
260 passed in 96.506 seconds. Both replay goldens unchanged:

- v3: `ea5b30c9e4a3f705c194188d6e5788c64990953aad73ff3c4b7573bc69eca27d`
- v8: `965f218bb5f81a57dbb51167624798f02eb79354b6cc042683b2b1ed9b1a868d`

Two previous assertions now describe the corrected observation semantics:
ACK cost is available immediately rather than after a later window; an empty
slot response after terminal counters is unknown context, not proof of a new
member. Terminal counter and fencing assertions remain unchanged.

A further assertion requires same-membership recovery to resume an existing
winner without another probe attempt. After that narrow refinement, 210 adaptive
and HTTP tests passed in 12.575 seconds and both replay tests passed again in
74.475 seconds. These are subsets of the 260 tests, not additional unique tests.
Execution source hashes are in SOURCE_DELTA_EXECUTION.json; the original
preflight source hashes and manifest remain preserved separately.

## Physical result

Execution PASS: 3/3 requests, 290.020564 s including runtime preparation and
cleanup. The exact same dev3 requests, arrivals, outputs, initial evidence,
model/shard artifacts, native binaries, CUDA graph mode, and matched desktop
parents were retained. No baseline or longer trace was run. Host quiet checks
passed before preflight and execution; no compiler process was observed in 378
one-second host samples. GDM and other users' processes were not disturbed.

At the nominal 4.5 W phone assumption, CPU package energy was 14.119678 kJ,
GPU board energy 8.557606 kJ, and assumed phone energy 0.649820 kJ. The phone
activity union was 109.255782 s; 180.764783 s used the unchanged 0.875 W idle
assumption. CPU/GPU energy was measured physically. Phone energy was not.

| Reference | Fleet energy at 4.5 W (kJ) | Duration (s) | New adaptive saving vs reference |
|---|---:|---:|---:|
| Clean upstream desktop | 31.866 | 371.198 | 26.80% |
| Matched modified desktop | 31.699 | 355.145 | 26.41% |
| Fixed GGG | 22.492 | 312.197 | -3.71% |
| Fixed GGQ | 25.337 | 315.719 | 7.93% |
| Fixed GQQ | 31.963 | 331.500 | 27.02% |
| Fixed QQQ | 33.325 | 384.840 | 30.00% |
| This adaptive run | 23.327 | 290.021 | n/a |

These are final source-v7 **historical references**, not a fresh matched A/B.
The strict comparison validator still rejects a matched-A/B claim because the
source identity differs. The independent historical report validates the unchanged
inputs, evidence, artifacts, native binaries, desktop parents, phone identities,
and accounting configuration. All decoded differences are in COMPARISON.json.
Controller changes also affect fixed phone arms; this is not isolated proof of
dynamic-placement superiority. Fixed GGG remains lower-energy by 3.71%.

| Assumed active phone power | Adaptive fleet energy (kJ) | Saving vs matched desktop | Saving vs fixed GGG |
|---|---:|---:|---:|
| 3 W | 23.163 | 26.93% | -3.63% |
| 4.5 W | 23.327 | 26.41% | -3.71% |
| 6 W | 23.491 | 25.89% | -3.80% |

## Request behavior

| Request | Execution start/end (s) | Service latency (s) | Phone tokens/all tokens | Fraction-weighted all-token coverage | Phone calls |
|---|---|---:|---:|---:|---:|
| Gemma 36 | 20.017 / 152.012 | 131.994 | 280/292 (95.89%) | 90.24% | 6,632 |
| Llama 37 | 155.450 / 157.014 | 1.564 | 0/292 | 0% | 0 |
| Qwen 50 | 219.278 / 282.507 | 63.229 | 11/71 (15.49%) | 15.49% | 66 |

Arrival-to-completion latency is 151.012 s, 96.014 s, and 191.507 s respectively;
it includes scheduler queueing/preparation, unlike service latency above. No
request or output was truncated. Llama legitimately used its desktop route.

Gemma executed 247 tokens at 100%, 11 each at 75%, 50%, and 25%, and 12 at 0%.
Its initial 11 assisted tokens used the already READY HTP0/HTP1 subset, covering
CPU FFN layers 0-15. The later helper covered CPU layers 0-23. Widths were
3,840/7,680/11,520/15,360 columns. Calls were HTP0=2,240, HTP1=2,240, HTP2=2,152.
The earlier Gemma unassisted tail did not recur. Weighted eligible coverage is
90.55% over 291 eligible tokens; token coverage does not mean full-model offload.

Qwen's helper covered only CPU FFN layers 12-17, at 17,408 columns. It did not
offload all CPU layers or the complete model. Its 11 assisted tokens were at
100%; the other 60 were desktop. Eligible weighted coverage was 15.71% over
70 tokens. This is lower than the previous build-confounded run's 25.35%.

### Remaining Qwen limitation

The new branch reaches the promising leader's verification decision instead of
exiting at an unaffordable next coarse fraction. A valid measured baseline window
(tokens 5-9) used 86.056188 J/token and 715,789 us/token. The valid assisted
window (16-20) used 73.335000 J/token and 640,695 us/token: a diagnostic 14.78%
energy reduction and 0.895 latency ratio. One pair is not qualification.

The remaining bound-resolution block did not fit at token 20, with 51 output
tokens left. The exact exported outcome is `INCONCLUSIVE`, reason
`BOUND_RESOLUTION_UNAFFORDABLE`, not measured energy-negative rejection. Qwen
stayed at 0% rather than manufacturing a winner or extending exploration limits.
The control ACK costs were 2.163505 s entering assistance and 3.133769 s returning
to baseline; measured warmup was 664,551 us/token. These costs are now visible
and feed later admission. This small run therefore does not establish sustained
qualified Qwen assistance. The next performance work is reducing control and
measurement overhead or accumulating valid compatible evidence within the same
limits, not forcing 100% or weakening uncertainty bounds.

## Telemetry and lifecycle

- 946/947 recorded health snapshots were VALID, including 944 using
  `phone-http-date`; there were zero STALE clock rejections. Valid observed ages
  ranged from 0.284475 to 2.515558 s, below the unchanged 5 s limit.
- One unavailable observation occurred during the normal ADB-to-FunctionFS
  handoff at 19.714865 s. The next saved valid observation was at 23.685886 s,
  giving a sampled recovery interval of 3.971021 s, not a continuous outage
  measurement. No reset/restart was used for recovery.
- Two end-of-request slot observations were unavailable (Gemma's slot already
  empty, Qwen's query timed out). Neither was fabricated into batch 1 -> 1.
  There were zero actual `DECODE_CONTEXT_CHANGED` events. The terminal windows
  remain accounted but cannot qualify evidence; exact terminal proofs passed.
- RESULT.json contains 508 direct helper events and 369 adaptive timing events.
  A separate adaptive-timing-events.json is also saved for failure diagnosis.

| Session load | Generation | Scheduler LOADING/READY observed (s) | Physical load-to-READY (s) | Physical weight read (s) |
|---|---:|---|---:|---:|
| Initial Gemma HTP0 | 1 | 5.614 / 23.604 | 9.902 | 8.731 |
| Initial Gemma HTP1 | 1 | 23.882 / 33.602 | 9.285 | 8.174 |
| Initial Gemma HTP2 | 1 | 34.123 / 44.411 | 9.950 | 8.659 |
| Replace HTP2 with Qwen | 2 | 153.546 / 179.969 | 25.958 | 24.578 |

The first demanded layout was proposed at 4.811950 s and preparation began
0.802526 s later. Desktop execution started at 20.017491 s, before the first
session's READY observation. Actual scheduler publication timestamps were
23.789217, 33.911516, 44.884407, and 180.251272 s. Physical durations above
use same-phone monotonic clocks, not subtraction across skewed wall clocks.
HTP initialization was 0.095-0.188 s and weight upload 0.863-1.067 s; the long
Qwen replacement is predominantly in the existing weight-read phase.

Gemma's first positive ACK was at 41.716545 s on HTP0/HTP1 while HTP2 still
loaded. Same-phone phase/call timestamps contain four call observations on each
retained session inside that initial HTP2 load interval. There is no all-session
attachment barrier. Generations finish at 1/1/2. Only HTP2 was replaced; the
retained sessions' artifacts, generations, leases and historical proofs were
not globally invalidated. Four loads succeeded, zero failed; no helper errors,
fallback, stale execution, USB reset or request execution recovery was recorded.

Replacement happened after Gemma completed. Consequently this workload is not
a new retained-active-request interruption test during replacement, nor a new
reverse/rollback fault gate. Those paths were not rebuilt or claimed as rerun.
The actual FFN shard files and all generation-keyed proofs are preserved. Final
resident weights total 8,870,952,960 bytes, admitted with the existing workspace
and memory checks. USB returned to ptp,adb, the qualified kernel remains active,
and GPU usage returned to 3,178 MiB with GDM untouched.

CUDA graph evidence is physical: 77 captures, 52 instantiations, 77 executable
updates, 2,387 launches, and 25 recaptures by the existing metric. All API return
values are successful. No graph-disabled reference is labeled default llama.cpp.

## Files and immutable evidence

Production files in the deployment delta:

- `_internal/adaptive_decode.py`, `_internal/adaptive_decode_contracts.py`,
  `_internal/adaptive_decode_state.py`
- `_internal/adaptive_decode_ops/admission.py`, `budgeting.py`, `helpers.py`,
  `promotion.py`, `reporting.py`, `sequencing.py`, `windows.py`
- `_unified/adaptive_decode_control.py`
- `adapters/energy.py`, `probes.py`, `http_backend.py`, `heterogeneous_rig.py`
- `campaigns/burstgpt/runner.py` (timing persistence only)

Tests: `test_adaptive_decode.py`, `test_llama_server_adapter.py`,
`test_phone_power_probe.py`, `test_sustained_assistance.py`,
`test_telemetry_recovery.py`. The energy adapter and its power-readiness tests
were the preceding fix, included here in the deployed delta. This report,
configuration/measurement wrapper, analysis files and research_dev/talks.md are
also updated. No unrelated source, native binary, baseline, or old result changed.

- Remote run: `/mnt/storage/s42-telemetry-qwen-20260910-v1/`
- Deployment: `/mnt/storage/s42-telemetry-qwen-20260910-v1-deploy/`
- Local full copy: `physical/` (1,943 files, hashes match the remote tree).
- RESULT.json SHA-256:
  `8ce2a096aed9a4a56d9ee3c34a6150159c235a7da59c13b06145ae4b54e972f3`
- Execution SOURCE_MANIFEST.json file SHA-256:
  `4b65600d33dabb9ba87db8fad1183e92822bf0e34b87c26cede76bb6fcebdc17`
- Canonical physical inventory SHA-256:
  `7e41f9fb4d4c64be83620bb259599191351add5a020ee73ef6af5b46fb33332c`

ARTIFACTS.json lists every physical path, size and SHA-256. SOURCE_DELTA_EXECUTION.json
lists every deployed source change and before/after hashes. DIAGNOSIS.json and
COMPARISON.json retain the decoded evidence and compatibility checks. Previous
failed v2/v3 artifacts remain untouched. Nothing committed, pushed, or submitted.
