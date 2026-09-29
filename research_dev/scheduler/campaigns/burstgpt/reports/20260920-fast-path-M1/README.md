# Fast-path M1: split and transfer inventory

Status: **PASS - amended M1 check, uncapped calibration and held-out arms**.
Both held-out predictions are within 10%, and all 64 tokens match at each
ubatch. Proceed to M2. M0's 11.97% saving remains a single-pair result; its three
alternating repeats stay scheduled alongside M5's baseline runs. M3 still needs
OP12 physically moved to the desktop.

The model, server and library hashes, request, launch settings and KV plan match
the original M1 arms. The same model content is read from
`/mnt/storage/Qwen3-14B-Q4KM-dequant-f16.gguf`; no server rebuild. New records are
in `physical/no-pressure/`. The original failed 18 GiB check and its command
sources remain preserved in `physical/` and `PRESSURE_LIMITED_COMMANDS.json`.

The user paused during traced-128; it finished and cleaned up while its launch
driver remained stopped. The driver resumed at 17:50:17 UTC after runtime hash
verification. No later arm launched during the pause (`PAUSED.json`, `RESUMED.json`).

## Uncapped acceptance result

The unchanged frozen checker returned exit 0 at 2026-09-20 18:15:46.730577 UTC
(`physical/no-pressure/CHECK_M1.json`, `check.log`, `check.exit`). Both held-out
predictions pass the 10% limit, with exact output tokens. Single runs throughout.

| ubatch | Frozen prediction s | Held-out prefill s | Absolute relative error | Tokens identical | Check |
| ---: | ---: | ---: | ---: | --- | --- |
| 128 | 291.318576 | 296.522748632 | 1.7550669066738758% | 64/64 | PASS |
| 1024 | 111.980349 | 113.784898427 | 1.5859305162167252% | 64/64 | PASS |

| Arm, single run | Prefill host J, measured | Request host J, measured | Decode ms/token | Phone W, assumed | Request phone J, assumed |
| --- | ---: | ---: | ---: | ---: | ---: |
| traced-128 | 30529.421736886812 | 36823.29911685669 | 782.900515625 | 0.875 idle | 300.126485155 |
| traced-1024 | 16331.824856169456 | 22494.619715904548 | 776.543546875 | 0.875 idle | 142.55258134075 |
| heldout-128 | 31233.436394658922 | 37431.483731909655 | 778.13453125 | 0.875 idle | 303.0371492925 |
| heldout-1024 | 16821.033381750723 | 23117.855078846358 | 775.8648125 | 0.875 idle | 143.01547419337498 |

Host energy is measured RAPL package plus NVML board. The phone executes no
FFN in M1; its idle-power assumption is separate. No measured-plus-assumed sum
is presented as measured energy. Model load and warmup are outside prefill.

| Arm | memory.events.max ready | Finish | Delta | memory.peak B |
| --- | ---: | ---: | ---: | ---: |
| traced-128 | 0 | 0 | 0 | 30877900800 |
| traced-1024 | 0 | 0 | 0 | 30952939520 |
| heldout-128 | 0 | 0 | 0 | 30925471744 |
| heldout-1024 | 0 | 0 | 0 | 31062175744 |

All scopes were uncapped with swap disabled, and all high/OOM/OOM-kill events
were zero. Global page scanning and file faults were not zero: request major-fault
deltas were 211 / 127 / 131 / 132 in table order, versus 13,121 / 7,399 / 24,188 /
24,232 under the old cap. `ARM_SUMMARY.json` preserves request and sampled prefill
stat deltas; removing cgroup-cap pressure is not a claim of zero paging activity.

Weight-copy host-call time falls 154.782345 s,
84.70185285990898% of the 182.737850205 s held-out prefill difference.
This identifies repeated weight streaming as the dominant difference. The two
traced accounting errors are 0.5536643176797519% and 1.0895660102371686%. One
traced/untraced pair per ubatch does not isolate instrumentation overhead from
run variation. No diagnostic repeat or prediction refit was needed.

At 18:16:30 UTC all four scopes were inactive, no owned server remained, the
rig lock was free, and OP15 was visible on ADB 5037 (`CLEANUP.json`). The same
transport identity and native runtime hashes were retained. Raw records, traces,
inventories and source snapshots are in `physical/no-pressure/`.

## Uncapped calibration

Fresh scopes use `MemoryMax=infinity` and `MemorySwapMax=0`, retaining isolated
memory counters. The same model is read from `/mnt/storage`; native hashes,
request tokens, launch settings and KV plans match the original M1 arms.
The checker requires zero `memory.events.max` and `memory.events.high` at both
ready and finish. Single runs throughout. No native rebuild or prediction fit.

Predictions frozen at 2026-09-20T17:58:01.543261+00:00 in
`physical/no-pressure/FROZEN_PREDICTIONS.json`. The repeated rig suite passed
73 tests in 5.963 s; pyflakes was clean before held-out launches.

| ubatch | Copy host s, including waits | Compute host-call s | Normal completion wait s | Frozen sum s | Traced prefill s | Weight-copy bytes | Weight-copy host s, excluding preceding waits |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 188.516451 | 102.776075 | 0.02605 | 291.318576 | 292.940482926 | 1255225267200 | 178.576956 |
| 1024 | 29.055967 | 82.820162 | 0.10422 | 111.980349 | 113.213889054 | 165161216000 | 23.794611 |

GPU host-call time is not device compute. Existing completion waits remain
separate, and mixed layer/family groups retain joint timings. The complete
ubatch sequence includes the final 9-token / 521-token remainder.

Exact commands: `physical/no-pressure/COMMANDS.txt`; script sources and hashes:
`physical/no-pressure/COMMAND_SOURCES.json`. The optional diagnostic command
is only for a failed 1024 held-out check and cannot replace that result.

## Original 18 GiB acceptance result

Predictions were frozen at 16:20:59.553197 UTC, before held-out launches. The
unchanged checker returned exit 1 at 16:35:59 UTC (`physical/CHECK_M1.json`,
`physical/check.log`, `physical/check.exit`). Single runs throughout.

| ubatch | Frozen prediction s | Traced prefill s | Held-out prefill s | Held-out absolute relative error | Tokens identical | Check |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 128 | 293.118023 | 294.798142643 | 321.684526758 | 8.880285304953539% | 64/64 | PASS |
| 1024 | 108.788073 | 110.039417596 | 143.271303902 | 24.06848403193644% | 64/64 | FAIL |

The split/copy sums explain the traced prefills within 0.5699220585099254% and
1.1371784977945002%. They do not predict both held-out runs within 10%. The
observed traced/untraced duration changes are -8.357997316801735% and
-23.195074938894367%; one pair per ubatch cannot isolate instrumentation overhead
from run variation. The amendment identifies memory-cap reclaim as a fixture
confound; the no-pressure recheck tests the prediction without that cap.

| Original arm | memory.events.max at ready | At finish | Request delta | memory.peak B |
| --- | ---: | ---: | ---: | ---: |
| Traced 128 | 4084 | 4586 | 502 | 19327352832 |
| Traced 1024 | 6287 | 6702 | 415 | 19327352832 |
| Held-out 128 | 4614 | 6713 | 2099 | 19327352832 |
| Held-out 1024 | 4049 | 6149 | 2100 | 19327352832 |

Weight-copy host-call time drops 160.918028 s between calibrations, versus a
178.413222856 s prefill difference between held-out arms (90.19400323813407%).
This identifies repeated weight streaming as the dominant difference, while
leaving the absolute 1024 timing prediction unqualified.

| Arm, single run | Prefill host J, measured | Request host J, measured | Decode ms/token | Phone W, assumed | Request phone J, assumed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Traced 128 | 30632.62391998164 | 36836.34979130128 | 778.456 | 0.875 idle | 301.54206854625 |
| Traced 1024 | 16779.772341230942 | 23034.08633601586 | 779.215171875 | 0.875 idle | 139.921546814875 |
| Held-out 128 | 31713.703344718502 | 37925.048007494195 | 779.54275 | 0.875 idle | 325.08877604087496 |
| Held-out 1024 | 18093.66540323799 | 24334.87074173741 | 773.740265625 | 0.875 idle | 168.692844159875 |

Host values are RAPL package plus NVML board. Phone assumptions remain separate.
`physical/MATCHED_SETTINGS.json` verifies complete launch-contract equality apart
from tracing, identical KV plans, request tokens/settings and runtime hashes at
each ubatch. Every arm has zero OOM/OOM-kill events and swap disabled. Both held-out
scope peaks are exactly 19,327,352,832 B. Full records, per-split JSONL, full
inventories and representative prefill/decode graphs are in `physical/`.

At 16:36:37 UTC all four scopes were inactive, no owned server remained, the rig
lock was free and OP15 was visible on ADB 5037 (`physical/CLEANUP.json`). No
foreign process was changed. No commit or push was made.

## Implementation

The existing `ggml_backend_sched_compute_splits` emits JSONL only when
`GGML_SCHED_TRACE` is set. Records identify the scheduler/graph/split, backend,
operator names and shapes, actual input-copy ranges and bytes, and host times.
Copy waits are separate from the copy calls. Existing scheduler synchronization
boundaries emit completion-wait records; tracing adds no device synchronization.
GPU enqueue time is not reported as device compute time. Mixed layer/family
groups keep joint split timing; the reader does not invent per-node shares.
The untraced template specialization performs no added timing or record
allocation. Tracing rejects pipeline-parallel schedulers and unusable output
paths. An incomplete or inconsistent trace cannot become an inventory.

`scheduler_trace_path` is an optional typed field in the existing server launch
contract, defaulting to `None`. The launcher clears inherited trace settings,
requires a native capability confirmation when enabled and prevents reuse across
trace settings. The setting enters the measurement's runtime identity and atlas
environment digest. `read_scheduler_inventory` extends the existing
`_internal/profile_materializer.py`; it retains per-device, layer/family groups
and individual weight-copy records. Phase filtering here assumes the checked
single-slot request: multi-token graphs are prefill and one-token graphs are
decode. It is not a phase detector for concurrent slots. No replacement scheduler
or cost model is introduced. `IMPLEMENTATION.patch` is relative to the shared tree immediately
before these edits, not to the branch's clean base.

## Original 18 GiB rig and method

Deploy: `/mnt/storage/s42-fast-path-M1-20260920-T3ypai` on
`zhihao@172.20.74.85`. Fresh CUDA build, 16 GPU layers, Qwen3-14B dequantized F16,
32,768 context, one slot, batch 2048, eight unpinned decode and batch threads.
CPU KV layers 0-31 and GPU KV layers 32-39, the same pair-v1 request with 9,737
prompt tokens and 64 generated tokens. Each arm gets a fresh 18 GiB scope,
swap disabled, a free server port with checked PID/argv, and the shared rig
lock. The gate drops the model file cache before server launch; prefill excludes
load and warmup. CUDA graph mode remains the default.

All arms are single runs. RAPL package plus NVML board measures host energy.
The phone executes no FFN in M1; its separately stated idle assumption is
0.875 W. Only ADB 5037 is used for transport identity materialization; phone
binaries and the prior six transport receipts are hash-checked. Materialization
binds the rebuilt server/libraries and actual transport source, without claiming
a new transport throughput measurement.

Calibration sums the actual ubatch sequence including the remainder and changing
attention context. `ANALYZE.py freeze` refuses to run after held-out output
directories exist, saves component sums and trace/source hashes, and writes
`physical/FROZEN_PREDICTIONS.json` exclusively. `ANALYZE.py check` verifies those
hashes, the same runtime, identical output tokens at each ubatch, and no OOM.
It checks both traced accounting and held-out prediction within 10%. The observed
traced/untraced duration difference includes single-run variation; it is not an
isolated instrumentation-overhead estimate. Full and representative prefill /
decode inventories remain in `physical/` alongside raw trace JSONL.

The fast-path plan specifies ubatch 128 and the M0 choice, 1024. This supersedes
the earlier architecture note's 128/512 example for this milestone.

## Original 18 GiB frozen calibration

Predictions were written at 2026-09-20 16:20:59.553197 UTC before either held-out
arm launched. The repeat check passed 73 rig tests in 5.394 s with pyflakes clean.
The 128-token arm has 76 full ubatches and a 9-token remainder; 1024 has nine full
ubatches and a 521-token remainder. Every trace contains 63 following decode graphs.

| ubatch | Copy host s, including waits | Compute host-call s | Normal completion wait s | Frozen sum s | Traced prefill s | Weight-copy bytes | Weight-copy host s, excluding preceding waits |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 197.819884 | 95.269275 | 0.028864 | 293.118023 | 294.798142643 | 1255225267200 | 188.178136 |
| 1024 | 32.563820 | 76.119895 | 0.104358 | 108.788073 | 110.039417596 | 165161216000 | 27.260108 |

Weight-copy host-call time falls 160.918028 s. FFN weights contribute
1,016,109,056,000 versus 133,698,560,000 copied bytes and 151.728923 versus
22.413633 s; attention/projection weights contribute the rest. These are actual
copy calls, including their host staging and page-fault costs, not an isolated
PCIe-bandwidth measurement. CUDA host submission is not device compute; completion
waits remain separately charged. The untraced Release specialization has no trace,
timer or emitter calls (`software/untraced-call-review.txt`).

## Original 18 GiB exact commands

From the shared workspace:

```sh
bash research_dev/scheduler/campaigns/burstgpt/reports/20260920-fast-path-M1/DEPLOY.sh
ssh -o BatchMode=yes zhihao@172.20.74.85 'bash /mnt/storage/s42-fast-path-M1-20260920-T3ypai/BUILD.sh'
ssh -o BatchMode=yes zhihao@172.20.74.85 'bash /mnt/storage/s42-fast-path-M1-20260920-T3ypai/MATERIALIZE_TRANSPORT.sh'
ssh -o BatchMode=yes zhihao@172.20.74.85 'bash /mnt/storage/s42-fast-path-M1-20260920-T3ypai/CHECK.sh'
ssh -o BatchMode=yes zhihao@172.20.74.85 'bash /mnt/storage/s42-fast-path-M1-20260920-T3ypai/RUN_INVENTORY.sh traced'
```

On the desktop, after traced arms finish and before held-out arms start:

```sh
cd /mnt/storage/s42-fast-path-M1-20260920-T3ypai/native-source
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 ../ANALYZE.py freeze --physical ../physical > ../physical/freeze.log
bash ../CHECK.sh
bash ../RUN_INVENTORY.sh heldout
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 ../ANALYZE.py check --physical ../physical > ../physical/check.log
```

The source copy includes the two historical profile fixtures used by
`test_profiles`; the first rig unit run omitted them and its failure is
preserved in `software/unit-rig-missing-fixture.log`. The corrected run passed
72 tests in 3.571 s, including mixed CPU/GPU tiny-model exact logits, observable
weight copies, incomplete/duplicate trace rejection, typed launch validation,
environment isolation, old-server rejection and runtime digest separation.
Local tests also passed; touched Python files pass pyflakes. The native
instrumentation built locally and on the CUDA rig.

Calibration exposed legitimate zero-row output tensors in intermediate Qwen
prefill batches. The reader now accepts zero dimensions while rejecting negative
ones; a regression test covers this case. The native records were unchanged.
The revised local suite passed 73 tests in 2.886 s; the rig then passed all 73
in 5.394 s before held-out execution, with pyflakes clean.

Materialized transport identity:
`sha256:82ac6e66272fabbf731af1f2d14cca3eeb563a6a883933bd590dc5011641bb14`.
The native build did not change after calibration began.
