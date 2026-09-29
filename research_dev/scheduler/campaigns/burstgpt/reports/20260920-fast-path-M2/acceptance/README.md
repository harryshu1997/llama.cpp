# M2 amended acceptance recheck

**M2: FAIL on the utilization gate.** All N=1/2/4/8 correctness and hang checks
pass under the amended rule, but phone compute time rises from 51.488 to 64.171
ms/token/slot between N=4 and N=8 (24.633%). RPC time also rises. The plan
requires phone time per token per slot to fall with N; M3 has not started.
[Aggregate check](../physical/acceptance/CHECK_M2.json).

Step 3.2 is closed: 90/90 matched-input calls have identical returned payloads,
agreement fraction 1.000. The N=1 saved-output regression is exact. Across the
15 long-pair slots, 14 are exact and one passes at its first near-tie. All
complete 576 outputs, with at least 567 full-cohort assisted steps per layer.

The original failed N=2 pair remains preserved and labeled **single runs from a
failed correctness pair**: 65.8 W versus 123.4 W and 21.2 kJ versus 45.0 kJ.
The Step 3 diagnostic energy rows (23.9 and 24.6 kJ per two-slot request, about
553 ms/token) remain **single runs with local shadows enabled**. The new N=2
pair supplies the accepted correctness/utilization point; it does not rewrite
those historical records.

The user chose first mismatch per slot as the acceptance boundary. The existing
[existing ANALYZE.py](../ANALYZE.py) now accepts EXACT tokens, or a first mismatch with host top-1/top-2
raw-logit margin <= 0.05 and full-logit NMSE <= 5e-4. Every differing position is
recorded with both host logits, margin and NMSE. Later positions are explicitly
comparisons after context divergence and do not decide acceptance.

Local validation: the 100-test suite passes with two live-server tests skipped. All 100
rig tests pass, including the capture on/off two-slot reference check; pyflakes
is clean. The initial tiny completion test failed because its default no-vocab
fixture cannot detokenize, even with capture disabled. The optional test vocabulary
resolves that fixture issue; no serving-code change was needed. First-failure
records are preserved under `../physical/acceptance/software/`.

Raw logits are captured before sampling through the default-off typed
`LlamaServerLaunchContract.logits_trace_path` field. It enters the runtime digest;
unsupported launches fail closed. The binary format is `S41LOG1\0`, then repeated
little-endian u32 slot, task, one-based output step and vocabulary count, followed
by that many f32 logits. Capture coverage is checked against every slot/step.
Both long paired arms capture logits; neither has local FFN shadows enabled.
These measurements include capture writes. Sampling offload is already disabled
by default in the serving configuration.

Deployment: `/mnt/storage/s42-fast-path-M2-acceptance-20260921-c09ea6`.
Native pack/unpack and acknowledgement behavior are unchanged. The ordered
fixture is the one used by the eight-case matrix: it waits for native allocation
and prefill-start records before submitting the next request. Request index is
stable and submission order is 0 through N-1.

| Arm set | Prompts | Outputs per slot | Diagnostics | Raw logits |
| --- | --- | ---: | ---: | --- |
| Two ordered N=2 phone repetitions | 256,257 | 64 | first 5 assisted steps | off |
| N=1 saved-output regression | pair-v1, 9737 tokens | 64 | off | off |
| Host-only / 100% phone pairs, N=1/2/4/8 | 256 through 256+N-1 | 576 | off | on |

All arms use ubatch 1024, batch 2048, eight unpinned threads, 16 GPU layers,
keep-cache + populate, and fresh uncapped scopes with swap disabled. Every arm
runs the touched-module unittests and pyflakes first and verifies transport hashes.
OP15 alone is used, with ADB 5037. Host energy is measured RAPL package plus NVML
board energy; assumed phone power is separate, 4.5 W active and 0.875 W idle.
`memory.events.max` at ready and finish is recorded for every arm.

Exact commands are in `DEPLOY.sh`, `BUILD.sh`, `MATERIALIZE_TRANSPORT.sh`,
`CHECK.sh`, `RUN_ARM.sh`, `RUN_CHECKS.sh` and `POSTPROCESS.sh`, with arm argv and source hashes in
`../physical/acceptance/*.command.json`. The native/Python change from the starting
shared tree is recorded in [ACCEPTANCE.patch](ACCEPTANCE.patch). No commit or push was made.

```sh
bash research_dev/scheduler/campaigns/burstgpt/reports/20260920-fast-path-M2/acceptance/DEPLOY.sh
ssh zhihao@172.20.74.85 'bash /mnt/storage/s42-fast-path-M2-acceptance-20260921-c09ea6/BUILD.sh'
ssh zhihao@172.20.74.85 'bash /mnt/storage/s42-fast-path-M2-acceptance-20260921-c09ea6/MATERIALIZE_TRANSPORT.sh'
ssh zhihao@172.20.74.85 'bash /mnt/storage/s42-fast-path-M2-acceptance-20260921-c09ea6/RUN_CHECKS.sh'
rsync -a research_dev/scheduler/campaigns/burstgpt/reports/20260920-fast-path-M2/acceptance/POSTPROCESS.sh zhihao@172.20.74.85:/mnt/storage/s42-fast-path-M2-acceptance-20260921-c09ea6/POSTPROCESS.sh
ssh zhihao@172.20.74.85 'bash /mnt/storage/s42-fast-path-M2-acceptance-20260921-c09ea6/POSTPROCESS.sh'
```

The run driver completed all arms; every paired checker returned 0. The final
postprocessing command returned 1 for the recorded utilization failure.

## Ordered Step 3.2: PASS, 2026-09-21 00:55:46 UTC

| Run | Request slots | Own acks | Diagnostic rows / calls | Max relative L2 | Rows above 1e-2 | events.max ready/finish |
| --- | --- | --- | --- | ---: | ---: | --- |
| run1 | 1/0 | 4/3 | 180 / 90 | 2.896e-4 | 0 | 0/0 |
| run2 | 1/0 | 4/3 | 180 / 90 | 2.896e-4 | 0 | 0/0 |

Both 64-output requests completed in both runs. All 90 corresponding complete
call inputs match, and all 90 corresponding returned payloads match (180/180
rows). Matched-call return agreement fraction: **1.000**. No nondeterminism was
observed in these first five assisted steps; this does not establish determinism
for every phone execution. The original event remains unexplained. This result
is descriptive, and the amended near-tie correctness rule remains in force.

Records: `../physical/acceptance/determinism/CHECK_DETERMINISM.json`, including every
call/row comparison, and each run's `CHECK_DIAGNOSTIC.json` with numeric metrics.

## N=1 saved-output regression: PASS, 2026-09-21 01:03:33 UTC

All 64 pair-v1 outputs match the saved M0 tuned ubatch-1024 host reference.
The 9,737 prompt token IDs also match. The native runtime is newer; this is
an output regression, not a new matched energy pair. No tolerance was needed.

| Single phone regression | Exact outputs | Request host kJ, measured | Decode ms/token | Phone W, assumed | events.max ready/finish |
| --- | --- | ---: | ---: | --- | --- |
| pair-v1, N=1 | 64/64 | 20.588 | 771.840 | 4.5 active; 0.875 idle | 0/0 |

The host request energy is measured RAPL package plus NVML board. The separate
phone assumption gives 315.291 J over the request. There were 61 assisted steps
on each of the 18 owned layers, with no watchdog event. Records are in
`../physical/acceptance/regression/CHECK_REGRESSION.json` and its adjacent files.
The decreasing-utilization and 512-step checks use the separate 576-output pairs.

## Utilization and power: FAIL

| N | Phone compute ms/token/slot | Phone RPC ms/token/slot | Host-only decode W, measured | Phone-arm host decode W, measured | Phone W, assumed | Full-cohort steps/layer |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 164.651 | 194.934 | 124.304 | 66.846 | 4.5 active; 0.875 idle | 573 |
| 2 | 82.999 | 103.113 | 124.962 | 67.197 | 4.5 active; 0.875 idle | 572 |
| 4 | 51.488 | 67.398 | 133.308 | 66.018 | 4.5 active; 0.875 idle | 571 |
| 8 | 64.171 | 77.996 | 145.646 | 57.087 | 4.5 active; 0.875 idle | 567 |

N=4 to N=8 compute rises 24.633%; RPC rises 15.724%. The plan requires phone time per token per slot to fall with N, so M2 fails. No cause is established by these single pairs.

The denominator is full-cohort physical calls * N / 18 owned layers. Partial start/drain calls remain in the records and are excluded from these fixed-N points.

## Per-slot correctness: PASS under the amended rule

| N | Request | Slot, both arms | Prompt tokens | Acceptance | Matching positions | First mismatch |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 0 | 256 | EXACT | 576/576 | - |
| 2 | 0 | 1 | 256 | EXACT | 576/576 | - |
| 2 | 1 | 0 | 257 | EXACT | 576/576 | - |
| 4 | 0 | 3 | 256 | EXACT | 576/576 | - |
| 4 | 1 | 2 | 257 | EXACT | 576/576 | - |
| 4 | 2 | 1 | 258 | NEAR_TIE | 141/576 | 80 |
| 4 | 3 | 0 | 259 | EXACT | 576/576 | - |
| 8 | 0 | 7 | 256 | EXACT | 576/576 | - |
| 8 | 1 | 6 | 257 | EXACT | 576/576 | - |
| 8 | 2 | 5 | 258 | EXACT | 576/576 | - |
| 8 | 3 | 4 | 259 | EXACT | 576/576 | - |
| 8 | 4 | 3 | 260 | EXACT | 576/576 | - |
| 8 | 5 | 2 | 261 | EXACT | 576/576 | - |
| 8 | 6 | 1 | 262 | EXACT | 576/576 | - |
| 8 | 7 | 0 | 263 | EXACT | 576/576 | - |

| N | Request / slot | Step | Host / phone token | Host top-1 logit | Host top-2 logit | Margin | NMSE | Later differing positions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 2 / 1 | 80 | 19 / 15 | 25.099 | 25.099 | 3.891e-04 | 8.226e-07 | 434 |

Both displayed N=4 logits round to the same value; the margin above uses the unrounded raw values. Every mismatch retains its step, both host logits, margin and NMSE in CHECK_N4_pair.json. Only the first mismatch decides acceptance; 434 later differences are labeled after_context_divergence.

## Single-pair energy and latency

| N | Arm | Request host kJ, measured | Decode ms/token by request index | Phone W, assumed | Request phone kJ, assumed |
| --- | --- | --- | --- | --- | --- |
| 1 | control | 44.457 | 616.324 | 0.875 idle | 0.316 |
| 1 | combined | 20.388 | 518.871 | 4.5 active; 0.875 idle | 1.346 |
| 2 | control | 45.628 | 629.805 / 625.968 | 0.875 idle | 0.324 |
| 2 | combined | 21.623 | 545.869 / 541.543 | 4.5 active; 0.875 idle | 1.408 |
| 4 | control | 50.662 | 657.491 / 653.627 / 649.476 / 645.610 | 0.875 idle | 0.339 |
| 4 | combined | 26.488 | 678.298 / 674.360 / 670.098 / 665.989 | 4.5 active; 0.875 idle | 1.738 |
| 8 | control | 60.732 | 725.834 / 721.996 / 717.085 / 713.368 / 709.529 / 705.589 / 701.570 / 697.547 | 0.875 idle | 0.376 |
| 8 | combined | 41.735 | 1223.169 / 1220.323 / 1217.133 / 1213.956 / 1210.076 / 1206.038 / 1201.857 / 1197.622 | 4.5 active; 0.875 idle | 3.126 |

Host energy is RAPL package plus NVML board, counted once per concurrent request group. Phone energy is separate and assumed over the union of active decode intervals plus idle time. Decode power uses the common decode interval; per-request ms/token is the server predicted_ms divided by output tokens. Both long arms write raw logits; these timings include that instrumentation. The phone arm has lower host power at every N but is slower end to end at N=4 and N=8.

## Memory and validation

| Arm | memory.peak GiB | memory.events.max ready | finish | oom_kill finish |
| --- | --- | --- | --- | --- |
| determinism/run1 | 27.793 | 0 | 0 | 0 |
| determinism/run2 | 27.799 | 0 | 0 | 0 |
| regression | 28.994 | 0 | 0 | 0 |
| n1-control | 27.963 | 0 | 0 | 0 |
| n1-combined | 28.246 | 0 | 0 | 0 |
| n2-control | 28.444 | 0 | 0 | 0 |
| n2-combined | 28.845 | 0 | 0 | 0 |
| n4-control | 29.083 | 0 | 0 | 0 |
| n4-combined | 29.073 | 0 | 0 | 0 |
| n8-control | 29.098 | 0 | 0 | 0 |
| n8-combined | 29.108 | 0 | 0 | 0 |

All 11 arms use fresh MemoryMax=infinity, MemorySwapMax=0 scopes. All 100 rig unittests pass and pyflakes is clean before each arm. Cleanup passed for all 11 scopes and owned server PIDs; all seven phone closes report terminal status 0 and RESTORED. The rig lock is free, the GPU has no compute process, and OP15 is visible on ADB 5037. No other phone was used.

## Records and disposition

Per-arm `*.command.json` files preserve argv, working directory, configuration
and source text/hashes. Each `SERVER_IDENTITY.json` records the free listener's
PID, answering command line, typed launch contract and native/model hashes.
`../physical/acceptance/software/deploy/TRANSPORT_QUALIFICATION_IDENTITY.json` is the identity materialized
after this build; the runner verifies it before every arm. The native build is
identical across the current pairs. Source hashes, build/test logs and the
initial tiny-fixture failure are preserved under `../physical/acceptance/software/`.
[CSV curve](../physical/acceptance/UTILIZATION_CURVE.csv),
[per-arm validation](../physical/acceptance/VALIDATION_SUMMARY.json), and
[source snapshot check](../physical/acceptance/software/WORKSPACE_SOURCE_CHECK.json)
provide the compact audit records. Full raw logits, streams, call proofs, energy
samples, memory observations and mismatch metrics are under `../physical/acceptance/` and the deployment's
`physical/` directory. Raw JSON and binary observations retain their precision;
report results use at most three decimals.

[Archive verification](../physical/acceptance/ARCHIVE_VERIFIED.json) passed at
2026-09-21 03:09:48 UTC: all 317 recorded files, 10.647 GB, match their desktop
SHA-256 hashes. The complete raw-logit captures are available locally. Exact
archive and verification commands from the controller are:

```sh
rsync -a --stats zhihao@172.20.74.85:/mnt/storage/s42-fast-path-M2-acceptance-20260921-c09ea6/physical/ research_dev/scheduler/campaigns/burstgpt/reports/20260920-fast-path-M2/physical/acceptance/
ssh zhihao@172.20.74.85 python3 - < research_dev/scheduler/campaigns/burstgpt/reports/20260920-fast-path-M2/physical/acceptance/software/ARCHIVE_COMMAND.py
rsync -a zhihao@172.20.74.85:/mnt/storage/s42-fast-path-M2-acceptance-20260921-c09ea6/physical/ARCHIVE_SHA256.json research_dev/scheduler/campaigns/burstgpt/reports/20260920-fast-path-M2/physical/acceptance/
python3 -u research_dev/scheduler/campaigns/burstgpt/reports/20260920-fast-path-M2/physical/acceptance/software/VERIFY_ARCHIVE.py research_dev/scheduler/campaigns/burstgpt/reports/20260920-fast-path-M2/physical/acceptance
```

The four final report/cleanup artifacts created while the bulk copy was active
were copied separately; the manifest verifies them with the other records.
Software/build logs and the saved M0 reference are additionally archived under
`../physical/acceptance/software/`.

[Cleanup](../physical/acceptance/CLEANUP.json) passed. No in-flight worker was
force-killed, no second phone was used, and no commit or push was made.
M2 remains failed solely on the decreasing phone-time check. Stop here for
the user's decision; no performance fix, rerun or M3 launch is inferred.
