# Results audit, 2026-09-27 05:00 UTC

Read-only audit of the completed `longtail_eval_v2` runs on
`zhihao@172.20.74.85`. No experiment, deployment, phone action, policy change,
commit or push was performed. The latest saved campaign result is g11, completed
2026-09-26 09:22 UTC. No newer energy result was found in this campaign root.

## Verdicts

| Check | Verdict | Evidence |
| --- | --- | --- |
| Saved host energy and request completion | PASS | All 12 audited arms completed 14/14 requests and contain 3,604 final output tokens; request parameters match the first legacy baseline |
| g11 retains the live server through Pixel loss | PASS | Two recoveries retain queue positions; mask and live reconnect use server generation 6; no mask-ended/server-exit event |
| Provisioning stall markers in tp2 | PASS | Zero `HELPER_REMATERIALIZATION_FAILED`, `no helper opportunity`, or `replacement source is not ready` occurrences in its decision log |
| Thermal exclusions reduce OP15 availability | PASS | Stored executor snapshots show qualification false during the long Qwen interval; g11 also records 528.898 s of explicit thermal deferrals |
| Relaxed thermal policy restores best energy | NOT VERIFIED | Policy remains off; no confirming arm exists |
| Repeatable 58.7% host saving | NOT VERIFIED | The three undisturbed two-phone runs save 58.687%, 47.210%, and 50.911% against the first legacy reference |
| Strict output identity | FAIL | Best two-phone run is 13/14 exact; tp2 and g11 are 12/14 |
| Working scheduler Python sources equal stage and deploy | PASS | All 463 non-report Python files match by SHA-256 |
| Fresh full test runner | FAIL | One 10 ms performance assertion measured 10.4753444 ms; isolated rerun PASS |

## Energy and output checks

Host energy is measured CPU-package RAPL plus GPU-board NVML energy over the paid
trace interval. It is not wall-socket energy. Phone energy remains an assumed-power
model and is excluded below. Savings use the same 228.534703 kJ first legacy
baseline throughout this table, including the repeat and fault arms.

| Arm | CPU kJ | GPU kJ | Host kJ | Duration s | Host saving | Exact outputs |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Legacy desktop r1 | 159.872 | 68.663 | 228.535 | 2244.348 | reference | reference |
| Desktop + dispatcher r1 | 123.458 | 54.695 | 178.153 | 1834.608 | 22.046% | 13/14 |
| OP15 r1 | 50.398 | 54.067 | 104.465 | 1821.864 | 54.289% | 12/14 |
| OP15 + Pixel r1 | 38.285 | 56.129 | 94.414 | 1832.951 | 58.687% | 13/14 |
| Legacy desktop r2 | 175.793 | 74.560 | 250.353 | 2428.830 | -9.547% | 12/14 |
| Desktop + dispatcher r2 | 120.089 | 53.978 | 174.066 | 1809.916 | 23.834% | 11/14 |
| OP15 r2 | 56.232 | 54.949 | 111.181 | 1836.064 | 51.351% | 11/14 |
| OP15 + Pixel r2 / ev8 | 66.391 | 54.253 | 120.644 | 1832.924 | 47.210% | 13/14 |
| OP15 + Pixel tp2 | 56.885 | 55.300 | 112.186 | 1848.969 | 50.911% | 12/14 |
| G1g: retire and reload | 59.000 | 53.884 | 112.884 | 1794.644 | 50.605% | 10/14 |
| g9: mask-out, old dispatcher | 77.513 | 60.163 | 137.677 | 1947.747 | 39.757% | 11/14 |
| g11: mask-out, fixed dispatcher | 66.662 | 54.777 | 121.439 | 1814.338 | 46.862% | 12/14 |

The best measured saving remains 58.687%, not a newly improved result. Incremental
Pixel saving in the first controls is 9.622% versus OP15 alone. These are single-run
comparisons. The repeat baseline itself used 9.547% more energy. Software and
native binary identities also changed between the original controls and tp2/g11;
the audit records those differences. The fault arms differ in fault timing and
number of affected requests. Their energy differences do not isolate the recovery
fix or establish that mask-out consumes less energy than retire/reload.

Every comparison above uses fresh token parsing of the canonical SSE streams,
not only length or output-text checksums. The desktop repeat's 12/14 identity
does not make the treatment identity failures pass. No saved logits establish
that the divergent outputs are numerical near-ties or semantically equivalent.

## g11 recovery evidence

`RESULT.json` records `HELPER_MASKED_OUT` at 505.537575 s for Pixel layers 18-23
on `physical:hot:desktop`, endpoint 18571, generation 6. It records
`HELPER_REATTACHED` with `mode=live_reconnect` at 570.268482 s on that same
generation: 64.730907 s between mask and reconnect. The membership log records
identity-verified re-admission.

| Request | Failed attempt | Recovery attempt | Discarded tokens | Discarded attempt time |
| --- | --- | --- | ---: | ---: |
| Qwen 003 | 1 | 2 | 71 | 47.018522 s |
| Qwen 004 | 0 | 1 | 72 | 47.224992 s |

Both recovery events specify `same_server_mask_out`. Both FALLBACK decisions
carry `RECOVERY_RETAINED_QUEUE_PLACE` with sequences 4 and 5, preserving the
waiting Gemma request behind them. The mask event list contains no
`HELPER_MASK_ENDED`, and the decision log contains no `SERVER_EXITED`.

The code in `adapters/runtime.py` defines `penalty_us` as the failed attempt's
finish time minus its start time. Therefore the quoted 47 s is discarded work,
not time from loss to restored service and not a measured end-to-end latency
increment versus an otherwise identical control. Recovery re-executes the prompt;
it does not preserve the partial generated sequence as a continuation checkpoint.

## Thermal findings and corrections

| Two-phone arm | Host kJ | OP15 Qwen calls | Observed OP15 qualification |
| --- | ---: | ---: | --- |
| r1 | 94.414 | 23,364 | No false qualification snapshot |
| r2 / ev8 | 120.644 | 2,940 | False for a 570.296 s sampled interval; provisioning defects also present |
| tp2 | 112.186 | 5,526 | False for 514.662 s during the Qwen window, plus a later unclosed episode |

The tp2 long interval is 735.856290-1250.517845 s on the saved snapshot clock.
Do not directly align that clock with a differently offset plot origin. A later
false qualification begins at snapshot time 1805.738731 s and is still false at
the last snapshot, 1841.388230 s, spanning another 35.649499 s without a recorded
clear. Thus 515 s describes the long interval, not all exclusions in that run.
The maximum saved OP15 executor temperature is 67.8 C, not 66.7 C. These are
sampled observations, not continuous thermal measurements.

The old snapshots retain `thermal_qualified` but no raw `thermal_status` values.
They establish exclusion, but do not independently verify the report's exact
LIGHT/status-1 attribution for every excluded interval. The current ADB probe
uses the maximum status over the platform and individual sensors. Its raw status
and new per-device policy need to be recorded in a confirming arm.

The scheduler applies the phone qualification to route feasibility. Accordingly,
the issue is a scheduler admission rule acting on phone telemetry; saying it is
entirely unrelated to the scheduler is misleading. Thermal gating is a confirmed
coverage mechanism. Proving that changing the limit removes the entire energy
gap requires the still-unrun intervention and repeats.

The configured 90 C threshold is used only when thermal qualification is unknown
in `_internal/route_generation/feasibility.py`. It is not an independent hard
ceiling when a known thermal verdict is accepted.

Counts depend on the artifact and nesting. The tp2 decision log has 706 literal
`THERMAL_LIMIT` and 272 `PHONE_HELPER_UNAVAILABLE` occurrences. They are not counts
of unique thermal events. The earlier 35,290/544 claim is not reproduced by that
counting scope. All three reported provisioning-stall strings are absent in tp2.

## Verification and remaining scope

The 463-file equality check covers scheduler Python sources excluding reports,
`__pycache__`, and `.venv`. It does not certify the entire repository or all native
build outputs. The checkout branch remains `wip/unified-scheduler-cleanup-20260813`;
the handoff's word "main" refers to the working checkout.

Fresh command: `PYTHONPATH=.:research_dev/scheduler/tests python3
research_dev/scheduler/tests/run_all.py`. It invoked 154 scripts/modules and
printed 150 unittest summaries, reporting 2,240 unittest cases. Exit status was
1, solely from `test_cached_synthetic_refinement_is_below_ten_milliseconds`:
10.4753444 ms against 10 ms. A single isolated rerun of that unchanged test passed.
The 12 recovery-dispatch and 18 thermal-policy tests passed. This is a
timing-sensitive failure, not a fresh blanket suite PASS. Python was 3.13.12.
Pyflakes passed for the audit collector and its embedded remote reader.

At this audit, the local root filesystem had 278 GiB available and 93% usage.
That verifies current headroom, not the exact amount freed by the earlier cleanup.
A read of the rig's lock table found no holder of the shared execution lock;
phone battery and thermal state were not polled, so the old 77% battery value
must not be presented as current.

The thermal override remains off. G2 OP15 loss, G3 OP15 re-plug, thermal-policy
confirmation, repeated energy confidence intervals, and strict output identity
remain open. This audit did not exercise those hardware gates.

Evidence: [AUDIT.json](AUDIT.json), [CSV table](RESULTS.csv),
[validation](VALIDATION.json), [collector](collect_audit.py),
[full test log](scheduler-tests.log), [isolated timing test](timing-recheck.log).
The JSON includes remote paths, raw RESULT and stream hashes, exact-token
comparison summaries, source/binary matching checks, recovery events, and thermal
snapshot transitions. All rig access by the collector is read-only.
