# S42 General Energy Scheduler V1 Results

Date: 2026-08-08 EDT.

Verdict: `PROTOTYPE_PASS; I3_THREE_PAIR_FLEET_ENERGY_PASS; MMLU64_NONINFERIOR; I3_COHORT_ROUTE_COMPILED; STAGE6_MONITORED_PAIR_PASS; UNIFIED_EXECUTION_PLAN_ADAPTER_PASS; QWEN_FULL_FFN_SCREEN_PASS; PRECOMMIT_PHONE_THERMAL_ATOMICITY_PENDING; GENERAL_PER_SHAPE_ENFORCE_BLOCKED`.

## Three-session Qwen full-FFN screen

On 2026-08-09, OP15 kept about 9225 MiB of Gemma and Qwen FFN weights warm
across HTP0, HTP1, and HTP2. HTP1 and HTP2 jointly replaced complete Qwen
SWIGLU FFNs for layers 0-11 at decode M=1. Both sessions share one serialized
HTP compute resource; their benefit is increased resident capacity.

Two matched ABBA repeats used the same RTX 4060 Ti placement and the same
connected, resident phone boundary in control and treatment:

| mean over two repeats | GPU plus CPU | GPU plus CPU plus OP15 | change |
| --- | ---: | ---: | ---: |
| paid duration | 32.291 s | 28.400 s | -12.05% |
| desktop CPU plus GPU energy | 3963.549 J | 2367.054 J | -40.28% |
| whole-phone energy | 73.805 J | 82.495 J | +11.77% |
| accounted fleet energy | 4037.354 J | 2449.548 J | -39.33% |

The treatment increased phone energy by 8.689 J but saved 1596.495 J on the
desktop. Pair savings were 39.01% and 39.65%. Exact output tokens, 1296 calls
per treatment, the 2 GiB phone memory floor, and zero resets all passed. This
is a three-request Qwen decode screen, not a full-trace certificate. Details
and the strict hash-bound result are under `multi_session_phone_v1/`.

## Unified scheduler-owned successor A/B

On 2026-08-08, one fresh matched pair replayed the exact I3 cohort through the
canonical scheduler in `research_dev/scheduler`. The scheduler emitted a plan
before device changes, selected the route, and supplied all server, model,
operator-split, bridge, phone-worker, DMA-BUF, and USB settings to the backend
adapter. Both results and runtime receipts bind the plan hash and route.

| metric | scheduler control | scheduler treatment | change |
| --- | ---: | ---: | ---: |
| trace makespan | 735.298 s | 631.548 s | -14.11% |
| CPU package energy | 115.527 kJ | 93.462 kJ | -19.10% |
| GPU board energy | 19.521 kJ | 18.850 kJ | -3.44% |
| connected phone energy | 0.615 kJ | 1.644 kJ | +167.20% |
| accounted fleet energy | 135.663 kJ | 113.956 kJ | -16.00% |
| completed work | 74 req / 11,605 tok | 74 req / 11,605 tok | equal |
| SLO requests met | 55 | 55 | equal |

The treatment plan selected `i3-cold-cpu-op15-ffn-v1` for
`VERIFIED_COHORT_ENERGY_SAVING`; the control selected
`i3-cold-cpu-control-v1`. The treatment executed 52,320 phone calls, exposed
2.45% mean join wait, and recorded zero reset recoveries. Pinned MMLU64 was
27 / 64 in both arms. Every pair gate passed.

The hash-bound record and compact report are
`physical_ab_v1/UNIFIED_SCHEDULER_BURSTGPT_4060TI_OP15_R5_V1.json` and
`physical_ab_v1/UNIFIED_SCHEDULER_BURSTGPT_4060TI_OP15_R5_V1.md`. This closes
the execution-plan adapter gap for the exact certified cohort. Phone thermal
state is still reduced from an on-device sidecar after ADB restoration, so an
atomic pre-dispatch phone-thermal transaction remains pending.

## Stage 6 monitored successor A/B

On 2026-08-07, one new full-length physical pair replayed the exact I3 cohort
with runtime topology, thermal, heartbeat, USB, and reset monitoring. The raw
trace remains on the RTX 4060 Ti host; compact hash-bound receipts are in
`physical_ab_v1/`.

| metric | CPU control | CPU plus OP15 | change |
| --- | ---: | ---: | ---: |
| trace makespan | 732.607 s | 631.293 s | -13.83% |
| CPU package energy | 115.573 kJ | 95.172 kJ | -17.65% |
| GPU board energy | 20.006 kJ | 19.290 kJ | -3.58% |
| connected phone energy | 0.625 kJ | 1.689 kJ | +170.21% |
| accounted fleet energy | 136.205 kJ | 116.151 kJ | -14.72% |
| completed work | 74 req / 11,605 tok | 74 req / 11,605 tok | equal |
| SLO requests met | 55 | 55 | equal |

The treatment executed 52,320 dynamic FFN calls, assigned 76.82% of eligible
dense-FFN MACs to OP15, exposed 2.31% arithmetic-mean join wait, and recorded
zero USB resets. The maximum executor topology-heartbeat gap was 1.051 seconds.
Android thermal status remained `NONE`; the maximum NPU-zone sample was
60.8 C. The pinned MMLU64 authority remains 27 / 64 in both arms.

This one pair validates the already promoted three-pair I3 epoch; it does not
replace the three-pair certificate or broaden admission to arbitrary shapes.

## Outcome

The standalone scheduler is implemented without modifying llama.cpp core or
the active llama-server integration. It supports data-driven task, layer, and
operator routes, arbitrary nonnegative workload features, composite resource
queues, quality gates, measured-energy uncertainty, capacity policy, and
current-readiness overrides. Energy-enforced multi-resource layer/operator
routes now also require measured overlap evidence whose conservative exposed
join-wait bound is at most 5%.

The successor I3 campaign on 2026-08-06 uses llama-server's default CPU
selection, normal CPU repacking outside view-safe dense FFN tensors, and the
batch-shape policy `1:9664,3:8192,8:4096,128:8192,512:11136`. Three real
alternating control/treatment pairs reduce average makespan by 14.38% and
accounted CPU-package, GPU-board, and whole-phone energy by 16.76%. The fixed
BurstGPT workload mix passes the 5% mean-overlap and 10% fleet-energy gates.
The pinned MMLU64 score is 27 / 64 in both arms. This is physical admission
evidence for the exact cohort route now imported by the unified scheduler; it
is not a per-request additive-energy profile. Details are in
`../s41_gemma_qwen_continuous_baseline/tp_operator_split_v1/RESULTS_LLAMA_SERVER_I3_ENERGY_V1.md`.

The original V1 physical profile was built from the existing RTX 4060 Ti plus
OP15 paired BurstGPT campaigns. The replay preserved all 74 arrivals, 18,211
input tokens, 2,175 requested output tokens, two workload roles, and the
30-second SLO.

## Physical loop I3 headline

This is the current real-device result. Both arms complete 74 requests,
33,843 input tokens, and 11,605 output tokens in every repetition.

| average over three pairs | CPU control | CPU plus OP15 | change |
| --- | ---: | ---: | ---: |
| total trace makespan | 736.468 s | 630.594 s | -14.38% |
| accounted fleet energy | 137.217 kJ | 114.217 kJ | -16.76% |
| accounted fleet J/output token | 11.824 J | 9.842 J | -16.76% |
| server CPU plus GPU energy | 136.144 kJ | 112.586 kJ | -17.30% |
| output throughput | 15.758 token/s | 18.404 token/s | +16.79% |
| cold mean prefill/request | 61.636 s | 46.875 s | -23.95% |
| cold mean decode/request | 241.689 s | 212.841 s | -11.94% |
| completed work | 74 req / 11,605 tok | 74 req / 11,605 tok | equal |
| SLO requests met | 55 | 55 | equal |

All three treatment repetitions improve both latency and fleet energy. OP15
executes 52,320 paid calls per treatment, 88.245 trillion MACs, and 76.82% of
eligible dense-FFN MACs. This percentage is not full-model work. Mean exposed
join wait is 2.67%, zero bridge resets occur, and every run cleans up.

The quality authority is bounded approximate/task quality. The pinned MMLU64
score is 27 / 64 and 64 / 64 parseable in both arms. A path-matched greedy
probe is exact for 2 / 4 sequences and 21 / 32 token positions, so exact-token
quality remains unclaimed.

The fixed workload-mix route passes the paper-strength 10% fleet-energy gate.
It does not create a universal per-shape certificate: rare M=1 through M=3
buckets individually expose about 14% to 21% join wait, while the frequent
M=4 through M=8 buckets hide nearly all phone time. A general scheduler must
bind admission to a qualified shape bucket and workload/profile epoch.
The route compiler now installs this evidence as one cohort/profile-epoch
route rather than assigning shared trace energy additively to requests. The
execution adapter now applies the selected hash-bound plan. The remaining
boundary is a fresh atomic pre-commit runtime snapshot that includes phone
thermal state while preserving the fail-closed CPU fallback.

```text
VERDICT: PASS
BEST_REAL_RESULT: 630.594164 s, accounted_energy_j=114217, energy_change_pct=-16.76, eligible_phone_work_pct=76.82
BLOCKER: none for the fixed BurstGPT-mix route; general per-shape admission remains unqualified
NEXT_ONE_CHANGE: add an atomic pre-dispatch runtime and phone-thermal snapshot
```

## Historical physical loop I0 headline

This table contains real-device values only. It does not contain replay or
fleet-scaling predictions.

| metric | CPU control | CPU plus OP15 | change |
| --- | ---: | ---: | ---: |
| total BurstGPT makespan | 327.704644 s | 197.321246 s | -39.79% |
| accounted fleet energy | **not measured** | **not measured** | no claim |
| completed work | 74 req / 2,175 tok | 74 req / 2,175 tok | equal |
| SLO requests met | 57 | 59 | +2 |
| phone-routed requests | 0 / 74 | 17 / 74 | +17 |
| phone share of eligible cold FFN MACs | 0% | 62.62% | +62.62 points |

The phone executed 34.161 trillion of 54.554 trillion eligible cold-FFN MACs,
24,240 paid calls excluding warmup, and transferred 2,367,774,720 bytes in
each direction. All 17 eligible cold requests used OP15, equal to 22.97% of
all trace requests. Full-model phone compute share is unclaimed because the
hot model and non-FFN graph were not counted under one common MAC denominator.

For the split island, physical p50 values were 1.247 ms phone compute,
1.532 ms complete phone RPC, 1.480 ms host branch, 0.073 ms join wait, and
1.573 ms overlapped island latency. At p50, branch imbalance is 3.35%, the
phone finishes 0.051 ms after the host branch, exposed join wait is 4.67% of
the island, and 95.20% of the phone RPC is hidden by host work. This meets the
provisional 5% overlap target at p50. The source lacks sum/count counters, so
no arithmetic-mean overlap pass is claimed.

The sampled GPU-board diagnostic is 7,807.2 J for control and 6,816.3 J for
treatment using mean board watts times makespan, a -12.69% change. It excludes
CPU, phone, DRAM, motherboard, storage, and USB-controller energy and is
therefore not the requested accounted fleet result.

I0 is incomplete: it has only one physical pair, no synchronized CPU/GPU/phone
energy receipt, and no arithmetic-mean overlap counters. Its exact output
agreement is 8 / 17 cold requests and 394 / 505 cold token positions, so it is
an approximate-quality performance iteration.

```text
VERDICT: INCOMPLETE
BEST_REAL_RESULT: 197.321246 s, accounted_energy_j=missing, energy_change_pct=missing, phone_work_pct=62.62
BLOCKER: synchronized CPU, GPU, and phone energy for at least three alternating pairs
NEXT_ONE_CHANGE: add synchronized component power and mean overlap counters without changing execution
```

## Historical physical loop I1 server-energy acquisition

On 2026-08-06, the integrated llama-server path ran one matched pair on the
real RTX 4060 Ti desktop and OP15. This is a new source-length workload, not a
replacement for frozen I0:

```text
trace: /home/zhihao/s41-dynamic-ffn-v1/campaign/input/REQUESTS_SEMANTIC_SOURCE.jsonl
sha256: b20a9ba66ee3558d835a0e19ed3cfa4c31a4a9e8b4f9c085b29a14f80250a0ff
requests: 74
input tokens: 33,843
output tokens: 11,605
```

| metric | CPU control | CPU plus OP15 | change |
| --- | ---: | ---: | ---: |
| total trace makespan | 733.730 s | 651.964 s | -11.14% |
| server CPU plus GPU energy | 136,186 J | 101,421 J | -25.53% |
| server CPU plus GPU J/output token | 11.735 | 8.739 | -25.53% |
| completed work | 74 req / 11,605 tok | 74 req / 11,605 tok | equal |
| output throughput | 15.816 token/s | 17.800 token/s | +12.54% |
| cold mean prefill/request | 61.598 s | 46.897 s | -23.87% |
| cold mean decode/request | 240.659 s | 221.586 s | -7.93% |

The server energy boundary synchronizes Intel package RAPL and RTX 4060 Ti
NVML board power over the paid trace. CPU package energy falls from 116,252 J
to 82,183 J; GPU board energy falls from 19,935 J to 19,238 J. This excludes
whole-phone energy and the platform terms listed in the S41 report. At an
unmeasured 4.5 W phone sensitivity point, treatment would be 104,355 J and
23.37% below control. The measured server saving has a 53.32 W phone-power
break-even point, but this calculation cannot authorize `enforce`.

The treatment invoked OP15 for all 17 cold requests, made 52,320 paid layer
calls, and assigned a call-weighted 64.61% of eligible dense-FFN columns to
the phone. It transferred 6,876,610,560 bytes in each direction and held
3,303.31 MiB of resident phone weights. Full-model compute share remains
unclaimed.

Arithmetic means are 5.246 ms phone RPC, 4.748 ms host branch, 1.382 ms join
wait, and 6.130 ms overlapped island. Exposed join wait is 22.55%, so this
route fails the 5% balance gate even though the complete trace improves. The
dominant M=6 and M=8 decode cuts are the next bounded rebalance targets.

Greedy token equality is also not an I1 quality authority. The cold route is
exact for 2 / 17 requests with a median 30-token common prefix, but the
untouched hot CUDA route is exact for only 42 / 57 across the two continuous-
batch runs. This proves cross-run batch-geometry sensitivity. BurstGPT has no
task answers, so semantic quality remains unmeasured.

```text
control RESULT sha256: 04c35a16bcce2ada5a794ece46ce3cd51248ef057779d97eed1ee1e57f587ab8
treat   RESULT sha256: c0587ed4ff71917bab4e534294cd343bf991b3fef5a8d3d760b007744c80597c
VERDICT: INCOMPLETE
BEST_REAL_RESULT: 651.964225 s, server_energy_j=101421.263, server_energy_change_pct=-25.53, eligible_phone_work_pct=64.61
BLOCKER: synchronized whole-phone energy, repeated pairs, 5% mean-wait gate, and semantic quality authority
NEXT_ONE_CHANGE: rebalance decode cuts from the measured per-shape arithmetic means
```

## F16 Gemma shape-balance follow-up

The proposed per-shape rebalance was implemented through the unified
scheduler and tested on the same 17 Gemma requests from the two-model F16
BurstGPT trace. The physical A-B-B-A keeps the 25-GPU-layer placement,
resident OP15 weights, exact input/output lengths, and synchronized
CPU-package plus GPU-board plus whole-phone boundary fixed.

| metric | fixed 6,144-column split | shape-balanced split | saving |
| --- | ---: | ---: | ---: |
| makespan | 694.686 s | 680.995 s | +1.97% |
| CPU package energy | 51.862 kJ | 52.946 kJ | -2.09% |
| GPU board energy | 24.899 kJ | 24.478 kJ | +1.69% |
| whole-phone energy | 1.768 kJ | 1.726 kJ | +2.39% |
| fleet energy | 78.529 kJ | 79.150 kJ | -0.79% |

Weighted join wait fell from about 0.860 ms to 0.035 ms, but narrower phone
widths increased host-branch work and mean CPU package power from about
74.66 W to 77.75 W. The paired fleet savings were +1.50% and -3.07%, so the
candidate is not repeatably energy-positive. All equal-work, placement,
unified-plan, synchronized-energy, zero-reset, and zero-swap gates passed.

```text
VERDICT: RETAIN_FIXED_POLICY
BEST_REAL_RESULT: mean makespan saving=1.97%, mean fleet energy saving=-0.79%
BLOCKER: latency balancing shifts enough FFN work to the CPU to increase fleet joules
NEXT_ONE_CHANGE: measure an energy-aware intermediate width before another promotion screen
```

The selected evidence is under
`full_fp16_burstgpt_v1/shape_balance_v1/results/physical_abba_v1/`. The failed
candidate was not promoted to the complete 74-request run; the existing
25.537% fleet-energy-saving policy remains the qualified fallback.

## Physical reproduction

| route campaign | physical makespan | predicted makespan | error | physical throughput | predicted throughput | physical / conservative SLO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU control | 327.704644 s | 327.737360 s | +0.010% | 6.6371 token/s | 6.6364 token/s | 57 / 57 |
| CPU plus OP15 FFN | 197.321246 s | 197.625155 s | +0.154% | 11.0226 token/s | 11.0057 token/s | 59 / 59 |

The control fit used 17 cold CPU observations. The phone fit used 17 cold
CPU-plus-OP15 observations. The hot CUDA fit pooled 114 observations from the
two campaigns. Maximum one-sided residuals are 4.889 s, 1.673 s, and 1.351 s,
respectively, and are included in conservative scheduling bounds.

This is an in-sample reproduction of the imported physical runs, not an
independent generalization result. Its value is that the event simulator,
resource queues, route mapping, and uncertainty gates reconstruct the observed
campaign before the policy is connected to a live executor.

## Historical V1 policy decisions on frozen I0 BurstGPT

| mode | phone-routed cold requests | predicted makespan | conservative SLO | claim |
| --- | ---: | ---: | ---: | --- |
| control | 0 / 17 | 327.737 s | 57 / 74 | baseline |
| enforce | 0 / 17 | 327.737 s | 57 / 74 | energy fail-closed |
| shadow | 17 / 17 | 197.625 s | 59 / 74 | latency opportunity only |
| capacity | 17 / 17 | 197.625 s | 59 / 74 | lower server busy time |

`enforce` rejected all 17 phone candidates with `ENERGY_NOT_MEASURED`.
Neither physical campaign has a synchronized CPU-package, GPU-board, and
whole-phone component set, so there is no fleet-joule comparison to authorize
offload. GPU utilization and board-power samples are retained as diagnostics
only.

After energy is measured, a split still fails closed with
`OVERLAP_NOT_MEASURED` or `OVERLAP_LIMIT` unless its arithmetic-mean overlap
record and uncertainty remain within the 5% join-wait limit. Shadow mode may
explore such a route but cannot promote it.

The phone route is also `approximate`: prior physical outputs were not all
token-exact. Replaying with `quality_requirement=exact` rejected all 17 phone
candidates with `QUALITY_INSUFFICIENT` in every policy mode.

## Live device check

The read-only capture at `2026-08-06T02:06:02+00:00` found:

- RTX 4060 Ti UUID `GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08`, driver
  595.84, P8, 0% utilization, 7.93 W, and 779 MiB used at the instant sampled;
- i9-12900K CPU identity;
- OP15 serial `3C15AU002CL00000`, CPH2749 / SM8850, on an existing ADB
  server at port 5037;
- USB device `22d9:2772` negotiated at 5,000 Mb/s; and
- another agent's active llama-server trace processes.

S42 did not modify or stop those processes. The current readiness overlay sets
`op15-htp=false` because no S42 worker lease and resident-weight receipt exist.
The live-state dry replay therefore routed zero requests to OP15 and recorded
17 `resource is not ready: op15-htp` rejections.

## Verification

- 231 / 231 unified scheduler and S42 tests pass in isolated processes.
- 24 / 24 I3 policy, quality, phone-energy, campaign, and server-energy tests
  pass in isolated processes.
- The strict I3 validator accepts both new physical arms and the paired Stage
  6 record.
- Python syntax checks pass for every S42 source and test.
- ASCII source and documentation scan passes.
- Profile, replay, live-probe, and analysis records carry internal SHA-256
  bindings.
- Analysis verdict is `pass=true`.
- Stage 6 intentionally launched the exact qualified models, worker, bridge,
  and AOA transport. It did not alter model, server, bridge, worker, kernel,
  frequency, or clock-policy artifacts.
- The fresh unified pair bound both results and runtime receipts to the
  scheduler-selected route and plan hash before accepting either arm.

Key bindings:

```text
profile  sha256:0f8765ed562ad1b50a0e50130e350d9f218ab9a22ba1ab47d00d47db679952d8
replay   sha256:9e2f145c11da0244a9b7d36fb93a3ec11680fa7fdb921d1d9865b33f600905d3
analysis sha256:a9e1bc0cd7cd81284547cdaa1d63a7576d0a0af0c93fcf278a065268a8e720f6
probe    sha256:f9ea90ce25c2d1d3d78fad424ee78b457980f12791d0104dfdc21d08bd2d4ba3
physical sha256:d96895d2a6f2097edc77d04a46cc19bcbc8c43bf8dc3704d6a5e94ef556623be
stage6  sha256:874fee36ae98d4fb27a9bdbee08f1f60f4cf57b5b4bf39f2d1348f4b21ea955e
unified sha256:49acd562d6abef02a393eb581e06a39bf124991c9e0ec0c5ae4c61979f6d868e
```

## Next bounded loop

1. Publish one fresh atomic pre-dispatch snapshot that includes host topology,
   USB identity, bridge readiness, and phone thermal state; retain the CPU
   fallback when any gate fails.
2. Add shape-bucket eligibility. Keep M=1 through M=3 fail-closed until a
   separate route passes the 5% wait gate; admit only observed qualified
   buckets and batch mixtures.
3. Replay a held-out BurstGPT window or a changed arrival-intensity mix before
   broadening the route beyond the measured trace distribution.
4. If publication requires whole-system rather than accounted-device energy,
   repeat the same pairs with one external AC boundary as a separate metric.
5. Add another execution alternative only after the general admission rule is
   validated. Avoid expanding to arbitrary online graph partitioning.
