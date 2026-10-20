# Frozen CUDA-graph reference set v1

Complete: all six references pass, each completing the unchanged three-request,
655-output-token workload. Real CUDA graph capture/replay, semantic-sanity
output validation, terminal proofs, exact desktop parents and frozen assignments
pass. There are zero fallbacks, USB resets, failed session loads or phone reloads.
No 24- or 84-request trace was run.

**Best tested fixed layout: `fixed-ggg`.** Including runtime preparation, it
uses 22.49 kJ at nominal phone power: 29.42% less than clean upstream and 29.05%
less than the matched modified runtime. This is one small run per arm, not a
replicated or steady-state claim, and does not establish an adaptive-residency
advantage over fixed placement.

Final artifacts are exclusively under [references-source-v7/](references-source-v7/).
The earlier completed source-v6 arms at the top level are preserved, but excluded
from this comparison. [COMPARISON.json](COMPARISON.json) is the validated machine
summary; [MEASUREMENT_DETAILS.json](MEASUREMENT_DETAILS.json) contains per-request
timelines, reasons, native settings, session phases and preparation measurements.

## Preparation-inclusive energy

CPU package and GPU board energy are measured. Phone active power is assumed at
3/4.5/6 W, with identical 0.875 W idle treatment in every arm. All figures include
runtime loading, warm-up, exploration, control, graph work and dynamic cleanup.

| Reference | Duration (s) | Fleet kJ, 3 W | Fleet kJ, 4.5 W | Fleet kJ, 6 W |
| --- | ---: | ---: | ---: | ---: |
| desktop-default-cuda | 371.20 | 31.866 | 31.866 | 31.866 |
| desktop-matched-cuda | 355.14 | 31.699 | 31.699 | 31.699 |
| fixed-ggg | 312.20 | 22.352 | 22.492 | 22.632 |
| fixed-qqq | 384.84 | 33.239 | 33.325 | 33.411 |
| fixed-ggq | 315.72 | 25.206 | 25.337 | 25.468 |
| fixed-gqq | 331.50 | 31.867 | 31.963 | 32.058 |

Nominal savings versus matched desktop are GGG **29.05%**, GGQ **20.07%**,
GQQ **-0.83%**, and QQQ **-5.13%**. GGG remains best at all three assumed phone
powers; its saving versus matched desktop spans **28.61-29.49%**, and versus
clean upstream **28.98-29.86%**. No arm's unfavorable result is discarded.
GGQ/GQQ/QQQ consume 12.65%/42.11%/48.16% more nominal energy than GGG.

The measured CPU/GPU components are 21.262/10.279 kJ for upstream,
21.117/10.272 kJ for matched desktop, and 12.464/9.417 kJ for GGG. All six
domain breakdowns are in COMPARISON.json. The phone estimate is additional.
These are CPU-package/GPU-board/assumed-phone totals, not wall-plug energy.

## Request latency

Service latency includes prefill and decode, from the execution receipt's start
to finish. It excludes earlier queueing/loading. Original arrivals remain
Gemma 36 at 1 s, Llama 37 at 61 s, Qwen 50 at 91 s.

| Reference | Gemma 36 (s) | Llama 37 (s) | Qwen 50 (s) |
| --- | ---: | ---: | ---: |
| desktop-default-cuda | 159.24 | 1.64 | 58.90 |
| desktop-matched-cuda | 160.20 | 1.62 | 57.91 |
| fixed-ggg | 135.39 | 1.75 | 61.30 |
| fixed-qqq | 175.04 | 1.75 | 51.81 |
| fixed-ggq | 137.04 | 1.87 | 61.19 |
| fixed-gqq | 169.40 | 1.73 | 56.20 |

Arrival-to-start delays are separately preserved in MEASUREMENT_DETAILS.json.
For example, Gemma's matched/GGG/QQQ delays are 42.26/48.66/76.69 s. These
include queueing and loading, not pure scheduler overhead. QQQ speeds up Qwen's
own service but loses end-to-end on this workload. Startup and service durations
vary; these runs do not isolate how much variation is caused by loading
contention, OS caches or phone assistance. No overlapping preparation window is
subtracted to manufacture a steady-state result or break-even estimate.

## Actual phone assistance

All percentages below use **all output decode tokens**, including unassisted
startup and tails. Any-assistance percentage and fraction-weighted percentage
are different: the latter sums the applied split fraction across decode tokens.
Llama is desktop-only in every reference; both desktop controls have zero
assisted tokens for all models.

| Fixed layout | Gemma assisted / 292 | Gemma weighted | Qwen assisted / 71 | Qwen weighted |
| --- | ---: | ---: | ---: | ---: |
| ggg | 280 (95.89%) | 90.24% | 0 (0%) | 0% |
| qqq | 0 (0%) | 0% | 59 (83.10%) | 83.10% |
| ggq | 279 (95.55%) | 89.90% | 11 (15.49%) | 15.49% |
| gqq | 51 (17.47%) | 11.82% | 59 (83.10%) | 83.10% |

Gemma's actual token distribution at 0/25/50/75/100% is:

- GGG: 12/11/11/11/247 tokens.
- GGQ: 13/11/11/11/246 tokens.
- GQQ: 241/11/11/11/18 tokens.

Qwen's positive calls use 100% of the resident column width; QQQ/GQQ have
12 zero-fraction tokens and 59 assisted tokens, GGQ has 60 and 11 respectively.
Native per-layer calls are matched to acknowledged policy generations and
terminal proofs, including calls outside closed measurement windows.

| Layout | HTP0 stored layers | HTP1 stored layers | HTP2 stored layers | Calls HTP0 / HTP1 / HTP2 |
| --- | --- | --- | --- | ---: |
| ggg | Gemma 0-7 | Gemma 8-15 | Gemma 16-23 | 2240 / 2240 / 2240 |
| qqq | Qwen 0-5 | Qwen 6-11 | Qwen 12-17 | 354 / 354 / 354 |
| ggq | Gemma 0-7 | Gemma 8-15 | Qwen 12-17 | 2232 / 2232 / 66 |
| gqq | Gemma 0-7 | Qwen 6-11 | Qwen 12-17 | 408 / 354 / 354 |

Gemma's executed widths are 3,840/7,680/11,520/15,360 columns for
25/50/75/100%; Qwen's executed positive width is 17,408. Every assisted layer
is CPU-resident under the exact frozen desktop parent. Gemma has 25 CPU FFN
layers and Qwen has 24: even a 100% split does not offload the whole model.

The adaptive controller is unchanged across fixed arms. GQQ Gemma returns to
0% with INCUMBENT_NO_LONGER_BENEFICIAL at token 60; it is not forced to remain
assisted. Short Qwen in GGQ records INSUFFICIENT_OPPORTUNITY, which must not
be relabeled as proof that every fraction is energy-negative. Exact helper
events, evidence, budgets and fraction changes remain in each RESULT.json.

## Preparation, memory and transitions

Each fixed arm loads the real FFN shard files once per selected session. The
assignment is explicit configuration, not a large hysteresis interval. Each
session progresses independently and remains generation 1. There are three
initial phone loads, zero policy replacements and zero subsequent reloads per
fixed arm; desktop controls have no phone loads. Session IDs, masks, artifact,
geometry, operator plan, leases and physical terminal proofs remain checked.

| Layout | Resident GB | Preparation (s) | First READY (s) | All READY (s) | Prep-window fleet kJ |
| --- | ---: | ---: | ---: | ---: | ---: |
| ggg | 8.493 | 47.14 | 25.41 | 51.65 | 2.495 |
| qqq | 9.626 | 50.18 | 26.02 | 54.64 | 1.863 |
| ggq | 8.871 | 65.03 | 24.99 | 68.17 | 4.328 |
| gqq | 9.248 | 57.14 | 24.17 | 61.03 | 3.571 |

READY times are publication times relative to the paid boundary. Preparation
duration is the enclosing measured interval, not simply the sum of stage
times. Prep-window energy is nominal and overlaps desktop loading/execution;
it is already included in the total, not an additive marginal phone cost.

Per-session stage-to-verification times, HTP0/HTP1/HTP2 respectively:
GGG 19.65/12.68/13.39 s; QQQ 20.13/13.70/14.90 s;
GGQ 30.36/11.76/19.45 s; GQQ 19.65/18.81/16.77 s.
Native weight read/upload, HTP initialization, verification and endpoint READY
timestamps are preserved without mixing phone and host clock domains.

All layouts fit the declared 10 GB shared phone pool with the existing
per-session and workspace admission checks. The planner records a 4,096-byte
base workspace; physical transition envelopes also carry their larger
execution-contract workspace demands. Neither is omitted from admission. There is no
Llama/OpenCL allocation in this set. Per-session weight_source=ffn_shard
proofs record opened paths, bytes loaded, index/shard/parent hashes, stored and
executed masks and column widths. The underlying weight files are not copied
into the report directory.

## CUDA graphs and bounded compatibility

GGML_CUDA_DISABLE_GRAPHS is absent from server environments. Nsight proves
actual graph execution rather than relying on USE_GRAPHS=1 in a log.

| Reference | Captures | Reused-executable recaptures* | GPU graph replays |
| --- | ---: | ---: | ---: |
| desktop-default-cuda | 116 | 48 | 2458 |
| desktop-matched-cuda | 116 | 49 | 2458 |
| fixed-ggg | 64 | 25 | 2400 |
| fixed-qqq | 86 | 25 | 2378 |
| fixed-ggq | 76 | 24 | 2388 |
| fixed-gqq | 84 | 25 | 2380 |

*Defined as executable updates minus initial instantiations, floored at zero.
All raw API counts and durations are retained. Counts include warm-up, prefill,
shape changes and control; they are not attributed solely to fraction changes.
Every graph API return code is zero and GPU graph activity matches launches.

Before the set, [bounded-check/](bounded-check/) completed Qwen 71/71 tokens,
59 assisted tokens and 1,062 calls, with one load/session and a 0-to-100%
control change. It records 78 captures and 1,796 GPU graph replays. Subsequent
Gemma fixed arms physically exercise 0/25/50/75/100% without reloads. Failed
bounded/mixed/preflight attempts remain in the artifact tree with their causes
documented in [CHANGES.md](CHANGES.md).

## Frozen identity, reproduction and limits

[Provenance](provenance/README.md) lists source, binary, model, shard and parent
hashes, qualifications, defaults, every class of override, permitted comparison
differences and accounting boundaries. Exact resolved commands, journals,
snapshots, native stderr, Nsight reports/SQLite and raw power samples accompany
each final reference. The clean upstream pin is
`4f31eedb0ccf546b7e8d6bb243b170f12522f54d`; the modified binary is only called
desktop-matched-cuda. GDM and unrelated processes were not stopped. Final GPU
inspection shows the original 3,178 MiB allocation and no inference process.

Reproduce without physical inference, using the archived source rather than
subsequent working-tree edits (run from this directory):

```sh
analysis_root=$(mktemp -d /tmp/cuda-graph-reference-XXXXXXXX)
tar -xzf provenance/reference-source-v7.tar.gz -C "$analysis_root"
python3 analyze_references.py --source-root "$analysis_root" --output "$analysis_root/COMPARISON.json"
cmp COMPARISON.json "$analysis_root/COMPARISON.json"
python3 freeze_artifacts.py --verify
```

The analyzer refuses incomplete sets, mismatched parents/artifacts/requests,
different initial evidence or boundaries, graph-disabled execution, changed
fixed assignments, reloads and invalid native coverage/proofs. Outputs are
created exclusively; existing completed references are never overwritten.
ARTIFACT_SHA256.json inventories every report and evidence file except itself
and generated Python caches. The origin/attempt mapping is in
[ARTIFACT_INDEX.json](ARTIFACT_INDEX.json).

Final focused validation: 74 residency/COW/fixed tests plus two unchanged
decision replay goldens. Earlier comparison/default-launch/preflight tests and
exact changes are recorded in CHANGES.md; no repeated full harness was run.
Later local runner cleanup is not part of the frozen source or these results.

Limitations: one sequential run per arm, uncontrolled OS cache/thermal variation,
Nsight/debug instrumentation, assumed phone power, no whole-phone Llama, and no
steady-state or preload-amortization experiment. The positive GGG result meets
25% on this small preparation-inclusive set; it is not a guarantee for a larger
trace. Future work must separately report benefit over clean upstream, matched
desktop, and this best tested fixed layout. No adaptive-residency win is claimed.
