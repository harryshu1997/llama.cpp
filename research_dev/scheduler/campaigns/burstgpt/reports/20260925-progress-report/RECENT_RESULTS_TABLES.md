# Recent results - 2026-09-25 17:14:10 UTC

Measured host energy is CPU-package RAPL plus GPU-board NVML. Phone energy is excluded.
Savings are relative to the baseline named for each trace; percentages across traces do not add.

## Latest matched trace: longtail_eval_v2

14 requests, 3,604 output tokens. Inputs, source-file hashes and native binaries match.
Every completed arm finished 14/14 requests with zero rejections.

| Arm | Completed | CPU kJ | GPU kJ | Host kJ | Duration s | Host saving | Exact sequences |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Legacy desktop | PASS | 159.87 | 68.66 | 228.53 | 2244.3 | Reference | Reference |
| OP15 | PASS | 50.40 | 54.07 | 104.46 | 1821.9 | 54.3% | 12/14 (FAIL) |
| OP15 + Pixel | PASS | 38.28 | 56.13 | 94.41 | 1833.0 | 58.7% | 13/14 (FAIL) |
| Desktop + dispatcher | PREFLIGHT | - | - | - | - | - | Pending |

The two-phone arm uses **10.051 kJ (9.62%) less** host energy than OP15 alone, with **0.61% longer** duration. This is one observation per configuration; repeatability is not established. The two phone configurations match each other on 12/14 exact sequences.

Strict output identity FAIL for both phone arms. No first-divergence logits or quality evaluation
justify calling these differences harmless. The dispatcher-only control remains pending.

## Token coverage in the two-phone arm

| Model | Output tokens | Any phone | Coverage | Also uses Pixel | Pixel coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3 14B | 1,524 | 1,298 | 85.17% | 1,196 | 78.48% |
| Gemma 4 12B | 2,034 | 1,687 | 82.94% | 0 | 0.00% |
| Llama 3.2 1B | 46 | 0 | 0.00% | 0 | 0.00% |
| All models | 3,604 | 2,985 | 82.82% | 1,196 | 33.19% |

Coverage reconstruction PASS. Tokens using both phones count once. OP15 serves every assisted
token in this run; Pixel participates on a subset of Qwen tokens. These counts describe FFN
assistance on selected layers, not whole-model offload or the percentage of model FLOPs offloaded.

## Longer trace: longtail_v1

31 requests, 8,207 output tokens. Baseline is legacy desktop. All requests complete;
strict exact-output identity FAIL for the dispatcher comparison.

| Arm | Completed | CPU kJ | GPU kJ | Host kJ | Duration s | Host saving | Exact sequences |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Legacy desktop | PASS | 377.00 | 159.32 | 536.32 | 4984.4 | Reference | Reference |
| Desktop + dispatcher | PASS | 279.58 | 113.46 | 393.04 | 3381.2 | 26.7% | 14/31 (FAIL) |

Model loads fall from 15 to 8, switches from 10 to 4, and load time from 630.3 to 225.2 s.
Neither arm uses a phone. This is a scheduling result.

## Short trace and repeat spread: longtail_dev_v2

9 requests, 1,420 output tokens. Baseline is desktop with the new dispatcher.
All arms complete 9/9; strict exact-output checks FAIL on all phone comparisons.

| Arm | Completed | CPU kJ | GPU kJ | Host kJ | Duration s | Host saving | Exact sequences |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Desktop + dispatcher | PASS | 44.56 | 19.41 | 63.97 | 627.1 | Reference | Reference |
| OP15 r1 | PASS | 15.04 | 19.45 | 34.49 | 634.6 | 46.1% | 5/9 (FAIL) |
| OP15 r2 | PASS | 22.47 | 19.39 | 41.86 | 630.5 | 34.6% | 5/9 (FAIL) |
| OP15 + Pixel r1 | PASS | 26.81 | 21.57 | 48.37 | 729.2 | 24.4% | 4/9 (FAIL) |
| OP15 + Pixel r2 | PASS | 15.83 | 19.81 | 35.64 | 622.1 | 44.3% | 5/9 (FAIL) |

| Configuration | Repeats | Mean host kJ | Saving vs desktop + dispatcher |
| --- | ---: | ---: | ---: |
| OP15 | 2 | 38.18 | 40.3% |
| OP15 + Pixel | 2 | 42.00 | 34.3% |

On this shorter trace the two-phone mean uses 10.0% more energy.
Helper readiness and model-load timing vary across repeats; do not report a universal Pixel gain.

## Pixel kernel/runtime improvement

Controlled Qwen FFN layer RPCs over ADB TCP at server-like cadence; 72 calls per batch size.
Before is the mean of two control medians; after is the optimized-arm median.

| Rows per call | Before ms | After ms | Speedup | Latency reduction |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 38.42 | 13.56 | 2.83x | 64.7% |
| 2 | 61.44 | 20.20 | 3.04x | 67.1% |
| 4 | 88.39 | 32.45 | 2.72x | 63.3% |

Numerical check PASS: byte-identical kernel outputs. Separate four-request server check PASS:
4/4 token-identical outputs. Its full-width Pixel arm saves 17.4% host request energy
against the mean of bracketing desktop controls; this is a short controlled test, not a trace result.

## Download and provenance

- [All trace energy rows CSV](trace_energy.csv).
- [Repeat means CSV](repeat_means.csv).
- [Token coverage CSV](recent_token_coverage.csv).
- [Pixel latency CSV](pixel_latency.csv).
- [Exact saved-token audit](sources/rig_audit_20260925T171405.json).
- [Full report and figures](README.md); [PDF report](progress_report.pdf).
- [Coverage and energy timeline graphs](timelines/README.md).

The OP15-only completion was checked in CHAIN-ev3 at 17:13:24 UTC. All activity for this
report was read-only on the rig; no campaign was launched, stopped or changed.
