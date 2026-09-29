# Scheduler and phone-offload progress - 2026-09-25

Read-only result audit: **2026-09-25 17:14:10 UTC**. No campaign, deployment, phone state or existing worker was changed.

## What can be reported today

**Three completed eval_v2 arms:** Legacy desktop 228.535 kJ; OP15 104.465 kJ (-54.29%); OP15+Pixel 94.414 kJ (-58.69%). Two phones use 9.62% less host energy than OP15 in this single comparison. Exact outputs vs legacy: OP15 12/14, two phones 13/14; both strict checks FAIL.

- **Measured host savings exceed 25%:** dispatcher-only longtail_v1 saves **26.7%**, and the recent
  dev_v2 phone arms save **34.6-46.1% (OP15)** and **24.4-44.3% (OP15+Pixel)** against their matched desktop controls.
- **Strict exact-output acceptance FAILS on those completed comparisons:** 14/31 for the longtail
  desktop comparison, 5/9 for both OP15 repeats, and 4/9 then 5/9 for OP15+Pixel.
  All completed arms retain the requested output lengths and matched inputs. These are measured
  workload-energy results, not a new all-token-identical acceptance result or a quality-equivalence proof.
- **Short-trace repeats did not show a consistent Pixel benefit.** Across the two dev_v2 repeats,
  OP15 averages **38.18 kJ** and OP15+Pixel **42.00 kJ**:
  two phones use **10.0% more** host energy on the mean.
  The pairwise changes are +40.3% and -14.9%; assistance availability and model-load timing vary substantially.
- **Pixel itself is much faster:** layer RPC latency improves **2.72-3.04x** with byte-identical
  kernel outputs. A separate four-request server test is 4/4 token-identical; its full-width Pixel
  request saves **17.4%** host request energy against the mean of bracketing desktop controls.
  This short controlled test is not a trace result.

## Today's longtail_eval_v2 run

14 requests (7 Qwen, 6 Gemma, 1 Llama), 3,604 output tokens, maximum output 614,
1,675 s arrival span. The 3 outputs above 512 tokens carry about 49.5% of output work.
The new window and current replan/fail-fast fixes make this a separate experiment from longtail_v1.

Two-phone completion PASS, 94.414 kJ in 1833.0 s; 58.69% host saving. Exact output check FAIL: 13/14.

| Arm | Completed | CPU kJ | GPU kJ | Host kJ | Duration s | Host saving | Exact sequences |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Legacy desktop | PASS | 159.87 | 68.66 | 228.53 | 2244.3 | Reference | Reference |
| OP15 | PASS | 50.40 | 54.07 | 104.46 | 1821.9 | 54.3% | 12/14 (FAIL) |
| OP15 + Pixel | PASS | 38.28 | 56.13 | 94.41 | 1833.0 | 58.7% | 13/14 (FAIL) |
| Desktop + dispatcher | PREFLIGHT | - | - | - | - | - | Pending |

The legacy baseline uses **159.872 CPU + 68.663 GPU = 228.535 kJ**, over **2,244.348 s**
(**101.827 W** average). Current run order: legacy desktop, OP15+Pixel, OP15, desktop+dispatcher.
The dispatcher-only control separates the scheduling contribution on this trace.
No partial energy extrapolation is presented as a final saving.

**Observed full-configuration saving: 134.121 kJ (58.687%).** Duration falls by 411.397 s (18.330%). Average host power is 101.8 -> 51.5 W.

Phone-execution proof PASS: OP15 63,852 calls; Pixel 7,176 calls. All 7 Qwen and all 6 Gemma requests are assisted. No request is rejected; all 3,604 requested output tokens are present.

Strict identity FAIL; differences:

- `burstgpt_longtail_eval_v2:004`: first difference at zero-based output token 137.

No first-divergence logits or quality evaluation establish an acceptance exception.

Treatment phone-energy estimates are **4.536 kJ OP15** and **2.939 kJ Pixel**. They are assumed-power bookkeeping, excluded from the measured host saving.

**New OP15-only control:** completion PASS 14/14, 104.465 kJ / 1821.864 s. Relative to OP15, two phones save **10.051 kJ (9.622%)**, while taking 11.087 s (0.609%) longer. Request inputs, source-file hashes and native binaries match across all three arms. This is an observed single-run configuration difference; repeatability is unverified. OP15 vs two-phone exact sequences: 12/14 (FAIL). OP15 differs from legacy at request 003 token 203 and request 004 token 233 (zero-based).

The outer chain stopped at **2026-09-25 16:34:07 UTC** with `RuntimeError('CANCEL requested')`. The two-phone arm had already reported PASS / exit 0. The saved cleanup check found zero Pixel workers and zero ADB forwards. The remaining controls were subsequently reordered under CHAIN-ev3. This reporting session did not request cancellation.

CHAIN-ev3 completed OP15-only at **17:13:24 UTC**, exit 0 and clean worker/forward checks. Desktop+dispatcher is **PREFLIGHT** at this snapshot.

## Completed longer trace: scheduling benefit without phones

longtail_v1: 31 requests, 8,207 output tokens, maximum output 1,100. Same source files and
native binaries within the pair; no phone calls in either arm.

| Arm | Completed | CPU kJ | GPU kJ | Host kJ | Duration s | Host saving | Exact sequences |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Legacy desktop | PASS | 377.00 | 159.32 | 536.32 | 4984.4 | Reference | Reference |
| Desktop + dispatcher | PASS | 279.58 | 113.46 | 393.04 | 3381.2 | 26.7% | 14/31 (FAIL) |

The dispatcher saves **143.282 kJ (26.716%)** and **1,603.192 s (32.164%)**.
Model reloads fall **15 -> 8**, model switches **10 -> 4**, and logged model-load time
**630.3 -> 225.2 s**. Average host power rises **107.6 -> 116.2 W**; finishing sooner
more than offsets that increase. This supports the scheduling mechanism, not a phone-energy claim.

The OP15 arm of this older chain failed; OP15+Pixel was not run on longtail_v1.
No energy result is inferred for either missing arm. The subsequent replan fix is tested offline;
today's eval_v2 chain is its current hardware follow-up.

## Completed short trace: show both phone repeats

longtail_dev_v2: 9 requests, 1,420 output tokens, maximum output 617. All five runs complete;
every comparison has matching request IDs, prompts, lengths, seeds, source arrivals and source SLOs,
matching source-file hashes and matching native binaries. Baseline is **desktop + dispatcher**.

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

Pixel execution PASS: **1,608 / 1,842** verified FFN calls in its two repeats. All 4 Qwen
requests are assisted in both. OP15 r2 loses all Qwen assistance; the two-phone r1 loses the
first two Gemma assists. The result reports identify helper readiness and model-cache timing
as major differences. Two repeats do not separate those effects from an incremental Pixel benefit.

The Qwen execution-window energies are OP15 r1 **9.99 kJ**, OP15+Pixel **10.01 / 9.11 kJ**.
These model windows can overlap and are not additive trace-energy components. They suggest a
possible local benefit, not a controlled whole-trace attribution.

## Recent implementation and qualification

| Milestone | Status | Evidence and limit |
| --- | --- | --- |
| Automatic two-phone FFN execution | PASS on dev_v2 | Pixel calls 1,608 / 1,842; OP15 remains the primary helper |
| Device-set policies per batch composition | PASS on dev_v2 | Start OP15-only, probe OP15+Pixel, keep it only within energy/latency bounds; B1 and B4 challengers accepted |
| Pixel CPU worker qualification | PASS | Byte-identical rows 1-4; 4/4 token-identical 64-token server outputs; 744 Pixel calls |
| Pixel CPU+GPU concurrency target >=1.4x | FAIL | Best measured cell 1.01x; every emulated server-cadence dual configuration slower than CPU-only |
| Replan crash correction and fail-fast | PASS in tests/replay | 141 modules / 2,031 tests, exit 0 in saved suite report; formerly failing replays complete 31/31 |
| Current eval_v2 matched two-phone energy | PASS | Final result available above |

The integrated Pixel helper uses the **packed CPU/NEON path over ADB TCP**, with a per-process
CPU utilization floor, bounded thread-pool polling, and batched weight-row reuse. OP15 uses its
qualified HTP/USB path. Pixel GPU, TPU, and the experimental accessory USB path are not the
backend or transport used in these two-phone traces.

## Pixel latency improvement

Per-layer RPC at server-like cadence, Qwen layers 18-23, width 17,408, ADB TCP.
Before = mean of two control-arm medians; after = one optimized-arm median, 72 calls per batch size.
These are layer-call latencies, not whole-model token latency.

| Rows per call | Before ms | After ms | Speedup | Latency reduction |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 38.42 | 13.56 | 2.83x | 64.7% |
| 2 | 61.44 | 20.20 | 3.04x | 67.1% |
| 4 | 88.39 | 32.45 | 2.72x | 63.3% |

The slow production path was dominated by low CPU/DSU clocks between short bursts. The adopted
fix keeps the CPU ready for those bursts and reuses decoded weight rows across batched inputs.
The controlled server test's full-width Pixel request uses **4035.5 J** vs desktop control
mean **4884.6 J**, and takes **39.15 s** vs **42.33 s** desktop mean.

## Interpretation limits

Host energy is measured CPU-package RAPL plus GPU-board NVML over the paid trace interval. It is not wall-plug energy and excludes phone energy. Phone power is only modeled (4.5 W active, 0.875 W idle), so no measured whole-system saving is claimed. Compare percentages only within a trace and its stated baseline. The two longtail desktop arms and the dev_v2 baseline are single measurements; there are two repeats of each dev_v2 phone configuration. No confidence interval is established.

Output differences also occur in host-only comparisons. This is consistent with scheduling/batch-dependent
floating-point differences, but these artifacts do not establish a first-divergence logits tolerance
or task-quality equivalence. No exact-token exception is granted by this report.

Saved execution-identity hashes differ because source manifests include arm-specific resolved
configuration. The audit separately compares all source-file hashes and native-binary hashes;
both match within each completed comparison.

## Files and provenance

- [Why the headline changed from about 25% to 58.7%](SAVINGS_EXPLANATION.md): expanded layer sets,
  CPU energy reduction, scheduling and workload differences.
- [Recent results as report-ready tables](RECENT_RESULTS_TABLES.md).
- [New token coverage and energy timelines](timelines/README.md), with per-phone coverage,
  [two-page PDF](timelines/eval_v2_timelines.pdf) and downloadable CSV data.
- [Today's eval_v2 comparison](eval_v2_comparison.png), [vector SVG](eval_v2_comparison.svg).

- [Energy comparison figure](energy_comparison.png), [vector SVG](energy_comparison.svg).
- [Pixel latency figure](pixel_latency.png), [vector SVG](pixel_latency.svg).
- [Printable report](progress_report.pdf).
- [Trace table CSV](trace_energy.csv), [repeat means CSV](repeat_means.csv), [Pixel latency CSV](pixel_latency.csv).
- [Structured data and source hashes](REPORT_DATA.json).
- [Raw-result audit with exact saved tokens](sources/rig_audit_20260925T171405.json).
- Source reports: [two-phone evaluation](../20260925-two-phone-eval/README.md),
  [Pixel qualification](../20260925-pixel-cpu-gpu/README.md),
  [replan diagnosis](../20260925-replan-reprojection-fix/PROGRESS.md),
  [fail-fast validation](../20260925-replan-reprojection-fix/failfast/README.md).

Reproduce without running hardware:

```sh
python3 collect_remote.py
python3 build_report.py
```

The collector only reads remote files. The builder uses saved artifacts locally. Each capture
is retained with its UTC timestamp. No commit, push or production code change was made.
