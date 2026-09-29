# longtail_eval_v2 coverage and energy timelines

Generated 2026-09-25 from the completed legacy desktop and OP15+Pixel pair.
Collection at 16:46:57 UTC read saved artifacts only. No campaign or device was changed.

## Figures

- [Token coverage timeline PNG](token_coverage_timeline.png), [SVG](token_coverage_timeline.svg).
- [Energy and savings timeline PNG](energy_savings_timeline.png), [SVG](energy_savings_timeline.svg).
- [Both figures as a two-page PDF](eval_v2_timelines.pdf).

![Token coverage timeline](token_coverage_timeline.png)

![Energy savings timeline](energy_savings_timeline.png)

## Results

Coverage reconstruction PASS: all 3,604 output tokens accounted for. The policy-derived
layer counts and per-device row totals match each assisted request's physical execution proof.
Tokens using both phones count once in the any-phone column.

| Model | Output tokens | Any phone / OP15 | Coverage | Also uses Pixel | Pixel coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3 14B | 1,524 | 1,298 | 85.17% | 1,196 | 78.48% |
| Gemma 4 12B | 2,034 | 1,687 | 82.94% | 0 | 0% |
| Llama 3.2 1B | 46 | 0 | 0% | 0 | 0% |
| Total | 3,604 | 2,985 | 82.82% | 1,196 | 33.19% |

These percentages count tokens receiving FFN assistance on selected layers, not the fraction
of all model operations or FLOPs executed by a phone. All first output tokens from prefill
are included in the denominator. Pixel participates only alongside OP15 in this run.

| Arm | CPU kJ | GPU kJ | Host kJ | Paid interval s |
| --- | ---: | ---: | ---: | ---: |
| Legacy desktop | 159.872 | 68.663 | 228.535 | 2244.348 |
| OP15 + Pixel full configuration | 38.285 | 56.129 | 94.414 | 1832.951 |

Energy reconstruction PASS: each CPU/GPU endpoint agrees with RESULT.json within
one microjoule (integer rounding). Final host saving is **134.121 kJ / 58.687%**;
duration falls **411.397 s / 18.330%**. This is a single pair and excludes phone energy.
The full configuration also changes scheduling; the incremental Pixel saving is not isolated.

Exact-output acceptance **FAIL: 13/14** sequences match. The result hashes match the prior
[exact-token audit](../sources/rig_audit_20260925T163624.json). Qwen request 004 first differs
at zero-based output token 137; quality equivalence has not been established.

## How the timelines are constructed

Coverage uses only the 13 adaptive groups named by these requests' physical proofs;
30 historical groups in the observation store are excluded. Acknowledged policy boundaries,
window token ranges and the final policy assign phone devices to each output position.
Per-layer counts and per-device logical rows are then checked against execution proofs.
The global phone window counter is not used to attribute tokens to a request.

There are 3,542 individually timestamped decode observations and one first-token timestamp.
The remaining 61 tokens (16 request-tail tokens and 45 Llama tokens) are placed at request
completion, without interpolating individual token times. The final counts are verified;
those 61 time positions are approximate. Ratios remain flat when a model emits no new tokens.

Energy uses each sensor's own monotonic timestamp. CPU RAPL counters are unwrapped and
interpolated at the exact paid boundaries. GPU NVML board power is integrated with the
trapezoidal rule and interpolated boundary power, matching `server_energy_summary`.
The plot exports a one-second grid with both exact run endpoints included.

Both arms start at elapsed time zero independently. The intermediate gap compares equal
elapsed time, not equal completed work. After 1832.951 s the treatment curve holds its final
paid total; this is an accounting extension, not a measurement or extrapolation of idle power.
Only the final endpoints compare the complete trace in both arms.

## Reproduction and data

From this directory, without accessing hardware:

```sh
python3 build_timelines.py
```

`collect_timelines.py` optionally reads the two completed remote arms again. It records
source-file SHA-256 hashes and creates a new timestamped local evidence file.

- [Token-level events CSV](token_events.csv), including timestamp-source flags.
- [Cumulative token coverage CSV](token_coverage_timeline.csv).
- [CPU/GPU/host energy CSV](host_energy_timeline.csv).
- [Energy gap and percentage CSV](energy_savings_timeline.csv).
- [Totals, methods provenance and source hashes](TIMELINE_SUMMARY.json).
- [Validation results](VALIDATION.json).
- [Saved compact evidence](sources/timeline_evidence_20260925T164657.json).

Data validation PASS; pyflakes PASS on the collector, embedded remote reader and plot builder.
PNG layout inspected; the PDF has two pages. No production code change, commit or push.
