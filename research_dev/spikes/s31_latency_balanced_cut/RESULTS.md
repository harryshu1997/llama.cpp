# S31 Latency-Balanced Phone Cut Results

Verdict: `S31_BALANCED_FULL_TRACE_PASS` for measured partition selection,
physical B32 execution, priority/SLO mechanics, and selected CUDA compute
relief. Numeric equivalence and total-system energy remain unproven.

## Why S29 was slow

S29 fixed R2 at OP12 `[0,6)` -> OP15 `[6,8)` -> CUDA `[8,48)`. In its final
B32 trace, OP12 averaged about 1.256 seconds per decode step while OP15
averaged about 0.223 seconds. OP12 was about 5.6x slower and determined the
pipeline cadence.

S31 keeps the same layer-8 output boundary and measures every cut that moves
work from the S29 baseline toward OP15 (`k=1..6`):

| Cut | OP12 p95 | OP15 p95 | Phone bottleneck | Balance ratio | Route p95 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.307 s | 0.399 s | 0.399 s | 0.771 | 3.144 s |
| 2 | 0.574 s | 0.445 s | 0.574 s | 0.776 | 4.069 s |
| 3 | 0.920 s | 0.439 s | 0.920 s | 0.477 | 4.821 s |
| 4 | 1.071 s | 0.399 s | 1.071 s | 0.373 | 6.047 s |
| 5 | 1.225 s | 0.310 s | 1.225 s | 0.253 | 6.187 s |
| 6 | 1.499 s | 0.267 s | 1.499 s | 0.178 | 6.635 s |

The deterministic selector chose cut 1: OP12 `[0,1)` and OP15 `[1,8)`.
Relative to the cut-6 baseline in the same sweep, this lowers the measured
phone bottleneck 73.40% and route p95 52.62%. No per-layer interpolation is
used.

## Fresh selected-route calibration

Each range is two physical repetitions for one input token and four output
steps. The selected R2 route is measured first at B32 to avoid a reproduced
HTP shape-growth stall when B32 followed B1/B4/B24 in one session.

| Route | B1 | B4 | B24 | B32 |
|---|---:|---:|---:|---:|
| R0 | 0.266-0.273 s | 0.283-0.302 s | 0.416-0.442 s | 0.382-0.486 s |
| R2 cut 1 | 0.589-0.895 s | 0.762-0.860 s | 2.904-3.226 s | 3.067-3.254 s |

The profile is digest-bound to the calibration and to a canonical replay of
all six cut measurements. A hand-written or mutated `CUT_SELECTED` label is
rejected.

## Full 60-request trace

| Metric | All-CUDA control | Selected-cut treatment | Result |
|---|---:|---:|---:|
| Completed | 60 | 60 | PASS |
| SLO misses | 0 | 0 | PASS |
| Routes | R0=60 | R0=28, R2=32 | B32 offload |
| Phone batches | none | OP12 B32 x4, OP15 B32 x4 | PASS |
| P0 p95 latency | 1.175 s | 0.643 s | -45.26%, PASS |
| Summed CUDA compute | 7.166 s | 5.095 s | -28.90%, PASS |
| Makespan | 4.414 s | 8.541 s | 1.93x control, cost |
| Makespan vs S29 cut 6 | - | 12.138 -> 8.541 s | -29.63%, PASS |

During the treatment B32 cohort, OP12's four physical steps averaged 0.285 s
and OP15's averaged 0.315 s. The slow-phone bottleneck is removed without
reducing batch size or offloaded depth. The treatment remains slower than the
all-CUDA control; the result is improved heterogeneous throughput and server
compute relief, not a latency win over local CUDA.

## Validation and limits

- Every worker retained one PID, boot nonce, and device boot ID across
  calibration DETACH, control DETACH, and treatment STOP.
- All placement certificates passed with zero missing-buffer compute nodes.
- All 60 requests completed once, all software leases and KV state drained,
  and urgent/background rows never shared a physical batch.
- The independent validator replay is byte-identical. New tests are 12/12 and
  inherited S29 tests are 10/10.
- Only 21/60 request token sequences match the Q8 CUDA control. The F16-phone
  route remains numerically uncertified.
- Phone, WiFi, USB, network, and total-system energy are unknown. Summed CUDA
  compute time is not an energy measurement.

Accepted artifacts are under
`results/selected_campaign_final_20260721T222102Z/`. The canonical selection
bundle is `results/selected_cut_bundle_20260721T215337Z/`. Two aborted selected
campaigns are retained as negative evidence: one overlapped an unrelated OP15
HTP experiment, and one reproduced the ascending-shape HTP stall. Neither
enters the accepted profile or validation.
