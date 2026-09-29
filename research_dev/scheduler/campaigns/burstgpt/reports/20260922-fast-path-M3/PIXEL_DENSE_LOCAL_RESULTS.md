# Pixel shader-body tuning: phone-local tests, 2026-09-23

Numerical audit PASS: 1,800 calls across15 arms, maximum relative L2
0.000325482692 against archived CPU references. Every control and both leading
candidates are byte-identical to the original Pixel outputs. Four slower
candidates change low bits while remaining within the0.01 numerical threshold.

All requests and response capture execute on Pixel over loopback. The desktop
does not run a model or regenerate CPU references. Latency is the worker's
internal graph build/upload/compute/download interval, not pure shader time.
USB round trips, server latency and energy are not measured. This saturated
local replay has a different cadence from earlier host-driven sweeps; compare
only against controls within the same run.

## First sweep: PASS numerical checks, exploratory latency

128 threads,128-lane subgroup,8 output rows; quantum4352. Each arm has120calls;
first24 are warmup. Means below use48 warm calls per width, covering six layers.
Each candidate is compared against the average of its surrounding row8 controls.

| Shader | Half-width ms | Full-width ms | Full control ms | Full latency reduction | Full achieved GFLOP/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| pair4_u4 | 13.207 | 22.682 | 21.021 | -7.90% | 23.58 |
| vec4_u1 | 11.332 | 19.834 | 21.021 | +5.65% | 26.96 |
| vec4_u2 | 11.071 | 19.598 | 21.021 | +6.77% | 27.29 |
| vec4_u4 | 12.837 | 22.738 | 21.387 | -6.32% | 23.52 |
| vec4_u8 | 15.649 | 27.861 | 21.387 | -30.27% | 19.19 |
| vec4_u4_a2 | 12.495 | 22.448 | 21.387 | -4.96% | 23.82 |
| vec4_u4_a4 | 13.689 | 24.157 | 21.097 | -14.50% | 22.14 |
| vec8_u2 | 12.950 | 22.341 | 21.097 | -5.90% | 23.94 |
| striped4_u4 | 31.843 | 61.653 | 21.097 | -192.24% | 8.67 |

Positive reduction means faster. GFLOP/s divides matrix FLOPs by the entire
worker interval; it is an achieved workload rate, not GPU peak throughput.

Unroll1/2 with native f16vec4 loads improve this workload. Larger unroll factors,
multiple accumulator chains and striped lane mapping regress. Register-pressure
or occupancy explanations are hypotheses; no hardware counters were collected.

## Confirmation: numerical PASS, modest average latency improvement

The two leading candidates were repeated in reversed order with20 repeats each
and interleaved row8 controls:2,160 phone calls. Every output is byte-identical
to the original Pixel kernel; CPU maximum relative L2 remains0.000325482692.
Warm measurements have108 samples per width/arm. Both candidates improve the
bracketed mean in both orders, but neither beats every individual control.

| Candidate | Half-width ms | Matched control ms | Reduction | Full-width ms | Matched control ms | Reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vec4_u1 | 11.696 | 12.277 | 4.728% | 20.455 | 21.244 | 3.714% |
| vec4_u2 | 11.798 | 12.277 | 3.900% | 20.679 | 21.244 | 2.661% |

Full-width controls span20.489-21.753ms. The second vec4_u1 arm is0.569% slower
than its following control, although2.439% faster than its surrounding-control
mean. This limits confidence in a small gain; the initial6.772% result should
not be presented as a confirmed speedup. No GPU clock/occupancy counters were
captured. Battery temperature was31.2-32.1C during the comparison.

Best repeated average: vec4_u1,26.144 matrix GFLOP/s per worker second. Across
both new runs,3,960 calls pass CPU checks; all2,160 confirmation outputs are
exact. Cleanup PASS: unchanged boot, all finite workers exited0, no adb forward
was created. New script pyflakes PASS, and the common auditor reproduces all
four earlier sweep reports exactly. The initial local lock failure executed
zero calls and is retained as FAIL in the raw archive.

No server-token qualification or production promotion is implied. The existing
server-qualified row8 selection is retained; the new library is isolated.
[Confirmation audit](PIXEL_DENSE_LOCAL_CONFIRM.json),
[averages and limits](PIXEL_DENSE_LOCAL_CONFIRM_SUMMARY.json),
[reproducible candidate settings](PIXEL_DENSE_LOCAL_CANDIDATE.json).

[Raw audit](PIXEL_DENSE_LOCAL_SWEEP.json),
[physical evidence](physical/pixel10pro-dense-local-1/run1/RESULT.json),
[local runner](pixel_local_sweep.py).
