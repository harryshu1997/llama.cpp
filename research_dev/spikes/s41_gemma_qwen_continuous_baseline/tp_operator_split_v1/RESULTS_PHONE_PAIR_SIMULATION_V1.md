# Two-OP15 one-operator simulation

Date: 2026-07-31 EDT

## Scope

This is not a real two-phone acquisition. Each complementary slice was run
independently on the same real OP15. A two-OP15 makespan is simulated as the
maximum of the two measured slice quantiles plus 0.0038 ms for the measured
host sum. The simulation assumes independent USB buses and does not include
cross-device launch or bus interference.

The FFN test uses Gemma-like Q8_0 geometry (K=3840, NFF=15360, B=1). The
attention test uses Qwen Q8_0 geometry (K=5120, 8 GQA groups, KV=8192, B=1).
They are separate operator tests and must not be added into one model layer.

## Results

| route | full OP15 median | simulated 2x OP15 | speedup | latency change | rel-L2 | argmax |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FFN OpenCL | 31.465 ms | 11.297 ms | 2.785x | -64.10% | 2.88e-4 | match |
| FFN HTP | 31.465 ms OpenCL oracle | 5.012 ms | 6.278x | -84.07% | 1.23e-2 | match |
| Attention OpenCL | 32.394 ms | 23.839 ms | 1.359x | -26.41% | 2.99e-4 | match |
| Attention HTP | 32.394 ms OpenCL oracle | 5.079 ms | 6.378x | -84.32% | 4.26e-1 | match |

At an assumed equal 4.5 W active power per phone, the normalized active-phone
energy is:

| route | one-phone energy | two-phone energy | change |
| --- | ---: | ---: | ---: |
| FFN OpenCL | 0.1416 J | 0.1017 J | -28.19% |
| FFN HTP | 0.1416 J | 0.0451 J | -68.14% |
| Attention OpenCL | 0.1458 J | 0.2146 J | +47.19% |
| Attention HTP | 0.1458 J | 0.0457 J | -68.64% |

These energy rows are normalized projections, not power measurements. HTP and
OpenCL may also have different active power.

## Verdict

The valid simulation is OpenCL. Two OP15-equivalent phones should reduce FFN
latency by about 64% and active-phone energy by about 28%. The OpenCL attention
split should reduce latency by about 26%, but it increases active-phone energy
because the speedup is below 2x.

HTP is not currently correctness-eligible. FFN is close but exceeds a 1%
relative-L2 bound. Attention is numerically unacceptable. Full-width HTP FFN
also returned all zeros while the DSP logged `VTCM-TOO-SMALL`, despite the
backend reporting graph success. HTP values therefore describe optimization
headroom only.

Raw calculation and per-run artifacts are under
`results/phone_pair_layer_v1/run_20260731T1712/`, including
`TWO_OP15_SIMULATION.json`.
