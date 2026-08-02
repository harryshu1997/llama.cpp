# Gemma global-attention xmem latency ceiling

Date: 2026-07-31 EDT.

Verdict:
`LATENCY_FIT_PASS; CORRECTNESS_INVALID; P90_FAIL; PHYSICAL_4060_TREATMENT_NOT_RUN`.

## Scope

This is a latency-only successor to the correctness-gated Gemma global-attention
context-shard probe. It measures one projected-Q plus resident-f16-KV attention
core with:

- 1,048,576 cached tokens;
- 16 query heads and one KV head;
- head dimension 512;
- an RTX A6000 CUDA prefix and an OP15 Adreno 840 suffix;
- direct AOA USB transport;
- a compact 16,428-byte request and 16,564-byte response.

It is not a complete layer or full-model result. The A6000 was locked at
990/5001 MHz. OP15 reported `max_gpuclk=500000000` and
`thermal_pwrlevel=9`; no thermal or clock protection was bypassed.

## Latency-only path

Both phone matmuls use the opt-in Adreno xmem f16-by-f32 GEMM:

1. xmem computes QK for the resident phone KV suffix;
2. the existing OpenCL softmax runs on the scores;
3. xmem computes probability-times-V;
4. the phone returns the normalized state and one score/probability anchor per
   head;
5. the host merges the CUDA and phone shards with the existing online-softmax
   merge.

The phone enables the existing weight-prepack cache and a new opt-in scratch
image cache. The image cache is disabled by default. It reuses the two xmem
source/destination images for repeated single-stream requests of the same
shape.

The host adds explicit `latency_corun` and `latency_steady_corun` modes. These
modes print all numerical errors but do not reject the timed loop. Their
verdict is always labeled `INVALID_LATENCY_ONLY` when the normal gate fails.
The worker also accepts a bounded request count so that it exits cleanly and
flushes OpenCL profiling evidence.

## Shard sweep

All rows use the real A6000 plus OP15 path. The 24K row predates the scratch
image cache; the 8K-10K rows enable it. Times are medians in milliseconds.

| Phone suffix | CUDA full | CUDA prefix | Phone total | Split total | Change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 24,576 | 9.286914 | 9.061260 | 16.897223 | 16.962749 | +82.652% |
| 10,240 | 9.284429 | 9.176451 | 9.072562 | 9.432584 | +1.596% |
| 9,216 | 9.282034 | 9.180368 | 8.769078 | 9.345306 | +0.682% |
| 8,192 | 9.282255 | 9.180550 | 8.244780 | 9.203865 | -0.845% |

The 8K row is the median of three independent release-build repetitions. Its
per-repetition changes were -0.851%, -0.823%, and -0.862%. The phone has about
0.94 ms of median slack under the CUDA prefix, so CUDA remains the critical
path.

The p90 result does not pass. Across the same repetitions, the median of the
control p90 values is 9.293427 ms and the median of the treatment p90 values is
10.810555 ms, a 16.325% regression. Most variation is in phone compute and USB
OUT/IN readiness.

## Phone breakdown

The median-of-repetition release result at the selected 8K point is:

| Component | Time |
| --- | ---: |
| OpenCL graph | 6.814 ms |
| USB OUT | 0.540 ms |
| Host USB IN wait, including phone execution | 7.673 ms |
| Phone end-to-end | 8.245 ms |
| CUDA prefix | 9.181 ms |
| Host merge | 0.0066 ms |

The profiled 8K graph attributes 4.473 ms to GPU kernels:

| Kernel group | Time |
| --- | ---: |
| QK xmem GEMM | 1.190 ms |
| Softmax | 0.148 ms |
| Probability-times-V xmem GEMM | 2.871 ms |
| Pack, store, concat, and anchor copies | 0.264 ms |

The controlled profiled comparison shows that scratch image reuse lowers graph
time from 6.953 to 6.311 ms (-9.23%) and phone end-to-end time from 8.195 to
7.842 ms (-4.31%). It changes split time only from 9.213 to 9.206 ms because
CUDA is already the median critical path.

## Correctness limitation

The selected 8K xmem path is not numerically eligible:

```text
phone relative L2   0.353681791
final relative L2   0.00515149041
non-finite values   0
```

The phone component exceeds its 0.03 gate by more than 10x. The final error is
smaller only because the invalid phone suffix contributes less than one percent
of the 1M-token context. This result cannot authorize runtime integration,
quality, energy, or full-model claims.

## Interpretation

The requested latency-first question is answered: xmem can make the real
phone leg fit under CUDA at a 1M context, but only for an 8K suffix under the
current 500 MHz phone cap. This removes about 0.078 ms from one global-attention
core. It offloads 0.78% of that layer's context-dependent attention work and
stores 16 MiB of KV for that layer on the phone.

Gemma-4-12B has eight global-attention layers. Even an ideal repetition of this
result across all eight saves only about 0.63 ms per decoded token before
full-model integration overhead. Tail latency and FP32 accumulation therefore
matter more than increasing the suffix today.

The physical RTX 4060 Ti server-only 1M control is 10.225790 ms, while this
A6000 control is about 9.282 ms. The A6000 treatment is not a physical 4060 Ti
treatment and must not be reported as one.

## Next bounded step

Keep the 8K shard and the latency-matched schedule fixed. Replace only the
xmem accumulator/probability precision needed to recover the phone component
gate, then rerun the same three repetitions. Do not expand to a full layer
until correctness and p90 both pass.

Raw evidence is under:

`results/gemma_global_attention_xmem_latency_v1/run_20260731T164535Z/`.
