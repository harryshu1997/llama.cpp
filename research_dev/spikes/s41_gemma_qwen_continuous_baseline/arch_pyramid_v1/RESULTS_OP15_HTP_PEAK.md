# OP15 HTP peak-compute probe

Date: 2026-08-05 EDT.

## Result

The real OP15 sustained a median 11.66 TFLOP/s on a saturated FP16 HMX
matrix multiplication. Three fresh-process repetitions were:

| repetition | shape `[N,K] x [K,M]` | time | work | throughput |
| ---: | --- | ---: | ---: | ---: |
| 1 | `[8192,2048] x [2048,4096]` | 11.842 ms | 137.44 GFLOP | 11.61 TFLOP/s |
| 2 | `[8192,2048] x [2048,4096]` | 11.776 ms | 137.44 GFLOP | 11.67 TFLOP/s |
| 3 | `[8192,2048] x [2048,4096]` | 11.792 ms | 137.44 GFLOP | 11.66 TFLOP/s |

The operation count treats one multiply and one add as two floating-point
operations. The HTP runtime reported eight HVX threads, one HMX unit, and
8 MiB VTCM. Every performance process exited successfully.

## Saturation sweep

The initial `[4096,4096] x [4096,M]` FP16 sweep showed why the earlier
decode-style GEMV result was not a peak-compute measurement:

| M | throughput |
| ---: | ---: |
| 32 | 0.851 TFLOP/s |
| 64 | 1.77 TFLOP/s |
| 128 | 3.51 TFLOP/s |
| 256 | 6.68 TFLOP/s |
| 512 | 6.61 TFLOP/s |
| 1024 | 8.21 TFLOP/s |
| 1536 | 8.66 TFLOP/s |
| 2048 | 9.74 TFLOP/s |
| 3072 | 9.98 TFLOP/s |
| 4096 | 9.89 TFLOP/s |

A shape sweep then found 11.40 TFLOP/s at
`[4096,2048] x [2048,4096]`, 10.94 TFLOP/s at
`[8192,1024] x [1024,4096]`, and 11.89 TFLOP/s in the exploratory
`[8192,2048] x [2048,4096]` pass. The three fresh repetitions above are the
reported result.

The Q8-weight HMX path reached 16.62 TOP/s-equivalent at
`[4096,4096] x [4096,3072]`. That value is not labeled TFLOP/s in the
diagram because the weights are quantized and its operation semantics differ
from the FP16 measurement.

## Command

The existing arbitrary-shape backend benchmark was used without source or
runtime changes:

```sh
TP_RUNS=1 \
TP_SLICE=f16:8192x2048@4096 \
LD_LIBRARY_PATH=. \
ADSP_LIBRARY_PATH=. \
./tbo-htp perf -b HTP0
```

This is an accelerator throughput microbenchmark, not whole-model speed and
not an accelerator-rail power measurement. Performance mode validates backend
execution and timing; it does not run a separate numerical oracle for this
exact large shape.
