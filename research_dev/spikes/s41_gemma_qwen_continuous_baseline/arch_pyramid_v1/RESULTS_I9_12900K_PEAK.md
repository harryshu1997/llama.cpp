# i9-12900K peak-compute probe

Date: 2026-08-05 EDT.

## Result

The physical desktop i9-12900K sustained a median 1.41 TFLOP/s on a dense
FP32 matrix multiplication using all 24 hardware threads. Three fresh-process
repetitions were:

| repetition | shape `[N,K] x [K,M]` | time | work | throughput |
| ---: | --- | ---: | ---: | ---: |
| 1 | `[4096,1024] x [1024,4096]` | 24.497 ms | 34.36 GFLOP | 1.40 TFLOP/s |
| 2 | `[4096,1024] x [1024,4096]` | 24.441 ms | 34.36 GFLOP | 1.41 TFLOP/s |
| 3 | `[4096,1024] x [1024,4096]` | 24.422 ms | 34.36 GFLOP | 1.41 TFLOP/s |

One multiply and one add count as two floating-point operations. Every
performance process exited successfully.

## Shape sweep

| shape `[N,K] x [K,M]` | throughput |
| --- | ---: |
| `[1024,1024] x [1024,1024]` | 1.38 TFLOP/s |
| `[2048,2048] x [2048,2048]` | 1.34 TFLOP/s |
| `[4096,1024] x [1024,4096]` | 1.40 TFLOP/s |
| `[4096,2048] x [2048,4096]` | 1.32 TFLOP/s |
| `[4096,4096] x [4096,4096]` | 1.13 TFLOP/s |
| `[8192,1024] x [1024,4096]` | 1.36 TFLOP/s |
| `[8192,2048] x [2048,4096]` | 1.17 TFLOP/s |

## Command

```sh
TP_RUNS=1 \
TP_SLICE=f32:4096x1024@4096 \
build-s21-cuda/bin/test-backend-ops perf -b CPU
```

This measures the llama.cpp CPU backend's dense FP32 GEMM path. It is not the
processor's theoretical instruction peak, whole-model speed, or wall-power
efficiency. Performance mode validates backend execution and timing; it does
not run a separate numerical oracle for this exact large shape.
