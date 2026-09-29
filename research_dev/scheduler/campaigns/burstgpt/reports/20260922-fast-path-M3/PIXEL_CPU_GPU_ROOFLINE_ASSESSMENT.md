# Pixel FFN bandwidth and compute headroom - 2026-09-23 20:51 UTC

Evidence/arithmetic **PASS**. Physical DRAM peak and sustained compute peak are **NOT ESTABLISHED**. No new hardware run.

| Metric | Full FFN | Half FFN |
| --- | ---: | ---: |
| Nominal F16 weight traffic | 534.774 MB | 267.387 MB |
| Matrix work | 534.774 MFLOP | 267.387 MFLOP |
| Measured worker latency | 13.05775 ms | 7.07694 ms |
| Effective weight bandwidth | 40.955 GB/s | 37.783 GB/s |
| Matrix work per worker-second | 40.955 GFLOP/s | 37.783 GFLOP/s |
| Relative to measured 46.515 GB/s streaming | 88.05% | 81.23% |
| Conditional memory-only time at that rate | 11.497 ms | 5.748 ms |
| Latency reduction to that conditional reference | 11.95% | 18.77% |

The stream result uses one CPU core plus GPU on disjoint 512 MiB allocations. The FFN uses four pinned CPU threads plus GPU and performs three projections, activation and merging. Clocks are unlocked. The percentages compare two workloads; they are not physical bus utilization. The conditional times omit all non-weight traffic/overheads and assume the stream rate is transferable. A new layout could also change the achievable rate, so this is not a hard limit.

For one row, three F16 matrices contain `3 * 5120 * 17408 * 2` bytes and require `3 * 2 * 5120 * 17408` matrix FLOPs. Nominal arithmetic intensity is therefore one FLOP per weight byte. Input/output traffic lowers it slightly. This explains why about41 GB/s yields about41 GFLOP/s for the full worker even when the arithmetic units could execute far more. This is the usual bandwidth-bound GEMV regime described in [NVIDIA's matrix multiplication guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html).

The [published DXT-48-1536 specification](https://www.imaginationtech.com/product/img-dxt-48-1536/) gives 1,536 FP32 FLOPs per clock. At1.094 GHz seen in a before/after snapshot, that implies a conditional theoretical GPU roof of1.680 TFLOP/s. The snapshot does not prove the frequency during FFN kernels. GPU branch matrix work divided by its backend interval is24.995 GFLOP/s; the full CPU+GPU worker rate is40.955 GFLOP/s. Neither is a standalone compute-throughput benchmark, and the combined number must not be treated as GPU utilization.

Candidate improvements, not measured new wins:

1. Jointly sweep CPU thread count/core placement and channel ratio. The fine ratio sweep fixed four CPU threads. The raw-stream winner uses one fast CPU core, but that does not establish the best FFN configuration. Test1/2/3/4/6 CPU threads with a ratio sweep for each.
2. For a larger single-token improvement, retain packed original quantized weights and fuse dequantization into dot kernels. This reduces weight bytes; simply changing arithmetic precision while retaining F16 weights does not. Metadata and decode costs matter. Preserve the intended F16-reference weight values/rounding and validate tensor outputs and eventual full-model tokens.
3. For higher useful throughput, batch independent same-model request rows2/4,then8 using tiled matrix kernels that reuse each weight across rows. Ideal matrix FLOPs per weight byte grows approximately with batch size. This does not guarantee equal per-request latency or eliminate queuing. The current custom shader fast path explicitly requires `NUM_COLS == 1`; the multi-row path needs its own optimization and qualification, including a ratio search for each batch size.
4. Profile and reduce dispatch/graph overhead; consider packing each backend's two full-width blocks into one, reducing full FFN GEMVs from12 to6 while retaining all weights. Half-width requests already use one block per backend. Larger down-projection K, weight layout and reduction order need new validation. Output merge itself is only0.0121 ms for the full FFN, so eliminating it cannot produce a large gain. The0.547 ms CPU wait includes useful remaining GPU work and is not all removable overhead.

Prior up-matvec/SwiGLU fusion passed correctness but regressed full GPU-only latency by3.191%; it removed only136 KiB of intermediate traffic versus510 MiB of weights. That experiment does not establish a mixed-path benefit. Avoid treating fewer dispatches alone as a guaranteed speedup.

A compute-only FMA/GEMM benchmark and memory-controller counters with clock telemetry would be needed to quantify sustained arithmetic peak and actual DRAM utilization. They have not been run here.

- [Exact calculations and proposed tests](PIXEL_CPU_GPU_ROOFLINE_ASSESSMENT.json)
- [Tuned FFN results](PIXEL_CPU_GPU_RATIO_RESULTS.md)
- [Streaming measurements](PIXEL_BANDWIDTH_RESULTS.md)
- [Earlier fusion result](PIXEL_SWIGLU_RESULTS.md)
