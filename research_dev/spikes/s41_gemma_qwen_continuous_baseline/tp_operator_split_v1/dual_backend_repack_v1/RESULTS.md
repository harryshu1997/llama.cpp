# OP15 dual-backend Q4 screen results

Tested on OP15 CPH2749 using Hexagon HTP v81 and Adreno OpenCL. The benchmark
uses the Gemma4 dense FFN dimensions K=3840 and NFF=15360. It assigns disjoint
columns from a 9,664-column phone suffix to HTP and GPU. All three FFN weight
tensors contain the same Q4_0 values on both paths; only their backend-native
layouts differ.

Each backend calls `ggml_backend_tensor_set` once per weight tensor before
warmup. There is no weight upload, quantization, or layout conversion in a
timed iteration. The input and partial outputs share one rpcmem allocation,
which OpenCL imports by DMA-BUF. Two persistent host threads launch the HTP and
GPU work concurrently, then the phone CPU sums the partial outputs.

## Correctness-passing decode results

The correctness threshold is relative L2 <= 0.005 against the intact HTP
control.

| GPU columns | HTP columns | HTP control p50 | Dual p50 | Speedup | Relative L2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 9,152 | 1.052 ms | 1.699 ms | 0.619x | 0.002939 |
| 512 | 9,152 | 1.057 ms | 1.688 ms | 0.626x | 0.002939 |
| 1,024 | 8,640 | 1.068 ms | 1.794 ms | 0.595x | 0.004525 |

The 512-column repeat is the best valid boundary measured. Its one-time pack
cost was 27.6 ms for the intact HTP control, 26.1 ms for the HTP shard, and
20.8 ms for the GPU shard. The disjoint HTP and GPU weights occupy 56.56 MiB
and 3.16 MiB, respectively, equal to the 59.73 MiB control rather than a second
full copy.

At this boundary, the HTP shard alone took 0.98-0.99 ms and the GPU shard alone
took 1.46-1.54 ms. Concurrent execution increased the GPU leg to about 1.65 ms,
while launch skew was about 0.004 ms and the CPU merge was below 0.003 ms. The
bottleneck is the Adreno Q4 matmul, with an additional shared-memory contention
cost; it is not repacking, launch synchronization, or result merging.

## Larger GPU shard diagnostics

These configurations exceeded the strict cross-backend numerical threshold,
so they are diagnostic timing data rather than valid outputs. None approached
a speedup.

| Batch M | GPU columns | HTP control p50 | GPU solo p50 | Dual p50 | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,472 | 1.054 ms | 1.628 ms | 1.901 ms | 0.555x |
| 1 | 3,072 | 1.042 ms | 2.414 ms | 2.488 ms | 0.419x |
| 8 | 3,072 | 2.678 ms | 9.103 ms | 8.831 ms | 0.303x |
| 1 | 8,192 | 1.056 ms | 3.265 ms | 3.313 ms | 0.319x |
| 32 | 8,192 | 2.931 ms | 13.087 ms | 13.292 ms | 0.221x |
| 128 | 8,192 | 3.207 ms | 32.888 ms | 33.239 ms | 0.096x |

## Decision

One-time repacking and zero-copy shared activations work as designed, but the
current OpenCL Q4 kernels cannot replace any measured HTP column range
profitably. Full-model GPU plus HTP scheduling is therefore not enabled from
this experiment.

A future screen should use a different prepared GPU representation or kernel,
such as a one-time Q4-to-F16 GPU layout for sufficiently large prefill batches,
or a persistent quantized matmul kernel. It should pass this one-layer control
before being integrated into full-model scheduling.
