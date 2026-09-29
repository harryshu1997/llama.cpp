# OP15 Q4-to-F16 xmem FFN results

Date: 2026-08-04 EDT.

Verdict: `XMEM_PATH_PASS; CORRECTNESS_LIMITED; NO_SPLIT_SPEEDUP`.

## Scope

This is a one-layer phone-internal screen, not a full-model or USB result. It
uses the Gemma4 dense FFN dimensions K=3840 and NFF=15360 and keeps a fixed
9,664-column phone suffix. HTP runs a Q4_0 shard. Adreno receives a disjoint
shard reconstructed in F16 from the same Q4_0 values.

The GPU preparation happens before warmup:

1. Generate the Q4_0 source values and dequantize them to F16 on the phone CPU.
2. Upload the three F16 FFN weight tensors once.
3. Run one untimed GPU graph to create the three cached xmem weight layouts and
   the reusable scratch images.
4. Reuse those allocations for every measured graph.

The input and partial outputs remain in shared rpcmem imported by OpenCL through
DMA-BUF. Persistent host threads launch HTP and GPU concurrently. No weight
conversion, upload, or xmem weight prepack occurs in a timed iteration.

Xmem requires token batch M >= 16 for these matmuls. Batch-1 decode therefore
continues to use the native Q4 path and is not part of this screen.

## Physical sweep

The physical OP15 CPH2749 sweep covered GPU column counts 64, 128, 256, 512,
1,024, 1,472, 2,048, 3,072, 4,096, and 8,192 at M=16, 32, 64, and 128. Each
row used three warmups and ten timed iterations. The table shows the lowest
dual p50 found at each batch among correctness-passing configurations.

| M | GPU columns | HTP control p50 | GPU solo p50 | Dual p50 | Speedup | Relative L2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 64 | 2.789 ms | 3.459 ms | 4.331 ms | 0.644x | 0.000907 |
| 32 | 128 | 2.948 ms | 3.671 ms | 4.704 ms | 0.627x | 0.001322 |
| 64 | 256 | 3.122 ms | 4.508 ms | 5.385 ms | 0.580x | 0.002088 |
| 128 | 64 | 3.753 ms | 4.776 ms | 5.840 ms | 0.643x | 0.001022 |

Every eligible split is slower than intact HTP. Even the smallest GPU shard is
slower by itself than the entire HTP control. Removing 64-256 columns also
barely shortens the HTP leg, so overlap cannot hide the GPU work.

Shared-LPDDR contention is material. At M=16 and 64 GPU columns, the GPU leg
changes from 3.459 ms solo to 4.139 ms concurrent. The CPU merge costs about
0.040 ms at M=16 and about 0.32 ms at M=128, so merging is not the primary
bottleneck.

## Does xmem accelerate Adreno?

Yes. Matched controls show that xmem accelerates Adreno relative to both the
native Q4 and generic F16 kernels. It still does not reach HTP speed.

| M | GPU columns | GPU representation | GPU solo p50 | Dual p50 | Relative L2 |
| ---: | ---: | --- | ---: | ---: | ---: |
| 16 | 512 | native Q4_0 | 6.573 ms | 6.861 ms | 0.002860 |
| 16 | 512 | F16 xmem | 4.208 ms | 5.312 ms | 0.002842 |
| 128 | 512 | native Q4_0 | 11.430 ms | 11.997 ms | 0.002971 |
| 128 | 512 | generic F16 | 20.406 ms | 20.882 ms | 0.000306 |
| 128 | 512 | F16 xmem | 7.997 ms | 8.776 ms | 0.002954 |
| 128 | 8,192 | native Q4_0 | 33.034 ms | 33.248 ms | 0.016369 |
| 128 | 8,192 | generic F16 | 37.852 ms | 38.153 ms | 0.000465 |
| 128 | 8,192 | F16 xmem | 32.518 ms | 32.770 ms | 0.016314 |

For the 512-column shard, xmem makes the Adreno leg 1.56x faster than native
Q4_0 at M=16 and 1.43x faster at M=128. At M=128 it is also 2.55x faster than
generic F16. At 8,192 columns, however, it is only 1.02x faster than native
Q4_0.

## One-time preparation and memory

For the M=16, 512-column case, GPU preparation took about 17.0 ms for host Q4
reconstruction, 3.7 ms for the F16 upload, and 15.2 ms for the first xmem graph.
The first graph includes cache creation and useful FFN computation; all of it is
excluded from steady-state timing.

The F16 source weights and cached xmem layout currently coexist:

| GPU columns | HTP Q4 shard | GPU F16 source | Xmem cache | Dual residency |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 59.33 MiB | 1.41 MiB | 1.41 MiB | 62.14 MiB |
| 512 | 56.56 MiB | 11.25 MiB | 11.25 MiB | 79.06 MiB |
| 8,192 | 9.10 MiB | 180.00 MiB | 180.00 MiB | 369.10 MiB |

The intact Q4 control is 59.73 MiB. A production cache could potentially own
the packed allocation and release its F16 source copy, but that cannot recover
the measured latency gap.

## Kernel-path proof

A current-source profiling build was run at M=16 with 512 GPU columns. The
profile recorded:

```text
3  adreno_xmem_prepack_weight_f16
12 adreno_xmem_pack_src_f32
12 kernel_gemm_xmem_f16_f32_os8
12 adreno_xmem_store_dst_f32
```

There are three prepack calls, one per FFN weight tensor, despite four GPU graph
executions in the profiled process. Each graph has three xmem GEMMs. This proves
that steady-state graphs reuse the prepared weights.

The profiled first prepack kernels took 0.998, 0.991, and 1.051 ms of device
time. A later graph spent about 1.19 ms, 1.19 ms, and 0.51 ms in its three xmem
GEMMs, before driver, input-pack, output-store, activation, and synchronization
overhead.

## Correctness limit and decision

The relative-L2 gate is 0.005 against intact HTP. GPU shares through 1,024
columns pass. The 1,472-column share is already slightly over the gate, and the
8,192-column share reaches about 0.0163. Generic F16 remains accurate at the
large share, which points to xmem's half-precision accumulation and output path
rather than the one-time Q4-to-F16 reconstruction.

Do not integrate this dense-FFN HTP plus GPU split into full-model scheduling.
Use HTP alone for this phone suffix. A future GPU attempt needs both a more
accurate accumulator and enough steady-state throughput for the GPU leg to fit
under the HTP shard; eliminating preparation or merge overhead is insufficient.
