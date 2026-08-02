# OP15-equivalent fleet projection and CPU comparison

Date: 2026-07-31 EDT

Status: `REAL_SINGLE_PHONE_SLICES; SIMULATED_FLEET; CPU_COMPONENT_MEASURED; NOT_END_TO_END`

## Scope

This report projects identical OP15 phones from real equal-slice measurements.
It does not claim a physical multi-phone run. Fleet latency is the measured
median of one representative equal slice plus a conservative serial host sum:

```
T_fleet(N) = T_OP15_slice(N) + (N - 1) * 0.0038 ms
```

The projection requires all phones to run concurrently with independent AOA
links and USB buses. It excludes launch skew, shared-bus contention, and phone
heterogeneity. N=2 uses the maximum of two measured complementary slices plus
the same host sum. N>=4 uses one representative slice and has not executed all
offsets or an aggregate correctness check.

The FFN test is a complete Gemma-like Q8_0 FFN residual at M=1 with K=3840 and
NFF=15360. The attention test is a complete Qwen Q8_0 attention residual at
M=1 with K=5120, eight GQA groups, and KV=8192. These belong to different model
geometries and must not be added to form one layer.

Raw phone records are under:

- `results/phone_pair_layer_v1/run_20260731T1712/`
- `results/phone_fleet_projection_v1/run_20260731T1745/`

## Gemma-like FFN versus all CPU

The i9-12900K Q8_0 matmul core was remeasured with `TP_SLICE`: gate and up are
1.26665 ms each and down is 1.22062 ms, for 3.75392 ms total. This is a lower
bound for the complete CPU FFN because it omits SiLU, multiply, residual, and
graph overhead.

OpenCL is the valid phone backend. The fleet active energy column assumes
4.5 W per phone and is a normalized estimate, not measured power.

| OP15-equivalent phones | projected OpenCL median | CPU / phones | fleet active energy | evidence status |
| ---: | ---: | ---: | ---: | --- |
| 1 | 31.465 ms | 0.119x | 0.1416 J | real full operator |
| 2 | 11.297 ms | 0.332x | 0.1017 J | two real slices, aggregate checked |
| 4 | 7.715 ms | 0.487x | 0.1389 J | one real representative slice |
| 8 | 5.225 ms | 0.718x | 0.1881 J | one real representative slice |
| 16 | 4.573 ms | 0.821x | 0.3293 J | one real representative slice |
| 32 | 4.284 ms | 0.876x | 0.6169 J | one real representative slice |
| all CPU | >=3.754 ms | 1.000x | not measured | real matmul core |

The current OpenCL path does not beat the CPU even at 32 idealized phones. Its
latency saturates near 4 ms because each phone still pays AOA, graph, and
OpenCL dispatch overhead. Active-phone energy is best at N=2 and then rises.

HTP gives the following optimization ceiling:

| OP15-equivalent phones | projected HTP median | CPU / phones | fleet active energy | numerical status |
| ---: | ---: | ---: | ---: | --- |
| 2 | 5.012 ms | 0.749x | 0.0451 J | aggregate rel-L2 1.225%, over 1% gate |
| 4 | 3.569 ms | 1.052x | 0.0642 J | representative component only |
| 8 | 3.022 ms | 1.242x | 0.1088 J | representative component only |
| 16 | 2.784 ms | 1.348x | 0.2005 J | representative component only |
| 32 | 2.751 ms | 1.364x | 0.3962 J | representative component only |

HTP could cross the CPU around N=4, but it is not eligible. The representative
HTP/OpenCL rel-L2 errors for N=4, 8, 16, and 32 are 1.240%, 1.182%, 1.157%,
and 1.051%. The full-width HTP FFN also returned all zeros after a
`VTCM-TOO-SMALL` DSP failure. These rows are performance headroom only.

## Qwen attention versus all CPU

The CPU Q8_0 projections and output matmuls measured 0.29652 and 0.11260 ms.
CPU flash attention at KV=8192 measured 0.24234 ms for GQA ratio 1 and
0.83511 ms for ratio 4. Linear interpolation to Qwen's ratio 5 is 1.03270 ms.
This gives:

- 1.44182 ms with hot projection weights.
- About 2.36963 ms when 66.85 MB of Q8 projection weights stream at the
  measured 50 GB/s CPU bandwidth.

Both are component estimates, not an exact complete CPU attention graph. The
table uses the more conservative 2.370 ms streaming estimate.

| OP15-equivalent phones | OpenCL median | CPU / phones | HTP median | CPU / HTP | status |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 32.394 ms | 0.073x | not run | - | real full OpenCL operator |
| 2 | 23.839 ms | 0.099x | 5.079 ms | 0.467x | OpenCL aggregate passes; HTP aggregate fails |
| 4 | 14.095 ms | 0.168x | 3.861 ms | 0.614x | representative group-count 2 slice |
| 8 | 10.039 ms | 0.236x | 3.067 ms | 0.773x | representative group-count 1 slice |
| all CPU | about 2.370 ms | 1.000x | - | - | streaming component estimate |

The CPU remains about 1.29x faster than even the projected eight-phone HTP
route. HTP group-count 4 is numerically invalid at 42.6% aggregate rel-L2.
The smaller representative group-count 2 and 1 components are below 1%, but
all offsets and their aggregate have not been checked. This decomposition
stops at eight phones because Qwen has eight KV groups. More phones require a
new intra-group head or context split and another merge.

## Whole-model CPU comparison

The true Gemma-4-12B Q8_0 all-CPU engine baseline on the same i9-12900K is
426 ms/token, or 2.35 tok/s. It remains 2.34 tok/s while the GPU decodes a
second model.

The following separate whole-model estimate reuses the measured OP15 Q8 slice
sweep, two Megatron-style collectives per layer, 48 layers, and the previously
measured 1.77x engine/component derate. It is not derived from the complete
FFN worker above and excludes a physical fleet run.

| OP15-equivalent phones | current two USB buses | versus CPU | one bus per phone | versus CPU |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 315 ms, 3.18 tok/s | 1.35x | 315 ms, 3.18 tok/s | 1.35x |
| 4 | 336 ms, 2.97 tok/s | 1.27x | 239 ms, 4.18 tok/s | 1.78x |
| 8 | 431 ms, 2.32 tok/s | 0.99x | 198 ms, 5.04 tok/s | 2.14x |
| 16 | 677 ms, 1.48 tok/s | 0.63x | 191 ms, 5.24 tok/s | 2.23x |
| 32 | 677 ms, 1.48 tok/s | 0.63x | 184 ms, 5.43 tok/s | 2.31x |
| all CPU | 426 ms, 2.35 tok/s | 1.00x | 426 ms, 2.35 tok/s | 1.00x |

With the current two-bus topology, N=2 to N=4 is the useful range and N>=8
falls to CPU parity or worse because collectives serialize. With one genuinely
independent link per phone, useful scaling reaches about N=16 and then
saturates. These full-model rows are planning bounds, not end-to-end results.

## Decision

More phones do not automatically save latency or energy. The next useful
physical experiment is N=2 on independent links. Before scaling further:

1. Fix HTP error reporting so DSP graph failure cannot publish a successful
   zero output.
2. Make HTP FFN pass an aggregate numerical gate for every slice.
3. Use one USB controller or AOA link per phone, or stop at N=2 to N=4.
4. Do not spend hardware effort on the current attention split; CPU is faster
   at KV=8192 even against the HTP projection.

