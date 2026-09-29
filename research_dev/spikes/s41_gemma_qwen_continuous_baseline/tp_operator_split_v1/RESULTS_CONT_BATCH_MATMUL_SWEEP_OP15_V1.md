# Continuous-batch matmul offload sweep on OP15

Date: 2026-08-05 EDT

This coarse sweep is superseded for Q/K/V and output-projection crossover
selection by `RESULTS_CONT_BATCH_MATMUL_FINE_OP15_V2.md`.

## Verdict

Continuous batching creates three additional useful offload regions beyond the
existing FFN split:

1. Split Q/K/V projection work at `M >= 32`.
2. Split the attention output projection at `M >= 128`.
3. Split the Q6_K vocabulary head for `M = 1..8`, with fewer phone shards as
   the batch grows.

Keep Q/K/V and output projection on the desktop CPU for decode and the common
`M = 8` batch. The modeled Q/K/V gain at `M = 8` is only 4.1%, which is below
a safe integration margin and would require 48 additional RPCs per graph.

These are physical CPU and HTP kernel measurements combined with the already
measured direct FunctionFS DMA transport. They are not yet a llama-server
end-to-end result. The combined rows assume perfect overlap between the CPU
branch and phone branch.

## Test setup

- Desktop: Intel i9-12900K, eight pinned P-core threads at default frequency.
- GPU: RTX 4060 Ti idle during the operator sweep.
- Phone: OP15, HTP v81, `HTP1`, four HVX threads and one HMX unit.
- Phone kernel: `6.12.23-android16-5-o-g227664cbe007-4k`.
- Model shapes: Gemma 4 12B, hidden 3840, Q 4096, K/V 2048 each,
  attention output 4096 x 3840, vocabulary 262144.
- Projection weights: Q4_0, resident and prepacked once.
- Vocabulary weights: Q6_K, resident and prepacked once.
- Samples: two warmups and seven measured iterations per shape.
- Direct DMA bandwidth used by the gate: 465 MB/s in both directions.
- Fixed direct-DMA RPC floor used by the gate: 0.235 ms.

The phone graph uses F16 DMA input and output for projections, including the
HTP casts in the measured compute interval. Q6_K consumes F32 activations, so
the vocabulary calculation includes HTP compute, HTP-to-phone output copies,
and phone top-k reduction. Only the F16 hidden state and compact top-k result
cross USB for the vocabulary split.

The phone started at 33.2 C and ended at 32.5 C. An `M=8` Q projection control
changed from 0.523 ms to 0.532 ms, and the Q6_K one-shard control remained at
2.55 ms. There was no sustained thermal slowdown.

## Projection result

The Q/K/V CPU baseline is four 3840 x 2048 Q4_0 matrices: two halves of Q plus
K and V. This preserves the exact total weight and operation count. The
separate full-Q and K+V measurements agree with this representation.

Times below are per layer. `Combined` is
`max(CPU remaining, phone compute + USB + RPC floor)`.

| M | CPU QKV | selected split | combined | gain | decision |
| ---: | ---: | --- | ---: | ---: | --- |
| 1 | 0.151 ms | none | 0.151 ms | 0.0% | CPU |
| 8 | 0.759 ms | half Q | 0.729 ms | 4.1% | reject: no margin |
| 32 | 2.943 ms | phone Q, CPU K+V | 1.880 ms | 36.1% | offload |
| 128 | 12.383 ms | phone K+V, CPU Q | 6.037 ms | 51.2% | offload |
| 512 | 48.154 ms | phone K+V, CPU Q | 23.942 ms | 50.3% | offload |

The output projection is split along K. The desktop and phone each evaluate a
2048 x 3840 partial result, then the desktop sums the two outputs. The table is
an ideal gate and does not include that final vector add.

| M | CPU output projection | selected split | combined before join | gain before join | decision |
| ---: | ---: | --- | ---: | ---: | --- |
| 1 | 0.085 ms | none | 0.085 ms | 0.0% | CPU |
| 8 | 0.383 ms | none | 0.383 ms | 0.0% | CPU |
| 32 | 1.472 ms | none | 1.472 ms | 0.0% | CPU |
| 128 | 6.080 ms | 50% K split | 3.946 ms | 35.1% | offload |
| 512 | 24.044 ms | 50% K split | 14.380 ms | 40.2% | offload |

The 2.13 ms and 9.66 ms margins in the last two rows are large enough to
justify a real transport-and-join test. The `M = 32` phone branch is already
1.495 ms before the join, so it cannot beat the 1.472 ms CPU path.

Across 48 layers, the ideal projection savings are about 51 ms at `M = 32`,
407 ms at `M = 128`, and 1.63 s at `M = 512`. These values cover Q/K/V and
output projections only, not the complete layer.

## Vocabulary-head result

Each phone shard contains 32768 Q6_K vocabulary rows. The HTP-repacked size is
150 MiB per shard. The desktop computes the remaining rows at the same time,
and the phone returns top-32 candidates per token.

| M | full CPU head | p50 optimum | offloaded rows | combined | gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 27.065 ms | 4 shards | 131072 | 13.243 ms | 51.1% |
| 2 | 27.842 ms | 4 shards | 131072 | 13.554 ms | 51.3% |
| 4 | 29.129 ms | 4 shards | 131072 | 17.969 ms | 38.3% |
| 8 | 32.200 ms | 2 shards | 65536 | 24.224 ms | 24.8% |

At `M = 4`, three shards give 18.124 ms and much more phone-side slack. That
is the safer initial policy because it gives up only 0.155 ms relative to the
p50 optimum. The recommended starting policy is therefore:

- `M = 1..2`: four shards, or 50% of vocabulary rows.
- `M = 3..4`: three shards, or 37.5% of vocabulary rows.
- `M = 5..8`: two shards, or 25% of vocabulary rows.

This vocabulary route is not transparent to every llama.cpp sampler. A
top-k-only response cannot reproduce arbitrary vocabulary-wide logit bias,
penalties, grammar masks, or samplers that need more candidates. Exact server
support needs either a compatible sampler contract or phone-side sampling.

## Resident-memory budget

An upper-bound HTP1 placement for all 48 layers is:

| resident state | HTP1 memory |
| --- | ---: |
| all Q/K/V projections | about 810 MiB |
| all output projections | about 405 MiB |
| four Q6_K vocabulary shards | 600 MiB |
| total | about 1815 MiB |

This fits within the observed 3200 MiB HTP1 virtual-memory budget. The
existing FFN suffix remains on HTP0. HTP0 and HTP1 are separate mappings but
share phone compute and memory resources, so dependent phases should remain
sequential.

## Integration gate

The initial scheduler policy should be:

```text
M <= 8:    CPU QKV, CPU output projection, dynamic LM-head split
9 <= M < 32: CPU QKV and output projection
32 <= M < 128: split QKV, CPU output projection
M >= 128: split QKV and split output projection
```

The interval `9 <= M < 32` remains conservative because it was not directly
swept. Before server integration, the passing projection cases need one real
FunctionFS DMA worker test that includes the output-projection join. Then the
policy can be added behind an opt-in switch and checked on the original
BurstGPT trace.

## Evidence

Reproducible harness:

- `continuous_matmul_sweep_v1/backend_matmul_bench.cpp`
- `continuous_matmul_sweep_v1/run_projection_sweep.sh`
- `continuous_matmul_sweep_v1/run_lm_head_sweep.sh`
- `continuous_matmul_sweep_v1/analyze_sweep.py`

Raw physical results and the generated `ANALYSIS.json` are retained on the
4060 Ti host under `/home/zhihao/s41-cont-matmul-v1/results/`.
