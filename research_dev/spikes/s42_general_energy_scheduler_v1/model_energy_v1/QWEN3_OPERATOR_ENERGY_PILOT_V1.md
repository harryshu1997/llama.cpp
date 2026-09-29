# Qwen3-14B Q4_K_M local-layer operator energy pilot V1

Date: 2026-08-06 EDT.

Verdict: `AGGREGATE_HELDOUT_PASS; PER_OPERATOR_ATTRIBUTION_ESTIMATED; QWEN_PHONE_ENERGY_PENDING`.

## Scope

This pilot measures one Qwen3-14B-shaped decode layer on the physical RTX
4060 Ti. The graph uses deterministic synthetic mixed-quantization block
weights, `M=1`, hidden width 5,120, FFN width 17,408, 40 query heads, eight
KV heads, and head width 128. Model loading is outside the paid interval.

The tensor types match the deployed Qwen3-14B Q4_K_M checkpoint: Q, K, O,
gate, and up use Q4_K; V and down use Q6_K. Q4_K uses 144 physical bytes per
256 weights and Q6_K uses 210 physical bytes per 256 weights.

No GGUF is loaded in this probe. It exercises the deployed dimensions,
quantization blocks, ggml graph, and CUDA kernels, but remains a complete-layer
proxy. It omits Q/K normalization, RoPE, embedding, the Q6_K vocabulary head,
sampling, and full llama-server overhead.

The table is for `KV=512`. It is not a complete-model or CPU-plus-phone energy
table.

## Estimated operator energy

The RTX 4060 Ti model is:

```text
E_layer(KV) = 105.900608 mJ + KV * 0.001486519 mJ
```

The fixed term is divided among operators using physical mixed-quantization
weight and activation traffic. The attention row also receives the
KV-dependent term.

| operator | shape or traffic at KV=512 | estimated mJ/layer | layer share |
| --- | --- | ---: | ---: |
| attention RMSNorm and scale | 0.098 MiB | 0.051 | 0.05% |
| Q projection | Q4_K, 14.102 MiB | 7.406 | 6.94% |
| K projection | Q4_K, 2.836 MiB | 1.490 | 1.40% |
| V projection | Q6_K, 4.125 MiB | 2.167 | 2.03% |
| QK, softmax, and AV | 0.078 MiB fixed plus 2.312 MiB KV | 0.802 | 0.75% |
| O projection | Q4_K, 14.102 MiB | 7.406 | 6.94% |
| residual and FFN RMSNorm | 0.289 MiB | 0.152 | 0.14% |
| FFN gate projection | Q4_K, 47.898 MiB | 25.157 | 23.59% |
| FFN up projection | Q4_K, 47.898 MiB | 25.157 | 23.59% |
| SiLU | 0.133 MiB | 0.070 | 0.07% |
| gate times up | 0.199 MiB | 0.105 | 0.10% |
| FFN down projection | Q6_K, 69.812 MiB | 36.667 | 34.38% |
| final residual add | 0.059 MiB | 0.031 | 0.03% |
| **predicted complete layer** | **203.941 MiB physical traffic** | **106.662** | **100.00%** |
| **measured complete layer** | **held-out KV=512** | **105.794** | **-** |

The three FFN projections account for 81.55% of predicted layer energy. The
down projection alone accounts for 34.38%, but it cannot be treated as an
independent measured row. Only the complete-layer sum has a physical accuracy
check.

## Physical validation

Three fresh executions were measured at each context. GPU-board energy uses
50 ms `nvidia-smi` samples, arrival timestamps, and zero-order-hold
integration over markers emitted by the layer runner. The GPU used normal
DVFS; no frequency lock was applied.

KV=136 and KV=8192 fit the fixed and KV coefficients. KV=512 is held out.

| KV entries | role | measured median mJ/layer | median latency | model mJ/layer | error |
| ---: | --- | ---: | ---: | ---: | ---: |
| 136 | fit | 106.103 | 0.842 ms | 106.103 | 0.00% |
| 512 | held out | 105.794 | 0.867 ms | 106.662 | +0.82% |
| 8192 | fit | 118.078 | 1.020 ms | 118.078 | 0.00% |

The held-out absolute error is 0.82%, below the 10% pilot gate. Repetition
half-range is 1.17% at KV=136, 0.60% at KV=512, and 1.18% at KV=8192. The
largest power-sample gap is 50.95 ms.

This validates the sum for one mixed-quantization layer proxy within the
measured context range. It does not independently validate each row. The
scheduler must label those rows `estimated`, apply a 10% upper uncertainty
bound, and avoid extrapolation outside the measured shape range.

Before energy acquisition, path-matched monolithic and two-phase executions
were checked on CPU and CUDA. Both produced `rel_l2=0`, `max_abs=0`, identical
argmax 2010, and no non-finite values. The mixed proxy deliberately disables
phone splitting because Qwen's gate/up and down tensors use different block
types.

## Cross-device diagnostic

An independent RTX A6000 GPU 1 series exercises the same model. GPU 1 had no
foreign compute process; GPU 0 was separately occupied, so this remains a
secondary diagnostic. Its endpoint fit is:

```text
E_layer(KV) = 100.565556 mJ + KV * 0.003240988 mJ
```

It predicts 102.225 mJ at held-out KV=512 versus 110.793 mJ measured, an
absolute error of 7.73%. This is a secondary mechanics check only. Its
coefficients are not used in the RTX 4060 Ti table or scheduler profile.

An earlier A6000 GPU 0 attempt was rejected when unrelated SGLang processes
appeared during acquisition. No result from that contaminated series is in
the evidence root.

## Phone status

The existing Qwen route has only a physical correctness check for one Q6_K
down-projection shape. It has no measured recurring HTP latency or phone
energy. The current OP15 HTP backend also lacks a qualified Q4_K kernel for
Qwen gate and up projections. Therefore this report does not copy the Gemma
fused-FFN phone estimate or claim a Qwen phone energy saving.

For the current scheduler, a resident Qwen layer remains CUDA-only. Q6_K down
offload stays a candidate until its complete CPU-plus-OP15 island has measured
latency, merge wait, and energy.

## Evidence

Primary RTX 4060 Ti results:

```text
rtx4060/ctx136/GPU_ENERGY.json
sha256:ff68b7f8f9d7d78fa4319ef8e111f41531b06795262dad2fc408d2a9060ea5b7

rtx4060/ctx512/GPU_ENERGY.json
sha256:1efc2265d556d64482ba6edfbd35e77e40e4f03b30aecaabf5f8e169ab84f401

rtx4060/ctx8192/GPU_ENERGY.json
sha256:b246e9eabdf1fefea5d3f5f34a9cc809c547b740a38656863e8655ec08e08099
```

Secondary A6000 GPU 1 results:

```text
a6000_gpu1/ctx136/GPU_ENERGY.json
sha256:cf6bceef2280768ae7dc808fe4be93c235f18449499ee0ef8eb85e30169163c0

a6000_gpu1/ctx512/GPU_ENERGY.json
sha256:925e8b46495b03e564ab4ab9470d3f4ddf5f50e2dd0ffc2cd17119e1a968dd8c

a6000_gpu1/ctx8192/GPU_ENERGY.json
sha256:b5bd96750f8f48c652c2b01df0119d5777f761b7b699f5ead3d671cebf923c90
```

All accepted raw logs, power samples, and the exact RTX 4060 source snapshot
are under:

```text
research_dev/spikes/s42_general_energy_scheduler_v1/model_energy_v1/results/
qwen3_operator_energy_v1/run_20260807T005324Z/
```

The RTX 4060 runner binary SHA-256 is
`47eed12ba95040d7620362154d1823f02af142023bd00a3771d94a8741409547`.

## Remaining calibration

Directly isolate the Q4_K and Q6_K GEMV rows, the KV scan, the CPU package
rows, and the complete Q6_K OP15 down island. Then validate the composed
profile on a held-out full Qwen route. Prefill batches, the Q6_K vocabulary
head, and model-switch energy require separate buckets.
