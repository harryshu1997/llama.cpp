# Gemma4 Q4_0 local-layer operator energy pilot V1

Date: 2026-08-06 EDT.

Verdict: `AGGREGATE_HELDOUT_PASS; PER_OPERATOR_ATTRIBUTION_ESTIMATED; CPU_AND_PHONE_PER_OPERATOR_PENDING`.

## Scope

This pilot measures one Gemma4-shaped local-attention decode layer on the
physical RTX 4060 Ti. The graph uses deterministic synthetic Q4_0 block
weights, `M=1`, hidden width 3,840,
FFN width 15,360, Q/K/V widths 4,096/2,048/2,048, and 16 query heads with
eight KV heads. Model loading is outside the paid interval.

No GGUF is loaded in this probe. It exercises the deployed operator shapes,
quantization, ggml graph, and CUDA kernels, but remains a complete-layer proxy
rather than a full Gemma request.

The table is for `KV=512`. It is not a complete-model or CPU-plus-phone energy
table. The eight global-attention layers, vocabulary head, embedding, runtime
overhead, CPU branch, USB path, and OP15 are outside this measurement.

## Estimated operator energy

The model has two calibrated terms:

```text
E_layer(KV) = 52.744627 mJ + KV * 0.002477031 mJ
```

The fixed term is divided among operators using physical Q4_0 weight and
activation traffic. The attention row also receives the KV-dependent term.
Q4_0 weights use 18 physical bytes per 32 values.

| operator | shape or traffic at KV=512 | estimated mJ/layer | layer share |
| --- | --- | ---: | ---: |
| attention RMSNorm and scale | 0.073 MiB | 0.032 | 0.06% |
| Q projection | 3840 x 4096 | 3.679 | 6.81% |
| K projection | 3840 x 2048 | 1.843 | 3.41% |
| V projection | 3840 x 2048 | 1.843 | 3.41% |
| QK, softmax, and AV | 0.188 MiB fixed plus 4.000 MiB KV | 1.350 | 2.50% |
| O projection | 4096 x 3840 | 3.679 | 6.81% |
| residual and FFN RMSNorm | 0.236 MiB | 0.103 | 0.19% |
| FFN gate projection | 3840 x 15360 | 13.779 | 25.51% |
| FFN up projection | 3840 x 15360 | 13.779 | 25.51% |
| SiLU | 0.117 MiB | 0.051 | 0.09% |
| gate times up | 0.176 MiB | 0.076 | 0.14% |
| FFN down projection | 15360 x 3840 | 13.779 | 25.51% |
| final residual add | 0.044 MiB | 0.019 | 0.04% |
| **predicted complete layer** | **125.394 MiB physical traffic** | **54.013** | **100.00%** |
| **measured complete layer** | **held-out KV=512** | **57.446** | **-** |

The FFN gate, up, and down projections account for 76.53% of predicted layer
energy. This supports treating the fused FFN as the primary split candidate.

## Physical validation

Three fresh executions were measured at each context. GPU-board energy uses
50 ms `nvidia-smi` samples, arrival timestamps, and zero-order-hold
integration over markers emitted by the layer runner. The GPU used its normal
DVFS policy; no frequency lock was applied.

KV=136 and KV=8192 fit the fixed and KV coefficients. KV=512 is held out.

| KV entries | role | measured median mJ/layer | median latency | model mJ/layer | error |
| ---: | --- | ---: | ---: | ---: | ---: |
| 136 | fit | 53.082 | 0.525 ms | 53.082 | 0.00% |
| 512 | held out | 57.446 | 0.546 ms | 54.013 | -5.98% |
| 8192 | fit | 73.036 | 0.770 ms | 73.036 | 0.00% |

The held-out absolute error is 5.98%, below the 10% pilot gate. Repetition
half-range is 2.69% at KV=136, 0.61% at KV=512, and 0.73% at KV=8192.

This validates the sum for one local-layer proxy within the measured context
range. It does not independently validate each row. Those rows remain
`estimated`, and the scheduler must use a 10% upper uncertainty bound and
must not extrapolate beyond the measured shape range.

## Existing phone estimate

The I3 CPU-plus-OP15 route provides one useful operator-island estimate. For
decode `M=1`, OP15 owns 9,664 FFN columns. Its real mean RPC is 1.867442 ms,
of which 1.356632 ms is HTP compute. At the requested conservative 5 W phone
power:

| fused phone FFN component | estimated mJ/call |
| --- | ---: |
| HTP compute | 6.783 |
| USB and runtime remainder | 2.554 |
| complete fused FFN RPC | 9.337 |

Across 48 layers this is about 448.2 mJ per `M=1` decode step on the phone.
It is a timing-times-5-W estimate, not measured per-operator phone energy.

## Evidence

Primary corrected Q4 results:

```text
q4_v2_ctx136/GPU_ENERGY.json
sha256:808ddf31b2ffd1f6d37e3e0bb4aabdd1e895be85aad82092935a9f4eda5e0f89

q4_v2_ctx512/GPU_ENERGY.json
sha256:d4d063fb7b9167b78ded4c8528fbec8bac128c652d95dea0a0bedfeffb23a461

q4_v2_ctx8192/GPU_ENERGY.json
sha256:850dcbb150c9723cb7e65d819c995609afcb45542160a35dac3d32e5ba29ce08
```

All raw logs and power samples are under:

```text
research_dev/spikes/s42_general_energy_scheduler_v1/model_energy_v1/results/
gemma4_operator_energy_v1/run_20260807T002904Z/
```

The first Q4 acquisition and the Q8 diagnostic are preserved in the same run
root. They are not the primary result because the first acquisition carried a
stale hard-coded sampler-period label and Q8 does not match the deployed block
precision.

## Remaining calibration

Before scheduler enforcement, directly isolate the large GEMV rows, the KV
scan, the CPU package rows, and the fused OP15 RPC. Then validate the composed
profile on a held-out complete Gemma route. Global attention, prefill batches,
and the Q6_K vocabulary head require separate shape buckets.
