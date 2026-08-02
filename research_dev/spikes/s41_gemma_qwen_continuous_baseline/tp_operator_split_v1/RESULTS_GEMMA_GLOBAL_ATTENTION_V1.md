# Gemma global-attention context sharding: A6000 plus OP15

Date: 2026-07-30 EST.

Verdict:
`MECHANISM_PASS_NARROW_STEADY_WIN_GAPPED_TAIL_FAIL`.

## Scope

This is a bounded real-device test of the context-dependent attention core
for one Gemma-4-12B global-attention layer. It starts with already-projected Q
and resident f16 K/V. It does not include Q/K/V projection, output projection,
the FFN, a real model, full-model decoding, or energy.

The geometry is Gemma-4-12B global attention:

- 16 query heads;
- one KV head;
- head dimension 512;
- 262,144 cached tokens;
- one decode query.

CUDA owns the prefix of the KV sequence and OP15 owns a suffix. Both shards
run concurrently. Each shard returns its normalized attention state and one
anchor logit/probability pair per head. The host derives the shard log-sum-exp
as:

```text
log_z = scaled_anchor_logit - log(anchor_probability)
```

It then performs the exact two-shard online-softmax merge. The AOA request is
16,428 bytes and the response is 16,564 bytes, independent of context length.

## Calibration

The same CUDA graph was run on the physical RTX 4060 Ti and the RTX A6000.
The A6000 memory clock stayed at 5001 MHz while graphics clock was calibrated.

| control | 262K CUDA median |
| --- | ---: |
| physical RTX 4060 Ti | 2.362369 ms |
| A6000 at 240/5001 | 6.724929 ms |
| A6000 at 900/5001 | 2.494560 ms |
| A6000 at 990/5001 | 2.354921 ms |

The frozen 990/5001 proxy is 0.315% faster than the physical 4060 Ti for this
exact operator. The earlier 240 MHz setting is not a valid 4060 proxy for this
operator.

Physical 4060 Ti controls scale nearly linearly:

| context | CUDA median |
| ---: | ---: |
| 65,536 | 0.595295 ms |
| 131,072 | 1.186375 ms |
| 262,144 | 2.362369 ms |

## Correctness

All real OP15 runs passed finite-value, KV hash, request hash, response hash,
and path-matched numerical gates.

| phone KV suffix | local split rel L2 | phone state rel L2 | final rel L2 |
| ---: | ---: | ---: | ---: |
| 4,096 | 0.0000701 | 0.0001744 | 0.0000702 |
| 8,192 | 0.0001600 | 0.0002236 | 0.0001602 |

The maximum phone-versus-CUDA log-sum-exp error was `9.54e-7`.

## Latency

Each aggregate below is the median of three independent 60-iteration runs.
The continuous schedule sends phone work back-to-back. The gapped schedule
places a CUDA-only layer between phone requests to expose USB wake behavior.

| schedule | phone KV | CUDA | CUDA + OP15 | median change | p90 change |
| --- | ---: | ---: | ---: | ---: | ---: |
| continuous | 8,192 | 2.353188 ms | 2.300496 ms | -2.239% | -1.705% |
| gapped | 8,192 | 2.356254 ms | 2.431027 ms | +3.173% | +10.471% |
| gapped | 4,096 | 2.355132 ms | 2.338630 ms | -0.701% | +10.588% |

The continuous 8K result repeated at -2.297%, -2.227%, and -2.187%. OP15
finished in 2.007 ms while the CUDA prefix took 2.281 ms, so phone work was
hidden.

The gapped 4K median result repeated at -0.593%, -0.691%, and -0.727%.
However, its phone p90 was 2.546 ms and treatment p90 was 2.611 ms, above the
2.361 ms CUDA control. Most of the variability appears in USB OUT:
continuous median was about 0.186 ms, while gapped medians were about
0.36-0.40 ms with much larger tails.

## Interpretation

The mechanism is real, but its useful regime is narrow:

- At 65K and 131K, the physical 4060 Ti finishes the entire attention core
  faster than the measured phone path, so this route cannot reduce M=1
  latency.
- At 262K, a continuously utilized phone saves about 0.053 ms for this
  attention core.
- A smaller 4K shard preserves a repeatable 0.017 ms median saving with gaps,
  but worsens tail latency.
- Adding the unchanged projections and FFN can only dilute these percentage
  gains. Therefore this evidence is not a complete-layer speedup claim.
- No energy conclusion follows. The phone adds power, and the server-side
  energy change is too small to infer from latency alone.

For Gemma-4-12B, only 8 of 48 layers are global-attention layers. An 8K f16
KV suffix is 16 MiB per global layer, or 128 MiB across all eight global
layers. One phone therefore provides only modest KV-capacity relief in this
configuration.

## Next bounded step

Do not integrate this route into the model runtime yet. First remove or hide
the gapped AOA wake variance using one of two bounded policies:

1. schedule global-attention shards continuously across already-batched
   requests; or
2. add a low-cost transport keep-warm mechanism and include its phone energy.

Then rerun the 4K and 8K suffixes with a mandatory p90 non-regression gate.
Only a route that passes median, p90, correctness, and measured total-energy
gates should proceed to a complete Gemma layer.

## Evidence

Raw evidence and the machine-readable summary are under:

`results/gemma_global_attention_v1/run_20260731T031712Z/`.

The implementation is:

- `gemma_global_attention_protocol.h`;
- `gemma_global_attention_worker.cpp`;
- `gemma_global_attention_host.cpp`.
