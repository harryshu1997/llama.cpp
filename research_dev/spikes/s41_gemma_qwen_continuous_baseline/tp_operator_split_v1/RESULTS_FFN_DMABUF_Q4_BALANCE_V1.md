# Gemma4 FFN DMA-BUF hybrid Q8/Q4 balance point

Date: 2026-08-03 EDT.

Verdict: `EXPLORATORY_HYBRID_RESULT; NOT_A_SAME_MODEL_COMPARISON; SUPERSEDED`.

> Correction: this experiment used Q8_0 weights for the desktop prefix and
> Q4_0 weights for the phone suffix. It measures a hybrid operator hardware
> ceiling, not an apples-to-apples model speedup. The corrected byte-identical
> Q4_0 comparison is in `RESULTS_FFN_DMABUF_Q4_SAME_MODEL_V1.md`.

## Result

A hybrid split with the desktop Gemma4 12B Q8_0 model and a Q4_0 FFN suffix
resident on the OP15 HTP reached the per-layer overlap balance point. The phone
evaluates columns `[4224,15360)` and the desktop evaluates `[0,4224)` in every
one of the 48 layers.

The phone therefore owns 11,136 of 15,360 FFN columns, or 72.5%. Median phone
RPC and desktop-prefix time were approximately 1.91 and 1.95 ms. Median desktop
wait was 0.014-0.022 ms in the two long processes.

Across ten paid treatment requests, median decode time fell from 6399.457 to
3330.153 ms. Decode throughput increased from 2.344 to 4.504 token/s. All ten
requests produced the same sensible 16-token sequence as the CPU control for
this prompt.

## Configuration

- Desktop model: Gemma4 12B Q8_0.
- Phone suffix model: Gemma4 12B Q4_0.
- Shape: K=3840, NFF=15360, 48 dense layers.
- Desktop prefix: 4224 columns, or 27.5% of each FFN.
- Phone suffix: 11136 columns, or 72.5% of each FFN.
- Phone resident weights: 3303.29 MiB in the HTP repack buffer.
- HTP virtual-memory limit: 3328 MiB, measured on this device and selected
  with `GGML_HEXAGON_VMEM=3328`.
- Desktop: Intel Core i9-12900K, 8 inference threads, `-ngl 0`.
- Transport: direct FunctionFS DMA-BUF and persistent libusb host memory.
- Wire size per layer: 15,488 bytes OUT plus 15,488 bytes IN.
- Each process: one warmup plus five paid 16-token requests.

The HTP backend natively repacks and evaluates Q4_0 matmuls. The phone model
was already present at:

```
/data/local/tmp/ls-s32/gemma-4-12B-it-Q4_0.gguf
```

No transport or model-graph code change was required. The existing worker
reads the quantization type from the GGUF tensors, and the existing client
selects the suffix width at runtime.

## Full-model timing

| metric | CPU only | CPU Q8 + OP15 Q4 | change |
| --- | ---: | ---: | ---: |
| 15-step decode time | 6399.457 ms | 3330.153 ms | -47.96% |
| decode throughput | 2.344 token/s | 4.504 token/s | +92.17% |
| 16-token request wall | 7250.514 ms | 4183.188 ms | -42.30% |
| request throughput | 2.207 token/s | 3.825 token/s | +73.33% |

The control contains five paid samples. The treatment contains ten paid
samples from two fresh phone workers and two fresh desktop model processes.
Treatment decode times ranged from 3320.582 to 3345.782 ms.

The measured width progression was:

| phone columns | quant | resident weights | desktop prefix p50 | phone RPC p50 | result |
| ---: | --- | ---: | ---: | ---: | --- |
| 5632 | Q8_0 | 3155.63 MiB | 4.176 ms | 1.93 ms | phone finishes too early |
| 9216 | Q4_0 | 2733.75 MiB | 2.721 ms | 1.715 ms | 1.006 ms slack |
| 10624 | Q4_0 | 3151.41 MiB | 2.152 ms | 1.874 ms | 0.278 ms slack |
| 10752 | Q4_0 | 3189.38 MiB | 2.102 ms | 1.892 ms | 0.210 ms slack |
| 11136 | Q4_0 | 3303.29 MiB | 1.945 ms | 1.906 ms | balanced |

The 11136-column split improved median decode by another 2.34% relative to
10752 columns. A wider split is not useful: it would consume the remaining HTP
mapping headroom and move the phone RPC onto the critical path.

## Overlap timing

| process | model RPC p50 | model RPC p90 | HTP compute p50 | desktop prefix p50 | desktop wait p50 | overlap p50 | USB round trip p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.915 ms | 2.031 ms | 1.403 ms | 1.948 ms | 0.0225 ms | 1.980 ms | 1.865 ms |
| 2 | 1.897 ms | 2.004 ms | 1.401 ms | 1.942 ms | 0.0143 ms | 1.961 ms | 1.859 ms |

This is effectively perfect overlap. The median branch difference is only
0.033-0.045 ms, and the observed join cost remains below 0.023 ms at p50.

For one layer and one decoded token, the phone suffix evaluates about
256.57 MFLOP. Across 48 layers, it evaluates about 12.32 GFLOP per token. The
activation boundary remains fixed because the input and partial output are
both K=3840 F32 vectors.

## HTP virtual-memory probe

The default HTP operation-batch mapping limit is 3200 MiB. Running the
backend's bounded capacity probe with `GGML_HEXAGON_VMEM=0` measured
3,489,660,928 bytes, or 3328 MiB, after backing off from the next failed
256 MiB mapping step. The balanced worker was then run with the measured
3328 MiB limit rather than probing on every launch.

The 3303.29 MiB Q4 weight buffer leaves little headroom, but both fresh
workers allocated successfully, warmed all 48 graphs, and completed 4320
requests with status 0.

## Correctness and safety receipts

- Both balanced workers completed 4320 verified DMA-BUF and HTP calls.
- All ten paid outputs matched one deterministic token sequence.
- The sequence also matched all five CPU control requests for this prompt.
- All protocol metadata, activation hashes, response hashes, and finite-value
  checks passed.
- The fixed-kernel fault filter was empty after the campaign.
- All 53 retained artifacts pass their SHA-256 manifest.
- The phone was restored to stock kernel
  `6.12.23-android16-5-gb3b66ace21e0-ab14672634-4k`, `ptp,adb`, the original
  SuperSpeed gadget, and no FFN worker.

Raw logs and checksums are stored at:

```
/home/zhihao/s41-opoffload-dmabuf-v1/q4-balance-v1/
```

## Planner setting

For this desktop CPU plus OP15 route, select:

```
phone_backend = HTP0
phone_ffn_quant = Q4_0
phone_ffn_columns = 11136
desktop_ffn_columns = 4224
phone_layers = 0-47
GGML_HEXAGON_VMEM = 3328
```

Do not use this hybrid profile as the same-model planner default. Use the
corrected Q4_0/Q4_0 setting in `RESULTS_FFN_DMABUF_Q4_SAME_MODEL_V1.md`.
