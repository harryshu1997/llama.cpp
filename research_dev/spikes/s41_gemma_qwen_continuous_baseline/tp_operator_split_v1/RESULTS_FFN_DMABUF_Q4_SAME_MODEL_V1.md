# Gemma4 FFN DMA-BUF same-model Q4 overlap

Date: 2026-08-03 EDT.

Verdict: `SAME_MODEL_Q4_PASS; EXACT_TOKENS_PASS; DECODE_LATENCY_MINUS_33.58_PERCENT; DECODE_THROUGHPUT_PLUS_50.56_PERCENT`.

The follow-up four-HVX and split-width calibration supersedes only the planner
setting in this report. It selects 9344 phone columns and is recorded in
`RESULTS_FFN_DMABUF_Q4_FASTPATH_V1.md`. The measurements below remain the
9088-column baseline for that comparison.

## Correction and result

The desktop CPU and OP15 now use byte-identical copies of the same Gemma4 12B
Q4_0 GGUF. The earlier Q8_0-desktop plus Q4_0-phone run was useful for finding
the hardware overlap ceiling, but it was a hybrid operator and was not an
apples-to-apples model comparison.

The corrected split sends 9088 of each layer's 15360 FFN columns to OP15 HTP0
and computes the complementary 6272 columns on the desktop. Across 15 paid
requests per arm, median decode time fell from 3555.619 to 2361.654 ms. Decode
throughput increased from 4.219 to 6.352 token/s. All 30 paid requests produced
the same deterministic 16-token sequence.

The two model files are 6,975,878,176 bytes and have the same SHA-256:

```
494518c2262a26e2a607af0e40bca11c4de5a0b108e7c21308906dbbfb1c6f8c
```

The HTP backend repacks Q4_0 blocks into its native in-memory layout. That is a
backend storage transformation, not a different model or quantization.

## CPU repack correction

The first same-Q4 attempt found a deterministic numerical failure. CPU model
loading had converted Q4_0 tensors to the shape-dependent `CPU_REPACK` layout,
while the split graph later created smaller `ggml_view_2d` tensors for the host
FFN prefix. The views changed the logical matrix shape without repacking the
underlying storage, so the host prefix read an invalid packed layout.

The custom driver now:

- accepts `--no-repack` for a matched CPU-only control;
- disables extra CPU weight buffers automatically when FFN submatrix views are
  active.

This reuses `llama_model_params::use_extra_bufts`; no new tensor format or copy
path was added. A 32-column desktop/desktop split changed the first decoded
tokens before the fix and matched `[100,45518,107]` after it. A corrected
8704-column desktop/desktop split then matched all 16 control tokens, isolating
the fault from USB and HTP before the phone campaign.

## Configuration

- Model: Gemma4 12B Q4_0, K=3840, NFF=15360, 48 dense layers.
- Desktop model:
  `/home/zhihao/models/gemma-4-12B-it-Q4_0-op15-exact.gguf`.
- Phone model: `/data/local/tmp/ls-s32/gemma-4-12B-it-Q4_0.gguf`.
- Desktop: Intel Core i9-12900K, 8 inference threads, `-ngl 0`.
- Installed RTX 4060 Ti 16 GiB: intentionally unused in this CPU comparison.
- Desktop prefix: columns `[0,6272)`, or 40.83% of each FFN.
- Phone suffix: columns `[6272,15360)`, or 59.17% of each FFN.
- Phone resident weights: 2695.79 MiB in the HTP Q4_0 repack buffer.
- HTP virtual-memory arena: default 3200 MiB.
- Transport: direct FunctionFS DMA-BUF and persistent libusb host memory.
- Wire size per call: 15,488 bytes OUT plus 15,488 bytes IN.
- Process protocol: one warmup plus five paid 16-token requests.
- Repetitions: three fresh CPU controls and three fresh phone workers.

The input and partial output remain K=3840 F32 vectors, so transfer size does
not grow with the number of offloaded columns. Per layer and decoded token, the
phone suffix evaluates about 209.39 MFLOP. Across 48 layers, that is about
10.05 GFLOP per token.

## Full-model timing

Per-process medians were:

| repetition | CPU decode | CPU + OP15 decode | decode reduction | CPU wall | CPU + OP15 wall | wall reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3552.811 ms | 2361.115 ms | 33.54% | 4477.215 ms | 3313.776 ms | 25.99% |
| 2 | 3559.382 ms | 2366.885 ms | 33.50% | 4520.749 ms | 3303.631 ms | 26.92% |
| 3 | 3554.957 ms | 2364.437 ms | 33.49% | 4479.612 ms | 3367.412 ms | 24.83% |

Across all 15 paid samples per arm:

| metric | Q4 CPU only | Q4 CPU + Q4 OP15 | change |
| --- | ---: | ---: | ---: |
| 15-step decode time | 3555.619 ms | 2361.654 ms | -33.58% |
| decode throughput | 4.219 token/s | 6.352 token/s | +50.56% |
| 16-token request wall | 4481.863 ms | 3314.770 ms | -26.04% |
| request throughput | 3.570 token/s | 4.827 token/s | +35.21% |

One treatment request had a 2628.427 ms decode outlier. The aggregate table
uses the median; the other 14 treatment samples ranged from 2349.429 to
2379.806 ms. The per-process decode reductions remain within 0.06 percentage
point of each other.

## Overlap and transport timing

| repetition | model RPC p50 | model RPC p90 | HTP compute p50 | desktop prefix p50 | desktop wait p50 | USB round trip p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.530 ms | 1.596 ms | 1.150 ms | 1.619 ms | 0.00022 ms | 1.494 ms |
| 2 | 1.541 ms | 1.611 ms | 1.155 ms | 1.616 ms | 0.00023 ms | 1.504 ms |
| 3 | 1.542 ms | 1.610 ms | 1.155 ms | 1.617 ms | 0.00021 ms | 1.502 ms |

The phone finishes about 0.08 ms before the desktop prefix at p50, so the USB
round trip and HTP work are hidden behind useful desktop work. A 9152-column
gate moved phone completion onto the join path and was slower. Shape-specific
HTP kernels make the nearby width curve non-linear; 9088 was the fastest
validated gate, not a linear extrapolation.

## Correctness and safety receipts

- The CPU and treatment arms used the same GGUF quantization and exact bytes.
- All 15 CPU and 15 treatment outputs matched this token sequence:

```
[100,45518,107,221662,111658,568,12553,769,107,15331,171194,568,8506,77469,769,107]
```

- Each treatment completed 4320 verified DMA-BUF and HTP calls with status 0.
- All protocol metadata, activation hashes, response hashes, and finite-value
  checks passed.
- The fault filter found two DWC clock/runtime-PM messages at 2.36 seconds of
  boot and no DMA-BUF, IOMMU, FunctionFS, DWC, panic, or BUG event during the
  timed campaign.
- All 64 retained artifacts pass their SHA-256 manifest.
- A normal reboot restored stock kernel
  `6.12.23-android16-5-gb3b66ace21e0-ab14672634-4k`, `ptp,adb`, the original
  SuperSpeed gadget, and zero FFN workers.

Raw logs and checksums are stored at:

```
/home/zhihao/s41-opoffload-dmabuf-v1/q4-same-model-fixed-v1/
```

## Historical planner setting

For this desktop CPU plus OP15 route, select:

```
desktop_model_quant = Q4_0
phone_model_quant = Q4_0
desktop_weight_repack = false
phone_backend = HTP0
phone_ffn_columns = 9088
desktop_ffn_columns = 6272
phone_layers = 0-47
```

This is the corrected same-model operating point for this hardware and prompt.
The exact-token result is still prompt-specific; broader logit, perplexity, and
generation testing is required before treating it as a general quality proof.
Use `RESULTS_FFN_DMABUF_Q4_FASTPATH_V1.md` for the current measured setting.
