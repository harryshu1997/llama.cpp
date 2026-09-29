# Gemma4 Q4 FFN DMA-BUF fast path

Date: 2026-08-03 EDT.

Update: the steady-state setting is superseded by the persistent-I/O,
FP16-activation profile in `RESULTS_FFN_DMABUF_F16_PACKED_V1.md`. This
report remains the F32 activation baseline and weight-packing analysis.

Verdict: `Q4_FASTPATH_PASS; EXACT_TOKENS_PASS; FOUR_HVX_SELECTED; PHONE_COLUMNS_9344; DECODE_LATENCY_MINUS_34.71_PERCENT_VS_CPU`.

## Result

The byte-identical Gemma4 12B Q4_0 desktop CPU plus OP15 profile was improved
in two steps:

- limit the HTP M=1 FFN kernel to four HVX threads;
- rebalance the split from 9088 phone columns to 9344 phone columns.

Across 15 paid requests from three fresh workers, median 15-step decode time
was 2321.403 ms. The prior 9088-column profile took 2361.654 ms, so this is a
further 1.70% latency reduction and 1.73% throughput increase. Against the
matched CPU-only median of 3555.619 ms, decode latency is 34.71% lower and
decode throughput is 53.17% higher.

| metric | CPU only | prior phone profile | calibrated profile |
| --- | ---: | ---: | ---: |
| phone columns | 0 | 9088 | 9344 |
| 15-step decode | 3555.619 ms | 2361.654 ms | 2321.403 ms |
| decode throughput | 4.219 token/s | 6.351 token/s | 6.462 token/s |
| 16-token request wall | 4481.863 ms | 3314.770 ms | 3276.087 ms |
| request throughput | 3.570 token/s | 4.827 token/s | 4.884 token/s |

The calibrated request wall time is 1.17% lower than the prior phone profile
and 26.90% lower than CPU only. Prefill remains on the desktop, so its wall
time improvement is smaller than the decode-only improvement.

## Configuration

- Desktop: Intel Core i9-12900K, eight inference threads, `-ngl 0`.
- Phone: OnePlus 15 HTP0, four HVX threads, one HMX unit available but unused
  by this M=1 quantized path.
- Model on both devices: the same 6,975,878,176-byte Q4_0 GGUF with SHA-256
  `494518c2262a26e2a607af0e40bca11c4de5a0b108e7c21308906dbbfb1c6f8c`.
- Dense FFN shape: K=3840, NFF=15360, 48 layers.
- Desktop prefix: columns `[0,6016)`, or 39.17%.
- Phone suffix: columns `[6016,15360)`, or 60.83%.
- Resident phone weight buffer: 2771.72 MiB in HTP Q4_0 tiled layout.
- Activation traffic per FFN call: 15,488 bytes OUT and 15,488 bytes IN.
- Transport: direct FunctionFS DMA-BUF with persistent libusb host memory.

The session script uses `GGML_HEXAGON_NHVX=4` when the caller does not set the
variable. Set it explicitly to another value, including `0` for backend
autoselection, to override the measured default.

## Width calibration

Each screening point used one warmup, two paid eight-token requests, and 1008
verified FFN calls. The 9088 default row and every wider row use the same Q4_0
model bytes and direct DMA path.

| HVX threads | phone columns | HTP compute p50 | phone RPC p50 | desktop branch p50 | overlap p50 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 9088 | 1.156 ms | 1.548 ms | 1.618 ms | 1.626 ms |
| 4 | 9088 | 1.116 ms | 1.506 ms | 1.622 ms | 1.624 ms |
| 4 | 9152 | 1.131 ms | 1.528 ms | 1.599 ms | 1.609 ms |
| 4 | 9216 | 1.131 ms | 1.519 ms | 1.588 ms | 1.597 ms |
| 4 | 9280 | 1.132 ms | 1.518 ms | 1.572 ms | 1.589 ms |
| 4 | 9344 | 1.131 ms | 1.513 ms | 1.558 ms | 1.575 ms |
| 4 | 9408 | 1.138 ms | 1.515 ms | 1.541 ms | 1.574 ms |

The 9408-column point has nearly the same overlap median, but its join wait
rises from 0.011 to 0.035 ms and its two short full-model decodes were slower.
The 9344-column point was therefore selected for the long validation.

The three long workers reported:

| repetition | HTP compute p50 | phone RPC p50 | desktop branch p50 | overlap p50 | USB p50 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.130 ms | 1.501 ms | 1.556 ms | 1.567 ms | 1.469 ms |
| 2 | 1.131 ms | 1.508 ms | 1.557 ms | 1.571 ms | 1.470 ms |
| 3 | 1.132 ms | 1.502 ms | 1.553 ms | 1.567 ms | 1.471 ms |

## Desktop packing decision

Desktop-side Q4 packing is technically possible. The HTP backend's
`repack_q4_0_tiled` routine is host C++ and produces deterministic 32x32
tiles. A desktop utility can produce those bytes before the phone session.

It does not improve steady-state phone computation in this path. The current
worker already repacks each weight once into `HTP0-REPACK` and keeps the
2771.72 MiB buffer resident for every token. Desktop-created bytes would be
the same tiles consumed by the same HTP kernel.

Sending that packed buffer at each startup is also unattractive. At the
measured 465 MB/s direct USB rate, 2771.72 MiB needs about 6.25 seconds of wire
time before validation and setup. Warm phone-local load and repack during this
campaign was already in the same or lower range. A useful implementation must
therefore be a persistent phone-side cache, not a per-session desktop upload.

Such a cache needs a versioned header containing at least the backend ABI,
HTP architecture, tile format, tensor type and shape, selected suffix,
source-GGUF hash, per-tensor offsets, and a payload hash. On a cache hit the
phone can copy the stored tiled bytes directly into the HTP shared buffer. It
can shorten cold preparation, but it should not be counted as a token-latency
optimization.

## Correctness and safety

- All 15 paid requests emitted the same token sequence as the CPU and prior
  phone profiles.
- Each of three fresh long workers completed 4320 calls with status zero, for
  12,960 verified calls total.
- Every model process reported `FFN_OVERLAP_OK`; every bridge reported status
  `ok`.
- The workers reported four HTP/HVX threads and the same resident-weight hash
  `734b605fdc9344d5`.
- The kernel log contained gadget rebind `ep0out` diagnostics, but no DMA-BUF,
  SMMU, IOMMU, panic, or BUG fault. No call failed.
- A normal reboot restored stock kernel
  `6.12.23-android16-5-gb3b66ace21e0-ab14672634-4k`, `ptp,adb`, the original
  SuperSpeed gadget, and no FFN worker.

Raw logs are stored on the desktop at:

```text
/home/zhihao/s41-opoffload-dmabuf-v1/q4-fastpath-v1/
```

## Planner setting

```text
desktop_model_quant = Q4_0
phone_model_quant = Q4_0
desktop_weight_repack = false
phone_backend = HTP0
phone_hvx_threads = 4
phone_ffn_columns = 9344
desktop_ffn_columns = 6016
phone_layers = 0-47
transport = functionfs_dmabuf
```

This exact-shape row should override the generic roofline solver for the
tested i9-12900K plus OP15 combination. Broader prompt, logit, perplexity,
long-context, and thermal testing is still required before treating it as a
general quality or sustained-performance default.
