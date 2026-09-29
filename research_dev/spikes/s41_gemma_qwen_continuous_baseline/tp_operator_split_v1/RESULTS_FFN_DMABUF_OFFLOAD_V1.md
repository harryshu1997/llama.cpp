# Gemma4 FFN operator split over direct USB DMA-BUF

Date: 2026-08-03 EDT.

Verdict: `REAL_MODEL_OPERATOR_OFFLOAD_PASS; EXACT_TOKENS_PASS; DECODE_LATENCY_MINUS_6.04_PERCENT; PHONE_SIDE_ACTIVATION_COPY_REMOVED`.

## Result

Splitting each Gemma4 dense FFN by its intermediate columns and running the
suffix concurrently on the OP15 HTP reduced median decode time by 6.04% versus
desktop CPU-only execution. Decode throughput increased by 6.43%. The full
request, including an unchanged CPU prefill, was 5.11% faster.

This is operator-level overlap, not layer placement. Every decode layer starts
the phone suffix from the normalized hidden state while the desktop computes
the complementary FFN prefix. The two 3840-element partial outputs are summed
before the normal post-FFN operations. Attention, normalization, the LM head,
and prefill remain on the desktop.

All 15 paid requests in each arm produced the same 16 greedy token IDs:

```
[100,45518,107,221662,111658,568,12553,769,107,15331,171194,568,8506,77469,769,107]
```

## Configuration

- Model: Gemma4 12B Q8_0, 48 dense layers, K=3840, NFF=15360.
- Desktop: Intel Core i9-12900K, 8 inference threads, 30 GiB RAM.
- Installed GPU: RTX 4060 Ti 16 GiB. It was intentionally unused with `-ngl 0`
  because this comparison is desktop CPU versus desktop CPU plus phone.
- Phone: OnePlus 15, HTP0, 1792 FFN columns resident for every layer.
- Desktop slice: columns `[0,13568)`, or 88.33% of each FFN.
- Phone slice: columns `[13568,15360)`, or 11.67% of each FFN.
- Resident phone weights: 1004.07 MiB, Q8_0 repacked HTP buffer.
- Transport: USB 5 Gbit/s FunctionFS with direct HTP DMA-BUF input and output,
  plus persistent libusb host memory.

For one layer and one decode token, the phone evaluates two 3840x1792 gate/up
matrix-vector products, GEGLU, and one 1792x3840 down matrix-vector product.
Counting a multiply-add as two operations, this is about 41.29 MFLOP. The
activation is 15,360 bytes in each direction. A 128-byte aligned protocol
prefix makes each USB transfer 15,488 bytes, or 30,976 wire bytes per call.

## DMA execution path

```
desktop hidden state
  -> persistent libusb buffer
  -> xHCI / USB / DWC3
  -> FunctionFS-imported HTP input DMA-BUF
  -> HTP FFN suffix
  -> HTP output DMA-BUF
  -> DWC3 / USB / xHCI
  -> desktop partial-output sum
```

The request header and activation share one aligned HTP input buffer. The
response header and partial result share one aligned HTP output buffer. Each
operator call therefore uses one USB OUT and one USB IN operation. The phone
does not call userspace `read`, `write`, or `memcpy` for activation payloads.
DMA fence waits and cache synchronization remain.

## Measurement protocol

The fixed FunctionFS kernel was booted temporarily with `fastboot boot`; no
partition was flashed. Its boot image SHA-256 was
`26e8d41808b10b70264d958582bb6f6c9fba634c34275e0bc28fc820a6d3fb8d`.
The stock kernel must not run this direct path because its FunctionFS OUT
DMA direction is reversed.

Three fresh CPU control and phone treatment process pairs were interleaved.
Each process ran one full warmup followed by five paid 16-token requests. The
prompt, model, context size, thread count, greedy token selection, and desktop
binary were identical. Each treatment executed 4320 verified phone FFN calls
including its warmup. The table reports the median of the five paid requests
inside each fresh process.

| repetition | CPU decode | CPU + OP15 decode | decode reduction | CPU request wall | CPU + OP15 request wall | wall reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6407.304 ms | 6020.481 ms | 6.04% | 7250.645 ms | 6912.146 ms | 4.67% |
| 2 | 6419.395 ms | 6019.760 ms | 6.23% | 7282.199 ms | 6898.079 ms | 5.27% |
| 3 | 6404.526 ms | 6023.471 ms | 5.95% | 7249.965 ms | 6873.205 ms | 5.20% |

Across all 15 paid samples per arm:

| metric | CPU-only | CPU + OP15 | change |
| --- | ---: | ---: | ---: |
| 15-step decode time | 6407.304 ms | 6020.481 ms | -6.04% |
| decode throughput | 2.341 token/s | 2.491 token/s | +6.43% |
| 16-token request wall time | 7255.010 ms | 6884.470 ms | -5.11% |
| request throughput | 2.205 token/s | 2.324 token/s | +5.38% |

## Overlap and transport timing

The per-process FFN medians were:

| repetition | model RPC | phone HTP compute | desktop FFN prefix | desktop wait | bridge USB round trip |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.154 ms | 0.503 ms | 5.726 ms | 0.00017 ms | 1.063 ms |
| 2 | 1.191 ms | 0.503 ms | 5.728 ms | 0.00019 ms | 1.075 ms |
| 3 | 1.178 ms | 0.503 ms | 5.726 ms | 0.00018 ms | 1.080 ms |

The phone result finished about 4.5 ms before the desktop prefix at the median,
so phone compute and communication were fully hidden. The remaining critical
path is the 13,568-column desktop FFN prefix. A matched short validation reduced
the model RPC median from 1.461 ms on the ADB/TCP path to 1.021 ms on the first
DMA run, but transport is no longer the critical path at this split width.

The end-to-end gain is smaller than the 11.67% FFN column reduction because
attention, norms, the LM head, graph callbacks, and other model work remain on
the desktop. Prefill is intentionally not split, which also makes the request
wall-time gain smaller than the decode-only gain.

## Correctness and safety receipts

- Three CPU and three treatment processes completed successfully.
- All 30 paid request token sequences matched exactly.
- Every treatment reported `FFN_OVERLAP_OK` and 4320 calls.
- Every phone worker exited with status zero after the host disconnect marker.
- The fixed-kernel dmesg fault filter was empty.
- A final build-from-current-source gate passed 96 more DMA calls with exact
  token IDs `[100,45518,107]`.
- A normal reboot restored stock kernel
  `6.12.23-android16-5-gb3b66ace21e0-ab14672634-4k`, `ptp,adb`, SuperSpeed,
  the original gadget, and zero FFN transport workers.

Raw logs and their SHA-256 receipts are on the desktop under
`/home/zhihao/s41-opoffload-dmabuf-v1/campaign/`.

## Limits and next step

This result covers one prompt, greedy decode, one phone, a short context, and
CPU-hosted model execution. It is not a CUDA comparison and it does not include
energy measurement. The follow-up width sweep is recorded in
`RESULTS_FFN_DMABUF_WIDTH_SWEEP_V1.md`. It reached a stable 5632-column split
and measured a 23.66% decode-time reduction, with the numerical-equivalence
caveat documented there. The corrected byte-identical Q4_0 desktop/phone test
is recorded in `RESULTS_FFN_DMABUF_Q4_SAME_MODEL_V1.md`; it reached a
9088-column phone suffix with a 33.58% decode-time reduction and exact tokens
for the measured prompt. Its calibrated successor is recorded in
`RESULTS_FFN_DMABUF_Q4_FASTPATH_V1.md`; four HVX threads and a 9344-column
phone suffix reduced decode time by another 1.70%.
