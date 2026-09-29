# Gemma4 Q4 FFN packed-activation DMA fast path

Date: 2026-08-03 EDT.

Verdict: `F16_PACKED_DMA_PASS; PERSISTENT_IO_PASS; EXACT_TESTED_TOKENS_PASS; PHONE_COLUMNS_9664; DECODE_THROUGHPUT_PLUS_56.08_PERCENT_VS_CPU`.

## Result

The desktop now packs the decode activation into FP16 in a persistent I/O
thread while its CPU branch computes the complementary FFN columns. The OP15
HTP graph reads the packed activation directly from the FunctionFS DMA-BUF,
casts it to F32, executes the same resident Q4_0 FFN weights, casts the partial
result to FP16, and returns it through the output DMA-BUF.

Across 15 paid requests from three fresh phone workers, median 15-step decode
time was 2278.041 ms. The prior F32 DMA profile took 2321.403 ms, so packing
reduces decode latency by 1.87% and increases decode throughput by 1.90%.
Against matched desktop CPU-only inference, decode latency is 35.93% lower
and decode throughput is 56.08% higher.

| metric | CPU only | prior F32 DMA | packed F16 DMA |
| --- | ---: | ---: | ---: |
| phone columns | 0 | 9344 | 9664 |
| desktop columns | 15360 | 6016 | 5696 |
| 15-step decode | 3555.619 ms | 2321.403 ms | 2278.041 ms |
| decode throughput | 4.219 token/s | 6.462 token/s | 6.585 token/s |
| 16-token request wall | 4481.863 ms | 3276.087 ms | 3220.108 ms |
| request throughput | 3.570 token/s | 4.884 token/s | 4.969 token/s |

The packed request wall time is 1.71% lower than the F32 DMA profile and
28.15% lower than CPU only. Prefill remains on the desktop and is not split,
so request-wall improvement is smaller than decode improvement.

## Data path

The steady-state path is:

    desktop F32 activation
      -> persistent I/O thread converts F32 to F16
      -> libusb OUT
      -> FunctionFS DMA-BUF
      -> HTP F16 input tensor
      -> HTP cast to F32
      -> Q4_0 gate/up, GeGLU, and down projection
      -> HTP cast partial output to F16
      -> FunctionFS DMA-BUF
      -> libusb IN
      -> persistent I/O thread converts F16 to F32
      -> desktop sums the two FFN partials

The model quantization is not different between devices. Both load the exact
same Q4_0 GGUF bytes. FP16 is only the transient activation wire format.
This introduces activation rounding relative to the F32 wire path, so the
tested token match does not replace broader perplexity or long-context
quality testing.

The client also keeps one I/O thread alive for the entire decode and reuses
its input, request, response, and conversion buffers. The earlier client
created and joined a C++ thread for every FFN call. At the same 9344-column
F32 split, the persistent client reduced short-run overlap p50 from 1.5940
to 1.5625 ms and median eight-token decode from 1093.457 to 1089.007 ms.

The two ends must select the format together. Launch the phone session with
`S41_FFN_F16_IO=1`, and launch the desktop client with
`--ffn-columns 9664 --ffn-f16-io`. The protocol handshake rejects a mixed
F32/F16 pair instead of interpreting one layout as the other.

## Why this packing format

For K=3840 and M=1:

| candidate activation representation | payload per direction | wire bytes with prefix | result |
| --- | ---: | ---: | --- |
| F32 | 15360 bytes | 15488 bytes | prior path |
| F16 | 7680 bytes | 7808 bytes | selected |
| flat Q8_0 | 4080 bytes | 4208 bytes | tested selector rejected the FFN MUL_MAT |
| HTP tiled Q8_0 row | 138240 bytes | 138368 bytes | rejected by size |

The HTP tiled Q8_0 layout is optimized for its matrix kernel, but an M=1
activation is padded into 120 large tiles. Sending that layout would be about
18 times larger than the selected F16 payload. A profile trace also measured
the first gate/up activation quantization at roughly 2 us, so eliminating
that phone-side conversion cannot repay the extra USB traffic.

Flat Q8_0 would be smaller, but forcing the tested direct matrix selector
made the 9344-column Q4 FFN graph unsupported. It is not a valid path on this
backend build. F16 therefore gives the best supported combination: compact
wire data, direct DMA storage, and only small casts inside the HTP graph.

Phone weight packing remains a startup-only operation. The worker repacks the
Q4_0 suffix once into `HTP0-REPACK` and keeps all 48 layers resident. Sending
about 2866.64 MiB of prepacked weights from the desktop would not change
steady-state compute and would add several seconds of startup transfer.

## Split calibration

Each F16 screening point used a fresh worker, one warmup request, three paid
eight-token requests, and 1344 verified phone calls.

| phone columns | desktop columns | HTP compute p50 | RPC p50 | desktop branch p50 | join wait p50 | overlap p50 | median short decode |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9344 | 6016 | 1.172 ms | 1.4473 ms | 1.5472 ms | 0.0000 ms | 1.5491 ms | 1076.660 ms |
| 9536 | 5824 | 1.196 ms | 1.4717 ms | 1.5065 ms | 0.0041 ms | 1.5250 ms | 1076.984 ms |
| 9664 | 5696 | 1.215 ms | 1.4748 ms | 1.4846 ms | 0.0208 ms | 1.5081 ms | 1070.305 ms |
| 9792 | 5568 | 1.231 ms | 1.4947 ms | 1.4471 ms | 0.0901 ms | 1.5397 ms | 1084.545 ms |

The 9664-column point is the best balance. At 9792 columns the desktop branch
is shorter, but phone completion is late enough that join wait grows and the
full decode regresses.

At the same 9344-column width, F16 reduced RPC p50 from 1.5164 to 1.4473 ms.
The extra HTP casts increased phone compute from about 1.142 to 1.172 ms, but
the smaller USB payload more than recovered that cost. Rebalancing then moved
another 320 columns to the phone.

## Long validation

Each repetition used a fresh phone worker, one warmup request, five paid
16-token requests, and 4320 verified FFN calls.

| repetition | HTP compute p50 | RPC p50 | desktop branch p50 | join wait p50 | overlap p50 | USB p50 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.221 ms | 1.4771 ms | 1.4767 ms | 0.0333 ms | 1.5087 ms | 1.4514 ms |
| 2 | 1.223 ms | 1.4783 ms | 1.4709 ms | 0.0403 ms | 1.5103 ms | 1.4532 ms |
| 3 | 1.217 ms | 1.4814 ms | 1.4947 ms | 0.0138 ms | 1.5385 ms | 1.4502 ms |

All three bridges reported 7808 wire bytes, about 0.042 ms OUT p50, and about
1.408 ms IN p50. IN includes waiting for phone computation before the response
becomes available.

## Correctness and safety

- All 15 paid requests emitted the CPU-reference token sequence
  `[100,45518,107,221662,111658,568,12553,769,107,15331,171194,568,8506,77469,769,107]`.
- Three fresh workers completed 4320 calls each with status zero, for 12,960
  verified calls total.
- Every model process reported `FFN_OVERLAP_OK`; every bridge reported
  `status=ok`.
- Every worker reported the same 9664-column resident-weight hash
  `eb1e9434d0c238a1`.
- The desktop and phone model files were both 6,975,878,176 bytes with SHA-256
  `494518c2262a26e2a607af0e40bca11c4de5a0b108e7c21308906dbbfb1c6f8c`.
- No DMA-BUF, SMMU, IOMMU, panic, or BUG fault was present after the campaign.
- A normal reboot restored stock kernel
  `6.12.23-android16-5-gb3b66ace21e0-ab14672634-4k`, USB `ptp,adb`,
  device `22d9:2772`, and no worker process.

Raw logs are stored on the desktop at:

    /home/zhihao/s41-opoffload-dmabuf-v1/persistent-io-v1/

## Planner setting

    desktop_model_quant = Q4_0
    phone_model_quant = Q4_0
    activation_wire = F16
    desktop_activation_pack_thread = persistent_io
    desktop_weight_repack = false
    phone_backend = HTP0
    phone_hvx_threads = 4
    phone_ffn_columns = 9664
    desktop_ffn_columns = 5696
    phone_layers = 0-47
    transport = functionfs_dmabuf

This measured row should override the generic roofline solver for Gemma4 12B
Q4_0 decode on the tested i9-12900K plus OP15. Broader prompts, perplexity,
long context, sustained thermal behavior, and energy still need validation
before making FP16 activation transport a general quality default.
