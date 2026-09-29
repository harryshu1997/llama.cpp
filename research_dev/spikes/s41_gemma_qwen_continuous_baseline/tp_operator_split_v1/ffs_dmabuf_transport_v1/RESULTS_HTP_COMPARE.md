# Matched FunctionFS copy versus direct HTP DMA-BUF

Date: 2026-08-03 EDT.

Verdict:
`MATCHED_HTP_DMABUF_COMPARE_PASS; PHONE_USERSPACE_COPY_REMOVED; SIZE_THRESHOLD_CONFIRMED; STOCK_KERNEL_UNSAFE`.

## Compared paths

All variants ran the same FP32 square graph on OP15 HTP and used the same
desktop libusb host. Every request changed its input and every output element
was checked.

1. `staged_malloc`: FunctionFS `read` into a staging vector, Hexagon
   `tensor_set`, HTP compute, `tensor_get`, then FunctionFS `write`.
2. `copy_malloc`: FunctionFS `read` and `write` directly against the CPU
   mappings of the HTP buffers. This removes the explicit staging vectors but
   still uses the normal FunctionFS kernel/userspace copies.
3. `dmabuf_malloc`: DWC3 DMA directly into and out of the HTP `rpcmem`
   DMA-BUFs, with ordinary host libusb memory.
4. `dmabuf_devmem`: the same direct phone path with host
   `libusb_dev_mem_alloc` memory.

The main campaign used three interleaved repetitions. The 10 KiB M=1 shape
also used six alternating copy/direct pairs because one main-campaign copy
run entered a different performance state.

## Result

The values below are median-of-run medians. Phone CPU is
`CLOCK_PROCESS_CPUTIME_ID` for the worker, so it excludes kernel interrupt and
DWC3 time.

| activation each way | selected direct host buffer | copied latency | direct latency | latency change | copied phone CPU | direct phone CPU |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1,280 B | malloc | 0.2177 ms | 0.2179 ms | +0.1% | 99.2 us | 108.9 us |
| 10,240 B | malloc | 0.3144 ms | 0.2133 ms | -32.2% | 192.0 us | 89.0 us |
| 81,920 B | malloc | 0.6820 ms | 0.4856 ms | -28.8% | 273.9 us | 97.7 us |
| 1,048,576 B | devmem | 6.2886 ms | 4.9812 ms | -20.8% | 1,249.3 us | 178.9 us |

The direct path won all six M=1 diagnostic pairs. At 80 KiB, the aligned
three-run change against copied FunctionFS was -28.1% and direct malloc won all
three runs. At 1 MiB, host persistent USB memory was required for the best
result; direct malloc improved latency by only 10.2%, while direct devmem
improved it by 20.8%.

At 1.28 KiB, the synchronous USB, fence, and HTP submission floor dominates.
The direct path shortens host-to-phone completion but lengthens the remaining
HTP plus return interval, producing no end-to-end gain. This mechanism should
not replace AOA for the smallest activation solely on this result.

## Why it is faster

The normal staged path has these phone data movements:

```
FunctionFS request buffer -> userspace staging -> HTP rpcmem
HTP rpcmem -> userspace staging -> FunctionFS request buffer
```

The direct path imports the scatter-gather table for the HTP allocation into
FunctionFS and uses it as the USB request buffer:

```
xHCI -> USB -> DWC3 -> HTP input rpcmem -> HTP compute
HTP output rpcmem -> DWC3 -> USB -> xHCI
```

`DMA_BUF_IOCTL_SYNC` waits for the FunctionFS reservation fence and invokes
the exporter cache-access hooks. `ggml_backend_synchronize` orders HTP output
before the USB IN queue. The gain is removal of phone endpoint `read`/`write`
copies and HTP staging copies. It does not remove the USB wire transfer, IOMMU
work, fences, HTP submission, or the desktop GPU-VRAM-to-host-buffer copy.

## Required kernel fix

The stock OP15 FunctionFS implementation maps endpoint directions backwards.
For USB OUT, DWC3 writes the buffer and the attachment must use
`DMA_FROM_DEVICE`; for USB IN, DWC3 reads it and the attachment must use
`DMA_TO_DEVICE`. The patch also fixes the matching reservation-fence usage,
serializes USB request cleanup against detach/close, and releases the initial
fence reference after adding it to `dma_resv`.

Do not queue this DMA-BUF path on the stock tested kernel. Its first prior OUT
queue caused a DWC3 SMMU write fault and kernel panic. This campaign used the
temporary, non-flashed fixed image and returned the phone to stock with a
normal reboot.

## Scope

The campaign contains 48 main captures and 12,000 paid HTP executions. The
M=1 diagnostic adds 12 captures and 6,000 paid executions. All outputs had
zero absolute error, all captured kernel-fault logs were empty, and every
case restored the stock gadget at SuperSpeed.

This validates the transport mechanism with a small HTP graph. It does not
measure a Gemma RMSNorm, FFN, attention operator, complete layer, CUDA overlap,
full-model latency, power, or energy. The integrated worker is synchronous at
queue depth 1. The next useful gate is a model-shaped operator that uses these
same imported buffers, followed by a multi-buffer overlap test.

Main analysis:
`../results/ffs_dmabuf_transport_v1/run_20260803T090334Z_htp_compare/ANALYSIS.json`.

M=1 diagnostic:
`../results/ffs_dmabuf_transport_v1/run_20260803T090334Z_htp_compare/M1_DIAGNOSTIC/SUMMARY.json`.

Evidence hashes:

- main `ANALYSIS.json`: `fc4fe7e9615d632ca5ea2c05c8fcf64d0515435bdf78bfab2573e00aa74334d2`
- M=1 `SUMMARY.json`: `adc2c7845ed86063fde5b6a176df3e2084689ef9f95dea046d0eb1d43c9e45b7`
