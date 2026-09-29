# FunctionFS DMA-BUF and direct HTP transport

Date: 2026-08-02 EDT (captures ended 2026-08-03 UTC).

Verdict:
`REAL_FUNCTIONFS_DMABUF_PASS; REAL_USB_DMABUF_HTP_DMABUF_PASS; PHONE_SIDE_COPY_REMOVED; HOST_DEVMEM_SIZE_DEPENDENT; FULL_MODEL_AND_ENERGY_NOT_RUN`.

## Outcome

The OP15 now has a working transport path with no phone-userspace activation
copy:

```
host buffer -> xHCI -> USB -> DWC3 -> phone DMA-BUF -> HTP
HTP output DMA-BUF -> DWC3 -> USB -> xHCI -> host buffer
```

FunctionFS imports the phone DMA-BUF scatter-gather table directly. The HTP
treatment allocates both activation buffers through the existing Hexagon
`rpcmem` path, maps them into FastRPC, exposes their file descriptors through
one narrow backend accessor, and attaches those same file descriptors to the
FunctionFS endpoints. No phone CPU `read`, `write`, or activation `memcpy` is
on that path. DMA fence waits and cache synchronization remain.

The host can use either ordinary memory or Linux usbfs persistent DMA memory
from `libusb_dev_mem_alloc`. Persistent memory is useful for larger payloads,
but is not the fastest choice for every small activation.

## Stock-kernel failure and fix

The first DMA-BUF attempt on the stock OP15 kernel caused a real kernel panic
on the first host-to-phone queue. The captured fault was:

```
arm-smmu 15000000.apps-smmu: Unhandled arm-smmu context fault from a600000.dwc3
FAR 0x00000000efbb0000
FSR 0x40000408 [PF W SS]
Kernel panic - not syncing: Oops - BUG: Fatal exception
```

The exact downstream source mapped FunctionFS OUT buffers with
`DMA_TO_DEVICE`. That is reversed: an OUT endpoint carries host-to-gadget data,
so DWC3 writes the buffer and needs `DMA_FROM_DEVICE`. Upstream Linux fixed
this exact IOMMU write-fault bug in
[`0145e7acd298`](https://github.com/torvalds/linux/commit/0145e7acd29855dfba4a2f387d455b5d9a520f0e).

The temporary test kernel backports that fix plus the adjacent upstream
request-lifetime and fence-reference fixes:

- [`2796646f6d89`](https://github.com/torvalds/linux/commit/2796646f6d892c1eb6818c7ca41fdfa12568e8d1)
- [`baa6b6068a3f`](https://github.com/torvalds/linux/commit/baa6b6068a3f2bf2ed525a1cb37975905dadc658)

The source patch is `functionfs_dmabuf_fixes.patch`. The temporary boot image
was built through the official isolated GKI target:

```
tools/bazel build //common:kernel_aarch64_gki_artifacts
```

The kernel config stayed byte-identical to the prior control. The phone was
booted with `fastboot boot`; no partition was flashed. The image also contains
the pre-existing AOA 64 KiB receive change in another gadget function. The
FunctionFS tests do not exercise that function.

| artifact | SHA-256 |
| --- | --- |
| kernel config | `9f03ed30a44329ebc6337dca7157f3eaa67c3143519883b026c51abd0d7dda43` |
| kernel Image | `648ce329264e1191682760e3997a644fcee6ad4ee8b04dec63d4dcd461370354` |
| temporary boot image | `26e8d41808b10b70264d958582bb6f6c9fba634c34275e0bc28fc820a6d3fb8d` |
| stock FunctionFS source | `356a44fc7424231bb88d0481d07b735e328aa48181ad1159bfd2ccfb24b42213` |
| fixed FunctionFS source | `229d79e869516834161cd8e4e5cc6b619ecd0e8930c096acc33fde5aa2cf4f63` |

No stock-kernel DMA-BUF retry was performed after the panic.

## Transport implementation

`ffs_dmabuf_phone.c` implements two phone modes behind the same FunctionFS
vendor interface:

- `copy`: ordinary endpoint `read` and `write` control;
- `dmabuf`: Qualcomm system DMA-heap buffers attached through
  `FUNCTIONFS_DMABUF_ATTACH` and queued through
  `FUNCTIONFS_DMABUF_TRANSFER`.

`ffs_dmabuf_host.cpp` supports synchronous and queued libusb transfers with
ordinary or persistent host memory. Every request and response carries exact
sequence, size, and sentinel checks. The asynchronous runner uses independent
buffers per slot.

`ffs_dmabuf_htp_worker.cpp` allocates separate HTP input and output buffers,
attaches their exact DMA-BUF file descriptors to USB, and executes an HTP FP32
square graph. The host changes every input on every request and validates every
output element. All chosen values and their squares are exactly representable,
so this mechanism gate requires zero absolute error.

The custom gadget advertises USB 3.2 and 16-packet bursts. A watchdog restores
the original `ptp,adb` gadget after every process, including failures.

## Measurement protocol

The physical path was OP15 serial `3C15AU002CL00000` directly attached at USB
5 Gbit/s to desktop `zhihao-Z690-C-ac`. Phone thermal state was not gated for
this prototype, as requested.

The transport campaign used six boundaries, three interleaved variants, and
three fresh processes per variant. Model-sized cases used 50 warmups and 300
paid requests. One MiB cases used 20 warmups and 100 paid requests. The queued
diagnostic added three fresh depth-4 runs at two sizes. In total, 60 captures
and 14,400 paid requests passed.

The HTP campaign used three activation sizes, two interleaved host allocators,
three fresh processes, 50 warmups, and 300 paid requests. In total, 18 captures
and 5,400 paid HTP executions passed.

Both independent analyzers recompute the timing statistics, check every raw
array length, verify rate arithmetic, verify every worker and session receipt,
require empty kernel-fault logs, and require the restored SuperSpeed terminal
state.

## Copy control versus phone DMA-BUF

Values are the median of three fresh-process medians. The change is the median
of the three repetition-aligned changes against the copy control. Negative is
better.

| boundary | request / response | copy + malloc | DMA-BUF + malloc | DMA-BUF + devmem | best aligned change |
| --- | ---: | ---: | ---: | ---: | ---: |
| attention state | 1,308 / 1,384 B | 0.2079 ms | **0.1147 ms** | 0.1542 ms | **-43.57%** |
| hidden, M=1 | 10,268 / 10,344 B | 0.2591 ms | **0.1485 ms** | 0.1695 ms | **-43.12%** |
| standalone SwiGLU | 69,660 / 34,920 B | 0.5555 ms | 0.3566 ms | **0.3526 ms** | **-36.52%** |
| hidden, M=8 | 81,948 / 82,024 B | 0.6546 ms | 0.5045 ms | **0.4837 ms** | **-26.14%** |
| 1 MiB host to phone | 1,048,604 / 104 B | 3.1629 ms | 2.7847 ms | **2.4860 ms** | **-21.84%** |
| 1 MiB phone to host | 64 / 1,048,680 B | 3.4514 ms | 2.8826 ms | **2.5707 ms** | **-26.89%** |

Every DMA-BUF variant beat its aligned copy control in all three repetitions.
Persistent host memory is not a universal latency win. Ordinary host memory is
best at 1.3 and 10 KiB; persistent memory wins from the roughly 100 KiB
bidirectional boundary upward in this campaign.

The earlier AOA 64 KiB campaign is not a paired control, but provides a useful
diagnostic. FunctionFS DMA-BUF reduced its 4.748 ms 1 MiB upload to 2.486 ms
and its 0.666 ms M=8 boundary to 0.484 ms. At the smallest attention boundary,
the earlier 0.109 ms AOA result remains slightly faster than the new 0.115 ms
FunctionFS result. There is no single best transport choice at every size.

## Queued throughput

Queue depth 4 overlaps independent requests. It is a throughput result, not a
single-request latency result.

| boundary | depth-1 aggregate | depth-4 aggregate | change | depth-4 request median |
| --- | ---: | ---: | ---: | ---: |
| hidden, M=1 | 124.89 MB/s | 440.08 MB/s | +252.37% | 0.1929 ms |
| hidden, M=8 | 326.37 MB/s | 823.95 MB/s | +152.46% | 0.7820 ms |

The aggregate metric sums both directions, which can exceed the nominal
one-direction USB line rate when IN and OUT overlap.

## Direct HTP DMA-BUF result

This path executes HTP directly on the USB-received `rpcmem` buffer and sends
the HTP output buffer directly back. It does not stage through a phone CPU
activation vector.

| activation | bytes each way | malloc | host devmem | selected result | max error |
| --- | ---: | ---: | ---: | ---: | ---: |
| attention state | 1,280 B | 0.2181 ms | **0.2171 ms** | devmem, -0.43% | 0 |
| hidden, M=1 | 10,240 B | 0.2135 ms | **0.2128 ms** | devmem, -0.41% | 0 |
| hidden, M=8 | 81,920 B | **0.4876 ms** | 0.5368 ms | malloc, devmem +7.25% | 0 |

The host-timestamped causal split for the selected variant is:

| activation | host to phone completion | post-upload HTP plus return | total |
| --- | ---: | ---: | ---: |
| attention state | 0.0792 ms | 0.1386 ms | 0.2171 ms |
| hidden, M=1 | 0.0775 ms | 0.1349 ms | 0.2128 ms |
| hidden, M=8 | 0.1926 ms | 0.2953 ms | 0.4876 ms |

The phone-reported HTP submit medians were 53 to 70 us for the two small
activations and 59 us for the selected M=8 variant. Its input-wait interval is
not additive to the host causal interval because it starts before the host
generates and timestamps the next request.

This establishes a real copy-free phone-side mechanism. It does not establish
CUDA-VRAM-to-phone zero copy: an RTX 4060 Ti still needs to place an activation
in a host USB buffer. It also does not establish a model-quality result because
the HTP gate uses an FP32 square graph rather than RMSNorm, FFN, or attention.

## What remains

The next bounded step is one real model-shaped operator on the same imported
buffers, preferably RMSNorm and then fused SwiGLU, measured concurrently with
the RTX 4060 Ti part. Only after that should the project run a complete layer,
BurstGPT, and wall-power measurements. The result here makes that operator test
worth doing; it does not yet justify a full-model latency or energy claim.

No CUDA compute, complete transformer layer, full model, BurstGPT trace, power,
or energy measurement ran in this experiment.

## Evidence and terminal state

Transport evidence:
`../results/ffs_dmabuf_transport_v1/run_20260803T021815Z/`.

HTP evidence:
`../results/ffs_dmabuf_transport_v1/run_20260803T023728Z_htp/`.

Key authorities:

- transport `ANALYSIS.json`: `b3a526605c65ab3b867423d294fcb9520ba676a81a19b8184b2bf89e84648dfd`
- HTP `ANALYSIS.json`: `31d7f5924e92c3beb27b143969a3de558a27975e681a2094ac224d074dfc574b`
- experimental Hexagon library: `36b355536a7b5fdcd6aed45d75d28fb9d168426382f8b05c9789c0952a21708b`
- HTP worker: `1524f048214dc8f30b6f21150adc8e849c00c69655197943a58f73f9e6a8f07f`

The raw stock panic is preserved under `FAILURE_STOCK_KERNEL/`. Both evidence
roots contain verified recursive SHA-256 manifests.

After testing, a normal reboot returned the phone to stock kernel
`6.12.23-android16-5-gb3b66ace21e0-ab14672634-4k`, `ptp,adb`, SuperSpeed,
the original gadget bound, the experimental gadget unbound, and zero transport
workers. No commit or push was performed.

## Matched copied-HTP follow-up

A 2026-08-03 follow-up compared staged FunctionFS, mapped-buffer FunctionFS,
and direct HTP DMA-BUF with the same HTP graph. Direct DMA-BUF had no benefit
at 1.28 KiB, reduced median latency by 32.2% at 10 KiB, by 28.8% at 80 KiB,
and by 20.8% at 1 MiB with host persistent USB memory. See
`RESULTS_HTP_COMPARE.md` for the controls, limitations, and evidence.
