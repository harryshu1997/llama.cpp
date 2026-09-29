# Desktop-to-phone H2D bandwidth

Date: 2026-08-03 EDT.

Verdict:
`H2D_FUNCTIONFS_DMABUF_BANDWIDTH_PASS`.

## Result

The best sustained one-way application-payload rate from the desktop to OP15
was **465.86 MB/s** with 1 MiB DMA-BUF requests, host persistent USB memory,
and queue depth 4. This is 3.73 Gbit/s, 93.17% of the 500 MB/s USB symbol-rate
ceiling, and 74.54% of the advertised 5 Gbit/s bit rate.

Values below are medians across three fresh-process repetitions. Queued results
are throughput measurements; their per-request timestamps include time waiting
behind earlier queue entries and are not single-request latency.

| request | copied FunctionFS | phone DMA-BUF | DMA-BUF + host devmem | queued DMA-BUF + devmem |
| --- | ---: | ---: | ---: | ---: |
| 1 MiB | 328.75 MB/s | 355.71 MB/s | 403.01 MB/s | **465.86 MB/s**, depth 4 |
| 4 MiB | 333.55 MB/s | 359.40 MB/s | 408.12 MB/s | **460.68 MB/s**, depth 3 |
| 15 MiB | 337.40 MB/s | 374.36 MB/s | **424.74 MB/s** | not allocated |

At 15 MiB and queue depth 1, the direct phone DMA-BUF path with host persistent
memory improved throughput by 25.73% over copied FunctionFS. Queuing independent
requests raised the 1 MiB result by 41.59% over the copied control.

## What the modes measure

- `copy_malloc`: ordinary desktop memory and phone FunctionFS `read`.
- `dmabuf_malloc`: ordinary desktop memory with DWC3 DMA directly into the
  phone DMA-BUF.
- `dmabuf_devmem`: libusb persistent desktop USB memory and direct phone
  DMA-BUF.
- `dmabuf_devmem_qN`: the same direct path with N independent requests in
  flight.

The 5 Gbit/s SuperSpeed link uses 8b/10b encoding. Before USB packet and link
overhead, its one-direction data-symbol ceiling is:

```
5 Gbit/s * 8 / 10 / 8 = 500 MB/s
```

The 625 MB/s value obtained by dividing 5 Gbit/s by eight does not account for
encoding. The measured 465.86 MB/s leaves about 6.83% for packet, link,
controller, and software overhead relative to the encoding-adjusted ceiling.

## Host allocation constraint

The desktop had `usbcore.usbfs_memory_mb=16`. A 16 MiB persistent allocation
and four 4 MiB persistent buffers do not fit once libusb request metadata and
alignment are included. The largest valid single buffer was therefore 15 MiB,
and the 4 MiB queued case used depth 3. This is an allocation-pool limit, not a
measured USB bandwidth limit. No host-wide kernel parameter was changed.

## Validation and scope

The campaign contains 33 captures and 3,870 paid requests. Every protocol
completion and worker status passed, every captured kernel-fault log was empty,
and every case restored `ptp,adb` at SuperSpeed.

This benchmark measures desktop host memory to phone transport only. It does
not include CUDA VRAM readback, HTP/GPU compute, or a result payload of model
size. The direct DMA-BUF path requires the tested FunctionFS kernel fix; the
stock OP15 kernel has the previously documented reversed DMA direction and is
unsafe for this operation.

Analysis:
`../results/ffs_dmabuf_transport_v1/run_20260803T093232Z_h2d_bandwidth/ANALYSIS.json`.

Analysis SHA-256:
`b152995191ce49f392c4c3961e9dd9b0dc67f0ef479ea7656e4926fb1a681422`.
