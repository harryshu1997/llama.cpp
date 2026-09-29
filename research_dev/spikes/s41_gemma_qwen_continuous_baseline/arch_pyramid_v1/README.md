# Desktop and phone data-movement pyramid

The diagram combines live probes from the RTX 4060 Ti desktop and OP15 with
existing sustained hardware and transport campaigns. It intentionally avoids
model-specific token rates, model names, and whole-model latency.

Files:

- `desktop_phone_pyramid.svg`: editable source and primary artifact.
- `desktop_phone_pyramid.png`: rendered 2400 x 1000 image.
- `cuda_link_probe.cpp`: pinned-host CUDA copy probe used on the desktop.
- `direct_read_probe.c`: `O_DIRECT` sequential read probe used on both devices.

## Live probe results

The 2026-08-05 EDT live probes produced:

| tier | result |
| --- | ---: |
| CUDA H2D, 256 MiB x 20 | 1.680 GB/s |
| CUDA D2H, 256 MiB x 20 | 1.705 GB/s |
| CUDA D2D logical copy | 123.5 GB/s |
| CUDA D2D memory traffic | 247.1 GB/s |
| desktop NVMe `O_DIRECT` read, 1 GiB | 1.669 GB/s |
| OP15 UFS `O_DIRECT` read, 1 GiB | 3.444 GB/s |

The live desktop inventory also reported:

| component | capacity or limit |
| --- | ---: |
| RTX 4060 Ti memory | 16,380 MiB |
| RTX 4060 Ti idle board power | 7.4 W |
| RTX 4060 Ti board power limit | 165 W |
| desktop RAM visible to the OS | 30 GiB |
| i9-12900K configured PL1 / PL2 | 180 W / 241 W |
| OP15 RAM visible to the OS | 14.8 GiB |

The GPU reported PCIe Gen1 x8 during the copy probe even though its capability
is Gen4 x8. The diagram therefore does not reuse the unverified 12 GB/s label
from the input sketch. The negotiated link should be fixed before a
publication-grade benchmark.

## Hardware-level inputs

- Direct FunctionFS DMA-BUF transport: 465.81 MB/s H2D and 465.15 MB/s D2H
  independently; 434.61 MB/s per direction simultaneously.
- RTX 4060 Ti: 22.1 TFLOP/s rated FP32 throughput and 288 GB/s rated memory
  bandwidth; the live copy probe sustained 247.1 GB/s of memory traffic.
- i9-12900K: 16 cores, 24 threads, up to 5.2 GHz, 1.41 TFLOP/s median
  sustained FP32 GEMM throughput, and 50 GB/s measured isolated streaming
  bandwidth. See `RESULTS_I9_12900K_PEAK.md`.
- OP15 Adreno 840: 1.1-2.2 TFLOP/s measured dense GEMM throughput and
  66-77 GB/s measured effective streaming bandwidth.
- OP15 Hexagon HTP v81: 11.66 TFLOP/s median sustained FP16 GEMM throughput
  for `[8192,2048] x [2048,4096]`. Three fresh processes measured 11.61,
  11.67, and 11.66 TFLOP/s. See `RESULTS_OP15_HTP_PEAK.md`.
- OP15 active power estimate: 3.99 W total and 1.74 W marginal over idle.
  This combines USB input and duty-scaled battery current; it is not an
  isolated accelerator-rail measurement. A conservative fleet budget remains
  4.5 W per active phone.

The compute figures use different precision and kernels. They describe each
hardware tier but are not direct cross-device arithmetic comparisons. Memory
bandwidth and transport bandwidth are the more comparable quantities.

The direct DMA-BUF bandwidth campaign used the temporary FunctionFS
DMA-direction kernel fix. The stock OP15 kernel was restored afterward, so
that requirement remains explicit in the figure.

## Repository evidence

- `../tp_slice_probe_v1/RESULTS.md`
- `../tp_operator_split_v1/RESULTS_Q6K_HTP_V1.md`
- `../tp_operator_split_v1/offload_solver_v1/README.md`
- `../tp_operator_split_v1/ffs_dmabuf_transport_v1/RESULTS.md`
- `../tp_operator_split_v1/ffs_dmabuf_transport_v1/RESULTS_H2D_BANDWIDTH.md`
- `../tp_operator_split_v1/ffs_dmabuf_transport_v1/RESULTS_BIDIRECTIONAL_BANDWIDTH.md`
