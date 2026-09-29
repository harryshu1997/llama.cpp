# RTX 4060 Ti plus OP15 kernel-energy profile V1

Date: 2026-08-06 EDT.

## Result

Stage 1 is complete. The physical campaign produced three qualified
repetitions for every required row and materialized:

- `../../scheduler/profiles/MEASURED_4060TI_OP15_KERNEL_PROFILE_V1.json`
- schema: `s42-kernel-energy-profile-v1`
- SHA-256: `fbc2375c9fea20c39c783778d27a0433d88f5c432f19a370684e36559da1092e`
- 3 idle domains, 16 kernel rows, 7 PCIe rows, 6 USB rows, 9 one-time
  preparation/load/switch rows, and 4 derived directional link models

The energy boundary is the connected fleet: i9-12900K package, RTX 4060 Ti
board, and whole OP15. Isolated rows retain the domains visible during their
physical acquisition. The materializer records a SHA-256 ID for every raw
power trace, paid-window marker, and benchmark result used by a row.

## Kernel buckets

Latency is the median of the three physical repetitions. Dynamic energy UCB
uses maximum observed active power and latency after subtracting measured idle
power. GOPS/s counts all three dense FFN matrix multiplications.

| Backend | M | FFN columns | Latency ms | Effective GOPS/s | Dynamic energy UCB, mJ | Latency LOO max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU | 1 | 15,360 | 3.100 | 114.2 | 384.409 | 1.00% |
| CPU | 8 | 15,360 | 6.710 | 421.9 | 872.415 | 0.85% |
| CPU | 32 | 15,360 | 27.277 | 415.2 | 3,090.412 | 0.65% |
| CPU | 128 | 15,360 | 105.341 | 430.0 | 12,072.084 | 3.23% |
| CUDA | 1 | 15,360 | 0.380 | 931.1 | 54.050 | 0.02% |
| CUDA | 8 | 15,360 | 0.440 | 6,440.3 | 78.671 | 0.66% |
| CUDA | 32 | 15,360 | 0.497 | 22,784.4 | 87.966 | 0.28% |
| CUDA | 128 | 15,360 | 0.866 | 52,305.0 | 155.169 | 0.90% |
| HTP | 1 | 9,664 | 1.146 | 194.3 | 6.573 | 11.92% |
| HTP | 8 | 9,664 | 2.725 | 653.7 | 9.987 | 4.14% |
| HTP | 32 | 9,664 | 2.883 | 2,471.6 | 11.408 | 0.53% |
| HTP | 128 | 9,664 | 3.733 | 7,635.0 | 13.052 | 0.87% |
| Adreno native Q4 | 1 | 512 | 1.634 | 7.2 | 2.871 | 1.43% |
| Adreno F16 xmem | 16 | 512 | 5.088 | 37.1 | 9.795 | 0.98% |
| Adreno F16 xmem | 32 | 512 | 5.376 | 70.2 | 9.894 | 0.10% |
| Adreno F16 xmem | 128 | 512 | 8.452 | 178.6 | 15.486 | 0.04% |

These are shape buckets, not an interpolation license. HTP M=1 has the
largest repeat variation, so a certified plan must use its maximum/UCB row.

## Transfer buckets

| Path | Payload | Median paid service | Median throughput | Dynamic energy UCB |
| --- | ---: | ---: | ---: | ---: |
| PCIe H2D | 8 KiB | 0.010 ms | 0.82 GB/s | 0.551 mJ |
| PCIe H2D | 1 MiB | 0.629 ms | 1.67 GB/s | 32.487 mJ |
| PCIe H2D | 4 MiB | 2.505 ms | 1.67 GB/s | 130.420 mJ |
| PCIe D2H | 8 KiB | 0.009 ms | 0.91 GB/s | 0.485 mJ |
| PCIe D2H | 1 MiB | 0.621 ms | 1.69 GB/s | 32.098 mJ |
| PCIe D2H | 4 MiB | 2.466 ms | 1.70 GB/s | 127.154 mJ |
| PCIe duplex | 2 MiB aggregate | 0.710 ms | 2.95 GB/s aggregate | 36.990 mJ |
| USB sync duplex | 16 KiB aggregate | 0.178 ms | 92.3 MB/s aggregate | 0.477 mJ |
| USB H2P | 1 MiB | 2.223 ms | 471.8 MB/s | 5.322 mJ |
| USB P2H | 1 MiB | 2.222 ms | 471.9 MB/s | 5.026 mJ |
| USB duplex | 2 MiB aggregate | 2.364 ms | 887.1 MB/s aggregate | 6.556 mJ |
| USB H2P | 4 MiB | 8.934 ms | 469.5 MB/s | 22.040 mJ |
| USB P2H | 4 MiB | 8.893 ms | 471.7 MB/s | 21.470 mJ |

The 4 MiB FunctionFS runs use queue depth 3. Depth 4 needs more than the
host's 16 MiB usbfs allocation pool; depth 3 fits that fixed pool and retains
the same approximately 470 MB/s directional throughput. The complete 4 MiB
RPC latency is about 26 ms because three requests are concurrently in flight,
while the scheduler's steady-state service cost is about 8.9 ms.

The derived conservative directional fits are 1.68 GB/s PCIe H2D, 1.70 GB/s
PCIe D2H, 469 MB/s USB H2P, and 472 MB/s USB P2H. They are valid only over
their measured payload ranges.

## One-time epoch costs

| Cost | Median latency | Maximum latency | Reported dynamic energy |
| --- | ---: | ---: | ---: |
| HTP Q4 pack, 9,664 columns | 49.240 ms | 49.426 ms | 94.206 mJ |
| Adreno Q4 upload, 512 columns | 18.468 ms | 18.511 ms | 32.246 mJ |
| Adreno Q4 to F16 reconstruction | 23.349 ms | 23.356 ms | 38.818 mJ |
| Adreno F16 upload, 512 columns | 3.795 ms | 3.807 ms | 6.666 mJ |
| Complete Adreno F16 xmem preparation | 56.836 ms | 58.622 ms | 100.595 mJ |
| Load Qwen3-14B Q4_K_M to CUDA | 6.778 s | 8.662 s | 306.216 J median dynamic |
| Load Gemma4-12B Q8_0 to CUDA | 9.153 s | 19.263 s | 466.825 J median dynamic |
| Switch Gemma to Qwen CUDA residency | 6.366 s | 6.498 s | 329.215 J median dynamic |
| Switch Qwen to Gemma CUDA residency | 9.373 s | 9.381 s | 484.697 J median dynamic |

Repacking and upload costs are charged once per matching residency/layout
epoch, not once per operator invocation. Model-load cold-cache variation is
large, especially for Gemma, so admission must retain the measured maximum
latency unless the model is already proven resident.

## Qualification boundary

- Exact measured shape and payload buckets may be used by the planner.
- Derived link equations may be used only inside their measured ranges.
- Every runtime energy decision must use the conservative maximum/UCB values.
- Arbitrary interpolation, unseen layouts, and unseen concurrency buckets fail
  closed.
- Additive kernel composition is not yet an executable certified route. It
  remains fail-closed until the held-out full-model validation in stage 3.

The raw campaign stays outside Git under
`/home/zhihao/s42-kernel-energy-v1/results`. Invalid and interrupted captures
were retained with `.invalid-*` suffixes and excluded by the materializer.
