# Bidirectional FunctionFS DMA-BUF bandwidth

Date: 2026-08-03 EDT.

Verdict:
`FUNCTIONFS_DMABUF_BIDIRECTIONAL_PASS`.

## Result

The DMA-BUF path reaches approximately 465 MB/s in either direction when each
direction is measured independently. With equal 1 MiB transfers active in both
directions, it sustains 869.21 MB/s aggregate, or 434.61 MB/s per direction.

Values are medians across three fresh-process repetitions. H2D and D2H rates
exclude the 64-byte control message in the opposite direction.

| payload and queue | H2D | D2H | D2H change versus H2D |
| --- | ---: | ---: | ---: |
| 1 MiB, depth 4 | 465.81 MB/s | 465.15 MB/s | -0.14% |
| 4 MiB, depth 3 | 462.44 MB/s | 463.09 MB/s | +0.14% |
| 15 MiB, depth 1 | 428.24 MB/s | 454.67 MB/s | +6.17% |

The simultaneous 1 MiB full-duplex result is:

| H2D | D2H | aggregate | full-duplex symbol-ceiling use |
| ---: | ---: | ---: | ---: |
| 434.61 MB/s | 434.61 MB/s | 869.21 MB/s | 86.92% |

The 5 Gbit/s SuperSpeed link has separate transmit and receive lanes. After
8b/10b encoding, each direction has a 500 MB/s symbol-rate ceiling, so the
full-duplex symbol-rate ceiling is 1,000 MB/s aggregate. The aggregate result
above 500 MB/s confirms that OUT and IN traffic overlapped on the wire.

Compared with the sum of the two independent 1 MiB rates, simultaneous duplex
traffic loses 6.63%. Each simultaneous direction still carries 3.48 Gbit/s of
application payload.

## Measurement

Every case used direct phone DMA-BUFs and libusb persistent host USB memory.
The primary 1 MiB cases used queue depth 4, the 4 MiB cases used depth 3, and
the 15 MiB cases used synchronous depth 1. The reduced depths keep persistent
allocations within the desktop's 16 MiB usbfs memory pool.

The campaign contains 21 captures and 2,580 paid request/response pairs. All
payload sequence and sentinel checks passed, all phone workers exited with
status zero, all captured kernel-fault logs were empty, and every case restored
the normal gadget at SuperSpeed.

Queued request timestamps include time resident behind other entries and
should not be interpreted as single-transfer latency. The result measures
desktop host memory to phone memory and back. It does not include CUDA VRAM
readback, HTP/GPU computation, or model execution.

The direct path requires the tested FunctionFS DMA-direction fix. Testing used
the verified temporary boot image and did not flash a partition. The phone was
returned to its stock kernel, `ptp,adb`, and the normal SuperSpeed gadget
afterward.

Analysis:
`../results/ffs_dmabuf_transport_v1/run_20260803T103913Z_bidirectional/ANALYSIS.json`.

Analysis SHA-256:
`8392a75f562a999d40a8b88b4534e4fb192a1fc2bcd10f09db2ff9b6c6be0bda`.
