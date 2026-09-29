# Pixel CPU plus GPU FFN feasibility

2026-09-23 19:10 UTC. Evidence/arithmetic review **PASS**. Concurrent CPU/GPU FFN
speedup **NOT VERIFIED**; no new hardware test or implementation in this review.

## Measured starting point

| Measurement | CPU | GPU | Concurrent |
| --- | ---: | ---: | ---: |
| Full 17408-column FFN worker | 18.291 ms | 20.610 ms | Not measured |
| Half 8704-column FFN worker | 9.162 ms | 11.848 ms | Not measured |
| Matched logical streaming, CPU6 configuration | 30.493 GB/s | 37.968 GB/s | 45.537 GB/s aggregate |

Under simultaneous streaming, CPU6 falls to 20.278 GB/s and GPU to
25.324 GB/s. Both retain about two thirds of their isolated rates.
The best repeated aggregate stream uses one CPU core plus GPU: 46.515
GB/s, 22.91% above its matched GPU controls. That one-core stream result
cannot be substituted for six-thread CPU FFN performance.

## What splitting would do

Divide the FFN intermediate columns between CPU and GPU, keeping the
corresponding gate, up and down weights resident on their assigned backend.
Both branches consume the same input and execute concurrently. Add their
FP32 down-projection outputs on the phone, then return one F16 response.
This exposes independent work within one layer; the next layer still waits
for the sum. Output rounding/reduction differences require qualification.

A full FFN currently comprises four 4352-column blocks. Assigning whole
blocks allows CPU/GPU fractions 25/75, 50/50 and 75/25 and preserves the
selected GPU kernel shapes. At 50/50, each backend executes the measured
half-width workload. Ignoring all contention gives 11.848 ms.
As a sensitivity check, scaling each branch by its measured streaming
slowdown gives CPU 13.778 ms and GPU 17.763 ms:
**17.763 ms before extra merge/synchronization**, only
2.89% below the tuned CPU.
This assumption is unvalidated: streams and FFNs have different compute,
scheduling, clock and fixed-cost behavior. It establishes neither a speedup
nor an upper/lower bound. Uneven partitions and CPU thread counts need testing.

Reading the full 510 MiB of weights at 46.515 GB/s would take
11.497 ms. That is a streaming-only reference,
not an attainable FFN latency or measured physical DRAM limit.

## Existing implementation and next qualification

The dirty `examples/layersplit/ffn-split-worker.cpp` already contains an
experimental secondary-backend path with a helper thread, disjoint weight
slices, parallel graph execution and FP32 output addition. It belongs to
ongoing work and was only inspected. It lacks the newly qualified Pixel CPU
persistent-pool settings, and subdivides each existing block. Its smaller
matrix shapes may miss the selected Pixel GPU fast path. It is not yet a
qualified Pixel CPU/GPU worker.

A bounded phone-local comparison should retain the six-thread persistent
CPU baseline, test whole-block assignments and vary CPU thread counts, and
record both branch times, total worker time, merge cost and numerical error.
The half-width phone workload needs its own comparison: its best measured
CPU time is already 9.162 ms, and launching both engines has proportionally
larger fixed costs. Server overlap and energy must be measured separately.
Phone-local testing uses the Pixel lock; it does not require the desktop
campaign lock. No test was started or queued for this assessment.

[CPU measurements](PIXEL_CPU_TUNE_RESULTS.md),
[bandwidth measurements](PIXEL_BANDWIDTH_RESULTS.md),
[calculations](PIXEL_CPU_GPU_FFN_ASSESSMENT.json),
[server overlap budget](PIXEL_CPU_SERVER_OVERLAP_ASSESSMENT.md).
