# GPU capacity overflow with OP15 FFN assistance

This adapter tests one oversized Gemma route selected by the unified scheduler.
The 23.832 GB model cannot fit entirely in the 16 GB RTX 4060 Ti. The capacity
planner rejects the full-GPU candidate, places layers 23-47 on CUDA, keeps
layers 0-22 on the CPU, and allows the OP15 to assist only eligible FFN slices
from those CPU layers.

## Physical result

One matched 17-request BurstGPT control/treatment pair passed on 2026-08-08.
Both arms replayed exactly 11,476 input tokens and 6,919 output tokens with the
same 25-GPU-layer/23-CPU-layer placement.

| Metric | GPU+CPU control | GPU+CPU+OP15 | Change |
| --- | ---: | ---: | ---: |
| Makespan | 788.001 s | 694.014 s | -11.93% |
| CPU package energy | 63.363 kJ | 51.163 kJ | -19.25% |
| GPU board energy | 27.080 kJ | 24.661 kJ | -8.93% |
| Server compute-device energy | 90.443 kJ | 75.824 kJ | -16.16% |
| Whole-phone energy | 0.572 kJ | 1.706 kJ | +198.30% |
| Accounted fleet energy | 91.015 kJ | 77.530 kJ | -14.82% |

The treatment served 26,220 phone RPCs, transferred 1,212,103,680 bytes in
each direction, covered 6,862 token rows per CPU layer, executed
11,170,747,514,880 phone MACs, and recorded zero USB reset recoveries. Work was
validated from the observed shape histogram rather than a fixed RPC count,
because continuous batching can change RPC morphology without changing the
trace's model-level token work.

The accounted boundary is CPU package RAPL plus GPU board NVML energy plus
synchronized whole-phone USB input and battery discharge over the paid trace
interval. It excludes DRAM outside package RAPL, motherboard, fans, storage,
and AC conversion.

The byte-exact pair record and raw accounting inputs are under
[`results/physical_pair_r1`](results/physical_pair_r1). The record SHA-256 is
`5d3769bd947b2479e7ed85d0076eae1921db8fb885a6cf08bbc9b468b3cefda7`.

This is one matched physical pair. It supports the route as shadow evidence,
but it is not sufficient to promote the route to general energy enforcement.
Repeat pairs and thermal-order controls are still required for promotion.

## Components

- `plan_gpu_overflow.py` builds the hash-bound capacity and offload plan.
- `run_gpu_overflow_arm.sh` configures the desktop, synchronized phone logger,
  staged FunctionFS worker, DMA-BUF bridge, and low-level trace runner.
- `analyze_gpu_overflow_pair.py` validates equal work and compares fleet energy.
- `test_gpu_overflow.py` covers placement, artifact capabilities, dynamic work
  conservation, tamper rejection, and fleet accounting.
