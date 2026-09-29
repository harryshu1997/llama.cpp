# Unified scheduler-owned BurstGPT physical A/B

Date: 2026-08-08 EDT.

Status: `PASS`.

This fresh matched pair ran on the RTX 4060 Ti desktop and OP15 through the
canonical control plane in `research_dev/scheduler`. The physical wrapper
created a plan before changing device state. The backend then derived the
model paths, runtime settings, operator split, bridge, phone worker, DMA-BUF
transport, and USB configuration from that plan. Each arm's result and
runtime-gate receipt bind the selected route and plan hash.

The trace contains 74 requests, 33,843 input tokens, and 11,605 output tokens.
Its SHA-256 is
`b20a9ba66ee3558d835a0e19ed3cfa4c31a4a9e8b4f9c085b29a14f80250a0ff`.

| metric | scheduler control | scheduler treatment | change |
| --- | ---: | ---: | ---: |
| trace makespan | 735.298 s | 631.548 s | -14.11% |
| CPU package energy | 115.527 kJ | 93.462 kJ | -19.10% |
| GPU board energy | 19.521 kJ | 18.850 kJ | -3.44% |
| connected phone energy | 0.615 kJ | 1.644 kJ | +167.20% |
| server energy | 135.048 kJ | 112.312 kJ | -16.84% |
| accounted fleet energy | 135.663 kJ | 113.956 kJ | -16.00% |
| completed work | 74 req / 11,605 tok | 74 req / 11,605 tok | equal |
| SLO requests met | 55 | 55 | equal |

## Scheduler bindings

| arm | route | reason | plan SHA-256 |
| --- | --- | --- | --- |
| control | `i3-cold-cpu-control-v1` | `CONTROL_BASELINE` | `sha256:7d6ec011bfdbada019e728c263cba74c16cc989dfefbb38a8557c087fa84f7bf` |
| treatment | `i3-cold-cpu-op15-ffn-v1` | `VERIFIED_COHORT_ENERGY_SAVING` | `sha256:df0aac19e12d9be077bab01f3244d11f6b9fca4db09dee5aaffdf755aa10f2be` |

The treatment executed 52,320 phone calls and 88.245 trillion phone MACs. It
transferred 6,876,610,560 bytes in each direction, assigned 76.82% of eligible
dense-FFN MACs to OP15, exposed 2.45% arithmetic-mean join wait, and recorded
zero reset recoveries. The maximum sampled phone temperatures were 29.4 C for
the battery and 55.0 C for the NPU zone. Pinned MMLU64 remained 27 / 64 in
both arms.

All pair gates passed: equal work, at least 10% fleet-energy saving,
non-regressed makespan and SLO count, at most 5% join wait, approximate quality
noninferiority, zero reset recovery, and both monitored runtime receipts.

## Scope

This result validates the exact measured cohort and bounded approximate
quality class. It does not authorize exact-token service, arbitrary trace
mixtures, or universal per-shape energy enforcement. Phone thermal telemetry
is collected locally during FunctionFS accessory mode and reduced after ADB is
restored. The remaining integration gap is an atomic pre-dispatch phone
thermal and runtime snapshot, not scheduler ownership of the execution plan.

The canonical record is
`UNIFIED_SCHEDULER_BURSTGPT_4060TI_OP15_R5_V1.json`. Its internal record
SHA-256 is `12183eb062294fdf0d225384ce121bc8dc6282da36895887fde385ed4e65c9a1`;
the file SHA-256 is
`49acd562d6abef02a393eb581e06a39bf124991c9e0ec0c5ae4c61979f6d868e`.
