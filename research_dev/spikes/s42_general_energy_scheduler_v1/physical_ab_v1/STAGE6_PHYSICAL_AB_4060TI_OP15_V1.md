# Stage 6 full BurstGPT physical A/B

Verdict: `PASS`.

| metric | CPU control | CPU plus OP15 | change |
| --- | ---: | ---: | ---: |
| makespan | 732.607 s | 631.293 s | -13.83% |
| CPU package energy | 115.573 kJ | 95.172 kJ | -17.65% |
| GPU board energy | 20.006 kJ | 19.290 kJ | -3.58% |
| connected phone energy | 0.625 kJ | 1.689 kJ | +170.21% |
| accounted fleet energy | 136.205 kJ | 116.151 kJ | -14.72% |
| completed work | 74 req / 11,605 tok | 74 req / 11,605 tok | equal |
| SLO requests met | 55 | 55 | +0 |

## Treatment checks

- Dynamic phone calls: 52320.
- Phone share of eligible dense-FFN MACs: 76.82%.
- Arithmetic-mean exposed join wait: 2.31%.
- Maximum executor topology-heartbeat gap: 1.051 s.
- Maximum RPC-progress gap, including idle intervals: 7.476 s.
- Bridge reset recoveries: 0.
- Pinned MMLU64: 27 / 64 control, 27 / 64 treatment.

one successor validation pair; existing I3 three-pair certificate remains the promotion authority.
