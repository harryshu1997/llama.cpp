# S17 results

Verdict: `COALESCING_KERNEL_MECHANICS_PASS_NUMERICAL_ELIGIBILITY_FAIL`

CP0 is complete. CP1 ran on the real OP15 HTP0 and the selected A6000 CUDA0.
The B64 coalescing mechanism works and is faster, but the predeclared 5e-3
route-matched residual gate fails. CP2 is not authorized.

## Artifacts and placement

Gemma `[6,12)` was materialized without block renumbering:

~~~text
sha256 e4fdd28f0f5fef0692a3c93c6308ea48ac971a97428f17199ecbc921020b1c36
bytes  2741158560
~~~

The local and OP15 copies match. Every completed middle run reports
`SCHEDULED_PLACEMENT_OK`, zero missing-buffer nodes, and all 2,076 AUTO-route
compute nodes on HTP0 or CUDA0 as declared. The explicit HTP route reports all
2,232 compute nodes on HTP0. Both phone prefix captures are finite, use HTP0
except the declared CPU GET_ROWS seam, and bind identical `[0,6)` shard hashes.

## Synthetic support screen

The first screen used deterministic finite residuals only to establish support.

| Metric | CUDA0 | OP15 HTP0 |
|---|---:|---:|
| B64 p50 | 9.70 ms | 282.21 ms |
| 2xB32 p50 | 14.12 ms | 432.18 ms |
| coalescing speedup | 1.46x | 1.53x |
| B64 vs 2xB32 relative L2 | 1.538e-2 | 0 |
| repeat relative L2 | 0 | 0 |

HTP B64 is bit-identical to two HTP B32 launches and faster. CUDA changes its
numerical tiling between B32 and B64; that diagnostic difference is reported
but is not used as an HTP cross-sequence gate. HTP B64 versus same-shape CUDA
B64 is 9.751e-3 relative L2 with 8/64 row-argmax mismatches, so the synthetic
screen correctly fails the cross-backend gate.

## Real phone-boundary screen

The second screen replaced synthetic inputs with actual B32 `[0,6)` outputs:

- OP15/v81 head B32: 254.07 ms round trip, HTP0 placement.
- OP12/v75 head B32: 1,115.87 ms round trip, HTP0 placement.
- The two 32-row tensors were concatenated without modification and sent to
  identical `[6,12)` HTP0 and CUDA0 middle contexts.

| Middle route | B64 p50 | 2xB32 p50 | speedup | HTP vs CUDA relative L2 | row argmax mismatch |
|---|---:|---:|---:|---:|---:|
| AUTO | 281.19 ms | 430.38 ms | 1.53x | 9.001e-3 | 1/64 |
| explicit attention | 314.69 ms | 473.42 ms | 1.50x | 8.886e-3 | 1/64 |

For both attention routes, HTP B64 is bit-identical to HTP 2xB32 and every
repeat is bit-identical. Explicit attention is slower and does not meet the
5e-3 cross-backend gate, so it is rejected rather than selected post hoc.

## Interpretation

The hardware result supports the batching premise: merging two ready B32
activation cohorts into one HTP B64 middle launch amortizes work by about 1.5x
without HTP cross-sequence bleed at this empty-context point. It does not yet
support the full hierarchical route. The HTP/CUDA residual discrepancy exceeds
the frozen threshold under both kernel routes, and no same-shape end-to-end
token/logit run or C>=512 decode-context run has been executed.

The next bounded work is numerical localization or a predeclared end-to-end
oracle revision. It must not silently raise the threshold. CP2 dual residency,
the multi-ingress runtime, mixed-trace evaluation, and energy remain stopped.

Phone, USB, host-wall, and total-system energy are UNKNOWN.

The claim-bearing reports are `frozen-auto-results/report.json` and
`frozen-explicit-results/report.json`. Each contains immutable copies of both
executed harness sources. `validate_cp1.py` reopens every bound artifact and
raw F32 tensor and reproduces both FAIL verdicts. Earlier `screen-results/` and
`real-boundary*-results/` directories are diagnostic only.
