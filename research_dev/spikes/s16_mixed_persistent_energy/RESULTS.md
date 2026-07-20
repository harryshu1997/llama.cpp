# S16 Results

Verdict: `MIXED_PERSISTENT_MECHANICS_PASS_GPU_BOARD_RELIEF_UNRESOLVED`

The six-row acquisition completed and independently revalidated. The scheduling
mechanism works on the real selected A6000 and OP15, but the measured board
relief is too small for the available NVML instrument.

## Matched work

Each row completed:

- 339,440 high-priority BGE encodes at the measured B16 knee.
- 20 Gemma cohorts x B32 x 8 tokens = 5,120 low-priority tokens.
- One resident CUDA process for all 20 cohorts.
- For P2, one resident OP15 HTP process for all 20 cohorts.

All 120 Gemma cohorts returned exact frozen reference tokens. P0 used CUDA0
with only `GET_ROWS` on CUDA_Host. P2 used OP15 HTP0 with only the declared
`GET_ROWS` CPU seam and a CUDA0 `[8,48)` tail. Every low cohort completed inside
the BGE paid interval and below the frozen 5 s SLO.

## Latency

| Metric | P0 server-only | P2 OP15 head | Result |
|---|---:|---:|---:|
| Median BGE p95 | 4,112 us | 3,983 us | 0.969x, PASS |
| Representative Gemma p95 | 619 ms | 2,889 ms | both below 5 s |
| Maximum Gemma cohort | 705 ms | 3,094 ms | both below 5 s |

The phone route is about 4.6x slower for one B32 cohort, but low priority and
the 80 s BGE interval provide enough slack to hide all twenty cohorts.

## Selected-GPU board energy

| Pair | P0 | P2 | Raw saving | Uncertainty-adjusted lower bound |
|---:|---:|---:|---:|---:|
| 0 | 24,335.4 J | 24,019.2 J | 1.30% | -560.4 J |
| 1 | 24,116.1 J | 23,952.2 J | 0.68% | -717.8 J |
| 2 | 24,139.0 J | 23,957.0 J | 0.75% | -701.3 J |

Every pair points in the same direction, but the median raw saving is only
0.75 percent. The comparison subtracts the Ampere `power.draw` 5 W uncertainty
floor from both members of each pair. Its median lower bound is negative, so no
selected-GPU energy saving is claimed.

Each row has 174-176 actual power-value changes, a maximum arrival gap below
146 ms, an invariant 300 W limit, and an 87.4-89.0 s paid window. The result is
not a short-trace or oversampling failure.

## Interpretation

The real mechanism is valid: READY phone work can overlap a busy mixed server,
preserve the high-priority class, meet an absolute low-priority SLO, and keep
weights/KV resident across repeated batches. A single shallow `[0,8)` island is
not enough to establish energy relief while BGE already keeps the A6000 near
its power frontier. It removes 16.7 percent of Gemma layers, while the BGE load
dominates the approximately 24 kJ paid window.

The next test must increase useful phone contribution without serially chaining
phones. Add OP12 as an independent READY request lane, replay two disjoint B32
low-priority cohorts, and evaluate both selected-GPU energy and iso-power useful
work. If the fleet still produces less than 10 percent relief/work gain, stop
the energy claim for this hardware and retain the system as a capacity/SLO
mechanism.

Phone, USB, host-wall, and total-system energy remain unknown.

## Verification

- Persistent monodriver input/reset suite: 10/10.
- S16 contract tests: 10/10.
- Short physical screen: PASS.
- Full independent report validator: PASS, verdict reproduced as FAIL_GATE.
- Report SHA-256: `31ca3ad85a92ad25d4221c2f1f298ec9a6a27d599d132c9f085f7fad6a28fc3f`.
- No commit or push.

