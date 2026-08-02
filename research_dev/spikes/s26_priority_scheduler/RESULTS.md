# S26 Priority Scheduler Results

Verdict: `S26_PRIORITY_PHYSICAL_PASS`

Scope: real request-level scheduling mechanics and selected RTX 4060 Ti CUDA
compute time. This is not a phone-energy, network-energy, GPU-board-energy, or
total-system-energy result.

## Physical setup

- R0: RTX 4060 Ti CUDA `[0,8)` -> CUDA `[8,16)` -> CUDA `[16,48)`.
- R2: OP12 HTP0 `[0,8)` -> OP15 HTP0 `[8,16)` -> RTX 4060 Ti CUDA `[16,48)`.
- Work: 12 Gemma-4 requests, one input token and four generated tokens each.
- Priorities/SLOs: four P0 at 2 s, four P1 at 8 s, four P2 at 15 s.
- Both runs used the same coordinator, `RouteRunner.run_group`, persistent
  workers, B4 route points, and 5 ms bounded gather.

## Bound route points

| Route | Batch | Makespan upper bound | CUDA work | Route-point output |
|---|---:|---:|---:|---|
| R0 | 1 | 349,395 us | 212,995 us | `532,236772,236772,564` |
| R0 | 4 | 319,699 us | 313,949 us | `532,236772,236772,107` |
| R2 | 1 | 1,418,965 us | 155,690 us | `532,236772,236772,564` |
| R2 | 4 | 2,203,547 us | 208,875 us | `532,236772,236772,564` |

The isolated lockstep B4 points passed physical placement, row lineage, KV
drain, and exact B4 checks. R2 B4 reduces CUDA work by 105,074 us (33.5%)
against the equal-batch R0 B4 point. Output equality across batch sizes is not
claimed; each executed route point is checked against its own frozen oracle.

## Matched result

| Metric | All-CUDA control | Priority offload | Delta |
|---|---:|---:|---:|
| Route groups | 3 x R0 B4 | 1 x R0 B4 + 2 x R2 B4 | - |
| Completed / SLO misses | 12 / 0 | 12 / 0 | equal |
| P0 p95 latency | 341,517 us | 241,203 us | -29.37% |
| Summed CUDA island compute | 770,196 us | 579,537 us | -24.75% |
| Workload makespan | 788,850 us | 4,300,810 us | +3.512 s |

The policy admits the P0 B4 group to R0 first. It concurrently admits the P1
B4 group to R2 because the measured same-batch CUDA relief is positive and its
8 s deadline is feasible. The P2 group waits for phone credits, then follows R2
and meets its 15 s SLO. All CUDA, OP12, and OP15 stage calls use B4.

The longer makespan is intentional and material: low-priority slack is spent to
reduce selected-server work while protecting urgent latency. This checkpoint
does not show higher total throughput.

## Validation

- S26 offline suite: 39/39 pass.
- S24 `RouteRunner` suite: 15/15 pass.
- All route-point and final worker placement certificates pass.
- Final validator gates: 9/9 pass.
- Token, physical-batch, and control-mode mutations each fail closed with exit 2.
- Final artifacts: `results/physical_final3_20260721T150457Z/`.
- Profile bundle SHA-256: `416abc8946444d7ca1debf45dede37dad46a740be1f3ddb3216683d277ad07c9`.

## Limits

- Priorities and SLOs are synthetic; the execution and timings are physical.
- Phone F16 to desktop Q8 boundaries remain numerically uncertified for model
  quality beyond the frozen route-point token oracle.
- This is one run of one dense 12-request trace, not a statistical claim.
- GPU-board energy is not measured. Phone, network, and total-system energy are
  unknown.
