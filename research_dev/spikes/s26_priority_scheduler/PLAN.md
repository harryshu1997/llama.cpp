# S26 Priority-Safe Online Scheduler

Status: `S26_PRIORITY_PHYSICAL_PASS`; selected CUDA compute measured, energy not measured.

## Purpose

Replace S24's `MAX_OFFLOAD_SLOWEST_FEASIBLE_FIXED_ROUTE` rule with a small
request-level controller that protects urgent work and admits phone work only
in a measured, server-relieving batch.

The first physical policy intentionally exposes only two routes:

- R0: CUDA `[0,8)` -> CUDA `[8,16)` -> CUDA `[16,48)`;
- R2: OP12 `[0,8)` -> OP15 `[8,16)` -> CUDA `[16,48)`.

R1 is excluded from the first gate. This keeps the mechanism test focused on
priority protection and one measured B4 phone route.

## Policy contract

1. Priority 0 is urgent and is considered before every lower-priority request.
2. Urgent requests use the largest measured R0 point already present in READY.
   Four ready urgent requests launch as B4; a lone urgent request launches B1
   immediately and never waits for missing peers.
3. Lower-priority requests target the measured R2 B4 point.
4. R2 launches only when its measured CUDA work is lower than the same-batch
   R0 point and every request is predicted to meet its deadline.
5. A lower-priority request may wait only until its latest safe start.
6. At latest safe start, a measured smaller point or immediate R0 fallback is
   allowed. An unmeasured batch size is never invented.
7. R2 atomically reserves OP12, OP15, and CUDA-tail credits before launch.
8. Four CUDA-prefix, CUDA-middle, and CUDA-tail credits remain protected for
   priority-0 work. Lower-priority fallback may borrow them only when no urgent
   request is pending or active.
9. A request keeps one route and route epoch until exact completion.
10. Unknown profile, insufficient relief, stale completion, capacity overflow,
    deadline overflow, and duplicate ownership fail closed.

## Checkpoints

- [x] CP0: bind R0 B1/B4 and R2 B1/B4 points to physical artifacts.
- [x] CP1: implement deterministic priority, latest-start, relief, and credit
      policy with mutation tests.
- [x] CP2: connect the policy to the existing physical RouteRunner without
      changing frozen S24 code or the StageNet protocol.
- [x] CP3: run the 12-request deterministic three-class trace on the real
      OP12, OP15, and RTX 4060 Ti.
- [x] CP4: compare priority-0 latency, SLO misses, CUDA work, route batches,
      and conservation against a matched all-CUDA controller run.

CP4 passed: no extra priority-0 miss, no priority-0 p95 regression, and lower
matched CUDA work. The next separate gate is repeated selected-GPU board-energy
acquisition. Phone, network, and total-system energy remain unknown.
