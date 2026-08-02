# S28 Priority-Safe Shared Tail

## Objective

Run one bounded request-level policy over the measured R0 and R2 routes while
using one physical CUDA-tail queue. Urgent P0 work must not wait behind queued
background work. Compatible P1/P2 work may batch within the background band.

This checkpoint does not add routes, change model residency, stream weights,
or implement arbitrary layer exits.

## Frozen routes

- R0: 4060 Ti `[0,8)` -> 4060 Ti `[8,16)` -> 4060 Ti `[16,48)`
- R2: OP12 `[0,8)` -> OP15 `[8,16)` -> 4060 Ti `[16,48)`

The policy may use only the previously measured B1 and B4 route points. The
60-request workload uses real-derived BurstGPT arrivals and observed demand
metadata, but the executed one-token/four-step shape, priorities, and SLOs are
synthetic.

## Gates

1. One tail queue object and one tail sequence-credit pool serve R0 and R2.
2. Queue order is priority, latest safe dispatch, then FIFO.
3. A physical batch never combines P0 and background rows.
4. All 60 requests have exactly one terminal outcome and no SLO miss.
5. P0 p95 latency in treatment is at most 1.10x the all-CUDA control.
6. Both phone workers execute only their declared ranges with no missing
   placement buffers.
7. Measured summed CUDA-island compute time is lower in treatment.

Gate 7 is a server-work result, not an energy result.

## Controls

- C0: all requests use R0.
- C1: P0 uses R0; P1/P2 use R2 when the measured route and SLO permit it.

The same resident processes execute C0 followed by C1. C0 ends with DETACH;
C1 ends with STOP. This avoids model reloads between the matched controls.

## Stop conditions

- Any unmeasured route or batch point.
- Any urgent/background batch mix.
- Any missing request, duplicate terminal outcome, stale route epoch, live KV
  state after cleanup, placement fallback, or SLO miss.
- Any claim of token correctness, phone energy, USB energy, or total-system
  energy from this checkpoint.
