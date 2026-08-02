# S25 Results

## Verdict

`THREE_DEVICE_CONTINUOUS_LIFECYCLE_PASS`

The actual OP12, OP15, and RTX 4060 Ti workers completed a changing request set
without restarting or clearing another request's KV state. This closes the
basic request-lifecycle gap. It does not show that the current route improves
energy, throughput, or SLO attainment.

## Physical run

- OP12: HTP0, layers `[0,8)`, four resident sequence slots.
- OP15: HTP0, layers `[8,16)`, four resident sequence slots.
- Desktop: RTX 4060 Ti CUDA0, layers `[16,48)`, eight resident sequence slots.
- Activation transport: desktop directly contacted each phone over the local
  network. The A6000 GPU did not execute the route.
- Dynamic membership: `AB, AB, CB, CB, CD, D`.
- Output lengths: `A=2, B=4, C=3, D=2`.
- Slot reuse: C reused A's slot while B remained live; D reused B's slot while C
  remained live.
- Physical batch size on every stage: `2,2,2,2,2,1`.
- Same-route B1 greedy token comparison: all four requests equal.
- Final state: zero active sequences on all workers, then DRAIN and STOP.
- End-to-end oracle plus treatment time: 6.057396 seconds.

Dynamic output tokens:

| Request | B1 oracle | Continuous treatment |
| --- | --- | --- |
| A | `532,236772` | `532,236772` |
| B | `532,236772,236772,564` | `532,236772,236772,564` |
| C | `532,236772,236772` | `532,236772,236772` |
| D | `532,236772` | `532,236772` |

All three placement certificates are `SCHEDULED_PLACEMENT_OK` with zero missing
compute buffers. OP12 used HTP0 except for its declared CPU `GET_ROWS`; OP15 used
HTP0 for all compute nodes; the tail used CUDA0 for all compute nodes.

## Preliminary failure retained

The first run used different arbitrary synthetic start tokens per request. The
lifecycle and placement completed, but request B's B2 greedy sequence differed
from B1. That attempt is not a passing artifact. It motivated persistent
mismatch records and the final run's use of the same token-2 mechanics input as
S24. Neither run is a real-prompt quality certificate.

## What is implemented

- Persistent resident weights on each worker.
- Per-request sequence identity and KV state.
- Variable-row physical batching on phone HTP and desktop CUDA.
- Per-request retirement and sequence-slot reuse.
- Layer-boundary activation handoff through all three devices.
- Request-level continuous admission derived from arrival and output length.
- Fail-closed layer, lineage, status, finite-value, drain, placement, and token
  checks.

The handoff is an accelerator-stage exit, not semantic early termination. Every
request still executes all 48 model layers.

## Next gate

Replace the fixed two-slot lifecycle with the online SLO scheduler while keeping
the proven mechanics. The scheduler must choose a route and a device-specific
batch wait from offline profiles, prioritize urgent work before FIFO insertion,
and never wait past latest safe start. S24 CP5 remains the governing negative
result: the previous fixed policy harmed priority-0 latency and energy.

