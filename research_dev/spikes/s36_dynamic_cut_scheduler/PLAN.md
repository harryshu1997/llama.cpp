# S36: SLO-Aware Dynamic Cut Scheduler

## Question

Can one selected A6000 and two phone NPUs use the S35 dynamic-cut mechanism
under a dense arrival trace while preserving request SLOs and reducing selected
CUDA work?

S36 is a request-level proof scheduler. It does not change the llama.cpp graph,
KV cache, backend scheduler, or kernels.

## Frozen topology

All routes execute the complete 48-layer Gemma 4 12B F16 model. A cut is a
handoff boundary, not skipped computation.

- OP12: resident `[0,8)`, active cut 4 or 8.
- OP15: resident `[0,8)`, active cut 4 or 8.
- selected A6000 prefix: resident `[0,4)` for the all-CUDA route.
- selected A6000 tail: resident `[4,48)`, active start 4 or 8.
- the second A6000 is excluded.

The finite routes are:

| Route | Head | Tail |
| --- | --- | --- |
| `cuda-c4` | CUDA `[0,4)` | CUDA `[4,48)` |
| `op12-c4` | OP12 `[0,4)` | CUDA `[4,48)` |
| `op12-c8` | OP12 `[0,8)` | CUDA `[8,48)` |
| `op15-c4` | OP15 `[0,4)` | CUDA `[4,48)` |
| `op15-c8` | OP15 `[0,8)` | CUDA `[8,48)` |

## Workload

The request arrivals and request identities come from the frozen S23 densest
two-second BurstGPT window. Source priority and deadline are unavailable, so
the existing deterministic S23 priority/SLO sidecar remains synthetic.

The physical execution proxy is frozen separately:

- prompt: four token-id 2 rows;
- generated tokens: four terminal steps;
- arrivals, request IDs, observed token demand, priority, and SLO retain their
  S23 values;
- the adapter binds the source trace by SHA-256 and labels the payload as
  synthetic.

This proxy is intentionally small enough for a physical proof run. It is not a
token-throughput claim for the observed BurstGPT prompt lengths.

## Fast-loop policy

1. Priority 0 always selects `cuda-c4`.
2. A phone route is eligible only when its measured conservative prediction,
   queue delay, transfer/tail prediction, and safety margin fit the remaining
   SLO.
3. Priority 1 selects the lowest predicted feasible phone latency, then
   same-cut queued rows, normalized active load, the deeper cut, and route ID.
   Priority 2 selects the deepest feasible cut, then same-cut queued rows,
   normalized active load, predicted duration, and route ID. This prevents the
   shallow cut from becoming a dominated, unexercised configuration.
4. If no phone route is feasible, use `cuda-c4` immediately. The server never
   waits for phone readiness.
5. A live request retains one route and cut for prompt and decode.

## Continuous batching contract

- this campaign opts into llama.cpp's unified KV layout so B32 owns one
  bounded shared cache instead of 32 independently 256-padded caches;
- each physical worker has one bounded queue and one compute thread;
- pending rows are partitioned by active cut and priority band;
- one physical `range_batch` contains exactly one cut and one priority band;
- prefill and decode rows may share that batch;
- release occurs at the measured batch knee, at the earliest latest-safe-start,
  or during an explicit drain;
- a batch never exceeds the StageNet row capacity;
- worker failure is latched and propagated to every pending future.

For a four-token prompt, eight requests produce 32 physical rows. Therefore a
row knee of 32 does not imply 32 prompt requests.

## Checkpoints

- [x] CP0: deterministic trace adapter and finite route/profile contracts.
- [x] CP1: cut-aware batcher and SLO policy pass fail-closed unit tests.
- [x] CP2: measured cut profiles for both phones and CUDA stages.
- [x] CP3: equal-work all-CUDA control on the frozen trace.
- [x] CP4: treatment on OP12, OP15, and one A6000.
- [x] CP5: artifacts and result labels frozen.

## Physical gates

The treatment passes only if all conditions hold:

1. all 60 requests reach exactly one terminal outcome;
2. treatment tokens equal the all-CUDA control per request;
3. no additional priority-0 SLO miss and priority-0 p95 latency is no more than
   5% worse than control;
4. both phone devices and both cuts execute at least one request;
5. at least one phone batch physically contains both prefill and decode rows;
6. no physical batch mixes cuts or priority 0 with background work;
7. selected-CUDA measured compute time is lower than control;
8. every worker finishes with zero active sequences and no software lease.

If the dense trace does not naturally create a mixed-phase batch, CP4 fails.
The gather window or policy may be changed only in a new, explicitly versioned
run, not after inspecting a paid run.

## Claims boundary

A passing S36 run would establish dynamic-cut scheduling mechanics and measured
selected-CUDA compute relief under the execution proxy. Phone energy, USB/WiFi
energy, total-system energy, production `llama-server` integration, and benefit
at observed BurstGPT token lengths remain unclaimed.

The physical campaign is repeated three times without policy changes. Mechanics
are reported independently from selected-CUDA relief. Relief is reproducible
only if the median treatment value is lower than the median control value.
