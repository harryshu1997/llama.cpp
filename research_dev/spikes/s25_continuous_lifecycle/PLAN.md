# S25 Continuous Request Lifecycle

Status: `THREE_DEVICE_CONTINUOUS_LIFECYCLE_PASS` on 2026-07-21.

## Question

Can the existing StageNet V3 implementation sustain a changing set of decode
requests across the complete OP12 `[0,8)` -> OP15 `[8,16)` -> CUDA `[16,48)`
route?

This checkpoint tests runtime mechanics only. It does not claim an energy or
throughput benefit.

## Required proof

1. [x] Start two requests in one physical batch.
2. [x] Retire the shorter request while the longer request remains active.
3. [x] Admit a new request into the released sequence slot without resetting the
   surviving request.
4. [x] Repeat the slot-reuse transition once more.
5. [x] Observe the same changing batch membership on OP12, OP15, and the CUDA tail.
6. [x] Remove every sequence and drain all three workers with zero active state.
7. [x] Match every greedy output token against B1 execution on the same physical
   route and the same worker session.

The fixed request set is:

| Request | Arrival step | Output steps | Sequence slot |
| --- | ---: | ---: | ---: |
| A | 0 | 2 | 0 |
| B | 0 | 4 | 1 |
| C | 2 | 3 | 0 after A retires |
| D | 4 | 2 | 1 after B retires |

Expected dynamic membership is `AB, AB, CB, CB, CD, D`. This is request-level
continuous batching: membership changes without unloading weights, restarting a
worker, or clearing another request's KV state.

## Scope

- The term "early exit" means exiting an accelerator at a declared layer
  boundary and handing the activation to the next stage. It does not skip model
  layers or terminate inference early.
- Routes remain fixed for a request because its KV state is owned by those
  resident stages.
- The proof uses a capacity of two even when a worker exposes more slots. A
  later scheduling checkpoint may select a larger device-specific batch knee.
- Token equality is a narrow same-route batching check, not a general numerical
  correctness certificate.

## Stop conditions

Stop and report failure on any protocol error, layer-range mismatch, non-finite
activation, lineage mismatch, active-count mismatch, premature slot reuse,
token mismatch, or nonzero final active count.
