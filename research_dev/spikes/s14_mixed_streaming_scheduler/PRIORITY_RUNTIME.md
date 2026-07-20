# S14 priority batch runtime substrate

Status: host mechanics only. No device, latency, relief, or energy result.

`priority_batch_runtime.py` connects the measured fast policy to an executor
without accepting caller-selected batches or caller-supplied completion labels.

## Workflow

1. Register a `RouteConfig` with a measured batch profile, roofline class,
   profile identity, and route epoch.
2. Enqueue a `WorkItem`. Its service, model, and island must match the route.
3. Call `decide()`. `WAIT` exposes the bounded wake time; `NO_FEASIBLE` leaves
   the request for server fallback; only `LAUNCH` grants device ownership.
4. Execute the exact `Launch.request_ids` as one native batch.
5. Call `complete()` with the same launch id and epoch plus exactly one
   `BoundaryCertificate` per request.
6. Admit only finite-time, in-deadline results whose identity, epoch,
   correctness, and D2H gates all pass. Failed gates require server fallback;
   late valid results are recorded as tardy and not admitted.

Independent routes have independent bounded lanes. This permits a server BGE
batch and a phone Gemma batch to run concurrently. A route cannot accept a
second launch until the first completes or fails.

## Fail-closed properties

- strict priority and compatibility isolation are inherited from
  `power_frontier_policy.choose_batch()`;
- duplicate, future, wrong-route, and already-in-flight requests are rejected;
- batch size and membership come only from the policy decision;
- stale launch ids and route epochs are rejected;
- completion certificate membership must exactly equal launch membership;
- certificate validation is atomic across a batch;
- failed launches release the lane and preserve every request for fallback.

## Verification

```sh
PYTHONDONTWRITEBYTECODE=1 python3 \
  research_dev/spikes/s14_mixed_streaming_scheduler/tests/test_priority_batch_runtime.py
```

The 15 tests cover SLO-bounded memory batching, measured-knee compute batching,
independent server/phone concurrency, priority blocking, route signatures,
ownership, stale epochs, exact and malformed certificate sets, late results,
and launch failure.

## CP-D integration boundary

The CP-D harness must provide two executors:

- selected-A6000 BGE batches, using the sequence-length-specific measured knee;
- OP15 -> OP12 -> selected-A6000 Gemma batches, using the certified batch
  profile and validated stage activations.

The harness must not infer completed work from process exit alone. It must
translate persisted placement, topology, correctness, transfer, and epoch
evidence into each `BoundaryCertificate`. The CP1 checklist remains open until
this physical binding is exercised on the frozen P0/P2 cohort.
