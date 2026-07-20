# S17 hierarchical activation coalescing

Status: CP0 complete; CP1 coalescing mechanics pass but the numerical gate
fails. Stop before CP2.

## 1. Goal

Turn the two independent phone lanes into a heterogeneous fan-in hierarchy.
Low-capacity devices compute small resident prefix islands while a faster phone
coalesces compatible boundary activations into a larger batch for a resident
middle island. The selected A6000 consumes the resulting larger tail batch.

This is not a fixed OP15 -> OP12 -> A6000 pipeline. Producers are independent,
and no producer is allowed to block a ready cohort past its latest start time.

~~~text
OP12 prefix [0,6) B32 ---------+
                                +-> OP15 middle [6,12) B32..B64
OP15 prefix [0,6) B32 ---------+               |
                                                +-> A6000 tail [12,48)
~~~

The OP15 prefix and middle are separate prepared images and contexts. A single
OP15 `[0,12)` HTP allocation is forbidden: the measured HTP single-buffer limit
already rejects that geometry.

## 2. Route set

The scheduler keeps all three routes. The funnel is an option, not a mandatory
critical path.

| Route | Topology | Use |
|---|---|---|
| R0 | A6000 `[0,48)` | immediate fallback and tight SLO |
| R1 | one phone prefix -> A6000 tail | sparse arrivals or a late producer |
| R2 | producer prefixes -> OP15 middle -> A6000 tail | compatible backlog with sufficient slack |

R2 may use one producer cohort when its deadline requires release. It must not
wait for a nominal B64 target after the earliest admitted latest-start time.

## 3. Compatibility and ownership

Only activations with the same compatibility key may coalesce:

~~~text
model_digest
input_cut and output_cut
decode_step and context class
tensor dtype, shape, and layout
attention and kernel route
sampling policy
correctness-contract version
~~~

Each request keeps the same layer and KV owner for its lifetime. OP12 owns KV
for `[0,6)`, OP15 owns KV for `[6,12)`, and the A6000 owns KV for `[12,48)`.
Moving a live request between routes requires an explicit, validated KV
transfer; S17 does not implement or assume one.

The release rule is:

~~~text
release when compatible_count >= measured_batch_knee
        or now >= min(latest_start_us of admitted requests)
~~~

The target is a measured knee, not an unconditional batch of 32 or 64. Credits
bound every producer queue, the OP15 middle queue, and the A6000 tail queue.

## 4. Correctness oracle

Batch-size-dependent floating-point tiling is already observed on the full
A6000 model. Therefore bit identity to a different batch geometry is not a
valid oracle by itself. S17 uses two controls:

1. Same-shape full-model control at the final batch size for token and logit
   agreement.
2. Route-matched segmented CUDA control with the same producer and fan-in batch
   geometry to isolate backend and transport error from batching noise.

An R2 row is eligible only when all requests complete, every boundary is finite,
the route-matched residual relative L2 is at most 5e-3, placement is certified,
and token/top-k agreement meets the predeclared gate. A mismatch is reported;
the gate is not relaxed after acquisition.

## 5. Checkpoints

### CP0 - freeze the route and controls

- [x] Define R0/R1/R2 and the compatibility key.
- [x] Define KV ownership, bounded credits, and deadline release.
- [x] Record the separate-image requirement on OP15.
- [x] Freeze the same-shape and route-matched correctness controls.

### CP1 - OP15 middle-island screen

- [x] Materialize and hash Gemma `[6,12)` without renumbering blocks.
- [x] Run HTP0 B={32,64} empty-context support and placement checks on OP15.
- [x] Compare B64 against 2xB32 on HTP0 and a matched CUDA middle island.
- [x] Repeat with real OP12/OP15 `[0,6)` boundary activations.
- [x] Screen AUTO and explicit-attention routes without changing the gate.
- [ ] Run the same-shape end-to-end token/logit oracle and realistic C>=512
      decode context. This remains blocked by the failed middle residual gate.
- [x] Stop because the route-matched residual threshold fails even though B64
      is supported and faster per request than two B32 middle executions.

The CP1 result is
`COALESCING_KERNEL_MECHANICS_PASS_NUMERICAL_ELIGIBILITY_FAIL`. It does not
authorize CP2 or a hierarchical runtime.

### CP2 - OP15 dual residency

- [ ] Hold `[0,6)` and `[6,12)` as separate resident images without reload.
- [ ] Verify memory accounting, preparation identity, and generation-qualified
      leases for both images.
- [ ] Measure prefix/middle interference on HTP0 and decide whether the OP15
      prefix lane remains enabled while the middle lane is active.

### CP3 - physical three-device funnel

- [ ] Run OP12 `[0,6)` B32 concurrently with an OP15 producer cohort.
- [ ] Coalesce compatible boundaries and execute OP15 `[6,12)` at the selected
      B32/B64 release size.
- [ ] Execute one persistent A6000 `[12,48)` tail with no duplicate tail weights.
- [ ] Prove exact request ownership, token/logit correctness, placement, reset,
      and persistent process identity for at least seven sessions.

### CP4 - scheduler integration

- [ ] Add the R2 compound route to the READY route registry.
- [ ] Implement deadline-aware coalescing, partial release, credits, epochs,
      cancellation, and fail-closed fallback.
- [ ] Keep priority isolation: high-priority BGE remains ahead of low-priority
      Gemma work on the selected A6000.

### CP5 - real mixed-trace comparison

- [ ] Replay the frozen mixed trace on one A6000 plus both phones.
- [ ] Compare R0, R1, and dynamic R0/R1/R2 with identical admitted work.
- [ ] Report p50/p95/p99, SLO attainment, achieved batch distribution, useful
      phone work, discarded late work, throughput, and A6000 utilization.

### CP6 - matched selected-GPU energy

- [ ] Run rotated matched control/treatment pairs only after CP5 passes.
- [ ] Report uncertainty-adjusted selected-A6000 board energy and iso-power
      SLO-valid work.
- [ ] Keep phone, USB, host-wall, and total-system energy UNKNOWN.
- [ ] Stop the energy claim on this hardware if neither metric improves by the
      frozen 10 percent fleet gate.

## 6. Build distance

There are seven checkpoints including CP0. A minimal physical funnel exists at
the end of CP3. The mixed-workload system exists at the end of CP5. CP6 is the
claim-bearing energy evaluation, not part of the runtime implementation.

## 7. Current evidence

- OP15 persistent `[0,8)` B32 plus A6000 tail is token-correct and SLO-valid.
- OP12 persistent `[0,6)` B32 plus A6000 tail is token-correct under a 12 s SLO.
- The old serial OP15 `[0,8)` -> OP12 `[8,12)` -> A6000 route is diagnostic only:
  B32 p50 was 8.42 s and a repeated mixed run failed.
- Two independent B1 phone heads feeding one tail context are exact when the
  tail executes B1 serially. The B2 tail differs from the full B2 oracle and is
  a frozen negative until the route-matched oracle explains the delta.
- S16 preserved the high-priority BGE p95 but obtained only 0.75 percent median
  raw selected-GPU saving from one shallow phone island. R2 is intended to
  increase useful offloaded depth and tail batch density; no saving is assumed.
