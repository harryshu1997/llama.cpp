# CPU plus phone operator offload V1

Date: 2026-07-31 EDT

## Objective

Test whether one real phone can reduce decode latency or CPU energy for one
bounded model operator while the host CPU computes the complementary shard.
Keep model weights resident on both sides and transfer only the activation and
a compact partial result.

This is an operator experiment. It does not authorize a full-model, BurstGPT,
multi-phone, or energy claim.

## Selection rule

A phone route is eligible only when all of the following hold:

1. The phone owns persistent weights or KV state. Weight transfer is outside
   the token critical path.
2. Host and phone shards are independent after a small activation is sent.
3. The result is a small residual, state, or top-k candidate list.
4. The phone p90 plus merge fits below the host-shard p90.
5. The globally tuned hybrid median and p90 beat the globally tuned CPU-only
   control. A same-thread comparison is diagnostic only.
6. The local partition, phone component, and merged output pass their explicit
   numerical gates. Backend success without valid output is a failure.
7. Energy is claimed only after synchronized CPU-package and phone-power
   measurements show lower joules per token.
8. At least 30 distinct input hashes must produce 30 independently checked
   outputs. A repeated constant input cannot establish accelerator coherence.

For a fleet of N independent phones, the latency ceiling is:

```
T_split(N) = max(T_host_remaining(N), max_i(T_phone_i)) + T_merge(N)
```

The projection is not an acquisition. It assumes independent links, concurrent
launch, all-offset correctness, and enough phone memory.

## Candidate order

| Priority | Route | Boundary | Reason |
| ---: | --- | --- | --- |
| 1 | sharded vocabulary head | one K activation, eight candidates back | the response is 64 bytes and work scales with resident vocabulary rows |
| 2 | complete FFN residual slice | one K activation, one K residual back | applies to every dense layer and is also an expert-shaped proxy |
| 3 | one GQA group or resident-KV context shard | query/state in, one K residual or normalized state back | only plausible at long context with KV already resident |
| 4 | complete MoE expert | routed activation in, one K residual back | promising topology, but no actual MoE model or valid full-expert HTP kernel is in this scope |

RMSNorm, elementwise activation, embedding lookup, and weight paging are not
tested. Their arithmetic is too small or their returned data is too large for
the measured AOA boundary.

## One-phone acquisition

- Model geometry: Gemma-like Q8_0, K=3840, NFF=15360, vocabulary=262144.
- Phone: OP15 HTP v81 through direct AOA USB.
- Host controls: ggml CPU backend with a thread-count sweep.
- FFN cache condition: explicit host cache eviction before each timed graph.
- Head cache condition: the approximately 1 GB vocabulary matrix naturally
  exceeds host last-level cache.
- Timing: three alternating control/treatment repetitions after warmup.
- FFN correctness: path-matched local partition, phone residual, merged
  residual, finite values, and row argmax.
- Head correctness: report greedy top-1 and global top-8 independently. A
  greedy pass does not authorize sampling.

Use the fastest correctness-valid backend. HTP is preferred when it passes.
OpenCL is the fallback. A faster path that misses its numerical gate remains a
latency-only diagnostic.

## Fleet projection

Project 1, 2, 4, and 8 phones for FFN columns and 1, 2, 4, and 5 phones for
vocabulary rows. Use physically measured i9-12900K remaining-shard times and
the conservative real contended OP15 path. Do not claim multi-phone
correctness or link scaling until every offset executes on real phones.

## Stop conditions

- Stop a route if the globally tuned one-phone treatment is slower.
- Stop HTP if the DSP reports an error, output is all zero, or the numerical
  gate fails, even when graph execution returns success.
- Do not build full-model scheduling from fleet projections.
- Do not report joules until readable RAPL or another synchronized CPU power
  source is available.

## Current gate state

- Greedy vocabulary head: real one-phone median and p90 pass; 30/30 changing
  global top-1 decisions pass.
- Sampling vocabulary head: fail; changing global top-8 is 27/30.
- Complete Gemma-like CPU layer on the 5995WX: relative-error gate passes,
  exact hidden-state argmax is 29/30, and globally tuned latency fails.
- Physical i9 complete-layer control: pass. The i9 plus OP15 result is composed
  until the phone cable is moved to that host.
- Multi-phone: projection only. Arbitrary-offset correctness and performance
  are not eligible.
- Energy: not acquired.

The HTP cached graph can replay the first input. V1 uses a graph-UID-zero
worker bypass for measurement. This is a temporary research mechanism, not a
Hexagon core fix, and it must remain enabled for all changing-input runs.

## Ordered next work

1. Move OP15 to the i9 host and repeat the one-phone complete-layer and greedy
   head acquisition with three alternating pairs.
2. Repair the Hexagon cached-input lifecycle in the backend, then reproduce
   the dynamic gates without the UID-zero bypass.
3. Implement a fast arbitrary-offset suffix and validate two disjoint slices
   before acquiring any fleet result.
4. Add wider candidate return plus CPU rescoring if sampling is required.
5. Only then acquire synchronized energy or try one actual MoE expert.
