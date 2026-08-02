# CPU plus OP15 operator offload V1

Date: 2026-07-31 EDT

Verdict:
`REAL_GREEDY_HEAD_PASS; DYNAMIC_TOP8_FAIL; COMPLETE_CPU_LAYER_RELATIVE_PASS_ARGMAX_FAIL; COMPLETE_CPU_LAYER_LATENCY_FAIL_ON_5995WX; I9_ONE_PHONE_COMPOSED_PROMISING; HTP_GRAPH_CACHE_WORKAROUND_ONLY; FLEET_PROJECTED_ONLY; ENERGY_NOT_MEASURED`.

## Scope and topology

The live concurrent tests use one decode item on:

- host: AMD Threadripper PRO 5995WX with ggml Q8_0 CPU kernels;
- phone: rooted OP15 with Hexagon HTP v81 or Adreno 840 OpenCL;
- transport: direct AOA USB;
- residency: operator weights remain resident on the phone;
- boundary: one K-wide activation is sent and a compact result is returned.

The target desktop is a physical i9-12900K with an RTX 4060 Ti at
`172.20.74.85`. The phone is cabled to the Threadripper host, so i9 plus phone
rows combine real component measurements but are not a physical co-run.

This is not a full-model, BurstGPT, multi-phone, capacity, or energy result.

## HTP changing-input defect

The original HTP worker could replay the first graph result after the external
input changed. A complete-layer input A passed, but a following standalone
input B returned A's phone contribution: component relative L2 was 2.12666,
the component argmax was 82 versus 2867, and final relative L2 was 0.7577.
Restarting the worker made the first request correct and the next distinct
request fail again.

The following experiments did not fix the defect:

- DMA_BUF synchronization around the host write;
- uncached rpcmem;
- disabling HOSTBUF, NHMX, or operation fusion;
- f32 instead of f16 activation and result transport.

For this research harness, setting the graph UID to zero disables the HTP
graph cache. With that bypass, A, B, and A are independently recomputed and
match their path-specific oracles. The bypass is in the two phone workers and
is enabled with `S41_DISABLE_GRAPH_CACHE=1`. It is not a production Hexagon
backend fix. Older constant-input HTP timings remain performance diagnostics,
but their changing-input correctness claims are superseded unless they used
this bypass or another independent dynamic-input check.

## Real one-phone vocabulary head

The phone owns 46,080 of 262,144 Gemma vocabulary rows. It receives one f16
K=3840 activation, executes the resident Q8_0 matmul on HTP, reduces its logits
to eight candidates, and returns 64 bytes. The CPU concurrently computes the
remaining 216,064 rows and merges the candidate lists.

Three alternating 96-thread control/treatment pairs used a cache-disabled HTP
worker and 100 paid iterations per process:

| Pair | CPU-only median | CPU plus OP15 median | CPU-only p90 | CPU plus OP15 p90 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 20.621 ms | 17.094 ms | 20.707 ms | 17.168 ms |
| 2 | 20.671 ms | 17.158 ms | 20.793 ms | 17.274 ms |
| 3 | 20.652 ms | 17.113 ms | 20.734 ms | 17.198 ms |
| median of pairs | 20.652 ms | 17.113 ms | 20.734 ms | 17.198 ms |

The median reduction is 17.14%, p90 falls 17.05%, and all 300 phone legs fit
under the CPU shard. Median phone time is 4.576 ms, median HTP matmul time is
2.672 ms, ARM top-8 reduction is about 0.285 ms, and the merge is below 0.001
ms. The matmul is about 0.354 GFLOP, or approximately 132 GFLOP/s.

A separate 30-input dynamic test gives:

| Gate | Result |
| --- | ---: |
| distinct host inputs | 30/30 |
| phone-local top-1 | 30/30 |
| merged global top-1 | 30/30 |
| exact phone-local top-8 list | 19/30 |
| matching phone-local top-8 positions | 217/240 |
| exact merged global top-8 list | 27/30 |
| maximum phone top-1 score relative error | 0.002150 |

Therefore the route passes for greedy decoding, not for exact top-k sampling.
A sampling-capable successor should return a wider approximate candidate set
and recompute those candidate scores with the CPU oracle before publication.
This head runs once per generated token, not once per transformer layer.

## Real one-phone complete layer

`causal_layer_host.cpp` executes a Gemma-like Q8_0 decode layer at KV=136:
RMS normalization, Q/K/V projection, GQA attention, output projection,
attention residual, FFN normalization, and a complete FFN residual. OP15 owns
a 1,792-column suffix of K=3840, NFF=15360 and performs gate, up, SiLU,
multiply, and down before returning one K-wide f16 residual.

The cache-disabled worker consumed 30 distinct full activation vectors. The
maximum final relative L2 was 0.001812 with no non-finite values, but one
hidden-state argmax differed. f32 transport reproduced the same mismatch at
the same input with relative L2 0.001733, so wire precision is not the cause.
The frozen exact-argmax gate is 29/30 and fails. The relative-error gate passes.

Three globally tuned process-level pairs on the 5995WX give:

| Metric | tuned CPU-only | CPU plus OP15 | Change |
| --- | ---: | ---: | ---: |
| median | 1.149 ms | 2.372 ms | +106.44% |
| p90 | 1.218 ms | 2.652 ms | +117.76% |

The isolated cache-disabled HTP leg is 0.743 ms median and 0.836 ms p90. Its
native graph compute is 0.456 ms for about 0.0413 GFLOP, or approximately
90.6 GFLOP/s. During the complete CPU co-run, memory and scheduling contention
raises the observed phone leg to about 1.812 ms. The tuned Threadripper CPU
finishes the complete control before the split boundary can pay for itself.

The built-in same-thread comparison can show a win at a slower CPU thread
count. That is diagnostic only and is not used for the verdict.

## Phone NPU versus GPU

The same 1,792-column resident complete-FFN graph was run on both phone
accelerators:

| Phone backend | Median | p90 | Correctness | Decision |
| --- | ---: | ---: | --- | --- |
| HTP v81, cache bypass | 0.743 ms | 0.836 ms | final rel L2 0.004992, fixed-input argmax exact | selected |
| Adreno 840 OpenCL | 2.810 ms | 3.481 ms | final rel L2 0.003483, fixed-input argmax exact | valid but 3.78x slower |

The OpenCL backend is named `GPUOpenCL`, not `OpenCL0`; the latter selector
correctly failed and printed the enumerated devices. The selected OpenCL build
uses the Adreno-optimized quantized kernels. HTP remains the fastest valid
backend for this shape.

Other observed backend limits are unchanged:

- a full-width HTP FFN can report `VTCM-TOO-SMALL` and return zeros even when
  graph execution reports success;
- the current arbitrary-offset HTP FFN is 4-5x slower than the suffix path and
  has about 1.0-1.2% component error;
- the faster Adreno xmem long-context path misses its numerical and p90 gates;
- OP12 is not rooted, so HTP and `/dev/usb_accessory` cannot be used there;
- OP15 USB adb is unauthorized, while wireless adb plus root and AOA data are
  working.

## Physical i9 controls and one-phone composition

The complete Gemma-like layer was measured directly on the i9 with a 128 MiB
cache eviction. A thread sweep selected eight threads:

| CPU threads | Complete-layer median |
| ---: | ---: |
| 4 | 7.667 ms |
| 8 | 7.506 ms |
| 12 | 8.113 ms |
| 16 | 8.210 ms |
| 20 | 8.166 ms |
| 24 | 8.189 ms |

A longer eight-thread run measured 7.519 ms for the monolithic graph and
7.590 ms for the explicit two-phase graph. With one 1,792-column suffix, the
physical i9 FFN remainder is 5.492 ms. Combining it with the conservative
1.812 ms real phone leg and 0.032 ms merge predicts 6.866 ms for the complete
layer, 8.68% below the 7.519 ms control.

This N=1 result is promising but composed. It does not include the target
desktop USB controller, simultaneous target-host execution, or target-host
power. Moving the phone cable is required before calling it a pass.

## Fleet latency ceilings

For N phones, the projection uses:

```
T = non_FFN_phase + max(i9_FFN_remainder, slowest_phone) + merge
```

Each phone is assumed to process one 1,792-column slice in 1.812 ms on an
independent link. The i9 remainder is physical; N greater than 1 is not.

### Complete-layer FFN slicing

| Phones | Columns offloaded | i9 FFN remainder | Projected layer | Versus 7.519 ms | Server compute-path reduction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,792 | 5.492 ms | 6.866 ms | -8.68% | 9.96% |
| 2 | 3,584 | 4.840 ms | 6.177 ms | -17.86% | 19.05% |
| 4 | 7,168 | 3.365 ms | 4.764 ms | -36.64% | 37.66% |
| 8 | 14,336 | 0.446 ms | 3.258 ms | -56.67% | 75.50% |

Only N=1 has a fast suffix kernel. N greater than 1 requires fast arbitrary
offsets, aggregate correctness, concurrent USB buses, and real devices. The
table is an optimization ceiling, not current system behavior.

### Vocabulary-head sharding

The i9 head control is 35.031 ms. Each projected phone owns 46,080 disjoint
rows and uses the real 4.576 ms phone leg.

| Phones | Physical i9 remainder | Projected head | Versus control | Server compute-path reduction |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 28.994 ms | 28.994 ms | -17.23% | 17.23% |
| 2 | 22.797 ms | 22.797 ms | -34.92% | 34.92% |
| 4 | 10.312 ms | 10.312 ms | -70.56% | 70.56% |
| 5 | 4.053 ms | 4.576 ms | -86.94% | 88.43% |

N greater than 1 still needs per-offset dynamic correctness and simultaneous
AOA. Exact top-k sampling is not eligible even at N=1.

## Energy bounds, not energy results

The i9 RAPL counters are root-readable, so synchronized package energy was not
acquired. At a conservative 4.5 W active power per phone, the analytical CPU
active-power break-even points are about 11.5-11.9 W for the complete-layer
FFN projections and 3.3-3.4 W for the head projections. These values assume
the CPU drops power after its shard finishes and phones consume active power
only for their measured leg.

Fixed system power, CPU DVFS, phone idle power, USB-controller power, link
contention, and merge scaling are omitted. No server-joule, fleet-joule, or
joules-per-token claim follows from this bound.

## Operator decisions

| Route | Current decision | Why |
| --- | --- | --- |
| sharded vocabulary head | first integration target for greedy decode | real 17.14% median and 17.05% p90 win, 64-byte result |
| complete dense FFN residual | target-desktop physical test next | real relative-error pass, local global latency fail, composed i9 win |
| complete MoE expert | promising but untested | ideal small activation/residual boundary, but neither tested model is MoE |
| GQA group or context shard | long-context diagnostic only | gains require resident KV and long context; CPU route is not yet competitive |
| RMSNorm and elementwise ops | reject | too little arithmetic to amortize dispatch and AOA |
| embedding lookup | reject | host lookup is cheaper than the boundary |
| model-weight paging | reject for latency | moving weights destroys the small-message invariant |

For MoE, route complete experts to separate phones after the router decision;
do not split one expert across a phone chain. Each selected expert receives the
same token activation and returns one weighted K-wide residual, so phones and
CPU can run concurrently without phone-to-phone communication. A real MoE
shape and routing distribution are required before making a claim.

For long-context attention, each phone must keep a disjoint KV segment
resident. The host sends projected queries and merges partial online-softmax
state. Initial KV placement is amortized over many decode steps; copying KV on
the token path is ineligible. Existing calibrated GPU plus OP15 evidence shows
only a 2.24% continuous-schedule median win for an 8K suffix at 262K context,
while gapped p90 is worse. It is not a CPU-host pass.

## Implementation and verification

Research harness changes are limited to the S41 spike:

- CPU backend selection, explicit thread counts, cache eviction, and CPU/phone
  overlap in the FFN, head, and complete-layer hosts;
- cache-disabled dynamic-input modes in the activation and head workers;
- a 30-input head oracle and a 30-input complete-layer oracle;
- f16/f32 phone I/O diagnosis for the complete-layer host;
- explicit backend-device listing on selector failure.

Source SHA256 values:

```
causal_activation_host.cpp   f498c724f86170e3f8ec718605dbd445747de7afaeb646dd4f6e2720ed93eb8a
causal_activation_worker.cpp 177b2035ae6867f502bc1ed4990805058e23c28527722bfe63ed2784ebe29555
causal_head_host.cpp         6eba73c57ba61f665a09a99b18e4d403e6e1b23a9ba744323b309043aba28734
causal_head_worker.cpp       ddbd39cf03791261bf3d11eae66644721fd34ae4a591c00653e6196af96b9479
causal_layer_host.cpp        98a9c9a91640d068b4b89b02b86def50744d373428156dca01633e785b0ee1a6
```

The Hexagon core was not changed as part of the cache workaround. The existing
unrelated worktree changes are preserved. No commit or push was performed.

## Next bounded experiment

1. Move the OP15 cable to the i9 desktop and acquire three alternating
   complete-layer and greedy-head control/treatment pairs.
2. If N=1 passes, repair arbitrary-offset HTP execution and validate two real
   disjoint slices before using any N greater than 1 projection.
3. For sampling, return a wider phone candidate set and perform exact CPU
   rescoring before publishing top-k.
4. Acquire synchronized i9 RAPL and phone power only after latency and dynamic
   correctness pass on the physical topology.
5. Test one actual MoE expert only after these bounded routes are stable.

Do not start a full-model or BurstGPT campaign from the fleet table alone.
