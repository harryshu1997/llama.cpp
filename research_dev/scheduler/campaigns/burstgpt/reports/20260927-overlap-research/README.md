# Computation and transfer overlap: system audit and research proposals

2026-09-27. Scope: source inspection, read-only extraction from the latest
undisturbed tp2 trace, and primary-source literature review. No new hardware
experiment or runtime change. All proposed performance and novelty claims remain
unverified. This is a research shortlist, not a claim of priority over all work.

## What the system does now

The desktop performs attention and the remaining transformer work. For selected
Qwen FFNs, OP15 owns layers 0-17 and Pixel owns 18-23. These are successive layers,
not two simultaneous FFN column partitions for the same token. OP15 uses the
FunctionFS USB path; the integrated Pixel uses its packed CPU worker over ADB
TCP. Pixel CPU+GPU execution failed the earlier performance gate and remains off.

The client submits at `ffn_norm`, runs any local FFN prefix, and waits at
`ffn_phone_partial`. It permits one pending call per client. The backend eval
callback synchronizes at its graph boundaries. The full-width graph contains
no local FFN prefix, so there is almost nothing between submit and join.

Code anchors:

- `examples/layersplit/ffn-split-client.cpp:1550`: pending-call guard, submission,
  synchronous input extraction on TCP; `1695`: completion wait and result publication.
- `src/llama-graph.cpp:1669`: full-width phone path has no local FFN computation.
- `tools/server/server.cpp:982`: dispatch to the helper owning the layer.
- `ggml/src/ggml-backend.cpp:1809`: backend synchronization before eval callback.
- `tools/server/server-context.cpp:3934`: dormant release and restoration retain
  the all-processing-slots policy check.
- `src/llama-model.cpp:1860` and `src/llama-mmap.cpp:612`: restore the released
  host ranges; optional population is synchronous after advisory readahead.
- `_unified/helper_preparation.py` and its operations: existing admission,
  preparation, residency generations, reservations and attachment machinery.

All five native source files hashed by the collector match the deployed source.
This does not certify every build output or every repository file.

## Measured opportunity

Call-weighted means from tp2's saved server shutdown summaries:

| Path | Calls represented | RPC ms | Worker compute ms | Host FFN branch ms | Host join wait ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen / OP15 | 4,608 | 10.913 | 9.175 | 0.0118 | 10.948 |
| Qwen / Pixel | 1,344 | 13.488 | 8.464 | 0.0136 | 13.661 |
| Gemma / OP15 | 42,096 | 7.929 | 6.587 | 0.0092 | 7.947 |

These summaries are a subset of the request-level proof calls; they must not be
reported as whole-trace coverage. The host branch is the interval between phone
submission and its FFN join, not all server compute. RPC and join wait overlap
and must not be added. RPC minus worker compute includes software, copying and
transport overhead; it is not pure USB transmission time.

The useful local overlap reported by these summaries is about 0.1% of the Qwen
RPC interval. Attention still runs before/after these boundaries. Increasing the
USB queue depth alone does not create an independent input for the next layer.

Four logged host-share restorations total only 0.515575 s, with a maximum of
174.952 ms. Flags were `drop_cache=0,populate=1`. Their 30.576 GB of range mentions
are not measured physical disk traffic. Relative to the 1848.969 s tp2 trace,
eliminating these logged intervals alone could remove at most about 0.028% of
trace wall time, even before accounting for overlap. Other page faults, model
loads and preparation are separate costs. Restore-only prefetch is consequently
a low priority for this warm-cache trace.

Most weight traffic in a resident phone FFN is phone DRAM-to-processor traffic,
not weights repeatedly sent over USB. Three transfer classes need separate
timelines: activation RPC, local weight streaming, and model/shard preparation.

## Proposal 1: jointly schedule remote FFNs and pending weight movement

Treat each remote FFN interval as an opportunity to perform useful host memory
work while its ordinary local FFN weight reads are absent. Admit bounded chunks
of an already-arrived request's model load, GPU upload, shard staging or packing
when their predicted interference fits the current decode deadline. Choose the
phone fraction and transfer rate together. A fraction that is inferior for an
isolated token could be better for total completion energy if it permits a
pending model to become ready earlier.

The proposed contribution is a scheduler coupling three quantities currently
handled at different levels: remote execution time, transfer bandwidth under
contention, and the lifetime of reserved/resident memory. A transfer must reserve
its destination before starting; released RSS does not automatically mean free
physical DRAM because the page cache may retain the data. A helper result gets
priority over a bulk transfer at bounded chunk boundaries. The policy also needs
an alternate execution plan if its expected remote interval ends early or the
device loses qualification. Timing predictions are estimates, not hard guarantees.

This would extend the existing preparation/lease machinery rather than add a
second allocator. It must retain the dormant release rule. The old S14-CP3/CP4
plan in `research_dev/NEXT_PLAN.md` already proposes two residency generations
and preemptible bulk staging; neither feature alone is new even within this repo.

Closest work: [HeteGen](https://arxiv.org/html/2403.01164v1) already overlaps
heterogeneous compute and parameter movement; [ChunkFlow](https://arxiv.org/abs/2605.11335)
already schedules chunked prefetch against communication contention. The narrow
hypothesis worth testing is whether phone execution can be chosen to create
useful host bandwidth intervals while enforcing shared-model residency and
device-availability constraints, improving completed-work energy under SLOs.
That combination is a candidate contribution, not an established novelty claim.

Decisive test: compare current execution, ordinary asynchronous prefetch, and the
joint policy at the same memory limit and request arrivals. Sweep transfer rate
and offload fraction; measure exposed preparation delay, token p99, DRAM/PCIe/USB
traffic where counters are available, unused prefetch bytes, and joules for the
whole completed trace. If ordinary prefetch achieves the same outcome, the extra
policy has no demonstrated value. Do not target the 0.516 s warm restore cost as
the source of a large gain.

## Proposal 2: stagger independent cohorts, then re-batch at phone boundaries

While request group A waits for its phone FFN, execute ready host work for group
B. When A reaches the Pixel-owned layers, OP15 can work on a different group's
earlier layer, with the host supplying each group's intervening attention.
Transfer B's activation while A computes where the device/link supports it.

The implementation needs suspended graph continuations, independent activation
buffers, sequence/KV ownership, tagged completion events, and a queue of pending
calls rather than one mutable pending slot. It cannot simply remove a wait.
Layer dependencies and autoregressive token dependencies remain intact.

The research problem is choosing when to keep requests together for weight reuse
and when to stagger them for overlap. In particular, splitting a good B2 call
into two B1 calls may double weight reads and lose energy. Re-batch compatible
rows at each phone boundary under a shared artifact, layout generation and
policy; use actual batch composition in the learning evidence. Do not invent
concurrency or future arrivals for the headline trace.

Closest work: [NEO](https://arxiv.org/html/2411.01142v1) uses asymmetric sub-batch
pipelining, and [FastPP](https://www.usenix.org/conference/osdi26/presentation/hwang)
uses online pipeline scheduling. The repo also contains older S22/S24 pipeline
experiments and a macro-tail fence facility; tp2 reports zero tail-fence calls.
Basic pipelining is therefore an engineering baseline. A candidate contribution
would be the policy-constrained choice between re-batching and staggering on
heterogeneous phone FFNs with different memory traffic and thermal availability.

Decisive test: first replay the real arrival trace offline with measured B1/B2/B4
costs, preserving batching efficiency and buffer capacity. Then compare current
coalesced batching, fixed staggering and adaptive re-batching with identical
arrivals, weights and resource limits. Record exposed wait, actual overlap,
weight bytes per completed token, token p99 and full-run energy. Low concurrency
can make this idea lose; a saturated synthetic test alone cannot establish the
campaign benefit.

## Proposal 3: stream FFN output into the next layer's projection

A more ambitious kernel/runtime experiment is to publish completed output-row
tiles from the phone down projection and start the next host projection before
the complete output arrives. Qwen's next input RMSNorm is a barrier, but its
scalar normalization can be deferred past a bias-free linear projection:

```text
z = residual + FFN(x)
s = 1 / sqrt(mean(z*z) + epsilon)
Q = s * sum_over_tiles(Wq[:, tile] * (gamma[tile] * z[tile]))
```

For each completed z tile, accumulate its squared norm and its Q/K/V projection
contribution. Apply the final scalar when all tiles are complete, then perform
Q/K normalization, RoPE and attention in the required order. This overlaps later
phone output production/transfer with host projection work without predicting a
future activation. Start with Qwen; Gemma has additional normalization sites.

The identity above is in real arithmetic. Moving scaling and changing reduction
order can change floating-point results; current token mismatches cannot serve
as a waiver. Partial FFN-column results are not complete z tiles: sum all owners
and the residual before updating the norm statistic, or include the required
cross terms. No downstream token may be committed from incomplete tiles.

[FlashNorm](https://arxiv.org/html/2407.09577v1) already establishes deferred
normalization. [Galaxy](https://arxiv.org/abs/2405.17245) overlaps edge communication
and tile computation, and [Syncopate](https://www.usenix.org/conference/osdi26/presentation/qiang)
automates fine-grained compute/communication overlap. Thus neither algebra nor
chunking is a new claim. The candidate extension is a schedule spanning phone
down-projection output, USB completion and the next host projection, with an
adaptive tile size and correct ownership/reduction semantics at small decode
batches. Its novelty confidence and performance confidence are lower than its
technical interest.

Decisive test: compare whole-vector return, tiled return only, and tiled return
plus deferred projection. Use one persistent transport exchange rather than a
new high-overhead RPC per tile. Sweep tile size and rows 1/2/4, measure complete
two-layer latency and energy, and check both FFN values and full generated
tokens. Small activation messages and extra GEMV launches may erase the benefit.

## Proposal 4: overlap preparation with service before a thermal handoff

While a phone is still qualified, prepare a bounded successor placement for
work likely to outlast its thermal headroom. Stage and qualify the necessary
weights while the old placement serves; transfer ownership for all compatible
decode rows at a common safe boundary. Reserve both generations during the
handoff, and keep the existing thermal exclusion threshold.

The first fallback can be the desktop. Giving Pixel any of OP15's layers requires
new shards, numerical qualification and capacity evidence: its present qualified
layers 18-23 are not a general substitute for OP15 layers 0-17. Do not assume
that host fallback recovers OP15's energy benefit. The initial hypothesis is
less exposed transition delay and fewer discarded tokens; sustained energy
improvement additionally requires a useful qualified alternate phone placement.

This connects tp2's long thermal exclusion and g11's loss/rejoin mechanism to the
overlap question. The proposed policy jointly chooses preparation lead time,
resident bytes and the cohort handoff boundary. Generic pre-copy migration,
thermal-aware placement and two generations are existing techniques. In
particular, [Sereno](https://www.usenix.org/conference/osdi26/presentation/xin)
already addresses interference in mobile LLM execution, although its mechanism
targets foreground QoS rather than this phone-to-host ownership transition.
The precise combination needs a wider migration/thermal literature check before
being used as a novelty claim.

Decisive test: compare reactive exclusion, fixed-lead preparation, and a thermal
forecast policy, all with the same temperature/status limits. Charge unused
preparation bytes and energy, transient memory, and mistaken forecasts. Measure
coverage, thermal-state duration, re-executed tokens, SLOs and total energy over
long repeats. A change in thermal thresholds is a separate intervention.

## Research priority

For practical speed, first evaluate proposal 2's concurrency-versus-batching
tradeoff using saved traces. For a system paper, proposal 1 provides the strongest
organizing question, with proposal 4 as an availability constraint. Proposal 3
is a bounded kernel experiment rather than the foundation of the energy claim.

Useful engineering baselines are asynchronous copies, double buffering,
event-based completion, packed-weight prefetch and matmul/activation fusion.
Those alone are not a defensible novelty claim. [PowerInfer-2](https://arxiv.org/html/2406.06282v2)
already pipelines neuron-cluster I/O and computation on phones;
[Kairox](https://www.usenix.org/conference/osdi26/presentation/jiang-yapeng)
already adapts CPU/GPU neuron balancing and prefetch to runtime conditions.

Every proposed gain must be compared with the current assisted system, not only
the older 228.535 kJ desktop baseline. Separate kernel/operator numerical checks
from strict full-output identity, and measured host energy from modeled phone
energy. No speedup, extra energy saving or novel contribution is established by
this inspection alone.

## Artifacts

[EVIDENCE.json](EVIDENCE.json) contains source hashes, saved summary rows and
restoration events. [collect_evidence.py](collect_evidence.py) reproduces the
read-only collection. Collector pyflakes and embedded-reader checks pass.
Existing baseline/tp2/g11 energy and correctness results are in the
[results audit](../20260927-results-audit/README.md).
