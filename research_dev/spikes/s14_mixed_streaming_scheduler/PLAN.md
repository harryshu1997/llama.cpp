# S14: mixed-workload residency and streaming scheduler

Status: CP1_SHARED_TAIL_POINT_PASS_PERSISTENT_SESSION_MECHANICS_PASS (energy not run).

The persistent-worker blocker is resolved: an opt-in versioned DETACH/STOP session
protocol (in `examples/layersplit/layersplit.cpp` only; no llama-graph /
ggml_backend_sched / kernel edits) lets one resident phone `stagenet` worker serve
many sequential client sessions with an exact per-session reset and no weight
reload. Real-device gate `PERSISTENT_SESSION_MECHANICS_PASS_PHYSICAL_ENERGY_NOT_RUN`
(two resident OP15 v81 + OP12 v75 `[0,6)` workers, 7 shared-tail B1 sessions, 6
DETACH + 1 STOP; per-stream tokens identical across all 7, one resident pid/nonce
per phone throughout, HTP0-only placement). Details in `persistence/RESULTS.md`.
Tail B2 remains a frozen negative and was not retried. Energy stays unmeasured.

Current checkpoint state is recorded in `CP1_REVIEW.md`. CP1-A passes. The
matched CP1-B acquisition passes placement and correctness but is ineligible
because 10/18 timing rows fail the frozen CoV gate. CP1-D is a replayable
11.854-percent selected-GPU counterfactual, not a live offload result. The first
live serial OP15 -> OP12 B32 run completed only one P0/P2 pair; treatment used
more GPU-board energy, was 13.83x slower for Gemma, and the next treatment did
not become ready. The corrected independent OP15 B1 route then passed seven
processes and a live mixed run: selected-GPU energy fell 5.43 percent and
high-priority BGE p95 was preserved, but low-priority Gemma p95 rose 2.639x and
failed the frozen 2.0x gate. No CP1 scheduler or total-energy PASS is claimed.
An independent OP12 B1 screen now rejects `[0,8)` on repeatability but accepts
`[0,6)`: seven of seven processes and 56/56 requests pass exact-token,
placement, thermal, and raw-log replay gates at 1.152 s median route wall and
0.0258 process CoV. This enables a two-phone independent-request experiment;
it does not change the failed CP1-F SLO verdict.
The first shared-tail implementation now runs both `[0,6)` phone heads
concurrently with one A6000 tail weight copy. A B2 tail repeatedly fails exact
tokens. A B1 tail passes exact tokens at a 1.34 s two-request group wall, but a
later reload attempt times out in first compute. Persistent resident phone
sessions are therefore the next gate; no two-phone energy run is authorized.

## 0. Purpose and claim boundary

S14 is the first experiment whose subject is the intended Q-PIM system rather
than one fixed model route. It asks whether a causal two-level scheduler can use
OP12 and OP15 as active far-memory accelerators for a continuous mixed workload:

1. the slow loop chooses which certified contiguous operator islands to retain,
   replicate, stream, or evict;
2. the fast loop dispatches only already-READY islands and never makes the server
   wait for an unfinished prefetch; and
3. background weight streaming overlaps useful server or phone computation,
   while activation/result traffic retains bounded priority.

The first objective is correct mixed-workload mechanics and useful server
relief. Energy is a later gate. S11-E0 remains immutable negative evidence: the
fixed serial OP15 `[0,2)` route reduced selected-A6000 power but increased its
equal-work board energy by 53.15 percent. S14 does not reinterpret or rescue
that experiment.

The next bounded screen uses exactly one selected A6000 plus OP12 and OP15. The
second installed GPU is excluded. Under concurrent load, high-priority
compute-bound BGE work competes with low-priority memory-bound Gemma decode.
Phones may execute complete READY islands or compatible decode batches while
the selected A6000 runs other work. Returned boundaries may join a compatible
server suffix batch only after validation. A lower A6000 cap or clock is an
explicit measured decision, not an automatic consequence of high utilization.

This plan does not claim coherent memory, byte-addressable PIM, production
`llama-server` integration, phone energy, server-wall energy, or total-system
energy. The server sends explicit commands and tensor boundaries to phone-local
weights. Unknown evidence remains ineligible.

## 1. Workload contract

The initial workload is the deterministic `mix-v1` composition defined by
`WORKLOAD_TRACES.md` and the S8 normalization contract:

- BurstGPT v2.0 generation requests; and
- RAGPulse RAG requests, decomposed into explicit DAG stages.

Each source must also be replayed alone. The mixed trace is
`semi_synthetic`: source arrivals and sizes remain real within each lane, while
the cross-source alignment is a committed deterministic transform. Trace rows
do not receive invented real deadlines, priorities, payloads, or component
latencies.

The priority/SLO screen adds a separately frozen semi-synthetic sidecar because
the source traces do not contain trustworthy comparable priority and deadline
fields. At minimum it defines high-priority BGE and low-priority Gemma classes,
per-class TTFT/completion limits, and a stable tie-break. Results must never
describe those fields as observed production SLOs.

S14 requires two executable service/model classes, not two labels routed through
one Gemma island. The first candidate is Gemma generation. The second candidate
must pass its support-first funnel, CPU-reference correctness, no-fallback,
boundary, latency, and memory gates before entering the scheduler. The current
BGE embedding/reranking funnel is the preferred candidate. If no second class
passes, report `SECOND_SERVICE_BLOCKED` and stop the mixed-system claim.

## 2. Candidate island geometry

The solver selects from a finite, predeclared catalog. It never invents a graph
cut at dispatch time.

For a transformer route, the initial stateful candidates are contiguous head
islands `[0,k)`. Each candidate binds:

~~~text
model/version and graph hash
layer range and attention-class vector
weight/prepared-image identities and bytes
input/output boundary schemas and bytes
KV/state ownership and lifetime rule
device/backend route and batch/shape envelope
correctness, latency, memory, thermal, and interference evidence
~~~

The request's layer boundary and KV owner are fixed for its lifetime. A larger
`k` is eligible only when its additional compute relief exceeds transfer,
verification, preparation, memory, interference, and critical-path cost.
Putting the maximum possible layer count on a phone is not an objective.

Stateless embedding, reranking, encoder, or FFN candidates may be complete
operator islands with explicit boundaries. Per-GEMM row/column splitting across
USB or WiFi remains excluded. A serial OP15-to-OP12 chain is also excluded from
the initial catalog unless separately measured to beat independent placement.

## 3. Placement modes across two phones

For each model and candidate island, the slow loop may choose:

- `SERVER_ONLY`: no phone residency;
- `SINGLE`: one phone owns one READY copy;
- `REPLICATED`: OP12 and OP15 hold the same island and drain independent jobs;
- `DIVERSE`: each phone holds a different model or island; or
- `STAGING`: a future generation is being transferred but cannot execute.

Replication is valuable for a hot repeated model and queue throughput. Diverse
placement is valuable for model-mix coverage and reduced churn. The scheduler
chooses between them from visible ANNOUNCED demand, measured service rates,
memory, thermal state, and transfer amortization. It does not force both phones
to run every interval.

The first physical two-phone mode is replicated independent service, not a
serial chain. OP15 owns a READY `[0,8)` B1 lane and OP12 owns a READY `[0,6)` B1
lane. A shared A6000 tail context must consume returned boundaries from either
lane without loading a second tail weight copy. Each request stays on one phone
for its complete head island; no request crosses OP15 then OP12.

## 4. Double-buffered weight generations

Each phone has at most one ACTIVE generation per island and one bounded STAGING
generation:

~~~text
G0: READY/ACTIVE, leased by current executions
G1: STAGING -> VERIFIED -> PUBLISHED -> PREPARED -> READY
~~~

The slow path transfers G1 while G0, another resident island, or the A6000 does
useful work. It uses the bounded S9 chunk protocol, verified-prefix resume,
content identity, durable publish, preparation, and generation-qualified
leases. G1 becomes dispatchable only after the complete island is atomically
published and prepared. Partial bytes and on-disk-only data never satisfy READY.

G0 and G1 require independent residency-slot generations. The current phone-PIM
worker has one global `ResidentState` and one generation; incrementing that
generation for G1 would incorrectly stale G0 while G0 still has live leases.
CP3 therefore adds explicit per-slot identity and generation before claiming
double-buffered overlap.

Large weights use USB H2P. Small commands, token IDs, sequence metadata, and
small input activations use WiFi H2P. Dense results use USB P2H. On each phone,
USB result/state traffic preempts background weight transfer at a bounded chunk
boundary. Transfer and backend preparation are separate stages because either
may interfere with compute. Their measured interference profiles determine
whether overlap is eligible.

Eviction first marks a generation DRAINING, rejects new dispatches, and waits for
all execution, state, result, and alias pins. The old generation is reclaimed
only after the replacement or declared fallback state is usable. No alias from a
previous model generation may survive reload.

## 5. Two-level decisions

### Slow loop

At a bounded epoch, choose:

~~~text
x[model,island,k,phone] in {absent, staging, ready_mirrored, ready_exclusive}
replica count and phone assignment
stream start/order and bounded byte credits
hold, drain, eviction, and ownership epochs
prepared HTP/GPU route and co-run envelope
~~~

The placement score uses only causal state:

~~~text
benefit =
    visible announced reuse relief
  + predicted server batch/power/HBM relief
  - transfer/verify/publish/prepare cost
  - eviction and memory opportunity cost
  - compute/transport interference cost
  - expected stale or unused prefetch cost
~~~

Forecast-only bytes are limited by a fixed exploration budget and earn no
benefit until used. The 216 MiB/s OP12 and 262 MiB/s OP15 figures are only
ADB-to-file staging controls; they exclude the complete durable
verify/publish/prepare path and cannot price READY residency. The repaired S9
full-shard protocol measured about 36-37 MiB/s at its best tested windows, with
OP15 reaching 95 C. Multi-GiB weights therefore require long reuse horizons,
and the solver must use the complete measured stage decomposition rather than a
raw cable or ADB rate.

### Fast loop

At every arrival and completion:

1. materialize concrete READY DAG nodes;
2. dispatch only against a current READY envelope and finite lane/link/state
   credits;
3. use native batches for compatible same-model work and independent active
   bursts for different models;
4. overlap phone work, server work, and eligible background streaming;
5. launch the SLO-safe server route when a phone or prefetch is not ready; and
6. record useful and wasted phone compute, prefetched bytes, activation bytes,
   queue delay, and terminal outcome.

Batch launch is SLO-driven. Memory-bound decode launches at the largest measured
compatible batch that can finish before the earliest member's latest start.
Compute-bound work launches at the smallest measured batch reaching its
throughput knee. Urgent requests bypass a batch wait. The selected A6000 may use
a lower measured cap or clock only when phone relief plus the resulting server
batch plan remains SLO-feasible and reduces energy per completed work.

The fast loop never waits for cold weights and never converts ANNOUNCED work
into an executable request.

## 6. Required controls

All controls consume the same closed request cohort and use the same correctness
and SLO rules:

| ID | Policy |
|---|---|
| C0 | optimized server-only DAG order and native batching |
| C1 | fixed static phone placement, no dynamic residency |
| C2 | mixed virtual queue with static READY residency |
| C3 | dynamic placement with transfers serialized outside useful compute |
| C4 | dynamic placement with double-buffered streaming overlap |
| C5 | clairvoyant bounded oracle using the same measured catalog |

Report the marginal effect of the virtual queue, dynamic placement, and overlap
separately. A win over eager FIFO but not C0 is not a system win.

For the one-GPU compute-pressure screen, evaluate four matched operating-point
subcontrols for C0 and the selected phone policy:

~~~text
P0 normal GPU state, server-only batching
P1 lower GPU state, server-only batching
P2 normal GPU state, phone relief plus server batching
P3 lower GPU state, phone relief plus server batching
~~~

All four bind the same selected GPU UUID and prove that the second GPU performs
zero experiment work.

The candidate UUID for the CP1/CP4 matched gate is
`GPU-45b611d6-7c5d-9e48-f260-4fbe5f8ef69f`; the excluded board is
`GPU-431a4567-fa90-7a73-625a-2ee6e7b5eaaf`. Earlier route and overlap smokes
that selected the latter board remain mechanics evidence only and cannot be
combined with the matched P0-P3 cohort.

## 7. Checkpoints

### CP0: executable trace and island catalog

- [x] Implement and hash-bind the deterministic S8 `mix-v1` transform.
      (`compose_mix.py`; frozen NORMALIZATION_SPEC section 7; byte-identical rerun.)
- [x] Replay BurstGPT, RAGPulse, and `mix-v1` structurally with byte-identical
      reruns and exact terminal accounting. DONE on the REAL pinned sources
      2026-07-17: BurstGPT_3.csv (231682327 B, sha256 2299986a) and RAGPulse
      0_trace.jsonl (1923473 B, cd371571) fetched + verified, normalized (BurstGPT
      5,344,021 records / 9874 bins), and composed to the committed
      `configs/mix-v1.config.json` (median lanes, offset 0/0). Real `mix-v1` =
      177 rows (165 burstgpt + 12 ragpulse), byte-identical rerun (output_sha256
      567d4af1), structural `replay_mix` PASS: order preserved, demand
      input 112911 / output 21595 / retrieved 60, services api_generation 152 +
      conversation_generation 13 + rag_qa 12. Run + hashes in
      `scratchpad/s8_mix_v1/`.
- [x] Complete the second-service support/correctness funnel. DONE 2026-07-17,
      `GATE1_HTP_BGE_FUSED_PASS` (`GATE1_HTP_BERT_OPS.md` sections 2-7). Op-level +
      exact-BGE-shape + END-TO-END: the fused bge-small-en-v1.5 encoder runs on
      HTP0 on BOTH OP15/v81 and OP12/v75, pooled-CLS cosine 0.9973 vs CPU (both FA
      modes), only GET_ROWS on CPU (declared exception, f16 embd table). A
      phone-resident BGE embedding island is a real second executable service
      class. NOT `TRACE_OR_SERVICE_BLOCKED`. A stable A6000 timing sweep now
      exists but is not yet atlas-eligible; phone latency and BGE energy remain
      unmeasured. F32-embd + approx-GELU are non-blocking follow-ups.
- [ ] Measure candidate `[0,k)` and stateless-island rows on A6000, OP12, and
      OP15, including boundary bytes and no-fallback placement. PARTIAL: the
      2026-07-17 consolidation (no fresh device run) bound all existing
      digest-pinned measured evidence (S11 gemma [0,2) stage latency/memory,
      Gate-1 BGE cosine 0.9973 + no-fallback op support) and computes boundary
      bytes structurally. Fresh-latency gaps (BGE per-encode, A6000 BGE baseline,
      a coherent gemma latency+PLACEMENTCERT run, larger k) are enumerated as the
      CP1 measured-atlas work-list in `CATALOG.md` section 5. The new A6000 BGE
      timing sweep remains outside the catalog until the CP1 evidence repairs
      below are complete.
- [x] Freeze the finite candidate catalog before scheduler results are viewed.
      DONE 2026-07-17. `island_catalog.json`, `catalog_hash`
      sha256:3cf13792185f41919af3a6ee47fdb41eb236b93e0967b31082ab062dd3d4b3d5,
      2 models / 4 islands / 5 rows, schema-validated + digest-bound + fail-closed
      (`build_catalog.py`, `validate_catalog.py`, 22 tests, `SHA256SUMS.txt`,
      `CATALOG.md`). A 5-lens adversarial review confirmed 2 binding defects on
      the gemma rows (unbound `fallback=none`; false binary identity); both fixed,
      so the HONEST frozen state is ZERO dispatch-eligible rows (no single run
      yields coherent latency + same-run HTP no-fallback certificate). NOT energy;
      no commit.

Stop if two executable service/model classes do not exist.

### CP1: static resident mixed runtime

- [x] Extend the bounded host harness, not `llama-server`, to admit the mixed DAG
      and model-specific reduction flow; add an optional live-mode dispatch hook that
      invokes `llama-phone-pim-fleet` with fail-closed parsing.
- [x] Enable a live two-session device capacity-probe path with both persistent
      phone sessions and bounded timeout. Returned records are strictly bound to
      the requested devices, model, route, backend, generation, jobs, and
      correctness limit. They remain `live_fleet` diagnostics and do not alter
      the offline reducer's terminal, SLO, or relief metrics.
- [ ] Pre-stage all selected weights before the paid interval.
- [ ] Exercise replicated and diverse placements with finite VQ, lane,
      activation, result, and KV/state credits.
- [x] Compare offline C0, C1, and C2 mechanics with identical modeled work and
      terminal conservation. This is not a physical output-equivalence result.
- [x] Repair the BGE server benchmark: use explicit sequence parallelism and
      direct per-request timing; report p50/p95/p99 and the measured throughput
      knee instead of treating token batch capacity as prompt count.
      The first repaired sweep produced stable timings, but is not yet an atlas
      row: persist all per-process samples and stderr, verify model/binary/source
      identities, emit a graph placement certificate, and bind the artifacts.
      Its FA-off graph builds attention over total `n_tokens x n_tokens` and the
      graph contains an equal-sequence stream-split TODO. Therefore the current
      independent-sequence `B x L^2` roofline is not the executed graph cost.
      The next-run profiler schema v3 now models total-token attention, reports
      masked excess work, and refuses to derive a regime from its weight-only
      intensity upper bound. The existing v2 result remains historical and
      must not be relabeled without a rerun.
      First profile/test an equal-sequence or block-diagonal attention split as
      a separate operator/graph optimization; do not infer the roofline class
      from the current analytic field.
- [x] Rebuild and deploy an instrumented OP12 binary; make the three-device
      harness fail unless OP12 and OP15 both emit passing placement certificates.
- [ ] Add priority-aware compatibility queues and real native batches to the
      reducer/runtime. Priority must affect dispatch order; `batch_size=1`
      actions do not satisfy this item.
      Host integration substrate now exists in `priority_batch_runtime.py`: it
      owns per-route queues and launches, binds every launch to a measured
      profile and route epoch, and admits completion only from an exact set of
      per-request boundary certificates. Its adversarial suite has 15 tests.
      This item remains open until those launches invoke the real BGE and
      independent phone executors and their outputs drive terminal results.
- [x] Implement the measurement-independent fast-policy primitives for that
      integration: strict priority ordering, compatibility-key isolation,
      SLO-bounded memory/compute batch release, validated boundary admission,
      and one-GPU P0-P3 selection (`power_frontier_policy.py`, 15 tests). This
      does not mark the reducer/runtime integration item complete.
- [x] Run the one-GPU priority screen: high-priority BGE on A6000 concurrently
      with low-priority Gemma decode. The first live serial OP15 -> OP12 attempt
      failed: its completed P2 cohort used 1.293x selected-GPU energy and had
      13.83x Gemma p95 latency, then repeat 1 failed before readiness. Continue
      The independent OP15 B1 rerun passes seven-process correctness and
      placement. Its live P2 reduces selected-GPU energy by 5.43 percent and
      preserves high-priority BGE p95, but raises low-priority Gemma p95 2.639x
      and fails the frozen 2.0x gate. OP12 next receives a separate READY island
      or request stream. P1/P3 remain unmeasured because this host cannot set a
      lower A6000 power or clock point.
- [ ] Repair or replace the batched OP15 correctness gate. The current-source
      independent `[0,8)` route is exact at B1 but differs from the same-batch
      full-model token stream at B4/B8/B32. Placement alone cannot authorize a
      batch. Until an activation/logit/task-quality certificate is defined and
      measured, only B1 is eligible and the batch-decode thesis is unproven.
- [x] Require correctness and placement evidence per exact batch in the live
      runtime. `CertifiedBatchPoint` carries both evidence IDs; bare latency
      points are rejected before dispatch.
- [ ] Measure and enforce each class's latest-start rule, useful batch fraction,
      p50/p95/p99 latency, SLO attainment, and wasted/late phone work.

Current real-device batching is deliberately not called integrated CP1. The
serial three-device route is token-correct in single acquisitions at
B={1,4,8,32}, but B16/B64 did not complete and B32 did not repeat. The result
reinforces the catalog rule: independent or replicated placement is the default;
a serial phone chain must first beat it. CP1 remains open until native batches
are selected by the priority/SLO runtime and their returned activations drive a
repeatable measured suffix.

Stop if static mixed placement cannot retain at least 0.90x C0 useful throughput
while providing a measurable server-compute or HBM-relief mechanism.

### CP2: tiny exact placement oracle and causal policy

- [ ] Extend the bounded S10 temporal model with model/island/phone placement,
      replica count, transfer generations, memory, and eviction.
- [ ] Compare exhaustive search and an independent checker on tiny fixtures.
- [ ] Add counterexamples where replication wins, diversity wins, prefetch loses,
      and a smaller `k` beats a larger `k`.
- [ ] Implement a deterministic causal slow-loop heuristic and report oracle gap.

Stop if the causal policy cannot retain the declared mechanics opportunity.

### CP3: overlapped weight streaming

- [ ] Add one bounded STAGING generation per phone to the live harness.
- [ ] Measure USB H2P streaming alone and concurrent with A6000, HTP, GPU, WiFi
      H2P, and USB P2H work.
- [ ] Enforce result-preempts-weight priority and bound its preemption delay.
- [ ] Prove resume, stale generation, crash, drain, rollback, and no-dispatch-
      before-READY behavior on both phones.
- [ ] Compare C3 and C4 under the same trace and placement decisions.

Mechanics require no correctness or fallback failures, zero dispatches from
partial/on-disk-only data, and at most 5 percent p95 activation/result slowdown
from background streaming. Report useful, unused, retried, and evicted bytes;
do not hide prediction waste.

### CP4: causal real-trace replay

- [ ] Run low, median, high, and burst windows from each source and `mix-v1`.
- [ ] Sweep model popularity, memory pressure, thermal state, and transfer rate.
- [ ] Compare C0-C5 and report SLO, throughput, HBM, batch density, queueing,
      churn, prefetch efficiency, phone utilization, and oracle gap.
- [ ] Sweep the selected A6000's measured normal/lower power or clock points and
      report whether phone relief causally changes the chosen operating point.
- [ ] Run at least 30 minutes for the final shortlisted resident/streaming policy.

Only a CP4 mechanics pass authorizes a narrow selected-A6000 energy diagnostic.

### CP5: energy and production decision

- [ ] Freeze a new rotated matched cohort; do not reuse S11-E0 measurements.
- [ ] Measure one selected A6000's board energy first, with phone energy UNKNOWN;
      verify the second GPU is excluded and idle experiment work is zero.
- [ ] Add server-wall and total-wall claims only after valid instruments exist.
- [ ] Integrate the smallest winning mechanism into `llama-server` only after a
      causal real-device result survives the optimized C0 control.

The paper-level target remains at least 10 percent lower synchronized total-wall
J/work at equal work/SLO, or 10 percent more SLO-valid work at equal wall power.
Until total-wall instrumentation exists, any GPU-board result is diagnostic and
must state the excluded phone/USB/relay boundary.

## 8. Implementation boundary

The first implementation stays in bounded research harnesses and reuses:

- S8 normalized trace and DAG contracts;
- S9 verified residency, resume, generation, and windowed transport;
- S12-V2 `KEEP`, `PREFETCH`, `REPLICATE`, pin-safe replacement, and cold-miss
  server fallback mechanics, plus the S12 asymmetric path model;
- S13 persistent two-phone protocol-v3 sessions and completion-driven fleet; and
- S10 bounded temporal oracle/checker infrastructure.

S12-V2 is the starting state reducer, not a second scheduler to rewrite. It is
currently symbolic: it lacks real multi-island DAGs, measured profiles, tail
batching, thermal state, and energy. S14 first adds the measured-profile adapter
and normalized mixed replay, then connects only the passing mechanics to S13's
live persistent phone sessions.

Do not modify `gemma4.cpp`, `llama-graph.cpp`, or `ggml_backend_sched` for CP0-
CP3. Do not add a new kernel unless a complete-island profile identifies a
kernel bottleneck and a separate operator test passes first. Do not commit,
push, or claim production integration without explicit human approval.

## 9. Verdict vocabulary

- `TRACE_OR_SERVICE_BLOCKED`: two executable service classes do not exist.
- `STATIC_MIXED_MECHANICS_FAIL`: CP1 cannot preserve useful throughput/relief.
- `PLACEMENT_POLICY_FAIL`: CP2 causal placement loses the bounded opportunity.
- `STREAMING_OVERLAP_FAIL`: CP3 interference or readiness mechanics fail.
- `MIXED_RUNTIME_MECHANICS_PASS`: CP4 passes; energy remains unmeasured.
- `GPU_BOARD_DIAGNOSTIC_RELIEF_PASS/FAIL`: selected-board boundary only.
- `TOTAL_SYSTEM_ENERGY_PASS`: reserved for synchronized total-wall evidence.
