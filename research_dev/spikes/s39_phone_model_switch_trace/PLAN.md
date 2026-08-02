# S39 Active warm-tier model-switch proof of concept

Status: `V2_4_NO_MODEL_TOPOLOGY_PASS; USB_LAUNCHER_MECHANICS_PASS; ARTIFACT_RECEIPT_IN_PROGRESS; ACQUISITION_NOT_RUN; CYCLE_BLOCKED`

Authority: [../../ACTIVE_WARM_TIER_DESIGN.md](../../ACTIVE_WARM_TIER_DESIGN.md).

S39 is the first bounded implementation of the active warm-tier design. It does
not implement a general multi-model cache, direct KV migration, or an energy
claim.

CP0-D `DESKTOP_TWO_MODEL_BASELINE`, defined in
`../s39_desktop_swap_baseline/PLAN.md`. It is a server-only control that must
precede further phone qualification. It authorized only the pinned Qwen3 8B
desktop download, exact digest verification, independent RTX 4060 Ti
qualification of Qwen3 8B and Qwen3 14B, measured non-co-residency, and
repeated warm-cache and cold-NVMe one-GPU swap replays. Both models and
non-co-residency pass. Warm cache passes 3/3; cold NVMe fails 3/3 with 57/74
requests stranded. It ran no phone command and created no phone shard. See
`../s39_desktop_swap_baseline/RESULTS.md`.

CP0-R1 `TWO_ROUTE_ELIGIBILITY` is the active phone gate.
`v24_readiness/CP0_R1_EVIDENCE_CONTRACT_V2_4.json` is the sole prospective
exit authority for Qwen3-14B `A_ONLY`. V1, V2, V2.1, V2.2, V2.3, W9, and the
CUDA replay-partition diagnostic are immutable parents or historical evidence.
They do not qualify a route or authorize a model-switch cycle. B_ONLY and PAIR
require later, separately frozen successors after the preceding phase passes.

## Question

Can OP15 and OP12 collectively serve a GPU-nonresident model while a 16 GiB
desktop GPU changes models, then move all live requests into one CUDA
continuous batch by replaying committed token histories without stopping
service?

After cutover, can the phones release the promoted model and prepare the
displaced model so the same transition can run in reverse?

## Historical and current model pairs

The original provisional pair was Gemma 4 12B IT plus Qwen3 14B. Gemma remains
`FAIL_CORRECTNESS`; it is historical negative evidence and is not a CP0-R1
candidate.

The only CP0-R1 pair is:

- model A: Qwen3 14B Q4_K_M, the incumbent `PROVISIONAL_BATCH` route;
- model B: Qwen3 8B Q8_0, the single permitted new candidate.

Qwen3 8B uses the official `Qwen/Qwen3-8B-GGUF` Q8_0 artifact at revision
`6cfbfc7d8ab95bf485c79fcc40be60930d5b4c8c`. The artifact has SHA-256
`408b955510e196121c1c375201744783b5c9a43c7956d73fc78df54c66e883d6`,
8,709,518,112 bytes, and 36 layers. This pair can establish a capacity result
only; it does not establish architecture diversity.

The current Qwen3 14B assignment remains a candidate, not an eligible route. Real
B1, B8, and B32 collective-phone cohorts match their same-artifact CUDA
control, but corpus, repeated-process, memory, and measured-network-path gates
remain. CP0-R1 must qualify it under the same contract as Qwen3 8B. S39 must
not route around either route's missing evidence.

Every checkpoint is stored on desktop NVMe and every assigned phone shard is
stored on phone UFS before a run. The treatment does not send a 9-12 GB model
from phones to the desktop during promotion.

Provisioning checkpoint, 2026-07-23:

- both frozen source GGUF hashes match `model_assignment.json`;
- OP15 stores Gemma `[0,36)` and Qwen `[0,32)`;
- OP12 stores Gemma `[24,48)` and Qwen `[24,40)`;
- the overlaps preserve Gemma candidate cuts 24-36 and Qwen cuts 24-32;
- all four final on-device SHA-256 values match
  `SHARD_MANIFEST.json`;
- readiness is `STAGED_ONLY`, not executable or scheduler-ready.

The partial-load graph and layer-filtered KV path now support dense Qwen3 as an
experimental LayerSplit route. Storage readiness remains distinct from
execution readiness. `CURRENT_ROUTE_READINESS.json` is derived from hashed raw
phone and CUDA records by `build_route_readiness.py`; it cannot issue `PASS`
from the current evidence.

## Frozen trace

`build_trace.py` verifies the pinned BurstGPT v2 source and scans aligned
20-minute windows without changing time. A successful request has
`output_tokens > 0`. The frozen source bin is 21,424:

- 20 source rows, including 19 successful requests and one recorded failure;
- 15 ChatGPT and 4 GPT-4 successful requests;
- 4,919 input tokens and 1,038 output tokens;
- arrivals from 12 to 1,143 seconds in a 1,200-second window;
- three same-session GPT-4 requests at 180, 198, and 234 seconds;
- one later GPT-4 request at 543 seconds.

The raw trace remains `real`. Mapping ChatGPT to model A and GPT-4 to model B is
`semi_synthetic`. Prompt payloads and SLO classes are also separate, hashed
sidecars. The recorded failed request remains in the trace and is not executed.

The second model-B request within 60 seconds, at trace time 198 seconds, is the
first provisional promotion trigger. The isolated model-B request at 543
seconds is the hysteresis control and should remain on phones because a model-A
request arrives two seconds later.

Two trace profiles are retained:

- `bundle/` is the low-load correctness trace from source bin 21,424.
- `bundle_frequent/` is the active frequent-switch trace from source bin
  16,522. It has 74 successful requests, five useful model-B promotion cycles,
  and nine target changes during arrivals.

`ACTIVE_TRACE.json` selects `bundle_frequent/` for current testing. The
correctness trace remains the fallback for debugging ownership, replay, and
token-conservation failures.

`build_replay_intents.py` deterministically converts the selected trace into
five promotion windows and nine ordered policy intents at 60, 129, 359, 421,
498, 546, 911, 989, and 1,154 seconds. The replay manifest binds the active
trace, assignment, shard manifest, builder, and output bytes. These are policy
intents only, not proof that a route is ready or that a transition executed.

## W0 route screen, 2026-07-23

Both phones ran one complete model route at cut 30 with GPUOpenCL and greedy
decoding. The same prompt and token budget ran on CUDA from the exact source
GGUF.

| Model | Phone route | CUDA control | Derived status |
|---|---:|---:|---|
| Qwen3 14B Q4_K_M | 8 tokens, exact match | 8 tokens | `PROVISIONAL_B1` |
| Gemma 4 12B Q4_0 | 8 tokens, mismatch | 8 tokens | `FAIL_CORRECTNESS` |

Qwen's phone TTFT was 1.574 seconds and service wall time was 3.668 seconds for
the five-token prompt plus eight generated tokens. Its CUDA control completed
in 0.163 seconds. Gemma's phone TTFT was 1.317 seconds and service wall time was
2.724 seconds; its CUDA control completed in 0.149 seconds, but token zero
already differed (`496` versus `9079`).

The Qwen placement certificates report 9,000 OP15 OpenCL compute nodes and
3,060 OP12 OpenCL compute nodes, with only declared OP15 `GET_ROWS` work on
CPU. Gemma placement is similarly clean, so placement alone does not repair
its numerical failure. A same-host CUDA split at cut 30 reproduces Gemma's
monolithic top-5 exactly, which narrows the failure to the heterogeneous phone
route or backend numerics rather than the layer boundary.

`warm_tier_controller.py` consumes the hash-bound intents and derived route
readiness. With current evidence it exits nonzero with
`E_ROUTE_NOT_READY`; with test-only PASS routes it exercises all nine
transitions through drain, load, catch-up, edge drain, reprepare, and atomic
ready publication. No physical promotion is authorized.

## Two-model execution invariant

The active replay must execute requests for both models. It is not sufficient
to execute Qwen on the phones while treating the Gemma rows and residency
changes as labels.

For an intent at the same timestamp as its source request, event ordering is:

1. admit the source request to the current phone-warm model;
2. begin the GPU promotion intent;
3. keep the phones authoritative while CUDA loads and reconstructs state;
4. commit one ownership frontier;
5. route later requests for the promoted model to CUDA;
6. prepare the displaced model on the phones before the reverse intent.

`validate_two_model_trace.py` binds the real request rows, model assignment,
and replay intents. The active trace contains 57 successful Gemma requests and
17 successful Qwen requests. Both models have a phone-side trigger witness and
a later CUDA-side request witness. The nine intents alternate the planned
state:

```text
GPU Gemma / phones Qwen
-> GPU Qwen / phones Gemma
-> GPU Gemma / phones Qwen
-> ...
```

`TWO_MODEL_TRACE_CONTRACT.json` records these obligations with status
`TRACE_CONTRACT_PASS_PHYSICAL_EXECUTION_NOT_RUN`. A physical result passes only
if all 74 successful requests execute exactly once, both models execute on
both tiers, and all nine residency transitions complete. The trace contract
is not execution evidence.

## W0 batch and transport screen, 2026-07-23

The Qwen workers remained resident for three sessions: B1 and B8 ended with
`DETACH`, and B32 ended with `STOP`. OP15 ran `[0,30)` and OP12 ran `[30,40)`.
Every cohort generated the same eight greedy tokens as the frozen CUDA
control.

| Requests | P50 service | Request rate | Gain over B1 | Activation payload |
|---:|---:|---:|---:|---:|
| 1 | 3.825 s | 0.261 req/s | 1.00x | 0.47 MiB |
| 8 | 24.859 s | 0.322 req/s | 1.23x | 3.75 MiB |
| 32 | 87.548 s | 0.365 req/s | 1.40x | 15.00 MiB |

The B32 prefill call contains 160 token rows. Each of its seven decode calls
contains 32 rows. This proves static cohort batching and persistent session
reset. It does not yet prove continuous admission across unequal arrivals,
inter-stage overlap, or a useful serving throughput.

Weights were provisioned by USB ADB before the paid route and loaded from phone
UFS. Runtime hidden states traveled by WiFi TCP through the host coordinator:
OP15 to host, then host to OP12. The route reports did not capture interface
counters or socket peers, so the exact path remains a post-run operator record.
See `TRANSPORT_CONTRACT.md`.

The host relay is the W0 control, not the target path. Direct OP15-to-OP12 WiFi
reachability is confirmed in both directions with zero packet loss. The next
runtime gate keeps host-side scheduling and ownership but sends reserved
activation batches directly to OP12.

`summarize_qwen_batch_wifi.py` re-derives the batch certificate from all three
reports, both worker logs, the CUDA control, the shard manifest, and the run
context. The derived readiness is `PROVISIONAL_BATCH`, and the controller still
refuses it.

## W1 direct phone chain, 2026-07-23

An additive relay now runs on OP15. The host sends token and lineage rows to
the relay, OP15 computes `[0,30)`, the relay sends each F32 cut activation
directly to OP12 over WiFi, OP12 computes `[30,40)`, and only terminal token
rows return to the host.

The corrected relay binary ran persistent B1 and B32 sessions on both phones.
All 264 generated-token checks matched the same-artifact CUDA sequence. B32
again formed one 160-row prefill batch and seven 32-row decode batches. The
direct relay moved 7,864,320 activation bytes to OP12 and reported zero host
activation payload bytes.

This is a mechanics result, not a latency result. The corrected direct B1
point was 5.653 seconds versus 5.696 seconds for one separately loaded
host-relay control, only 0.76 percent lower. Direct B32 was 90.796 seconds.
Process-to-process variation is larger than the apparent B1 difference, and
phone compute remains dominant. The current status is
`DIRECT_CHAIN_MECHANICS_PASS_REPEATS_PENDING`.

The relay enforces model identity, a contiguous cut, route capacities, lineage,
sequence cleanup, and persistent reset. It still relies on a single-client
implicit reservation. Interface counters and a host-issued cryptographic
batch descriptor were not captured, so W1 does not promote Qwen to an eligible
route. See `RESULTS_W1.md` and `results/w1_direct_phone_chain/`.

## W2 mixed-phase direct chain, 2026-07-24

The direct route now has a bounded phase-aware admission queue for one fixed
model and cut. Decode rows are physically ordered first. Prefill rows can fill
the remaining row capacity until the batch knee or earliest dispatch deadline.
Accepted groups are atomic, queue capacity is finite, and expired prefill
cannot starve behind a continuing decode stream.

One real-device run first admitted 16 Qwen requests, then submitted 16 live
decode rows with 80 new prefill rows. The batcher emitted one 96-row
`llama_decode` with the 16 decode rows first and all 80 prefill rows after
them. It continued all 32 sequences with six B32 decode calls and one final
B16 call. All 256 generated-token checks matched the CUDA sequence.

The physical batch sequence was:

```text
80P, 16D+80P, 32D, 32D, 32D, 32D, 32D, 32D, 16D
```

The route moved 7,864,320 activation bytes directly from OP15 to OP12 and zero
activation bytes through the host. Worker placement, session-step
conservation, sequence cleanup, and relay row counts passed the independent
reducer. Status is `DIRECT_MIXED_BATCH_MECHANICS_PASS`.

W2 does not yet prove a throughput or latency gain. Its 23.219-second maximum
is much lower than W1's 90.796-second unordered B32 point, but ordering,
prefill shape, and device state changed together. The next acquisition is a
matched sorted-versus-shuffled row-order control. Inter-stage overlap remains
unimplemented because the relay permits only one in-flight batch.

## W3 matched row-order gate, 2026-07-24

W3 held the model, cuts, workers, physical batch shapes, row count, activation
bytes, tokens, and KV lifecycle fixed. It changed only the sequence-row order
and reversed the treatment order on a second freshly loaded worker pair.

Canonical rows took 24.911 and 23.580 seconds of route compute. The fixed
shuffled permutation took 110.115 and 109.498 seconds. The aggregate effect is
4.528x, and both AB and BA pairs exceed the predeclared 1.20x gate. All 1,024
generated-token comparisons pass. Mean treatment-start temperature differs by
only 0.55 C on OP12 and 0.20 C on OP15.

Worker certificates explain the effect. Canonical order produces eight graph
executions per session. Shuffled order produces 136, expanding OP15 compute
nodes from 5,528 to 93,976 and OP12 nodes from 1,880 to 31,960. Physical batch
size alone was therefore not a sufficient batching certificate.

Canonical phase, sequence, and position ordering is now a route invariant.
This is an implementation prerequisite, not the system's primary novelty.
Inter-stage overlap remains unimplemented, but it is subordinate to the
two-model switch proof. See `RESULTS_W3.md`.

The shared `MixedPhaseBatcher` now enforces that invariant after admission.
Priority and deadlines select batch membership; physical rows are then ordered
by phase, sequence ID, and position without changing future ownership. A new
real-device mixed 16+16 session passes all 256 token checks and placement
gates with 23.493 s maximum completion. Host tests also prove that shuffled
B32 submission is canonicalized before the backend call. Status is
`CANONICAL_ROW_ORDER_INTEGRATED_REAL_DEVICE_PASS`.

## W4 second-model quality screen, 2026-07-24

Qwen2.5 14B Instruct Q4_0 was the first replacement candidate for the failed
Gemma phone route. The exact full artifact is 8,517,726,048 bytes with SHA-256
`924a4c39ef9fc6c139875ab6771c2e8172a3b40ffec5c720eca69ad7a0edfae7`.
Experimental Qwen2 partial-layer loading and graph boundaries were added using
the existing Qwen3 LayerSplit pattern. A Qwen2.5 0.5B host control matched a
monolithic run exactly across the boundary before the 14B phone screen.

The gate used 128 deterministically selected and Qwen-tokenized WikiText
prompts. Each prompt contains eight input tokens and eight greedy token
decisions. Four independent B32 cohorts ran through the complete direct phone
route and through a same-artifact CUDA split reference. Prefill was submitted
as four sequence-major 64-row calls per cohort, followed by seven B32 decode
calls. This preserves the eight-token workload while avoiding an unsupported
256-row phone prefill call. The frozen thresholds were:

- first-token agreement at least 95 percent;
- token-decision agreement at least 95 percent;
- exact eight-token sequence agreement at least 80 percent.

All three phone routes fail:

| Backend route | First token | Token decisions | Exact sequences | Verdict |
|---|---:|---:|---:|---|
| HTP cut 30, fused attention | 97.66% | 89.65% | 75.00% | fail |
| HTP cut 30, explicit attention | 96.88% | 90.04% | 77.34% | fail |
| GPUOpenCL cut 32, fused attention | 95.31% | 85.94% | 70.31% | fail |

The GPUOpenCL run completed 44 physical batches and 1,920 rows. Its direct
relay moved 39,321,600 activation bytes from OP15 to OP12 and zero activation
bytes through the host. Both phone placement certificates report
`SCHEDULED_PLACEMENT_OK`, zero missing buffers, and only the declared OP15
`GET_ROWS` host operation. OP12 reports an optional OpenCL kernel compile
failure, but every realized compute node remains on OpenCL. The CUDA reference
has the same cut, batch sequence, row count, and model identity.

`validate_qwen25_quality.py` independently recomputes every agreement count,
token digest, route row count, activation byte count, and placement predicate
from the persisted evidence. A post-run record binds the staged shard and
runtime hashes to the same device boot IDs in the session certificates. This
is same-boot attribution, not a pre-run acquisition record. Six adversarial
tests reject token, verdict, relay, placement, boot, and JSON mutations. The
derived certificate is `QUALITY_FAIL`, with `scheduler_eligible=false`.

One exact B32 prompt was therefore not representative of corpus behavior.
Qwen2.5 Q4_0 is rejected as the second warm-tier model. No repeat, energy
measurement, trace integration, or bidirectional replay is authorized for
this route. See `RESULTS_W4.md`.

## Controls

- `C0 server_queue`: drain A, load B, then begin B work.
- `C1 phone_finish`: phones serve B during the switch; phone-started requests
  finish on phones.
- `C2 catchup_handoff`: phones serve B while CUDA loads; CUDA batch-prefills
  committed histories, consumes the token delta, and takes ownership.
- `C3 rotating_warm_tier`: C2, followed by phone preparation of A and one
  reverse A promotion.
- `C4 two_gpu_oracle`: keep both models on separate GPUs as a performance bound,
  not a resource-matched baseline.
- `C5 host_warm_executor`: execute the alternate model from desktop CPU/RAM
  while the single target GPU changes residency.

## Historical post-W9 proposal

The earlier plan made one reduced bidirectional cycle conditional on W9. W9
failed and the CUDA-only diagnostic later showed why its cross-geometry oracle
was invalid. That proposal is retained as history, not as the next gate.

Gemma and Qwen2.5 remain rejected routes. Do not rerun their phone quality
screens, weaken their old gates, or reclassify them from the new contract.

## CP0-R1 two-route eligibility - current bounded gate

`CP0_R1_TWO_ROUTE_ELIGIBILITY_CONTRACT.json` remains the immutable v1
model-independent policy. `CP0_R1_CANDIDATE.json` binds the one allowed
candidate attempt. V1 accepted internally consistent summary records and is
therefore not an evidence-admission authority.

`CP0_R1_EVIDENCE_CONTRACT_V2.json` is the frozen first raw-evidence successor.
It required A, B, and pair evidence in one bundle, so it could not qualify the
incumbent before acquiring the one allowed candidate.

`CP0_R1_EVIDENCE_CONTRACT_V2_1.json` is a frozen historical parent.
`cp0_r1_evidence_v21.py` reopens and hashes every raw artifact once, parses
those same bytes, rejects supplied summaries, and evaluates three separate
interval-bound phases in order: `A_ONLY`, `B_ONLY`, and `PAIR`. B re-evaluates
the raw A bundle; the pair re-evaluates both raw model bundles. Each phase
binds its own prospective lock and every event to one `HOST_MONOTONIC_RAW`
interval. `cp0_r1_phase_preflight_v21.py` runs the exact recorded argv for the
target GPU, both ADB servers, both phone identities, the complete desktop
model, and complete phone shards. The evaluator requires that full readiness
to finish no more than five seconds before acquisition begins.

V2.1 additionally derives exact B8 call-row coverage and equal comparison
lengths, exact CUDA used-plus-free accounting and target-process attribution,
corpus-to-item output linkage, phone-publication and CUDA-ready linkage, exact
F32 activation bytes for every direct transfer, and an exact full-shard
local-UFS read in each reprepare. The checked-in v2 no-model preflight remains
identity evidence only; it is not a V2.1 qualification phase.

`CP0_R1_EVIDENCE_CONTRACT_V2_2.json` is a frozen parent. It retains every
V2.1 gate and additionally:

- exact-checks the canonical 64-row corpus generated from all pinned MMLU test
  parquets;
- requires CUDA to answer at least 25/64 items correctly;
- exact-binds incumbent A to `GPUOpenCL`, cut 30, stored ranges `[0,32)` and
  `[24,40)`, and both candidate shard hashes;
- requires exactly eight continuation tokens per request on the phone route,
  tested CUDA route, and independent CUDA oracle;
- orders linked phone completion, publication, and CUDA readiness;
- requires distinct A, B, and PAIR phase IDs;
- authorizes a cycle only by reopening all three V2.2 roots.

`v24_readiness/CP0_R1_EVIDENCE_CONTRACT_V2_4.json` is the current A_ONLY
successor. It adds the canonical B8 token-history geometry, exact monolithic
CUDA identity, runtime bundle roots, producer process provenance, and the
source/stat/receipt closure for every acquisition stage. Its authority
reopens the raw V2.2-compatible predicates and the V2.4 orchestration record.
The offline prospective route is non-executable: it contains explicit
identity and network sentinels. After the mandatory phone reboot and phase
lock, a separate source-bound stage creates an immutable bound root using the
observed boot IDs and WiFi addresses. Fast readiness and every producer consume
that bound root. V2.4 cannot authorize B_ONLY, PAIR, a cycle, trace replay, or
energy work.

Run the gate in this order:

1. Finish the V2.4 concrete production plan and no-model preflight, then run
   one V2.4 `A_ONLY` phase for Qwen3 14B Q4_K_M.
2. Only after A passes, acquire and screen Qwen3 8B Q8_0 as the only new
   candidate under a separately frozen `B_ONLY` successor.
3. Stop the candidate search immediately if Qwen3 8B fails.
4. Only after A and B pass, freeze and run `PAIR` and derive measured
   non-co-residency on the target RTX 4060 Ti.
5. In that pair phase, qualify both local-UFS reprepare directions.
6. Only if the then-current authority re-evaluates the A, B, and PAIR bundle
   roots and emits
   `ONE_REDUCED_A_TO_B_TO_A_CYCLE_AUTHORIZED`, authorize one reduced cycle.

Each model independently must:

- fit and serve B8 on the bound RTX 4060 Ti with the frozen KV and serving
  envelope and at least 512 MiB free VRAM;
- execute one complete contiguous OP15 -> OP12 route with direct WiFi
  activations, valid placement, at least 512 MiB available memory per phone,
  zero process swap, and zero system swap growth;
- preserve exact history, position, ownership, and cleanup records;
- match an independent monolithic CUDA oracle that uses the same history and
  batch partition as the tested path;
- pass the prospectively frozen 64-item MMLU task noninferiority gate. Phone
  versus CUDA greedy-token agreement and cross-geometry agreement are recorded
  diagnostics, not pass criteria;
- publish useful B8 phone work before the corresponding target-GPU model-ready
  instant.

The pair must have a measured, non-shareable model-plus-KV lower bound that
exceeds the 16 GiB GPU after the frozen 512 MiB serving headroom. Both
Qwen3-14B -> Qwen3-8B and Qwen3-8B -> Qwen3-14B phone reprepares must release
all request state, read weights only from phone-local UFS, advance the
readiness generation, transfer zero weight bytes over USB or WiFi in the paid
interval, and finish within 30 seconds.

The current V2.2 contract check status is
`V2_2_EVIDENCE_READY_ACQUISITION_NOT_RUN`, and the historical
identity-only preflight status is `NO_MODEL_PREFLIGHT_PASS`. Neither is route
eligibility.
No Qwen2.5 phone acquisition, Qwen3 8B download, trace replay, controller
integration, model-switch cycle, or energy acquisition is authorized by this
setup work.

## W4 Qwen2.5 replacement screen, 2026-07-24

Qwen2.5 14B Q4_0 now has an executable two-phone route. The Qwen2 architecture
uses the same bounded partial-load graph, layer-filtered KV, StageV3 protocol,
direct phone-to-phone relay, and canonical row order as Qwen3. OP15 stores
`[0,32)` and OP12 stores `[32,48)`. The full and both shard artifacts are
content-addressed in `RESULTS_W4.md`.

One repeated-prompt B32 route is exact against a matched B32 CUDA control. It
executes eight B32 physical calls, moves 5,242,880 activation bytes directly
from OP15 to OP12, and moves zero activation bytes through the host. A B1
control generates different tokens at decisions seven and eight on both CUDA
and phones. This is batch-sensitive greedy output, not a phone-only mismatch;
all route correctness gates must therefore use a matched batch control.

The frozen 128-prompt corpus rejects Qwen2.5 Q4_0:

| Phone backend | First token | Token decisions | Exact sequences | Gate |
|---|---:|---:|---:|---|
| HTP0 fused attention | 97.66% | 89.65% | 75.00% | FAIL |
| HTP0 explicit attention | 96.88% | 90.04% | 77.34% | FAIL |
| GPUOpenCL | 95.31% | 85.94% | 70.31% | FAIL |

The frozen gates remain 95 percent first-token agreement, 95 percent token
decision agreement, and 80 percent exact-sequence agreement. No threshold is
changed after observing these rows. Qwen2.5 Q4_0 is not scheduler-eligible.

Qwen2.5 14B Q8_0 from pinned repository revision
`05244aa5d871c661c80082a15d3bce44714d068d` is 15,701,598,336 bytes. A
tail-only load optimization removes the unused 827,228,160-byte token
embedding from the `[30,48)` shard. OP15 stores `[0,30)` and OP12 stores
`[30,48)`. The route completed all 44 physical calls with clean placement and
direct OP15-to-OP12 activations, but the unchanged quality gate rejected it:

| Route | First token | Token decisions | Exact sequences | Gate |
|---|---:|---:|---:|---|
| Q8_0 GPUOpenCL cut 30 | 96.09% | 85.16% | 70.31% | FAIL |

Two fresh acquisitions produced identical physical and CUDA token hashes. The
independent validator derives the same failure from the persisted logs. Q8_0
is not scheduler-eligible, and its failure stops this replacement path.

## W5 batched phone-to-CUDA handoff, 2026-07-24

W5 separates handoff correctness from the failed Q8 task-quality decision. It
does not weaken or replace the W4 quality gate. The prospective
`W5_HANDOFF_CONTRACT.json` was frozen before acquisition with
`scope=MECHANICS_ONLY` and `scheduler_eligible_on_pass=false`.

One real B8 cohort ran on the Qwen2.5 14B Q8_0 route:

1. OP15 `[0,30)` and OP12 `[30,48)` generated four authoritative phone tokens
   per request.
2. CUDA reconstructed the exact prompt plus committed-phone-token history.
3. A control used two-token history chunks; catch-up used eight-token chunks.
4. CUDA generated eight speculative continuation tokens while the phone
   ownership epoch remained authoritative.
5. The coordinator removed all phone sequence state, committed one CUDA owner
   epoch, then removed CUDA state after the bounded continuation.

The catch-up continuation equals the same-history CUDA control for all eight
requests. Catch-up used two history batches versus six for the control and
took 408,641 us versus 555,645 us in this single run. This timing is
informational, not a repeated performance claim. State counts were 8 phone
sequences at the frozen frontier, 8 prepared CUDA sequences, and zero on both
routes after completion.

Both phone workers emitted `SCHEDULED_PLACEMENT_OK` with zero missing buffers;
both CUDA workers did the same. The phone direct relay executed 7 batches and
88 rows. The CUDA direct relay executed 22 batches and 304 rows. All saved
artifacts verify. The reportable provenance-hardened repeat is selected by
`W5_ACTIVE_RESULT.json` and is stored under
`results/w5_phone_cuda_handoff/run_20260724T193106Z/`.

This closes the isolated exact-history replay seam at B8. It does not prove:

- task quality for the Q8 phone route;
- CUDA loading concurrent with ongoing phone decode;
- a generated-token delta after the frozen frontier;
- N=1 or N=32 catch-up;
- unequal histories, model rewarming, or a reverse transition;
- scheduler eligibility, trace benefit, or energy savings.

## W6 concurrent phone delta, 2026-07-24

W6 keeps the phones authoritative for two more generated tokens while CUDA
reconstructs the four-token snapshot state. CUDA then ingests only the
two-token delta, commits a durable CUDA owner record, releases all phone
sequence state, and continues eight tokens.

The selected real B8 run passes all eight continuation comparisons, records
99.81 percent overlap of the shorter concurrent leg, and returns every state
counter to zero. CUDA snapshot replay took 174,719 us and was hidden behind
2,698,645 us of phone progress. After the phone frontier, delta ingestion plus
CUDA continuation took 333,057 us versus 522,081 us for full-history replay
plus continuation, a 36.21 percent single-run critical-path reduction.

The reportable rerun also passes a separately frozen physical gate. Its
validator consumes the realized session certificates, exact worker/shard
hashes, device boot identities, and executed step counts. Core phone compute
is on OpenCL and core server compute is on CUDA0.

This remains mechanics-only. CUDA weights were already resident, only fixed
B8 equal histories ran, and the independent Q8 task-quality gate still fails.
See `RESULTS_W6.md`.

## W7 process-cold CUDA promotion, 2026-07-24

W7 started a new B8 request on the collective-phone Qwen route before two
fresh CUDA model processes launched. The phones continued until CUDA became
ready, then executed the W6 replay, concurrent two-token delta, durable
cutover, and continuation path.

The prospective gate failed twice. The first run exposed that weight residency
did not imply prepared OpenCL kernels. R1 added an explicit two-token B8
preparation session ending in `DETACH`, then proved the paid request used the
same resident worker PID and boot nonce on each phone. R1 still failed:
process-cold CUDA became ready before the prepared phone route produced the
first token for a new prompt.

This rejects new-request prefill as a bridge for the measured A6000 warm-cache
load regime. It does not reject continuity for an already-live session whose
phone KV exists before promotion. W8 starts the paid interval at a decode
boundary and keeps the W7 load, ownership, placement, and matched-control
requirements. See `RESULTS_W7.md`.

## W8 live-session promotion, 2026-07-25

W8 begins with B8 phone KV and two committed tokens already live. The paid
clock starts at the next decode boundary, before two fresh CUDA model
processes launch.

The first W8 run delivered two phone tokens before CUDA readiness and completed
an exact same-frontier CUDA handoff, but failed its independent greedy-control
gate. Three primary cross-backend token decisions diverged and cascaded to 29
mismatches among 104 tokens.

R1 froze a separate teacher-forced trace control. The real R1 run passes
placement, ownership, exact trace replay, state cleanup, and source/artifact
binding. Phone service reduces promotion-trigger-to-next-token latency from
3.283 seconds to 1.357 seconds and delivers two decode rounds per request, 16
tokens in aggregate, before CUDA is ready. This is not new-request TTFT because
phone KV and two committed output tokens per request exist before the paid
clock. Full completion grows from 3.756 seconds to 7.256 seconds because a
fixed two-token-per-request post-frontier phone delta takes 2.762 seconds while
CUDA frontier replay takes only 0.173 seconds. The fresh CUDA control is
teacher-forced trace replay. It evaluates 104 decisions while feeding the first
96 post-start trace token IDs, and its predictions agree on 101 of 104
decisions. Agreement remains diagnostic.

This is a live-service mechanics signal, not an end-to-end latency or
scheduler pass. The run used process-cold CUDA workers with a warm host page
cache on an RTX A6000 with 48,530 MiB, not the target RTX 4060 Ti. It did not
drain a hot model, perform a model replacement, or enter one native continuous
CUDA server.

W9 must prospectively select the intentionally scheduled extra phone-batch
count from measured leg times. The measured rule selects `k_extra=0`. This does
not mean zero phone work after readiness: an optional in-flight phone batch may
finish, while CUDA must begin replaying the last committed frontier
immediately. See `RESULTS_W8.md`.

## W9 profile-driven zero-extra cutover - historical failed gate

Execution outcome, 2026-07-25: `P1.T` crossed the prospective paid marker and
failed the mandatory same-frontier CUDA continuation comparison. The
no-replacement rule stopped the campaign before `P1.C` and P2-P4. W9 therefore
fails and cannot issue a four-pair mechanics or performance certificate. The
immutable attempt and independent failure audit are in `RESULTS_W9.md`.

W9 closes only the profile-driven zero-extra handoff mechanism for the same
real B8 live session. It does not demonstrate adaptation to a second real
regime. It does not integrate the multi-model controller, repair the rejected
Qwen2.5 Q8 quality route, execute a model swap, run on the target GPU unless
the recorded CUDA0 actually resolves to it, or authorize trace and energy
experiments.

### Prospective policy contract

Freeze `W9_PROFILED_CUTOVER_CONTRACT.json` before any paid acquisition. Bind
the W8 source artifacts, exact binaries and scripts, model and shard digests,
device identities, token budget, sampler, timing gates, predictor inputs, and
the nonzero `cutover_margin_us=50000`.

Define:

- `F0`: the vector of last fully committed phone positions for the frozen B8
  cohort at the readiness linearization point;
- `F1`: the vector of final phone positions acknowledged after quiescence;
- `d_inflight`: the rectangular zero-or-one-token-per-request result in
  `F1 - F0` from an optional batch already in flight at CUDA readiness;
- `k_extra`: complete phone decode batches deliberately started after that
  optional batch. In this fixed B8 cohort, one batch adds one token per request;
- `d_actual`: the exact per-request delta in `F1 - F0`, including in-flight and
  extra work;
- `K_max`: the finite maximum permitted by the remaining 13-token output budget
  while reserving at least one autonomous CUDA continuation token.

The frozen decision is:

```text
K_max = max(
    0,
    13 - phone_tokens_at_F0 - max_inflight_tokens
       - min_cuda_continuation_tokens
)

phone_side_us(k) =
    predicted_inflight_remaining_us + predicted_phone_extra_us(k)

predicted_delta_tokens(k) = max_inflight_tokens + k

cutover_us(k) =
    max(predicted_cuda_replay_us, phone_side_us(k))
    + predicted_delta_ingest_us(predicted_delta_tokens(k))
    + predicted_commit_us

completion_us(k) =
    cutover_us(k) + predicted_cuda_remaining_us(k)

k_extra = max(
    {0} union
    {integer k in [1, K_max] where
        phone_side_us(k)
            <= predicted_cuda_replay_us - cutover_margin_us
        and completion_us(k) <= completion_us(0)}
)
```

Freeze `max_inflight_tokens=1`, `min_cuda_continuation_tokens=1`, the finite
candidate vector, and every cumulative predictor value as an integer number of
microseconds. Predictor vectors must be nonnegative and monotone and round
conservative estimates up. Bind each input either to a W8 raw record and field
or label it as a prospectively frozen conservative bound. At the snapshot,
`predicted_inflight_remaining_us` is zero when no batch exists; otherwise it is
the nonnegative frozen batch-duration estimate minus elapsed coordinator time.
Persist the batch start, estimate, elapsed time, and realized residual.

The current W8 profile must select `k_extra=0`. Compute and persist the
decision from the frozen inputs; do not hardcode zero in the execution path.
Synthetic unit profiles must exercise positive selections, but no real
`k_extra>0` acquisition or general production scheduler is part of W9.

### Treatment event order

`F0` and `F1` are per-request position vectors over one frozen cohort, not one
scalar. The phone owner uses epoch `e`; the prospective CUDA owner uses `e+1`.
One coordinator-local lock is the linearization authority. A rectangular phone
batch committed under that lock before the readiness snapshot belongs to
`F0`. A batch accepted after the snapshot belongs only to `F1 - F0`.
Cross-device timestamps are evidence but never decide ownership order. Record
both `cuda_ready_ns` and the later coordinator `f0_snapshot_ns`.

1. Reuse the W8 preparation path to establish a real OP15 and OP12 session.
   B8 phone KV and exactly two committed output tokens exist before the paid
   clock.
2. Start the paid interval at a decode boundary and launch fresh CUDA head and
   tail workers after the frozen launch offset. The phones remain the only
   output owner.
3. When both CUDA workers are ready, acquire the coordinator lock, freeze
   further phone submission, snapshot `F0`, and start CUDA history replay
   immediately. There is no StageV3 wire-level quiesce operation; quiescence
   means the coordinator cannot submit a next phone batch.
4. Finish zero or one phone batch already in flight. Since this profile selects
   `k_extra=0`, do not launch another phone decode batch.
5. If no batch was in flight, record disposition `ABSENT` and `d_inflight=0`.
   If one was submitted, require one all-or-none rectangular B8 result, publish
   it under phone epoch `e`, and record disposition `PUBLISHED` with
   `d_inflight=1`. Cancellation, partial completion, or silent discard is out
   of scope and fails W9.
6. Acknowledge `F1`, its per-request positions, exact token-history digests, and
   phone epoch `e`. At `F1`, phones cease dispatch and publication but retain
   their KV for rollback. With `k_extra=0`, `d_actual` must equal
   `d_inflight` and be either zero or one token per request.
7. CUDA finishes replay of `F0` while an optional phone batch is completing,
   then consumes exactly `F1 - F0`. Implement `d_actual=0` as an explicit no-op
   branch; the existing positive-width token-ingestion interface must not be
   called with an empty delta.
8. After CUDA reaches `F1`, fsync the CUDA owner record for epoch `e+1` and
   return its exact durable-completion timestamp. CUDA must not publish any
   token before that instant. Release phone sequence state only afterward.
9. Continue autonomous CUDA decoding and release all CUDA state. Increase CUDA
   continuation as needed so every request still has exactly 13 post-start
   tokens.

CUDA replay must start from `F0` without waiting for `F1`. When
`d_inflight=1`, require positive overlap between replay and the remaining
in-flight phone batch. When `d_inflight=0`, record the conditional overlap as
not applicable rather than inventing a zero-duration leg. At least one of the
four treatments must exercise and pass the real `d_inflight=1` branch.

On any failure before the durable epoch-`e+1` commit, discard CUDA state and
retain or resume the phone owner under epoch `e`. After the commit, epoch `e`
can never publish again, even if phone cleanup fails.

Persist separate per-request counts for preexisting tokens, pre-ready phone
tokens, phone tokens in `F0`, post-snapshot in-flight phone tokens, deliberately
scheduled extra phone tokens, CUDA continuation tokens, and total post-start
tokens. Preexisting tokens do not count toward the fixed 13-token post-start
budget. For every request:

```text
phone_tokens_at_F0 + d_actual + cuda_continuation_tokens = 13
d_actual = d_inflight + k_extra
cuda_continuation_tokens >= 1
```

### Repetitions and matched control

Run four prospective pairs in this dependency order:

```text
P1.T, P1.C, P2.T, P2.C, P3.T, P3.C, P4.T, P4.C
```

Each `Pi.C` is a fresh-process CUDA teacher-forced replay of the exact trace
from `Pi.T`; treatment must therefore precede its paired control. Do not
describe this as randomized AB/BA ordering. Preserve each pair in a unique run
directory. The prospective contract names exactly four pair ordinals.

A setup abort occurs only before the durable paid-start marker. Preserve its
artifacts and cause; the same ordinal may restart after the prerequisite is
fixed. Once the paid-start marker exists, that ordinal is consumed. Any
treatment or control failure then fails W9 and cannot be replaced or excluded.
A pass requires exactly four paid treatments and their four paired controls,
all valid and passing. Do not silently retry or overwrite any attempt.

All paid legs are process-cold and host-page-cache-warm. Before each leg,
sequentially read every byte of the exact CUDA model files outside the paid
clock, record the byte counts and file identities, and start no CUDA process.
Use the same preconditioning method for treatment and control and never drop
host caches during the series. The intended
request-to-CUDA-launch delay is 100,000 us for both treatment and control. Each
leg must remain under the existing 250,000 us bound and the paired delay
difference must be at most 20,000 us. Record phone thermal state before and
after every treatment. Require complete same-boot captures and at most 3,000
millicelsius difference between the coolest and hottest treatment-start
`gpu_max_millic` on each phone. This is a new W9 range gate that reuses W3's
numeric limit, not W3's group-mean calculation.
Prove the preceding leg's CUDA processes and sequence state are absent before
starting the next leg.

Before each leg, bind the resolved CUDA device name, UUID, total VRAM, PCI bus
ID, driver, power limit, host boot ID, `CUDA_DEVICE_ORDER`,
`CUDA_VISIBLE_DEVICES`, and the physical-index/UUID/logical-CUDA0 mapping.
Prefer UUID binding and set `CUDA_DEVICE_ORDER=PCI_BUS_ID`. Save the raw query,
prelaunch and ready-time process lists, and memory inventory.

Before every launch, require five idle samples over at least one second with
zero target-GPU utilization and no compute process. Persist temperature,
P-state, SM and memory clocks, power, and memory for the pre- and post-leg
brackets; cool down until the next pre-leg bracket passes. The paired
treatment/control `cuda_ready_us` values are comparable only when their
absolute difference is at most the greater of 100,000 us and five percent of
the larger value. A device mismatch or unexplained competing process found
before the paid marker is a setup abort; after that marker it is a paid
failure. W8's CUDA0 was an RTX A6000; no result may relabel it as a 4060 Ti.

### Prospective pass criteria

Every treatment/control pair must pass all mechanics requirements:

- at least one phone token per request is published with coordinator
  `published_ns < cuda_ready_ns`;
- the frozen scheduler inputs reproduce `k_extra=0`;
- each treatment publishes exactly 13 post-start tokens per request and each
  control evaluates the same 13 post-start token positions;
- `F0` and `F1` are rectangular B8 vectors ordered by the coordinator
  linearization rule, with `d_actual` exactly zero or one;
- replay of `F0` and ingestion of the realized `F1 - F0` are token-exact;
- all autonomous post-cutover CUDA tokens match a same-frontier autonomous
  CUDA continuation control for all eight requests; oracle work is outside the
  paid treatment path and cannot delay publication;
- the paired fresh CUDA teacher-forced control replays the pre-trigger history,
  evaluates all 13 treatment trace positions per request, and feeds exactly
  the first 12 known post-start tokens per request to advance between
  decisions. Thus it makes 104 predictions and feeds 96 post-start token IDs;
  the frozen B8 geometry is 80 pre-trigger plus 96 post-start CUDA rows in 17
  calls. Greedy agreement is reported only as a diagnostic;
- there are zero duplicate, missing, stale, or conflicting tokens;
- the publication ledger and ownership journal reach `COMPLETE`, publication
  ownership changes once from epoch `e` to `e+1`, and CUDA publishes nothing
  before the exact durable ownership-commit instant;
- all phone and CUDA placement certificates pass, source and artifact hashes
  match, every terminal sequence-state count is zero, and the GPU
  identity/idle/readiness-comparability gates pass.

Compute the four individual paired ratios without intermediate rounding. Sort
the four exact rational values and define the even-sample median as the
arithmetic mean of the middle two. Both prospective performance gates must
pass:

```text
median(treatment_promotion_next_token_us
       / control_promotion_next_token_us) <= 0.75

median(treatment_completion_us
       / control_completion_us) <= 1.25
```

The approximately 4.5 second W9 completion is a hypothesis, not a pass
criterion. Do not discard a valid slower observation.

### Required measurements and validation

Use the following metric names and literal boundaries:

- `promotion_next_token_us`: promotion trigger to the first post-trigger
  published token for the already-live session;
- `cuda_ready_us`: promotion trigger to both fresh CUDA workers ready;
- `handoff_gap_us`: last phone publication to first CUDA publication;
- `completion_us`: promotion trigger to all 13 post-start tokens complete;
- `new_request_ttft_us`: reserved and unmeasured in W9.

For the teacher-forced control, the next-token analog ends when CUDA has
replayed only the prompt plus two preexisting output tokens and produced the
prediction for the first post-trigger trace position. The first known
post-trigger token is teacher-forced only in the following evaluation. Control
completion ends at the thirteenth prediction after the first 12 known
post-trigger tokens have been fed. These are matched decision boundaries, not
generated-token or new-request-TTFT claims.

For each request, maximum paid-interval inter-token gap is the maximum of
`promotion_next_token_us` and every interval between consecutive post-trigger
publication timestamps, including the phone-to-CUDA boundary. Do not substitute
the idle time between physical batch calls.

Also persist every token publication timestamp, the maximum paid-interval
inter-token gap, request-to-launch delays, CUDA load/readiness, `F0` replay,
`f0_snapshot_ns`, `cuda_replay_start_ns`, in-flight disposition and wait,
`d_actual`, delta ingestion or zero-delta no-op, the exact durable ownership
commit, CUDA continuation, full completion, useful pre-ready phone tokens,
phone batch durations, phone thermal brackets, and GPU idle/thermal/clock
brackets.

Extend existing W8/W6 code paths minimally, but do not mutate W6's frozen
fixed-delta contract or evidence. Add a W9-specific schema with
`max_inflight_tokens=1`, runtime `d_actual in {0,1}`, and dynamic CUDA
continuation. Replace W8's completed-batch readiness polling only in W9 with a
bounded coordinator: submit at most one phone batch in a worker/future, observe
CUDA readiness concurrently, freeze further submissions under the
linearization lock, launch `F0` replay, then join and publish the optional
rectangular batch. Reuse replay, positive delta ingestion, continuation,
placement, and cleanup primitives rather than cloning them.

The current six-record journal is insufficient for W9 causal evidence. Extend
it or add one minimal hash-chained, fsynced W9 publication ledger containing
each request, position, token, owner epoch, coordinator `published_ns`, `F0`,
`F1`, readiness, replay start, catch-up, durable `CUDA_COMMITTED`,
`PHONE_RELEASED`, and `COMPLETE` events. The commit API must return the instant
after the `CUDA_COMMITTED` record is durable and before phone removal begins.
Validators must rebuild ownership and publication order from this ledger.

Add one four-pair launcher using absolute monotonic 100 ms launch deadlines for
both legs. It must precheck that every output path is absent, retain every
failed attempt, and emit an aggregate failure manifest rather than overwrite
or select runs.

Before device acquisition, add focused tests for:

- current-profile selection of zero, exact policy boundary and margin,
  hypothetical faster-phone selection above zero, `K_max=0`, flat predictors
  over a finite domain, no feasible positive value, and deterministic
  decisions;
- zero actual delta, one in-flight delta, rejection of an unexpected extra
  batch, partial-B8 failure, budget exhaustion, positive CUDA continuation,
  and conservation of the 13-token equal-output budget;
- race-order determinism at the `F0` lock, conditional overlap for
  `d_inflight=0`, and positive overlap for `d_inflight=1`;
- failure status that cannot produce a pass certificate;
- CUDA identity or mapping mismatch, an unexpected prelaunch CUDA process,
  non-idle GPU, readiness mismatch, launch-deadline mismatch, and missing
  page-cache evidence;
- causal timing: paid phone work starts after the paid clock, treatment CUDA
  replay starts after readiness, control prefill starts after readiness, and
  CUDA publication starts after ownership commit;
- strict pre-ready classification where equality is not early, and a control
  next-token timestamp at pre-trigger-history completion before the first
  teacher-forced post-trigger feed;
- corrupt digest, stale epoch, duplicate or missing token, partial ledger,
  ledger tamper, precommit CUDA publication, old-epoch postcommit publication,
  precommit failure that preserves phone epoch `e`, source mutation, placement
  failure, terminal-state leak, and manifest mutation;
- four-value median, pre-clock abort versus paid failure, no paid replacement,
  and failure of any one of the exact four pairs.

Run the focused W9 tests, the full S39 host test suite, and `bash -n` on changed
shell launchers. Independently regenerate the aggregate certificate from raw
records and require byte-identical output. Verify the final manifest with
`sha256sum -c`. The validator must derive the decision, token conservation,
timing ratios, placement, ownership, and terminal state without trusting the
launcher summary.

### W9 stop and claim boundary

Fail closed if any paid pair fails, a treatment has no pre-ready phone token,
the four runs never exercise the real in-flight branch, the policy selects or
executes an unexpected extra batch, either performance bound fails, work
changes, publication order or rollback is invalid, or any exactness, placement,
provenance, device, launch, page-cache, thermal, or terminal-state check fails.
Do not weaken the frozen rule or threshold after acquisition.

A pass may claim only repeated profile-driven zero-extra live-session cutover
mechanics for the recorded OP15, OP12, model artifact, and resolved CUDA
device. It does not establish a real adaptive `k_extra>0` regime. Keep
`scope=MECHANICS_ONLY` and `scheduler_eligible=false`; make no new-request TTFT,
task-quality, multi-model capacity, target-4060-Ti, or energy claim.

The prospective text originally made a reduced forward/reverse cycle
conditional on a W9 pass. W9 did not pass. The current successor is CP0-R1
two-route eligibility above. Controller integration, a model-switch cycle, the
full trace, and energy remain blocked.

## CP0-R1 - Qualify exactly two routes

- [x] Freeze one model-independent eligibility contract.
- [x] Bind Qwen3 14B plus exactly one Qwen3 8B Q8_0 candidate attempt.
- [x] Freeze the v2 raw-evidence successor without changing any v1 artifact.
- [x] Close summary-only admission with required role tags, secure read-once
      hashing, and independent derivations.
- [x] Reverify the exact RTX 4060 Ti, OP15, and OP12 in the historical V2.2
      no-model preflight.
- [x] Freeze V2.1 as the phased exit authority without changing V1 or V2.
- [x] Support A-only, B-after-A, and pair-after-A/B raw evidence bundles.
- [x] Bind every phase lock, readiness probe, and acquired event to its own
      host-clock interval.
- [x] Close corpus/output, CUDA memory, oracle length and B8 geometry, bridge,
      exact transfer-size, and full-shard local-UFS admission gaps.
- [x] Exact-check every readiness argv and require a fresh full readiness pass
      immediately before each acquisition.
- [x] Freeze V2.2 as the sole exit authority without changing V1, V2, or
      V2.1.
- [x] Freeze and exact-check the canonical pinned-revision 64-row MMLU corpus
      and the 25/64 CUDA sanity floor.
- [x] Exact-bind incumbent A's candidate route and require eight continuation
      tokens per request on all three paths.
- [x] Enforce completion-to-publication-to-readiness order and distinct phase
      IDs.
- [x] Require final cycle authorization to reopen all three V2.2 bundle roots;
      reject legacy status-only results.
- [x] Pass the V2.4 no-model three-device identity and network preflight.
- [x] Finish the versioned physical-USB runtime launcher and prospective
      A_ONLY adapter.
- [ ] Finish the long artifact receipt, post-reboot fresh receipt, and
      concrete production plan.
- [ ] Run and pass one V2.4 A_ONLY qualification for Qwen3 14B.
- [ ] Finish every per-route gate for Qwen3 14B; its current status is only
      `PROVISIONAL_BATCH`.
- [ ] Materialize and hash Qwen3 8B Q8_0, then freeze its one route cut and
      both phone shard identities before paid acquisition.
- [ ] Run every per-route gate for Qwen3 8B. Stop candidate search on any
      failure.
- [ ] Prove each model independently serves B8 on the RTX 4060 Ti with the
      frozen KV and headroom.
- [ ] Prove the pair cannot coexist with the same KV and headroom.
- [ ] Record direct activation transfer, placement, positive phone memory,
      zero swap growth, exact state mechanics, and the path-matched CUDA oracle
      for both routes.
- [ ] Run the frozen task-quality noninferiority gate for both routes.
- [ ] Prove useful phone publication precedes CUDA readiness for each model.
- [ ] Prove both local-UFS reprepare directions within 30 seconds.

Exit: V2.4 is the sole current authority for A_ONLY and cannot authorize a
cycle. After A_ONLY passes, B_ONLY and PAIR require separately frozen
successors. A reduced cycle may run only when the then-current successor
reopens the A_ONLY, B_ONLY, and PAIR bundle roots and emits
`ONE_REDUCED_A_TO_B_TO_A_CYCLE_AUTHORIZED`. V1, V2, V2.1, V2.2, V2.3, and
legacy status-only results cannot authorize that cycle. Storage-only,
provisional, incorrect, cross-geometry-only, or partial-server evidence
remains ineligible.

## CP1 - Measure the transition budget

- [ ] Measure desktop model unload and load from warm host cache.
- [ ] Measure desktop model load from cold NVMe.
- [ ] Measure phone ready-to-first-token at `N={1,8,32}`.
- [ ] Measure phone continuous decode rate and useful batch candidates.
- [x] Isolate sorted versus shuffled sequence order with matched workers,
      batch shapes, and thermal brackets.
- [ ] After the reduced bidirectional cycle passes, add bounded downstream
      credits and overlap OP15 batch `k+1` with OP12 batch `k`.
- [ ] Measure CUDA batched prefill of the exact S39 histories.
- [ ] Measure phone rewarm from local UFS after releasing the other model.

Primary gate:

```text
live_phone_next_token_time < desktop_model_ready_time
```

New-request phone TTFT is a separate measurement. If phone prefill loses to
desktop readiness, do not claim a new-request bridge. If the live-session gate
also fails, retain only the low-rate phone-residency hypothesis: phones must
avoid a GPU switch rather than bridge it.

The W0 cohort screen is not this gate. It records end-to-end service time but
does not separate first-token readiness, repeated-process variance, or
continuous-arrival throughput.

## CP2 - One-request catch-up

- [x] Add a durable versioned ownership record with request, model, route,
      owner, epoch,
      committed position, and token-history digest.
- [ ] Keep an already-live phone route decoding while CUDA loads.
- [x] Snapshot one committed frontier in an isolated real B8 gate.
- [x] Reconstruct native CUDA KV from prompt plus committed token IDs.
- [x] Consume a fixed two-token phone delta without publishing it twice.
- [x] Durably commit one CUDA owner before releasing the fixed B8 phone state.
- [ ] Recover correctly from a process crash at every partial journal phase.
- [ ] Continue at least 32 CUDA tokens.

Use greedy decoding. Direct KV import is out of scope. Any duplicate, missing,
stale, or conflicting token fails the gate.

## CP3 - Batched catch-up

- [ ] Run `N={1,8,32}` live phone requests.
- [x] Batch CUDA history reconstruction at N=8.
- [x] Keep phones as the publication owner during the N=8 reconstruction.
- [x] Catch up the fixed B8 delta and switch the group at a token boundary.
- [ ] Admit unequal prompt and output lengths.
- [x] Prove exact conservation and sequence-slot cleanup for the fixed N=8
      cohort.

Exit: one CUDA batch absorbs independently progressing phone requests, and a
useful `N` beats sequential reconstruction.

## CP4 - Rewarm the displaced model

- [ ] Drain and release all phone state for promoted model B.
- [ ] Load and prepare model A shards from phone-local UFS.
- [ ] Publish a new boot/residency/prepared-image readiness generation.
- [ ] Reject stale B readiness and incomplete A readiness.
- [ ] Execute a reverse A promotion and catch-up.

Exit:

```text
A hot / B warm
-> B hot / A unavailable
-> B hot / A warm
-> A hot / B warm
```

No step may claim both models executable in phone RAM unless physical
measurement proves dual residency.

## CP5 - Trace replay

- [ ] Execute all 74 successful requests, including 57 Gemma and 17 Qwen.
- [ ] Execute at least one request for each model on phones and on CUDA.
- [ ] Complete all nine alternating residency intents.
- [ ] Run C0-C3 and C5 over identical request identities and token budgets;
      report C4 as a performance bound.
- [ ] Evaluate only the three frozen demand regimes: sparse alternate-model
      demand, one sustained alternate-model burst, and oscillating A/B demand.
- [ ] Predeclare a finite promotion and hysteresis sweep.
- [ ] Preserve model-A steady-state SLO while model B is warm.
- [ ] Record every queue, load, phone batch, replay batch, delta, ownership
      transition, and rewarm event.
- [ ] Run failure injections for phone disconnect, CUDA load failure, stale
      epoch, and rewarm failure.

Report:

- switch-period SLO-valid admitted capacity, SLO goodput, and new-request TTFT;
- promotion-trigger-to-next-token latency for already-live sessions;
- common promotion-period response gap, from request arrival or the previous
  token to the next publication, for every control;
- route-specific last-phone-to-first-CUDA handoff gap;
- useful phone tokens during loading;
- model unload/load, replay, catch-up, and rewarm times;
- phone and CUDA batch distributions;
- duplicate/missing/stale token counts;
- bytes by payload class;
- number of avoided GPU switches;
- selected-GPU energy only after mechanics pass.

## Decision

S39 passes the mechanism gate only if C2 or C3:

- conserves every request and token exactly once;
- has zero stale ownership or false readiness;
- improves switch-period SLO goodput or SLO-valid admitted capacity by at least
  20 percent over the best of C0, C1, and C5;
- keeps P95 new-request TTFT and the common promotion-period response gap at or
  below 110 percent of the best of C0, C1, and C5;
- does not regress the steady-state hot-model SLO;
- completes phone rewarming before the measured reverse-switch opportunity.

Phone, host, network, and total-system energy remain unknown. A latency pass
does not imply an energy pass.
