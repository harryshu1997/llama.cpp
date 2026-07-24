# Project Log - Active Warm-Tier Multi-Model Serving

> Running progress record. **Current status** is at the top and kept up to date.
> The **log** below is newest-first; every entry is timestamped.
> Goal: keep one GPU-nonresident model executable across OP12 and OP15, serve it
> during a GPU model change, batch-reconstruct live request state on CUDA, and
> rotate the displaced model back into the phone warm tier.

---

## Current status - `2026-07-24 EDT`

**DIRECTION RESET TO ACTIVE WARM-TIER MULTI-MODEL SERVING.** One desktop GPU
holds one hot large model while OP15 and OP12 collectively hold one other
executable warm model. Phones serve the warm model while the GPU drains and
loads its local checkpoint. CUDA then batch-prefills prompt and committed token
histories, catches the small phone token delta, and takes ownership at an exact
token boundary. Phones subsequently prepare the displaced GPU model for the
reverse switch. Direct KV migration and runtime checkpoint transfer are not the
first path.

**S39 W2 DIRECT MIXED-PHASE BATCH MECHANICS PASS; OVERLAP AND BENEFIT OPEN.**
The direct Qwen route now admits new prefill while older requests retain live
decode KV. One real OP15-to-OP12 run formed a 96-row `llama_decode` containing
16 decode rows first and 80 prefill rows after them, then continued all 32
streams in B32 decode calls. All 256 generated-token checks matched CUDA.
Across nine calls the relay sent 7,864,320 activation bytes directly from OP15
to OP12 and zero activation bytes through the host. Both worker lifecycle and
placement certificates pass the independent reducer. Maximum completion was
23.219 s, but no latency gain is claimed because W1's row ordering and device
state differed. The relay is still synchronous and cannot overlap OP15 batch
`k+1` with OP12 batch `k`. See
`spikes/s39_phone_model_switch_trace/RESULTS_W2.md`.

**S39 W1 DIRECT PHONE CHAIN MECHANICS PASS; PERFORMANCE AND ROUTE READINESS
OPEN.** An additive relay on OP15 now keeps host-side admission, batching, route
epochs, and result ownership while sending Qwen cut activations directly from
OP15 `[0,30)` to OP12 `[30,40)` over WiFi. The corrected relay binary completed
persistent B1 and B32 sessions with all 264 token checks exact. B32 formed one
160-row prefill batch plus seven 32-row decode batches, moved 7,864,320
activation bytes OP15-to-OP12, and moved zero activation bytes through the
host. Both workers preserved PID/nonce across detach and stop with clean
placement. The direct B1 point was only 0.76% faster than one separately loaded
host-relay control, within observed run variation; B32 took 90.796 s. Status is
`DIRECT_CHAIN_MECHANICS_PASS_REPEATS_PENDING`, not a latency or energy pass.
Interface counters and a host-issued cryptographic reservation descriptor are
still absent, so Qwen remains provisional. See
`spikes/s39_phone_model_switch_trace/RESULTS_W1.md`.

**S39 TRACE REPLAY READY; QWEN BATCH PROVISIONAL; W0 TWO-MODEL GATE BLOCKED.**
Dense Qwen3 partial loading, layer-bounded graph execution, and layer-filtered
KV run across OP15 `[0,30)` and OP12 `[30,40)`. Persistent workers completed
B1, B8, and B32 cohorts with all 328 generated-token checks matching the
same-artifact CUDA sequence. B32 executed one 160-row prefill and seven 32-row
decode calls. Request throughput improved only 1.40x over B1, so useful
continuous service is not yet proven. USB provisioned the weights before the
run; 15.00 MiB of B32 activation payload traversed the two WiFi relay legs.
Because endpoint/interface evidence was not captured by the runner, Qwen is
`PROVISIONAL_BATCH`, not `PASS`. Gemma Q4 still differs from CUDA at token zero.
The hash-bound controller refuses both routes with `E_ROUTE_NOT_READY`. No
switch, handoff, SLO, latency, or energy benefit is claimed. See
`spikes/s39_phone_model_switch_trace/RESULTS_W0.md`.

**S38 F16 LONG-CONTEXT MIXED MECHANICS PASS; MATCHED RAG ROUTE BLOCKED.** The
S36 runtime now releases long prompts in 64-token quanta and stops on declared
EOG tokens. On OP15, two real S38 prompts of 1,507 and 2,532 tokens produced six
physical B64 HTP calls containing one live decode row plus 63 prefill rows.
All 16 bounded token decisions match a split-CUDA F16 control, placement is
`SCHEDULED_PLACEMENT_OK`, and selected-CUDA compute time falls 23.01%. This is
not yet an SLO or energy win: the 500 ms mechanics gather window makes request
latency worse, F16 does not match the frozen Q8_0 C0 artifact, and raw StageNet
argmax does not reproduce llama-server's reasoning-budget sampler. Positive
reasoning-budget work now falls back before phone dispatch. Matched Q8 C2/C3,
answer quality, and energy remain blocked. See
`spikes/s38_distributed_rag_trace/RESULTS.md`.

**S38 MATCHED SERVER-ONLY RAG CONTROL PASS; MATCHED PHONE CONTROLS NOT RUN.** A frozen
320-request cohort proportionally covers every RAG question/evidence stratum
and replays 151.55 s of offered arrivals. One A6000 and one RTX 4060 Ti each ran
the full all-local BGE embed -> 7,008-chunk retrieval -> BGE rerank -> Gemma-4
12B Q8_0 generation DAG using identical trace, index, payload, and model hashes.
Both meet the requested run bound: 10.20 and 17.70 min. Throughput is 0.523 and
0.301 req/s, versus 2.112 req/s offered; corrected response p95 is 443.86 and
869.22 s because queueing is real. Answer EM is 55.94% on both. Generation is
over 99.7% of median service time. The matched validator passes all 320 request
identities and timing equations. No SLO, phone, or energy benefit is claimed;
next certify the phone reranker and run C1 distributed retrieval. See
`spikes/s38_distributed_rag_trace/`.

**S38 MIXED-GENERATION ADAPTER READY; MATCHED PHONE ROUTE BLOCKED.** The S38
generation seam now uses llama-server's exact chat template and tokenizer,
then admits eligible work to the existing S36 `DynamicRouteRunner`. Prefill
and decode rows therefore enter the same cut-homogeneous continuous batcher,
with a request-pinned layer cut through decode. Admission binds the live route
device, cut, GGUF type, context, stream count, and row capacity and rejects
missing correctness, placement, latency, shape, identity, or SLO evidence.
The frozen Q8_0 RAG control still selects the server for all requests: its real
prompts are 1,507 to 2,532 tokens, the prior mixed phone proof used four-token
F16 prompts and an eight-token context, and the measured same-Q8_0 HTP route
failed quality. The auxiliary F16 mechanics point above expands the context
envelope but does not repair that Q8_0 quality gate. Nine adapter tests, three
subset tests, six baseline tests, and all 41 S36 tests pass. Next produce a
matched-quality Q8_0 phone route and compatible sampler before C2/C3; do not
copy the Qwen-only non-SWA S33 wavefront into Gemma.

**S36/S37 REAL CONTINUOUS-BATCH AND ARBITRARY-CUT MECHANICS PASS; CUDA RELIEF
FAILS.** Both phones now run B32 with llama.cpp unified KV: the previous OP12
469 MiB independent-stream allocation becomes a 14 MiB, 256-cell shared cache.
On the frozen 60-request mixed-priority trace, each of three treatment runs
used both phones and cuts 4/8, preserved all tokens and SLOs, drained all state,
and produced a real OP12 HTP batch mixing three decode rows with 32 prefill
rows in one `llama_decode`. The follow-up physical sweep ran every jointly
resident cut 4, 5, 6, 7, and 8 twice on each phone with identical tokens.
However, selected-CUDA stage relief reproduced in only one of three pairs;
median treatment was 1.86% slower. The cut-specific tails fragmented CUDA work.
Next implement the already scoped cut lift to canonical cut 8 and one shared
tail; do not claim energy savings from S36/S37. See
`spikes/s36_dynamic_cut_scheduler/RESULTS.md` and
`spikes/s37_arbitrary_layer_exit/RESULTS.md`.

**S35 MIXED PHONE BATCH AND DYNAMIC LAYER HANDOFF PASS.** One physical B5
`llama_decode` call on each phone mixed one decode-shaped row from an existing
sequence with four prompt rows from a newly admitted sequence. Against a
same-worker serial oracle, maximum relative L2 was 4.03e-7 on OP15 and 7.00e-7
on OP12; both placement certificates passed with zero missing compute buffers.
The experimental StageNet path now also selects any active Gemma-4 interval
inside already resident weights. A sequence pins its interval, batches group
only equal intervals, and graph reuse includes the interval. Both phones
switched `[0,1)` -> `[0,2)` -> `[0,1)` without a weight reload and rejected a
live-sequence cut mutation. Finally, one resident OP12 `[0,8)` worker and one
resident A6000 `[4,48)` tail completed full-model routes with cuts 4 and 8;
both produced tokens `[236761,236744,236761]`. This is a mechanics result, not
an energy, SLO-policy, throughput, or semantic early-termination claim. See
`spikes/s35_mixed_prefill_dynamic_cut/RESULTS.md`.

**S34 SAME-QUANTIZATION ADMISSION PASS; NUMERICAL REPAIR STILL OPEN.** The
StageNet worker now has an opt-in model-identity capability that reports the
GGUF `general.file_type` and launcher-verified SHA-256. The S31 physical route
requires one exact Q8_0 identity across every CUDA and phone stage before it
creates batchers. Its launch and evidence paths also reject unequal hashes.
The new protocol was built for CUDA and Android and queried on an RTX 4060 Ti,
OP12, and OP15: all reported file type 7 and Q8_0 SHA-256
`7b56cbd0...3d492848`. The desktop's older same-type but different-digest Q8
file was rejected, and the exact 12.67 GB artifact is now installed there.
Future S31 campaigns use the full shared Q8_0 GGUF on both phones instead of
the historical F16 shards. This closes mixed-quantization admission, not S33's
separate HTP-versus-CUDA numerical divergence; quantized routes remain outside
the eligible scheduler until that kernel-quality gate passes.

**S33 FULL QUANTIZED CAPACITY PASS; QUALITY FAIL.** Full Q4_0 and Q8_0 files
are stored on both phones, but neither full 48-layer graph passes the zero-swap
execution gate. Partial Q4_0 residency reaches OP12 `[0,12)` and OP15
`[4,24)` at B32 with zero process swap, creating an eight-layer overlap.
However, the frozen 128-prompt, eight-output same-GGUF quality gate fails:
Q4_0 reaches 80.5% first-token, 61.6% token-decision, and 43.0% exact-sequence
agreement; Q8_0 reaches 82.0%, 70.4%, and 58.6%. Both routes are also 13-20x
slower per B32 cohort than the one-A6000 reference. The independent binder
installs zero scheduler rows. Keep F16 phone execution until the quantized HTP
kernel path is repaired. Evidence is under `spikes/s33_full_quantized_routes/`.

**S32 QUANTIZED CAPACITY GAIN; EXACT ROUTE FAIL.** Q8_0 expanded the largest
zero-swap tested B32 windows to OP12 `[0,4)` and OP15 `[4,16)`, with median
stage times 762.1 and 222.8 ms. Larger windows used process swap or aborted in
DSP execution. Q8_0 and Q4_0 both failed the frozen same-GGUF CPU-versus-HTP
relative-L2 gate (Q8_0 1.15-1.36%; Q4_0 1.15-1.93%). A real same-Q8 route
`OP12 [0,4) -> OP15 [4,16) -> A6000 [16,48)` matched a CUDA reference for
32/32 cloned BOS requests, but only 18/32 distinct-input four-token requests
(101/128 token decisions). The fail-closed binder therefore produced no
eligible case. Q8/Q4 capacity is measured, but quantized phone routes stay out
of the exact scheduler. Evidence is under
`spikes/s32_quantized_overlap_residency/`.

**S31 LATENCY-BALANCED CUT PASS; ENERGY AND QUALITY OPEN.** A finite real-device
cut sweep replaced S29's fixed OP12 `[0,6)` -> OP15 `[6,8)` partition with the
measured winner OP12 `[0,1)` -> OP15 `[1,8)`, keeping the CUDA tail at
`[8,48)`. At B32, the phone-stage p95 bottleneck fell 1.499 -> 0.399 s and the
balance ratio rose 0.178 -> 0.771. A fresh matched 60-request run completed
R0=28/R2=32 with zero synthetic SLO misses and B32 on both phones. Selected
CUDA compute fell 7.166 -> 5.095 s (-28.90%), P0 p95 fell 1.175 -> 0.643 s,
and treatment makespan improved 29.63% versus S29, although it remains 1.93x
the all-CUDA control. Only 21/60 token sequences match across F16-phone and
Q8-CUDA routes; phone/network/total energy remain unknown. Evidence is under
`spikes/s31_latency_balanced_cut/`.

**S29 REAL B32 PRIORITY TRACE PASS; THROUGHPUT, ENERGY, AND QUALITY OPEN.**
The RTX 4060 Ti, OP12, and OP15 completed the frozen 60-request trace with 32
resident slots per worker. R2 is OP12 `[0,6)` -> OP15 `[6,8)` -> CUDA `[8,48)`;
R0 uses matching CUDA cuts and the same tail. Fresh B1/B4/B24/B32 calibration
preceded the run. Control completed R0=60; treatment completed R0=28/R2=32.
Both phones executed B32 for all four decode steps, all 60 requests completed,
and no synthetic SLO was missed. A one-second online P0-quiet guard preserved
priority: P0 p95 improved 1.386 -> 0.636 s. Summed CUDA compute fell 6.668 ->
5.609 s (-15.88%), while makespan rose 4.505 -> 12.138 s (2.69x). Therefore
this is a server-work/priority mechanics pass, not a throughput win. Phone,
network, and total energy remain unknown; F16-phone/Q8-CUDA tokens match only
6/60, so numeric quality is uncertified. Evidence is under
`spikes/s29_large_batch_trace/`.

**S28 REAL PRIORITY-SAFE SHARED TAIL PASS; ENERGY AND NUMERIC QUALITY OPEN.**
One RTX 4060 Ti tail queue served urgent R0 and background R2 rows from OP12
HTP0 `[0,8)` and OP15 HTP0 `[8,16)` on the frozen 60-request dense mechanics
trace. Control completed R0=60; treatment completed R0=10/R2=50. Both had zero
synthetic SLO misses, P0 p95 changed 860,246 -> 850,127 us, and summed CUDA
island compute fell 5,766,619 -> 4,697,047 us (-18.55%). OP12/OP15 mean batch
was 3.846/4. The tail interleaved R0/R2 seven times and never mixed P0 with
background work. The cost is a 5.007 -> 41.977 s makespan increase from waiting
toward latest-safe start. F16-phone/Q8-server tokens are uncertified (10/60
matched); no energy boundary was measured. Evidence is under
`spikes/s28_priority_shared_tail/`.

**S26 PRIORITY-SAFE PHYSICAL SCHEDULER PASS; ENERGY NOT MEASURED.** A matched
real-device run used one coordinator and lockstep B4 execution for both the
all-CUDA control and treatment. Control ran three R0 B4 groups on the RTX 4060
Ti. Treatment kept the four P0 requests on R0 B4 and sent the P1/P2 groups over
OP12 HTP0 `[0,8)` -> OP15 HTP0 `[8,16)` -> CUDA `[16,48)`. All 12 requests met
their 2/8/15 s SLOs. P0 p95 was preserved (observed 341,517 -> 241,203 us) and
summed CUDA island compute fell 770,196 -> 579,537 us (-24.75%). Low-priority
slack paid the cost: makespan rose 0.789 -> 4.301 s. Every active stage used B4;
placement, lineage, KV drain, worker persistence, route-point token oracles,
and three fail-closed mutations passed. This is selected-CUDA-compute evidence,
not GPU-board or total-system energy. Evidence is under
`spikes/s26_priority_scheduler/`.

**S25 REAL CONTINUOUS REQUEST LIFECYCLE PASS.** The actual OP12 HTP0 `[0,8)`,
OP15 HTP0 `[8,16)`, and RTX 4060 Ti CUDA0 `[16,48)` workers executed unequal
request lengths with changing physical memberships `AB, AB, CB, CB, CD, D`.
C reused A's sequence slot while B remained live; D reused B's slot while C
remained live. Every stage observed B2 until the final B1 row, all four dynamic
greedy sequences matched same-route B1, every placement certificate passed,
and all workers drained to zero and stopped. Verdict:
`THREE_DEVICE_CONTINUOUS_LIFECYCLE_PASS`. This is real runtime mechanics, not
simulation, semantic early termination, or an energy/throughput claim. Evidence
is under `spikes/s25_continuous_lifecycle/`.

**S24 FIXED-DIAMOND MECHANICS PASS; BENEFIT GATE FAIL.** A real RTX 4060 Ti,
OP12, and OP15 execution completed the frozen R0/R1/R2 diamond. CP4 passed all
11 physical sessions: OP12 and OP15 selected B4 knees, R2 reached B4 across
both phones and the CUDA tail, OP15 formed mixed R1/R2 batches, and the CUDA
tail formed mixed R0/R1/R2 batches. Placement, lineage, finite outputs, KV
cleanup, leases, and same-route repeatability passed. Phone-F16 versus
desktop-Q8 boundaries remain numerically uncertified even though this
synthetic screen produced equal greedy tokens.

CP5 then rejected the system claim on 12 real equal-work control runs. Shared
convergence improved OP15 mean batch from 2.0 to 3.0 and reduced median
makespan 34.3%, but priority-0 TTFT and latency regressed 30.0% and 32.5%.
The SLO router increased summed CUDA island compute 48.6%, introduced two SLO
misses, and raised median selected-GPU board energy from 35.18 J to 163.26 J.
Final verdict: `BENEFIT_GATE_FAIL`. CP6 is stopped; phone, network, A6000 host,
and total-system energy remain unknown. Evidence is under
`spikes/s24_overlap_handoff_poc/results/{cp4_fixed_diamond,cp5_controls}/`.

**S22 CP7 BATCH-SHAPE ORACLE REPAIRED; PHONE QUALITY STILL OPEN.** The prior
chunked-prefill token mismatch is not by itself an exactness failure. A new
same-process boundary probe found byte-identical sequential repeats. Chunked
versus sequential layer-8 activations have maximum relative L2 0.219% on CPU
F16 and 0.307% on A6000 F16, both below the existing 0.5% gate; 4060 Ti Q8 is
5.317% and fails. More importantly, the unsplit full-F16 A6000 model itself
changes the frozen prompt's last greedy token between sequential and chunk-4
execution while both sequential repeats match. Exact token agreement across
batch shapes is therefore not a valid standalone oracle. The mixed route stays
numerically uncertified until both phone HTP boundary rows and a real-prompt
output-quality set pass. The terminal V3 worker now also supports a full
`[0,48)` server-control route. Python tests are 42/42; CPU, CUDA, and Android
builds pass.

**S22 MIXED-SLO ROUTER MECHANICS PASS; NUMERIC QUALITY OPEN.** A finite
profile-driven router selected CUDA `[0,8)` for two 1.4 s requests, OP15
`[0,8)` for two 2.2 s requests, and OP12 `[0,8)` for two 3.5 s requests. All
six ran concurrently into one 4060 Ti `[8,48)` tail and met SLO: actual maxima
were 1.219 s, 1.299 s, and 2.228 s. Slack-bounded gather raised tail mean batch
from 1.04 to 2.40 and max B2 to B4 without a global barrier. Placement passed
on all lanes. The CUDA route is explicitly numerically uncertified: its last
token differs from the phone routes and needs a mechanics-only override.
Four-token prompt batching reaches B8 on both phones and the tail, but the
same-route OP15 sequential versus chunk-4 control produces different tokens;
chunked prefill is mechanics-pass but not yet quality-certified. The Python
suite at this checkpoint was 30/30. No energy or arbitrary-cut claim.

**S22 CP3 REAL THREE-DEVICE ASYNC BATCH MECHANICS PASS.**
Current-source Gemma-4-12B executed on OP15 HTP0 `[0,8)`, OP12 HTP0 `[0,8)`,
and one shared RTX 4060 Ti CUDA0 `[8,48)` tail. Four live requests per phone
formed B4 locally for four decode steps; the shared tail formed eight B4
batches at the two independent arrival cadences without a global phone
barrier. All eight requests completed, met the configured 30 s mechanics gate,
and returned the same four token IDs. Placement is certified on all three
devices. This proves resident distributed KV, per-sequence admission/removal,
bounded asynchronous fan-in, and device-local ready-row rebatching. It does not
yet prove multi-token prefill, a desktop-only control route, mixed-SLO route
selection, arbitrary early exits, or energy savings. OP12 also fails closed at
32 resident 512-token slots because its HTP KV allocation is too large; the
successful run uses eight slots. Evidence is under
`spikes/s22_slo_overlap_pipeline/`.

**Authoritative direction: FOCUSED Q-PIM FUNNEL.** The paper-critical system has
one generation model, one foreground embedding service, one selected A6000, and
two phones. OP12 executes Gemma `[0,6)` and a CUDA `[6,8)` bridge normalizes its
activation; OP15 executes `[0,8)` directly. Both enter one continuously batched
CUDA `[8,48)` tail. The core mechanism is stateful cut lifting plus batch
morphing: each phone uses an independent measured batch and cadence, the bridge
advances OP12 rows to the canonical cut, and the tail reforms compatible ready
rows into a new batch with exact row lineage and distributed per-layer KV
ownership. Credit- and slack-bounded fan-in prevents a slow phone from creating
a global barrier but is policy, not a standalone contribution. Ordinary
continuous batching and dynamic rebatching are reused ideas, not the novelty. The slow
external desktop link and second A6000 are outside the paper-critical path.
Dynamic weight replacement,
general DAG scheduling, R2 phone-to-phone execution, solver optimality, and
DVFS are future work, not current gates.

**S20 SERVER-ONLY REAL-TRACE TIMELINE PASS.** Replayed the frozen 32-request
BurstGPT cohort through current-source Gemma-4-12B F16 `llama-server` on one
A6000 with continuous batching. The cohort preserves 21 arrivals at relative
0 s, 11 at 1 s, 21,203 observed input tokens, and 1,898 observed output tokens;
only token values are synthetic because BurstGPT publishes no text. Nsight
GA10x metrics sampled at 1 kHz show 8.1 s prefill-only (mean tensor active
44.1%, DRAM pressure 42.0%), 3.0 s mixed prefill+decode, then 11.7 s decode-only
(mean tensor active 6.8%, DRAM 76.0%, peak 89.3%). All 32 exact token-count
responses completed; GPU1 stayed at 0% utilization over 328 control samples.
Verdict `SERVER_TRACE_TIMELINE_PASS`; phase-derived model-level classification,
not a per-kernel roofline certificate. Graph and harness are in
`spikes/s20_server_trace_roofline/`; raw evidence is in scratchpad. The first
acquisition was rejected for a `/slots.next_token` response-shape parser error;
the repaired full rerun alone enters the result.

**S19 CONTINUOUS-BATCH REUSE AUDIT COMPLETE; RUNTIME NOT YET CLAIMED.** The
llama-server implementation confirms the correct lifecycle: one slot per
sequence, a shared logical `(seq_id, token, position)` batch rebuilt on every
update, compatible prompt admission beside active decode, and per-sequence KV
removal. LayerSplit already has variable-row decode but still globally resets
fixed cohorts and cannot prefill an arbitrary free slot. S19 therefore borrows
the server mechanics, not its HTTP/task stack: a versioned row-manifest prefill,
per-sequence remove, worker shadow slot table, and distributed host slot loop
are the next code gate. The audit and physical CP1-CP6 order are frozen in
`spikes/s19_dynamic_batch_runtime/`. The audit also rejects the prior assumption
that one native tail batch can mix layer-6 and layer-8 entry activations. The
focused system normalizes OP12 rows with a CUDA `[6,8)` bridge, then coalesces
them with OP15 layer-8 rows in one `[8,48)` tail.

**S19 BATCH ATLAS + VARIABLE-COHORT DISPATCHER MECHANICS PASS (sibling spike).**
A separate effort from the concurrent continuous-batching audit above; its files
live in `spikes/s19_batch_atlas_dispatch/` because `s19_dynamic_batch_runtime/`
was already held by that audit (left untouched). Measured the real per-device
batch service atlas over B={4,8,16,24,32,48,64} using the existing persistent
LayerSplit drivers (no C++ change). Eligible operating points: CUDA_R0 all seven
(knee B64, 869 tok/s), OP15 `[0,8)` {4,8,24,32,48} (knee B48, 123 tok/s), OP12
`[0,6)` {4,8,24,48,64} (knee B48, 30 tok/s). The frozen same-batch CUDA
correctness gate is load-bearing: OP15 B16/B64 and OP12 B16/B32 are INELIGIBLE.
OP15 B16/B64 diverge from the same-batch CUDA greedy path (near-tie argmax,
batched-GEMM accumulation order; OP12 B64 by contrast MATCHES CUDA B64, so the
divergence is per-(device,batch)), and OP12 B16/B32 hit reproducible v75 HTP
exchange hangs. A fail-closed online dispatcher never selects an unmeasured
batch, reserves downstream CUDA-tail + phone + USB credit before any phone
launch, splits oversized queues into measured microbatches, forces deadline
release, and conserves every request. 9/9 unit/adversarial tests pass, the
decision log is byte-identical across PYTHONHASHSEED, and an independent
validator re-derives conservation, credit, and no-unmeasured-batch. A real-device
proof executed the selected cohorts on hardware (varying CUDA batch B8/B16/B32, a
concurrent OP15+OP12 B8 launch with distinct worker PIDs and 12.4 s wall overlap,
an R0 fallback) - all exact tokens, HTP0/CUDA0 placement OK. Verdict
`S19_CP1_BATCH_ATLAS_MEASURED` + `S19_CP2_VARIABLE_COHORT_BATCHING_MECHANICS_PASS`.
One static cohort per persistent exchange, NOT continuous batching; no energy of
any kind; two CUDA tail images still resident (S18 carryover, peak ~45 GiB). No
commit.

**S18 REAL THREE-DEVICE R1 MECHANICS PASS; RELIEF IS INSUFFICIENT.** Six matched
rows ran on one selected A6000, OP15, and OP12. Every row completed 339,440 BGE
encodes plus 384 Gemma requests and 3,072 exact tokens. OP15 and OP12 owned
independent B32 groups, overlapped for the two OP12-credit rounds, and met their
5 s and 12 s classes. Median BGE p95 is effectively unchanged: 3,875 us for P0
and 3,877 us for P4.

The selected-GPU result is negative. Raw savings are 0.075-0.610 percent, with
a 0.342 percent median and a -764.1 J uncertainty-adjusted lower bound. Two
route-specific CUDA tail processes raise peak selected-GPU memory from 26,555
to 45,674 MiB. Verdict:
`S18_R1_FLEET_MECHANICS_PASS_RELIEF_INSUFFICIENT`. The focused next physical
gate uses a CUDA `[6,8)` bridge for OP12 and one `[8,48)` tail shared by OP12,
OP15, and server-head rows. It must remove the duplicated common suffix without
serializing either SLO class.

B32/B64 are experiment points, not scheduler constants. The next runtime must
continuously admit and retire decode sequences at token boundaries on every
phone and the server tail. It selects only measured batch candidates, jointly
constrained by device memory/compute/thermal state, earliest SLO, transfer and
activation credits, and every later stage's batch capacity. Small batches may
run for an imminent SLO, but B1/B2 is not a target phone operating point.

S17 R2 remains stopped and does not enter S18. Its OP15 B64 middle island is
1.50-1.53x faster than 2xB32, but its HTP0/CUDA0 residual exceeds the frozen
5e-3 numerical gate and one of 64 row argmaxes differs.

Skipped GPU-us receives no energy credit. The baseline is optimized server-only
continuous batching over the same arrivals and SLOs. The focused interim gate
requires at least 10 percent lower selected-GPU J/work or 10 percent more
SLO-valid work at the same selected-GPU power boundary. This remains a
GPU_BOARD result, not a total-system claim.
Mirrored phone weights can save compute but not HBM; exclusive ownership can
release HBM but has no hidden immediate server copy. Per-GEMM network splitting
and token-prefix KV ownership are excluded.

**S16 REAL MIXED PERSISTENT GATE IS COMPLETE.** Six rotated rows ran on the
selected A6000 and OP15. Every row completed identical work: 339,440
high-priority BGE encodes at B16 plus twenty low-priority Gemma B32 x 8-token
cohorts. The full-model CUDA control and OP15 `[0,8)` plus CUDA `[8,48)`
treatment kept weights, contexts, and PIDs resident across all twenty cohorts.
All 3,840 request results are token-exact, placement is certified, and every
low cohort fits the synthetic 5 s SLO. Median BGE p95 improves from 4,112 us to
3,983 us (0.969x).

The selected-GPU energy gate does not pass. All three pairs have positive raw
savings (0.68-1.30 percent), but the median is only 0.75 percent and is below
the Ampere NVML 5 W uncertainty floor. The independent validator reopens all
six power traces, 120 low results, 120 host placements, 60 phone sessions, and
BGE outputs and reproduces `S16_FAIL_GATE`. The honest verdict is
`MIXED_PERSISTENT_MECHANICS_PASS_GPU_BOARD_RELIEF_UNRESOLVED`. The next bounded
gate is an independent persistent OP12 `[0,6)` B32 lane, followed by a
two-phone equal-work test. Do not serially chain the phones.

The OP12 prerequisite now passes. Two persistent B32 x 8 exchanges over OP12
`[0,6)` plus CUDA `[6,48)` take 9.409 s and 9.144 s, return 64/64 exact
requests, preserve one host and phone PID across DETACH/STOP, and pass
independent raw replay. OP12 therefore enters only a synthetic 12 s
lower-priority class; it remains ineligible for OP15's 5 s class. The next code
checkpoint is the two-independent-lane coordinator, not another device profile.

**S15 TYPED PERSISTENT B32 DISPATCH PASSED THE PREVIOUS LIVE CHECKPOINT.** A frozen
32-request cohort from observed BurstGPT arrivals now drives the real
typed `ExecutionRequest`, persistent transport, live evidence adapter, OP15
`[0,8)` HTP head, and selected-A6000 `[8,48)` tail. One OP15 PID/nonce and one
A6000 PID survive DETACH and are reused for a second STOP exchange. Both B32
exchanges return exact CUDA-reference tokens and 32 admitted boundary
certificates. Child elapsed times are 3.208313 s and 2.792293 s. The phone uses
HTP0 except the declared GET_ROWS CPU seam; all 9288 tail compute nodes per
exchange use CUDA0. An independent validator reopens the typed replies, bound
artifacts, raw mux streams, process identities, and placement certificates.
Arrival timestamps are observed; payload, priority, and SLO are synthetic.
Energy remains UNKNOWN.

The phone-worker persistence path now passes the same B32 route over seven real
sessions: one OP15 PID/nonce, DETACH x6 plus STOP, 224/224 exact requests, HTP0
compute with CPU only GET_ROWS, 3.676122 s conservative prompt-to-host-exit, and
3.020 percent CoV. This is a distinct epoch-14 profile with frozen binaries and
libraries. That seven-session gate reloads the host tail before each prompt.
The server-residency gate adds a JSONL command loop to one A6000 tail process.
Its direct C++ run and the typed run now both pass. The typed gate closes the
previous mechanics gap: success is admitted only after the bridge binds exact
tokens, phone SESSIONCERT, host PLACEMENTCERT, epochs, persistent PIDs, and all
request boundary certificates.

Weight streaming remains a later slow-loop feature; the current worker still
lacks per-slot residency generations needed for safe compute/stream overlap.

### 2026-07-19 EDT - S18 real two-phone R1 passes mechanics, not relief

- Ran six matched real rows on one selected A6000, OP15, and OP12. Each row has
  339,440 BGE encodes plus six 64-request Gemma rounds at eight tokens/request.
- All request tokens, placement, session/reset, route-credit, overlap, SLO, and
  BGE isolation gates pass independent raw replay. OP12 is capped at its two
  measured exchanges; OP15 serves later loose-SLO groups.
- Median BGE p95 is 3,875 us P0 versus 3,877 us P4. OP15 route median is 2.875 s;
  OP12 is 9.412 s. Tight and loose class maxima are 3.140 s and 9.703 s.
- Selected-GPU raw saving is only 0.342 percent median and below uncertainty.
  Duplicate CUDA tails raise peak HBM from 26,555 to 45,674 MiB. The next gate
  is a single multi-ingress superset tail, not another duplicated-tail run.

### 2026-07-19 EDT - S17 B64 middle coalescing is faster; numerical gate fails

- Froze the seven-checkpoint hierarchical activation-coalescing plan. R2 is an
  optional fan-in route beside direct R0/R1; it never waits past latest-start.
- Materialized Gemma `[6,12)` (2,741,158,560 bytes, sha256 e4fdd28f...) and ran
  real CUDA0/OP15 HTP0 B64 versus 2xB32 StageNet controls.
- HTP B64 is bit-identical to HTP 2xB32 and 1.50-1.53x faster. Real input came
  from measured OP15 and OP12 B32 `[0,6)` outputs; all compute placement passes.
- AUTO and explicit attention both exceed the unchanged 5e-3 HTP/CUDA residual
  gate (9.001e-3 and 8.886e-3). CP2 is stopped; no pipeline or energy claim.

### 2026-07-19 EDT - S16 real mixed OP15 mechanics pass; energy unresolved

- Added a persistent full-model B32 CUDA control to `llama-layersplit`, reusing
  the existing monobatch path and clearing KV before every exchange. CPU and
  CUDA builds pass; the two-exchange reset/PID/placement regression makes the
  live input suite 10/10.
- Ran a real short P0/P2 screen and the frozen six-row acquisition. Each full
  row carries 339,440 BGE encodes and 5,120 Gemma tokens; all tokens, placement,
  persistence, thermal, overlap, and absolute SLO gates pass.
- High-priority BGE p95 P2/P0 is 0.969. P2 Gemma B32 p95 is about 2.89 s versus
  0.62 s for P0, but remains below the frozen 5 s low-priority SLO.
- Raw selected-GPU savings are 1.30, 0.68, and 0.75 percent. Each trace has
  174-176 real power changes and sub-146 ms maximum gaps, but the paired 5 W
  uncertainty-adjusted lower bounds are negative. No energy saving is claimed.
- Independent raw-artifact replay passes. Phone, USB, server-wall, and total
  system energy remain UNKNOWN. See `spikes/s16_mixed_persistent_energy/`.

### 2026-07-19 EDT - OP12 persistent B32 independent lane passes

- Ran the frozen B32 cohort twice through OP12 `[0,6)` HTP0 plus A6000
  `[6,48)`: DETACH then STOP, one phone PID/nonce and one host PID.
- Route walls are 9.409 s and 9.144 s. Both pass the predeclared 12 s
  lower-priority SLO; OP12 is explicitly not admitted to the 5 s class.
- All 64 results are exact. Phone placement is HTP0 plus GET_ROWS on CPU; the
  v75 path correctly uses explicit attention rather than broken fused FA.
- Independent replay passes. Verdict: `OP12_PERSISTENT_B32_LANE_PASS`.

### 2026-07-18 EDT - S15 typed persistent OP15 B32 route passes on real devices

- Added canonical persistent readiness/results to `llama-layersplit`, plus a
  bounded physical mux that owns one real OP15 StageNet worker and one real
  selected-A6000 tail process.
- Ran the frozen 32-request BurstGPT cohort twice through the typed executor:
  launch 1 DETACH, launch 2 STOP. Host PID 3815676 and worker PID 21616 remain
  unchanged; session ids are 1,2 and cumulative phone steps are 384,768.
- Both launches are token-exact for all 64 request completions. Phone placement
  is HTP0 plus declared GET_ROWS on CPU; each host tail certificate records
  9288 CUDA0 compute nodes and no fallback. Maximum child elapsed time is
  3.208313 s; HMX thermal rises from 32.9 C to 37.2 C.
- The independent validator and 11 local mux/evidence tests pass. Exact local
  executable digests, remote runtime hashes, model/shard hashes, and raw streams
  are retained under `spikes/s15_persistent_typed_gate/`.
- Verdict: `TYPED_PERSISTENT_OP15_B32_PHYSICAL_PASS_ENERGY_UNKNOWN`. This is a
  production-path mechanics proof, not a mixed-workload energy result. Next is
  repeated one-A6000 BGE+Gemma control/treatment acquisition; OP12 remains an
  independent request lane for the following checkpoint.

**ONE-GPU LIVE RELIEF PASSES; LOW-PRIORITY SLO FAILS. - 2026-07-18.** The
independent OP15 `[0,8)` B1 route passes 7 processes, 56/56 exact requests,
scheduled HTP placement, thermal gates, and 0.0135 process CoV. In the live
three-repeat mixed run, each P0/P2 cohort completes 50,928 high-priority BGE
encodes and 64 low-priority Gemma tokens on one selected A6000 plus live OP15.
P2 reduces selected-GPU board energy 5.43 percent and preserves BGE p95
(1.0003x), but Gemma p95 rises 2.639x and fails the frozen 2.0x gate. Therefore
the verdict is `LIVE_OP15_FAIL_GATE`, not a scheduler or total-energy PASS.
Phone, USB, host-wall, and total-system energy remain unknown.

The serial OP15 -> OP12 route is rejected as the default: its completed B32 P2
cohort used 1.293x selected-GPU energy and had 13.83x Gemma p95, then the next
P2 failed before readiness. OP12 must next run a separate READY island or
independent request stream. The older B4/B8/B32 measurements remain ineligible
and cannot be inferred from B1. S15 now adds one independently measured,
post-load exact B32 point under a distinct profile and route epoch; B4 and B8
remain ineligible. The runtime requires a digest-bound `CertifiedBatchPoint`
for every exact batch.

**OP12 INDEPENDENT B1 ROUTE PASSES AT SIX LAYERS; EIGHT LAYERS IS REJECTED. -
2026-07-18.** A new fail-closed OP12/v75 wrapper binds exact workload settings,
binary/source/shard identities, process intervals, thermals, and raw phone/host
logs. OP12 `[0,6)` passes 7/7 independent processes and 56/56 exact requests:
median route wall 1.151677 s, median phone stage 0.874626 s, process CoV 0.02582,
13,392 HTP0 nodes plus 72 declared CPU GET_ROWS nodes, zero missing buffers, and
38.4-39.9 C end thermals. The adapter independently replays ROUTEJSON and
PLACEMENTCERT instead of trusting stored PASS booleans. A single `[0,8)` screen
passed, but its predeclared repeatability cohort timed out on process 3 after
two passes, so `[0,8)` is INELIGIBLE. The next physical step is independent
OP15 `[0,8)` and OP12 `[0,6)` request lanes feeding one shared A6000 tail
context; a serial phone chain and two duplicated tail processes remain rejected.

**TWO PARALLEL PHONE HEADS -> ONE SHARED CUDA TAIL: EXACT B1 POINT PASSES, B2
FAILS, PERSISTENCE BLOCKS REPEATABILITY. - 2026-07-18.** `layersplit.cpp` now has
an opt-in `--parallel-heads` driver: OP15 and OP12 each run `[0,6)` B1 on
independent threads, and one CUDA `[6,48)` context consumes both returned
activations. This is the first real implementation with both phones concurrent
and one server-tail weight copy. Native tail B2 is mechanically clean but
repeatedly token-incorrect versus the full B2 oracle, so it is rejected. Exact
tail B1 in the same shared context passes both streams and every placement and
thermal gate at about 1.34 s for two requests. After several phone model
unload/reload cycles, a later audit timed out in first compute. Verdict:
`SHARED_TAIL_POINT_PASS_PERSISTENT_WORKER_BLOCKED`; no scheduler or energy PASS.
Next: versioned DETACH/reconnect sessions that retain weights and context, then
7/7 host sessions before any selected-GPU energy acquisition.

The measurement-independent S14 fast-policy core is present in
`spikes/s14_mixed_streaming_scheduler/power_frontier_policy.py`: strict priority
ordering, compatibility-key isolation, SLO-bounded batch release for measured
memory/compute profiles, four-gate phone-boundary admission, and one-selected-GPU
P0-P3 selection. The S14 scheduler/catalog suite is 36/36 and the energy and
evidence suite is 32/32. The physical harness uses the certified profile but is
not yet driven by `PriorityBatchRuntime`, so integrated trace dispatch remains
open.

**S14 ENERGY STAGE A: A6000 GPU-BOARD ENERGY SAVING MEASURED (real hardware). - 2026-07-17.**
First real energy number in the whole program. On 1x RTX A6000 via NVML, running
Gemma-4-12B batched decode: offloading the head `[0,k)` to a phone (A6000 runs
only tail `[k,48)`, partial load) saves GPU-board energy per token, at matched
work (24000 tok/run, 3 rotated repeats, CoV <0.1%):

```text
 A6000 runs        mJ/tok   power  util   HBM        GPU energy saved
 full  [0,48)      859.9    297W   99%    22713 MiB  baseline
 tail  [6,48)      763.1    297W   99%    20114 MiB  11.3%  (offload 12.5% layers)
 tail  [12,48)     663.5    297W  100%    17515 MiB  22.8%  (offload 25% layers, HBM -5.2GB)
```

Power is FLAT (297W, near the 300W cap) and util stays ~99-100%: the saving is
purely FEWER GPU-seconds (fewer layers), not throttling; saved ~= 0.9 x layer
fraction (0.9 = fixed tail overhead lm_head+norm+embd). HONEST scope: selected-
A6000 GPU_BOARD only; phone/USB/total-wall UNKNOWN; assumes overlap (saturated,
gap_ms=0) so idle-wait energy is NOT modelled (that was the S11-E0 +53% failure);
this is the CEILING under perfect overlap, a diagnostic not a total-system claim.
Next: Stage B (phone can hold+run [0,12) ~7.4GB at a real latency + placement
cert -> realisable) then Stage C (fold into the mixed CP1 runtime at relaxed SLO).
Frozen in `spikes/s14_mixed_streaming_scheduler/energy/` (RESULTS.md + SHA256SUMS).
No commit; phones untouched (A6000-only measurement).

**S14-CP0c CANDIDATE ISLAND CATALOG: FROZEN (mechanics, no energy). - 2026-07-17.**
The finite candidate geometry is frozen before any scheduler runs.
`island_catalog.json` (`catalog_hash` sha256:3cf13792) instantiates the
predeclared `ATLAS_MATRIX` row contract as a schema-validated, digest-bound,
fail-closed artifact: 2 models / 4 islands / 5 rows. No fresh device run; it
binds existing evidence (S11 gemma [0,2) stage latency/memory/boundary, Gate-1
BGE cosine 0.9973 + no-fallback) and computes boundary bytes structurally. An
independent validator re-derives every content-address (descriptor/graph/weight/
catalog hashes), cross-refs, boundary bounds, a fail-closed PASS predicate, and
on-disk digests; 22 tests green.

```text
                         verdict       eligibility
 gemma_head_0_2  OP15/HTP0 LOWER_BOUND  INELIGIBLE (fallback unknown)
 gemma_layer_2_3 OP12/HTP0 LOWER_BOUND  INELIGIBLE (fallback unknown)
 gemma_head_0_3  OP15/HTP0 UNKNOWN      INELIGIBLE_UNMEASURED (declared)
 bge_encoder_0_12 OP15/HTP0 LOWER_BOUND INELIGIBLE_NO_LATENCY
 bge_encoder_0_12 OP12/HTP0 LOWER_BOUND INELIGIBLE_NO_LATENCY
```

A 5-lens / 13-agent adversarial review flipped the headline. The first draft
called `gemma_head_0_2` ELIGIBLE_COARSE on `fallback=none`; the review confirmed
that predicate was UNBOUND (neither cited artifact certifies HTP placement, and
the batched sweep explicitly disclaims it; exact tokens do not prove HTP
execution) and that the row stamped a binary that produced neither of its
numbers. Both fixed: gemma rows are now `fallback=unknown`, per-role binaries are
pinned, and the real E0 route-feasibility placement cert is bound at catalog
level. HONEST result: ZERO dispatch-eligible rows. The single measurement that
would change that -- one coherent gemma run emitting latency AND a same-run
PLACEMENTCERT on one binary -- is named as the top CP1 gap. No commit; no energy.
See `spikes/s14_mixed_streaming_scheduler/CATALOG.md`.

Follow-up audit hardened the validator before CP1: dispatch eligibility now also
requires `post_transfer_slo_feasible=true`, and duplicate JSON keys plus NaN/
Infinity constants are rejected at load time. The catalog suite is 22/22 and all
SHA256SUMS entries verify; the five-row eligibility result remains unchanged.

**S14 DESIGN FREEZE - 2026-07-17.** The implementation order is now explicit:
mixed trace composition and a two-service measured island catalog; static mixed
residency through the S12-V2 reducer and S13 fleet; exact discrete placement and
replication policy; then per-slot double-buffered weight streaming. USB H2P is
for large weights, WiFi H2P for small commands/input, and USB P2H results
preempt bulk weights. The measured ADB staging rates are not protocol READY
rates; the full verified path is about 36-37 MiB/s in the tested windowed rows.
No runtime, capacity, or energy claim is made by this design entry.

**S14-CP1 OFFLINE SINGLE-ISLAND PREVIEW: RELIEF INSUFFICIENT - 2026-07-17.**
The adapter and S12-V2 replay are deterministic and conserve all 177 terminal
outcomes, including timed-out rows. The only timed phone row is Gemma `[0,2)` on
OP15, so RAG requests reuse that same island; BGE has no latency row, C1 is not a
distinct control, OP12 is unused, and no S13 device dispatch occurs. The replay
therefore is not the planned two-service/two-phone CP1 gate. It shows 0 HBM
relief, 0.1 percent server-compute relief, doubled p50 latency, and a 0.667
useful-completion ratio at the synthetic den=50 contention point. The preview
supports the deeper-island measurement gap; it does not support an energy claim.

**S14-CP0a REAL mix-v1 (BurstGPT + RAGPulse): COMPOSED + REPLAYED. - 2026-07-17.**
The pinned raw sources were staged and the real headline mix produced. BurstGPT_3.csv
(231682327 B, sha256 2299986a) + RAGPulse 0_trace.jsonl (1923473 B, cd371571)
fetched + byte-verified; normalized with the committed configs (BurstGPT
5,344,021 records over 9874 15-min bins; median bin 22706 = 165 rows; RAGPulse
median = 12 rows). Committed `configs/mix-v1.config.json` (median lanes, scale 1/1,
offset 0/0) drives `compose_mix` -> real `mix-v1.jsonl` = 177 rows, byte-identical
on rerun (output_sha256 567d4af1, run_id mix-0dcd4054). `replay_mix` PASS: order
preserved, total demand input 112911 / output 21595 / retrieved_chunks 60, services
api_generation 152 + conversation_generation 13 + rag_qa 12, real component hashes
bound. This retires the S8 `MIX_COMPOSITION_NOT_IMPLEMENTED` gap on real data. Run +
SHA256SUMS in `scratchpad/s8_mix_v1/`. No scheduler/device/energy claim.

**S14-CP0a MIX-V1 COMPOSITION + STRUCTURAL REPLAY: MECHANICS PASS
(device-independent). - 2026-07-17.** The frozen `mix-v1` transform
(NORMALIZATION_SPEC section 7) is now implemented and hash-bound. `compose_mix.py`
superposes two already-normalized real components into one `semi_synthetic`
trace using the committed integer time map `t_mix = floor(t*num/den)+offset`,
merge key `(t_mix, rank, source_row_id)`, provenance rewrite, and
`mix:<rank>:<source>:<row>` event IDs; it verifies each component against the
committed `input_output_sha256`/`input_manifest_sha256`, publishes atomically, and
is byte-identical on rerun (same `output_sha256` + `run_id`). `structural_replay.py`
gains a mixed path (`replay_mix`) that validates schema, canonical bytes, the full
hash chain, per-stream component bindings, arrival order, service-to-DAG mapping,
and demand accounting; the old `E_MIX_UNSUPPORTED` guard is retired. New schemas
`mix_config` + `structural_replay_mix_result` validate under jsonschema 4.10.3 and
ajv 5. The frozen single-source `normalize_trace.py` is untouched (24 tests) and
the single-source replay is unchanged (32 tests). New: 15 composer + 11 mixed-
replay tests, +7 schema fixtures (59 total). This is Gate-A structural mechanics
on synthesized components; the headline real BurstGPT+RAGPulse `mix-v1` still needs
the pinned raw sources staged, and no scheduler/device/energy claim is made.

**S14-CP0b GATE-1 FUSED-GRAPH CLOSURE: `GATE1_HTP_BGE_FUSED_PASS`. - 2026-07-17.**
The gold-standard follow-up ran end to end. Fetched BAAI bge-small-en-v1.5
(BertModel 384/12L/12H/ff1536), converted `--outtype f16` -> 67MB gguf (sha256
4cd429b8). `test-export-graph-ops` emitted the exact 27-op BGE graph;
`test-backend-ops support|test --test-file` on HTP0 both phones: every op
supported (contiguous LayerNorm included, no crash - the isolated non-contig crash
never occurs in the real graph), 23/25 op cases OK, with GET_ROWS->CPU (f16 embd
table = declared exception) and GELU tripping the 1e-7 harness threshold at 1.8e-4
(approximate HTP kernel, immaterial). END-TO-END: `llama-embedding --device HTP0
-ngl 99 --pooling cls` on device, pooled-CLS cosine vs CPU reference = 0.997328
(OP15/v81 fa=auto), 0.997331 (v81 fa=off), 0.99733 (OP12/v75 fa=auto), 0.997328
(v75 fa=off) - all ~0.9973 on BOTH phones, both attention modes; HTP engagement
confirmed by timing (1244ms HTP vs 179ms CPU-only). BGE-shape FLASH_ATTN_EXT is
correct even on v75 (shape-specific, unlike Gemma-shape FA). VERDICT: a
phone-resident BGE embedding island is a REAL second executable service class;
S14 now has two phone-resident classes (Gemma head + BGE embed). Non-blocking
follow-ups: F32 embeddings to keep get_rows on HTP; approx-GELU note; energy/
latency unmeasured. Detail: `GATE1_HTP_BERT_OPS.md` section 7; logs
`scratchpad/gate1_bert_ops/`. Phones restored.

**S14-CP0b GATE-1 (HTP BERT-op support+correctness): PASS on BOTH phones,
pending fused-graph. - 2026-07-17.** On-device `test-backend-ops support|test` for
the BGE/BERT op set on OP15/v81 (build-snapdragon) and OP12/v75 (npu-harness),
`support`==`ggml_backend_supports_op` (no-fallback gate), `test`==CPU-ref
correctness. Result: every mandatory BERT op is supported with NO CPU fallback and
numerically correct vs CPU on contiguous F32 inputs, on BOTH devices - LayerNorm
(NORM), L2_NORM, GELU, non-causal SOFT_MAX, GET_ROWS, SCALE, MUL, RMS_NORM,
MUL_MAT; zero numerical FAILs. One localized defect: NORM and L2_NORM hard-crash
(`dspqueue_read 0x2e` in `flush_pending`) ONLY on the `noncontig_rows=1` variant on
both v81 and v75 - a `supports_op` FAIL-OPEN (unary predicate allows non-contiguous
src0) that a well-formed BGE graph never hits (BGE norm is contiguous); flagged to
fix defensively. Confirmed mitigable fallbacks: f16 get_rows -> use F32 embeddings;
non-32-aligned softmax -> pad seq to 32. Verdict
`GATE1_HTP_BERT_OPS_PASS_PENDING_FUSED_GRAPH`: a phone-resident BGE island is
op-level feasible on both phones; NOT `TRACE_OR_SERVICE_BLOCKED`. Still open: this
is isolated single-op (not fused), and no end-to-end BGE model was run - gold
standard is convert bge-small-en-v1.5 (F32 embeddings) -> test-export-graph-ops ->
`--test-file` on both phones + pooled-CLS cosine vs HF/CPU. Phones restored (temp
dirs removed, PIM workers were not running). Raw logs: `scratchpad/gate1_bert_ops/`,
detail in `spikes/s8_operator_island_affinity/GATE1_HTP_BERT_OPS.md`.

**S14-CP0b SECOND SERVICE CLASS: NOT `TRACE_OR_SERVICE_BLOCKED`; PHONE ISLAND
UNPROVEN. - 2026-07-17.** Feasibility audit of the BGE embedding/reranking
candidate. Server-side is `FEASIBLE_NOW`: `llama-embedding` builds (build-cpu, exit
0), BERT arch + `bge-small-en-v1.5` recognized, `--pooling cls|rank` + `--reranking`
present, `conversion/bert.py` covers BGE and the XLM-Roberta reranker, and the
network is reachable to fetch/convert weights. The `EMBEDDING_MODEL_FUNNEL.md`
premise that HTP is Gemma-decode-only is partly refuted: the Hexagon `supports_op`
switch already accepts NORM (LayerNorm), L2_NORM, GELU, non-causal SOFT_MAX, and
GET_ROWS. Residual phone-side risk is real but narrow: f16 `get_rows` and non-32-
aligned softmax fall back to CPU (addressable via F32 embeddings + seq padding),
and DSP kernel correctness for these BERT ops is unmeasured (v75 has a precedent of
supported-but-broken RMS_NORM/ROPE and wrong fused-FA). Verdict: a phone-resident
second island is `FEASIBLE_WITH_WORK / UNKNOWN-until-Gate-1`, gated on one bounded
on-device `test-backend-ops`/cb-eval op-support+correctness probe on OP15/v81 and
OP12/v75. Do not declare `TRACE_OR_SERVICE_BLOCKED`; do not yet claim a phone-side
second service.

**S13 LIVE TWO-PHONE FFN FLEET: RUNTIME MECHANICS PASS; SERVER BENEFIT NOT
TESTED.** The project now has a compiled runtime path, not only an S12 replay.
`llama-phone-pim-fleet` owns one persistent protocol-v3 session per phone,
`llama-phone-pim-fleet` owns one persistent protocol-v3 session per phone,
discovers the authoritative generation with STATUS, idempotently PREPAREs the
exact Gemma4 `blk.2` FFN identity, and drains a shared real-work queue across
OP12 and OP15. Every returned activation is checked against a local CPU FFN
oracle.

Both phones independently passed the production llama.cpp callback oracle at
about 2.9e-4 relative L2. A final-code cold eight-job fleet run assigned 4 jobs
to each phone, completed validated dispatch in 222.221 ms, and stayed below
3.05e-4 relative L2. Cold remote setup was 18.621 s because OP15's observed
load took 18.571 s; this transient is reported, not generalized. A
warm-resident 16-job run used a deliberately stale generation hint, adopted
generation 7 from STATUS, completed remote setup in 53.790 ms and validated
dispatch in 408.023 ms, and assigned 9/7 jobs to OP12/OP15.
Protocol/client/store/stream
plus fail-closed CLI tests pass 6/6 in release and ASan/UBSan builds; the new
targets are warning-clean. The final binary, schema-v2 cold/warm records, worker
logs, and production-oracle records are digest pinned.

This is one real dense-FFN island and a completion-driven harness. It is not
mixed-model serving, A6000 offload relief, or `llama-server` integration. The
current ADB forwards carry both activation directions; S12's separate WiFi H2P
and USB P2H runtime remains unimplemented. No latency, capacity, HBM, or energy
claim is authorized. See `spikes/s13_runtime_fleet/`.

**S11-B STATIC BATCHED ROUTE: MECHANICS + KV CAPACITY PASS; SERVER THROUGHPUT
RELIEF FAIL; ENERGY NOT RUN.** The resident OP15 `[0,2)` route now carries
bounded multi-sequence prefill and decode commands with explicit sequence IDs,
distinct KV state, exact batch/group/stream identity, and an equally batched
A6000 control. Exact greedy output passes at B=1,2,4,8,16. The complete route
scales from 3.18 req/s at B=1 to 25.49 req/s at B=16, an 8.01x increase while
group latency rises 2.00x. This proves that static batching uses the phone much
more efficiently.

It does not provide additive server throughput. The phone route reaches only
41-51 percent of the equally batched A6000 control. A three-pair B=8 run with
two measured groups per route is exact for 48/48 treatment requests, with
452.36 ms median group wall, 17.12 req/s, and 4.74 percent CoV versus the
A6000's 202.07 ms and 39.67 req/s. Phone thermal status stayed 0.
The complete-route CoV is below 5 percent, but the six OP15 stage groups have
8.52 percent CoV, so a stable sustained phone service rate is not established.

The Gemma-4 KV cache now follows the same layer window as weights and graph
execution. OP15 `[0,2)` fell from 1280 MiB to 4 MiB at B=1; OP12 `[2,3)` fell
to 2 MiB. The two-phone route remains exact and releases 1288 MiB on the
selected A6000. Keep batch execution as a scheduler primitive only under
memory, admission, or future measured power pressure. See
`spikes/s11_batched_route_poc/`.

**S11-B FAIL-CLOSED REPAIR: PASS.** The runner is now a versioned v2 evidence
path, and the stage protocol exchanges a versioned hello before work. It rejects
wrong stage roles, layer gaps/overlaps, model dimensions, a non-tail host,
oversized B=1 prefills, premature EOF, duplicate JSON keys, pair truncation, and
handed measurement-validity labels. CPU, CUDA, Android, and ASan/UBSan builds
pass; 26 runner tests pass. A fresh B=8 OP15 route remains exact at 449.66 ms
versus 204.76 ms server-only, with 888 MiB A6000 relief. The repaired two-phone
chain is also exact and releases 1288 MiB. Neither run changes the throughput
failure or authorizes energy. Exact per-node HTP placement remains unproven.

**S11-E0 SELECTED-A6000 BOARD ENERGY: FIXED SERIAL ROUTE FAIL.** The frozen
eight-pair cohort completed without selection or rerun. All 16 slots are exact,
evidence-valid, and within the 3.5 s p95 SLO. The OP15 `[0,2)` route releases
888 MiB on the selected A6000 and lowers its average board power from 289 W to
192 W, but aggregate runtime grows from 683 s to 1,576 s. Equal-work board
energy rises from 197.4 kJ to 302.3 kJ (+53.15 percent); the conservative
control-minus-treatment bound is -125.8 kJ. Verdict:
`GPU_BOARD_DIAGNOSTIC_RELIEF_FAIL`.

Reject the fixed serial route as an energy-saving primitive and do not sweep
B=16 or another boundary. Carry forward only the exact resident-phone,
scheduled-placement, thermal, and 888 MiB relief mechanisms. The next mechanism
must overlap phone work with useful server work across concurrent batches or
mixed jobs, preferably using both phones as replicated READY islands. NVML
still excludes CPU, DRAM, PSU, USB, and phone energy, so
`PHONE_ENERGY_UNKNOWN`, `TOTAL_SYSTEM_ENERGY_UNKNOWN`, and `formal_claim=NONE`
remain mandatory.

**S8 REAL TRACE COMPONENTS: GATE-A PASS; MIX COMPOSITION BLOCKED.** The final
normalizer scans the pinned 5,344,021-row BurstGPT source and 7,106-row
RAGPulse source, selects deterministic load windows, and publishes atomic
hash-bound bundles. Current-code BurstGPT median and every RAGPulse window
reproduce byte-for-byte. A fail-closed structural reader now binds the raw
source, config, normalizer code, every sibling trace and sidecar, artifact
replay digest, and static service DAG before accounting demand. Real replay
passes for BurstGPT median (165 requests) and RAGPulse low/median/high/burst
(2/12/27/50 requests). A certifying replay also reruns normalization from the
pinned raw source and byte-compares the complete bundle. Normalizer 24/24,
structural replay 32/32, and all 17 schemas pass both validators. The frozen
mixed-component transform is not implemented; semi-synthetic replay rejects
explicitly, so mixed Gate A remains blocked.

**S12-V0 TRACE-DRIVEN VIRTUAL QUEUE: MECHANICS PASS; REAL PROFILE COVERAGE ZERO.**
Four frozen V0 policies run deterministically: clairvoyant server-only,
causal server batching, fixed phone, and memory-admission-triggered phone.
The memory policy selects a phone only when no currently dispatchable server
batch fits A6000 HBM. Thirty unit tests, two CLI negatives, five hash-seed runs,
trace-snapshot binding, and active-route HBM accounting pass. A synthetic
shape-shadow demonstrates the mechanics only. Exact coverage on the real
normalized windows is 0/165 for BurstGPT median and 0/12 for RAGPulse median,
because the traces do not bind the S11 model, prompt, context, or payload.
No real request is assigned an S11 latency and energy remains NOT_RUN.

**S12-V1 ASYMMETRIC DATA PATH: MECHANICS PASS; PHYSICAL PROFILE BLOCKED.** The
selected topology now sends host-to-phone input over the shared WiFi LAN and
returns phone results over each phone's USB connection. A versioned replay
models independent WiFi H2P and USB P2H lanes, bounded phone/host buffers, the
phone phase, and an A6000 tail queue. Prefill and each decode step repeat the
ordered path `WiFi -> phone -> USB -> tail`; a decode input is released only by
the preceding tail.

Traffic classes are now explicit: small commands, token IDs, and sequence
metadata use WiFi H2P; large verified weight segments use resumable USB H2P;
dense results use USB P2H. Result returns preempt background weight streaming
on the same phone link. S12 still assumes weights are ready; composing the S9
slow loop with this priority rule is a later gate.

The adversarial review removed two false overlap assumptions. Each V1 run now
freezes either `FULL_MODEL` server residency or `TAIL_ONLY` phone-route
residency for the full horizon; dynamic route mixing is blocked until measured
host load/unload transitions exist. OP15 has one `llama_context`, so the replay
hard-limits the phone to one KV-owning group and reports zero cross-group
WiFi/USB overlap. Separate paths are modeled, but topology alone does not create
executable overlap.

The frozen synthetic fixture is not a new-topology latency result. WiFi/USB
rates are assumed and the phone phase reuses the old single-socket stage wall
as an undecomposed proxy.
The current route sends small token/metadata input to the OP15 head and returns
the dense cut activation; middle islands must bind separate dense input/output
bytes. LayerSplit still has one bidirectional socket, so paired WiFi ingress and
USB egress runtime connections are not implemented. Multiple leased contexts
or measured KV state switching are also required before the independent links
can overlap. Energy remains `NOT_RUN`.

**S11 FIXED-ROUTE FOUNDATION (RETAINED): EXACT ROUTE + GPU MEMORY PASS; LATENCY
FAIL; ENERGY NOT RUN.** A new resident `monodriver` control and resettable one/two-phone
`pipedriver` path now compare the same greedy workload after all weights and
warmups are resident. The live OP15 HTP route owns Gemma-4 12B F16 layers
`[0,2)` and returns only the cut activation to the A6000 tail.

Seven measured requests were exact: both routes generated
`100,45518,107,236829`. The phone-owned island reduced selected-A6000 memory
from 24,580 to 23,724 MiB, releasing 856 MiB. Batched 28-token prefill replaced
the original sequential B=1 prefill plumbing. It did not reduce latency:
server-only median was 156.95 ms/request versus 380.50 ms with OP15, because the
OP15 stage median was 223.56 ms. A repeated run put treatment at 345.50 ms, and
the final OP15-stage CoV was 9.23 percent, so the phone path is not stationary
enough for a latency or energy claim. This is a mechanics and memory proof, not a
general performance win. The scheduler must use this placement only under
memory/capacity or independently measured power pressure, or after concurrent
streams can batch the phone decode work.

The harness is fail-closed: ABBA ordering, exact token equality, explicit GPU
UUID, ready/done measurement barriers, integer ZOH NVML integration, stable
pstate, 100 independent updates, 250 ms maximum gap, and +/-5 W uncertainty.
No energy run was made, and no total-energy label is authorized. OP12 was
disconnected, so the compiled two-phone route remains unexecuted.

**S10-E2A R4 CURRENT STATUS: ROUTE-DAG MECHANICS PASS; PHYSICAL CLAIM BLOCKED.**
This supersedes R3. A final audit proved that R3's route digests were opaque
labels and that decorative phone work could pass while a server EXEC produced
the result. Active schema v4 resolves exact control/treatment `RouteSchedule`
records pinned by the anchored plan.

Each route freezes its action IDs, devices, backends, operator islands, request
sets, byte/duration and lease requirements, plus control/data dependency edges.
The realized lifecycle must match that DAG exactly. Every assisted request needs
a DATA path from phone H2D through HTP/OpenCL execution and D2H into the result;
a server continuation is valid only downstream of phone D2H. The reproduced
decorative-phone exploit now fails `E_ROUTE_NODE_EXTRA`. Resolver-issued route
evidence stores canonical bytes, so post-validation dictionary mutation cannot
rewrite the route.

Verification: E2A 215/215, R4 14/14, CLI negatives 42/42. Fixture replay is
byte-identical, and the 45/28 pinned E1/E2 baseline files remain unchanged. No
measurement was run. The production fixture fails closed at
`E_ANCHOR_TRUST_ROOT`; no eligible external verifier, witnessed launcher,
calibrated server-wall instrument, or physical acquisition exists yet.

**S10-E2A ALL-PAIRS AGGREGATE: MECHANICS PASS, EXTERNAL ANCHOR BLOCKED. A TIMESTAMP
AUTHORITY WOULD NOT HELP.** E2A builds the `SUM_ALL_PAIRS_V1` evaluator E2 left
unbuilt, and hits a THIRD blocker, independent of the two below. An all-pairs sum is
only worth something if the cohort was fixed before the results were seen. That needs
two properties, and **they do not covary**:

~~~text
  P1 PRECEDENCE   the plan existed before the runs   <- RFC3161 buys this, in full
  P2 EXCLUSIVITY  exactly ONE plan was committed     <- RFC3161 buys NOTHING of this

  A TSA is a RESPONDER, not a LOG. It does not publish or enumerate what it signs.
  -> anchor 32 candidate plans, run everything, reveal the one that fits.
  -> every per-record check passes.

  A token is a lower bound on a plan's AGE. Never an upper bound on its COUNT.
~~~

freetsa.org is genuinely independent (third-party key, third-party clock) and ~10
minutes of provisioning away -- a live probe returned `Status: Granted` and the chain
verifies once `cacert.pem` is fetched. **Provisioning it would still not unblock E2A.**
Encoded as `ANCHOR_INDEPENDENT[RFC3161]=True` beside `ANCHOR_ENUMERABLE[RFC3161]=False`;
that pair of lines IS the finding. Closing P2 needs an ENUMERABLE commitment
(pre-registration, or a transparency log with a reviewable identity binding). TPM
exists but is permission-denied and custodially ours; git is our own force-pushable
fork; the host's only `[tsa]` config points at `./demoCA`, i.e. openssl configures you
to be your OWN authority. Also honest: **E2A implements no cryptographic verifier at
all** -- it types capability, parses no token, checks no signature.

The adversarial review found **3 CRITICAL, all reproduced, all fixed**, two of them
verbatim recurrences of bugs this codebase documents as fixed: (1) the independent
checker CLI printed `SERVER_RELIEF_PASS` + exit 0 on twenty lines of hand-written JSON
-- E2's "handed a conclusion" bug, in the artifact a reviewer runs; (2) `__pycache__`
defeated the pinned canon -- `exec_module()` ran poisoned bytecode while the source
digest still matched, killing the type gate (`is_int(1.0)` OLD=True / NEW=False, same
digest); (3) `validate_aggregate`, billed as "what makes the label unfakeable", raised
`E_TYPE` on its OWN output and had never once run. Design result kept: **structural
checks first, policy gates last** -- the anchor gate ran first and masked 13 CLI
negatives, so a deleted check and a working one looked identical.

**S10-V0-R-E2 MATCHED TIMELINE: MECHANICS PASS, MEASUREMENT NOT RUN. NO SERVER-WALL
INSTRUMENT EXISTS ON THIS HOST.** E1 rejects every `MEASURED` instance on purpose --
its solver is additive per-device while a wall/board measurement is an AGGREGATE
timeline, and feeding an aggregate into an additive solver double-counts shared power.
E2 is the other shape: it compares two REALIZED timelines POST HOC (optimized
server-only control vs Q-PIM treatment) at the same boundary. E2 v1 proves aggregate
accounting closure (`met+tardy+rejected+canceled == offered`) and exact equality of
opaque workload/SLO digests; it does not yet prove per-request output equivalence or
that lifecycle work stayed inside the paid window. Frozen: left-edge
zero-order-hold integer integration, gross energy only (no invented idle baseline),
conservative decision (`treatment+unc < control-unc`) with a 10% gate by integer cross
multiplication (`t*10 <= c*9`, no float, no division). `SYSTEM_ENERGY_SAVING` is not a
value in any E2 schema -- it is inexpressible, not merely disallowed.

**The instrument audit is the load-bearing result.** First-hand on the live host: NVML
is GPU-board ONLY (TWO A6000 boards, so a timeline must name which; `power.draw` on
Ampere is a **1-second average** with a vendor-stated **+/-5 W** accuracy, now the
enforced uncertainty floor). RAPL is unusable twice over -- `energy_uj` is root-only
(sudo needs a password) AND only `package-0`/`core` exist (no dram, no psys), so it is
a component counter that can never be a wall. No BMC, IPMI, PDU, or external meter
exists. **`SERVER_RELIEF_PASS_TOTAL_ENERGY_BLOCKED` is therefore UNREACHABLE without
new hardware**, not merely unmeasured. A GPU-board delta is named `boundary_delta_nj`
and its `server_wall_delta_nj`/break-even budget are NULL: a board sensor cannot
establish how CPU, DRAM, fans, or PSU losses moved, so its delta is not a server delta.

**Physical labels are unreachable by construction here**: a single pair is diagnostic
only (`PAIR_ONLY_NO_AGGREGATE_CLAIM`); a label needs `SUM_ALL_PAIRS_V1` over a complete
predeclared repetition set, and that evaluator is deliberately not built. The synthetic
fixture computes `relief=true` at exactly -20% and is still `MEASUREMENT_INVALID /
SYNTHETIC_NO_PHYSICAL_CLAIM`. The one existing A6000 trace stays NEGATIVE evidence: 323
rows hold only **57 value changes** (a 10 Hz poll of a ~1.7 Hz sensor), it spans P0/P2/
P3/P8, and it is not a matched pair -- rejected on four independent grounds, gate not
lowered.

**Ten fail-open paths were found and closed.** I found one myself: recomputation was
opt-in (`samples=None`) and no caller opted in, so a forged `energy_nj=1` validated on
the strength of a correct artifact hash -- E1's "a signed number is never proof of
itself", repeated. An independent red team found nine more, all real: `build_comparison`
accepted a LABEL and checked only set membership, so importing the module stamped a
sealed `SERVER_RELIEF_PASS` onto junk whose treatment burned 1e15 nJ MORE (critical the
moment the aggregate evaluator lands, since it calls exactly that function); quality
gates counted the WHOLE artifact while energy integrates only the window, so padding
outside the paid window was free and admitted the real 57-update trace; a TOCTOU gap
between hashing a path and re-opening it (won 74/400 with no privileges); a timeline's
own `status` was never read, so `FAILED` runs reached the decision; `E_STATUS_CHANGE`
was opt-in via an optional scalar `pstate` and was **dead code across all 104 tests**.
Fixes: label DERIVED not supplied, quality measured over the paid window, read-once
artifacts, `status != OK` refused, `pstates` required one-per-sample. The reproduced
kill-chain stages are pinned as unit regressions. One claim I could not
fix and did not pretend to: a label split across two non-adjacent free-form fields
evades any contiguous scan -- recorded as an executable test, since the closed enum is
what actually blocks it. A later audit also cross-bound raw power, execution, and
timeline evidence by run nonce and paid-payload digest; rejected global record-digest
reuse, stale sample brackets, malformed P-states, and mismatched repetition scope; and
made `MatchedComparison` v1 diagnostic-only in both schema and runtime. A final live
audit closed hostile preloading of the canonical type gate, resealed comparison
arithmetic/reason mutations, representation-dependent paid-payload identity, and
inexact or over-wide NVML parsing, plus a validator exception leak. It also made
two next-gate blockers explicit: resolved per-request same-work proof and paid
lifecycle/drain closure. Suite: 152 E2
tests + 30 CLI negatives, deterministic across 5 processes/seeds; the final 45-file E1
baseline is byte-identical before and after. E1 passes 201 tests (28 foundation + 173
evidence), 39 evidence negatives, 18 CLI negatives, and 1187 differential comparisons
with zero mismatches. See `spikes/s10_matched_energy_e2/`. No measurement, no
commit/push.

**S10-V0-R-E1 EVIDENCE INTEGRITY: PASS FOR MECHANICS; ALL PHYSICAL CLAIMS BLOCKED.
C0-C5 AND PF1 REMAIN UNAUTHORIZED.** The repaired live path runs isolated draft-2020
schema validation, hashes actual artifact files, binds exact token/KV shapes, pins
route/correctness/reference/boundary identity, and separates H2D, D2H, and produced
output bytes. Merged-batch profiles are checked against every legal member subset so a
zero-output or uncharged-energy boundary cannot back a real merged action. Required
bindings are derived from the INSTANCE, so deleting one is an error. Certificates are
schema-checked before indexing. The evidence-bound v3 path still reproduces the frozen
147250000 and 2974000 nJ mechanics optima, with every v2 mechanics file byte-unchanged.

The atlas remains EMPTY: zero eligible measured rows exist, the A6000 trace is
GPU_BOARD rather than wall power and has too few independent samples, and phone energy
is physically unmeasurable. More importantly, E1 now rejects every `MEASURED` instance:
its solver adds per-device and per-route terms, while SERVER_WALL and TOTAL_WALL are
aggregate timelines. No GPU-board, server-relief, or system-energy label is emitted
until a separate typed matched control/treatment record exists. A second adversarial
pass reproduced and closed live schema bypass (including hostile `PYTHONPATH`), absent
artifact bytes, disjoint KV/correctness envelopes, phantom fallback/reference proofs,
route-boundary revision/time/byte mismatches, under-typed power and uncertainty, and a
batch route that previously selected an invalid 1 us merged action. Final targeted
review found no remaining executable blocker in those paths. Suite: 201 tests (173
evidence + 28 foundation), 39 evidence negatives, 18 CLI negatives,
fixture/hash-seed determinism, and the foundation's 1187-case differential digests
remain byte-identical. This is tested
evidence integrity, not certification of arbitrary live dispatch. See
`spikes/s10_power_frontier_repair/{EVIDENCE_CONTRACT,EVIDENCE_MATRIX,RESULTS_E1}.md`.
No commit/push.

**S10-V0-R TEMPORAL FOUNDATION: PASS (retained).** The historical S10-V0 `FAIL` is
still INVALID/INCONCLUSIVE and is retained as a historical tree; it is not a
falsification of Q-PIM. The earlier exact recursive-hash claim was withdrawn because no
byte-exact recipe and pre-run manifest were persisted; a persisted, self-checking recipe
now exists (`HISTORICAL_MANIFEST.txt`, 98 files excluding bytecode -- the old "101" had
counted 3 `__pycache__` entries). The two exactness defects that blocked the foundation are now closed. The
solver has two exact modes: earliest-start is kept only where it is provably exact
(zero wake/idle/transition and zero output bytes, where server energy is independent
of start times), and everything else enumerates every legal integer start time across
routes, compatible batch partitions and device orders, bounded by the horizon rather
than by deadlines (TARDY is legal). It therefore now SELECTS the delayed 147250000 nJ
placement over the earliest 162000000 nJ at identical zero-miss/zero-lateness
outcomes (one merged P0 window instead of two), and finds a delayed
activation-feasible placement where the earliest one breaks the memory bound (peak
200 > 150 -> 100). Optimality is no longer taken on the solver's word: an
checker-owned reference with structurally separate enumeration (stdlib only, no
oracle imports, no incumbent/objective pruning) re-derives the optimum, so default
checker mode ACCEPTS proven optima and REJECTS feasible-but-suboptimal certificates;
the signed completeness marker is never proof, extra search metadata is rejected, and
`--feasibility-only` stays separate and claims nothing. Suite: 25 tests plus 1187
compared generated cases across four processes and multiple PYTHONHASHSEED values
(0 mismatches, deterministic digests), 14 CLI negatives with no tracebacks, and
fail-closed out-of-domain/state-cap behaviour. An adversarial review also closed
empty identifier acceptance, forged signed search metadata, vacuous differential
coverage, and error-classification holes. Typed evidence, C0-C5, PF1, runtime,
capacity, and energy remain unauthorized. See
`spikes/s10_power_frontier_repair/{V0_AUDIT,PLAN,RESULTS}.md`. No commit/push.

**Phase: S10-V0-R-E2 MATCHED CONTROL/TREATMENT TIMELINE GATE - DONE, VERDICT
E2_MATCHED_TIMELINE_MECHANICS_PASS_MEASUREMENT_NOT_RUN (2026-07-15, ASCII).** Built a
separate post-hoc matched-comparison mechanism under `spikes/s10_matched_energy_e2/`:
versioned records (RealizedTimeline, MatchedComparison, RepetitionSet,
ServerWallCapability), a deterministic integer integrator, a conservative comparator, an
instrument audit, and an adversarial suite. No physical measurement was run and none is
authorized. No server relief and no energy saving is claimed.

Architecture: E1 rejects every `MEASURED` instance CORRECTLY, because its solver is
additive per-device while a wall/board reading is an AGGREGATE timeline of a whole
boundary. E2 never feeds an aggregate back into the solver; it compares two REALIZED
timelines post hoc at the same boundary, over the same CLOSED work, with identical SLO
outcomes. `assert_not_additive_input` makes that rule executable (and now covers all
three aggregate kinds, not just the timeline). E2 reuses E1 only for canonical
JSON/SHA-256, read-only; a test asserts no E1 decision logic is imported.

Frozen mechanics: left-edge zero-order-hold integer integration clipped exactly to the
window; gross energy only (no invented idle baseline); `treatment+unc < control-unc`
plus a 10% gate by integer cross multiplication (`t*10 <= c*9`) -- no float, no
division; uncertainty never optional; work must be closed; scope/rails/boards/clock
epoch must match. `SYSTEM_ENERGY_SAVING` is absent from every schema and enum:
inexpressible, not merely disallowed.

**Instrument audit (the load-bearing result, first-hand):** NVML is GPU-board ONLY --
TWO A6000 boards on this host, and `power.draw` on Ampere is a **1-second average** with
a vendor-stated **+/-5 W** accuracy (now the enforced uncertainty floor, charged per
board). RAPL is unusable twice over: `energy_uj` is `-r-------- root root` with no
sudo, AND only `package-0`/`core` exist (no dram, no psys) -- a component counter, never
a wall. No BMC/IPMI/PDU/external meter. **SERVER_RELIEF is UNREACHABLE without new
hardware**, not merely unmeasured. Consequently a GPU-board delta is `boundary_delta_nj`
with `server_wall_delta_nj` and the phone break-even budget NULL -- a board sensor
cannot establish how CPU/DRAM/fans/PSU moved.

Physical labels are unreachable BY CONSTRUCTION here: a pair is diagnostic only
(`PAIR_ONLY_NO_AGGREGATE_CLAIM`); a label needs `SUM_ALL_PAIRS_V1` over a complete
predeclared repetition set, deliberately not built. The synthetic fixture computes
`relief=true` at exactly -20% and stays `MEASUREMENT_INVALID /
SYNTHETIC_NO_PHYSICAL_CLAIM`. CP5: the existing A6000 trace stays NEGATIVE evidence --
323 rows hold **57 value changes** (10 Hz poll of a ~1.7 Hz sensor), it spans P0/P2/P3/
P8, and it is one timeline, not a pair. Rejected on four independent grounds; the gate
was frozen before the trace was read and was not lowered.

**Ten fail-open paths found and closed.** Mine, before the audit: recomputation was
opt-in (`samples=None`) and NO caller opted in, so a forged `energy_nj=1` validated on a
correct artifact hash -- E1's "a signed number is never proof of itself", repeated.
Red team found nine more, all real: (F1) `build_comparison` accepted a LABEL and checked
only set membership, so importing the module sealed a `SERVER_RELIEF_PASS` onto junk
whose treatment burned 1e15 nJ MORE -- critical because the next checkpoint's aggregate
evaluator calls exactly it; (F2) quality gates counted the WHOLE artifact while energy
integrates only the window, so padding outside the paid window was free and admitted the
real 57-update trace; (F3) TOCTOU between hashing a path and re-opening it, won 74/400
unprivileged; (F4) a timeline's own `status` was never read, so `FAILED` runs reached
the decision; (F5) `E_STATUS_CHANGE` was opt-in via an optional scalar `pstate` -- dead
code across all 104 tests; plus additive guard covering one kind of three, an evadable
label scan, empty rails accepted, and `normalizer_digest` required but never checked.
Fixes: label DERIVED never supplied, quality measured over the paid window, read-once
artifacts, `status != OK` refused, `pstates` required one-per-sample. The composed kill
chain is dead at every stage, verified against the red team's own `x4_chain.py`.

Honest limits recorded rather than papered over: a label split across two non-adjacent
free-form fields evades any contiguous scan (canonical JSON sorts keys) -- kept as an
executable test, since the closed enum is the real block; E2 cannot detect a FABRICATED
artifact, only make a lie digest-pinned and attributable; `route_schedule_digest` is
recorded provenance, deliberately not matched. Suite: 114 E2 tests + 30 CLI negatives,
deterministic across 5 processes/seeds, E1 byte-identical before AND after in the same
run. E1 itself moved 185->192->194 tests under concurrent review while E2 was written;
I did not modify it, and both manifests are kept so the drift is a diff rather than a
silence. Next: the aggregate evaluator, then a separately authorized GPU-board A/B
designed around ~1.7 Hz and +/-5 W. No commit/push.

**Phase: S10-V0-R-E1 POST-REVIEW REPAIR - DONE, VERDICT
TYPED_EVIDENCE_INTEGRITY_PASS_PHYSICAL_CLAIMS_BLOCKED (2026-07-15, ASCII).** Repaired
the worker's fail-open live path without touching the frozen v2 solver/checker. Live
schemas now run under isolated system Python; artifacts are resolved, contained, and
byte-hashed; token/KV and correctness envelopes are exact; reference and fallback
proofs name distinct real artifacts; route/boundary identity, revision, wall time,
H2D/D2H/output geometry, and energy are coherent; certificates fail schema before
indexing; and batch rows cover every legal subset without silently dropping boundary
energy. All physical claims are disabled in E1 because additive solver inputs cannot
represent aggregate wall timelines. 185 tests, 39+14 CLI negatives, 1187 differential
comparisons with no mismatch. No commit/push.

**Superseded worker report: S10-V0-R-E1 TYPED EVIDENCE BINDING - originally reported
TYPED_EVIDENCE_CONTRACT_PASS_ATLAS_BLOCKED (2026-07-15, ASCII).** Built the versioned
typed-evidence contract (schema v3, 7 immutable record types, 24 stable `E_*` codes),
a deterministic binder, a strict fail-closed validator, an honest evidence inventory,
and an adversarial suite. `evidence.scope=MECHANICS_ONLY` was a label; it is now a
mechanism.

Structure: required bindings are computed FROM THE INSTANCE, never from the binding
list, so an omitted binding is `E_BINDING_MISSING` rather than an unchecked number. A
v3 instance is a v2 core plus a binding block; the binder projects to v2 IN MEMORY
ONLY to reuse the proven solver, and the certificate signs the FULL v3 instance plus
the bundle and binding digests, so a swapped projection cannot validate. All v2
mechanics files are byte-unchanged and the evidence path reproduces both frozen optima
(147250000 nJ with `server_p0_intervals [[800,1150]]`; 2974000 nJ at peak 100).

Energy boundaries frozen (contract sections 4, 5b, 5c): GPU_BOARD -> GPU-board relief
only; SERVER_WALL -> server relief only; TOTAL_WALL required for SYSTEM_ENERGY_SAVING;
`delta_E_server` is a break-even BUDGET for excluded phone/USB/charger/relay energy,
not a saving; unknown phone energy is UNKNOWN with a reason, never 0.

**Atlas: EMPTY. Zero eligible measured rows.** No PowerProfile of any device passes,
so no `MEASURED` instance is constructible and no energy claim is authorized. The
A6000 board trace is a real measurement refused twice (GPU_BOARD scope; ~57
independent samples at ~1.77 Hz effective vs `MIN_POWER_SAMPLES=100`). Phone energy is
UNKNOWN and physically unmeasurable (USB rail pinned, coulomb dead at 99% Charging, no
root on op12, `pwr_sampler.sh` never run). Gates were frozen BEFORE the atlas was read
and were not lowered when it came back empty.

Also recorded (not this spike's to fix): several published claims are contradicted by
their own cited artifacts -- S6 "xmem GEMM confirmed" (cited CSV has 0 hits for
xmem/os8/prepack; the 126x kernel is the stock `kernel_mul_mm_f16_f32_l4_lm`), S6 "HMX
every M>=5" (only M=1 and M=8 exist on disk; the "7 hmx" are 7 MUL_MATs of one M=8
graph), S6 fused-FA "native HMX" (all 8 FA lines are unit `----`), the S6-L ffnmerge
16+48 row (splices a speedup from a no-correctness run onto a correctness value from
different shapes), and adb-push "262 OP15 / 216 OP12" (device-swapped and ~3x high;
the artifact says OP12 249 / OP15 86).

**An independent red team broke the first implementation TEN ways. All real, all
closed, each with its own regression.** Two classes I had missed entirely:
(1) **float/bool type confusion** defeated EVERY eligibility gate -- guards written
`if is_int(a) and is_int(b) and <bad>` SKIP on a wrong type, and `900000.0 == 900000`,
so a route certified at 90% error against a 0.5% threshold bound and solved cleanly;
JSON Schema provably cannot catch it (draft6+ `integer` accepts any zero-fraction
number), and my own `test_bool_is_not_an_integer` asserted the exact property that
guaranteed the bypass. (2) **record selection** -- the contract froze the STATISTIC to
stop cherry-picking, then left WHICH RECORD free, so cherry-picking returned one level
up: a server with wake=50 and transition=0 that no record describes (147250000 ->
146250000), a COLD route and a STEADY route in one schedule (-> 122500000), a
USB_VBUS twin record double-counting a phone, and any PASS route erasing the activation
bound. Also: GPU-board energy inside a certified SYSTEM claim, boundary records with no
identity at all, `MAX_INT` never enforced, `None == None` validating an absent field,
and tracebacks escaping instead of refusals. Fixes: one type gate at load before any
comparison, every guard inverted, `E_INCOHERENT` record coherence (one device -> one
power record, one thermal condition, one build), boundary identity + direction, boundary
scope folded into claim classification, `canon.MISSING`, and `validate_safe`.

What the red team could NOT break is worth as much: `required_targets()` enumeration is
complete, verified empirically by perturbing every integer leaf and re-solving. A
structural guard now walks every integer in the instance and fails if one is neither
evidence-derived nor declared workload -- including a test proving the guard can fail.

Suite: 154 tests (25 foundation + 129 evidence), 37 evidence CLI negatives + 14
foundation negatives (no tracebacks), determinism across 5 processes/PYTHONHASHSEED
values, and the foundation's four 1187-case differential digests BYTE-IDENTICAL to the
pre-edit baseline. Honest limits recorded: no ajv on this host; the gate binds numbers
to artifacts but cannot detect a FABRICATED artifact; `activation_mem_bound_bytes` is a
capacity bound to a designated `CAPACITY_PROBE` route as a proxy; `horizon_us` is
workload-declared yet multiplies idle energy, so absolute certificate energy is not
fully evidence-derived. HEAD `933c722f6` unchanged, historical tree verified under a
persisted self-checking recipe, no commit/push. Even PASS does not authorize C0-C5.

**Phase: S10-V0-R ADVERSARIAL REVIEW CLOSURE - DONE, TEMPORAL FOUNDATION STILL
PASS (2026-07-15, ASCII).** Independent review reproduced the full suite and found
no exactness defect in the declared bounded temporal model, including an additional
1200-seed slice and prune-on/off comparisons. It did find two contract holes: genuine
optima could carry forged search counters, and empty device/route identifiers were
accepted. Signed certificates now contain only `search.complete`; default checker
mode re-proves optimality and rejects extra search metadata. Schemas and semantic
validation reject empty identifiers. Regression coverage now fails on unexpected
solver/reference errors instead of calling them jointly infeasible, asserts all 1200
differential seeds are accounted for with at least 1000 exact comparisons, and makes
CLI setup fail closed. Reproduced result: 25 unit tests; 14 CLI negatives; 1187
compared + 13 jointly infeasible + 0 skipped + 0 mismatches; deterministic digests;
marker `S10_V0R_TEMPORAL_FOUNDATION_TESTS_PASS`. Stale status prose was corrected,
the unsupported historical recursive-hash and zero-pruning-count claims were
withdrawn, and the active handoff now targets typed evidence with separately labeled
GPU-board/server relief while total energy remains blocked by unknown phone energy.
C0-C5 remain unauthorized. No commit/push.

**Direction reset - 2026-07-15: Q-PIM POWER-FRONTIER DESIGN FROZEN; S10-V0 NOT
RUN.** Replaced the prior capacity-first MW0-MW7 order with PF0-PF5. The new
mechanism jointly schedules topological DAG order, phone active-weight
residency, complete-island placement, A6000 lazy claims/batches/power states,
and phone pacing. Offload value is evaluated as a counterfactual power-trigger
bundle at the complete wall boundary rather than a sum of skipped GPU work.
S9-V1A-R is preserved as completed bounded transport substrate. The first test
is deliberately small and fail-closed: two/three frozen DAG templates, measured
server/phone power surfaces, exact tiny enumeration, independent checker,
bounded causal beam search, and one controlled real replay. Possible verdicts
are PASS, MECHANISM_PASS_ENERGY_BLOCKED, or FAIL; only PASS authorizes PF1.
No code was implemented for this direction reset.

**Phase: S10-V0-R TEMPORAL FOUNDATION - DONE, VERDICT TEMPORAL_FOUNDATION_PASS (ASCII).**
Baseline before editing: HEAD 933c722f6, `scripts/run_tests.sh` 19 tests exit 0 marker
`S10_V0R_CURRENT_MECHANICS_TESTS_PASS`; historical `s10_power_frontier/` retained as
101 files. The earlier byte-identical recursive-hash claim is withdrawn because its
recipe and a pre-run manifest were not persisted. CP1 froze the exact temporal
domain in PLAN.md: integer us/mW/nJ; EARLIEST mode kept ONLY where provably exact (zero
wake/idle/transition and zero output_bytes => server energy = p8*H + (p0-p8)*sum(durations),
independent of start times, and earliest-start minimises every finish of a fixed order);
TEMPORAL mode otherwise. CP2 implemented full integer start-time enumeration over routes x
compatible batch partitions x per-device orders x delay, with feasibility windows from
releases/precedence/wake/HORIZON only - never deadlines, since TARDY is a legal outcome and a
deadline bound would discard feasible schedules. Declared bound TEMPORAL_MAX_NODES=6,
TEMPORAL_MAX_ACTIONS=6, TEMPORAL_MAX_WINDOW_PRODUCT=8e6 accumulated and checked before
recursing into each device order; crossing it raises without a certificate; max_states exhaustion raises;
`complete=true` only after exhaustion. Two documented B&B rules (R1 lateness/miss bound, R2
energy floor) are lexicographic lower bounds and are tested against `prune=False`. RESULT:
the frozen counterexample now SELECTS n0[850,950]+n1[1000,1100] = one merged P0 window
[800,1150] = **147250000 nJ**, beating the earliest **162000000 nJ** at identical [0,0,-2]
outcomes. New frozen `activation_delay_counterexample.json`: earliest peak
200 > bound 150 (infeasible), oracle finds delayed p1[10,14] -> peak 100. CP3: `checker/
reference.py` is structurally separate (stdlib-only, no incumbent/objective pruning,
no oracle imports) and
independently confirms both fixtures. Default checker mode now ACCEPTS proven
optima and REJECTS feasible-but-suboptimal certs (stable SUBOPTIMAL diagnostic) even when they
carry the genuine optimum's completeness marker; the marker is never proof; `--feasibility-only`
stays separate (optimality_verified=false). partial_partition (horizon 5000 => 19.6M combos vs
8M bound) is OUT of the reference domain, so default mode fails closed with that specific
reason and its optimum is instead proved by the pre-existing batch slow_reference - no frozen
fixture rests on the oracle's own word. CP4: 25 unit tests; 1187 compared generated cases
(1200 seeds, 13 agreed-infeasible, 0 out-of-domain, 0 mismatches) over 4 processes at
PYTHONHASHSEED 0/1/42/12345, digests byte-identical again at seed 99 and `random`; regressions
for equal-busy/different-gap, wake+idle boundary equality (touching windows merge to [50,450],
+1us splits them; equal 400us active, delta == exactly one transition_nj), horizon/release/
deadline/precedence/lane-overlap/duration/batch-identity/activation/terminal/energy/objective
mutations, incomplete-search and state-cap exhaustion; 14 CLI negatives all nonzero with stable
ORACLE_FAIL/CHECK_FAIL and zero tracebacks. An adversarial reviewer also caught that my
partial_partition guard was vacuously green and that PLAN.md's tail contradicted its new
header; both fixed. A final review removed unverifiable signed counters, rejected empty
identifiers, required full differential accounting, and stopped classifying arbitrary paired
runtime errors as agreed infeasible. `git diff --check` clean, ASCII-only, bytecode removed. Marker is now
`S10_V0R_TEMPORAL_FOUNDATION_TESTS_PASS`. Typed evidence binding is the next separate gate;
C0-C5, PF1, runtime, capacity and energy remain unauthorized. No commit/push. See
`spikes/s10_power_frontier_repair/{RESULTS,PLAN}.md`.

**Phase (historical, superseded): S10-V0 SCREEN - INVALID/INCONCLUSIVE, NOT A
Q-PIM FALSIFICATION (ASCII).**
CP0 integrity: HEAD 933c722f6 unchanged, pre-existing dirty tree preserved byte-for-byte,
phone-pim CTest 3/3 release + 3/3 ASan (pre and post edit), smallest resident dense-FFN
reproduced on both phones with the certified worker 0a50ca72..e749 (OP15/OP12
PRESTAGED_FFN_PASS, rel-L2 2.9e-4). Physical boundary = ENERGY_BLOCKED: no synchronized
wall meter (host+PSU+USB+phone), only A6000 board power at ~1.5 Hz (too coarse for few-ms
islands), power caps unsettable (no root), no A6000 state below auto-P8 (25 W), phone
charger unmeterable. CP2 atlas MEASURED: A6000 FFN island (n_embd 3840, n_ff 15360, 354 MB
f16) compute p50 517 us (M1) -> 545 (M16) -> 998 (M256) -> 3625 (M1024), batching nearly
free (marginal 1.887 us/token); board idle 25 W, sustained active mean 281 W / ceiling
300 W; phone same island e2e OP15 27.1 ms / OP12 45.0 ms (43-71x slower). INFERRED power
is mechanism-favorable (phone 2 W, USB 0 W, GPU 300 W) so a FAIL is conservative. CP3:
exact symmetry-reduced oracle + STANDALONE checker (imports no solver/sim/objective);
13/13 adversarial mutations caught; 1200 generated fixtures all oracle-valid and all 1200
corruptions rejected. CP4: policies C0..C5, every one of ~300 certs re-validated by the
independent checker. Result: under CONSERVATIVE measured-plausible power the oracle offloads
NOTHING (+0.0% vs optimized server-only in all 22 bins); under FAVORABLE blocked power C4
beats C1 only by offloading LONE unbatchable islands (primary +20.4%, lone+slack up to
+47%), and larger-server-batch=False in EVERY instance. Opportunity gate met only in
favorable+lone+slack (non-robust: 0% conservative). MECHANISM gate FAILS: none of the three
certified levers (denser batch / lower cap / break-even low-power interval) is available or
triggered on the measured hardware; the favorable-power gain is skipped-GPU-us per-island
offload, which the design excludes and credits at zero. CP5 physical reproduction NOT run
(gates failed). Verdict FAIL -> stop Q-PIM runtime / PF1; keep S9 as substrate. No commit/
push; protected scope untouched (no tools/server, model graph, KV, ggml_sched, kernels, or
v3 wire). See `spikes/s10_power_frontier/{RESULTS,MANIFEST,PLAN}.md` + `artifacts/`,
`cp_verdict.json`.

**Phase: S9-V1A-R PIPELINED-TRANSPORT EVIDENCE REPAIR - DONE (ASCII).** Repaired the V1A
profiling and re-grounded the evidence. Profiling is now opt-in (worker --profile-recv, host
--profile-transport; default OFF), stage-scoped/per-opcode (excludes HELLO/PREPARE/EXECUTE/
RELEASE/SHUTDOWN and oracle gaps), splits host outer-frame SHA (envelope+data) from the actual
socket write (only ~75 ms), measures worker prefix-hash separately, labels hash domains, and NEVER
sums host-CPU and phone-CPU timers as a wall fraction. WITHDRAWN V1A claims: "61%/69% SHA of wall"
(illegal cross-CPU sum), "four hashes over identical bytes" (really six passes, differing domains),
"26%/33% RTT" (process-lifetime, contaminated). Repaired window=1 accounting: HOST-side serial
timeline 99.86% (OP12) / 99.96% (OP15) of wall, single-CPU valid; phone-side timers reported
separately. Profiling overhead ~1.00x (measured ON vs OFF). Fail-closed analyzer (analyze_sweep.py,
18 mutation self-tests) validates device/worker-SHA/model-SHA/bytes/chunks/verdict/source/rel-L2/
accepted/duplicate/wasted and the UNROUNDED speedup. adversarial_tests.py: structural JSON, nonzero
exit on any fail, persists host stdout/stderr/exit + recovery records, computes retry/waste from
persisted first+resumed runs. Added: byte accounting (attempted/written/durable/wasted/retried),
FAIL_PROVISION record on disconnect, envelope-inclusive checked 64 MiB bound, --stage-window frozen
to {1,2,4,8}, bench sequence/summary/agreement validation, and an ASan/UBSan test of the real host
window>1 path (windows 1/2/4/8 + resume, zero diagnostics). Counterbalanced matrix (Latin square),
full provenance, thermal/frequency snapshots (NOT energy). Full-shard gate (5 reps/window, 0 errors,
0 duplicate/wasted bytes): OP12 2.61x median / 1.89x conservative (best w4); OP15 2.28x / 1.27x
(best w8) - OVERALL GATE PASS on both median and conservative min(best)/max(w1). 64 MiB 2.21x
(OP12) / 1.66x (OP15); 256 MiB stable at w8; simultaneous both phones on separate USB buses with no
cross-interference (OP12 37.45, OP15 22.86 MiB/s). Adversarial T1/T2/T3 PASS both phones (T2 recovers
from the worker verified prefix, which on OP15 exceeded the host-acked prefix by one chunk, proving
no in-flight guessing). Honest labels: gate PASS but partly a DVFS effect (OP15 throttled to 95 C /
<=1.3 GHz); contract INCOMPLETE; capacity UNPROVEN; energy DEFERRED. SHA de-dup, protocol v4,
scheduler, llama-server - reported, NOT implemented. No commit/push. See
`spikes/s9_pipelined_transport/{RESULTS_R,PLAN,MANIFEST}.md` + `artifacts_r/`.

**Phase (superseded by S9-V1A-R): S9-V1A PIPELINED TRANSPORT - PROFILED +
GATED, PASS.** The stop-and-wait
dynamic-provisioning path was profiled to 99.85% (OP12) / 99.80% (OP15) of the
window=1 wall using measurement-only host timers plus an opt-in worker `recv_profile`
(no v3 wire change; certified worker `0a50ca72…` untouched). Finding: SHA-256 is
**61% (OP12) / 69% (OP15)** of the wall and mostly **redundant** — each 4 MiB chunk is
hashed 4× (host verify + host rolling-prefix; worker frame-SHA + worker store-hash) plus
a full re-verify at commit; round-trip idle is 26%/33%; the wire is NOT the bottleneck
(a standalone memory-sink bench moves 64 MiB at 120/105 MiB/s, ~8× the full-stage rate).
A bounded `--stage-window N` (FIFO, ≤64 MiB outstanding, per-request durable-prefix digest
retained until its ACK, window=1 byte-identical) was added on the HOST only. Gate: median
provisioning goodput **2.44× (OP12) / 6.71× (OP15)** at window 8 (window 4 recommended),
both ≥1.20×, **0 correctness failures / 0 wasted bytes over 40 runs**, bit-identical
published SHA `5cfba18d…` and rel-L2 < 5e-3. Windowing adversarial T1 (resume-after-partial),
T2 (SIGKILL mid-flight → recover from the worker's verified prefix, resumed @163.6/138.4 MB),
T3 (window-8 durable identity) all PASS on both phones; ASan/UBSan + release CTest 3/3. Honest
caveat: part of the win is DVFS (a busy window keeps the CPU clocked up, so the dominant SHA
runs faster). Bigger levers — **SHA de-dup** (61–69%) and **batched ACKs** (protocol v4) —
are profiled and reported, NOT implemented (need review). No commit/push. See
`spikes/s9_pipelined_transport/{RESULTS,PLAN,MANIFEST}.md`.

**Phase: S9-V0-R3 STATIC REPAIR COMPLETE; SEQUENTIAL DYNAMIC PHONE RUNTIME
MECHANICS PASS; LIVE SCHEDULER BLOCKED.** Append-only schema/bundle v5 closes the static v4
holes reproduced for complete digest binding, exact manifest/segment membership,
layer and correctness I/O envelopes, SoC floors, snapshot causality, ticket/chunk
ranges, duplicates, and ReadyCertificate physical claims. Final v5 evidence is
24 schema fixtures, 28 bundle fixtures, and 34 red-v4/green-v5 checks, all green.
This remains immutable-snapshot coherence, not atomic live scheduling. Separately,
`examples/phone-pim/` now provides the first real trusted-localhost command path:
the phone accepts one ticketed sequential model stream, ACKs durable verified
prefixes, resumes after disconnect/restart, atomically publishes a content-addressed
read-only shard, loads and warms one complete dense FFN, executes HTP0 on bounded
activation commands, and returns results to an independent production Gemma4
oracle. Exact-final clean uploads pass on OP15 v81 and OP12 v75 at M=16 (rel-L2
2.92e-4 / 2.95e-4). Host suites are 22 protocol, 45 storage, and 45 real-process
integration checks; Android protocol/storage suites pass on both phones. This is
mechanics, not capacity: the stop-and-wait ADB-forwarded path is only 4.7-14.9
MiB/s versus separately measured `adb push` at about 216-262 MiB/s. Multi-model
residency, native pipelined transport, llama-server integration, live leases/
credits, capacity, and energy remain unimplemented. See
`spikes/s9_dynamic_weight_residency/V0R3_REPAIR.md` and
`spikes/s9_phone_pim_runtime/DYNAMIC_RESULTS.md`.

**Phase: S9-V0-R1 targeted suite closure COMPLETE; scheduler dispatch
certification BLOCKED.** An independent adversarial pass
found the official V0-R suites PASSED while the mechanics were STILL fail-OPEN, so V0-R
is relabeled `SUITES PASS; MECHANICS CERTIFICATION BLOCKED; CAPACITY UNPROVEN`. R1 freezes
V0-R as RED evidence (golden/v0r_historical/, replay af1501b6...), introduces a VERSIONED
R1/v3 contract (schemas/v3; v1+v2 untouched), and closes its targeted SIXTEEN cases with
red-before/green-after evidence (each imports the frozen V0-R code AND the R1 code and
shows V0-R has the hole, R1 closes it; 17/17). Validator (bundle_version 3): PRIVATE
per-call temp files (race-free), the dispatch island must exist, each satisfied tuple must
form one coherent chain WeightSet->CanonicalAllocation->PreparedImage->ReadyCertificate->
ResidencyLease matched on device/backend/boot/generation/ids/digests, DispatchDecision
must match the island, sticky/rebuildable needs a matching StateLease (stateless needs
null), and alias/live-lease sets are DERIVED from records with exact set equality +
refcounts + reclaimable (5 new stable codes, 26 total). Simulator: backend/generation-
qualified readiness (no HTP residency dispatched to GPU), stale_epoch -> DRAINING +
non-dispatchable, stale pipeline rolls back once + releases the contention domain + wakes
the queue, full epoch stack rechecked after D2H, per-request/session sticky state with
DEFERRED reset while pinned, bounded lane admission (rejected terminal on overflow),
horizon-bounded event loop, explicit verified-offset partial failures, link_drop
resumable(bulk)/terminal(activation/compute/d2h), server relief credited only after a
valid completion, and measured interference REJECTED as unsupported. Suites: 100 schema
fixtures (53 v1 + 33 v2 + 14 v3, both validators) + v2/v3 bundle selftests + 23 semantic +
26 PRESERVED frozen-V0-R behavior + 17 R1 red/green + 10 golden/preservation, all green.
R1 golden replay 9a69a7f6...; V0/V0-R goldens preserved and still reproducible. THEN a
SECOND independent adversarial pass found further validator and simulator holes. The
DRAINING lease now rejects, and the 2026-07-14 review also binds StateLease residency and
ReadyCertificate correctness to the exact dispatched chain. Six simulator defects were
repaired: shared-domain transfer ownership, phone-scoped bulk faults, deferred eviction,
single-owner sticky mutation, horizon cleanup, and admitted-only server GPU accounting;
per-event lane/ledger/pin conservation is asserted. `test_r1_holes.py` is now 16/16 and the
v3 bundle index is 18/18. This is not closure of the contract: eleven new mutation probes
still pass across DeviceInventory/expiry authority, executable identity, transport epochs,
and state-to-ledger binding. Freeze v3 and repair these in v4 before any dispatch claim.
S9 remains contract/simulator evidence with CAPACITY UNPROVEN. HEAD 933c722f6, nothing
committed. See `spikes/s9_dynamic_weight_residency/V0R1_REPAIR.md`.

**Phase (superseded): S9-V0-R (contract + simulator closure) COMPLETE.** Repaired the fail-OPEN
contracts and simulator mechanics the review found in S9-V0, under a VERSIONED v2
schema bundle (schema_version 2; the v1 schemas stay frozen). Record model: a NEW
CanonicalAllocation separates the one canonical byte-charge from backend prepared
images that ALIAS it (refcounted, drain-gated); PreparedImage identity now binds
preparation algorithm + source allocation + derived_payload_sha256 + sharing mode;
dispatch is an EXACT tuple set (required == satisfied); domain-separated digests for
island/cert/leases/allocation; a NEW CorrectnessCertificate binds island + kernel
route + backend build + profile-row + exact shape envelope; partial ranges get an
explicit contiguous/disjoint/overlap coverage policy; bulk frames bind
ticket/segment/chunk/offset/length/digest and a nonzero resume needs a verified prefix
digest. ONE strict bundle validator always runs JSON Schema + semantic + cross-record
with 21 stable error codes (rejects unknown kind/version, duplicate JSON keys, missing
records, mismatched digests). Simulator rewritten: every request reaches exactly one
terminal outcome (completed_phone/completed_server/fallback/rejected/timed_out, sum
asserted); pipelines/dispatches/completions are epoch-bound (boot/generation/route/
state) with stale drops + fail-closed stale-completion rejects; horizon + bounded
prefetch/server/lane queues; real multi-device selection over PER-DOMAIN links
(separate USB buses additive -- replaces V0's controller/phone-count division);
reserve-on-acquire + rollback for UFS/LPDDR/derived/scratch/state with persistent
sticky state; resume re-sends only the remaining range while retaining link ownership
and counting retry bytes; the dimensionally-invalid causal score is replaced by a
documented LEXICOGRAPHIC integer objective; interference applies ONLY over the actual
overlap window; D2H is an explicit completion blocker. Every repair has red-before/
green-after evidence: sim/test_regressions.py imports the FROZEN V0 module and the
repaired one and shows V0 has the bug and V0-R is correct on the same scenario (14/14).
Suites all green: 86 schema fixtures (53 v1 + 33 v2, both validators) + 15 bundle + 23
semantic + 26 sim + 14 red/green + 6 cross-process golden-replay (+mutation). V0-R
golden replay `af1501b6...`; frozen V0 replay `64fcad3b...` preserved as regression
evidence. Relabel: V0 = SUITES PASS / MECHANICS NOT YET CERTIFIED / CAPACITY UNPROVEN;
V0-R = SUITES PASS / MECHANICS CERTIFIED / CAPACITY UNPROVEN (device rates symbolic;
directional H2D/D2H decomposition is the S9-V1 blocker; no real-trace claim until S8
Gate A). Nothing committed; HEAD 933c722f6. See
`spikes/s9_dynamic_weight_residency/V0R_REPAIR.md`.

**Phase (superseded): S9-V0 (dynamic weight residency for a PIM-style phone accelerator --
contracts + schemas + deterministic simulator + tests) COMPLETE.** A separate
research track from S8: a HOST-MANAGED phone accelerator where weights are
prefetched to UFS, promoted to LPDDR, prepared for HTP/OpenCL, leased, and executed
under server commands (PIM-STYLE, not cache-coherent PIM; the host transfers weights
+ boundary tensors). A 7-agent read-only substrate audit (file/line-cited) found the
crux: the current tree has NO SHA-256 of any weight payload -- the only content hash
is FNV-1a 64-bit in ggml-rpc, served on a cache hit WITHOUT re-comparing bytes -- so
VERIFYING, the derived-image identity, durable atomic publish, and
generation-qualified share/prepack teardown are all NET-NEW around reusable byte
plumbing. Froze 15 self-contained JSON Schemas (ModelManifest, WeightSegment+chunks,
atomic WeightSet, backend PreparedImage binding 9 identity fields, IslandExecutable,
TransferTicket, fail-closed ReadyCertificate + DispatchDecision, ResidencyLease,
StateLease, DeviceInventory ledger, bounded TransportFrame, sim config/manifest),
two linked state machines, and four contracts (residency/transport/prefetch/sim).
Built a deterministic integer-us simulator: 8 baselines, a 40-550 MiB/s goodput sweep,
the causal residency score, and the never-wait candidate; byte-identical replay
(64fcad3b...). Tests: 53 schema fixtures + 23 semantic (both validators, 0 fail) and
28 sim checks (all ten required behaviors: hash/partial recovery, dup/stale/reorder
rejection, lease-safe eviction + exact byte ledger, no dispatch from on-disk, no
live-state eviction, activation-preempts-bulk, unknown-profile/unsupported-backend
fail-closed). Both phones now negotiate USB 3.2 Gen 1 (5 Gbps) on separate root
buses. Verified 1 GiB ADB staging measured H2D medians 215.9/261.9 MiB/s
(OP12/OP15) and D2H 189.8/226.8 MiB/s; concurrent fleet makespan rates were
409.4/392.6 MiB/s. These include filesystem ingestion effects and are not raw
transport rates. S9-V1 must add per-device directionality, contention domains,
no-double-count staged paths, and D2H result transfer. MECHANICS VALIDATED;
capacity + energy UNPROVEN (remaining
rates symbolic; V0 link topology incomplete; no real-trace claim until S8 Gate A).
Nothing committed; HEAD 933c722f6. See
`spikes/s9_dynamic_weight_residency/`.

**Phase (superseded): S8-V0b-P0 (source pinning + final Gate-A input contract) COMPLETE; the
normalizer + server-only replay are NOT started.** Both real sources are now
fetched OUTSIDE git, inspected by a full pass, and pinned: BurstGPT v2.0
`BurstGPT_3.csv` (231682327 B, sha256 2299986a..., 5344021 rows, CC-BY-4.0,
387963/7.26% zero-response failures KEPT) and RAGPulse `data/0_trace.jsonl`
@ commit 99a62769... (1923473 B, sha256 cd371571..., 7106 records, MIT). Exact
parsing/mappings are frozen in `configs/{burstgpt,ragpulse}.config.json` (validated
by `schemas/source_config.schema.json` under both validators), with a per-source
timestamp policy discovered by inspection (BurstGPT require_nondecreasing / 0
inversions; RAGPulse sort_stable / 1 inversion). Quantiles are rational-integer,
the mix formula + hash preimages + server-only replay are frozen, and a semantic
manifest validator was added. MW1 atlas and the oracle stay BLOCKED until Gate A
passes.

**Phase (superseded): S8-V0a-R2 (executable machine-validated schema contract)
COMPLETE.** The V0a-R draft was machine-hardened into an executable Gate-A
contract: 15 self-contained JSON Schemas that validate standalone under both
`jsonschema` 4.10.3 and `ajv-cli@5` (no preloaded refs), a fail-closed request /
trace-manifest / profile-row / route-action schema set, 43 fixtures (19 valid +
24 adversarial) and a `run_schema_tests.py` runner that exercises BOTH validators
(0 failures, exit 0). Gate-A normalization is frozen (aligned load windows, source
parsing, explicit mix offsets, canonical bytes, full hash binding, deadlines kept
null). The DECISION_CONTRACT + decision/action/lease schemas are explicitly
DRAFT-BLOCKED-BEFORE-V0c. MW1 atlas and the oracle stay BLOCKED until Gate A
passes.

**Phase (superseded by the line above): S8-V0a-R (contract repair after review)
COMPLETE.** A review returned 10 blocking findings on the V0a draft (S6
over-certified, prose-not-schemas, underspecified determinism, "where" not
"what", non-deterministic objective, one-layer overgeneralization, second service
not executable, unsafe REUSE, BurstGPT metadata, unbounded admission). All 10 are
repaired as documentation (versioned JSON Schemas, `NORMALIZATION_SPEC.md`,
DAG-cover + compound-action schemas, frozen objective + p95 formula + tie-break,
downgraded S6/substrate evidence, `EMBEDDING_MODEL_FUNNEL.md`, corrected trace
metadata, finite admission credits). MW1 atlas and the oracle stay BLOCKED until
Gate A passes.

**Phase (superseded by the line above): S8-V0a (inspect + contract) COMPLETE.**
The primary target is mixed-workload reverse offload, not a fixed Gemma
partition. Phones first download, verify, load, and warm model shards, then
advertise READY operator islands on HTP and/or GPU. The host decides what, how,
and when to offload using server resource pressure, SLO risk, weight/state
residency, measured interference, thermal/link state, and later total-system
energy.

**V0a froze the contracts and audited the substrate (no code ported, no gates
run):** six documents under `spikes/s8_operator_island_affinity/` --
`S6_EVIDENCE_AUDIT.md` (which S6/S7 results may enter the atlas), file/line-cited
`SUBSTRATE_AUDIT.md`, `TRACE_SOURCE_AUDIT.md`, `SCHEMA_CONTRACT.md` (v1 frozen),
`DECISION_CONTRACT.md` (slow/fast loops), and `ATLAS_MATRIX.md`. Gate B is NOT
met yet: a certified-correct decode island exists on both phones (FA-off path)
but has no 7-process latency row, and no RAG embed/rerank or vision-encoder
island is profiled -- a second eligible service class is the top MW1 gap.

**Primary systems hypothesis:** cross-workload slack pooling can move a
coordinated set of deadline-flexible islands to phones, improve server batch
formation, release GPU-ms/HBM, and eventually create a real server low-power
window. Scattering isolated operators while the A6000 remains in the same power
state is not an energy saving. Capacity and energy are separate claims.

**New sources of truth:** `MIXED_WORKLOAD_DESIGN.md`, `WORKLOAD_TRACES.md`, and
`NEXT_PLAN.md`. The first bounded experiment is
`spikes/s8_operator_island_affinity/PLAN.md`: normalize real traces, build a
resident-island profile atlas, and run an offline capacity oracle before any
live scheduler or VQ port.

**Reusable substrate:** route `A0` is the live 12B OP15 -> OP12 -> A6000
pipeline with local KV; GGUF sharding and per-tensor HTP/OpenCL weight sharing
work; static HTP batching works at tested shapes; raw TCP stages and dual HTP/GPU
workers exist. The old fork also contains a process-local VQ byte table,
CONWIP, co-execution harnesses, and an experimental HEFT queue. These are audit
inputs, not a distributed scheduler.

**Evidence constraining the new design:** S3 row split failed; S4 GPU attention
failed at realistic context; S5 isolated phone operators did not add useful
A6000 throughput; S6 overlap is only saturated evidence and its FFN result is
incomplete; S7 ragged HMX is only an isolated kernel pass. Candidate islands
must amortize transport/state and use certified kernels.

**Trace decision:** no one public trace contains real arrivals, all modalities,
true priority, and true deadlines. Use unchanged real scenarios (BurstGPT/Azure
LLM, Azure LMM, RAGPulse) plus a clearly labeled deterministic superposition.
Never use observed completion latency as a deadline.

**Energy remains BLOCKED.** No phone/server/link energy claim is authorized
until MW5 validates synchronized physical boundaries and the A6000 power-state
break-even interval.

---

## S38 matched server-only RAG baseline - `2026-07-22 EDT`

`MATCHED_SERVER_C0_PASS; PHONE_ASSISTED_CONTROLS_NOT_RUN`. Built an exact-token
BGE index over all 609 MultiHop-RAG documents (7,008 chunks), then froze a
deterministic 320-request cohort from the dense-512 RAGPulse/MultiHop trace. The
cohort covers every question/evidence stratum, carries 943,165 requested input
and 79,685 requested output tokens, and replays 151.55 s of offered arrivals.

One A6000 and one RTX 4060 Ti each ran the same all-local pipeline and exact
model hashes: BGE-small F16 embed, cosine top-20, bge-reranker-base F16 top-6,
and two-slot continuously batched Gemma-4 12B IT Q8_0. A6000 wall time was
612.25 s at 0.523 req/s; desktop wall time was 1,062.10 s at 0.301 req/s. Both
answer EM values were 55.94%. The offered 2.112 req/s overloaded the hosts by
4.04x and 7.01x. Corrected response p95 values are 443.86 and 869.22 s; the
earlier harness field that omitted executor queue time was replaced before the
final runs by separate queue, service, and response fields.

The matched validator proves all 320 request identities, model/index/trace
bindings, manifest hashes, and timestamp equations. A6000 is 1.735x faster in
requests/s. Cross-GPU retrieved chunk sets agree 97.81%, reranked sets 97.50%,
and normalized answers 86.56%, while aggregate recall and answer quality remain
nearly identical. No phone, SLO, GPU-energy, or total-energy claim is made.

## S26 priority-safe matched physical scheduler - `2026-07-21 EDT`

`S26_PRIORITY_PHYSICAL_PASS`. Added a measured request-level controller with
strict priority, same-batch CUDA-relief admission, latest-start bounds, exact
resource credits, immutable route epochs, and grouped completion. Isolated
physical R0/R2 B4 profiles made admission like-for-like: R0 used 313,949 us of
CUDA work and R2 used 208,875 us. The final matched run used identical grouped
execution in control and treatment; it completed all 12 requests with zero SLO
misses and reduced selected CUDA compute 24.75% while preserving P0 p95. The
low-priority makespan increase to 4.301 s is recorded as the cost. GPU-board,
phone, network, and total energy remain unmeasured.

## S15 persistent OP15 plus A6000 tail: two real B32 exchanges pass - `2026-07-18 EDT`

`PERSISTENT_OP15_B32_HOST_AND_PHONE_PASS_ENERGY_UNKNOWN`. Added opt-in
`pipedriver --persistent-jsonl`: after one model/context load, the selected
A6000 tail accepts bounded commands with a strictly increasing launch ID,
prompt, exact request count, token limit, and DETACH/STOP ending. It reuses the
existing B32 pipeline, clears host and phone KV per exchange, suppresses human
stdout, emits one canonical result, and closes stderr attribution with a
launch-bound `PERSISTENT_DRIVER_EXCHANGE_END` marker. Invalid input, incomplete
result/placement, compute failure, or DETACH failure stops the worker.

The real OP15 `[0,8)` plus selected-A6000 `[8,48)` gate ran two B32 exchanges
through one host PID 3788992 and one phone PID 19868. DETACH then STOP passed;
64/64 requests matched current same-batch CUDA tokens; each produced eight
tokens. Exchange times were 2.798729 s and 2.598388 s. Each host session
observed 9,288 CUDA0 compute nodes and no fallback; each phone session observed
HTP0 compute with CPU only GET_ROWS, contiguous 384/768 steps, and zero missing
buffers. HMX temperature was 30.2-38.7 C. CPU/CUDA/Android builds pass; C++ seam
tests 9/9 and independent evidence tests 9/9 pass.

Scope remains mechanics-only: the runner writes directly to the C++ JSONL
contract, repeats one synthetic prompt, and does not traverse the typed
`PersistentPreparedTransport`/`StageNetSessionAdapter`. Selected-GPU, phone,
USB, server-wall, and total-system energy are UNKNOWN. The next gate is the same
two physical exchanges through the typed launcher, followed by repeated frozen
BGE+Gemma matched energy windows.

---

## S15 persistent stagenet sessions: DETACH/STOP protocol + reset-exactness gate - `2026-07-18 EDT`

`PERSISTENT_SESSION_MECHANICS_PASS_PHYSICAL_ENERGY_NOT_RUN`. Answers the CP-H
"next gate": persistent stagenet sessions that detach clients without unloading
weights. Only `examples/layersplit/layersplit.cpp` changed; llama-graph,
ggml_backend_sched, and all kernels untouched.

- Added opt-in `STAGE_DETACH=-7`: the resident `stagenet` worker resets
  request-local KV, emits a per-session `SESSIONCERT` (v2: contiguous session_id,
  worker pid/boot-nonce, device_boot_id, HTP0 layer range, placement tally, reset
  ack), acks, closes only the client, re-`accept()`s, and keeps weights/backends
  resident. `STAGE_STOP=-1` drains + terminates (byte-unchanged). Host driver
  gets `--session-end detach|stop` (default stop). HELLO wire response left
  byte-identical (v1) so legacy/serial routes are unaffected; version 2 is
  advertised only in the session cert.
- Rebuilt host (build-cuda, clean) + Android arm64 (snapdragon docker, `--force`,
  identical binary for both phones; v75 skel op12, v81 op15). Froze every executed
  binary/skel/script into `persistence/artifacts/` + SHA256SUMS BEFORE hashing;
  deployed to `/data/local/tmp/ls-s14-persistent/` (ls-s14-cpe untouched);
  on-device sha == frozen sha `d26075bc...`.
- Real-device gate (`persistence/run_persistence_gate.py`): two resident
  `stagenet [0,6)` workers (OP15 v81 + OP12 v75, HTP0) + host parallel-head
  shared tail `[6,48)` on the selected A6000; 7 sequential B1 sessions, 6 DETACH +
  1 STOP (Tail B2 NOT retried). `certified:true`, problems none: all 7 rc=0;
  one resident pid/nonce per phone across all 7 sessions, terminates only on STOP
  (`exit after 182 steps`); per-stream token ids identical across all 7 sessions;
  every cert SCHEDULED_PLACEMENT_OK, missing_buffer=0, HTP0-only + declared
  GET_ROWS on CPU.
- Isolated tail-independent proof: a wire client drove each worker through
  6 DETACH + 1 STOP with a fixed decode; raw head hidden-state bytes bit-identical
  across all 7 sessions on both phones (op15 `bada7465...`, op12 `0d5bf42e...`).
- MECHANICS only: no phone/server energy, latency, throughput, or shared-tail
  mono-correctness claim. Details in
  `research_dev/spikes/s14_mixed_streaming_scheduler/persistence/RESULTS.md`.

---

## S14 energy Stage B + C: deep-head phone feasibility ceiling + mixed GPU-board saving - `2026-07-17 EDT`

Measured the whole A->B->C chain on real hardware (OP15 Hexagon v81 / HTP0 + one
A6000). `research_dev/spikes/s14_mixed_streaming_scheduler/energy/`, SHA256SUMS
frozen, no commit.

**Stage B (device, MEASURED) - how deep a head a phone can actually run.** For
gemma-4-12B-it-f16, drive a phone `stagenet` head `[0,k)` on HTP0 (placement cert
on) with the A6000 `pipedriver` tail `[k,48)`; reuse existing binaries, do NOT
touch the frozen S11-E0 harness.

| head `[0,k)` | placement | HTP0 weights | resident | phone head p50 | tail p50 | tokens |
|---|---|---:|---:|---:|---:|---|
| `[0,2)` | SCHEDULED_PLACEMENT_OK | 855 MiB | 2775 MiB | 443 ms | 292 ms | match |
| `[0,6)` | SCHEDULED_PLACEMENT_OK | 2599 MiB | 4519 MiB | 783 ms | 270 ms | match |
| `[0,8)` | SCHEDULED_PLACEMENT_OK | 3454 MiB | 5374 MiB | 934 ms | 259 ms | match |
| `[0,10)` | DSP_QUEUE_ABORT | 4309 MiB (loaded) | -- | -- | -- | -- |
| `[0,12)` | DSP_QUEUE_ABORT | 5198 MiB (loaded) | -- | -- | -- | -- |

- `[0,2)/[0,6)/[0,8)` fully certified: ~100% HTP0 compute (only the f16 token_embd
  GET_ROWS on CPU, declared), `missing_buffer=0`, head+tail == mono token-for-token.
- **Max feasible single-phone head = `[0,8)`.** `[0,10)/[0,12)` LOAD but abort on
  the first forward: `ggml-hex: dspqueue_read failed 0x2e` in `flush_pending` -- a
  ~4 GiB (2^32 B) cap on ONE HTP weight buffer (3454 MiB passes, 4309 MiB fails).
- Two fixes found: FA must stay ON (forcing `GGML_DECODE_NO_FA` hangs the
  global-attn layers >=5 on v81); batched multi-seq HTP decode is a separate S1
  hang, so measured single-stream (matches the frozen GREEDY_SINGLE_STREAM method).

**Stage A ext (A6000, MEASURED).** Added `[0,8)`: 730.0 vs 859.9 mJ/tok =
**15.1% GPU-board energy saved**, HBM -3454 MiB (== the HTP0 buffer). k=6 reconfirms
11.2%. So the realisable single-phone saving is 15.1% at `[0,8)`, NOT the 22.8%
`[0,12)` ceiling (that head DSP-aborts on one phone).

**Stage C (DERIVED from measured).** Fold the measured per-token energies onto the
real mix-v1 workload (21,595 decode tokens, 87.9% generation) at `k*=[0,8)`. Power
is flat so `saving(f)=f*s(8)`. Routing all generation to the phone head realises
**13.3% mixed-workload A6000 GPU-board decode saving** (server-only 18,570 J ->
16,103 J); ceiling 15.1% if rag_qa also offloads. Realisable only at a RELAXED SLO
(phone route ~1193 ms/req serial vs ~292 ms server-only) and as an OVERLAP ceiling
(idle-wait not modelled).

**Net:** the phone genuinely runs a certified, token-correct deep head and the
A6000 spends measurably less energy -- but a single phone caps at `[0,8)` = 15.1%
(13.3% on the mix), not 22.8%. Reaching 22.8% needs the two-phone split
(op15 `[0,8)` + op12 `[8,12)`, a 4-layer ~1.8 GiB mid slice under the cap).

**Stage D (device, MEASURED) -- is the overlap real? does it save?** Two live
processes + NVML: GPU0 runs a saturated full-model backlog ("server busy"), OP15
runs `[0,8)` heads concurrently (tail on GPU1 so GPU0's own work is byte-identical).

| GPU0 board | energy/tok | tput | power |
|---|---:|---:|---:|
| phone IDLE (control) | 799.1 mJ | 369 tok/s | 294 W |
| phone `[0,8)` decoding ‖ (treatment) | 802.8 mJ | 367 tok/s | 295 W |

- **Overlap is CLEAN**: GPU0 changes -0.3% tput / +0.46% energy/tok with the phone
  pipeline live -- inside NVML noise. The phone runs alongside a busy server for free.
- **But throughput-bounded**: the phone sustains 8.2 tok/s (`[0,8)`, single-stream,
  token-correct) vs the A6000's 369 tok/s => carries only **2.2%** => realised
  GPU-board saving at a saturated server = **0.34%**. The 13.3% Stage C ceiling needs
  the phone to carry 87.9% of tokens; it carries 2.2%. Full offload at saturation
  needs ~45 phones (S5 additive-capacity, energetically). Batched HTP heads (S1 hang)
  or a slower/edge server or the light-load regime (A6000 idle-dominated, unmeasured)
  are the only ways one phone matters. **Answer: overlap yes + clean; savings real
  per token but ~0.34% realised with one phone against a busy A6000.**

**Stage D addendum -- batching the phone head (MEASURED).** Since the A6000 is ~45x
faster single-stream, batch the phone `[0,8)` head (decode is bandwidth-bound: batch
B loads the 3.4 GiB weights ONCE, emits B tokens). Measured on OP15/HTP0, FA on:
8.4(b1) -> 29(b4) -> 39(b8) -> 66(b16) -> **112 tok/s (b32, 13.4x)**; forward time only
118->285 ms across b1->b32, still scaling at 32. The initial "token mismatch vs mono"
was a WRONG-REFERENCE artifact, not a bug: batched greedy decode is not bit-identical
to single-stream (float non-associativity flips argmax at near-ties) -- the A6000 full
model ALONE shows the identical batch-size-dependent divergence, the phone route at
batch 2 == GPU full model at batch 2 exactly, batch 1 is bit-exact, and output is
coherent text at every batch. Certified. Batching lifts one phone from 2.2% -> **30%**
of the A6000 (realised ~4.6%/phone), two phones on hand -> ~61% -> **~9.1% realised**,
approaching the 13.3% mix ceiling; fleet for full offload drops from ~40 phones to ~3.

**Stage D addendum2 -- extended batch sweep to the binary cap (MEASURED). - 2026-07-18.**
Pushed `[0,8)` past b32 to find saturation (`batch_sweep_ext.py` ->
`batch_sweep_ext_result.json`): **b48 = 148 tok/s, b64 = 167 tok/s (19.9x
single-stream)**; forward 285->324->383 ms b32->b48->b64. Returns diminish (+32%
then +13% -> compute-bound, near saturation); weight buffer fixed at 3454 MiB,
`missing_buffer=0` at every batch. b32 re-measured at 112.5 == the prior 112.2, so
prompt length does not affect decode throughput. **b64 is the binary ceiling**:
`layersplit.cpp` rejects `--driver-batch > 64` and caps `n_ubatch` at
`min(n_batch,512)` (so `prompt_tokens*batch <= 512`; b64 needs a <=8-token prompt);
b96/b128 need a host+phone rebuild. Overlap math at b64: one phone carries
167/369 = **45%** of the A6000, and the TWO phones on hand carry **~90% -> ~13.7%
realised** GPU-board decode saving -- at/above the 13.3% mix ceiling (itself capped
by [0,8) depth s(8)=15.1%). So the two batched phones essentially cover full
decode-head offload against a saturated A6000; the single-stream ~40-phone fleet
collapses to ~2-3. Phone ENERGY still UNKNOWN. Frozen (SHA256SUMS, 16 artifacts).
No commit.

**S14 CHECKPOINTS A-D: BGE second service + priority-differentiated offload (MEASURED). - 2026-07-18.**
Brought up BGE embedding (`bge-small-en-v1.5-f16`, sha 4cd429b8 == frozen CP0 catalog pin)
as the second executable service class and ran the priority-differentiated mixed workload.
Tooling: added an env-gated `BGEPROF` bench + `PLACEMENTCERT` (reusing layersplit's cb_eval
tally) to `examples/embedding/embedding.cpp`; built for CUDA + Android-hexagon (docker
snapdragon toolchain) from the current checkout; deployed protocol-matched binaries
(`d44adb3f`) to both phones.
- **CP-A (A6000 BGE atlas, 0 fail):** 7 procs x 20 reps; measured throughput knee (policy
  `throughput_knee`) = batch 16 / 2 / 1 at seq 31 / 132 / 499. Long seq is compute-bound at
  B1 (batching hurts); short seq wants B~16. The measured knee is the scheduler input; the
  weight-only AI is an optimistic upper bound only (does NOT classify -- FA-off does masked
  cross-sequence attention over the whole physical batch). Cosine 0.999999-1.0.
- **CP-B (phone BGE atlas, 0 fail):** same shapes on OP15/v81 + OP12/v75 HTP0; every shape
  `SCHEDULED_PLACEMENT_OK`, 296 HTP0 nodes, 0 missing-buffer, only declared GET_ROWS on CPU,
  cosine 0.9959-0.9978. Fills the empty p50/p95/p99 on catalog island `bge_encoder_0_12`
  (append-only; catalog not mutated) -> island now scheduler-eligible. Compute-bound on-device.
- **CP-C (3-device Gemma route CERTIFIED):** OP15 [0,8) -> OP12 [8,12) -> host [12,48).
  `pipe3_device.py` fail-closed; certified batch-1 run passes 7/7 gates (both stage certs OK,
  ranges match, token-correct). OP15 3712 HTP0 + 16 CPU(GET_ROWS); OP12 1968 HTP0 + 0 CPU.
  Negative-gate suite 10/10. Label: token-correct MECHANICS only.
- **CP-D (priority-differentiated P0 vs P2, digest-pinned):** hi-pri BGE (compute-bound) stays
  on A6000 at knee batch; lo-pri Gemma-12B decode (memory-bound) at max batch, P0 full [0,48)
  vs P2 tail [8,48) (head offloaded). `power_frontier_policy` drives priority order, roofline
  batching (BGE->knee, Gemma->max), compat isolation, and the phone->server BoundaryCertificate.
  MEASURED selected-A6000 paid-window energy: **P0 6160 J -> P2 5411 J = -12.16%** (Gemma
  764.5 -> 646.7 mJ/tok, -15.4%) while **hi-pri BGE p50 preserved (3709 -> 3754 us, +1.2%)**.
  P1/P3 (power-cap plans) UNMEASURED -- no controllable A6000 power state on this host. Phone/
  USB/total-wall energy UNKNOWN; GPU_BOARD is the strongest boundary. Frozen: RESULTS_CP_ABCD.md
  + SHA256SUMS (27 artifacts). No commit.

---

## S11-E0 board-energy harness hardened; host builds + placement cert live; Android BLOCKED - `2026-07-16 EDT`

Hardened the S11-E0 selected-A6000 GPU-board energy diagnostic harness so a
measured timeline is fail-CLOSED on every piece of evidence recomputed from its
hashed on-disk bytes, then adversarially red-teamed the whole change set. No
measured acquisition was run; `formal_claim` stays `NONE`; `PHONE_ENERGY_UNKNOWN`
and `TOTAL_SYSTEM_ENERGY_UNKNOWN`.

Evidence pipeline (each artifact reopened once as immutable bytes, SHA-256 +
count verified, then recomputed; aggregation consumes the recomputed validity,
never a stored boolean a slot wrote about itself):

| Evidence | Recomputed gate | Fail-closed on |
|---|---|---|
| Power (+limit) | ZOH energy, >=100 in-window updates, <=250ms gap, power-limit invariant from bytes, uncertainty from the recomputed limit | metadata-only limit forgery, mixed 16-slot limit |
| Process | continuous selected-GPU compute-app monitor; bounded probe; errors persisted into the hashed bytes | any non-driver PID, coverage gap, monitor error |
| Thermal (treatment only; control NOT_APPLICABLE) | on-device OP15 logger (no ADB in the paid window), boot id + monotonic uptime, bracketed continuous coverage | status!=0, empty sensors, gap, logger error, wrong phone |
| Placement (CP1.5) | executed backend-placement certificate per run | missing/duplicate/zero-compute/CPU-fallback/wrong-backend |

CP1.5 executed-placement certificate: the smallest change confined to
`examples/layersplit/layersplit.cpp` -- an OBSERVE-ONLY ggml eval callback wired
through the public `cb_eval` seam (no edits to gemma4.cpp / llama-graph.cpp /
ggml_backend_sched). Returning false at `ask` keeps the scheduler batching each
split (no forced per-node execution); it reads each node's realized output buffer
`t->buffer` and tallies the heavy GEMMs vs copy/metadata. Host-validated: a
monodriver control on CUDA emitted `observed_backends:["CUDA0"],
cpu_fallback_nodes:0, status:PLACEMENT_OK` over 31584 GEMMs, and the runner
parsed/evaluated/reintegrated the real cert as valid.

CP2 builds: host CUDA / CPU / ASan+UBSan release **PASS** (rc=0; the C++ compiles
under all three; binary SHA-256s recorded). Android release **BLOCKED** -- the
toolchain is gone from this host (NDK r28b removed, Hexagon SDK `6.4.0.2` removed,
`build-s11-android` root-owned). The staged phone binary predates CP1.5, so the
on-device treatment readiness pair cannot run and no 90-120s request count / p95
SLO is frozen. Not fabricated.

Adversarial review (5-dimension find+verify workflow, 25 agents): 6 confirmed
findings, **0 on the C++ placement instrumentation**. Fixed: `reverify_pairs` now
CONJOINS the runner's stored validity so runtime-only vetoes (E_SAMPLER, done-
boundary contamination, phone-boundary) survive the byte recompute -- and a benign
single dropped power sample now invalidates its slot instead of aborting the whole
experiment; guarded a `headers[0]` IndexError. Added the missing load-bearing
tests (thermal-validity consumption, end-to-end relief label from real bytes with
per-evidence flips, main() exit-0 / exit-2 contract, thermal+process record
schema). Offline suite 34 -> 93, all green; red-before/green-after demonstrated
for every gate.

STOP after CP2 for review, per plan. CP3 acquisition (and the Android toolchain
restore the on-device readiness pair needs) awaits go-ahead.

---

## S9-V0-R2 v4 static bundle-coherence closure - `2026-07-14 EDT`

The 2026-07-14 review proved v3 still VALIDATES static bundles that disagree with authoritative
device state, executable identity, transport epochs, or physical reservations (11 fail-open
blockers). R2 freezes v1/v2/v3 byte-for-byte, adds a versioned **v4** contract (schema_version 4,
bundle_version 4) with a v4-only domain-separated digest family (`s9:<kind>:v4`) that additionally
binds `PreparedImage.source_weight_set_id`, `TransferTicket.issued_boot/gen`, and the three
`StateLease` reservation fields. `cross_record_v4` binds each DISPATCH to exactly one DeviceInventory
snapshot (present, boot/status_seq pinned, accepting/not-draining/not-stale/thermal-eligible, backend
supported), rejects expired leases at a decision timestamp, derives epoch/credit gates from records
(booleans are not authority), binds full executable identity, binds BULK/EXECUTE/RESULT frame epoch
stacks, and DERIVES the physical ledger single-copy from live records. All 11 blockers close with
red-v3/green-v4 fixtures (frozen v3 accepts each; v4 rejects with an exact code). Label:
`STATIC_SNAPSHOT_COHERENT` only -- NOT live dispatch certification.

**Same meta-lesson as R1: the first v4 pass was INCOMPLETE.** An 8-lens adversarial hole-hunt
workflow + skeptical verify found 6 more fail-open holes it missed (RESULT frames unbound; identity
fail-open when the manifest is absent; arch/soc/layout_version never cross-checked; DI-less devices
escaping the ledger; a cert attesting a footprint > device LPDDR; duplicate dispatch decisions per
request), plus 2 that the record-derived ledger already covered. All closed with 10 more red-v3/
green-v4 fixtures. Suites green: 116 schema + (v2 15 + v3 18 + v4 27) bundle + 23 semantic + 26/17/16/13
sim (untouched) + 33 v4 regression/effectiveness. 35 stable codes (9 new v4). Report:
`spikes/s9_dynamic_weight_residency/V0R2_REPAIR.md`. NOT a completeness proof; live dispatch =
V0c. Nothing committed.

---

## Two-level design adversarial correction - `2026-07-13 EDT`

Three independent read-only reviews found and corrected load-bearing gaps in the
first two-level draft. The causal oracle now has explicit non-anticipativity and
envelope publication times; future arrivals, output lengths, failures, thermal
state, and link rates are clairvoyant-only. Request end-to-end p95 may not be
formed by adding stage p95 values; V0c must freeze request/DAG SLO milestones and
a calibrated integer scenario method before the oracle runs.

The execution envelope now binds exact plan digests, server and phone weight
identity, physical tile ownership, expiry, fallback, and split correctness.
Dispatch atomically pins every residency/prepared-image tuple; DRAINING rejects
new pins. Compound state uses explicit disjoint state partitions or is ineligible.
Phone-exclusive HBM relief requires an already READY alternate; a reload restores
future capacity and never blocks the current request. Any fallback reservation is
subtracted from relief.

The capacity comparison now uses one closed cohort with terminal conservation.
Resource relief requires equal completed work and no worse rejection/SLO;
throughput requires at least 10 percent more completed work. A clairvoyant or
exact bound cannot authorize runtime integration: the deployable causal policy
must pass Gate C. The new v3 DRAINING-lease validator probe is recorded as an S9
dispatch-certification blocker. Documentation only; no runtime/schema/code fix.

---

## Two-level scheduler design decision - `2026-07-13 EDT`

The system scheduler is now explicitly split at the weight-readiness boundary.
The slow residency/envelope planner sees forecast operator demand, atlas rows,
server HBM/GPU pressure, phone inventories, and measured link/load costs. It
chooses whole or partial weight placement, prepared HTP/GPU images, ownership
mode, minimum-hold leases, background transfer, drain/evict, and an elastic set
of certified discrete split ratios. A planned placement is not READY; observed
verify, prepare, warm, correctness, ledger, and lease completion must publish it.

The fast execution scheduler owns the bounded virtual operator queue. For each
concrete DAG-ready island it chooses admission, bounded batch wait, committed
cover, READY whole-island/stage/split plan, and atomic lane/link/state/fallback
credits. It cannot fetch weights or invent a graph cut/split. Split shares balance
predicted completion including queues and boundary transfer, not equal bytes; an
HBM-relief policy may choose the largest certified phone share that still meets
the SLO.

The offline problem is modeled as finite candidate-plan selection plus residency,
precedence, memory, state, lane, link, batch, and deadline constraints. It is
NP-hard. The evaluation sequence is exhaustive tiny ground truth, independent
solution validation, pinned CP-SAT exact agreement, causal rolling-horizon oracle,
then a separately labeled clairvoyant upper bound. Solver use alone is not the
novelty; the hypothesis is joint future-demand-driven partial residency, elastic
execution envelopes, online batching/split selection, and measured server HBM or
capacity relief from command-driven phone memory.

Updated `MIXED_WORKLOAD_DESIGN.md`, added `TWO_LEVEL_SCHEDULER.md`, revised
`NEXT_PLAN.md`, and refreshed `README.md`. No runtime, model, backend, kernel, or
schema was changed. Capacity and energy remain unproven.

---

## S9-L0 dual-phone USB baseline and V1 plan update - `2026-07-13 EDT`

Replaced both phone cables and measured the resulting path without disrupting the
other user's ADB server (dedicated server on port 5038). OP12 path 6-2/Bus 006 and
OP15 path 8-3/Bus 008 each negotiate 5000M on separate SuperSpeed roots. A 1 GiB
incompressible payload, `adb -Z`, three solo reps, and SHA-256 verification produced:

| path | OP12 | OP15 | concurrent fleet makespan |
|---|---:|---:|---:|
| host -> phone file | 215.9 MiB/s | 261.9 MiB/s | 409.4 MiB/s |
| phone file -> host | 189.8 MiB/s | 226.8 MiB/s | 392.6 MiB/s |

This makes background residency useful on a seconds-ahead horizon (900 MiB is
about 4.2 s on OP12 and 3.4 s on OP15), but does not authorize request-critical
weight fetch. ADB push already includes file-ingestion effects, so inserting it as
V0 raw link goodput would double-count part of V0's UFS stage.

Plan updated for S9-V1: freeze V0 replay/results; version the schema to per-device
H2D/D2H evidence-bound profiles and contention domains; dynamically share only
active same-domain streams; add D2H result transfer; distinguish staged-file from
decomposed native-buffer paths; add a measured two-phone scenario; then measure
native transport, UFS, durable publish, hash, materialize, prepare, warmup, and
transfer/compute interference. Future 800/1000 MiB/s USB 10 Gbps points remain
unmeasured sensitivity only. Capacity and energy remain unproven.

Full evidence and exact elapsed samples:
`spikes/s9_dynamic_weight_residency/CURRENT_SLOW_LINK.md`. No runtime or simulator
code changed in this plan update.

---

## S9-V0 dynamic weight residency (PIM-style phone accelerator) - `2026-07-13 EDT`

New research track, separate from S8's mixed-workload atlas: a host-managed phone
accelerator that prefetches weights to UFS, promotes to LPDDR, prepares for
HTP/OpenCL, leases, and executes under server commands. PIM-STYLE (host transfers
weights + boundary tensors; the phone cannot read server memory). Contracts +
schemas + a deterministic simulator + tests -- no daemon/kernel/scheduler, no
energy/novelty/PIM-hardware claim, nothing committed.

Substrate audit (7 read-only agents, file/line-cited; 12 citations re-verified by
hand). The crux finding: NO weight payload is cryptographically verified anywhere.

| substrate | class | why |
|---|---|---|
| loader offsets / load_data_for | REUSE/ADAPT | locator good; mmap-alias + no hash + not transactional |
| GGUF split + shard_gguf.py | ADAPT | count-only completeness; no per-shard sha256/manifest/coverage KV |
| downloader (resume/rename) | ADAPT | zero fsync, zero payload sha256 (grep-confirmed), delete-before-redownload |
| ggml-rpc SET_TENSOR_HASH | REF-ONLY | FNV-1a 64-bit (not SHA), served on hit WITHOUT re-compare |
| per-tensor HTP<->OpenCL share | REF-ONLY/ADAPT | name-keyed, first-insert-wins, NO teardown -> stale-alias + leaks |
| OpenCL xmem prepack cache | REJECT | pointer-keyed, no eviction, deliberate leak, f16-accum ~1.86% rel_L2 |
| partial load (LLAMA_LAYER_*) | REJECT-arbitrary | gemma4-ONLY (load + graph); a hard eligibility gate |

Froze 15 self-contained JSON Schemas (validate under BOTH jsonschema 4.10.3 +
ajv-cli 5): ModelManifest, WeightSegment+chunks, atomic WeightSet, backend
PreparedImage (derived-image digest binds 9 fields), IslandExecutable,
TransferTicket, fail-closed ReadyCertificate + DispatchDecision, ResidencyLease,
StateLease, DeviceInventory ledger, bounded TransportFrame, sim config/manifest.
Two linked state machines (ABSENT..READY..EVICTING) + the dispatch eligibility
rule. Four contracts: residency / transport (bounded versioned LE frames, chunk
hashes, resumable verified ranges, fsync+dir-fsync+atomic publish, credits,
activation-preempts-bulk, epochs, drain-before-evict; does NOT reuse LayerSplit/
ggml-rpc framing) / prefetch (8 baselines + causal score + never-wait) / simulator.

Deterministic simulator (integer-us, stable order): server GPU queue+HBM, shared
USB controller + per-phone links, UFS/verify/LPDDR/prepare, bounded HTP+GPU lanes,
canonical vs backend-derived RAM ledger, activation/state/thermal/deterministic
failures, interference only-when-measured. Sweep 40/100/250/400/550 MiB/s; the
never-wait candidate keeps p95 ~5-13 ms and gains relief with goodput, while the
waiting diagnostic pays 3.5-23.7 s p95. Byte-identical replay 64fcad3b...

Tests: schema 53 (0 fail, both validators) + semantic 23 (0 fail) + missing-fixture
guard (exit 1) + sim 28 (all ten required behaviors: hash/partial recovery,
dup/stale/reorder rejection, lease-safe eviction + exact ledger, no on-disk
dispatch, no live-state eviction, activation-preempts-bulk, unknown-profile/
unsupported-backend fail-closed). Initial link capture showed both phones on USB
3.2 Gen 1 (5 Gbps); the newer S9-L0 entry above supersedes the deferred-goodput
status with measured ADB staging evidence. MECHANICS VALIDATED; capacity/energy
UNPROVEN (V0 rates/topology are not a measured capacity model; no real-trace claim). See
`spikes/s9_dynamic_weight_residency/`.

---

## S8-V0b-P0 source pinning + final Gate-A input contract - `2026-07-13 EDT`

**Sources pinned + inspected (outside git); contract frozen; nothing committed.**
Stopped before the normalizer / server-only replay.

- Task 1 -- real sources fetched to `/home/myid/zs89458/Documents/s8_sources/`
  (outside the worktree) and characterized by a FULL pass. BurstGPT v2.0
  `BurstGPT_3.csv`: 231682327 B, sha256 `2299986a...a43b8f`, 5344021 rows,
  CC-BY-4.0; services {Conversation log 233617, API log 5110404}; models {GPT-4,
  ChatGPT}; 387963 zero-response failures (7.26%, KEPT as burstgpt_failed);
  timestamps float-seconds, 0 inversions; API-log rows have blank Session ID ->
  null. RAGPulse `data/0_trace.jsonl` @ immutable commit `99a62769...` : 1923473
  B, sha256 `cd371571...801e65`, 7106 records, MIT; hash_ids keys
  [sys_prompt, passages_ids, history, web_search, user_input]; timestamps integer
  seconds, **1 non-monotonic pair**. -> `SOURCE_PINS.md`.
- Task 2/3 -- `schemas/source_config.schema.json` + validated
  `configs/{burstgpt,ragpulse}.config.json` freeze exact header/keys, column/value
  maps, session-blank->null, retrieved_chunks=len(passages_ids), namespaced
  cache_keys order, modality defaults, source_fields, and a const `gate_a` block
  (deadline/priority always null, provenance none). One authoritative contract
  (schemas+configs > NORMALIZATION_SPEC > SCHEMA_CONTRACT > informative prose); the
  "WORKLOAD_TRACES.md wins" cycle removed. Decimal quantiles -> rational-integer.
- Per-source timestamp policy: BurstGPT `require_nondecreasing`; RAGPulse
  `sort_stable` (the 1 inversion means a single global strict rule would wrongly
  reject it).
- Task 4 -- mix frozen: `t_mix=floor(t*scale_num/scale_den)+offset_us`, checked
  arithmetic, stretch vs compress stated, unique ranks, streams serialized by
  rank, provenance rewriting to semi_synthetic, rank-namespaced event IDs,
  duplicate-ID rejection.
- Task 5 -- hash preimages frozen (source/output/config/normalizer-code-bundle/
  component/sidecar/replay), each stating exactly which bytes it covers;
  artifact `kind=normalize` requires nonempty outputs + sidecar binding.
- Task 7 -- `validate_manifests.py` semantic validator + 15 fixtures (bin
  arithmetic, first<=last, quantile bounds, nonmonotonic<->policy, mix unique/
  sorted ranks + component binding, safe-integer bounds, normalize binding); fixed
  the previously impossible bin-index fixture (t_start = bin_index*W).
- Task 8 -- server-only replay DEFINED as a strictly structural reader/order/DAG
  pass (schema+hash validation, arrival-order preservation, service->DAG mapping,
  demand accounting); explicitly NOT inference, no synthetic text/profile/deadline/
  perf.
- Task 6 -- `run_schema_tests.py` hardened: pins+verifies jsonschema 4.10.3 +
  ajv-cli 5.0.0, prechecks existence/regular-file/JSON-parse/index-completeness,
  `--index` override PROVES a missing expected-invalid fixture fails (exit 1).

Test outputs: schema suite 51 fixtures (21 valid + 30 invalid) 0 failures exit 0;
semantic 15 fixtures (3+12) 0 failures exit 0; missing-fixture proof exit 1; both
configs valid under both validators. `RESULTS.md` updated. Datasets held outside
git. No runtime/model/graph/KV/scheduler/backend/kernel/S6/S7 edit; no commit.

---

## S8-V0a-R2 executable Gate-A contract repair - `2026-07-12 EDT`

**Documentation + schema/test only; nothing measured, nothing committed.**
Review-2 said V0a-R was a useful draft but not yet a frozen executable contract
(10 blocking findings). All repaired:

- Checkpoint 1 (schema bundle): all 15 schemas made self-contained (local
  `#/$defs` only) so each validates from its own path under `jsonschema` 4.10.3
  AND `ajv-cli@5 --spec=draft2020` with no `-r` preload. `request.schema.json` is
  fail-closed (all canonical keys required; bidirectional deadline/priority-vs-
  provenance; integer upper bounds; scalar-only `source_fields`; no unknown
  fields). `trace_manifest.schema.json` splits real (rejects `streams`) vs
  `semi_synthetic` mix (requires `streams[]`, each binding component trace +
  sidecar hashes, rank, rational scale, explicit integer offset).
  `profile_row.schema.json` makes `verdict:PASS` fail-closed (requires
  correctness=pass, fallback=none, n_proc>=7, non-null timing/memory/boundary,
  server_control, server_relief, post_transfer_slo_feasible, non-empty
  artifacts). Added `fixtures/` (19 valid + 24 adversarial) and
  `run_schema_tests.py` running BOTH validators: 15/15 schemas compile+load, 43
  fixtures 0 failures, exit 0 (nonzero on any unexpected result).
- Checkpoint 2 (Gate-A normalization frozen): single authority `schema_version` +
  provenance `semi_synthetic` across all live docs (WORKLOAD_TRACES,
  TRACE_SOURCE_AUDIT, SCHEMA_CONTRACT, schemas); row-index quantiles replaced by
  aligned 15-minute offered-token load windows (nearest-rank + full tie-break);
  frozen BurstGPT/RAGPulse parsing (encoding/BOM/newline/dialect/columns/grammar/
  monotonicity/exclusions, header verified vs pinned revision); mix offsets are
  explicit committed integers (SplitMix64 removed); canonical bytes forbid float/
  NaN/empty in trace records with checked integer arithmetic; Gate-A v1 keeps
  deadline/priority null (synthetic SLO deferred -- no MW1 dependency); full hash
  binding (source+revision+config+code+output+sidecar; mix binds every component).
- Checkpoint 3 (consistency): DECISION_CONTRACT + decision/action/lease schemas
  marked DRAFT-BLOCKED-BEFORE-V0c with 6 unresolved items; embedding funnel
  corrected (CLS+L2 pooling, HF-FP32->CPU reference chain, cosine + retrieval
  top-k, rerank score + ranking agreement, RAGPulse-no-text synthetic payloads,
  truncation coverage); Gemma atlas rows split into homogeneous SWA/global
  islands or an exact per-layer attention vector; stale audit fixed -- NO OP15
  decode artifact exists (dumps are device-ambiguous/OP12-consistent),
  oplayerprof/resdiff/dualengine/ffnmerge are component tools PENDING repairs, FFN
  boundary equality separated from weight-copy count, SUBSTRATE summaries match
  the downgraded detail, per-tensor sharing noted existing/works-single-load but
  unsafe across reloads.

`RESULTS.md` updated to V0a-R2. Next is V0b / Gate A ONLY. MW1 and the oracle stay
blocked until Gate A passes; DECISION_CONTRACT stays DRAFT-BLOCKED until V0c. No
graph/KV/scheduler/kernel edits; no commit or push.

---

## S8-V0a-R contract repair after review - `2026-07-12 EDT`

**Documentation-only repair of the V0a draft; nothing measured, nothing
committed.** A review found V0a was a useful draft but not a frozen executable
contract and returned 10 blocking findings. All repaired:

- Prose "schemas" replaced by 15 versioned JSON Schema files under
  `spikes/s8_operator_island_affinity/schemas/` (draft 2020-12, all valid, all
  `$ref`s resolve). Records SPLIT into immutable island descriptor,
  model-residency lease (slow loop), request-state lease (fast loop), and
  telemetry. `schema_version` is the single authority (no `trace_version`);
  provenance value is `semi_synthetic`.
- `NORMALIZATION_SPEC.md` (new): exact deterministic rules -- window selection,
  integer time scaling, total-order sort key, splitmix64 mix offsets, no-PRNG
  rule, canonical JSONL bytes, source+output hash binding, negative tests.
- `DECISION_CONTRACT.md` rewritten: added the DAG-cover partition rule (the
  "what"), compound `route_action`s (chain_a0 / corun_pair / merged_batch) with
  all-or-nothing multi-lane reservation, a frozen lexicographic capacity
  objective with explicit committed shadow prices, an exact predicted-p95
  finish-time formula, a total-order tie-break, and finite host+lane admission
  credits with explicit terminal outcomes (no unbounded wait).
- S6 evidence DOWNGRADED: the cited raw v2 records are absent; SERVICE is a
  single-stream (B=1) decode over positions 0-63 with no correctness check; FFN
  merge is an unequal-boundary mechanism signal; only a narrow blk.2 B1/C512
  causal_local_swa correctness POINT survives (does NOT certify a 48-layer
  island).
- Substrate classifications DOWNGRADED: socket helpers REUSE->ADAPT
  (SIGPIPE/deadline/typed-error); OverlapLeg/OverlapBarrier ADAPT->REFERENCE_ONLY
  (ignored-failure-reports-done + diagnostic data race); VQ ring
  REUSE->ADAPT-concept/REFERENCE_ONLY-code (publish-before-write +
  overwrite-live-slot).
- `EMBEDDING_MODEL_FUNNEL.md` (new): exact `bge-small-en-v1.5` embed +
  `bge-reranker-base`; support-first funnel (CPU ref -> op-support/no-fallback ->
  correctness -> latency) with the HTP-BERT-op-support risk stated up front; a
  new `embprof` harness spec (oplayerprof cannot measure RAG/vision).
- `TRACE_SOURCE_AUDIT.md`: BurstGPT_3 corrected to ~220 MB / ~5.34M rows,
  failures = zero response tokens; normalized traces bound to source+sidecar
  hashes.
- Atlas rows now scoped by exact layer_range + attention_class + graph_hash +
  shape_envelope; Gate B amended to require a matched A6000 control,
  post-transfer SLO feasibility, and measured server GPU/HBM relief. Gate B is
  NOT met.

`RESULTS.md` updated to V0a-R. Next step is V0b / Gate A ONLY (fetch pinned
sources, deterministic normalization, negative tests, server-only replay, compute
Gate A, then stop). MW1 profiling and the oracle stay blocked until Gate A
passes. No graph/KV/scheduler/kernel edits; no commit or push.

---

## S8-V0a inspect + contract checkpoint - `2026-07-12 EDT`

**Documentation and contract work only; nothing measured, nothing committed.**
Delivered six frozen documents under `spikes/s8_operator_island_affinity/`:

- `S6_EVIDENCE_AUDIT.md`: maps S6/S7 results onto atlas eligibility. INELIGIBLE
  as passing rows: SERVICE latency (FAIL +19%), FFN merge (LOWER_BOUND, 2 weight
  copies), v81 fused-FA (5.02e-3 marginal FAIL), all energy (DEFERRED). Preserved
  as correctness certificates: OP15 FA-off decode 2.95e-3 and OP12 v75 AUTO
  explicit-attention decode 3.61e-3 (both still need 7-process latency rows).
  Negative evidence (S3/S4/S5 fails, v75 broken FA) retained.
- `SUBSTRATE_AUDIT.md`: file/line-cited audit of current + historical mechanisms,
  each classed REUSE/ADAPT/REFERENCE_ONLY/REJECT. Four parallel read-only
  investigations covered llama-server queue/slot/cancel, layersplit TCP +
  dualengine workers, ggml-rpc + weight readiness (download verify/fsync/rollback
  ABSENT; per-tensor share has no version qualifier -> stale-alias across
  reloads), and the route2 VQ/CONWIP + Unifer 16-byte telemetry. Enforced the
  local-occupancy vs distributed-readiness split: neither prototype provides a
  reliable readiness/lease/manifest plane -- that is net-new.
- `TRACE_SOURCE_AUDIT.md`: BurstGPT v2.0 + Azure LLM24 (generation), RAGPulse
  (RAG DAG), Azure LMM25 (vision encoder). URL/release/license/fields/checksum
  procedure recorded; no dataset downloaded; observed vs synthetic
  (deadline/priority) fields classified.
- `SCHEMA_CONTRACT.md`: v1 schemas for request, service DAG, island, model
  manifest, phone capability, READY lease, profile row, route decision, artifact
  manifest. Rule: any null measurement makes a route ineligible; no silent
  estimation into a PASS.
- `DECISION_CONTRACT.md`: slow loop (placement/download/verify/warm/READY/RAM
  admission/lease horizon/eviction) and fast loop (admission/batching wait/island
  selection/credits/state affinity/fallback/reason codes). HTP and GPU are
  separate bounded lanes; co-run eligible only for a measured compatible pair;
  `energy_gain` reason code reserved and unusable until MW5.
- `ATLAS_MATRIX.md`: first experiment matrix (generation, RAG, encoder x
  controls C1-C7). Gate B NOT met; second service class is the top gap.

`RESULTS.md` updated to V0a-complete / Gates A-C NOT RUN. Stop for human review
before V0b (normalizer) / MW1 (atlas) / V0c (oracle). No graph/KV/scheduler/
kernel edits; no commit or push.

---

## Historical Design A/S7 status snapshot - `2026-07-11 EDT` (superseded as current direction)

**Phase: S7-V0 - ragged HMX decode tile skipping passes the isolated operator gate.** On OP15,
the default-off kernel prototype passes CPU-reference correctness and reduces C=512 attention by
1.49x for the deterministic ragged-prefix batch at both B=16 and B=32. Dense overhead stays below
1%. The next gate is a real Gemma layer driven by a real arrival/output-length trace; no scheduler,
end-to-end, or energy claim is authorized yet.

**Claim boundary corrected.** PowerInfer-2 already demonstrates mobile concurrent batched decode,
heterogeneous attention/FFN partitioning, and phone J/token. NanoFlow already pipelines decode
attention with projections across operation nano-batches, and variable-length attention is not a
new primitive. The open system question is whether trace-driven ragged HMX work elimination on
phone stages increases useful server capacity and reduces total service energy under latency SLOs.
Do not use "first" language.

**Worktree status.** S7 adds deterministic ragged cases to `test-backend-ops` and a default-off HMX
KV-block worklist. The pre-existing `layersplit.cpp` decode-FA toggle and
`ggml-hexagon.cpp` fused-FA/per-tensor-sharing edits remain uncommitted. No Gemma graph, KV, backend
scheduler, or production runtime has been changed.

**Phase: M2/M3 streaming pipeline LIVE on 12B + per-tensor sharing shipped; M5 energy still gated on a physical enabler.** Two wins today: (1) **the 3-device streaming pipeline runs the 12B end-to-end** — `op15[0,2) → op12[2,3) → A6000[3,48)`, persistent KV, incremental O(N) decode over TCP/USB, answered *"…capital of France?"* → **"Paris"** at ~10.5 tok/s (log below). (2) **per-tensor weight sharing** replaces the fragile whole-buffer size-match so a phone can hold **many layers at true 1× weight** (validated op15 L=8, log below). Below: the M5 crossover findings (energy blocked remotely).

**Phase: M5 crossover — measured; the verdict is nuanced.** All three requirements are BUILT and validated (dualengine + static batch + one-copy sharing, commits below). M5 tried to push many layers onto op15 to find an energy crossover vs the A6000 (0.15 J/tok). Findings:
- **Throughput scales LINEARLY with layers (reliable):** clean (xmem-off) op15 decode L=4 → **239 ms/round**, L=8 → **485 ms/round** = **2.03×** ≈ ~60 ms/layer. *(The earlier L=8 = 1717 ms was an xmem-ON RAM-thrash artifact — the `os8` prepack tile + dual models near-OOM'd; turning xmem off removed the cliff.)*
- **No defensible J/tok obtainable remotely:** battery coulomb reads **0** (phone Full, USB-powered); the USB rail shows a near-**constant ~1.7–1.8 W** regardless of L (a sustained-power ceiling, sign-inverted) — it cannot resolve per-layer energy. A real number needs a **physical unplug + WiFi-adb + battery-discharge slope** (⚠️ user's physical action). **Cannot claim a phone-vs-server crossover from this data.**
- **Req-3 one-copy sharing L-scaling limit — ✅ FIXED (per-tensor sharing):** the old whole-buffer size-match only worked for a one-buffer shard (Hexagon ~1 GB cap → 4 buffers vs OpenCL ~1.9 GB cap → 2 buffers diverge → 2nd-copy fallback → L=16 OOM). Now sharing is keyed by **tensor NAME**: Hexagon publishes each weight's `{fd,offset,size}`, OpenCL imports each distinct fd once and aliases every weight by name (zero own weight bytes; `set_tensor` skipped for aliases; name-miss lazily promotes one real buffer for compute/KV/token_embd). xmem untouched. **VALIDATED op15 L=8 [2,10): all 4 fds imported once, 0 promotions (100% aliased), correctness bit-identical to no-share.** Designed by a 17-agent workflow.
- **The L=8 correctness "FAIL" is fp accumulation, NOT a sharing bug (rigorously confirmed):** the harness compares batched(B=16) vs B-single **on the decode engine only** (never touches the shared prefill weights). rel_L2 = **2.755e-3** (< 1e-2 → passes L2); only **1/16** argmax-of-residual flips (a near-tie), tripping the strict `argmax==0` gate. **Identical rel_L2 whether import succeeded OR fell back to a 2nd copy** → the discrepancy is inherent 16-wide-GEMM vs 1-wide-GEMV fp over 8 layers; no cross-seq bleed, no sharing corruption.

*(prior M4)* **DUAL-ENGINE (one session, two backends) live on BOTH real phones.** `dualengine` mode in `llama-layersplit` loads the shard on two devices and runs **NPU decode (HTP0) concurrently with GPU prefill (GPUOpenCL)** on two threads. Its concurrent wall equals the longer co-running leg (op12 **1.76x**, op15 **1.92x** versus serialized leg sums), proving full wall overlap. The harness did not measure equivalent solo leg times, so it does not yet prove zero interference. Static **B=16 batched decode** is bit-correct (rel_L2 ~5e-4, 0/16), and the op15 S1 hang did not reproduce. One-copy weights are built via `--share-weights`.

*(prior)* **M2 done — gemma-4 12B fp16 live on the actual phones over USB.** `op15[0,2) → op12[2,3) → A6000[3,48)` answers *"…capital of France?"* → **"The capital of France is Paris."** Each phone stores ONLY its shard (op15 2.72 GB, op12 0.43 GB) via partial-load (`c615983dd`) + `shard_gguf.py`. Single-engine pipeline: **NPU 194 / CPU 219 / GPU 264 ms/tok**.

```
 EXPLORE ✅ ─── DESIGN ✅ ─── M0 🔵 ─── M1 ✅ ─── M2 ✅ ─── M3 🔵 ─── M4 ✅ ─── M5 🔵
 (benchmarks   (research_dev/  (ground-truth   static    inter-    batched   dual-    energy
  + reviews)    DESIGN.md…)     + S1/S2 spikes) pipeline  connect   decode    engine   crossover)
                                                        ▲ dualengine + 1-copy share here
```

| Milestone | State | Exit gate |
|---|---|---|
| Explore (benchmarks, reviews) | ✅ done | — |
| Design (`research_dev/`) | ✅ done | DESIGN.md + MILESTONES.md + README.md |
| **M0 — ground truth + vetoes** | 🔵 partial | `gguf_dump` 12B ✅; **S1** localized (HTP bug, not deadlock); **S2** shared-weights ✅ |
| **M1 — static pipeline correct** | ✅ done | 3-way head→mid→tail (cuts 2,3) **bit-exact** vs mono; k≤13 ceiling N-invariant |
| M2 — interconnect + overlap | ✅ done | dualengine wall == max(co-running leg); solo/co-run interference still needs H0 |
| M3 — batched continuous decode | 🔵 partial | static B=16 lockstep bit-correct, no hang; continuous `n_parallel` still HTP-bugged (S1) |
| M4 — dual-engine + one-copy | ✅ done | reqs (1)(2)(3) built + validated; per-tensor sharing validated through op15 L=8 |
| M5 — rebalance to a real win | 🔵 measured | linear throughput scaling ✅; **defensible J/tok blocked remotely**; serving capacity still needs KV/scratch/thermal budgets |

**Next execution plan:** replay a public request trace into dynamic batches, preserve each stream's
actual KV length, and run one real Gemma SWA layer at B=16/32. Compare rectangular HMX, ragged HMX
with the same admitted batch, and compacted smaller batches including compaction cost. Proceed to
continuous-pipeline integration only if correctness holds and the ragged path remains at least
1.10x faster on useful trace windows without hurting dense windows by more than 5%. The fleet
energy verdict still needs matched physical power boundaries and sustained thermal runs.

---

## Historical Design A decisions

- **Build Design A** (server→OP15→OP12→server pipeline) per direction. *(A review preferred hub-spoke B; we build A and mitigate its costs.)*
- **Phones own the FIRST layers, A6000 is the terminal stage** (llama.cpp pins `lm_head`+sampler to the last device). Baseline: OP15 = layers 0–1, OP12 = layer 2, A6000 = embed + 3–47 + head + sampler.
- **Backend assignment is measurement-driven.** Static HTP batched decode works. Sequence-affine HTP/OpenCL routes remain the coarse control; S4 tests GPU-owned attention/KV inside an HTP layer only after per-phone vetoes pass.
- **Weights: pre-downloaded mmap shards**, one copy shared by NPU (fastRPC/dmabuf) + GPU (OpenCL import). No runtime weight RPC.
- **Only the residual `[n_embd, n_tokens]` crosses the wire; KV stays on each stage.**

## Historical Design A architecture at a glance

```
 8 fps -> [A6000 embed] -> [OP15 L0-1] -> [OP12 L2]
            ^                                  |
            | next-token IDs                   v residual
            +-- [A6000 L3-47 + norm + head + sample]

   CURRENT PHONE DECODE: intact HTP   |   S4 CANDIDATE: HTP-A -> GPU-B/KV -> HTP-C
```

## Historical measured numbers

| Thing | Value | Source |
|---|---|---|
| op12 NPU prefill peak | **7.12 TFLOPS** @ batch 512 | roofline sweep |
| op12 GPU decode (stock) | ~0.40 TFLOPS | roofline sweep |
| op12 GPU decode (xmem+cache) | **0.95 TFLOPS** @ 128, 2.35× stock | xmem re-run |
| A6000 12B fp16 decode floor | **~0.15 J/tok** (net) @ batch 256 | user's table |
| op15 dualengine decode, B=16 | **239 ms/round @ L=4, 485 @ L=8** (~60 ms/layer, linear) | M5 sweep (xmem-off) |
| op15 dualengine power (USB rail) | ~1.7–1.8 W near-constant (ceiling, not per-L) | M5 sweep — **can't resolve J/tok** |
| phone J/tok crossover vs A6000 | **UNMEASURED remotely** — needs unplug + discharge | M5 blocker |

---

## Log

### 2026-07-24 EDT - S39 W2 direct mixed decode/prefill batch passes

- Added a bounded route-local batcher that orders decode rows first, fills
  remaining capacity with prefill, dispatches at the batch knee or earliest
  deadline, and fails closed on queue or compute errors.
- Executed Qwen3 14B Q4_K_M on OP15 `[0,30)` and OP12 `[30,40)`. The physical
  pattern was `80P, 16D+80P, 6x32D, 16D`.
- All 32 requests produced the same eight tokens as CUDA. Relay, session,
  placement, row, and activation-byte conservation passed.
- Recorded 23.219 s maximum completion and 20.169 s for the newly admitted
  cohort. This is not compared against W1 because ordering and device state
  were not held constant.
- The next gate is a matched sorted-versus-shuffled ordering experiment,
  followed by a bounded multi-inflight relay for inter-stage overlap.

### 2026-07-23 EDT - S39 W1 direct OP15-to-OP12 route passes mechanics

- Added an opt-in terminal relay on OP15 without changing the legacy StageNet
  wire path. It validates both worker identities, cut continuity, capacity,
  lineage, status, and sequence cleanup.
- Real OP15 `[0,30)` -> OP12 `[30,40)` B1 and B32 runs produced 264/264 exact
  Qwen token checks. B32 used one 160-row prefill batch and seven 32-row decode
  batches.
- The relay moved 7,864,320 B of B32 activation directly to OP12 and returned
  only tokens to the host. Both phone workers persisted across the two sessions
  and stopped cleanly.
- A corrected-binary B1 point was 5.653 s versus 5.696 s for one host-relay
  control. This is within run variation, so no latency win is claimed.
- Strict, ASan, and UBSan builds pass. The relay self-test covers the happy
  path, `n_batch` overflow, and invalid status capacity. The evidence reducer
  and eight mutation tests pass.
- Honest status:
  `DIRECT_CHAIN_MECHANICS_PASS_REPEATS_PENDING`. Qwen route readiness remains
  provisional and the two-model W0 gate remains blocked.

### 2026-07-23 EDT - S39 W0 trace/controller mechanics pass; routes blocked

- Implemented dense Qwen3 partial weight load, partial graph execution,
  injected activation input, and layer-filtered KV. Host full/split checks are
  exact at cuts 24, 30, and 32; Android and CUDA builds pass.
- Real OP15 `[0,30)` plus OP12 `[30,40)` Qwen B1 execution matches all eight
  same-artifact CUDA tokens. Phone TTFT is 1.574 s and phone service wall is
  3.668 s versus 0.163 s on CUDA. This is provisional, not an eligible route.
- Real Gemma Q4 at cut 30 has clean phone placement but produces different
  tokens from CUDA. A same-host CUDA split matches monolithic output, narrowing
  the failure to the heterogeneous phone route or backend numerics.
- The active trace reduces deterministically to five promotion windows and
  nine target changes. Replay rebuild is byte-identical.
- Added a hash-before-parse route reducer and a fail-closed promotion
  controller. Tests pass: replay 18, readiness 6, controller 11. The real
  readiness file causes the controller to exit 2 with `E_ROUTE_NOT_READY`.
- Verdict: `TRACE_REPLAY_MECHANICS_PASS; TWO_MODEL_ROUTE_GATE_BLOCKED`.

### 2026-07-23 EDT - Active warm-tier direction and S39 gates frozen

The primary target changed from fixed phone prefixes and a shared CUDA suffix
to cyclic multi-model residency. One desktop GPU holds one hot model; OP15 and
OP12 collectively hold one executable warm model. During promotion, phones
continue serving while CUDA loads, batch-prefills committed token histories,
catches the token delta, and takes ownership at a token boundary. After
cutover, phones prepare the displaced model. `ACTIVE_WARM_TIER_DESIGN.md`
freezes the state machine, ownership invariants, controls, scheduler scope,
metrics, W0-W5 milestones, and stop rules. S39 retains its real trace but now
stops first at the honest two-model eligibility atlas. No direct KV, latency,
SLO, or energy result is claimed.

### 2026-07-21 EDT - S33 full quantized files add residency, not valid routes

Stored identical full Q4_0 and Q8_0 GGUF files on OP12 and OP15, screened full
graphs, and measured partial B32 windows. Q4_0 reaches clean windows of
OP12 `[0,12)` and OP15 `[4,24)`, but full-graph Q4_0 and Q8_0 both violate the
zero-swap execution gate.

Froze 128 natural WikiText prompts before measurement and compared physical
two-phone routes with same-GGUF one-A6000 references for eight greedy output
tokens. Q4_0 and Q8_0 both miss all three quality thresholds by large margins.
The binder recomputes metrics and validates model hashes, placement, topology,
and worker steps; it refuses both routes. No thresholds were changed and no
quantized route was added to the scheduler.

### 2026-07-21 EDT - S31 measured cut removes the OP12 bottleneck

Measured every `k=1..6` OP12/OP15 cut that shifts work from the S29 baseline
toward OP15 while preserving the layer-8 phone output boundary. A
deterministic, digest-bound selector minimizes the physical B32
phone-stage p95 bottleneck and chose OP12 `[0,1)` plus OP15 `[1,8)`. The
selected bottleneck is 0.399 s versus 1.499 s for S29's cut 6, and route p95 is
3.144 s versus 6.635 s. Selection replay, worker ranges, batch size, drain,
placement, model hashes, and SLO are fail-closed.

The fresh three-session campaign retained all five worker identities and ran
the same 60-request control/treatment trace. Treatment formed one R2 B32
cohort, cut selected CUDA compute 28.90%, preserved all SLOs, and reduced
makespan 29.63% versus S29. It is still 1.93x slower than all-CUDA. An unrelated
OP15 HTP run invalidated one attempt, and ascending B1/B4/B24/B32 reproduced a
shape-growth stall; both attempts are excluded. Numeric quality and
total-system energy remain unclaimed.

### 2026-07-21 EDT - S28 real priority-safe shared CUDA tail passes

Replaced S26's route-isolated tail queues with one priority-aware physical
queue. P0 is ordered before background by latest-safe start and FIFO, while
P0 and P1/P2 are forbidden from sharing a physical batch. Route lineage now
carries priority into every stage event, with mutation tests for ordering,
isolation, topology aliasing, conservation, and persisted-result validation.

The same resident OP12, OP15, and RTX 4060 Ti workers ran an all-CUDA control
then a phone-offload treatment over the 60-request dense mechanics trace. All
60 requests completed with no synthetic SLO miss in each run. Treatment sent
50 background requests through OP12 -> OP15 -> CUDA tail, reduced summed CUDA
island compute 18.55%, and preserved P0 p95. Both phone stages averaged B3.846
with max B4. The shared tail switched between R0 and R2 seven times without an
urgent/background batch mix. Treatment used background slack aggressively and
took 41.977 s versus 5.007 s control. Numeric quality and every energy boundary
remain unclaimed.

### 2026-07-18 EDT - S15 persistent OP15 B32 passes seven real sessions

Extended opt-in DETACH to the single-phone batched pipedriver and repaired a
failure path that previously detached even after a failed session. DETACH now
requires a clean session and a bounded ACK; failures use STOP and cannot be
silently reused. CPU, CUDA, and Android builds pass, with six focused CLI/input
tests.

One OP15 `[0,8)` worker then served seven B32 sessions with DETACH x6 and legacy
STOP x1. The PID, nonce, boot ID, weights, and backend contexts remain constant;
all 224 requests match current same-batch CUDA, every session is HTP0-only except
GET_ROWS, median prompt-to-host-exit is 3.620283 s, maximum is 3.676122 s, and
HMX ends at 38.4 C. Frozen artifacts and an independent validator pass. The
result is runtime route epoch 14; the runtime suite is 79/79. Energy is UNKNOWN,
and the A6000 tail still reloads outside each paid prompt window.

### 2026-07-18 EDT - S15 real arrival-faithful B32 dispatch passes

Froze a 32-request cohort from the densest observed BurstGPT API-generation
window and recertified OP15 `[0,8)` plus selected-A6000 `[8,48)` after model and
context load but before prompt submission. Seven fresh physical processes are
224/224 exact versus same-batch CUDA, all phone compute is on HTP0 with CPU only
for GET_ROWS, the conservative full-response profile is 3.968367 s, and CoV is
0.462 percent.

Integrated that profile into the real runtime registry at a distinct epoch and
ran one physical coordinator-triggered replay. It produces exactly 31 WAIT
decisions followed by B32, exposes the prompt only after typed EXECUTE, and
completes 32/32 requests in 3.577135 s with 422865 us of the synthetic SLO
remaining. An independent validator derives decisions, timing, identity,
terminal ownership, token equality, placement, D2H completion, and thermal
validity from the raw artifacts. The trace replay is in logical arrival time,
not wall-clock paced. Payload, priority, and deadline are synthetic; energy is
UNKNOWN. No commit or push.

### 2026-07-18 EDT - S14 CP-F live OP15 relief passes, SLO gate fails

Corrected the worker's overclaim. The matched phone BGE rerun is ineligible
(10/18 rows exceed CoV 0.05), and the replayable 11.854-percent CP-D number is a
GPU-only counterfactual with no live phone. The serial OP15 -> OP12 B32 live run
failed repeatability and increased both selected-GPU energy and critical-path
latency in its completed treatment cohort.

Screened the independent OP15 `[0,8)` route with current binaries. B1 is exact;
B4/B8/B32 are rejected for same-batch token divergence. Repaired Stage-B so
token mismatch, incomplete request sets, undeclared CPU work, and invalid
thermals return nonzero. Seven final B1 processes complete 56/56 exact requests,
p50 wall median 1.254 s, CoV 0.0135. The live mixed P0/P2 run then measures 5.43
percent selected-A6000 board relief with high-priority BGE p95 preserved, but
low-priority Gemma p95 is 2.639x and fails the frozen 2.0x gate. Independent raw
replay passes. No phone or total-system energy claim is made.

### 2026-07-17 EDT - S11-E0 selected-board cohort rejects fixed serial route

Executed the single authorized B=8 cohort: eight ABBA pairs, 512 requests per
timeline, 32 generated tokens per request, and a pre-frozen 3.5 s p95 SLO. All
4,096 requests per route and 131,072 generated tokens per route match exactly.
All placement, process, power, power-limit, thermal, and SLO gates pass.

The phone route reduces selected-A6000 average board power by about one third
and releases 888 MiB, but it is 2.31x longer in aggregate. Server-only consumes
197.42 kJ; treatment consumes 302.33 kJ, a 53.15 percent increase. The gross
relief is -104.92 kJ and the conservative lower bound is -125.82 kJ. Recorded
the frozen inputs in `ACQUISITION_FREEZE.json` and the digest-pinned result in
`ACQUISITION_RESULT.json`. Per stop rule, no rescue sweep is authorized.

### 2026-07-17 EDT - S11-E0 real long readiness passes

Recovered the Android/Hexagon build through the containerized Snapdragon
toolchain, deployed a hash-matched OP15 binary, and repaired the fail-closed
evidence path. Runner v4 binds raw GPU UUIDs in power/process records, continuous
phone boot/thermal identity, strict scheduled op/buffer placement, and a distinct
non-energy readiness mode. CUDA, CPU, ASan/UBSan, and Android builds pass; the
offline suite is 104/104.

The first real B=8/32-token treatment diverged at token 9 with v81 fused FA.
Disabling FA only on the phone stage restored exact output. The sustained
512-request pair then passed all exactness, placement, process-boundary, and
thermal gates: 85.084 s control, 198.724 s treatment, 888 MiB A6000 relief,
1.332/3.236 s control/treatment p95, thermal status 0, HMX max 49.2 C. Frozen
the next diagnostic at 512 requests and a 3.5 s p95 SLO. Energy remains unrun.

### 2026-07-16 EDT - S11-E0 selected-A6000 diagnostic plan

The next physical checkpoint is frozen as a selected-A6000 `GPU_BOARD` A/B,
not a server-wall or total-system energy claim. Control is the resident full
Gemma-4 12B route; treatment is the resident OP15 `[0,2)` plus A6000 tail route.
The first and only authorized cohort is B=8 with at least 32 generated tokens,
eight ABBA pairs, a predeclared p95 SLO, and a conservative 10 percent
sum-all-pairs decision. All attempts are retained and the run stops after B=8.

The v3 runner now binds raw power traces to one GPU UUID, rejects reuse or
mixed-board aggregation, records p-state transitions, includes Ampere's
one-second power-average boundary uncertainty, and reports exact normalized
energy denominators. Unit tests pass 33/33. Acquisition remains blocked on
continuous competing-process observation, continuous phone thermal telemetry,
aggregate reintegration of hashed raw power, unchanged power-limit binding,
and a no-fallback placement certificate. The frozen OP15 B=8 measurement route
is now enforced. No physical energy was measured.

### 2026-07-16 EDT - S12-V1 asymmetric WiFi-input and USB-result scheduler

The scheduler topology now treats server-to-phone and phone-to-server traffic
as separate directional resources. A phone group advances through bounded
WiFi input, phone compute, USB result, and A6000-tail phases; returned results
hold explicit host buffer credits and receive tail priority.

Review found that the first V1 draft overclaimed executable overlap. S11's
full-model and tail-only HBM numbers come from separate processes, so V1 now
freezes one residency for a whole replay and rejects dynamic route mixing.
LayerSplit has one phone KV context, so V1 also limits OP15 to one in-flight
group and reports zero WiFi/USB overlap. The old S12 replay and hash remain
unchanged. The revised suite has 57 unit tests, three CLI negatives, and ten
hash-seed runs. Rates and interference are unmeasured, the old phone-stage time
is only a proxy, and no runtime, latency, capacity, or energy claim is made.

### 2026-07-16 EDT - S8/S12 real trace substrate and bounded virtual queue

The pinned BurstGPT and RAGPulse sources now normalize into atomic,
byte-reproducible component bundles. A structural replay binds the raw source,
config, every output/sidecar, artifact digest, normalizer/replay source, and
validated service DAG before reporting only arrival order and observed demand.
It reruns the pinned normalizer and byte-compares the full bundle. BurstGPT
median and all four RAGPulse windows pass. The frozen mix composition is still
unimplemented and rejects fail-closed.

The S12 virtual queue implements four bounded policies and exact terminal,
activation, and HBM ledgers. Mechanics pass 30 unit tests plus deterministic
multi-process replay. Strict checks over final real components produce zero
eligible S11 rows, so no real-trace latency or energy result is claimed.

### 2026-07-16 EDT - S11-B fail-closed runner and stage-chain repair

Adversarial testing found five mechanics holes: a B=1 prefill could exceed its
configured allocation and abort; a middle stage accepted token-only input; EOF
could report success; peer failure could raise SIGPIPE; and a full-model host
could ignore the returned activation while still labeling the route offloaded.

The current path checks the complete ownership chain before work using a
versioned, timeout-bounded stage hello and strict mode ranges. The runner v2
binds source, binary, and model hashes and derives exactness and measurement
eligibility from strict records. The final repaired B=8 checkpoint remains exact
at 0.455x server throughput with 888 MiB A6000 relief. A repaired OP15+OP12
checkpoint remains exact at 0.390x with 1288 MiB relief. Energy remains
`NOT_RUN`.

### 2026-07-16 EDT - S11-B: exact static phone batches and layer-window KV

The fixed OP15 `[0,2)` route now supports real independent sequence rows for
batched prefill and decode. The phone and A6000 tail use the same sequence IDs,
and the server-only control uses the same batch. The fail-closed harness records
group timing separately from request identity and rejects incomplete groups,
negative or non-closing timing, and empty aggregates.

Exact output passes at B=1,2,4,8,16. Treatment throughput rises
3.18 -> 25.49 req/s from B=1 -> B=16, but remains below the equally batched
A6000 at every point. The repeated B=8 gate is exact for 48/48 requests:
452.36 ms median group wall, 17.12 req/s, and 4.74 percent complete-route CoV.
The OP15 stage alone has 8.52 percent CoV across its six groups. This is phone
utilization and A6000-memory capacity, not a latency, stable phone-rate, or
aggregate-throughput win.

A Gemma-4 KV filter now applies the existing LayerSplit range during memory
construction. Real HTP allocation falls from 1280 MiB on each phone to 4 MiB
for OP15's two layers and 2 MiB for OP12's one layer at B=1. The existing
two-phone route remains token-exact and releases 1288 MiB on the selected
A6000. No energy run was made.

### 2026-07-16 EDT - S10-E2A R4: exact route DAG and phone-result path

A final read-only audit reproduced two critical defects after R3. Route digests
were only labels, so an action set was not actually precommitted. More seriously,
a zero-duration phone EXEC plus unrelated transfers and a real server CUDA EXEC
could pass as Q-PIM.

R4 adds plan-digested control and treatment `RouteSchedule` artifacts. Lifecycle
actions must match the exact route nodes, and every dependency edge is replayed
as an ACK-before-start constraint. Each phone-assisted request needs a DATA path
`H2D -> phone EXEC -> D2H -> result`; an optional server continuation must be a
descendant of D2H. Extra server work, missing phone work, zero-duration compute,
broken data paths, cycles, and operator-island substitutions all fail closed.

Verification: E2A 215/215, R4 14/14, CLI negatives 42/42; deterministic fixture
digest `05b679d52d36e014ea0ed1114f69262065aed2862ea7007568a3a8d0ba736962`.
The E1/E2 pinned baselines are unchanged. No measurement was run, so the verdict
is `E2A_R4_ROUTE_DAG_MECHANICS_PASS_PHYSICAL_CLAIM_BLOCKED`.

### 2026-07-16 EDT - S10-E2A R3: route and evidence boundary repaired

Independent review found that R2 could still accept a server-only treatment,
reuse an unproven SERVER_WALL scope, leave input and generation identity
underbound, overwrite duplicate bundle slots, ignore warmup semantics, reopen
paths through an intermediate-directory race, and accept incomplete anchor
identity. Active v3 records close those paths.

The plan now pins distinct routes and server/phone device sets. Control forbids
phone actions; treatment requires request-covered HTP/OpenCL execution bracketed
by H2D and D2H. Lifecycle actions carry exact leases and model/operator identity.
Run outputs bind input, prompt, decode, and stop-set digests. SERVER_WALL resolves
and validates the E2 capability record. Artifact reads remain below one retained
root dirfd, and E2 consumes the secured buffers rather than reopening paths.

Verification: E2A 201/201, R3 18/18, CLI negatives 39/39; E2 152/152 and E1
201/201 pass; E1 differential 1187/1187 with zero mismatch. The production
fixture still refuses at `E_ANCHOR_TRUST_ROOT`. No measurement was run, so the
verdict is `E2A_R3_INTERNAL_CLAIM_PATH_PASS_PHYSICAL_CLAIM_BLOCKED`.

### 2026-07-16 EDT - S10-E2A R2: semantic evidence chain repaired

An independent audit reproduced seven surviving fail-open paths after the first
E2A report: transitive poisoned bytecode under pinned E2 sources, vacuous
same-work thresholds, RESULT before EXEC and reversed leases, an unrelated ledger
clock, dispatch-only SLO accounting, raw-outcome bypass, and mutable/reused
artifact paths. The final review also closed early lease release, forged resolved
wrappers, and the lifecycle helper's omitted-evidence mode. All are closed under
v2, with 19 focused regressions. Two of those regressions cover overlapping
windows and reused evidence.

The full E2A suite is 183/183; 32 CLI negatives pass; deterministic regeneration
and the 45/28 pinned E1/E2 baseline files are unchanged. The original E1 and E2
suites also pass independently. This is still mechanics-only: no physical
measurement or energy-saving conclusion was produced.

### 2026-07-15 EDT - S10-E2A: all-pairs aggregate built; a TSA would not unblock it

Built `SUM_ALL_PAIRS_V1` (the evaluator E2 deliberately left unbuilt) plus the
pre-run plan, both anchor receipts, the request manifest/outcomes, the append-only
attempt ledger, and lifecycle resolution. Verdict:
`E2A_AGGREGATE_MECHANICS_PASS_EXTERNAL_ANCHOR_BLOCKED`. No measurement run.

**The result that matters is a negative about anchors, and it was not the expected
one.** The question was "is there an independent external anchor?" The useful answer
is sharper:

| property | what it means | RFC3161 (freetsa) |
|---|---|---|
| P1 PRECEDENCE | the plan existed before the runs | **YES**, in full |
| P2 EXCLUSIVITY | exactly ONE plan was committed | **NO**, not at all |

RFC3161 is independent by the spec's own test (third-party key, third-party clock),
and ~10 min of provisioning away: a live probe got `Status: Granted`, chain verifies
once `cacert.pem` is fetched. It is still useless here, because a TSA is a
**responder, not a log**: nothing enumerates what it signed, so anchor-32-reveal-1
passes every check. **Provisioning a TSA would not unblock E2A** -- it buys the
property that was never in doubt. Closing P2 needs enumerability: pre-registration or
a transparency log with a reviewable identity binding. Neither exists.

Encoded as a typed map (E2's instrument-typing lesson applied to integrity -- a
`verifier_name: "freetsa.org"` string grants nothing):

~~~text
ANCHOR_INDEPENDENT["RFC3161_TSA"] = True     <- passes the independence test
ANCHOR_ENUMERABLE ["RFC3161_TSA"] = False    <- and still cannot carry an aggregate
                                                that pair of lines IS the finding
REQUIRED_PROPERTY = ORDERING_AND_ENUMERABLE  -> E_ANCHOR_UNENUMERABLE
~~~

Three blockers now stack, independent -- clearing any one alone changes nothing:
no wall instrument (E2), no enumerable anchor (E2A), no cryptographic verifier
implemented at all (E2A, stated rather than hidden: `ANCHOR_VERIFIERS` is empty).

**Adversarial review: 3 CRITICAL, all reproduced, all fixed.** Two were verbatim
recurrences of bugs this codebase documents as fixed:

- **checker CLI printed a physical label on hand-written JSON** (`SERVER_RELIEF_PASS`,
  exit 0; no bundle, no anchor, no artifacts). E2's "never be handed a conclusion",
  recurring in the one artifact a reviewer actually runs. It survived because the file
  is scrupulously honest in its docstring and the CLI ignored the docstring.
- **`__pycache__` defeated the pinned canon.** The pin hashed `canon.py` and then
  `exec_module()`'d it -- which runs the CACHED BYTECODE when the pyc header matches.
  Same digest, dead type gate. E2's artifact lesson at module scope: hashing one thing
  and consuming another.

~~~text
  source digest unchanged : b2a3bfde28a22033  (pin passes, cleanly)
  OLD exec_module()       : is_int(1.0) = True   <- poisoned bytecode ran
  NEW compile(data)       : is_int(1.0) = False  <- source ran, pyc ignored
~~~

- **`validate_aggregate` had never run.** Billed as "what makes the label unfakeable",
  it raised `E_TYPE` on its own sealed output (the type gate rejects bools; the record
  has two). No test called it -- which is how it and a no-op additive guard survived
  145 green tests. A gate that rejects the thing it protects is not strict, it is absent.

**Design result worth keeping: structural checks first, policy gates last.** The anchor
gate originally ran first; it masked 13 CLI negatives (a broken ledger, a truncated
cohort, a forged energy and a missing slot all reported `E_ANCHOR_UNENUMERABLE`), so a
deleted check and a working one were indistinguishable -- and those checks only become
load-bearing the day an enumerable anchor exists, i.e. the day nobody would notice they
had rotted. Moved after resolution, the refusal also says something stronger: the bundle
is impeccable and STILL cannot support the claim. The same mistake recurred within the
session when a new verifier check was added early and masked six anchor checks.

**Finding about E2 (CP5):** its `repetition_set.json` declares 8 pairs and validates
cleanly under E2's own rules, but 14 of its 16 declared timeline digests are digests of
nothing -- only pair 0 exists on disk. A shape check cannot notice that 7/8 of a cohort
is absent. Relatedly `attempted_pairs == len(pairs)` is a tautology (the producer writes
both numbers), so E2A checks slot coverage against the anchored plan's 2N slots instead.

145 tests + 32 CLI negatives, deterministic fixtures, E1 (45 files) and E2 (28 files)
byte-identical before and after. Open items are listed in RESULTS.md section 3 rather
than smoothed away: no window disjointness, realized work asserted rather than derived
from output artifacts, and several required-but-unread fields.
Tree: `research_dev/spikes/s10_matched_energy_e2_aggregate/`.


### 2026-07-14 EDT - Sequential dynamic phone provisioning mechanics pass

Extended `examples/phone-pim/` from a pre-staged-only command path to protocol v3
sequential provisioning. The host binds a ticket, manifest, object digest, ordered
chunk map, route epoch, and generation. The worker reserves the complete object,
ACKs only a synced verified prefix, reconstructs that prefix after disconnect or
process restart, verifies the full file, publishes with no-replace rename plus
directory sync, and exposes only the mode-0400 content-addressed final. PREPARE
explicitly selects `published_store`; it cannot silently fall back to a pre-staged
path. A verified descriptor cache removes the duplicate hash between lookup and
PREPARE without trusting a mutable path.

Final host suites pass: protocol 22, store 45, and a real worker/socket lifecycle
suite 45. Android protocol/store suites pass on both phones. Exact-final clean
464,114,176-byte uploads and production-oracle HTP execution pass on OP15 v81 and
OP12 v75 with rel-L2 2.92e-4 and 2.95e-4. The important negative result is
transport efficiency: useful goodput is only 4.7 and 14.9 MiB/s in the final
clean runs, so this is
`DYNAMIC_PROVISIONING_MECHANICS_PASS; CAPACITY_UNPROVEN`. The next gate is a
bounded native/pipelined bulk path with decomposed H2D, UFS, hash, materialize,
warmup, D2H, and two-domain measurements. No scheduler or energy claim was made.
See `spikes/s9_phone_pim_runtime/DYNAMIC_RESULTS.md`. Nothing committed or pushed.

### 2026-07-14 EDT - S9 v5 static closure and first real phone FFN runtime

Reviewed the worker's S9 contracts and simulator instead of accepting the green
suites as certification. A new mutation pass reproduced additional v4 fail-open
records. Added append-only v5: shape-first validation, all-field content digests,
exact dispatch/manifest/segment/range/I/O/SoC/causality/transfer/duplicate and
ReadyCertificate-ledger checks. Final evidence: 24 v5 schema fixtures, 28 v5
bundle fixtures, and 34 red-v4/green-v5 checks, all pass. Historical simulator,
bundle, semantic, and golden suites remain preserved. Label stays static snapshot
coherence only; live atomic dispatch is blocked.

Implemented `examples/phone-pim/` as the smallest actual PIM-style command path.
The trusted-localhost worker verifies one pre-staged Gemma4 shard by bytes and
SHA-256, loads a complete dense FFN into one persistent backend graph, warms it,
and accepts bounded epoch-guarded activation commands. The host oracle captures
the real Gemma4 FFN boundary through `llama_decode`; it does not reuse the worker
builder. Replay after reconnect/restart, stale generation, corrupt file, backend
failure, oracle failure, non-loopback binding, and terminal generation all fail
closed. Host/Android protocol tests pass; OP15 v81 and OP12 v75 HTP0 both pass
M=16 correctness. The result is `PRESTAGED_FFN_MECHANICS_PASS`, not dynamic
streaming, scheduler, capacity, or energy. Full reports are in
`spikes/s9_phone_pim_runtime/` and `spikes/s9_dynamic_weight_residency/
V0R3_REPAIR.md`. Nothing committed or pushed.

### `2026-07-12 EDT` - S6-L repair v2: fail-closed harnesses re-run on real devices, honest verdicts

Repaired the S6 measurement infrastructure to be fail-closed (Phases 1-4) and re-ran the smallest
decisive tests on both phones. Corrected verdicts (full detail in
[spikes/s6_latency_scheduler/RESULTS.md](spikes/s6_latency_scheduler/RESULTS.md)):

- **Profiler + oracle (Phase 1):** oplayerprof now proves graph placement via `cb_eval` (fails on any
  CPU-compute fallback), reset returns {ok,method}, correctness/dump are fatal with checked I/O + IEEE
  finite, and a canonical `resdiff.py` rejects empty/truncated/mismatched/non-finite files. Device:
  OP15 HTP vs CPU **2.95e-3 PASS**, placement `[HTP0]`.
- **Dualengine (Phase 2):** owned generation handshake + timed waits + a process watchdog replace the
  raw-pointer barrier (the prior intermittent hang did NOT reproduce); errors latch and a bad phase
  emits NO gate metrics; compute-only makespan is separated from the reset-inclusive cycle; a real
  CPU cross-backend check and an advancing-KV SERVICE microtrace were added. Device (OP15, B16/C512):
  **SATURATED 1.93x but decode CoV 0.060 -> PROVISIONAL**; **SERVICE decode +19% under contention ->
  FAIL** (the honest request-latency result). Host: 10k-epoch worker stress + 50ms-reset invariant +
  parser negative tests all pass.
- **FFN merge (Phase 3):** verdict ladder now makes a false PASS impossible (UNSUPPORTED without a GPU,
  SYNTHETIC_UNCERTIFIED without real residuals, LOWER_BOUND without a proven single copy). Real
  post-attention residuals are captured by oplayerprof `--dump-attn-out` (gemma4 `attn_out`, via
  `cb_eval`, no model edit). Device (OP15, real residual 16+48): all controls correct 2.18e-3, merged
  1.91x vs serial-HTP, mem 354/354MB -> **LOWER_BOUND**.
- **Hexagon FA:** OP12 **v75 AUTO revalidated** - placement `[HTP0]` no CPU fallback, v75 vs OP12 CPU
  **3.61e-3 PASS**. OP15 **v81 strict-gate result for review**: FA-ON B16/C512 vs CPU **5.02e-3**
  (marginally over 5e-3), FA-OFF explicit **2.98e-3 PASS**; v81 support policy left UNCHANGED pending
  human review. Old same-engine batched-vs-single check (5e-4) is blind to this; the CPU cross-backend
  check is what catches it.
- **Energy (Phase 4):** `run_energy.sh` refuses (DEFERRED, exit 3); `energy_align.py` gates hardened
  (bracketing, max-gap, real-sample count, zero-USB, coulomb-when-unplugged, both paths); 16/16
  synthetic cases pass. NO physical J claimed.

Remaining (set up, not fully run): 7-process OP15 sweep, v81 FA at more B/C, OP12 dualengine (RAM),
FFN 32+96/32+480, S7-V1 trace. Protected files (gemma4.cpp / llama-graph.cpp / ggml_backend_sched)
untouched. Nothing committed. Raw data under `scratchpad/s6_latency_repair_v2/`.

### `2026-07-12 EDT` - Direction decision: mixed-workload resident operator islands

The project target changed from one fixed Gemma pipeline to a mixed-service
server-to-phone accelerator pool. Design A is retained as route `A0` and proven
substrate. The active design introduces resident operator islands, versioned
phone weight readiness, a host virtual queue with bounded per-backend credits,
device/backend status plus cache readiness, rolling residency leases, and
separate capacity and energy objectives.

S8-V0 is the first gate: public real-trace normalization, existing-code audit,
resident-island profile atlas, and an offline capacity oracle. No scheduler,
backend VQ port, graph/KV change, or energy claim starts before that gate.

### `2026-07-12 EDT` - 🔧 S6-L REPAIR: corrected verdicts (the prior "both PASS" was wrong)

A review flagged the earlier S6-L "both hypotheses PASS" as untrustworthy. Repaired the harnesses and
reran; **corrected verdicts** (`spikes/s6_latency_scheduler/RESULTS.md` sec 0, artifacts
`scratchpad/s6_latency_repair/`):

| hypothesis | corrected verdict |
|---|---|
| overlap **saturated throughput** | **PROVISIONAL** ~1.9x (near-zero interference), but stock-GPU prefill CoV 4-8% trips the 5% gate on 2/6 configs; only 3 procs/config (shared device) |
| overlap **pair (request) latency** | **FAIL** 1.03-1.38x; 5/6 fail -- decode leg slowed 13-19% by a concurrent prefill, or speedup ~1.03x once prefill dominates |
| complete-FFN merge | **LOWER_BOUND** -- correct now (rel_L2 2e-3, HMX x3), merged beats best control 1.26-1.81x, but the route-affine control needs a 2nd weight copy (no per-tensor sharing) |
| Hexagon FA | **v75 FIXED (gate `opt_arch==75`)**, v81 fine -- op12 on/off/auto now 3.3e-3, AUTO stays on HTP |
| energy | **DEFERRED** -- offline math synth-validated (8 cases), no physical J |

**What was wrong before:** (1) the "1.9x request-latency overlap" was actually *saturated throughput*
(balanced backlogs); real per-request pair latency is 1.03-1.38x and fails the gate. (2) the FFN
"PASS 1.35-1.92x" hid non-finite output behind a fast-math `std::isfinite` (a bit check exposed it;
this gemma's ffn_norm gain mean~25/max~139 overflows F32 for arbitrary synthetic input), used no
post-FFN norm, and had no GPU-resident boundary -> now LOWER_BOUND. (3) the FA gate `opt_arch<81`
blessed all future archs -> narrowed to `==75`. Also: dualengine decode default AUTO; correctness now
returns nonzero; ffnmerge builds on host (missing `<ctime>`); energy adds interpolation/prefill-denom/
coulomb/battery-only/missing-ilim/non-monotonic/status cases. **Known bug:** the overlap harness
intermittently hangs (worker futex race) -- fix before a wider run. Nothing committed; stopped for review.

### `2026-07-11 EDT` - S7-V0 ragged HMX decode tile skip passes isolated operator gate

Built a default-off HMX flash-attention prototype and tested it with deterministic, model-free
Gemma-4 SWA tensors in `test-backend-ops`. The kernel builds a per-sequence active KV-block list
before K/V DMA and skips only blocks whose broadcast fp16 mask is exactly all negative infinity.
Partially valid blocks and the original flag-off path are retained.

On OP15 v81, flag off and on both pass all CPU-reference cases at B=8/16. At C=512, ragged-prefix
attention improves by **1.489x at B=16** and **1.492x at B=32** (five-process CoV below 0.5%). An
interior masked hole improves by 1.280x/1.274x. The all-valid control changes by +0.2%/-0.7%, within
the 5% overhead gate. This is an isolated kernel PASS only; no real-layer, trace, end-to-end, or
energy claim is made. See [S7-V0 results](spikes/s7_ragged_attention/RESULTS.md).

### `2026-07-11 EDT` - 🛠️ S6-L: measurement infra repaired (A-E), both latency hypotheses PASS, real OP12 FA bug found+fixed (SUPERSEDED by the 2026-07-12 repair above -- the "both PASS" verdict did not hold)

Repaired the S6 harnesses, then ran a **latency-only** operator-overlap screen (energy stays
DEFERRED). Spike: `spikes/s6_latency_scheduler/`. Nothing committed.

**Infra repairs.** `oplayerprof` now has explicit **decode/prefill modes** timing only
`llama_decode` (KV-clear/build/compile/xmem-prepack excluded), fixed graph shape, `seq_rm` fixed-C
reset outside the timed window, `--min-samples` gating, CPU-force, status+exit propagation,
truncate-by-default (CoV now 0.3-4%). Correctness is now **cross-backend vs CPU** at real context
(not pos 0): stock OpenCL 2.9e-3 PASS, **xmem os8 1.88e-2 -> PERF_ONLY** (excluded). `run_dualengine`
rewritten to **4 genuine cases** (D solo, P solo, directly-measured serial, D||P barrier-concurrent)
with persistent workers -- concurrent legs are no longer mislabelled "alone".

**S6-L results (OP15).**

| hypothesis | result |
|---|---|
| HTP-decode \|\| stock-GPU-prefill overlap | **PASS** -- 1.91-1.97x speedup, conc/max_solo ~1.01, per-leg slowdown <=1.03, CoV <3.5%, all B={16,32} x T={64,256,512} |
| complete-FFN merge (16+48, 32+96, 32+480) | **PASS** -- 1.35-1.92x, merged p95 0.52-0.75x sep, xfer <9%, rel_L2=0, all 3 GEMMs HMX |

**Real bug found (OP12 v75).** Cross-backend-vs-CPU exposed that **OP12 Hexagon v75
FLASH_ATTN_EXT is numerically broken** (rel_L2 ~0.5-0.8) while the non-FA path on the same v75
engine is correct (3.3e-3); v81/OP15 FA is fine. The old same-engine batched-vs-serial check could
not see it (both legs share the broken kernel). **Fix:** FA support predicate restricted to
`opt_arch >= 81`; validated -- OP12 + `--fa auto` now decodes at 3.3e-3. Use `flash_attn_type=AUTO`
on v75. Flagged for review (changes the op12 decode path).

**Energy still DEFERRED.** Offline scripts repaired (trapezoid + charge-counter/coulomb, gross vs
incremental J/layer-token, strict validity gates, `set -euo pipefail`, EXIT cleanup) and
**synthetically validated** (`synth_validate.py` recovers gross=50 J / incr=40 J / coulomb=40 J).
No physical J/token is claimed. `xmem` cache left disabled for S6-L; a lifetime-safe design is
proposed for review, not implemented.

---

### `2026-07-11 EDT` - ⚡ Energy measurement: plugged-in whole-device wattmeter is viable; harness built + pipeline validated

**Question:** can we measure per-op / whole-device energy on the phones, and without unplugging?

**Per-op directly: no.** No retail-phone sensor is both fast enough (PMIC updates ~2-6 Hz) and per-rail enough to catch a µs-ms op. The HTP/GPU rails (CX/MX, GFX) are not exposed to userspace on retail OnePlus. The tractable method is the **amortized differential**: average power of a steady loop of one layer on one backend, minus idle.

**Whole-device without unplug: yes, with a caveat -- and it refines the m5 dead-end.** Probed op15 (root; op12 CPH2583 has `no su`). `usb/current_now` is a **live, load-tracking node**, not a fixed ceiling:

| state | usb V x I | note |
|---|---|---|
| idle | 5.08 V x ~130 mA ~= **0.66-0.71 W** | tracks load |
| all 8 cores | 5.04 V x ~496 mA ~= **2.50 W** | **pinned at `input_current_limit` (5 V/500 mA SDP)** |

So the m5 "1.7-1.8 W near-constant ceiling" was **not a broken sensor** -- it was the input-current cap clipping a live reading (dP idle->cap = **1.78 W**, exactly the m5 number). Real blocker = **negotiated input power < workload draw**, not "plugged in." Fix: **Full battery + high-wattage PD/SUPERVOOC charger + WiFi-adb** -> `usb V x I ~= system power`, `pin -> 0`, no unplug/discharge needed (easier than the m5-mandated unplug+coulomb path). Battery-coulomb path still works too (sampler logs `bat_*`), but needs unplug + battery off Full.

**Built the harness (`research_dev/energy/`), ready to drop on either path:**
- `oplayerprof.cpp` energy mode: `--idle-secs` (idle baseline window per (B,C)) + every `OPLAYERPROF_MARK` now carries `mono=<sec>` from `/proc/uptime`. Harness stays normal-shell (root would break HTP/OpenCL SELinux domain). Android binary rebuilt.
- `pwr_sampler.sh` (root) -> usb+battery rails @ ~6-10 Hz, `/proc/uptime`-stamped. `energy_align.py` -> `dP = P_compute - P_idle`, E/round + E/tok, and a **validity gate** (`usb_pinned_frac > 5% -> valid=false <usb_capped>` = the "is the charger big enough?" check). `run_energy.sh` orchestrates.

**Pipeline validated end-to-end (op15, dev-port):** against a known idle->all-core-load pattern it recovered **P_idle=0.71 / P_compute=2.50 / dP=1.78 W** and correctly flagged **`usb_pinned_frac=1.0 -> <usb_capped>`**. Sensor + clock-alignment + gate all confirmed. Awaiting a high-wattage charger for an unclipped inference number; op12 needs rooting to participate.

### `2026-07-11 EDT` - S4-V0 offline schedule bound: FAIL (op15 SWA) -> STOP before implementation

Built a measurement-only harness `examples/layersplit/oplayerprof.cpp` (intact
single-layer (B,C) decode via public llama API; per-op HTP times from
GGML_HEXAGON_PROFILE, per-op GPU times from a GGML_OPENCL_PROFILING build; no
graph/KV/scheduler edits). Measured H_A/A_H/H_C (HTP) and G_B = GPU KV-store+fused-FA
(Adreno) for blk.2 SWA on op15. Ideal speedup = wall / max(H_A+H_C, G_B).

| B | C | wall ms | attn% | H_A+H_C | G_B (Adreno FA) | ideal | verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | 32 | 20.0 | 22.8% | 15.5 | 7.7 | 1.30 | pass (empty KV) |
| 16 | 512 | 20.7 | 23.3% | 15.9 | 25.4 | 0.82 | FAIL |
| 16 | 1024 | 23.3 | 32.2% | 15.8 | 44.2 | 0.53 | FAIL |
| 32 | 512 | 26.5 | 37.8% | 16.5 | 56.7 | 0.47 | FAIL |

**Verdict: FAIL at realistic context.** The split clears >=1.20x only at C=32
(near-empty KV). Root cause: the Adreno OpenCL decode flash-attention kernel is slow
and scales ~linearly with KV (G_B 7.7->25.4->44.2 ms as C 32->512->1024 at B=16),
while Hexagon HMX FA scales ~1.6x (A_H 4.6->7.5 ms) and HTP does the whole rest of the
layer in ~15-16 ms flat. Moving attention off HMX onto Adreno replaces a cheap,
well-scaling op with an expensive, poorly-scaling one -> GPU attention becomes the
bottleneck and the pipeline is SLOWER than intact HTP (down to 0.47x) at any C real
decode runs in. Both required batches (16, 32) fail at C>=512.

op15 is the FASTER GPU (Adreno 840); op12/Adreno 750 expected no better (older GPU +
FA split-variant won't compile). FULL class can't run GPU attention at all (head_dim
512 fallback). So S4 operator-type split is non-viable on both classes. Per PLAN stop
rule (S>=2 below 1.20x ideal at B=16/32 for same class+C -> stop before implementation):
**STOP; do not proceed to V1/V2 or graph integration.** Device runs hit repeated adb
dropoffs from MEMORY PRESSURE (usable RAM ceiling ~6 GB op12 / ~10 GB op15): running
HTP-prof and OpenCL-G_B processes concurrently (two shard maps + OpenCL image/xmem) and
the OpenCL backend accumulating per-shape prepack buffers across the B×C sweep exceeded
the ceiling -> OOM. Handled via one-process-at-a-time detached nohup runs + adb-server
restarts. The failing gate points (B=16/32 at C>=512) are tiny-memory (KV <=140 MB) and
reliable; op12's full grid still pending but moot given the solo bound fails. NOTE the
6/10 GB ceiling is itself a standing serving constraint (batch x context x layers per
phone).
Full table + raw artifacts in
[spikes/s4_streamed_batch_decode/RESULTS.md](spikes/s4_streamed_batch_decode/RESULTS.md).

### `2026-07-11 EDT` - S4-V0 GPU fused-attention veto: BOTH phones PASS (SWA decode); OP12 veto was STALE

Ran the one authorized device slice (existing `llama-layersplit`, no new code):
force the batched decode of the SWA shard blk.2 (head_dim 256) onto the Adreno GPU
with FA enabled, `GGML_SCHED_DEBUG=2`, watch OpenCL compile + op placement.
`--mode dualengine --dev-decode GPUOpenCL --dev-prefill CPU -m 12b-f16-mid-2-3.gguf -b 8 -n 2`.

| Phone | FA compile | Placement | splits | Correctness | Verdict |
|---|---|---|---|---|---|
| OP12 / Adreno 750 | non-split OK; **split fails** (sub_group_shuffle_xor, NON-FATAL) | FLASH_ATTN on `[OpenC]`, no CPU node | 1 | PASS rel_L2 1.8e-5, argmax 0/8 | **PASS (SWA decode)** |
| OP15 / Adreno 840 | both variants OK | FLASH_ATTN on `[OpenC]` | 1 | PASS rel_L2 1.8e-5, argmax 0/8 | **PASS (clean)** |

**The historical OP12 flash-attn veto (this log:168, AGENT_HANDOFF:132-134) is STALE
for the decode path.** In-tree non-fatal FA compile skips the failing split variant;
the non-split f32_f16 kernel runs the whole SWA decode attention on the Adreno 750
with no CPU fallback, correct. Neither phone is vetoed for the SWA class.

Caveats carried forward: (1) RESOLVED to 512-ctx — a follow-up probe put a 512-token
GPU prefill + 100-round decode on OP12's Adreno 750 (both contexts GPUOpenCL): 880
FLASH_ATTN ops all on-GPU, 0 CPU, splits=1, correct. The failing split variant is
never actually required; the non-split kernel serves prefill (n_q=512) and decode.
Only C=1024 (SWA-window top) unconfirmed, no cliff expected. (2) FULL/global class (head_dim 512)
not runtime-tested (no blk.5 shard on device) but source-conclusive: absent from the
OpenCL supported_dims table -> GPU-B CPU-falls-back on all 8 global layers on BOTH
phones. S4 GPU-owned attention covers at most the 40 SWA layers. Raw logs +
gate table in [spikes/s4_streamed_batch_decode/RESULTS.md](spikes/s4_streamed_batch_decode/RESULTS.md).

### `2026-07-11 EDT` - S4-V0 Checkpoint 1: graph-cut + capability inventory (inspect-only, source-verified)

Completed the pre-code inventory (PLAN.md checkpoint 1). Nothing edited. Two
load-bearing claims verified directly in source.

**Graph cut (dense gemma-4 12B).** HTP-A ends at gemma4.cpp:344 (post-RoPE Q/K,
post-rms V, retained residual). GPU-B is the interior of the single
build_attn(iswa) overload (llama-graph.cpp:2869): KV store cpy_k/cpy_v
(:2919/:2925) + fused ggml_flash_attn_ext (:2426), ending at the kqv_out marker
(:2935). HTP-C = wo + post-norms + residual + FFN. The single straddle PLAN.md
warned about is CONFIRMED: `wo` is fused inside build_attn at :2941-2942, one line
past the kqv_out marker — the cut must return kqv_out and relocate wo to HTP-C.
Mask + KV indices are built HOST-side (set_input).

**Metadata.** 48 layers, PLAIN dense (no per-layer-embd, no shared-KV, no MoE),
5 SWA : 1 FULL repeat. SWA class (40 layers): GQA 16:8, head_dim 256, window 1024,
seed blk.2. FULL class (8 layers): MQA 16:1, head_dim 512, V-LESS "alternative
attention" (V = reused K-proj, weightless rms_norm, no V-RoPE), seed blk.5.

**NEW constraint — S4 attention offload covers at most 40/48 layers.** The OpenCL
FA supported_dims table (ggml-opencl.cpp:5836-5840, VERIFIED) tops out at 256;
head_dim 512 is absent -> GPU-B fused attention silently CPU-falls-back on all 8
global layers on BOTH phones. So GPU-owned attention is viable only for the 40 SWA
layers; the 8 global layers stay intact-HTP, and throughput/energy accounting must
reflect that. Independent of the OP12 shuffle issue.

**Veto calls (confirm at Checkpoint 2 with device build logs).** OP12 (Adreno-750):
predicted veto — historical sub_group_shuffle_xor FA compile failure (in-tree
mitigations may now let gemma-F16 FA compile, but a missing variant HARD-ABORTS at
GGML_ASSERT(kernel!=NULL) ggml-opencl.cpp:12785), plus the head_dim-512 FULL-class
fallback. OP15 (Adreno-840): proceeds, SWA-only.

**Gate deliverability.** Ideal compute-only >=1.20x replay is fully deliverable at
Checkpoint 2 (needs only H_A/H_C/G_B/A_H). Bounded >=1.15x is BLOCKED on handoff
(X_AG/X_GC) + metadata (X_META) TIMES, each needing a reviewed measurement-only
edit (X_META one touches llama-graph.cpp set_input -> propose-and-stop). Details in
[spikes/s4_streamed_batch_decode/RESULTS.md](spikes/s4_streamed_batch_decode/RESULTS.md).
STOP here for review; no code, no device runs, no edits.

### `2026-07-11 EDT` - S4 pivot: operator-type multi-stream decode, veto before implementation

The output-row result does not support continuing with column or row splits of
one dense weight. More importantly, that mechanism is not the research claim:
PowerInfer-2 already covers heterogeneous mobile batched decode and phone
J/token, while NanoFlow directly covers attention/projection overlap across
operation nano-batches. S4 therefore asks a narrower systems question: can an
HTP-Adreno inter-operator pipeline improve real continuous batch decode and
gross fleet J/completed-token in the existing server -> OP15 -> OP12 -> server
layer pipeline?

The initial phone-local DAG for request group `s` and layer `l` is:

```text
HTP-A(s,l): norm + Q/K/V projections + Q/K/V norm + RoPE
GPU-B(s,l): GPU-exclusive KV update + fused attention
HTP-C(s,l): output projection + residual/norm + complete FFN

HTP-A(s,l) -> GPU-B(s,l) -> HTP-C(s,l) -> HTP-A(s,l+1)
```

There is one serial HTP worker, one serial GPU worker, and multiple disjoint
request groups occupying the cross-backend pipeline. GPU owns each participating
sequence's attention KV from allocation through eviction. HTP never mirrors or
mutates that KV. An S4 request must take this route from admission so GPU-B also
populates its KV during prefill; decode cannot inherit an HTP-built cache. The
primary control is an intact full-B HTP decode, not serial microbatches that
reread weights.

The first agent assignment is V0 only. It must first label the real SWA/full
attention and KV-sharing layer classes, reproduce correct GPU fused attention,
profile real B=4/5/8/16/32 and context=32/256/512/1024 task costs, and
offline-replay S=1/2/4 schedules including HMX fragmentation, repeated weight
reads, handoff, fill/drain, and co-run interference. OP12 is `UNSUPPORTED` if
the known OpenCL flash-attention compile failure reproduces; no slow fallback is
allowed. No integration code starts unless the same S>=2 schedule at B=16 and
B=32 for the same layer class and context logs real HMX execution and reaches
1.20x ideal and 1.15x handoff-bounded predicted layer-throughput. Full contracts are
in [S4 PLAN](spikes/s4_streamed_batch_decode/PLAN.md), [RELATED_WORK](spikes/s4_streamed_batch_decode/RELATED_WORK.md), and [RESULTS](spikes/s4_streamed_batch_decode/RESULTS.md). The executable handoff is [AGENT_HANDOFF.md](AGENT_HANDOFF.md).

This entry also supersedes the H1 wording immediately below. OP15 is a valid
failure and OP12 xmem-off loses. OP12 xmem-on is `INVALID`, not a platform
failure, because the static xmem prepack cache was keyed only by the recycled
OpenCL allocation handle and offset. Its merged split output is wrong, so its
timing cannot support either a win or a loss. S3 remains archived because the
research direction changed, not because every heterogeneous operator pipeline
was disproved. No commits were made.

### `2026-07-11 EDT` - S3-H1: output-row split FAILS on both phones -> STOP (GPU too slow to complement HTP)
Built the standalone `llama-phone-microop` (new `examples/layersplit/microop.cpp` + CMake target + `sweep.sh`; bare HTP0+GPUOpenCL ggml backends, no llama/sched) and ran the HeteroInfer-style output-row split screen on the real `blk.2.ffn_gate.weight` [3840,15360] F16: HTP computes rows [0:n_h], GPU rows [n_h:N], host merge, no reduction. Host CPU smoke bit-exact (projL2=0), Android build via snapdragon docker, device runs on both phones.

Full matrix (M=1,2,4,5,8,16,32,64 x split 0/25/50/75/100 x {xmem off,on} x 2 phones = 160 rows):

```
best interior split is 75% HTP / 25% GPU at every useful M (GPU is 2-3x slower).
                   complete-op speedup vs best intact backend
        M=5   M=8   M=16  M=32  M=64      (>=1.10x at 2 adjacent useful B REQUIRED)
op12 off 0.94  0.95  0.89  0.82  0.68
op12 on  0.96  0.90  1.03* 0.98  0.82    (*M=16 1.03x but projL2=9e-3 > 5e-3 -> correctness FAIL)
op15 off 0.93  0.73  0.69  0.64  0.39
op15 on  0.93  0.73  0.88  0.81  0.64
```

- **H1 gate = FAIL on both OP12 and OP15.** Max complete-op speedup at any useful B over all 160 rows = **1.03x** (needs >=1.10x), and that single point violates the projection rel-L2 <= 5e-3 gate. Everywhere else the split is SLOWER than the best single backend. Correctness otherwise clean (projL2 ~2e-4 xmem-off, split rel-L2 <=2.8e-4, repeat=0, sentinel PASS).
- Root cause: the OpenCL GPU F16 GEMV is 2-3x slower than the Hexagon HTP for this projection (GPU-only ~20 ms xmem-off / ~10-14 ms xmem-on at M>=16 vs HTP ~6-8 ms). The completion-balanced 75/25 split makes the concurrent wall ~= HTP's 75%-share time -- barely below HTP alone -- and fanout+merge overhead erases it. xmem halves GPU time at M>=16 but still misses 10% AND its prepack GEMM pushes cross-backend rel-L2 to ~9e-3.
- Per PLAN + handoff ("if Gate A fails everywhere, stop; whole-request/layer-stage scheduling is the better granularity"): **STOPPED.** No H2/H3, no FFN branch split, no mutable activations, no graph integration. Micro-op output-row splitting is not viable on these phones for this workload. Verdict + 160-row data + eff-GBps in [RESULTS.md](spikes/s3_microop_schedule/RESULTS.md). No commits. Directional takeaway: keep HTP as the decode engine and use the GPU for a *different phase* (prefill), not for co-splitting one memory-bound GEMV.

### `2026-07-11 EDT` - S3-H0: real B-way batch-decode baseline measured (HTP >> GPU), gate PASS
Ran H0 (no code edits) via existing `dualengine` on the identical 1-layer shard `12b-f16-mid-2-3.gguf` (blk.2, head-less/nextn), B=1,2,4,5,8,16,32,64, 30 rounds, distinct seq_id[j]=j. One llama_decode/round; correctness = batched vs serial replay on the same engine.

```
mean ms/round   B=1   B=2   B=4   B=5   B=8  B=16  B=32  B=64    (ALL correctness PASS)
op12 HTP0(FA)   10.7  11.3  17.2  25.6  27.6  31.8  39.6   OOM    rel_L2 ~5e-4
op12 GPU(noFA)  18.8  86.7  87.5  88.6  89.3  94.3 102.5 111.6    rel_L2 ~1-2e-5
op15 HTP0(FA)   11.1  10.3  12.2  26.5  24.8  35.8  34.3  49.1    rel_L2 ~5e-4
op15 GPU(noFA)  16.9  83.8  85.0  84.0  88.3  96.2 111.4 132.2    rel_L2 ~1-2e-5
A6000 CUDA0      1.3   1.5   1.5   1.5   1.5   1.6   1.8   2.1     per-layer ref
```

- HTP dominates GPU decode 2.6-8x on both phones (GPU is attention-bound; B=1->2 GPU cliff = kernel-path switch). HMX boundary at B=5 confirmed on both. Phones 8-23x slower/layer than A6000 (near-flat over batch) -> phones can only win on energy.
- Correctness PASS everywhere; rel_L2 FLAT across B=1..64 => no cross-seq bleed, KV isolation holds; 0/B argmax mismatch.
- On the reviewer's "dualengine labels co-run legs as alone" caveat: tested, not assumed. op12 HTP0 B=8 decode = 26.2 ms @ prefill-1tok vs 24.9 ms @ prefill-64tok (4x heavier) -> decode leg load-insensitive => co-run leg == solo for NPU-decode||GPU-prefill (zero interference). A true single-worker solo intact-decode over a head-less shard is not available without editing layersplit.cpp (flagged).
- op12 HTP B=64 OOM (2GB HTP buf map; KV over-allocates all 48 layers). op12 Adreno-750 GPU flash-attn kernel fails to compile (sub_group_shuffle_xor) -> GPU decode FA-off.
- Measured tables written to [RESULTS.md](spikes/s3_microop_schedule/RESULTS.md); raw JSONL in scratchpad. H0 gate PASS. Next = build standalone `llama-phone-microop` for the H1 output-row split screen on `blk.2.ffn_gate`, pending review. No commits.

### `2026-07-10 EDT` - S3 corrected to test real batch decode first

The target workload is static B-way decode, so a synthetic GEMM `M` sweep alone
is insufficient. H0 now runs one real `llama_decode` per round with B distinct
sequences, one token per sequence, private KV histories, and advancing positions
on each phone backend. It compares B-way output with serial replay and records KV
length, correctness, hangs, fallback, and backend failures.

H1 keeps the bounded output-row experiment, but uses `M_kernel=B` and the same
batch occupancies as the real decode control, including B=5 at the HMX boundary.
Its result is explicitly a batch-shaped operator screen. A batch-decode claim
requires a separately reviewed one-layer integration that runs the split inside
a real B-way graph. No runtime code was changed for this correction.

### `2026-07-10 EDT` - S3 corrected from related work: bandwidth provenance + output-row split first

Reviewed the primary HeteroInfer and llm.npu papers plus the llm.npu artifact. The important correction is that HeteroInfer's decode result splits one `MUL_MAT` weight along output rows and runs complementary GPU/NPU slices; it does not demonstrate two request streams. It reports GPU-only 43.3 GB/s versus GPU+NPU 59.5 GB/s on Snapdragon 8 Gen 3, but does not disclose the DDR counter/tool or byte-accounting formula, so those numbers are not a reproducible method for this tree. llm.npu reports NPU bubble/critical-path utilization, not DRAM bandwidth; its transferable idea is offline subgraph profiling plus input-ready task ordering.

The revised [related-work note](spikes/s3_microop_schedule/RELATED_WORK.md), [S3 plan](spikes/s3_microop_schedule/PLAN.md), [results template](spikes/s3_microop_schedule/RESULTS.md), and [agent handoff](AGENT_HANDOFF.md) define H0-H3. H0 first measures true solo and co-running legs because current `dualengine` labels co-running leg times as `alone` and gives prefill no equivalent warmup. OP15 exposes an aggregate `dcvs/bw_hwmon_meas` tracepoint; comparable OP12 trace access is denied, so OP12 uses a documented profiler capture or explicitly reports no direct DDR result. The portable fallback is named `effective_min_weight_read_GBps`, never physical bandwidth.

H1 then splits `[K,N]` along ggml `ne1=N`, sweeps HTP shares 0/25/50/75/100 percent, concatenates disjoint output channels, and tests private slices, one-copy parent views, and a full-copy control. H2 compares row split, FFN gate/up, independent streams, and single-backend execution at equal total M. H3 is only a bounded ready-task trace replay. Full-layer 15 percent and whole-phone J/token 10 percent gates remain mandatory. No code was written for this revision.

### `2026-07-10 EDT` — 🔬 S3 checkpoint 1: micro-op FFN harness sketch + full API recon (design only, no code, awaiting review) ⏸️
Started the bounded **S3 micro-operator spike** ([PLAN](spikes/s3_microop_schedule/PLAN.md)) — the one question before any generic scheduler: *can gemma-4-12B's independent FFN `gate` and `up` projections run concurrently on **HTP ∥ OpenCL** with enough full-FFN win to justify mutable-activation sharing?* This is **Gate A** (concurrent projections ≥10% vs best serial) + **Gate B** (complete FFN ≥15% vs best intact backend) **only**. Per the plan I stopped at **checkpoint 1 = file/interface sketch**; **nothing was built or committed**, and the uncommitted Fused-FA change + `gemma4.cpp` + `layersplit.cpp` + `ggml_backend_sched` were **not touched**.

**Deliverable = a standalone `examples/layersplit/microop.cpp` → `llama-phone-microop`** that makes ONE bare `HTP0` and ONE `GPUOpenCL` `ggml_backend_t` (no llama, no sched), preads the 3 real `blk.2` FFN weights, and hand-builds direct graphs. Faithful span (verified against the model graph, *not* assumed): `up=mul_mat(x)`, `gate=mul_mat(x)` (same `x`), `h=geglu_split(gate,up)`, `out=mul_mat(down,h)`.

**A 6-agent recon workflow mapped the exact current APIs and surfaced 10 plan-vs-reality corrections** the reviewer should know — the plan is feasible, but the wording was off in load-bearing places:
```
· activation is ONE fused ggml_geglu_split(gate,up) — tanh-GELU on arg-1(gate) × up — NOT gelu+mul
· both HTP0 AND GPUOpenCL report device-type GPU → must select by NAME, never by type
· env latches at first REGISTRY access (not dev_init) → "xmem off" vs "xmem on+cache" = 2 processes (mandatory)
· must build via ggml_build_forward_expand or graph_compute SILENTLY skips nodes (repo-specific COMPUTE-flag gate)
· HTP join needs the remote projection copied into an HTP-buffer MIRROR (every HTP operand must live on the HTP session buffer) — cross-backend copy is host-staged, one ggml_backend_tensor_copy call
· xmem GEMM only FIRES at token N>=16 & M>=64 & K%8==0 → decode M<16 silently uses the l4_lm buffer kernel even with xmem on (log the ACTUAL path)
· op-refuse knob is GGML_HEXAGON_OPFILTER, not the plan's GGML_HEXAGON_OPMASK
· HTP run needs ADSP_LIBRARY_PATH=. (for libggml-htp-vNN.so) — the CPU pipeline scripts omit it
· F16 weights need NO repack/prepack: plain [K,N] F16 into the default buft feeds both standard + xmem paths
· link ggml ALONE (pulls ggml-base+gguf, cpu, hexagon, opencl; self-registers HTP0/GPUOpenCL) — no llama/llama-common
```

**Ground truth confirmed on disk + devices:** `blk.2.ffn_{gate,up}.weight` = `[3840,15360]` F16, `ffn_down` = `[15360,3840]` F16, **112.5 MiB each** (matches plan). Read by `pread(data_offset + tensor_offset, nbytes)` — metadata-only gguf open, blob never touched; verified bit-exact. A **layer-2 shard already exists** (`12b-f16-mid-2-3.gguf` on op12; host has `mid-2-{6,10,18}`). Both phones online (op12 `5ae7a43d`, op15 `3C15AU002CL00000`, 18 `.so` staged each).

**Experiment matrix** (per M in `1,2,4,5,8,16,32,64,128,256,512`, x {xmem off, xmem on+cache}, x {op12, op15}):
```
1 HTP-only FFN   2 OCL-only FFN   (intact references)
3 gate=HTP up=OCL join=HTP    4 gate=HTP up=OCL join=OCL
5 gate=OCL up=HTP join=HTP    6 gate=OCL up=HTP join=OCL
```
Correctness before performance (proj rel_L2 <= 5e-3, FFN rel_L2 <= 1e-2, repeat <= 1e-7, no CPU fallback), then p50/p95/mean/sd/min/max for fanout · gate · up · concurrent-wall · remote-transfer · join · down · full-FFN → one JSONL record per config.

**Status: ⏸️ stopped for review.** Two decisions requested before checkpoint 2 (host build + CPU smoke): (1) link `ggml`-only *(recommended)* vs `llama` to match the sibling target; (2) generate a fresh `[2,3)` shard vs reuse `mid-2-6`. On approval → write `microop.cpp` + the 5-line CMake target + `sweep.sh`, host-smoke on the CPU backend, return the checkpoint-2 report (diff, build output, run cmd, smoke log). Gate C (mutable shared activations) stays unbuilt unless Gate A passes, Gate B loses on copy cost, **and** a reviewer approves.

### `2026-07-09 EDT` — ⚡ Attention 2.3× more: run FLASH_ATTN on the NPU for batched decode (root-cause fix) ✅
The v_trans fix below killed the `v_cont` but left attention as explicit `kq`/`kqv` score matmuls on **HVX (15.8 ms/layer at B=32)**. Tried the *other* lever — the user's insight — and it wins: **keep flash-attn ON the NPU**.

**Root cause (correcting the entry below):** Hexagon *does* have a working `flash_attn_ext` kernel (`flash-attn-ops.c` HVX + `hmx-flash-attn-ops.c` HMX). The only thing blocking it for batched decode was one over-conservative host gate — `ggml_hexagon_supported_flash_attn_ext`: `if (dst->ne[3] != 1) return false`. Batched decode packs the `npl` sequences into `ne[3]` (n_seqs), so that gate rejected exactly the B>1 case → auto-FA saw a device mismatch → downgraded flash off → the `v_cont`. **Single-token decode (B=1, ne[3]=1) always passed the gate** — so stagenet never actually had a v_cont problem (the "no FA kernel" claim below was wrong).

**Fix:** delete the gate (both FA kernels already iterate `ne[3]` — HMX `for (ib3=0; ib3<neq3; ++ib3)`, HVX `qrows=neq1*neq2*neq3` with per-row `iq3` + GQA broadcast), and set the NPU decode context to `flash_attn = ENABLED`. Now the whole attention is **one fused `FLASH_ATTN_EXT` op on HMX**, V stays in its natural layout (no transposed 65536-row scatter write).

**A/B on op12 (gemma-4-12B, B=32 batched decode, one layer), all bit-identical (rel_L2 5.23e-4, 0/32 PASS):**
```
                        (A) v_trans/explicit     (B) NPU fused FA
attention block            15.8 ms/layer            6.9 ms/layer   ← 2.3×
  · kq (HVX)                7.30 ms                  ─┐
  · kqv (HVX)              6.30 ms                   ├ one HMX FA op
  · softmax + kqv_out CONT  1.22 ms                  ─┘
  · V write                0.98 ms (65536×1)        0.03 ms (32×2048, natural)
full layer (+21 ms weights) ~37 ms                  ~29 ms          ← ~22%
CPU fallback                0                        0  (gate relaxed → all 35 FA ops on DSP)
```
Control run (`Bcpu`: FA on, gate NOT relaxed) proved the gate is load-bearing — 3 batched FA ops dropped off the DSP (35→32, i.e. to CPU). The advantage **grows with context**: `kq`/`kqv` (config A, HVX) scale with `n_kv`; the fused HMX FA streams it. **Adopted B as the default** (`GGML_DECODE_NO_FA=1` still selects the v_trans path for A/B). Bonus: the gate deletion means AUTO now resolves to FA-on-NPU everywhere, so any Hexagon-decode mode (stagenet/tail/…) is fixed for free. Full chain from the original bug: **193 → 37 → 29 ms/layer**.

### `2026-07-09 EDT` — ⚡ Decode 5.2× faster: kill the attention V-repack by storing V transposed ✅
Per-op profiling (`GGML_HEXAGON_PROFILE`) of the B=32 NPU decode on op12 exposed a shocker: one op, **`v_cont` (the attention V materialization), was 154 ms/layer — 80% of a 193 ms layer** — while the weight matmuls (q/k/v/o/ffn) were already fast on **HMX (~21 ms total)**.

**Root cause — a latent flash-attn/`v_trans` ordering mismatch:** the KV cache is created with `v_trans = !flash_attn`, but with the AUTO default `flash_attn` starts *true* → V stored **non-transposed**; then auto-FA downgrades `flash_attn` to *off* on the Hexagon decode device ("Flash-Attn tensor assigned to CPU, missing support"). Net: flash off **but** V non-transposed → the explicit attention path (`build_attn_mha`, the branch llama.cpp itself flags *"note: avoid this branch"*) **transposes the whole V cache every step**, and it scales with batch (M=1 = 4.85 ms → M=32 = 154 ms).

**Fix (1 line):** set `flash_attn_type = DISABLED` on the decode context up front → cache stores V **transposed** (`v_trans=true`) → `kqv = mul_mat(v, kq)` reads it directly, no `v_cont`; the transpose becomes a cheap per-token write. Prefill context keeps AUTO (flash-attn IS useful there and works on the Adreno GPU). Verified by a 4-agent adversarial workflow (Hexagon supports the transposed-V layout, no CPU fallback) + on-device re-profile.

**Re-profiled on op12 (B=32, 1 layer):**
```
                     before → after
v_cont (V repack):   153 837 µs → GONE (0)
LAYER TOTAL:          193.0 ms  → 37.3 ms     (5.2× faster)
correctness:          argmax 2/32 FAIL → 0/32 PASS, rel_L2 2.7e-3 → 5.2e-4   (also cleaner!)
```
The decode is now **HMX-weight-bound** (weights ~21 ms = 57%); the remaining attention cost is the `kq`/`kqv` score matmuls (~14 ms, HVX-flat) — a future flash-attn-on-Hexagon target, but no longer catastrophic. **Why flash-attn wasn't the answer for decode:** batched decode is B *independent block-diagonal* attentions (small per-stream `kq`), so FA's avoid-the-big-score-matrix win doesn't apply — the prize was the V transpose, which `v_trans` gets for free.

### `2026-07-09 EDT` — 🌐 3-DEVICE STREAMING PIPELINE runs the 12B end-to-end → correct answer ✅
The persistent hub-and-spoke pipeline (`stagenet` on each phone + `pipedriver` on the server) is **live on the real 12B across all three devices**. Unlike the old O(N²) file-relay (`pipeline_3dev.sh`, model reload per stage per token), this keeps **KV resident on every stage** and decodes **incrementally (O(N))** over TCP-over-USB (adb forward).

```
op15 stagenet[0,2)  ──hidden──►  op12 stagenet[2,3)  ──hidden──►  A6000 pipedriver tail[3,48)+sample
     (2 layers, CPU)                  (1 layer, CPU)                     (45 layers, CUDA)
        ▲ relayed token+residual dual-batch (12B PLAIN arch ignores the token; uses residual only)
```

**Run (12B fp16, chat template):**
```
prompt: "What is the capital of France? Answer in one word."
OUTPUT: <|channel>thought  The user is asking for the capital of France. The user requested a
        one-word answer. The capital of France is Paris. "Paris" is one word.<channel|>Paris<turn|>
```
**→ "Paris".** Correct, coherent gemma-4-it channel reasoning — a numerically-broken split would emit token salad, so this end-to-end validates the streaming relay on the 12B.

**Per-token decode breakdown (41 steps):** op15 (RTT+compute) **27.8 ms** ∥ op12 **30.4 ms** ∥ A6000 tail **36.6 ms** → **94.8 ms/tok (~10.5 tok/s)**. At this tiny 3-layer phone split the phone legs are USB-RTT-bound, not compute-bound — the rebalance (many phone layers, enabled now by [[per-tensor sharing]]) is the next step. Notes: host build rebuilt to match the fresh phone binary (protocol parity); port 5555 hit a TIME_WAIT bind snag → moved op15 to 5557; phones ran CPU for this first correctness pass (NPU/GPU dualengine-per-stage is the follow-on).

### `2026-07-09 EDT` — 🧩 PER-TENSOR weight sharing: req-3 one-copy now scales to any shard size ✅
Replaced the fragile whole-buffer size-match (which fell back to a 2× copy once Hexagon's ~1 GB and OpenCL's ~1.9 GB buffer caps diverged — see the M5 entry) with **name-keyed per-tensor aliasing**. Design came from a **17-agent workflow** (5 parallel code readers → 3-approach design panel → 3-lens judges → 5 adversarial verifiers → high-effort synthesis; ~1.15M tokens). Chosen: Design A + 2 verify-panel hardenings.

**Mechanism (8 edits, `ggml-hexagon.cpp` +51, `ggml-opencl.cpp` +199, behind the same `--share-weights` gate):**
- **Hexagon** `init_tensor` publishes each F16/F32 `.weight` as `name → {fd, base, import_size, offset=t->data−sbuf->base, size}`; exports `ggml_hexagon_shared_tensor_lookup()`.
- **OpenCL** import mode allocates a **1-byte dummy** buffer (reports full size so ggml-alloc's fake address space stays consistent); `init_tensor` resolves each weight by name, imports each **distinct fd once** (process-global registry), and points `extra->{data_device=alias, offset=hexagon_offset}` — kernels already take `data_device + offset` (no sub-buffer → no `MEM_BASE_ADDR_ALIGN` issue).
- **Hardening 1:** `set_tensor` **skips** the write for aliased weights → any offset bug is a benign mislocated *read*, never a scribble into Hexagon's live rpcmem.
- **Hardening 2:** a name-**miss** (compute/KV, or `token_embd` on a head phone) **lazily promotes** the dummy to one real requested-size buffer (keeps ggml-alloc peak-reuse; wires the `CL_LARGE_BUFFER_QCOM` retry for >2 GB token_embd).
- **xmem left byte-for-byte untouched** — its `{data_device, offset0}` os8-cache key stays unique because distinct weights carry distinct Hexagon offsets.

**Validated on op15, L=8 [2,10) — the exact case the old code FAILED:**
```
Hexagon published 4 buffers (fd 26/27/28/29, ~991/1050/1015/566 MB)
OpenCL: imported fd=26 once ... fd=27 once ... fd=28 once ... fd=29 once   (all 4)
        promoted-to-real = 0   (100% of 56 weight tensors aliased → ZERO OpenCL weight bytes)
        import FAILED     = 0   (all 4 concurrent EXT_HOST_PTR imports OK — resolves the multi-fd risk)
correctness rel_L2 = 2.755e-3  == bit-identical to --share-weights OFF  ⇒ numerically correct
```
So req-3 is no longer capped at single-buffer shards — a phone can now hold **many layers at true 1× weight**, which is exactly what M5's rebalance needs. Open items (from the verify panel): uncached-ion coherency relies on sequential load + the added `clFinish` (fine for dualengine); name-key assumes both models cover the same abs layer range; xmem-on-alias correctness still to be spot-checked on-device.

### `2026-07-09 EDT` — 📉 M5 crossover measured: linear throughput, but energy blocked remotely + sharing has an L-cap
Pushed op15 to more layers to look for the phone-vs-A6000 energy crossover. Four results, one positive, three limiting — all cross-checked on the real phone:

**1. Throughput scales linearly (reliable).** Clean, xmem-off op15 dualengine decode:

| Layers on phone | decode ms/round | ratio |
|---|---|---|
| L=4 [2,6) | 239 | 1.0× |
| L=8 [2,10) | 485 | 2.03× |

≈ **60 ms/layer**, linear. The scary L=8 = **1717 ms** seen in the first sweep was **not** scaling — it was xmem-**ON** RAM thrash: the `os8` prepack tile + two model copies drove op15 to ~540 MB free and it paged weights. xmem-off (decode is GEMV, doesn't need the os8 GEMM tile) removed the cliff.

**2. Energy: no defensible J/tok remotely.** Battery `charge_counter`/`current_now` = **0** (op15 is Full and USB carries the whole load). The only responsive rail is `usb/current_now × voltage_now` ≈ **1.7–1.8 W, near-constant across L** and sign-inverted — a sustained input-power ceiling, not a per-workload signal. A real J/tok needs the clean method: **unplug op15, drive it over WiFi-adb, integrate the battery discharge.** That is a physical action only the user can take; **I did not fabricate a crossover number.**

**3. Req-3 one-copy sharing works only in the single-buffer regime (new limitation).** The import matches Hexagon↔OpenCL weight buffers by **exact byte size** (offsets then line up via shared 128-byte alignment). That holds when the whole shard is one buffer (op12 [2,3) 448 MB, op15 [0,2) 855 MB → PASS). But the two backends **partition large weights differently**:
```
L=8 shard (3.6 GB):
  Hexagon publishes 4 buffers: 991 / 1050 / 1015 / 566 MB   (cap = sess->max_bufsize ≈ 1 GB)
  OpenCL   requests  2 buffers: 1923 / 1699 MB              (cap = CL_DEVICE_MAX_MEM_ALLOC_SIZE ≈ 1.9 GB)
  → no size matches → import falls back to a 2nd copy (correct, but 2× RAM)
```
Independent of xmem (verified). This is why **L=16 (2×7.3 GB) OOM-thrashes** a 16 GB phone. Fix is per-**tensor** publish/import, or equalizing the two caps — future work.

**4. The L=8 "FAIL" is fp accumulation, not a sharing bug (proven, not assumed).** The dualengine correctness check runs batched(B=16) vs B-single **entirely on the decode/Hexagon engine** — it never reads the shared prefill weights. rel_L2 = **2.755e-3** (well under the 1e-2 L2 gate) with **1/16** argmax-of-residual flips (a near-tie) tripping the strict `argmax==0`. Decisive control: with `--share-weights` **but import falling back to a 2nd copy** (xmem-off L=8), rel_L2 was **bit-identical** to the xmem-on *shared* run — so sharing cannot be the cause. It's inherent 16-wide-GEMM vs 1-wide-GEMV rounding over 8 layers. No cross-seq bleed.

**Verdict:** the Design-A machinery (reqs 1/2/3) runs at full-efficiency overlap and scales linearly, but (a) a defensible fleet J/tok is gated on physically unplugging op15, (b) op15's RAM + the per-buffer sharing cap limit clean many-layer runs to L≈4–8 today. The honest M5 answer awaits the unplug measurement; the machinery is ready for it.

### `2026-07-08 EDT` — 🧩 Build 3: ONE weight copy for both engines — LIVE (requirement 3 done) ✅
Wired the S2 proof into the real dualengine. Turned out to need only **3 small env-gated edits** (default behavior + xmem prepack-cache untouched), because the two-model structure means the OpenCL model's weight buffers stay normal OpenCL buffers — just backed by imported memory — so **no `supports_buft`, `init_tensor`, or loader changes were needed**:
- **ggml-hexagon** (`+36`): when `GGML_PHONE_SHARE_PUBLISH`, `alloc_buffer` publishes each rpcmem weight buffer's `{fd,base,size}`; exported `ggml_hexagon_shared_weight_take()`.
- **ggml-opencl** (`+51`): when `GGML_PHONE_SHARE_IMPORT`, `alloc_buffer` dlsym's the publisher, claims the matching buffer, and imports the rpcmem fd via the S2-proven QCOM path (`ion + UNCACHED + CL_MEM_EXT_HOST_PTR_QCOM|USE_HOST_PTR`) instead of `clCreateBuffer` — **no second copy**.
- **layersplit** (`+21`): `--share-weights` scopes PUBLISH to the decode model's weight load and IMPORT to the prefill model's, so KV/compute buffers stay private per engine.

**Why it's bit-correct:** both backends use **128-byte alignment** (verified on both phones: `CL_DEVICE_MEM_BASE_ADDR_ALIGN`=1024 bits), and Hexagon indexes weights by `t->data − sbuf->base` while OpenCL indexes by `t->data − get_base()` (fake base = alignment) — **both resolve to the same `tensor_offset` into the same physical rpcmem.** The prefill model writes each F16 weight exactly where HMX reads it (identical shard → identical bytes anyway).

**op12 [2,3) (448 MB weights):** handshake `published fd=23 … imported fd=23 — no second copy`; **CORRECTNESS PASS** (rel_L2 5.27e-4, 0/16); **1.80× overlap** (unchanged); **peak RSS 1570→1155 MiB, saved 415 MiB** ≈ the whole second weight buffer.
**op15 [0,2) head shard:** `published fd=24 (855 MiB) → imported fd=24 — no second copy`; **CORRECTNESS PASS** (rel_L2 5.04e-4, 0/16); **1.48× overlap** (855 MiB layer weights shared; the 1.9 GB `token_embd` stays on CPU/mmap for both — page-cache shared).

**→ All three requirements now BUILT + validated on real hardware: (1) one session/two backends, (2) static batch decode ∥ prefill, (3) one weight copy.** Env-gated so it composes with the existing pipeline; default path unchanged.

### `2026-07-08 EDT` — 🔑 S2 one-copy weight-share PROVEN on both phones (req 3 feasible) ✅
Standalone probe (`research_dev/spikes/s2_shared_weights/s2_shared_probe.c`, all-dlopen, no vendor link libs): a Hexagon **rpcmem** dmabuf written by the CPU is imported into OpenCL and read **BIT-EXACT by an Adreno GPU kernel — 0/1048576 mismatches on op12 AND op15**. Since `ggml-hexagon` already reads rpcmem natively for HMX, the GPU reading the *same fd* proves **one physical f16 copy can serve both engines** (requirement 3).

**The import combo that works (both devices, identical):** `cl_mem_ion_host_ptr{ allocation_type=CL_MEM_ION_HOST_PTR_QCOM, host_cache_policy=UNCACHED, ion_filedesc=fd, ion_hostptr=base }` + `clCreateBuffer(CL_MEM_EXT_HOST_PTR_QCOM | CL_MEM_USE_HOST_PTR, size, &h)`. Gotchas found by matrix-sweep: `USE_HOST_PTR` is mandatory (without it → -30); use **ion** allocation_type not dmabuf (the rpcmem fd imports as ION even though the ext string says dmabuf; dmabuf type → -59). Device caps (`cl_ext_probe.c`): both Adreno 750/840 expose `cl_qcom_dmabuf_host_ptr`+`cl_qcom_ext_host_ptr`(+iocoherent); `clImportMemoryARM` absent; page=4096, ext_mem_padding=0. Open for Build 3: clean RSS/smaps 1×-copy measurement; CACHED+WRITEBACK+one-time-flush vs UNCACHED for GPU read bandwidth.

**→ All three requirements now proven achievable: (1)+(2) BUILT on hardware, (3) SPIKE-PROVEN. Remaining: Build 3 = ggml shared buffer-type (rpcmem alloc + ION import), which edits the protected xmem files — needs coordination.**

### `2026-07-08 EDT` — ⚡ DUAL-ENGINE: one session, NPU decode ∥ GPU prefill, on BOTH real phones ✅
Built the `dualengine` mode (`examples/layersplit/layersplit.cpp`) — the user's design requirements (1)+(2), realized in ONE process:
- **Two `llama_context` over two `llama_model`** (one per device), two `std::thread` workers. Decode engine pinned `--dev-decode HTP0`, prefill engine `--dev-prefill GPUOpenCL`. Agent-confirmed safe: `llama_model` weights are read-only during decode (`build_graph` is `const`), each context owns its own KV/sched/backends, and distinct devices don't contend. The `events=false` pipeline-parallel gate is orthogonal (it only governs intra-context micro-batch overlap) — the two-thread approach sidesteps it.
- **Static B-way batched decode** (accumulate B, one `llama_decode` with distinct `seq_id`s) — clears the HMX B≥5 gate; prefill stays one-request-at-a-time.
- **Self-validating**: (A) batched decode vs B single decodes on the SAME engine → L2-relative diff (fp noise ~5e-4, argmax 0/16) proves no cross-seq bleed; (B) times each engine alone vs overlapped wall.

**Real-hardware numbers (12B fp16 shard, B=16, 8 rounds):**

| Phone | decode HTP0 alone | prefill GPUOpenCL alone | wall (overlapped) | speedup | correctness |
|---|---|---|---|---|---|
| op12 (v75, [2,3) mid/inject) | 914 ms | 1194 ms | **1197 ms** ≈ max(·) | **1.76×** | rel_L2 5.3e-4, 0/16 |
| op15 (v81, [0,2) head/token) | 1077 ms | 993 ms | **1078 ms** ≈ max(·) | **1.92×** | rel_L2 5.0e-4, 0/16 |

Wall == the *longer* leg on both → **zero-interference concurrent execution** (reproduces [[cross-engine-coschedule-npu-gpu]] inside one process). **op15 did NOT hang** at B=16 — the static lockstep batch avoids the continuous-batching path that hung at `n_parallel=2` ([[s1-npu-batch-decode-hang-localized]]); the user's "static batch first" call was right. Still 2× weight (one shard per engine); requirement (3) one-copy is next.

**Research settled two internals (2 Explore agents):** (i) two contexts genuinely share one read-only model's weights; the blocker to one-copy across HTP0+GPUOpenCL is `supports_buft` (session/context identity) + private repack layouts — but that repack is only for QUANTIZED types. (ii) **F16 weights are stored NATIVE-LINEAR on BOTH backends** (Hexagon skips repack for F16/F32; OpenCL f16 write is plain-linear) → the linear bytes ARE shareable. Hexagon already exports a dmabuf fd (`rpcmem_alloc2`→`rpcmem_to_fd`, `ggml_hexagon_shared_buffer{base,fd,size}`); OpenCL can import it via `clImportMemoryARM(CL_IMPORT_TYPE_DMA_BUF_ARM, fd)` (declared in the linked NDK headers, unused today). The xmem prepack reads that linear f16 `cl_mem`, so an import-alias feeds it directly → one shared linear copy + a small derived os8 tile. Gate = does the Adreno driver honor the ARM import (S2 probe).

### `2026-07-08 EDT` — 🚀 gemma-4 **12B** deployed to the ACTUAL PHONES over USB — correct coherent output ✅
The real target model, sharded across the fleet, generating correct text end-to-end:

```
 "What is the capital of France?"  --chat (channel template)
   [op15] L0-1 (shard 2.72 GB) ─USB─► [op12] L2 (shard 0.43 GB) ─USB─► [A6000] L3-47 + lm_head
   → "The capital of France is Paris."   ✓ (matches full model via llama-cli)
```

Each phone stores **ONLY its slice** (op15 2.72 GB, op12 **0.43 GB** — not the 24 GB model). Ran on all three phone engines, all correct:

| phones engine | 12B ms/tok (incl prefill+USB) |
|---|---|
| **NPU** (HTP0)   | **194** |
| CPU              | 219 |
| GPU (GPUOpenCL)  | 264 |

**What it took:** `--chat` (apply the model's channel Jinja template — raw prompts degenerate; missing BOS was a gotcha), the plain-arch injection fix (`3f7784540`), partial-load + shard tool (`c615983dd`), f16 conversion, and a phone-lib rebuild with the injection fix. The injection fix holds on op12's **real NPU** (middle stage, no segfault). Phones cleaned up after (0 procs, 0 forwards). *xmem files untouched.*

### `2026-07-08 EDT` — 12B split validated end-to-end on host; fixed a plain-arch injection bug 🐛✅ (`3f7784540`)
Ran the **12B f16** pipeline on host CUDA from real shards: `op15[0,2)` + `op12[2,3)` (shards) → server `[3,48)` (full f16, partial load). Caught + fixed a real bug **before** phone deploy (the point of host validation):
- **Bug:** a 12B *middle* stage segfaulted — the `ls>0` injection path always did `ggml_get_rows(model.tok_embd, inj_tokens)` (an E2B per-layer-rebuild leftover), but 12B has no per-layer embd and a middle stage doesn't load `tok_embd` → null deref.
- **Fix:** branch the injection on `model.per_layer_tok_embd`. Plain arch (12B) builds only the injected-residual input (`inj_h`) and uses it as `inpL` — no token, no `tok_embd`, no orphaned input. Guarded `llm_graph_input_embd_h::set_input` for the null token/embd tensors; loader keeps `tok_embd` when `n_embd_per_layer>0` (E2B).

**Correctness: the split reproduces the full f16 model's argmax bit-for-bit** (verified on several raw prompts — split and mono-full both give the same token id). Per-hop timing (12B): op15 1.8 ms, op12 1.1 ms, A6000 tail 35.6 ms.

**Gotcha found:** raw-prompt output is **degenerate** (`a a a…`) — but so is the *full* bf16 AND f16 model (identical), because **gemma-4-12B-it is instruction-tuned + "any-to-any"** with a complex **channel-based Jinja chat template** (`<|channel>`, `<|"|>`, function-calling). Raw completion prompts are out-of-distribution. Not a pipeline bug — coherent output needs the model's chat template applied (separate task). Also: 12B loads as `LLM_TYPE_UNKNOWN` (48 not in the gemma4 n_layer switch) — cosmetic, inference unaffected (E2B path identical).

**Also:** phone lib set rebuilt with the partial-load loader; f16 12B shards staged (op15 2.72 GB, op12 0.43 GB). Ready for phone deploy.

### `2026-07-08 EDT` — Phone stores ONLY its layer slice (partial load + gguf shard tool) ✅ (`c615983dd`)
Requirement: 12B fp16 (~24 GB) can't fit a phone → each stage must hold only its layers. Built two pieces in our repo:
1. **`gemma4.cpp` partial load** — `load_arch_tensors` now reads the same `LLAMA_LAYER_START/END` as the graph and creates ONLY layers `[ls,le)` (+ `tok_embd` for head/terminal, `output`+`output_norm` for terminal). Out-of-range tensors are never created → never allocated, never loaded. `llama-model.cpp` passes `done_getting_tensors(partial=true)` when the env is set so a full-gguf partial load doesn't trip the tensor-count check.
2. **`research_dev/shard_gguf.py`** — extracts a layer slice into a per-stage gguf, keeping **original block indices + all metadata** (block_count, SWA pattern, rope, tokenizer) so per-layer SWA/rope indexing is bit-identical.

**Validated:** E2B shards load (312/601 tensors for tail; `is_swa` correct per *absolute* index). **12B shard sizes:** op12 `[2,3)` = **0.43 GB** (1.9% of 22 GB), op15 `[0,2)` = 2.72 GB (2 layers ~0.9 GB + `tok_embd` ~1.9 GB). Also: **f16 12B conversion done** (bf16→f16, mandatory per the kernel audit).

**Note:** op15's 1.9 GB is the `tok_embd` table, kept because the head currently EMBEDS (`ls==0`). The intended design has the **server embed** and send the residual to op15 → then op15 needs no `tok_embd` (~0.9 GB). That's a driver/topology change (next). E2B shards stay large because the MatFormer `per_layer_token_embd` table (~1.3 GB) is global — a 12B non-issue.

### `2026-07-07 EDT` — Model = gemma-4 12B fp16; kernel-shape audit + S1 hang did NOT reproduce 🔎
On-device (op15, before the user reclaimed it): the recorded **S1 hang did NOT reproduce**. Lockstep batched decode (our `tailbench`, full model) ran B=1→**64** (27→51 tok/s); continuous-batching `batched-bench` (Q4_0) ran npl=1→**8** (TG 21→31 t/s) — all clean. The old "npl=2 hangs on op15" was the **fp16** sweep; fp16 npl=2 was the one run in progress when op15 was reclaimed (no verdict). So the hang is at worst fp16-specific, not multi-seq-attention-general. **#4 (static lockstep batch) is proven feasible on the NPU today.** ([[s1-npu-batch-decode-hang-localized]])

**Model locked: gemma-4 12B fp16.** Dumped the local gguf: n_layer=48, n_embd=3840, n_ff=15360, n_head=16/kv=8, head_dim=256, vocab=262144. **12B is the PLAIN arch** — `per_layer_token_embd=0` (no E2B token-relay hack) + `shared_kv_layers=0` (cut ANY layer). Simpler to split than E2B.

**Kernel-shape audit (code-grounded) — user's "be careful about shapes" concern resolved:**
- **bf16→f16 MANDATORY.** NPU `supports_mul_mat` has no BF16 case → whole matmul → CPU (`ggml-hexagon.cpp:2672`). The local file is bf16; must convert to f16 (~24 GB) before deploy.
- **All 12B GEMMs are kernel-clean** (every K,N ÷32; xmem K%16 + out≥64). Prefill→**xmem** (M≥16), decode→**HMX** (B≥5). Only lm_head falls off xmem (harmless, terminal).
- **HMX gate: decode B≥5** (`m≤4→HVX`). Static batch 32/64 is well clear. ✓
- **xmem triple-gated**: compile `-DGGML_OPENCL_USE_ADRENO_KERNELS` + env `GGML_OPENCL_ADRENO_XMEM_GEMM` + Adreno. Default OFF → l4_lm (~3–4× slower).
- **CORRECTION to earlier claim:** peak-kernels vs one-shared-weight is NOT strictly either/or. HMX reads the **native-linear f16 in place**; xmem **prepacks from that same linear copy**. With S2 dmabuf-import + **uncached** xmem you keep **both peak kernels at 1× persistent RAM** (small per-call repack tax); **cached** os8 = 2× RAM, zero tax (fine at a 2–3-layer split, ~2–3 GB). Only degrade GPU→l4_lm if the tax bites; HMX never lost. ([[gemma-4-12b-arch-kernel-shapes]], [[npu-decode-gpu-prefill-conditional]])

### `2026-07-07 EDT` — Design check: NPU-batch-decode ∥ GPU-prefill — verdict + S1 hang localized 🔎
Question raised: *"can we form a decode batch, run it on the phone NPU, and prefill on the GPU concurrently?"* Ran a 6-agent code+roofline review (4 readers of the actual backends + synthesis + adversary). **Verdict: partly on the same page — right goal, wrong as a static rule, currently unbuildable.**

**On the same page (correct):** batched decode is graph-feasible (gemma-4 graph is `n_seqs`-general; `tailbench` already assembles a real B-way batch, `llama_batch_init(n_streams,…)`, `seq_id[j]=j`, one `llama_decode`). Batching is exactly what pushes decode to high-M where the NPU/HMX wins. The utilization idea (NPU holds sustained decode, GPU absorbs bursty prefill) is legitimate.

**Corrections (why it's not a fixed rule):**
1. **It inverts the locked baseline** (decode→GPU, prefill→NPU) and is right *only* at sustained **B>4** (HMX gate) inside an NPU-favorable band — energy is **non-monotone**: NPU B≤16, **GPU B32–64**, NPU B128. Below B=4 it loses on *both* phases (NPU decode → HVX/no-win; prefill stranded on the 17× weaker GPU, 403 GFLOPS vs 7.12 TFLOPS). ⇒ must be an **adaptive router on batch occupancy**, baseline as low-load default.
2. **"Simultaneously" ⇒ two processes, not one.** One `ggml_backend_sched` serializes splits; pipeline-parallel overlap is force-disabled because OpenCL & Hexagon both report `events=false` ([opencl:8906](../ggml/src/ggml-opencl/ggml-opencl.cpp#L8906), [hexagon:3640](../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L3640), gate [llama-context:385](../src/llama-context.cpp#L385)). Two pinned contexts ⇒ **2× weight RAM** (no cross-engine dmabuf import = the S2 veto). 12B likely infeasible at 2× without S2.
3. **Zero-interference may not transfer** — it was NPU-*compute* ∥ GPU-*memory*; this pairing is likely *memory ∥ memory* (M=1 decode is bandwidth-bound), which our data says **contends** on the bus. Re-measure.

**S1 hang localized (the blocker).** `n_parallel=2` hung on op15 — *not* a llama deadlock (batching is lock-free, `n_seqs`-general). It's an **HTP backend bug**: host `flush_pending()` infinite-retries on the 1 s DSP timeout with no watchdog (`AEE_EEXPIRED → continue`, [hexagon:1516-1550](../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L1516)); DSP worker-pool busy-spins `while(atomic_load(&n_pending))` (`worker-pool.c:216`); `FLASH_ATTN_EXT` `supports_op` accepts a 2-seq attention op with no mask/KV validation ([hexagon:1883-1916](../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L1883)). **Fixable, and the floor** — nothing else builds until real B>4 decode completes on the NPU.

**Also found:** the serving path isn't ready — `stagenet`/`pipedriver` are strictly single-seq (`llama_batch_init(1,…)`, `seq_id 0`); a B-way NPU decode needs `tailbench`'s batch-assembly ported into the persistent socket loop (protocol carrying B tokens+hidden, `n_seq_max=B`, B residual replies).

### `2026-07-07 EDT` — Pipeline on phone NPU + GPU, A6000 CUDA terminal — per-hop timing ✅
Ran the persistent pipeline with the phone stages on each engine, host tail on the **A6000/CUDA** (was silently `-ngl 0`=CPU; added `HNGL` to `pipeline_persistent.sh`, default 99). Added per-hop timing to `pipedriver` (times only the 32 generation-phase steps). All three engines emit the same correct text.

Split: **op15=[0,2) → op12=[2,3) → A6000=[3,35)**, gemma-4-E2B Q4_0, 32-token decode.

| phones engine | stageA op15 (2L) | stageB op12 (1L) | tail A6000 (32L) | Σ decode/tok | full ms/tok¹ |
|---|---|---|---|---|---|
| **NPU** (HTP0)   | **34.2** ms | 7.4 ms  | 6.4 ms | 48.0 ms | 164 |
| **GPU** (OpenCL) | 16.5 ms | 13.1 ms | 6.4 ms | 36.0 ms | 111 |
| **CPU**          | 18.1 ms | 11.3 ms | 6.3 ms | 35.7 ms |  67 |

¹ full pipeline wall-clock/tok incl. prompt prefill + first-forward warmup (NPU graph / OpenCL kernel compile) — NPU pays the most here.

```
 per-token decode path (steady state), NPU config:
   op15 NPU 2L ──34ms──►  op12 NPU 1L ──7ms──►  A6000 CUDA 32L ──6.4ms──► sample
   └────────────── phone stages dominate ──────────────┘   └─ tail is cheap ─┘
   (USB RTT is tiny: residual = 1536×f32 = 6 KB/hop, <2 ms — the cost is phone dispatch+compute)
```

**Findings.**
- **The A6000 tail is not the bottleneck** — 32 of 35 layers decode in **6.4 ms** on CUDA. The 2–3 phone layers cost 5–6× more.
- **At this split the accelerators LOSE to CPU.** Single-token decode of 2 layers has too little compute to amortize the NPU's fastRPC dispatch (~34 ms on op15) or the GPU's kernel-enqueue overhead. Phone CPU is fastest end-to-end (67 ms/tok).
- **op15 (head) ≫ op12** disproportionately — the head also builds the per-layer token-embedding projection for all 35 layers, not just its 2.
- **op12 GPU** can't compile the split flash-attn kernel (`sub_group_shuffle_xor` unsupported on Adreno v75-era) → ran via fallback. Portability caveat.
- **Implication (R2/M5):** a 3-layer split is plumbing, not a win. The NPU/GPU only pays off after rebalancing many layers onto the phones (compute amortizes dispatch) and/or batching decode (S1). Backends confirmed live: op15 Hexagon v81 HTP0 (hvx 8, hmx 1, vtcm 8 MB) + Adreno 840 OpenCL.

### `2026-07-06 17:30 EDT` — Persistent pipeline: KV-resident incremental decode over USB ✅ (`a50edb599`)
Replaced the stateless act-file relay with **persistent stages** — each phone runs a long-lived server holding its model + KV; only the residual+token cross USB per step.

```
 host pipedriver (tail [3,35), drives)
   │  adb forward tcp:15555 (USB)          │  adb forward tcp:15556 (USB)
   ▼                                       ▼
 [op15] stagenet [0,2)  ──residual+tok──► [op12] stagenet [2,3)  ──► back to host tail ──► sample ─┐
   ▲ KV-resident                            ▲ KV-resident                                          │
   └────────────────── next token (feeds op15 for pos+1) ◄──────────────────────────────────────┘
```

- **op15[0,2) → op12[2,3) → server[3,35)** generated `"The quick brown fox jumps over the lazy dog and then runs away."` at **~320 ms/tok** (CPU) — vs ~6 s/tok for the re-prefill version (**~18×**).
- New driver modes `stagenet` + `pipedriver` (+ `connect_to`, `--port2`); frame `{pos,tok,nh,hidden}` carries the token id so each stage injects a **DUAL batch** (the per-layer-embd fix). Runner: [research_dev/pipeline_persistent.sh](pipeline_persistent.sh).
- Gotchas: killing the host-side `adb shell` doesn't kill the on-device process (use `adb shell pkill -9 -f layersplit`); **device port 5555 is adbd's wireless-adb listener** → use 15555/15556.
- Next: swap phones to NPU/GPU in the pipeline; measure per-stage USB RTT vs compute; then throughput at real cut ratios.

### `2026-07-06 16:45 EDT` — 3-device USB pipeline BUILT + generating text ✅
All phones are USB-connected to the server (they can't peer → server-mediated relay). Built an orchestrator ([research_dev/pipeline_3dev.sh](pipeline_3dev.sh) + `_gen.sh`) that chains the stages over **adb (USB)** with no new C++ transport — reuses head/mid/tail + a tiny `--tokens-file` for exact token feedback.

```
 input ─► [op15] head layers [0,2) ─residual(USB/adb)─► server ─► [op12] mid [2,3)
                                                                        │ residual (USB)
                                            next-token ◄── [server] tail [3,35) ◄┘
```

- **Single forward:** pipeline next-token == whole-model `mono` → **top-1 MATCH ✅** (heterogeneous op15-arm64 → op12-arm64 → server-x86; residuals cross as fp32, ~73 KB/hop, negligible).
- **Generation:** greedy decode over the pipeline produced coherent text —
  `"The quick brown fox jumps over the lazy dog and then runs away."`
- Stateless stages (no persistent KV) → each step re-prefills the full seq (O(N²) + a model reload per stage; ~2 s/stage). **This is the correctness/plumbing build.** Throughput build = persistent stages holding KV over `adb forward` TCP (the socket modes still need the dual-batch + token-relay fix + a `midnet`).
- ⚠️ paused mid-experiment: op15 in use by another agent.

### `2026-07-06 16:15 EDT` — On-device validation: op15 NPU + GPU + CPU ✅
Built the phone lib set from **this checkout** via the npu-harness snapdragon-docker (`scripts/build_npu_op12.sh --force` — builds all htp-vNN + opencl + libllama-with-fix, ABI-matched). Deployed to op15 (Hexagon v81 / Adreno 840) and ran the oracle on a 12-token prompt:

```
 op15 engine        mono         2-way k=13    3-way (2,3)     verdict
 NPU (Hexagon v81)  8784 @10.076 8784 @10.076  8784 @10.076    ✅ BIT-EXACT
 CPU                8784 @9.743  8784 @9.743   8784 @9.743     ✅ BIT-EXACT
 GPU (Adreno 840)   8784 @9.267  8784 @9.267   8784 @9.162     top-1 exact (3way logit = FP-noise)
```

- **All three engines predict the same next token** (8784 ' runs'); NPU + CPU reproduce the whole model to the bit through the 3-stage split → the injection/extraction survives the real **CPU↔NPU offload boundary**.
- GPU: 2-way is bit-exact; the 3-way logit drifts ~1% (extra CPU↔GPU residual round-trip at the mid stage = OpenCL reduction-order non-determinism, not a fix bug).
- Also ran on **op12** (Hexagon v75) earlier: NPU bit-exact too. Both phones covered; **op15 is the focus device.**
- Build recipe note: docker build compiled all of `libggml-htp-{v73,v75,v79,v81}.so` — deploy the arch matching the SoC (op15→v81, op12→v75); the arm64 `llama-layersplit` + `libllama`(fix) + `libggml-opencl` are shared.

### `2026-07-06 15:40 EDT` — Multi-token + 3-way pipeline bit-exact ✅ (commit `99224d3a7`)
Generalized the driver to N-token prefill and added a **middle stage** — the full op15→op12→server chain now validates end-to-end. A parallel 4-agent read-workflow first confirmed the **graph needs zero changes** (already N-token-general; already supports `ls>0 && le<n_layer`); all work was driver-only.

```
 12-tok prompt "The quick brown fox jumps over the lazy dog and then"
 MONO (whole model, last-token) ............ id=8784  logit=11.035674   ← reference

 2-way head[0,k)→tail[k,35):  k=3 ✅  k=10 ✅  k=13 ✅ | k=14 ✗   (k≤13 ceiling holds for N tokens)
 3-way head[0,2)→ mid[2,3) →tail[3,35): ...  id=8784  logit=11.035674   ✅ BIT-EXACT
        (= op15 → op12 → server baseline, through a real middle stage)
 single-token back-compat (--tok) ................................... ✅
```

- **act-file v2**: `{n_embd, N, tokens[N], residual[N*n_embd]}` — relays N cut residuals + the N token ids (needed at every hop to rebuild per-layer embeddings). New `mid` mode injects + runs head-less + relays onward.
- **Caveat for real transport:** the *socket* relay modes (tailnet/headnet/*stream) still inject `token==NULL` = the old residual-only path → would resurrect the per-layer-embd bug for a layer-split tail. They need the dual-batch + token-relay before use across devices. (Next.)

### `2026-07-06 15:05 EDT` — M1 FIXED → the split is bit-exact ✅ (commit `48b020120`)
Made the gemma-4 cross-device cut numerically correct. **The tail now decodes a DUAL batch** — the relayed input token rebuilds the per-layer embeddings exactly, the injected residual becomes `inpL`:

```
 HEAD [0,k)  --(act-file: n_embd, TOKEN_ID, residual[n_embd])-->  TAIL [k,35)
   token batch                                                     DUAL batch {token+embd}
   dumps residual @cut                                             token → per-layer embd (exact)
                                                                   embd  → inpL for layers [k,35)
```

- **3 files:** `gemma4.cpp` (tail builds token-embd manually — `build_inp_embd` prunes its embd tensor on a dual batch — then swaps `inpL` to the injected residual after the per-layer projection); `llama-graph.cpp` (null-guard `llm_graph_input_embd::set_input` so the token-only per-layer input tolerates `ubatch.embd`; no-op elsewhere); `layersplit.cpp` (relay token id + dual batch).
- **Reused existing infra:** the injected residual rides the MTP `embd_h.h` hidden-state input — **no new core class/API**.
- **Result — full cut sweep, gemma-4-E2B, 3 tokens:**

```
 cut k :  1  2  3  4  5  6  7  8  9 10 11 12 13 | 14 ...........34
 match : ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ |  ✗  (wrong)
                     bit-exact id+logit          ^ shared-KV boundary
```

- **Second finding — shared-KV cut ceiling.** `shared_kv_layers=20` ⇒ layers 0–14 own KV, 15–34 **reuse** it (last SWA owner = L13). So a cut is valid only for **k≤13**; k≥14 leaves the tail's KV cache missing reused entries. **Baseline cuts k=2 & k=3 are exact** → phones-hold-first-layers is safe; just don't cut inside the shared-KV tail. Memory: `gemma3n-perlayer-breaks-layersplit`.

### `2026-07-06 14:40 EDT` — M1 validation FAILED → found an architectural blocker
Ran the oracle on `gemma-4-E2B-it-Q4_0` (35 layers, CPU): `mono` vs `head→tail`.

```
 mono  (tok=1000)                  ARGMAX id=1000  logit=-12.27   ← reference
 head[0,k) → tail[k,35) :
   k=10  id=107   k=17  id=140   k=25  id=2   k=30  id=1000(logit -5.35!)   k=34  id=105
                                                    └ top-1 matches by luck, logit still wrong
```

**Every cut point is numerically wrong.** Root cause (proven in code, not just empirically):

```
 gemma-4 = Gemma-3n style. Each layer adds a PER-LAYER TOKEN EMBEDDING:
   inp_per_layer = per_layer_token_embd[ token_id ]          ← needs the INPUT TOKEN
   inp_per_layer = project(inp_per_layer, scaled_tok_embd)   ← needs the TOKEN EMBEDDING
   ...added at layer il for every il.
 A residual-only cut carries neither. The token-less TAIL stage silently hits the
 else-branch → uses the PADDING token (id 0) embedding for ALL its layers, and
 mis-projects the deep residual (±53) as if it were the token embedding. → garbage.
```

- **Not a port defect:** the fork's `gemma4.cpp` has byte-identical logic (`project_per_layer_inputs(inpL,…)` + padding fallback) → the fork's LayerSplit was **never numerically validated on a per-layer-embd model**.
- **This is exactly what M1 is for.** Caught before we built the batched/energy layers on top of a wrong pipeline.
- **Fix options** (in [PORT.md](PORT.md)): ① relay token IDs + inject residual (cheapest, recommended) · ② relay the projected per-layer tensor · ③ plain-arch fallback model. Memory: `gemma3n-perlayer-breaks-layersplit`.

### `2026-07-06 14:19 EDT` — Ported the LayerSplit driver → functional pipeline
Copied `examples/layersplit/` (1009-line driver) from the fork. **Built + ran with zero code changes** — this repo already ships `llama-ext.h` + the `embeddings_nextn` C API it needs. Committed `1f1b3f60a`.
- Modes: `mono` (reference) · `head`/`tail` (cut-activation correctness oracle) · `tailnet`/`headnet` (**raw-TCP** cross-device stage relay) · `tailbench` (batched tail decode).
- **Milestone:** with the LayerSplit hooks (`0a577ab99`) + this driver, the **single-model cross-device pipeline is buildable and functional** (M1 skeleton). Next: validate logits (mono vs head→tail) on a real model.

### `2026-07-06 14:03 EDT` — Started porting reuse code → branch `plan-a-port`
Bringing the Unifer fork's Plan-A code into this repo as clean ports (fork is at a different base — b9531 vs our b9850, so replay feature diffs, not copy). Tracker: [PORT.md](PORT.md).
- ✅ **RPC `tensor_extras`** (`ggml-rpc.cpp`, 5 hunks) — carries Adreno/HTP repack metadata across the RPC round-trip.
- ✅ **LayerSplit hooks** (`gemma4.cpp`, 5 edits) — `LLAMA_LAYER_START/END` bounds + head-less cut via `res->t_h_nextn`. **De-risked:** the `t_h_nextn`/`embeddings_nextn` plumbing already exists upstream here, so no new API needed; dropped the bundled fused-QKV optimization.
- **Difficulty map (PORT.md):** dma-buf zero-copy (our S2) is the hard one — its file drifted ~3500 lines and holds our xmem changes → do it **last**. RPC/LayerSplit were easy/moderate.
- ✅ **Build-verified + committed** on `plan-a-port`: native CPU+RPC build compiled both TUs clean (0 errors/warnings); commits `3f42a8504` (RPC) + `0a577ab99` (LayerSplit). xmem changes untouched.
- **Next:** the `layersplit` driver (`examples/layersplit/`), then transport, then dma-buf.

### `2026-07-06 13:45 EDT` — Reuse audit: most of Plan A already exists

Read Unifer's `PLAN_A_REUSE.md`; verified every component in the fork `~/Documents/llama.cpp @ route2-b9531` (= Unifer's `third_party/llama.cpp`).

- **Reuse — committed:** GPU↔NPU **dma-buf zero-copy** *(≈ our S2, already done!)*, VQ admission gate, HTP vision encoder (op15 only), MTMD embd inject; continuous batching is upstream.
- **Reuse — ⚠️ uncommitted (commit first):** layer-split (`gemma4.cpp` LAYER_START/END + `examples/layersplit/`), RPC `tensor_extras` patch, `examples/cdsd/` transport, split-K GEMV kernel.
- **Corrections:** op12 NPU fragile (vision-unusable; LLM matmul works — we measured it); **op15 is shared** (no broadcast pkill); USB-2 ~40 MB/s, RTT <2 ms; the doc independently confirms Plan A = **throughput** play, not an energy story at large batch.
- **Heads-up:** design docs live in `llama.cpp-release/research_dev`, but code + `project_*` memories live in **Unifer** → consolidate.

### `2026-07-06 13:33 EDT` — Design A system plan written → `research_dev/`
Ran an 8-subsystem design + adversarial feasibility workflow (agents read the real llama.cpp source). Output: **DESIGN.md**, **MILESTONES.md**, **README.md**.
- Feasibility: 2 feasible, 13 feasible-with-caveats, **1 risky-unproven** (batched phone decode).
- Code-grounded surprises: phones must hold the *first* layers (device-order pins `lm_head` to terminal); Gemma-4 shared-KV tail layers constrain the cut; llama.cpp pipeline overlap auto-disables; no OpenCL dmabuf-import path exists (needs new code).

### `2026-07-06` — Is batched decode even possible on the phone?
Checked the Hexagon backend: ops (MUL_MAT, FLASH_ATTN_EXT, softmax) *support* batched decode, **but** real multi-sequence continuous batching is unproven and `n_parallel=2` was recorded to **hang** on op15. The roofline "batch M" is one synthetic matmul — **not** N sequences with separate KV + masked attention. ⇒ This is the #1 risk; de-risk with `llama-batched-bench` (spike S1).

### `2026-07-06` — A6000 J/tok corrected; overflow scenario
Corrected my earlier over-optimistic server number: real A6000 12B decode floor is **~0.15 J/tok**, not ~0.03. In the *server-saturated / spillover* framing, phones become attractive because the alternative is lighting up another under-utilized 300 W box. Energy win is real but conditional on batched phone decode working.

### `2026-07-06` — Design review: A vs B
5-lens adversarial review. **B (hub-spoke replication + router) preferred**; A (pipeline) has real costs (host-relayed phone↔phone hop, bubbles, always-hot server). Prefill offload = robust energy win (~1.3–1.8×); decode offload = capacity/KV-relief, energy only if batched. User chose to build A anyway.

### `2026-07-06` — GPU efficient-kernel check + xmem re-run
Confirmed op12 GPU already uses the efficient `l4_lm` tiled GEMM (not a slow path); the ~400 GFLOPS ceiling is real, small-batch weakness is 64-wide tile underfill. Enabling **xmem image-GEMM + prepack cache** lifted GPU peak to **0.95 TFLOPS** (2.35× stock), engaging at batch ≥16.

### `2026-07-06` — Batch-size roofline sweep (op12)
Swept GEMM batch on NPU vs GPU. **NPU knee M=512 → 7.12 TFLOPS; GPU knee ~M=64–128 → ~0.4 TFLOPS.** NPU wins every batch except M=1. Produced a line-graph artifact. This is the empirical basis for "prefill→NPU, decode→GPU".
