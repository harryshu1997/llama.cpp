# Active warm-tier executable research plan

Status: `S39_TRACE_AND_SHARDS_READY; QWEN_BATCH_PROVISIONAL; W0_ELIGIBLE_TWO_MODEL_ATLAS_BLOCKED`

The current paper-critical system is defined in
[ACTIVE_WARM_TIER_DESIGN.md](ACTIVE_WARM_TIER_DESIGN.md). The first bounded
implementation is
[S39](spikes/s39_phone_model_switch_trace/PLAN.md).

One desktop GPU holds one hot large model. OP15 and OP12 collectively hold one
other executable warm model. When demand shifts, phones serve the warm model
while the GPU drains and loads its local copy. CUDA then batch-prefills the
prompt and committed phone token histories, consumes the small token delta, and
takes ownership at one exact token boundary. The phones release that model and
prepare the displaced GPU model for a later reverse switch.

The immediate gate order is:

1. **W0 eligible two-model atlas.** Freeze exact model identities and prove one
   complete CUDA route and one complete collective-phone route for each model.
   Measure phone warm TTFT/decode, CUDA load/unload, CUDA batched prefill, and
   phone rewarm. Storage-only residency is ineligible.
2. **W1 one-request catch-up.** Keep one request decoding on phones while CUDA
   loads and reconstructs native KV from tokens. Switch ownership once and
   continue for 32 tokens with no duplicate, missing, or stale token.
3. **W2 batched catch-up.** Repeat at `N={1,8,32}` with unequal request lengths,
   one authoritative phone frontier, batched CUDA reconstruction, delta
   catch-up, and exact sequence cleanup.
4. **W3 symmetric rewarm.** Release the promoted model from phones, prepare the
   displaced model from local UFS, publish new readiness, and execute a reverse
   switch.
5. **W4 trace comparison.** Run server queue, phone finish, catch-up handoff,
   and rotating-warm-tier controls on the frozen S39 trace with a finite,
   predeclared promotion/hysteresis sweep.
6. **W5 benefit and robustness.** Only after mechanics pass, expand arrival
   regimes, inject failures, and measure selected-GPU energy.

Do not implement direct KV assembly first. Token-history replay is the primary
handoff because it constructs native CUDA KV and transfers little data. Do not
transfer full checkpoints during a switch; desktop checkpoints and phone shards
are provisioned before the run.

Use USB ADB as the bulk provisioning plane and WiFi TCP as the runtime
command/activation plane. Weight readiness requires a verified phone-UFS
artifact before dispatch. The next real acquisition must capture socket peers
and interface counters; W0's WiFi endpoints are a post-run operator record.

The current blockers are concrete:

- Qwen3 B1/B8/B32 is token-exact but lacks a prompt corpus, repeated-process
  variance, zero-swap evidence, and a pre-captured network-path certificate;
- existing Gemma Q4/Q8 HTP routes failed the prior numerical-quality gate;
- no two-model load/serve/replay/rewarm timing atlas exists;
- no token-boundary ownership transfer exists;
- no symmetric reverse switch has run.

Everything below this line is retained historical Q-PIM, RAG, scheduler, and
evidence work. It supplies mechanisms and controls but is not the live roadmap.

---

Status: S36/S37 closed the next runtime gates on 2026-07-22. The physical
SLO-aware scheduler completed the frozen 60-request trace on one A6000, OP12,
and OP15. Every treatment used both phones, preserved all tokens and SLOs, and
produced one real phone batch containing prefill and decode rows. A separate
physical sweep completed every jointly resident handoff cut 4 through 8 twice
on both phones. See `spikes/s36_dynamic_cut_scheduler/RESULTS.md` and
`spikes/s37_arbitrary_layer_exit/RESULTS.md`.

The benefit gate failed. Across three paired runs, selected-CUDA stage relief
was +1.78%, -4.34%, and -1.53%; median treatment was 1.86% slower. Different
cuts fragmented terminal CUDA work into smaller `[cut,48)` calls. The next
finite gate is therefore the already scoped canonical cut lift:

1. Freeze canonical cut `K=8` and retain request-selected phone cuts 4..8.
2. Add one CUDA lift queue for exact `[cut,8)` execution and one shared tail
   queue for `[8,48)`; the shared tail must hold one weight image.
3. Preserve request, route epoch, position, token, and per-layer KV ownership
   through lift and tail. A request keeps one phone cut for prefill and decode.
4. Batch by `(device, cut, priority-band)` before lift and by
   `(canonical-cut, priority-band)` after lift. Prefill and decode rows may mix
   in one physical call; graph ranges and priority zero may not mix.
5. Repeat the same three paired physical runs. Require all S36 mechanics gates
   plus reproducibly lower median selected-CUDA stage time before measuring
   GPU-board energy.

Do not call the selectable cut a semantic early exit. Every request still
executes all 48 layers; the cut chooses where the exact activation and KV
ownership hand off from a phone prefix to the CUDA suffix.

Status: S28 completed the first real priority-safe shared-tail gate on
2026-07-21. The RTX 4060 Ti, OP12 `[0,8)`, and OP15 `[8,16)` processed the
frozen 60-request dense trace through one CUDA `[16,48)` tail queue. The
all-CUDA control completed 60 R0 requests; treatment kept 10 P0 requests on R0
and sent 50 P1/P2 requests through R2. Both had zero synthetic SLO misses.
Treatment reduced summed CUDA-island compute from 5.767 to 4.697 s (-18.55%)
and preserved P0 p95 (860.246 to 850.127 ms). OP12 and OP15 mean batches were
3.846/4. The shared tail interleaved route classes seven times and never mixed
P0 with background work in one physical batch.

This is a real request-level scheduler and server-work proof, not an energy or
accuracy proof. The latest-safe batching policy increased makespan from 5.007
to 41.977 s, although every synthetic SLO passed. F16-phone/Q8-server output is
uncertified (10/60 token sequences matched), and phone/network/total energy was
not measured. See `spikes/s28_priority_shared_tail/RESULTS.md`.

The next gate is not more scheduler scope. First make the server control
precision-compatible with the phone shards and repeat the same finite R0/R2
experiment. Then acquire matched 4060 Ti GPU-board energy with the already
validated request, placement, and session gates. Only if server energy falls
with preserved P0 SLO and acceptable quality should the work expand to richer
shapes or model residency changes.

Status: S25 completed the missing real continuous-request lifecycle proof on
2026-07-21. OP12 `[0,8)`, OP15 `[8,16)`, and the RTX 4060 Ti CUDA `[16,48)`
tail executed unequal output lengths with physical memberships
`AB, AB, CB, CB, CD, D`. C reused A's sequence slot while B remained live, D
reused B's slot while C remained live, all three workers drained to zero, and
all four dynamic greedy sequences matched same-route B1. Placement was
HTP0/HTP0/CUDA0 with zero missing compute buffers. See
`spikes/s25_continuous_lifecycle/RESULTS.md`.

This closes the basic implementation question: resident phone and desktop
workers support variable-row batches, per-request KV ownership, request-level
admission and retirement, sequence-slot reuse, and layer-boundary handoff. The
handoff is not semantic early termination; every request still runs all 48
layers. S25 is a real mechanics result, not a simulation or a benefit claim.

The next gate is policy, not another batching primitive. S24 proved that the
fixed SLO router is harmful even though shared batching works. Build one bounded
online controller that:

1. orders admission by priority and latest safe start before FIFO insertion;
2. selects only a measured route and device-specific batch candidate;
3. admits phone work only when predicted CUDA work or HBM residency decreases;
4. reserves downstream sequence and time credits before phone dispatch;
5. releases at the measured knee or the earliest latest-safe start, whichever
   occurs first; and
6. falls back to CUDA without waiting if those conditions are not met.

Run that controller first on a small real unequal-length mixed-priority trace.
Compare it against all-CUDA and S24's failed fixed policy. Require zero extra
priority-0 misses, at most 5 percent priority-0 p95 regression, and strictly
less matched CUDA work or resident HBM before acquiring GPU-board energy.

The S24 status below is retained as the negative policy baseline.

Status: S24 completed the real RTX 4060 Ti + OP12 + OP15 fixed-diamond proof of
concept on 2026-07-21. Physical batching and convergence mechanics pass, but
the frozen benefit gate fails. CP6 is not authorized for this policy.

The positive mechanism result is narrow: heterogeneous upstream routes can
converge into shared physical batches without a cohort barrier. The measured
fixed policy is not a useful system result. Relative to route-isolated queues,
shared OP15 batching raised mean batch from 2.0 to 3.0 and reduced makespan,
but it exceeded the priority regression bound. The SLO router increased CUDA
island compute, introduced two misses, and consumed 4.64x the selected-GPU
board energy of the all-CUDA control. F16-phone/Q8-desktop quality also remains
numerically uncertified.

Do not extend this exact diamond to broader traces, deeper overlap, or direct
phone transfer. A new checkpoint must first change one load-bearing condition:
use a precision-compatible server control and replace the route policy with a
priority-safe admission rule that can prove predicted CUDA relief before
dispatch. Freeze that as a separate experiment; do not reinterpret S24.

The S18/S19 plan below is retained as historical context.

The independent-lane design now runs end to end on one selected A6000, OP15,
and OP12. Six matched rows preserve 339,440 high-priority BGE encodes plus 384
low-priority Gemma requests per row. All 3,072 Gemma tokens per row are exact,
the phone routes overlap, both SLO classes pass, and median BGE p95 changes from
3,875 to 3,877 us. OP12 is fail-closed at its measured credit of two exchanges;
later loose-SLO groups use OP15.

The configuration does not pass the energy or memory goal. Median selected-GPU
board saving is only 0.342 percent and its uncertainty-adjusted lower bound is
-764.1 J. The two independent routes currently load two CUDA tail images, so
peak selected-GPU memory rises from 26,555 to 45,674 MiB. Verdict:
`S18_R1_FLEET_MECHANICS_PASS_RELIEF_INSUFFICIENT`. See
`spikes/s18_two_phone_r1_mixed/RESULTS.md`.

The next physical gate is the focused Q-PIM Funnel runtime: OP12 `[0,6)` rows
pass through a CUDA `[6,8)` bridge, OP15 `[0,8)` rows enter directly, and both
feed one variable-batch `[8,48)` CUDA tail. The paper-critical hardware remains
one selected A6000 plus OP12 and OP15; the slow external desktop link and the
second A6000 are excluded. A native llama batch cannot mix
boundary rows entering at different layers, so the bridge normalizes both
routes to the same layer-8 cut. The B32 S18 geometry remains a regression point
only. Production dispatch may choose only measured batch candidates, but it
must choose among them online rather than hard-code B32/B64.

The new mechanism name is heterogeneous-cut batch morphing. Continuous batching
within one executor is reused llama-server substrate. OP12 and OP15 release
independent stage-local batches at their measured candidates or latest-start
limits; after normalization, the tail forms a new compatible batch whose size
and membership may differ from either upstream batch. Exact row lineage and
distributed per-layer KV ownership make this transformation safe.

The request-lifecycle substrate follows llama-server's proven pattern: one slot
per sequence, one logical batch manifest rebuilt at every update, compatibility
filtering, prompt admission into residual capacity, and per-sequence KV removal.
It does not copy the server HTTP/task stack or expose activations through a
public server API. Q-PIM extends the compatibility key with route/cut,
residency, backend, activation-layout, and epoch identities, and adds only the
priority/SLO merge-release policy needed by the focused system. The bounded
implementation and physical gates are frozen in
`spikes/s19_dynamic_batch_runtime/PLAN.md`.

1. Freeze one CUDA `[6,8)` bridge and one CUDA `[8,48)` tail. Their weights are
   disjoint; `[8,48)` is loaded exactly once.
2. Keep request ownership, KV state, session epochs, and device credits isolated
   for OP12 prefix, OP15 prefix, CUDA bridge, and CUDA tail.
3. Add bounded KV-slot tables and token-boundary admission/retirement to both
   persistent phone heads and the server tail. Every step carries an exact
   request/position/epoch manifest; a request never changes its cut or KV owner.
4. Measure per-device batch candidates across relevant context, memory, thermal,
   and co-run states. The scheduler selects the largest useful SLO-feasible
   decode batch or the measured compute saturation point, subject to reserved
   downstream credits. Batch 1/2 is an urgent fallback, not a target.
5. Prove the frozen B32 route as a regression, then run arrival-varying tests in
   which active batch membership and selected size change between token steps.
6. Prove a physical split/merge event: unequal phone releases merge into one
   tail batch, or one upstream release is consumed by multiple tail batches,
   without a cross-phone barrier or token mismatch.
7. Repeat the S18 mixed screen and require lower peak HBM before any full energy
   acquisition. Stop if the shared tail or continuous-batch handoff serializes
   the lanes past their SLOs.
8. Only after that screen passes, rerun an equal-work selected-GPU test across a
   frozen workload-ratio sweep. Do not reconnect the general DAG/residency
   machinery in the paper-critical path.

Historical S17 status follows. Its optional hierarchical R2 middle island is
still stopped: HTP0 B64 is 1.50-1.53x faster than 2xB32, but HTP0 versus CUDA0
relative L2 is 8.886e-3 to 9.001e-3 against the frozen 5e-3 gate, with one of 64
row-argmax mismatches. No R2 result enters the S18 R1 claim.

Everything below this point is historical evidence and deferred design context,
not an active paper-critical implementation gate.

Previous S16 status follows.

The previous capacity-first mixed-workload scheduler plan is superseded. The new
target is dependency-aware power-frontier scheduling: reorder independent
multi-model DAG islands to unlock phone-resident work, then use power-trigger
bundles to create denser A6000 batches and measured lower-power intervals.

The current executable checkpoint is no longer simulation-only. S16 ran six
rotated real mixed BGE+Gemma rows on one selected A6000 and OP15. Every row
completed the same 339,440 high-priority BGE encodes plus 20 low-priority
Gemma B32 x 8-token cohorts. The P0 full-model CUDA control and the P2 OP15
`[0,8)` plus CUDA `[8,48)` treatment both kept their processes resident across
all twenty cohorts. All tokens and placement certificates pass independent
replay. BGE p95 is preserved (P2/P0 = 0.969), and every low cohort meets the
synthetic 5 s SLO.

The energy result is deliberately negative: all three pairs show a raw
selected-GPU reduction, but the median is only 0.75 percent and the 5 W Ampere
uncertainty-adjusted lower bound is negative. Verdict:
`MIXED_PERSISTENT_MECHANICS_PASS_GPU_BOARD_RELIEF_UNRESOLVED`. Phone, USB,
server-wall, and total-system energy remain unknown. See
`spikes/s16_mixed_persistent_energy/`.

Immediate next physical gate:

1. DONE: certify one persistent OP12 `[0,6)` B32 route with exact tokens,
   placement, reset, and process identity. Its 9.14-9.41 s route passes a
   predeclared 12 s lower-priority SLO; it is not eligible for the 5 s class.
2. Add OP12 as an independent READY request lane. Do not construct a serial
   OP15-to-OP12 chain.
3. Replay two disjoint low-priority B32 cohorts concurrently: OP15 `[0,8)` and
   OP12 `[0,6)`, each with its own CUDA tail work and bounded credits.
4. Keep BGE B16 on the selected A6000. Compare equal work under the persistent
   server-only baseline and the two-phone treatment.
5. Report both uncertainty-aware selected-GPU energy and iso-power SLO-valid
   work. Stop the energy direction on this hardware if neither reaches 10
   percent under the frozen fleet gate.

This gate tests the mixed-workload energy mechanism. It does not authorize a
total-system energy claim or dynamic weight streaming.

No production scheduler integration or total-wall energy claim is authorized.
Historical S10-V0 is invalid/inconclusive. S10-V0-R passes its bounded
temporal-enumeration and independent-optimality foundation.

**E1 (typed evidence binding) is COMPLETE**:
`TYPED_EVIDENCE_INTEGRITY_PASS_PHYSICAL_CLAIMS_BLOCKED`. Every oracle input binds
to a digest-pinned hashed artifact, and the measured atlas is empty. E1 rejects
every `MEASURED` instance on purpose: its solver is additive per-device, while
`SERVER_WALL` and `GPU_BOARD` measurements are aggregate timelines of a whole
boundary, and feeding an aggregate into an additive solver double-counts shared
power. See `spikes/s10_power_frontier_repair/`.

**E2 (matched control/treatment timelines) is COMPLETE**:
`E2_MATCHED_TIMELINE_MECHANICS_PASS_MEASUREMENT_NOT_RUN`. E2 compares two realized
timelines POST HOC instead of feeding aggregates back into the solver. The
mechanics pass and no measurement was run. Two blockers are now concrete rather
than vague, both from a first-hand instrument audit:

- **No SERVER_WALL instrument exists on this host.** NVML is GPU-board only
  (two A6000 boards; `power.draw` is a 1 s average, +/-5 W). RAPL is root-only AND
  package/core only (no dram, no psys). No BMC, IPMI, PDU, or external meter.
  `SERVER_RELIEF_PASS_TOTAL_ENERGY_BLOCKED` is UNREACHABLE without new hardware.
- **A physical label needs an aggregate evaluator that is not built.** A single
  pair is diagnostic only (`PAIR_ONLY_NO_AGGREGATE_CLAIM`); a label requires
  `SUM_ALL_PAIRS_V1` over a complete predeclared repetition set.

The one existing A6000 trace is negative evidence: 323 rows contain only 57
power-value changes (a 10 Hz poll of a ~1.7 Hz sensor), it spans four p-states,
and it is not a matched pair. See `spikes/s10_matched_energy_e2/`.

**E2A (all-pairs aggregate) remains the total-wall physical-claim evidence gate**:
`E2A_R4_ROUTE_DAG_MECHANICS_PASS_PHYSICAL_CLAIM_BLOCKED`. E2A builds the
`SUM_ALL_PAIRS_V1` evaluator E2 deliberately left unbuilt. Active v4 records bind
same-work identity, exact resolved control/treatment route DAGs, devices,
operator islands, request sets, phone-result data paths, leases, SERVER_WALL
capability, and complete plan-set commitment fields. The internal mechanics pass,
but three external blockers remain:

- **No INDEPENDENT EXTERNAL ANCHOR exists for the experiment plan.** An all-pairs
  sum is only worth something if the cohort was fixed before the results were
  seen; otherwise run 20 pairs, keep the best 8, and declare a plan of exactly
  those 8. A commitment needs PRECEDENCE (the plan predates the runs) and
  EXCLUSIVITY (only ONE plan was committed). **These do not covary.** RFC3161 via
  freetsa.org is genuinely independent and about ten minutes of provisioning away,
  and it buys precedence ONLY: a TSA is a responder, not a log, so nobody can
  enumerate how many other plans were anchored beside the one revealed. A token is
  a lower bound on a plan's AGE, never an upper bound on a plan's COUNT.
  **Provisioning a TSA would therefore not unblock E2A.** Closing it needs an
  enumerable commitment: third-party pre-registration, or a transparency log with
  a reviewable identity binding. TPM is present but permission-denied and
  custodially ours; git is our own force-pushable fork.
- **No eligible cryptographic verifier or independent trust root is registered.**
  The v4 interface carries the experiment identity, namespace, checkpoint,
  identity binding, complete plan count, and plan-set digest, but the production
  fixture still refuses at `E_ANCHOR_TRUST_ROOT`.
- **No witnessed acquisition launcher exists.** Internal lifecycle records can
  prove their own consistency, but cannot prove that physical execution followed
  the externally committed plan without a witnessed launch relation.

The three blockers stack and are independent: clearing any one alone changes
nothing. See `spikes/s10_matched_energy_e2_aggregate/` (ANCHOR_AUDIT.md is the
load-bearing document; RESULTS.md section 3 lists what is still open).

**S11-B (bounded implementation checkpoint) is COMPLETE**:
`BATCH_MECHANICS_PASS; KV_CAPACITY_PASS; SERVER_THROUGHPUT_RELIEF_FAIL;
PHONE_LEG_STABILITY_NOT_ESTABLISHED; ENERGY_NOT_RUN`. Exact static phone batches
pass at B=1,2,4,8,16. The complete OP15 `[0,2)` route scales from 3.18 to 25.49
req/s, but reaches only 41-51 percent of the equally batched A6000 control. The
complete repeated B=8 route has 4.74 percent CoV, while the OP15 leg alone has
8.52 percent CoV. The Gemma-4 layer-window KV repair reduces OP15 `[0,2)` from
1280 MiB to 4 MiB and OP12 `[2,3)` to 2 MiB at B=1, with exact two-phone output
retained. See `spikes/s11_batched_route_poc/`.

The post-sweep integrity repair is also complete: runner v2 is hash-bound and
fail-closed, and a versioned stage hello validates the exact layer chain before
work. Fresh repaired checkpoints remain exact at B=8 on OP15 and B=1 across
OP15+OP12, with the same 888 MiB and 1288 MiB A6000 relief respectively. These
checks do not change the throughput failure or certify per-node HTP placement.
The hello binds topology and execution capabilities only; S9 model-manifest and
prepared-image digests remain to be carried by the live PF3 protocol.

This authorizes the batch path only as an experimental scheduler primitive for
memory-pressure and admission tests. It does not authorize ordinary latency
routing, production integration, or a physical energy label.

**S11-E0 selected-A6000 board-energy diagnostic is COMPLETE and the fixed
serial route FAILS**: `GPU_BOARD_DIAGNOSTIC_RELIEF_FAIL`. All eight frozen ABBA
pairs pass exactness, SLO, placement, power, process, and thermal gates. The
OP15 `[0,2)` route releases 888 MiB and lowers average selected-board power from
289 W to 192 W, but equal-work runtime grows 2.31x and A6000 board energy rises
from 197.4 kJ to 302.3 kJ (+53.15 percent). The conservative relief lower bound
is -125.8 kJ.

Do not rescue this primitive with B=16 or a boundary sweep. It proves resident
phone execution and real A6000 residency/power changes, but also proves that a
serial phone barrier wastes too much server time. The next physical mechanism
must overlap phone work with useful server work across independent ready jobs.
Phone and total-system energy remain unknown. See
`spikes/s11_fixed_route_poc/{ACQUISITION_RESULT.json,RESULTS.md}`.

**S8/S12 trace substrate is PARTIAL and fail-closed**:
`REAL_COMPONENT_GATE_A_PASS; MIX_COMPOSITION_PASS;
VQ_MECHANICS_PASS; REAL_PROFILE_COVERAGE_PARTIAL; ENERGY_NOT_RUN`.
BurstGPT median and all RAGPulse component windows normalize reproducibly and
pass the structural reader. The frozen S12-V0 queue implements clairvoyant
server-only, causal server batching, fixed-phone, and memory-triggered policies
with bounded queues and exact HBM/activation/terminal ledgers. S12-V2 separately
passes synthetic two-model/two-phone `KEEP`, `PREFETCH`, `REPLICATE`, pin-safe
replacement, generation, and cold-miss server-fallback mechanics. It is the
state-reducer substrate for S14, not measured performance: it lacks real
multi-island DAGs, measured profiles, tail batching, thermal state, and energy.
The real normalized windows have zero exact S11 profile coverage, so they
receive no latency.

S14 has executed the deterministic mix transform and structural replay:
`mix-v1` contains 177 requests from the pinned BurstGPT and RAGPulse sources and
replays byte-identically. BGE support/correctness establishes a second service
class, while its server/phone latency atlas and a coherent Gemma
latency-plus-placement row remain the device-dependent gaps. Do not integrate
the VQ into `llama-server` until exact real-trace profile coverage and the
priority/SLO native-batch runtime exist.

**S12-V1 asymmetric data path is the current scheduler topology**:
`DUAL_PATH_CAUSAL_MECHANICS_PASS; STATIC_HOST_RESIDENCY_ONLY;
DYNAMIC_MIXED_POLICY_BLOCKED; SINGLE_PHONE_CONTEXT_NO_CROSS_GROUP_OVERLAP;
PATH_RATES_AND_INTERFERENCE_UNMEASURED; RUNTIME_TWO_SOCKET_PATH_NOT_IMPLEMENTED`.
Fast-loop input travels host-to-phone over the shared WiFi LAN; phone results
return on each phone's USB connection. These are separate directional
resources, but the current runtime cannot yet exploit their independence.
The slow residency loop uses USB H2P for large verified weight segments, while
the fast loop uses WiFi H2P for small commands, token IDs, and sequence
metadata. Dense phone results use USB P2H and preempt background weight traffic
on that phone's link.

The S12 replay now models bounded WiFi input, phone input/result buffers, phone
compute, per-phone USB result, host result, and A6000-tail phases. A phone
request cannot complete before its USB result and server tail complete. Each
run freezes either a full-model server residency or a tail-only phone-route
residency, charged for the whole horizon. Dynamic mixing is blocked until host
load/unload transitions exist. OP15 also has one KV context, so V1 limits the
phone to one in-flight group and reports zero cross-group path overlap. The
current executable route is only OP15 A0. OP12 needs its own complete-route
profile before the fleet scheduler can dispatch it.

This is not yet a latency result. The fixture uses assumed path rates and an old
single-socket phone-stage wall time as a proxy. The physical gate must measure
WiFi H2P, USB P2H, shared-WiFi contention, simultaneous WiFi/USB behavior, and
compute/link interference. Runtime realization needs correlated WiFi ingress
and USB egress sockets carrying one request and epoch identity, plus multiple
leased KV contexts or measured state switching before link overlap is useful. See
`spikes/s12_trace_vq/DUAL_PATH_DESIGN.md`.

S14's bounded C0-C5 harness controls are authorized only in the checkpoint order
below. PF1 physical claims and production runtime integration remain blocked.

## 1. Claim boundary

Primary claim:

~~~text
At equal closed work and end-to-end SLO:
  total wall J/completed work is at least 10 percent lower

or, at equal total wall power:
  SLO-valid completed work is at least 10 percent higher
~~~

The baseline is not eager llama.cpp. It is the best valid server-only policy
with the same DAG reordering, lazy batching, DVFS/power caps, and SLO knowledge.

Before a wall-power instrument exists, selected-A6000 `GPU_BOARD` energy is an
interim mechanism diagnostic only. It can falsify an offload route but cannot
satisfy the primary total-wall gate or establish net energy savings.

Skipped GPU-us, greater phone utilization, longer idle time, or lower modeled
energy is not sufficient. A claimed win must be explained by an actual batch or
power-state change at the complete wall boundary.

## 2. Preserved substrate and evidence

Reusable substrate:

- Design A route A0, stage-local KV, GGUF shards, and three-device transport;
- S8 source pins, schemas, normalization contract, and mixed-service DAG work;
- S9 versioned weight identity, residency, prepared-image, and lease contracts;
- examples/phone-pim durable provision/resume/publish/PREPARE/EXECUTE path;
- one independently checked resident Gemma dense-FFN route on both phones;
- separate OP12/OP15 USB 3.2 Gen 1 contention domains;
- a shared WiFi H2P input domain independent of the USB P2H result domains; and
- existing CUDA, HTP, OpenCL, transfer, and interference harnesses.

S9-V1A-R closes the current transport measurement slice:

~~~text
full-shard window gate:
  OP12 2.61x median, 1.89x conservative, best window 4
  OP15 2.28x median, 1.27x conservative, best window 8

0 gate errors, duplicate chunks, or wasted bytes
T1/T2/T3 resume/durability tests PASS on both phones
~~~

The result is partly DVFS-sensitive and OP15 reached 95 C. It proves bounded
pipelining, not contract completeness, multi-model capacity, or energy. Freeze
it as substrate. Do not implement SHA de-duplication or protocol v4 unless S10
shows transport is a selected schedule's bottleneck.

Negative evidence remains binding:

- S3 rejects generic phone-backend output-row GEMV splitting.
- S4 rejects Adreno attention at realistic KV context.
- S5 rejects treating phones as raw additive A6000 operator throughput.
- S6 saturated overlap is not a request-latency or energy result.
- KV ownership requires a contiguous layer range and explicit lifetime lease.

## 3. New gate order

~~~text
PF0  S10 small power-frontier opportunity screen
  |
  +-- FAIL -> stop Q-PIM runtime work
  |
  v
PF1  reproducible traces, DAGs, power/route atlas
  v
PF2  real-trace exact and causal oracle
  v
PF3  smallest live Q-PIM runtime
  v
PF4  exclusive HBM ownership and stateful routes
  v
PF5  scale, thermal, and optional grouped-HMX bonus
~~~

## 4. PF0: S10-V0-R foundation and small falsification screen

Detailed contract: spikes/s10_power_frontier_repair/PLAN.md.

Purpose: test the mechanism before changing llama-server, model graphs, KV
internals, or the production backend scheduler.

### PF0-A: freeze the controlled instance

- [ ] Bind the experiment to one selected A6000; administratively exclude the
      second installed GPU from scheduling, model placement, and accounting.
- [ ] Define two or three small DAG templates with explicit server pre/suffix
      islands and at least one phone-eligible complete island.
- [ ] Include at least two model/weight identities; same-model repetitions may
      provide native batches, while different models provide active bursts.
- [ ] Use concrete READY inputs and measured tensor sizes; no future-token or
      dependency prediction.
- [ ] Bind every CUDA/HTP/OpenCL route to current correctness and no-fallback
      evidence; UNKNOWN routes are excluded.
- [ ] Fix arrival, dependency, deadline/slack, model-mix, and load sweeps.
- [ ] Include concurrent high-priority BGE encode and low-priority Gemma decode;
      label priority and SLO fields synthetic when the source trace lacks them.
- [ ] Freeze server-only, placement-only, frontier-only, full-Q-PIM, and oracle
      controls before seeing the result.

### PF0-B: measure the minimum atlas

- [ ] Measure A6000 latency and energy versus batch and supported power
      cap/clock/state for every selected server island.
- [ ] Classify each selected route with measured roofline/throughput evidence.
      For decode, find the largest useful SLO-feasible batch; for compute-bound
      work, find the smallest batch at the throughput knee.
- [x] Run the S11-E0 selected-A6000 diagnostic as the first narrow screen;
      retain `formal_claim=NONE` and all excluded energy as UNKNOWN.
- [ ] Measure actual idle states, wake/transition latency and energy, and
      break-even gap. Do not assume a deep sleep state.
- [ ] Measure phone HTP/GPU latency and energy/thermal state for selected islands
      at available pacing controls.
- [ ] Include WiFi H2P input, USB P2H result, verification, host relay, and USB
      VBUS. Bind each direction to its physical path and contention domain.
- [ ] Measure shared-WiFi contention across OP12/OP15 and simultaneous WiFi H2P
      plus USB P2H. Do not infer no-interference from distinct hardware alone.
- [ ] Use an external synchronized wall boundary. If phones are host-powered,
      include their VBUS draw in the server-wall measurement and avoid adding it
      twice.
- [ ] Establish whether instrumentation can distinguish a 10 percent effect.

### PF0-C: independent tiny oracle

- [ ] Implement a standard-library exhaustive enumerator for the frozen tiny
      DAG set.
- [ ] Implement a standalone solution checker sharing no candidate or objective
      code with the enumerator.
- [ ] Enumerate all topologically legal orders, READY routes, allowed batches,
      power-trigger bundles, and power states within fixed small bounds.
- [ ] Include compute-pressure bundles that offload READY low-priority work and
      evaluate normal/lower GPU operating points with and without phone relief.
- [ ] Reject activation-memory overflow, invalid state, transfer omission,
      thermal-profile mismatch, and unfinished horizon work.
- [ ] Compare against optimized server-only, not eager FIFO.
- [ ] Add mutation tests for precedence, latest claim, wall-energy accounting,
      mirrored/exclusive credit, and terminal accounting.

### PF0-D: causal bounded policy

- [ ] Implement only a measurement/simulation policy: H-hop dependency lookahead
      plus deterministic bounded beam search.
- [ ] Generate unlocker sets, phone trigger bundles, and A6000 batch/power plans.
- [ ] Execute one simulated action and replan without future arrivals.
- [ ] Report oracle gap and separate prediction from correctness.

### PF0-E: controlled real-device reproduction

- [ ] Reproduce the smallest winning schedule with exactly one selected A6000,
      OP12, and OP15; verify the second GPU performs zero experiment work.
- [ ] Use resident weights and the existing bounded phone-PIM command path.
- [ ] Compare optimized server-only, fixed phone placement, and Q-PIM in rotated
      order with identical offered work.
- [ ] Run the decisive priority-differentiated overlap: high-priority BGE on the
      selected A6000 while low-priority Gemma decode forms native phone batches;
      admit returned boundaries to compatible server suffix batches.
- [ ] Compare normal-state server-only, lower-state server-only, normal-state
      phone relief, and lower-state phone relief as separate treatments.
- [ ] Run enough repeated processes for confidence, then a sustained thermal run.
- [ ] Account every activation/result byte and every late/canceled phone result.

PF0 opportunity gate:

1. perfect-future oracle improves total wall energy by at least 15 percent in two
   adjacent declared load bins versus optimized server-only;
2. the causal bounded policy retains at least a 10 percent improvement;
3. the controlled real run shows at least 10 percent lower wall J/work or 10
   percent higher iso-power SLO-valid work;
4. SLO attainment is no worse and all offered work has one terminal outcome;
5. a measured batch-density or A6000 power-state change causally explains the
   result; and
6. the result survives conservative transfer, thermal, and measurement error.

Possible verdicts:

- PASS: all six gates pass; authorize PF1 only.
- MECHANISM_PASS_ENERGY_BLOCKED: ordering/overlap works but wall power is invalid;
  do not build the full runtime.
- FAIL: oracle, causal policy, or real mechanism misses the gate; stop Q-PIM.

## 5. PF1: traces, DAGs, and complete measured atlas

Goal: generalize a passing mechanism beyond the tiny controlled instance.

- [ ] Finish S8-V0b deterministic normalization and structural replay.
- [ ] Pass Gate A with byte-identical BurstGPT and RAGPulse outputs.
- [ ] Freeze request DAG semantics for generation, RAG/embedding-rerank, and one
      encoder/background class.
- [ ] Materialize conditional DAG branches only when their inputs are known.
- [ ] Freeze TTFT, TBT, completion, priority, and synthetic-SLO provenance.
- [ ] Profile A6000/OP12/OP15 routes for at least two service classes.
- [ ] Add batch, active-burst, power-state, phone-pace, thermal, boundary, and
      pairwise-interference surfaces.
- [ ] Measure one-GPU compute-bound pressure points and verify that phone relief
      changes a real batch, power-cap, clock, or active-burst decision.
- [ ] Record full wall-power validity and uncertainty for every energy row.
- [ ] Run 30-minute resident/thermal stability for shortlisted routes.

Exit:

- at least two service classes have certified complete phone islands;
- every route has correctness, boundary, state, latency, power, and thermal data;
- Gate A and the amended two-class atlas gate pass; and
- the PF0 mechanism remains feasible under the expanded atlas.

If the result supports only one model, narrow the claim before proceeding.

## 6. PF2: real-trace exact and causal oracle

Goal: determine whether power-frontier scheduling survives real burstiness,
model mix, dependencies, and residency.

- [ ] Freeze instance and solution-certificate schemas.
- [ ] Add pinned one-worker CP-SAT only after exhaustive/checker agreement on
      all tiny fixtures and at least 1000 generated instances.
- [ ] Model slow residency and fast frontier decisions separately.
- [ ] Model ANNOUNCED weight demand and READY concrete execution separately.
- [ ] Include mirrored/exclusive ownership, drain/reload, state leases, USB
      domains, activation memory, thermal duty, and power transitions.
- [ ] Run causal rolling-horizon and separately labeled clairvoyant oracles.
- [ ] Evaluate the deterministic bounded online policy through the same checker.

Required policies:

1. eager server-only;
2. optimized server-only DAG order plus lazy batching/DVFS;
3. whole-request/fixed phone placement;
4. phone placement without frontier shaping;
5. frontier shaping without power-trigger bundle credit;
6. full Q-PIM;
7. clairvoyant upper bound.

Exit:

- full Q-PIM beats optimized server-only by the primary 10 percent gate in at
  least two trace scenarios;
- the causal bounded policy, not merely CP-SAT, passes;
- gains survive arrival, profile, thermal, and power uncertainty; and
- activation memory, transfer, and scheduler overhead do not erase the result.

Stop before runtime integration if the causal policy fails.

## 7. PF3: smallest live Q-PIM runtime

Goal: reproduce only the winning PF2 mechanism.

PF3-A now exists as a bounded compiled substrate. The
`examples/phone-pim/llama-phone-pim-fleet` target keeps one session per phone,
performs authoritative STATUS plus exact PREPARE, checks every FFN result
against a CPU oracle, and completion-schedules real jobs across OP12 and OP15.
This closes the simulator-to-device execution gap for one stateless operator
island only. It does not yet consume the S12 policy, execute a mixed-model DAG,
or alter A6000 work.

- [ ] Add a host DAG/VQ orchestrator outside ggml backend policy.
- [ ] Feed admitted request/island milestones from a bounded harness first.
- [x] Reuse verified residency, PREPARE, EXECUTE, and D2H result for one
      pre-staged Gemma4 dense FFN island.
- [ ] Implement authoritative READY, ownership, state, lane, link, and thermal
      reservations with generation-qualified epochs.
- [ ] Implement H-hop frontier construction and the bounded winning policy.
- [ ] Implement A6000 batch/power control and synchronized wall telemetry.
- [ ] Keep physical backend queues shallow and fail closed.
- [ ] Integrate one server path only after harness replay matches PF2.

Exit:

- live decisions match replay within declared timing/energy error;
- primary wall-energy or iso-power gate remains at least 10 percent;
- queues, state, and memory stay bounded for 30 minutes;
- phone loss, stale results, and thermal changes take declared outcomes; and
- no result relies on hidden fallback, duplicated HBM credit, or omitted power.

## 8. PF4: exclusive memory and stateful routes

Goal: add HBM relief without conflating it with mirrored fallback.

- [ ] Promote only proven mirrored routes to EXCLUSIVE_ACTIVE.
- [ ] Remove and measure exact server HBM allocations.
- [ ] Require drain plus completed reload before returning ownership.
- [ ] Add contiguous-layer KV leases only; reject token-prefix geometry.
- [ ] Include state handback bytes and replay cost.
- [ ] Evaluate low-priority batch decode and A0 as stateful controls.

Exit:

- at least 10 percent peak HBM or HBM byte-us relief at equal work/SLO;
- no immediate fallback copy is counted as released memory;
- state survives lease, drain, failure, and replay tests; and
- energy remains separately measured.

## 9. PF5: scale and optional kernel contribution

- [ ] Add devices through manifests and measured contention domains.
- [ ] Test one, two, and N phones with disappearance/rejoin and thermal rotation.
- [ ] Test model churn, cache pressure, fairness, and starvation.
- [ ] Screen cross-model descriptor-grouped HMX using test-backend-ops first.
- [ ] Include grouped HMX only if it improves a complete island by at least
      1.20x at two adjacent useful workloads with no fallback or hidden padding.

Kernel failure does not invalidate Q-PIM. It remains a measured bonus.

## 10. Global measurement and integrity rules

- Seven independent processes are the default latency protocol unless the spike
  freezes a stricter alternative.
- Controls rotate; first-run/cold effects are reported, not silently removed.
- Raw commands, stdout/stderr, exit status, hashes, device IDs, USB paths,
  thermal state, and exact sample counts are retained.
- Total wall energy uses synchronized windows and identical completed work.
- Missing or invalid power becomes UNKNOWN, never zero.
- Host, phone, and overlapping timers are not illegally summed.
- Capacity, HBM, time, and energy claims remain separately labeled.
- No commit, push, or upstream integration without explicit human approval.

## 11. Immediate instruction

Do not integrate `llama-server` or claim server/total energy relief. Historical
S11-E0 reached its stop rule and permanently rejected the fixed serial `[0,2)`
energy route. Do not run a post-hoc S11 boundary sweep or relabel a deeper range
as an S11 rescue.

The active bounded implementation target is S14, defined in
`spikes/s14_mixed_streaming_scheduler/PLAN.md`. It extends the existing S12-V2
state reducer rather than creating another scheduler. Execute these gates in
order:

### S14-CP0: mixed input and measured catalog

1. implement the frozen deterministic `mix-v1` transform and structural replay
   for BurstGPT plus RAGPulse;
2. make a second service/model class executable through its support-first,
   CPU-reference correctness, boundary, latency, memory, and no-fallback funnel;
3. profile a finite catalog of independently certified stateless islands and
   Gemma head cuts `[0,k)` on A6000, OP12, and OP15; and
4. freeze every candidate range before scheduler results. A stateful request
   pins its cut, phone, residency generation, and KV owner for its lifetime.

Stop with `TRACE_OR_SERVICE_BLOCKED` if two executable service/model classes do
not exist. S12-V2's symbolic two-model fixture is not a substitute.

### S14-CP1/CP2: static mixed runtime and placement policy

1. adapt measured rows into S12-V2 and compare optimized server-only, fixed
   placement, and mixed-VQ static residency with all weights pre-staged;
2. connect only passing decisions to S13's persistent OP12/OP15 sessions in a
   bounded harness outside `llama-server`;
3. let the slow loop choose single, replicated, or diverse phone placement and a
   certified layer/island range from visible ANNOUNCED demand;
4. keep finite phone, server, activation, result, KV/state, and queue credits;
5. require exact output, no fallback, measured placement, stable thermal state,
   and at least 0.90x optimized server-only useful throughput together with a
   concrete server-compute, batch, power, or HBM-relief mechanism; and
6. extend the bounded S10 temporal oracle with discrete placement, replica,
   memory, generation, and transfer decisions, then report causal-policy gap.

Before the general policy, run a one-GPU bounded screen with two priority
classes. High-priority BGE remains on the selected A6000 unless another route is
measured better. Low-priority Gemma decode may wait only within declared slack
to form the largest useful compatible OP12/OP15 batch. The scheduler admits a
returned phone boundary to a server suffix batch only after identity, epoch,
correctness, and D2H validation. Sweep normal and lower A6000 power/clock points;
do not lower the GPU state unless the complete joint schedule remains SLO-valid
and reduces measured energy per completed work.

Stop before streaming if static mixed mechanics fail. Replication serves a hot
repeated model; diverse placement covers different model demand. Both phones are
available, but forced utilization is not an objective.

### S14-CP3/CP4: overlapped residency streaming

1. add independent per-slot generations so READY/leased G0 keeps serving while
   one bounded G1 stages, verifies, publishes, prepares, and becomes READY;
2. do not reuse the current worker's single global generation, which would stale
   G0 leases during replacement;
3. use USB H2P for large weights, WiFi H2P for small commands/input, and USB P2H
   for dense results; result/state traffic preempts bulk weights at bounded chunk
   boundaries;
4. measure transfer, verify, UFS, publish, prepare, HTP/GPU, WiFi, result, and
   A6000 interference rather than assuming overlap is free;
5. dispatch only complete READY generations and make the server take its SLO-safe
   route instead of waiting for unfinished prefetch;
6. compare dynamic placement without overlap against dynamic placement with
   overlap on separate real source replays and low/median/high/burst `mix-v1`;
   and
7. report useful, unused, retried, and evicted prefetch bytes, queue/SLO outcomes,
   batch density, HBM, throughput, thermal state, and oracle gap.

The 216/262 MiB/s numbers are ADB-to-file staging controls, not READY-weight
throughput. The repaired S9 full-shard protocol reached about 36-37 MiB/s at its
best tested windows and OP15 reached 95 C. Weight reuse must amortize the full
transfer/verify/publish/prepare path.

The path-decomposition measurements remain mandatory: per-phone WiFi H2P, USB
P2H, shared-WiFi contention, simultaneous WiFi/USB behavior, and compute/link/
prepare interference. Replace S12 proxy rates only with digest-bound physical
rows.

Only an S14 mixed-runtime mechanics pass authorizes a new selected-A6000 energy
diagnostic. Do not reuse S11-E0 energy or SLO evidence. Phone energy remains
UNKNOWN, and server-wall/total-wall labels remain blocked until valid physical
instruments exist.

Preserve the E2A-R4 fixtures. The external commitment, witnessed acquisition,
and wall-power instrument are still required before any physical paper claim.
