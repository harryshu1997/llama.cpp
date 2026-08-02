# Active Warm-Tier Multi-Model Serving

Status: S40 Qwen/Qwen warm-tier path active; one real no-reboot joint B8
prototype passes execution with 60/64 diagnostic phone/CUDA token agreement;
formal V2.4/V2.6 Qwen3-14B A_ONLY remains blocked before model launch.

Operator-split checkpoint, 2026-07-29: one Qwen3-14B q8_0 FFN layer now runs
with distinct M-row inputs across CUDA and OP15 HTP over AOA. Three live
A6000 240/5001 plus OP15 repetitions improve median latency by 9.79%, 7.43%,
and 5.24% at M=1,2,4, reach only 0.61% at M=8, and lose 40.71% at M=16.
The physical 4060 is much more efficient than the throttled A6000 at M>=4; a
physical dual-CUDA delay-injection profile predicts that M=8 is already 6.81%
slower. This mechanism is therefore eligible only as an M<=4 candidate and
does not authorize prefill, energy, full-model, or switch claims. Direct AOA
on the physical 4060 remains required. See
`spikes/s41_gemma_qwen_continuous_baseline/tp_operator_split_v1/RESULTS.md`.

Operator-island successor, 2026-07-29: three single-row routes now pass on
real OP15 HTP plus the 240/5001 MHz A6000 proxy. A complete FFN residual slice
improves median latency by 7.54% for Qwen3-14B and 9.64% for Gemma-4-12B. A
sharded vocabulary head that reduces the phone suffix to one token and score
improves it by 16.10% and 17.12%. One Qwen GQA group with an in-graph current
K/V cache write improves median and p90 by about 9.6% at 8,192 cache entries,
but is rejected below that context because latency or tails regress. Each
selected result has three fresh 300-iteration workers and passes exact final
argmax plus the route-specific numerical gate.

These probes do not authorize a full-model or energy claim. The next narrow
core change is the disabled-by-default sharded greedy vocabulary head, followed
by complete FFN residual slices. Attention remains diagnostic until native
Q/K normalization, RoPE, masking, and rolling-cache semantics are present.
The direct AOA treatment must then run on the physical RTX 4060 Ti before a
BurstGPT or energy campaign.

Prototype checkpoint, 2026-07-27: Qwen3-14B executed concurrently on the
RTX 4060 Ti and the OP15 `[0,30)` plus OP12 `[30,40)` OpenCL route. Both
routes completed eight requests with eight continuation tokens and cleaned
from eight to zero live sequences. The phone route took 45.239 seconds versus
1.601 seconds for concurrent CUDA and relayed 15,626,240 activation bytes
directly over phone-to-phone Wi-Fi. This is functional evidence, not a
performance, quality, energy, or V2.4/V2.6 qualification pass. See
`spikes/s39_phone_model_switch_trace/v26_readiness/prototype_v1/RESULTS.md`.

Current checkpoint, 2026-07-26: CP0-R1 V2.4 is the sole prospective A_ONLY
exit authority for the S40 Qwen3-14B/Qwen3-8B path. It binds the canonical
64-item MMLU corpus and B8 token histories, exact RTX 4060 Ti and phone
identities, runtime bundles, producer process provenance, four orchestration
programs, their complete source support closure, and the final raw-bundle
reevaluation. The production sequence is artifact root -> phone reboot
preparation -> phase lock and exact preflight -> post-reboot identity and
network binding -> fast fresh readiness -> monolithic CUDA and joint
phone/CUDA capture -> fan-in -> authority. The prospective route remains
immutable; a separate bound root carries the observed boot IDs and WiFi
addresses used by the executed route. The live no-model topology now passes
on the exact RTX 4060 Ti, both physical USB phones, and all six WiFi
reachability edges. The prospective A_ONLY adapter passes its tests and the
frozen originator accepts the route shape. A versioned wrapper now adapts the
historical launcher to exact physical USB selectors without changing its
remaining validation or execution. Long pre-reboot and short post-reboot
artifact receipts remain unfinished. The sequence has not executed a model or
qualified a route.
Qwen3-8B phone provisioning, B_ONLY, PAIR, the reduced A -> B -> A cycle,
trace campaigns, and energy comparison remain blocked until Qwen3-14B A_ONLY
passes.

Preserved alternative checkpoint, 2026-07-26: the server-only S41 baseline uses
Gemma 4 12B IT Q8_0 and Qwen3 14B Q4_K_M on the real RTX 4060 Ti. Both models
pass independent full-CUDA B8 service and physical non-co-residency in both
orders. Two valid warm-cache GPU-switch repetitions complete 74/74 requests;
all three cold-NVMe repetitions strand the 57 Gemma requests. A dual-ready
Gemma-GPU/Qwen-CPU run completes but fails the zero-swap gate, while the
Qwen-GPU/Gemma-CPU direction passes resource gates but is CPU-bound. The raw
trace, normalized metrics, and server-only graphs are in
`spikes/s41_gemma_qwen_continuous_baseline/RESULTS.md`. No phone inference is
included in these baselines.

Implementation checkpoint, 2026-07-26: S40 now uses one experimental
llama.cpp router/controller with interchangeable GPU, CPU, and phone executors.
The feature is disabled by default. Its deterministic mechanics suite passes
249/249 tests, and both pinned desktop models pass real B1 and B8 execution
through the same native executor on the exact RTX 4060 Ti. Qwen3 8B Q8_0
completed B1/B8 in 0.336/0.983 seconds; Qwen3 14B Q4_K_M completed them in
0.349/1.452 seconds. Both produced exactly eight continuation tokens per
request and unloaded cleanly. At that checkpoint, the next physical gate was
the prospectively frozen V2.3 Qwen3 14B A_ONLY qualification across the RTX
4060 Ti, OP15, and OP12. Qwen3 8B phone provisioning, B_ONLY, PAIR, the
A -> B -> A cycle, trace campaigns, and energy comparison remained forbidden
until their preceding gates passed. This is the historical S40 Qwen/Qwen path;
its V2.3 authority cannot authorize S41 Gemma. See
`spikes/s40_shared_warm_tier_server/PLAN.md`.

Implementation checkpoint, 2026-07-25: CP0-D is complete on the target RTX
4060 Ti. Qwen3 8B Q8_0 and Qwen3 14B Q4_K_M each pass the independent B8
serving envelope, and simultaneous load attempts prove non-co-residency in
both orders. All three warm-page-cache replays complete 74/74 requests and
meet 74/74 synthetic SLOs. All three cold-NVMe replays fail closed with 17/74
requests complete, 2/74 within SLO, and 57 Qwen3 8B requests stranded. Mean
cold load is about 8 seconds versus about 1.7 seconds warm, so target changes
become overdue before useful admission. Selected-GPU board energy is measured;
server-wall and total-system energy remain unknown. See
`spikes/s39_desktop_swap_baseline/RESULTS.md`.

Implementation checkpoint, 2026-07-25: phone qualification is paused. The next
gate was a desktop-only two-model control on the target RTX 4060 Ti. It
measures what a one-GPU server does without a phone warm tier: Qwen3 8B Q8_0
and Qwen3 14B Q4_K_M run independently, their required serving envelopes are
tested for non-co-residency, and the frozen frequent-switch trace runs under
one drain, unload, load, and publish policy. Three warm-page-cache and three
cold-NVMe repetitions use the same request bytes, arrival order, SLOs, server
arguments, and switch intents. The gate records request TTFT and completion,
queueing, model publication gaps, SLO goodput, load/unload time, VRAM/RAM, and
selected-GPU board energy. Whole-server energy remains unknown without a wall
instrument. Only the pinned Qwen3 8B desktop artifact was acquired; this
gate created no phone shards and executed no phone commands. See
`spikes/s39_desktop_swap_baseline/PLAN.md`.

Historical evidence checkpoint, 2026-07-25: CP0-R1 V2.2 introduced the raw
phase bundle and remains an immutable parent. V2.3 became the Qwen/Qwen
readiness authority for A_ONLY and B_ONLY because it additionally binds long artifact
hashing, fresh post-lock identities, boot IDs, runtime executables, the exact
RTX 4060 Ti, and activation-byte interface counters. V2.2 permits an A-only
qualification, then a B qualification that re-evaluates
the raw A bundle, then a pair phase that re-evaluates both model bundles. Every
phase lock, readiness probe, and acquired event is bound to its own
`HOST_MONOTONIC_RAW` interval. The evaluator closes the corpus/output, exact
CUDA memory, oracle-length and B8 geometry, bridge linkage, exact activation
size, and full-shard UFS gaps. V2.2 additionally exact-checks the frozen
64-row pinned-revision MMLU corpus, a 25/64 CUDA sanity floor, incumbent A's
candidate route, eight-token continuations, causal publication timing, and
distinct phase IDs. Full device and artifact readiness uses
exact-checked commands and must finish immediately before each acquisition.
Final cycle authorization must reopen all three raw roots and pass the current
authority; status-only results are rejected.
This contract work does not qualify either phone route. The subsequent
desktop-only CP0-D gate acquired the pinned Qwen3 8B desktop artifact and ran
desktop control measurements; it did not authorize phone acquisition, phone
execution, or a phone-assisted switch cycle. See
`spikes/s39_phone_model_switch_trace/RESULTS_CP0_R1_V2_2.md`.

Historical initial CP0-R1 checkpoint, 2026-07-25: the next gate was
`TWO_ROUTE_ELIGIBILITY`, not a forward/reverse model-switch cycle. One
model-independent contract now applies symmetrically to Qwen3 14B Q4_K_M and
the only new candidate, Qwen3 8B Q8_0. Qwen3 14B remains
`PROVISIONAL_BATCH`; Qwen3 8B is selected but not acquired. Both must pass B8
target-4060 service, measured non-co-residency, complete direct two-phone
execution, memory and zero-swap gates, exact state mechanics, an independent
path-matched CUDA oracle, prospective task-quality noninferiority, pre-ready
live publication, and bounded local-UFS reprepare. Only a derived
`TWO_ROUTE_ELIGIBILITY_PASS` authorizes one reduced A -> B -> A cycle. See
`spikes/s39_phone_model_switch_trace/CP0_R1_TWO_ROUTE_ELIGIBILITY_CONTRACT.json`.

Implementation checkpoint, 2026-07-25: the separate CUDA-only replay-partition
diagnostic explains W9's exactness refusal without rerunning or repairing W9.
Two fresh `[8,3]+[1]` repetitions, two fresh `6 x [2]` repetitions, two fresh
`[8,4]` repetitions, and one same-process remove/replay sequence are each
internally exact. The incremental path matches W9's ledger, same-process
cleanup matches fresh execution, and both one-shot F1 geometries match each
other. Only incremental F0-plus-delta versus one-shot F1 differs. Future gates
must use a path-matched exact oracle and treat cross-geometry agreement as
diagnostic. The Qwen Q8 route remains stopped by task quality. See
`spikes/s39_replay_partition_diagnostic/RESULTS.md`.

Implementation checkpoint, 2026-07-25: S39 W9 failed closed on the first
prospective paid treatment. The real B8 route reached `F0`, exercised one
in-flight phone batch, replayed and caught up CUDA, durably changed ownership,
and published 13 post-start tokens per request. Its incremental CUDA
continuation then disagreed with the mandatory fresh same-frontier CUDA replay.
The frozen no-replacement rule stopped before the paired control and P2-P4, so
W9 has no repeated latency result and no pass certificate. See
`spikes/s39_phone_model_switch_trace/RESULTS_W9.md`.

Implementation checkpoint, 2026-07-25: S39 W8-R1 passed one real B8
live-session trace-mechanics gate. OP15 and OP12 held active KV and delivered
two decode rounds per request, 16 tokens in aggregate, before fresh CUDA
workers became ready. Promotion-trigger-to-next-token latency was 1.357 seconds
versus 3.283 seconds for the fresh CUDA teacher-forced trace control. This is
not new-request TTFT because prompt KV and two output tokens per request existed
before the paid clock. CUDA replayed the dynamic frontier, took ownership, and
continued exactly with zero leaked state. Full completion was 93.2 percent
slower because the fixed two-token-per-request post-frontier phone delta took
2.762 seconds while CUDA replay took 0.173 seconds. At that historical
checkpoint, the next bounded gate selected the number of intentionally
scheduled extra phone batches, `k_extra`; this measured regime selected
`k_extra=0`.

W8-R1 ran process-cold with a warm host page cache on CUDA0, an RTX A6000 with
48,530 MiB, not the target 16 GiB RTX 4060 Ti. It did not drain or replace an
already-resident model and used separate CUDA head and tail processes rather
than one native continuous server instance. The Qwen2.5 Q8 phone route remains
scheduler-ineligible after its separate task-quality failure, so no
multi-model, capacity, scheduler, target-GPU, trace, or energy pass is
authorized.

This document defines the paper-critical system. Earlier Q-PIM layer-split,
continuous-batch, transport, and mixed-workload results are retained as
substrate and controls. They are not the current headline.

## Research question

Can a low-power phone fleet act as an executable warm tier for a memory-limited
GPU, serving a non-resident model during a model switch and then transferring
live requests into one continuously batched GPU instance without a service
blackout?

The initial system has one desktop GPU that can hold one large model and two
phones that collectively hold one other executable model:

```text
                  one hot model                 one warm model
              +------------------+          +--------------------+
requests ---> | desktop GPU      |          | OP15 + OP12        |
              | continuous batch |          | sharded execution  |
              +------------------+          +--------------------+
                       ^                              |
                       +-- batched catch-up handoff --+
```

The system is a rotating executable cache:

```text
initial:  GPU = model A HOT        phones = model B WARM
switch:   GPU loads model B        phones keep serving model B
cutover:  GPU catches up B         phones transfer request ownership
rearm:    GPU = model B HOT        phones prepare model A WARM
```

The phone tier is not the only durable copy of a model. The desktop keeps every
checkpoint on local NVMe or host memory, and the phones keep assigned shards on
local UFS. Runtime promotion does not transfer an entire checkpoint over WiFi
or USB.

## Transport planes

The first implementation uses two independent physical paths:

```text
bulk plane:     host -- USB ADB --> phone UFS --> prepared phone weights
control plane:  host <-------- WiFi TCP --------> phones
data plane:     OP15 --------- WiFi TCP --------> OP12
result plane:   OP12 --------- WiFi TCP --------> host
```

USB carries model shards, binaries, and other large immutable artifacts before
they are needed. A shard becomes storage-ready only after its on-phone digest
matches the manifest. USB transfer is outside the paid serving window for the
first proof of concept.

WiFi carries runtime commands, request metadata, hidden-state activations, and
token results. It does not carry model weights. The W0 baseline relays an
upstream phone activation through the host coordinator. The target path sends
that activation directly from OP15 to OP12 after the host reserves downstream
credits and publishes a batch descriptor. OP12 returns terminal tokens and a
completion record to the host.

The host remains the control-plane authority. Direct transfer does not permit
OP15 to choose a downstream batch, route, or ownership epoch. Each direct frame
must bind the model digest, route epoch, batch ID, request IDs, positions,
layer boundary, tensor shape, and payload integrity. OP12 must reject a frame
without a matching host-issued reservation.

The paths may operate concurrently: USB may prepare the next warm model while
WiFi serves the current warm model. Readiness must bind the model generation so
an incomplete USB transfer can never become executable. A benefit claim must
measure both paths and must not hide provisioning inside an unmeasured setup
interval.

## Contributions

The work claims at most three contributions.

1. **Executable warm residency.** A non-resident GPU model remains immediately
   executable on a collectively sharded phone tier. Low-rate requests can stay
   on the phones, and a burst can trigger GPU promotion without first creating
   a serving blackout.
2. **Non-blocking batched catch-up.** Phones remain the output owner while the
   GPU loads the model and reconstructs native KV for multiple live requests
   from their prompt and committed token histories. The GPU catches up to the
   phone frontier and ownership changes atomically at a token boundary.
3. **Symmetric hot/warm rotation.** After promotion, the phones release the
   promoted model's runtime state and prepare the displaced GPU model. A
   hysteretic policy chooses which model occupies the one-entry executable warm
   tier and prevents model-switch thrashing.

Continuous batching, token replay, model loading, layer sharding, and SLO-aware
routing are required substrate. They are not individually claimed as novel.

## Scope

The first proof of concept is intentionally narrow:

- one target RTX 4060 Ti for paper-level evaluation; a selected A6000 may be
  used only for labeled mechanics development;
- OP15 and OP12 as one collective phone tier;
- two decoder models that cannot reside together in GPU memory;
- one executable model resident across the phones at a time;
- identical parent model digest, tokenizer, chat template, quantization, KV
  types, and context parameters on every route for that model; phone shard
  digests and layer ranges are independently derivation-bound to that parent;
- greedy decoding first, then deterministic stochastic sampling;
- direct phone-to-phone activation transfer with host-issued reservations;
- no total-system energy claim until phone and host energy are measurable.

The S39 Gemma HTP result remains negative evidence, but it does not exclude a
different phone backend. The current bounded pair is Gemma 4 12B IT Q8_0 plus
Qwen3 14B Q4_K_M. Gemma CUDA uses the exact parent Q8_0 artifact; each phone
uses an independently hashed Q8_0 shard whose layer range and derivation bind
back to that parent. Its first new gate is an all-phone Adreno OpenCL capacity
and kernel screen; the known HTP windows do not cover the full model without a
CUDA tail.
Qwen3 remains on GPUOpenCL because the phone HTP backend does not support
Q4_K_M. A model enters a physical switch experiment only after both its CUDA
route and complete collective-phone route pass identity, placement,
task-quality, memory, continuous-batching, and latency gates. Storage
residency alone is not eligibility.

The primary paper objective is service capacity and continuity at a fixed
one-GPU memory budget. The final evaluation must measure and show that both
models cannot coexist on the target GPU under the exact serving configuration.
Energy is secondary. The work can stop with a capacity result; an energy
contribution requires whole-server wall energy plus both phone chargers and
networking, not GPU board power alone.

## Residency and transition state

GPU residency, edge residency, and request ownership are orthogonal. A model
may briefly be ready on both CUDA and the phones during catch-up, so one
per-model enum would be incorrect.

```text
gpu_residency:  ABSENT | LOADING | READY | DRAINING
edge_residency: ABSENT | STAGING | READY | DRAINING
request_owner:  PHONE | CATCHING_UP | CUDA
```

The first system follows these global phases:

```text
A_GPU_READY / B_EDGE_READY
        |
        | promote B; drain A; phones serve B
        v
B_GPU_LOADING / B_EDGE_READY
        |
        | CUDA ready; phones still own B requests
        v
B_CATCHING_UP / B_EDGE_READY
        |
        | token-boundary ownership commit
        v
B_GPU_READY / B_EDGE_DRAINING
        |
        | release B phone state; prepare A
        v
B_GPU_READY / A_EDGE_STAGING
        |
        v
B_GPU_READY / A_EDGE_READY
```

Global invariants:

- at most one model has `gpu_residency=READY`;
- at most one model has `edge_residency=READY` in the first implementation;
- zero GPU-ready models is legal during an unload/load interval;
- the same model may be GPU-ready and edge-ready only during bounded catch-up
  and edge draining;
- a model is dispatchable only with a generation-qualified ready certificate;
- a phone allocation is not reclaimed while it owns a live request;
- only one execution owner may commit a token for a request and generation;
- no request waits for an unready phone or an unbounded model load;
- no runtime request depends on an unfinished USB weight transfer;
- failed promotion leaves the phone route authoritative and fails closed;
- failed rewarming changes readiness to false rather than preserving stale
  readiness.

## Promotion workflow

Suppose model A is hot and model B is warm.

1. A B request that triggers promotion is admitted to the still-authoritative
   phone tier before the same-timestamp promotion intent is applied.
2. Existing A requests continue on the GPU. The first prototype drains them
   rather than migrating them.
3. New B requests are admitted to the phone fleet and continuously batched.
4. The slow loop observes sustained B demand and begins B promotion.
5. The GPU stops admitting work that would make A impossible to drain within
   the promotion bound.
6. The desktop loads B from local NVMe or host memory. Phones continue serving
   B throughout this interval.
7. When B is ready on CUDA, the coordinator snapshots each live B request's
   last committed phone frontier `F0` and immediately starts CUDA replay from
   `F0`.
8. The coordinator stops new phone submissions after the zero or one batch
   already in flight, plus only the extra batches authorized by the frozen
   cutover policy. Phones remain the only publication owner during this
   interval.
9. The phones acknowledge a final frontier `F1` with token-history digest and
   ownership epoch. At `F1` they cease dispatch and publication but retain KV
   under the old epoch for rollback. CUDA consumes exactly `F1 - F0`; a
   zero-width delta is a valid explicit no-op path.
10. Once CUDA reaches `F1`, the coordinator durably commits the new ownership
    epoch. CUDA cannot publish before this commit. Only then may the phone KV
    be released and CUDA join the requests to its continuous decode batch.
11. After every B request and buffer is released from the phone tier, the
    phones prepare model A from local UFS and publish a new ready certificate.

The trace gate requires real requests for both A and B. Each model must execute
on the phone tier while warm and on CUDA after promotion. Alternating residency
records without corresponding request/result ownership records do not count as
a model-switch result.

If A demand returns before A is ready on the phones, the request follows an
explicit bounded fallback. The system must not report A as warm.

For the bounded cutover policy, `k_extra` counts intentionally scheduled
complete phone decode batches after an optional zero-or-one batch already in
flight at CUDA readiness. It does not count that optional batch. For a fixed
homogeneous cohort, one such batch produces one token per live request.
`K_max` is finite and comes from the remaining output budget while preserving
at least one CUDA continuation token:

```text
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

Predictors are cumulative, monotone, integer microsecond estimates rounded up.
The margin is nonzero and all inputs are frozen before acquisition. The final
realized delta is the optional in-flight result plus `k_extra`, and may be zero
when no batch was in flight. CUDA replay of `F0` overlaps a remaining phone
batch when one exists; it must not wait for `F1` before starting. A later
scheduler may optimize a broader SLO objective, but the first prospective gate
evaluates only this bounded rule.

On any failure before the durable CUDA commit, CUDA state is discarded and the
phones remain authoritative under the old epoch. After the commit, the old
phone epoch can never publish again.

## Request-state contract

Every live request carries:

```text
request_id
model_digest
tokenizer_digest
route_epoch
ownership_epoch
current_owner
prompt_token_ids
committed_output_token_ids
last_committed_position
sampler_state
stop_state
deadline_us
priority_class
```

The coordinator already knows the prompt and every committed output token, so
the primary handoff transfers only a bounded delta and metadata. Hidden
activations or final logits cannot reconstruct historical per-layer KV.

Direct KV migration is a later optional optimization. It is not required for
the first system. If added, the scheduler chooses the cheaper exact path:

```text
handoff_cost = min(
    export_KV + network_KV + import_KV,
    batched_native_prefill + token_delta_catchup
)
```

Direct KV is eligible only when the complete state layout is version-compatible
and a same-route continuation oracle passes. Token replay is the fail-closed
fallback.

## Sampling and ownership

The first gate uses greedy decoding. Stochastic decoding must use either a
migrated sampler state or a counter-based random stream keyed by
`(request_id, token_position, sampling_config_digest)`.

During catch-up:

- phones are the only output owner;
- CUDA may compute state but cannot publish tokens;
- every committed token is identified by request, position, and ownership
  epoch;
- a duplicate, gap, stale epoch, or conflicting owner aborts the cutover;
- after the cutover record is durable, CUDA becomes the only output owner.

## Two-level scheduler

### Slow loop: residency and promotion

The slow loop runs at model-load timescale. It selects the phone warm model,
starts preparation, and promotes or demotes models.

For model `m`, the warm value is based on:

```text
warm_value(m) =
    predicted_arrivals(m)
  * server_switch_penalty(m)
  * expected_SLO_loss_without_warmth(m)
  - phone_prepare_cost(m)
  - eviction_cost(current_warm)
  - thermal_risk(m)
```

Promotion requires sustained queue pressure, not one arrival. Demotion uses a
different lower threshold and a minimum residency interval.

### Fast loop: request and batch ownership

The fast loop:

- orders work by priority and latest safe start;
- admits only to a ready, identity-compatible model route;
- forms model-homogeneous continuous phone and CUDA batches;
- releases at a measured useful batch or the earliest latest-safe start;
- reserves downstream sequence and catch-up credits;
- preserves exact per-request ownership;
- falls back without waiting when a route becomes unsafe.

Different models do not share one physical transformer batch. They may overlap
on different devices.

## Feasibility conditions

The live-session bridge is useful only when all of the following hold:

```text
live_phone_next_token_time < GPU_model_ready_time
phone_capacity >= arrivals_during_promotion
CUDA_decode_rate > phone_decode_rate
catchup_time <= remaining_request_slack
phone_rewarm_time < expected_time_to_next_reverse_switch
```

New-request TTFT is a separate gate. If phone prefill does not beat GPU
readiness, do not claim a new-request bridge; already-live sessions may still
benefit, and phones may still serve low-rate models by avoiding promotion
entirely. If both live-session continuity and avoided switches fail, stop the
direction.

## Controls

- `C0 server_queue`: queue model B while A drains and B loads.
- `C1 phone_finish`: phones serve B during the switch; every phone-started
  request finishes on the phones.
- `C2 catchup_handoff`: phones bridge the load; CUDA batch-prefills histories
  and takes over live B requests.
- `C3 rotating_warm_tier`: C2 plus preparation of displaced model A and a later
  reverse switch.
- `C4 oracle`: both models resident on separate GPUs. This is a performance
  bound, not the resource-matched baseline.
- `C5 host_warm_executor`: the alternate model is executable from desktop
  CPU/RAM while the one GPU changes residency. This is the simplest
  resource-matched alternative to the phone tier.

All controls execute the same request identities and token budgets.

## Metrics and claim boundary

Primary:

- switch-period SLO goodput;
- P50/P95/P99 TTFT for genuinely new requests;
- promotion-trigger-to-next-token latency for already-live sessions;
- common promotion-period response gap: request arrival or previous published
  token to the next publication, defined for every control;
- route-specific last-phone-to-first-CUDA handoff gap;
- per-request TPOT and aggregate decode throughput;
- model-load blackout duration;
- useful tokens produced by phones during promotion;
- CUDA replay and delta-catch-up time;
- phone and CUDA batch-size distributions;
- promotion, demotion, and rewarm time;
- duplicate, missing, or stale token count;
- bytes transferred by class;
- peak GPU memory and host staging memory.

Secondary:

- selected-GPU board energy for matched equal work;
- number of GPU model switches avoided;
- phone useful-work ratio;
- wasted preparation and canceled bytes.

`PHONE_ENERGY`, `HOST_ENERGY`, and `TOTAL_SYSTEM_ENERGY` remain unknown until a
valid physical boundary exists. Selected-GPU energy is reported as such.

Persist unambiguous metric names. `new_request_ttft_us` is reserved for a
request whose prompt was not prepared before the clock.
`promotion_next_token_us` measures an already-live session from promotion
trigger to its next published token. `promotion_response_gap_us` is the common
arrival-or-previous-token to next-publication metric. `cuda_ready_us`,
`handoff_gap_us`, and `completion_us` retain their literal event boundaries.
Do not relabel promotion latency as TTFT.

## Milestones

### S41-R1 - Gemma-Qwen two-route eligibility

- Freeze a new S41 successor contract before any paid phone attempt. Reuse the
  S39 evidence mechanics, but do not let its Qwen/Qwen V2.3 authority authorize
  Gemma.
- Bind the Gemma Q8_0 and Qwen3 Q4_K_M parent artifacts, plus independently
  hashed OP15 and OP12 shards with exact derivation and layer ranges.
- Preserve the completed target-4060 qualification and physical
  non-co-residency evidence for both parent models.
- Run only the remaining Gemma cut-31 capacity attempt. Stop the current
  Gemma/Qwen phone campaign if either phone swaps or misses the memory or
  placement gate.
- If capacity passes, prospectively select the phone kernel using B1, B8,
  mixed prefill/decode, thermal, and quality evidence. Then prove dynamic
  continuous admission, retirement, and slot reuse.
- Refresh Qwen3 phone evidence and apply one pinned task-quality
  noninferiority gate against each same-parent CUDA route. Cross-backend and
  cross-geometry greedy agreement remains diagnostic only.
- Prove direct activation transfer, exact state mechanics, path-matched CUDA
  replay, useful phone publication before CUDA readiness, and both local-UFS
  reprepare directions.

Exit: a new S41 evaluator reopens the raw Gemma, Qwen, pair, and reprepare
roots and emits one S41-specific reduced-cycle authorization. S39 V1, V2,
V2.1, V2.2, V2.3, and status-only records cannot authorize that cycle. No
scheduler may treat storage-only or provisional residency as executable.

### W1 - One-request non-blocking catch-up

- Keep one request decoding on phones while CUDA loads.
- Reconstruct CUDA KV from committed tokens.
- Consume the phone token delta and switch ownership once.
- Prove no duplicate, missing, or stale token and bounded inter-token gap.

Exit: one same-model request crosses from phones to CUDA and continues for 32
tokens with a valid continuation oracle.

### W2 - Batched catch-up

- Repeat with `N={1,8,32}` live requests.
- Batch native CUDA reconstruction and delta catch-up.
- Admit and retire unequal request lengths.
- Keep phone output authoritative until one atomic batch cutover.

Exit: all requests are conserved exactly once and batched handoff is faster
than sequential replay at a useful `N`.

### W3 - Symmetric rewarm and reverse switch

- Release the promoted model from phones.
- Prepare the displaced model from local UFS.
- Publish a new generation-qualified readiness certificate.
- Reverse the workload and execute the same catch-up path.

Exit: `A hot/B warm -> B hot/A warm -> A hot/B warm` completes without stale
weights, stale KV, or false readiness.

### W4 - Frozen multi-model trace

- Replay the S39 trace or a denser trace with the same frozen provenance.
- Run C0-C3 and C5 with identical work; report C4 as an upper bound.
- Sweep promotion threshold and hysteresis from a predeclared finite set.
- Report queueing, handoff, rewarm, batching, and switch avoidance.

Exit: C2 or C3 improves switch-period SLO goodput or SLO-valid admitted
capacity by at least 20 percent over the best resource-matched control, without
losing requests or regressing steady-state hot-model SLO. P95 new-request TTFT
and the common promotion-period response gap must each be at most 110 percent
of the best resource-matched control. Report route-specific handoff gap as a
diagnostic, not as an independent paper pass.

### W5 - Benefit and robustness

- Repeat only the three predeclared regimes: sparse alternate-model demand,
  one sustained alternate-model burst, and oscillating A/B demand.
- Inject phone disconnect, failed load, stale epoch, and delayed rewarm.
- Measure selected-GPU energy only after the latency mechanism passes.

Exit: report the regime where the warm tier is beneficial and the regime where
it is not. Do not generalize beyond measured devices and models.

## Stop rules

Stop or narrow the design when:

- no second model passes a complete collective-phone execution gate;
- phones cannot publish useful live-session work before measured GPU readiness
  and cannot avoid a GPU switch for low-rate demand;
- CUDA cannot catch up to ongoing phone decode;
- batched replay dominates the entire model-switch interval;
- the phone tier cannot rewarm before realistic reverse demand;
- workload oscillation causes more wasted preparation than useful service;
- output ownership cannot be made exact.

## Prior work boundary

The comparison must include:

- Prima.cpp for heterogeneous single-model local-storage paging and layer-ring
  execution; phone sharding alone is not a contribution;
- ServerlessLLM and HydraServe for checkpoint loading and cold-start overlap;
  token-history replay while a source continues is not a contribution;
- Llumnix for live request migration;
- EdgeShard for static heterogeneous layer placement;
- DroidSpeak and edge handover work for KV reuse, transfer, and recomputation.

The claimed distinction is not heterogeneous inference or live migration
alone. It is an asymmetric executable standby that substitutes for a second
server GPU during hot-set transitions: collectively sharded low-power devices
remain authoritative while a memory-limited GPU changes model residency, then
transfer live requests and cyclically prepare the displaced model.
