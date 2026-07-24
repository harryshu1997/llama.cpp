# Active Warm-Tier Multi-Model Serving

Status: current primary research direction, frozen 2026-07-23.

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

- one RTX 4060 Ti or one selected A6000;
- OP15 and OP12 as one collective phone tier;
- two decoder models that cannot reside together in GPU memory;
- one executable model resident across the phones at a time;
- identical model digest, tokenizer, chat template, quantization, KV types, and
  context parameters on every route for that model;
- greedy decoding first, then deterministic stochastic sampling;
- direct phone-to-phone activation transfer with host-issued reservations;
- no total-system energy claim until phone and host energy are measurable.

The existing Gemma and Qwen assignment in S39 is provisional. A model enters a
physical switch experiment only after both its CUDA route and complete
collective-phone route pass identity, placement, correctness, memory, and
latency gates. Storage residency alone is not eligibility.

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

1. Existing A requests continue on the GPU. The first prototype drains them
   rather than migrating them.
2. New B requests are admitted to the phone fleet and continuously batched.
3. The slow loop observes sustained B demand and begins B promotion.
4. The GPU stops admitting work that would make A impossible to drain within
   the promotion bound.
5. The desktop loads B from local NVMe or host memory. Phones continue serving
   B throughout this interval.
6. When B is ready on CUDA, the coordinator snapshots each live B request's
   committed token frontier.
7. CUDA batch-prefills the prompts and committed token histories to construct
   native CUDA KV. This is a logical batched reconstruction phase and may be
   split into bounded ubatches.
8. Phones remain authoritative and may generate a small token delta while CUDA
   prefill runs. CUDA consumes that delta without publishing output.
9. Once CUDA reaches the same frontier, the coordinator commits a new ownership
   epoch. Phones stop B at that exact token boundary and CUDA joins the
   requests to its continuous decode batch.
10. After every B request and buffer is released from the phone tier, the
    phones prepare model A from local UFS and publish a new ready certificate.

If A demand returns before A is ready on the phones, the request follows an
explicit bounded fallback. The system must not report A as warm.

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

The mechanism is useful only when all of the following hold:

```text
phone_warm_TTFT < GPU_model_ready_time
phone_capacity >= arrivals_during_promotion
CUDA_decode_rate > phone_decode_rate
catchup_time <= remaining_request_slack
phone_rewarm_time < expected_time_to_next_reverse_switch
```

If the first inequality fails, phones may still be useful for low-rate models
by avoiding promotion entirely. If both bridge service and avoided switches
fail, stop the direction.

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

All controls execute the same request identities and token budgets.

## Metrics and claim boundary

Primary:

- switch-period SLO goodput;
- P50/P95/P99 TTFT;
- maximum inter-token gap across handoff;
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

## Milestones

### W0 - Eligible two-model atlas

- Freeze exact model, tokenizer, template, quantization, and context digests.
- Prove one complete collective-phone route and one CUDA route per model.
- Measure memory, placement, correctness, warm TTFT, decode rate, and useful
  batch candidates.
- Measure CUDA warm-load, cold-load, unload, and context construction.

Exit: two executable models exist, or the experiment narrows to a smaller
second model. No scheduler may treat storage-only residency as executable.

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
- Run C0-C3 with identical work.
- Sweep promotion threshold and hysteresis from a predeclared finite set.
- Report queueing, handoff, rewarm, batching, and switch avoidance.

Exit: C2 or C3 improves switch-period SLO goodput or P95 TTFT by at least 20
percent without losing requests or regressing steady-state hot-model SLO.

### W5 - Benefit and robustness

- Repeat across at least three arrival regimes and model-switch frequencies.
- Inject phone disconnect, failed load, stale epoch, and delayed rewarm.
- Measure selected-GPU energy only after the latency mechanism passes.

Exit: report the regime where the warm tier is beneficial and the regime where
it is not. Do not generalize beyond measured devices and models.

## Stop rules

Stop or narrow the design when:

- no second model passes a complete collective-phone execution gate;
- phone warm TTFT is not earlier than measured GPU readiness;
- CUDA cannot catch up to ongoing phone decode;
- batched replay dominates the entire model-switch interval;
- the phone tier cannot rewarm before realistic reverse demand;
- workload oscillation causes more wasted preparation than useful service;
- output ownership cannot be made exact.

## Prior work boundary

The comparison must include:

- PRIMA for fixed local-storage paging and layer-ring execution;
- ServerlessLLM and HydraServe for checkpoint loading and cold-start overlap;
- Llumnix for live request migration;
- EdgeShard for static heterogeneous layer placement;
- DroidSpeak and edge handover work for KV reuse, transfer, and recomputation.

The claimed distinction is not live migration alone. It is an asymmetric,
collectively sharded, low-power executable warm tier with continuous service
during model promotion, batch-preserving token catch-up, and cyclic preparation
of the displaced model.
