# Active Warm-Tier Multi-Model Serving

The primary system uses OP12 and OP15 as a collectively sharded executable warm
tier for one memory-limited desktop GPU. The GPU holds one hot model. The phone
fleet holds one other ready model, serves it while the GPU drains and changes
models, and then transfers live requests into one CUDA continuous batch through
batched token-history replay. After cutover, the phones prepare the displaced
GPU model for the next switch.

```text
initial:  GPU = model A HOT        phones = model B WARM
bridge:   GPU loads B              phones serve B requests
handoff:  GPU batch-prefills B     phones continue B until catch-up
cutover:  GPU = model B HOT        phones release B
rearm:    GPU = model B HOT        phones prepare model A WARM
```

This is an executable model cache, not remote coherent memory and not a fixed
layer pipeline.

## Primary question

Can a low-power, collectively sharded phone tier eliminate the service blackout
of multi-model GPU residency changes while preserving continuous batching and
request SLOs?

The first claim is latency and SLO goodput. Selected-GPU board energy is
secondary. Phone, host, network, and total-system energy remain unknown until a
valid physical measurement boundary exists.

## Current authority

| File | Purpose |
|---|---|
| [ACTIVE_WARM_TIER_DESIGN.md](ACTIVE_WARM_TIER_DESIGN.md) | Current system contract, invariants, scheduler, controls, and milestones |
| [spikes/s39_phone_model_switch_trace/PLAN.md](spikes/s39_phone_model_switch_trace/PLAN.md) | First bounded physical proof of concept |
| [talks.md](talks.md) | Live status and newest-first experiment log |
| [NEXT_PLAN.md](NEXT_PLAN.md) | Active gate order followed by preserved historical plans |
| [WORKLOAD_TRACES.md](WORKLOAD_TRACES.md) | Trace provenance and workload sources |
| [MIXED_WORKLOAD_DESIGN.md](MIXED_WORKLOAD_DESIGN.md) | Historical Q-PIM focused design and reusable scheduler substrate |
| [DESIGN.md](DESIGN.md) | Historical Design A layer-pipeline substrate |
| [MILESTONES.md](MILESTONES.md) | Historical Design A milestones |

## What is new

The system combines three mechanisms:

1. **Executable warm residency.** A GPU-nonresident model remains usable on
   phone-resident shards instead of waiting in storage.
2. **Non-blocking batched catch-up.** Phones remain the token owner while CUDA
   loads and reconstructs native KV for multiple requests from prompt and
   committed token IDs. CUDA consumes the small token delta and takes ownership
   at an exact token boundary.
3. **Symmetric rewarming.** Once model B becomes hot on the GPU, phones prepare
   displaced model A. A later demand reversal runs the same transition in the
   other direction.

Model loading, continuous batching, SLO routing, token replay, and layer
sharding are substrate, not standalone novelty.

## What moves

All checkpoints are provisioned before a measured run:

```text
desktop NVMe or host memory: all model checkpoints
phone UFS:                  assigned shards for eligible models
desktop VRAM:               one hot model
phone RAM and NPU buffers:  one collectively executable warm model
```

A normal promotion does not transfer a complete 9-12 GB checkpoint from phones
to the desktop. The desktop loads its local copy. The coordinator already owns
the prompt and emitted token history, so catch-up transfers only bounded token
deltas and control metadata.

The physical transport is split by payload. USB ADB provisions large weight
shards to phone UFS before they become ready. WiFi TCP carries runtime commands,
hidden-state activations, and token results. The current phone-stage path is
relayed by the host coordinator; it is not direct phone-to-phone transfer.

Direct KV transfer is optional future work. The primary path reconstructs
native CUDA KV by batch-prefilling token histories, which avoids cross-backend
KV-layout dependence.

## Runtime invariants

- At most one model is hot on the GPU; zero is legal during replacement.
- At most one other model is executable on the phone fleet initially.
- Storage residency never implies execution readiness.
- Model, tokenizer, template, quantization, KV type, context, and route digests
  must match their certificates.
- Only one owner may commit a request token for an ownership epoch.
- Phones remain authoritative during CUDA catch-up.
- CUDA becomes authoritative only after a durable token-boundary cutover.
- A failed promotion leaves the phone route authoritative.
- A failed rewarm publishes not-ready and cannot retain stale readiness.
- No request waits indefinitely for a phone, load, replay, or handoff.

## Evidence carried forward

The earlier program provides useful mechanisms:

- persistent phone workers retain prepared weights across sessions;
- OP12 and OP15 execute real stage-local continuous batches;
- mixed prefill and decode rows have run in one physical phone batch;
- arbitrary resident layer intervals and per-request KV lifecycle work;
- request identity, position, epoch, placement, and reset checks exist;
- the three-device routes prove activation transport and exact request
  conservation;
- S39 already freezes a real BurstGPT-derived 20-minute model-switch trace.

The earlier program also provides stop conditions:

- output-row splitting one GEMV across phone backends lost;
- moving attention to Adreno lost at realistic context;
- fixed phone prefixes generally increased end-to-end latency;
- Q4/Q8 HTP routes have not passed the prior same-artifact numerical-quality
  gate;
- phone storage capacity is not proof that a complete second model can execute;
- selected-GPU energy reductions do not establish total-system savings.

## Honest current status

S39 has a deterministic BurstGPT-derived trace and provisional Gemma/Qwen model
mapping. Its active frequent-switch profile now reduces byte-identically to
five promotion windows and nine target changes. Overlapping shard supersets
are hash-verified on both phones. Dense Qwen3 partial-stage execution is
implemented, and one real B1 collective-phone route matches eight CUDA tokens,
but it remains `PROVISIONAL_B1`. The corresponding Gemma Q4 route has clean
placement but fails token correctness. The active system has not passed W0:

- neither decoder has a complete, repeated-process collective-phone execution
  certificate;
- CUDA and phone transition times have not been measured as one comparable
  atlas;
- non-blocking phone-to-CUDA token catch-up is not implemented;
- symmetric rewarming and reverse handoff are not implemented;
- no latency, SLO, or energy benefit is claimed.

The hash-bound warm-tier controller is implemented and intentionally refuses
all nine physical transitions with `E_ROUTE_NOT_READY`. The first task remains
the W0 two-model eligibility and timing gate, not a general scheduler.

## Start here

1. Read the top of [talks.md](talks.md).
2. Read [ACTIVE_WARM_TIER_DESIGN.md](ACTIVE_WARM_TIER_DESIGN.md).
3. Execute [the S39 plan](spikes/s39_phone_model_switch_trace/PLAN.md) in order.
4. Read S33-S38 results before reusing a quantized or mixed-generation route.

Do not start direct KV assembly, energy acquisition, a general solver, or a
multi-model cache replacement policy before one-request and batched token
catch-up pass on real devices.
