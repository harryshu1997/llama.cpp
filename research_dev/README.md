# Q-PIM: power-frontier scheduling over active mobile memory

The server runs concurrent multi-model inference DAGs. Pre-provisioned phones
expose READY resident operator islands on HTP and GPU. Q-PIM topologically
reorders independent DAG islands to unlock phone-resident work early, then
selects coordinated power-trigger bundles that reshape the remaining A6000 work
into dense batches and measured lower-power intervals.

Design A, `server -> OP15 -> OP12 -> server`, remains a proven route named `A0`.
It supplies useful pipeline, sharding, local-KV, transport, and shared-weight
mechanisms. It is not the new system novelty.

## Primary question

Can dependency-aware frontier shaping over active phone-resident weights reduce
complete-wall joules at equal work/SLO, or increase SLO-valid work at equal wall
power, beyond an optimized server-only DAG-ordering, batching, and DVFS control?

Skipped GPU-us, utilization, and modeled idle time are not energy evidence. A
phone bundle earns energy credit only when a measured batch or A6000 power-state
change reduces synchronized total-wall energy.

## Sources of truth

| File | Authority |
|---|---|
| [talks.md](talks.md) | Live status and newest-first experiment log |
| [MIXED_WORKLOAD_DESIGN.md](MIXED_WORKLOAD_DESIGN.md) | Authoritative Q-PIM architecture and claim boundary |
| [TWO_LEVEL_SCHEDULER.md](TWO_LEVEL_SCHEDULER.md) | Slow residency planner and fast power-frontier scheduler |
| [WORKLOAD_TRACES.md](WORKLOAD_TRACES.md) | Public dataset catalog and trace orientation; S8 schemas/spec freeze bytes |
| [NEXT_PLAN.md](NEXT_PLAN.md) | Current PF0-PF5 research and build gate order |
| [spikes/s10_power_frontier_repair/PLAN.md](spikes/s10_power_frontier_repair/PLAN.md) | Only authorized S10 foundation repair |
| [spikes/s10_power_frontier_repair/V0_AUDIT.md](spikes/s10_power_frontier_repair/V0_AUDIT.md) | Why the historical S10-V0 verdict is invalid/inconclusive |
| [spikes/s9_phone_pim_runtime/DYNAMIC_RESULTS.md](spikes/s9_phone_pim_runtime/DYNAMIC_RESULTS.md) | Bounded sequential provisioning/runtime evidence |
| [spikes/s9_pipelined_transport/RESULTS_R.md](spikes/s9_pipelined_transport/RESULTS_R.md) | Repaired bounded pipelined-transport evidence |
| [spikes/s10_matched_energy_e2_aggregate/RESULTS.md](spikes/s10_matched_energy_e2_aggregate/RESULTS.md) | Current evidence-chain verdict and physical blockers |
| [DESIGN.md](DESIGN.md) | Historical Design A technical substrate |
| [MILESTONES.md](MILESTONES.md) | Historical Design A M0-M5 record |
| [PORT.md](PORT.md) | Port provenance and reusable historical components |

## What, how, and when

**What:** a certified resident operator island in an admitted request DAG. The
virtual queue separates ANNOUNCED future weight demand from READY executable
work.

**How:** prioritize topologically legal unlockers, execute complete READY islands
on certified phone routes, and cluster the remaining A6000 islands into native
batches or contiguous active bursts. Per-GEMM network splitting is excluded.

**When:** select a sleep, batch-shaping, power-cap, or memory bundle only when its
counterfactual complete-wall benefit is positive under the SLO. The server uses
latest SLO-safe claims rather than eager execution when doing so creates a
measured useful batch or power interval.

The slow loop plans residency, prepared images, ownership, leases, and route/
power envelopes. The fast loop performs bounded H-hop DAG frontier shaping,
phone-bundle selection, A6000 batching/power decisions, and atomic dispatch from
READY capabilities. Neither loop fetches weights on the request critical path.

Maximum phone use means maximum useful parallelism, not forced utilization.

## Evidence carried forward

- The persistent three-device 12B route `A0` runs end to end with stage-local KV.
- Phone GGUF shards preserve absolute layer indices and store only local slices.
- Read-only weights can be shared by Hexagon and OpenCL per tensor.
- Static HTP batched decode works at tested shapes.
- S3 output-row splitting failed; do not split one GEMV over phone backends.
- S4 GPU attention failed at realistic context; HTP remains the tested decode
  attention engine.
- S5 showed isolated phone operators do not add useful raw A6000 throughput over
  adb at the measured point.
- S6 found saturated HTP-decode/GPU-prefill overlap, but request-pair latency and
  FFN boundary evidence are not scheduler authorization.
- S7 ragged HMX attention passed an isolated operator gate and still needs a real
  layer and trace.
- The protocol-v3 phone-PIM prototype can durably provision, resume, publish,
  prepare, and execute one Gemma4 dense-FFN island on both phones. S9-V1A-R
  proves bounded windowed provisioning on both phones, but still provides no
  multi-model capacity or energy claim.
- Fleet energy remains unmeasured because the physical phone power boundary is
  invalid in the current setup.

## Existing implementation relevant to the new target

- Current tree: server queues/slots, LayerSplit TCP stages, GGUF sharding,
  download/cache helpers, per-tensor weight sharing, and dual HTP/GPU workers.
- `examples/phone-pim`: bounded protocol-v3 sequential provisioning, verified
  restart resume, content-addressed publication, and one resident FFN command
  path. It is a prototype substrate, not a server scheduler.
- Historical `route2-b9531`: process-local VQ byte table, CONWIP admission,
  backend singleton/fairness, co-execution harnesses, and an experimental HEFT
  virtual queue.
- Historical Unifer design: a 16-byte remote telemetry roster and trace replay.

The historical VQ is local occupancy instrumentation, not a distributed status
or lease system. It must be audited and minimally extracted before any port.

## Honest current status

The Q-PIM direction and power-frontier scheduler are defined. S9-V1A-R passes
its bounded full-shard windowing gate on both phones and is frozen as transport
substrate; it remains capacity- and energy-unproven. Historical S10-V0 is
invalid/inconclusive. The repaired temporal foundation, typed evidence binding,
matched-timeline comparison, and E2A all-pairs evidence chain pass their bounded
mechanics tests. No physical measurement was run. E2A still blocks a physical
claim because this host has no enumerable independent plan commitment,
registered verifier, witnessed launcher, or server-wall power instrument. S8
Gate A has not run, no two-service power/route atlas exists, and no scheduler
runtime exists. No capacity or energy benefit is claimed.

## Start here

1. Read the top of [talks.md](talks.md).
2. Read [MIXED_WORKLOAD_DESIGN.md](MIXED_WORKLOAD_DESIGN.md).
3. Read [TWO_LEVEL_SCHEDULER.md](TWO_LEVEL_SCHEDULER.md).
4. Read [NEXT_PLAN.md](NEXT_PLAN.md).
5. Read the current E2A verdict before authorizing another experiment.

Do not implement C0-C5, a production scheduler, model/KV changes, or a historical
VQ port until the external-anchor, measurement, opportunity, causal-policy, and
controlled physical gates pass.
