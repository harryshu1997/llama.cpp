# Q-PIM Funnel: heterogeneous continuous activation batching

The focused system uses OP12 and OP15 as resident Gemma prefix accelerators for
one selected A6000. A small CUDA bridge normalizes their different prefix cuts,
then one shared CUDA tail continuously batches ready activation rows while a
high-priority BGE service remains on the A6000. The scheduler releases at a
measured useful batch or the earliest SLO-safe time.

Design A, `server -> OP15 -> OP12 -> server`, remains a proven route named `A0`.
It supplies useful pipeline, sharding, local-KV, transport, and shared-weight
mechanisms. It is not the new system novelty.

## Primary question

Can heterogeneous phone prefixes feed one continuously batched GPU suffix while
preserving mixed-priority SLOs and reducing one-GPU HBM or selected-GPU board
J/completed equal-work request?

Skipped GPU-us, utilization, and modeled idle time are not energy evidence. The
current experiment reports only matched selected-GPU board energy. A
total-system claim still requires synchronized phone, USB, host, and GPU power.

## Sources of truth

| File | Authority |
|---|---|
| [talks.md](talks.md) | Live status and newest-first experiment log |
| [MIXED_WORKLOAD_DESIGN.md](MIXED_WORKLOAD_DESIGN.md) | Authoritative Q-PIM architecture and claim boundary |
| [TWO_LEVEL_SCHEDULER.md](TWO_LEVEL_SCHEDULER.md) | Deferred broad scheduler design; not paper-critical |
| [WORKLOAD_TRACES.md](WORKLOAD_TRACES.md) | Public dataset catalog and trace orientation; S8 schemas/spec freeze bytes |
| [NEXT_PLAN.md](NEXT_PLAN.md) | Current focused executable gate order |
| [spikes/s19_dynamic_batch_runtime/PLAN.md](spikes/s19_dynamic_batch_runtime/PLAN.md) | Active continuous-batch and shared-tail implementation plan |
| [spikes/s10_power_frontier_repair/PLAN.md](spikes/s10_power_frontier_repair/PLAN.md) | Historical S10 foundation repair and oracle substrate |
| [spikes/s10_power_frontier_repair/V0_AUDIT.md](spikes/s10_power_frontier_repair/V0_AUDIT.md) | Why the historical S10-V0 verdict is invalid/inconclusive |
| [spikes/s9_phone_pim_runtime/DYNAMIC_RESULTS.md](spikes/s9_phone_pim_runtime/DYNAMIC_RESULTS.md) | Bounded sequential provisioning/runtime evidence |
| [spikes/s9_pipelined_transport/RESULTS_R.md](spikes/s9_pipelined_transport/RESULTS_R.md) | Repaired bounded pipelined-transport evidence |
| [spikes/s13_runtime_fleet/RESULTS.md](spikes/s13_runtime_fleet/RESULTS.md) | Live two-phone FFN fleet runtime evidence |
| [spikes/s10_matched_energy_e2_aggregate/RESULTS.md](spikes/s10_matched_energy_e2_aggregate/RESULTS.md) | Current evidence-chain verdict and physical blockers |
| [DESIGN.md](DESIGN.md) | Historical Design A technical substrate |
| [MILESTONES.md](MILESTONES.md) | Historical Design A M0-M5 record |
| [PORT.md](PORT.md) | Port provenance and reusable historical components |

## What, how, and when

**What:** low-priority Gemma prefix work for OP12 `[0,6)` or OP15 `[0,8)`.
High-priority BGE stays on the selected A6000.

**How:** normalize OP12 results through CUDA `[6,8)`, combine all layer-8 rows in
one continuously batched CUDA `[8,48)` tail, and maintain per-request KV in each
stage.

**When:** release a phone or tail batch when it reaches a measured useful batch
candidate or the earliest admitted request reaches its latest safe start.

Weights are fixed and resident before the run. No request waits for a download,
general DAG solve, or phone-to-phone middle stage.

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
- S11-E0 proved exact resident OP15 `[0,2)` execution and 888 MiB selected-A6000
  relief, but rejected the fixed serial route for energy: lower A6000 power did
  not offset 2.31x runtime, so equal-work board energy increased 53.15 percent.
  The next mechanism must overlap phone work with useful server work.
- The protocol-v3 phone-PIM prototype can durably provision, resume, publish,
  prepare, and execute one Gemma4 dense-FFN island on both phones. S9-V1A-R
  proves bounded windowed provisioning on both phones, but still provides no
  multi-model capacity or energy claim.
- Fleet energy remains unmeasured because the physical phone power boundary is
  invalid in the current setup.

The next bounded build is S19: versioned per-sequence StageNet commands,
LayerSplit-local continuous admission, one-phone physical validation, then the
OP12 `[6,8)` bridge plus one shared `[8,48)` tail. No simulator, weight-streaming
planner, general solver, or phone-to-phone route can substitute for those
physical gates.

## Existing implementation relevant to the new target

- Current tree: server queues/slots, LayerSplit TCP stages, GGUF sharding,
  download/cache helpers, per-tensor weight sharing, and dual HTP/GPU workers.
- `examples/phone-pim`: bounded protocol-v3 sequential provisioning, verified
  restart resume, content-addressed publication, one resident FFN command path,
  and a compiled multi-phone work-stealing harness. It is a live prototype
  substrate, not a mixed-model server scheduler.
- Historical `route2-b9531`: process-local VQ byte table, CONWIP admission,
  backend singleton/fairness, co-execution harnesses, and an experimental HEFT
  virtual queue.
- Historical Unifer design: a 16-byte remote telemetry roster and trace replay.

The historical VQ is local occupancy instrumentation, not a distributed status
or lease system. It must be audited and minimally extracted before any port.

## Honest current status

S18 runs one selected A6000, OP15, and OP12 with exact Gemma tokens and preserved
BGE p95, but its independent fixed-B32 routes duplicate CUDA tail weights and
save only 0.342 percent median selected-GPU board energy. It is a mechanics
pass and a relief failure. S19 has audited llama-server continuous batching and
frozen the focused implementation, but arbitrary-sequence admission,
heterogeneous-cut normalization, and the one-copy shared tail are not yet
implemented. No total-system energy benefit is claimed.

## Start here

1. Read the top of [talks.md](talks.md).
2. Read [MIXED_WORKLOAD_DESIGN.md](MIXED_WORKLOAD_DESIGN.md).
3. Read [spikes/s19_dynamic_batch_runtime/PLAN.md](spikes/s19_dynamic_batch_runtime/PLAN.md).
4. Read [NEXT_PLAN.md](NEXT_PLAN.md) for historical evidence and gate order.
5. Read the S18 result before interpreting the current energy numbers.

Do not copy llama-server's HTTP/task stack. Borrow its slot, logical-batch,
continuous-admission, and per-sequence KV lifecycle inside the experimental
LayerSplit executor. Do not start an energy acquisition before the S19 physical
continuous-batch and one-copy-tail gates pass.
