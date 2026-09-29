# S42 matmul virtual-queue scheduler V1

Date: 2026-08-07 EDT.

## Status and boundary

`matmul_vq_scheduler.py` is an op-level shadow planner for one desktop CPU,
one desktop GPU, and one phone accelerator. It does not dispatch kernels or
change llama-server placement. Its output is marked
`PLANNED_NOT_RUNTIME_CERTIFIED`; a route still needs the existing S42 runtime
certificate and physical validation before enforcement.

The planner accepts a finite queue of ordered model programs. It considers
only dense matmuls. A shard-safe non-matmul op inherits the preceding matmul
placement at zero modeled cost. A non-shard-safe op is an explicit barrier and
gathers its activation to CPU.

The fixed policy assumptions requested for this stage are:

- all weights already exist in device-local storage;
- weight acquisition has zero latency and zero energy;
- materialized weight slices still consume accelerator memory;
- the GPU memory capacity defaults to 16 GiB;
- the phone memory capacity defaults to 10 GiB;
- the desktop CPU package stays at the configured active power, 115 W in the
  physical-profile materializer, for the complete selected op route;
- CPU, GPU, and phone branches may overlap;
- every program's final activation is gathered to CPU.

## Per-matmul input

Each matmul supplies the facts that do not depend on a backend:

```text
M, K, N
input_bytes
output_bytes
weight_bytes
compute_ops                  normally 2*M*K*N
kernel_family
quantization
weight_id
N split alignment
allowed devices
```

Backend rows supply effective compute throughput, effective memory
throughput, launch time, active powers, legal N range, split alignment,
memory capacity, and evidence status. Directed link rows supply fixed time,
effective bandwidth, fixed energy, byte energy, active powers, and the
measured payload range.

## Exact split contract

V1 searches output-column, or N-axis, splits:

```text
N_cpu + N_gpu + N_phone = N
```

For branch `d`:

```text
ops_d        = ceil(compute_ops * N_d / N)
weight_d     = ceil(weight_bytes * N_d / N)
output_d     = proportional output byte slice
kernel_mem_d = input_bytes + weight_d + output_d

compute_us_d = ceil(ops_d * 1e6 / effective_ops_per_s_d)
memory_us_d  = ceil(kernel_mem_d * 1e6 / effective_bytes_per_s_d)
kernel_us_d  = launch_us_d + max(compute_us_d, memory_us_d)
```

N-axis splitting is exact for a dense `M x K` by `K x N` matmul. Every branch
reads the complete input and produces disjoint output columns. It needs no
cross-device numerical reduction. K-axis reduction splits and M-axis batch
splits are intentionally outside V1.

The search includes CPU-only, full GPU, full phone, CPU plus GPU, CPU plus
phone, GPU plus phone, and three-engine cuts when the shape, memory, and
profile gates permit them. Large ranges are sampled at deterministic aligned
points. Exact complements are added so a sampled GPU plus phone full-offload
cut is not accidentally omitted.

## Activation residency and transfer cost

The output stays sharded after a matmul. Shard-safe small ops retain the same
shards. At the next dense matmul, each selected branch needs the complete
input. The scheduler therefore copies only missing shards:

1. A remote source shard is gathered to CPU only if CPU or another remote
   target needs it.
2. A GPU or phone target keeps its local shard.
3. CPU uploads only the target's missing bytes after the required gathers.
4. CPU, GPU, and phone compute begin when their own input is complete.
5. The new output remains sharded until the next barrier or final CPU gather.

For directed link `l`:

```text
link_us = fixed_link_us + ceil(bytes * 1e6 / bandwidth_bytes_per_s)

scalar_link_energy_uj = fixed_energy_uj
                      + ceil(bytes * dynamic_pj_per_byte / 1e6)
```

PCIe H2D/D2H and USB H2P/P2H are separate directed models. Directions may
share one interval-calendar resource, which conservatively serializes them.

## Virtual queue and overlap

Programs enter a finite host queue. The planned state sequence is:

```text
ANNOUNCED -> READY -> ASSIGNED -> RUNNING -> COMPLETE
```

`run()` remains the finite offline convenience path. The runtime-facing path
is `schedule_next(now_us)`, which assigns exactly one READY queue entry. It
advances the causally earliest entry, with deadline and enqueue order as
deterministic tie breakers. If no dependency-ready entry exists at `now_us`,
it returns `None` without changing the plan. `enqueue_and_schedule()` admits a
new program and applies the same queue arbitration.

Physical device queues are not filled. Instead, the existing S42
`ResourceTimeline` transactionally previews and commits intervals for:

```text
compute:cpu
compute:gpu
compute:phone
link:pcie
link:usb
```

All phases in a candidate have offsets from one route start. Transfers and
independent compute branches overlap when they use different resources. The
candidate wall time is the maximum branch completion, including the needed
input transfers.

Before each assignment, the scheduler reads the current interval calendars,
device readiness, link readiness, and memory ledger. A resource forecast has
two distinct timestamps:

```text
earliest_available_us = first one-microsecond calendar gap
predicted_free_us     = end of all currently committed device work
```

These are diagnostics. Candidate simulation still tests the actual phase
duration against every interval, so it can use a sufficiently large internal
gap or wait for the blocking reservation. Each matmul decision records the
pre-assignment device free-time forecast, exact route start, queue delay,
per-resource delay, and blocking resources.

The live update operations are:

```python
scheduler.reserve_external_resource(
    "compute:gpu", "hot-model", now_us, predicted_finish_us
)
scheduler.update_external_memory(
    "gpu", "hot-model-weights", current_vram_bytes, now_us
)
scheduler.set_device_ready("phone", phone_is_ready, now_us)
scheduler.set_resource_ready("link:usb", usb_is_ready, now_us)

scheduler.enqueue(new_program)
decision = scheduler.schedule_next(now_us)
```

`release_resource_lease(token, actual_end_us)` shortens a committed interval
when work completes early. The next task immediately sees that capacity.
Setting an unavailable resource back to ready enables it for the next
unassigned op. Already assigned work is never silently migrated. Revoking a
resource returns the affected owner IDs and changes result status to
`REPLAN_REQUIRED`; the outer runtime must cancel or recover those assignments.

## Energy objective

The planner does not add 115 W separately for two ops whose host-active
intervals overlap. It maintains one power envelope per physical domain:

```text
P_domain(t) = max(P_idle, all active intervals in that domain at t)

E_domain = integral(P_domain(t) dt)

E_fleet = sum(E_domain) + sum(unattributed scalar link energy)
```

For each ready matmul, the CPU-only route defines the current queue-aware
latency baseline. A candidate is feasible only when:

```text
candidate_finish <= program_deadline

candidate_elapsed <= CPU_baseline_elapsed * latency_limit_ppm / 1e6
```

The selected route minimizes incremental fleet energy, followed by finish
time and a deterministic route ID. With the default latency limit of 1.0x,
energy savings cannot be purchased by making the operator slower than its
current CPU-only baseline.

If no later matmul can consume the current shards before a barrier or program
completion, candidate comparison also previews the mandatory CPU gather. Its
time, link energy, host-active interval, and resource queue delay are included
in both the energy objective and latency gate before placement is chosen. The
planner therefore cannot select a remote last matmul by hiding its return
cost outside the decision.

This accounting captures the requested phone-overlap rule. If CPU work and a
phone branch overlap, the host interval is charged once at 115 W and the
phone's measured or configured active interval is added in its own domain. A
full phone route still charges the host active interval while it waits or
executes other virtual-queue work.

## Memory rule

GPU and phone admission checks include:

```text
reserved/background allocations
+ persistent selected weight slices
+ current program activation shards
+ missing full input needed during this op
+ output shard allocated before the input is released
```

The peak and post-op totals must both fit. A weight allocation is keyed by
`model_id`, `weight_id`, and device. It grows when a later decision selects a
larger slice and does not shrink automatically. This implements the earlier
"prepare the largest split, execute any smaller split" rule. Reusing the same
model and weight ID does not allocate a duplicate copy.

Background VRAM or phone use can be registered with
`reserve_external_memory()` before scheduling. At runtime,
`update_external_memory()` adds, changes, or releases the same allocation; a
zero-byte update releases it. This is how a hot GPU model, display allocation,
phone worker heap, or other consumer changes the capacity seen by the next
unassigned op.

## Build the current 4060 Ti plus OP15 shadow profile

The checked-in kernel campaign can be converted without editing scheduler
code:

```sh
python3 \
  research_dev/spikes/s42_general_energy_scheduler_v1/materialize_matmul_vq_profile.py \
  --input research_dev/scheduler/profiles/MEASURED_4060TI_OP15_KERNEL_PROFILE_V1.json \
  --output /tmp/4060ti-op15-matmul-vq.json \
  --generic-family
```

`--generic-family` makes K and the operator-family selector generic so a new
model can be explored immediately. Those rows remain `estimated`: they borrow
effective rates from the measured Gemma fused-FFN campaign and cannot be used
as an enforcement claim. Omitting the flag preserves the exact family and K
bindings from the source campaign.

To account for a hot model already using 12 GiB of VRAM:

```sh
python3 \
  research_dev/spikes/s42_general_energy_scheduler_v1/materialize_matmul_vq_profile.py \
  --input research_dev/scheduler/profiles/MEASURED_4060TI_OP15_KERNEL_PROFILE_V1.json \
  --output /tmp/4060ti-op15-hot-vq.json \
  --generic-family \
  --gpu-reserved-bytes 12884901888
```

The generated profile uses 17,179,869,184 bytes for GPU capacity and
10,737,418,240 bytes for phone capacity.

## Run a model program

```sh
python3 \
  -m research_dev.scheduler matmul \
  --profile /tmp/4060ti-op15-matmul-vq.json \
  --workload research_dev/spikes/s42_general_energy_scheduler_v1/MATMUL_VQ_EXAMPLE_WORKLOAD_V1.json \
  --output /tmp/matmul-vq-result.json
```

The result reports every op's dimensions, byte counts, compute count, exact
column cut, residency before and after, kernel estimates, transfers, resource
leases, queue delay, CPU baseline, energy increment, memory growth, and
evidence status. It also reports fleet makespan, energy by domain, peak/current
memory, and all persistent weight slices.

## Current limitations

- This is a planner, not a llama-server executor.
- Programs are ordered dependency chains. A future adapter must represent
  true Q/K/V, gate/up, expert, and multi-request DAG readiness rather than
  serializing independent matmuls.
- Placement is causal and energy-greedy at each ready matmul. It accounts for
  the mandatory next CPU gather, but it does not claim a globally optimal
  multi-program schedule across all future matmul residency choices.
- Non-matmul latency and energy are zero in V1. Only operators explicitly
  marked shard-safe may inherit a split; all others must be barriers.
- N-axis splitting is implemented. K reduction, M batching, attention head
  constraints, MoE routing, KV ownership, and vocabulary top-k compression
  need separate exact contracts.
- Generic materialized rows are estimates. New model families and shape
  buckets need physical CPU, CUDA, and phone profiles plus held-out full-model
  validation.
- Weight acquisition is free only because this stage explicitly assumes it.
  Runtime mapping, repack, page fault, eviction, and reload costs must be
  restored when that assumption is removed.
- The memory ledger is conservative and reserves planned persistent state. A
  time-indexed allocator can recover capacity when many programs overlap.
- Early lease release updates the resource calendar. V1 retains the planned
  power intervals, so post-completion energy is conservative until runtime
  power receipts replace the shadow estimate.
- Resource revocation reports already assigned owners but does not reconstruct
  their dependency state or migrate them. That recovery remains an outer
  runtime transaction.
- Runtime health, thermal, USB epoch, cancellation, sampler, grammar, KV, and
  quality gates remain in the outer S42 scheduler and route certificate.

## Tests

`test_matmul.py` covers energy-first GPU placement, CPU plus phone
branch overlap, deferred USB transfer across a shard-safe chain, explicit CPU
barriers, phone and hot-GPU memory limits, persistent weight reuse, resource
queueing, one-entry runtime dispatch, device free-time forecasts, external busy
windows, early release, runtime readiness and memory updates, measured-only
shape gating, finite queue admission, physical-profile materialization,
mandatory final-return pricing, final CPU ownership, and non-duplicated host
power envelopes.

Run the complete S42 suite:

```sh
python3 research_dev/scheduler/tests/run_all.py
```
