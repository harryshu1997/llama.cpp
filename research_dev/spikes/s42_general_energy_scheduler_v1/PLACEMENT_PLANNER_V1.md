# S42 hierarchical energy placement planner V1

Date: 2026-08-06 EDT.

## Status

The scheduler is not production-complete. The placement optimization core,
the first real continuous-ubatch graph adapter, and an epoch-bound route
compiler are implemented. The compiler certifies the directly measured I3
CPU control and CPU plus OP15 cohort routes. It keeps newly composed
per-operator routes fail closed until their host complements, transfers,
merges, and held-out full-model behavior are measured.

The hierarchy is:

1. `HierarchicalPlacementPlanner.plan_task` compares whole-request routes and
   includes model load time and energy when the route is not resident.
2. `plan_sequence` performs layer placement by retaining activation location
   across an ordered operator graph. A device boundary is charged only when
   the activation crosses it.
3. Each operator lists certified local, full-offload, or fork-join split
   candidates. The solver selects the lowest-energy feasible candidate and
   reports the exact cut.
4. `UnifiedScheduler` remains the outer online admission and resource gate.
   A compiled placement is not executable until the profiler qualifies it and
   publishes it as a route for the current device, worker, transport, model,
   and thermal epoch.

This separation is intentional. The placement compiler may search estimated
profiles, but only the existing measured-route gate can authorize production
energy enforcement.

## Objective

For a task route `r`, operator candidate `c`, device energy domain `d`, and
directed transfer link `l`:

```text
T_kernel = launches * launch_us
         + max(compute_ops / effective_ops_per_us,
               memory_bytes / effective_bytes_per_us)

E_dynamic_kernel = (P_active - P_idle) * T_kernel

T_link = fixed_link_us + bytes / link_bytes_per_us

E_link = fixed_link_energy
       + bytes * link_energy_per_byte
       + link_domain_dynamic_energy

T_split = max(T_branch_0, T_branch_1, ...) + T_tail

E_split = sum(E_branch_i) + E_tail

E_route = E_load
        + sum(E_dynamic_kernel)
        + sum(E_link)
        + sum(P_idle[d] * T_route for every boundary domain d)
```

The optimization is:

```text
minimize    E_route

subject to  T_load + T_route <= request deadline
            resident weights + peak workspace <= each memory-pool capacity
            claimed staged allocations are adoptable by the execution process
            requested quality <= candidate quality
            every device and directed link is ready
            every candidate is placement-verified
            every kernel, link, load, and cut is measured in enforce mode
```

Every alternative charges the same `idle_charge_domains` set. For the current
fleet boundary this includes the CPU package, GPU board, and connected phone,
even when an alternative leaves one engine idle. This prevents a false energy
win caused by silently dropping an unused but still powered device from one
side of the comparison. A profile may use a smaller set only when it describes
a different, explicitly power-gated boundary.

This is constrained energy minimization, not a weighted `energy + lambda *
latency` score. A strict deadline can select a faster, higher-energy device.
With deadline slack, the planner selects the lowest-joule plan. This makes the
priority policy explicit and avoids silently changing latency versus energy
tradeoffs when units or coefficients change.

The dynamic program retains nondominated latency, dynamic-energy, active-time,
memory, domain, and activation-location states. Its default frontier limit is
4,096 states per operator. The output sets `search_optimal=true` only when no
frontier was truncated; otherwise it is the best bounded-search plan found and
must not be described as a proof of global optimality.

## Transfer and memory topology

The transfer graph is directed. CUDA to phone is not a direct edge unless a
physical direct path is measured. With the current topology, a candidate may
need:

```text
CUDA VRAM -> desktop RAM -> FunctionFS DMA-BUF -> phone memory
```

The solver enumerates simple paths and retains nondominated latency-energy
choices. H2D and D2H may use different link profiles. Fixed RPC time and the
payload-dependent term are both charged.

Memory belongs to pools, not compute engines. OP15 HTP and Adreno therefore
reference the same phone-DRAM pool. One truly shared allocation is counted
once when it has the same allocation ID and layout. HTP-packed and GPU-packed
copies use different allocation IDs and are both counted. Concurrent HTP and
GPU workspaces are added within the shared pool. Each device may also declare
an `allocation_limit_bytes`; this separately enforces limits such as the OP15
HTP mapping budget even when free shared phone DRAM remains.

## Candidate representation

An operator candidate defines:

- activation owner on entry and exit;
- one or more parallel branches;
- sequential compute and transfer steps in each branch;
- merge or post-processing steps after the fork-join;
- resident weight allocations and peak workspace;
- quality, evidence status, and evidence IDs; and
- split axis, selected amount, and full amount.

A normal CPU, CUDA, HTP, or Adreno operator is one branch with one profiled
kernel. A CPU plus HTP FFN split is two branches:

```text
host branch:   CPU complementary FFN columns
phone branch:  upload -> HTP FFN columns -> download
tail:          CPU merge
```

Latency uses the maximum branch time. Energy adds both branches. Two parallel
branches may not independently use the same energy domain; such an execution
must be represented by one physically measured fused-domain profile to avoid
double counting.

## Current RTX 4060 Ti plus OP15 coverage

| Component | Placement usable now | Energy status | Consequence |
| --- | --- | --- | --- |
| RTX 4060 Ti Gemma local layer | CUDA task/layer shortlist | Complete-layer measured, per-op attribution estimated | Planning only per operator |
| RTX 4060 Ti Qwen local layer | CUDA task/layer shortlist | Complete-layer measured, per-op attribution estimated | Planning only per operator |
| i9-12900K operators | CPU baseline and measured performance routes | Gemma dense FFN M=1, 8, 32, 128 package energy measured | Exact buckets available; other shapes remain shadow-only |
| OP15 HTP Gemma FFN | Certified CPU plus HTP execution route | 9,664-column M=1, 8, 32, 128 phone energy measured | Exact buckets available; arbitrary shapes cannot enforce |
| OP15 HTP Q/K/V, output, head | Performance shortlist by measured shapes | Per-op phone energy pending | Shadow only until energy acquisition |
| OP15 Adreno 840 | Native Q4 M=1 and F16 xmem M=16, 32, 128 at 512 columns | Isolated phone energy measured | Exact buckets available; other shapes remain shadow-only |
| CPU RAM <-> CUDA VRAM | Directional 8 KiB, 1 MiB, 4 MiB and duplex buckets measured | Transfer latency and energy measured; bounded directional fits derived | Use exact buckets or in-range conservative fit only |
| Desktop <-> OP15 DMA-BUF | Directional 1 MiB, 4 MiB and duplex/sync buckets measured | Fleet link latency and energy measured | Use exact buckets or in-range conservative fit only |
| HTP <-> Adreno shared memory | Shared physical pool known | Cache sync, dispatch, and energy by layout pending | Do not assume zero-cost engine switching |
| Model load, repack, eviction | Qwen/Gemma CUDA load/switch and selected phone layout preparation measured | Exact epoch costs measured; eviction and reuse prediction not generalized | Charge matching epoch cost or keep route resident |

A task route may claim staged CUDA bytes only when the process that executes
the route can adopt the exact allocation. File-page residency and CUDA memory
owned by another process do not qualify. An unadoptable claim is rejected as
`STAGED_ALLOCATION_NOT_ADOPTABLE`; it is not converted into an optimistic
PCIe-bandwidth saving. Even an adoptable route must use a measured reduced
load latency and energy row.

The full CPU plus OP15 BurstGPT campaign remains strong system evidence: on
the fixed I3 workload it reduced mean makespan by 14.38% and accounted energy
by 16.76%. That validates the exact cohort route. It does not make every FFN
shape, phone-GPU path, or CUDA staging path a measured additive energy row.

## What remains for the runtime scheduler

The following work is still required before calling the scheduler complete:

1. Extend the current graph-to-profile binding beyond its exact Gemma dense
   FFN buckets. Shared-memory transitions and merge kernels still require
   exact physical rows.
2. Validate every composed route on held-out full-model work. Per-operator
   rows become measured only when their physical acquisition and composed
   route error pass the declared bound.
3. Cache certified plans by model, quantization, shape bucket, device epoch,
   residency epoch, transport epoch, thermal bucket, and concurrency bucket.
4. Calibrate additional qualified thermal and contention buckets instead of
   rejecting every state outside the current I3 envelope.
5. Plan continuous batches or cohorts as the unit of work. Shared prefill and
   decode kernels are non-additive, so independently summing request energy is
   invalid.
6. Physically qualify grammar, context shifting, cooperative cancellation,
   speculative decoding, and KV migration before enabling those capabilities.
7. Run alternating physical A/B campaigns on the original BurstGPT lengths
    with the selected policy, including load/switch energy and all completed
    work.

## Tests

`test_placement.py` covers:

- lower-energy CPU versus faster CUDA selection under loose and tight
  deadlines;
- multi-hop CUDA-to-phone and phone-to-CUDA transfer charging;
- fork-join latency with additive CPU plus HTP energy;
- shared HTP and Adreno phone-memory capacity;
- measured-only enforcement versus estimated planning;
- task, layer, and operator scope classification;
- task-route load energy; and
- rejection of cross-process or otherwise unadoptable staged allocations; and
- structured operator, layer, transfer, memory, and energy output.

Run all S42 tests from the repository root:

```sh
python3 research_dev/scheduler/tests/run_all.py
```
