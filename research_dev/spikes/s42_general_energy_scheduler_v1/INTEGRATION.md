# S42 Integration Boundary

The execution workstream should treat S42 as a policy library. It should not
copy S42's replay runtime into `llama-server`.

## Admission flow

1. The server adapter maps a request to a stable `workload_id` and supplies
   shape features such as input tokens, expected output tokens, batch,
   context, bytes, FLOPs, or active experts.
2. The route manager snapshots device identity, model residency, worker epoch,
   transport epoch, thermal eligibility, and resource leases.
3. The scheduler evaluates only routes certified for that snapshot.
4. The route manager atomically acquires every declared resource before
   execution. A CPU-plus-phone operator route acquires the CPU lane, phone
   backend, transport, and any shared USB-root resource.
5. The executor runs the already-built route and reports actual completion,
   correctness, placement, energy, and failure state.

Do not let a scheduler decision itself prove readiness. Readiness comes from
the executor's current epoch and lease record. The S42 live probe deliberately
marks `op15-htp=false` until a worker owns resident weights.

## Route hierarchy

Use the coarsest eligible route:

1. task-level service on a resident phone;
2. a contiguous layer island with one entry and one exit transfer;
3. an operator island only when its compact boundary, merge, and concurrency
   were measured end to end; or
4. the server baseline.

The scheduler never builds a new cut. A profiler or offline solver may propose
one, but it becomes selectable only after physical qualification creates a
route profile.

`placement_planner.py` is that offline compiler. It may compare measured and
estimated CPU, CUDA, HTP, and phone-GPU candidates in planning mode. Enforce
mode accepts only measured kernels, links, loads, and placement-verified cuts.
Its output must still be validated end to end and published as a `RouteProfile`
before `UnifiedScheduler` can dispatch it.

The compiler operates on the real scheduling unit. For continuous batching,
that unit is the physical ubatch or a measured cohort, not an independently
modeled logical request. The graph adapter must emit the combined prefill and
decode shapes, activation bytes, and KV owner seen by the backend.

## Overlap-first execution

For a certified fork-join island, define:

```text
T_phone = T_host_to_phone + T_phone_queue + T_phone_compute + T_phone_to_host
T_island = max(T_host, T_phone) + T_merge
E_trace = sum(P_device_average * T_trace)
```

The executor dispatches the host and phone branches from the same ready input
without a blocking launch between them. It releases each CPU, GPU, phone, and
transport resource at that resource's actual completion, rather than holding
all resources until the join. Work from another ready request may fill an
otherwise idle interval.

Offline cut selection aims for the phone's conservative completion bound to
land just before the host branch. Online scheduling only chooses among those
certified cuts. It does not offload when queueing moves the phone past the host
slack. The physical promotion gate requires arithmetic-mean exposed join wait
at or below 5% of island time; p50 alone is diagnostic.

## Multi-device representation

Each physical compute engine, phone, transport, and contention domain is a
resource. For example, two phones on one USB controller declare separate NPU
resources and the same `usb-root-0` resource. This prevents a false scaling
projection that counts both links at full independent bandwidth.

Memory uses a separate pool topology. HTP and Adreno reference the same OP15
DRAM pool even though they are separate compute resources. Equal allocation
IDs represent a genuinely shared layout and are counted once; repacked HTP and
GPU layouts use different IDs and consume memory twice. A CUDA-to-phone
activation follows every profiled directed hop, normally CUDA D2H followed by
desktop-to-phone transfer, unless a direct path has physical evidence.

## Energy promotion

`enforce` accepts only `energy.status=measured`. Promote a route to measured
only after matched work under one synchronized boundary, at least three
alternating control/treatment repetitions, valid device identities, no lost
work, and uncertainty bounds. The route energy upper bound must be at least
5% below the baseline lower bound. Server power reduction alone is not enough.

V1 sums server CPU-package, server GPU-board, and whole-phone average power
times the same paid trace duration. The control charges the connected phone's
idle power for its full interval. The component set and omissions must match
between arms. A gross AC meter covering the desktop and USB-powered phone is a
stronger cross-check; it is a separate boundary and is never added to the
component sum.

The placement profile enforces this with one common `idle_charge_domains` set.
It must include every still-powered domain in the declared boundary for every
alternative. Omitting an unused GPU or connected phone from only one candidate
would create a false energy reduction.

The resident BGE batch-32 profile in `small_model_phone_v1/` also has a CUDA
lifecycle boundary. The route manager may use its reused-epoch variant only
when the same atomic precommit snapshot contains a tail-charge receipt for the
active or queued CUDA epoch. Unknown state uses the epoch-open variant. If that
variant selects CUDA, the receipt must be created in the same commit that
acquires the CUDA lease; if it selects OP15, no CUDA tail was opened and no
receipt is created.

## V1 limitations for integration

- The scheduler now reserves and releases phase-specific interval leases for
  CPU, GPU, USB, HTP, and Adreno resources. Early completion immediately
  updates queue predictions. Runtime revocation returns affected request owners
  without silently rerunning partially completed work.
- Model coefficients are immutable within a profile epoch. Online learning
  should publish a successor profile rather than mutate active evidence.
- Worker failure, stale heartbeat, thermal or contention mismatch, USB reset,
  model eviction, epoch mismatch, or unsupported request semantics makes the
  route ineligible and falls back before work starts.
- The I3 CPU-plus-OP15 route has three alternating matched pairs with
  synchronized CPU-package, GPU-board, and whole-phone energy. It reduces
  accounted fleet energy by 16.76% and makespan by 14.38%, has 2.67% trace-
  wide mean exposed wait, and matches the 27 / 64 MMLU64 control score. It may
  now compiled as the exact measured workload/profile-epoch cohort route with
  bounded approximate quality. Stage 6 independently validates one monitored
  successor pair at -13.83% makespan and -14.72% accounted fleet energy. The
  route is not an exact-token or universal per-shape certificate. M=1 through
  M=3 individually exceed the wait limit, and shared continuous-batch energy
  remains at cohort/epoch scope rather than being assigned additively to
  individual requests. The production executor handshake and pre-commit live
  snapshot remain outside this standalone prototype.
