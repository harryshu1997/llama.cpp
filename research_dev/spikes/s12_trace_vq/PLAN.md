# S12-V0 Trace-Driven Virtual Queue

## S12-V2 Mixed-Residency Extension

V2 adds a separate synthetic two-model, two-phone engine in
`two_level_vq.py`. The server keeps full fallback residency. A causal slow loop
chooses `KEEP`, `PREFETCH`, `REPLICATE`, or deferred eviction/prefetch from an
immutable current-state snapshot. A fast loop dispatches only matching READY
replicas; otherwise a free server takes the request.

The current gate proves state mechanics only:

- independent OP12/OP15 capacity, generation, and activation ledgers;
- symbolic content and backend identity on residency and dispatch;
- `RECEIVING -> VERIFYING -> PREPARING -> READY` transitions;
- pin-safe `LEASED -> DRAINING -> EVICTING` replacement;
- exact two-device instantaneous assignment under bounded synthetic scores;
- finite queue and horizon outcomes with terminal conservation; and
- deterministic replay without changing the V0/V1 hashes.

V2 does not use priority for ordering; deadlines classify terminal outcomes.
It does not model a multi-island DAG, tail batching, physical link faults,
thermal state, real capacity, or energy. The next authorized slice is the
measured-profile adapter and mixed-trace replay, not production integration.

Run it with:

```sh
python3 research_dev/spikes/s12_trace_vq/two_level_vq.py \
  --config research_dev/spikes/s12_trace_vq/configs/two_level_fixture.json
```

## S12-V1 Dual-Path Extension

S12-V0 below is frozen. S12-V1 adds a separate executable replay in
`dual_path_vq.py` for the newly selected physical topology:

```text
server -> phone input: WiFi H2P
phone -> server result: per-phone USB P2H
```

The paths use distinct physical lanes. V1 represents them independently, but
does not overlap them because the current OP15 stage has one KV context and V1
allows one in-flight group. Phone compute remains serialized against both paths
until a real interference matrix is measured.

Each phone group advances through bounded WiFi, phone-input, compute,
phone-result, USB, host-result, and A6000-tail states once for prefill and once
for every decode step. A decode input is not eligible for WiFi until the
preceding A6000 tail produces its token. Terminal completion is impossible
before the final USB return and tail complete. Horizon cleanup releases all
buffer and in-flight credits.

Each V1 policy also freezes host residency for the whole replay. Server-only
policies use `FULL_MODEL`; fixed-phone uses `TAIL_ONLY`. V1 rejects dynamic
mixing because S11 measured those allocations in separate processes and no
load/unload transition is implemented. The static allocation is charged even
while the A6000 compute lane is idle.

The current OP15 head route receives token commands over WiFi and returns a
dense cut activation over USB. A generic middle island may receive and return
dense activations, so route profiles must carry independent directional byte
counts.

The eventual S9/S12 composition assigns small latency-sensitive commands and
token metadata to WiFi H2P, large resumable weight segments to USB H2P, and
dense results to USB P2H. USB result traffic preempts background weight
prefetch because both directions share one phone link unless a measured duplex
profile proves otherwise. V1 does not yet simulate weight transfers; it assumes
the required S9 residency certificate already exists.

V1 has three policies: offline server-only, causal server-only, and fixed
phone. The V0 memory-trigger policy remains historical evidence but is not
admitted into V1. V1 is mechanics-only. Its path rates are assumptions and `phone_stage_us` is an
old-path wall-time proxy, not isolated phone compute. No latency, throughput,
capacity, or energy conclusion may use its output. See `DUAL_PATH_DESIGN.md`.

## Scope

Build a bounded offline replay outside `llama-server` that answers one narrow
question:

Can a causal virtual queue use the exact S11 phone route only as an A6000
memory-admission fallback, while preserving finite queues, exact byte
accounting, and explicit terminal outcomes?

This is a mechanics gate. It does not integrate production runtime code,
authorize a real-trace performance claim, or measure energy.

## Evidence Boundary

`profiles/s11_batched_route.json` binds the five measured S11 rows for
`B={1,2,4,8,16}` to their plan and summary digests. The only eligible payload
is:

```text
model:       Gemma-4 12B IT F16, exact model digest
prompt:      exact S11 prompt digest
input:       28 tokens
output:      4 greedy tokens
context:     96 tokens per sequence
phone route: OP15 HTP0 layers [0,2), then A6000 layers [2,48)
```

The S11 rows are `MECHANICS_ONLY_SINGLE_PROCESS_PAIR`. They are not a service
atlas, sustained throughput certificate, backend-placement certificate, or
energy profile.

Every phone batch charges exactly:

```text
activation_bytes = B * (28 + 3) * 3840 * 4
```

The 31 rows are 28 prefill rows plus the first three decode activations. All
time, byte, and MiB fields are integers.

## Trace Modes

### `strict_real`

A row is profiled only if its token shape, exact model digest, prompt digest,
context, and chat mode all match S11. Shape equality alone is insufficient.
An unprofiled row receives no server or phone latency and terminates as
`unprofiled`.

The pinned BurstGPT and RAGPulse fields do not contain the S11 model, prompt,
or context identity. Therefore the strict real profile coverage is zero. This
is the expected fail-closed result until varied payloads are measured.

### `semi_synthetic_shape_shadow`

This mode preserves event IDs and arrival timestamps but replaces every
payload, model, and shape with the exact S11 profile. Every output carries:

```text
SYNTHETIC_ONLY_NO_REAL_TRACE_PERFORMANCE_CLAIM
```

It exists only to test queue mechanics before S8 normalization and a varied
payload atlas are available.

## V0 Policies (Frozen Historical Replay)

1. `server_only_optimized`: exhaustive offline FCFS partition reference. It
   reads all future arrivals, optimizes `(makespan, sum_completion, batches,
   partition)`, and is explicitly `clairvoyant=true`.
2. `causal_server_batch`: holds the oldest request for a fixed bounded interval,
   then dispatches the largest currently queued measured server batch that fits
   current A6000 HBM.
3. `fixed_phone`: sends every profiled request to the resident S11 phone route.
   It is a diagnostic baseline, not a deployable default.
4. `memory_admission_triggered`: selects a server batch whenever any currently
   dispatchable server batch fits. It may select a phone batch only when no
   server batch fits current HBM.

The three V0 causal policies receive only current queue state. They do not receive
future request arrivals. The offline reference is the only exception and is
labeled at every result.

## Resource Model

- One finite virtual queue per policy, with explicit overflow terminals.
- One A6000 lane. A phone route reserves both OP15 and the A6000 lane for the
  complete measured group time. No unmeasured inter-group pipeline overlap is
  assumed.
- OP15 weights are already resident and readiness is frozen in the config.
- A time-indexed exogenous HBM reservation models other server work.
- A background-HBM increase that oversubscribes an active route fails the run.
- Offline peak HBM is evaluated at route start and every reservation transition
  inside the active half-open interval.
- The replay stops at a finite horizon and terminalizes every remaining
  request.
- Terminal conservation is asserted:

```text
completed_server + completed_phone + rejected_queue_full
  + timed_out + unprofiled == arrivals
```

- Energy is always `NOT_RUN`; all energy fields are null.

Stable event order at equal integer microsecond timestamps is:

```text
completion, HBM change, arrival, flush, insertion sequence
```

Routes occupy half-open intervals `[start,finish)`, so a completion at the
exact timestamp of a new background reservation releases memory first.

## Gates

- [x] Strict duplicate-key JSON and ASCII canonical output.
- [x] Trace bytes are read once; parsing and the recorded digest share that
      immutable snapshot.
- [x] Five S11 rows bind exact source and raw artifact digests.
- [x] Exact activation, HBM MiB, and integer-microsecond accounting.
- [x] Finite queues, finite horizon, and terminal conservation.
- [x] Stable event order and cross-`PYTHONHASHSEED` replay.
- [x] Offline reference labeled clairvoyant.
- [x] Causal policies have no future-arrival input.
- [x] Memory policy phone decisions require server admission failure.
- [x] Background-HBM overcommit fails closed.
- [x] Rising background HBM inside an offline route contributes to measured
      peak occupancy.
- [x] Strict unprofiled rows receive no phone latency.
- [x] Shape shadow is labeled synthetic-only.
- [x] Energy is `NOT_RUN`.

## Commands

```sh
python3 research_dev/spikes/s12_trace_vq/run_tests.py

python3 research_dev/spikes/s12_trace_vq/profile_coverage.py \
  --trace research_dev/spikes/s12_trace_vq/fixtures/varied_arrivals.jsonl \
  --profile research_dev/spikes/s12_trace_vq/profiles/s11_batched_route.json \
  --mode strict_real

python3 research_dev/spikes/s12_trace_vq/vq_sim.py \
  --config research_dev/spikes/s12_trace_vq/configs/shadow_fixture.json

python3 research_dev/spikes/s12_trace_vq/dual_path_vq.py \
  --config research_dev/spikes/s12_trace_vq/configs/dual_path_shadow_fixture.json
```

Stop after mechanics and coverage. Do not edit the S8 normalizer, integrate the
production server, or claim real workload, throughput, capacity, or energy
benefits.
