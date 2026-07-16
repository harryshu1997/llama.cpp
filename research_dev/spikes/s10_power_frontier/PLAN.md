# S10-V0: power-frontier opportunity screen

Status: NOT RUN. This is the next and only authorized implementation slice.

## 1. Question

Can topologically valid reordering of concurrent DAG islands unlock resident
phone work early enough to change the optimized A6000 batch/power schedule and
reduce synchronized total wall energy at equal work and SLO?

This spike screens the proposed Q-PIM mechanism before any full scheduler,
llama-server, model-graph, KV, ggml backend-scheduler, or production-kernel work.

## 2. Hypotheses

H1 - Frontier:

~~~text
dependency-aware unlocker ordering increases timely phone-completed islands
over identical fixed placement with FIFO/EDF ordering
~~~

H2 - Causal energy:

~~~text
selected phone power-trigger bundles create a measured A6000 batch-density,
power-cap, or low-power-state change that lowers total wall J/work
~~~

H3 - Causal policy:

~~~text
a bounded no-future-arrival policy retains enough of the perfect-future oracle
gain to justify runtime implementation
~~~

H4 - Optional bonus:

~~~text
grouping multiple ready phone islands improves the complete phone leg
~~~

H4 is not required and does not authorize a kernel change.

## 3. Protected scope

Allowed:

- new measurement/simulation/checker code under this spike directory;
- a narrowly scoped opt-in harness or target under examples/phone-pim;
- existing public CUDA/ggml backend APIs and the certified phone-PIM protocol;
- measurement-only power, clock, queue, and thermal instrumentation; and
- documentation/status updates.

Forbidden:

- tools/server or llama-server scheduling changes;
- llama model graph or KV-cache changes;
- ggml backend scheduler policy changes;
- HTP/OpenCL/CUDA kernel changes;
- protocol-v3 semantic or wire changes;
- SHA de-duplication, protocol v4, or unrelated S9 optimization;
- commits or pushes.

Default behavior of every reused binary remains unchanged. New instrumentation
and scheduling modes are opt-in and fail closed.

## 4. Evidence frozen from S9

Use RESULTS_R.md as the only S9-V1A transport authority. Do not repeat withdrawn
V1A claims.

- OP12 full-shard best-window gate: 2.61x median, 1.89x conservative.
- OP15 full-shard best-window gate: 2.28x median, 1.27x conservative.
- T1/T2/T3 durability/resume PASS on both phones.
- Windowing is bounded to 64 MiB outstanding and tested values 1/2/4/8.
- OP12 and OP15 use separate USB SuperSpeed domains.
- OP15 was thermally throttled at 95 C in part of the matrix.
- Capacity and energy are unproven.

Do not optimize transport during S10 unless the selected winning schedule is
measured to be transport-bound.

## 5. Checkpoint 0: integrity and physical boundary

Before edits:

- record HEAD, full dirty status, relevant source/binary hashes, phone serials,
  worker hash, ADB endpoint, USB topology, A6000 identity/driver, and clocks;
- reproduce release and ASan/UBSan phone-pim tests;
- reproduce the smallest current OP12/OP15 FFN correctness command;
- inventory available A6000 power caps, application clocks, P-states, and
  transition controls without changing persistent system configuration; and
- identify the synchronized external wall-power source and sample rate.

The primary wall boundary must include:

~~~text
A6000 + host CPU/DRAM/PSU + USB controllers/relay + phone/charger draw
~~~

If phones draw from host USB, that draw is already inside host wall power and
must not be added again. If separately powered, use synchronized wall-side phone
meters. Battery current under a capped charging rail is not a valid substitute.

CP0 outcomes:

- READY: complete wall boundary, synchronization, and uncertainty are valid.
- ENERGY_BLOCKED: no valid wall boundary or uncertainty cannot resolve 10
  percent. Continue only through the analytic mechanism screen; do not claim
  S10 PASS or authorize PF1.

## 6. Checkpoint 1: freeze the controlled DAG instance

Freeze before measuring policy results:

- two or three DAG templates;
- at least 12 concurrent instances and a bounded maximum of 24 for the tiny
  oracle;
- explicit release times, dependency edges, latest SLO-safe milestones, and
  terminal outcomes;
- one or more server pre-islands, one phone-eligible complete middle island, and
  one server suffix where the real dependency requires it;
- at least two distinct weight_set_id values;
- same-weight repetitions that can form a native server batch;
- different-weight work that cannot be mislabeled as one tensor batch;
- exact activation/result/state bytes; and
- fixed load, burstiness, model-mix, and slack sweeps.

Preferred real route:

~~~text
server CUDA prerequisite
  -> resident complete phone FFN or stateless service island
  -> server CUDA suffix
~~~

Concrete input tensors must exist before an island becomes READY. A synthetic
tensor generator is allowed only when its dtype, shape, byte size, and
correctness oracle are frozen and labeled. A two-weight same-model screen may
falsify the mechanism, but a positive general mixed-model verdict requires at
least two model_id values before PF1 authorization.

Every policy receives the identical closed cohort. Work unfinished at the
horizon is timed out, never omitted.

## 7. Checkpoint 2: minimum route and power atlas

For every selected server island, measure:

- latency and completed work versus batch size;
- supported A6000 cap/clock/P-state;
- board and complete-wall energy;
- idle/low-state power;
- transition/wake latency and transition energy; and
- 7-process p50/p95/p99/CoV unless a stricter frozen protocol is justified.

For every selected phone route, measure:

- HTP/GPU kernel provenance and no fallback;
- CPU-reference correctness, finite/repeat checks, and exact output bytes;
- input H2D, compute, output D2H, and verification separately;
- available pacing control and latency/energy point;
- temperature, clocks, throttle state, and sustained duty; and
- pair interference only if the test actually selects concurrent lanes.

Compute the measured A6000 break-even gap:

~~~text
t_break_even =
  wake_latency
  + transition_energy / (idle_power - lower_state_power)
~~~

Do not proceed to a physical energy gate when no measured lower state exists.

## 8. Checkpoint 3: independent exact tiny oracle

Implement:

1. a standard-library exhaustive or branch-and-bound enumerator;
2. a standalone solution-certificate checker with no imported solver,
   candidate-generator, simulator, or objective code; and
3. canonical JSON instance/result records with deterministic hashes.

Bounds are frozen in the instance schema and small enough to enumerate. The
oracle covers:

- all topologically legal orders;
- server/phone route assignment from the certified catalog;
- same-weight server batches and different-model active bursts;
- mirrored race or committed route semantics;
- sleep, batch-shaping, and power-cap trigger bundles;
- A6000 and phone operating points;
- link/lane/memory/state/thermal capacity;
- latest SLO-safe claims;
- every transition and total wall-energy interval; and
- exactly one terminal outcome per job.

Mutation tests must independently catch:

- a removed dependency;
- execution before READY;
- an omitted H2D/D2H/state byte;
- double-counted or missing USB/phone wall energy;
- HBM credit in mirrored mode;
- instant or free A6000 state transition;
- a server claim after latest_start;
- activation-memory overflow;
- stale/duplicate completion; and
- unfinished work omitted at the horizon.

Run at least 1000 deterministic generated tiny fixtures after all hand-written
adversarial fixtures pass.

## 9. Checkpoint 4: policies and opportunity sweep

Frozen policies:

~~~text
C0 eager server-only FIFO
C1 optimized server-only DAG order + lazy batch + power control
C2 fixed phone placement, no frontier shaping
C3 dependency frontier shaping, no causal power-bundle credit
C4 full Q-PIM perfect-future oracle
C5 full Q-PIM causal bounded policy
~~~

The causal policy:

- sees only arrived DAG nodes, current READY state, and current measurements;
- uses bounded H-hop lookahead and deterministic K-state beam search;
- enumerates only certified power-trigger bundles;
- commits one action and replans;
- never sees future arrivals, failures, output lengths, or thermal changes; and
- returns the best checked feasible incumbent on timeout.

Sweep at least:

- offered load from low to overload;
- bursty and smooth arrivals;
- tight and slack-rich SLOs;
- same-model-heavy and mixed-model-heavy cohorts;
- cold, warm, and conservative throttled phone profiles; and
- measured transfer/power uncertainty bounds.

Report full Pareto curves, not only the best point.

## 10. Checkpoint 5: smallest real-device schedule

Only when C4 and C5 pass their analytic gates:

- reproduce one smallest winning DAG/order/bundle on A6000 + OP12 + OP15;
- use already resident verified weights;
- rotate C1, C2, and C5 order;
- retain all host/device stdout, stderr, exit status, commands, clocks, thermal,
  power samples, transferred bytes, and output hashes;
- require identical completed work and one terminal outcome per instance;
- run repeated short trials, then at least one 30-minute sustained trial; and
- compare synchronized complete-wall joules and iso-power goodput.

No simulated sleep is a physical result. If the A6000 cannot enter the modeled
state, that power-trigger bundle fails.

## 11. Gates and stop rules

Opportunity gate:

- C4 saves at least 15 percent total wall energy versus C1 in two adjacent
  declared load bins with no worse SLO.

Causal gate:

- C5 saves at least 10 percent versus C1 in the same bins and retains at least
  two thirds of C4's absolute gain.

Mechanism gate:

- C5 increases timely accepted phone islands over C2; and
- a measured larger server batch, lower cap, or break-even low-power interval
  explains the benefit.

Physical gate:

- upper 95 percent confidence bound of C5 wall J/work is at most 0.90 times C1;
  or C5 completes at least 1.10 times SLO-valid work at equal wall power;
- all activation/result/state/link/host/phone energy is included; and
- the benefit survives the sustained thermal run.

Verdict:

- PASS only when opportunity, causal, mechanism, and physical gates all pass.
- MECHANISM_PASS_ENERGY_BLOCKED when ordering works but physical energy is
  invalid; stop before PF1/full runtime.
- FAIL at the first missed analytic or real gate; stop and preserve evidence.

H4 grouped-HMX is tested only after the core PASS and has its own 1.20x
complete-island gate at two adjacent useful workloads.

## 12. Deliverables

~~~text
research_dev/spikes/s10_power_frontier/
  PLAN.md
  RESULTS.md
  MANIFEST.md
  schemas/
  fixtures/
  oracle/
  checker/
  policies/
  scripts/
  artifacts/
~~~

RESULTS.md must lead with the verdict, list every deviation/blocked cell, and
separate measured, inferred, and simulated values. Nothing is committed or
pushed. Stop for review after the first decisive verdict.
