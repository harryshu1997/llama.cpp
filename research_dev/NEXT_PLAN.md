# Q-PIM executable research plan

Status: authoritative gate order as of 2026-07-16.

The previous capacity-first mixed-workload scheduler plan is superseded. The new
target is dependency-aware power-frontier scheduling: reorder independent
multi-model DAG islands to unlock phone-resident work, then use power-trigger
bundles to create denser A6000 batches and measured lower-power intervals.

No full scheduler is authorized. Historical S10-V0 is invalid/inconclusive.
S10-V0-R passes its bounded temporal-enumeration and independent-optimality
foundation.

**E1 (typed evidence binding) is COMPLETE**:
`TYPED_EVIDENCE_INTEGRITY_PASS_PHYSICAL_CLAIMS_BLOCKED`. Every oracle input binds
to a digest-pinned hashed artifact, and the measured atlas is empty. E1 rejects
every `MEASURED` instance on purpose: its solver is additive per-device, while
`SERVER_WALL` and `GPU_BOARD` measurements are aggregate timelines of a whole
boundary, and feeding an aggregate into an additive solver double-counts shared
power. See `spikes/s10_power_frontier_repair/`.

**E2 (matched control/treatment timelines) is COMPLETE**:
`E2_MATCHED_TIMELINE_MECHANICS_PASS_MEASUREMENT_NOT_RUN`. E2 compares two realized
timelines POST HOC instead of feeding aggregates back into the solver. The
mechanics pass and no measurement was run. Two blockers are now concrete rather
than vague, both from a first-hand instrument audit:

- **No SERVER_WALL instrument exists on this host.** NVML is GPU-board only
  (two A6000 boards; `power.draw` is a 1 s average, +/-5 W). RAPL is root-only AND
  package/core only (no dram, no psys). No BMC, IPMI, PDU, or external meter.
  `SERVER_RELIEF_PASS_TOTAL_ENERGY_BLOCKED` is UNREACHABLE without new hardware.
- **A physical label needs an aggregate evaluator that is not built.** A single
  pair is diagnostic only (`PAIR_ONLY_NO_AGGREGATE_CLAIM`); a label requires
  `SUM_ALL_PAIRS_V1` over a complete predeclared repetition set.

The one existing A6000 trace is negative evidence: 323 rows contain only 57
power-value changes (a 10 Hz poll of a ~1.7 Hz sensor), it spans four p-states,
and it is not a matched pair. See `spikes/s10_matched_energy_e2/`.

**E2A (all-pairs aggregate) is the CURRENT gate**:
`E2A_R2_TARGETED_MECHANICS_PASS_PHYSICAL_CLAIM_BLOCKED`. E2A builds the
`SUM_ALL_PAIRS_V1` evaluator E2 deliberately left unbuilt, and adds a third
blocker that is independent of the two above and was not anticipated:

- **No INDEPENDENT EXTERNAL ANCHOR exists for the experiment plan.** An all-pairs
  sum is only worth something if the cohort was fixed before the results were
  seen; otherwise run 20 pairs, keep the best 8, and declare a plan of exactly
  those 8. A commitment needs PRECEDENCE (the plan predates the runs) and
  EXCLUSIVITY (only ONE plan was committed). **These do not covary.** RFC3161 via
  freetsa.org is genuinely independent and about ten minutes of provisioning away,
  and it buys precedence ONLY: a TSA is a responder, not a log, so nobody can
  enumerate how many other plans were anchored beside the one revealed. A token is
  a lower bound on a plan's AGE, never an upper bound on a plan's COUNT.
  **Provisioning a TSA would therefore not unblock E2A.** Closing it needs an
  enumerable commitment: third-party pre-registration, or a transparency log with
  a reviewable identity binding. TPM is present but permission-denied and
  custodially ours; git is our own force-pushable fork.
- **E2A implements no cryptographic verifier at all.** It types anchor capability;
  it parses no token, checks no signature, validates no inclusion proof.
  `ANCHOR_VERIFIERS` is empty and every kind is refused `E_ANCHOR_NO_VERIFIER`.

The three blockers stack and are independent: clearing any one alone changes
nothing. See `spikes/s10_matched_energy_e2_aggregate/` (ANCHOR_AUDIT.md is the
load-bearing document; RESULTS.md section 3 lists what is still open).

C0-C5, PF1, measurements, and runtime work all remain blocked.

## 1. Claim boundary

Primary claim:

~~~text
At equal closed work and end-to-end SLO:
  total wall J/completed work is at least 10 percent lower

or, at equal total wall power:
  SLO-valid completed work is at least 10 percent higher
~~~

The baseline is not eager llama.cpp. It is the best valid server-only policy
with the same DAG reordering, lazy batching, DVFS/power caps, and SLO knowledge.

Skipped GPU-us, greater phone utilization, longer idle time, or lower modeled
energy is not sufficient. A claimed win must be explained by an actual batch or
power-state change at the complete wall boundary.

## 2. Preserved substrate and evidence

Reusable substrate:

- Design A route A0, stage-local KV, GGUF shards, and three-device transport;
- S8 source pins, schemas, normalization contract, and mixed-service DAG work;
- S9 versioned weight identity, residency, prepared-image, and lease contracts;
- examples/phone-pim durable provision/resume/publish/PREPARE/EXECUTE path;
- one independently checked resident Gemma dense-FFN route on both phones;
- separate OP12/OP15 USB 3.2 Gen 1 contention domains; and
- existing CUDA, HTP, OpenCL, transfer, and interference harnesses.

S9-V1A-R closes the current transport measurement slice:

~~~text
full-shard window gate:
  OP12 2.61x median, 1.89x conservative, best window 4
  OP15 2.28x median, 1.27x conservative, best window 8

0 gate errors, duplicate chunks, or wasted bytes
T1/T2/T3 resume/durability tests PASS on both phones
~~~

The result is partly DVFS-sensitive and OP15 reached 95 C. It proves bounded
pipelining, not contract completeness, multi-model capacity, or energy. Freeze
it as substrate. Do not implement SHA de-duplication or protocol v4 unless S10
shows transport is a selected schedule's bottleneck.

Negative evidence remains binding:

- S3 rejects generic phone-backend output-row GEMV splitting.
- S4 rejects Adreno attention at realistic KV context.
- S5 rejects treating phones as raw additive A6000 operator throughput.
- S6 saturated overlap is not a request-latency or energy result.
- KV ownership requires a contiguous layer range and explicit lifetime lease.

## 3. New gate order

~~~text
PF0  S10 small power-frontier opportunity screen
  |
  +-- FAIL -> stop Q-PIM runtime work
  |
  v
PF1  reproducible traces, DAGs, power/route atlas
  v
PF2  real-trace exact and causal oracle
  v
PF3  smallest live Q-PIM runtime
  v
PF4  exclusive HBM ownership and stateful routes
  v
PF5  scale, thermal, and optional grouped-HMX bonus
~~~

## 4. PF0: S10-V0-R foundation and small falsification screen

Detailed contract: spikes/s10_power_frontier_repair/PLAN.md.

Purpose: test the mechanism before changing llama-server, model graphs, KV
internals, or the production backend scheduler.

### PF0-A: freeze the controlled instance

- [ ] Define two or three small DAG templates with explicit server pre/suffix
      islands and at least one phone-eligible complete island.
- [ ] Include at least two model/weight identities; same-model repetitions may
      provide native batches, while different models provide active bursts.
- [ ] Use concrete READY inputs and measured tensor sizes; no future-token or
      dependency prediction.
- [ ] Bind every CUDA/HTP/OpenCL route to current correctness and no-fallback
      evidence; UNKNOWN routes are excluded.
- [ ] Fix arrival, dependency, deadline/slack, model-mix, and load sweeps.
- [ ] Freeze server-only, placement-only, frontier-only, full-Q-PIM, and oracle
      controls before seeing the result.

### PF0-B: measure the minimum atlas

- [ ] Measure A6000 latency and energy versus batch and supported power
      cap/clock/state for every selected server island.
- [ ] Measure actual idle states, wake/transition latency and energy, and
      break-even gap. Do not assume a deep sleep state.
- [ ] Measure phone HTP/GPU latency and energy/thermal state for selected islands
      at available pacing controls.
- [ ] Include H2D activation, D2H result, verification, host relay, and USB VBUS.
- [ ] Use an external synchronized wall boundary. If phones are host-powered,
      include their VBUS draw in the server-wall measurement and avoid adding it
      twice.
- [ ] Establish whether instrumentation can distinguish a 10 percent effect.

### PF0-C: independent tiny oracle

- [ ] Implement a standard-library exhaustive enumerator for the frozen tiny
      DAG set.
- [ ] Implement a standalone solution checker sharing no candidate or objective
      code with the enumerator.
- [ ] Enumerate all topologically legal orders, READY routes, allowed batches,
      power-trigger bundles, and power states within fixed small bounds.
- [ ] Reject activation-memory overflow, invalid state, transfer omission,
      thermal-profile mismatch, and unfinished horizon work.
- [ ] Compare against optimized server-only, not eager FIFO.
- [ ] Add mutation tests for precedence, latest claim, wall-energy accounting,
      mirrored/exclusive credit, and terminal accounting.

### PF0-D: causal bounded policy

- [ ] Implement only a measurement/simulation policy: H-hop dependency lookahead
      plus deterministic bounded beam search.
- [ ] Generate unlocker sets, phone trigger bundles, and A6000 batch/power plans.
- [ ] Execute one simulated action and replan without future arrivals.
- [ ] Report oracle gap and separate prediction from correctness.

### PF0-E: controlled real-device reproduction

- [ ] Reproduce the smallest winning schedule with the A6000, OP12, and OP15.
- [ ] Use resident weights and the existing bounded phone-PIM command path.
- [ ] Compare optimized server-only, fixed phone placement, and Q-PIM in rotated
      order with identical offered work.
- [ ] Run enough repeated processes for confidence, then a sustained thermal run.
- [ ] Account every activation/result byte and every late/canceled phone result.

PF0 opportunity gate:

1. perfect-future oracle improves total wall energy by at least 15 percent in two
   adjacent declared load bins versus optimized server-only;
2. the causal bounded policy retains at least a 10 percent improvement;
3. the controlled real run shows at least 10 percent lower wall J/work or 10
   percent higher iso-power SLO-valid work;
4. SLO attainment is no worse and all offered work has one terminal outcome;
5. a measured batch-density or A6000 power-state change causally explains the
   result; and
6. the result survives conservative transfer, thermal, and measurement error.

Possible verdicts:

- PASS: all six gates pass; authorize PF1 only.
- MECHANISM_PASS_ENERGY_BLOCKED: ordering/overlap works but wall power is invalid;
  do not build the full runtime.
- FAIL: oracle, causal policy, or real mechanism misses the gate; stop Q-PIM.

## 5. PF1: traces, DAGs, and complete measured atlas

Goal: generalize a passing mechanism beyond the tiny controlled instance.

- [ ] Finish S8-V0b deterministic normalization and structural replay.
- [ ] Pass Gate A with byte-identical BurstGPT and RAGPulse outputs.
- [ ] Freeze request DAG semantics for generation, RAG/embedding-rerank, and one
      encoder/background class.
- [ ] Materialize conditional DAG branches only when their inputs are known.
- [ ] Freeze TTFT, TBT, completion, priority, and synthetic-SLO provenance.
- [ ] Profile A6000/OP12/OP15 routes for at least two service classes.
- [ ] Add batch, active-burst, power-state, phone-pace, thermal, boundary, and
      pairwise-interference surfaces.
- [ ] Record full wall-power validity and uncertainty for every energy row.
- [ ] Run 30-minute resident/thermal stability for shortlisted routes.

Exit:

- at least two service classes have certified complete phone islands;
- every route has correctness, boundary, state, latency, power, and thermal data;
- Gate A and the amended two-class atlas gate pass; and
- the PF0 mechanism remains feasible under the expanded atlas.

If the result supports only one model, narrow the claim before proceeding.

## 6. PF2: real-trace exact and causal oracle

Goal: determine whether power-frontier scheduling survives real burstiness,
model mix, dependencies, and residency.

- [ ] Freeze instance and solution-certificate schemas.
- [ ] Add pinned one-worker CP-SAT only after exhaustive/checker agreement on
      all tiny fixtures and at least 1000 generated instances.
- [ ] Model slow residency and fast frontier decisions separately.
- [ ] Model ANNOUNCED weight demand and READY concrete execution separately.
- [ ] Include mirrored/exclusive ownership, drain/reload, state leases, USB
      domains, activation memory, thermal duty, and power transitions.
- [ ] Run causal rolling-horizon and separately labeled clairvoyant oracles.
- [ ] Evaluate the deterministic bounded online policy through the same checker.

Required policies:

1. eager server-only;
2. optimized server-only DAG order plus lazy batching/DVFS;
3. whole-request/fixed phone placement;
4. phone placement without frontier shaping;
5. frontier shaping without power-trigger bundle credit;
6. full Q-PIM;
7. clairvoyant upper bound.

Exit:

- full Q-PIM beats optimized server-only by the primary 10 percent gate in at
  least two trace scenarios;
- the causal bounded policy, not merely CP-SAT, passes;
- gains survive arrival, profile, thermal, and power uncertainty; and
- activation memory, transfer, and scheduler overhead do not erase the result.

Stop before runtime integration if the causal policy fails.

## 7. PF3: smallest live Q-PIM runtime

Goal: reproduce only the winning PF2 mechanism.

- [ ] Add a host DAG/VQ orchestrator outside ggml backend policy.
- [ ] Feed admitted request/island milestones from a bounded harness first.
- [ ] Reuse S9 manifests, verified residency, PREPARE, EXECUTE, and D2H result.
- [ ] Implement authoritative READY, ownership, state, lane, link, and thermal
      reservations with generation-qualified epochs.
- [ ] Implement H-hop frontier construction and the bounded winning policy.
- [ ] Implement A6000 batch/power control and synchronized wall telemetry.
- [ ] Keep physical backend queues shallow and fail closed.
- [ ] Integrate one server path only after harness replay matches PF2.

Exit:

- live decisions match replay within declared timing/energy error;
- primary wall-energy or iso-power gate remains at least 10 percent;
- queues, state, and memory stay bounded for 30 minutes;
- phone loss, stale results, and thermal changes take declared outcomes; and
- no result relies on hidden fallback, duplicated HBM credit, or omitted power.

## 8. PF4: exclusive memory and stateful routes

Goal: add HBM relief without conflating it with mirrored fallback.

- [ ] Promote only proven mirrored routes to EXCLUSIVE_ACTIVE.
- [ ] Remove and measure exact server HBM allocations.
- [ ] Require drain plus completed reload before returning ownership.
- [ ] Add contiguous-layer KV leases only; reject token-prefix geometry.
- [ ] Include state handback bytes and replay cost.
- [ ] Evaluate low-priority batch decode and A0 as stateful controls.

Exit:

- at least 10 percent peak HBM or HBM byte-us relief at equal work/SLO;
- no immediate fallback copy is counted as released memory;
- state survives lease, drain, failure, and replay tests; and
- energy remains separately measured.

## 9. PF5: scale and optional kernel contribution

- [ ] Add devices through manifests and measured contention domains.
- [ ] Test one, two, and N phones with disappearance/rejoin and thermal rotation.
- [ ] Test model churn, cache pressure, fairness, and starvation.
- [ ] Screen cross-model descriptor-grouped HMX using test-backend-ops first.
- [ ] Include grouped HMX only if it improves a complete island by at least
      1.20x at two adjacent useful workloads with no fallback or hidden padding.

Kernel failure does not invalidate Q-PIM. It remains a measured bonus.

## 10. Global measurement and integrity rules

- Seven independent processes are the default latency protocol unless the spike
  freezes a stricter alternative.
- Controls rotate; first-run/cold effects are reported, not silently removed.
- Raw commands, stdout/stderr, exit status, hashes, device IDs, USB paths,
  thermal state, and exact sample counts are retained.
- Total wall energy uses synchronized windows and identical completed work.
- Missing or invalid power becomes UNKNOWN, never zero.
- Host, phone, and overlapping timers are not illegally summed.
- Capacity, HBM, time, and energy claims remain separately labeled.
- No commit, push, or upstream integration without explicit human approval.

## 11. Immediate instruction

Do not start C0-C5 or claim physical relief. The next controlled experiment
requires an enumerable independent plan commitment, a registered verifier, a
witnessed pre-run launch relation, and a valid power instrument. Until those
exist, preserve the E2A-R2 fixtures and keep runtime, llama-server, model-graph,
KV, backend-scheduler, protocol, and production-kernel work separate.
