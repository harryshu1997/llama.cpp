# S9-V0 Results

> UPDATE (S9-V0-R3 plus runtime slice, 2026-07-14): a post-v4 mutation audit found
> further static holes. Append-only bundle/schema v5 now validates every record
> shape before dereference, binds all schema-allowed content fields, and closes
> exact manifest/segment/range/I/O/SoC/causality/transfer/duplicate/physical-ledger
> cases. Evidence: 24 v5 schema fixtures, 28 v5 bundle fixtures, and 34 red-v4/
> green-v5 checks, all passing. This is still static snapshot coherence, not live
> scheduler atomicity. Separately, `examples/phone-pim/` now implements the first
> real trusted-localhost pre-staged FFN command runtime. OP12 and OP15 both pass an
> independent production Gemma4 oracle on HTP0. It has no dynamic weight stream,
> multi-model scheduler, capacity, or energy claim. Reports: `V0R3_REPAIR.md` and
> `../s9_phone_pim_runtime/RESULTS.md`.

> UPDATE (S9-V0-R2, 2026-07-14): the 2026-07-14 review found eleven static v3 fail-open
> blockers (authoritative DeviceInventory gates, lease expiry, executable identity, transport
> epochs, and physical-ledger reservations). R2 freezes v1/v2/v3 byte-for-byte and closes all
> eleven under a versioned **v4** contract (schema_version/bundle_version 4, a v4-only
> `s9:<kind>:v4` digest family that additionally binds source_weight_set_id, ticket issued
> boot/gen, and the state-lease reservation fields). `cross_record_v4` binds each DISPATCH to
> one authoritative DeviceInventory snapshot, rejects expired leases, derives epoch/credit
> gates from records, binds full executable identity, binds BULK/EXECUTE/RESULT frame epoch
> stacks, and derives a single-copy physical ledger from live records. The pre-R2 v3 validator
> + digest lib are frozen at `golden/v0r1_historical/` as the red side. An 8-lens adversarial
> hole-hunt then found SIX more fail-open holes the first v4 pass missed (RESULT frames
> unbound; identity fail-open when the manifest is absent; arch/soc/layout_version never
> cross-checked; DI-less devices escaping the ledger; a cert attesting a footprint > device
> LPDDR; duplicate dispatch decisions per request) -- all closed with red-v3/green-v4 fixtures.
> Label: `STATIC_SNAPSHOT_COHERENT` only; NOT a completeness proof and NOT live dispatch
> (atomic snapshot acquisition, compare-and-reserve use pins, and completion races remain V0c).
> Suites green: 116 schema + (v2 15 + v3 18 + v4 27) bundle + 23 semantic + 26/17/16/13 sim
> (untouched) + 33 v4 regression/effectiveness. Full report: `V0R2_REPAIR.md`. Nothing committed.

> UPDATE (S9-V0-R1): an independent adversarial pass found the V0-R suites PASSED while
> the mechanics were STILL fail-open, so V0-R is relabeled
> `SUITES PASS; MECHANICS CERTIFICATION BLOCKED; CAPACITY UNPROVEN`. The R1 round freezes
> V0-R as RED evidence (golden/v0r_historical/, replay `af1501b6...`), introduces a
> versioned R1/v3 contract, and CLOSES all sixteen fail-open cases with red-before/
> green-after evidence (sim/test_r1_regressions.py 17/17). R1 golden replay `9a69a7f6...`.
> Suites: 100 schema fixtures (v1+v2+v3, both validators) + v2/v3 bundle selftests + 23
> semantic + 26 preserved V0-R behavior + 17 R1 regressions + 10 golden/preservation, all
> green. A SECOND independent adversarial pass (an 8-agent workflow + a review DRAINING-lease
> dispatch probe) then found 10 FURTHER fail-open holes that survived the first R1 pass
> (e.g. the DRAINING lease, which `bundle_validate.py` accepted); all 10 are now closed with
> regression evidence (the DRAINING probe now
> returns E_CHAIN_BROKEN). This is NOT a completeness proof -- every adversarial round has
> found new holes and the validator checks a static bundle, not live dispatch -- so R1 closes
> its targeted + adversarially-found cases but is NOT claimed as complete scheduler dispatch
> authority. A 2026-07-14 review then closed two more validator bindings and six simulator
> defects, expanding `test_r1_holes.py` to 16 checks and the v3 bundle index to 18. Eleven
> additional v3 coherence mutations still validate, so a frozen-v3/versioned-v4 repair is
> required. Full R1 report: `V0R1_REPAIR.md`. Production correctness and capacity remain UNPROVEN.

> UPDATE (S9-V0-R): the fail-open contracts and simulator mechanics the review found
> are now REPAIRED under a versioned v2 schema bundle, with red-before/green-after
> evidence for every repair. The frozen V0 result below is relabeled
> `SUITES PASS; MECHANICS NOT YET CERTIFIED; CAPACITY UNPROVEN`; V0-R is
> `SUITES PASS; MECHANICS CERTIFIED (repaired + adversarially regression-tested);
> CAPACITY UNPROVEN`. Full V0-R report: `V0R_REPAIR.md`. V0-R golden replay
> `sha256:af1501b6...`; suites: 86 schema fixtures (both validators) + 15 bundle + 23
> semantic + 26 sim + 14 red/green regressions + 6 golden-replay, all green. The
> numbers in the rest of this file describe the FROZEN V0 run
> (golden/v0_historical/, replay `sha256:64fcad3b...`).

Status: contracts/simulator plus append-only v5 static repair COMPLETE; S9-L0
dual-phone ADB link evidence COMPLETE; pre-staged FFN runtime mechanics PASS;
dynamic provisioning and live scheduling PLANNED. No downloader service,
llama-server integration, production scheduler, capacity, energy, novelty, or
PIM-hardware claim. Synthetic workload fixtures only (S8 Gate A not independently
confirmed). Nothing committed or pushed.

## Verdict

`TARGETED SUITES PASS; SCHEDULER DISPATCH CERTIFICATION BLOCKED; CAPACITY
UNPROVEN`. The residency, transport, and prefetch contracts are machine-checkable
research artifacts, and the simulator exercises its modeled state machines,
ledger, preemption, recovery, determinism, and never-wait behavior. Two adversarial
passes found 16 + 10 fail-open holes (including the DRAINING-lease probe), all now
closed with regression evidence; a later review found eleven additional static v3
coherence holes. Because every pass has found new gaps and the validator is not a
live authoritative dispatch gate, no complete
production-dispatch-certification claim is made. USB negotiation and
end-to-end ADB staging goodput are now measured,
but the decomposed transport, storage, preparation, compute, and interference rates
are not. The V0 simulator also assumes one symmetric shared controller and omits
D2H result transfer. It therefore does NOT prove a capacity number. A
capacity/energy claim stays BLOCKED behind a record-derived dispatch repair,
S9-V1 link-model measurements, and confirmed real traces.

## Files (all new, under research_dev/spikes/s9_dynamic_weight_residency/)

- Docs: `PLAN.md`, `RESULTS.md`, `SUBSTRATE_AUDIT.md`, `WEIGHT_RESIDENCY_CONTRACT.md`,
  `TRANSPORT_CONTRACT.md`, `PREFETCH_POLICY.md`, `SIMULATOR_SPEC.md`,
  `CURRENT_SLOW_LINK.md`.
- Schemas: `schemas/*.schema.json` (15 records + informative `_defs`).
- Fixtures + tooling: `make_fixtures.py`, `run_schema_tests.py`,
  `validate_manifests.py`, `fixtures/` (53 schema + 23 semantic + 2 indexes).
- Simulator: `sim/residency_sim.py`, `sim/make_scenarios.py`,
  `sim/test_residency_sim.py`, `sim/scenarios/baseline_sweep.config.json`.
- Documentation integration: `research_dev/talks.md`,
  `research_dev/MIXED_WORKLOAD_DESIGN.md`, and `research_dev/NEXT_PLAN.md`. This
  spike is otherwise contained under its S9 directory.

## Tests + exact commands

Run from `research_dev/spikes/s9_dynamic_weight_residency/`:

```
python3 make_fixtures.py                 # regenerate 53 schema + 23 semantic fixtures
python3 run_schema_tests.py              # both validators over all schema fixtures
python3 validate_manifests.py --selftest # semantic cross-field checks
python3 sim/make_scenarios.py            # emit the committed sweep scenario
python3 sim/test_residency_sim.py        # 28 checks: 10 required behaviors + determinism
python3 sim/residency_sim.py sim/scenarios/baseline_sweep.config.json --out /tmp/run.json
```

Outputs (verified):
```
run_schema_tests.py : validators pinned OK (jsonschema 4.10.3, ajv-cli 5.0.0);
                      prechecks OK (53 entries); 15/15 schemas compile+load;
                      fixtures 53 (22 valid + 31 invalid) failures 0; exit 0
run_schema_tests.py --index /tmp/s9_bad_idx.json : exit 1 (phantom fixture -> guard proven)
validate_manifests.py --selftest : semantic 23 (9 valid + 14 invalid) failures 0; exit 0
sim/test_residency_sim.py : sim tests 28 failures 0; exit 0
                            byte-identical replay sha256:64fcad3b...e59d1e (stable across processes)
```

Required-test coverage (all PASS in sim/test_residency_sim.py):
1. schemas compile under BOTH validators -- run_schema_tests.py (15 schemas, 53 fixtures).
2. valid + adversarial fixtures -- 31 adversarial schema fixtures + 14 adversarial semantic.
3. byte-identical deterministic replay -- same config+seed -> identical replay hash.
4. hash-mismatch + partial-transfer recovery -- quarantine + fallback; resume-not-restart.
5. duplicate / stale-epoch / reordered rejection -- FrameSequencer (TRANSPORT_CONTRACT s7).
6. lease-safe eviction + exact physical-byte accounting -- ledger partition invariant + eviction.
7. no dispatch from partial/on-disk weights -- VERIFIED_ON_DISK yields no cert -> fallback.
8. no live-state eviction -- pinned residency never evicted (live_state_evictions == 0).
9. activation preempts bulk prefetch -- send_activation delays in-flight transfer.
10. unknown-profile / unsupported-backend fail closed -- hard gate -> fallback.

## Contract invariants (machine-checked)

- Content identity is SHA-256; the derived-image identity binds nine fields
  (model_version, tensor_digest, graph_hash, backend_build, soc, arch,
  layout_version, boot_epoch, residency_generation); `derived_image_digest`
  recomputation is checked in validate_manifests.py.
- ReadyCertificate is fail-closed by construction (warmup_passed const true,
  correctness.verdict const pass) -- an un-ready cert is unrepresentable.
- DispatchDecision `verdict==DISPATCH` requires every gate: non-empty certs,
  residency lease present, all epoch_match true, all five credit classes ok
  (weights/derived/scratch/activations/state), correctness pass, empty hard gates,
  reason dispatch_ok. No path from unknown/partial input to DISPATCH.
- WeightSet is atomic (all-or-none); set_digest + total_bytes recomputed.
- Physical-byte ledger partitions LPDDR exactly:
  weights+derived+scratch+activations+mutable_state+free == total; canonical vs
  backend-derived tracked separately (F16 canonical HTP<->GPU shareable; GPU xmem
  prepacked NOT).
- TransportFrame: bounded (16 MiB / 65536 control), checksummed, LE; bulk requires
  payload_sha256 + BULK_CHUNK + background priority; activation/result outrank
  background weights.
- ModelManifest partial_load_supported=false => exactly one full-range weight set
  (arbitrary-model REJECT gate; gemma4-only sharding per SUBSTRATE_AUDIT G3).

## Simulator inputs + baseline results

Scenario `baseline_sweep`: 1 phone (op15), 1 model (900 MB canonical, HTP), 12
decode requests 2 s apart, 5-point V0 sensitivity sweep, 8 baselines. Its controller
ceiling is the negotiated 5 Gbps endpoint ceiling, not measured application
goodput. Result (dispatched-to-phone / server-GPU-us-freed /
completion-p95-us):

```
goodput    server_only   per_request_fetch    never-wait cache/predictive/clairvoyant
40 MiB/s   0 /0/ 2000     12 /24000/ 23674256   0 /0/ 2000        (transfer >> horizon)
100 MiB/s  0 /0/ 2000     12 /24000/ 10571250   6 /12000/ 12875
250 MiB/s  0 /0/ 2000     12 /24000/  5374750   9 /18000/ 7250
400 MiB/s  0 /0/ 2000     12 /24000/  4083532   9 /18000/ 5844
550 MiB/s  0 /0/ 2000     12 /24000/  3494410  10 /20000/ 5205
```

The numeric sweep uses binary MiB/s despite the original `MB/s` labels: 40/100 are
degraded-link sensitivity points, 250 is close to the measured OP15 staged rate,
and 400/550 are optimistic transport sensitivities. None is a measured decomposed
link profile.

Reading: `per_request_fetch` (the diagnostic losing baseline) dispatches everything
to the phone but pays 3.5-23.7 s p95 because it WAITS for weights on the critical
path. The never-wait policies keep p95 at 5-13 ms and gain server relief that scales
with goodput (0 -> 20000 GPU-us freed). `server_only` is the flat floor. In this
single-model, no-eviction fixture the four never-wait policies and clairvoyant
coincide; they diverge only under RAM/eviction pressure and multi-model reuse, which
the eviction scenario in the test harness exercises separately. These are MECHANICS,
not a capacity claim. The existing V0 one-controller model must not be used to make
a two-phone conclusion from the new measurements.

## Unknown / symbolic parameters (honest)

Measured: endpoint negotiation, physical bus topology, and ADB end-to-end staged
file rates in both directions. Still symbolic: native link goodput, UFS-only rates,
verify/materialize/prepare rates, `warmup_us`, `phone_compute_us`, per-class server
GPU-us and HBM bytes, and derived-image bytes (the xmem ~1x-extra ratio informs the
fixture; exact per-tensor values are symbolic). Interference is pinned at 1000
permille (no interference) because no measured HTP<->GPU or transfer<->compute
matrix exists; the simulator honors interference ONLY when
`interference.measured==true`. No energy is emitted; energy stays DISABLED.

## Current-link evidence

Both phones negotiate 5000 Mbps and occupy separate SuperSpeed root buses: OP12 on
Bus 006/path 6-2 and OP15 on Bus 008/path 8-3. A non-disruptive ADB server on port
5038 enabled a verified 1 GiB incompressible transfer test with three solo reps.
Host-to-phone-file medians were 215.9 MiB/s (OP12) and 261.9 MiB/s (OP15);
phone-file-to-host medians were 189.8 and 226.8 MiB/s. Concurrent fleet makespan
goodput was 409.4
MiB/s H2D and 392.6 MiB/s D2H, with matching SHA-256 on both devices. Full method,
elapsed samples, integer rates, and hash are in `CURRENT_SLOW_LINK.md`.

These are practical ADB-to-file provisioning rates, not raw USB rates. Push already
includes phone file-write/page-cache effects, so using it as V0
`goodput_bytes_per_s` while also charging V0's UFS stage would double-count part of
the storage path. Pull similarly is not a
memory-resident result path. The measurement updates the plan and cold-fill time
scale; it does not upgrade the V0 capacity verdict.

## Gate verdicts

- Contract Gate (schemas + semantic + sim mechanics): PASS -- 15 schemas validate
  under both validators; 53 + 23 fixtures pass; 28 sim checks pass; deterministic
  replay stable.
- Capacity Gate: NOT RUN -- requires measured device rates + a confirmed real trace
  (S8 Gate A). Symbolic-rate sweep is mechanics only.
- Energy Gate: NOT RUN -- DISABLED; needs synchronized physical instrumentation.
- S8 Gate A dependency: NOT independently confirmed here; S9-V0 uses only synthetic
  fixtures and makes no real-trace claim.

## Smallest next implementation slice

Implement S9-V1 in the research simulator and schemas only:

1. Add versioned per-device H2D/D2H link profiles, evidence IDs, path kind,
   contention-domain IDs, and directional domain caps.
2. Allocate bandwidth dynamically among active streams in the same domain; do not
   divide one global controller by the number of configured phones.
3. Add D2H `output_bytes` transfer before request completion/resource release.
4. Add a staged-file mode that does not blindly charge UFS twice, plus a measured
   two-phone ADB scenario bound to the evidence in `CURRENT_SLOW_LINK.md`.
5. Measure native buffer transport, UFS, durable fsync/publish, hash, materialize,
   prepare, warmup, and transfer/compute interference as separate legs before a
   decomposed capacity run.

Keep 40-550 MiB/s as broad V0 sensitivity. Optional 800/1000 MiB/s points represent
an unmeasured future USB 10 Gbps device class, not OP12/OP15. Do not build a daemon
or production scheduler until the V1 model tests pass and S8 Gate A is confirmed.

## Git integrity + preserved pre-existing changes

- HEAD unchanged: 933c722f6. Nothing staged, committed, or pushed.
- The only tracked diff from this plan update is `research_dev/talks.md`;
  `MIXED_WORKLOAD_DESIGN.md`, `NEXT_PLAN.md`, and all S9 content are already
  untracked research files in this dirty worktree.
- The pre-existing dirty worktree (S6/S7 changes to examples/layersplit/*,
  ggml/src/ggml-hexagon/*, ggml/src/ggml-hexagon/htp/*, tests/test-backend-ops.cpp,
  and the other untracked research docs) is preserved; this update changed only the
  S9/link plan documents named above.
