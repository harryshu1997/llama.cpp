# S10-V0-R-E1 typed evidence results

## VERDICT: TYPED_EVIDENCE_INTEGRITY_PASS_PHYSICAL_CLAIMS_BLOCKED

This verdict applies to the repaired E1 implementation, not the worker's first
green report. A follow-up review reproduced that report and then broke the live
path through schema bypass, absent artifact files, disjoint KV envelopes,
phantom no-fallback evidence, selectable boundary records, and additive reuse of
whole-wall power. Those paths are now closed and regression-tested.

The mechanics evidence contract and deterministic binder are in place. The
measured atlas is still EMPTY by the on-disk audit: zero eligible rows exist,
and phone energy is physically unmeasurable. In addition, E1 now rejects every
`MEASURED` instance because its additive solver cannot faithfully consume
aggregate server-wall or total-wall timelines. Physical relief needs a separate
typed matched control/treatment record.

This does NOT authorize C0-C5. It does not authorize measurements, PF1, or
runtime work. Nothing was committed or pushed.

The temporal scheduling and energy arithmetic remain unchanged and green. The
standalone oracle and checker now share a dependency-free live gate for the
checked-in instance schema; valid-input optima and differential results are
unchanged.

A final adversarial pass closed three additional defects: `metric=EXACT` now
requires zero observed error and zero threshold, thermal evidence is bound to the
route's backend binary, and schema-invalid input cannot launch the independent
optimality enumeration. The first two previously produced a valid certificate;
the third was fail-closed but wasted bounded search time.

## Repair addendum

The original 129-test suite was green but did not exercise the same schemas or
artifact bytes in the live binder. The repaired path now:

- runs all three draft-2020 schemas before semantic field access and refuses if
  the schema engine is unavailable;
- verifies every artifact's real bytes under a trusted root, including synthetic
  fixtures;
- carries workload `kv_tokens`, requires route applicability at the exact
  token/KV point, and requires correctness to cover the route envelope;
- links no-fallback proofs to an existing artifact listed by the certificate;
- pins each RouteProfile to one BoundaryProfile and requires end-to-end latency;
- separates H2D, D2H, and produced-output bytes, and checks batch-profile output
  geometry, direction, and omitted energy across every legal merged subset;
- rejects SERVER_WALL and TOTAL_WALL records as additive per-device solver inputs, uses typed
  instrument capabilities, and requires globally qualified disjoint rail ids;
- schema-validates certificates before indexing them, including `search.complete`;
- isolates the schema worker from `PYTHONPATH` injection;
- makes relief comparison inputs and shell negative-test setup fail closed;
- rejects target/reference payload aliases, non-server capacity probes, stale
  route/correctness/thermal revision sets, zero-time `NONE` transport carrying
  bytes, and duplicate node IDs before projection; and
- applies the checked-in foundation instance schema in both standalone executable
  paths, including canonical profile keys and numeric bounds.

The v2 solver's scheduling and accounting logic was not changed. The oracle and
checker entry paths were edited only to add the shared live schema gate.

## CP0 - baseline reproduced

```
bash scripts/run_tests.sh          # exit 0
  -> Ran 26 foundation tests ... OK
  -> S10_V0R_TEMPORAL_FOUNDATION_TESTS_PASS
  -> differential: 1187 compared, 13 jointly infeasible, 0 skipped, 0 mismatches
```

Matches the handoff exactly. One correction recorded rather than repeated: the
earlier claim of a recursive hash over "101 files" of the historical tree was made
without a persisted recipe, and 3 of those 101 were `__pycache__` bytecode. The
recipe is now persisted and self-checking (see Integrity).

## CP1/CP2 - the frozen contract

`EVIDENCE_CONTRACT.md`, schema version 3. Seven record types
(ArtifactDescriptor, CorrectnessCertificate, RouteProfile, BoundaryProfile,
ThermalInterferenceProfile, PowerProfile, InstanceEvidenceBinding), bounded
strings/arrays/integers, `additionalProperties: false` throughout, status one of
PASS/FAIL/UNKNOWN/INELIGIBLE with a stable reason code, and 31 stable `E_*`
diagnostics.

Energy boundaries are frozen and never equated, but E1 emits no physical relief
label. `SERVER_WALL` and `TOTAL_WALL` timelines are aggregate and cannot enter
the per-device solver; `GPU_BOARD` is incomplete. Every `MEASURED` instance is
therefore refused. `GPU_BOARD_RELIEF`, `SERVER_RELIEF`, and
`SYSTEM_ENERGY_SAVING` need a later matched comparison record. The equations
`delta_E_server = E_server_control - E_server_qpim` and
`phone_plus_external_break_even_budget = delta_E_server` are frozen and
deliberately NOT measured. Only a complete `SERVER_WALL` delta can define that
budget. A GPU-board delta describes that board alone and cannot establish a
server-wide or phone-energy budget.

The load-bearing structural rule: **required bindings are computed from the
INSTANCE, never from the binding list**, so dropping a binding is
`E_BINDING_MISSING`, not a skipped check.

## CP3 - binder and validator

`evidence/{canon,validator,binder,boundary}.py`. One canonical-JSON and SHA-256
implementation, proven byte-identical to the frozen oracle's and checker's on a
corpus rather than asserted in prose.

The v2 scheduling and accounting mechanics are unchanged. A v3 instance is a v2
core plus an evidence binding block; the binder projects it to the v2 core IN
MEMORY ONLY to reuse the proven solver. The projection is never the signed identity: the
certificate binds `instance_sha256` of the FULL v3 instance plus
`bundle_sha256`/`binding_sha256`, so a swapped projection cannot validate (tested:
naming the projection digest is rejected with an explicit "projected" diagnostic).

**The evidence-bound path reproduces both frozen temporal optima**, so the
contract demonstrably does not disturb the proven mechanics:

| fixture | via evidence | expected | claim |
|---|---|---|---|
| `transition_v3` | `[0,0,-2,147250000]`, `server_p0_intervals [[800,1150]]` | 147250000 | `NONE_MECHANICS_ONLY` |
| `activation_v3` | `[0,0,-2,2974000]`, peak 100 | 2974000 | `NONE_MECHANICS_ONLY` |

## CP4 - the atlas is empty

`EVIDENCE_MATRIX.md`. Two independent read-only sweeps of S6, S8, S9, historical
S10, `research_dev/energy/`, and the untracked `scratchpad/` trees, reconciled
against the frozen gates. **Zero eligible rows.** No PowerProfile of any device
passes, so no instance can be `MEASURED`, so no energy claim of any kind is
authorized.

The most instructive row: the A6000 board power trace is a real, reproducible
measurement, and the contract refuses it TWICE - `GPU_BOARD` scope (NVML cannot
see wall power), and ~57 independent samples in 32.3 s (the sensor moves at
~1.77 Hz against a 10 Hz poll) against `MIN_POWER_SAMPLES=100`.

The audit also found published claims contradicted by their own cited artifacts.
Recorded in EVIDENCE_MATRIX.md section 4, none of it in this spike's scope to fix:
S6's "xmem GEMM confirmed" is refuted by the very CSV it cites (0 hits for
`xmem`/`os8`/`prepack`; the kernel is the stock `l4_lm`) - independently verified,
not taken on report; S6's "HMX at every M>=5" exists on disk only at M=8; the
S6-L ffnmerge 16+48 row splices a speedup from a run with no correctness onto a
correctness value from different shapes; and the adb-push "262 MiB/s OP15 /
216 OP12" figures are device-swapped and ~3x high against the only artifact
(OP12 249, OP15 86). A memory asserting the S6 kernels were "CERTIFIED" has been
corrected.

## CP5 - adversarial verification

201 unit tests (28 foundation + 173 evidence), exit 0. 173 evidence tests across
14 classes, the great majority adversarial - far above the required 30 mutations:

```
DigestAndSwapMutations 19   IdentityRebindMutations 11   EligibilityMutations 16
DerivedValueMutations  12   BoundarySemantics       18   MalformedInput       16
EvidenceRepairRegressions 36 RedTeamRegressions     18   ProjectionSwap        5
FrozenEquations         9
EnumerationIsExhaustive 4   ValidBundle              4   FrozenOptima          3
ProjectionCarriesNoUnboundNumber 2
```

Plus 39 negative evidence CLI cases (several parameterised),
each exiting nonzero with a stable `EVIDENCE_FAIL`/`BIND_FAIL` prefix and no
traceback, and 18 foundation CLI negatives.

### The first audit found ten holes. All were real; all are closed.

An independent red team was given the contract and told to break it. It did, in
two independent classes I had missed entirely. Every one of these PASSED
validation before the audit, and each now has its own regression test:

| # | hole | effect before the fix |
|---|---|---|
| F1 | **float/bool type confusion at every eligibility gate** | a route certified at 90% error against a 0.5% threshold bound and solved cleanly |
| F2 | `classify_energy_claim` ignored BoundaryProfile scope | `SYSTEM_ENERGY_SAVING` certified with 500000 nJ of GPU-board energy inside `total_nj` |
| F3 | BoundaryProfile had no identity fields at all | any boundary record backed any node; a PCIE/GPU record backed a phone route |
| F4 | `E_DOUBLE_COUNT` watched only `devices.SERVER.active_mw` | a twin record hid `USB_VBUS`; phone counted twice |
| F5 | per-field record selection | a server with wake=50 and transition=0 that no record described; 147250000 -> 146250000 |
| F6 | thermal state evidenced but never constrained | COLD n0 + STEADY n1 in one schedule; 147250000 -> 122500000 |
| F7 | `activation_mem_bound_bytes` bound any PASS route | activation limit erased by an OP15 route's footprint |
| F8 | `MAX_INT` never enforced on a live path | `duration_us = 2**60` validated |
| F9 | absent record field resolved to `None` | a binding with `value: null` validated by `None == None` |
| F10 | internal errors escaped as tracebacks | a crash is not a rejection |

**F1 is the one worth reading twice.** The guards were written
`if is_int(a) and is_int(b) and <bad>`, so a wrong type SKIPPED the check rather
than failing it - at six sites. Since `900000.0 == 900000`, a float then satisfied
every binding equality too. Worse, my own `test_bool_is_not_an_integer` asserted
`is_int(True) is False` as though that were a rejection, when it is precisely what
guaranteed the bypass. And JSON Schema cannot help: draft6+ defines `integer` as
any number with a zero fractional part, so `900000.0` validates as an integer -
now proven executably against `/usr/bin/jsonschema` rather than argued.

Fixes: one type gate (`canon.check_integers`) applied once at load before any
comparison; every guard inverted to fail closed on a wrong type; record coherence
(`E_INCOHERENT`: one device, one power record; one thermal condition; one build);
boundary identity + direction; boundary scope folded into claim classification;
`canon.MISSING` instead of `None`; and `validate_safe` so an internal error is a
refusal, not a traceback.

The root cause worth naming: the contract froze the STATISTIC to stop
cherry-picking, then left RECORD selection free - so cherry-picking returned one
level up, across records, thermal states, builds, and rail sets. Section 5b of the
contract now closes that class.

What the audit could NOT break, which is worth as much: `required_targets()`
enumeration is complete. It was verified empirically - perturbing every integer
leaf of both projected cores and re-solving - not by hand-audit. The only
solver-visible unenumerated fields are exactly the declared workload set. A
structural guard (`EnumerationIsExhaustive`) now walks every integer in the
instance and fails if one is neither evidence-derived nor explicitly declared
workload, and it includes a test proving the guard itself can fail.

### Determinism

Bundle, binding, and certificate digests are identical across 5 separate
processes at `PYTHONHASHSEED` 0/1/42/12345/random. Fixtures are a deterministic
function of their generator (`make_evidence_fixtures.py --check`, also run under a
foreign hash seed).

### Post-handoff repair

A second review of the worker's reported PASS found more executable holes: the
live binder did not run the JSON Schemas, artifact descriptors were never checked
against real files, decode KV shape was absent, no-fallback and CPU-reference
artifacts could be phantom or reused, route and boundary records could disagree
on revisions, bytes, or wall time, certificates were indexed before schema
validation, aggregate wall power and uncertainty could enter the additive model,
and the schema subprocess trusted hostile `PYTHONPATH`. A final batch-specific
reproduction selected a 1 us merged action whose boundary claimed zero output and
whose energy term the solver would have dropped.

All reproduced paths now fail closed. The batch validator enumerates every legal
SERVER subset for a batch profile, requires one aggregate output geometry, pins
that geometry to the route boundary, requires a server-compatible direction, and
refuses nonzero batch-boundary energy until the solver models it. An independent
rerun reproduced the hostile schema and malformed-batch exploits before the fix
and verified their rejection afterward. This is evidence for the tested attack
surface, not a claim that static validation proves arbitrary live dispatch safety.

The final repair also closes schema-module shadowing when hostile `PYTHONPATH`
already contains the real schema directory, rejects zero-duration thermal
profiles in both schema and runtime gates, and binds artifact validity intervals
to the signed bundle evaluation epoch.

## Commands and counts

```
bash scripts/run_tests.sh                       # exit 0
  -> Ran 201 tests ... OK                       (28 foundation + 173 evidence)
  -> S10_V0R_TEMPORAL_FOUNDATION_TESTS_PASS     (foundation still green)
  -> S10_V0R_TYPED_EVIDENCE_TESTS_PASS
  -> S10_V0R_EVIDENCE_FIXTURES_OK               (deterministic regeneration)
  -> S10_V0R_EVIDENCE_NEGATIVE_PASS (39 cases)
  -> S10_V0R_CLI_NEGATIVE_PASS      (18 cases)
  -> differential: 1187 compared, 0 mismatches, digests IDENTICAL to CP0
  -> evidence determinism: identical across 5 processes / hash seeds
git diff --check                                # clean
ASCII scan of the repair directory              # all ASCII
sha256sum -c HISTORICAL_MANIFEST.txt            # 98/98 OK
```

## Changed and added files

Added: `EVIDENCE_CONTRACT.md`, `EVIDENCE_MATRIX.md`, `RESULTS_E1.md`,
`HISTORICAL_MANIFEST.txt`,
`evidence/{canon,validator,binder,boundary,schema_gate}.py`,
`schemas/{evidence_bundle.v3,instance.v3,certificate.v3}.schema.json`,
`fixtures/evidence/{mechanics_bundle,transition_v3,activation_v3}.json` plus seven
materialized synthetic artifact payloads,
`tests/{test_evidence,make_evidence_fixtures}.py`, `scripts/evidence_negative.sh`.

Changed: `scripts/run_tests.sh` (evidence gate plus one fail-closed temporary
workspace), `PLAN.md` (status + gate checkbox), `oracle/exact.py`,
`checker/checker.py`, and `tests/test_foundation.py` (live schema-gate repairs and
regressions).

**Unchanged and verified byte-identical**: every v2 mechanics file -
`schemas/{instance,certificate}.schema.json`, all three `fixtures/*.json`,
`checker/reference.py`, `policies/causal.py`,
`tests/{slow_reference,gen_cases,differential,make_suboptimal}.py`,
`scripts/cli_negative.sh`.

## Deviations and honest limits

- **ajv draft-2020 is not installed on this host.** Only `/usr/bin/jsonschema`
  runs. Not installed, because that is outside this checkpoint. Recorded, not
  papered over. It would not have caught F1 anyway - that is now proven, not
  assumed.
- **The contract binds numbers to hashed artifacts. It cannot detect a fabricated
  artifact.** A record that lies about the conditions it was measured under, with
  an artifact that agrees, is accepted. What the gate provides is that the lie is
  recorded, digest-pinned, and attributable - not that it is impossible. The
  audit's F6 fix closes MIXING two honestly-different records; it does not close
  one dishonest record. Artifact lifetimes are checked against a signed bundle
  evaluation epoch, including future, expired, zero-validity, and invalid-calendar
  cases. That epoch is an attributable as-of assertion, not independent proof of
  wall-clock recency.
- `activation_mem_bound_bytes` is a device capacity but binds a designated
  `CAPACITY_PROBE` RouteProfile's `memory_bytes`, because no DeviceCapacityProfile
  record type exists yet. The selection is now enforced (F7); the record type is
  still a proxy. This matters: it is the one field where a LARGER value makes a
  schedule more feasible.
- `horizon_us` is workload-declared but multiplies server idle energy, so the
  ABSOLUTE energy in a certificate is not fully evidence-derived. Differences at
  equal horizon are. Found by the audit; recorded rather than hidden.
- The only valid bundle is fully synthetic. **No measured row exists to bind.**
  Even if the atlas gains rows, E1 deliberately refuses every MEASURED instance
  until a typed matched wall-timeline comparison replaces additive wall inputs.
- `MIN_PROCESSES=3`, `MIN_SAMPLES=30`, `MIN_POWER_SAMPLES=100` are judgement
  calls, frozen before the atlas was read so they could not be tuned to admit a
  favoured row. They were not lowered when the atlas came back empty.
- The historical tree hash is claimed only under the persisted recipe in
  `HISTORICAL_MANIFEST.txt`, excluding `*.pyc`.

## Still not implemented or run

- Any measurement. No phones, no A6000, no energy, no trace replay.
- C0-C5, PF1, CP-SAT, a causal policy, a live scheduler, residency decisions.
- A DeviceCapacityProfile record type; a measured second service route.

## Integrity

- HEAD `933c722f6`, unchanged. No reset, checkout, revert, stage, commit, push, PR.
- Historical `../s10_power_frontier/` byte-for-byte unchanged under a PERSISTED,
  self-checking recipe: `find ... -type f ! -name '*.pyc' | sort | xargs sha256sum`
  over 98 files -> `HISTORICAL_MANIFEST.txt`, whose own digest is
  `39d49c68a07e07eb0275fbd3104339634bde2ca397ba86a1cbf21c0321daa70f` before and
  after. Verify with `sha256sum -c HISTORICAL_MANIFEST.txt`.
- No edits to tools/server, llama-server, model graphs, KV internals,
  ggml_backend_sched, the phone-PIM protocol/runtime, or any HTP/OpenCL/CUDA
  kernel.
- Pre-existing S6-S10 dirty files preserved; the only tracked file touched is
  `research_dev/talks.md`.
- ASCII only; generated bytecode removed.

Stop for human review.
