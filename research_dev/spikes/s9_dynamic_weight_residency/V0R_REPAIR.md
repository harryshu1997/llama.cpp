# S9-V0-R: contract and simulator closure

This round repairs the fail-OPEN contracts and simulator mechanics the review found in
S9-V0, under a versioned v2 schema bundle, and proves each repair with a red-before /
green-after regression. The pre-repair V0 is frozen for provenance; nothing here is a
capacity or energy claim.

## Status relabel

- S9-V0 (frozen): `SUITES PASS; MECHANICS NOT YET CERTIFIED; CAPACITY UNPROVEN`.
- S9-V0-R (this round): `SUITES PASS; MECHANICS CERTIFIED (repaired + adversarially
  regression-tested); CAPACITY UNPROVEN`.

Capacity stays UNPROVEN: device pipeline rates are symbolic, and the directional
H2D/D2H link decomposition is deferred to S9-V1 (see "Remaining S9-V1 blockers").

## Golden hashes

| artifact | hash |
|---|---|
| V0 frozen code (residency_sim_v0.py) | `sha256:4491827b4d77fc6d513bf7608e90a293dbc518325c3b48eb18253288dd615313` |
| V0 frozen replay (baseline_sweep)    | `sha256:64fcad3b400024e9c3b87a748f1219c65477a9c8106e45ae670a2d3e28e59d1e` |
| V0-R code (residency_sim.py)         | `sha256:6393e227831e0632addc6716d54b79606a632f0f37f0a49895aa6486ebc9b7eb` |
| V0-R golden replay (baseline_sweep)  | `sha256:af1501b6b766a7a70afa41460de87b379bf47b7bf133017799bb13058b840339` |

`golden/v0r/baseline_sweep.v0r.manifest.json` is the stored golden;
`sim/test_golden_replay.py` recomputes the run in a SEPARATE process and asserts
byte-identity against it, and a one-integer mutation of the stored file makes that test
fail.

## Findings repaired (Checkpoint 2, record model + oracle)

Repaired records live under `schemas/v2/` at `schema_version: 2`; the v1 schemas stay
frozen under `schemas/`. Digests come from one shared module, `s9lib.py`, so the
fixture generator and the validator cannot drift.

1. Canonical allocation vs prepared-image ownership -- NEW `canonical_allocation.schema.json`.
   One byte-charge per (device, weight-set, generation); backend images ALIAS it
   (`alias_refcount == len(alias_prepared_image_ids)`); not reclaimable while any
   alias or lease holds it (`alias_refcount>=1 OR lease_refcount>=1 => reclaimable=false`).
2. Prepared-image identity -- `schemas/v2/prepared_image.schema.json` + `s9lib.prepared_image_digest`.
   `derived_image_digest` now binds backend, image_class, preparation_algorithm,
   backend_build, layout, source_allocation_id, source_weight_set_digest, derived_bytes,
   sharing_mode, boot/generation, AND derived_payload_sha256. Relabel or payload
   corruption breaks the digest.
3. Exact dispatch tuples -- `schemas/v2/dispatch_decision.schema.json` + bundle validator.
   Requirement is a tuple set `(weight_set, prepared_image, ready_certificate,
   residency_lease)`; `verdict==DISPATCH` requires `required_tuples == satisfied_tuples`
   exactly (E_TUPLE_MISMATCH otherwise).
4. Domain-separated digests -- `s9lib.domain_digest` prefixes `s9:<kind>:v2` so an
   island digest, lease digest, allocation digest, correctness digest, etc. live in
   disjoint pre-image spaces.
5. Partial-range validation -- `schemas/v2/model_manifest.schema.json` (`coverage_policy`)
   + `bundle_validate._check_ranges`: `start<end<=n_layer_total`, shared n_layer_total,
   unique ids, per-set total_bytes/set_digest match the WeightSet record, and an
   explicit contiguous_partition / sharded_disjoint / overlap_allowed gap-overlap policy.
6. One strict bundle validator -- `bundle_validate.py` ALWAYS runs JSON Schema + semantic
   + cross-record and rejects with 21 stable error codes: unknown kinds (E_UNKNOWN_KIND),
   unknown versions (E_UNKNOWN_VERSION), duplicate JSON keys (E_DUPLICATE_KEY), missing
   records (E_MISSING_RECORD), mismatched digests (E_DIGEST_MISMATCH), and the rest.
7. Correctness binding -- NEW `correctness_certificate.schema.json`: `correctness_digest`
   binds island, kernel route, backend build, profile-row digest, and the exact shape
   envelope; the island/ready-cert reference it by id+digest.
8. Bulk-frame + resume binding -- `schemas/v2/transport_frame.schema.json` binds a bulk
   frame to ticket/segment/chunk/offset/length/digest; `transfer_ticket.schema.json`
   requires a verified prefix digest for any nonzero resume; EXECUTE/RESULT frames must
   carry payload_sha256 and all strings are length-bounded.

## Findings repaired (Checkpoint 3, simulator mechanics)

`sim/residency_sim.py` was rewritten (frozen V0 kept at
`golden/v0_historical/residency_sim_v0.py`).

1. Epoch binding -- every pipeline stage carries (pipeline_ver, generation, boot);
   every dispatch carries (boot, generation, route, state). A stale stage is dropped
   (`stale_pipeline_drops`); a stale completion is rejected fail-closed
   (`stale_completion_rejects`) with its bytes rolled back. DRAINING rejects new work.
2. Horizon + bounded queues + terminal outcomes -- `horizon_us` enforced; bounded
   prefetch queue (drops logged), server queue (rejects overflow), and HTP/GPU lanes.
   Every request ends in exactly one of completed_phone / completed_server / fallback /
   rejected / timed_out (asserted: `sum(buckets) == requests`).
3. Real multi-device/backend selection over per-domain links -- candidates enumerated
   over (device, backend); links contend per `contention_domain` (separate USB buses are
   additive), replacing V0's "divide one controller by phone count." static /
   fastest_ready / relief_predictive / clairvoyant have distinct, tested behavior.
4. Byte reserve + rollback -- UFS on RECEIVING, LPDDR-canonical on MATERIALIZING, derived
   on PREPARING, scratch on publish, activation+state on dispatch; any failure rolls the
   pipeline back (`oversub_rollbacks`). Sticky state persists until an explicit reset.
5. Resume -- re-sends only the remaining verified range while RETAINING link ownership,
   counts retry bytes, and stays activation-preemptible.
6. Objective -- the dimensionally-invalid V0 causal score is replaced by a documented
   LEXICOGRAPHIC objective over integer keys (SLO-feasible, server relief us, -cold cost
   us, reuse permille, -eviction bytes, -interference risk, device order); relief and
   cold-cost are also summed separately, both in microseconds.
7. Interference -- applied ONLY over the actual resource-overlap window, and only when
   `interference.measured` is true.
8. D2H -- an EXPLICIT result-return blocker on the link; a phone request is not complete
   until D2H finishes. The directional H2D/D2H rate split is the S9-V1 blocker.

## Red-before / green-after evidence

`sim/test_regressions.py` imports BOTH the frozen V0 module and the repaired module and,
per repair, shows V0 exhibits the bug and V0-R is correct on the same scenario
(14/14 PASS):

| # | scenario | RED (frozen V0) | GREEN (V0-R) |
|---|---|---|---|
| 1 | post-stale completion | completes stale work | rejected, bytes rolled back |
| 2 | stale loading pipeline | publishes READY under bumped epoch | stale_pipeline_drops>=1 |
| 3 | two required sets, one cert | v1 schema accepts | E_TUPLE_MISMATCH |
| 4 | prepared-image relabel | v1 digest blind to image_class | E_DIGEST_MISMATCH |
| 5 | per-domain link | halves goodput per phone | full goodput per domain |
| 6 | impossible RAM/HBM | request silently lost | rejected (explicit terminal) |
| 7 | horizon | completes past horizon | timed_out |
| 8 | sticky state | freed at completion | persists until reset; no live-state eviction |
| 9 | LPDDR oversubscription | no mid-pipeline rollback | oversub_rollbacks>=1, ledger valid |
| 10 | partial resume | resend not counted | retry_bytes == canonical/4 |
| 11 | two-phone placement | uses one device only | dispatches to both phones |
| 12 | full epoch stack | no route/state epoch | rejects on route change |
| 13 | unknown validator kind | no kind gate | E_UNKNOWN_KIND |
| 14 | golden replay mutation | two-run determinism blind to stored file | mutation caught vs artifact |

## Suite counts (all green)

Run from `research_dev/spikes/s9_dynamic_weight_residency/`:

```
python3 run_schema_tests.py           # 86 fixtures (53 v1 + 33 v2), both validators, 0 fail
python3 bundle_validate.py --selftest # 15 bundle fixtures (1 valid + 14 adversarial), 0 fail
python3 validate_manifests.py --selftest   # 23 v1 semantic, 0 fail
python3 sim/test_residency_sim.py     # 26 checks (10 required + V0-R additions), 0 fail
python3 sim/test_regressions.py       # 14 red-before/green-after, 0 fail
python3 sim/test_golden_replay.py     # 6 (cross-process golden + mutation), 0 fail
```

17 v2 schemas, 21 stable bundle error codes. V0-R golden replay
`af1501b6...`, stable across separate processes.

## Remaining S9-V1 blockers

1. Directional link model: split H2D and D2H into versioned, evidence-bound per-device
   goodput profiles with per-domain directional caps; the V0-R sim uses one symbolic
   rate both directions (D2H is present as an explicit blocker but not yet decomposed).
2. Measured decomposition: native buffer transport, UFS write, durable fsync/publish,
   hash/verify, materialize, prepare, warmup, and a measured transfer<->compute and
   HTP<->GPU interference matrix (interference stays at 1000 permille until measured).
3. Confirmed real traces (S8 Gate A) before any capacity or energy number; V0-R uses
   synthetic fixtures only.
4. A staged-file transport mode that does not double-charge the UFS write already folded
   into the measured ADB push rate (see CURRENT_SLOW_LINK.md).

Do not build a daemon, add the directional USB model, or claim capacity/energy until
these pass. Nothing in this round was committed or pushed.
