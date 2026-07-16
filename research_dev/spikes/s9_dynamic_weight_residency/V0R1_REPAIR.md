# S9-V0-R1: fail-closed closure

> Review update (2026-07-14): the DRAINING-lease probe now returns E_CHAIN_BROKEN, and two
> additional cross-record gaps found during review are closed: a StateLease must depend on a
> dispatched residency tuple, and a ReadyCertificate must use the dispatched island's
> correctness certificate. Six further simulator defects are repaired with per-event
> conservation checks. `sim/test_r1_holes.py` now has 16 checks and the v3 bundle index has
> 18 fixtures. This is still NOT a completeness proof. Eleven new v3 mutation probes remain
> fail-open across authoritative DeviceInventory/expiry gates, executable identity,
> transport epochs, and state-to-ledger binding. R1 status therefore stays honest:
> `SUITES PASS; TARGETED + ADVERSARIALLY-FOUND CASES CLOSED; SCHEDULER DISPATCH CERTIFICATION
> NOT CLAIMED COMPLETE; CAPACITY UNPROVEN`.

An independent adversarial pass found that the official V0-R suites PASSED while the
mechanics were still fail-OPEN. This round relabels V0-R, freezes it as RED evidence,
introduces a versioned R1/v3 contract, and closes all sixteen fail-open cases with
red-before / green-after evidence. These are the 16 targeted cases, not complete
scheduler certification; the later DRAINING-lease blocker is recorded above. No
capacity or energy claim; nothing committed.

## Status relabel

- S9-V0 (frozen): `SUITES PASS; MECHANICS NOT YET CERTIFIED; CAPACITY UNPROVEN`.
- S9-V0-R (frozen, RELABELED): `SUITES PASS; MECHANICS CERTIFICATION BLOCKED; CAPACITY UNPROVEN`.
- S9-V0-R1 (this round): `SUITES PASS; 16 TARGETED + 10 ADVERSARIALLY-FOUND FAIL-OPEN CASES
  CLOSED (red/green + a fresh independent adversarial pass); SCHEDULER DISPATCH CERTIFICATION
  NOT CLAIMED COMPLETE; CAPACITY UNPROVEN`.

Capacity stays UNPROVEN: device rates are symbolic and the directional H2D/D2H link
model is still S9-V1.

## Versioning + golden hashes

The v1 (frozen) and v2 (frozen) schema bundles are untouched. The repaired records are a
NEW v3 bundle (`schemas/v3/`, schema_version 3; record shapes identical to v2, the R1
teeth are in the validator + simulator). The pre-R1 code is frozen under
`golden/v0r_historical/`.

| artifact | hash |
|---|---|
| V0 golden replay (preserved) | `sha256:64fcad3b...` |
| V0-R golden replay (preserved, frozen sim reproduces it) | `sha256:af1501b6...` |
| frozen V0-R sim (residency_sim_v0r.py) | `sha256:6393e227...` |
| frozen V0-R validator (bundle_validate_v0r.py) | `sha256:b763c384...` |
| reviewed R1 simulator (residency_sim.py) | `sha256:f10d79e4...` |
| reviewed R1 validator (bundle_validate.py) | `sha256:11e90b68...` |
| R1 golden replay (new) | `sha256:9a69a7f6...` |

`golden/v0r1/baseline_sweep.v0r1.manifest.json` is the R1 golden;
`sim/test_golden_replay.py` reproduces it cross-process, catches a one-integer mutation,
AND re-verifies that the frozen V0-R sim on its frozen config snapshot still reproduces
`af1501b6...` and that the V0 golden file stays self-consistent.

## Contract repairs (R1 validator, bundle_version 3 in bundle_validate.py)

The R1 validator uses PRIVATE per-call temp files (`tempfile.mkstemp`) so concurrent
validators cannot race, and its `cross_record_r1` derives ONE coherent chain and adds 5
stable error codes (26 total: `E_ISLAND_ABSENT`, `E_CHAIN_BROKEN`, `E_DISPATCH_MISMATCH`,
`E_STATE_LEASE`, `E_ALIAS_SET`):

- the dispatch island MUST exist;
- each satisfied tuple must form `WeightSet -> CanonicalAllocation -> PreparedImage ->
  ReadyCertificate -> ResidencyLease`, matched on device, backend, boot epoch, residency
  generation, ids, and digests;
- DispatchDecision device/backend/route-epoch/tuple-coverage must match the island;
- a sticky/rebuildable island requires a StateLease matching request/island/device/
  backend/route_epoch/state_policy; a stateless island requires a null StateLease;
- alias and live-lease sets are DERIVED from the bundle records and must equal the
  allocation's declared sets and refcounts exactly, and `reclaimable` must follow from
  those records.

bundle_version 2 stays byte-identical (the v2 fixtures pass/fail exactly as before).

## Simulator repairs (R1 sim, sim/residency_sim.py)

1. readiness is backend/generation-qualified (`is_ready(dev, ws, back)`); dispatch to a
   wrong-backend residency is refused.
2. `stale_epoch` sets the device DRAINING and non-dispatchable.
3. a stale pipeline stage rolls back EXACTLY ONCE (`stale_cleaned`), releases its
   contention domain, and wakes the bounded prefetch queue (`domain_releases_on_stale`).
4. the full epoch stack is rechecked after D2H before the output is accepted
   (`post_d2h_epoch_rejects`).
5. mutable state is stored per request/session (`sticky_sessions`); reset is DEFERRED
   while the session is pinned (`reset_deferred`) and executed at unpin.
6. HTP/GPU lane admission is bounded (`lane_depth`); overflow -> explicit rejected
   terminal (`lane_rejects`).
7. event processing STOPS at the horizon (no unbounded polling).
8. a partial failure carries an explicit verified offset; failures at different offsets
   produce different retry bytes and completion times.
9. `link_drop` is resumable in the bulk phase (`link_drop_resumed`) and terminal
   fail-closed in the activation/compute/d2h phases (`link_drop_terminal`).
10. server relief is credited ONLY after a valid successful phone completion (in
    `d2h_done`, not at dispatch).
11. measured interference is REJECTED as unsupported (`MeasuredInterferenceUnsupported`)
    rather than emitting asymmetric, misleading timing.

Every request still reaches exactly one terminal outcome and the run asserts
`completed_phone + completed_server + fallback + rejected + timed_out == arrivals`.

## Red-before / green-after evidence (sim/test_r1_regressions.py, 17/17)

Each imports BOTH the frozen V0-R code and the R1 code and shows V0-R exhibits the hole
and R1 closes it, asserting actual state/bytes/ownership/terminals (not counters alone):

| # | case | RED (frozen V0-R) | GREEN (R1) |
|---|---|---|---|
| 1 | wrong device/backend/request/route | validator accepts | E_CHAIN_BROKEN / E_DISPATCH_MISMATCH / E_STATE_LEASE |
| 2 | null/foreign StateLease | validator accepts | E_STATE_LEASE |
| 3 | incoherent tuple (records exist) | validator accepts | E_CHAIN_BROKEN |
| 4 | nonexistent island | validator accepts | E_ISLAND_ABSENT |
| 5 | PI missing from alias set | validator accepts | E_ALIAS_SET |
| 6 | reclaimable while referenced | validator accepts | E_ALIAS_SET |
| 7 | parallel validators, shared /tmp | 6 wrong verdicts under load | 0 wrong (private temp files) |
| 8 | stale RECEIVING | domain leaked, queue stuck | rolled back, domain released, queue woken |
| 9 | request after stale_epoch | re-prefetch onto stale device | device DRAINING, falls back |
| 10 | epoch change during D2H | stale output accepted | rejected (post_d2h_epoch_rejects) |
| 11 | HTP residency to GPU | dispatched | refused (both directions) |
| 12 | two sticky sessions | collapse to one scalar (4096) | {s1:4096, s2:8192} |
| 13 | reset while pinned | reset lost | deferred, executed at unpin, state->0 |
| 14 | lane overflow | unbounded, all complete | bounded, 1 rejected terminal |
| 15 | multi-offset partial | single canonical/4 resume | exact per-offset retry (300MB+100MB) |
| 16 | link_drop | ignored (pass) | resumable(bulk) / terminal(act,compute,d2h) |

## Suite counts (all green)

```
python3 run_schema_tests.py            # 100 fixtures (53 v1 + 33 v2 + 14 v3), both validators, 0 fail
python3 bundle_validate.py --selftest  # v2 (15) + v3 (18) bundle fixtures, 0 fail
python3 validate_manifests.py --selftest    # 23 v1 semantic, 0 fail
python3 sim/test_residency_sim.py      # 26 FROZEN V0-R behavior checks (preserved), 0 fail
python3 sim/test_r1_regressions.py     # 16 red/green regressions + 1 current invariant, 0 fail
python3 sim/test_r1_holes.py           # 10 reference/mutant closures + 6 current checks, 0 fail
python3 sim/test_golden_replay.py      # 13 replay/provenance checks, 0 fail
```

17 v3 schemas, 26 stable bundle error codes.

## Independent adversarial re-verification (second pass)

The first R1 round's "16 closed" claim was PREMATURE. An 8-agent adversarial workflow
(each agent building and running concrete repros against the LIVE R1 code, then a skeptical
verify pass) found **9 confirmed surviving fail-open holes** (2 candidates correctly refuted
as not-bugs), and a review dispatch probe found a **10th** (a DRAINING ResidencyLease still
accepted for DISPATCH). All ten are now closed with regression evidence
(`sim/test_r1_holes.py` 8/8 + 5 new v3 fixtures); they dedup to 4 validator + 3 simulator fixes:

Validator (`bundle_validate.py cross_record_r1`):
- island `required_weight_set_digests` / `required_prepared_image_ids` / `..._digests` were
  bound into the signed island_digest but NEVER checked against the dispatched tuple, so a
  signed island could be served against phantom / different-content weight sets or prepared
  images. Now the dispatched tuples must cover the island's declared digest sets exactly
  (E_DISPATCH_MISMATCH).
- a ResidencyLease's weight set was not matched to its named CanonicalAllocation outside a
  DISPATCH, letting a ws2 lease be charged to a ws1 allocation and a ws2 allocation derive
  reclaimable while ws2 residency was live (use-after-free). Now checked standalone, and a
  lease counts toward an allocation only when their weight sets match (E_CHAIN_BROKEN).
- a StateLease's boot_epoch was exempt from the chain boot coherence; a lease minted in a
  prior boot generation attached to a live chain. Now sl.boot_epoch must match its residency
  lease (E_STATE_LEASE).
- a DRAINING/EVICTING/ERROR/QUARANTINED ResidencyLease was accepted for DISPATCH. Now a
  dispatched tuple's lease must be READY or LEASED (E_CHAIN_BROKEN).

Simulator (`sim/residency_sim.py`):
- the terminal `exec_done` branch unconditionally re-released a lane already freed by a
  compute-phase `link_drop` reject, inflating `htp_free`/`gpu_free` above capacity and
  defeating bounded lane admission. Now the terminal branch releases only if the request
  still holds the lane, and `run()` asserts `0 <= *_free <= *_lane_slots`.
- `link_drop(compute)` re-scanned `req_res` while `_release_lane` promoted a queued request
  to compute mid-loop, folding activation-phase queued requests into the drop. Now the victim
  set is snapshotted before mutating.
- `link_drop(bulk)` resume let the stale original transfer event roll back the entire UFS
  reservation while the resumed pipeline reached READY without re-reserving it (silent UFS
  over-commit). Now a bare version bump (preemption / resume) is a superseded no-op that keeps
  the reservation, and `run()` asserts `sum(res_ufs) == ufs_used`.

Honesty note: a later review found eleven additional valid-but-incoherent v3 bundles. They
cover absent/stale/ineligible DeviceInventory state and lease expiry, incompatible
model/graph/backend-build identities, unbound EXECUTE/BULK/ticket epochs, and mutable-state
reservations not bound into identity or reconciled with the physical ledger. These require a
versioned v4 repair. v3 remains static regression evidence, not dispatch authority.

## Remaining S9-V1 blockers

Unchanged from V0-R: directional H2D/D2H per-device link profiles with domain caps; a
staged-file mode that does not double-charge UFS; the decomposed per-leg measurements
(transport/UFS/fsync/verify/materialize/prepare/warmup + a measured interference matrix);
confirmed real traces (S8 Gate A). Do not start the directional link model or claim
capacity/energy until these pass. Nothing was committed or pushed.
