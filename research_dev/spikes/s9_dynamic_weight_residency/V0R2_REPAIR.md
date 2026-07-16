# S9-V0-R2: v4 static bundle-coherence closure

The 2026-07-14 review (`REVIEW_2026-07-14.md`) proved that the R1/v3 suites pass while v3 still
VALIDATES static bundles that disagree with authoritative device state, executable identity,
transport epochs, or physical reservations -- eleven concrete fail-open cases. This round freezes
v1/v2/v3 byte-for-byte, introduces a versioned **v4** contract (`schema_version 4`,
`bundle_version 4`), and closes all eleven with a v4-only domain-separated digest family and
red-v3 / green-v4 evidence.

**Label: `STATIC_SNAPSHOT_COHERENT` only.** v4 is NOT live dispatch certification. Atomic snapshot
acquisition, compare-and-reserve pins, and completion races remain V0c runtime work. No capacity,
energy, PIM-hardware, or dispatch-certification claim follows from these synthetic static tests.

## Freeze + versioning

- v1, v2, v3 schemas, fixtures, and goldens are untouched (v2/v3 selftests pass byte-for-byte, and
  every v2/v3 digest OUTPUT is unchanged -- the v4 digest family is append-only in `s9lib.py`).
- The pre-R2 (v3) validator + digest library are frozen under `golden/v0r1_historical/` as the RED
  side; the red-v3/green-v4 suite imports the FROZEN validator, so "v3 accepts" is reproducible.

| artifact | sha256 |
|---|---|
| frozen v3 validator (`bundle_validate_v0r1.py`) | `06b962c5...` |
| frozen v3 digest lib (`s9lib_v0r1.py`) | `6cfe043e...` |
| live R2 validator (`bundle_validate.py`, after both hunt rounds) | `4a6a5d56...` |
| live R2 digest lib (`s9lib.py`) | `f0d86f5c...` |
| v4 coherent reference bundle (`fixtures/v4/bundles/valid/dispatchable.json`) | `e6f05c55...` |
| R1 golden replay (preserved; sim untouched) | `9a69a7f6...` |
| GIT_HEAD | `933c722f6` |

## The v4 digest family (`s9lib.py`, append-only)

`domain_digest_v4` uses a DISJOINT pre-image tag `s9:<kind>:v4\n`. Eight `*_v4` builders re-domain
the v2 preimages so a whole v4 bundle is one coherent chain; three of them additionally BIND fields
the review found unbound in v3:

- `prepared_image_digest_v4` adds `source_weight_set_id` (repair 5).
- `transfer_ticket_digest_v4` adds `issued_boot_epoch` + `issued_residency_generation` (repair 6).
- `state_lease_digest_v4` adds `reserved_state_bytes` + `reserved_activation_bytes` +
  `in_flight_mutation_seq` (repair 9).

`sim/test_v4_regressions.py` proves each of the six new fields is EFFECTIVE: changing it changes the
v4 digest and does NOT change the v3 digest.

## Schemas (`schemas/v4/`, 15 files via `make_v4_schemas.py`)

13 record schemas + the bundle envelope are const-bumped 3->4. Two are structurally extended so the
dispatch commits to the state it decided against:

- `dispatch_decision` gains `decision_ts_us` (repair 2) and `device_status_ref{device_id,
  boot_epoch, status_seq}` (repair 1), both required.
- `transport_frame` gains a required `device_id` (repairs 7/8).

`sim_config` / `sim_run_manifest` are NOT emitted for v4: v4 does not touch the simulator.

## Validator repairs (`bundle_validate.py`, `cross_record_v4`)

`bundle_version 4` runs the full v2 + v3 (`cross_record` + `cross_record_r1`) checks AND
`cross_record_v4`, with the v4 digest builders. Nine new stable codes (35 total):
`E_DEVICE_ABSENT`, `E_DEVICE_STALE`, `E_DEVICE_INELIGIBLE`, `E_LEASE_EXPIRED`, `E_GATE_UNDERIVED`,
`E_IDENTITY_MISMATCH`, `E_PI_SOURCE`, `E_FRAME_EPOCH`, `E_LEDGER_DERIVED`.

| repair | v4 check | code |
|---|---|---|
| 1 | DISPATCH binds exactly one DeviceInventory snapshot (present, ref boot/status_seq pinned, dispatched residency boot == device boot) | E_DEVICE_ABSENT / E_DEVICE_STALE |
| 2 | device accepting=true, draining=false, stale=false, thermal.eligible=true, backend supported | E_DEVICE_INELIGIBLE |
| 3 | residency lease live at `decision_ts_us` | E_LEASE_EXPIRED |
| 4 | caller `epoch_match`/`credits` booleans EQUAL the record-derived truth (defense in depth) | E_GATE_UNDERIVED |
| 5 | manifest/model/graph (manifest must be present), model_version/graph/backend/backend_build, and arch/soc/layout_version across prepared-image / weight-set / allocation / ready-cert / device | E_IDENTITY_MISMATCH |
| 6 | `PreparedImage.source_weight_set_id` == the served weight set | E_PI_SOURCE |
| 7 | BULK frame device/boot/residency == its ticket; EXECUTE and RESULT frames device/route/boot/residency/state == their dispatch + state lease | E_FRAME_EPOCH |
| 8 | DeviceInventory ledger DERIVED from live records by single-copy canonical accounting; missing or excess charges rejected | E_LEDGER_DERIVED |

`E_GATE_UNDERIVED` is a defense-in-depth cross-check: a DISPATCH cannot assert a gate its records do
not support. It co-fires by design with the specific code for the same incoherence (e.g. a
device-absent bundle emits `{E_DEVICE_ABSENT, E_GATE_UNDERIVED}`); the fixture index records the
exact co-fire set for each case.

## Red-v3 / green-v4 evidence

`fixtures/v4/bundles/` has 1 coherent bundle + 16 adversarials (one per review blocker, some blockers
split into facets), each with an EXACT expected-code set in `index.json`. `sim/test_v4_regressions.py`
(23 checks) shows for each: the FROZEN v3 validator ACCEPTS the equivalent bundle (fail-open) and the
LIVE v4 validator REJECTS with the recorded code(s).

| blocker | v4 fixture | expected codes |
|---|---|---|
| 1 DeviceInventory absent | device_absent.json | E_DEVICE_ABSENT, E_GATE_UNDERIVED |
| 1 wrong boot pinned | device_boot_mismatch.json | E_DEVICE_STALE, E_GATE_UNDERIVED |
| 1 wrong status_seq pinned | device_status_seq_mismatch.json | E_DEVICE_STALE, E_GATE_UNDERIVED |
| 2 draining | device_draining.json | E_DEVICE_INELIGIBLE |
| 2 thermal ineligible | device_thermal_ineligible.json | E_DEVICE_INELIGIBLE |
| 2 backend unsupported | device_backend_unsupported.json | E_DEVICE_INELIGIBLE |
| 3 lease expired | lease_expired.json | E_LEASE_EXPIRED |
| 4 backend_build mismatch | backend_build_mismatch.json | E_IDENTITY_MISMATCH |
| 5 island graph mismatch | island_graph_mismatch.json | E_IDENTITY_MISMATCH |
| 6 prepared-image source foreign | pi_source_foreign.json | E_PI_SOURCE |
| 7 bulk frame stale epoch | bulk_frame_stale_epoch.json | E_FRAME_EPOCH |
| 7 exec frame stale route | exec_frame_stale_route.json | E_FRAME_EPOCH |
| 8 ticket issued-boot now bound | ticket_issued_boot_unbound.json | E_DIGEST_MISMATCH |
| 9 state mutation-seq now bound | state_mutation_seq_unbound.json | E_DIGEST_MISMATCH |
| 10 reservation exceeds ledger | state_reservation_exceeds_ledger.json | E_GATE_UNDERIVED, E_LEDGER_DERIVED |
| 11 zero-live ledger | ledger_zero_live.json | E_GATE_UNDERIVED, E_LEDGER_DERIVED |

## Suite counts (all green)

```
python3 run_schema_tests.py            # 116 fixtures (53 v1 + 33 v2 + 14 v3 + 16 v4), both validators, 0 fail
python3 bundle_validate.py --selftest  # v2 (15) + v3 (18) + v4 (36) bundle fixtures, 0 fail
python3 validate_manifests.py --selftest    # 23 v1 semantic, 0 fail  (unchanged)
python3 sim/test_residency_sim.py      # 26 FROZEN V0-R behavior checks, 0 fail  (sim untouched)
python3 sim/test_r1_regressions.py     # 17 R1 red/green, 0 fail       (sim untouched)
python3 sim/test_r1_holes.py           # 16 R1 hole closures, 0 fail   (sim untouched)
python3 sim/test_golden_replay.py      # 13 replay/provenance, 0 fail  (sim untouched)
python3 sim/test_v4_regressions.py     # 35 red-v3/green-v4 + 6 digest-effectiveness + 1 valid = 42, 0 fail
git diff --check                       # clean; no generated bytecode
```

The v4 bundle set is 1 coherent + 35 adversarials: 16 review-blocker fixtures plus 19 hole-hunt
closures (10 from round 1, 9 from round 2). `test_v4_regressions.py`'s 35 red-v3/green-v4 checks pair
each adversarial with a proof that the FROZEN v3 validator accepts the equivalent bundle.

## Independent hole hunt

After the first v4 green pass, an 8-lens adversarial hole-hunt workflow (each agent building and
running concrete repros against the LIVE v4 validator, then a skeptical verify pass) found that the
first v4 pass was INCOMPLETE. It surfaced SIX distinct new fail-open holes (2 further candidates were
refuted as not-bugs, and 2 more -- a duplicate canonical allocation and a second uncounted state
lease -- were found ALREADY closed by the record-derived ledger). This is the same "every adversarial
round finds more" pattern R1 hit; the finding is reported, not hidden, and completeness is NOT claimed.

All six are now closed with red-v3/green-v4 fixtures + regression checks:

| # | hole (v4 first pass ACCEPTED an incoherent bundle) | fix | code |
|---|---|---|---|
| 1 | a RESULT frame (a live-payload frame like EXECUTE) dodged the EXECUTE-only epoch binding | bind EXECUTE and RESULT symmetrically | E_FRAME_EPOCH |
| 2 | island model identity check was gated on `mm is not None`, so an absent/foreign manifest skipped it | fail CLOSED when the island's model has no manifest | E_IDENTITY_MISMATCH |
| 3 | a device carrying live allocation/image/lease/state bytes but NO DeviceInventory escaped the ledger | derive the ledger over EVERY referenced device; uncounted live bytes are a missing charge | E_LEDGER_DERIVED |
| 4 | a ReadyCertificate could attest a footprint larger than the device LPDDR; an RL weight reservation could differ from its allocation | reconcile RC.physical_bytes.total <= lpddr and RL.reserved weights == allocation | E_LEDGER_DERIVED |
| 5 | `arch`, `soc`, `layout_version` (all digest-bound identity fields) were never cross-checked, so a v75 image on a v81 device / a foreign arch / a foreign layout validated | cross-check arch/soc/layout_version across prepared image / weight set / allocation / ready cert / device | E_IDENTITY_MISMATCH |
| 6 | two DISPATCH decisions (or a DISPATCH + FALLBACK) for the same request validated | a request has exactly one dispatch decision | E_DISPATCH_MISMATCH |

No new stable codes were needed; the fixes reuse the existing v4 codes. Fixtures: 10 additional
red-v3/green-v4 adversarials (`result_frame_stale`, `island_no_manifest`, `ledger_diless_device`,
`rc_footprint_exceeds_lpddr`, `rl_weights_mismatch`, `pi_arch_foreign`, `pi_layout_foreign`,
`pi_soc_foreign`, `di_soc_mismatch`, `duplicate_dispatch_request`).

A focused SECOND-round hunt (4 lenses: identity residue, ledger residue, frame/dispatch residue, a
completeness critic) then found ELEVEN more confirmed fail-open holes in the coherence-field family --
decisive confirmation that the STATIC surface is not, and cannot be presumed, complete. NINE were
closed (again reusing existing codes):

| round-2 hole | fix | code |
|---|---|---|
| served SoC not a MEMBER of `manifest.compatible_soc` (equality alone passed) | device soc in compatible_soc | E_IDENTITY_MISMATCH |
| `image_class` / `preparation_algorithm` not legal for the serving backend (a GPU format on an htp image) | format prefix must match backend | E_IDENTITY_MISMATCH |
| served `(backend, backend_build)` not in `manifest.required_backends` | membership required | E_IDENTITY_MISMATCH |
| `dtype` never cross-checked across allocation / weight set / manifest | dtype must agree | E_IDENTITY_MISMATCH |
| `ResidencyLease.reserved_bytes.derived` escaped the ledger | == derived bytes of images on its allocation | E_LEDGER_DERIVED |
| `canonical_allocation.canonical_bytes` could be SMALLER than the weight set (under-reservation) | == weight set total_bytes | E_LEDGER_DERIVED |
| two canonical allocations of the same content on one device at one generation (single-copy) | (weight_set_digest, device, generation) unique | E_LEDGER_DERIVED |
| `ReadyCertificate.correctness.metric_digest`/verdict could disagree with its cert | embedded == certificate | E_CORRECTNESS_BINDING |
| `TransferTicket.issued_boot_epoch` not anchored to the device it targets | == device boot_epoch | E_FRAME_EPOCH |

The remaining round-2 confirmed items are recorded as **KNOWN-OPEN static-coherence gaps** and are NOT
closed here, because closing them risks over-constraining legitimate static bundles or needs a modeling
decision (they are flagged for the reviewer, not silently accepted):

- other live control frames (LEASE/RELEASE/ALLOC_STATE/FREE_STATE/RESET_STATE/CREDIT/DRAIN) are not
  epoch-bound to a dispatch (only EXECUTE/RESULT are) -- some are not per-request, so blanket binding
  could reject legal bundles;
- a stateless island's EXECUTE/RESULT `state_epoch` is unchecked (there is no StateLease to bind to);
- `island.io_schema` is not reconciled with the certified `correctness.shape_envelope`;
- boot coherence between a residency record and its DeviceInventory is enforced only inside a DISPATCH.

**Honest label: `STATIC_SNAPSHOT_COHERENT; TARGETED + TWO ADVERSARIAL ROUNDS OF FOUND CASES CLOSED;
NOT A COMPLETENESS PROOF; KNOWN-OPEN GAPS LISTED; NOT LIVE DISPATCH CERTIFICATION; CAPACITY UNPROVEN.`**
Every adversarial round (v3->v4->hunt1->hunt2) found new holes; a static validator over this record
schema has an open-ended coherence surface. Completeness is NOT claimed. A record-derived LIVE-dispatch
model with atomic snapshot acquisition, compare-and-reserve use pins, and completion validation is V0c.

## Remaining limitations (still S9-V1 / V0c)

v4 is STATIC_SNAPSHOT_COHERENT only. It validates that ONE serialized bundle is internally
impossible to fault; it does NOT model atomic snapshot acquisition, compare-and-reserve pins on live
device state, or completion races -- those are V0c runtime work. Capacity/energy stay UNPROVEN:
device rates are symbolic and the directional H2D/D2H link model is still S9-V1. Nothing was
committed or pushed.
