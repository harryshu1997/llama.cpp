# S9-V0-R3 Static Bundle Repair

Status: bundle/schema version 5 implemented and tested on 2026-07-14. Versions
1-4 remain frozen as historical evidence.

## Why v5 Exists

The v4 suites were green, but an independent mutation audit found more coherent
records that v4 accepted: unbound lease/certificate fields, foreign model and
weight identities, uncovered island ranges, future snapshots, unenforced SoC
floors, tickets outside their segment/chunk/device authority, duplicate frames,
and malformed tuples that reached cross-record code and raised `KeyError`.

A second audit of the first v5 pass found missing or unlisted ticket segments,
island ranges and I/O outside their manifest/correctness envelope, duplicated
set identities, and false ReadyCertificate physical claims. Those cases are also
closed and included in the final v5 fixtures.

## Repairs

- Schema validation completes for every record before digest or cross-record
  dereference. Any schema failure returns `E_SCHEMA`; malformed records never
  reach `_tset` or other shape-assuming code.
- Each content-addressed v5 digest binds every schema-allowed field except the
  record envelope and digest field. Unknown fields remain schema-rejected.
- Dispatch chains bind ReadyCertificate and ResidencyLease to the exact served
  WeightSet, model, manifest entry, segment records, backend build, SoC floor,
  layer coverage, and correctness I/O envelope.
- ReadyCertificate physical partitions are derived from the canonical allocation,
  prepared image, residency lease, and state lease. Its free-RAM value must match
  the DeviceInventory from the same receiver timestamp.
- Device and certificate timestamps cannot occur after the dispatch decision or
  violate READY-before-lease causality.
- Transfer tickets require a present device, model, exact WeightSet member,
  contiguous whole-chunk byte range, and matching frame request/chunk/length.
- Duplicate frame sequence/idempotency identities and duplicate set/tuple
  identities are rejected.
- `ModelManifest.required_backends[].min_soc` is mandatory in v5.

## Evidence

- v5 schemas: 15, self-contained draft-2020.
- v5 schema fixtures: 24 (16 valid, 8 invalid), checked by jsonschema 4.10.3 and
  ajv-cli 5.0.0.
- v5 bundle fixtures: 28 (1 valid, 27 adversarial), exact error-code sets.
- red-v4/green-v5 regression and digest checks: 34/34.
- full schema run: 140 fixtures across v1-v5, with zero failures under both
  validators.
- deterministic v5 schema/fixture regeneration: aggregate SHA-256
  `1b63d39748645f6d807a42286cb4641bc7517e0f5fd8ec9ea23fbbe202452838`
  before and after regeneration.
- historical v2/v3/v4 bundle suites remain green.

Run from this directory:

```sh
python3 make_v5_schemas.py
python3 make_fixtures_v5.py
python3 run_schema_tests.py
python3 bundle_validate.py --selftest
python3 sim/test_v5_regressions.py
```

## Claim Boundary

v5 is a static, immutable-snapshot coherence check for the modeled dispatch. It
does not authenticate a producer and does not implement a live scheduler. Atomic
snapshot acquisition, compare-and-reserve pins, credit consumption, connection
ordering, payload hashing, deadlines, cancellation, and completion races remain
runtime responsibilities. The phone prototype covers a separate bounded subset
of those runtime mechanics.

Verdict: `TARGETED STATIC INVARIANTS PASS; LIVE DISPATCH AND CAPACITY UNPROVEN`.
