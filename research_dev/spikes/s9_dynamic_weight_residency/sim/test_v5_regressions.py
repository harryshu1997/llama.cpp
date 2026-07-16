#!/usr/bin/env python3
"""Red-v4/green-v5 evidence for the S9 v5 static-bundle repairs."""
import copy
import json
import os
import sys
import tempfile


HERE = os.path.dirname(os.path.abspath(__file__))
SPIKE = os.path.dirname(HERE)
sys.path.insert(0, SPIKE)

import bundle_validate as validator
import make_fixtures_v4 as F4
import make_fixtures_v5 as F5
import s9lib


def validate_object(obj):
    fd, path = tempfile.mkstemp(prefix="s9_v5_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f)
        return validator.validate_bundle(path)
    finally:
        os.unlink(path)


def refresh_v4(records):
    return F4.refresh_chain_v4(records)


def v4_red_accepts(mutator):
    records = F4.build_records_v4()
    mutator(records, refresh_v4)
    ordered = F5.core(records)
    if "EXTRA" in records:
        ordered.append(records["EXTRA"])
    return validate_object(F4.bundle("v4-red", ordered)) == []


def v5_green_codes(name):
    path = os.path.join(SPIKE, "fixtures", "v5", "bundles", "invalid", name)
    return sorted({code for code, _ in validator.validate_bundle(path)})


results = []
for name, mutator, expected in F5.CASES:
    red = v4_red_accepts(mutator)
    got = v5_green_codes(name)
    green = got == sorted(expected)
    results.append((name, red and green))
    print(f"  {'PASS' if red and green else 'FAIL'} {name:38} "
          f"v4={'accepts' if red else 'REJECTS'} v5={got} expected={sorted(expected)}")


valid_path = os.path.join(SPIKE, "fixtures", "v5", "bundles", "valid", "dispatchable.json")
valid_errors = validator.validate_bundle(valid_path)
results.append(("valid v5 bundle", valid_errors == []))
print(f"  {'PASS' if valid_errors == [] else 'FAIL'} valid v5 bundle errors={valid_errors}")


malformed_path = os.path.join(SPIKE, "fixtures", "v5", "bundles", "invalid", "schema_invalid_tuple.json")
try:
    malformed_codes = sorted({code for code, _ in validator.validate_bundle(malformed_path)})
    malformed_ok = malformed_codes == ["E_SCHEMA"]
except Exception as ex:
    malformed_codes = [f"CRASH:{type(ex).__name__}"]
    malformed_ok = False
results.append(("schema-invalid returns, never raises", malformed_ok))
print(f"  {'PASS' if malformed_ok else 'FAIL'} schema-invalid returns, never raises codes={malformed_codes}")


records = F5.build_records_v5()
for key, kind, field, value in (
        ("RL1", "residency_lease", "horizon", {"start_us": 1000000, "end_us": 62000000}),
        ("RL1", "residency_lease", "model_id", "foreign-model"),
        ("RL1", "residency_lease", "reserved_bytes", {"weights": 400, "derived": 0, "scratch": 101}),
        ("RC1", "ready_certificate", "free_ram_after_bytes", 9399),
        ("RC1", "ready_certificate", "issued_receiver_ts_us", 1000001),
        ("RC1", "ready_certificate", "physical_bytes",
         {"canonical": 400, "derived": 0, "scratch": 100,
          "activations_reserved": 40, "state_reserved": 61, "total": 601})):
    original = records[key]
    changed = copy.deepcopy(original)
    changed[field] = value
    v5_builder = s9lib.DIGEST_BUILDERS_V5[kind][1]
    v4_builder = s9lib.DIGEST_BUILDERS_V4[kind][1]
    ok = v5_builder(original) != v5_builder(changed) and v4_builder(original) == v4_builder(changed)
    label = f"digest binds {kind}.{field}"
    results.append((label, ok))
    print(f"  {'PASS' if ok else 'FAIL'} {label}")


failures = sum(1 for _, ok in results if not ok)
print(f"\nv5 regressions: {len(results)} checks  failures: {failures}")
sys.exit(1 if failures else 0)
