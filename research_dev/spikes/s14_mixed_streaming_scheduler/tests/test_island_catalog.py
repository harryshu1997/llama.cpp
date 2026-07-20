#!/usr/bin/env python3
"""Fail-closed tests for the S14 CP0-c island catalog.

Run under /usr/bin/python3 (jsonschema present there; absent in the venv):
    /usr/bin/python3 tests/test_island_catalog.py

Covers build determinism, the happy path, and one negative case per structural
guard. Negative cases RESEAL catalog_hash after mutating (mimicking an adversary
who re-freezes a tampered catalog) so the targeted structural check fires rather
than being masked by the top-level hash check.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MOD_DIR = HERE.parent
sys.path.insert(0, str(MOD_DIR))

import catalog_common as cc  # noqa: E402
import build_catalog as bc  # noqa: E402
import validate_catalog as vc  # noqa: E402

REPO_ROOT = MOD_DIR.parents[2]


def _fresh_catalog() -> dict:
    digests = {path: bc._digest(path) for (_, path, _) in bc.SOURCES}
    return bc.build(digests)


def _reseal(catalog: dict) -> dict:
    catalog["catalog_hash"] = cc.sha256_of({k: v for k, v in catalog.items() if k != "catalog_hash"})
    return catalog


def _errors(catalog: dict, verify_artifacts: bool = False) -> list[str]:
    with tempfile.NamedTemporaryFile("wb", suffix=".json", delete=False) as fh:
        fh.write(cc.canonical_json(catalog))
        path = Path(fh.name)
    try:
        return vc.validate(path, verify_artifacts, REPO_ROOT)
    finally:
        path.unlink(missing_ok=True)


def _errors_from_path(path: Path, verify_artifacts: bool = False) -> list[str]:
    return vc.validate(path, verify_artifacts, REPO_ROOT)


def _has(errors: list[str], needle: str) -> bool:
    return any(needle in e for e in errors)


# ---- tests ----

def test_build_is_deterministic():
    a = cc.canonical_json(_fresh_catalog())
    b = cc.canonical_json(_fresh_catalog())
    assert a == b, "build must be byte-identical across runs"


def test_happy_path_valid():
    assert _errors(_fresh_catalog()) == []


def test_happy_path_valid_with_artifacts():
    assert _errors(_fresh_catalog(), verify_artifacts=True) == []


def test_catalog_hash_tamper():
    cat = _fresh_catalog()
    cat["catalog_hash"] = "sha256:" + "0" * 64
    assert _has(_errors(cat), "catalog_hash: mismatch")


def test_nonfinite_json_rejected():
    cat = _fresh_catalog()
    cat["profile_rows"][3]["correctness_metric"]["value"] = float("nan")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(cat, fh, separators=(",", ":"))
        path = Path(fh.name)
    try:
        assert _has(_errors_from_path(path), "invalid JSON")
    finally:
        path.unlink(missing_ok=True)


def test_duplicate_json_key_rejected():
    with tempfile.NamedTemporaryFile("wb", suffix=".json", delete=False) as fh:
        fh.write(b'{"schema_version":1,"schema_version":1}')
        path = Path(fh.name)
    try:
        assert _has(_errors_from_path(path), "invalid JSON")
    finally:
        path.unlink(missing_ok=True)


def test_descriptor_hash_tamper():
    cat = _fresh_catalog()
    # mutate a descriptor field but do NOT recompute descriptor_hash; reseal catalog_hash
    cat["islands"][0]["service_class"] = "background"
    _reseal(cat)
    assert _has(_errors(cat), "descriptor_hash mismatch")


def test_graph_hash_not_derived():
    cat = _fresh_catalog()
    isl = cat["islands"][0]
    isl["graph_hash"] = "sha256:" + "1" * 64
    isl["descriptor_hash"] = cc.descriptor_hash(isl)  # re-address so the hash check passes
    cat["profile_rows"][0]["graph_hash"] = isl["graph_hash"]
    cat["profile_rows"][0]["island_descriptor_ref"] = isl["descriptor_hash"]
    _reseal(cat)
    assert _has(_errors(cat), "graph_hash not the v0 derivation")


def test_row_ref_unresolved():
    cat = _fresh_catalog()
    cat["profile_rows"][0]["island_descriptor_ref"] = "sha256:" + "2" * 64
    _reseal(cat)
    assert _has(_errors(cat), "island_descriptor_ref does not resolve")


def test_row_model_version_mismatch():
    cat = _fresh_catalog()
    # keep ref resolvable but flip model_version so the cross-check fires
    cat["profile_rows"][0]["model_version"] = bc.BGE_MODEL_VERSION
    _reseal(cat)
    assert _has(_errors(cat), "model_version != descriptor")


def test_verdict_pass_overclaim():
    cat = _fresh_catalog()
    # gemma_head_0_2 has n_proc=1; claiming PASS must be rejected as an over-claim
    cat["profile_rows"][0]["verdict"] = "PASS"
    _reseal(cat)
    assert _has(_errors(cat), "PASS predicate fails (over-claim)")


def test_boundary_exceeds_cap():
    cat = _fresh_catalog()
    cap = sum(t["max_bytes"] for t in cat["islands"][0]["boundary_in"]["tensors"])
    cat["profile_rows"][0]["boundary_in_bytes"] = cap + 1
    _reseal(cat)
    assert _has(_errors(cat), "exceeds descriptor cap")


def test_boundary_nonpositive():
    cat = _fresh_catalog()
    cat["profile_rows"][0]["boundary_in_bytes"] = 0
    _reseal(cat)
    # jsonschema nint allows 0; our positivity guard must still fire
    assert _has(_errors(cat), "must be positive")


def test_bad_scope_constant():
    cat = _fresh_catalog()
    cat["scope"] = "SOMETHING_ELSE"
    _reseal(cat)
    # envelope schema const fires first
    assert _errors(cat), "wrong scope must fail"


def test_artifact_digest_tamper_under_verify():
    cat = _fresh_catalog()
    cat["profile_rows"][0]["artifact_hashes"][0] = "sha256:" + "3" * 64
    _reseal(cat)
    assert _has(_errors(cat, verify_artifacts=True), "artifact digest mismatch")


def test_source_binding_digest_tamper_under_verify():
    cat = _fresh_catalog()
    cat["source_bindings"][0]["sha256"] = "sha256:" + "4" * 64
    _reseal(cat)
    assert _has(_errors(cat, verify_artifacts=True), "digest mismatch")


def test_mixed_attention_requires_vector():
    cat = _fresh_catalog()
    isl = cat["islands"][0]
    isl["attention_class"] = "mixed"  # no attention_class_by_layer -> descriptor schema fails
    isl["descriptor_hash"] = cc.descriptor_hash(isl)
    cat["profile_rows"][0]["attention_class"] = "mixed"
    cat["profile_rows"][0]["island_descriptor_ref"] = isl["descriptor_hash"]
    _reseal(cat)
    assert _errors(cat), "mixed without per-layer vector must fail"


def test_eligibility_statuses():
    cat = _fresh_catalog()
    by_key = {(r["island_id"], r["device_backend_id"]): vc.eligibility(r)[0] for r in cat["profile_rows"]}
    # No row is dispatch-eligible: no single run gives coherent latency + same-run
    # HTP no-fallback certificate (see red-team finding, CATALOG.md).
    assert by_key[("gemma_head_0_2", "OP15/HTP0")] == "INELIGIBLE"
    assert by_key[("gemma_layer_2_3", "OP12/HTP0")] == "INELIGIBLE"
    assert by_key[("gemma_head_0_3", "OP15/HTP0")] == "INELIGIBLE_UNMEASURED"
    assert by_key[("bge_encoder_0_12", "OP15/HTP0")] == "INELIGIBLE_NO_LATENCY"
    assert by_key[("bge_encoder_0_12", "OP12/HTP0")] == "INELIGIBLE_NO_LATENCY"
    assert all(vc.eligibility(r)[0] != "ELIGIBLE" and "ELIGIBLE_COARSE" != vc.eligibility(r)[0]
               for r in cat["profile_rows"]), "freeze must have zero dispatch-eligible rows"


def test_eligibility_requires_slo_feasibility():
    """No-fallback evidence is insufficient when transfer misses the SLO."""
    cat = _fresh_catalog()
    row = cat["profile_rows"][0]
    row["fallback"] = "none"
    row["supported"] = True
    status, reason = vc.eligibility(row)
    assert status == "INELIGIBLE_SLO", (status, reason)


def test_fallback_none_only_where_bound():
    """Regression for the red-team binding finding: a row may assert
    fallback=none only where a no-fallback certificate is actually bound. Gemma
    rows (no same-run placement cert) must be unknown; BGE rows (Gate-1 op-support
    gate bound in artifact_paths) keep none."""
    cat = _fresh_catalog()
    for r in cat["profile_rows"]:
        if r["island_id"].startswith("gemma"):
            assert r["fallback"] != "none", f"{r['island_id']}: gemma no-fallback is not bound; must not claim none"
        if r["island_id"] == "bge_encoder_0_12":
            assert r["fallback"] == "none"
            assert any("GATE1" in p for p in r["artifact_paths"]), "BGE fallback=none must bind the Gate-1 cert"


def test_gemma_row_binary_matches_its_measurement():
    """Regression: the row with a phone latency pins the sweep binary that
    produced it, not the repaired-checkpoint binary."""
    cat = _fresh_catalog()
    g02 = next(r for r in cat["profile_rows"] if r["island_id"] == "gemma_head_0_2")
    assert g02["p50_us"] == 154962
    assert g02["build_hash"] == bc.GEMMA_SWEEP_BUILD
    assert g02["device_binary_hash"] == bc.GEMMA_SWEEP_DEVICE


def test_committed_catalog_matches_builder():
    """The frozen island_catalog.json on disk must equal a fresh build."""
    on_disk = (MOD_DIR / "island_catalog.json").read_bytes()
    fresh = cc.canonical_json(_fresh_catalog())
    assert on_disk == fresh, "island_catalog.json is stale; rerun build_catalog.py"


def _run() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
