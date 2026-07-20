#!/usr/bin/env python3
"""Independent fail-closed validator for the S14 CP0-c island catalog.

Shares only the primitive helpers (canonical_json/sha256/derivations) with the
builder; it RE-DERIVES and RE-CHECKS every asserted value rather than reusing
construction logic:

  1. schema:   envelope vs island_catalog.schema.json, each island vs
               island_descriptor.schema.json, each row vs profile_row.schema.json
               (jsonschema Draft 2020-12).
  2. identity: descriptor_hash, graph_hash, weight_set_id, and catalog_hash are
               recomputed and must match.
  3. cross:    every row.island_descriptor_ref resolves to a descriptor; the
               row's (model_version, graph_hash, layer_range, attention_class)
               equal that descriptor's; every descriptor.model_version is a
               declared model; mixed attention carries a per-layer vector of the
               right length.
  4. boundary: non-null row boundary bytes are positive and within the
               descriptor's declared tensorset max_bytes.
  5. verdict:  the profile_row PASS predicate is re-evaluated; a row labelled
               PASS that fails it is an over-claim and fails the catalog.
  6. binding:  with --verify-artifacts, every source_binding and every row
               artifact_hash must match the on-disk file it names.
  7. frozen:   scope/frozen_status/energy_status constants are enforced.

Then it computes and prints per-row scheduler eligibility. Requires jsonschema
(present in /usr/bin/python3 on this host; absent in the npu-harness venv).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import catalog_common as cc

try:
    from jsonschema import Draft202012Validator
except Exception as exc:  # pragma: no cover - environment guard
    print(f"FATAL: jsonschema is required (use /usr/bin/python3): {exc}", file=sys.stderr)
    raise SystemExit(2)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SCHEMAS_DIR = HERE.parent / "s8_operator_island_affinity" / "schemas"

PASS_REQUIRED_NONNULL = (
    "p50_us", "p95_us", "p99_us", "cov", "resident_bytes",
    "boundary_in_bytes", "boundary_out_bytes", "transfer_measured_us",
    "correctness_metric", "server_control", "server_relief",
)


def _load_validator(name: str) -> Draft202012Validator:
    schema = cc.load_json(SCHEMAS_DIR / name)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _schema_errors(validator: Draft202012Validator, instance: Any, label: str) -> list[str]:
    out = []
    for err in sorted(validator.iter_errors(instance), key=lambda e: list(e.path)):
        loc = "/".join(str(p) for p in err.path)
        out.append(f"{label}[{loc}]: {err.message}")
    return out


def _pass_predicate_ok(row: dict[str, Any]) -> bool:
    """Independent re-implementation of profile_row PASS (must not fail open)."""
    if row.get("correctness") != "pass":
        return False
    if row.get("fallback") != "none":
        return False
    if row.get("supported") is not True:
        return False
    if not isinstance(row.get("n_proc"), int) or row["n_proc"] < 7:
        return False
    if row.get("post_transfer_slo_feasible") is not True:
        return False
    for key in PASS_REQUIRED_NONNULL:
        if row.get(key) is None:
            return False
    if not row.get("artifact_paths"):
        return False
    return True


def eligibility(row: dict[str, Any]) -> tuple[str, str]:
    if row.get("verdict") == "UNKNOWN":
        return "INELIGIBLE_UNMEASURED", "verdict=UNKNOWN"
    if row.get("verdict") not in ("PASS", "LOWER_BOUND"):
        return "INELIGIBLE", f"verdict={row.get('verdict')}"
    if row.get("correctness") != "pass":
        return "INELIGIBLE", f"correctness={row.get('correctness')}"
    if row.get("fallback") != "none":
        return "INELIGIBLE", f"fallback={row.get('fallback')}"
    if row.get("supported") is not True:
        return "INELIGIBLE", "supported!=true"
    if row.get("p50_us") is None:
        return "INELIGIBLE_NO_LATENCY", "p50_us is null"
    if row.get("boundary_in_bytes") is None or row.get("boundary_out_bytes") is None:
        return "INELIGIBLE", "boundary bytes null"
    if row.get("post_transfer_slo_feasible") is not True:
        return "INELIGIBLE_SLO", "post_transfer_slo_feasible!=true"
    if not isinstance(row.get("n_proc"), int) or row["n_proc"] < 7:
        return "ELIGIBLE_COARSE", f"n_proc={row.get('n_proc')}<7 (provisional single-process)"
    return "ELIGIBLE", "full evidence"


def validate(catalog_path: Path, verify_artifacts: bool, repo_root: Path) -> list[str]:
    errors: list[str] = []
    try:
        catalog = cc.load_json(catalog_path)
    except (OSError, UnicodeDecodeError, ValueError, cc.CatalogError) as exc:
        return [f"catalog: invalid JSON: {exc}"]

    env_v = _load_validator("island_catalog.schema.json")
    desc_v = _load_validator("island_descriptor.schema.json")
    row_v = _load_validator("profile_row.schema.json")

    errors += _schema_errors(env_v, catalog, "catalog")
    if errors:
        return errors  # envelope malformed; deeper checks unsafe

    # constants
    if catalog["scope"] != "CP0C_ISLAND_CATALOG_MECHANICS_NO_ENERGY":
        errors.append("catalog.scope: wrong constant")
    if catalog["frozen_status"] != "FROZEN_BEFORE_SCHEDULER_RESULTS":
        errors.append("catalog.frozen_status: wrong constant")
    if catalog["energy_status"] != "NOT_RUN":
        errors.append("catalog.energy_status: wrong constant")

    # catalog_hash recompute
    recomputed = cc.sha256_of({k: v for k, v in catalog.items() if k != "catalog_hash"})
    if recomputed != catalog["catalog_hash"]:
        errors.append(f"catalog.catalog_hash: mismatch (recomputed {recomputed})")

    models = {m["model_id"]: m for m in catalog["models"]}
    model_versions = {m["model_version"] for m in catalog["models"]}

    # islands
    descriptors_by_hash: dict[str, dict[str, Any]] = {}
    for idx, island in enumerate(catalog["islands"]):
        island_errs = _schema_errors(desc_v, island, f"island[{idx}]")
        errors += island_errs
        if island_errs:
            continue
        body = {k: v for k, v in island.items() if k != "descriptor_hash"}
        exp_hash = cc.sha256_of(body)
        if exp_hash != island["descriptor_hash"]:
            errors.append(f"island[{idx}] {island['island_id']}: descriptor_hash mismatch")
        exp_graph = cc.derived_graph_hash(island["model_version"], island["layer_range"], island["attention_class"])
        if exp_graph != island["graph_hash"]:
            errors.append(f"island[{idx}] {island['island_id']}: graph_hash not the v0 derivation")
        exp_weight = cc.derived_weight_set_id(island["model_version"], island["layer_range"])
        if exp_weight != island["weight_set_id"]:
            errors.append(f"island[{idx}] {island['island_id']}: weight_set_id not the v0 derivation")
        if island["model_version"] not in model_versions:
            errors.append(f"island[{idx}] {island['island_id']}: model_version not a declared model")
        lr = island["layer_range"]
        if not (0 <= lr["start"] < lr["end"] <= lr["n_layer_total"]):
            errors.append(f"island[{idx}] {island['island_id']}: bad layer_range ordering")
        if island["attention_class"] == "mixed":
            vec = island.get("attention_class_by_layer")
            if not vec or len(vec) != lr["end"] - lr["start"]:
                errors.append(f"island[{idx}] {island['island_id']}: mixed needs per-layer vector of length end-start")
        if island["descriptor_hash"] in descriptors_by_hash:
            errors.append(f"island[{idx}] {island['island_id']}: duplicate descriptor_hash")
        descriptors_by_hash[island["descriptor_hash"]] = island

    # rows
    for idx, row in enumerate(catalog["profile_rows"]):
        row_errs = _schema_errors(row_v, row, f"row[{idx}]")
        errors += row_errs
        # Independent PASS over-claim check: runs even when the schema also
        # errored, so a loosened schema can never fail this open. Uses .get()
        # so it is safe on a malformed row.
        if row.get("verdict") == "PASS" and not _pass_predicate_ok(row):
            errors.append(f"row[{idx}] {row.get('island_id')}: verdict=PASS but PASS predicate fails (over-claim)")
        if row_errs:
            continue
        ref = row["island_descriptor_ref"]
        desc = descriptors_by_hash.get(ref)
        if desc is None:
            errors.append(f"row[{idx}] {row['island_id']}: island_descriptor_ref does not resolve")
            continue
        if row["island_id"] != desc["island_id"]:
            errors.append(f"row[{idx}]: island_id != descriptor island_id")
        for key in ("model_version", "graph_hash", "attention_class"):
            if row[key] != desc[key]:
                errors.append(f"row[{idx}] {row['island_id']}: {key} != descriptor")
        if row["layer_range"] != desc["layer_range"]:
            errors.append(f"row[{idx}] {row['island_id']}: layer_range != descriptor")

        # boundary bounds vs descriptor tensorsets
        in_cap = sum(t["max_bytes"] for t in desc["boundary_in"]["tensors"])
        out_cap = sum(t["max_bytes"] for t in desc["boundary_out"]["tensors"])
        for field, cap in (("boundary_in_bytes", in_cap), ("boundary_out_bytes", out_cap)):
            val = row.get(field)
            if val is not None:
                if val <= 0:
                    errors.append(f"row[{idx}] {row['island_id']}: {field} must be positive when set")
                elif val > cap:
                    errors.append(f"row[{idx}] {row['island_id']}: {field}={val} exceeds descriptor cap {cap}")

        # artifact_hashes align with artifact_paths length
        if len(row["artifact_paths"]) != len(row.get("artifact_hashes", [])):
            errors.append(f"row[{idx}] {row['island_id']}: artifact_paths/artifact_hashes length mismatch")
        if verify_artifacts:
            for path, digest in zip(row["artifact_paths"], row.get("artifact_hashes", [])):
                actual = _safe_digest(repo_root / path)
                if actual != digest:
                    errors.append(f"row[{idx}] {row['island_id']}: artifact digest mismatch for {path}")

    # source_bindings resolve
    seen_paths: set[str] = set()
    for idx, binding in enumerate(catalog["source_bindings"]):
        if binding["path"] in seen_paths:
            errors.append(f"source_bindings[{idx}]: duplicate path {binding['path']}")
        seen_paths.add(binding["path"])
        if verify_artifacts:
            actual = _safe_digest(repo_root / binding["path"])
            if actual != binding["sha256"]:
                errors.append(f"source_bindings[{idx}]: digest mismatch for {binding['path']}")

    return errors


def _safe_digest(path: Path) -> str | None:
    try:
        return cc.sha256_file(path)
    except OSError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the S14 CP0-c frozen island catalog")
    parser.add_argument("--catalog", default=str(HERE / "island_catalog.json"))
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--verify-artifacts", action="store_true")
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    errors = validate(catalog_path, args.verify_artifacts, Path(args.repo_root))
    if errors:
        print("CATALOG INVALID:")
        for e in errors:
            print(f"  - {e}")
        return 1

    catalog = cc.load_json(catalog_path)
    print(f"CATALOG VALID: {catalog['catalog_id']}")
    print(f"catalog_hash {catalog['catalog_hash']}")
    print(f"islands {len(catalog['islands'])} rows {len(catalog['profile_rows'])} "
          f"verify_artifacts={args.verify_artifacts}")
    print("eligibility:")
    for row in catalog["profile_rows"]:
        status, reason = eligibility(row)
        print(f"  {row['island_id']:20s} {row['device_backend_id']:10s} "
              f"{row['verdict']:12s} -> {status} ({reason})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
