#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import validate_cp_e_result as CPEV


class EvidenceError(ValueError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise EvidenceError(f"duplicate key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=no_duplicates)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(str(exc)) from exc
    if type(value) is not dict:
        raise EvidenceError("top level must be an object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise EvidenceError(f"artifact unavailable: {path}") from exc
    return "sha256:" + digest.hexdigest()


def validate_thermal(records: object, limit: int) -> None:
    try:
        CPEV.validate_thermal_pair(records, limit)
    except CPEV.EvidenceError as exc:
        raise EvidenceError(str(exc)) from exc


def validate_artifacts(artifacts: object) -> None:
    if type(artifacts) is not dict or type(artifacts.get("host")) is not dict:
        raise EvidenceError("artifact bundle missing")
    base = copy.deepcopy(artifacts)
    batch_source = base["host"].pop("cp_e_batch_cert.py", None)
    if type(batch_source) is not dict or set(batch_source) != {"path", "sha256"}:
        raise EvidenceError("batch-screen source artifact missing")
    if sha256_file(Path(batch_source["path"])) != batch_source["sha256"]:
        raise EvidenceError("batch-screen source artifact mismatch")
    try:
        CPEV.validate_artifacts(base)
    except CPEV.EvidenceError as exc:
        raise EvidenceError(str(exc)) from exc


def validate(result: dict[str, Any]) -> list[int]:
    expected_batches = [1, 4, 8, 16, 32, 64]
    if result.get("schema") != "s14-cp-e-batch-cert-v1" \
            or result.get("status") != "BATCH_CERT_SCREEN_COMPLETE":
        raise EvidenceError("batch-screen schema or status mismatch")
    if result.get("batches_requested") != expected_batches:
        raise EvidenceError("batch-screen request set mismatch")
    if result.get("route") != {"op15": [0, 8], "op12": [8, 12], "server": [12, 48]}:
        raise EvidenceError("route mismatch")
    validate_artifacts(result.get("artifacts"))
    rows = result.get("rows")
    if type(rows) is not list or len(rows) != len(expected_batches):
        raise EvidenceError("expected six batch rows")
    if [row.get("batch") for row in rows if type(row) is dict] != expected_batches:
        raise EvidenceError("batch rows are incomplete or reordered")
    certified = []
    for row in rows:
        batch = row["batch"]
        validate_thermal(row.get("thermal_start"), 60_000)
        validate_thermal(row.get("thermal_end"), 85_000)
        path_value = row.get("result_path")
        digest_value = row.get("result_sha256")
        nested = None
        if type(path_value) is str and Path(path_value).exists():
            if sha256_file(Path(path_value)) != digest_value:
                raise EvidenceError(f"nested result digest mismatch: B{batch}")
            nested = load_json(Path(path_value))
        elif digest_value is not None:
            raise EvidenceError(f"digest without nested result: B{batch}")
        if row.get("certified") is True:
            if row.get("returncode") != 0 or type(nested) is not dict \
                    or nested.get("certified") is not True \
                    or nested.get("status") != "CERTIFIED_3DEVICE_TOKEN_CORRECT_MECHANICS" \
                    or nested.get("batch") != batch:
                raise EvidenceError(f"certified row lacks a certified nested result: B{batch}")
            checks = nested.get("checks")
            if type(checks) is not dict or not checks or not all(value is True for value in checks.values()):
                raise EvidenceError(f"nested checks failed: B{batch}")
            if row.get("checks") != checks or row.get("result_status") != nested.get("status"):
                raise EvidenceError(f"copied nested fields mismatch: B{batch}")
            certified.append(batch)
        elif row.get("returncode") == 0 and type(nested) is dict and nested.get("certified") is True:
            raise EvidenceError(f"passing nested result labeled ineligible: B{batch}")
    if result.get("certified_batches") != certified:
        raise EvidenceError("certified-batch summary mismatch")
    expected_max = max(certified) if certified else None
    if result.get("largest_certified_batch") != expected_max:
        raise EvidenceError("largest-certified-batch mismatch")
    return certified


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    try:
        certified = validate(load_json(args.result))
    except (EvidenceError, CPEV.EvidenceError, TypeError, ValueError) as exc:
        print(f"INVALID: {exc}")
        return 2
    print("VALID_BATCH_SCREEN certified=" + ",".join(str(value) for value in certified))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
