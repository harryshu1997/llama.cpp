#!/usr/bin/env python3
"""Thin V2.6 outer authority for the bounded Qwen3-14B A_ONLY phase."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
S39 = HERE.parent
V24 = S39 / "v24_readiness"
SCHEMA = "s39-cp0-r1-evidence-result-v2.6"
PHASE = "A_ONLY"


class EvidenceError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def exact(value: Any, expected: Any, field: str) -> None:
    require(type(value) is type(expected) and value == expected, f"E_VALUE: {field}")


def canonical(value: Any) -> bytes:
    try:
        return (json.dumps(value, ensure_ascii=True, sort_keys=True,
                           separators=(",", ":")) + "\n").encode("ascii")
    except (TypeError, UnicodeEncodeError) as error:
        raise EvidenceError("E_CANONICAL") from error


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"E_DUPLICATE_KEY: {key}")
        result[key] = value
    return result


def read_canonical(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"), object_pairs_hook=_strict_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"E_JSON: {path}") from error
    require(type(value) is dict, f"E_TYPE: {path}")
    exact(canonical(value), raw, f"E_CANONICAL: {path}")
    return value, raw


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"E_IMPORT: {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_contract(path: Path) -> tuple[dict[str, Any], bytes]:
    value, raw = read_canonical(path)
    exact(value.get("schema"), "s39-cp0-r1-evidence-contract-v2.6", "contract.schema")
    exact(value.get("version"), "2.6", "contract.version")
    exact(value.get("claim_boundary", {}).get("authorized_phase"), PHASE, "contract.phase")
    exact(value.get("claim_boundary", {}).get("b_only_authorized"), False, "contract.b_only")
    exact(value.get("claim_boundary", {}).get("pair_authorized"), False, "contract.pair")
    require(value["claim_boundary"]["raw_predicates_must_be_recomputed"] is True, "E_RAW_RECOMPUTE")
    return value, raw


def validate_composition(contract: dict[str, Any]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for group_name, group in contract.get("composition", {}).items():
        require(type(group) is dict, f"E_COMPOSITION: {group_name}")
        for name, descriptor in group.items():
            require(type(descriptor) is dict, f"E_COMPOSITION: {group_name}.{name}")
            path_text = descriptor.get("path")
            require(type(path_text) is str, f"E_COMPOSITION_PATH: {group_name}.{name}")
            path = S39 / path_text
            try:
                raw = path.read_bytes()
            except OSError as error:
                raise EvidenceError(f"E_COMPOSITION_READ: {group_name}.{name}") from error
            exact(len(raw), descriptor.get("bytes"), f"E_COMPOSITION_BYTES: {group_name}.{name}")
            exact(sha256(raw), descriptor.get("sha256"), f"E_COMPOSITION_SHA: {group_name}.{name}")
            paths[f"{group_name}.{name}"] = str(path)
    return paths


def validate_inventory(path: Path, contract: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
    value, raw = read_canonical(path)
    inventory = _load("s39_v26_inventory", HERE / "runtime_inventory_v26.py")
    try:
        inventory.validate_runtime_inventory(value)
    except Exception as error:
        raise EvidenceError(f"E_INVENTORY: {error}") from error
    body = value["inventory"]
    exact(body["phase"], PHASE, "inventory.phase")
    require(
        body["phase_id"].startswith(contract["phase_protocol"]["phase_id_prefix"]),
        "E_INVENTORY_PHASE_ID",
    )
    require(type(body["components"]) is list and body["components"], "E_INVENTORY_COMPONENTS")
    return value, raw


def validate_capture_receipt(
    path: Path,
    *,
    contract: dict[str, Any],
    expected_kind: str,
    expected_role: str,
    phase_id: str,
    lock_ns: int,
) -> tuple[dict[str, Any], bytes]:
    value, raw = read_canonical(path)
    exact(value.get("capture_kind"), expected_kind, f"receipt.kind: {path}")
    exact(value.get("producer_role"), expected_role, f"receipt.role: {path}")
    exact(value.get("phase"), PHASE, f"receipt.phase: {path}")
    exact(value.get("phase_id"), phase_id, f"receipt.phase_id: {path}")
    exact(value.get("contract_sha256"), sha256(canonical(contract)), f"receipt.contract: {path}")
    producer_key = "artifact_root" if expected_kind == "artifact_root" else "fast_fresh_readiness"
    expected_source = contract["composition"]["capture_producers"][producer_key]
    exact(value["source"]["path"].replace("\\", "/").endswith(expected_source["path"]), True, f"receipt.source.path: {path}")
    exact(value["source"]["sha256"], expected_source["sha256"], f"receipt.source.sha256: {path}")
    require(value["started_monotonic_ns"] >= lock_ns, f"E_RECEIPT_BEFORE_LOCK: {path}")
    require(value["completed_monotonic_ns"] > value["started_monotonic_ns"], f"E_RECEIPT_INTERVAL: {path}")
    receipt = _load("s39_v26_receipt", HERE / "capture_execution_receipt_v1.py")
    try:
        receipt.validate_receipt(
            value,
            source_path=Path(value["source"]["path"]),
            result_path=Path(value["result"]["path"]),
        )
    except Exception as error:
        raise EvidenceError(f"E_RECEIPT: {error}") from error
    return value, raw


def authorize_a_only(
    *,
    contract_path: Path,
    inventory_path: Path,
    phase_lock_path: Path,
    artifact_receipt_path: Path,
    fresh_receipt_path: Path,
    raw_bundle_root: Path,
    v24_chain_kwargs: dict[str, Path],
) -> dict[str, Any]:
    contract, contract_raw = load_contract(contract_path)
    composition = validate_composition(contract)
    inventory, inventory_raw = validate_inventory(inventory_path, contract)
    lock, lock_raw = read_canonical(phase_lock_path)
    exact(lock.get("phase"), PHASE, "phase_lock.phase")
    phase_id = lock.get("phase_id")
    require(type(phase_id) is str and phase_id.startswith(contract["phase_protocol"]["phase_id_prefix"]), "E_PHASE_ID")
    lock_ns = lock.get("event_ns")
    require(type(lock_ns) is int and lock_ns > 0, "E_PHASE_LOCK_TIME")
    exact(lock.get("phase_id"), inventory["inventory"]["phase_id"], "phase_lock.inventory_phase")
    artifact_receipt, artifact_raw = validate_capture_receipt(
        artifact_receipt_path,
        contract=contract,
        expected_kind="artifact_root",
        expected_role="capture.artifact_root",
        phase_id=phase_id,
        lock_ns=0,
    )
    fresh_receipt, fresh_raw = validate_capture_receipt(
        fresh_receipt_path,
        contract=contract,
        expected_kind="fast_fresh_readiness",
        expected_role="capture.fast_fresh_readiness",
        phase_id=phase_id,
        lock_ns=lock_ns,
    )
    require(artifact_receipt["completed_monotonic_ns"] <= lock_ns, "E_ARTIFACT_AFTER_LOCK")
    require(lock_ns < fresh_receipt["started_monotonic_ns"], "E_FRESH_BEFORE_LOCK")
    v24 = _load("s39_v24_authority", V24 / "cp0_r1_evidence_v24.py")
    try:
        result = v24.authorize_a_only(
            bundle_root=raw_bundle_root,
            **v24_chain_kwargs,
        )
    except Exception as error:
        raise EvidenceError(f"E_V24_RAW_AUTHORITY: {error}") from error
    exact(result.get("status"), "MODEL_A_QUALIFICATION_PASS_V2_4", "v24.status")
    return {
        "artifact_receipt_sha256": sha256(artifact_raw),
        "composition": composition,
        "contract_sha256": sha256(contract_raw),
        "fresh_receipt_sha256": sha256(fresh_raw),
        "inventory_sha256": sha256(inventory_raw),
        "phase_id": phase_id,
        "schema": SCHEMA,
        "status": "MODEL_A_QUALIFICATION_PASS_V2_6",
        "v24_result_sha256": sha256(canonical(result)),
    }


def main() -> int:
    print("V2_6_AUTHORITY_REQUIRES_EXPLICIT_A_ONLY_INPUTS", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
