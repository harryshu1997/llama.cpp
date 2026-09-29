#!/usr/bin/env python3
"""Evaluate compiled route contracts against one immutable runtime snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import (  # noqa: E402
    RequestSemantics,
    RouteRuntimeContract,
    RuntimeGateError,
    RuntimeSnapshot,
    evaluate_runtime_gate,
)


OUTPUT_SCHEMA = "s42-runtime-gate-audit-v1"


class AuditError(ValueError):
    pass


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def load_object(path: Path) -> tuple[Mapping[str, Any], str]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read {path}: {exc}") from exc
    if type(value) is not dict:
        raise AuditError(f"{path} must contain an object")
    return value, "sha256:" + hashlib.sha256(raw).hexdigest()


def request_semantics(value: object) -> RequestSemantics:
    if value is None:
        return RequestSemantics()
    if type(value) is not dict:
        raise AuditError("request semantics must be an object")
    allowed = {
        "cancelled",
        "cancellation_required",
        "kv_owner",
        "kv_migration_required",
        "context_shift_required",
        "full_logits_required",
        "grammar_required",
        "sampler_location",
        "speculative_decode",
    }
    unknown = set(value) - allowed
    if unknown:
        raise AuditError("request semantics contain unknown fields")
    try:
        result = RequestSemantics(**value)
        result.validate()
    except (TypeError, RuntimeGateError) as exc:
        raise AuditError(f"invalid request semantics: {exc}") from exc
    return result


def audit(
    contracts_document: Mapping[str, Any],
    contracts_sha256: str,
    snapshot_document: Mapping[str, Any],
    snapshot_sha256: str,
    semantics: RequestSemantics,
    now_us: int,
) -> Mapping[str, Any]:
    if contracts_document.get("schema") != "s42-i3-runtime-gate-contracts-v1":
        raise AuditError("contract document schema mismatch")
    raw_routes = contracts_document.get("routes")
    if type(raw_routes) is not dict or not raw_routes:
        raise AuditError("contract document routes are missing")
    try:
        snapshot = RuntimeSnapshot.from_json(snapshot_document)
        routes = {
            route_id: RouteRuntimeContract.from_json(value)
            for route_id, value in raw_routes.items()
        }
    except RuntimeGateError as exc:
        raise AuditError(str(exc)) from exc

    results: dict[str, object] = {}
    for route_id, contract in sorted(routes.items()):
        receipt = evaluate_runtime_gate(contract, semantics, snapshot, now_us)
        results[route_id] = {
            "admitted": receipt.admitted,
            "checked_resources": list(receipt.checked_resources),
            "reason": receipt.reason,
        }
    result: dict[str, Any] = {
        "contracts_sha256": contracts_sha256,
        "now_us": now_us,
        "routes": results,
        "schema": OUTPUT_SCHEMA,
        "snapshot": {
            "epoch_key": snapshot.epoch_key,
            "file_sha256": snapshot_sha256,
            "generation": snapshot.generation,
            "snapshot_id": snapshot.snapshot_id,
        },
        "status": "PASS",
    }
    result["audit_sha256"] = "sha256:" + hashlib.sha256(
        canonical_bytes(result)
    ).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contracts", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--now-us", type=int, required=True)
    parser.add_argument("--request-semantics", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.now_us < 0:
        parser.error("now-us must be nonnegative")
    if args.output is not None and args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        contracts, contracts_sha256 = load_object(args.contracts)
        snapshot, snapshot_sha256 = load_object(args.snapshot)
        raw_semantics = None
        if args.request_semantics is not None:
            raw_semantics, _ = load_object(args.request_semantics)
        result = audit(
            contracts,
            contracts_sha256,
            snapshot,
            snapshot_sha256,
            request_semantics(raw_semantics),
            args.now_us,
        )
        payload = canonical_bytes(result)
        if args.output is None:
            print(payload.decode("ascii"), end="")
        else:
            args.output.write_bytes(payload)
    except (AuditError, RuntimeGateError) as exc:
        parser.exit(2, f"runtime gate audit failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
