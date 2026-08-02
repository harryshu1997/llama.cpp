#!/usr/bin/env python3
"""Build the bounded CP0-R1 V2.6 A_ONLY evidence contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
S39 = HERE.parent
V24 = S39 / "v24_readiness"

DEFAULT_OUTPUT = HERE / "CP0_R1_EVIDENCE_CONTRACT_V2_6.json"
CANDIDATE = S39 / "CP0_R1_CANDIDATE.json"
CORPUS = S39 / "CP0_R1_MMLU64_CORPUS_V2_2.jsonl"
TOKEN_HISTORY = (
    V24 / "results" / "prephase_20260726T0915Z" / "token-history.json"
)
TOKENIZER_PLAN = (
    V24 / "results" / "prephase_20260726T0915Z" / "tokenizer-plan.json"
)
V24_CONTRACT = V24 / "CP0_R1_EVIDENCE_CONTRACT_V2_4.json"

INNER_V24 = {
    "authority": V24 / "cp0_r1_evidence_v24.py",
    "common": V24 / "v24_common.py",
    "contract": V24_CONTRACT,
    "contract_builder": V24 / "build_contract_v24.py",
}
V26_PROGRAMS = {
    "authority": HERE / "cp0_r1_evidence_v26.py",
    "capture_execution_receipt": HERE / "capture_execution_receipt_v1.py",
    "capture_execution_schema": HERE / "CAPTURE_EXECUTION_RECEIPT_V1.schema.json",
    "contract_builder": Path(__file__).resolve(),
    "managed_plan_gate": HERE / "managed_plan_gate_v1.py",
    "runtime_inventory": HERE / "runtime_inventory_v26.py",
}
CAPTURE_PRODUCERS = {
    "artifact_root": (
        V24 / "desktop_deployment_v1" / "artifact_root_capture_v1.py"
    ),
    "fast_fresh_readiness": (
        V24 / "desktop_deployment_v1" / "fast_fresh_capture_v1.py"
    ),
}
REQUIRED_ARTIFACT_ROLES = sorted(
    {
        "capture.execution_plan",
        "capture.artifact_root.receipt",
        "capture.fast_fresh_readiness.receipt",
        "inner.raw_manifest",
        "inner.token_history",
        "phase.lock",
        "runtime.inventory",
    }
)


class ContractError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"E_DUPLICATE_KEY: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ContractError(f"E_JSON_NUMBER: {value}")


def _reject_float(value: str) -> None:
    raise ContractError(f"E_JSON_FLOAT: {value}")


def canonical_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, UnicodeEncodeError) as error:
        raise ContractError("E_CANONICAL") from error


def read_canonical(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ContractError(f"E_READ: {path}: {error}") from error
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
            parse_float=_reject_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"E_JSON: {path}: {error}") from error
    require(type(value) is dict, f"E_TYPE: {path}")
    require(canonical_bytes(value) == raw, f"E_CANONICAL: {path}")
    return value, raw


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ContractError(f"E_READ: {path}: {error}") from error
    require(bool(raw), f"E_EMPTY: {path}")
    try:
        relative = path.resolve().relative_to(S39.resolve())
    except ValueError as error:
        raise ContractError(f"E_SOURCE_OUTSIDE_S39: {path}") from error
    return {
        "bytes": len(raw),
        "path": str(relative),
        "sha256": sha256(raw),
    }


def _count_corpus_rows(raw: bytes) -> int:
    rows = raw.splitlines(keepends=True)
    require(len(rows) == 64, "E_CORPUS_ROWS")
    require(all(row.endswith(b"\n") for row in rows), "E_CORPUS_NEWLINE")
    for index, row in enumerate(rows):
        try:
            value = json.loads(
                row.decode("ascii"),
                object_pairs_hook=_strict_object,
                parse_constant=_reject_constant,
                parse_float=_reject_float,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ContractError(f"E_CORPUS_JSON: {index}: {error}") from error
        require(type(value) is dict, f"E_CORPUS_TYPE: {index}")
        require(canonical_bytes(value) == row, f"E_CORPUS_CANONICAL: {index}")
    return len(rows)


def build_contract() -> dict[str, Any]:
    candidate, candidate_raw = read_canonical(CANDIDATE)
    v24, v24_raw = read_canonical(V24_CONTRACT)
    _, history_raw = read_canonical(TOKEN_HISTORY)
    _, tokenizer_raw = read_canonical(TOKENIZER_PLAN)
    corpus_raw = CORPUS.read_bytes()
    corpus_rows = _count_corpus_rows(corpus_raw)

    require(candidate.get("schema") == "s39-cp0-r1-candidate-v1", "E_CANDIDATE")
    require(v24.get("schema") == "s39-cp0-r1-evidence-contract-v2.4", "E_V24")
    models = [
        value
        for value in candidate.get("models", [])
        if type(value) is dict and value.get("slot") == "A"
    ]
    require(len(models) == 1, "E_MODEL_A")
    model = models[0]
    require(model.get("model_id") == "qwen3-14b-q4_k_m", "E_MODEL_A_ID")
    require(
        v24["candidate_lock"]["sha256"] == sha256(candidate_raw),
        "E_V24_CANDIDATE",
    )
    require(
        v24["quality_corpus"]["sha256"] == sha256(corpus_raw),
        "E_V24_CORPUS",
    )

    return {
        "candidate_lock": {
            "bytes": len(candidate_raw),
            "model": model,
            "sha256": sha256(candidate_raw),
        },
        "claim_boundary": {
            "acquisition_authorized": True,
            "authorized_phase": "A_ONLY",
            "b_only_authorized": False,
            "energy_claim": "NONE",
            "formal_claim": "NONE",
            "legacy_status_is_not_evidence": True,
            "pair_authorized": False,
            "raw_predicates_must_be_recomputed": True,
        },
        "composition": {
            "capture_producers": {
                name: artifact(path)
                for name, path in sorted(CAPTURE_PRODUCERS.items())
            },
            "inner_v24": {
                name: artifact(path)
                for name, path in sorted(INNER_V24.items())
            },
            "v26": {
                name: artifact(path)
                for name, path in sorted(V26_PROGRAMS.items())
            },
        },
        "devices": v24["devices"],
        "model_route_lock": {
            "geometry": v24["model_geometry"][model["model_id"]],
            "route": v24["incumbent_route_lock"],
        },
        "phase_protocol": {
            "artifact_root_scope": "PRE_REBOOT_OUTSIDE_PHASE",
            "authorized_phases": ["A_ONLY"],
            "clock_id": "HOST_MONOTONIC_RAW",
            "phase_id_prefix": "cp0-r1-v26-a-only-",
            "phase_local_roles": [
                "capture.fast_fresh_readiness.receipt",
                "inner.raw_manifest",
                "phase.lock",
                "runtime.inventory",
            ],
            "required_artifact_roles": REQUIRED_ARTIFACT_ROLES,
        },
        "quality": {
            "continuation_tokens_per_request": 8,
            "corpus": {
                "bytes": len(corpus_raw),
                "path": str(CORPUS.relative_to(S39)),
                "rows": corpus_rows,
                "sha256": sha256(corpus_raw),
            },
            "cuda_minimum_correct_items": 25,
            "items": 64,
        },
        "raw_predicate": {
            "expected_result_schema": (
                "s39-cp0-r1-raw-predicate-result-v2.4"
            ),
            "expected_status": "MODEL_A_QUALIFICATION_PASS",
            "manifest_name": "EVIDENCE_BUNDLE_V2_4.json",
            "v24_contract_sha256": sha256(v24_raw),
        },
        "schema": "s39-cp0-r1-evidence-contract-v2.6",
        "scope": "A_ONLY_OUTER_EVIDENCE_AUTHORITY",
        "serving_envelope": v24["serving_envelope"],
        "static_inputs": {
            "candidate": artifact(CANDIDATE),
            "corpus": artifact(CORPUS),
            "token_history": artifact(TOKEN_HISTORY),
            "tokenizer_plan": artifact(TOKENIZER_PLAN),
        },
        "status": "FROZEN_BEFORE_A_ONLY_ACQUISITION",
        "version": "2.6",
    }


def write_exclusive(path: Path, value: Any) -> None:
    raw = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o644,
    )
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        write_exclusive(args.output, build_contract())
    except (ContractError, OSError) as error:
        print(f"V2_6_CONTRACT_REFUSED: {error}", file=sys.stderr)
        return 2
    print(f"V2_6_CONTRACT_WRITTEN: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
