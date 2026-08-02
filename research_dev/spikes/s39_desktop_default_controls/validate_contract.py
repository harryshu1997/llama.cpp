#!/usr/bin/env python3
"""Validate the frozen stock-default control before acquisition."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "s39_desktop_swap_baseline"


class ContractError(RuntimeError):
    pass


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract() -> dict[str, Any]:
    value = json.loads(
        (HERE / "DEFAULT_CONTROL_CONTRACT.json").read_text(encoding="ascii")
    )
    if type(value) is not dict \
            or value.get("schema") != "s39-stock-default-controls-v1":
        raise ContractError("invalid contract identity")
    return value


def validate() -> dict[str, Any]:
    contract = load_contract()
    source = contract["source"]
    bindings = (
        (
            BASE / "DESKTOP_BASELINE_CONTRACT.json",
            source["cp0d_contract_sha256"],
        ),
        (BASE / "DESKTOP_REQUESTS.jsonl", source["requests_sha256"]),
        (BASE / "DESKTOP_SWITCHES.jsonl", source["switches_sha256"]),
    )
    for path, expected in bindings:
        actual = digest_file(path)
        if actual != expected:
            raise ContractError(f"digest mismatch: {path}")
    requests = [
        json.loads(line)
        for line in (BASE / "DESKTOP_REQUESTS.jsonl").read_text(
            encoding="ascii"
        ).splitlines()
    ]
    switches = [
        json.loads(line)
        for line in (BASE / "DESKTOP_SWITCHES.jsonl").read_text(
            encoding="ascii"
        ).splitlines()
    ]
    if len(requests) != source["request_count"] \
            or len(switches) != source["switch_count"]:
        raise ContractError("trace count mismatch")
    if [row["request_index"] for row in requests] != list(range(74)) \
            or [row["intent_index"] for row in switches] != list(range(9)):
        raise ContractError("trace identity is not contiguous")
    required = contract["required_server_arguments"]
    forbidden = contract["tuning_arguments_forbidden"]
    if len(required) != len(set(required)) \
            or len(forbidden) != len(set(forbidden)) \
            or set(required) & set(forbidden):
        raise ContractError("server argument sets are invalid")
    controls = contract["controls"]
    if [row["control_id"] for row in controls] != [
            "STOCK_DEFAULT_SWAP_WARM",
            "STOCK_DEFAULT_SWAP_COLD",
            "STOCK_DEFAULT_DUAL_WARM"]:
        raise ContractError("control order mismatch")
    if any(row.get("repetitions") != 3 for row in controls):
        raise ContractError("control repetition mismatch")
    return {
        "contract_sha256": digest_file(
            HERE / "DEFAULT_CONTROL_CONTRACT.json"
        ),
        "request_count": len(requests),
        "status": "STOCK_DEFAULT_CONTRACT_VALID",
        "switch_count": len(switches),
    }


if __name__ == "__main__":
    try:
        print(json.dumps(validate(), sort_keys=True))
    except (ContractError, KeyError, TypeError, ValueError) as exc:
        print(f"DEFAULT_CONTRACT_ERROR: {exc}")
        raise SystemExit(2)
