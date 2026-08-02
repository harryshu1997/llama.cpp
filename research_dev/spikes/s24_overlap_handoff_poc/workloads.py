#!/usr/bin/env python3
"""Build the three S24 fixed-diamond workloads from pinned inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
S23 = HERE.parent / "s23_dense_trace_runtime"
if str(S23) not in sys.path:
    sys.path.insert(0, str(S23))

from dense_trace import validate as validate_s23


SCHEMA = "s24-fixed-diamond-trace-v1"
MECHANICS_SCOPE = (
    "REAL_ARRIVALS_AND_OBSERVED_DEMAND_"
    "SYNTHETIC_FIXED_TOKEN_PRIORITY_SLO_ONE_TOKEN_FOUR_STEP_MECHANICS_ONLY"
)
OBSERVED_SCOPE = (
    "REAL_ARRIVALS_AND_OBSERVED_LENGTHS_"
    "SYNTHETIC_FIXED_TOKEN_PRIORITY_SLO_CONTEXT_600"
)
DETERMINISTIC_SCOPE = "SYNTHETIC_THREE_CLASS_FIXED_ROUTE_PROOF"
ROUTE_BY_PRIORITY = {0: "R0", 1: "R1", 2: "R2"}
SYNTHETIC_TOKEN = 2
OBSERVED_CONTEXT = 600


class WorkloadError(RuntimeError):
    pass


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def content_digest(value: dict[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("trace_hash", None)
    return "sha256:" + hashlib.sha256(canonical_bytes(unsigned)).hexdigest()


def load_s23(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkloadError(f"cannot load S23 trace: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkloadError("S23 trace must be an object")
    validate_s23(value, verify_source=False)
    return value


def _request(
    request_id: int,
    arrival_us: int,
    priority: int,
    slo_us: int,
    input_tokens: int,
    output_steps: int,
    observed_input_tokens: int,
    observed_output_tokens: int,
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "arrival_us": arrival_us,
        "priority": priority,
        "slo_us": slo_us,
        "route_hint": ROUTE_BY_PRIORITY[priority],
        "input_tokens": input_tokens,
        "output_steps": output_steps,
        "synthetic_token": SYNTHETIC_TOKEN,
        "observed_input_tokens": observed_input_tokens,
        "observed_output_tokens": observed_output_tokens,
    }


def _trace(
    scope: str,
    source: dict[str, Any],
    requests: list[dict[str, Any]],
    context_limit: int,
) -> dict[str, Any]:
    value = {
        "schema": SCHEMA,
        "scope": scope,
        "source": source,
        "context_limit": context_limit,
        "route_by_priority": {
            str(priority): route_id
            for priority, route_id in ROUTE_BY_PRIORITY.items()
        },
        "requests": requests,
    }
    value["trace_hash"] = content_digest(value)
    validate(value)
    return value


def deterministic_three_class() -> dict[str, Any]:
    requests = []
    synthetic_slos = {
        0: 2_000_000,
        1: 8_000_000,
        2: 15_000_000,
    }
    request_id = 24_001
    for priority in range(3):
        for _index in range(4):
            requests.append(_request(
                request_id=request_id,
                arrival_us=0,
                priority=priority,
                slo_us=synthetic_slos[priority],
                input_tokens=1,
                output_steps=4,
                observed_input_tokens=1,
                observed_output_tokens=4,
            ))
            request_id += 1
    return _trace(
        DETERMINISTIC_SCOPE,
        {
            "provenance": "s24-authored",
            "claims": "route-mechanics-and-convergence-only",
        },
        requests,
        512,
    )


def dense_mechanics(s23: dict[str, Any]) -> dict[str, Any]:
    requests = [
        _request(
            request_id=row["request_id"],
            arrival_us=row["arrival_us"],
            priority=row["priority"],
            slo_us=row["slo_us"],
            input_tokens=row["execution_input_tokens"],
            output_steps=row["execution_steps"],
            observed_input_tokens=row["observed_input_tokens"],
            observed_output_tokens=row["observed_output_tokens"],
        )
        for row in s23["requests"]
    ]
    return _trace(
        MECHANICS_SCOPE,
        {
            "provenance": "s23-burstgpt-dense-60",
            "s23_trace_hash": s23["trace_hash"],
            "claims": "arrival-pressure-and-runtime-mechanics-only",
        },
        requests,
        512,
    )


def observed_context_cohort(s23: dict[str, Any]) -> dict[str, Any]:
    selected = [
        row for row in s23["requests"]
        if (
            row["observed_input_tokens"] + row["observed_output_tokens"]
            <= OBSERVED_CONTEXT
        )
    ]
    if len(selected) != 28:
        raise WorkloadError(
            f"context-{OBSERVED_CONTEXT} cohort has {len(selected)} rows, not 28"
        )
    requests = [
        _request(
            request_id=row["request_id"],
            arrival_us=row["arrival_us"],
            priority=row["priority"],
            slo_us=row["slo_us"],
            input_tokens=row["observed_input_tokens"],
            output_steps=row["observed_output_tokens"],
            observed_input_tokens=row["observed_input_tokens"],
            observed_output_tokens=row["observed_output_tokens"],
        )
        for row in selected
    ]
    return _trace(
        OBSERVED_SCOPE,
        {
            "provenance": "s23-burstgpt-dense-60",
            "s23_trace_hash": s23["trace_hash"],
            "selection_rule": (
                "preserve-source-order-where-observed-input-plus-output-"
                "is-at-most-600"
            ),
            "claims": "observed-length-runtime-with-synthetic-token-values",
        },
        requests,
        OBSERVED_CONTEXT,
    )


def validate(value: dict[str, Any]) -> None:
    expected_top = {
        "schema",
        "scope",
        "source",
        "context_limit",
        "route_by_priority",
        "requests",
        "trace_hash",
    }
    if set(value) != expected_top or value.get("schema") != SCHEMA:
        raise WorkloadError("trace top-level contract mismatch")
    if value.get("trace_hash") != content_digest(value):
        raise WorkloadError("trace content digest mismatch")
    if value.get("scope") not in {
        DETERMINISTIC_SCOPE, MECHANICS_SCOPE, OBSERVED_SCOPE,
    }:
        raise WorkloadError("trace scope is invalid")
    context_limit = value.get("context_limit")
    if (
        isinstance(context_limit, bool)
        or not isinstance(context_limit, int)
        or context_limit <= 0
    ):
        raise WorkloadError("trace context limit is invalid")
    if value.get("route_by_priority") != {
        "0": "R0", "1": "R1", "2": "R2",
    }:
        raise WorkloadError("priority route binding is invalid")
    requests = value.get("requests")
    if not isinstance(requests, list) or not requests:
        raise WorkloadError("trace requests must be nonempty")
    expected_request = {
        "request_id",
        "arrival_us",
        "priority",
        "slo_us",
        "route_hint",
        "input_tokens",
        "output_steps",
        "synthetic_token",
        "observed_input_tokens",
        "observed_output_tokens",
    }
    identities = set()
    order = []
    for row in requests:
        if not isinstance(row, dict) or set(row) != expected_request:
            raise WorkloadError("trace request contract mismatch")
        for key in (
            "request_id", "arrival_us", "priority", "slo_us",
            "input_tokens", "output_steps", "synthetic_token",
            "observed_input_tokens", "observed_output_tokens",
        ):
            if isinstance(row[key], bool) or not isinstance(row[key], int):
                raise WorkloadError(f"{key} must be an integer")
        if (
            row["request_id"] <= 0
            or row["arrival_us"] < 0
            or row["priority"] not in ROUTE_BY_PRIORITY
            or row["slo_us"] <= 0
            or row["input_tokens"] <= 0
            or row["output_steps"] <= 0
            or row["synthetic_token"] != SYNTHETIC_TOKEN
            or row["observed_input_tokens"] <= 0
            or row["observed_output_tokens"] <= 0
        ):
            raise WorkloadError("trace request value is out of range")
        if row["route_hint"] != ROUTE_BY_PRIORITY[row["priority"]]:
            raise WorkloadError("route hint differs from priority class")
        if row["input_tokens"] + row["output_steps"] > context_limit:
            raise WorkloadError("request exceeds the context envelope")
        if row["request_id"] in identities:
            raise WorkloadError("request identity is duplicated")
        identities.add(row["request_id"])
        order.append((row["arrival_us"], row["request_id"]))
    if order != sorted(order):
        raise WorkloadError("trace requests are not in source arrival order")
    if value["scope"] == MECHANICS_SCOPE and (
        len(requests) != 60
        or [sum(row["arrival_us"] == arrival for row in requests)
            for arrival in (0, 1_000_000, 2_000_000)] != [21, 17, 22]
        or any(
            row["input_tokens"] != 1 or row["output_steps"] != 4
            for row in requests
        )
    ):
        raise WorkloadError("dense mechanics trace contract mismatch")
    if value["scope"] == OBSERVED_SCOPE and (
        len(requests) != 28
        or context_limit != OBSERVED_CONTEXT
        or any(
            row["input_tokens"] != row["observed_input_tokens"]
            or row["output_steps"] != row["observed_output_tokens"]
            for row in requests
        )
    ):
        raise WorkloadError("observed context cohort contract mismatch")


def write(path: Path, value: dict[str, Any]) -> None:
    path.write_bytes(canonical_bytes(value))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--s23",
        type=Path,
        default=S23 / "burstgpt-dense-60.json",
    )
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    s23 = load_s23(args.s23)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "deterministic-three-class.json": deterministic_three_class(),
        "burstgpt-dense-mechanics.json": dense_mechanics(s23),
        "burstgpt-observed-context-600.json": observed_context_cohort(s23),
    }
    for name, value in outputs.items():
        write(args.output_dir / name, value)
    print(json.dumps({
        name: value["trace_hash"] for name, value in outputs.items()
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WorkloadError as exc:
        print(json.dumps({
            "verdict": "FAIL", "error": str(exc),
        }, sort_keys=True))
        raise SystemExit(2)
