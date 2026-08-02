#!/usr/bin/env python3
"""Build the frozen S36 execution proxy from the S23 dense trace."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "s36-dynamic-cut-trace-v1"
SOURCE_SCHEMA = "s23-burstgpt-dense-trace-v1"
PROMPT_TOKENS = [2, 2, 2, 2]


class TraceError(RuntimeError):
    pass


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TraceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicates,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TraceError(f"cannot read source trace: {exc}") from exc
    if type(value) is not dict:
        raise TraceError("source trace must be an object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def build_trace(source_path: Path) -> dict[str, Any]:
    source = load_json(source_path)
    if source.get("schema") != SOURCE_SCHEMA:
        raise TraceError("unexpected source trace schema")
    rows = source.get("requests")
    if type(rows) is not list or len(rows) != 60:
        raise TraceError("source trace must contain 60 requests")
    requests = []
    seen: set[int] = set()
    previous_key: tuple[int, int] | None = None
    for row in rows:
        if type(row) is not dict:
            raise TraceError("source request must be an object")
        required_ints = (
            "arrival_us", "request_id", "priority", "slo_us",
            "observed_input_tokens", "observed_output_tokens",
            "execution_steps",
        )
        if any(type(row.get(key)) is not int for key in required_ints):
            raise TraceError("source request has an invalid integer field")
        request_id = row["request_id"]
        key = (row["arrival_us"], request_id)
        if (
            request_id < 0
            or request_id in seen
            or row["arrival_us"] < 0
            or not 0 <= row["priority"] <= 2
            or row["slo_us"] <= 0
            or row["execution_steps"] != 4
            or row["observed_input_tokens"] <= 0
            or row["observed_output_tokens"] <= 0
            or (previous_key is not None and key < previous_key)
        ):
            raise TraceError("source request violates the frozen contract")
        seen.add(request_id)
        previous_key = key
        requests.append({
            "request_id": request_id,
            "arrival_us": row["arrival_us"],
            "priority": row["priority"],
            "slo_us": row["slo_us"],
            "prompt_tokens": list(PROMPT_TOKENS),
            "output_steps": 4,
            "observed_input_tokens": row["observed_input_tokens"],
            "observed_output_tokens": row["observed_output_tokens"],
            "source_event_id": row.get("event_id"),
            "source_row_sha256": row.get("source_row_sha256"),
        })
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "scope": (
            "OBSERVED_ARRIVALS_AND_DEMAND_SYNTHETIC_PAYLOAD_PRIORITY_SLO_"
            "EXECUTION_PROXY"
        ),
        "source": {
            "path": str(source_path),
            "file_sha256": sha256_file(source_path),
            "trace_hash": source.get("trace_hash"),
            "schema": SOURCE_SCHEMA,
        },
        "execution_proxy": {
            "prompt_token_id": 2,
            "prompt_tokens": 4,
            "output_steps": 4,
            "priority_slo_provenance": source.get("synthetic_sidecar", {}).get(
                "provenance"
            ),
        },
        "requests": requests,
    }
    report["trace_hash"] = "sha256:" + hashlib.sha256(
        canonical_bytes(report)
    ).hexdigest()
    return report


def validate_built_trace(value: object) -> dict[str, Any]:
    if type(value) is not dict or value.get("schema") != SCHEMA:
        raise TraceError("S36 trace schema mismatch")
    supplied_hash = value.get("trace_hash")
    if type(supplied_hash) is not str or not supplied_hash.startswith("sha256:"):
        raise TraceError("S36 trace hash is missing")
    unhashed = dict(value)
    del unhashed["trace_hash"]
    expected_hash = "sha256:" + hashlib.sha256(canonical_bytes(unhashed)).hexdigest()
    if supplied_hash != expected_hash:
        raise TraceError("S36 trace hash mismatch")
    rows = value.get("requests")
    if type(rows) is not list or len(rows) != 60:
        raise TraceError("S36 trace must contain 60 requests")
    seen: set[int] = set()
    previous: tuple[int, int] | None = None
    for row in rows:
        if type(row) is not dict:
            raise TraceError("S36 request must be an object")
        request_id = row.get("request_id")
        arrival_us = row.get("arrival_us")
        priority = row.get("priority")
        slo_us = row.get("slo_us")
        if (
            type(request_id) is not int
            or request_id <= 0
            or request_id in seen
            or type(arrival_us) is not int
            or arrival_us < 0
            or type(priority) is not int
            or not 0 <= priority <= 2
            or type(slo_us) is not int
            or slo_us <= 0
            or row.get("prompt_tokens") != PROMPT_TOKENS
            or row.get("output_steps") != 4
        ):
            raise TraceError("S36 request violates the frozen contract")
        key = (arrival_us, request_id)
        if previous is not None and key < previous:
            raise TraceError("S36 requests are not in arrival order")
        seen.add(request_id)
        previous = key
    return value


def load_built_trace(path: Path) -> dict[str, Any]:
    return validate_built_trace(load_json(path))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        report = build_trace(args.source)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_bytes(report))
    except TraceError as exc:
        parser.exit(2, f"trace adapter failed: {exc}\n")
    print(json.dumps({
        "output": str(args.output),
        "requests": len(report["requests"]),
        "trace_hash": report["trace_hash"],
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
