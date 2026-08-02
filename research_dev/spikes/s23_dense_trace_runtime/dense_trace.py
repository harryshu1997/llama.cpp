#!/usr/bin/env python3
"""Build and validate a dense BurstGPT workload for the S22 runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRACE = ROOT / "scratchpad/s8_gate_a/run-a/burstgpt-burst.jsonl"
DEFAULT_MANIFEST = ROOT / "scratchpad/s8_gate_a/run-a/burstgpt-burst.manifest.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "burstgpt-dense-60.json"
SCHEMA = "s23-burstgpt-dense-trace-v1"
SCOPE = (
    "OBSERVED_ARRIVAL_AND_TOKEN_DEMAND_"
    "SYNTHETIC_PAYLOAD_PRIORITY_SLO_EXECUTION_PROXY"
)
MAX_INT = (1 << 63) - 1


class TraceError(RuntimeError):
    pass


def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise TraceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="ascii"), object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TraceError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TraceError(f"{path} must contain one JSON object")
    return value


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def content_digest(value: dict[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("trace_hash", None)
    return "sha256:" + hashlib.sha256(canonical_bytes(unsigned)).hexdigest()


def require_int(name: str, value: object, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TraceError(f"{name} must be an integer")
    if value < minimum or value > MAX_INT:
        raise TraceError(f"{name} is out of range")
    return value


def normalized_rows(path: Path) -> list[tuple[dict[str, Any], str]]:
    rows: list[tuple[dict[str, Any], str]] = []
    try:
        with path.open("r", encoding="ascii", newline="") as stream:
            for line_number, raw in enumerate(stream, 1):
                if not raw.endswith("\n") or raw.endswith("\r\n"):
                    raise TraceError(f"source row {line_number} has a noncanonical newline")
                payload = raw[:-1]
                try:
                    row = json.loads(payload, object_pairs_hook=reject_duplicates)
                except json.JSONDecodeError as exc:
                    raise TraceError(f"source row {line_number} is invalid JSON") from exc
                if not isinstance(row, dict):
                    raise TraceError(f"source row {line_number} is not an object")
                rows.append((row, "sha256:" + hashlib.sha256(payload.encode("ascii")).hexdigest()))
    except (OSError, UnicodeError) as exc:
        raise TraceError(f"cannot read normalized source: {exc}") from exc
    if not rows:
        raise TraceError("normalized source is empty")
    return rows


def eligible(row: dict[str, Any]) -> bool:
    source_fields = row.get("source_fields")
    return (
        row.get("schema_version") == 2
        and row.get("source") == "burstgpt-v2"
        and row.get("service") == "api_generation"
        and row.get("provenance") == "real"
        and row.get("deadline_us") is None
        and row.get("priority_class") is None
        and isinstance(source_fields, dict)
        and source_fields.get("burstgpt_failed") is False
        and type(row.get("input_tokens")) is int
        and row["input_tokens"] > 0
        and type(row.get("output_tokens")) is int
        and row["output_tokens"] >= 8
        and type(row.get("t_us")) is int
        and row["t_us"] >= 0
        and type(row.get("source_row_id")) is int
        and row["source_row_id"] > 0
        and isinstance(row.get("event_id"), str)
    )


def select_window(
    ordered: list[tuple[dict[str, Any], str]], width_us: int, count: int,
) -> tuple[list[tuple[dict[str, Any], str]], dict[str, Any]]:
    require_int("width_us", width_us, 1)
    require_int("count", count, 1)
    candidates = [item for item in ordered if eligible(item[0])]
    candidates.sort(key=lambda item: (
        item[0]["t_us"], item[0].get("source_rank", 0),
        item[0]["source_row_id"], item[0]["event_id"],
    ))
    if len(candidates) < count:
        raise TraceError("fewer eligible source requests than requested")

    left = 0
    best: tuple[tuple[object, ...], int, int] | None = None
    for right, (row, _) in enumerate(candidates):
        while row["t_us"] - candidates[left][0]["t_us"] > width_us:
            left += 1
        first = candidates[left][0]
        key = (-(right - left + 1), first["t_us"], first["event_id"], row["event_id"])
        if best is None or key < best[0]:
            best = (key, left, right)
    if best is None:
        raise TraceError("no dense source window exists")
    _, left, right = best
    dense = candidates[left:right + 1]
    if len(dense) < count:
        raise TraceError("densest source window is smaller than requested")
    selected = dense[:count]
    return selected, {
        "eligible_request_count": len(candidates),
        "window_width_us": width_us,
        "densest_window_count": len(dense),
        "densest_window_start_us": dense[0][0]["t_us"],
        "densest_window_end_us": dense[-1][0]["t_us"],
        "selected_count": count,
        "selection_rule": "first-N-in-earliest-max-count-window-v1",
        "eligibility": (
            "real api_generation; nonfailure; input_tokens>0; output_tokens>=8; "
            "null source priority/deadline"
        ),
    }


def synthetic_class(event_id: str) -> tuple[int, int]:
    bucket = int.from_bytes(hashlib.sha256(event_id.encode("ascii")).digest()[:8], "big") % 10
    if bucket < 2:
        return 0, 15_000_000
    if bucket < 5:
        return 1, 25_000_000
    return 2, 40_000_000


def percentile(values: Iterable[int], numerator: int, denominator: int) -> int:
    ordered = sorted(values)
    if not ordered:
        raise TraceError("cannot summarize an empty vector")
    rank = (numerator * len(ordered) + denominator - 1) // denominator
    return ordered[max(1, rank) - 1]


def build(
    trace_path: Path = DEFAULT_TRACE,
    manifest_path: Path = DEFAULT_MANIFEST,
    width_us: int = 2_000_000,
    count: int = 60,
) -> dict[str, Any]:
    manifest = strict_object(manifest_path)
    trace_sha = file_digest(trace_path)
    if (
        manifest.get("schema_version") != 2
        or manifest.get("source") != "burstgpt-v2"
        or manifest.get("provenance") != "real"
        or manifest.get("output_sha256") != trace_sha
        or manifest.get("source_sha256")
        != "sha256:2299986a07388aa303ec2c41d1131e756db650a39ed6ef9dfe7cc3d7f9a43b8f"
    ):
        raise TraceError("normalized source manifest does not match the pinned BurstGPT trace")

    selected, selection = select_window(normalized_rows(trace_path), width_us, count)
    first_us = selected[0][0]["t_us"]
    requests = []
    for row, row_sha in selected:
        priority, slo_us = synthetic_class(row["event_id"])
        requests.append({
            "request_id": row["source_row_id"],
            "event_id": row["event_id"],
            "source_row_id": row["source_row_id"],
            "source_row_sha256": row_sha,
            "arrival_us": row["t_us"] - first_us,
            "observed_input_tokens": row["input_tokens"],
            "observed_output_tokens": row["output_tokens"],
            "priority": priority,
            "slo_us": slo_us,
            "execution_input_tokens": 1,
            "execution_steps": 4,
        })

    input_lengths = [row["observed_input_tokens"] for row in requests]
    output_lengths = [row["observed_output_tokens"] for row in requests]
    priorities = {str(priority): sum(row["priority"] == priority for row in requests)
                  for priority in range(3)}
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "scope": SCOPE,
        "source": {
            "trace_path": str(trace_path.resolve().relative_to(ROOT)),
            "trace_sha256": trace_sha,
            "manifest_path": str(manifest_path.resolve().relative_to(ROOT)),
            "manifest_sha256": file_digest(manifest_path),
            "source_revision": "v2.0",
            "source_sha256": manifest["source_sha256"],
        },
        "selection": selection,
        "observed_summary": {
            "arrival_span_us": requests[-1]["arrival_us"],
            "input_tokens_total": sum(input_lengths),
            "input_tokens_p50": percentile(input_lengths, 1, 2),
            "input_tokens_p95": percentile(input_lengths, 95, 100),
            "input_tokens_max": max(input_lengths),
            "output_tokens_total": sum(output_lengths),
            "output_tokens_p50": percentile(output_lengths, 1, 2),
            "output_tokens_p95": percentile(output_lengths, 95, 100),
            "output_tokens_max": max(output_lengths),
        },
        "synthetic_sidecar": {
            "provenance": "s23-synthetic-sidecar",
            "class_rule": "sha256-event-id-first-u64-mod-10-v1",
            "classes": [
                {"priority": 0, "bucket_count": 2, "slo_us": 15_000_000},
                {"priority": 1, "bucket_count": 3, "slo_us": 25_000_000},
                {"priority": 2, "bucket_count": 5, "slo_us": 40_000_000},
            ],
            "selected_class_counts": priorities,
        },
        "execution_proxy": {
            "payload_provenance": "synthetic-fixed-token",
            "reason": "BurstGPT publishes token lengths but not prompt text",
            "input_tokens_per_request": 1,
            "output_steps_per_request": 4,
            "claims": "arrival-pressure-and-runtime-mechanics-only",
        },
        "requests": requests,
    }
    result["trace_hash"] = content_digest(result)
    return result


def validate(value: dict[str, Any], verify_source: bool = True) -> None:
    expected_top = {
        "schema", "scope", "source", "selection", "observed_summary",
        "synthetic_sidecar", "execution_proxy", "requests", "trace_hash",
    }
    if set(value) != expected_top or value.get("schema") != SCHEMA or value.get("scope") != SCOPE:
        raise TraceError("dense trace top-level contract mismatch")
    if value.get("trace_hash") != content_digest(value):
        raise TraceError("dense trace content digest mismatch")

    source = value.get("source")
    selection = value.get("selection")
    proxy = value.get("execution_proxy")
    sidecar = value.get("synthetic_sidecar")
    requests = value.get("requests")
    if not all(isinstance(item, dict) for item in (source, selection, proxy, sidecar)):
        raise TraceError("dense trace metadata must be objects")
    if not isinstance(requests, list) or not requests:
        raise TraceError("dense trace requests must be nonempty")
    if selection.get("selected_count") != len(requests):
        raise TraceError("dense trace selected count mismatch")
    if proxy.get("payload_provenance") != "synthetic-fixed-token" \
            or proxy.get("claims") != "arrival-pressure-and-runtime-mechanics-only":
        raise TraceError("execution proxy provenance mismatch")
    if sidecar.get("provenance") != "s23-synthetic-sidecar":
        raise TraceError("synthetic sidecar provenance mismatch")
    if set(source) != {
        "trace_path", "trace_sha256", "manifest_path", "manifest_sha256",
        "source_revision", "source_sha256",
    }:
        raise TraceError("dense trace source contract mismatch")
    if set(selection) != {
        "eligible_request_count", "window_width_us", "densest_window_count",
        "densest_window_start_us", "densest_window_end_us", "selected_count",
        "selection_rule", "eligibility",
    }:
        raise TraceError("dense trace selection contract mismatch")
    if set(proxy) != {
        "payload_provenance", "reason", "input_tokens_per_request",
        "output_steps_per_request", "claims",
    }:
        raise TraceError("dense trace execution proxy contract mismatch")
    if set(sidecar) != {"provenance", "class_rule", "classes", "selected_class_counts"}:
        raise TraceError("dense trace sidecar contract mismatch")
    if require_int("selected_count", selection.get("selected_count"), 1) != len(requests):
        raise TraceError("dense trace selected count mismatch")
    require_int("eligible_request_count", selection.get("eligible_request_count"), len(requests))
    require_int("window_width_us", selection.get("window_width_us"), 1)
    dense_count = require_int("densest_window_count", selection.get("densest_window_count"), 1)
    dense_start = require_int("densest_window_start_us", selection.get("densest_window_start_us"))
    dense_end = require_int("densest_window_end_us", selection.get("densest_window_end_us"))
    if dense_count < len(requests) or dense_end < dense_start:
        raise TraceError("dense trace selection bounds are inconsistent")
    if proxy.get("input_tokens_per_request") != 1 \
            or type(proxy.get("input_tokens_per_request")) is not int \
            or proxy.get("output_steps_per_request") != 4 \
            or type(proxy.get("output_steps_per_request")) is not int:
        raise TraceError("execution proxy shape metadata changed")
    classes = sidecar.get("classes")
    expected_classes = [
        {"priority": 0, "bucket_count": 2, "slo_us": 15_000_000},
        {"priority": 1, "bucket_count": 3, "slo_us": 25_000_000},
        {"priority": 2, "bucket_count": 5, "slo_us": 40_000_000},
    ]
    if sidecar.get("class_rule") != "sha256-event-id-first-u64-mod-10-v1" \
            or classes != expected_classes:
        raise TraceError("synthetic class rule changed")

    expected_request = {
        "request_id", "event_id", "source_row_id", "source_row_sha256",
        "arrival_us", "observed_input_tokens", "observed_output_tokens",
        "priority", "slo_us", "execution_input_tokens", "execution_steps",
    }
    seen_ids: set[int] = set()
    seen_events: set[str] = set()
    previous_key: tuple[int, int, str] | None = None
    for row in requests:
        if not isinstance(row, dict) or set(row) != expected_request:
            raise TraceError("dense trace request contract mismatch")
        request_id = require_int("request_id", row["request_id"], 1)
        source_row_id = require_int("source_row_id", row["source_row_id"], 1)
        arrival_us = require_int("arrival_us", row["arrival_us"])
        require_int("observed_input_tokens", row["observed_input_tokens"], 1)
        require_int("observed_output_tokens", row["observed_output_tokens"], 8)
        priority = require_int("priority", row["priority"])
        require_int("slo_us", row["slo_us"], 1)
        if priority not in (0, 1, 2):
            raise TraceError("priority is outside the frozen class set")
        execution_input = require_int(
            "execution_input_tokens", row["execution_input_tokens"], 1,
        )
        execution_steps = require_int("execution_steps", row["execution_steps"], 1)
        if execution_input != 1 or execution_steps != 4:
            raise TraceError("execution proxy shape changed")
        event_id = row["event_id"]
        if not isinstance(event_id, str) or not event_id:
            raise TraceError("event_id must be nonempty")
        if request_id != source_row_id or request_id in seen_ids or event_id in seen_events:
            raise TraceError("request identity is not unique and source-derived")
        expected_priority, expected_slo = synthetic_class(event_id)
        if (priority, row["slo_us"]) != (expected_priority, expected_slo):
            raise TraceError("synthetic class binding mismatch")
        if not isinstance(row["source_row_sha256"], str) \
                or not row["source_row_sha256"].startswith("sha256:") \
                or len(row["source_row_sha256"]) != 71:
            raise TraceError("source row digest is invalid")
        key = (arrival_us, source_row_id, event_id)
        if previous_key is not None and key < previous_key:
            raise TraceError("dense requests are not in stable arrival order")
        previous_key = key
        seen_ids.add(request_id)
        seen_events.add(event_id)

    input_lengths = [row["observed_input_tokens"] for row in requests]
    output_lengths = [row["observed_output_tokens"] for row in requests]
    expected_summary = {
        "arrival_span_us": requests[-1]["arrival_us"],
        "input_tokens_total": sum(input_lengths),
        "input_tokens_p50": percentile(input_lengths, 1, 2),
        "input_tokens_p95": percentile(input_lengths, 95, 100),
        "input_tokens_max": max(input_lengths),
        "output_tokens_total": sum(output_lengths),
        "output_tokens_p50": percentile(output_lengths, 1, 2),
        "output_tokens_p95": percentile(output_lengths, 95, 100),
        "output_tokens_max": max(output_lengths),
    }
    if value.get("observed_summary") != expected_summary:
        raise TraceError("observed demand summary mismatch")
    expected_counts = {
        str(priority): sum(row["priority"] == priority for row in requests)
        for priority in range(3)
    }
    if sidecar.get("selected_class_counts") != expected_counts:
        raise TraceError("synthetic class counts mismatch")

    if verify_source:
        trace_path = (ROOT / source["trace_path"]).resolve()
        manifest_path = (ROOT / source["manifest_path"]).resolve()
        if not trace_path.is_relative_to(ROOT) or not manifest_path.is_relative_to(ROOT):
            raise TraceError("source path escapes the repository")
        if file_digest(trace_path) != source.get("trace_sha256") \
                or file_digest(manifest_path) != source.get("manifest_sha256"):
            raise TraceError("source artifact digest mismatch")
        rebuilt = build(
            trace_path, manifest_path,
            require_int("window_width_us", selection.get("window_width_us"), 1),
            len(requests),
        )
        if canonical_bytes(rebuilt) != canonical_bytes(value):
            raise TraceError("dense trace does not reproduce from the pinned source")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width-us", type=int, default=2_000_000)
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.validate_only:
            validate(strict_object(args.output))
        else:
            value = build(args.trace, args.manifest, args.width_us, args.count)
            validate(value)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(canonical_bytes(value))
        print(json.dumps({"verdict": "PASS", "output": str(args.output)}, sort_keys=True))
        return 0
    except (KeyError, OSError, TraceError, ValueError) as exc:
        print(json.dumps({"verdict": "FAIL", "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
