#!/usr/bin/env python3
"""Validate an S25 report against all three executed worker certificates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from continuous_lifecycle import DEFAULT_REQUESTS, build_lifecycle_plan, output_mismatches


class ValidationError(RuntimeError):
    pass


def _pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload, object_pairs_hook=_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"JSON root is not an object: {path}")
    return value, hashlib.sha256(payload).hexdigest()


def load_certificate(path: Path, prefix: str) -> dict[str, Any]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            try:
                records.append(json.loads(
                    line[len(prefix):], object_pairs_hook=_pairs,
                ))
            except json.JSONDecodeError as exc:
                raise ValidationError(f"invalid {prefix.strip()} in {path}") from exc
    if len(records) != 1 or not isinstance(records[0], dict):
        raise ValidationError(f"expected exactly one {prefix.strip()} in {path}")
    return records[0]


def validate_report(report: dict[str, Any]) -> None:
    if report.get("schema") != "s25-continuous-lifecycle-v1":
        raise ValidationError("report schema mismatch")
    if report.get("verdict") != "PASS":
        raise ValidationError("physical report does not pass")
    if report.get("claim_scope") != "REAL_RUNTIME_MECHANICS_ONLY":
        raise ValidationError("physical report scope changed")
    expected_requests = [asdict(item) for item in DEFAULT_REQUESTS]
    if report.get("requests") != expected_requests:
        raise ValidationError("request set differs from the frozen lifecycle")

    plan = build_lifecycle_plan(DEFAULT_REQUESTS, 2)
    expected_memberships = [
        [row.request for row in step.rows] for step in plan.steps if step.rows
    ]
    if report.get("plan") != {
        "capacity": 2,
        "memberships": expected_memberships,
    }:
        raise ValidationError("reported plan differs from the derived plan")
    dynamic = report.get("dynamic")
    serial = report.get("serial")
    if not isinstance(dynamic, dict) or not isinstance(serial, dict):
        raise ValidationError("execution records are missing")
    mismatches = output_mismatches(
        serial.get("outputs", {}), dynamic.get("outputs", {}),
    )
    if mismatches or report.get("token_mismatches") != []:
        raise ValidationError("same-route B1 token screen failed")

    events = dynamic.get("events")
    if not isinstance(events, list) or len(events) != len(plan.steps):
        raise ValidationError("dynamic event count differs from the plan")
    for expected, actual in zip(plan.steps, events):
        if actual.get("step") != expected.index:
            raise ValidationError("dynamic event step changed")
        if actual.get("admissions") != list(expected.admissions):
            raise ValidationError("dynamic admissions changed")
        removals = actual.get("removals")
        if not isinstance(removals, list) or [
            item.get("request") for item in removals if isinstance(item, dict)
        ] != list(expected.removals):
            raise ValidationError("dynamic removals changed")
        members = [row.request for row in expected.rows]
        if actual.get("members") != members:
            raise ValidationError("dynamic physical membership changed")
        if members:
            sizes = actual.get("physical_batch_sizes")
            if sizes != {name: len(members) for name in ("op12", "op15", "tail")}:
                raise ValidationError("physical stage batch sizes differ")
            statuses = actual.get("statuses")
            if not isinstance(statuses, dict):
                raise ValidationError("post-batch statuses are missing")
            for status in statuses.values():
                if status.get("active_sequences") != len(members):
                    raise ValidationError("post-batch active count differs")

    proofs = report.get("proofs")
    if not isinstance(proofs, dict) or not proofs or not all(
        value is True for value in proofs.values()
    ):
        raise ValidationError("mechanics proof set is incomplete")
    drained = report.get("drained")
    if not isinstance(drained, dict) or set(drained) != {"op12", "op15", "tail"}:
        raise ValidationError("drain records are incomplete")
    for status in drained.values():
        if status.get("active_sequences") != 0 or status.get("draining") is not True:
            raise ValidationError("worker did not drain to zero")


def validate_worker(
    log: Path, name: str, layer_start: int, layer_end: int, backend: str,
) -> dict[str, Any]:
    session = load_certificate(log, "SESSIONCERT ")
    placement = load_certificate(log, "PLACEMENTCERT ")
    if session.get("schema") != "ls-stagenet-session-v2":
        raise ValidationError(f"{name} session schema changed")
    expected_steps = 2 * sum(item.output_steps for item in DEFAULT_REQUESTS)
    expected_session = {
        "session_end": "STOP",
        "expected_backend": backend,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "n_layer": 48,
        "steps_session": expected_steps,
        "steps_total": expected_steps,
        "reset_applied": False,
        "missing_buffer_compute_nodes": 0,
        "placement_status": "SCHEDULED_PLACEMENT_OK",
    }
    for key, value in expected_session.items():
        if session.get(key) != value:
            raise ValidationError(f"{name} session field changed: {key}")
    if placement.get("status") != "SCHEDULED_PLACEMENT_OK":
        raise ValidationError(f"{name} placement did not pass")
    if placement.get("run_rc") != 0 or placement.get("missing_buffer_compute_nodes") != 0:
        raise ValidationError(f"{name} placement is incomplete")
    if (placement.get("layer_start"), placement.get("layer_end")) != (
        layer_start, layer_end,
    ):
        raise ValidationError(f"{name} placement range changed")
    by_buffer = placement.get("compute_by_buffer_type")
    if not isinstance(by_buffer, dict) or by_buffer.get(backend, 0) <= 0:
        raise ValidationError(f"{name} has no compute on {backend}")
    unexpected = set(by_buffer) - {backend}
    if name == "op12":
        unexpected -= {"CPU"}
        by_op = placement.get("compute_by_op_and_buffer", {})
        if set(by_op.get("GET_ROWS", {})) != {"CPU"}:
            raise ValidationError("OP12 CPU exception differs from GET_ROWS")
    if unexpected:
        raise ValidationError(f"{name} used an unexpected compute backend")
    return {"session": session, "placement": placement}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--op12-log", type=Path, required=True)
    parser.add_argument("--op15-log", type=Path, required=True)
    parser.add_argument("--tail-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValidationError("validation output already exists")
        report, report_sha = load_json(args.report)
        validate_report(report)
        workers = {
            "op12": validate_worker(args.op12_log, "op12", 0, 8, "HTP0"),
            "op15": validate_worker(args.op15_log, "op15", 8, 16, "HTP0"),
            "tail": validate_worker(args.tail_log, "tail", 16, 48, "CUDA0"),
        }
        result = {
            "schema": "s25-continuous-lifecycle-validation-v1",
            "verdict": "PASS",
            "report_sha256": "sha256:" + report_sha,
            "worker_logs": {
                name: {
                    "path": str(path),
                    "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for name, path in {
                    "op12": args.op12_log,
                    "op15": args.op15_log,
                    "tail": args.tail_log,
                }.items()
            },
            "worker_sessions": {
                name: record["session"] for name, record in workers.items()
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes((
            json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("ascii"))
    except (OSError, ValidationError, ValueError) as exc:
        print(json.dumps({
            "verdict": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }, sort_keys=True, separators=(",", ":")))
        return 2
    print(json.dumps({
        "verdict": "PASS", "output": str(args.output),
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

