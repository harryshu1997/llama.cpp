#!/usr/bin/env python3
"""Validate the matched S28 real-device run from persisted artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SCHEMA = "s28-priority-shared-tail-v1"
TRACE_HASH = (
    "sha256:9079e3f939068d17fcd356ed05173910f23dead75e879e439d4ff85ea65f832c"
)
WORKERS = {
    "OP12": ("HTP0", 0, 8, 0, 200),
    "OP15": ("HTP0", 8, 16, 0, 200),
    "cuda-prefix": ("CUDA0", 0, 8, 240, 40),
    "cuda-mid": ("CUDA0", 8, 16, 240, 40),
    "cuda-tail": ("CUDA0", 16, 48, 240, 240),
}
EVENT_WORKERS = {
    "cuda-prefix", "cuda-mid", "op12-prefix", "op15-mid", "cuda-tail",
}
SESSION_PREFIX = "SESSIONCERT "
PLACEMENT_PREFIX = "PLACEMENTCERT "


class ValidationError(RuntimeError):
    pass


def pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="ascii"), object_pairs_hook=pairs,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot load {path}: {exc}") from exc
    if type(value) is not dict:
        raise ValidationError(f"{path} is not a JSON object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(root: Path) -> None:
    manifest = root / "SHA256SUMS.txt"
    if not manifest.is_file():
        raise ValidationError(f"missing manifest: {manifest}")
    seen = set()
    for line in manifest.read_text(encoding="ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ValidationError(f"invalid manifest line in {manifest}")
        expected, name = match.groups()
        if name in seen or name == "SHA256SUMS.txt":
            raise ValidationError(f"invalid manifest member: {name}")
        seen.add(name)
        relative = name[2:] if name.startswith("./") else name
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValidationError(f"manifest mismatch: {path}")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path != manifest
    }
    normalized = {name[2:] if name.startswith("./") else name for name in seen}
    if actual != normalized:
        raise ValidationError(f"manifest coverage differs in {root}")


def records(path: Path, prefix: str) -> list[dict[str, Any]]:
    result = []
    for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        if line.startswith(prefix):
            try:
                value = json.loads(line[len(prefix):], object_pairs_hook=pairs)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"invalid certificate in {path}") from exc
            if type(value) is not dict:
                raise ValidationError(f"invalid certificate type in {path}")
            result.append(value)
    return result


def validate_worker_log(
    path: Path,
    worker: str,
    expected_backend: str,
    layer_start: int,
    layer_end: int,
    control_steps: int,
    treatment_steps: int,
) -> None:
    allowed = {expected_backend}
    if worker == "OP12":
        allowed.add("CPU")
    if worker == "cuda-prefix":
        allowed.add("CUDA_Host")

    def validate_backend_map(record: dict[str, Any], steps: int) -> None:
        by_op = record.get("compute_by_op_and_buffer")
        if type(by_op) is not dict:
            raise ValidationError(f"{worker} has no per-op placement evidence")
        observed = set()
        for op, by_buffer in by_op.items():
            if type(op) is not str or not op or type(by_buffer) is not dict:
                raise ValidationError(f"{worker} placement map type differs")
            for buffer, count in by_buffer.items():
                if (
                    buffer not in allowed
                    or type(count) is not int
                    or count <= 0
                    or (buffer != expected_backend and op != "GET_ROWS")
                ):
                    raise ValidationError(
                        f"{worker} has an undeclared compute fallback"
                    )
                observed.add(buffer)
        if steps == 0 and observed:
            raise ValidationError(f"{worker} has compute in an empty session")
        if steps > 0 and expected_backend not in observed:
            raise ValidationError(f"{worker} did not use its expected backend")

    text = path.read_text(encoding="utf-8", errors="strict")
    if "unexpected EOF" in text or '"status":"RUN_FAILED"' in text:
        raise ValidationError(f"{worker} contains a failed session")
    sessions = records(path, SESSION_PREFIX)
    placements = records(path, PLACEMENT_PREFIX)
    if len(sessions) != 2 or len(placements) != 1:
        raise ValidationError(f"{worker} certificate count differs")
    if [row.get("session_id") for row in sessions] != [1, 2]:
        raise ValidationError(f"{worker} session sequence differs")
    if [row.get("session_end") for row in sessions] != ["DETACH", "STOP"]:
        raise ValidationError(f"{worker} session termination differs")
    if sessions[0].get("reset_applied") is not True:
        raise ValidationError(f"{worker} detach did not reset request state")
    if sessions[1].get("reset_applied") is not False:
        raise ValidationError(f"{worker} stop reset flag differs")
    identity = (
        sessions[0].get("worker_pid"),
        sessions[0].get("worker_boot_nonce"),
        sessions[0].get("device_boot_id"),
    )
    if any(
        (
            row.get("worker_pid"),
            row.get("worker_boot_nonce"),
            row.get("device_boot_id"),
        ) != identity
        for row in sessions
    ):
        raise ValidationError(f"{worker} was not resident across controls")
    expected_steps = [control_steps, treatment_steps]
    total = 0
    for row, steps in zip(sessions, expected_steps):
        total += steps
        if (
            row.get("expected_backend") != expected_backend
            or row.get("layer_start") != layer_start
            or row.get("layer_end") != layer_end
            or row.get("steps_session") != steps
            or row.get("steps_total") != total
            or row.get("missing_buffer_compute_nodes") != 0
        ):
            raise ValidationError(f"{worker} session evidence differs")
        expected_status = "PLACEMENT_UNOBSERVED" if steps == 0 else "SCHEDULED_PLACEMENT_OK"
        if row.get("placement_status") != expected_status:
            raise ValidationError(f"{worker} placement status differs")
        validate_backend_map(row, steps)
    placement = placements[0]
    if (
        placement.get("status") != "SCHEDULED_PLACEMENT_OK"
        or placement.get("run_rc") != 0
        or placement.get("layer_start") != layer_start
        or placement.get("layer_end") != layer_end
        or placement.get("missing_buffer_compute_nodes") != 0
    ):
        raise ValidationError(f"{worker} final placement certificate differs")
    buffers = set(placement.get("compute_by_buffer_type", {}))
    if not buffers or not buffers <= allowed or expected_backend not in buffers:
        raise ValidationError(f"{worker} used an unexpected compute backend")
    validate_backend_map(placement, treatment_steps)


def request_map(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = report.get("runtime", {}).get("requests")
    if type(rows) is not list or len(rows) != 60:
        raise ValidationError("request result count differs")
    result = {}
    for row in rows:
        request_id = row.get("request_id")
        if type(request_id) is not int or request_id in result:
            raise ValidationError("request result identity differs")
        result[request_id] = row
    return result


def validate_report(report: dict[str, Any], mode: str) -> None:
    if report.get("schema") != SCHEMA or report.get("status") != "RUN_COMPLETE":
        raise ValidationError(f"{mode} report status differs")
    if report.get("trace", {}).get("trace_hash") != TRACE_HASH:
        raise ValidationError(f"{mode} trace binding differs")
    configuration = report.get("configuration", {})
    if (
        configuration.get("control_mode") != mode
        or configuration.get("shared_tail") is not True
        or configuration.get("urgent_background_batch_isolation") is not True
    ):
        raise ValidationError(f"{mode} configuration differs")
    summary = report.get("summary", {})
    if (
        summary.get("completed_requests") != 60
        or summary.get("rejected_requests") != 0
        or summary.get("slo_misses") != 0
    ):
        raise ValidationError(f"{mode} terminal outcomes differ")
    events = report.get("batch_events")
    if type(events) is not dict or set(events) != EVENT_WORKERS:
        raise ValidationError(f"{mode} physical queue set differs")
    for worker, worker_events in events.items():
        if type(worker_events) is not list:
            raise ValidationError(f"{worker} event stream type differs")
        for event in worker_events:
            priorities = event.get("priorities")
            if (
                event.get("status") != "OK"
                or type(priorities) is not list
                or not priorities
                or any(type(value) is not int or value < 0 for value in priorities)
                or (0 in priorities and any(value != 0 for value in priorities))
            ):
                raise ValidationError(f"{worker} priority batch evidence differs")


def validate_pair(control: dict[str, Any], treatment: dict[str, Any]) -> dict[str, Any]:
    validate_report(control, "all-cuda")
    validate_report(treatment, "priority")
    control_rows = request_map(control)
    treatment_rows = request_map(treatment)
    if set(control_rows) != set(treatment_rows):
        raise ValidationError("control and treatment request sets differ")
    if any(row.get("route_id") != "R0" for row in control_rows.values()):
        raise ValidationError("control used a phone route")
    for row in treatment_rows.values():
        expected = "R0" if row.get("priority") == 0 else "R2"
        if row.get("route_id") != expected:
            raise ValidationError("treatment route differs from priority policy")
    if control.get("profile", {}).get("sha256") != treatment.get("profile", {}).get("sha256"):
        raise ValidationError("control and treatment profiles differ")
    tail = treatment["batch_events"]["cuda-tail"]
    route_sequence = [tuple(event["contributing_routes"]) for event in tail]
    if set(route_sequence) != {("R0",), ("R2",)}:
        raise ValidationError("shared tail did not execute both routes")
    transitions = sum(
        route_sequence[index] != route_sequence[index - 1]
        for index in range(1, len(route_sequence))
    )
    if transitions < 2:
        raise ValidationError("shared tail routes did not interleave")
    for worker in ("op12-prefix", "op15-mid"):
        if control["batch_events"][worker] or not treatment["batch_events"][worker]:
            raise ValidationError(f"{worker} control/treatment activity differs")
    control_cuda = control["summary"]["summed_cuda_island_compute_us"]
    treatment_cuda = treatment["summary"]["summed_cuda_island_compute_us"]
    if (
        type(control_cuda) is not int
        or type(treatment_cuda) is not int
        or not 0 < treatment_cuda < control_cuda
    ):
        raise ValidationError("measured CUDA-island work did not decrease")
    control_p0 = control["summary"]["priority"]["0"]["latency_us"]["p95"]
    treatment_p0 = treatment["summary"]["priority"]["0"]["latency_us"]["p95"]
    if treatment_p0 * 10 > control_p0 * 11:
        raise ValidationError("P0 p95 latency regressed by more than 10 percent")
    token_matches = sum(
        control_rows[request_id].get("output_tokens")
        == treatment_rows[request_id].get("output_tokens")
        for request_id in control_rows
    )
    return {
        "completed": 60,
        "slo_misses": 0,
        "control_makespan_us": control["summary"]["makespan_us"],
        "treatment_makespan_us": treatment["summary"]["makespan_us"],
        "control_cuda_compute_us": control_cuda,
        "treatment_cuda_compute_us": treatment_cuda,
        "cuda_compute_relief_percent": (
            (control_cuda - treatment_cuda) * 100.0 / control_cuda
        ),
        "control_p0_p95_us": control_p0,
        "treatment_p0_p95_us": treatment_p0,
        "tail_route_transitions": transitions,
        "token_matches": token_matches,
        "token_total": 60,
        "numeric_verdict": "UNCERTIFIED_F16_PHONE_Q8_SERVER",
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--control", type=Path, required=True)
    result.add_argument("--treatment", type=Path, required=True)
    result.add_argument("--phone-session", type=Path, required=True)
    result.add_argument("--desktop-session", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if args.output.exists():
            raise ValidationError("output already exists")
        verify_manifest(args.phone_session)
        verify_manifest(args.desktop_session)
        for worker, values in WORKERS.items():
            root = args.phone_session if worker.startswith("OP") else args.desktop_session
            validate_worker_log(root / f"{worker}.log", worker, *values)
        result = validate_pair(load_json(args.control), load_json(args.treatment))
        report = {
            "schema": "s28-priority-shared-tail-validation-v1",
            "status": "PHYSICAL_PRIORITY_SHARED_TAIL_PASS",
            "control_sha256": "sha256:" + sha256(args.control),
            "treatment_sha256": "sha256:" + sha256(args.treatment),
            "phone_manifest_sha256": "sha256:" + sha256(
                args.phone_session / "SHA256SUMS.txt"
            ),
            "desktop_manifest_sha256": "sha256:" + sha256(
                args.desktop_session / "SHA256SUMS.txt"
            ),
            "result": result,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="ascii",
        )
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return 0
    except (OSError, UnicodeError, ValidationError) as exc:
        print(json.dumps({
            "status": "VALIDATION_FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }, sort_keys=True, separators=(",", ":")))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
