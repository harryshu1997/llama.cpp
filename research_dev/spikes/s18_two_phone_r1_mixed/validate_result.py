#!/usr/bin/env python3
"""Independently reopen and validate the S18 report and raw evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiment_contract import (
    BATCHES_PER_ROUND, CONTROL_BATCH, FULL_ROUNDS, FULL_SCHEDULE, N_GEN,
    OP12_EXCHANGE_CREDIT, PHONE_BATCH, SCREEN_ROUNDS, SCREEN_SCHEDULE,
    ContractError, digest_file, integrate_power, parse_power_jsonl, strict_object,
    summarize, validate_result_record,
)


HERE = Path(__file__).resolve().parent
HOST_END_PREFIX = b"PERSISTENT_DRIVER_EXCHANGE_END "
SESSION_PREFIX = b"SESSIONCERT "


class ValidationError(RuntimeError):
    pass


def parse_after(line: bytes, prefix: bytes, label: str) -> dict[str, Any]:
    if not line.startswith(prefix):
        raise ValidationError(f"missing {label} prefix")
    return strict_object(line[len(prefix):], label, False)


def resolve(path_text: str) -> Path:
    if type(path_text) is not str or not path_text:
        raise ValidationError("artifact path is invalid")
    relative = Path(path_text)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValidationError("artifact path escapes the spike")
    probe = HERE
    for part in relative.parts:
        probe = probe / part
        if probe.is_symlink():
            raise ValidationError("artifact path contains a symlink")
    result = probe.resolve(strict=True)
    if not result.is_file() or HERE.resolve() not in result.parents:
        raise ValidationError("artifact is not a regular in-spike file")
    return result


def validate_host_placement(value: dict[str, Any], role: str, start: int,
                            pid: int) -> None:
    mode = "monodriver" if role == "monodriver" else "pipedriver"
    if value.get("schema") != "layersplit-scheduled-placement-v2" \
            or value.get("role") != role or value.get("mode") != mode \
            or value.get("layer_start") != start or value.get("layer_end") != 48 \
            or value.get("n_layer") != 48 or value.get("pid") != pid \
            or value.get("run_rc") != 0 \
            or value.get("status") != "SCHEDULED_PLACEMENT_OK" \
            or value.get("missing_buffer_compute_nodes") != 0 \
            or type(value.get("compute_nodes")) is not int or value["compute_nodes"] <= 0:
        raise ValidationError("host placement identity failed")
    by_op = value.get("compute_by_op_and_buffer")
    if type(by_op) is not dict or not by_op:
        raise ValidationError("host placement operation tally is absent")
    for op_name, buffers in by_op.items():
        if type(buffers) is not dict or not buffers:
            raise ValidationError("host placement operation tally is malformed")
        for backend, count in buffers.items():
            if type(count) is not int or count <= 0:
                raise ValidationError("host placement node count is invalid")
            allowed = backend == "CUDA0" or (
                role == "monodriver" and op_name == "GET_ROWS" and backend == "CUDA_Host")
            if not allowed:
                raise ValidationError(f"undeclared host fallback {op_name}@{backend}")


def validate_session(value: dict[str, Any], name: str, layer_end: int,
                     launch_id: int, session_end: str) -> None:
    if value.get("schema") != "ls-stagenet-session-v2" \
            or value.get("session_id") != launch_id \
            or value.get("session_end") != session_end \
            or value.get("expected_backend") != "HTP0" \
            or value.get("layer_start") != 0 or value.get("layer_end") != layer_end \
            or value.get("n_layer") != 48 \
            or value.get("placement_status") != "SCHEDULED_PLACEMENT_OK" \
            or value.get("missing_buffer_compute_nodes") != 0 \
            or value.get("reset_applied") is not (session_end == "DETACH"):
        raise ValidationError(f"{name} phone session identity failed")
    by_op = value.get("compute_by_op_and_buffer")
    if type(by_op) is not dict or not by_op:
        raise ValidationError(f"{name} phone operation tally is absent")
    for op_name, buffers in by_op.items():
        if type(buffers) is not dict or not buffers:
            raise ValidationError(f"{name} phone operation tally is malformed")
        for backend, count in buffers.items():
            if type(count) is not int or count <= 0:
                raise ValidationError(f"{name} phone node count is invalid")
            if backend != "HTP0" and not (
                    op_name == "GET_ROWS" and backend == "CPU"):
                raise ValidationError(f"undeclared {name} fallback {op_name}@{backend}")


def validate_high(row: dict[str, Any], row_dir: Path) -> None:
    stdout = (row_dir / "high.stdout").read_text(encoding="ascii").splitlines()
    stderr = (row_dir / "high.stderr").read_text(encoding="ascii").splitlines()
    records = [strict_object(
        (line[len("BGEPROF "):] + "\n").encode("ascii"), "BGE record", False)
        for line in stdout if line.startswith("BGEPROF ")]
    placements = [strict_object(
        (line[len("PLACEMENTCERT "):] + "\n").encode("ascii"),
        "BGE placement", False)
        for line in stderr if line.startswith("PLACEMENTCERT ")]
    if len(records) != 1 or len(placements) != 1:
        raise ValidationError("BGE raw record or placement count failed")
    record = records[0]
    reported = row.get("bge")
    if type(reported) is not dict or record.get("finite") is not True \
            or record.get("batch") != reported.get("batch") \
            or record.get("reps") != reported.get("reps") \
            or record.get("total_tokens") != reported.get("batch") * reported.get("seq_len_exact") \
            or record.get("paid_start_s") != reported.get("paid_start_s") \
            or record.get("paid_end_s") != reported.get("paid_end_s") \
            or record.get("us") != reported.get("latency_samples_us"):
        raise ValidationError("BGE report differs from raw output")
    placement = placements[0]
    if placement.get("status") != "SCHEDULED_PLACEMENT_OK" \
            or placement.get("missing_buffer_compute_nodes") != 0:
        raise ValidationError("BGE placement status failed")
    by_buffer = placement.get("by_buffer")
    if type(by_buffer) is not dict or not by_buffer \
            or any("CUDA" not in backend for backend in by_buffer):
        raise ValidationError("BGE compute escaped CUDA")


def validate_manifest(row: dict[str, Any], row_dir: Path) -> None:
    expected = {
        path.name: digest_file(path)
        for path in row_dir.iterdir()
        if path.is_file() and path.name != "power.jsonl" and path.stat().st_size > 0
    }
    if row.get("raw_artifact_sha256") != dict(sorted(expected.items())):
        raise ValidationError("raw artifact manifest differs from disk")


def validate_control(row: dict[str, Any], row_dir: Path,
                     reference: list[int], rounds: int) -> None:
    stdout = (row_dir / "control.stdout.bin").read_bytes().splitlines(keepends=True)
    stderr = (row_dir / "control.stderr.bin").read_bytes().splitlines(keepends=True)
    count = rounds * BATCHES_PER_ROUND
    placements = [line for line in stderr if line.startswith(b"PLACEMENTCERT ")]
    markers = [line for line in stderr if line.startswith(HOST_END_PREFIX)]
    if len(stdout) != count or len(placements) != count or len(markers) != count:
        raise ValidationError("control raw evidence count failed")
    results = []
    placement_values = []
    pids = set()
    for index, payload in enumerate(stdout, start=1):
        session_end = "STOP" if index == count else "DETACH"
        result = strict_object(payload, "control result")
        validate_result_record(
            result, reference, CONTROL_BATCH, index, session_end)
        placement = parse_after(
            placements[index - 1], b"PLACEMENTCERT ", "control placement")
        validate_host_placement(placement, "monodriver", 0, result["host_pid"])
        marker = parse_after(markers[index - 1], HOST_END_PREFIX, "control marker")
        if marker != {"launch_id": index}:
            raise ValidationError("control marker identity failed")
        results.append(result)
        placement_values.append(placement)
        pids.add(result["host_pid"])
    if len(pids) != 1 or results != row.get("control_results") \
            or placement_values != row.get("control_placements") \
            or [item["route_wall_us"] for item in results] \
            != row.get("control_route_wall_us"):
        raise ValidationError("control report differs from raw evidence")


def validate_phone_lane(row: dict[str, Any], row_dir: Path, reference: list[int],
                        rounds: int, name: str, layer_end: int) -> None:
    host_stdout_path = row_dir / f"{name}-host.stdout.bin"
    host_stderr_path = row_dir / f"{name}-host.stderr.bin"
    phone_stdout_path = row_dir / f"{name}-phone.stdout.bin"
    phone_stderr_path = row_dir / f"{name}-phone.stderr.bin"
    if phone_stdout_path.read_bytes() != b"":
        raise ValidationError(f"{name} phone stdout is unexpectedly non-empty")
    stdout = host_stdout_path.read_bytes().splitlines(keepends=True)
    host_stderr = host_stderr_path.read_bytes().splitlines(keepends=True)
    phone_stderr = phone_stderr_path.read_bytes().splitlines(keepends=True)
    placements = [line for line in host_stderr if line.startswith(b"PLACEMENTCERT ")]
    markers = [line for line in host_stderr if line.startswith(HOST_END_PREFIX)]
    sessions = [line for line in phone_stderr if line.startswith(SESSION_PREFIX)]
    if len(stdout) != rounds or len(placements) != rounds \
            or len(markers) != rounds or len(sessions) != rounds:
        raise ValidationError(f"{name} raw evidence count failed")
    results = []
    session_values = []
    host_pids = set()
    worker_pids = set()
    nonces = set()
    for index, payload in enumerate(stdout, start=1):
        session_end = "STOP" if index == rounds else "DETACH"
        result = strict_object(payload, f"{name} result")
        validate_result_record(
            result, reference, PHONE_BATCH, index, session_end)
        placement = parse_after(
            placements[index - 1], b"PLACEMENTCERT ", f"{name} host placement")
        validate_host_placement(placement, "host_tail", layer_end, result["host_pid"])
        marker = parse_after(markers[index - 1], HOST_END_PREFIX, f"{name} marker")
        if marker != {"launch_id": index}:
            raise ValidationError(f"{name} marker identity failed")
        session = parse_after(
            sessions[index - 1], SESSION_PREFIX, f"{name} phone session")
        validate_session(session, name, layer_end, index, session_end)
        results.append(result)
        session_values.append(session)
        host_pids.add(result["host_pid"])
        worker_pids.add(session["worker_pid"])
        nonces.add(session["worker_boot_nonce"])
    if len(host_pids) != 1 or len(worker_pids) != 1 or len(nonces) != 1:
        raise ValidationError(f"{name} process persistence failed")
    if results != row.get("phone_results", {}).get(name) \
            or session_values != row.get("phone_sessions", {}).get(name) \
            or [item["route_wall_us"] for item in results] \
            != row.get("phone_route_wall_us", {}).get(name):
        raise ValidationError(f"{name} report differs from raw evidence")


def validate(path: Path) -> dict[str, Any]:
    report = strict_object(path.read_bytes(), "report")
    if report.get("schema") != "s18-two-phone-r1-mixed-v1" \
            or report.get("mode") not in ("screen", "full"):
        raise ValidationError("report identity failed")
    full = report["mode"] == "full"
    schedule = FULL_SCHEDULE if full else SCREEN_SCHEDULE
    rounds = FULL_ROUNDS if full else SCREEN_ROUNDS
    rows = report.get("rows")
    reference = report.get("reference_tokens")
    if type(rows) is not list or len(rows) != len(schedule) \
            or type(reference) is not list or len(reference) != N_GEN:
        raise ValidationError("report row or reference count failed")
    for row, expected in zip(rows, schedule):
        if (row.get("label"), row.get("pair")) != expected:
            raise ValidationError("report schedule failed")
        power_path = resolve(row.get("power_artifact"))
        row_dir = power_path.parent
        if digest_file(power_path) != row.get("power_sha256"):
            raise ValidationError("power artifact digest failed")
        paid_start = int(round(float(row["bge"]["paid_start_s"]) * 1_000_000))
        paid_end = int(round(float(row["bge"]["paid_end_s"]) * 1_000_000))
        recomputed_power = integrate_power(
            parse_power_jsonl(power_path.read_bytes()), paid_start, paid_end, full)
        if recomputed_power != row.get("power"):
            raise ValidationError("reported power differs from raw evidence")
        validate_manifest(row, row_dir)
        validate_high(row, row_dir)
        if row["label"] == "P0":
            validate_control(row, row_dir, reference, rounds)
            windows = row.get("control_windows_us")
        else:
            op12_count = min(rounds, OP12_EXCHANGE_CREDIT)
            op15_count = rounds * BATCHES_PER_ROUND - op12_count
            validate_phone_lane(row, row_dir, reference, op15_count, "op15", 8)
            validate_phone_lane(row, row_dir, reference, op12_count, "op12", 6)
            windows = [
                window
                for name in ("op15", "op12")
                for window in row.get("phone_windows_us", {}).get(name, [])
            ]
        if type(windows) is not list or not windows \
                or any(type(window) is not list or len(window) != 2
                       or window[0] < paid_start or window[1] > paid_end
                       or window[1] <= window[0] for window in windows):
            raise ValidationError("low-work window escapes the paid interval")
    recomputed_summary = summarize(rows, full)
    if recomputed_summary != report.get("summary"):
        raise ValidationError("reported summary does not reproduce")
    expected_status = "S18_R1_FLEET_SCREEN_PASS" \
        if not full and recomputed_summary["screen_pass"] else \
        "S18_R1_FLEET_GPU_BOARD_PASS" \
        if full and recomputed_summary["overall_pass"] else \
        "S18_R1_FLEET_MECHANICS_PASS_RELIEF_INSUFFICIENT" \
        if full and recomputed_summary["high_priority_gate"] else "S18_FAIL_GATE"
    if report.get("status") != expected_status \
            or report.get("second_gpu_idle_at_end") is not True \
            or report.get("formal_total_energy_claim") != "NONE" \
            or report.get("phone_energy") != "UNKNOWN" \
            or report.get("usb_energy") != "UNKNOWN" \
            or report.get("total_system_energy") != "UNKNOWN":
        raise ValidationError("verdict or energy scope failed")
    return {
        "status": expected_status,
        "rows": len(rows),
        "report_sha256": digest_file(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.report), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, ValidationError, OSError, ValueError, TypeError) as exc:
        print(f"S18_VALIDATE_ERROR {exc}")
        raise SystemExit(2)
