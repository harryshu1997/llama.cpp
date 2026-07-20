#!/usr/bin/env python3
"""Independently reopen and validate the S16 report and raw evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiment_contract import (
    BATCH, FULL_SCHEDULE, LOW_COHORTS, N_GEN, ContractError, digest_file,
    integrate_power, parse_power_jsonl, strict_object, summarize,
    validate_result_record,
)


HERE = Path(__file__).resolve().parent


class ValidationError(RuntimeError):
    pass


def parse_prefixed(line: bytes, prefix: bytes, label: str) -> dict[str, Any]:
    if not line.startswith(prefix):
        raise ValidationError(f"missing {label} prefix")
    return strict_object(line[len(prefix):], label, False)


def resolve_artifact(path_text: str) -> Path:
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
    resolved = probe.resolve(strict=True)
    if not resolved.is_file() or HERE.resolve() not in resolved.parents:
        raise ValidationError("artifact is not a regular in-spike file")
    return resolved


def validate_placement(value: dict[str, Any], label: str, pid: int) -> None:
    role = "monodriver" if label == "P0" else "host_tail"
    mode = "monodriver" if label == "P0" else "pipedriver"
    start = 0 if label == "P0" else 8
    if value.get("schema") != "layersplit-scheduled-placement-v2" \
            or value.get("role") != role or value.get("mode") != mode \
            or value.get("layer_start") != start or value.get("layer_end") != 48 \
            or value.get("n_layer") != 48 or value.get("pid") != pid \
            or value.get("run_rc") != 0 or value.get("status") != "SCHEDULED_PLACEMENT_OK" \
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
                label == "P0" and op_name == "GET_ROWS" and backend == "CUDA_Host")
            if not allowed:
                raise ValidationError(f"undeclared host fallback {op_name}@{backend}")


def validate_session(value: dict[str, Any], launch_id: int, end: str) -> None:
    if value.get("schema") != "ls-stagenet-session-v2" \
            or value.get("session_id") != launch_id or value.get("session_end") != end \
            or value.get("expected_backend") != "HTP0" \
            or value.get("layer_start") != 0 or value.get("layer_end") != 8 \
            or value.get("n_layer") != 48 \
            or value.get("placement_status") != "SCHEDULED_PLACEMENT_OK" \
            or value.get("missing_buffer_compute_nodes") != 0 \
            or value.get("reset_applied") is not (end == "DETACH"):
        raise ValidationError("phone session identity failed")
    by_op = value.get("compute_by_op_and_buffer")
    if type(by_op) is not dict or not by_op:
        raise ValidationError("phone operation tally is absent")
    for op_name, buffers in by_op.items():
        for backend, count in buffers.items():
            if type(count) is not int or count <= 0:
                raise ValidationError("phone node count is invalid")
            if backend != "HTP0" and not (op_name == "GET_ROWS" and backend == "CPU"):
                raise ValidationError(f"undeclared phone fallback {op_name}@{backend}")


def validate_high(row: dict[str, Any], row_dir: Path) -> None:
    stdout = (row_dir / "high.stdout").read_text(encoding="ascii").splitlines()
    stderr = (row_dir / "high.stderr").read_text(encoding="ascii").splitlines()
    records = [strict_object(
        (line[len("BGEPROF "):] + "\n").encode("ascii"), "BGE record", False)
        for line in stdout if line.startswith("BGEPROF ")]
    placements = [strict_object(
        (line[len("PLACEMENTCERT "):] + "\n").encode("ascii"), "BGE placement", False)
        for line in stderr if line.startswith("PLACEMENTCERT ")]
    if len(records) != 1 or len(placements) != 1:
        raise ValidationError("BGE raw record or placement count failed")
    record = records[0]
    reported = row["bge"]
    if record.get("finite") is not True or record.get("batch") != reported.get("batch") \
            or record.get("reps") != reported.get("reps") \
            or record.get("total_tokens") != BATCH // 2 * reported.get("seq_len_exact") \
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
    non_htp = placement.get("non_htp_ops")
    if type(non_htp) is not list \
            or any(not value.startswith("GET_ROWS@CPU:") for value in non_htp):
        raise ValidationError("BGE contains undeclared CPU work")


def validate_low(row: dict[str, Any], row_dir: Path, reference: list[int]) -> None:
    stdout = (row_dir / "low.stdout.bin").read_bytes().splitlines(keepends=True)
    stderr = (row_dir / "low.stderr.bin").read_bytes().splitlines(keepends=True)
    count = row["gemma_batches"]
    if len(stdout) != count:
        raise ValidationError("low raw result count failed")
    placements = [line for line in stderr if line.startswith(b"PLACEMENTCERT ")]
    sessions = [line for line in stderr if line.startswith(b"SESSIONCERT ")]
    markers = [line for line in stderr if line.startswith(b"PERSISTENT_DRIVER_EXCHANGE_END ")]
    expected_sessions = count if row["label"] == "P2" else 0
    if len(placements) != count or len(markers) != count or len(sessions) != expected_sessions:
        raise ValidationError("low raw evidence count failed")
    pids = set()
    worker_pids = set()
    worker_nonces = set()
    walls = []
    for index, payload in enumerate(stdout, start=1):
        end = "STOP" if index == count else "DETACH"
        result = strict_object(payload, "low result")
        validate_result_record(result, reference, row["label"], index, end)
        placement = parse_prefixed(placements[index - 1], b"PLACEMENTCERT ", "host placement")
        validate_placement(placement, row["label"], result["host_pid"])
        marker = parse_prefixed(markers[index - 1], b"PERSISTENT_DRIVER_EXCHANGE_END ", "marker")
        if marker != {"launch_id": index}:
            raise ValidationError("low exchange marker identity failed")
        pids.add(result["host_pid"])
        walls.append(result["route_wall_us"])
        if sessions:
            session = parse_prefixed(sessions[index - 1], b"SESSIONCERT ", "phone session")
            validate_session(session, index, end)
            worker_pids.add(session["worker_pid"])
            worker_nonces.add(session["worker_boot_nonce"])
    if len(pids) != 1 or next(iter(pids)) != row["host_pid"] \
            or walls != row["low_route_wall_us"]:
        raise ValidationError("low host persistence or latency binding failed")
    if sessions and (len(worker_pids) != 1 or len(worker_nonces) != 1 \
                     or next(iter(worker_pids)) != row["worker_pid"]):
        raise ValidationError("low phone persistence binding failed")


def validate(path: Path) -> dict[str, Any]:
    report = strict_object(path.read_bytes(), "report")
    if report.get("schema") != "s16-mixed-persistent-energy-v1" \
            or report.get("mode") not in ("screen", "full"):
        raise ValidationError("report identity failed")
    full = report["mode"] == "full"
    schedule = FULL_SCHEDULE if full else (("P0", 0), ("P2", 0))
    rows = report.get("rows")
    if type(rows) is not list or len(rows) != len(schedule):
        raise ValidationError("report row count failed")
    reference = report.get("reference_tokens")
    if type(reference) is not list or len(reference) != N_GEN:
        raise ValidationError("report token reference failed")
    expected_cohorts = LOW_COHORTS if full else 2
    for row, expected in zip(rows, schedule):
        if (row.get("label"), row.get("pair")) != expected \
                or row.get("gemma_batches") != expected_cohorts:
            raise ValidationError("row order or work failed")
        power_path = resolve_artifact(row["power_artifact"])
        if digest_file(power_path) != row.get("power_sha256"):
            raise ValidationError("power artifact digest failed")
        paid_start = int(round(float(row["bge"]["paid_start_s"]) * 1_000_000))
        paid_end = int(round(float(row["bge"]["paid_end_s"]) * 1_000_000))
        recomputed = integrate_power(
            parse_power_jsonl(power_path.read_bytes()), paid_start, paid_end, full)
        if recomputed != row.get("power"):
            raise ValidationError("reported power differs from raw artifact")
        validate_high(row, power_path.parent)
        validate_low(row, power_path.parent, reference)
        windows = row.get("low_exchange_windows_us")
        if type(windows) is not list or len(windows) != expected_cohorts \
                or any(type(window) is not list or len(window) != 2 \
                       or window[0] < paid_start or window[1] > paid_end or window[1] <= window[0]
                       for window in windows):
            raise ValidationError("low exchange windows escape BGE paid work")
    recomputed_summary = summarize(rows, full)
    if recomputed_summary != report.get("summary"):
        raise ValidationError("reported summary does not reproduce")
    expected_status = "S16_SCREEN_PASS" if not full and recomputed_summary["screen_pass"] else \
        "S16_MIXED_GPU_BOARD_DIAGNOSTIC_PASS" if full and recomputed_summary["overall_pass"] else \
        "S16_FAIL_GATE"
    if report.get("status") != expected_status \
            or report.get("formal_total_energy_claim") != "NONE" \
            or report.get("phone_energy") != "UNKNOWN" \
            or report.get("usb_energy") != "UNKNOWN" \
            or report.get("total_system_energy") != "UNKNOWN":
        raise ValidationError("verdict or energy scope failed")
    return {"status": expected_status, "rows": len(rows), "report_sha256": digest_file(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.report), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, ValidationError, OSError, ValueError) as exc:
        print(f"S16_VALIDATE_ERROR {exc}")
        raise SystemExit(2)
