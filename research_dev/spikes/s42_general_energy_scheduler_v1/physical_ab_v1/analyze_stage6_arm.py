#!/usr/bin/env python3
"""Validate runtime gates around one full-length Stage 6 physical arm."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import load_execution_plan  # noqa: E402


SCHEMA = "s42-stage6-runtime-gate-receipt-v1"
GPU_UUID = "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08"
EXPECTED_CALLS = 52320
EXPECTED_MODEL_HASHES = {
    "cold_model_sha256": (
        "494518c2262a26e2a607af0e40bca11c4de5a0b108e7c21308906dbbfb1c6f8c"
    ),
    "hot_model_sha256": (
        "500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0"
    ),
}


class GateError(ValueError):
    pass


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot read {path}: {exc}") from exc
    if type(value) is not dict:
        raise GateError(f"{path} must contain an object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        for line in path.read_text(encoding="ascii").splitlines():
            value = json.loads(line)
            if type(value) is not dict:
                raise GateError(f"{path} contains a non-object")
            rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot read {path}: {exc}") from exc
    if not rows:
        raise GateError(f"{path} is empty")
    return rows


def load_phone(path: Path) -> list[dict[str, int | float]]:
    rows: list[dict[str, int | float]] = []
    try:
        with path.open(encoding="ascii", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            expected = {
                "battery_millic",
                "cpu_millic",
                "ddr_millic",
                "epoch_s",
                "gpu_millic",
                "npu_millic",
                "shell_millic",
                "thermal_status",
                "uptime_s",
            }
            if set(reader.fieldnames or []) != expected:
                raise GateError("phone telemetry columns do not match")
            for raw in reader:
                rows.append({
                    key: float(value) if key == "uptime_s" else int(value)
                    for key, value in raw.items()
                })
    except (OSError, UnicodeError, ValueError) as exc:
        raise GateError(f"cannot read {path}: {exc}") from exc
    if not rows:
        raise GateError("phone telemetry is empty")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze(
    mode: str,
    repeat_index: int,
    result_path: Path,
    host_path: Path,
    phone_path: Path,
    heartbeat_limit_s: float,
    scheduler_plan_path: Path | None = None,
) -> dict[str, Any]:
    result = load_json(result_path)
    if not (
        result.get("schema") == "s41-burstgpt-llama-server-result-v1"
        and result.get("status") == "PASS"
        and result.get("mode") == mode
        and result.get("repeat_index") == repeat_index
    ):
        raise GateError("result identity does not match")
    preflight = result.get("preflight")
    if type(preflight) is not dict or any(
        preflight.get(key) != value for key, value in EXPECTED_MODEL_HASHES.items()
    ):
        raise GateError("model epoch does not match")
    paid_start = result.get("paid_start_ns")
    paid_end = result.get("paid_end_ns")
    if type(paid_start) is not int or type(paid_end) is not int or paid_end <= paid_start:
        raise GateError("paid interval is invalid")

    host_rows = load_jsonl(host_path)
    telemetry = [
        row for row in host_rows
        if row.get("kind") == "telemetry"
        and isinstance(row.get("monotonic_ns"), int)
        and paid_start <= row["monotonic_ns"] <= paid_end
    ]
    if not telemetry:
        raise GateError("no host telemetry covers the paid interval")
    telemetry.sort(key=lambda row: int(row["monotonic_ns"]))
    telemetry_gaps_s = [
        (telemetry[0]["monotonic_ns"] - paid_start) / 1e9,
        (paid_end - telemetry[-1]["monotonic_ns"]) / 1e9,
        *[
            (right["monotonic_ns"] - left["monotonic_ns"]) / 1e9
            for left, right in zip(telemetry, telemetry[1:])
        ],
    ]
    executor_heartbeat_ok = max(telemetry_gaps_s) <= heartbeat_limit_s
    expected_processes = {"bridge": 1 if mode == "op15" else 0, "cold": 1, "hot": 1}
    topology_ok = all(
        type(row.get("processes")) is dict
        and all(
            type(row["processes"].get(key)) is list
            and len(row["processes"][key]) == count
            for key, count in expected_processes.items()
        )
        for row in telemetry
    )
    gpu_rows = [row.get("gpu") for row in telemetry]
    gpu_ok = all(
        type(gpu) is dict
        and gpu.get("uuid") == GPU_UUID
        and gpu.get("software_thermal_slowdown") == "Not Active"
        and gpu.get("hardware_thermal_slowdown") == "Not Active"
        for gpu in gpu_rows
    )
    gpu_temperatures = [
        int(gpu["temperature_c"])
        for gpu in gpu_rows
        if type(gpu) is dict and type(gpu.get("temperature_c")) is int
    ]
    throttle_counts = [
        row.get("cpu_package_throttle_count") for row in telemetry
    ]
    cpu_ok = all(type(value) is int for value in throttle_counts) and len(
        set(throttle_counts)
    ) == 1
    wanted_usb = "18d1:2d00" if mode == "op15" else "22d9:2772"
    usb_ok = all(
        any(
            device.get("vendor_product") == wanted_usb
            and device.get("speed_mbps") == 5000
            for device in row.get("usb", [])
            if type(device) is dict
        )
        for row in telemetry
    )

    phone_rows = load_phone(phone_path)
    thermal_status_ok = all(row["thermal_status"] == 0 for row in phone_rows)
    temperature_fields = (
        "battery_millic",
        "cpu_millic",
        "ddr_millic",
        "gpu_millic",
        "npu_millic",
        "shell_millic",
    )
    phone_max = {
        key: max(int(row[key]) for row in phone_rows) for key in temperature_fields
    }

    progress = [row for row in host_rows if row.get("kind") == "bridge_progress"]
    heartbeat_gaps: list[float] = []
    heartbeat_ok = mode == "cpu"
    last_calls = 0
    if mode == "op15":
        all_call_progress = [
            row for row in progress
            if type(row.get("calls")) is int
            and row["calls"] > 0
        ]
        call_progress = [
            row for row in all_call_progress
            if type(row.get("monotonic_ns")) is int
            and paid_start <= row["monotonic_ns"] <= paid_end
        ]
        all_call_progress.sort(key=lambda row: int(row["file_mtime_ns"]))
        call_progress.sort(key=lambda row: int(row["file_mtime_ns"]))
        for left, right in zip(call_progress, call_progress[1:]):
            heartbeat_gaps.append(
                (right["file_mtime_ns"] - left["file_mtime_ns"]) / 1e9
            )
        if all_call_progress:
            last_calls = max(int(row["calls"]) for row in all_call_progress)
        phone = result.get("phone")
        bridge = phone.get("bridge") if type(phone) is dict else None
        heartbeat_ok = bool(
            all_call_progress
            and last_calls == EXPECTED_CALLS
            and executor_heartbeat_ok
            and type(bridge) is dict
            and bridge.get("calls") == EXPECTED_CALLS
            and bridge.get("reset_recoveries") == 0
            and all(int(row.get("reset_recoveries", 0)) == 0 for row in progress)
        )

    gates = {
        "cpu_not_throttled": cpu_ok,
        "exact_paid_topology": topology_ok,
        "gpu_not_thermally_throttled": gpu_ok,
        "op15_android_thermal_status_none": thermal_status_ok,
        "qualified_usb_5gbps": usb_ok,
        "worker_transport_heartbeat": heartbeat_ok,
    }
    scheduler = None
    if scheduler_plan_path is not None:
        plan = load_execution_plan(scheduler_plan_path)
        scheduler_ok = bool(
            plan.execution_mode == mode
            and result.get("scheduler_plan_sha256") == plan.plan_sha256
            and result.get("preflight", {}).get("scheduler", {}).get(
                "plan_sha256"
            ) == plan.plan_sha256
            and result.get("preflight", {}).get("scheduler", {}).get(
                "route_id"
            ) == plan.decision.route_id
        )
        gates["unified_scheduler_plan_bound"] = scheduler_ok
        scheduler = {
            "decision_reason": plan.decision.reason,
            "plan_sha256": plan.plan_sha256,
            "route_id": plan.decision.route_id,
        }
    receipt: dict[str, Any] = {
        "epoch_key": "sha256:27d004d2958af9df39e4543eda7853cca68f9293df7aba3515bed3585775a1e5",
        "gates": gates,
        "host": {
            "executor_heartbeat_max_gap_s": max(telemetry_gaps_s),
            "gpu_temperature_max_c": max(gpu_temperatures, default=None),
            "paid_samples": len(telemetry),
        },
        "mode": mode,
        "phone": {
            "max_temperature_millic": phone_max,
            "samples": len(phone_rows),
            "thermal_status_values": sorted({
                int(row["thermal_status"]) for row in phone_rows
            }),
        },
        "repeat_index": repeat_index,
        "result": {
            "path": str(result_path),
            "sha256": sha256(result_path),
        },
        "schema": SCHEMA,
        "scheduler": scheduler,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "transport": {
            "heartbeat_limit_s": heartbeat_limit_s,
            "progress_max_gap_s": max(heartbeat_gaps, default=0.0),
            "last_observed_calls": last_calls,
            "paid_progress_events": len([
                row for row in progress
                if type(row.get("monotonic_ns")) is int
                and paid_start <= row["monotonic_ns"] <= paid_end
            ]),
            "progress_events": len(progress),
        },
    }
    receipt["receipt_sha256"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "op15"), required=True)
    parser.add_argument("--repeat-index", type=int, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--host-telemetry", type=Path, required=True)
    parser.add_argument("--phone-telemetry", type=Path, required=True)
    parser.add_argument("--scheduler-plan", type=Path)
    parser.add_argument("--heartbeat-limit-s", type=float, default=1.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeat_index < 1 or not math.isfinite(args.heartbeat_limit_s) or args.heartbeat_limit_s <= 0:
        parser.error("invalid repeat index or heartbeat limit")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        value = analyze(
            args.mode,
            args.repeat_index,
            args.result,
            args.host_telemetry,
            args.phone_telemetry,
            args.heartbeat_limit_s,
            args.scheduler_plan,
        )
        args.output.write_bytes(canonical_bytes(value))
    except GateError as exc:
        parser.exit(2, f"stage6 gate analysis failed: {exc}\n")
    print(json.dumps({
        "output": str(args.output),
        "receipt_sha256": value["receipt_sha256"],
        "status": value["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
