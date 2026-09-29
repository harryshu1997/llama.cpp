#!/usr/bin/env python3
"""Validate the repeated Qwen continuous-batch M=1 through M=4 screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


INDICES = [7, 21, 20, 12]
TRACE_SHA256 = (
    "b20a9ba66ee3558d835a0e19ed3cfa4"
    "c31a4a9e8b4f9c085b29a14f80250a0ff"
)
PHONE_SERIAL = "3C15AU002CL00000"
MIN_AVAILABLE_KIB = 2 * 1024 * 1024
LAYER_COUNT = 12
EXPECTED_LAYER_MASK = "0000000000000fff"
EXPECTED_SHAPES = {1, 2, 3, 4}


class AnalysisError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read {path}: {exc}") from exc
    require(type(value) is dict, f"object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while block := source.read(4 * 1024 * 1024):
                digest.update(block)
    except OSError as exc:
        raise AnalysisError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def close(name: str, left: float, right: float) -> None:
    if not math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9):
        raise AnalysisError(f"{name}: {left} != {right}")


def positive(name: str, value: object) -> float:
    require(
        type(value) in (int, float)
        and math.isfinite(value)
        and value > 0,
        f"positive {name}",
    )
    return float(value)


def prefixed_objects(path: Path, prefix: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise AnalysisError(f"cannot read {path}: {exc}") from exc
    result = []
    for line in lines:
        if not line.startswith(prefix):
            continue
        payload = line.removeprefix(prefix)
        if not payload.startswith("{"):
            continue
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise AnalysisError(f"malformed {prefix.strip()}: {path}") from exc
        require(type(value) is dict, f"{prefix.strip()} object: {path}")
        result.append(value)
    return result


def one_prefixed(path: Path, prefix: str) -> dict[str, Any]:
    rows = prefixed_objects(path, prefix)
    require(len(rows) == 1, f"one {prefix.strip()}: {path}")
    return rows[0]


def validate_result(path: Path, arm: str, repeat: int) -> dict[str, Any]:
    value = load_object(path)
    require(
        value.get("schema") == "s41-burstgpt-llama-server-result-v1"
        and value.get("status") == "PASS"
        and value.get("arm") == arm
        and value.get("repeat_index") == repeat
        and value.get("dispatch") == "concurrent"
        and value.get("indices") == INDICES,
        f"result identity: {path}",
    )
    rows = value.get("request_results")
    require(
        type(rows) is list
        and len(rows) == len(INDICES)
        and value.get("metrics", {}).get("requests") == len(INDICES),
        f"completed work: {path}",
    )
    for expected_index, row in zip(INDICES, rows, strict=True):
        require(
            type(row) is dict
            and row.get("request_index") == expected_index
            and type(row.get("tokens")) is list
            and len(row["tokens"]) == row.get("output_tokens")
            and all(type(token) is int for token in row["tokens"]),
            f"request work: {path}",
        )
    start = value.get("paid_start_ns")
    end = value.get("paid_end_ns")
    duration = positive("duration", value["metrics"].get("duration_s"))
    require(type(start) is int and type(end) is int and end > start,
            f"paid interval: {path}")
    close("paid duration", duration, (end - start) / 1e9)
    energy = value.get("server_energy")
    require(
        type(energy) is dict
        and energy.get("boundary") == "paid_trace_interval",
        f"server energy boundary: {path}",
    )
    cpu = positive("CPU energy", energy.get("cpu_package_energy_j"))
    gpu = positive("GPU energy", energy.get("gpu_board_energy_j"))
    server = positive(
        "server energy", energy.get("server_compute_device_energy_j")
    )
    close("server energy sum", server, cpu + gpu)
    return value


def validate_phone(path: Path, result_path: Path,
                   duration_s: float) -> dict[str, Any]:
    value = load_object(path)
    require(
        value.get("schema") == "s41-phone-energy-v3"
        and value.get("status") == "PASS"
        and value.get("serial") == PHONE_SERIAL
        and value.get("boundary") == "paid_trace_interval"
        and value.get("battery_discharge_only") is True,
        f"phone energy identity: {path}",
    )
    close("phone duration", positive("phone duration", value.get("duration_s")),
          duration_s)
    usb = positive("USB energy", value.get("usb_input_energy_j"))
    battery = value.get("battery_discharge_energy_j")
    require(
        type(battery) in (int, float)
        and math.isfinite(battery)
        and battery >= 0,
        f"battery energy: {path}",
    )
    whole = positive("whole-phone energy", value.get("whole_phone_energy_j"))
    close("phone energy sum", whole, usb + float(battery))
    require(
        value.get("input_sha256", {}).get("trace_result")
        == sha256(result_path),
        f"phone result binding: {path}",
    )
    return value


def validate_logs(capture: Path, treatment: bool) -> dict[str, Any]:
    bridge = one_prefixed(capture / "bridge.stderr", "FFNDMABUF ")
    router_arm = one_prefixed(capture / "router.log", "RESIDENTARM ")
    router = one_prefixed(capture / "router.log", "RESIDENTROUTER ")
    workers = one_prefixed(
        capture / "resident-workers.log", "RESIDENTWORKERS "
    )
    server_rows = prefixed_objects(
        capture / "server.stderr", "S41SERVERFFN "
    )
    shapes = prefixed_objects(
        capture / "server.stderr", "S41SERVERFFNSHAPE "
    )
    calls = int(bridge.get("calls", -1))
    require(
        bridge.get("status") == "ok"
        and bridge.get("reset_recoveries") == 0
        and router.get("status") == "ok"
        and router.get("sessions") == 1
        and router.get("requests") == calls
        and router_arm.get("status") == "WARM"
        and router_arm.get("target_count") == 2
        and router_arm.get("layer_mask") == EXPECTED_LAYER_MASK
        and router_arm.get("n_embd") == 5120
        and router_arm.get("columns") == 17408
        and workers.get("status") == "WARM"
        and workers.get("sessions") == ["HTP0", "HTP1", "HTP2"],
        f"resident transport receipt: {capture}",
    )
    if treatment:
        require(len(server_rows) == 1 and calls > 0,
                f"treatment FFN summary: {capture}")
        summary = server_rows[0]
        require(
            summary.get("status") == "ok"
            and summary.get("calls") == calls
            and summary.get("decode_calls", 0)
                + summary.get("prefill_calls", 0) == calls,
            f"treatment call conservation: {capture}",
        )
        shape_ids = set()
        for row in shapes:
            tokens = row.get("tokens")
            shape_calls = row.get("calls")
            require(
                type(tokens) is int
                and tokens in EXPECTED_SHAPES
                and row.get("columns") == 17408
                and type(shape_calls) is int
                and shape_calls > 0
                and shape_calls % LAYER_COUNT == 0,
                f"qualified shape receipt: {capture}",
            )
            shape_ids.add(tokens)
        require(shape_ids == EXPECTED_SHAPES,
                f"missing M=1..4 shape: {capture}")
    else:
        require(calls == 0 and not server_rows and not shapes,
                f"resident control work: {capture}")
    session_text = (capture / "session.log").read_text(encoding="ascii")
    memory = re.findall(r"mem_available_kib=(\d+)", session_text)
    require(len(memory) == 1 and int(memory[0]) >= MIN_AVAILABLE_KIB,
            f"phone memory reserve: {capture}")
    return {
        "calls": calls,
        "mem_available_kib": int(memory[0]),
        "reset_recoveries": 0,
        "shapes": sorted(EXPECTED_SHAPES) if treatment else [],
    }


def validate_arm(root: Path, arm: str, repeat: int) -> dict[str, Any]:
    result_path = root / "RESULT.json"
    capture = Path(f"{root}.phone-capture")
    result = validate_result(result_path, arm, repeat)
    duration = float(result["metrics"]["duration_s"])
    phone_path = capture / "PHONE_ENERGY_V3.json"
    phone = validate_phone(phone_path, result_path, duration)
    logs = validate_logs(capture, arm == "op15")
    server = result["server_energy"]
    server_j = float(server["server_compute_device_energy_j"])
    phone_j = float(phone["whole_phone_energy_j"])
    return {
        "artifacts": {
            "phone_energy_sha256": sha256(phone_path),
            "result_sha256": sha256(result_path),
        },
        "cpu_package_j": float(server["cpu_package_energy_j"]),
        "duration_s": duration,
        "fleet_j": server_j + phone_j,
        "gpu_board_j": float(server["gpu_board_energy_j"]),
        "logs": logs,
        "phone_j": phone_j,
        "request_results": result["request_results"],
        "server_j": server_j,
    }


def signature(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "input_tokens": row["input_tokens"],
        "output_tokens": row["output_tokens"],
        "request_index": row["request_index"],
        "tokens": row["tokens"],
    } for row in rows]


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def change(control: float, treatment: float) -> float:
    return 100.0 * (treatment / control - 1.0)


def analyze(roots: dict[str, Path], trace: Path) -> dict[str, Any]:
    require(sha256(trace) == TRACE_SHA256, "trace identity")
    runs = {
        "control_r1": validate_arm(roots["control_r1"], "control", 1),
        "treatment_r1": validate_arm(roots["treatment_r1"], "op15", 1),
        "treatment_r2": validate_arm(roots["treatment_r2"], "op15", 2),
        "control_r2": validate_arm(roots["control_r2"], "control", 2),
    }
    control_signatures = [
        signature(runs["control_r1"]["request_results"]),
        signature(runs["control_r2"]["request_results"]),
    ]
    treatment_signatures = [
        signature(runs["treatment_r1"]["request_results"]),
        signature(runs["treatment_r2"]["request_results"]),
    ]
    controls = [runs["control_r1"], runs["control_r2"]]
    treatments = [runs["treatment_r1"], runs["treatment_r2"]]
    keys = (
        "duration_s", "cpu_package_j", "gpu_board_j", "server_j",
        "phone_j", "fleet_j",
    )
    control_mean = {key: mean(controls, key) for key in keys}
    treatment_mean = {key: mean(treatments, key) for key in keys}
    changes = {
        key + "_change_pct": change(control_mean[key], treatment_mean[key])
        for key in keys
    }
    pair_savings = [
        100.0 * (1.0 - treatments[index]["fleet_j"]
                 / controls[index]["fleet_j"])
        for index in range(2)
    ]
    gates = {
        "each_pair_fleet_energy_saving": all(row > 0 for row in pair_savings),
        "exact_requested_token_counts": True,
        "makespan_not_regressed": changes["duration_s_change_pct"] <= 0,
        "m1_through_m4_observed": all(
            row["logs"]["shapes"] == [1, 2, 3, 4]
            for row in treatments
        ),
        "resident_control_zero_calls": all(
            row["logs"]["calls"] == 0 for row in controls
        ),
        "treatment_matches_control_reference": any(
            treatment_signatures[0] == row for row in control_signatures
        ),
        "treatment_repeats_exact_tokens": (
            treatment_signatures[0] == treatment_signatures[1]
        ),
        "zero_reset_recoveries": all(
            row["logs"]["reset_recoveries"] == 0
            for row in runs.values()
        ),
    }
    compact_runs = {
        name: {key: value for key, value in row.items()
               if key != "request_results"}
        for name, row in runs.items()
    }
    result: dict[str, Any] = {
        "comparison": {
            "changes": changes,
            "control_mean": control_mean,
            "pair_fleet_energy_savings_pct": pair_savings,
            "treatment_mean": treatment_mean,
        },
        "gates": gates,
        "measurement_boundary": (
            "paid concurrent BurstGPT cohort after warmup; CPU package plus "
            "GPU board plus synchronized whole-phone energy"
        ),
        "observations": {
            "control_repeats_exact_tokens": (
                control_signatures[0] == control_signatures[1]
            ),
        },
        "repetition_order": [
            "control_r1", "treatment_r1", "treatment_r2", "control_r2"
        ],
        "runs": compact_runs,
        "schema": "s42-qwen-full-ffn-m1-m4-energy-screen-v1",
        "scope": (
            "four concurrent source-length Qwen BurstGPT rows; qualifies "
            "full-FFN replacement only for physical decode M=1 through M=4"
        ),
        "status": "PASS" if all(gates.values()) else "FAIL",
        "trace": {"indices": INDICES, "sha256": TRACE_SHA256},
    }
    result["record_sha256"] = hashlib.sha256(canonical(result)).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("control-r1", "treatment-r1", "treatment-r2", "control-r2"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.exists():
        parser.error("output must be an unused absolute path")
    roots = {
        "control_r1": args.control_r1,
        "treatment_r1": args.treatment_r1,
        "treatment_r2": args.treatment_r2,
        "control_r2": args.control_r2,
    }
    try:
        result = analyze(roots, args.trace)
        args.output.write_bytes(canonical(result))
    except (AnalysisError, OSError, UnicodeError) as exc:
        parser.exit(2, f"Qwen M=1..4 analysis failed: {exc}\n")
    print(json.dumps({
        "output": str(args.output),
        "record_sha256": result["record_sha256"],
        "status": result["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
