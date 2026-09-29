#!/usr/bin/env python3
"""Validate and summarize the repeated Qwen full-FFN energy screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


SCHEMA = "s42-qwen-full-ffn-energy-screen-v1"
INDICES = [52, 53, 31]
PHONE_SERIAL = "3C15AU002CL00000"
MIN_AVAILABLE_KIB = 2 * 1024 * 1024
EXPECTED_LAYER_MASK = "0000000000000fff"
EXPECTED_CALLS = 1296


class AnalysisError(ValueError):
    pass


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"),
                   sort_keys=True)
        + "\n"
    ).encode("ascii")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while block := source.read(4 * 1024 * 1024):
                digest.update(block)
    except OSError as exc:
        raise AnalysisError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read {path}: {exc}") from exc
    if type(value) is not dict:
        raise AnalysisError(f"{path} must contain an object")
    return value


def positive_number(name: str, value: object) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise AnalysisError(f"{name} must be a positive finite number")
    return float(value)


def close(name: str, left: float, right: float) -> None:
    if not math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9):
        raise AnalysisError(f"{name} mismatch: {left} != {right}")


def prefixed_object(path: Path, prefix: str) -> dict[str, Any]:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise AnalysisError(f"cannot read {path}: {exc}") from exc
    matches = [line.removeprefix(prefix) for line in lines
               if line.startswith(prefix)]
    if len(matches) != 1:
        raise AnalysisError(f"{path}: expected one {prefix.strip()} record")
    try:
        value = json.loads(matches[0])
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"{path}: malformed {prefix.strip()} record") from exc
    if type(value) is not dict:
        raise AnalysisError(f"{path}: {prefix.strip()} must be an object")
    return value


def validate_result(path: Path, arm: str, repeat: int) -> dict[str, Any]:
    value = load_object(path)
    if not (
        value.get("schema") == "s41-burstgpt-llama-server-result-v1"
        and value.get("status") == "PASS"
        and value.get("arm") == arm
        and value.get("repeat_index") == repeat
        and value.get("indices") == INDICES
    ):
        raise AnalysisError(f"{path}: result identity")
    rows = value.get("request_results")
    metrics = value.get("metrics")
    if not (
        type(rows) is list
        and len(rows) == len(INDICES)
        and type(metrics) is dict
        and metrics.get("requests") == len(INDICES)
    ):
        raise AnalysisError(f"{path}: completed work")
    for expected_index, row in zip(INDICES, rows, strict=True):
        if not (
            type(row) is dict
            and row.get("request_index") == expected_index
            and type(row.get("tokens")) is list
            and len(row["tokens"]) == row.get("output_tokens")
            and all(type(token) is int for token in row["tokens"])
        ):
            raise AnalysisError(f"{path}: request result")
    start = value.get("paid_start_ns")
    end = value.get("paid_end_ns")
    if type(start) is not int or type(end) is not int or end <= start:
        raise AnalysisError(f"{path}: paid interval")
    duration = positive_number("duration_s", metrics.get("duration_s"))
    close("paid duration", duration, (end - start) / 1e9)
    energy = value.get("server_energy")
    if not (
        type(energy) is dict
        and energy.get("boundary") == "paid_trace_interval"
    ):
        raise AnalysisError(f"{path}: server energy boundary")
    cpu = positive_number("cpu package energy", energy.get(
        "cpu_package_energy_j"))
    gpu = positive_number("GPU board energy", energy.get(
        "gpu_board_energy_j"))
    server = positive_number("server energy", energy.get(
        "server_compute_device_energy_j"))
    close("server component sum", server, cpu + gpu)
    return value


def validate_phone(path: Path, result_path: Path,
                   duration_s: float) -> dict[str, Any]:
    value = load_object(path)
    if not (
        value.get("schema") == "s41-phone-energy-v3"
        and value.get("status") == "PASS"
        and value.get("serial") == PHONE_SERIAL
        and value.get("boundary") == "paid_trace_interval"
        and value.get("battery_discharge_only") is True
    ):
        raise AnalysisError(f"{path}: phone energy identity")
    close("phone duration", positive_number(
        "phone duration", value.get("duration_s")), duration_s)
    usb = positive_number("USB input energy", value.get("usb_input_energy_j"))
    battery = value.get("battery_discharge_energy_j")
    if type(battery) not in (int, float) or not math.isfinite(battery) or battery < 0:
        raise AnalysisError(f"{path}: battery energy")
    whole = positive_number("whole-phone energy", value.get(
        "whole_phone_energy_j"))
    close("whole-phone component sum", whole, usb + battery)
    hashes = value.get("input_sha256")
    if not (type(hashes) is dict
            and hashes.get("trace_result") == sha256(result_path)):
        raise AnalysisError(f"{path}: result hash binding")
    if type(value.get("sample_count_total")) is not int or value[
            "sample_count_total"] < 50:
        raise AnalysisError(f"{path}: too few phone samples")
    return value


def validate_logs(capture: Path, treatment: bool) -> dict[str, Any]:
    bridge = prefixed_object(capture / "bridge.stderr", "FFNDMABUF ")
    router_arm = prefixed_object(capture / "router.log", "RESIDENTARM ")
    router = prefixed_object(capture / "router.log", "RESIDENTROUTER ")
    workers = prefixed_object(
        capture / "resident-workers.log", "RESIDENTWORKERS ")
    workers_text = (capture / "resident-workers.log").read_text(
        encoding="ascii")
    expected_calls = EXPECTED_CALLS if treatment else 0
    if not (
        bridge.get("status") == "ok"
        and bridge.get("calls") == expected_calls
        and bridge.get("decode_calls") == expected_calls
        and bridge.get("prefill_calls") == 0
        and bridge.get("reset_recoveries") == 0
        and router.get("status") == "ok"
        and router.get("sessions") == 1
        and router.get("requests") == expected_calls
    ):
        raise AnalysisError(f"{capture}: transport work receipt")
    if not (
        router_arm.get("status") == "WARM"
        and router_arm.get("target_count") == 2
        and router_arm.get("layer_mask") == EXPECTED_LAYER_MASK
        and router_arm.get("n_embd") == 5120
        and router_arm.get("columns") == 17408
        and workers.get("status") == "WARM"
        and workers.get("sessions") == ["HTP0", "HTP1", "HTP2"]
        and workers.get("qwen_layers") == "0-11"
        and workers.get("resident_mib_approx") == 9225
    ):
        raise AnalysisError(f"{capture}: resident arm receipt")
    worker_identities = (
        ("HTP0", "layers=23 mask=00000000007fffff K=3840 NFF=15360 "
         "slice=[9216,15360) type=f16 io=f16 activation=geglu "
         "max_tokens=512 quantum=512 alternate=0 blocks=12 "
         "weights=3105.05 MiB hash=0210c3a2d001c2ff"),
        ("HTP1", "layers=6 mask=000000000000003f K=5120 NFF=17408 "
         "slice=[0,17408) type=f16 io=f16 activation=swiglu "
         "max_tokens=512 quantum=17408 alternate=0 blocks=1 "
         "weights=3060.00 MiB hash=c8dfbd19d741a902"),
        ("HTP2", "layers=6 mask=0000000000000fc0 K=5120 NFF=17408 "
         "slice=[0,17408) type=f16 io=f16 activation=swiglu "
         "max_tokens=512 quantum=17408 alternate=0 blocks=1 "
         "weights=3060.00 MiB hash=53c4b4acb071c452"),
    )
    for backend, identity in worker_identities:
        if (workers_text.count(f"weight buffers={backend} count=") != 1
                or workers_text.count(identity) != 1):
            raise AnalysisError(f"{capture}: {backend} weight identity")
    session_text = (capture / "session.log").read_text(encoding="ascii")
    memory_matches = re.findall(r"mem_available_kib=(\d+)", session_text)
    if len(memory_matches) != 1:
        raise AnalysisError(f"{capture}: memory receipt")
    available_kib = int(memory_matches[0])
    if available_kib < MIN_AVAILABLE_KIB:
        raise AnalysisError(f"{capture}: memory reserve")
    return {
        "bridge_calls": expected_calls,
        "mem_available_kib": available_kib,
        "reset_recoveries": 0,
        "resident_mib_approx": workers["resident_mib_approx"],
        "router_requests": expected_calls,
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
            "phone_samples_sha256": sha256(capture / "phone-samples.tsv"),
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


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def work_signature(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
            "request_index": row["request_index"],
            "tokens": row["tokens"],
        }
        for row in rows
    ]


def percent(treatment: float, control: float) -> float:
    return 100.0 * (treatment / control - 1.0)


def parse_identity(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path or not name.isascii():
        raise argparse.ArgumentTypeError("identity must be ASCII NAME=PATH")
    return name, Path(path)


def analyze(roots: dict[str, Path], identities: list[tuple[str, Path]],
            trace: Path) -> dict[str, Any]:
    runs = {
        "control_r1": validate_arm(roots["control_r1"], "control", 1),
        "treatment_r1": validate_arm(roots["treatment_r1"], "op15", 1),
        "treatment_r2": validate_arm(roots["treatment_r2"], "op15", 2),
        "control_r2": validate_arm(roots["control_r2"], "control", 2),
    }
    reference_work = work_signature(runs["control_r1"]["request_results"])
    equal_outputs = all(
        work_signature(row["request_results"]) == reference_work
        for row in runs.values()
    )
    control = [runs["control_r1"], runs["control_r2"]]
    treatment = [runs["treatment_r1"], runs["treatment_r2"]]
    keys = ("duration_s", "cpu_package_j", "gpu_board_j", "server_j",
            "phone_j", "fleet_j")
    control_mean = {key: mean(control, key) for key in keys}
    treatment_mean = {key: mean(treatment, key) for key in keys}
    changes = {
        key + "_change_pct": percent(treatment_mean[key], control_mean[key])
        for key in keys
    }
    pair_savings = [
        100.0 * (1.0 - treatment[index]["fleet_j"] /
                 control[index]["fleet_j"])
        for index in range(2)
    ]
    gates = {
        "each_pair_fleet_energy_saving_at_least_10_percent": all(
            saving >= 10.0 for saving in pair_savings),
        "equal_exact_output_tokens": equal_outputs,
        "makespan_not_regressed": changes["duration_s_change_pct"] <= 0.0,
        "memory_reserve_at_least_2_gib": all(
            row["logs"]["mem_available_kib"] >= MIN_AVAILABLE_KIB
            for row in runs.values()),
        "resident_control_has_zero_calls": all(
            row["logs"]["bridge_calls"] == 0 for row in control),
        "treatment_call_count_exact": all(
            row["logs"]["bridge_calls"] == EXPECTED_CALLS
            for row in treatment),
        "zero_reset_recoveries": all(
            row["logs"]["reset_recoveries"] == 0
            for row in runs.values()),
    }
    identity_hashes: dict[str, str] = {}
    for name, path in identities:
        if name in identity_hashes:
            raise AnalysisError(f"duplicate identity: {name}")
        identity_hashes[name] = sha256(path)
    compact_runs = {
        name: {key: value for key, value in row.items()
               if key != "request_results"}
        for name, row in runs.items()
    }
    result: dict[str, Any] = {
        "comparison": {
            "changes": changes,
            "control_mean": control_mean,
            "fleet_delta_j": (
                treatment_mean["fleet_j"] - control_mean["fleet_j"]),
            "pair_fleet_energy_savings_pct": pair_savings,
            "phone_delta_j": (
                treatment_mean["phone_j"] - control_mean["phone_j"]),
            "server_delta_j": (
                treatment_mean["server_j"] - control_mean["server_j"]),
            "treatment_mean": treatment_mean,
        },
        "gates": gates,
        "identity_sha256": identity_hashes,
        "measurement_boundary": (
            "paid trace interval after one warm replay; desktop CPU package "
            "plus GPU board plus simultaneous whole-phone USB input and "
            "battery discharge"
        ),
        "repetition_order": [
            "control_r1", "treatment_r1", "treatment_r2", "control_r2"
        ],
        "runs": compact_runs,
        "schema": SCHEMA,
        "scope": (
            "matched three-request BurstGPT screen for Qwen decode M=1; "
            "not a full-trace or general-shape certificate"
        ),
        "status": "PASS" if all(gates.values()) else "FAIL",
        "trace": {
            "indices": INDICES,
            "sha256": sha256(trace),
        },
    }
    result["record_sha256"] = hashlib.sha256(canonical_bytes(result)).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("control-r1", "treatment-r1", "treatment-r2", "control-r2"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--identity", action="append", type=parse_identity,
                        default=[])
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    roots = {
        "control_r1": args.control_r1,
        "treatment_r1": args.treatment_r1,
        "treatment_r2": args.treatment_r2,
        "control_r2": args.control_r2,
    }
    try:
        result = analyze(roots, args.identity, args.trace)
        args.output.write_bytes(canonical_bytes(result))
    except (AnalysisError, OSError, UnicodeError) as exc:
        parser.exit(2, f"Qwen full-FFN analysis failed: {exc}\n")
    print(json.dumps({
        "output": str(args.output),
        "record_sha256": result["record_sha256"],
        "status": result["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
