#!/usr/bin/env python3
"""Validate and aggregate the I3 4060 Ti plus OP15 physical campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import run_trace


TRACE_SHA256 = (
    "b20a9ba66ee3558d835a0e19ed3cfa4c"
    "31a4a9e8b4f9c085b29a14f80250a0ff"
)
MMLU_SHA256 = (
    "3ffafee1615ae2de690a2726b880823e"
    "167a3d9c210c5faed86d8f0e93ecff4f"
)
POLICY_ID = "i3-hidden-wait"
POLICY_TABLE = "1:9664,3:8192,8:4096,128:8192,512:11136"
EXPECTED_REQUESTS = 74
EXPECTED_INPUT_TOKENS = 33843
EXPECTED_OUTPUT_TOKENS = 11605
EXPECTED_COLD_REQUESTS = 17
EXPECTED_HOT_REQUESTS = 57
EXPECTED_COLD_OUTPUT_TOKENS = 6919
EXPECTED_HOT_OUTPUT_TOKENS = 4686
EXPECTED_PHONE_CALLS = 52320
PHONE_SERIAL = "3C15AU002CL00000"
GPU_UUID = "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08"
N_EMBD = 3840
N_FF = 11136


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise run_trace.RunError(f"cannot read {path}: {error}") from error
    run_trace.require(type(value) is dict, f"{path}: JSON object")
    return value


def finite_positive(value: object, label: str) -> float:
    run_trace.require(
        isinstance(value, (int, float)) and
        not isinstance(value, bool) and
        math.isfinite(float(value)) and
        float(value) > 0,
        label,
    )
    return float(value)


def percent(treatment: float, control: float) -> float:
    return 100.0 * (treatment / control - 1.0)


def validate_trace(path: Path) -> dict[str, dict[str, Any]]:
    run_trace.require(run_trace.digest_file(path) == TRACE_SHA256, "trace hash")
    rows = run_trace.read_jsonl(path)
    run_trace.require(len(rows) == EXPECTED_REQUESTS, "trace request count")
    run_trace.require(
        sum(row["input_tokens"] for row in rows) == EXPECTED_INPUT_TOKENS,
        "trace input work",
    )
    run_trace.require(
        sum(row["output_tokens"] for row in rows) == EXPECTED_OUTPUT_TOKENS,
        "trace output work",
    )
    by_id = {row["event_id"]: row for row in rows}
    run_trace.require(len(by_id) == len(rows), "trace event identity")
    return by_id


def validate_phone_energy(path: Path, result_path: Path,
                          start_ns: int, end_ns: int) -> dict[str, float]:
    value = load_json(path)
    duration_s = (end_ns - start_ns) / 1e9
    run_trace.require(
        value.get("schema") == "s41-phone-energy-v3" and
        value.get("status") == "PASS" and
        value.get("serial") == PHONE_SERIAL and
        value.get("boundary") == "paid_trace_interval" and
        value.get("input_sha256", {}).get("trace_result") ==
        run_trace.digest_file(result_path),
        f"{path}: binding",
    )
    measured_duration = finite_positive(value.get("duration_s"), f"{path}: duration")
    run_trace.require(
        math.isclose(measured_duration, duration_s, rel_tol=0.0, abs_tol=1e-9),
        f"{path}: interval",
    )
    usb_j = finite_positive(value.get("usb_input_energy_j"), f"{path}: USB energy")
    battery_j = value.get("battery_discharge_energy_j")
    run_trace.require(
        isinstance(battery_j, (int, float)) and
        not isinstance(battery_j, bool) and
        math.isfinite(float(battery_j)) and
        float(battery_j) >= 0,
        f"{path}: battery energy",
    )
    total_j = finite_positive(value.get("whole_phone_energy_j"), f"{path}: total")
    run_trace.require(
        math.isclose(total_j, usb_j + float(battery_j), rel_tol=1e-9),
        f"{path}: energy sum",
    )
    power_w = finite_positive(
        value.get("whole_phone_average_power_w"), f"{path}: power"
    )
    run_trace.require(
        math.isclose(total_j, power_w * duration_s, rel_tol=1e-9),
        f"{path}: power times duration",
    )
    return {
        "battery_discharge_j": float(battery_j),
        "energy_j": total_j,
        "power_w": power_w,
        "usb_input_j": usb_j,
    }


def validate_server_energy(value: object, duration_s: float,
                           label: str) -> dict[str, float]:
    run_trace.require(type(value) is dict, f"{label}: server energy")
    cpu_j = finite_positive(value.get("cpu_package_energy_j"), f"{label}: CPU energy")
    gpu_j = finite_positive(value.get("gpu_board_energy_j"), f"{label}: GPU energy")
    total_j = finite_positive(
        value.get("server_compute_device_energy_j"), f"{label}: server total"
    )
    cpu_w = finite_positive(
        value.get("cpu_package_average_power_w"), f"{label}: CPU power"
    )
    gpu_w = finite_positive(
        value.get("gpu_board_average_power_w"), f"{label}: GPU power"
    )
    run_trace.require(
        value.get("boundary") == "paid_trace_interval" and
        value.get("unaccounted") == [
            "AC conversion",
            "DRAM outside package RAPL",
            "fans",
            "motherboard",
            "storage",
        ],
        f"{label}: server boundary",
    )
    run_trace.require(
        math.isclose(total_j, cpu_j + gpu_j, rel_tol=1e-9) and
        math.isclose(cpu_j, cpu_w * duration_s, rel_tol=1e-9) and
        math.isclose(gpu_j, gpu_w * duration_s, rel_tol=1e-9),
        f"{label}: server energy reconciliation",
    )
    return {
        "cpu_package_j": cpu_j,
        "cpu_package_w": cpu_w,
        "gpu_board_j": gpu_j,
        "gpu_board_w": gpu_w,
        "server_j": total_j,
    }


def validate_result(root: Path, mode: str, repeat_index: int,
                    trace: dict[str, dict[str, Any]]) -> dict[str, Any]:
    run_trace.require(root.is_absolute(), "result root must be absolute")
    result_path = root / "RESULT.json"
    phone_path = Path(f"{root}.phone-capture") / "PHONE_ENERGY_V3.json"
    value = load_json(result_path)
    run_trace.require(
        value.get("schema") == "s41-burstgpt-llama-server-result-v1" and
        value.get("status") == "PASS" and
        value.get("mode") == mode and
        value.get("repeat_index") == repeat_index,
        f"{root}: result identity",
    )
    preflight = value.get("preflight")
    metrics = value.get("metrics")
    resources = value.get("resources")
    run_trace.require(
        type(preflight) is dict and
        preflight.get("requests_sha256") == TRACE_SHA256 and
        preflight.get("mode") == mode and
        preflight.get("cold_runtime") == {
            "batch_size": 4096,
            "context": 32768,
            "parallel": 8,
            "repack": "default",
            "thread_selection": "llama_server_default",
            "threads": -1,
            "ubatch_size": 512,
        } and
        preflight.get("split_policy", {}).get("id") == POLICY_ID and
        preflight.get("split_policy", {}).get("table") == POLICY_TABLE,
        f"{root}: preflight",
    )
    run_trace.require(
        preflight.get("cpu_affinity") == {
            "bridge": None,
            "cold": None,
            "control": None,
            "hot": None,
        } and
        preflight.get("gpu", {}).get("uuid") == GPU_UUID and
        preflight.get("gpu", {}).get("name") == "NVIDIA GeForce RTX 4060 Ti" and
        preflight.get("hot_runtime") == {"context": 24576, "parallel": 4},
        f"{root}: default CPU and GPU identity",
    )
    runtime_identity = {
        key: preflight.get(key)
        for key in (
            "cold_model_sha256",
            "cold_runtime",
            "cold_runtime_manifest",
            "cold_server_sha256",
            "hot_model_sha256",
            "hot_runtime",
            "hot_runtime_manifest",
            "hot_server_sha256",
        )
    }
    run_trace.require(
        all(
            isinstance(runtime_identity[key], str) and
            len(runtime_identity[key]) == 64
            for key in (
                "cold_model_sha256",
                "cold_server_sha256",
                "hot_model_sha256",
                "hot_server_sha256",
            )
        ) and
        all(
            isinstance(runtime_identity[key], list) and runtime_identity[key]
            for key in ("cold_runtime_manifest", "hot_runtime_manifest")
        ),
        f"{root}: runtime identity",
    )
    by_role = metrics.get("by_role") if type(metrics) is dict else None
    run_trace.require(
        type(metrics) is dict and
        metrics.get("completed") == EXPECTED_REQUESTS and
        metrics.get("output_tokens") == EXPECTED_OUTPUT_TOKENS and
        type(metrics.get("slo_met")) is int and
        type(by_role) is dict and
        by_role.get("cold", {}).get("completed") == EXPECTED_COLD_REQUESTS and
        by_role.get("cold", {}).get("output_tokens") ==
        EXPECTED_COLD_OUTPUT_TOKENS and
        by_role.get("hot", {}).get("completed") == EXPECTED_HOT_REQUESTS and
        by_role.get("hot", {}).get("output_tokens") ==
        EXPECTED_HOT_OUTPUT_TOKENS and
        type(resources) is dict and
        resources.get("cold_swap_max_bytes") == 0 and
        resources.get("hot_swap_max_bytes") == 0,
        f"{root}: completed work and swap",
    )
    start_ns = value.get("paid_start_ns")
    end_ns = value.get("paid_end_ns")
    run_trace.require(
        type(start_ns) is int and type(end_ns) is int and end_ns > start_ns,
        f"{root}: paid interval",
    )
    duration_s = finite_positive(metrics.get("duration_s"), f"{root}: duration")
    run_trace.require(
        math.isclose(duration_s, (end_ns - start_ns) / 1e9,
                     rel_tol=0.0, abs_tol=1e-9),
        f"{root}: duration binding",
    )
    rows = value.get("request_results")
    run_trace.require(
        type(rows) is list and len(rows) == EXPECTED_REQUESTS,
        f"{root}: request rows",
    )
    tokens: dict[str, tuple[int, ...]] = {}
    role_counts = {"cold": 0, "hot": 0}
    role_by_event: dict[str, str] = {}
    for row in rows:
        event_id = row.get("event_id") if type(row) is dict else None
        row_tokens = row.get("tokens") if type(row) is dict else None
        role = row.get("role") if type(row) is dict else None
        run_trace.require(
            type(event_id) is str and event_id in trace and event_id not in tokens and
            role in role_counts and type(row_tokens) is list and
            all(type(token) is int for token in row_tokens) and
            len(row_tokens) == trace[event_id]["output_tokens"],
            f"{root}: request output",
        )
        tokens[event_id] = tuple(row_tokens)
        role_counts[role] += 1
        role_by_event[event_id] = role
    run_trace.require(
        role_counts == {
            "cold": EXPECTED_COLD_REQUESTS,
            "hot": EXPECTED_HOT_REQUESTS,
        },
        f"{root}: roles",
    )
    server = validate_server_energy(value.get("server_energy"), duration_s, str(root))
    phone = validate_phone_energy(phone_path, result_path, start_ns, end_ns)
    if mode == "cpu":
        run_trace.require(
            value.get("phone") == {"bridge": None, "ffn": None, "shapes": []},
            f"{root}: control phone work",
        )
        work = None
    else:
        phone_work = value.get("phone")
        run_trace.require(type(phone_work) is dict, f"{root}: phone work")
        ffn = phone_work.get("ffn")
        bridge = phone_work.get("bridge")
        shapes = phone_work.get("shapes")
        run_trace.require(
            type(ffn) is dict and ffn.get("status") == "ok" and
            type(bridge) is dict and bridge.get("status") == "ok" and
            bridge.get("reset_recoveries") == 0 and
            type(shapes) is list and shapes and
            ffn.get("calls") == bridge.get("calls") ==
            sum(shape["calls"] for shape in shapes) and
            ffn.get("upload_bytes") == bridge.get("upload_bytes") and
            ffn.get("download_bytes") == bridge.get("download_bytes"),
            f"{root}: phone counters",
        )
        policy = run_trace.parse_prefill_policy(POLICY_TABLE)
        run_trace.require(
            all(
                type(shape) is dict and
                type(shape.get("calls")) is int and shape["calls"] > 0 and
                type(shape.get("tokens")) is int and shape["tokens"] > 0 and
                shape.get("columns") ==
                run_trace.policy_columns(policy, shape["tokens"])
                for shape in shapes
            ),
            f"{root}: phone shape policy",
        )
        token_rows = sum(shape["calls"] * shape["tokens"] for shape in shapes)
        run_trace.require(
            ffn["calls"] == EXPECTED_PHONE_CALLS and
            ffn["upload_bytes"] == token_rows * N_EMBD * 2 and
            ffn["download_bytes"] == token_rows * N_EMBD * 2,
            f"{root}: exact phone work",
        )
        wait_fraction = finite_positive(
            ffn.get("wait_mean_ms"), f"{root}: mean wait"
        ) / finite_positive(ffn.get("overlap_mean_ms"), f"{root}: mean overlap")
        weighted_columns = sum(
            shape["calls"] * shape["tokens"] * shape["columns"]
            for shape in shapes
        )
        eligible_columns = sum(
            shape["calls"] * shape["tokens"] * N_FF for shape in shapes
        )
        work = {
            "calls": ffn["calls"],
            "download_bytes": ffn["download_bytes"],
            "eligible_ffn_mac_fraction": weighted_columns / eligible_columns,
            "exposed_join_wait_fraction": wait_fraction,
            "phone_macs": 3 * N_EMBD * weighted_columns,
            "reset_recoveries": bridge["reset_recoveries"],
            "upload_bytes": ffn["upload_bytes"],
        }
    return {
        "duration_s": duration_s,
        "fleet_j": server["server_j"] + phone["energy_j"],
        "mode": mode,
        "paid_end_ns": end_ns,
        "paid_start_ns": start_ns,
        "path": str(result_path),
        "phone": phone,
        "phone_energy_path": str(phone_path),
        "phone_energy_sha256": run_trace.digest_file(phone_path),
        "repeat_index": repeat_index,
        "result_sha256": run_trace.digest_file(result_path),
        "roles": role_by_event,
        "runtime_identity": runtime_identity,
        "server": server,
        "slo_met": metrics["slo_met"],
        "tokens": tokens,
        "work": work,
    }


def validate_mmlu(path: Path, mode: str) -> dict[str, Any]:
    value = load_json(path)
    run_trace.require(
        value.get("schema") == "s41-gemma-op15-mmlu64-v1" and
        value.get("status") == "PASS" and value.get("mode") == mode and
        value.get("corpus_sha256") == MMLU_SHA256 and
        value.get("parseable") == 64 and len(value.get("items", [])) == 64,
        f"{path}: MMLU identity",
    )
    correct = sum(item.get("correct") is True for item in value["items"])
    run_trace.require(correct == value.get("correct"), f"{path}: MMLU score")
    manifest = value.get("server_runtime_manifest")
    run_trace.require(type(manifest) is list and manifest, f"{path}: runtime")
    manifest_signature = tuple(
        (entry.get("sha256"), entry.get("size_bytes"))
        for entry in manifest
        if type(entry) is dict
    )
    run_trace.require(
        len(manifest_signature) == len(manifest) and
        all(
            type(digest) is str and len(digest) == 64 and
            type(size) is int and size > 0
            for digest, size in manifest_signature
        ),
        f"{path}: runtime manifest",
    )
    if mode == "cpu":
        run_trace.require(
            value.get("phone") == {"bridge": None, "ffn": None, "shapes": []},
            f"{path}: CPU phone work",
        )
    else:
        run_trace.require(
            value.get("split_policy") == {
                "id": POLICY_ID,
                "table": POLICY_TABLE,
            } and
            value.get("phone", {}).get("bridge", {}).get("status") == "ok" and
            value.get("phone", {}).get("bridge", {}).get("reset_recoveries") == 0 and
            value.get("phone", {}).get("ffn", {}).get("status") == "ok",
            f"{path}: treatment route",
        )
    return {
        "answers": [item["answer"] for item in value["items"]],
        "correct": correct,
        "duration_s": finite_positive(value.get("duration_s"), f"{path}: duration"),
        "path": str(path),
        "runtime_manifest_signature": manifest_signature,
        "sha256": run_trace.digest_file(path),
    }


def arithmetic_mean(rows: list[dict[str, Any]], path: tuple[str, ...]) -> float:
    values = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value[key]
        values.append(float(value))
    return statistics.fmean(values)


def token_diagnostic(control: dict[str, Any],
                     treatment: dict[str, Any]) -> dict[str, int]:
    result = {}
    for role in ("cold", "hot"):
        event_ids = [
            event_id for event_id, event_role in control["roles"].items()
            if event_role == role
        ]
        result[f"{role}_requests_exact"] = sum(
            control["tokens"][event_id] == treatment["tokens"][event_id]
            for event_id in event_ids
        )
        result[f"{role}_requests_total"] = len(event_ids)
    return result


def alternating_intervals(controls: list[dict[str, Any]],
                          treatments: list[dict[str, Any]]) -> bool:
    for index, (control, treatment) in enumerate(zip(controls, treatments)):
        if not (
            control["paid_start_ns"] < control["paid_end_ns"] <
            treatment["paid_start_ns"] < treatment["paid_end_ns"]
        ):
            return False
        if (
            index + 1 < len(controls) and
            treatment["paid_end_ns"] >= controls[index + 1]["paid_start_ns"]
        ):
            return False
    return True


def summarize(controls: list[dict[str, Any]], treatments: list[dict[str, Any]],
              quality_control: dict[str, Any], quality_treatment: dict[str, Any],
              trace: dict[str, dict[str, Any]]) -> dict[str, Any]:
    run_trace.require(
        len(controls) == len(treatments) and len(controls) >= 3,
        "minimum three pairs",
    )
    control_duration = arithmetic_mean(controls, ("duration_s",))
    treatment_duration = arithmetic_mean(treatments, ("duration_s",))
    control_server = arithmetic_mean(controls, ("server", "server_j"))
    treatment_server = arithmetic_mean(treatments, ("server", "server_j"))
    control_fleet = arithmetic_mean(controls, ("fleet_j",))
    treatment_fleet = arithmetic_mean(treatments, ("fleet_j",))
    wait_fractions = [row["work"]["exposed_join_wait_fraction"] for row in treatments]
    fleet_change = percent(treatment_fleet, control_fleet)
    duration_change = percent(treatment_duration, control_duration)
    mmlu_delta = quality_treatment["correct"] - quality_control["correct"]
    run_trace.require(
        quality_control["runtime_manifest_signature"] ==
        quality_treatment["runtime_manifest_signature"],
        "MMLU runtime differs between arms",
    )
    runtime_identity = controls[0]["runtime_identity"]
    run_trace.require(
        all(
            row["runtime_identity"] == runtime_identity
            for row in controls + treatments
        ),
        "trace runtime differs between arms",
    )
    gates = {
        "all_pairs_reduce_fleet_energy": all(
            treatment["fleet_j"] < control["fleet_j"]
            for control, treatment in zip(controls, treatments)
        ),
        "all_pairs_reduce_makespan": all(
            treatment["duration_s"] < control["duration_s"]
            for control, treatment in zip(controls, treatments)
        ),
        "fleet_energy_saving_at_least_10_percent": fleet_change <= -10.0,
        "mean_join_wait_at_most_5_percent":
            statistics.fmean(wait_fractions) <= 0.05,
        "mmlu64_absolute_floor": quality_treatment["correct"] >= 25,
        "mmlu64_regression_at_most_one": mmlu_delta >= -1,
        "slo_count_not_lower": all(
            treatment["slo_met"] >= control["slo_met"]
            for control, treatment in zip(controls, treatments)
        ),
        "three_alternating_pairs":
            len(controls) >= 3 and alternating_intervals(controls, treatments),
        "zero_bridge_recoveries": all(
            row["work"]["reset_recoveries"] == 0 for row in treatments
        ),
    }
    pairs = []
    for control, treatment in zip(controls, treatments):
        pairs.append({
            "control": {
                "duration_s": control["duration_s"],
                "fleet_j": control["fleet_j"],
                "path": control["path"],
                "phone_energy_path": control["phone_energy_path"],
                "phone_energy_sha256": control["phone_energy_sha256"],
                "result_sha256": control["result_sha256"],
                "server_j": control["server"]["server_j"],
            },
            "diagnostic_token_equality": token_diagnostic(control, treatment),
            "fleet_energy_change_pct": percent(
                treatment["fleet_j"], control["fleet_j"]
            ),
            "makespan_change_pct": percent(
                treatment["duration_s"], control["duration_s"]
            ),
            "repeat_index": control["repeat_index"],
            "treatment": {
                "duration_s": treatment["duration_s"],
                "fleet_j": treatment["fleet_j"],
                "path": treatment["path"],
                "phone_energy_path": treatment["phone_energy_path"],
                "phone_energy_sha256": treatment["phone_energy_sha256"],
                "result_sha256": treatment["result_sha256"],
                "server_j": treatment["server"]["server_j"],
            },
        })
    work = treatments[0]["work"]
    run_trace.require(
        all(
            row["work"][key] == work[key]
            for row in treatments[1:]
            for key in ("calls", "download_bytes", "phone_macs", "upload_bytes")
        ),
        "treatment work differs across pairs",
    )
    result = {
        "boundary": {
            "accounted": [
                "Intel package RAPL",
                "RTX 4060 Ti NVML board power",
                "OP15 USB input plus simultaneous battery discharge",
            ],
            "excluded": [
                "AC conversion",
                "DRAM outside package RAPL",
                "fans",
                "motherboard",
                "storage",
            ],
            "scope": "accounted compute-device fleet energy, not AC wall energy",
        },
        "gates": gates,
        "headline": {
            "control": {
                "cpu_package_j": arithmetic_mean(controls, ("server", "cpu_package_j")),
                "duration_s": control_duration,
                "fleet_j": control_fleet,
                "gpu_board_j": arithmetic_mean(controls, ("server", "gpu_board_j")),
                "phone_j": arithmetic_mean(controls, ("phone", "energy_j")),
                "server_j": control_server,
            },
            "fleet_energy_change_pct": fleet_change,
            "makespan_change_pct": duration_change,
            "server_energy_change_pct": percent(treatment_server, control_server),
            "treatment": {
                "cpu_package_j": arithmetic_mean(treatments, ("server", "cpu_package_j")),
                "duration_s": treatment_duration,
                "fleet_j": treatment_fleet,
                "gpu_board_j": arithmetic_mean(treatments, ("server", "gpu_board_j")),
                "phone_j": arithmetic_mean(treatments, ("phone", "energy_j")),
                "server_j": treatment_server,
            },
        },
        "iteration": "I3",
        "pairs": pairs,
        "phone_work": {
            **work,
            "mean_exposed_join_wait_fraction": statistics.fmean(wait_fractions),
        },
        "quality": {
            "control": quality_control,
            "diagnostic_only_trace_tokens": True,
            "mmlu64_score_delta": mmlu_delta,
            "treatment": quality_treatment,
        },
        "repetitions": len(controls),
        "schema": "s41-i3-physical-campaign-v1",
        "trace": {
            "input_tokens": EXPECTED_INPUT_TOKENS,
            "output_tokens": EXPECTED_OUTPUT_TOKENS,
            "requests": EXPECTED_REQUESTS,
            "sha256": TRACE_SHA256,
        },
        "verdict": "PASS" if all(gates.values()) else "FAIL",
    }
    digest_value = dict(result)
    result["record_sha256"] = hashlib.sha256(run_trace.canonical(digest_value)).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, action="append", required=True)
    parser.add_argument("--treatment-root", type=Path, action="append", required=True)
    parser.add_argument("--mmlu-control", type=Path, required=True)
    parser.add_argument("--mmlu-treatment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_trace.require(
        args.output.is_absolute() and not args.output.exists(), "output path"
    )
    run_trace.require(
        len(args.control_root) == len(args.treatment_root), "pair count"
    )
    trace = validate_trace(args.trace)
    controls = [
        validate_result(root, "cpu", index, trace)
        for index, root in enumerate(args.control_root, 1)
    ]
    treatments = [
        validate_result(root, "op15", index, trace)
        for index, root in enumerate(args.treatment_root, 1)
    ]
    result = summarize(
        controls,
        treatments,
        validate_mmlu(args.mmlu_control, "cpu"),
        validate_mmlu(args.mmlu_treatment, "op15"),
        trace,
    )
    run_trace.write_json(args.output, result)
    print(json.dumps({
        "output": str(args.output),
        "record_sha256": result["record_sha256"],
        "verdict": result["verdict"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
