#!/usr/bin/env python3
"""Validate and compare the two-F16-model BurstGPT ABBA campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


TRACE_SHA256 = (
    "b20a9ba66ee3558d835a0e19ed3cfa4"
    "c31a4a9e8b4f9c085b29a14f80250a0ff"
)
QWEN_SHA256 = (
    "d89e9e823744222e595e0b3c8fd5436c"
    "e5d3a6a446fa42492ebce6064dfa9718"
)
GEMMA_SHA256 = (
    "ed76f2183d2d1d65091986033023e6c7"
    "8d27f6276c1b0c5826cc92acf73538cf"
)
PHONE_SERIAL = "3C15AU002CL00000"
MIN_AVAILABLE_KIB = 2 * 1024 * 1024
PHASES = {
    "qwen": {
        "columns": 17408,
        "layer_count": 12,
        "layer_mask": "0000000000000fff",
        "max_tokens": 4,
        "n_embd": 5120,
        "route_control": "qwen_cuda_cpu_f16",
        "route_treatment": "qwen_cuda_cpu_op15_f16",
        "server_log": "hot-server.stderr",
        "bridge_log": "hot-phone/dmabuf-bridge.stderr",
    },
    "gemma": {
        "columns": 6144,
        "layer_count": 23,
        "layer_mask": "00000000007fffff",
        "max_tokens": 16,
        "n_embd": 3840,
        "route_control": "gemma_cuda_cpu_f16",
        "route_treatment": "gemma_cuda_cpu_op15_f16",
        "server_log": "cold-server.stderr",
        "bridge_log": "cold-phone/dmabuf-bridge.stderr",
    },
}


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


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read {path}: {exc}") from exc
    require(type(value) is dict, f"object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise AnalysisError(f"cannot read {path}: {exc}") from exc
    rows = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AnalysisError(f"malformed JSONL: {path}") from exc
        require(type(value) is dict, f"JSONL object: {path}")
        rows.append(value)
    require(rows, f"nonempty JSONL: {path}")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while block := source.read(4 * 1024 * 1024):
                digest.update(block)
    except OSError as exc:
        raise AnalysisError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def positive(name: str, value: object) -> float:
    require(
        type(value) in (int, float)
        and math.isfinite(value)
        and value > 0,
        f"positive {name}",
    )
    return float(value)


def close(name: str, left: float, right: float, tolerance: float = 1e-6) -> None:
    if not math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance):
        raise AnalysisError(f"{name}: {left} != {right}")


def prefixed_objects(path: Path, prefix: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    except (OSError, UnicodeError) as exc:
        raise AnalysisError(f"cannot read {path}: {exc}") from exc
    rows = []
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
        rows.append(value)
    return rows


def one_prefixed(path: Path, prefix: str) -> dict[str, Any]:
    rows = prefixed_objects(path, prefix)
    require(len(rows) == 1, f"one {prefix.strip()}: {path}")
    return rows[0]


def validate_plan(path: Path) -> dict[str, Any]:
    plan = read_object(path)
    claimed_hash = plan.get("plan_sha256")
    unsigned = dict(plan)
    unsigned.pop("plan_sha256", None)
    schema = plan.get("schema")
    require(
        schema in {
            "s42-full-fp16-burstgpt-plan-v1",
            "s42-full-fp16-burstgpt-plan-v2",
        }
        and plan.get("status") == "PASS"
        and type(claimed_hash) is str
        and hashlib.sha256(canonical(unsigned)).hexdigest() == claimed_hash,
        f"scheduler plan identity: {path}",
    )
    require(
        plan.get("trace") == {
            "input_tokens": 33_843,
            "output_tokens": 11_605,
            "requests": 74,
            "sha256": TRACE_SHA256,
        }
        and plan.get("artifacts", {}).get("qwen") == {
            "bytes": 29_543_423_360,
            "sha256": QWEN_SHA256,
        }
        and plan.get("artifacts", {}).get("gemma") == {
            "bytes": 23_832_065_056,
            "sha256": GEMMA_SHA256,
        }
        and plan.get("placement") == {
            "gemma_gpu_layers": 25,
            "qwen_gpu_layers": 18,
        },
        f"scheduler workload identity: {path}",
    )
    qwen = plan.get("phases", {}).get("qwen")
    gemma = plan.get("phases", {}).get("gemma")
    require(
        type(qwen) is dict
        and qwen.get("phone_policy") == "4:17408,512:0"
        and qwen.get("physical_m_max") == 4
        and qwen.get("execution_mode") == "full_replacement"
        and qwen.get("decision", {}).get("decision_reason")
            == "ENERGY_POSITIVE_RESIDENT_PHONE_ROUTE"
        and type(gemma) is dict
        and gemma.get("phone_policy") == "1:6144,16:6144,512:0"
        and gemma.get("physical_m_max") == 16
        and gemma.get("execution_mode") == "parallel_split"
        and gemma.get("decision", {}).get("decision_reason")
            == "ENERGY_POSITIVE_RESIDENT_PHONE_ROUTE",
        f"scheduler phase policy: {path}",
    )
    if schema == "s42-full-fp16-burstgpt-plan-v2":
        validate_runtime_placement(plan, path)
    return plan


def validate_runtime_placement(plan: dict[str, Any], path: Path) -> None:
    decision = plan.get("runtime_placement")
    require(type(decision) is dict, f"runtime placement receipt: {path}")
    claimed_hash = decision.get("decision_sha256")
    unsigned = dict(decision)
    unsigned.pop("decision_sha256", None)
    calculated_hash = "sha256:" + hashlib.sha256(
        canonical(unsigned).removesuffix(b"\n")
    ).hexdigest()
    baseline = decision.get("baseline")
    selected = decision.get("selected")
    snapshot = decision.get("snapshot")
    require(
        decision.get("schema") == "research-scheduler-runtime-placement-v1"
        and type(claimed_hash) is str
        and claimed_hash == calculated_hash
        and type(baseline) is dict
        and type(selected) is dict
        and type(snapshot) is dict,
        f"runtime placement identity: {path}",
    )
    require(
        baseline.get("candidate_id")
            == "fp16-server-gpu-cpu-switch-v1"
        and selected.get("candidate_id")
            == "fp16-server-gpu-cpu-op15-switch-v1"
        and baseline.get("work_set_sha256") == "sha256:" + TRACE_SHA256
        and selected.get("work_set_sha256") == "sha256:" + TRACE_SHA256
        and baseline.get("energy_boundary_id")
            == "cpu-package+gpu-board+whole-phone"
        and selected.get("energy_boundary_id")
            == "cpu-package+gpu-board+whole-phone"
        and selected.get("placement_verified") is True
        and selected.get("workload_verified") is True
        and selected.get("status") == "measured"
        and decision.get("conservative_energy_saving_ppm", -1)
            >= decision.get("minimum_energy_saving_ppm", 0)
        and decision.get("conservative_latency_change_ppm", 1) < 0,
        f"runtime placement selection: {path}",
    )
    bindings = selected.get("runtime_bindings")
    placement = plan.get("placement")
    phases = plan.get("phases")
    require(
        type(bindings) is dict
        and type(placement) is dict
        and type(phases) is dict
        and bindings.get("phone_execution") == "qualified"
        and bindings.get("placement_mode")
            == "sequential-partial-model-switch"
        and bindings.get("qwen_gpu_layers")
            == placement.get("qwen_gpu_layers")
        and bindings.get("gemma_gpu_layers")
            == placement.get("gemma_gpu_layers"),
        f"runtime placement binding: {path}",
    )
    for phase in ("qwen", "gemma"):
        phase_plan = phases.get(phase)
        require(
            type(phase_plan) is dict
            and bindings.get(f"{phase}_execution_mode")
                == phase_plan.get("execution_mode")
            and bindings.get(f"{phase}_n_embd")
                == phase_plan.get("n_embd")
            and bindings.get(f"{phase}_phone_columns")
                == phase_plan.get("phone_columns")
            and bindings.get(f"{phase}_phone_layer_mask")
                == phase_plan.get("layer_mask")
            and bindings.get(f"{phase}_phone_policy")
                == phase_plan.get("phone_policy")
            and bindings.get(f"{phase}_physical_m_max")
                == phase_plan.get("physical_m_max"),
            f"runtime {phase} binding: {path}",
        )
    capacities = snapshot.get("capacities")
    additional = selected.get("additional_bytes")
    require(
        type(capacities) is list and type(additional) is dict,
        f"runtime capacity receipt: {path}",
    )
    available = {
        row.get("resource_id"): row.get("available_bytes")
        for row in capacities
        if type(row) is dict
    }
    require(
        all(
            type(resource_id) is str
            and type(required_bytes) is int
            and type(available.get(resource_id)) is int
            and required_bytes <= available[resource_id]
            for resource_id, required_bytes in additional.items()
        ),
        f"selected runtime placement capacity: {path}",
    )


def validate_phone(path: Path, result_path: Path, duration_s: float) -> dict[str, Any]:
    value = read_object(path)
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
    usb = positive("USB input energy", value.get("usb_input_energy_j"))
    battery = value.get("battery_discharge_energy_j")
    require(
        type(battery) in (int, float)
        and math.isfinite(battery)
        and battery >= 0,
        f"phone battery energy: {path}",
    )
    whole = positive("whole-phone energy", value.get("whole_phone_energy_j"))
    close("whole-phone sum", whole, usb + float(battery))
    require(
        value.get("input_sha256", {}).get("trace_result")
            == sha256(result_path),
        f"phone result binding: {path}",
    )
    return value


def validate_resources(path: Path, start_ns: int, end_ns: int) -> dict[str, Any]:
    rows = read_jsonl(path)
    require(
        rows[0].get("t_ns", end_ns) <= start_ns
        and rows[-1].get("t_ns", start_ns) >= end_ns,
        f"resource interval coverage: {path}",
    )
    swap_max = 0
    available_min = None
    for row in rows:
        require(
            row.get("schema") == "s41-hierarchical-resource-v1"
            and type(row.get("pids")) is dict
            and type(row.get("system")) is dict,
            f"resource row identity: {path}",
        )
        for process in row["pids"].values():
            require(type(process) is dict, f"resource process: {path}")
            swap = process.get("swap_bytes")
            require(type(swap) is int and swap >= 0, f"process swap: {path}")
            swap_max = max(swap_max, swap)
        available = row["system"].get("available_bytes")
        require(type(available) is int and available > 0,
                f"system memory: {path}")
        available_min = available if available_min is None else min(
            available_min, available
        )
    require(swap_max == 0, f"nonzero process swap: {path}")
    return {
        "process_swap_max_bytes": swap_max,
        "samples": len(rows),
        "system_available_min_bytes": available_min,
    }


def validate_phase_logs(
    root: Path,
    phase: str,
    treatment: bool,
) -> dict[str, Any]:
    spec = PHASES[phase]
    bridge = one_prefixed(root / spec["bridge_log"], "FFNDMABUF ")
    server_rows = prefixed_objects(
        root / spec["server_log"], "S41SERVERFFN "
    )
    shapes = prefixed_objects(
        root / spec["server_log"], "S41SERVERFFNSHAPE "
    )
    calls = bridge.get("calls")
    require(
        bridge.get("status") == "ok"
        and bridge.get("allocator") == "malloc-split"
        and type(calls) is int
        and calls >= 0
        and bridge.get("reset_recoveries") == 0,
        f"{phase} bridge receipt: {root}",
    )
    if not treatment:
        require(calls == 0 and not server_rows and not shapes,
                f"{phase} control phone work: {root}")
        return {"calls": 0, "reset_recoveries": 0, "shapes": []}

    require(len(server_rows) == 1 and calls > 0,
            f"{phase} treatment summary: {root}")
    summary = server_rows[0]
    require(
        summary.get("status") == "ok"
        and summary.get("calls") == calls
        and summary.get("decode_calls", 0)
            + summary.get("prefill_calls", 0) == calls,
        f"{phase} treatment call conservation: {root}",
    )
    observed = set()
    shape_calls = 0
    for row in shapes:
        tokens = row.get("tokens")
        count = row.get("calls")
        require(
            type(tokens) is int
            and 1 <= tokens <= spec["max_tokens"]
            and row.get("columns") == spec["columns"]
            and type(count) is int
            and count > 0
            and count % spec["layer_count"] == 0,
            f"{phase} qualified shape: {root}",
        )
        observed.add(tokens)
        shape_calls += count
    require(
        shapes and 1 in observed and shape_calls == calls,
        f"{phase} shape accounting: {root}",
    )
    return {
        "calls": calls,
        "reset_recoveries": 0,
        "shapes": sorted(observed),
    }


def validate_resident_logs(
    capture: Path,
    phase_logs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    arms = prefixed_objects(capture / "router.log", "RESIDENTARM ")
    router = one_prefixed(capture / "router.log", "RESIDENTROUTER ")
    workers = one_prefixed(
        capture / "resident-workers.log", "RESIDENTWORKERS "
    )
    require(len(arms) == 2, f"two resident arms: {capture}")
    expected = {
        (
            spec["layer_mask"], spec["n_embd"], spec["columns"]
        )
        for spec in PHASES.values()
    }
    observed = {
        (row.get("layer_mask"), row.get("n_embd"), row.get("columns"))
        for row in arms
    }
    require(
        all(
            row.get("status") == "WARM"
            and row.get("target_count") in (1, 2)
            for row in arms
        )
        and observed == expected
        and router.get("status") == "ok"
        and router.get("sessions") == 2
        and router.get("requests")
            == sum(row["calls"] for row in phase_logs.values())
        and workers.get("status") == "WARM"
        and workers.get("sessions") == ["HTP0", "HTP1", "HTP2"],
        f"resident transport receipt: {capture}",
    )
    session_text = (capture / "session.log").read_text(encoding="ascii")
    memory = re.findall(r"mem_available_kib=(\d+)", session_text)
    require(
        len(memory) == 1 and int(memory[0]) >= MIN_AVAILABLE_KIB,
        f"phone memory reserve: {capture}",
    )
    return {
        "mem_available_kib": int(memory[0]),
        "requests": router["requests"],
        "sessions": router["sessions"],
    }


def validate_result(
    root: Path,
    arm: str,
    repeat: int,
) -> dict[str, Any]:
    treatment = arm == "op15"
    result_path = root / "RESULT.json"
    capture = Path(str(root) + ".phone-capture")
    plan_path = capture / "EXECUTION_PLAN.json"
    result = read_object(result_path)
    plan = validate_plan(plan_path)
    scheduler = result.get("fp16_resident_scheduler")
    require(
        result.get("schema") == "s41-hierarchical-burstgpt-result-v1"
        and result.get("status") == "PASS"
        and result.get("mode") == "fp16-switch"
        and type(scheduler) is dict
        and scheduler.get("arm") == arm
        and scheduler.get("plan_sha256") == plan["plan_sha256"]
        and scheduler.get("phase_decisions") == plan["phases"],
        f"trace scheduler binding: {root}",
    )
    if plan.get("schema") == "s42-full-fp16-burstgpt-plan-v2":
        require(
            scheduler.get("runtime_placement")
                == plan.get("runtime_placement"),
            f"runtime placement execution binding: {root}",
        )
    rows = result.get("request_results")
    metrics = result.get("metrics")
    require(
        type(rows) is list
        and len(rows) == 74
        and type(metrics) is dict
        and metrics.get("completed") == 74
        and metrics.get("output_tokens") == 11_605
        and {row.get("request_index") for row in rows} == set(range(74)),
        f"completed trace work: {root}",
    )
    expected_routes = {
        "hot": PHASES["qwen"][
            "route_treatment" if treatment else "route_control"
        ],
        "cold": PHASES["gemma"][
            "route_treatment" if treatment else "route_control"
        ],
    }
    counts = {"hot": 0, "cold": 0}
    input_tokens = {"hot": 0, "cold": 0}
    output_tokens = {"hot": 0, "cold": 0}
    for row in rows:
        role = row.get("role")
        tokens = row.get("tokens")
        require(
            role in counts
            and row.get("route") == expected_routes[role]
            and type(row.get("input_tokens")) is int
            and type(row.get("output_tokens")) is int
            and type(tokens) is list
            and len(tokens) == row["output_tokens"]
            and all(type(token) is int for token in tokens),
            f"request accounting: {root}",
        )
        counts[role] += 1
        input_tokens[role] += row["input_tokens"]
        output_tokens[role] += row["output_tokens"]
    require(
        counts == {"hot": 57, "cold": 17}
        and input_tokens == {"hot": 22_367, "cold": 11_476}
        and output_tokens == {"hot": 4_686, "cold": 6_919},
        f"trace token geometry: {root}",
    )
    start_ns = result.get("paid_start_ns")
    end_ns = result.get("paid_end_ns")
    require(
        type(start_ns) is int and type(end_ns) is int and end_ns > start_ns,
        f"paid trace interval: {root}",
    )
    duration_s = (end_ns - start_ns) / 1e9
    close("trace duration", duration_s,
          positive("trace duration", metrics.get("duration_s")))
    server = result.get("server_energy")
    require(
        type(server) is dict
        and server.get("boundary") == "paid_trace_interval",
        f"server energy boundary: {root}",
    )
    cpu_j = positive("CPU package energy", server.get("cpu_package_energy_j"))
    gpu_j = positive("GPU board energy", server.get("gpu_board_energy_j"))
    server_j = positive(
        "server compute energy", server.get("server_compute_device_energy_j")
    )
    close("server energy sum", server_j, cpu_j + gpu_j)
    switch = result.get("switch")
    require(type(switch) is dict, f"switch receipt: {root}")
    hot_end_s = positive("protected phase end", switch.get("hot_end_s"))
    gpu_ready_s = positive("promoted phase ready", switch.get("gpu_ready_s"))
    require(gpu_ready_s > hot_end_s, f"switch interval: {root}")
    phone = validate_phone(
        capture / "PHONE_ENERGY_V3.json", result_path, duration_s
    )
    phase_logs = {
        phase: validate_phase_logs(root, phase, treatment)
        for phase in PHASES
    }
    resident = validate_resident_logs(capture, phase_logs)
    resources = validate_resources(
        root / "resource-samples.jsonl", start_ns, end_ns
    )
    phone_j = float(phone["whole_phone_energy_j"])
    return {
        "arm": arm,
        "artifacts": {
            "phone_energy_sha256": sha256(capture / "PHONE_ENERGY_V3.json"),
            "plan_sha256": sha256(plan_path),
            "result_sha256": sha256(result_path),
            "resource_samples_sha256": sha256(
                root / "resource-samples.jsonl"
            ),
        },
        "cpu_package_j": cpu_j,
        "duration_s": duration_s,
        "fleet_j": server_j + phone_j,
        "gpu_board_j": gpu_j,
        "phase_logs": phase_logs,
        "phone_j": phone_j,
        "repeat_index": repeat,
        "resident": resident,
        "resources": resources,
        "server_j": server_j,
        "slo_met": metrics.get("slo_met"),
        "switch_s": gpu_ready_s - hot_end_s,
    }


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def change(control: float, treatment: float) -> float:
    return 100.0 * (treatment / control - 1.0)


def compact(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "arm", "artifacts", "cpu_package_j", "duration_s", "fleet_j",
        "gpu_board_j", "phase_logs", "phone_j", "repeat_index",
        "resident", "resources", "server_j", "slo_met", "switch_s",
    )
    return {key: row[key] for key in keys}


def analyze(roots: dict[str, Path]) -> dict[str, Any]:
    runs = {
        "control_r1": validate_result(roots["control_r1"], "control", 1),
        "treatment_r1": validate_result(roots["treatment_r1"], "op15", 1),
        "treatment_r2": validate_result(roots["treatment_r2"], "op15", 2),
        "control_r2": validate_result(roots["control_r2"], "control", 2),
    }
    controls = [runs["control_r1"], runs["control_r2"]]
    treatments = [runs["treatment_r1"], runs["treatment_r2"]]
    keys = (
        "duration_s", "cpu_package_j", "gpu_board_j", "server_j",
        "phone_j", "fleet_j", "switch_s",
    )
    control_mean = {key: mean(controls, key) for key in keys}
    treatment_mean = {key: mean(treatments, key) for key in keys}
    changes = {
        key + "_change_pct": change(control_mean[key], treatment_mean[key])
        for key in keys
    }
    pair_savings = [
        100.0 * (
            1.0 - treatments[index]["fleet_j"] / controls[index]["fleet_j"]
        )
        for index in range(2)
    ]
    validity_gates = {
        "equal_completed_work": True,
        "matched_two_f16_model_placement": True,
        "resident_sessions_two_phases": True,
        "synchronized_whole_phone_energy": True,
        "treatment_qualified_shapes_only": True,
        "treatment_zero_reset_recoveries": True,
        "unified_scheduler_plan_bound": True,
        "zero_process_swap": True,
    }
    outcome_gates = {
        "each_pair_fleet_energy_saving": all(row > 0 for row in pair_savings),
        "mean_fleet_energy_saving": changes["fleet_j_change_pct"] < 0,
        "mean_makespan_not_regressed": changes["duration_s_change_pct"] <= 0,
    }
    output = {
        "boundary": "cpu-package+gpu-board+whole-phone-paid-full-trace-v1",
        "changes": changes,
        "control_mean": control_mean,
        "energy_verdict": (
            "SAVING" if outcome_gates["mean_fleet_energy_saving"]
            else "REGRESSION"
        ),
        "outcome_gates": outcome_gates,
        "pair_fleet_energy_saving_pct": pair_savings,
        "run_order": [
            "control_r1", "treatment_r1", "treatment_r2", "control_r2"
        ],
        "runs": {name: compact(row) for name, row in runs.items()},
        "schema": "s42-full-fp16-burstgpt-abba-v1",
        "status": "PASS",
        "trace": {
            "input_tokens": 33_843,
            "output_tokens": 11_605,
            "requests": 74,
            "source_sha256": TRACE_SHA256,
        },
        "treatment_mean": treatment_mean,
        "validity_gates": validity_gates,
    }
    output["record_sha256"] = hashlib.sha256(canonical(output)).hexdigest()
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-r1", type=Path, required=True)
    parser.add_argument("--treatment-r1", type=Path, required=True)
    parser.add_argument("--treatment-r2", type=Path, required=True)
    parser.add_argument("--control-r2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.exists():
        parser.error("output must be an absolute new path")
    roots = {
        "control_r1": args.control_r1,
        "treatment_r1": args.treatment_r1,
        "treatment_r2": args.treatment_r2,
        "control_r2": args.control_r2,
    }
    try:
        value = analyze(roots)
        args.output.write_bytes(canonical(value))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"full FP16 ABBA analysis failed: {exc}\n")
    print(json.dumps({
        "fleet_energy_change_pct": value["changes"][
            "fleet_j_change_pct"
        ],
        "makespan_change_pct": value["changes"][
            "duration_s_change_pct"
        ],
        "record_sha256": value["record_sha256"],
        "verdict": value["energy_verdict"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
