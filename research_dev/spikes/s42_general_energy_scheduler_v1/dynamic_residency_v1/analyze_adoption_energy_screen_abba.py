#!/usr/bin/env python3
"""Validate the matched adoption energy screen A-B-B-A."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


TRACE_SHA256 = (
    "b20a9ba66ee3558d835a0e19ed3cfa4"
    "c31a4a9e8b4f9c085b29a14f80250a0ff"
)
QWEN_INDICES = [52, 53, 31, 54, 55, 47]
QWEN_INPUT_TOKENS = 936
QWEN_OUTPUT_TOKENS = 117
GEMMA_INPUT_TOKENS = 271
GEMMA_OUTPUT_TOKENS = 41
GPU_RESERVE_BYTES = 536_870_912
SAFETY_MARGIN_PCT = 2.0


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def finite_positive(value: object, name: str) -> float:
    require(
        type(value) in (int, float)
        and math.isfinite(value)
        and value > 0,
        name,
    )
    return float(value)


def validate_run(root: Path, arm: str) -> dict[str, Any]:
    result_path = root / "RESULT.json"
    capture = Path(str(root) + ".phone-capture")
    phone_path = capture / "PHONE_ENERGY_V3.json"
    cgroup_path = capture / "CGROUP_MEMORY_V1.json"
    result = read_object(result_path)
    claimed_hash = result.get("record_sha256")
    unsigned = dict(result)
    unsigned.pop("record_sha256", None)
    require(
        result.get("schema") == "s42-adoption-energy-screen-run-v1"
        and result.get("status") == "PASS"
        and result.get("arm") == arm
        and type(result.get("repeat_index")) is int
        and result["repeat_index"] > 0
        and type(claimed_hash) is str
        and hashlib.sha256(canonical(unsigned)).hexdigest() == claimed_hash,
        f"screen result identity: {root}",
    )
    require(
        result.get("artifacts", {}).get("source_trace_sha256") == TRACE_SHA256
        and result.get("placement") == {
            "gemma_gpu_layers": 1,
            "qwen_gpu_layers": 15,
        }
        and result.get("warmup_inside_boundary") is True
        and result.get("workload") == {
            "gemma_input_tokens": GEMMA_INPUT_TOKENS,
            "gemma_output_tokens": GEMMA_OUTPUT_TOKENS,
            "gemma_request_index": 50,
            "qwen_indices": QWEN_INDICES,
            "qwen_input_tokens": QWEN_INPUT_TOKENS,
            "qwen_output_tokens": QWEN_OUTPUT_TOKENS,
            "requests": 7,
        },
        f"screen workload identity: {root}",
    )
    requests = result.get("request_results")
    require(
        type(requests) is list
        and len(requests) == 7
        and {row.get("request_index") for row in requests}
            == set(QWEN_INDICES + [50])
        and sum(row.get("input_tokens", 0) for row in requests)
            == QWEN_INPUT_TOKENS + GEMMA_INPUT_TOKENS
        and sum(row.get("output_tokens", 0) for row in requests)
            == QWEN_OUTPUT_TOKENS + GEMMA_OUTPUT_TOKENS
        and all(
            type(row.get("tokens")) is list
            and len(row["tokens"]) == row["output_tokens"]
            and hashlib.sha256(canonical(row["tokens"])).hexdigest()
                == row.get("tokens_sha256")
            for row in requests
        ),
        f"screen request conservation: {root}",
    )
    start_ns = result.get("paid_start_ns")
    end_ns = result.get("paid_end_ns")
    source_ready_ns = result.get("source_ready_ns")
    gemma_ready_ns = result.get("gemma_ready_ns")
    require(
        type(start_ns) is int
        and type(source_ready_ns) is int
        and type(gemma_ready_ns) is int
        and type(end_ns) is int
        and start_ns < source_ready_ns < gemma_ready_ns < end_ns,
        f"screen transition timeline: {root}",
    )
    transition = result.get("transition")
    require(
        type(transition) is dict
        and transition.get("bytes") == 2_013_265_920
        and transition.get("copied_chunks") == 480
        and transition.get("copy_windows") == 54
        and transition.get("gpu_free_min_bytes", 0) >= GPU_RESERVE_BYTES
        and transition.get("gpu_reserve_bytes") == GPU_RESERVE_BYTES,
        f"screen transition receipt: {root}",
    )
    server = result.get("server_energy")
    require(
        type(server) is dict
        and server.get("boundary") == "paid_trace_interval",
        f"screen server energy: {root}",
    )
    cpu_j = finite_positive(server.get("cpu_package_energy_j"), "CPU energy")
    gpu_j = finite_positive(server.get("gpu_board_energy_j"), "GPU energy")
    server_j = finite_positive(
        server.get("server_compute_device_energy_j"), "server energy"
    )
    require(math.isclose(cpu_j + gpu_j, server_j), f"server sum: {root}")

    phone = read_object(phone_path)
    require(
        phone.get("schema") == "s41-phone-energy-v3"
        and phone.get("status") == "PASS"
        and phone.get("boundary") == "paid_trace_interval"
        and phone.get("battery_discharge_only") is True,
        f"screen phone energy: {root}",
    )
    phone_j = finite_positive(phone.get("whole_phone_energy_j"), "phone energy")
    duration_s = (end_ns - start_ns) / 1e9
    require(
        math.isclose(duration_s, float(phone.get("duration_s"))),
        f"phone duration: {root}",
    )

    cgroup = read_object(cgroup_path)
    cgroup_memory = cgroup.get("memory", {})
    require(
        cgroup.get("schema") == "s42-cgroup-memory-receipt-v1"
        and cgroup.get("status") == "PASS"
        and cgroup_memory.get("swap_max_bytes") == 0
        and cgroup_memory.get("swap_current_bytes") == 0
        and cgroup_memory.get("events", {}).get("oom", 0) == 0
        and cgroup_memory.get("events", {}).get("oom_kill", 0) == 0
        and cgroup_memory.get("events", {}).get("oom_group_kill", 0) == 0
        and result.get("process_swap_max_bytes") == 0,
        f"screen memory gates: {root}",
    )
    return {
        "arm": arm,
        "artifacts": {
            "cgroup_sha256": sha256(cgroup_path),
            "phone_sha256": sha256(phone_path),
            "result_sha256": sha256(result_path),
        },
        "cpu_package_j": cpu_j,
        "duration_s": duration_s,
        "fleet_j": server_j + phone_j,
        "gemma_ready_s": (gemma_ready_ns - start_ns) / 1e9,
        "gpu_board_j": gpu_j,
        "phone_j": phone_j,
        "repeat_index": result["repeat_index"],
        "request_token_hashes": {
            str(row["request_index"]): row["tokens_sha256"]
            for row in requests
        },
        "server_j": server_j,
        "source_ready_s": (source_ready_ns - start_ns) / 1e9,
        "transition": transition,
    }


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def change(control: float, treatment: float) -> float:
    return 100.0 * (treatment / control - 1.0)


def analyze(roots: dict[str, Path]) -> dict[str, Any]:
    runs = {
        "control_r1": validate_run(roots["control_r1"], "control"),
        "dynamic_r1": validate_run(roots["dynamic_r1"], "dynamic"),
        "dynamic_r2": validate_run(roots["dynamic_r2"], "dynamic"),
        "control_r2": validate_run(roots["control_r2"], "control"),
    }
    require(
        runs["control_r1"]["repeat_index"]
            == runs["dynamic_r1"]["repeat_index"]
        and runs["control_r2"]["repeat_index"]
            == runs["dynamic_r2"]["repeat_index"]
        and runs["control_r1"]["repeat_index"]
            != runs["control_r2"]["repeat_index"],
        "paired distinct repeat indices",
    )
    controls = [runs["control_r1"], runs["control_r2"]]
    dynamics = [runs["dynamic_r1"], runs["dynamic_r2"]]
    reference_hashes = runs["control_r1"]["request_token_hashes"]
    exact_equal_work = all(
        row["request_token_hashes"] == reference_hashes
        for row in runs.values()
    )
    keys = (
        "duration_s", "cpu_package_j", "gpu_board_j", "server_j",
        "phone_j", "fleet_j", "source_ready_s", "gemma_ready_s",
    )
    control_mean = {key: mean(controls, key) for key in keys}
    dynamic_mean = {key: mean(dynamics, key) for key in keys}
    changes = {
        key + "_change_pct": change(control_mean[key], dynamic_mean[key])
        for key in keys
    }
    pair_savings = [
        100.0 * (1.0 - dynamics[index]["fleet_j"] / controls[index]["fleet_j"])
        for index in range(2)
    ]
    validity_gates = {
        "complete_server_and_phone_boundary": True,
        "equal_exact_request_outputs": exact_equal_work,
        "same_verified_transition_bytes": True,
        "source_preparation_inside_boundary": True,
        "zero_cgroup_and_process_swap": True,
        "zero_oom": True,
    }
    outcome_gates = {
        "each_pair_exceeds_safety_margin": all(
            saving > SAFETY_MARGIN_PCT for saving in pair_savings
        ),
        "mean_exceeds_safety_margin": (
            -changes["fleet_j_change_pct"] > SAFETY_MARGIN_PCT
        ),
        "mean_makespan_not_regressed": changes["duration_s_change_pct"] <= 0,
    }
    authorized = all(validity_gates.values()) and all(outcome_gates.values())
    output: dict[str, Any] = {
        "admission": (
            "ENERGY_SCREEN_PASS_FALLBACK_AND_RESTORE_RECEIPTS_PENDING"
            if authorized
            else "ENERGY_SCREEN_FAIL_FULL_TRACE_BLOCKED"
        ),
        "changes": changes,
        "control_mean": control_mean,
        "dynamic_mean": dynamic_mean,
        "full_trace_authorized": False,
        "next_gate": (
            "MEASURE_FALLBACK_ACQUIRE_AND_SOURCE_EPOCH_RESTORE"
            if authorized
            else "DO_NOT_RUN_FULL_TRACE_DIAGNOSE_ENERGY_OR_LATENCY"
        ),
        "outcome_gates": outcome_gates,
        "pair_fleet_energy_saving_pct": pair_savings,
        "run_order": [
            "control_r1", "dynamic_r1", "dynamic_r2", "control_r2"
        ],
        "runs": runs,
        "safety_margin_pct": SAFETY_MARGIN_PCT,
        "schema": "s42-adoption-energy-screen-abba-v1",
        "status": "PASS" if all(validity_gates.values()) else "INVALID",
        "validity_gates": validity_gates,
        "workload": {
            "gemma_input_tokens": GEMMA_INPUT_TOKENS,
            "gemma_output_tokens": GEMMA_OUTPUT_TOKENS,
            "qwen_input_tokens": QWEN_INPUT_TOKENS,
            "qwen_output_tokens": QWEN_OUTPUT_TOKENS,
            "requests": 7,
            "source_trace_sha256": TRACE_SHA256,
        },
    }
    output["record_sha256"] = hashlib.sha256(canonical(output)).hexdigest()
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-r1", type=Path, required=True)
    parser.add_argument("--dynamic-r1", type=Path, required=True)
    parser.add_argument("--dynamic-r2", type=Path, required=True)
    parser.add_argument("--control-r2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.exists():
        parser.error("output must be an unused absolute path")
    roots = {
        "control_r1": args.control_r1,
        "dynamic_r1": args.dynamic_r1,
        "dynamic_r2": args.dynamic_r2,
        "control_r2": args.control_r2,
    }
    try:
        value = analyze(roots)
        args.output.write_bytes(canonical(value))
    except (OSError, AnalysisError) as exc:
        parser.exit(2, f"adoption energy screen analysis failed: {exc}\n")
    print(json.dumps({
        "admission": value["admission"],
        "fleet_energy_change_pct": value["changes"][
            "fleet_j_change_pct"
        ],
        "full_trace_authorized": value["full_trace_authorized"],
        "makespan_change_pct": value["changes"]["duration_s_change_pct"],
        "status": value["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
