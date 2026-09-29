#!/usr/bin/env python3
"""Validate and compare fixed and shape-balanced Gemma split policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import load_execution_plan  # noqa: E402
from research_dev.spikes.s42_general_energy_scheduler_v1.hybrid_overflow_v1.analyze_gpu_overflow_pair import (  # noqa: E402
    validate_phone_work,
)
from research_dev.spikes.s42_general_energy_scheduler_v1.full_fp16_burstgpt_v1.shape_balance_v1.policy_adapter import (  # noqa: E402
    QUALIFIED_TABLE,
    materialize_gemma_policy,
)


MODEL_SHA256 = (
    "ed76f2183d2d1d65091986033023e6c7"
    "8d27f6276c1b0c5826cc92acf73538cf"
)
TRACE_SHA256 = (
    "b20a9ba66ee3558d835a0e19ed3cfa4"
    "c31a4a9e8b4f9c085b29a14f80250a0ff"
)


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
    value = json.loads(path.read_text(encoding="ascii"))
    require(type(value) is dict, f"object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def positive(name: str, value: object) -> float:
    require(
        type(value) in (int, float)
        and math.isfinite(value)
        and value > 0,
        f"positive {name}",
    )
    return float(value)


def validate_run(root: Path, variant: str, repeat_index: int) -> dict[str, Any]:
    result_path = root / "RESULT.json"
    archived_capture = root / "PHONE_ENERGY_V3.json"
    capture = (
        root
        if archived_capture.is_file()
        else Path(str(root) + ".phone-capture")
    )
    phone_path = capture / "PHONE_ENERGY_V3.json"
    plan_path = capture / "EXECUTION_PLAN.json"
    result = read_object(result_path)
    phone = read_object(phone_path)
    plan = load_execution_plan(plan_path)
    expected_policy, balance, calibration_hash = materialize_gemma_policy(
        variant
    )
    require(
        result.get("schema") == "s41-gemma-gpu-trace-v2"
        and result.get("status") == "PASS"
        and result.get("mode") == "cuda-cpu-op15"
        and result.get("repeat_index") == repeat_index,
        f"{variant} result identity",
    )
    require(
        phone.get("schema") == "s41-phone-energy-v3"
        and phone.get("status") == "PASS"
        and phone.get("boundary") == "paid_trace_interval",
        f"{variant} phone energy identity",
    )
    require(
        plan.execution_mode == "cuda-cpu-op15"
        and plan.offload is not None
        and plan.offload.split == expected_policy
        and result.get("scheduler_plan_sha256") == plan.plan_sha256
        and result.get("preflight", {}).get("scheduler", {}).get("plan_sha256")
            == plan.plan_sha256,
        f"{variant} unified scheduler binding",
    )
    require(
        plan.trace_sha256 == "sha256:" + TRACE_SHA256
        and plan.model_hashes == {"cold": "sha256:" + MODEL_SHA256}
        and result.get("model", {}).get("sha256") == MODEL_SHA256,
        f"{variant} workload epoch",
    )
    placement = plan.layer_placement
    require(
        placement is not None
        and placement.cpu_layer_spec == "0-22"
        and placement.gpu_layer_spec == "23-47"
        and placement.selected.runtime_gpu_layers == 25
        and int(result.get("n_gpu_layers")) == 25,
        f"{variant} placement",
    )
    rows = result.get("request_results")
    metrics = result.get("metrics")
    require(
        type(rows) is list
        and len(rows) == 17
        and type(metrics) is dict
        and metrics.get("completed") == 17
        and metrics.get("output_tokens") == 6_919
        and sum(row.get("input_tokens", 0) for row in rows) == 11_476
        and sum(row.get("output_tokens", 0) for row in rows) == 6_919
        and all(
            type(row.get("tokens")) is list
            and len(row["tokens"]) == row.get("output_tokens")
            for row in rows
        ),
        f"{variant} completed work",
    )
    validate_phone_work(plan, result.get("phone"))
    resources = result.get("resources")
    require(
        type(resources) is dict
        and resources.get("gpu_memory_total_bytes") == 17_175_674_880
        and resources.get("process_swap_max_bytes") == 0,
        f"{variant} resource gate",
    )
    server = result.get("server_energy")
    require(
        type(server) is dict
        and server.get("boundary") == "paid_trace_interval",
        f"{variant} server energy",
    )
    cpu_j = positive("CPU package energy", server.get("cpu_package_energy_j"))
    gpu_j = positive("GPU board energy", server.get("gpu_board_energy_j"))
    server_j = positive(
        "server energy", server.get("server_compute_device_energy_j")
    )
    require(
        math.isclose(server_j, cpu_j + gpu_j, rel_tol=1e-6),
        f"{variant} server energy sum",
    )
    duration_s = (result["paid_end_ns"] - result["paid_start_ns"]) / 1e9
    require(
        math.isclose(duration_s, metrics["makespan_s"], abs_tol=0.001)
        and math.isclose(duration_s, phone.get("duration_s"), abs_tol=0.001),
        f"{variant} paid interval",
    )
    phone_j = positive("whole-phone energy", phone.get("whole_phone_energy_j"))
    token_hash = hashlib.sha256(canonical({
        str(row["request_index"]): row["tokens"] for row in rows
    })).hexdigest()
    return {
        "artifacts": {
            "phone_energy_sha256": sha256(phone_path),
            "plan_file_sha256": sha256(plan_path),
            "result_sha256": sha256(result_path),
        },
        "calibration_sha256": calibration_hash,
        "cpu_package_j": cpu_j,
        "duration_s": duration_s,
        "fleet_j": server_j + phone_j,
        "gpu_board_j": gpu_j,
        "phone_calls": result["phone"]["bridge"]["calls"],
        "phone_j": phone_j,
        "policy": plan.offload.split.table,
        "predicted_saving_ppm": (
            None if balance is None else balance.predicted_saving_ppm
        ),
        "repeat_index": repeat_index,
        "server_j": server_j,
        "token_sequences_sha256": token_hash,
        "variant": variant,
    }


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def saving_pct(fixed: float, tuned: float) -> float:
    return 100.0 * (1.0 - tuned / fixed)


def analyze(roots: dict[str, Path]) -> dict[str, Any]:
    runs = {
        "fixed_r1": validate_run(roots["fixed_r1"], "qualified", 1),
        "tuned_r1": validate_run(
            roots["tuned_r1"], "shape-balanced", 1
        ),
        "tuned_r2": validate_run(
            roots["tuned_r2"], "shape-balanced", 2
        ),
        "fixed_r2": validate_run(roots["fixed_r2"], "qualified", 2),
    }
    fixed = [runs["fixed_r1"], runs["fixed_r2"]]
    tuned = [runs["tuned_r1"], runs["tuned_r2"]]
    keys = (
        "duration_s", "cpu_package_j", "gpu_board_j", "server_j",
        "phone_j", "fleet_j",
    )
    fixed_mean = {key: mean(fixed, key) for key in keys}
    tuned_mean = {key: mean(tuned, key) for key in keys}
    savings = {
        key + "_saving_pct": saving_pct(fixed_mean[key], tuned_mean[key])
        for key in keys
    }
    fleet_pairs = [
        saving_pct(fixed[index]["fleet_j"], tuned[index]["fleet_j"])
        for index in range(2)
    ]
    duration_pairs = [
        saving_pct(fixed[index]["duration_s"], tuned[index]["duration_s"])
        for index in range(2)
    ]
    outcome_gates = {
        "each_pair_fleet_energy_saving": all(value > 0 for value in fleet_pairs),
        "each_pair_makespan_not_regressed": all(
            value >= 0 for value in duration_pairs
        ),
        "mean_fleet_energy_saving_at_least_0_5_pct": (
            savings["fleet_j_saving_pct"] >= 0.5
        ),
        "mean_makespan_saving": savings["duration_s_saving_pct"] > 0,
    }
    admitted = all(outcome_gates.values())
    output: dict[str, Any] = {
        "admission": "ADMIT_FULL_TRACE" if admitted else "RETAIN_FIXED_POLICY",
        "boundary": "cpu-package+gpu-board+whole-phone-paid-gemma17-v1",
        "fixed_mean": fixed_mean,
        "outcome_gates": outcome_gates,
        "pair_duration_saving_pct": duration_pairs,
        "pair_fleet_energy_saving_pct": fleet_pairs,
        "policies": {
            "fixed": QUALIFIED_TABLE,
            "tuned": runs["tuned_r1"]["policy"],
        },
        "run_order": ["fixed_r1", "tuned_r1", "tuned_r2", "fixed_r2"],
        "runs": runs,
        "savings": savings,
        "schema": "s42-gemma-shape-balance-abba-v1",
        "status": "PASS",
        "trace": {
            "input_tokens": 11_476,
            "output_tokens": 6_919,
            "requests": 17,
            "source_sha256": TRACE_SHA256,
        },
        "tuned_mean": tuned_mean,
        "validity_gates": {
            "equal_completed_work": True,
            "matched_gpu_cpu_phone_placement": True,
            "synchronized_whole_phone_energy": True,
            "unified_scheduler_plans_bound": True,
            "zero_phone_reset_recoveries": True,
            "zero_process_swap": True,
        },
    }
    output["record_sha256"] = hashlib.sha256(canonical(output)).hexdigest()
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixed-r1", "tuned-r1", "tuned-r2", "fixed-r2"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.exists():
        parser.error("output must be an unused absolute path")
    roots = {
        "fixed_r1": args.fixed_r1,
        "tuned_r1": args.tuned_r1,
        "tuned_r2": args.tuned_r2,
        "fixed_r2": args.fixed_r2,
    }
    try:
        value = analyze(roots)
        args.output.write_bytes(canonical(value))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"shape balance ABBA analysis failed: {exc}\n")
    print(json.dumps({
        "admission": value["admission"],
        "fleet_energy_saving_pct": value["savings"]["fleet_j_saving_pct"],
        "makespan_saving_pct": value["savings"]["duration_s_saving_pct"],
        "record_sha256": value["record_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
