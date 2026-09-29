#!/usr/bin/env python3
"""Validate the Qwen/Gemma dual-residency service-only A-B-B-A screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable


HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from dynamic_residency_v1.analyze_dual_residency_pair import (  # noqa: E402
    AnalysisError,
    EXPECTED_INPUTS,
    canonical,
    change_pct,
    read_object,
    token_quality,
    validate_arm,
)


DEFAULT_ROOTS = {
    "control_r1": (
        HERE / "results/RTX4060TI_QWEN18_GEMMA0_ENERGY_DIAGNOSTIC_V1"
    ),
    "treatment_r1": (
        HERE / "results/RTX4060TI_QWEN15_GEMMA1_ENERGY_DIAGNOSTIC_V1"
    ),
    "treatment_r2": (
        HERE / "results/RTX4060TI_QWEN15_GEMMA1_ENERGY_DIAGNOSTIC_V2"
    ),
    "control_r2": (
        HERE / "results/RTX4060TI_QWEN18_GEMMA0_ENERGY_DIAGNOSTIC_V2"
    ),
}
RUN_ORDER = ("control_r1", "treatment_r1", "treatment_r2", "control_r2")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def run_start_ns(root: Path) -> int:
    result = read_object(root / "RESULT.json")
    requests = result.get("service_smoke", {}).get("requests", {})
    starts = [
        row.get("started_monotonic_ns")
        for row in requests.values()
        if type(row) is dict
    ]
    require(
        len(starts) == 2 and all(type(value) is int for value in starts),
        f"service start receipts: {root}",
    )
    return min(starts)


def mean(
    rows: list[dict[str, Any]],
    accessor: Callable[[dict[str, Any]], float],
) -> float:
    return sum(accessor(row) for row in rows) / len(rows)


def metric_mean(rows: list[dict[str, Any]], *path: str) -> float:
    def access(row: dict[str, Any]) -> float:
        value: object = row
        for key in path:
            require(type(value) is dict and key in value, "mean metric path")
            value = value[key]
        require(type(value) in (int, float), "mean numeric metric")
        return float(value)

    return mean(rows, access)


def compact_run(row: dict[str, Any]) -> dict[str, Any]:
    requests = {}
    for model, request in row["requests"].items():
        requests[model] = {
            key: value
            for key, value in request.items()
            if key != "tokens"
        }
    return {
        "artifact_sha256": row["artifact_sha256"],
        "configuration": row["configuration"],
        "energy": row["energy"],
        "gpu": row["gpu"],
        "requests": requests,
        "service": row["service"],
        "stages": row["stages"],
    }


def analyze(roots: dict[str, Path]) -> dict[str, Any]:
    require(set(roots) == set(RUN_ORDER), "A-B-B-A roots")
    runs = {
        "control_r1": validate_arm(roots["control_r1"], "control"),
        "treatment_r1": validate_arm(
            roots["treatment_r1"], "treatment"
        ),
        "treatment_r2": validate_arm(
            roots["treatment_r2"], "treatment"
        ),
        "control_r2": validate_arm(roots["control_r2"], "control"),
    }
    starts = [run_start_ns(roots[name]) for name in RUN_ORDER]
    require(starts == sorted(starts) and len(set(starts)) == 4,
            "physical A-B-B-A order")
    input_epochs = [runs[name]["inputs"] for name in RUN_ORDER]
    require(
        all(value == input_epochs[0] for value in input_epochs[1:]),
        "matched input epoch",
    )
    for model in ("qwen", "gemma"):
        prompts = [
            runs[name]["requests"][model]["prompt_sha256"]
            for name in RUN_ORDER
        ]
        require(len(set(prompts)) == 1, f"matched prompt: {model}")
        for arm_names in (
            ("control_r1", "control_r2"),
            ("treatment_r1", "treatment_r2"),
        ):
            hashes = [
                runs[name]["requests"][model]["tokens_sha256"]
                for name in arm_names
            ]
            require(hashes[0] == hashes[1],
                    f"repeat token stability: {model}")
    pair_quality = []
    for control_name, treatment_name in (
        ("control_r1", "treatment_r1"),
        ("control_r2", "treatment_r2"),
    ):
        pair_quality.append({
            model: token_quality(
                runs[control_name]["requests"][model]["tokens"],
                runs[treatment_name]["requests"][model]["tokens"],
            )
            for model in ("qwen", "gemma")
        })
    require(pair_quality[0] == pair_quality[1],
            "repeat output-quality stability")

    controls = [runs["control_r1"], runs["control_r2"]]
    treatments = [runs["treatment_r1"], runs["treatment_r2"]]
    metric_paths = {
        "cpu_package_energy": ("energy", "cpu_package_j"),
        "gemma_first_token": ("requests", "gemma", "first_token_s"),
        "gemma_service": ("requests", "gemma", "service_s"),
        "gpu_board_energy": ("energy", "gpu_board_j"),
        "qwen_first_token": ("requests", "qwen", "first_token_s"),
        "qwen_service": ("requests", "qwen", "service_s"),
        "server_energy": ("energy", "server_j"),
        "wall_service": ("service", "wall_s"),
    }
    control_mean = {
        name: metric_mean(controls, *path)
        for name, path in metric_paths.items()
    }
    treatment_mean = {
        name: metric_mean(treatments, *path)
        for name, path in metric_paths.items()
    }
    changes = {
        name + "_pct": change_pct(control_mean[name], treatment_mean[name])
        for name in metric_paths
    }
    pair_server_savings = [
        100.0 * (
            1.0
            - treatments[index]["energy"]["server_j"]
            / controls[index]["energy"]["server_j"]
        )
        for index in range(2)
    ]
    pair_wall_savings = [
        100.0 * (
            1.0
            - treatments[index]["service"]["wall_s"]
            / controls[index]["service"]["wall_s"]
        )
        for index in range(2)
    ]
    screen_gates = {
        "each_pair_server_energy_lower": all(
            value > 0 for value in pair_server_savings
        ),
        "each_pair_wall_service_lower": all(
            value > 0 for value in pair_wall_savings
        ),
        "mean_server_energy_lower": changes["server_energy_pct"] < 0,
        "mean_wall_service_lower": changes["wall_service_pct"] < 0,
        "qwen_first_token_not_regressed": (
            changes["qwen_first_token_pct"] <= 0
        ),
    }
    direction_pass = all(
        screen_gates[key]
        for key in (
            "each_pair_server_energy_lower",
            "each_pair_wall_service_lower",
            "mean_server_energy_lower",
            "mean_wall_service_lower",
        )
    )
    output: dict[str, Any] = {
        "admission": {
            "decision": "COLLECT_TRANSITION_PHONE_AND_FENCE_RECEIPTS",
            "dynamic_energy_claim": None,
            "eligible": False,
            "missing": [
                "FULL_TRACE_EQUAL_WORK",
                "LOAD_TRANSITION_AND_RESTORE_ENERGY",
                "OP15_SYNCHRONIZED_ENERGY",
                "PROTECTED_GPU_FENCE_RECEIPTS",
                "ATOMIC_TENSOR_SLICE_MANIFEST",
                "OUTPUT_QUALITY_ADMISSION_RULE",
            ],
        },
        "boundary": "cpu-package+gpu-board-concurrent-two-request-service-v1",
        "changes": changes,
        "claim_gates": {
            "atomic_tensor_slice_manifest": False,
            "full_trace_equal_work": False,
            "load_transition_restore_energy_included": False,
            "op15_energy_included": False,
            "output_quality_admitted": False,
            "protected_gpu_fence_receipts": False,
            "service_abba_repeated": True,
        },
        "control_mean_of_two": control_mean,
        "pair_server_energy_saving_pct": pair_server_savings,
        "pair_wall_service_saving_pct": pair_wall_savings,
        "quality": pair_quality[0],
        "run_order": list(RUN_ORDER),
        "runs": {name: compact_run(runs[name]) for name in RUN_ORDER},
        "schema": "s42-dual-residency-service-abba-v1",
        "screen_gates": screen_gates,
        "screen_outcome": (
            "ENERGY_AND_WALL_DIRECTION_REPEATED_QWEN_TTFT_REGRESSED"
            if direction_pass
            and not screen_gates["qwen_first_token_not_regressed"]
            else "DIRECTION_PASS" if direction_pass else "DIRECTION_FAIL"
        ),
        "status": (
            "REPEATED_SERVICE_DIRECTION_PASS_NO_ADMISSION"
            if direction_pass else "REPEATED_SERVICE_DIRECTION_FAIL"
        ),
        "treatment_mean_of_two": treatment_mean,
        "validity_gates": {
            "all_artifact_manifests_verified": True,
            "all_cleanup_passed": True,
            "all_gpu_reserves_preserved": True,
            "all_zero_observed_process_swap": True,
            "physical_abba_order": True,
            "same_energy_boundary": True,
            "same_input_epoch_and_work": True,
            "stable_tokens_within_each_arm": True,
        },
        "workload_per_run": {
            "input_tokens": 287,
            "models": 2,
            "output_tokens": 50,
            "requests": 2,
            "trace_source_sha256": EXPECTED_INPUTS["trace"]["sha256"],
        },
    }
    output["record_sha256"] = hashlib.sha256(canonical(output)).hexdigest()
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--control-r1", type=Path, default=DEFAULT_ROOTS["control_r1"]
    )
    parser.add_argument(
        "--treatment-r1", type=Path, default=DEFAULT_ROOTS["treatment_r1"]
    )
    parser.add_argument(
        "--treatment-r2", type=Path, default=DEFAULT_ROOTS["treatment_r2"]
    )
    parser.add_argument(
        "--control-r2", type=Path, default=DEFAULT_ROOTS["control_r2"]
    )
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
        parser.exit(2, f"dual-residency A-B-B-A analysis failed: {exc}\n")
    print(json.dumps({
        "decision": value["admission"]["decision"],
        "record_sha256": value["record_sha256"],
        "server_energy_change_pct": value["changes"][
            "server_energy_pct"
        ],
        "status": value["status"],
        "wall_service_change_pct": value["changes"][
            "wall_service_pct"
        ],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
