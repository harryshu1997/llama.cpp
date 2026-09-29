#!/usr/bin/env python3
"""Compare two physical runs of the immutable three-model trace."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


RESULT_SCHEMA = "s42-three-model-physical-result-v1"
PHONE_SCHEMA = "s41-phone-energy-v3"
COMPARISON_SCHEMA = "s42-three-model-physical-comparison-v1"


class ComparisonError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ComparisonError(message)


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


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    require(type(value) is dict, f"object expected: {path}")
    return value


def validate_pair(
    result_path: Path, phone_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = load(result_path)
    phone = load(phone_path)
    require(
        result.get("schema") == RESULT_SCHEMA
        and result.get("status") == "PASS",
        "physical result status",
    )
    require(
        phone.get("schema") == PHONE_SCHEMA
        and phone.get("status") == "PASS"
        and phone.get("boundary") == "paid_trace_interval",
        "phone energy status",
    )
    require(
        abs(
            phone["duration_s"] - result["metrics"]["duration_s"]
        ) < 1e-6,
        "phone and server paid intervals differ",
    )
    require(
        result["metrics"]["completed"] == 84
        and result["metrics"]["output_tokens"] == 13132,
        "request or token conservation",
    )
    require(
        all(
            value == 0
            for value in result["resources"][
                "process_swap_max_bytes"
            ].values()
        ),
        "inference process swap",
    )
    return result, phone


def summary(result: dict[str, Any], phone: dict[str, Any]) -> dict[str, Any]:
    server_j = result["server_energy"]["server_compute_device_energy_j"]
    phone_j = phone["whole_phone_energy_j"]
    return {
        "cpu_package_energy_j": result["server_energy"][
            "cpu_package_energy_j"
        ],
        "duration_s": result["metrics"]["duration_s"],
        "fleet_compute_energy_j": server_j + phone_j,
        "gpu_board_energy_j": result["server_energy"][
            "gpu_board_energy_j"
        ],
        "gpu_utilization_mean_pct": result["resources"][
            "gpu_utilization_pct"
        ]["mean"],
        "output_throughput_tokens_s": result["metrics"][
            "output_throughput_tokens_s"
        ],
        "phone_energy_j": phone_j,
        "server_compute_energy_j": server_j,
        "slo_met": result["metrics"]["slo_met"],
        "small_model": result["metrics"]["by_model"][
            "llama-3.2-1b-instruct-q4_0"
        ],
    }


def compare(
    baseline_result_path: Path,
    baseline_phone_path: Path,
    scheduled_result_path: Path,
    scheduled_phone_path: Path,
) -> dict[str, Any]:
    baseline_result, baseline_phone = validate_pair(
        baseline_result_path, baseline_phone_path
    )
    scheduled_result, scheduled_phone = validate_pair(
        scheduled_result_path, scheduled_phone_path
    )
    require(
        baseline_result["trace_sha256"]
        == scheduled_result["trace_sha256"],
        "trace identity differs",
    )
    require(
        baseline_result["model_identities"]
        == scheduled_result["model_identities"],
        "model identities differ",
    )
    baseline = summary(baseline_result, baseline_phone)
    scheduled = summary(scheduled_result, scheduled_phone)
    energy_saving_j = (
        baseline["fleet_compute_energy_j"]
        - scheduled["fleet_compute_energy_j"]
    )
    duration_saving_s = baseline["duration_s"] - scheduled["duration_s"]
    return {
        "baseline": baseline,
        "delta": {
            "duration_saving_pct": (
                100 * duration_saving_s / baseline["duration_s"]
            ),
            "duration_saving_s": duration_saving_s,
            "fleet_compute_energy_saving_j": energy_saving_j,
            "fleet_compute_energy_saving_pct": (
                100
                * energy_saving_j
                / baseline["fleet_compute_energy_j"]
            ),
            "slo_gain": scheduled["slo_met"] - baseline["slo_met"],
            "throughput_gain_pct": 100 * (
                scheduled["output_throughput_tokens_s"]
                / baseline["output_throughput_tokens_s"]
                - 1
            ),
        },
        "input_sha256": {
            "baseline_phone": digest(baseline_phone_path),
            "baseline_result": digest(baseline_result_path),
            "scheduled_phone": digest(scheduled_phone_path),
            "scheduled_result": digest(scheduled_result_path),
        },
        "outcome": {
            "duration_improved": duration_saving_s > 0,
            "fleet_compute_energy_improved": energy_saving_j > 0,
            "slo_not_regressed": (
                scheduled["slo_met"] >= baseline["slo_met"]
            ),
        },
        "scheduled": scheduled,
        "schema": COMPARISON_SCHEMA,
        "status": "PASS",
        "trace_sha256": baseline_result["trace_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument("--baseline-phone", type=Path, required=True)
    parser.add_argument("--scheduled-result", type=Path, required=True)
    parser.add_argument("--scheduled-phone", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(
        args.output.is_absolute() and not args.output.exists(),
        "output must be a new absolute path",
    )
    result = compare(
        args.baseline_result,
        args.baseline_phone,
        args.scheduled_result,
        args.scheduled_phone,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical(result))
    print(json.dumps({
        "delta": result["delta"],
        "output": str(args.output),
        "status": "PASS",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
