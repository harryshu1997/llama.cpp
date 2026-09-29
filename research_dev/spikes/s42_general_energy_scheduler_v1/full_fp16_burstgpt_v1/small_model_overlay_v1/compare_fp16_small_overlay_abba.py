#!/usr/bin/env python3
"""Aggregate two matched static/runtime pairs in A-B-B-A order."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import compare_fp16_small_overlay as pair


SCHEMA = "s42-full-fp16-llama1b-scheduler-abba-v1"
T95_DF1 = 12.706204736


def require(condition: bool, message: str) -> None:
    if not condition:
        raise pair.ComparisonError(message)


def mean_ci95(values: list[float]) -> dict[str, float | int]:
    require(len(values) == 2, "two paired values required")
    mean = sum(values) / 2
    sample_sd = abs(values[0] - values[1]) / math.sqrt(2)
    half_width = T95_DF1 * sample_sd / math.sqrt(2)
    return {
        "ci95_high": mean + half_width,
        "ci95_low": mean - half_width,
        "mean": mean,
        "samples": 2,
    }


def aggregate(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    require(
        len(comparisons) == 2
        and all(value.get("status") == "PASS" for value in comparisons),
        "matched comparison results",
    )
    first, second = comparisons
    require(
        first["large_model_policy"] == second["large_model_policy"]
        and first["work_identity"] == second["work_identity"]
        and first["work_receipts"] == second["work_receipts"],
        "ABBA work or policy identity",
    )
    fleet_savings = [
        value["delta"]["fleet_energy_saving_pct"]
        for value in comparisons
    ]
    duration_savings = [
        value["delta"]["duration_saving_pct"]
        for value in comparisons
    ]
    fleet_ci = mean_ci95(fleet_savings)
    duration_ci = mean_ci95(duration_savings)
    return {
        "claim_scope": (
            "small-model scheduler at fixed large-model policy"
        ),
        "confidence_intervals": {
            "duration_saving_pct": duration_ci,
            "fleet_energy_saving_pct": fleet_ci,
        },
        "energy_claim_eligible": fleet_ci["ci95_low"] > 0,
        "large_model_policy": first["large_model_policy"],
        "pair_results": comparisons,
        "run_order": [
            "static-r1",
            "runtime-r1",
            "runtime-r2",
            "static-r2",
        ],
        "schema": SCHEMA,
        "status": "PASS",
        "work_identity": first["work_identity"],
        "work_receipts": first["work_receipts"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ("static-r1", "runtime-r1", "runtime-r2", "static-r2"):
        parser.add_argument(f"--{arm}-result", type=Path, required=True)
        parser.add_argument(f"--{arm}-phone", type=Path, required=True)
        parser.add_argument(
            f"--{arm}-qualification", type=Path, required=True
        )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(
        args.output.is_absolute() and not args.output.exists(),
        "new absolute output path",
    )
    runs = {
        name: {
            kind: getattr(args, f"{name.replace('-', '_')}_{kind}")
            for kind in ("result", "phone", "qualification")
        }
        for name in ("static-r1", "runtime-r1", "runtime-r2", "static-r2")
    }
    comparisons = []
    for static_name, runtime_name in (
        ("static-r1", "runtime-r1"),
        ("static-r2", "runtime-r2"),
    ):
        static = runs[static_name]
        runtime = runs[runtime_name]
        comparisons.append(pair.compare(
            static["result"],
            static["phone"],
            runtime["result"],
            runtime["phone"],
            static["qualification"],
            runtime["qualification"],
        ))
    value = aggregate(comparisons)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(pair.canonical(value))
    print(json.dumps({
        "energy_claim_eligible": value["energy_claim_eligible"],
        "fleet_energy_saving_pct": value["confidence_intervals"]
            ["fleet_energy_saving_pct"],
        "output": str(args.output),
        "status": "PASS",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
