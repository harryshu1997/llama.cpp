#!/usr/bin/env python3
"""Combine the two fixed-policy ABBA results into a 2x2 report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import compare_fp16_small_overlay_abba as abba


SCHEMA = "s42-full-fp16-llama1b-scheduler-2x2-abba-v1"
ABBA_SCHEMA = "s42-full-fp16-llama1b-scheduler-abba-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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


def load(path: Path, expected_policy: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    require(
        type(value) is dict
        and value.get("schema") == ABBA_SCHEMA
        and value.get("status") == "PASS"
        and value.get("large_model_policy") == expected_policy
        and len(value.get("pair_results", [])) == 2,
        f"ABBA identity: {path}",
    )
    return value


def cell_mean(value: dict[str, Any], field: str) -> dict[str, object]:
    rows = [pair[field] for pair in value["pair_results"]]
    numeric = (
        "cpu_package_energy_j",
        "duration_s",
        "fleet_energy_j",
        "gpu_board_energy_j",
        "phone_energy_j",
        "server_energy_j",
        "slo_met",
        "throughput_tokens_s",
    )
    return {
        name: sum(float(row[name]) for row in rows) / len(rows)
        for name in numeric
    } | {
        "route_counts": [row["route_counts"] for row in rows],
        "samples": len(rows),
    }


def interaction(
    cpu: dict[str, Any],
    op15: dict[str, Any],
    metric: str,
) -> dict[str, float | int]:
    differences = [
        cpu_pair["delta"][metric] - op15_pair["delta"][metric]
        for cpu_pair, op15_pair in zip(
            cpu["pair_results"], op15["pair_results"]
        )
    ]
    return abba.mean_ci95(differences)


def combine(
    cpu: dict[str, Any],
    op15: dict[str, Any],
    cpu_path: Path,
    op15_path: Path,
) -> dict[str, Any]:
    require(
        cpu["work_identity"] == op15["work_identity"]
        and cpu["work_receipts"] == op15["work_receipts"],
        "2x2 work identity",
    )
    energy_by_policy = {
        "cpu-overflow": cpu["confidence_intervals"]
            ["fleet_energy_saving_pct"],
        "op15-assistance": op15["confidence_intervals"]
            ["fleet_energy_saving_pct"],
    }
    energy_eligible = {
        policy: interval["ci95_low"] > 0
        for policy, interval in energy_by_policy.items()
    }
    energy_status = "PASS" if all(energy_eligible.values()) else "FAIL"
    return {
        "cells": {
            "cpu-overflow+runtime-scheduler": cell_mean(
                cpu, "runtime_scheduler"
            ),
            "cpu-overflow+static-cpu": cell_mean(cpu, "static_cpu"),
            "op15-assistance+runtime-scheduler": cell_mean(
                op15, "runtime_scheduler"
            ),
            "op15-assistance+static-cpu": cell_mean(
                op15, "static_cpu"
            ),
        },
        "claim_scope": (
            "small-model runtime scheduler effect at each fixed "
            "large-model policy"
        ),
        "input_sha256": {
            "cpu_overflow_abba": digest(cpu_path),
            "op15_assistance_abba": digest(op15_path),
        },
        "interaction": {
            "duration_saving_pct_cpu_minus_op15": interaction(
                cpu, op15, "duration_saving_pct"
            ),
            "fleet_energy_saving_pct_cpu_minus_op15": interaction(
                cpu, op15, "fleet_energy_saving_pct"
            ),
        },
        "run_order": [
            "cpu-overflow+static-cpu-r1",
            "cpu-overflow+runtime-scheduler-r1",
            "op15-assistance+static-cpu-r1",
            "op15-assistance+runtime-scheduler-r1",
            "op15-assistance+runtime-scheduler-r2",
            "op15-assistance+static-cpu-r2",
            "cpu-overflow+runtime-scheduler-r2",
            "cpu-overflow+static-cpu-r2",
        ],
        "scheduler_effect": {
            "cpu-overflow": cpu["confidence_intervals"],
            "op15-assistance": op15["confidence_intervals"],
        },
        "energy_qualification_status": energy_status,
        "scheduler_energy_claim_eligible": energy_eligible,
        "schema": SCHEMA,
        "status": energy_status,
        "work_identity": cpu["work_identity"],
        "work_receipts": cpu["work_receipts"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-overflow-abba", type=Path, required=True)
    parser.add_argument("--op15-assistance-abba", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(
        args.output.is_absolute() and not args.output.exists(),
        "new absolute output path",
    )
    cpu = load(args.cpu_overflow_abba, "cpu-overflow")
    op15 = load(args.op15_assistance_abba, "op15-assistance")
    value = combine(
        cpu,
        op15,
        args.cpu_overflow_abba,
        args.op15_assistance_abba,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical(value))
    print(json.dumps({
        "output": str(args.output),
        "scheduler_energy_claim_eligible": value[
            "scheduler_energy_claim_eligible"
        ],
        "energy_qualification_status": value[
            "energy_qualification_status"
        ],
        "status": value["status"],
    }, sort_keys=True))
    return 0 if value["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
