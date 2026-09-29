#!/usr/bin/env python3
"""Attach clock-aligned OP15 energy to every paid model-energy case."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
S41_SUPPORT = (
    HERE.parents[1]
    / "s41_gemma_qwen_continuous_baseline"
    / "tp_operator_split_v1"
    / "burstgpt_gpu_cpu_op15_v1"
)
if S41_SUPPORT.is_dir() and str(S41_SUPPORT) not in sys.path:
    sys.path.insert(0, str(S41_SUPPORT))

import analyze_phone_energy  # noqa: E402
import run_trace  # noqa: E402


def load_result(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise run_trace.RunError(f"cannot read result: {error}") from error
    run_trace.require(
        type(value) is dict and
        value.get("schema") == "s42-model-energy-result-v1" and
        value.get("status") == "PASS" and
        type(value.get("cases")) is list and value["cases"],
        "model-energy result",
    )
    return value


def case_energy(
    rows: list[dict[str, float]], case: dict[str, Any]
) -> dict[str, Any]:
    spec = case.get("case")
    start_ns = case.get("paid_start_ns")
    end_ns = case.get("paid_end_ns")
    run_trace.require(
        type(spec) is dict and type(spec.get("case_id")) is str and
        type(start_ns) is int and type(end_ns) is int and end_ns > start_ns,
        "case interval",
    )
    usb_j = analyze_phone_energy.integrate(rows, "usb_input_w", start_ns, end_ns)
    battery_j = analyze_phone_energy.integrate(
        rows, "battery_discharge_w", start_ns, end_ns
    )
    duration_s = (end_ns - start_ns) / 1e9
    total_j = usb_j + battery_j
    run_trace.require(
        math.isfinite(total_j) and total_j > 0,
        "phone case energy",
    )
    return {
        "battery_discharge_energy_j": battery_j,
        "case_id": spec["case_id"],
        "duration_s": duration_s,
        "paid_end_ns": end_ns,
        "paid_start_ns": start_ns,
        "usb_input_energy_j": usb_j,
        "whole_phone_average_power_w": total_j / duration_s,
        "whole_phone_energy_j": total_j,
    }


def build_receipt(
    *,
    result_path: Path,
    samples_path: Path,
    clock_before_path: Path,
    clock_after_path: Path,
) -> dict[str, Any]:
    result = load_result(result_path)
    before = analyze_phone_energy.read_anchor(clock_before_path)
    after = analyze_phone_energy.read_anchor(clock_after_path)
    raw_rows = analyze_phone_energy.read_samples(samples_path)
    rows, slope = analyze_phone_energy.map_samples(raw_rows, before, after)
    start_ns = min(case["paid_start_ns"] for case in result["cases"])
    end_ns = max(case["paid_end_ns"] for case in result["cases"])
    run_trace.require(
        rows[0]["host_ns"] <= start_ns < end_ns <= rows[-1]["host_ns"],
        "phone sample coverage",
    )
    battery_current_mean_ma = sum(
        row["battery_current_ma"] for row in raw_rows
    ) / len(raw_rows)
    charge_counter_delta_uah = int(
        raw_rows[-1]["battery_charge_counter_uah"] -
        raw_rows[0]["battery_charge_counter_uah"]
    )
    if (
        abs(charge_counter_delta_uah) >= 1000 and
        abs(battery_current_mean_ma) >= 1
    ):
        run_trace.require(
            charge_counter_delta_uah * battery_current_mean_ma <= 0,
            "OPLUS battery current sign",
        )
    cases = [case_energy(rows, case) for case in result["cases"]]
    return {
        "battery_current_mean_ma": battery_current_mean_ma,
        "battery_current_positive_is_discharge": True,
        "cases": cases,
        "charge_counter_delta_uah": charge_counter_delta_uah,
        "clock_slope_host_per_phone": slope,
        "input_sha256": {
            "clock_after": run_trace.digest_file(clock_after_path),
            "clock_before": run_trace.digest_file(clock_before_path),
            "phone_samples": run_trace.digest_file(samples_path),
        },
        "method": "per-case trapezoidal USB input plus battery discharge",
        "result_sha256": run_trace.digest_file(result_path),
        "schema": "s42-model-energy-phone-v1",
        "serial": analyze_phone_energy.SERIAL,
        "status": "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--clock-before", type=Path, required=True)
    parser.add_argument("--clock-after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_trace.require(
        args.output.is_absolute() and not args.output.exists(), "output path"
    )
    run_trace.write_json(args.output, build_receipt(
        result_path=args.result,
        samples_path=args.samples,
        clock_before_path=args.clock_before,
        clock_after_path=args.clock_after,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

