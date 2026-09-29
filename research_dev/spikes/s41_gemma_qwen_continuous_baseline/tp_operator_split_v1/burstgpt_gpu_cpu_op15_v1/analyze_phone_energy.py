#!/usr/bin/env python3
"""Integrate OP15 USB input plus battery discharge over a paid trace."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import run_trace


SERIAL = "3C15AU002CL00000"


def read_anchor(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    run_trace.require(
        value.get("schema") == "s41-phone-clock-anchor-v1" and
        value.get("serial") == SERIAL and
        type(value.get("phone_uptime_ns")) is int and
        type(value.get("host_midpoint_ns")) is int,
        "clock anchor",
    )
    return value


def read_samples(
    path: Path, legacy_battery_current_ma: bool = False
) -> list[dict[str, float]]:
    rows = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            battery_current_field = (
                "battery_current_ua"
                if legacy_battery_current_ma
                else "battery_current_ma"
            )
            expected = {
                "uptime_s", "usb_current_ua", "usb_voltage_uv",
                battery_current_field, "battery_voltage_uv",
            }
            if not legacy_battery_current_ma:
                expected.add("battery_charge_counter_uah")
            run_trace.require(
                set(row) == expected,
                "phone sample fields",
            )
            uptime_ns = int(round(float(row["uptime_s"]) * 1e9))
            usb_current_ua = int(row["usb_current_ua"])
            usb_voltage_uv = int(row["usb_voltage_uv"])
            battery_current_ma = int(row[battery_current_field])
            battery_voltage_uv = int(row["battery_voltage_uv"])
            run_trace.require(
                uptime_ns > 0 and usb_current_ua >= 0 and
                usb_voltage_uv > 0 and battery_voltage_uv > 0,
                "phone sample bounds",
            )
            rows.append({
                "battery_discharge_w": (
                    max(0, battery_current_ma) * battery_voltage_uv / 1e9
                ),
                "battery_current_ma": battery_current_ma,
                "battery_charge_counter_uah": (
                    None
                    if legacy_battery_current_ma
                    else int(row["battery_charge_counter_uah"])
                ),
                "phone_uptime_ns": uptime_ns,
                "usb_input_w": usb_current_ua * usb_voltage_uv / 1e12,
            })
    run_trace.require(len(rows) >= 3, "phone sample count")
    run_trace.require(
        all(
            rows[index]["phone_uptime_ns"] > rows[index - 1]["phone_uptime_ns"]
            for index in range(1, len(rows))
        ),
        "phone sample order",
    )
    return rows


def map_samples(
    rows: list[dict[str, float]],
    before: dict[str, Any],
    after: dict[str, Any],
) -> tuple[list[dict[str, float]], float]:
    run_trace.require(before["boot_id"] == after["boot_id"], "phone reboot")
    phone_delta = after["phone_uptime_ns"] - before["phone_uptime_ns"]
    host_delta = after["host_midpoint_ns"] - before["host_midpoint_ns"]
    run_trace.require(phone_delta > 0 and host_delta > 0, "clock direction")
    slope = host_delta / phone_delta
    run_trace.require(abs(slope - 1.0) <= 0.002, "clock drift")
    mapped = []
    for row in rows:
        host_ns = before["host_midpoint_ns"] + int(round(
            (row["phone_uptime_ns"] - before["phone_uptime_ns"]) * slope
        ))
        mapped.append({**row, "host_ns": host_ns})
    return mapped, slope


def integrate(
    rows: list[dict[str, float]], key: str, start_ns: int, end_ns: int
) -> float:
    proxy = [{
        "gpu": {
            "power_mw": row[key] * 1000.0,
            "sample_t_ns": row["host_ns"],
        }
    } for row in rows]
    return run_trace.integrate_power_samples(proxy, start_ns, end_ns)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--clock-before", type=Path, required=True)
    parser.add_argument("--clock-after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-battery-current-ma", action="store_true")
    args = parser.parse_args()
    run_trace.require(args.output.is_absolute() and not args.output.exists(), "output")
    result = json.loads(args.result.read_text())
    run_trace.require(
        result.get("schema") in (
            "s41-burstgpt-llama-server-result-v1",
            "s41-gemma-gpu-trace-v2",
            "s41-hierarchical-burstgpt-result-v1",
            "s42-six-model-physical-result-v1",
            "s42-three-model-physical-result-v1",
            "s42-full-fp16-llama1b-combined-result-v1",
            "s42-full-fp16-llama1b-combined-result-v2",
            "s42-unified-fp16-llama-overlay-result-v1",
        ) and
        result.get("status") == "PASS",
        "trace result",
    )
    start_ns = result["paid_start_ns"]
    end_ns = result["paid_end_ns"]
    before = read_anchor(args.clock_before)
    after = read_anchor(args.clock_after)
    rows, slope = map_samples(
        read_samples(args.samples, args.legacy_battery_current_ma), before, after
    )
    run_trace.require(
        rows[0]["host_ns"] <= start_ns < end_ns <= rows[-1]["host_ns"],
        "phone samples do not cover paid interval",
    )
    usb_j = integrate(rows, "usb_input_w", start_ns, end_ns)
    battery_j = integrate(rows, "battery_discharge_w", start_ns, end_ns)
    duration_s = (end_ns - start_ns) / 1e9
    total_j = usb_j + battery_j
    charge_counter_delta_uah = None
    battery_current_mean_ma = sum(
        row["battery_current_ma"] for row in rows
    ) / len(rows)
    if not args.legacy_battery_current_ma:
        charge_counter_delta_uah = int(
            rows[-1]["battery_charge_counter_uah"] -
            rows[0]["battery_charge_counter_uah"]
        )
        if (
            abs(charge_counter_delta_uah) >= 1000 and
            abs(battery_current_mean_ma) >= 1
        ):
            run_trace.require(
                charge_counter_delta_uah * battery_current_mean_ma <= 0,
                "OPLUS battery current sign",
            )
    intervals = [
        (rows[index]["phone_uptime_ns"] - rows[index - 1]["phone_uptime_ns"]) / 1e9
        for index in range(1, len(rows))
    ]
    output = {
        "battery_discharge_energy_j": battery_j,
        "boundary": "paid_trace_interval",
        "clock_slope_host_per_phone": slope,
        "duration_s": duration_s,
        "input_sha256": {
            "clock_after": run_trace.digest_file(args.clock_after),
            "clock_before": run_trace.digest_file(args.clock_before),
            "phone_samples": run_trace.digest_file(args.samples),
            "trace_result": run_trace.digest_file(args.result),
        },
        "method": "trapezoidal USB input plus simultaneous battery discharge",
        "sample_count_total": len(rows),
        "sample_interval_s": run_trace.stats(intervals),
        "battery_current_source": (
            "OPLUS vendor battery/current_now in mA, positive is discharge; legacy v1 column was mislabeled"
            if args.legacy_battery_current_ma
            else "OPLUS vendor battery/current_now in mA, positive is discharge"
        ),
        "battery_current_positive_is_discharge": True,
        "battery_current_mean_ma": battery_current_mean_ma,
        "charge_counter_delta_uah": charge_counter_delta_uah,
        "battery_discharge_only": True,
        "schema": "s41-phone-energy-v3",
        "serial": SERIAL,
        "status": "PASS",
        "usb_input_energy_j": usb_j,
        "whole_phone_average_power_w": total_j / duration_s,
        "whole_phone_energy_j": total_j,
    }
    run_trace.write_json(args.output, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
