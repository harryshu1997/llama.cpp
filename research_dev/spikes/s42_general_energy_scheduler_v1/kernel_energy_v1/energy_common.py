#!/usr/bin/env python3
"""Shared integration helpers for the S42 kernel-energy campaign."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


class EnergyError(RuntimeError):
    pass


PHONE_MARKER = re.compile(
    r"PHONE_ENERGY_WINDOW_(START|END)\s+"
    r"phone_uptime_(ns|s)=([0-9]+(?:\.[0-9]+)?)"
)
DESKTOP_MARKER = re.compile(
    r"ENERGY_WINDOW_(START|END)\s+unix_ns=([0-9]+)"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EnergyError(message)


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def percentile(values: Iterable[float], fraction: float) -> float:
    rows = sorted(values)
    require(rows, "empty percentile input")
    require(0.0 <= fraction <= 1.0, "invalid percentile")
    index = min(len(rows) - 1, int(math.floor(fraction * len(rows))))
    return rows[index]


def interpolate(points: list[tuple[int, float]], target_ns: int) -> float:
    require(len(points) >= 2, "energy sample count")
    require(points[0][0] <= target_ns <= points[-1][0], "sample coverage")
    for index in range(1, len(points)):
        left_t, left_value = points[index - 1]
        right_t, right_value = points[index]
        require(right_t > left_t, "sample order")
        if target_ns <= right_t:
            fraction = (target_ns - left_t) / (right_t - left_t)
            return left_value + fraction * (right_value - left_value)
    raise EnergyError("interpolation coverage")


def integrate_power(
    points: list[tuple[int, float]], start_ns: int, end_ns: int
) -> float:
    require(end_ns > start_ns, "invalid energy interval")
    points = sorted(points)
    start_w = interpolate(points, start_ns)
    end_w = interpolate(points, end_ns)
    bounded = [(start_ns, start_w)]
    bounded.extend(point for point in points if start_ns < point[0] < end_ns)
    bounded.append((end_ns, end_w))
    energy_j = 0.0
    for left, right in zip(bounded, bounded[1:]):
        duration_s = (right[0] - left[0]) / 1e9
        energy_j += duration_s * (left[1] + right[1]) / 2.0
    return energy_j


def integrate_rapl(
    rows: list[dict[str, Any]], start_ns: int, end_ns: int, clock: str
) -> float:
    samples = sorted(
        (
            int(row[clock]),
            int(row["rapl_energy_uj"]),
            int(row["rapl_max_energy_range_uj"]),
        )
        for row in rows
    )
    require(len(samples) >= 2, "RAPL sample count")
    max_range = samples[0][2]
    require(max_range > 0, "RAPL range")
    require(all(row[2] == max_range for row in samples), "RAPL range changed")
    unwrapped: list[tuple[int, float]] = [(samples[0][0], 0.0)]
    prior = samples[0][1]
    total = 0
    for sample_t, energy_uj, _ in samples[1:]:
        delta = energy_uj - prior
        if delta < 0:
            delta += max_range
        require(0 <= delta < max_range, "RAPL counter delta")
        total += delta
        unwrapped.append((sample_t, float(total)))
        prior = energy_uj
    start_uj = interpolate(unwrapped, start_ns)
    end_uj = interpolate(unwrapped, end_ns)
    require(end_uj >= start_uj, "RAPL energy direction")
    return (end_uj - start_uj) / 1e6


def parse_desktop_window(text: str) -> tuple[int, int]:
    values: dict[str, list[int]] = {"START": [], "END": []}
    for match in DESKTOP_MARKER.finditer(text):
        values[match.group(1)].append(int(match.group(2)))
    require(len(values["START"]) == 1, "desktop start marker")
    require(len(values["END"]) == 1, "desktop end marker")
    start_ns = values["START"][0]
    end_ns = values["END"][0]
    require(end_ns > start_ns, "desktop marker order")
    return start_ns, end_ns


def parse_phone_window(text: str) -> tuple[int, int]:
    values: dict[str, list[int]] = {"START": [], "END": []}
    for match in PHONE_MARKER.finditer(text):
        value = float(match.group(3))
        ns = int(round(value if match.group(2) == "ns" else value * 1e9))
        values[match.group(1)].append(ns)
    require(len(values["START"]) == 1, "phone start marker")
    require(len(values["END"]) == 1, "phone end marker")
    start_ns = values["START"][0]
    end_ns = values["END"][0]
    require(end_ns > start_ns, "phone marker order")
    return start_ns, end_ns


def read_phone_samples(path: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    with path.open(newline="") as stream:
        for raw in csv.DictReader(stream, delimiter="\t"):
            expected = {
                "uptime_s",
                "usb_current_ua",
                "usb_voltage_uv",
                "battery_current_ma",
                "battery_voltage_uv",
                "battery_charge_counter_uah",
            }
            require(set(raw) == expected, "phone sample fields")
            usb_current_ua = int(raw["usb_current_ua"])
            usb_voltage_uv = int(raw["usb_voltage_uv"])
            battery_current_ma = int(raw["battery_current_ma"])
            battery_voltage_uv = int(raw["battery_voltage_uv"])
            uptime_ns = int(round(float(raw["uptime_s"]) * 1e9))
            require(
                uptime_ns > 0
                and usb_current_ua >= 0
                and usb_voltage_uv > 0
                and battery_voltage_uv > 0,
                "phone sample bounds",
            )
            usb_w = usb_current_ua * usb_voltage_uv / 1e12
            battery_w = max(0, battery_current_ma) * battery_voltage_uv / 1e9
            rows.append({
                "battery_current_ma": battery_current_ma,
                "battery_discharge_w": battery_w,
                "phone_uptime_ns": uptime_ns,
                "total_w": usb_w + battery_w,
                "usb_input_w": usb_w,
            })
    coalesced: list[dict[str, float | int]] = []
    coalesced_counts: list[int] = []
    for row in rows:
        if (
            coalesced
            and row["phone_uptime_ns"] == coalesced[-1]["phone_uptime_ns"]
        ):
            prior = coalesced[-1]
            count = coalesced_counts[-1]
            for name in (
                "battery_current_ma",
                "battery_discharge_w",
                "total_w",
                "usb_input_w",
            ):
                prior[name] = (
                    float(prior[name]) * count + float(row[name])
                ) / (count + 1)
            coalesced_counts[-1] += 1
        else:
            coalesced.append(row)
            coalesced_counts.append(1)
    rows = coalesced
    require(len(rows) >= 3, "phone sample count")
    require(
        all(
            int(rows[index]["phone_uptime_ns"])
            > int(rows[index - 1]["phone_uptime_ns"])
            for index in range(1, len(rows))
        ),
        "phone sample order",
    )
    return rows


def phone_energy_summary(
    rows: list[dict[str, float | int]], start_ns: int, end_ns: int
) -> dict[str, Any]:
    def points(name: str) -> list[tuple[int, float]]:
        return [
            (int(row["phone_uptime_ns"]), float(row[name])) for row in rows
        ]

    usb_j = integrate_power(points("usb_input_w"), start_ns, end_ns)
    battery_j = integrate_power(
        points("battery_discharge_w"), start_ns, end_ns
    )
    duration_s = (end_ns - start_ns) / 1e9
    total_j = usb_j + battery_j
    return {
        "battery_discharge_energy_j": battery_j,
        "duration_s": duration_s,
        "end_phone_uptime_ns": end_ns,
        "sample_count": len(rows),
        "start_phone_uptime_ns": start_ns,
        "usb_input_energy_j": usb_j,
        "whole_phone_average_power_w": total_j / duration_s,
        "whole_phone_energy_j": total_j,
    }
