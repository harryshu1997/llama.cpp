#!/usr/bin/env python3
"""Render aligned GPU, CPU, and OP15 activity for one physical trace."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from html import escape
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable


WIDTH = 1600
HEIGHT = 1160
LEFT = 150
RIGHT = 105
PLOT_WIDTH = WIDTH - LEFT - RIGHT
PANEL_HEIGHT = 220
PANEL_TOPS = (165, 475, 785)


class PlotError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PlotError(message)


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PlotError(f"cannot read {path}: {exc}") from exc
    require(type(value) is dict, f"JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        for line in path.read_text(encoding="ascii").splitlines():
            value = json.loads(line)
            require(type(value) is dict, f"JSONL object: {path}")
            rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PlotError(f"cannot read {path}: {exc}") from exc
    require(rows, f"nonempty JSONL: {path}")
    return rows


def mean_bins(
    points: Iterable[tuple[float, float]],
    width_s: float,
    duration_s: float,
) -> list[tuple[float, float]]:
    count = int(math.ceil(duration_s / width_s))
    values: list[list[float]] = [[] for _ in range(count)]
    for time_s, value in points:
        if not 0 <= time_s <= duration_s or not math.isfinite(value):
            continue
        index = min(int(time_s // width_s), count - 1)
        values[index].append(value)
    result = []
    for index, rows in enumerate(values):
        if rows:
            result.append((
                min(duration_s, (index + 0.5) * width_s),
                statistics.fmean(rows),
            ))
    return result


def percentile(values: list[float], fraction: float) -> float:
    require(values and 0 <= fraction <= 1, "percentile input")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def map_phone_samples(
    path: Path,
    before: dict[str, Any],
    after: dict[str, Any],
    paid_start_ns: int,
) -> list[tuple[float, float]]:
    require(
        before.get("schema") == "s41-phone-clock-anchor-v1"
        and after.get("schema") == "s41-phone-clock-anchor-v1"
        and before.get("boot_id") == after.get("boot_id"),
        "phone clock anchors",
    )
    phone_delta = after["phone_uptime_ns"] - before["phone_uptime_ns"]
    host_delta = after["host_midpoint_ns"] - before["host_midpoint_ns"]
    require(phone_delta > 0 and host_delta > 0, "phone clock direction")
    slope = host_delta / phone_delta
    require(abs(slope - 1.0) <= 0.002, "phone clock drift")
    points = []
    with path.open(newline="", encoding="ascii") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            uptime_ns = int(round(float(row["uptime_s"]) * 1e9))
            host_ns = before["host_midpoint_ns"] + int(round(
                (uptime_ns - before["phone_uptime_ns"]) * slope
            ))
            usb_w = (
                int(row["usb_current_ua"])
                * int(row["usb_voltage_uv"])
                / 1e12
            )
            battery_w = (
                max(0, int(row["battery_current_ma"]))
                * int(row["battery_voltage_uv"])
                / 1e9
            )
            points.append(((host_ns - paid_start_ns) / 1e9, usb_w + battery_w))
    require(len(points) >= 3, "phone samples")
    return points


def svg_path(
    points: list[tuple[float, float]],
    x_scale: Any,
    y_scale: Any,
) -> str:
    return " ".join(
        ("M" if index == 0 else "L")
        + f"{x_scale(x):.2f},{y_scale(y):.2f}"
        for index, (x, y) in enumerate(points)
    )


def render(
    result_path: Path,
    resources_path: Path,
    phone_samples_path: Path,
    clock_before_path: Path,
    clock_after_path: Path,
    phone_energy_path: Path,
    cpu_sar_path: Path,
) -> str:
    result = read_object(result_path)
    before = read_object(clock_before_path)
    after = read_object(clock_after_path)
    phone_energy = read_object(phone_energy_path)
    cpu_sar = read_object(cpu_sar_path)
    require(
        result.get("schema") == "s41-hierarchical-burstgpt-result-v1"
        and result.get("status") == "PASS"
        and phone_energy.get("schema") == "s41-phone-energy-v3"
        and phone_energy.get("status") == "PASS"
        and cpu_sar.get("schema") == "s42-runtime-placement-cpu-sar-v1",
        "input identity",
    )
    start_ns = result["paid_start_ns"]
    end_ns = result["paid_end_ns"]
    duration_s = (end_ns - start_ns) / 1e9
    require(
        math.isclose(duration_s, result["metrics"]["duration_s"])
        and math.isclose(duration_s, phone_energy["duration_s"]),
        "paid interval binding",
    )
    switch_start_s = result["switch"]["hot_end_s"]
    switch_end_s = result["switch"]["gpu_ready_s"]
    require(
        0 < switch_start_s < switch_end_s < duration_s,
        "phase boundaries",
    )

    resources = read_jsonl(resources_path)
    gpu_points = []
    cpu_power_points = []
    previous = None
    for row in resources:
        require(
            row.get("schema") == "s41-hierarchical-resource-v1",
            "resource schema",
        )
        gpu = row.get("gpu")
        rapl = row.get("rapl_package")
        require(type(gpu) is dict and type(rapl) is dict, "resource sample")
        gpu_points.append((
            (gpu["sample_t_ns"] - start_ns) / 1e9,
            float(gpu["utilization_pct"]),
        ))
        if previous is not None:
            t0 = previous["sample_t_ns"]
            t1 = rapl["sample_t_ns"]
            if t1 > t0:
                energy_delta = rapl["energy_uj"] - previous["energy_uj"]
                if energy_delta < 0:
                    energy_delta += rapl["max_energy_range_uj"]
                cpu_power_points.append((
                    ((t0 + t1) / 2 - start_ns) / 1e9,
                    energy_delta / 1e6 / ((t1 - t0) / 1e9),
                ))
        previous = rapl
    gpu_bins = mean_bins(gpu_points, 5.0, duration_s)
    cpu_power_bins = mean_bins(cpu_power_points, 10.0, duration_s)
    require(gpu_bins and cpu_power_bins, "desktop plot samples")

    phone_points = map_phone_samples(
        phone_samples_path, before, after, start_ns
    )
    phone_bins = mean_bins(phone_points, 10.0, duration_s)
    idle_values = [
        value for time_s, value in phone_points
        if switch_start_s <= time_s <= switch_end_s
    ]
    active_values = [
        value for time_s, value in phone_bins
        if not switch_start_s <= time_s <= switch_end_s
    ]
    require(idle_values and active_values, "OP15 activity calibration")
    phone_idle_w = statistics.fmean(idle_values)
    phone_active_w = percentile(active_values, 0.95)
    require(phone_active_w > phone_idle_w, "OP15 proxy range")
    op15_points = [
        (
            time_s,
            min(100.0, max(
                0.0,
                100.0 * (power_w - phone_idle_w)
                / (phone_active_w - phone_idle_w),
            )),
        )
        for time_s, power_w in phone_bins
    ]

    mapping = cpu_sar["trace_clock_mapping"]
    require(
        mapping["paid_start_monotonic_ns"] == start_ns
        and mapping["paid_end_monotonic_ns"] == end_ns,
        "CPU sar trace binding",
    )
    wall_start = datetime.fromisoformat(mapping["paid_start_local"])
    cpu_intervals = []
    for row in cpu_sar["intervals"]:
        low = (datetime.fromisoformat(row["start"]) - wall_start).total_seconds()
        high = (datetime.fromisoformat(row["end"]) - wall_start).total_seconds()
        low = max(0.0, low)
        high = min(duration_s, high)
        if high > low:
            cpu_intervals.append((low, high, float(row["busy_pct"])))
    require(cpu_intervals, "CPU sar coverage")
    covered_s = sum(high - low for low, high, _ in cpu_intervals)
    require(covered_s >= duration_s - 1.0, "CPU sar interval coverage")
    cpu_weighted_mean = sum(
        (high - low) * value for low, high, value in cpu_intervals
    ) / covered_s

    x = lambda value: LEFT + PLOT_WIDTH * value / duration_s
    y_gpu = lambda value: PANEL_TOPS[0] + PANEL_HEIGHT * (1 - value / 100)
    y_cpu = lambda value: PANEL_TOPS[1] + PANEL_HEIGHT * (1 - value / 30)
    y_cpu_w = lambda value: PANEL_TOPS[1] + PANEL_HEIGHT * (1 - value / 160)
    y_op15 = lambda value: PANEL_TOPS[2] + PANEL_HEIGHT * (1 - value / 100)
    lines: list[str] = []
    add = lines.append
    add(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
        f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">'
    )
    metadata = {
        "cpu_definition": "system-wide sar interval average",
        "input_sha256": {
            "clock_after": sha256(clock_after_path),
            "clock_before": sha256(clock_before_path),
            "cpu_sar": sha256(cpu_sar_path),
            "phone_energy": sha256(phone_energy_path),
            "phone_samples": sha256(phone_samples_path),
            "resources": sha256(resources_path),
            "result": sha256(result_path),
        },
        "op15_definition": (
            "10-second whole-phone power normalized from resident-idle mean "
            "to active-phase p95; not an HTP hardware counter"
        ),
        "schema": "s42-runtime-device-utilization-graph-v1",
    }
    add("<metadata>" + escape(json.dumps(metadata, sort_keys=True)) + "</metadata>")
    add("""
<defs>
  <marker id="arrow-blue" markerWidth="10" markerHeight="8" refX="9"
          refY="4" orient="auto">
    <path d="M0,0 L10,4 L0,8 z" fill="#2563eb"/>
  </marker>
  <marker id="arrow-green" markerWidth="10" markerHeight="8" refX="9"
          refY="4" orient="auto">
    <path d="M0,0 L10,4 L0,8 z" fill="#059669"/>
  </marker>
  <marker id="arrow-orange" markerWidth="10" markerHeight="8" refX="9"
          refY="4" orient="auto">
    <path d="M0,0 L10,4 L0,8 z" fill="#d97706"/>
  </marker>
  <style>
    text { font-family: Inter, DejaVu Sans, Arial, sans-serif; fill: #172033; }
    .title { font-size: 29px; font-weight: 700; }
    .subtitle { font-size: 15px; fill: #4b5563; }
    .panel-title { font-size: 19px; font-weight: 700; }
    .axis { font-size: 13px; fill: #526070; }
    .small { font-size: 12px; fill: #5f6b7a; }
    .note { font-size: 13px; fill: #4b5563; }
    .transition { font-size: 13px; font-weight: 650; }
  </style>
</defs>
<rect x="0" y="0" width="1600" height="1160" fill="#ffffff"/>
""")
    add(
        '<text x="70" y="48" class="title">'
        'Runtime device utilization - FP16 BurstGPT</text>'
    )
    add(
        '<text x="70" y="76" class="subtitle">'
        'RTX 4060 Ti + 24-thread desktop CPU + OP15, 74 requests, '
        f'{duration_s / 60:.2f} min paid trace</text>'
    )

    phase_rows = (
        (0.0, switch_start_s, "#dbeafe"),
        (switch_start_s, switch_end_s, "#ffedd5"),
        (switch_end_s, duration_s, "#dcfce7"),
    )
    for panel_index, top in enumerate(PANEL_TOPS):
        add(
            f'<rect x="{LEFT}" y="{top}" width="{PLOT_WIDTH}" '
            f'height="{PANEL_HEIGHT}" rx="6" fill="#fbfdff" '
            f'stroke="#cbd5e1"/>'
        )
        for low, high, color in phase_rows:
            add(
                f'<rect x="{x(low):.2f}" y="{top}" '
                f'width="{x(high) - x(low):.2f}" height="{PANEL_HEIGHT}" '
                f'fill="{color}" opacity="0.42"/>'
            )
        for minute in range(0, 46, 5):
            time_s = minute * 60
            if time_s > duration_s:
                continue
            add(
                f'<line x1="{x(time_s):.2f}" y1="{top}" '
                f'x2="{x(time_s):.2f}" y2="{top + PANEL_HEIGHT}" '
                'stroke="#dbe3ec" stroke-width="1"/>'
            )
            if panel_index == 2:
                add(
                    f'<text x="{x(time_s):.2f}" '
                    f'y="{top + PANEL_HEIGHT + 25}" class="axis" '
                    f'text-anchor="middle">{minute}</text>'
                )

    for boundary in (switch_start_s, switch_end_s):
        add(
            f'<line x1="{x(boundary):.2f}" y1="135" '
            f'x2="{x(boundary):.2f}" y2="{PANEL_TOPS[2] + PANEL_HEIGHT}" '
            'stroke="#9a6700" stroke-width="1.5" stroke-dasharray="7 5"/>'
        )

    labels = (
        (PANEL_TOPS[0], "GPU utilization (%)", "NVML, 5 s display bins"),
        (
            PANEL_TOPS[1],
            "CPU utilization (%)",
            "system-wide sar intervals; dashed line is RAPL package power",
        ),
        (
            PANEL_TOPS[2],
            "OP15 activity proxy (%)",
            "normalized 10 s whole-phone power; HTP utilization not captured",
        ),
    )
    for top, title, subtitle in labels:
        add(f'<text x="70" y="{top + 25}" class="panel-title">{escape(title)}</text>')
        add(f'<text x="70" y="{top + 46}" class="small">{escape(subtitle)}</text>')

    for value in (0, 25, 50, 75, 100):
        position = y_gpu(value)
        add(
            f'<line x1="{LEFT}" y1="{position:.2f}" '
            f'x2="{LEFT + PLOT_WIDTH}" y2="{position:.2f}" '
            'stroke="#dbe3ec"/>'
        )
        add(
            f'<text x="{LEFT - 12}" y="{position + 5:.2f}" '
            f'class="axis" text-anchor="end">{value}</text>'
        )
    gpu_path = svg_path(gpu_bins, x, y_gpu)
    gpu_area = (
        f'M{x(gpu_bins[0][0]):.2f},{y_gpu(0):.2f} '
        + gpu_path
        + f' L{x(gpu_bins[-1][0]):.2f},{y_gpu(0):.2f} Z'
    )
    add(f'<path d="{gpu_area}" fill="#3b82f6" opacity="0.20"/>')
    add(
        f'<path d="{gpu_path}" fill="none" stroke="#2563eb" '
        'stroke-width="2.2" stroke-linejoin="round"/>'
    )
    gpu_mean = result["resources"]["gpu_utilization_pct"]["mean"]
    add(
        f'<text x="{LEFT + PLOT_WIDTH - 12}" y="{PANEL_TOPS[0] + 25}" '
        f'class="transition" text-anchor="end">mean {gpu_mean:.1f}%, '
        'p95 100%</text>'
    )

    for value in (0, 10, 20, 30):
        position = y_cpu(value)
        add(
            f'<line x1="{LEFT}" y1="{position:.2f}" '
            f'x2="{LEFT + PLOT_WIDTH}" y2="{position:.2f}" '
            'stroke="#dbe3ec"/>'
        )
        add(
            f'<text x="{LEFT - 12}" y="{position + 5:.2f}" '
            f'class="axis" text-anchor="end">{value}</text>'
        )
    for low, high, value in cpu_intervals:
        top = y_cpu(value)
        add(
            f'<rect x="{x(low):.2f}" y="{top:.2f}" '
            f'width="{x(high) - x(low):.2f}" '
            f'height="{y_cpu(0) - top:.2f}" fill="#f59e0b" opacity="0.32"/>'
        )
        add(
            f'<line x1="{x(low):.2f}" y1="{top:.2f}" '
            f'x2="{x(high):.2f}" y2="{top:.2f}" '
            'stroke="#d97706" stroke-width="2.4"/>'
        )
    cpu_power_path = svg_path(cpu_power_bins, x, y_cpu_w)
    add(
        f'<path d="{cpu_power_path}" fill="none" stroke="#9a3412" '
        'stroke-width="1.5" stroke-dasharray="6 4" opacity="0.85"/>'
    )
    for value in (0, 40, 80, 120, 160):
        position = y_cpu_w(value)
        add(
            f'<text x="{LEFT + PLOT_WIDTH + 12}" y="{position + 5:.2f}" '
            f'class="axis">{value} W</text>'
        )
    cpu_power_mean = result["server_energy"]["cpu_package_average_power_w"]
    add(
        f'<text x="{LEFT + PLOT_WIDTH - 12}" y="{PANEL_TOPS[1] + 25}" '
        f'class="transition" text-anchor="end">sar weighted mean '
        f'{cpu_weighted_mean:.1f}%; package mean {cpu_power_mean:.1f} W</text>'
    )

    for value in (0, 25, 50, 75, 100):
        position = y_op15(value)
        add(
            f'<line x1="{LEFT}" y1="{position:.2f}" '
            f'x2="{LEFT + PLOT_WIDTH}" y2="{position:.2f}" '
            'stroke="#dbe3ec"/>'
        )
        add(
            f'<text x="{LEFT - 12}" y="{position + 5:.2f}" '
            f'class="axis" text-anchor="end">{value}</text>'
        )
    op15_path = svg_path(op15_points, x, y_op15)
    op15_area = (
        f'M{x(op15_points[0][0]):.2f},{y_op15(0):.2f} '
        + op15_path
        + f' L{x(op15_points[-1][0]):.2f},{y_op15(0):.2f} Z'
    )
    add(f'<path d="{op15_area}" fill="#10b981" opacity="0.22"/>')
    add(
        f'<path d="{op15_path}" fill="none" stroke="#059669" '
        'stroke-width="2.2" stroke-linejoin="round"/>'
    )
    add(
        f'<text x="{LEFT + PLOT_WIDTH - 12}" y="{PANEL_TOPS[2] + 25}" '
        f'class="transition" text-anchor="end">whole-phone mean '
        f'{phone_energy["whole_phone_average_power_w"]:.2f} W; '
        f'resident-idle calibration {phone_idle_w:.2f} W</text>'
    )

    switch_x1 = x(switch_start_s)
    switch_x2 = x(switch_end_s)
    add(
        f'<line x1="{switch_x1:.2f}" y1="405" x2="{switch_x2:.2f}" '
        'y2="405" stroke="#d97706" stroke-width="3" '
        'marker-end="url(#arrow-orange)"/>'
    )
    add(
        f'<line x1="{switch_x1:.2f}" y1="385" x2="{switch_x1:.2f}" '
        'y2="448" stroke="#2563eb" stroke-width="2" '
        'marker-end="url(#arrow-blue)"/>'
    )
    add(
        f'<text x="{switch_x1 - 10:.2f}" y="425" class="transition" '
        'text-anchor="end">Qwen complete</text>'
    )
    add(
        f'<line x1="{switch_x2:.2f}" y1="448" x2="{switch_x2:.2f}" '
        'y2="385" stroke="#059669" stroke-width="2" '
        'marker-end="url(#arrow-green)"/>'
    )
    add(
        f'<text x="{switch_x2 + 10:.2f}" y="425" class="transition">'
        'Gemma CUDA ready</text>'
    )
    add(
        f'<text x="{(switch_x1 + switch_x2) / 2:.2f}" y="454" '
        'class="small" text-anchor="middle">49.8 s GPU model transition</text>'
    )

    for time_s, label, color, marker in (
        (
            switch_start_s * 0.48,
            "Qwen FFN -> HTP1/HTP2",
            "#2563eb",
            "arrow-blue",
        ),
        (
            switch_end_s + (duration_s - switch_end_s) * 0.55,
            "Gemma suffix -> HTP0",
            "#059669",
            "arrow-green",
        ),
    ):
        position = x(time_s)
        add(
            f'<line x1="{position:.2f}" y1="710" x2="{position:.2f}" '
            f'y2="760" stroke="{color}" stroke-width="2.2" '
            f'marker-end="url(#{marker})"/>'
        )
        add(
            f'<text x="{position + 10:.2f}" y="738" '
            f'class="transition">{escape(label)}</text>'
        )

    qwen_mid = x(switch_start_s / 2)
    gemma_mid = x((switch_end_s + duration_s) / 2)
    add(
        f'<text x="{qwen_mid:.2f}" y="130" class="transition" '
        'text-anchor="middle">Qwen: CUDA18 + CPU + OP15 HTP1/2</text>'
    )
    add(
        f'<text x="{gemma_mid:.2f}" y="130" class="transition" '
        'text-anchor="middle">Gemma: CUDA25 + CPU + OP15 HTP0</text>'
    )
    add(
        f'<text x="{LEFT + PLOT_WIDTH / 2:.2f}" y="1045" '
        'class="axis" text-anchor="middle">Elapsed paid-trace time (minutes)</text>'
    )
    add(
        '<text x="70" y="1085" class="note">GPU is measured NVML '
        'utilization. CPU bars are measured 10-minute sar interval averages; '
        'the dashed line is 10-second RAPL package power.</text>'
    )
    add(
        '<text x="70" y="1110" class="note">OP15 is a power-derived '
        'activity proxy because this run did not capture Qualcomm HTP '
        'hardware-utilization counters. It must not be read as DSP occupancy.</text>'
    )
    add(
        '<text x="70" y="1136" class="small">Source: runtime placement '
        'retest op15-r51; exact paid interval and synchronized phone clocks.</text>'
    )
    add("</svg>")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--phone-samples", type=Path, required=True)
    parser.add_argument("--clock-before", type=Path, required=True)
    parser.add_argument("--clock-after", type=Path, required=True)
    parser.add_argument("--phone-energy", type=Path, required=True)
    parser.add_argument("--cpu-sar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.exists():
        parser.error("output must be an absolute new path")
    try:
        output = render(
            args.result,
            args.resources,
            args.phone_samples,
            args.clock_before,
            args.clock_after,
            args.phone_energy,
            args.cpu_sar,
        )
        args.output.write_text(output, encoding="ascii")
    except (OSError, ValueError) as exc:
        parser.exit(2, f"runtime utilization plot failed: {exc}\n")
    print(json.dumps({
        "output": str(args.output),
        "sha256": sha256(args.output),
        "status": "PASS",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
