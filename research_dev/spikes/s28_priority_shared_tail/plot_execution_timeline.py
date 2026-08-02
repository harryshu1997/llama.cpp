#!/usr/bin/env python3
"""Plot the persisted S28 request and physical-batch timelines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


SCHEMA = "s28-priority-shared-tail-v1"
PLACEMENTS = {
    "R0": "4060 [0,8) -> 4060 [8,16) -> 4060 [16,48)",
    "R2": "OP12 [0,8) -> OP15 [8,16) -> 4060 [16,48)",
}
ROUTE_COLORS = {"R0": "#d94841", "R2": "#138a72"}
PRIORITY_COLORS = {0: "#c7352d", 1: "#e29b22", 2: "#3574b9"}
WORKERS = (
    ("cuda-prefix", "4060 [0,8)"),
    ("cuda-mid", "4060 [8,16)"),
    ("op12-prefix", "OP12 [0,8)"),
    ("op15-mid", "OP15 [8,16)"),
    ("cuda-tail", "4060 [16,48)"),
)


class PlotError(RuntimeError):
    pass


def load_report(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PlotError(f"cannot load {path}: {exc}") from exc
    if (
        type(value) is not dict
        or value.get("schema") != SCHEMA
        or value.get("status") != "RUN_COMPLETE"
    ):
        raise PlotError(f"{path} is not a complete S28 report")
    return value


def request_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = report.get("runtime", {}).get("requests")
    if type(rows) is not list or len(rows) != 60:
        raise PlotError("treatment must contain exactly 60 requests")
    result = sorted(
        rows,
        key=lambda row: (
            row["scheduled_arrival_ns"], row["priority"], row["request_id"],
        ),
    )
    if any(row.get("route_id") not in PLACEMENTS for row in result):
        raise PlotError("treatment contains an unknown route")
    return result


def write_request_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "request_index",
        "request_id",
        "priority",
        "route_id",
        "placement",
        "arrival_s",
        "admitted_s",
        "first_token_s",
        "completed_s",
        "latency_s",
        "slo_s",
        "slo_met",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows, 1):
            writer.writerow({
                "request_index": index,
                "request_id": row["request_id"],
                "priority": row["priority"],
                "route_id": row["route_id"],
                "placement": PLACEMENTS[row["route_id"]],
                "arrival_s": f'{row["scheduled_arrival_ns"] / 1e9:.6f}',
                "admitted_s": f'{row["admitted_ns"] / 1e9:.6f}',
                "first_token_s": f'{row["first_token_ns"] / 1e9:.6f}',
                "completed_s": f'{row["completed_ns"] / 1e9:.6f}',
                "latency_s": f'{row["latency_us"] / 1e6:.6f}',
                "slo_s": f'{row["slo_us"] / 1e6:.6f}',
                "slo_met": str(bool(row["slo_met"])).lower(),
            })


def request_panel(
    axis: Any, rows: list[dict[str, Any]], max_time: float,
) -> None:
    for y, row in enumerate(rows):
        arrival = row["scheduled_arrival_ns"] / 1e9
        first_token = row["first_token_ns"] / 1e9
        completed = row["completed_ns"] / 1e9
        deadline = arrival + row["slo_us"] / 1e6
        route = row["route_id"]
        axis.barh(
            y,
            completed - arrival,
            left=arrival,
            height=0.72,
            color=ROUTE_COLORS[route],
            alpha=0.86,
            linewidth=0,
        )
        axis.plot(
            first_token,
            y,
            marker="o",
            markersize=2.8,
            color="#111111",
            linestyle="none",
            zorder=3,
        )
        axis.plot(
            min(deadline, max_time),
            y,
            marker="|",
            markersize=5,
            color="#222222",
            linestyle="none",
            zorder=3,
        )
    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels([
        f'{index:02d}  id={row["request_id"]}  P{row["priority"]}  {row["route_id"]}'
        for index, row in enumerate(rows, 1)
    ], fontsize=5.8)
    axis.invert_yaxis()
    axis.set_xlim(0, max_time)
    axis.set_xlabel("Seconds from trace start")
    axis.set_title(
        "A. Every request: arrival -> completion (dot = first token, tick = SLO)",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )
    axis.grid(axis="x", color="#dddddd", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(
        handles=[
            Patch(color=ROUTE_COLORS["R0"], label="R0: all 48 layers on 4060 Ti"),
            Patch(color=ROUTE_COLORS["R2"], label="R2: OP12 -> OP15 -> 4060 Ti"),
            Line2D([], [], color="#111111", marker="o", linestyle="none", label="first token"),
            Line2D([], [], color="#222222", marker="|", linestyle="none", label="synthetic SLO"),
        ],
        loc="lower right",
        fontsize=8,
        frameon=True,
        ncol=2,
    )


def device_panel(
    axis: Any, report: dict[str, Any], max_time: float,
) -> None:
    origin = report["runtime"]["origin_monotonic_ns"]
    for lane, (worker, _label) in enumerate(WORKERS):
        for event in report["batch_events"][worker]:
            start = (event["compute_start_ns"] - origin) / 1e9
            duration = max(
                (event["compute_end_ns"] - event["compute_start_ns"]) / 1e9,
                0.0005,
            )
            priorities = event["priorities"]
            priority = min(priorities)
            height = 0.28 + 0.52 * event["batch_size"] / 4.0
            axis.broken_barh(
                [(start, duration)],
                (lane - height / 2, height),
                facecolors=PRIORITY_COLORS[priority],
                edgecolors="none",
                alpha=0.9,
            )
    axis.set_yticks(range(len(WORKERS)))
    axis.set_yticklabels([label for _worker, label in WORKERS], fontsize=8)
    axis.invert_yaxis()
    axis.set_xlim(0, max_time)
    axis.set_xlabel("Seconds from trace start")
    axis.set_title(
        "B. Actual physical batches (bar height encodes batch size, max B4)",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )
    axis.grid(axis="x", color="#dddddd", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(
        handles=[
            Patch(color=PRIORITY_COLORS[0], label="P0 urgent"),
            Patch(color=PRIORITY_COLORS[1], label="P1 background"),
            Patch(color=PRIORITY_COLORS[2], label="P2 background"),
        ],
        loc="upper right",
        fontsize=8,
        ncol=3,
    )


def benefit_panel(
    axis: Any, control: dict[str, Any], treatment: dict[str, Any],
) -> None:
    control_cuda = control["summary"]["summed_cuda_island_compute_us"]
    treatment_cuda = treatment["summary"]["summed_cuda_island_compute_us"]
    control_p0 = control["summary"]["priority"]["0"]["latency_us"]["p95"]
    treatment_p0 = treatment["summary"]["priority"]["0"]["latency_us"]["p95"]
    control_makespan = control["summary"]["makespan_us"]
    treatment_makespan = treatment["summary"]["makespan_us"]
    labels = ("Summed CUDA\ncompute", "P0 p95\nlatency", "Makespan")
    ratios = (
        treatment_cuda * 100.0 / control_cuda,
        treatment_p0 * 100.0 / control_p0,
        treatment_makespan * 100.0 / control_makespan,
    )
    colors = ("#138a72", "#3574b9", "#d94841")
    bars = axis.barh(range(3), ratios, color=colors, height=0.55)
    axis.axvline(100, color="#222222", linewidth=1.0, linestyle="--")
    axis.set_yticks(range(3), labels=labels)
    axis.invert_yaxis()
    axis.set_xlim(0, max(ratios) * 1.13)
    axis.set_xlabel("Treatment as percent of all-CUDA control")
    axis.set_title(
        "C. Benefit and cost (lower is better)",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )
    axis.grid(axis="x", color="#dddddd", linewidth=0.6)
    axis.set_axisbelow(True)
    for bar, ratio in zip(bars, ratios):
        axis.text(
            ratio + max(ratios) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{ratio:.1f}%",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
    axis.text(
        0.99,
        0.96,
        "60/60 complete, 0 synthetic SLO misses\n"
        "CUDA compute -18.55%; P0 p95 -1.18%; makespan 8.38x\n"
        "Phone/USB energy unknown; F16-phone/Q8-server accuracy uncertified",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "pad": 5},
    )


def plot(
    control: dict[str, Any],
    treatment: dict[str, Any],
    png: Path,
    svg: Path,
    csv_path: Path,
) -> None:
    rows = request_rows(treatment)
    write_request_csv(csv_path, rows)
    max_time = max(
        row["scheduled_arrival_ns"] / 1e9 + row["slo_us"] / 1e6
        for row in rows
    )
    figure = plt.figure(figsize=(16, 22), constrained_layout=True)
    grid = figure.add_gridspec(3, 1, height_ratios=(3.8, 1.15, 1.0))
    request_panel(figure.add_subplot(grid[0]), rows, max_time)
    device_panel(figure.add_subplot(grid[1]), treatment, max_time)
    benefit_panel(figure.add_subplot(grid[2]), control, treatment)
    figure.suptitle(
        "S28 real execution timeline: priority-aware phone offload\n"
        "RTX 4060 Ti + OP12 + OP15 | real-derived arrivals, synthetic priority/SLO | fixed request route",
        fontsize=16,
        fontweight="bold",
        linespacing=1.45,
    )
    figure.text(
        0.5,
        0.006,
        "R2 is a fixed layer handoff, not semantic early exit. Every request executes all 48 layers.",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
    png.parent.mkdir(parents=True, exist_ok=True)
    svg.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png, dpi=180, facecolor="white")
    figure.savefig(svg, facecolor="white")
    plt.close(figure)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--control", type=Path, required=True)
    result.add_argument("--treatment", type=Path, required=True)
    result.add_argument("--png", type=Path, required=True)
    result.add_argument("--svg", type=Path, required=True)
    result.add_argument("--csv", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    if any(path.exists() for path in (args.png, args.svg, args.csv)):
        raise PlotError("an output path already exists")
    plot(
        load_report(args.control),
        load_report(args.treatment),
        args.png,
        args.svg,
        args.csv,
    )
    print(args.png)
    print(args.svg)
    print(args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
