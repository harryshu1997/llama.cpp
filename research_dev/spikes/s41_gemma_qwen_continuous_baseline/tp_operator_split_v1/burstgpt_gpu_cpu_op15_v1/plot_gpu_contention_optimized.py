#!/usr/bin/env python3
"""Plot baseline and optimized GPU-contention operating points."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


WIDTH = 1600
HEIGHT = 1050
TEXT = "#172033"
MUTED = "#64748b"
GRID = "#dbe2ea"
PANEL = "#f8fafc"
GROUPS = (
    ("resident-idle", "GPU idle control", "#4c78a8"),
    ("busy-baseline", "Busy baseline", "#e45756"),
    ("throughput-priority", "Optimized: keep Qwen speed", "#2a9d6f"),
    ("cold-priority", "Optimized: minimum cold latency", "#7b61a8"),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def text(
    x: float,
    y: float,
    value: object,
    css_class: str,
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" class="{css_class}" '
        f'text-anchor="{anchor}">{esc(value)}</text>'
    )


def rect(
    x: float,
    y: float,
    width: float,
    height: float,
    fill: str,
    radius: float = 0,
    opacity: float = 1.0,
    stroke: str | None = None,
) -> str:
    stroke_attr = "" if stroke is None else f' stroke="{stroke}"'
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(width, 0):.2f}" '
        f'height="{height:.2f}" rx="{radius:.2f}" fill="{fill}" '
        f'opacity="{opacity:.3f}"{stroke_attr}/>'
    )


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    stroke: str,
    width: float = 1.0,
    dash: str | None = None,
) -> str:
    dash_attr = "" if dash is None else f' stroke-dasharray="{dash}"'
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" '
        f'y2="{y2:.2f}" stroke="{stroke}" stroke-width="{width:.2f}"'
        f'{dash_attr}/>'
    )


def circle(
    x: float,
    y: float,
    radius: float,
    fill: str,
    stroke: str = "#ffffff",
) -> str:
    return (
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="3"/>'
    )


def metric(
    analysis: dict[str, Any],
    name: str,
    group: str,
) -> dict[str, Any]:
    value = analysis.get("aggregate", {}).get(name, {}).get(group)
    require(type(value) is dict, f"missing metric: {name} {group}")
    return value


def median(analysis: dict[str, Any], name: str, group: str) -> float:
    value = metric(analysis, name, group).get("median")
    require(type(value) in (int, float), f"missing median: {name} {group}")
    return float(value)


def change(
    analysis: dict[str, Any],
    group: str,
    name: str,
) -> float:
    value = analysis.get("changes_vs_busy_baseline_pct", {}).get(
        group, {}
    ).get(name)
    require(type(value) in (int, float), f"missing change: {group} {name}")
    return float(value)


def render(analysis: dict[str, Any]) -> str:
    require(
        analysis.get("schema") ==
            "s41-gpu-contention-optimization-analysis-v2"
        and analysis.get("status") == "PASS"
        and analysis.get("repeat_count") == 3
        and analysis.get("groups") == [group for group, _, _ in GROUPS]
        and analysis.get("token_sequences_identical_optimized") is True,
        "analysis identity",
    )
    idle_duration = median(analysis, "cold_duration_s", "resident-idle")
    busy_duration = median(analysis, "cold_duration_s", "busy-baseline")
    throughput_duration = median(
        analysis, "cold_duration_s", "throughput-priority"
    )
    cold_duration = median(analysis, "cold_duration_s", "cold-priority")
    throughput_hot = median(
        analysis, "hot_output_throughput_tokens_s", "throughput-priority"
    )
    cold_hot = median(
        analysis, "hot_output_throughput_tokens_s", "cold-priority"
    )

    output = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
            f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" '
            'role="img" aria-labelledby="title description">'
        ),
        '<title id="title">Optimized busy-GPU comparison for CPU plus OP15 inference</title>',
        (
            '<desc id="description">Three-run medians compare the idle control, '
            'the original busy-GPU baseline, and two optimized CPU-affinity '
            'policies for concurrent Qwen and Gemma inference.</desc>'
        ),
        "<style>",
        "text { font-family: Inter, DejaVu Sans, Arial, sans-serif; }",
        f".title {{ font-size: 30px; font-weight: 700; fill: {TEXT}; }}",
        f".subtitle {{ font-size: 15px; fill: {MUTED}; }}",
        f".section {{ font-size: 18px; font-weight: 700; fill: {TEXT}; }}",
        f".label {{ font-size: 14px; fill: {TEXT}; }}",
        f".small {{ font-size: 13px; fill: {TEXT}; }}",
        f".tick {{ font-size: 12px; fill: {MUTED}; }}",
        f".value {{ font-size: 14px; font-weight: 650; fill: {TEXT}; }}",
        f".card-value {{ font-size: 24px; font-weight: 700; fill: {TEXT}; }}",
        f".card-label {{ font-size: 13px; fill: {MUTED}; }}",
        f".callout {{ font-size: 15px; font-weight: 650; fill: {TEXT}; }}",
        f".note {{ font-size: 12px; fill: {MUTED}; }}",
        "</style>",
        rect(0, 0, WIDTH, HEIGHT, "#ffffff"),
        text(
            WIDTH / 2,
            42,
            "Busy-GPU interference recovered with two CPU policies",
            "title",
            "middle",
        ),
        text(
            WIDTH / 2,
            69,
            "RTX 4060 Ti + i9-12900K + OP15 | default clocks | three-run medians",
            "subtitle",
            "middle",
        ),
    ]

    cards = (
        (
            "Original busy penalty",
            f"{idle_duration:.1f}s -> {busy_duration:.1f}s",
            "+28.8% cold makespan",
            "#fff4f2",
        ),
        (
            "Keep Qwen speed",
            f"{throughput_duration:.1f}s | {throughput_hot:.2f} hot tok/s",
            "Gemma +11.6%; Qwen preserved",
            "#effaf5",
        ),
        (
            "Minimum cold latency",
            f"{cold_duration:.1f}s | {cold_hot:.2f} hot tok/s",
            "Gemma +24.6%; Qwen -3.9%",
            "#f6f2ff",
        ),
    )
    card_y = 94
    card_width = 470
    card_gap = 35
    card_x0 = (WIDTH - 3 * card_width - 2 * card_gap) / 2
    for index, (label, value, detail, fill) in enumerate(cards):
        x = card_x0 + index * (card_width + card_gap)
        output.append(rect(x, card_y, card_width, 105, fill, 12, 1, GRID))
        output.append(text(x + 22, card_y + 27, label, "card-label"))
        output.append(text(x + 22, card_y + 64, value, "card-value"))
        output.append(text(x + 22, card_y + 89, detail, "small"))

    left_x = 55
    left_width = 900
    output.append(rect(left_x, 225, left_width, 355, "#ffffff", 10, 1, GRID))
    output.append(text(80, 260, "Cold trace makespan", "section"))
    output.append(text(925, 260, "seconds; lower is better", "tick", "end"))
    plot_left = 330
    plot_right = 910
    plot_top = 300
    plot_bottom = 540
    time_max = 260.0
    for tick_value in (0, 50, 100, 150, 200, 250):
        x = plot_left + (plot_right - plot_left) * tick_value / time_max
        output.append(line(x, plot_top, x, plot_bottom, GRID))
        output.append(text(x, plot_top - 10, tick_value, "tick", "middle"))
    for index, (group, label, color) in enumerate(GROUPS):
        record = metric(analysis, "cold_duration_s", group)
        value = float(record["median"])
        y = 327 + index * 55
        width = (plot_right - plot_left) * value / time_max
        min_x = plot_left + (
            plot_right - plot_left
        ) * float(record["min"]) / time_max
        max_x = plot_left + (
            plot_right - plot_left
        ) * float(record["max"]) / time_max
        output.append(text(plot_left - 18, y + 6, label, "label", "end"))
        output.append(rect(plot_left, y - 15, width, 30, color, 4))
        output.append(line(min_x, y, max_x, y, TEXT, 2))
        output.append(line(min_x, y - 6, min_x, y + 6, TEXT, 2))
        output.append(line(max_x, y - 6, max_x, y + 6, TEXT, 2))
        output.append(text(
            min(plot_left + width + 10, plot_right - 3),
            y + 5,
            f"{value:.1f}",
            "value",
            "end" if plot_left + width + 55 > plot_right else "start",
        ))

    output.append(rect(left_x, 600, left_width, 355, "#ffffff", 10, 1, GRID))
    output.append(text(80, 635, "Cold phase latency", "section"))
    output.append(text(
        925,
        635,
        "p50 seconds; prefill and decode shown separately",
        "tick",
        "end",
    ))
    phase_left = 330
    phase_right = 910
    phase_top = 680
    phase_bottom = 885
    phase_max = 20.0
    for tick_value in (0, 5, 10, 15, 20):
        x = phase_left + (
            phase_right - phase_left
        ) * tick_value / phase_max
        output.append(line(x, phase_top, x, phase_bottom, GRID))
        output.append(text(x, phase_top - 10, tick_value, "tick", "middle"))
    for index, (group, label, color) in enumerate(GROUPS):
        prefill = median(analysis, "cold_prefill_p50_s", group)
        decode = median(analysis, "cold_decode_p50_s", group)
        y = 704 + index * 50
        prefill_width = (
            phase_right - phase_left
        ) * prefill / phase_max
        decode_width = (
            phase_right - phase_left
        ) * decode / phase_max
        output.append(text(phase_left - 18, y + 6, label, "label", "end"))
        output.append(rect(
            phase_left, y - 14, prefill_width, 28, color, 4
        ))
        output.append(rect(
            phase_left + prefill_width,
            y - 14,
            decode_width,
            28,
            color,
            4,
            0.42,
        ))
        output.append(text(
            phase_left + prefill_width / 2,
            y + 5,
            f"prefill {prefill:.2f}",
            "small",
            "middle",
        ))
        output.append(text(
            phase_left + prefill_width + decode_width / 2,
            y + 5,
            f"decode {decode:.2f}",
            "small",
            "middle",
        ))
    output.append(rect(330, 910, 18, 12, "#64748b", 2))
    output.append(text(356, 921, "prefill", "tick"))
    output.append(rect(430, 910, 18, 12, "#64748b", 2, 0.42))
    output.append(text(456, 921, "decode", "tick"))

    right_x = 980
    right_width = 565
    output.append(rect(right_x, 225, right_width, 730, "#ffffff", 10, 1, GRID))
    output.append(text(1005, 260, "Concurrent throughput tradeoff", "section"))
    output.append(text(
        1520,
        260,
        "top-right is better",
        "tick",
        "end",
    ))
    scatter_left = 1065
    scatter_right = 1505
    scatter_top = 315
    scatter_bottom = 690
    hot_min = 47.5
    hot_max = 50.8
    cold_min = 2.0
    cold_max = 2.7
    for tick_value in (48, 49, 50):
        x = scatter_left + (
            scatter_right - scatter_left
        ) * (tick_value - hot_min) / (hot_max - hot_min)
        output.append(line(x, scatter_top, x, scatter_bottom, GRID))
        output.append(text(x, scatter_bottom + 22, tick_value, "tick", "middle"))
    for tick_value in (2.0, 2.2, 2.4, 2.6):
        y = scatter_bottom - (
            scatter_bottom - scatter_top
        ) * (tick_value - cold_min) / (cold_max - cold_min)
        output.append(line(scatter_left, y, scatter_right, y, GRID))
        output.append(text(
            scatter_left - 12, y + 4, f"{tick_value:.1f}", "tick", "end"
        ))
    output.append(text(
        (scatter_left + scatter_right) / 2,
        scatter_bottom + 48,
        "Qwen hot throughput (token/s)",
        "label",
        "middle",
    ))
    output.append(text(
        1008,
        (scatter_top + scatter_bottom) / 2,
        "Gemma cold",
        "label",
    ))
    output.append(text(
        1008,
        (scatter_top + scatter_bottom) / 2 + 20,
        "throughput",
        "label",
    ))
    output.append(text(
        1008,
        (scatter_top + scatter_bottom) / 2 + 40,
        "(token/s)",
        "label",
    ))

    scatter_groups = GROUPS[1:]
    points = []
    for group, label, color in scatter_groups:
        hot = median(analysis, "hot_output_throughput_tokens_s", group)
        cold = median(analysis, "cold_output_throughput_tokens_s", group)
        x = scatter_left + (
            scatter_right - scatter_left
        ) * (hot - hot_min) / (hot_max - hot_min)
        y = scatter_bottom - (
            scatter_bottom - scatter_top
        ) * (cold - cold_min) / (cold_max - cold_min)
        points.append((x, y, group, label, color, hot, cold))
    output.append(line(
        points[0][0], points[0][1], points[1][0], points[1][1], MUTED, 2, "6,5"
    ))
    output.append(line(
        points[1][0], points[1][1], points[2][0], points[2][1], MUTED, 2, "6,5"
    ))
    label_offsets = {
        "busy-baseline": (-12, 31, "end"),
        "throughput-priority": (-12, -17, "end"),
        "cold-priority": (14, -17, "start"),
    }
    for x, y, group, label, color, hot, cold in points:
        output.append(circle(x, y, 11, color))
        dx, dy, anchor = label_offsets[group]
        output.append(text(x + dx, y + dy, label, "value", anchor))
        output.append(text(
            x + dx,
            y + dy + 18,
            f"{hot:.2f} hot | {cold:.3f} cold",
            "tick",
            anchor,
        ))

    output.append(rect(1010, 775, 505, 145, "#f8fafc", 9, 1, GRID))
    output.append(text(1032, 807, "Choose policy by service objective", "callout"))
    output.append(text(
        1032,
        838,
        "Keep Qwen speed: share P-cores; Gemma +11.6%.",
        "small",
    ))
    output.append(text(
        1032,
        866,
        "Minimize cold latency: isolate P-core primaries; Gemma +24.6%.",
        "small",
    ))
    output.append(text(
        1032,
        894,
        "Both use the same dynamic 8192 / 11136 prefill and 9664 decode split.",
        "tick",
    ))

    output.append(text(
        55,
        990,
        "Workload: 17 BurstGPT-timed Gemma4-12B Q4_0 requests (505 output tokens) on CPU + OP15; Qwen3-14B Q4_K_M saturates the GPU.",
        "note",
    ))
    output.append(text(
        55,
        1014,
        "Bars and points are three-run medians; makespan whiskers show min-max. All optimized runs returned the same token sequences.",
        "note",
    ))
    output.append(text(
        55,
        1038,
        "The idle control has no active Qwen requests and is omitted from the throughput scatter.",
        "note",
    ))
    output.append("</svg>")
    return "\n".join(output) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.output.suffix.lower() == ".svg", "output must be SVG")
    analysis = json.loads(args.analysis.read_text(encoding="ascii"))
    args.output.write_text(render(analysis), encoding="ascii")
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
