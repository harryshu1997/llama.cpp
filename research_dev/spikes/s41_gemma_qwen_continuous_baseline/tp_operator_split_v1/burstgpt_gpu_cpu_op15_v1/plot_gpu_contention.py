#!/usr/bin/env python3
"""Plot the GPU resident-idle versus saturated contention result."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


WIDTH = 1600
HEIGHT = 980
BLUE = "#0072b2"
ORANGE = "#d55e00"
TEXT = "#172033"
MUTED = "#64748b"
GRID = "#dbe2ea"
PANEL = "#f8fafc"


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


def metric(analysis: dict[str, Any], name: str) -> dict[str, Any]:
    value = analysis.get("aggregate", {}).get(name)
    require(type(value) is dict, f"missing metric: {name}")
    return value


def median(analysis: dict[str, Any], name: str, mode: str) -> float:
    value = metric(analysis, name).get(mode, {}).get("median")
    require(type(value) in (int, float), f"missing median: {name} {mode}")
    return float(value)


def change(analysis: dict[str, Any], name: str) -> float:
    value = metric(analysis, name).get("saturated_change_pct")
    require(type(value) in (int, float), f"missing change: {name}")
    return float(value)


def render(analysis: dict[str, Any]) -> str:
    require(
        analysis.get("schema") == "s41-gpu-contention-analysis-v1"
        and analysis.get("status") == "PASS"
        and analysis.get("repeat_count") == 3
        and analysis.get("token_sequences_identical") is True,
        "analysis identity",
    )

    latency_rows = (
        ("Cold makespan", "cold_duration_s"),
        ("Completion p50", "cold_completion_p50_s"),
        ("Queue p50", "cold_queue_p50_s"),
        ("Service p50", "cold_service_p50_s"),
        ("Prefill p50", "cold_prefill_p50_s"),
        ("Decode p50", "cold_decode_p50_s"),
    )
    operator_rows = (
        ("Prefill FFN overlap", "prefill_overlap_p50_ms"),
        ("Prefill phone RPC", "prefill_phone_rpc_p50_ms"),
        ("Prefill HTP compute", "prefill_phone_compute_p50_ms"),
        ("Prefill USB interval", "prefill_usb_p50_ms"),
        ("Decode FFN overlap", "decode_overlap_p50_ms"),
    )

    idle_throughput = median(
        analysis, "cold_output_throughput_tokens_s", "resident-idle"
    )
    busy_throughput = median(
        analysis, "cold_output_throughput_tokens_s", "saturated"
    )
    idle_duration = median(analysis, "cold_duration_s", "resident-idle")
    busy_duration = median(analysis, "cold_duration_s", "saturated")
    busy_gpu = median(analysis, "gpu_utilization_p50_pct", "saturated")

    output = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
            f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" '
            'role="img" aria-labelledby="title description">'
        ),
        '<title id="title">GPU contention comparison for CPU plus OP15 cold inference</title>',
        (
            '<desc id="description">Three-run median comparison of a resident-idle '
            'and saturated Qwen model on the RTX 4060 Ti while Gemma runs on the '
            'desktop CPU and OP15.</desc>'
        ),
        "<style>",
        "text { font-family: Inter, DejaVu Sans, Arial, sans-serif; }",
        f".title {{ font-size: 30px; font-weight: 700; fill: {TEXT}; }}",
        f".subtitle {{ font-size: 15px; fill: {MUTED}; }}",
        f".section {{ font-size: 18px; font-weight: 700; fill: {TEXT}; }}",
        f".label {{ font-size: 14px; fill: {TEXT}; }}",
        f".tick {{ font-size: 12px; fill: {MUTED}; }}",
        f".value {{ font-size: 13px; font-weight: 600; fill: {TEXT}; }}",
        f".card-value {{ font-size: 26px; font-weight: 700; fill: {TEXT}; }}",
        f".card-label {{ font-size: 13px; fill: {MUTED}; }}",
        f".callout {{ font-size: 16px; font-weight: 600; fill: {TEXT}; }}",
        f".note {{ font-size: 12px; fill: {MUTED}; }}",
        "</style>",
        rect(0, 0, WIDTH, HEIGHT, "#ffffff"),
        text(
            WIDTH / 2,
            43,
            "Busy RTX 4060 Ti slows CPU + OP15 cold inference",
            "title",
            "middle",
        ),
        text(
            WIDTH / 2,
            70,
            "same resident Qwen3-14B model | 3-run medians | default desktop frequency",
            "subtitle",
            "middle",
        ),
    ]

    cards = (
        (
            "Cold makespan",
            f"{idle_duration:.1f}s to {busy_duration:.1f}s",
            f"+{change(analysis, 'cold_duration_s'):.1f}%",
        ),
        (
            "Cold throughput",
            f"{idle_throughput:.3f} to {busy_throughput:.3f} tok/s",
            f"{change(analysis, 'cold_output_throughput_tokens_s'):.1f}%",
        ),
        (
            "Saturated GPU load",
            f"{busy_gpu:.0f}% utilization p50",
            "100% p95",
        ),
    )
    card_y = 100
    card_width = 465
    card_gap = 30
    card_x0 = (WIDTH - (3 * card_width + 2 * card_gap)) / 2
    for index, (label, value, delta) in enumerate(cards):
        x = card_x0 + index * (card_width + card_gap)
        output.append(rect(x, card_y, card_width, 104, PANEL, 12, 1, GRID))
        output.append(text(x + 22, card_y + 28, label, "card-label"))
        output.append(text(x + 22, card_y + 66, value, "card-value"))
        output.append(text(x + card_width - 22, card_y + 88, delta, "value", "end"))

    output.append(rect(55, 230, 950, 635, "#ffffff", 10, 1, GRID))
    output.append(text(80, 265, "Cold route latency", "section"))
    output.append(text(980, 265, "seconds", "tick", "end"))
    plot_left = 250
    plot_right = 960
    plot_top = 305
    plot_height = 505
    time_max = 250.0
    for tick_value in (0, 50, 100, 150, 200, 250):
        x = plot_left + (plot_right - plot_left) * tick_value / time_max
        output.append(line(x, plot_top, x, plot_top + plot_height, GRID, 1))
        output.append(text(x, plot_top - 9, tick_value, "tick", "middle"))

    row_step = 82
    bar_height = 21
    for index, (label, name) in enumerate(latency_rows):
        center_y = plot_top + 42 + index * row_step
        idle = median(analysis, name, "resident-idle")
        busy = median(analysis, name, "saturated")
        idle_width = (plot_right - plot_left) * idle / time_max
        busy_width = (plot_right - plot_left) * busy / time_max
        output.append(text(plot_left - 18, center_y + 5, label, "label", "end"))
        output.append(rect(plot_left, center_y - 24, idle_width, bar_height, BLUE, 3))
        output.append(rect(plot_left, center_y + 5, busy_width, bar_height, ORANGE, 3))
        output.append(text(
            min(plot_left + idle_width + 8, plot_right - 4),
            center_y - 8,
            f"{idle:.3f}",
            "value",
            "end" if plot_left + idle_width + 70 > plot_right else "start",
        ))
        output.append(text(
            min(plot_left + busy_width + 8, plot_right - 4),
            center_y + 21,
            f"{busy:.3f} ({change(analysis, name):+.1f}%)",
            "value",
            "end" if plot_left + busy_width + 120 > plot_right else "start",
        ))

    legend_y = 840
    output.append(rect(250, legend_y - 13, 24, 13, BLUE, 2))
    output.append(text(282, legend_y, "Qwen resident-idle", "label"))
    output.append(rect(450, legend_y - 13, 24, 13, ORANGE, 2))
    output.append(text(482, legend_y, "Qwen saturated", "label"))

    output.append(rect(1030, 230, 515, 635, "#ffffff", 10, 1, GRID))
    output.append(text(1055, 265, "Where the slowdown occurs", "section"))
    output.append(text(
        1055,
        291,
        "busy time relative to resident-idle = 100%",
        "tick",
    ))
    operator_left = 1240
    operator_right = 1510
    operator_top = 330
    operator_max = 160.0
    for tick_value in (100, 120, 140, 160):
        x = operator_left + (
            operator_right - operator_left
        ) * (tick_value - 100) / (operator_max - 100)
        output.append(line(x, operator_top - 16, x, 690, GRID, 1))
        output.append(text(x, operator_top - 23, f"{tick_value}%", "tick", "middle"))
    output.append(line(operator_left, operator_top - 16, operator_left, 690, BLUE, 2))

    operator_step = 70
    for index, (label, name) in enumerate(operator_rows):
        y = operator_top + index * operator_step
        idle = median(analysis, name, "resident-idle")
        busy = median(analysis, name, "saturated")
        ratio = busy / idle * 100
        width = (operator_right - operator_left) * (
            ratio - 100
        ) / (operator_max - 100)
        output.append(text(operator_left - 16, y + 5, label, "label", "end"))
        output.append(rect(operator_left, y - 13, max(width, 2), 25, ORANGE, 3))
        output.append(text(
            min(operator_left + width + 8, operator_right),
            y + 5,
            f"{idle:.2f} to {busy:.2f} ms ({ratio - 100:+.1f}%)",
            "value",
            "end" if operator_left + width + 155 > operator_right else "start",
        ))

    output.append(rect(1055, 720, 465, 105, "#fff7ed", 9))
    output.append(text(
        1288,
        754,
        "Prefill FFN overlap: +48.9%",
        "callout",
        "middle",
    ))
    output.append(text(
        1288,
        780,
        "Phone RPC, HTP, and USB: all below +1%",
        "label",
        "middle",
    ))
    output.append(text(
        1288,
        806,
        "The desktop prefill branch is the bottleneck.",
        "label",
        "middle",
    ))

    output.append(text(
        55,
        910,
        "Workload: 17 serialized Gemma4-12B Q4_0 cold requests, 505 output tokens, dynamic FFN split on CPU + OP15.",
        "note",
    ))
    output.append(text(
        55,
        934,
        "Treatment: one Qwen3-14B Q4_K_M model, eight continuously refilled GPU slots. Error ranges are in the report.",
        "note",
    ))
    output.append(text(
        55,
        958,
        "All six runs returned identical cold token sequences; frequencies were not changed or locked.",
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
