#!/usr/bin/env python3

import html
import json
import shutil
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "source_length_2x2_v1"
ANALYSIS = RESULTS / "ANALYSIS.json"
SVG = RESULTS / "SOURCE_LENGTH_2X2_COMPARISON_V1.svg"
PNG = RESULTS / "SOURCE_LENGTH_2X2_COMPARISON_V1.png"

WIDTH = 1800
HEIGHT = 720
PREFILL = "#2A9D8F"
DECODE = "#457B9D"
HOT = "#E76F51"
TEXT = "#243447"
MUTED = "#526577"
GRID = "#DCE3EA"


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def add_text(
    svg: list[str], x: float, y: float, value: str, size: int = 14,
    anchor: str = "start", color: str = TEXT, weight: str = "normal",
) -> None:
    lines = value.split("\n")
    svg.append(
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'text-anchor="{anchor}" fill="{color}" font-weight="{weight}">'
    )
    for index, line in enumerate(lines):
        dy = 0 if index == 0 else size * 1.2
        svg.append(
            f'<tspan x="{x:.1f}" dy="{dy:.1f}">{esc(line)}</tspan>'
        )
    svg.append("</text>")


def add_rotated_text(
    svg: list[str], x: float, y: float, value: str, size: int = 11,
    color: str = MUTED,
) -> None:
    svg.append(
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'text-anchor="middle" fill="{color}" '
        f'transform="rotate(-90 {x:.1f} {y:.1f})">{esc(value)}</text>'
    )


def add_rect(
    svg: list[str], x: float, y: float, width: float, height: float,
    fill: str, stroke: str = "none", radius: float = 0,
) -> None:
    svg.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" '
        f'height="{height:.1f}" fill="{fill}" stroke="{stroke}" '
        f'rx="{radius:.1f}"/>'
    )


def add_line(
    svg: list[str], x1: float, y1: float, x2: float, y2: float,
    color: str = GRID, width: float = 1,
) -> None:
    svg.append(
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" '
        f'y2="{y2:.1f}" stroke="{color}" stroke-width="{width:.1f}"/>'
    )


def draw_stacked_panel(
    svg: list[str], panel_x: float, title: str, labels: list[str],
    cases: list[dict], maximum: float, tick: float, rates=None, note=None,
) -> None:
    panel_y = 95
    panel_w = 560
    panel_h = 565
    chart_left = panel_x + 58
    chart_right = panel_x + panel_w - 20
    chart_top = panel_y + 115
    chart_bottom = panel_y + panel_h - 70
    chart_h = chart_bottom - chart_top
    add_rect(svg, panel_x, panel_y, panel_w, panel_h, "#FFFFFF", GRID, 12)
    add_text(svg, panel_x + 18, panel_y + 30, title, 17, weight="bold")
    if note:
        add_text(svg, panel_x + 18, panel_y + 57, note, 12, color=MUTED)

    value = 0.0
    while value <= maximum + 1e-9:
        y = chart_bottom - chart_h * value / maximum
        add_line(svg, chart_left, y, chart_right, y)
        add_text(svg, chart_left - 8, y + 4, f"{value:.0f}", 10, "end", MUTED)
        value += tick
    add_rotated_text(
        svg, panel_x + 17, chart_top + chart_h / 2, "cold service time (s)"
    )

    count = len(labels)
    usable = chart_right - chart_left
    slot = usable / count
    bar_w = min(90.0, slot * 0.62)
    for index, (label, case) in enumerate(zip(labels, cases)):
        center = chart_left + slot * (index + 0.5)
        prefill_h = chart_h * case["prefill_s"] / maximum
        decode_h = chart_h * case["decode_s"] / maximum
        add_rect(svg, center - bar_w / 2, chart_bottom - prefill_h, bar_w, prefill_h, PREFILL)
        add_rect(
            svg, center - bar_w / 2, chart_bottom - prefill_h - decode_h,
            bar_w, decode_h, DECODE,
        )
        total = case["service_s"]
        add_text(
            svg, center, chart_bottom - chart_h * total / maximum - 8,
            f"{total:.1f} s", 12, "middle", TEXT, "bold",
        )
        if rates is not None:
            add_text(
                svg, center, chart_bottom - prefill_h - decode_h * 0.48,
                f"{rates[index]:.3f}\nout tok/s", 11, "middle", "#FFFFFF", "bold",
            )
        add_text(svg, center, chart_bottom + 23, label, 11, "middle", TEXT)


def draw_delta_panel(svg: list[str], panel_x: float, groups: list[dict]) -> None:
    panel_y = 95
    panel_w = 610
    panel_h = 565
    chart_left = panel_x + 60
    chart_right = panel_x + panel_w - 20
    chart_top = panel_y + 120
    chart_bottom = panel_y + panel_h - 70
    chart_h = chart_bottom - chart_top
    minimum = -2.0
    maximum = 21.0
    add_rect(svg, panel_x, panel_y, panel_w, panel_h, "#FFFFFF", GRID, 12)
    add_text(svg, panel_x + 18, panel_y + 30, "C. GPU effect on CPU + OP15", 17, weight="bold")
    add_text(
        svg, panel_x + 18, panel_y + 57,
        "Only overlapping prefill slows; resident Qwen is noise-level.",
        12, color=MUTED,
    )

    def y_for(value: float) -> float:
        return chart_bottom - chart_h * (value - minimum) / (maximum - minimum)

    for value in (-2, 0, 5, 10, 15, 20):
        y = y_for(value)
        add_line(svg, chart_left, y, chart_right, y, TEXT if value == 0 else GRID)
        add_text(svg, chart_left - 8, y + 4, f"{value:+d}%", 10, "end", MUTED)
    add_rotated_text(
        svg, panel_x + 17, chart_top + chart_h / 2, "hot latency delta (%)"
    )

    labels = ["Req 0\n92% overlap", "Req 1\n53% overlap", "Req 2-65\n0% overlap"]
    series = [
        ("Service", "service_s", HOT),
        ("Prefill", "prefill_s", PREFILL),
        ("Decode", "decode_s", DECODE),
    ]
    slot = (chart_right - chart_left) / len(groups)
    bar_w = 34
    zero = y_for(0)
    for group_index, (label, group) in enumerate(zip(labels, groups)):
        center = chart_left + slot * (group_index + 0.5)
        for series_index, (_, key, color) in enumerate(series):
            value = group[key]["delta_pct"]
            x = center + (series_index - 1) * (bar_w + 5) - bar_w / 2
            y = y_for(max(value, 0))
            height = abs(y_for(value) - zero)
            add_rect(svg, x, y if value >= 0 else zero, bar_w, max(height, 1.5), color)
            label_y = y_for(value) - 7 if value >= 0 else y_for(value) + 14
            add_text(svg, x + bar_w / 2, label_y, f"{value:+.2f}%", 10, "middle", TEXT)
        add_text(svg, center, chart_bottom + 23, label, 11, "middle", TEXT)

    legend_x = panel_x + 110
    for index, (label, _, color) in enumerate(series):
        x = legend_x + index * 130
        add_rect(svg, x, panel_y + 77, 18, 12, color, radius=2)
        add_text(svg, x + 25, panel_y + 88, label, 11, color=MUTED)


def main() -> None:
    data = json.loads(ANALYSIS.read_text())
    full = data["full_17_request_cases"]
    prefix = data["matched_14_request_prefix"]
    effects = prefix["op15_hot_effect_by_overlap"]

    svg = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
        f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="#F7F9FB"/>',
        '<g font-family="DejaVu Sans, Arial, sans-serif">',
    ]
    add_text(
        svg, WIDTH / 2, 35,
        "Real BurstGPT source lengths: Gemma cold path on CPU and OP15",
        23, "middle", TEXT, "bold",
    )
    add_text(
        svg, WIDTH / 2, 62,
        "RTX 4060 Ti runs the Qwen hot trace; default clocks; one physical OP15; one repeat",
        13, "middle", MUTED,
    )

    full_cases = [
        full["cpu_hot_absent"],
        full["cpu_hot_trace"],
        full["op15_hot_absent"],
    ]
    draw_stacked_panel(
        svg, 20, "A. Complete 17-request cold run",
        ["CPU\nGPU absent", "CPU\nQwen trace", "CPU + OP15\nGPU absent"],
        full_cases, 2400, 400,
        [case["cold_output_tokens_s"] for case in full_cases],
        "CPU + OP15: 17.1% less makespan, 20.6% more output tok/s",
    )

    prefix_cases = [
        prefix["cases"]["cpu_hot_absent"],
        prefix["cases"]["cpu_hot_trace"],
        prefix["cases"]["op15_hot_absent"],
        prefix["cases"]["op15_hot_trace"],
    ]
    draw_stacked_panel(
        svg, 600, "B. Matched 14-request valid prefix",
        ["CPU\nGPU absent", "CPU\nQwen trace", "CPU + OP15\nGPU absent", "CPU + OP15\nQwen trace"],
        prefix_cases, 1800, 300, note="OP15 hot vs absent: +0.07% service before the USB reset",
    )

    draw_delta_panel(
        svg,
        1180,
        [
            effects["request_0_full_overlap"],
            effects["request_1_partial_overlap"],
            effects["requests_2_to_65_resident_only"],
        ],
    )

    add_rect(svg, 655, 671, 18, 12, PREFILL, radius=2)
    add_text(svg, 680, 682, "Prefill", 11, color=MUTED)
    add_rect(svg, 755, 671, 18, 12, DECODE, radius=2)
    add_text(svg, 780, 682, "Decode", 11, color=MUTED)
    add_text(
        svg, WIDTH / 2, 708,
        "Censored hot + OP15 arm: host reset SuperSpeed USB after request 65; two physical attempts failed.",
        12, "middle", "#9C2F2F", "bold",
    )
    svg.extend(["</g>", "</svg>"])
    SVG.write_text("\n".join(svg) + "\n")

    converter = shutil.which("convert")
    if converter:
        subprocess.run(
            [converter, "-background", "white", "-density", "160", str(SVG), str(PNG)],
            check=True,
        )
    print(SVG)
    if PNG.exists():
        print(PNG)


if __name__ == "__main__":
    main()
