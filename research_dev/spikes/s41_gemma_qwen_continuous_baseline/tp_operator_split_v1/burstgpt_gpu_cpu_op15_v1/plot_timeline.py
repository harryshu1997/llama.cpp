#!/usr/bin/env python3
"""Plot the paired BurstGPT cold-route timelines as a standalone SVG."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import html
import json
import math
from pathlib import Path
from typing import Any


WIDTH = 1660
HEIGHT = 1020
LEFT = 85
RIGHT = 55
PLOT_WIDTH = WIDTH - LEFT - RIGHT

COLORS = {
    "background": "#ffffff",
    "grid": "#dbe2ea",
    "axis": "#475569",
    "text": "#172033",
    "muted": "#64748b",
    "cpu": "#0072b2",
    "op15": "#d55e00",
    "arrival": "#555555",
}


@dataclass(frozen=True)
class RequestSpan:
    request_index: int
    input_tokens: int
    output_tokens: int
    arrival_s: float
    dispatch_s: float
    prefill_end_s: float
    decode_end_s: float
    completion_s: float

    @property
    def queue_s(self) -> float:
        return self.dispatch_s - self.arrival_s

    @property
    def prefill_s(self) -> float:
        return self.prefill_end_s - self.dispatch_s

    @property
    def decode_s(self) -> float:
        return self.decode_end_s - self.prefill_end_s

    @property
    def overhead_s(self) -> float:
        return self.completion_s - self.decode_end_s


@dataclass(frozen=True)
class Run:
    label: str
    mode: str
    duration_s: float
    throughput: float
    prefill_p50_s: float
    decode_p50_s: float
    service_p50_s: float
    spans: tuple[RequestSpan, ...]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_requests(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="ascii") as source:
        for line_number, line in enumerate(source, 1):
            row = json.loads(line)
            require(type(row) is dict, f"requests line {line_number}")
            index = row.get("request_index")
            require(type(index) is int and index not in rows, "request index")
            rows[index] = row
    return rows


def load_run(
    path: Path,
    expected_mode: str,
    label: str,
    requests: dict[int, dict[str, Any]],
) -> Run:
    value = json.loads(path.read_text(encoding="ascii"))
    require(value.get("status") == "PASS", f"{label}: status")
    require(value.get("mode") == expected_mode, f"{label}: mode")
    paid_start_ns = value.get("paid_start_ns")
    paid_end_ns = value.get("paid_end_ns")
    require(type(paid_start_ns) is int, f"{label}: paid start")
    require(type(paid_end_ns) is int and paid_end_ns > paid_start_ns,
            f"{label}: paid end")
    metrics = value.get("metrics")
    require(type(metrics) is dict, f"{label}: metrics")
    cold_metrics = metrics.get("by_role", {}).get("cold")
    require(type(cold_metrics) is dict, f"{label}: cold metrics")

    spans: list[RequestSpan] = []
    for result in value.get("request_results", []):
        if result.get("role") != "cold":
            continue
        index = result.get("request_index")
        require(type(index) is int and index in requests,
                f"{label}: result request index")
        trace_row = requests[index]
        arrival_s = (result["scheduled_arrival_ns"] - paid_start_ns) / 1e9
        dispatch_s = (result["dispatch_ns"] - paid_start_ns) / 1e9
        prefill_end_s = dispatch_s + result["prefill_us"] / 1e6
        decode_end_s = prefill_end_s + result["decode_us"] / 1e6
        completion_s = (result["completion_ns"] - paid_start_ns) / 1e9
        require(
            0 <= arrival_s <= dispatch_s <= prefill_end_s <= decode_end_s
            <= completion_s + 1e-6,
            f"{label}: request {index} timeline order",
        )
        spans.append(RequestSpan(
            request_index=index,
            input_tokens=trace_row["input_tokens"],
            output_tokens=trace_row["output_tokens"],
            arrival_s=arrival_s,
            dispatch_s=dispatch_s,
            prefill_end_s=prefill_end_s,
            decode_end_s=decode_end_s,
            completion_s=completion_s,
        ))
    spans.sort(key=lambda span: (span.arrival_s, span.request_index))
    require(len(spans) == 17, f"{label}: cold request count")
    return Run(
        label=label,
        mode=expected_mode,
        duration_s=(paid_end_ns - paid_start_ns) / 1e9,
        throughput=float(metrics["throughput_tokens_s"]),
        prefill_p50_s=float(cold_metrics["prefill_s"]["p50"]),
        decode_p50_s=float(cold_metrics["decode_s"]["p50"]),
        service_p50_s=float(cold_metrics["service_s"]["p50"]),
        spans=tuple(spans),
    )


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def text_element(
    x: float,
    y: float,
    text: object,
    css_class: str,
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" class="{css_class}" '
        f'text-anchor="{anchor}">{esc(text)}</text>'
    )


def rect(x: float, y: float, width: float, height: float, fill: str,
         radius: float = 0, opacity: float = 1.0) -> str:
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(0.0, width):.2f}" '
        f'height="{height:.2f}" rx="{radius:.2f}" fill="{fill}" '
        f'opacity="{opacity:.3f}"/>'
    )


def line(x1: float, y1: float, x2: float, y2: float, stroke: str,
         width: float = 1, dash: str | None = None,
         opacity: float = 1.0) -> str:
    dash_attr = "" if dash is None else f' stroke-dasharray="{dash}"'
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" '
        f'y2="{y2:.2f}" stroke="{stroke}" stroke-width="{width:.2f}"'
        f'{dash_attr} opacity="{opacity:.3f}"/>'
    )


def render(cpu: Run, op15: Run) -> str:
    time_max_s = max(cpu.duration_s, op15.duration_s)
    plot_y = 155
    plot_height = 520
    total_tokens = sum(span.output_tokens for span in cpu.spans)
    require(
        total_tokens == sum(span.output_tokens for span in op15.spans),
        "paired output-token count",
    )
    y_max = math.ceil(total_tokens / 100) * 100
    cpu_mean = total_tokens / cpu.duration_s
    op15_mean = total_tokens / op15.duration_s
    time_saved_s = cpu.duration_s - op15.duration_s
    cpu_tokens_at_op15_finish = sum(
        span.output_tokens
        for span in cpu.spans
        if span.completion_s <= op15.duration_s
    )

    def sx(seconds: float) -> float:
        return LEFT + seconds / time_max_s * PLOT_WIDTH

    def sy(tokens: float) -> float:
        return plot_y + plot_height * (1 - tokens / y_max)

    def cumulative_path(run: Run) -> str:
        commands = [f"M {sx(0):.1f} {sy(0):.1f}"]
        completed = 0
        for span in sorted(run.spans, key=lambda item: item.completion_s):
            commands.append(f"H {sx(span.completion_s):.1f}")
            completed += span.output_tokens
            commands.append(f"V {sy(completed):.1f}")
        commands.append(f"H {sx(time_max_s):.1f}")
        return " ".join(commands)

    output = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
            f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" '
            f'role="img" aria-labelledby="title description">'
        ),
        '<title id="title">Gemma4 cumulative CPU versus CPU plus OP15 timeline</title>',
        (
            '<desc id="description">Cumulative completed output tokens for one '
            'Gemma model, with measured execution spans and request markers.</desc>'
        ),
        "<style>",
        "text { font-family: Inter, DejaVu Sans, Arial, sans-serif; }",
        ".title { font-size: 30px; font-weight: 700; fill: #172033; }",
        ".subtitle { font-size: 15px; font-style: italic; fill: #475569; }",
        ".axis { font-size: 13px; font-style: italic; fill: #334155; }",
        ".tick { font-size: 12px; font-style: italic; fill: #334155; }",
        ".legend { font-size: 13px; font-style: italic; fill: #334155; }",
        ".lane { font-size: 13px; font-style: italic; fill: #334155; }",
        ".note { font-size: 11px; fill: #64748b; }",
        "</style>",
        rect(0, 0, WIDTH, HEIGHT, COLORS["background"]),
        text_element(WIDTH / 2, 43,
                     "Cumulative throughput: Gemma4 CPU vs CPU + OP15",
                     "title", "middle"),
        text_element(
            WIDTH / 2, 70,
            f"one Gemma 4 12B Q4_0 model | OP15 finishes {time_saved_s:.1f}s earlier | "
            f"CPU has {cpu_tokens_at_op15_finish}/{total_tokens} tokens at that time",
            "subtitle", "middle",
        ),
    ]

    legend_y = 111
    output.append(line(95, legend_y, 139, legend_y,
                       COLORS["cpu"], 4))
    output.append(text_element(
        150, legend_y + 5,
        f"full CPU (mean {cpu_mean:.2f} token/s)", "legend"))
    output.append(line(555, legend_y, 599, legend_y,
                       COLORS["op15"], 4))
    output.append(text_element(
        610, legend_y + 5,
        f"CPU + OP15 (mean {op15_mean:.2f} token/s)", "legend"))
    output.append(rect(1050, legend_y - 8, 28, 13,
                       COLORS["op15"], 0, 0.85))
    output.append(text_element(1088, legend_y + 5,
                               "request executing", "legend"))
    output.append(
        f'<circle cx="1340" cy="{legend_y}" r="5" fill="#ffffff" '
        f'stroke="{COLORS["arrival"]}" stroke-width="2"/>'
    )
    output.append(text_element(1352, legend_y + 5, "arrival", "legend"))
    output.append(
        f'<circle cx="1465" cy="{legend_y}" r="5" '
        f'fill="{COLORS["op15"]}"/>'
    )
    output.append(text_element(1477, legend_y + 5, "completion", "legend"))

    output.append(rect(LEFT, plot_y, PLOT_WIDTH, plot_height,
                       "#ffffff"))
    for tick in range(7):
        value = y_max * tick / 6
        y = sy(value)
        output.append(line(LEFT, y, LEFT + PLOT_WIDTH, y,
                           COLORS["grid"], 1.2))
        output.append(text_element(LEFT - 12, y + 5, f"{value:.0f}",
                                   "tick", "end"))
    axis_center_y = plot_y + plot_height / 2
    output.append(
        f'<text x="22" y="{axis_center_y:.1f}" class="axis" '
        f'text-anchor="middle" transform="rotate(-90 22 {axis_center_y:.1f})">'
        'completed output tokens</text>'
    )
    for tick in range(6):
        seconds = time_max_s * tick / 5
        x = sx(seconds)
        output.append(line(x, plot_y, x, plot_y + plot_height,
                           COLORS["grid"], 1.2))
        output.append(text_element(x, plot_y + plot_height + 26,
                                   f"{seconds:.1f}", "tick", "middle"))
    output.append(text_element(
        LEFT + PLOT_WIDTH / 2, plot_y + plot_height + 54,
        "time since paid start (s)", "axis", "middle"))

    last_arrival = max(span.arrival_s for span in cpu.spans)
    output.append(line(sx(last_arrival), plot_y,
                       sx(last_arrival), plot_y + plot_height,
                       COLORS["arrival"], 1.2, "5 4", 0.7))
    output.append(text_element(
        sx(last_arrival) + 6, plot_y + 19,
        f"last arrival {last_arrival:.1f}s", "note"))
    for run, color in ((cpu, COLORS["cpu"]), (op15, COLORS["op15"])):
        output.append(line(sx(run.duration_s), plot_y,
                           sx(run.duration_s), plot_y + plot_height,
                           color, 1.4, "3 4", 0.75))

    output.append(rect(
        sx(op15.duration_s), plot_y,
        sx(cpu.duration_s) - sx(op15.duration_s), plot_height,
        "#fff7ed",
    ))
    bracket_y = sy(total_tokens + 50)
    output.append(line(sx(op15.duration_s), bracket_y,
                       sx(cpu.duration_s), bracket_y,
                       COLORS["op15"], 2))
    output.append(line(sx(op15.duration_s), bracket_y - 7,
                       sx(op15.duration_s), bracket_y + 7,
                       COLORS["op15"], 2))
    output.append(line(sx(cpu.duration_s), bracket_y - 7,
                       sx(cpu.duration_s), bracket_y + 7,
                       COLORS["op15"], 2))
    output.append(text_element(
        (sx(op15.duration_s) + sx(cpu.duration_s)) / 2,
        bracket_y - 8, f"{time_saved_s:.1f}s earlier",
        "legend", "middle"))

    output.append(
        f'<path d="{cumulative_path(cpu)}" fill="none" '
        f'stroke="{COLORS["cpu"]}" stroke-width="3.4"/>'
    )
    output.append(
        f'<path d="{cumulative_path(op15)}" fill="none" '
        f'stroke="{COLORS["op15"]}" stroke-width="3.4"/>'
    )
    output.append(
        f'<circle cx="{sx(op15.duration_s):.2f}" '
        f'cy="{sy(total_tokens):.2f}" r="5" fill="{COLORS["op15"]}"/>'
    )
    output.append(text_element(
        sx(op15.duration_s) + 8, sy(total_tokens) + 20,
        f"OP15: {total_tokens} tokens at {op15.duration_s:.1f}s",
        "legend"))
    output.append(
        f'<circle cx="{sx(cpu.duration_s):.2f}" '
        f'cy="{sy(total_tokens):.2f}" r="5" fill="{COLORS["cpu"]}"/>'
    )
    output.append(text_element(
        sx(cpu.duration_s) - 8, sy(total_tokens) + 40,
        f"CPU: {total_tokens} tokens at {cpu.duration_s:.1f}s",
        "legend", "end"))

    lane_specs = (
        (cpu, COLORS["cpu"], 790,
         f"full CPU executing: {len(cpu.spans)} measured cold request spans"),
        (op15, COLORS["op15"], 838,
         f"CPU + OP15 executing: {len(op15.spans)} measured cold request spans"),
    )
    for run, color, lane_y, label in lane_specs:
        output.append(text_element(LEFT, lane_y - 9, label, "lane"))
        output.append(rect(LEFT, lane_y, PLOT_WIDTH, 20,
                           "#f1f5f9"))
        for span in run.spans:
            start_x = sx(span.dispatch_s)
            width = max(1.2, sx(span.completion_s) - start_x)
            output.append(
                f'<rect x="{start_x:.2f}" y="{lane_y}" '
                f'width="{width:.2f}" height="20" fill="{color}" '
                'fill-opacity="0.85" stroke="#ffffff" stroke-width="0.7">'
                f'<title>R{span.request_index:02d}: '
                f'{span.input_tokens} prompt, {span.output_tokens} output; '
                f'execution {span.completion_s - span.dispatch_s:.3f}s</title>'
                '</rect>'
            )

    marker_y = 910
    output.append(text_element(
        LEFT, marker_y - 18,
        "cold request arrivals and completions", "lane"))
    output.append(line(LEFT, marker_y, LEFT + PLOT_WIDTH, marker_y,
                       COLORS["grid"], 1.2))
    for span in cpu.spans:
        output.append(
            f'<circle cx="{sx(span.arrival_s):.2f}" cy="{marker_y}" '
            f'r="5" fill="#ffffff" stroke="{COLORS["arrival"]}" '
            'stroke-width="2"/>'
        )
    for span in cpu.spans:
        output.append(
            f'<circle cx="{sx(span.completion_s):.2f}" '
            f'cy="{marker_y - 8}" r="4.5" fill="{COLORS["cpu"]}" '
            'stroke="#ffffff" stroke-width="0.8"/>'
        )
    for span in op15.spans:
        output.append(
            f'<circle cx="{sx(span.completion_s):.2f}" '
            f'cy="{marker_y + 8}" r="4.5" fill="{COLORS["op15"]}" '
            'stroke="#ffffff" stroke-width="0.8"/>'
        )
    output.append(text_element(
        LEFT, 958,
        "Blue completion markers are full CPU; orange markers are CPU + OP15. Hollow markers are shared arrivals.",
        "note"))
    output.append(text_element(
        LEFT, 978,
        "Curves step at exact request completion; generated-token publication timestamps were not recorded.",
        "note"))
    output.append(text_element(
        LEFT, 998,
        "The concurrent hot model is omitted. Both Gemma runs use --no-repack; CPU + OP15 is operator-level FFN overlap.",
        "note"))
    output.append("</svg>")
    return "\n".join(output) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--op15", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.output.suffix.lower() == ".svg", "output must be SVG")
    requests = load_requests(args.requests)
    cpu = load_run(args.cpu, "cpu", "Full Gemma model on desktop CPU",
                   requests)
    op15 = load_run(args.op15, "op15",
                    "Desktop CPU + OP15 dynamic FFN offload", requests)
    args.output.write_text(render(cpu, op15), encoding="ascii")
    print(json.dumps({
        "cpu_duration_s": cpu.duration_s,
        "op15_duration_s": op15.duration_s,
        "output": str(args.output),
        "request_count_per_panel": len(cpu.spans),
        "speedup": cpu.duration_s / op15.duration_s,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
