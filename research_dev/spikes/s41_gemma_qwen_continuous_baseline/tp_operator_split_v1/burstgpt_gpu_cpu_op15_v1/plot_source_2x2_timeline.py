#!/usr/bin/env python3
"""Plot the source-length BurstGPT comparison as aligned execution timelines."""

from __future__ import annotations

from dataclasses import dataclass
import html
import json
import shutil
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
REQUESTS = HERE / "REQUESTS_SEMANTIC_SOURCE.jsonl"
RESULTS = HERE / "results" / "source_length_2x2_v1"
ANALYSIS = RESULTS / "ANALYSIS.json"
SVG = RESULTS / "SOURCE_LENGTH_2X2_TIMELINE_V1.svg"
PNG = RESULTS / "SOURCE_LENGTH_2X2_TIMELINE_V1.png"

WIDTH = 1800
HEIGHT = 1120
LEFT = 286
RIGHT = 72
PLOT_WIDTH = WIDTH - LEFT - RIGHT

BACKGROUND = "#F7F9FB"
PANEL = "#FFFFFF"
TEXT = "#203044"
MUTED = "#5D6D7E"
GRID = "#DCE3EA"
PREFILL = "#2A9D8F"
DECODE = "#3F78B5"
HOT = "#E76F51"
ARRIVALS = "#94A3B8"
FAILURE = "#B8323C"


@dataclass(frozen=True)
class RequestSpan:
    request_index: int
    input_tokens: int
    output_tokens: int
    dispatch_s: float
    prefill_end_s: float
    decode_end_s: float
    completion_s: float

    @property
    def service_s(self) -> float:
        return self.decode_end_s - self.dispatch_s


@dataclass(frozen=True)
class Run:
    key: str
    label: str
    detail: str
    spans: tuple[RequestSpan, ...]
    finish_s: float
    hot_window: tuple[float, float] | None = None
    reset_s: float | None = None


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
    opacity: float = 1.0,
    stroke: str = "none",
    stroke_width: float = 0,
    radius: float = 0,
) -> str:
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(0.0, width):.2f}" '
        f'height="{height:.2f}" rx="{radius:.2f}" fill="{fill}" '
        f'fill-opacity="{opacity:.3f}" stroke="{stroke}" '
        f'stroke-width="{stroke_width:.2f}"/>'
    )


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str,
    width: float = 1,
    dash: str | None = None,
    opacity: float = 1.0,
) -> str:
    dash_attribute = "" if dash is None else f' stroke-dasharray="{dash}"'
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" '
        f'y2="{y2:.2f}" stroke="{color}" stroke-width="{width:.2f}"'
        f'{dash_attribute} opacity="{opacity:.3f}"/>'
    )


def load_request_specs() -> dict[int, dict]:
    specs = {}
    for raw_line in REQUESTS.read_text(encoding="ascii").splitlines():
        row = json.loads(raw_line)
        index = row["request_index"]
        require(index not in specs, f"duplicate request {index}")
        specs[index] = row
    return specs


def result_to_span(result: dict, origin_ns: int, specs: dict[int, dict]) -> RequestSpan:
    index = result["request_index"]
    spec = specs[index]
    dispatch_s = (result["dispatch_ns"] - origin_ns) / 1e9
    prefill_end_s = dispatch_s + result["prefill_us"] / 1e6
    decode_end_s = prefill_end_s + result["decode_us"] / 1e6
    completion_s = (result["completion_ns"] - origin_ns) / 1e9
    require(
        0 <= dispatch_s <= prefill_end_s <= decode_end_s <= completion_s + 1e-4,
        f"request {index} phase order",
    )
    return RequestSpan(
        request_index=index,
        input_tokens=spec["input_tokens"],
        output_tokens=spec["output_tokens"],
        dispatch_s=dispatch_s,
        prefill_end_s=prefill_end_s,
        decode_end_s=decode_end_s,
        completion_s=completion_s,
    )


def load_complete_run(
    key: str,
    label: str,
    detail: str,
    specs: dict[int, dict],
    hot_window: tuple[float, float] | None = None,
) -> Run:
    value = json.loads((RESULTS / key / "RESULT.json").read_text())
    require(value["status"] == "PASS", f"{key}: incomplete result")
    origin_ns = value["paid_start_ns"]
    spans = tuple(
        sorted(
            (
                result_to_span(row, origin_ns, specs)
                for row in value["request_results"]
                if row["role"] == "cold"
            ),
            key=lambda span: span.dispatch_s,
        )
    )
    require(len(spans) == 17, f"{key}: expected 17 cold requests")
    return Run(
        key=key,
        label=label,
        detail=detail,
        spans=spans,
        finish_s=(value["paid_end_ns"] - origin_ns) / 1e9,
        hot_window=hot_window,
    )


def load_censored_run(
    specs: dict[int, dict],
    analysis: dict,
    op15_no_hot: Run,
) -> Run:
    rows = [
        json.loads(raw_line)
        for raw_line in (
            RESULTS / "op15_hot_trace_valid_prefix" / "events.jsonl"
        ).read_text().splitlines()
    ]
    origin_ns = next(row["t_ns"] for row in rows if row.get("kind") == "trace_start")
    valid_indices = set(analysis["matched_14_request_prefix"]["indices"])
    valid_results = [
        row
        for row in rows
        if row.get("kind") == "request_complete"
        and row.get("role") == "cold"
        and row["request_index"] in valid_indices
    ]
    spans = tuple(
        sorted(
            (result_to_span(row, origin_ns, specs) for row in valid_results),
            key=lambda span: span.dispatch_s,
        )
    )
    require(len(spans) == 14, "op15 hot trace: expected 14 valid requests")

    failed_result = next(
        row
        for row in rows
        if row.get("kind") == "request_complete" and row.get("request_index") == 66
    )
    baseline_66 = next(span for span in op15_no_hot.spans if span.request_index == 66)
    calls_into_request = 248924 - 248544
    calls_per_layer_sweep = 48
    prefill_calls = calls_per_layer_sweep * (
        (specs[66]["input_tokens"] + 511) // 512
    )
    completed_decode_tokens = (
        calls_into_request - prefill_calls
    ) / calls_per_layer_sweep
    decode_s_per_token = (
        baseline_66.decode_end_s - baseline_66.prefill_end_s
    ) / specs[66]["output_tokens"]
    reset_offset_s = (
        baseline_66.prefill_end_s - baseline_66.dispatch_s
        + completed_decode_tokens * decode_s_per_token
    )
    failed_dispatch_s = (failed_result["dispatch_ns"] - origin_ns) / 1e9
    reset_s = failed_dispatch_s + reset_offset_s
    hot = analysis["hot_service_windows"]["op15_hot_trace"]
    return Run(
        key="op15_hot_trace",
        label="CPU + OP15 | Qwen trace",
        detail="14 valid; USB reset during R66",
        spans=spans,
        finish_s=spans[-1].completion_s,
        hot_window=(hot["start_s"], hot["end_s"]),
        reset_s=reset_s,
    )


def add_panel_background(
    output: list[str],
    top: float,
    bottom: float,
    time_max_s: float,
    tick_s: float,
    title_value: str,
    subtitle_value: str,
    arrival_start_s: float,
    arrival_end_s: float,
) -> tuple:
    plot_top = top + 92
    plot_bottom = bottom - 57
    output.append(rect(28, top, WIDTH - 56, bottom - top, PANEL, 1, GRID, 1, 12))
    output.append(text(52, top + 32, title_value, "panel-title"))
    output.append(text(52, top + 57, subtitle_value, "panel-subtitle"))

    def sx(seconds: float) -> float:
        return LEFT + seconds / time_max_s * PLOT_WIDTH

    output.append(
        rect(
            sx(max(0, arrival_start_s)),
            plot_top,
            sx(min(time_max_s, arrival_end_s)) - sx(max(0, arrival_start_s)),
            plot_bottom - plot_top,
            ARRIVALS,
            0.11,
        )
    )
    tick_value = 0.0
    while tick_value <= time_max_s + 1e-9:
        x = sx(tick_value)
        output.append(line(x, plot_top, x, plot_bottom, GRID, 1))
        output.append(text(x, plot_bottom + 23, f"{tick_value:.0f}", "tick", "middle"))
        tick_value += tick_s
    output.append(text(
        LEFT + PLOT_WIDTH / 2,
        bottom - 14,
        "time since trace start (s)",
        "axis",
        "middle",
    ))
    return sx, plot_top, plot_bottom


def add_lane(
    output: list[str],
    run: Run,
    y: float,
    sx,
    time_max_s: float,
    zoom: bool,
) -> None:
    bar_height = 32 if zoom else 30
    output.append(text(LEFT - 14, y + 3, run.label, "lane-label", "end"))
    output.append(text(LEFT - 14, y + 22, run.detail, "lane-detail", "end"))
    output.append(rect(LEFT, y - 14, PLOT_WIDTH, bar_height, "#EEF2F6", 1, GRID, 0.6, 3))
    if run.hot_window is not None:
        hot_start, hot_end = run.hot_window
        output.append(
            rect(
                sx(max(0, hot_start)),
                y - 20,
                sx(min(time_max_s, hot_end)) - sx(max(0, hot_start)),
                bar_height + 12,
                HOT,
                0.16,
                HOT,
                0.8,
                2,
            )
        )

    def add_phase(start_s: float, end_s: float, color: str, tooltip: str) -> None:
        visible_start_s = max(0, start_s)
        visible_end_s = min(time_max_s, end_s)
        if visible_end_s <= visible_start_s:
            return
        output.append(
            f'<rect x="{sx(visible_start_s):.2f}" y="{y - 13}" '
            f'width="{max(0.8, sx(visible_end_s) - sx(visible_start_s)):.2f}" '
            f'height="{bar_height - 2}" fill="{color}" '
            'stroke="#FFFFFF" stroke-width="0.65">'
            f'<title>{esc(tooltip)}</title></rect>'
        )

    for span in run.spans:
        if span.dispatch_s > time_max_s:
            continue
        tooltip = (
            f"R{span.request_index}: {span.input_tokens} prompt, "
            f"{span.output_tokens} output; {span.service_s:.3f}s service"
        )
        add_phase(span.dispatch_s, span.prefill_end_s, PREFILL, tooltip + "; prefill")
        add_phase(span.prefill_end_s, span.decode_end_s, DECODE, tooltip + "; decode")
        if zoom and span.request_index in (0, 1):
            visible_start = max(span.dispatch_s, 0)
            visible_end = min(span.completion_s, time_max_s)
            if visible_end - visible_start > 18:
                label = f"R{span.request_index}"
                if span.request_index == 0:
                    label += f": {span.service_s:.1f}s"
                output.append(text(
                    (sx(visible_start) + sx(visible_end)) / 2,
                    y + 4,
                    label,
                    "bar-label",
                    "middle",
                ))
    if run.reset_s is not None and run.reset_s <= time_max_s:
        partial_start = run.finish_s
        output.append(rect(
            sx(partial_start),
            y - 13,
            sx(run.reset_s) - sx(partial_start),
            bar_height - 2,
            "#FFFFFF",
            1,
            FAILURE,
            1.5,
        ))
        x = sx(run.reset_s)
        output.append(line(x - 7, y - 20, x + 7, y + 20, FAILURE, 3))
        output.append(line(x + 7, y - 20, x - 7, y + 20, FAILURE, 3))


def render(runs: tuple[Run, ...], specs: dict[int, dict], analysis: dict) -> str:
    arrival_values = [row["arrival_us"] / 1e6 for row in specs.values()]
    arrival_start_s = min(arrival_values)
    arrival_end_s = max(arrival_values)
    no_hot = analysis["no_hot_cpu_to_op15"]["makespan"]
    cpu_hot_delta = analysis["cpu_hot_effect"]["service_s"]["delta_pct"]
    prefix_hot_delta = analysis["matched_14_request_prefix"]["op15_hot_effect"]["service_s"]["delta_pct"]

    output = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
            f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" '
            'role="img" aria-labelledby="title description">'
        ),
        '<title id="title">Real BurstGPT cold inference timeline comparison</title>',
        (
            '<desc id="description">Aligned prefill and decode timelines for '
            'desktop CPU and CPU plus OP15, with and without a concurrent Qwen '
            'hot-model trace.</desc>'
        ),
        "<style>",
        "text { font-family: Inter, DejaVu Sans, Arial, sans-serif; }",
        ".title { font-size: 28px; font-weight: 700; fill: #203044; }",
        ".subtitle { font-size: 14px; fill: #5D6D7E; }",
        ".panel-title { font-size: 18px; font-weight: 700; fill: #203044; }",
        ".panel-subtitle { font-size: 13px; fill: #5D6D7E; }",
        ".lane-label { font-size: 14px; font-weight: 700; fill: #203044; }",
        ".lane-detail { font-size: 11px; fill: #5D6D7E; }",
        ".tick { font-size: 11px; fill: #5D6D7E; }",
        ".axis { font-size: 12px; font-style: italic; fill: #46576A; }",
        ".legend { font-size: 12px; fill: #46576A; }",
        ".bar-label { font-size: 12px; font-weight: 700; fill: #FFFFFF; }",
        ".duration-label { font-size: 11px; font-weight: 700; fill: #203044; }",
        ".callout { font-size: 13px; font-weight: 700; fill: #203044; }",
        ".note { font-size: 12px; fill: #5D6D7E; }",
        ".failure { font-size: 12px; font-weight: 700; fill: #B8323C; }",
        "</style>",
        rect(0, 0, WIDTH, HEIGHT, BACKGROUND),
        text(WIDTH / 2, 39, "Real BurstGPT timeline: Gemma cold inference", "title", "middle"),
        text(
            WIDTH / 2,
            65,
            "RTX 4060 Ti runs Qwen hot load; desktop CPU and one OP15 run Gemma 4 12B Q4_0; default clocks",
            "subtitle",
            "middle",
        ),
    ]

    legend_y = 98
    legend_items = (
        (PREFILL, "Gemma prefill"),
        (DECODE, "Gemma decode"),
        (HOT, "Qwen executing"),
        (ARRIVALS, "trace arrival window"),
    )
    legend_x = 420
    for color, label in legend_items:
        output.append(rect(legend_x, legend_y - 11, 22, 13, color, 0.88, radius=2))
        output.append(text(legend_x + 30, legend_y, label, "legend"))
        legend_x += 235
    output.append(line(legend_x, legend_y - 8, legend_x + 16, legend_y + 8, FAILURE, 2.5))
    output.append(line(legend_x + 16, legend_y - 8, legend_x, legend_y + 8, FAILURE, 2.5))
    output.append(text(legend_x + 26, legend_y, "USB reset / censored", "legend"))

    full_sx, _, _ = add_panel_background(
        output,
        122,
        600,
        2200,
        300,
        "A. Full trace - completion time is directly comparable",
        (
            f"CPU + OP15 finishes {runs[0].finish_s - runs[2].finish_s:.1f}s earlier "
            f"({no_hot['reduction_pct']:.1f}% less makespan); all 74 arrivals occur in the shaded burst."
        ),
        arrival_start_s,
        arrival_end_s,
    )
    full_lane_y = (244, 322, 400, 478)
    for run, y in zip(runs, full_lane_y):
        add_lane(output, run, y, full_sx, 2200, False)

    output.append(line(
        full_sx(runs[2].finish_s),
        535,
        full_sx(runs[0].finish_s),
        535,
        DECODE,
        2,
    ))
    output.append(line(full_sx(runs[2].finish_s), 527, full_sx(runs[2].finish_s), 543, DECODE, 2))
    output.append(line(full_sx(runs[0].finish_s), 527, full_sx(runs[0].finish_s), 543, DECODE, 2))
    output.append(text(
        (full_sx(runs[2].finish_s) + full_sx(runs[0].finish_s)) / 2,
        526,
        f"{runs[0].finish_s - runs[2].finish_s:.1f}s saved",
        "callout",
        "middle",
    ))
    reset_run = runs[3]
    require(reset_run.reset_s is not None, "reset marker")
    output.append(text(
        full_sx(reset_run.reset_s) + 13,
        full_lane_y[3] - 22,
        f"reset during R66 at about {reset_run.reset_s:.1f}s",
        "failure",
    ))

    zoom_sx, _, _ = add_panel_background(
        output,
        620,
        1028,
        180,
        30,
        "B. First 180 seconds - Qwen overlap and phase detail",
        (
            f"Qwen changes full CPU service by {cpu_hot_delta:+.2f}%; before reset, "
            f"it changes matched CPU + OP15 service by {prefix_hot_delta:+.2f}%."
        ),
        arrival_start_s,
        arrival_end_s,
    )
    zoom_lane_y = (735, 802, 869, 936)
    for run, y in zip(runs, zoom_lane_y):
        add_lane(output, run, y, zoom_sx, 180, True)

    output.append(text(
        52,
        1061,
        "Bars start at cold-request dispatch. Green is prefill; blue is decode. White dividers are request boundaries.",
        "note",
    ))
    output.append(text(
        52,
        1083,
        "The CPU + OP15 hot arm is valid through request 65. Failed request 66 and all post-reset timings are excluded.",
        "note",
    ))
    output.append(text(
        WIDTH - 52,
        1083,
        "Hot trace: 57 Qwen requests; cold trace: 17 Gemma requests",
        "note",
        "end",
    ))
    output.append("</svg>")
    return "\n".join(output) + "\n"


def main() -> int:
    specs = load_request_specs()
    analysis = json.loads(ANALYSIS.read_text())
    cpu_hot = analysis["hot_service_windows"]["cpu_hot_trace"]
    runs = [
        load_complete_run(
            "cpu_hot_absent",
            "CPU | GPU absent",
            "17/17 complete; finish 2131.2s",
            specs,
        ),
        load_complete_run(
            "cpu_hot_trace",
            "CPU | Qwen trace",
            "17/17 complete; finish 2136.8s",
            specs,
            (cpu_hot["start_s"], cpu_hot["end_s"]),
        ),
        load_complete_run(
            "op15_hot_absent",
            "CPU + OP15 | GPU absent",
            "17/17 complete; finish 1767.3s",
            specs,
        ),
    ]
    runs.append(load_censored_run(specs, analysis, runs[2]))
    SVG.write_text(render(tuple(runs), specs, analysis), encoding="ascii")

    converter = shutil.which("convert")
    if converter:
        subprocess.run(
            [
                converter,
                "-background",
                "white",
                "-density",
                "150",
                str(SVG),
                str(PNG),
            ],
            check=True,
        )
    print(SVG)
    if PNG.exists():
        print(PNG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
