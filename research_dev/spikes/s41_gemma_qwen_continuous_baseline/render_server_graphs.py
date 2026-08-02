#!/usr/bin/env python3
"""Render paper-readable S41 server-only comparison figures."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from html import escape
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Callable

import cairosvg


SCHEMA = "s41-server-baseline-normalized-v1"
COMPLETE_ENERGY_SCOPE = "SELECTED_GPU_BOARD_COMPLETE_RUN"
PREFIX_ENERGY_SCOPE = "SELECTED_GPU_BOARD_BRACKETED_PREFIX"
WIDTH = 1240
HEIGHT = 650
COLORS = ("#0072b2", "#d55e00")
GOODPUT = "#009e73"
LATENCY_COLORS = ("#0072b2", "#d55e00", "#7b2cbf")
GRID = "#d9dde3"
TEXT = "#20242a"
FAIL = "#b91c1c"


class GraphError(RuntimeError):
    pass


def require_int(value: Any, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise GraphError(f"{field}: expected integer >= {minimum}")
    return value


def require_string(value: Any, field: str) -> str:
    if type(value) is not str or not value:
        raise GraphError(f"{field}: expected non-empty string")
    return value


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GraphError(f"{path}: invalid JSON") from error
    if type(value) is not dict:
        raise GraphError(f"{path}: expected object")
    return value


def _require_nullable_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    return require_int(value, field)


def validate_summary(summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("schema") != SCHEMA:
        raise GraphError("summary.schema: unsupported")
    if summary.get("energy_claim") != (
            "Selected GPU board only; not server-wall or "
            "total-system energy."):
        raise GraphError("summary.energy_claim: precise scope is required")
    models = summary.get("models")
    if not isinstance(models, list) or len(models) != 2:
        raise GraphError("summary.models: exactly two models required")
    model_ids = []
    for index, model in enumerate(models):
        if type(model) is not dict:
            raise GraphError(f"summary.models[{index}]: expected object")
        model_ids.append(require_string(
            model.get("id"), f"summary.models[{index}].id"))
        require_string(model.get("label"), f"summary.models[{index}].label")
    if len(set(model_ids)) != 2:
        raise GraphError("summary.models: duplicate id")
    runs = summary.get("runs")
    if not isinstance(runs, list) or not runs:
        raise GraphError("summary.runs: expected non-empty array")
    seen: set[tuple[str, int, str]] = set()
    for index, run in enumerate(runs):
        field = f"summary.runs[{index}]"
        if type(run) is not dict:
            raise GraphError(f"{field}: expected object")
        mode = require_string(run.get("mode"), f"{field}.mode")
        repeat = require_int(run.get("repeat_index"), f"{field}.repeat_index")
        cache = require_string(
            run.get("cache_regime"), f"{field}.cache_regime")
        identity = (mode, repeat, cache)
        if identity in seen:
            raise GraphError(f"{field}: duplicate run identity")
        seen.add(identity)
        require_string(run.get("label"), f"{field}.label")
        require_string(run.get("verdict"), f"{field}.verdict")
        offered = require_int(
            run.get("offered_request_count"), f"{field}.offered_request_count",
            1)
        completed = require_int(
            run.get("completed_request_count"),
            f"{field}.completed_request_count")
        stranded = require_int(
            run.get("stranded_request_count"),
            f"{field}.stranded_request_count")
        if completed + stranded != offered:
            raise GraphError(f"{field}: request conservation mismatch")
        output_tokens = require_int(
            run.get("completed_output_tokens"),
            f"{field}.completed_output_tokens")
        if (completed == 0) != (output_tokens == 0):
            raise GraphError(f"{field}: completed-token mismatch")
        require_int(run.get("paid_ns"), f"{field}.paid_ns", 1)
        require_int(
            run.get("slo_goodput_milli_rps"),
            f"{field}.slo_goodput_milli_rps")
        require_int(run.get("slo_met_count"), f"{field}.slo_met_count")
        counts = run.get("model_completed_request_counts")
        throughput = run.get("model_throughput_milli_tps")
        if type(counts) is not dict or set(counts) != set(model_ids):
            raise GraphError(f"{field}: model count key mismatch")
        if type(throughput) is not dict or set(throughput) != set(model_ids):
            raise GraphError(f"{field}: throughput key mismatch")
        for model_id in model_ids:
            require_int(counts[model_id], f"{field}.counts.{model_id}")
            require_int(
                throughput[model_id], f"{field}.throughput.{model_id}")
        if sum(counts.values()) != completed:
            raise GraphError(f"{field}: model completion count mismatch")
        ttft = _require_nullable_int(run.get("ttft_p95_ns"),
                                     f"{field}.ttft_p95_ns")
        completion = _require_nullable_int(
            run.get("completion_latency_p95_ns"),
            f"{field}.completion_latency_p95_ns")
        if completed and (ttft is None or completion is None):
            raise GraphError(f"{field}: completed run lacks tail latency")
        gap = _require_nullable_int(
            run.get("maximum_publication_gap_ns"),
            f"{field}.maximum_publication_gap_ns")
        censored = run.get("publication_gap_censored")
        if type(censored) is not bool:
            raise GraphError(f"{field}.publication_gap_censored: expected bool")
        timeline = run.get("timeline")
        if type(timeline) is not dict:
            raise GraphError(f"{field}.timeline: expected object")
        switches = timeline.get("switches")
        if not isinstance(switches, list):
            raise GraphError(f"{field}.timeline.switches: expected array")
        if switches and gap is None and not censored:
            raise GraphError(f"{field}: switching run lacks publication gap")
        if not switches and (gap is not None or censored):
            raise GraphError(f"{field}: invalid no-switch publication metric")
        _validate_timeline(timeline, model_ids, field)
        energy = require_int(
            run.get("selected_gpu_board_energy_nj"),
            f"{field}.selected_gpu_board_energy_nj", 1)
        scope = run.get("gpu_energy_scope")
        eligible = run.get("energy_equal_work_eligible")
        lower = run.get("energy_is_lower_bound")
        if type(eligible) is not bool or type(lower) is not bool:
            raise GraphError(f"{field}: invalid energy eligibility flags")
        expected_scope = (
            COMPLETE_ENERGY_SCOPE if eligible else PREFIX_ENERGY_SCOPE)
        if scope != expected_scope or lower == eligible:
            raise GraphError(f"{field}: energy scope/eligibility mismatch")
        energy_rows = timeline.get("energy")
        if not isinstance(energy_rows, list) or not energy_rows:
            raise GraphError(f"{field}.timeline.energy: empty")
        last_energy = require_int(
            energy_rows[-1].get(
                "cumulative_selected_gpu_board_energy_nj"),
            f"{field}.timeline.energy[-1]")
        if last_energy != energy:
            raise GraphError(f"{field}: energy timeline/total mismatch")
        _require_nullable_int(
            run.get("peak_server_rss_bytes"),
            f"{field}.peak_server_rss_bytes")
        _require_nullable_int(
            run.get("server_cpu_utilization_milli_pct_p50"),
            f"{field}.server_cpu_utilization_milli_pct_p50")
    equal_work = {
        (run["completed_request_count"], run["completed_output_tokens"])
        for run in runs if run["energy_equal_work_eligible"]
    }
    if len(equal_work) > 1:
        raise GraphError(
            "summary.runs: equal-work energy rows have different work")
    return summary


def _validate_timeline(
    timeline: dict[str, Any],
    model_ids: list[str],
    field: str,
) -> None:
    require_int(timeline.get("bin_width_ns"),
                f"{field}.timeline.bin_width_ns", 1)
    throughput = timeline.get("throughput")
    resources = timeline.get("resources")
    loads = timeline.get("loads")
    if not isinstance(throughput, list) or not throughput:
        raise GraphError(f"{field}.timeline.throughput: empty")
    if not isinstance(resources, list) or not resources:
        raise GraphError(f"{field}.timeline.resources: empty")
    if not isinstance(loads, list):
        raise GraphError(f"{field}.timeline.loads: expected array")
    last_t = -1
    for index, row in enumerate(throughput):
        if type(row) is not dict:
            raise GraphError(f"{field}.throughput[{index}]: expected object")
        t_ns = require_int(
            row.get("t_offset_ns"), f"{field}.throughput[{index}].t", 1)
        if t_ns <= last_t:
            raise GraphError(f"{field}.throughput: non-monotonic")
        last_t = t_ns
        values = row.get("model_milli_tokens_per_second")
        if type(values) is not dict or set(values) != set(model_ids):
            raise GraphError(f"{field}.throughput[{index}]: model mismatch")
        for model_id in model_ids:
            require_int(
                values[model_id],
                f"{field}.throughput[{index}].{model_id}")
    for index, row in enumerate(resources):
        require_int(row.get("t_offset_ns"),
                    f"{field}.resources[{index}].t")
        require_int(row.get("server_rss_bytes"),
                    f"{field}.resources[{index}].rss")
        if "cpu_utilization_milli_pct" in row:
            require_int(row["cpu_utilization_milli_pct"],
                        f"{field}.resources[{index}].cpu")
    for index, row in enumerate(loads):
        if row.get("model_id") not in model_ids:
            raise GraphError(f"{field}.loads[{index}]: model mismatch")
        require_int(row.get("start_t_offset_ns"),
                    f"{field}.loads[{index}].start")
        if row.get("end_t_offset_ns") is not None:
            require_int(row["end_t_offset_ns"],
                        f"{field}.loads[{index}].end")


def _modes(runs: list[dict[str, Any]]) -> OrderedDict[str, list[dict[str, Any]]]:
    result: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for run in runs:
        result.setdefault(run["mode"], []).append(run)
    return result


def _median(values: list[float]) -> float:
    if not values:
        raise GraphError("median: no values")
    return float(statistics.median(values))


def _begin(title: str, subtitle: str = "") -> list[str]:
    lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
            f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">'
        ),
        "<style>text{font-family:Arial,sans-serif;fill:#20242a}"
        ".title{font-size:22px;font-weight:700}"
        ".subtitle{font-size:13px}.axis{font-size:12px}"
        ".small{font-size:11px}.mode{font-size:11px;font-weight:600}"
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        (
            f'<text class="title" x="{WIDTH // 2}" y="30" '
            f'text-anchor="middle">{escape(title)}</text>'
        ),
    ]
    if subtitle:
        lines.append(
            f'<text class="subtitle" x="{WIDTH // 2}" y="51" '
            f'text-anchor="middle">{escape(subtitle)}</text>'
        )
    return lines


def _finish(lines: list[str]) -> str:
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _panel_axis(
    lines: list[str],
    x: int,
    y: int,
    width: int,
    height: int,
    maximum: float,
    title: str,
    y_label: str,
) -> None:
    maximum = max(maximum, 1e-9)
    if title:
        lines.append(
            f'<text x="{x + width // 2}" y="{y - 23}" '
            f'text-anchor="middle" font-size="16" font-weight="700">'
            f'{escape(title)}</text>')
    for tick in range(5):
        py = y + height - height * tick / 4
        value = maximum * tick / 4
        lines.append(
            f'<line x1="{x}" y1="{py:.1f}" x2="{x + width}" '
            f'y2="{py:.1f}" stroke="{GRID}"/>')
        lines.append(
            f'<text class="axis" x="{x - 8}" y="{py + 4:.1f}" '
            f'text-anchor="end">{value:.2f}</text>')
    lines.extend([
        f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + height}" '
        f'stroke="{TEXT}"/>',
        f'<line x1="{x}" y1="{y + height}" x2="{x + width}" '
        f'y2="{y + height}" stroke="{TEXT}"/>',
        f'<text class="axis" x="{x - 48}" y="{y + height // 2}" '
        f'text-anchor="middle" transform="rotate(-90 {x - 48} '
        f'{y + height // 2})">{escape(y_label)}</text>',
    ])


def _x_labels(
    lines: list[str],
    modes: list[str],
    x: int,
    y: int,
    width: int,
) -> None:
    short_labels = {
        "C1_GPU_SWITCH_COLD": ("C1 cold",),
        "C1_GPU_SWITCH_WARM": ("C1 warm",),
        "C2_GEMMA_GPU_QWEN_CPU": ("C2 Gemma", "GPU"),
        "C2_QWEN_GPU_GEMMA_CPU": ("C2 Qwen", "GPU"),
    }
    step = width / len(modes)
    for index, mode in enumerate(modes):
        px = x + step * (index + 0.5)
        labels = short_labels.get(mode, (mode.replace("_", " "),))
        label_lines = tuple(
            label[:22] + ".." if len(label) > 24 else label
            for label in labels
        )
        body = "".join(
            f'<tspan x="{px:.1f}" dy="{0 if line_index == 0 else 13}">'
            f'{escape(label)}</tspan>'
            for line_index, label in enumerate(label_lines)
        )
        lines.append(
            f'<text class="mode" x="{px:.1f}" y="{y}" '
            f'text-anchor="middle">{body}</text>')


def throughput_comparison_svg(summary: dict[str, Any]) -> str:
    models = summary["models"]
    groups = _modes(summary["runs"])
    modes = list(groups)
    goodput_max = max(
        run["slo_goodput_milli_rps"] / 1000
        for run in summary["runs"]) * 1.15
    throughput_max = max(
        sum(run["model_throughput_milli_tps"].values()) / 1000
        for run in summary["runs"]) * 1.15
    lines = _begin(
        "Server-only SLO goodput and per-model output throughput",
        "Thin bars are repetitions; outlined bars are medians.",
    )
    panels = [
        (90, 105, 470, 415, goodput_max, "SLO goodput", "requests/s"),
        (690, 105, 470, 415, throughput_max,
         "Per-model output throughput", "output tokens/s"),
    ]
    for panel in panels:
        _panel_axis(lines, *panel)
    for panel_index, (x, y, width, height, maximum, _, _) in enumerate(panels):
        step = width / len(modes)
        for mode_index, mode in enumerate(modes):
            runs = groups[mode]
            center = x + step * (mode_index + 0.5)
            spread = min(14, step / max(5, len(runs) + 2))
            if panel_index == 0:
                values = [
                    run["slo_goodput_milli_rps"] / 1000 for run in runs]
                for rep_index, value in enumerate(values):
                    px = center + (rep_index - (len(values) - 1) / 2) * spread
                    ph = value / maximum * height
                    lines.append(
                        f'<rect x="{px - 4:.1f}" y="{y + height - ph:.1f}" '
                        f'width="8" height="{ph:.1f}" fill="{GOODPUT}" '
                        'fill-opacity="0.42"/>')
                median = _median(values)
                ph = median / maximum * height
                lines.append(
                    f'<rect x="{center - 16:.1f}" y="{y + height - ph:.1f}" '
                    f'width="32" height="{ph:.1f}" fill="none" '
                    f'stroke="{GOODPUT}" stroke-width="3"/>')
            else:
                for rep_index, run in enumerate(runs):
                    px = center + (rep_index - (len(runs) - 1) / 2) * spread
                    base = y + height
                    for model_index, model in enumerate(models):
                        value = (
                            run["model_throughput_milli_tps"][model["id"]]
                            / 1000
                        )
                        ph = value / maximum * height
                        base -= ph
                        lines.append(
                            f'<rect x="{px - 4:.1f}" y="{base:.1f}" '
                            f'width="8" height="{ph:.1f}" '
                            f'fill="{COLORS[model_index]}" '
                            'fill-opacity="0.42"/>')
                base = y + height
                for model_index, model in enumerate(models):
                    values = [
                        run["model_throughput_milli_tps"][model["id"]] / 1000
                        for run in runs
                    ]
                    ph = _median(values) / maximum * height
                    base -= ph
                    lines.append(
                        f'<rect x="{center - 16:.1f}" y="{base:.1f}" '
                        f'width="32" height="{ph:.1f}" fill="none" '
                        f'stroke="{COLORS[model_index]}" stroke-width="3"/>')
            stranded = max(run["stranded_request_count"] for run in runs)
            if stranded:
                lines.append(
                    f'<text class="small" x="{center:.1f}" y="{y + 18}" '
                    f'text-anchor="middle" style="fill:{FAIL};font-weight:700">'
                    f'{stranded} stranded</text>')
            elif any(run["verdict"].startswith("RESOURCE_FAIL")
                     for run in runs):
                lines.append(
                    f'<text class="small" x="{center:.1f}" y="{y + 18}" '
                    f'text-anchor="middle" style="fill:{FAIL};font-weight:700">'
                    'swap gate failed</text>')
        _x_labels(lines, modes, x, y + height + 24, width)
    legend_x = 690
    for index, model in enumerate(models):
        lines.append(
            f'<rect x="{legend_x + index * 245}" y="568" width="14" '
            f'height="14" fill="{COLORS[index]}"/>')
        lines.append(
            f'<text class="axis" x="{legend_x + 20 + index * 245}" y="580">'
            f'{escape(model["label"])}</text>')
    lines.append(
        f'<rect x="100" y="568" width="14" height="14" fill="{GOODPUT}"/>')
    lines.append(
        '<text class="axis" x="120" y="580">SLO-met requests</text>')
    lines.append(
        '<text class="small" x="620" y="626" text-anchor="middle">'
        'A stranded cold run is a failed run, not a lower-throughput success.'
        '</text>')
    return _finish(lines)


def latency_comparison_svg(summary: dict[str, Any]) -> str:
    groups = _modes(summary["runs"])
    modes = list(groups)
    metrics: list[tuple[str, str, str]] = [
        ("ttft_p95_ns", "P95 TTFT", "seconds"),
        ("completion_latency_p95_ns", "P95 completion", "seconds"),
        ("maximum_publication_gap_ns", "Maximum publication gap", "seconds"),
    ]
    maxima = []
    for field, _, _ in metrics:
        values = [
            run[field] / 1e9 for run in summary["runs"]
            if run[field] is not None
        ]
        maxima.append(max(values, default=1.0) * 1.15)
    lines = _begin(
        "Server-only tail latency and model publication delay",
        "Dots are repetitions; thick horizontal marks are medians.",
    )
    panel_width = 330
    for metric_index, ((field, title, unit), maximum) in enumerate(
            zip(metrics, maxima)):
        x = 72 + metric_index * 397
        y = 110
        height = 400
        _panel_axis(lines, x, y, panel_width, height, maximum, title, unit)
        step = panel_width / len(modes)
        for mode_index, mode in enumerate(modes):
            runs = groups[mode]
            center = x + step * (mode_index + 0.5)
            values = []
            for rep_index, run in enumerate(runs):
                value = run[field]
                if value is None:
                    continue
                seconds = value / 1e9
                values.append(seconds)
                px = center + (rep_index - (len(runs) - 1) / 2) * 8
                py = y + height - seconds / maximum * height
                color = FAIL if run["publication_gap_censored"] \
                    and field == "maximum_publication_gap_ns" \
                    else LATENCY_COLORS[metric_index]
                lines.append(
                    f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" '
                    f'fill="{color}" fill-opacity="0.72"/>')
            if values:
                median = _median(values)
                py = y + height - median / maximum * height
                lines.append(
                    f'<line x1="{center - 16:.1f}" y1="{py:.1f}" '
                    f'x2="{center + 16:.1f}" y2="{py:.1f}" '
                    f'stroke="{LATENCY_COLORS[metric_index]}" '
                    'stroke-width="4"/>')
            elif field == "maximum_publication_gap_ns":
                note = (
                    "censored" if any(
                        run["publication_gap_censored"] for run in runs)
                    else "N/A"
                )
                color = FAIL if note == "censored" else TEXT
                lines.append(
                    f'<text class="small" x="{center:.1f}" y="{y + 22}" '
                    f'text-anchor="middle" style="fill:{color}">'
                    f'{note}</text>')
        _x_labels(lines, modes, x, y + height + 24, panel_width)
    lines.append(
        '<text class="small" x="620" y="588" text-anchor="middle">'
        'Publication gap is N/A for modes with no model transition; '
        'incomplete transitions are marked censored.</text>')
    stranded_runs = [
        run for run in summary["runs"] if run["stranded_request_count"]
    ]
    if stranded_runs:
        completed_values = {
            run["completed_request_count"] for run in stranded_runs
        }
        stranded_values = {
            run["stranded_request_count"] for run in stranded_runs
        }
        if len(completed_values) == 1 and len(stranded_values) == 1:
            qualifier = (
                f'Failed-run P95 covers {completed_values.pop()} completions '
                f'only; {stranded_values.pop()} requests stranded.'
            )
        else:
            qualifier = (
                'Failed-run P95 covers completed requests only; '
                'stranded work is excluded.'
            )
        lines.append(
            f'<text class="small" x="620" y="610" text-anchor="middle" '
            f'style="fill:{FAIL};font-weight:700">'
            f'{qualifier}</text>')
    if any(run["verdict"].startswith("RESOURCE_FAIL")
           for run in summary["runs"]):
        lines.append(
            f'<text class="small" x="620" y="630" text-anchor="middle" '
            f'style="fill:{FAIL};font-weight:700">'
            'C2 Gemma GPU is latency-complete but failed the zero-swap gate.'
            '</text>')
    return _finish(lines)


def _representatives(
    runs: list[dict[str, Any]],
    requested_modes: list[str] | None,
) -> list[dict[str, Any]]:
    cells: OrderedDict[tuple[str, str], list[dict[str, Any]]] = OrderedDict()
    for run in runs:
        if requested_modes and run["mode"] not in requested_modes:
            continue
        cells.setdefault((run["mode"], run["cache_regime"]), []).append(run)
    if requested_modes:
        missing = set(requested_modes) - {key[0] for key in cells}
        if missing:
            raise GraphError(
                "timeline mode missing: " + ", ".join(sorted(missing)))
    result = []
    for cell_runs in cells.values():
        median = _median([
            run["slo_goodput_milli_rps"] for run in cell_runs])
        result.append(min(
            cell_runs,
            key=lambda run: (
                abs(run["slo_goodput_milli_rps"] - median),
                run["repeat_index"],
            ),
        ))
    return result


def timeline_svg(
    summary: dict[str, Any],
    run: dict[str, Any],
) -> str:
    models = summary["models"]
    rows = run["timeline"]["throughput"]
    x_max = max(row["t_offset_ns"] for row in rows)
    y_max = max(
        value for row in rows
        for value in row["model_milli_tokens_per_second"].values()
    ) / 1000
    y_max = max(1.0, y_max * 1.15)
    x, y, width, height = 88, 105, 1080, 430
    lines = _begin(
        f'Throughput timeline: {run["label"]}',
        f'{run["mode"]} | {run["cache_regime"]} | repeat '
        f'{run["repeat_index"]}',
    )
    _panel_axis(lines, x, y, width, height, y_max, "", "output tokens/s")
    for tick in range(6):
        px = x + width * tick / 5
        seconds = x_max / 1e9 * tick / 5
        lines.append(
            f'<line x1="{px:.1f}" y1="{y}" x2="{px:.1f}" '
            f'y2="{y + height}" stroke="{GRID}"/>')
        lines.append(
            f'<text class="axis" x="{px:.1f}" y="{y + height + 22}" '
            f'text-anchor="middle">{seconds:.1f}</text>')
    lines.append(
        f'<text class="axis" x="{x + width // 2}" y="{y + height + 45}" '
        'text-anchor="middle">time since paid start (s)</text>')
    for load in run["timeline"]["loads"]:
        start = load["start_t_offset_ns"]
        end = load["end_t_offset_ns"]
        if end is None:
            end = x_max
        model_index = next(
            index for index, model in enumerate(models)
            if model["id"] == load["model_id"])
        px = x + start / x_max * width
        pw = max(2, (end - start) / x_max * width)
        lines.append(
            f'<rect x="{px:.1f}" y="{y}" width="{pw:.1f}" '
            f'height="{height}" fill="{COLORS[model_index]}" '
            'fill-opacity="0.10">'
            f'<title>load {escape(models[model_index]["label"])}</title>'
            '</rect>')
    for switch in run["timeline"]["switches"]:
        px = x + switch["intent_t_offset_ns"] / x_max * width
        lines.append(
            f'<line x1="{px:.1f}" y1="{y}" x2="{px:.1f}" '
            f'y2="{y + height}" stroke="#666" stroke-dasharray="5,4">'
            '<title>switch intent</title></line>')
        publication = switch["publication_t_offset_ns"]
        if publication is not None:
            px = x + publication / x_max * width
            lines.append(
                f'<line x1="{px:.1f}" y1="{y}" x2="{px:.1f}" '
                f'y2="{y + height}" stroke="{GOODPUT}" '
                'stroke-dasharray="2,3">'
                '<title>model publication</title></line>')
    for model_index, model in enumerate(models):
        points = []
        for row in rows:
            px = x + row["t_offset_ns"] / x_max * width
            value = row["model_milli_tokens_per_second"][model["id"]] / 1000
            py = y + height - value / y_max * height
            points.append(f"{px:.1f},{py:.1f}")
        lines.append(
            f'<polyline points="{" ".join(points)}" fill="none" '
            f'stroke="{COLORS[model_index]}" stroke-width="3"/>')
        lines.append(
            f'<line x1="{100 + model_index * 300}" y1="75" '
            f'x2="{132 + model_index * 300}" y2="75" '
            f'stroke="{COLORS[model_index]}" stroke-width="4"/>')
        lines.append(
            f'<text class="axis" x="{140 + model_index * 300}" y="79">'
            f'{escape(model["label"])}</text>')
    lines.append(
        '<line x1="710" y1="75" x2="742" y2="75" stroke="#666" '
        'stroke-dasharray="5,4"/>')
    lines.append(
        '<text class="axis" x="750" y="79">intent</text>')
    lines.append(
        f'<line x1="835" y1="75" x2="867" y2="75" stroke="{GOODPUT}" '
        'stroke-dasharray="2,3"/>')
    lines.append(
        '<text class="axis" x="875" y="79">publication</text>')
    failure_note = None
    if run["stranded_request_count"]:
        failure_note = (
            f'FAILED: {run["stranded_request_count"]} requests stranded'
        )
    elif run["verdict"].startswith("RESOURCE_FAIL"):
        failure_note = "INVALID CONTROL: zero-swap gate failed"
    if failure_note is not None:
        lines.append(
            f'<rect x="88" y="595" width="1080" height="34" '
            f'fill="{FAIL}" fill-opacity="0.10"/>')
        lines.append(
            f'<text x="628" y="618" text-anchor="middle" '
            f'style="fill:{FAIL};font-weight:700">'
            f'{failure_note}</text>')
    return _finish(lines)


def energy_comparison_svg(summary: dict[str, Any]) -> str:
    groups = _modes(summary["runs"])
    modes = list(groups)
    eligible_values = [
        run["selected_gpu_board_energy_nj"] / 1e9
        / run["completed_output_tokens"]
        for run in summary["runs"]
        if run["energy_equal_work_eligible"]
    ]
    if not eligible_values:
        raise GraphError("energy graph: no equal-work eligible run")
    maximum = max(eligible_values) * 1.15
    x, y, width, height = 110, 105, 1040, 420
    lines = _begin(
        "Selected-GPU board energy for equal completed work",
        "Only complete runs are normalized; partial cold prefixes are excluded.",
    )
    _panel_axis(lines, x, y, width, height, maximum,
                "Selected-GPU board energy", "J/completed output token")
    step = width / len(modes)
    for mode_index, mode in enumerate(modes):
        runs = groups[mode]
        center = x + step * (mode_index + 0.5)
        values = []
        for rep_index, run in enumerate(runs):
            if not run["energy_equal_work_eligible"]:
                continue
            value = (
                run["selected_gpu_board_energy_nj"] / 1e9
                / run["completed_output_tokens"]
            )
            values.append(value)
            px = center + (rep_index - (len(runs) - 1) / 2) * 10
            py = y + height - value / maximum * height
            lines.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" '
                f'fill="{GOODPUT}" fill-opacity="0.72"/>')
        if values:
            py = y + height - _median(values) / maximum * height
            lines.append(
                f'<line x1="{center - 20:.1f}" y1="{py:.1f}" '
                f'x2="{center + 20:.1f}" y2="{py:.1f}" '
                f'stroke="{GOODPUT}" stroke-width="4"/>')
        excluded = sum(
            not run["energy_equal_work_eligible"] for run in runs)
        if excluded:
            lines.append(
                f'<text class="small" x="{center:.1f}" y="{y + 20}" '
                f'text-anchor="middle" style="fill:{FAIL}">'
                f'{excluded} partial run{"s" if excluded != 1 else ""} '
                'excluded</text>')
        elif any(run["verdict"].startswith("RESOURCE_FAIL")
                 for run in runs):
            lines.append(
                f'<text class="small" x="{center:.1f}" y="{y + 20}" '
                f'text-anchor="middle" style="fill:{FAIL};font-weight:700">'
                'swap gate failed</text>')
    _x_labels(lines, modes, x, y + height + 24, width)
    lines.append(
        '<text class="small" x="620" y="612" text-anchor="middle" '
        'style="font-weight:700">Selected GPU board only; not server-wall '
        'or total-system energy.</text>')
    return _finish(lines)


def resources_comparison_svg(summary: dict[str, Any]) -> str:
    groups = _modes(summary["runs"])
    modes = list(groups)
    rss_values = [
        run["peak_server_rss_bytes"] / (1024 ** 3)
        for run in summary["runs"] if run["peak_server_rss_bytes"] is not None
    ]
    rss_max = max(rss_values, default=1.0) * 1.15
    cpu_values = [
        run["server_cpu_utilization_milli_pct_p50"] / 1000
        for run in summary["runs"]
        if run["server_cpu_utilization_milli_pct_p50"] is not None
    ]
    lines = _begin(
        "Sampled process memory and CPU utilization",
        "Dual-route RSS is not a process sum; CPU appears only when sampled.",
    )
    panels = [
        (90, 105, 470, 415, rss_max, "Peak sampled-process RSS", "GiB")
    ]
    if cpu_values:
        panels.append(
            (690, 105, 470, 415, max(cpu_values) * 1.15,
             "Median server CPU utilization", "percent"))
    for panel in panels:
        _panel_axis(lines, *panel)
    for panel_index, (x, y, width, height, maximum, _, _) in enumerate(panels):
        step = width / len(modes)
        for mode_index, mode in enumerate(modes):
            runs = groups[mode]
            center = x + step * (mode_index + 0.5)
            values = []
            for rep_index, run in enumerate(runs):
                if panel_index == 0:
                    raw = run["peak_server_rss_bytes"]
                    if raw is None:
                        continue
                    value = raw / (1024 ** 3)
                else:
                    raw = run["server_cpu_utilization_milli_pct_p50"]
                    if raw is None:
                        continue
                    value = raw / 1000
                values.append(value)
                px = center + (rep_index - (len(runs) - 1) / 2) * 10
                py = y + height - value / maximum * height
                lines.append(
                    f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" '
                    f'fill="{COLORS[panel_index]}" fill-opacity="0.72"/>')
            if values:
                py = y + height - _median(values) / maximum * height
                lines.append(
                    f'<line x1="{center - 18:.1f}" y1="{py:.1f}" '
                    f'x2="{center + 18:.1f}" y2="{py:.1f}" '
                    f'stroke="{COLORS[panel_index]}" stroke-width="4"/>')
            elif panel_index in (0, 1):
                lines.append(
                    f'<text class="small" x="{center:.1f}" y="{y + 20}" '
                    'text-anchor="middle">not sampled</text>')
        _x_labels(lines, modes, x, y + height + 24, width)
    if not cpu_values:
        lines.append(
            '<rect x="690" y="105" width="470" height="415" '
            'fill="#f7f7f7" stroke="#cccccc"/>')
        lines.append(
            '<text x="925" y="300" text-anchor="middle" '
            'font-size="17">CPU utilization not present in raw samples</text>')
    if any(run["verdict"].startswith("RESOURCE_FAIL")
           for run in summary["runs"]):
        lines.append(
            f'<text class="small" x="325" y="610" text-anchor="middle" '
            f'style="fill:{FAIL};font-weight:700">'
            'C2 Gemma GPU failed the zero-swap gate.</text>')
    return _finish(lines)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "run"


def _write_pair(path: Path, svg: str) -> list[Path]:
    svg_path = path.with_suffix(".svg")
    png_path = path.with_suffix(".png")
    svg_path.write_text(svg, encoding="utf-8")
    try:
        cairosvg.svg2png(
            bytestring=svg.encode("utf-8"),
            write_to=str(png_path),
            output_width=WIDTH,
            output_height=HEIGHT,
        )
    except Exception as error:
        raise GraphError(f"{png_path}: PNG conversion failed") from error
    return [svg_path, png_path]


def render(
    summary_path: Path,
    output_dir: Path,
    timeline_modes: list[str] | None = None,
) -> list[Path]:
    summary = validate_summary(read_json(summary_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    static: list[tuple[str, Callable[[dict[str, Any]], str]]] = [
        ("01_slo_goodput_throughput", throughput_comparison_svg),
        ("02_tail_latency_publication", latency_comparison_svg),
        ("04_selected_gpu_board_energy", energy_comparison_svg),
        ("05_server_resources", resources_comparison_svg),
    ]
    for name, function in static:
        outputs.extend(_write_pair(output_dir / name, function(summary)))
    for run in _representatives(summary["runs"], timeline_modes):
        name = (
            "03_timeline_" + _slug(run["mode"]) + "_"
            + _slug(run["cache_regime"])
        )
        outputs.extend(_write_pair(
            output_dir / name, timeline_svg(summary, run)))
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeline-mode", action="append")
    args = parser.parse_args()
    try:
        for path in render(
                args.summary, args.output_dir, args.timeline_mode):
            print(path)
    except (GraphError, OSError) as error:
        print(f"ERROR: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
