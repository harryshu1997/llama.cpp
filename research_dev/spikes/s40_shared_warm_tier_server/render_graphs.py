#!/usr/bin/env python3
"""Render deterministic S40 throughput and selected-GPU energy timelines."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
from typing import Any

from evidence_common import EvidenceError, read_json, require, require_int


WIDTH = 1000
HEIGHT = 440
LEFT = 82
RIGHT = 30
TOP = 44
BOTTOM = 62
COLORS = ("#0072b2", "#d55e00", "#009e73", "#cc79a7")


def _number(value: Any, field: str) -> int:
    return require_int(value, field)


def _polyline(
        points: list[tuple[int, int]],
        color: str,
        label: str) -> str:
    encoded = " ".join(f"{x},{y}" for x, y in points)
    return (
        f'<polyline fill="none" stroke="{color}" stroke-width="3" '
        f'points="{encoded}"><title>{escape(label)}</title></polyline>'
    )


def _base_svg(
        title: str,
        x_label: str,
        y_label: str,
        x_max: int,
        y_max: int,
        y_scale_divisor: int = 1) -> list[str]:
    plot_width = WIDTH - LEFT - RIGHT
    plot_height = HEIGHT - TOP - BOTTOM
    lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
            f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">'
        ),
        "<style>text{font-family:Arial,sans-serif;fill:#222}"
        ".tick{font-size:12px}.label{font-size:14px}"
        ".title{font-size:18px;font-weight:700}</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        (
            f'<text class="title" x="{WIDTH // 2}" y="25" '
            f'text-anchor="middle">{escape(title)}</text>'
        ),
    ]
    for index in range(6):
        x = LEFT + plot_width * index // 5
        y = TOP + plot_height * index // 5
        x_value = x_max * index / 5 / 1_000_000_000
        y_value = y_max * (5 - index) / 5 / y_scale_divisor
        lines.extend([
            (
                f'<line x1="{x}" y1="{TOP}" x2="{x}" '
                f'y2="{TOP + plot_height}" stroke="#eeeeee"/>'
            ),
            (
                f'<text class="tick" x="{x}" y="{TOP + plot_height + 22}" '
                f'text-anchor="middle">{x_value:.1f}</text>'
            ),
            (
                f'<line x1="{LEFT}" y1="{y}" x2="{LEFT + plot_width}" '
                f'y2="{y}" stroke="#eeeeee"/>'
            ),
            (
                f'<text class="tick" x="{LEFT - 10}" y="{y + 4}" '
                f'text-anchor="end">{y_value:.2f}</text>'
            ),
        ])
    lines.extend([
        (
            f'<line x1="{LEFT}" y1="{TOP + plot_height}" '
            f'x2="{LEFT + plot_width}" y2="{TOP + plot_height}" '
            f'stroke="#222"/>'
        ),
        (
            f'<line x1="{LEFT}" y1="{TOP}" x2="{LEFT}" '
            f'y2="{TOP + plot_height}" stroke="#222"/>'
        ),
        (
            f'<text class="label" x="{WIDTH // 2}" y="{HEIGHT - 15}" '
            f'text-anchor="middle">{escape(x_label)}</text>'
        ),
        (
            f'<text class="label" x="18" y="{HEIGHT // 2}" '
            f'text-anchor="middle" '
            f'transform="rotate(-90 18 {HEIGHT // 2})">'
            f'{escape(y_label)}</text>'
        ),
    ])
    return lines


def _xy(t_ns: int, value: int, x_max: int, y_max: int) -> tuple[int, int]:
    plot_width = WIDTH - LEFT - RIGHT
    plot_height = HEIGHT - TOP - BOTTOM
    x = LEFT + t_ns * plot_width // max(1, x_max)
    y = TOP + plot_height - value * plot_height // max(1, y_max)
    return x, y


def _switch_markers(
        switches: list[dict[str, Any]],
        x_max: int) -> list[str]:
    plot_height = HEIGHT - TOP - BOTTOM
    result = []
    for switch in switches:
        intent = _number(
            switch.get("intent_t_offset_ns"), "switch.intent_t_offset_ns")
        x, _ = _xy(intent, 0, x_max, 1)
        result.append(
            f'<line x1="{x}" y1="{TOP}" x2="{x}" '
            f'y2="{TOP + plot_height}" stroke="#666666" '
            f'stroke-dasharray="5,4"><title>switch intent</title></line>'
        )
        publication = switch.get("publication_t_offset_ns")
        if publication is not None:
            publication = _number(
                publication, "switch.publication_t_offset_ns")
            x, _ = _xy(publication, 0, x_max, 1)
            result.append(
                f'<line x1="{x}" y1="{TOP}" x2="{x}" '
                f'y2="{TOP + plot_height}" stroke="#009e73" '
                f'stroke-dasharray="2,3">'
                f'<title>atomic publication</title></line>'
            )
    return result


def throughput_svg(summary: dict[str, Any]) -> str:
    timeline = summary.get("timeline")
    require(
        isinstance(timeline, dict)
        and timeline.get("schema") == "s40-reduced-timeline-v1",
        "summary.timeline: unsupported",
    )
    rows = timeline.get("model_token_throughput")
    switches = timeline.get("switches")
    require(isinstance(rows, list) and rows, "timeline throughput: empty")
    require(isinstance(switches, list), "timeline switches: expected array")
    models = sorted(rows[0].get("model_milli_tokens_per_second", {}))
    require(bool(models), "timeline throughput: no models")
    x_max = max(_number(row.get("t_offset_ns"), "throughput.t_offset_ns")
                for row in rows)
    values: dict[str, list[int]] = {model_id: [] for model_id in models}
    for index, row in enumerate(rows):
        current = row.get("model_milli_tokens_per_second")
        require(
            isinstance(current, dict) and sorted(current) == models,
            f"throughput[{index}]: model set mismatch",
        )
        for model_id in models:
            values[model_id].append(
                _number(current[model_id], f"throughput[{index}].{model_id}"))
    y_max = max(1, max(value for items in values.values() for value in items))
    lines = _base_svg(
        "Per-model throughput over time",
        "Time since run start (s)",
        "Trailing 5 s throughput (tokens/s)",
        x_max,
        y_max,
        1000,
    )
    lines.extend(_switch_markers(switches, x_max))
    for index, model_id in enumerate(models):
        points = [
            _xy(
                _number(row["t_offset_ns"], "throughput.t_offset_ns"),
                values[model_id][row_index],
                x_max,
                y_max,
            )
            for row_index, row in enumerate(rows)
        ]
        color = COLORS[index % len(COLORS)]
        lines.append(_polyline(points, color, model_id))
        lines.append(
            f'<text class="label" x="{LEFT + 15 + index * 230}" y="42" '
            f'style="fill:{color}">{escape(model_id)}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def energy_svg(summary: dict[str, Any]) -> str:
    energy = summary.get("energy")
    require(isinstance(energy, dict), "summary.energy: expected object")
    require(
        energy.get("gpu_energy_scope")
        == "SELECTED_GPU_BOARD_DEVELOPMENT_ONLY",
        "summary.energy: selected-GPU samples required",
    )
    rows = energy.get("timeline")
    timeline = summary.get("timeline")
    require(isinstance(rows, list) and rows, "energy timeline: empty")
    require(isinstance(timeline, dict), "summary.timeline: expected object")
    switches = timeline.get("switches")
    require(isinstance(switches, list), "timeline switches: expected array")
    x_max = max(_number(row.get("t_offset_ns"), "energy.t_offset_ns")
                for row in rows)
    energy_millijoules = [
        _number(row.get("cumulative_gpu_energy_nj"), "energy.cumulative")
        // 1_000_000
        for row in rows
    ]
    y_max = max(1, max(energy_millijoules))
    lines = _base_svg(
        "Selected-GPU board energy over time (development only)",
        "Time since run start (s)",
        "Cumulative selected-GPU energy (mJ)",
        x_max,
        y_max,
    )
    lines.extend(_switch_markers(switches, x_max))
    points = [
        _xy(
            _number(row["t_offset_ns"], "energy.t_offset_ns"),
            energy_millijoules[index],
            x_max,
            y_max,
        )
        for index, row in enumerate(rows)
    ]
    lines.append(_polyline(points, COLORS[2], "selected GPU board"))
    lines.append(
        '<text class="tick" x="975" y="425" text-anchor="end">'
        'Not server-wall or total-system energy</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def render(summary_path: Path, output_dir: Path) -> list[Path]:
    summary = read_json(summary_path, "summary")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_dir / "throughput_timeline.svg",
        output_dir / "selected_gpu_energy_timeline.svg",
    ]
    outputs[0].write_text(throughput_svg(summary), encoding="ascii")
    outputs[1].write_text(energy_svg(summary), encoding="ascii")
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        for path in render(args.summary, args.output_dir):
            print(path)
    except (EvidenceError, OSError) as error:
        print(f"ERROR: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
