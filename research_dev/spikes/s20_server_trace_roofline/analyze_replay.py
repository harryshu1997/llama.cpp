#!/usr/bin/env python3
"""Reduce S20 Nsight/runtime evidence and render a self-contained SVG graph."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
import sqlite3
from typing import Any


class AnalysisError(RuntimeError):
    pass


REQUIRED_GPU_FIELDS = (
    "DRAM Read Throughput",
    "DRAM Write Throughput",
    "SM Active",
    "SM Issue",
    "Tensor Active",
)

REGIME_COLORS = {
    "COMPUTE_DOMINANT": "#fee2c5",
    "MEMORY_DOMINANT": "#dbeafe",
    "MIXED": "#ede9fe",
    "IDLE_OR_TRANSITION": "#f3f4f6",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open("r", encoding="ascii") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AnalysisError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise AnalysisError(f"{path}:{line_number}: record must be an object")
            values.append(value)
    if not values:
        raise AnalysisError(f"{path}: no records")
    return values


def replay_range(connection: sqlite3.Connection) -> tuple[int, int]:
    rows = connection.execute(
        "SELECT start, end FROM NVTX_EVENTS WHERE text = 'MEASURED_REPLAY'"
    ).fetchall()
    if len(rows) != 1:
        raise AnalysisError(f"expected one MEASURED_REPLAY NVTX range, found {len(rows)}")
    start, end = rows[0]
    if type(start) is not int or type(end) is not int or end <= start:
        raise AnalysisError("invalid MEASURED_REPLAY range")
    return start, end


def load_gpu_samples(connection: sqlite3.Connection, start: int,
                     end: int) -> list[dict[str, float]]:
    rows = connection.execute(
        "SELECT rawTimestamp, data FROM GENERIC_EVENTS "
        "WHERE rawTimestamp >= ? AND rawTimestamp <= ? ORDER BY rawTimestamp",
        (start, end),
    ).fetchall()
    samples: list[dict[str, float]] = []
    for timestamp, payload in rows:
        value = json.loads(payload)
        if not all(field in value for field in REQUIRED_GPU_FIELDS):
            continue
        sample = {"t_ns": float(timestamp - start)}
        for field in REQUIRED_GPU_FIELDS:
            try:
                sample[field] = float(value[field])
            except (TypeError, ValueError) as exc:
                raise AnalysisError(f"invalid Nsight field {field}: {value[field]!r}") from exc
        samples.append(sample)
    if len(samples) < 10:
        raise AnalysisError("insufficient Nsight GPU metric samples")
    return samples


def request_intervals(events: list[dict[str, Any]]) -> list[dict[str, int]]:
    by_id: dict[str, dict[str, int]] = {}
    field_for_kind = {
        "request_start": "start_ns",
        "first_token": "first_token_ns",
        "request_end": "end_ns",
    }
    for event in events:
        kind = event.get("kind")
        if kind not in field_for_kind:
            continue
        event_id = event.get("event_id")
        t_ns = event.get("t_ns")
        if not isinstance(event_id, str) or type(t_ns) is not int:
            raise AnalysisError("invalid request event")
        record = by_id.setdefault(event_id, {})
        field = field_for_kind[kind]
        if field in record:
            raise AnalysisError(f"duplicate {kind} for {event_id}")
        record[field] = t_ns
    intervals: list[dict[str, int]] = []
    for event_id, record in by_id.items():
        if set(record) != {"start_ns", "first_token_ns", "end_ns"}:
            raise AnalysisError(f"incomplete request interval for {event_id}")
        if not record["start_ns"] <= record["first_token_ns"] <= record["end_ns"]:
            raise AnalysisError(f"unordered request interval for {event_id}")
        intervals.append(record)
    if not intervals:
        raise AnalysisError("no complete request intervals")
    return intervals


def phase_counts(intervals: list[dict[str, int]], t_ns: int) -> tuple[int, int]:
    prefill = sum(
        item["start_ns"] <= t_ns < item["first_token_ns"] for item in intervals
    )
    decode = sum(
        item["first_token_ns"] <= t_ns < item["end_ns"] for item in intervals
    )
    return prefill, decode


def classify_counts(prefill: int, decode: int) -> str:
    if prefill > 0 and decode > 0:
        return "MIXED"
    if prefill > 0:
        return "COMPUTE_DOMINANT"
    if decode > 0:
        return "MEMORY_DOMINANT"
    return "IDLE_OR_TRANSITION"


def reduce_timeline(gpu_samples: list[dict[str, float]],
                    intervals: list[dict[str, int]], duration_ns: int,
                    bin_ms: int) -> list[dict[str, Any]]:
    width_ns = bin_ms * 1_000_000
    bins: list[dict[str, Any]] = []
    for left in range(0, duration_ns, width_ns):
        right = min(duration_ns, left + width_ns)
        metrics = [sample for sample in gpu_samples if left <= sample["t_ns"] < right]
        if not metrics:
            continue
        prefill, decode = phase_counts(intervals, (left + right) // 2)
        mean = lambda field: sum(sample[field] for sample in metrics) / len(metrics)
        dram_read = mean("DRAM Read Throughput")
        dram_write = mean("DRAM Write Throughput")
        bins.append({
            "t_s": (left + right) / 2e9,
            "dram_read_pct": dram_read,
            "dram_write_pct": dram_write,
            "dram_total_pct": min(100.0, dram_read + dram_write),
            "sm_active_pct": mean("SM Active"),
            "sm_issue_pct": mean("SM Issue"),
            "tensor_active_pct": mean("Tensor Active"),
            "active_slots": prefill + decode,
            "prefill_slots": prefill,
            "decode_slots": decode,
            "regime": classify_counts(prefill, decode),
        })
    if not bins:
        raise AnalysisError("timeline reduction produced no bins")
    return bins


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def summarize(timeline: list[dict[str, Any]], events: list[dict[str, Any]],
              manifest: dict[str, Any], bin_ms: int) -> dict[str, Any]:
    counts = {name: 0 for name in REGIME_COLORS}
    for point in timeline:
        counts[point["regime"]] += 1
    completed = [event for event in events if event.get("kind") == "request_end"]
    if len(completed) != manifest.get("request_count"):
        raise AnalysisError("event/manifest completion count mismatch")
    latencies_ms = [(event["end_ns"] - event["start_ns"]) / 1e6 for event in completed]
    total_bins = len(timeline)
    return {
        "schema": "s20-server-trace-summary-v1",
        "verdict": "SERVER_TRACE_TIMELINE_PASS",
        "classification_scope": "PHASE_DERIVED_WITH_NSIGHT_PRESSURE_NOT_KERNEL_ROOFLINE",
        "bin_ms": bin_ms,
        "request_count": manifest["request_count"],
        "input_tokens": manifest["input_tokens"],
        "output_tokens": manifest["output_tokens"],
        "replay_wall_s": manifest["replay_wall_ns"] / 1e9,
        "request_latency_ms_p50": percentile(latencies_ms, 0.50),
        "request_latency_ms_p95": percentile(latencies_ms, 0.95),
        "request_latency_ms_max": max(latencies_ms),
        "regime_fraction": {
            name: count / total_bins for name, count in counts.items()
        },
        "gpu_metric_mean": {
            "dram_total_pct": sum(p["dram_total_pct"] for p in timeline) / total_bins,
            "sm_issue_pct": sum(p["sm_issue_pct"] for p in timeline) / total_bins,
            "tensor_active_pct": sum(p["tensor_active_pct"] for p in timeline) / total_bins,
        },
        "gpu_metric_peak": {
            "dram_total_pct": max(p["dram_total_pct"] for p in timeline),
            "sm_issue_pct": max(p["sm_issue_pct"] for p in timeline),
            "tensor_active_pct": max(p["tensor_active_pct"] for p in timeline),
        },
    }


def polyline(points: list[dict[str, Any]], field: str, x0: float, y0: float,
             width: float, height: float, duration: float, maximum: float) -> str:
    coords = []
    for point in points:
        x = x0 + width * point["t_s"] / duration
        y = y0 + height * (1.0 - min(max(float(point[field]), 0.0), maximum) / maximum)
        coords.append(f"{x:.2f},{y:.2f}")
    return " ".join(coords)


def render_svg(timeline: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    width = 1440
    height = 850
    left = 92
    right = 36
    plot_width = width - left - right
    top_y = 116
    top_h = 350
    lower_y = 548
    lower_h = 190
    duration = max(point["t_s"] for point in timeline)
    duration = max(duration, 0.1)
    max_slots = max(32, max(point["active_slots"] for point in timeline))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172033;letter-spacing:0}.title{font-size:28px;font-weight:700}.sub{font-size:14px;fill:#4b5563}.axis{font-size:12px;fill:#64748b}.legend{font-size:13px}.grid{stroke:#dbe2ea;stroke-width:1}.line{fill:none;stroke-width:2.5;stroke-linejoin:round;stroke-linecap:round}</style>',
        '<text x="92" y="46" class="title">Gemma-4-12B: real BurstGPT burst on one A6000</text>',
        '<text x="92" y="75" class="sub">32 observed arrivals, 21,203 observed input tokens, 1,898 observed output tokens; synthetic token values only</text>',
        '<text x="92" y="96" class="sub">Background = executed slot phase; lines = Nsight GA10x hardware pressure sampled at 1 kHz and reduced to 100 ms</text>',
    ]

    for point in timeline:
        bin_width = plot_width / len(timeline) + 0.5
        x = left + plot_width * point["t_s"] / duration - bin_width / 2
        parts.append(
            f'<rect x="{x:.2f}" y="{top_y}" width="{bin_width:.2f}" height="{top_h}" fill="{REGIME_COLORS[point["regime"]]}"/>'
        )
        parts.append(
            f'<rect x="{x:.2f}" y="{lower_y}" width="{bin_width:.2f}" height="{lower_h}" fill="{REGIME_COLORS[point["regime"]]}"/>'
        )

    for value in range(0, 101, 20):
        y = top_y + top_h * (1 - value / 100)
        parts.append(f'<line x1="{left}" x2="{left + plot_width}" y1="{y:.2f}" y2="{y:.2f}" class="grid"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" class="axis">{value}%</text>')
    for value in (0, 8, 16, 24, 32):
        y = lower_y + lower_h * (1 - value / max_slots)
        parts.append(f'<line x1="{left}" x2="{left + plot_width}" y1="{y:.2f}" y2="{y:.2f}" class="grid"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" class="axis">{value}</text>')
    tick = max(1, math.ceil(duration / 10))
    for second in range(0, math.ceil(duration) + 1, tick):
        x = left + plot_width * second / duration
        parts.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{top_y}" y2="{lower_y + lower_h}" class="grid"/>')
        parts.append(f'<text x="{x:.2f}" y="{lower_y + lower_h + 24}" text-anchor="middle" class="axis">{second}s</text>')

    lines = (
        ("dram_total_pct", "#2563eb", "DRAM read + write"),
        ("tensor_active_pct", "#dc2626", "Tensor active"),
        ("sm_issue_pct", "#f59e0b", "SM issue"),
    )
    for field, color, _ in lines:
        points = polyline(timeline, field, left, top_y, plot_width, top_h, duration, 100)
        parts.append(f'<polyline points="{points}" class="line" stroke="{color}"/>')
    slot_lines = (
        ("prefill_slots", "#ea580c", "Prefill slots"),
        ("decode_slots", "#0891b2", "Decode slots"),
        ("active_slots", "#334155", "All active slots"),
    )
    for field, color, _ in slot_lines:
        points = polyline(timeline, field, left, lower_y, plot_width, lower_h, duration, max_slots)
        parts.append(f'<polyline points="{points}" class="line" stroke="{color}"/>')

    parts.extend([
        f'<rect x="{left}" y="{top_y}" width="{plot_width}" height="{top_h}" fill="none" stroke="#94a3b8"/>',
        f'<rect x="{left}" y="{lower_y}" width="{plot_width}" height="{lower_h}" fill="none" stroke="#94a3b8"/>',
        f'<text x="24" y="{top_y + top_h / 2}" transform="rotate(-90 24 {top_y + top_h / 2})" text-anchor="middle" class="axis">Hardware pressure (% of peak)</text>',
        f'<text x="24" y="{lower_y + lower_h / 2}" transform="rotate(-90 24 {lower_y + lower_h / 2})" text-anchor="middle" class="axis">Concurrent slots</text>',
        f'<text x="{left + plot_width / 2}" y="{height - 26}" text-anchor="middle" class="axis">Time from first observed arrival</text>',
    ])

    legend_x = left + 12
    legend_y = top_y + 22
    for _, color, label in lines:
        parts.append(f'<line x1="{legend_x}" x2="{legend_x + 28}" y1="{legend_y}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 36}" y="{legend_y + 4}" class="legend">{label}</text>')
        legend_x += 180
    legend_x = left + 12
    legend_y = lower_y + 22
    for _, color, label in slot_lines:
        parts.append(f'<line x1="{legend_x}" x2="{legend_x + 28}" y1="{legend_y}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 36}" y="{legend_y + 4}" class="legend">{label}</text>')
        legend_x += 180

    legend_x = width - 510
    legend_y = 30
    for label, color in REGIME_COLORS.items():
        parts.append(f'<rect x="{legend_x}" y="{legend_y - 12}" width="14" height="14" fill="{color}" stroke="#cbd5e1"/>')
        parts.append(f'<text x="{legend_x + 20}" y="{legend_y}" class="axis">{label.replace("_", " ").title()}</text>')
        legend_x += 125
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def render_html(svg: str) -> str:
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gemma-4-12B server trace</title><style>body{margin:0;background:#eef2f6;color:#172033;font-family:Arial,sans-serif;letter-spacing:0}.wrap{max-width:1480px;margin:24px auto;background:white;border:1px solid #d8e0e8;padding:18px}svg{width:100%;height:auto;display:block}p{line-height:1.5;margin:10px 30px}code{background:#eef2f6;padding:2px 5px}</style></head>
<body><main class="wrap">""" + svg + """<p><strong>Interpretation.</strong> Orange background means active slots are still in prefill; blue means all active slots are decoding; purple means both coexist. Hardware lines are independent Nsight GA10x samples. The labels are phase-derived resource regimes, not a formal per-kernel roofline certificate.</p>
<p><strong>Provenance.</strong> Arrival times and token counts are observed BurstGPT fields. Prompt token values are deterministic synthetic replacements because the public trace contains no prompt text. One A6000 and one Gemma-4-12B F16 server are used.</p>
</main></body></html>\n"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bin-ms", type=int, default=100)
    args = parser.parse_args()
    if args.bin_ms < 50 or args.bin_ms > 1000:
        raise AnalysisError("bin-ms must be in [50, 1000]")

    read_jsonl(args.run_dir / "runtime_samples.jsonl")
    events = read_jsonl(args.run_dir / "events.jsonl")
    with (args.run_dir / "run_manifest.json").open("r", encoding="ascii") as stream:
        manifest = json.load(stream)
    with sqlite3.connect(args.sqlite) as connection:
        start, end = replay_range(connection)
        gpu = load_gpu_samples(connection, start, end)
    intervals = request_intervals(events)
    if len(intervals) != manifest.get("request_count"):
        raise AnalysisError("request interval/manifest count mismatch")
    timeline = reduce_timeline(gpu, intervals, end - start, args.bin_ms)
    summary = summarize(timeline, events, manifest, args.bin_ms)

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "timeline.json").write_text(
        json.dumps(timeline, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    svg = render_svg(timeline, summary)
    (args.output / "server_trace.svg").write_text(svg, encoding="ascii")
    (args.output / "server_trace.html").write_text(render_html(svg), encoding="ascii")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AnalysisError as exc:
        print(f"S20_ANALYSIS_ERROR: {exc}")
        raise SystemExit(2)
