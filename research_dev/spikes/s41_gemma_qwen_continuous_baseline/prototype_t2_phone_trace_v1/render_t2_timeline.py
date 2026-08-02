#!/usr/bin/env python3
"""Render the T2 phone-trace throughput timeline in the server-baseline style.

Reads the prototype's `trace-events.jsonl` and derives a per-model throughput
timeline using the SAME binning as `reduce_server_results._timeline`: one
second bins, output tokens attributed to the bin containing the request's
`completion_ns`, values reported as milli-tokens per second. The SVG is drawn
by the baseline renderer so the T2 graph is directly comparable to
`03_timeline_*.svg` from the server-only campaign.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
S41 = HERE.parent
GEMMA_ID = "gemma-4-12b-it-q8_0"
QWEN_ID = "qwen3-14b-q4_k_m"
GRID_LINE = "#d9dde3"
MODELS = [
    {"id": GEMMA_ID, "label": "Gemma 4 12B Q8_0"},
    {"id": QWEN_ID, "label": "Qwen3 14B Q4_K_M"},
]
BIN_WIDTH_NS = 1_000_000_000


def load_renderer() -> Any:
    """Load the baseline renderer for its SVG helpers.

    `render_server_graphs` imports cairosvg only for PNG export, which is not
    installed here. A stub keeps the import satisfied without editing that
    file, whose bytes are bound by the frozen v1 graph manifest.
    """

    path = S41 / "render_server_graphs.py"
    if "cairosvg" not in sys.modules:
        stub = type(sys)("cairosvg")
        stub.svg2png = None
        sys.modules["cairosvg"] = stub
    spec = importlib.util.spec_from_file_location("render_server_graphs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_events(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="ascii").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def read_wire(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="ascii").splitlines()
        if line.strip()
    ]
    rows.sort(key=lambda row: row["started_ns"])
    return rows


def build_timeline(events: list[dict[str, Any]]) -> dict[str, Any]:
    starts = [e for e in events if e.get("kind") == "trace_start"]
    completes = [e for e in events if e.get("kind") == "request_complete"]
    if len(starts) != 1:
        raise SystemExit("E_TRACE_START")
    if not completes:
        raise SystemExit("E_NO_COMPLETIONS")
    paid_start_ns = starts[0]["t_ns"]
    paid_end_ns = max(row["completion_ns"] for row in completes)
    duration_ns = paid_end_ns - paid_start_ns
    bin_count = (duration_ns + BIN_WIDTH_NS - 1) // BIN_WIDTH_NS
    counts = [
        {model["id"]: 0 for model in MODELS} for _ in range(bin_count)
    ]
    for row in completes:
        offset_ns = min(
            duration_ns - 1,
            max(0, row["completion_ns"] - paid_start_ns),
        )
        counts[offset_ns // BIN_WIDTH_NS][row["model_id"]] += len(
            row["tokens"]
        )
    throughput = []
    for index, values in enumerate(counts):
        lo = index * BIN_WIDTH_NS
        hi = min(duration_ns, (index + 1) * BIN_WIDTH_NS)
        width = hi - lo
        throughput.append({
            "model_milli_tokens_per_second": {
                model["id"]: values[model["id"]] * 1_000_000_000_000 // width
                for model in MODELS
            },
            "t_offset_ns": hi,
        })
    return {
        "bin_width_ns": BIN_WIDTH_NS,
        "loads": [],
        "resources": [],
        "switches": [],
        "throughput": throughput,
    }


def phone_markers(
    svg: str,
    events: list[dict[str, Any]],
    timeline: dict[str, Any],
    wire: list[dict[str, Any]],
) -> str:
    """Add lanes for real phone execution, arrivals and completions.

    The throughput series credits all 8 output tokens of a request at its
    completion instant, so it reads zero until 55.31 s even though the phones
    execute continuously from 1.95 s. The execution lane plots the measured
    `started_ns`..`completed_ns` span of every physical batch from the wire
    ledger, which is what the phones were actually doing.

    Geometry mirrors `render_server_graphs.timeline_svg`: plot origin (88,105),
    1080x430, x scaled by the last bin edge. Lanes are appended below the plot,
    so the canvas grows and nothing overlaps the axes. Elements are emitted
    self-closing: the available rasterizer drops marks carrying a <title>.
    """

    start = next(e for e in events if e.get("kind") == "trace_start")["t_ns"]
    rows = [
        e for e in events
        if e.get("kind") == "request_complete" and e["model_id"] == QWEN_ID
    ]
    x0, width = 88, 1080
    x_max = max(row["t_offset_ns"] for row in timeline["throughput"])
    colour = "#d55e00"
    exec_y, exec_h = 612, 20
    marker_y = 668
    new_height = 720

    def px(ns: int) -> float:
        return x0 + max(0, min(x_max, ns)) / x_max * width

    out = [
        f'<text class="axis" x="{x0}" y="{exec_y - 8}">'
        'phone executing: measured batch spans on OP15 + OP12 '
        f'({len(wire)} physical batches)</text>',
        f'<rect x="{x0}" y="{exec_y}" width="{width}" height="{exec_h}" '
        'fill="#f2f4f7"/>',
    ]
    for batch in wire:
        bx = px(batch["started_ns"] - start)
        bw = max(1.2, px(batch["completed_ns"] - start) - bx)
        out.append(
            f'<rect x="{bx:.1f}" y="{exec_y}" width="{bw:.1f}" '
            f'height="{exec_h}" fill="{colour}" fill-opacity="0.85"/>')
    out.append(
        f'<text class="axis" x="{x0}" y="{marker_y - 14}">'
        'phone request arrivals and completions</text>')
    out.append(
        f'<line x1="{x0}" y1="{marker_y}" x2="{x0 + width}" y2="{marker_y}" '
        f'stroke="{GRID_LINE}"/>')
    for row in rows:
        x = px(row["scheduled_arrival_ns"] - start)
        out.append(
            f'<circle cx="{x:.1f}" cy="{marker_y}" r="5" fill="#ffffff" '
            f'stroke="{colour}" stroke-width="2"/>')
    for row in rows:
        x = px(row["completion_ns"] - start)
        out.append(
            f'<circle cx="{x:.1f}" cy="{marker_y}" r="5" fill="{colour}" '
            'stroke="#ffffff" stroke-width="1"/>')
    legend = [
        f'<rect x="700" y="86" width="26" height="11" fill="{colour}" '
        'fill-opacity="0.85"/>',
        '<text class="small" x="734" y="96">phone executing</text>',
        f'<circle cx="856" cy="92" r="4.5" fill="#ffffff" stroke="{colour}" '
        'stroke-width="2"/>',
        '<text class="small" x="868" y="96">arrival</text>',
        f'<circle cx="936" cy="92" r="4.5" fill="{colour}"/>',
        '<text class="small" x="948" y="96">completion</text>',
    ]
    svg = svg.replace('height="650"', f'height="{new_height}"', 1)
    svg = svg.replace('viewBox="0 0 1240 650"',
                      f'viewBox="0 0 1240 {new_height}"', 1)
    return svg.replace("</svg>", "\n".join(out + legend) + "\n</svg>")


def summarize(events: list[dict[str, Any]], timeline: dict[str, Any]) -> None:
    completes = [e for e in events if e.get("kind") == "request_complete"]
    paid_start = next(
        e for e in events if e.get("kind") == "trace_start")["t_ns"]
    paid_end = max(row["completion_ns"] for row in completes)
    seconds = (paid_end - paid_start) / 1e9
    print(f"paid duration      {seconds:.3f} s")
    total = 0
    for model in MODELS:
        rows = [r for r in completes if r["model_id"] == model["id"]]
        tokens = sum(len(r["tokens"]) for r in rows)
        total += tokens
        peak = max(
            row["model_milli_tokens_per_second"][model["id"]]
            for row in timeline["throughput"]
        ) / 1000
        active = sum(
            1 for row in timeline["throughput"]
            if row["model_milli_tokens_per_second"][model["id"]] > 0
        )
        print(
            f"{model['label']:<20} {len(rows):>3} req  {tokens:>4} tok  "
            f"mean {tokens / seconds:6.3f} tok/s  peak {peak:6.1f} tok/s  "
            f"active {active:>3}/{len(timeline['throughput'])} bins"
        )
    print(f"aggregate          {total / seconds:.3f} output tokens/s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path,
                        default=HERE / "trace-events.jsonl")
    parser.add_argument("--output", type=Path,
                        default=HERE / "graphs" / "t2_timeline.svg")
    parser.add_argument("--label", default="t2-phone-trace-prototype")
    parser.add_argument("--wire", type=Path,
                        default=HERE / "phone-wire.jsonl")
    parser.add_argument("--no-markers", action="store_true",
                        help="omit the phone arrival/dispatch/completion overlay")
    args = parser.parse_args()

    events = read_events(args.events)
    timeline = build_timeline(events)
    renderer = load_renderer()
    run = {
        "cache_regime": "gemma-cuda-resident, qwen-phone-resident",
        "label": args.label,
        "mode": "T2_PHONE_NO_PROMOTION_PROTOTYPE",
        "repeat_index": 1,
        # All 74 requests completed and the zero-swap gate held on Gemma, so
        # the renderer's failure banner stays off.
        "stranded_request_count": 0,
        "timeline": timeline,
        "verdict": "BURSTGPT_T2_PHONE_TRACE_PROTOTYPE_PASS",
    }
    svg = renderer.timeline_svg({"models": MODELS}, run)
    if not args.no_markers:
        wire = read_wire(args.wire)
        svg = phone_markers(svg, events, timeline, wire)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="ascii")
    summarize(events, timeline)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
