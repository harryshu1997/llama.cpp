#!/usr/bin/env python3
"""Render when the phone route is working and in which stage.

The throughput timeline bins every output token at its request's completion
instant, which hides ~43 s of real generation on this run. This figure shows
the per-request lifecycle instead, split into the three intervals the trace
records directly:

    queued   scheduled_arrival -> dispatch   (waiting for one of 8 slots)
    prefill  dispatch          -> first_token
    decode   first_token       -> completion

The lower panel counts slots occupied over time, so "is the phone actually
working" is answered by occupancy rather than by completed-token bins.
Colours avoid the model palette used by the throughput graphs (blue Gemma /
orange Qwen) so the two figures can sit side by side without a colour clash.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape


HERE = Path(__file__).resolve().parent
QWEN_ID = "qwen3-14b-q4_k_m"
WIDTH = 1240
QUEUED = "#9aa3ad"
PREFILL = "#009e73"
DECODE = "#7b2cbf"
GRID = "#d9dde3"
TEXT = "#20242a"
SLOTS = 8
STYLE = (
    "<style>text{font-family:Arial,sans-serif;fill:#20242a}"
    ".title{font-size:22px;font-weight:700}"
    ".subtitle{font-size:13px}.axis{font-size:12px}"
    ".small{font-size:11px}.mode{font-size:11px;font-weight:600}"
    "</style>"
)


def read_requests(path: Path) -> tuple[list[dict], int]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="ascii").splitlines()
        if line.strip()
    ]
    start = next(r for r in rows if r.get("kind") == "trace_start")["t_ns"]
    requests = [
        r for r in rows
        if r.get("kind") == "request_complete" and r["model_id"] == QWEN_ID
    ]
    requests.sort(key=lambda r: r["scheduled_arrival_ns"])
    return requests, start


def occupancy(requests: list[dict], start: int) -> list[tuple[float, int]]:
    events: list[tuple[float, int]] = []
    for row in requests:
        events.append(((row["dispatch_ns"] - start) / 1e9, 1))
        events.append(((row["completion_ns"] - start) / 1e9, -1))
    events.sort()
    series: list[tuple[float, int]] = []
    busy = 0
    for moment, delta in events:
        busy += delta
        series.append((moment, busy))
    return series


def render(requests: list[dict], start: int, label: str) -> str:
    span = max((r["completion_ns"] - start) / 1e9 for r in requests)
    rows = len(requests)
    left, right = 92, 1168
    plot_w = right - left
    top = 104
    row_h = 20
    gantt_h = rows * row_h
    occ_top = top + gantt_h + 74
    occ_h = 108

    def px(seconds: float) -> float:
        return left + seconds / span * plot_w

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
        f'height="{occ_top + occ_h + 78}" viewBox="0 0 {WIDTH} '
        f'{occ_top + occ_h + 78}">',
        STYLE,
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text class="title" x="{WIDTH // 2}" y="30" text-anchor="middle">'
        f'{escape("Phone route activity and stage: " + label)}</text>',
        f'<text class="subtitle" x="{WIDTH // 2}" y="52" '
        'text-anchor="middle">OP15 [0,30) + OP12 [30,40) OpenCL | '
        f'{rows} Qwen3 14B Q4_K_M requests | {SLOTS} concurrent slots | '
        '8 output tokens each</text>',
    ]

    for index, (name, colour) in enumerate(
        (("queued (waiting for a slot)", QUEUED),
         ("prefill (dispatch to first token)", PREFILL),
         ("decode (first token to completion)", DECODE))
    ):
        lx = 92 + index * 340
        lines.append(
            f'<rect x="{lx}" y="68" width="26" height="12" fill="{colour}"/>')
        lines.append(
            f'<text class="axis" x="{lx + 34}" y="79">{escape(name)}</text>')

    for tick in range(7):
        seconds = span * tick / 6
        x = px(seconds)
        lines.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" '
            f'y2="{occ_top + occ_h}" stroke="{GRID}"/>')
        lines.append(
            f'<text class="axis" x="{x:.1f}" y="{occ_top + occ_h + 22}" '
            f'text-anchor="middle">{seconds:.1f}</text>')
    lines.append(
        f'<text class="axis" x="{(left + right) // 2}" '
        f'y="{occ_top + occ_h + 46}" text-anchor="middle">'
        'time since paid start (s)</text>')

    for index, row in enumerate(requests):
        y = top + index * row_h
        arrival = (row["scheduled_arrival_ns"] - start) / 1e9
        dispatch = (row["dispatch_ns"] - start) / 1e9
        first = (row["first_token_ns"] - start) / 1e9
        done = (row["completion_ns"] - start) / 1e9
        bar = row_h - 6
        for x0, x1, colour, name in (
            (arrival, dispatch, QUEUED, "queued"),
            (dispatch, first, PREFILL, "prefill"),
            (first, done, DECODE, "decode"),
        ):
            if x1 <= x0:
                continue
            lines.append(
                f'<rect x="{px(x0):.1f}" y="{y}" '
                f'width="{max(1.0, px(x1) - px(x0)):.1f}" height="{bar}" '
                f'fill="{colour}" rx="2">'
                f'<title>request {row["request_index"]} {name} '
                f'{x1 - x0:.1f} s</title></rect>')
        lines.append(
            f'<text class="small" x="{left - 8}" y="{y + bar - 3}" '
            f'text-anchor="end" style="fill:#5b6672">'
            f'{row["request_index"]}</text>')
    lines.append(
        f'<text class="axis" x="26" y="{top + gantt_h // 2}" '
        f'transform="rotate(-90 26 {top + gantt_h // 2})" '
        'text-anchor="middle">request index (by arrival)</text>')

    series = occupancy(requests, start)
    base = occ_top + occ_h
    lines.append(
        f'<text class="axis" x="{left}" y="{occ_top - 12}">'
        'phone slots occupied (dispatch to completion)</text>')
    for level in range(0, SLOTS + 1, 2):
        y = base - level / SLOTS * occ_h
        lines.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
            f'stroke="{GRID}"/>')
        lines.append(
            f'<text class="axis" x="{left - 8}" y="{y + 4:.1f}" '
            f'text-anchor="end">{level}</text>')
    points = [f"{left:.1f},{base:.1f}"]
    previous = 0
    for moment, busy in series:
        x = px(moment)
        points.append(f"{x:.1f},{base - previous / SLOTS * occ_h:.1f}")
        points.append(f"{x:.1f},{base - busy / SLOTS * occ_h:.1f}")
        previous = busy
    points.append(f"{right:.1f},{base - previous / SLOTS * occ_h:.1f}")
    points.append(f"{right:.1f},{base:.1f}")
    lines.append(
        f'<polygon points="{" ".join(points)}" fill="{DECODE}" '
        'fill-opacity="0.18"/>')
    lines.append(
        f'<polyline points="{" ".join(points[:-1])}" fill="none" '
        f'stroke="{DECODE}" stroke-width="2"/>')
    saturated = base - occ_h
    lines.append(
        f'<line x1="{left}" y1="{saturated:.1f}" x2="{right}" '
        f'y2="{saturated:.1f}" stroke="#b91c1c" stroke-dasharray="5,4"/>')
    lines.append(
        f'<text class="small" x="{right - 4}" y="{saturated - 6:.1f}" '
        'text-anchor="end" style="fill:#b91c1c">all 8 slots busy</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path,
                        default=HERE / "trace-events.jsonl")
    parser.add_argument("--output", type=Path,
                        default=HERE / "graphs" / "t2_phone_stages.svg")
    parser.add_argument("--label", default="t2-phone-trace-prototype")
    args = parser.parse_args()
    requests, start = read_requests(args.events)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(requests, start, args.label),
                           encoding="ascii")

    span = max((r["completion_ns"] - start) / 1e9 for r in requests)
    first_token = min((r["first_token_ns"] - start) / 1e9 for r in requests)
    first_done = min((r["completion_ns"] - start) / 1e9 for r in requests)
    series = occupancy(requests, start)
    busy_from = min(m for m, b in series if b > 0)
    full = sum(
        1 for _, b in series if b == SLOTS)
    print(f"first token emitted      {first_token:.2f} s")
    print(f"first request completed  {first_done:.2f} s")
    print(f"phones first busy        {busy_from:.2f} s")
    print(f"span                     {span:.2f} s")
    print(f"points at full occupancy {full} of {len(series)} transitions")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
