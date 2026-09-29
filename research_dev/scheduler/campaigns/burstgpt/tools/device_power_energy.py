#!/usr/bin/env python3
"""Host energy per device-power state of one runner output directory.

    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.device_power_energy --run-dir run/ [--json]

Reads ``RESULT.json`` (``paid_start_ns``, ``paid_end_ns``), the host samples
``resource-samples.jsonl`` and the ``DEVICE_POWER_STATE`` events (``DEVICE_POWER_EVENTS.json``
when the controller wrote one, else ``RESULT.device_power_events``, else none). Every event at
``at_us`` is a boundary at ``paid_start_ns + at_us * 1000`` on the sample clock; the GPU board
energy (``_integrate_gpu``) and the RAPL package energy (``_integrate_rapl``) of every segment
between consecutive boundaries inside the paid window are summed per state. A run without the
policy has one state (``UNCONTROLLED``) over the whole paid window. Segments the samples do not
cover are reported, not integrated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from research_dev.scheduler.adapters.contracts import PhysicalAdapterError
from research_dev.scheduler.adapters.host_runtime import _integrate_gpu, _integrate_rapl

UNCONTROLLED = "UNCONTROLLED"
EVENTS_FILE = "DEVICE_POWER_EVENTS.json"
SCHEMA = "s42-device-power-energy-v1"


def load_events(run_dir: Path, result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """DEVICE_POWER_STATE rows: the incremental file (complete, includes close) else RESULT's."""
    path = run_dir / EVENTS_FILE
    rows = json.loads(path.read_text()) if path.is_file() else result.get("device_power_events", [])
    if type(rows) is not list or any(type(row) is not dict or row.get("kind") != "DEVICE_POWER_STATE"
                                     or type(row.get("at_us")) is not int for row in rows):
        raise ValueError("device power events are invalid")
    return sorted(rows, key=lambda row: row["at_us"])


def load_samples(path: Path) -> list[dict[str, Any]]:
    """Host sample rows with both a GPU and a RAPL sample (the integrators need both)."""
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if type(row.get("gpu")) is dict and type(row.get("rapl_package")) is dict:
                    rows.append(row)
    return rows


def state_segments(events: Sequence[Mapping[str, Any]], paid_start_ns: int, paid_end_ns: int) -> list[dict[str, Any]]:
    """``[{state, start_ns, end_ns}]`` covering the paid window; the state at its start is the
    ``to`` of the last event at or before it (OFF when none), no events -> one UNCONTROLLED."""
    if not events:
        return [{"state": UNCONTROLLED, "start_ns": paid_start_ns, "end_ns": paid_end_ns}]
    state = "OFF"
    boundaries: list[tuple[int, str]] = []
    for row in events:
        at_ns = paid_start_ns + row["at_us"] * 1000
        if at_ns <= paid_start_ns:
            state = row["to"]
        elif at_ns < paid_end_ns:
            boundaries.append((at_ns, row["to"]))
    segments = []
    cursor = paid_start_ns
    for at_ns, next_state in boundaries:
        if at_ns > cursor:
            segments.append({"state": state, "start_ns": cursor, "end_ns": at_ns})
        cursor, state = at_ns, next_state
    if paid_end_ns > cursor:
        segments.append({"state": state, "start_ns": cursor, "end_ns": paid_end_ns})
    return segments


def integrate_segments(samples: Sequence[Mapping[str, Any]], segments: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Add ``gpu_j``/``cpu_j`` to every segment, or ``uncovered`` when the samples do not span it."""
    rows = []
    for segment in segments:
        row = dict(segment)
        try:
            row["gpu_j"] = _integrate_gpu(samples, segment["start_ns"], segment["end_ns"])
            row["cpu_j"] = _integrate_rapl(samples, segment["start_ns"], segment["end_ns"])
        except PhysicalAdapterError as error:
            row["uncovered"] = str(error)
        rows.append(row)
    return rows


def per_state(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    totals: dict[str, dict[str, Any]] = {}
    for row in rows:
        total = totals.setdefault(row["state"], {"seconds": 0.0, "gpu_j": 0.0, "cpu_j": 0.0, "segments": 0, "uncovered": 0})
        total["segments"] += 1
        if "uncovered" in row:
            total["uncovered"] += 1
            continue
        total["seconds"] += (row["end_ns"] - row["start_ns"]) / 1e9
        total["gpu_j"] += row["gpu_j"]
        total["cpu_j"] += row["cpu_j"]
    for total in totals.values():
        seconds = total["seconds"]
        total["gpu_w"] = total["gpu_j"] / seconds if seconds > 0 else None
        total["cpu_w"] = total["cpu_j"] / seconds if seconds > 0 else None
    return totals


def readbacks(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The clock/EPP readback of every command of every event, in order."""
    rows = []
    for event in events:
        for result in event.get("result", ()):
            rows.append({"at_us": event["at_us"], "to": event["to"], "reason": event["reason"],
                         "command": " ".join(str(part) for part in result.get("command", ())),
                         "returncode": result.get("returncode"), "readback": result.get("readback")})
    return rows


def analyze(run_dir: Path) -> dict[str, Any]:
    result = json.loads((run_dir / "RESULT.json").read_text())
    paid_start_ns, paid_end_ns = result["paid_start_ns"], result["paid_end_ns"]
    if type(paid_start_ns) is not int or type(paid_end_ns) is not int or paid_end_ns <= paid_start_ns:
        raise ValueError("RESULT paid window is invalid")
    events = load_events(run_dir, result)
    samples = load_samples(run_dir / "resource-samples.jsonl")
    segments = integrate_segments(samples, state_segments(events, paid_start_ns, paid_end_ns))
    return {
        "schema": SCHEMA,
        "run_dir": str(run_dir),
        "paid_seconds": (paid_end_ns - paid_start_ns) / 1e9,
        "policy_present": "device_power_events" in result or (run_dir / EVENTS_FILE).is_file(),
        "event_count": len(events),
        "per_state": per_state(segments),
        "segments": segments,
        "readbacks": readbacks(events),
    }


def render(report: Mapping[str, Any]) -> str:
    lines = [f"run {report['run_dir']}: paid {report['paid_seconds']:.1f} s, {report['event_count']} device power events"]
    lines.append(f"{'state':<12} {'seconds':>10} {'gpu_kj':>9} {'gpu_w':>7} {'cpu_kj':>9} {'cpu_w':>7} {'segs':>5} {'uncov':>5}")
    for state, total in sorted(report["per_state"].items()):
        gpu_w = "-" if total["gpu_w"] is None else f"{total['gpu_w']:.1f}"
        cpu_w = "-" if total["cpu_w"] is None else f"{total['cpu_w']:.1f}"
        lines.append(f"{state:<12} {total['seconds']:>10.1f} {total['gpu_j'] / 1000:>9.3f} {gpu_w:>7} "
                     f"{total['cpu_j'] / 1000:>9.3f} {cpu_w:>7} {total['segments']:>5} {total['uncovered']:>5}")
    if report["readbacks"]:
        lines.append("readbacks (at_us, to, reason, command -> returncode, readback):")
        for row in report["readbacks"]:
            lines.append(f"  {row['at_us']:>12} {row['to']:<11} {row['reason']:<22} {row['command']} -> "
                         f"{row['returncode']} {json.dumps(row['readback'], sort_keys=True)}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="the runner output directory (RESULT.json)")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)
    report = analyze(args.run_dir)
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
