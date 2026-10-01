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

Every report also carries ``transition_costs`` (commands and their measured durations per event
reason) and ``requests`` (time to first token and latency from ``RESULT.request_results``). A run
with ``device_power.arrival_information`` has ``DEVICE_POWER_TELEMETRY.json`` (else
``RESULT.device_power_telemetry``); its report adds ``idle_split`` (energy per state inside and
outside the GPU-idle intervals) and ``prediction_quality``: idle opportunities (idle intervals of
at least ``idle.min_gap_s``), idle seconds captured in IDLE_MIN and missed, missed opportunities
(no IDLE_MIN at all), false idle drops (IDLE_MIN left within ``min_gap_s`` for a reason other than
the end of the trace) and the synchronous restore waits charged to GPU execution starts.

    ... device_power_energy --compare always-on=runA/ online=runB/ oracle=runC/ [--json]

prints the headline columns of several runs side by side.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from research_dev.scheduler.adapters.contracts import PhysicalAdapterError
from research_dev.scheduler.adapters.host_runtime import _integrate_gpu, _integrate_rapl

UNCONTROLLED = "UNCONTROLLED"
EVENTS_FILE = "DEVICE_POWER_EVENTS.json"
TELEMETRY_FILE = "DEVICE_POWER_TELEMETRY.json"
SCHEMA = "s42-device-power-energy-v1"
COMPARE_SCHEMA = "s42-device-power-compare-v1"
# the trace is over: an IDLE_MIN left for these reasons is not a false drop
_END_REASONS = ("end_trace", "close", "close_best_effort")


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


def load_telemetry(run_dir: Path, result: Mapping[str, Any]) -> dict[str, Any] | None:
    """The controller telemetry (file first, then RESULT), None for a run without it."""
    path = run_dir / TELEMETRY_FILE
    row = json.loads(path.read_text()) if path.is_file() else result.get("device_power_telemetry")
    if row is None:
        return None
    if type(row) is not dict or type(row.get("idle_intervals")) is not list or type(row.get("execution_waits")) is not list:
        raise ValueError("device power telemetry is invalid")
    return row


def state_timeline_us(events: Sequence[Mapping[str, Any]], end_us: int) -> list[dict[str, Any]]:
    """``[{state, reason, start_us, end_us}]`` over ``[0, end_us)`` on the RESULT clock; ``reason``
    is the entry event's reason (OFF / ``before_first_event`` before any event)."""
    state, reason, cursor = "OFF", "before_first_event", 0
    rows: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda row: row["at_us"]):
        at_us = max(0, min(end_us, event["at_us"]))
        if at_us > cursor:
            rows.append({"state": state, "reason": reason, "start_us": cursor, "end_us": at_us})
            cursor = at_us
        state, reason = event["to"], event["reason"]
    if end_us > cursor:
        rows.append({"state": state, "reason": reason, "start_us": cursor, "end_us": end_us})
    return rows


def idle_min_episodes(events: Sequence[Mapping[str, Any]], end_us: int) -> list[dict[str, Any]]:
    """Every stay in IDLE_MIN: ``{start_us, end_us, dwell_us, entry_reason, exit_reason}``
    (``exit_reason`` None when the window ends first)."""
    episodes: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for event in sorted(events, key=lambda row: row["at_us"]):
        at_us = max(0, min(end_us, event["at_us"]))
        if current is None and event["to"] == "IDLE_MIN":
            current = {"start_us": at_us, "entry_reason": event["reason"]}
        elif current is not None and event["to"] != "IDLE_MIN":
            current.update(end_us=at_us, exit_reason=event["reason"])
            episodes.append(current)
            current = None
    if current is not None:
        current.update(end_us=end_us, exit_reason=None)
        episodes.append(current)
    for row in episodes:
        row["dwell_us"] = row["end_us"] - row["start_us"]
    return episodes


def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def prediction_quality(events: Sequence[Mapping[str, Any]], telemetry: Mapping[str, Any], end_us: int,
                       horizon_us: int | None = None) -> dict[str, Any]:
    """Idle opportunities, captured/missed idle time, false idle drops and restore waits; the
    horizon (smallest idle worth a drop) is the policy's ``min_gap_s`` unless given."""
    if horizon_us is None:
        horizon_us = telemetry.get("horizon_us")
    horizon_us = 0 if horizon_us is None else int(horizon_us)
    episodes = idle_min_episodes(events, end_us)
    intervals = []
    for row in telemetry["idle_intervals"]:
        start, stop = max(0, int(row["start_us"])), min(end_us, int(row["end_us"]))
        if stop <= start:
            continue
        captured = sum(_overlap(start, stop, episode["start_us"], episode["end_us"]) for episode in episodes)
        first_drop = min((max(start, episode["start_us"]) for episode in episodes
                          if _overlap(start, stop, episode["start_us"], episode["end_us"]) > 0), default=None)
        intervals.append({"start_us": start, "end_us": stop, "length_us": stop - start, "end_reason": row["end_reason"],
                          "captured_us": captured, "drop_delay_us": None if first_drop is None else first_drop - start,
                          "opportunity": stop - start >= horizon_us})
    opportunities = [row for row in intervals if row["opportunity"]]
    false_drops = [row for row in episodes
                   if row["exit_reason"] is not None and row["exit_reason"] not in _END_REASONS
                   and row["dwell_us"] < horizon_us]
    waits = telemetry["execution_waits"]
    charged = [row for row in waits if row.get("synchronous")]
    return {
        "horizon_us": horizon_us,
        "idle_intervals": len(intervals),
        "idle_us": sum(row["length_us"] for row in intervals),
        "opportunities": len(opportunities),
        "opportunity_us": sum(row["length_us"] for row in opportunities),
        "captured_us": sum(row["captured_us"] for row in intervals),
        "captured_opportunity_us": sum(row["captured_us"] for row in opportunities),
        "missed_opportunity_us": sum(row["length_us"] - row["captured_us"] for row in opportunities),
        "missed_opportunities": sum(1 for row in opportunities if row["captured_us"] == 0),
        "idle_min_episodes": len(episodes),
        "idle_min_entry_reasons": _count(row["entry_reason"] for row in episodes),
        "false_idle_drops": len(false_drops),
        "false_idle_drop_exit_reasons": _count(row["exit_reason"] for row in false_drops),
        "false_idle_drop_us": sum(row["dwell_us"] for row in false_drops),
        "drop_delay_us": sorted(row["drop_delay_us"] for row in opportunities if row["drop_delay_us"] is not None),
        "execution_starts": len(waits),
        "restore_waits": len(charged),
        "restore_wait_us": sum(int(row["wait_us"]) for row in charged),
        "restore_wait_max_us": max((int(row["wait_us"]) for row in charged), default=0),
        "restore_waits_by_request": [{"request_id": row["request_id"], "wait_us": row["wait_us"],
                                      "state_before": row["state_before"]} for row in charged],
        "observed_arrivals": len(telemetry.get("observed_arrivals", ())),
        "intervals": intervals,
        "idle_min_episode_rows": episodes,
    }


def _count(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))


def transition_costs(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Commands per event reason and their measured durations (``sudo`` + ``nvidia-smi`` wall
    time, including the readback query that follows every successful command)."""
    by_reason: dict[str, dict[str, Any]] = {}
    for event in events:
        if event["reason"] in ("capability_probe", "capability_probe_failed"):
            continue
        row = by_reason.setdefault(event["reason"], {"events": 0, "commands": 0, "command_us": 0, "max_event_us": 0})
        results = event.get("result", ())
        duration = sum(int(result.get("duration_us") or 0) for result in results)
        row["events"] += 1
        row["commands"] += len(results)
        row["command_us"] += duration
        row["max_event_us"] = max(row["max_event_us"], duration)
    total_events = sum(row["events"] for row in by_reason.values())
    total_us = sum(row["command_us"] for row in by_reason.values())
    return {"events": total_events, "command_us": total_us,
            "mean_event_us": total_us / total_events if total_events else None,
            "by_reason": dict(sorted(by_reason.items()))}


def _quantile(values: Sequence[float], q: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]


def request_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    """Time to first token (arrival -> first token) and arrival -> completion latency, seconds."""
    rows = result.get("request_results")
    if type(rows) is not list:
        return {"requests": 0}
    paid_start_ns = result["paid_start_ns"]
    ttft, latency = [], []
    for row in rows:
        arrival_us = row.get("replay_arrival_us")
        if type(arrival_us) is not int:
            continue
        if type(row.get("first_token_ns")) is int:
            ttft.append((row["first_token_ns"] - paid_start_ns) / 1e9 - arrival_us / 1e6)
        if type(row.get("actual_latency_us")) is int:
            latency.append(row["actual_latency_us"] / 1e6)
    return {"requests": len(rows), "ttft_p50_s": _quantile(ttft, 0.5), "ttft_p90_s": _quantile(ttft, 0.9),
            "ttft_sum_s": sum(ttft), "latency_p50_s": _quantile(latency, 0.5), "latency_p90_s": _quantile(latency, 0.9)}


def idle_split(samples: Sequence[Mapping[str, Any]], segments: Sequence[Mapping[str, Any]],
               intervals: Sequence[Mapping[str, Any]], paid_start_ns: int) -> dict[str, dict[str, Any]]:
    """GPU/CPU energy per state split into the GPU-idle intervals (``idle``) and the rest."""
    windows = sorted((paid_start_ns + int(row["start_us"]) * 1000, paid_start_ns + int(row["end_us"]) * 1000)
                     for row in intervals if int(row["end_us"]) > int(row["start_us"]))
    pieces = []
    for segment in segments:
        cursor = segment["start_ns"]
        for start, stop in windows:
            start, stop = max(start, segment["start_ns"]), min(stop, segment["end_ns"])
            if stop <= start:
                continue
            if start > cursor:
                pieces.append({"state": segment["state"] + ":busy", "start_ns": cursor, "end_ns": start})
            pieces.append({"state": segment["state"] + ":idle", "start_ns": start, "end_ns": stop})
            cursor = stop
        if segment["end_ns"] > cursor:
            pieces.append({"state": segment["state"] + ":busy", "start_ns": cursor, "end_ns": segment["end_ns"]})
    return per_state(integrate_segments(samples, pieces))


def analyze(run_dir: Path) -> dict[str, Any]:
    result = json.loads((run_dir / "RESULT.json").read_text())
    paid_start_ns, paid_end_ns = result["paid_start_ns"], result["paid_end_ns"]
    if type(paid_start_ns) is not int or type(paid_end_ns) is not int or paid_end_ns <= paid_start_ns:
        raise ValueError("RESULT paid window is invalid")
    events = load_events(run_dir, result)
    samples = load_samples(run_dir / "resource-samples.jsonl")
    segments = integrate_segments(samples, state_segments(events, paid_start_ns, paid_end_ns))
    telemetry = load_telemetry(run_dir, result)
    end_us = (paid_end_ns - paid_start_ns) // 1000
    report = {
        "schema": SCHEMA,
        "run_dir": str(run_dir),
        "paid_seconds": (paid_end_ns - paid_start_ns) / 1e9,
        "policy_present": "device_power_events" in result or (run_dir / EVENTS_FILE).is_file(),
        "event_count": len(events),
        "per_state": per_state(segments),
        "segments": segments,
        "readbacks": readbacks(events),
        "transition_costs": transition_costs(events),
        "requests": request_summary(result),
        "arrival_information": None if telemetry is None else telemetry.get("arrival_information"),
    }
    if telemetry is not None:
        report["idle_split"] = idle_split(samples, [row for row in segments if "uncovered" not in row],
                                          telemetry["idle_intervals"], paid_start_ns)
        report["prediction_quality"] = prediction_quality(events, telemetry, end_us)
    return report


def compare(runs: Sequence[tuple[str, Path]]) -> dict[str, Any]:
    """Headline columns of several runs (for example always-on / online / oracle)."""
    rows = []
    for label, run_dir in runs:
        report = analyze(run_dir)
        totals = report["per_state"].values()
        quality = report.get("prediction_quality", {})
        rows.append({
            "label": label,
            "run_dir": str(run_dir),
            "arrival_information": report["arrival_information"],
            "paid_seconds": report["paid_seconds"],
            "gpu_kj": sum(row["gpu_j"] for row in totals) / 1000,
            "cpu_kj": sum(row["cpu_j"] for row in totals) / 1000,
            "state_seconds": {state: row["seconds"] for state, row in sorted(report["per_state"].items())},
            "captured_idle_s": quality.get("captured_us", 0) / 1e6 if quality else None,
            "missed_opportunity_s": quality.get("missed_opportunity_us", 0) / 1e6 if quality else None,
            "false_idle_drops": quality.get("false_idle_drops") if quality else None,
            "restore_wait_s": quality.get("restore_wait_us", 0) / 1e6 if quality else None,
            "command_s": report["transition_costs"]["command_us"] / 1e6,
            "ttft_p50_s": report["requests"].get("ttft_p50_s"),
            "latency_p50_s": report["requests"].get("latency_p50_s"),
        })
    return {"schema": COMPARE_SCHEMA, "runs": rows}


def render_compare(report: Mapping[str, Any]) -> str:
    def cell(value, digits=1):
        return "-" if value is None else (f"{value:.{digits}f}" if isinstance(value, float) else str(value))
    header = (f"{'arm':<12} {'info':<7} {'paid_s':>8} {'gpu_kj':>8} {'cpu_kj':>8} {'idle_cap_s':>10} "
              f"{'missed_s':>9} {'false':>5} {'wait_s':>7} {'cmd_s':>6} {'ttft50':>7} {'lat50':>7}")
    lines = [header]
    for row in report["runs"]:
        lines.append(f"{row['label']:<12} {cell(row['arrival_information']):<7} {row['paid_seconds']:>8.1f} "
                     f"{row['gpu_kj']:>8.3f} {row['cpu_kj']:>8.3f} {cell(row['captured_idle_s']):>10} "
                     f"{cell(row['missed_opportunity_s']):>9} {cell(row['false_idle_drops']):>5} "
                     f"{cell(row['restore_wait_s'], 3):>7} {cell(row['command_s'], 2):>6} "
                     f"{cell(row['ttft_p50_s']):>7} {cell(row['latency_p50_s']):>7}")
    for row in report["runs"]:
        lines.append(f"  {row['label']}: " + ", ".join(f"{state} {seconds:.1f} s" for state, seconds in row["state_seconds"].items()))
    return "\n".join(lines)


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
    costs = report.get("transition_costs")
    if costs is not None and costs["events"]:
        lines.append(f"transition costs: {costs['events']} events, {costs['command_us'] / 1e6:.2f} s of commands")
        for reason, row in costs["by_reason"].items():
            lines.append(f"  {reason:<22} {row['events']:>4} events {row['commands']:>4} commands "
                         f"{row['command_us'] / 1e3:>9.1f} ms (max {row['max_event_us'] / 1e3:.1f} ms)")
    quality = report.get("prediction_quality")
    if quality is not None:
        lines.append(f"prediction quality ({report['arrival_information']}, horizon {quality['horizon_us'] / 1e6:.1f} s): "
                     f"{quality['idle_intervals']} idle intervals {quality['idle_us'] / 1e6:.1f} s, "
                     f"{quality['opportunities']} opportunities {quality['opportunity_us'] / 1e6:.1f} s, "
                     f"captured {quality['captured_us'] / 1e6:.1f} s, missed {quality['missed_opportunity_us'] / 1e6:.1f} s "
                     f"({quality['missed_opportunities']} missed entirely), false idle drops {quality['false_idle_drops']}, "
                     f"restore waits {quality['restore_waits']}/{quality['execution_starts']} starts "
                     f"{quality['restore_wait_us'] / 1e3:.1f} ms (max {quality['restore_wait_max_us'] / 1e3:.1f} ms)")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--run-dir", type=Path, help="the runner output directory (RESULT.json)")
    target.add_argument("--compare", nargs="+", metavar="LABEL=DIR", help="several runs side by side")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)
    if args.compare is not None:
        runs = []
        for item in args.compare:
            label, separator, path = item.partition("=")
            if not separator or not label or not path:
                parser.error("--compare takes LABEL=DIR")
            runs.append((label, Path(path)))
        report = compare(runs)
        print(json.dumps(report, indent=2, sort_keys=True) if args.json else render_compare(report))
        return 0
    report = analyze(args.run_dir)
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
