#!/usr/bin/env python3
"""Fault-injection series report: one row per run with an injected helper loss (WS13, read-only).

    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.fault_series_report \\
        --run f1=$R/inputs-derived-mpaoa-p6f1flt --run f2=... [--baseline u1=$R/inputs-derived-mpaoa-p5m1new ...] \\
        [--meter f1=$R/meter/p6f1flt-derived-mpaoa/METER.json ...] [--json OUT.json] [--md OUT.md]

A run directory is an arm's inputs directory (``run-eval/run/`` holds RESULT.json, CO_HELPER_LIFECYCLE.json and the
injector's FAULT_INJECTED.json) or the run directory itself. Times are seconds on the run's paid clock (0 =
``paid_start_ns``, host CLOCK_MONOTONIC), which is also the clock of the co-helper launch receipts
(``monotonic_ns``), of ``helper_mask_events`` / ``device_membership_events`` (``at_us``) and of REQUEST_RECOVERED
``failed_at_us``. Per run:

* injection: FAULT_INJECTED status/condition and the signal time (``t=``/``active=`` conditions: the launch
  anchor + elapsed seconds + the wall gap between the condition and the signal; ``request=``: unknown);
* detection: the first HELPER_MASKED_OUT / DEVICE_QUARANTINED / REQUEST_RECOVERED failure at or after the signal,
  and the delay from the signal;
* recovery: REQUEST_RECOVERED rows (requests, tokens discarded, failed-attempt seconds = ``penalty_us``), servers
  retired (SERVER_EXITED; 0 under mask_out) and completed requests;
* re-join: DEVICE_READMITTED and HELPER_REATTACHED after the loss, the lifecycle JOINED receipt, and the time from
  the loss to the re-admission;
* energy: fleet energy (tools/fleet_energy.py: desktop CPU+GPU + both phones) and, with ``--baseline`` runs of the
  same configuration without a fault, the difference from their mean.

g11 (adb, 2026-09-26) is the reference shape: masked at 505.5 s, two in-flight Qwen requests recovered on the same
server (71/72 tokens discarded, 47.0/47.2 s failed-attempt time), no server stop, Pixel re-attached 64.7 s later.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

_REPO = Path(__file__).resolve().parents[5]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research_dev.scheduler.campaigns.burstgpt.tools import fleet_energy  # noqa: E402

SCHEMA = "ws13-fault-series-report-v1"
DEVICE = "pixel10pro-phone"


def run_directory(path: Path) -> Path:
    for candidate in (path / "run-eval" / "run", path):
        if (candidate / "RESULT.json").is_file():
            return candidate
    raise FileNotFoundError(f"no RESULT.json under {path}")


def _read(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def _seconds(value_us: object) -> float | None:
    return value_us / 1e6 if type(value_us) is int else None


def injection_time_s(fault: Mapping[str, Any] | None, paid_start_ns: int) -> float | None:
    """Signal time on the paid clock for ``t=``/``active=`` conditions (None for ``request=`` or no record)."""
    if not fault:
        return None
    condition = fault.get("condition") or {}
    origin, elapsed = condition.get("trace_origin_monotonic_ns"), condition.get("elapsed_s")
    if type(origin) is not int or not isinstance(elapsed, (int, float)):
        return None
    stamps = fault.get("timestamps") or {}
    gap = 0.0
    if isinstance(stamps.get("signal_sent_wall_s"), (int, float)) and isinstance(
            stamps.get("condition_met_wall_s"), (int, float)):
        gap = max(0.0, stamps["signal_sent_wall_s"] - stamps["condition_met_wall_s"])
    return (origin - paid_start_ns) / 1e9 + elapsed + gap


def fault_row(label: str, directory: Path, *, device: str = DEVICE, meter: Path | None = None,
              energy: bool = True) -> dict[str, Any]:
    run = run_directory(directory)
    result = _read(run / "RESULT.json", {})
    fault = _read(run / "FAULT_INJECTED.json")
    lifecycle = _read(run / "CO_HELPER_LIFECYCLE.json", [])
    paid_start = result["paid_start_ns"]
    injected = injection_time_s(fault, paid_start)
    masks = [row for row in result.get("helper_mask_events", ()) if row.get("device") == device]
    membership = [row for row in result.get("device_membership_events", ()) if row.get("device_id") == device]
    recovered = [row for row in result.get("request_recovered_events", ()) if row.get("kind") == "REQUEST_RECOVERED"]
    after = (lambda seconds: seconds is not None and (injected is None or seconds >= injected - 1.0))
    masked = [_seconds(row.get("at_us")) for row in masks if row.get("kind") == "HELPER_MASKED_OUT"]
    quarantined = [_seconds(row.get("at_us")) for row in membership if row.get("kind") == "DEVICE_QUARANTINED"]
    failed = [_seconds(row.get("failed_at_us")) for row in recovered]
    candidates = [value for value in (*masked, *quarantined, *failed) if after(value)]
    detected = min(candidates) if candidates else None
    readmitted = [_seconds(row.get("at_us")) for row in membership if row.get("kind") == "DEVICE_READMITTED"]
    reattached = [_seconds(row.get("at_us")) for row in masks if row.get("kind") == "HELPER_REATTACHED"]
    joined = [(row["monotonic_ns"] - paid_start) / 1e9 for row in lifecycle
              if isinstance(row, dict) and row.get("kind") == "JOINED" and row.get("device_id") == device
              and type(row.get("monotonic_ns")) is int]
    readmit = next((value for value in sorted(v for v in readmitted if v is not None)
                    if detected is not None and value >= detected), None)
    requests = result.get("request_results", ())
    row: dict[str, Any] = {
        "label": label, "directory": str(directory), "status": result.get("status"),
        "fault": None if not fault else {
            "status": fault.get("status"), "when": fault.get("when"), "signalled_pids": fault.get("signalled_pids"),
            "reason": fault.get("reason")},
        "injected_s": injected,
        "detected_s": detected,
        "detection_delay_s": None if detected is None or injected is None else detected - injected,
        "masked_out_s": min((v for v in masked if after(v)), default=None),
        "quarantined_s": min((v for v in quarantined if after(v)), default=None),
        "recovered_requests": [row.get("request") for row in recovered],
        "tokens_discarded": sum(int(row.get("tokens_discarded") or 0) for row in recovered),
        "failed_attempt_s": sum((row.get("penalty_us") or 0) for row in recovered) / 1e6,
        "recovery_actions": sorted({str(row.get("quarantine_action")) for row in recovered}),
        "servers_retired": len(result.get("server_exited_events", ())),
        "readmitted_s": readmit,
        "rejoin_after_loss_s": None if readmit is None or detected is None else readmit - detected,
        "reattached_s": min((v for v in reattached if v is not None and (detected is None or v >= detected)),
                            default=None),
        "joined_s": min((v for v in joined if detected is None or v >= detected), default=None),
        "requests": len(requests),
        "completed": sum(1 for row in requests if row.get("completion")),
        "duration_s": (result["paid_end_ns"] - paid_start) / 1e9,
    }
    if energy:
        report = fleet_energy.account_run(label, directory, meter_path=meter)
        row["fleet_kj"] = None if report["fleet_j"] is None else report["fleet_j"] / 1e3
        row["host_kj"] = None if report["desktop"]["host_j"] is None else report["desktop"]["host_j"] / 1e3
    return row


def build_report(runs: Mapping[str, Path], baselines: Mapping[str, Path], meters: Mapping[str, Path],
                 *, device: str = DEVICE, energy: bool = True) -> dict[str, Any]:
    rows = [fault_row(label, path, device=device, meter=meters.get(label), energy=energy)
            for label, path in runs.items()]
    base = []
    if energy:
        for label, path in baselines.items():
            report = fleet_energy.account_run(label, path, meter_path=meters.get(label))
            base.append({"label": label, "fleet_kj": None if report["fleet_j"] is None else report["fleet_j"] / 1e3})
        values = [row["fleet_kj"] for row in base if row["fleet_kj"] is not None]
        mean = statistics.mean(values) if values else None
        for row in rows:
            row["fleet_delta_vs_baseline_kj"] = (None if mean is None or row.get("fleet_kj") is None
                                                 else row["fleet_kj"] - mean)
    return {"schema": SCHEMA, "device": device, "runs": rows, "baselines": base}


def _fmt(value: object, digits: int = 1) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown(report: Mapping[str, Any]) -> str:
    lines = ["| run | fault | injected s | detected (+delay) s | recovered | tokens discarded | failed-attempt s "
             "| servers retired | re-admitted (+after loss) s | completed | fleet kJ (vs baseline) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in report["runs"]:
        fault = row["fault"] or {}
        when = fault.get("when") or {}
        lines.append(
            f"| {row['label']} | {fault.get('status', 'NO RECORD')} {when.get('kind', '')}={_fmt(when.get('value'))} "
            f"| {_fmt(row['injected_s'])} | {_fmt(row['detected_s'])} (+{_fmt(row['detection_delay_s'], 2)}) "
            f"| {len(row['recovered_requests'])} {','.join(map(str, row['recovered_requests']))} "
            f"| {row['tokens_discarded']} | {_fmt(row['failed_attempt_s'])} | {row['servers_retired']} "
            f"| {_fmt(row['readmitted_s'])} (+{_fmt(row['rejoin_after_loss_s'])}) "
            f"| {row['completed']}/{row['requests']} "
            f"| {_fmt(row.get('fleet_kj'))} ({_fmt(row.get('fleet_delta_vs_baseline_kj'))}) |")
    return "\n".join(lines) + "\n"


def _specs(values: Sequence[str]) -> dict[str, Path]:
    specs = {}
    for value in values:
        label, separator, path = value.partition("=")
        if not separator or not label or label in specs:
            raise SystemExit(f"bad label=path: {value!r}")
        specs[label] = Path(path)
    return specs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", action="append", required=True, help="label=arm inputs directory (fault run)")
    parser.add_argument("--baseline", action="append", default=[], help="label=undisturbed run of the same arm")
    parser.add_argument("--meter", action="append", default=[], help="label=METER.json kept outside the run")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--no-energy", action="store_true")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--md", type=Path)
    args = parser.parse_args(argv)
    report = build_report(_specs(args.run), _specs(args.baseline), _specs(args.meter), device=args.device,
                          energy=not args.no_energy)
    text = markdown(report)
    if args.json:
        args.json.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    if args.md:
        args.md.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
