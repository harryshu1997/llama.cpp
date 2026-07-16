#!/usr/bin/env python3
"""Import an existing nvidia-smi CSV power trace as an E2 RealizedTimeline.

Used ONLY as a negative/control case in this checkpoint (CP5). The existing
A6000 trace is a single unmatched GPU_BOARD timeline whose sensor updates far too
slowly to meet the frozen gate. This importer exists so that rejection is
demonstrated by the real gate on the real bytes, rather than asserted in prose.

It performs NO measurement. It reads an artifact that already exists.

Input format (nvidia-smi --format=csv,noheader, as found on disk):
    2026/07/15 00:32:30.718, 25.88, P8, 210, 0
    timestamp, power_W, pstate, clocks_MHz, utilization_pct

Watts are converted to integer milliwatts exactly: the vendor prints 2 decimals,
so W*1000 is an exact integer number of mW. A value with more precision than the
sensor reports would be a fabrication, and a float would defeat the type gate.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

import e2_canon as canon  # noqa: E402
import integrator  # noqa: E402


def parse_csv(path):
    """Parse the nvidia-smi CSV into integer (us, mW) samples plus pstates."""
    with open(path, encoding="ascii", newline="") as handle:
        rows = [row for row in csv.reader(handle) if row]
    if len(rows) < 2:
        raise integrator.TimelineError("E_SCHEMA", f"{path} has {len(rows)} rows")
    samples, pstates, utils = [], [], []
    base = None
    for index, row in enumerate(rows):
        if len(row) != 5:
            raise integrator.TimelineError(
                "E_SCHEMA", f"row {index} has {len(row)} fields, expected 5")
        stamp = datetime.datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
        if base is None:
            base = stamp
        delta = stamp - base
        offset_us = ((delta.days * 86400 + delta.seconds) * 1_000_000
                     + delta.microseconds)
        watts = row[1].strip()
        # Exact decimal -> integer mW. No float rounding: the vendor prints
        # hundredths of a watt, so scaling by 1000 is exact in decimal.
        if re.fullmatch(r"[0-9]+(?:\.[0-9]{1,2})?", watts) is None:
            raise integrator.TimelineError(
                "E_SCHEMA", f"row {index} has invalid power value {watts!r}")
        if "." in watts:
            whole, frac = watts.split(".", 1)
        else:
            whole, frac = watts, ""
        frac = frac.ljust(3, "0")
        power_mw = int(whole) * 1000 + int(frac)
        samples.append([offset_us, power_mw])
        pstates.append(row[2].strip())
        utils.append(int(row[4].strip()))
    return samples, pstates, utils


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="import an nvidia-smi CSV trace as an E2 timeline (negative "
                    "control only)")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--timeline-id", default="tl.imported.a6000")
    parser.add_argument("--role", default="OPTIMIZED_SERVER_ONLY_CONTROL")
    parser.add_argument("--json", action="store_true",
                        help="emit a machine-readable report")
    args = parser.parse_args(argv)

    report = {"csv": args.csv, "accepted": False, "failures": []}
    try:
        samples, pstates, utils = parse_csv(args.csv)
        normalized = integrator.normalize_samples(samples)
    except (integrator.TimelineError, ValueError, OSError) as exc:
        report["failures"].append(str(exc))
        print(json.dumps(report, sort_keys=True) if args.json
              else f"E2_IMPORT_REJECTED: {exc}")
        return 1

    updates = integrator.independent_updates(normalized)
    gap = integrator.max_gap_us(normalized)
    span = normalized[-1][0] - normalized[0][0]
    report.update({
        "rows": len(normalized),
        "independent_updates": updates,
        "span_us": span,
        "max_gap_us": gap,
        "distinct_pstates": sorted(set(pstates)),
        "effective_update_hz_milli": (updates * 1_000_000_000 // span) if span else 0,
        "min_independent_updates_gate": integrator.MIN_INDEPENDENT_UPDATES,
        "scope": "GPU_BOARD",
        "instrument_kind": "NVML_BOARD",
    })

    # Every gate this artifact must clear, applied honestly.
    try:
        integrator.gate_samples(normalized, "NVML_BOARD")
    except integrator.TimelineError as exc:
        report["failures"].append(str(exc))
    try:
        integrator.check_scope("NVML_BOARD", "SERVER_WALL")
    except integrator.TimelineError as exc:
        report["failures"].append(f"(promotion attempt) {exc}")
    if len(set(pstates)) > 1:
        report["failures"].append(
            f"E_STATUS_CHANGE: the device changed power state {sorted(set(pstates))} "
            f"inside the trace; a single window cannot span them")
    report["failures"].append(
        "E_PAIRS: a single timeline is not a matched comparison; E2 requires a "
        f"control/treatment pair and at least {integrator.MIN_PAIRS} balanced "
        "repetitions")
    report["accepted"] = False

    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(f"rows={report['rows']} independent_updates={updates} "
              f"span_us={span} max_gap_us={gap}")
        print(f"effective update rate ~"
              f"{report['effective_update_hz_milli'] / 1000:.2f} Hz")
        for failure in report["failures"]:
            print(f"E2_IMPORT_REJECTED: {failure}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
