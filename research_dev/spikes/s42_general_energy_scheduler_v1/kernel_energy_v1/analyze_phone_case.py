#!/usr/bin/env python3
"""Integrate one captured OP15 energy window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import energy_common


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--marker-log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    energy_common.require(args.output.is_absolute(), "absolute output")
    energy_common.require(not args.output.exists(), "output already exists")
    metadata = json.loads(args.capture.joinpath("capture.json").read_text())
    energy_common.require(metadata.get("status") == "PASS", "capture status")
    marker_path = args.marker_log or args.capture.joinpath("workload.log")
    marker_text = marker_path.read_text(encoding="ascii")
    try:
        start_ns, end_ns = energy_common.parse_phone_window(marker_text)
        boundary = "workload_markers"
    except energy_common.EnergyError:
        start_ns = metadata.get("start_phone_uptime_ns")
        end_ns = metadata.get("end_phone_uptime_ns")
        energy_common.require(
            type(start_ns) is int and type(end_ns) is int,
            "phone energy window",
        )
        boundary = "adb_command"
    rows = energy_common.read_phone_samples(
        args.capture.joinpath("phone-samples.tsv")
    )
    summary = energy_common.phone_energy_summary(rows, start_ns, end_ns)
    result = {
        **summary,
        "boundary": boundary,
        "case_id": metadata["case_id"],
        "method": "trapezoidal USB input plus battery discharge",
        "schema": "s42-phone-energy-case-v1",
        "serial": metadata["serial"],
        "status": "PASS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(energy_common.canonical(result))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except energy_common.EnergyError as error:
        print(f"S42_PHONE_ANALYZE_ERROR: {error}")
        raise SystemExit(2)
