#!/usr/bin/env python3
"""Measured phone power (USB input + battery discharge) in phone-FFN-active vs idle samples.

Active = a phone FFN call (S41SERVERFFNUSB, host monotonic ns) within +-150 ms of the sample's
host_sample_t_ns. Diagnostic only: the phone is USB powered (500 mA) with the OPLUS 80 % hold, so the
split between USB input and battery changes with the charger state.

    python3 phone_power_windows.py LABEL=RUN_DIR ...
"""
import bisect
import json
import pathlib
import re
import statistics
import sys

LINE = re.compile(r"S41SERVERFFNUSB .* started_ns=(\d+) h2d_completed_ns=\d+ d2h_completed_ns=(\d+) compute_us=\d+")
WINDOW_NS = 150_000_000


def summary(values):
    if not values:
        return "n=0"
    return "n=%d total_mean=%.0f mW (usb %.0f + battery %.0f) charging_fraction=%.2f" % (
        len(values), statistics.fmean(v[0] for v in values), statistics.fmean(v[1] for v in values),
        statistics.fmean(v[2] for v in values), statistics.fmean(1.0 if v[3] else 0.0 for v in values))


def main():
    for spec in sys.argv[1:]:
        label, _, run_dir = spec.partition("=")
        run = pathlib.Path(run_dir)
        calls = []
        for path in run.glob("large-model-*.stderr"):
            for line in path.read_text(errors="replace").splitlines():
                match = LINE.search(line)
                if match:
                    calls.append((int(match.group(1)), int(match.group(2))))
        calls.sort()
        starts = [call[0] for call in calls]
        rows = json.loads((run / "phone-power-diagnostics.json").read_text())["rows"]
        active, idle = [], []
        for row in rows:
            t = row.get("host_sample_t_ns")
            usb = row.get("usb_input_power_mw")
            battery = row.get("battery_discharge_power_mw")
            if not isinstance(t, int) or not isinstance(usb, (int, float)) or not isinstance(battery, (int, float)):
                continue
            index = bisect.bisect_left(starts, t - WINDOW_NS)
            hit = index < len(calls) and calls[index][0] <= t + WINDOW_NS
            (active if hit else idle).append((usb + battery, usb, battery, row.get("charging")))
        span = (calls[-1][1] - calls[0][0]) / 1e9 if calls else 0.0
        print(f"{label}: calls={len(calls)} call_span_s={span:.0f} first_call_ns={calls[0][0] if calls else None} "
              f"samples_ns=[{rows[0].get('host_sample_t_ns') if rows else None}, {rows[-1].get('host_sample_t_ns') if rows else None}]")
        print(f"  active: {summary(active)}")
        print(f"  idle:   {summary(idle)}")


if __name__ == "__main__":
    main()
