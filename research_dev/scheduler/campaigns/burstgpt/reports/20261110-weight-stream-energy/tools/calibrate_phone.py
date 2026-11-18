#!/usr/bin/env python3
"""Session calibration of the OP15 fuel-gauge current against its coulomb counter.

    calibrate_phone.py SESSION_POWER.txt [--json OUT]

ratio = integral(current_now x voltage) / (counter drop x voltage) over the whole session (1 s samples).
The counter is the reference (campaign convention, fleet_energy.py); current_now reads about half of it.
The integral runs between two counter steps that are far apart, so step quantization is < 1 %.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("power", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--capacity-scale", type=float, default=1.0,
                    help="design capacity x SOH / gauge full-charge capacity: converts the counter's mAh to "
                         "design-capacity mAh (OP15 7300/7212 = 1.012; OP12 5400 x 0.97 / 3642 = 1.438)")
    ap.add_argument("--battery-ma-factor", type=float, default=1.0,
                    help="raw battery/current_now -> mA, discharge positive (Pixel: -0.001)")
    ap.add_argument("--runs", type=Path, help="campaign directory: also calibrate over the run windows only "
                                              "(launch - 3 s .. client done + 8 s), where the battery discharges")
    args = ap.parse_args()
    rows = []
    for line in args.power.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        p = line.split()
        d = {"t": float(p[0])}
        for kv in p[1:]:
            k, _, v = kv.partition("=")
            try:
                d[k] = float(v)
            except ValueError:
                pass
        if "cc_uah" in d and "bat_ma" in d:
            d["bat_ma"] *= args.battery_ma_factor
            rows.append(d)
    steps = [i for i in range(1, len(rows)) if rows[i]["cc_uah"] != rows[i - 1]["cc_uah"]]
    if len(steps) < 2:
        print("fewer than two counter steps")
        return 1
    first, last = steps[0], steps[-1]
    sizes = sorted({abs(rows[i]["cc_uah"] - rows[i - 1]["cc_uah"]) for i in steps})
    current_j = 0.0
    counter_j = 0.0
    for i in range(first + 1, last + 1):
        a, b = rows[i - 1], rows[i]
        dt = b["t"] - a["t"]
        current_j += 0.5 * (a["bat_ma"] * a["bat_uv"] + b["bat_ma"] * b["bat_uv"]) * 1e-9 * dt
        counter_j += (a["cc_uah"] - b["cc_uah"]) * 3.6e-3 * 0.5 * (a["bat_uv"] + b["bat_uv"]) * 1e-6
    drop = rows[first]["cc_uah"] - rows[last]["cc_uah"]
    out = {"samples": len(rows), "span_s": rows[last]["t"] - rows[first]["t"], "counter_steps": len(steps),
           "step_sizes_uah": sizes, "counter_drop_mah": drop / 1e3, "counter_j": counter_j,
           "current_integral_j": current_j, "ratio": current_j / counter_j,
           "ratio_uncertainty": 2 * max(sizes) / drop if drop else None,
           "usb_j": sum(0.5 * (rows[i - 1]["usb_ua"] * rows[i - 1]["usb_uv"] + rows[i]["usb_ua"] * rows[i]["usb_uv"]) * 1e-12
                        * (rows[i]["t"] - rows[i - 1]["t"]) for i in range(first + 1, last + 1))}
    if args.runs:
        cur = cnt = 0.0
        windows = 0
        for marks_path in sorted(args.runs.glob("*/marks.txt")):
            marks = dict(line.split(None, 1) for line in marks_path.read_text().splitlines() if " " in line)
            if "client_done_uptime" not in marks:
                continue
            t0, t1 = float(marks["launch_uptime"]) - 3.0, float(marks["client_done_uptime"]) + 8.0
            inside = [r for r in rows if t0 <= r["t"] <= t1]
            if len(inside) < 5:
                continue
            windows += 1
            for a, b in zip(inside, inside[1:]):
                cur += 0.5 * (a["bat_ma"] * a["bat_uv"] + b["bat_ma"] * b["bat_uv"]) * 1e-9 * (b["t"] - a["t"])
                cnt += (a["cc_uah"] - b["cc_uah"]) * 3.6e-3 * 0.5 * (a["bat_uv"] + b["bat_uv"]) * 1e-6
        out["run_windows"] = {"windows": windows, "counter_j": cnt, "current_integral_j": cur,
                              "ratio": cur / cnt if cnt else None,
                              "ratio_uncertainty": (windows ** 0.5) * max(sizes) * 3.6e-3 * 4.0 / cnt if cnt else None}
    best = out.get("run_windows", {}).get("ratio") or out["ratio"]
    out["capacity_scale"] = args.capacity_scale
    out["ratio_for_power"] = best / args.capacity_scale   # current_now x V / true battery power
    print(json.dumps(out, indent=1))
    if args.json:
        args.json.write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
