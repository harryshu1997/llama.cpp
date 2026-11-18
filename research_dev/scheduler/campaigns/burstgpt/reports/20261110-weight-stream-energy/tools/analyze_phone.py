#!/usr/bin/env python3
"""Per-run analysis of the phone legs: analyze_phone.py CAMPAIGN_DIR [--ratio R] [--json OUT]

Phone terminal power = USB input (usb V x I) + battery discharge (battery current_now x voltage / R).
R is the fuel gauge's current-to-counter ratio (OP15 current_now reads ~0.5x the coulomb counter); pass the
session value from calibrate_phone.py. current_now refreshes every ~5.2 s, so a battery reading is used only when
a full refresh interval has passed since the window (or the load) began.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REFRESH_S = 5.4          # OPLUS gauges refresh current_now every ~5.2 s (--refresh-s; Pixel max77779: < 0.5 s)
BATTERY_MA = 1.0         # raw battery/current_now -> mA, discharge positive (--battery-ma-factor; Pixel: -0.001)
HOT_ZONE, SKIN_ZONE = "nsphvx-0", "shell_back"
STANDARD_CAP = {"m0": 1.0, "m6": 2438400 / 4608000}


def parse_power(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        row = {"t": float(parts[0])}
        for kv in parts[1:]:
            k, _, v = kv.partition("=")
            try:
                row[k] = float(v)
            except ValueError:
                pass
        rows.append(row)
    return rows


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def window_power(rows, t0, t1, ratio, clean_from, robust=False):
    """Mean USB W over [t0, t1]; battery W from the readings taken at or after clean_from (one refresh interval
    after the state of the window began)."""
    inside = [r for r in rows if t0 <= r["t"] <= t1]
    if not inside or "usb_ua" not in inside[0]:   # unrooted phone: no power nodes
        return {"usb_w": None, "bat_w": None, "w": None, "samples": len(inside), "battery_samples": 0,
                "battery_values_ma": []}
    usb_values = [r["usb_ua"] * r["usb_uv"] * 1e-12 for r in inside]
    # idle windows: the median rejects the short USB bursts of adb / the sampler start
    usb = (statistics.median(usb_values) if robust else mean(usb_values)) if usb_values else None
    clean = [r for r in inside if r["t"] >= clean_from]
    bat = mean(r["bat_ma"] * BATTERY_MA * 1e-3 * r["bat_uv"] * 1e-6 / ratio for r in clean)
    return {"usb_w": usb, "bat_w": bat, "w": (usb + bat) if usb is not None and bat is not None else None,
            "samples": len(inside), "battery_samples": len(clean),
            "battery_values_ma": sorted({round(r["bat_ma"] * BATTERY_MA) for r in clean})[:12]}


def analyze(run: Path, meta: dict, ratio: float, standard_cap: dict | None) -> dict:
    rows = parse_power(run / "power.txt")
    marks = dict(line.split(None, 1) for line in (run / "marks.txt").read_text().splitlines() if " " in line)
    idle = measure = None
    calls = []
    header = None
    for line in (run / "calls.csv").read_text().splitlines():
        if line.startswith("# idle"):
            idle = {k: int(v) for k, v in (kv.split("=") for kv in line.split()[2:])}
        elif line.startswith("# measure"):
            measure = {k: int(v) for k, v in (kv.split("=") for kv in line.split()[2:])}
        elif line.startswith("phase"):
            header = line.split(",")
        elif line and not line.startswith("#"):
            p = line.split(",")
            calls.append((p[0], int(p[2]), int(p[3]), int(p[4])))
    measured = [c for c in calls if c[0] == "measure"]
    warm_start = calls[0][1] / 1e6
    m0, m1 = measure["t0_us"] / 1e6, measure["t1_us"] / 1e6
    compute_ms = [c[3] / 1e3 for c in measured]
    period_ms = (m1 - m0) * 1e3 / len(measured)
    fifth = max(1, len(compute_ms) // 5)
    drift = statistics.median(compute_ms[-fifth:]) / statistics.median(compute_ms[:fifth]) - 1
    freq_keys = sorted(k for k in rows[-1] if k[0] == "m" and k[1:].isdigit())
    launch = float(marks["launch_uptime"])
    # the phone has been idle for the whole cool-down before the sampler starts (>= 45 s)
    pre = window_power(rows, rows[0]["t"] + 1.0, launch - 0.2, ratio, clean_from=rows[0]["t"] + 1.0, robust=True)
    resident = window_power(rows, idle["t0_us"] / 1e6 + 1.0, idle["t1_us"] / 1e6, ratio,
                            clean_from=idle["t0_us"] / 1e6 + 2 * REFRESH_S, robust=True)
    # a reading shown at t was refreshed within the last interval and covers the interval before that refresh
    run_w = window_power(rows, m0, m1, ratio, clean_from=warm_start + 2 * REFRESH_S)
    inside = [r for r in rows if m0 <= r["t"] <= m1]
    cool_keys = [k for k in rows[0] if k.startswith("cd_")]
    cooling_max = {k[3:]: max(r[k] for r in inside) for k in cool_keys}
    known = ("nsphvx-0", "nsphmx-0", "ddr", "cpu-1-0-0", "cpu-0-0-0", "shell_back",
             "BIG", "BIG_MID", "MID", "LITTLE", "soc_therm", "MEM", "quiet_therm", "nspss-0", "cpu-1-0", "cpu-0-0")
    zones = [z for z in known if z in rows[-1]]
    before = [r for r in rows if r["t"] <= launch]
    temps = {z: {"pre_c": before[-1][z] / 1e3 if before else None, "start_c": inside[0][z] / 1e3,
                 "end_c": inside[-1][z] / 1e3, "max_c": max(r[z] for r in inside) / 1e3} for z in zones}
    # CPU frequency cap: median scaling_max_freq of a policy over the measured window / its largest value in the
    # file. Every OP15 run shows the vendor policy's standard cap from the first second of load (prime cluster
    # policy6 at 2,438 MHz of 4,608; policy0 uncapped); a deeper cap is the status-2 mitigation.
    cpu_cap = {k: statistics.median([r[k] for r in inside if k in r]) / max(r[k] for r in rows if k in r)
               for k in freq_keys}
    deep_cap = standard_cap is not None and any(v < (standard_cap.get(k, 1.0) - 0.01) for k, v in cpu_cap.items())
    cpu_mhz = {"f" + k[1:]: mean(r["f" + k[1:]] for r in inside if "f" + k[1:] in r) / 1e3 for k in freq_keys}
    bus = {k[4:]: {"mean_mhz": mean(r[k] for r in inside if k in r) / 1e3,
                   "values_mhz": sorted({int(r[k] / 1e3) for r in inside if k in r})}
           for k in sorted(rows[-1]) if k.startswith("bus_")}
    status = marks.get("status", "")
    mb = meta["layers"] * meta["layer_bytes"] / 1e6
    comp = mean(compute_ms)
    # throttled = a cooling device engaged or the calls slowed down; the Android thermal status is reported apart
    throttled = any(v > 0 for v in cooling_max.values()) or drift > 0.05 or deep_cap
    android_status = int(status.strip().split(":")[-1]) if ":" in status else -1
    throttled = throttled or android_status >= 2
    out = {
        "tag": meta["tag"], "model": meta["model"], "layers": meta["layers"], "mb": mb,
        "calls": len(measured), "compute_ms": comp, "compute_p50_ms": statistics.median(compute_ms),
        "compute_p90_ms": sorted(compute_ms)[int(0.9 * (len(compute_ms) - 1))], "period_ms": period_ms,
        "gb_per_s": mb / 1e3 / (comp / 1e3), "duty": comp / period_ms,
        "watts": run_w["w"], "usb_w": run_w["usb_w"], "battery_w": run_w["bat_w"],
        "battery_samples": run_w["battery_samples"], "battery_values_ma": run_w["battery_values_ma"],
        "idle_w": pre["w"], "idle_battery_samples": pre["battery_samples"],
        "resident_idle_w": resident["w"], "resident_battery_samples": resident["battery_samples"],
        "j_per_call": run_w["w"] * comp / 1e3 if run_w["w"] is not None else None,
        "j_per_gb": run_w["w"] * comp / 1e3 / (mb / 1e3) if run_w["w"] is not None else None,
        "net_j_per_gb": (run_w["w"] - pre["w"]) * comp / 1e3 / (mb / 1e3)
        if run_w["w"] is not None and pre["w"] is not None else None,
        "drift": drift, "bus": bus, "cooling_max": cooling_max, "cpu_cap": cpu_cap, "cpu_mhz": cpu_mhz, "android_thermal_status": android_status,
        "temps": temps, "hot": temps.get(HOT_ZONE), "skin": temps.get(SKIN_ZONE),
        "hot_zone": HOT_ZONE, "skin_zone": SKIN_ZONE,
        "battery_dc": [inside[0].get("bat_dc", float("nan")), inside[-1].get("bat_dc", float("nan"))],
        "cool": meta.get("cool"), "cool_wait_s": meta.get("cool_wait_s"), "throttled": throttled,
        # the run changed operating point while measured (calls sped up): its power and time windows differ
        "unsteady": drift < -0.05,
        "load_s": float(marks["ready_uptime"]) - launch, "streaming_s": m1 - warm_start,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign", type=Path)
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--cap-policy", choices=["op15", "none"], default="op15",
                    help="op15: flag a CPU cap deeper than the OP15's standing one; none: the phone moves "
                         "scaling_max_freq in normal operation (OnePlus 12), so caps are not a throttle flag")
    ap.add_argument("--battery-ma-factor", type=float, default=1.0)
    ap.add_argument("--refresh-s", type=float, default=5.4)
    ap.add_argument("--hot-zone", default="nsphvx-0")
    ap.add_argument("--skin-zone", default="shell_back")
    args = ap.parse_args()
    global REFRESH_S, BATTERY_MA, HOT_ZONE, SKIN_ZONE
    REFRESH_S, BATTERY_MA, HOT_ZONE, SKIN_ZONE = args.refresh_s, args.battery_ma_factor, args.hot_zone, args.skin_zone
    standard_cap = STANDARD_CAP if args.cap_policy == "op15" else None
    results = []
    for meta_path in sorted(args.campaign.glob("*.json"), key=lambda p: p.stat().st_mtime):
        if meta_path.name in ("RESULTS.json", "CALIBRATION.json"):
            continue
        meta = json.loads(meta_path.read_text())
        run = args.campaign / meta["tag"]
        if not (run / "calls.csv").exists():
            print(f"{meta['tag']}: no calls.csv (rc {meta.get('rc')})", file=sys.stderr)
            continue
        try:
            results.append(analyze(run, meta, args.ratio, standard_cap))
        except Exception as error:  # a broken run must not hide the others
            print(f"{meta['tag']}: {error!r}", file=sys.stderr)
    print("tag                 MB   calls  comp_ms   GB/s  duty   W(usb+bat)       idleW resW  J/call   J/GB  netJ/GB  SoC C pre>start>end  bat C  drift  thr st")
    for r in results:
        t = r["hot"]
        print(f"{r['tag']:<17} {r['mb']:7.1f} {r['calls']:6d} {r['compute_ms']:8.3f} {r['gb_per_s']:6.1f} {r['duty']:5.2f} "
              f"{r['watts'] or float('nan'):5.2f} ({r['usb_w'] or float('nan'):.2f}+{r['battery_w'] or float('nan'):.2f}) "
              f"{r['idle_w'] or float('nan'):5.2f} {r['resident_idle_w'] or float('nan'):5.2f} "
              f"{r['j_per_call'] or float('nan'):7.4f} {r['j_per_gb'] or float('nan'):6.3f} {r['net_j_per_gb'] or float('nan'):6.3f}  "
              f"{t['pre_c']:.0f}>{t['start_c']:.0f}>{t['end_c']:.0f} (max {t['max_c']:.0f})  {r['battery_dc'][1]/10:.1f}  {r['drift']*100:+5.1f}%  {'YES' if r['throttled'] else 'no'}  {r['android_thermal_status']}")
    if args.json:
        args.json.write_text(json.dumps(results, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
