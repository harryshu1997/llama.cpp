#!/usr/bin/env python3
"""Per-run analysis of the host GPU leg: analyze_gpu.py CAMPAIGN_DIR [--json OUT]
GPU board power = NVML instantaneous board power (nvidia-smi power.draw.instant, 10 Hz), time-weighted mean over the
window; the 1-s NVML average (power.draw.average) over the same window is the cross-check. The CPU package power
(RAPL) over the same window is reported next to it: the host thread that feeds the GPU, not counted as device energy."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from analyze_host import energy_w, load_power

# NVML clock event reasons. A thermal or hardware slowdown is throttling; the software power cap (0x4) means the
# board runs at its configured power limit, which is its normal operating point under this load (reported separately).
SLOWDOWN_BITS = {0x8: "hw_slowdown", 0x20: "sw_thermal", 0x40: "hw_thermal", 0x80: "hw_power_brake"}
POWER_CAP_BIT = 0x4


def load_gpu(path: Path):
    rows = []
    for line in path.read_text().splitlines()[1:]:
        p = line.split(",")
        if len(p) < 12:
            continue
        try:
            rows.append({"t": int(p[0]), "inst": float(p[1]), "avg": float(p[2]), "temp": float(p[3]), "util": float(p[4]),
                         "mem_util": float(p[5]), "mem_mib": float(p[6]), "gr_mhz": float(p[7]), "mem_mhz": float(p[8]),
                         "pstate": p[9], "reasons": int(p[10], 16), "fan": float(p[11]) if p[11].replace(".", "").isdigit() else -1})
        except ValueError:
            continue
    return rows


def mean_w(rows, t0, t1, key="inst"):
    inside = [r for r in rows if t0 <= r["t"] <= t1]
    if len(inside) < 2:
        return None
    total = sum((b["t"] - a["t"]) * (a[key] + b[key]) / 2 for a, b in zip(inside, inside[1:]))
    return total / (inside[-1]["t"] - inside[0]["t"])


def analyze(run_dir: Path, meta: dict) -> dict:
    tag = meta["tag"]
    lines = [json.loads(line) for line in (run_dir / f"{tag}.jsonl").read_text().splitlines()]
    head = lines[0]
    idle = next(r for r in lines if r["kind"] == "idle")
    measure = next(r for r in lines if r["kind"] == "measure")
    calls = [r for r in lines if r["kind"] == "call" and r["phase"] == "measure"]
    compute_ms = [(c["t_end_ns"] - c["t_start_ns"]) / 1e6 for c in calls]
    gpu = load_gpu(run_dir / f"{tag}.gpu")
    m0, m1 = measure["t0_ns"], measure["t1_ns"]
    run_w = mean_w(gpu, m0, m1)
    run_w_avg = mean_w(gpu, m0 + int(1e9), m1, "avg")          # trailing 1-s average: skip its first second
    pre_w = mean_w(gpu, gpu[0]["t"], gpu[0]["t"] + int(4e9))    # sampler starts 5 s before the bench: no CUDA context yet
    res_w = mean_w(gpu, idle["t0_ns"] + int(2e9), idle["t1_ns"])  # weights resident in GPU memory, no work
    inside = [r for r in gpu if m0 <= r["t"] <= m1]
    reasons = 0
    for r in inside:
        reasons |= r["reasons"]
    slow = [name for bit, name in SLOWDOWN_BITS.items() if reasons & bit]
    capped_share = sum(1 for r in inside if r["reasons"] & POWER_CAP_BIT) / len(inside)
    rows, wrap = load_power(run_dir / f"{tag}.power")
    have_rapl = any(r[1] >= 0 for r in rows)
    cpu_w = energy_w(rows, wrap, m0, m1) if have_rapl else None
    cpu_idle_w = energy_w(rows, wrap, rows[0][0], rows[0][0] + int(4e9)) if have_rapl else None
    fifth = max(1, len(compute_ms) // 5)
    drift = statistics.median(compute_ms[-fifth:]) / statistics.median(compute_ms[:fifth]) - 1
    mb = head["weight_bytes"] / 1e6
    comp = statistics.mean(compute_ms)
    gb = mb / 1e3
    return {
        "tag": tag, "model": meta["model"], "layers": meta["layers"], "backend": head.get("backend"), "mb": mb,
        "calls": len(calls), "compute_ms": comp, "compute_p50_ms": statistics.median(compute_ms),
        "compute_p90_ms": sorted(compute_ms)[int(0.9 * (len(compute_ms) - 1))],
        "gb_per_s": gb / (comp / 1e3), "watts": run_w, "watts_nvml_1s_average": run_w_avg, "idle_w": pre_w,
        "resident_idle_w": res_w,
        "j_per_call": run_w * comp / 1e3, "j_per_gb": run_w * comp / 1e3 / gb,
        "net_j_per_gb": (run_w - pre_w) * comp / 1e3 / gb if pre_w is not None else None,
        "cpu_package_w": cpu_w, "cpu_package_idle_w": cpu_idle_w,
        "system_j_per_gb": (run_w + cpu_w) * comp / 1e3 / gb if cpu_w is not None else None,
        "temp_start_c": inside[0]["temp"], "temp_end_c": inside[-1]["temp"], "temp_max_c": max(r["temp"] for r in inside),
        "pre_temp_c": meta.get("pre_gpu_temp_c"), "gr_mhz": statistics.mean(r["gr_mhz"] for r in inside),
        "mem_mhz": statistics.mean(r["mem_mhz"] for r in inside), "util": statistics.mean(r["util"] for r in inside),
        "mem_util": statistics.mean(r["mem_util"] for r in inside), "mem_mib": max(r["mem_mib"] for r in inside),
        "fan_max": max(r["fan"] for r in inside), "pstate": sorted({r["pstate"] for r in inside}),
        "clock_event_reasons": f"0x{reasons:x}", "slowdown_reasons": slow, "drift": drift,
        "power_capped": capped_share > 0, "power_capped_share": capped_share,
        "throttled": bool(slow) or drift > 0.05, "cool": meta.get("cool"), "pre_load": meta.get("pre_load"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    results = []
    for meta_path in sorted(args.campaign.glob("*-L*.json"), key=lambda p: p.stat().st_mtime):
        meta = json.loads(meta_path.read_text())
        try:
            results.append(analyze(args.campaign, meta))
        except Exception as error:
            print(f"{meta.get('tag')}: {error!r}", file=sys.stderr)
    nan = float("nan")
    print("tag                 MB   calls  comp_ms   GB/s  GPU W (1s avg) idleW  resW   J/call    J/GB  netJ/GB  cpuW  sysJ/GB  GPU C pre>start>end  grMHz memMHz util memutil fan  drift  reasons cap thr")
    for r in results:
        print(f"{r['tag']:<17} {r['mb']:7.1f} {r['calls']:6d} {r['compute_ms']:8.3f} {r['gb_per_s']:6.1f} "
              f"{r['watts']:6.1f} ({r['watts_nvml_1s_average'] or nan:5.1f}) {r['idle_w'] or nan:5.1f} {r['resident_idle_w'] or nan:5.1f} "
              f"{r['j_per_call']:8.4f} {r['j_per_gb']:7.3f} {r['net_j_per_gb'] or nan:7.3f} {r['cpu_package_w'] or nan:5.1f} "
              f"{r['system_j_per_gb'] or nan:7.3f}  {r['pre_temp_c']}>{r['temp_start_c']:.0f}>{r['temp_end_c']:.0f} (max {r['temp_max_c']:.0f}) "
              f"{r['gr_mhz']:5.0f} {r['mem_mhz']:5.0f} {r['util']:4.0f} {r['mem_util']:4.0f} {r['fan_max']:3.0f} {r['drift']*100:+5.1f}%  "
              f"{r['clock_event_reasons']} {r['power_capped_share'] * 100:3.0f}% {'YES' if r['throttled'] else 'no'}")
    if args.json:
        args.json.write_text(json.dumps(results, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
