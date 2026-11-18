#!/usr/bin/env python3
"""Per-run analysis of the host CPU legs: analyze_host.py CAMPAIGN_DIR [--json OUT]
CPU package power = RAPL package energy counter delta / time over the window (CLOCK_MONOTONIC)."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def load_power(path: Path):
    header = path.read_text().splitlines()[0]
    wrap = int(header.split("max_energy_range_uj")[1].split()[0]) if "max_energy_range_uj" in header else 0
    rows = []
    for line in path.read_text().splitlines()[1:]:
        p = line.split()
        if len(p) >= 6:
            rows.append([int(v) for v in p[:6]])
    return rows, wrap


def energy_w(rows, wrap, t0, t1, col=1):
    """Mean watts of counter column col over [t0, t1] (ns); the samples just outside bound the window."""
    inside = [r for r in rows if t0 <= r[0] <= t1 and r[col] >= 0]
    if len(inside) < 2:
        return None
    total = 0
    for a, b in zip(inside, inside[1:]):
        d = b[col] - a[col]
        if d < 0 and wrap > 0:
            d += wrap
        total += d
    return total * 1e-6 / ((inside[-1][0] - inside[0][0]) * 1e-9)


def analyze(run_dir: Path, meta: dict) -> dict:
    tag = meta["tag"]
    lines = [json.loads(line) for line in (run_dir / f"{tag}.jsonl").read_text().splitlines()]
    head = lines[0]
    idle = next(r for r in lines if r["kind"] == "idle")
    measure = next(r for r in lines if r["kind"] == "measure")
    calls = [r for r in lines if r["kind"] == "call" and r["phase"] == "measure"]
    compute_ms = [(c["t_end_ns"] - c["t_start_ns"]) / 1e6 for c in calls]
    rows, wrap = load_power(run_dir / f"{tag}.power")
    m0, m1 = measure["t0_ns"], measure["t1_ns"]
    have_rapl = any(r[1] >= 0 for r in rows)
    run_w = energy_w(rows, wrap, m0, m1) if have_rapl else None
    core_w = energy_w(rows, wrap, m0, m1, col=2) if have_rapl else None
    # idle before the run: the sampler starts 5 s before the bench; the bench then loads weights and idles
    pre_w = energy_w(rows, wrap, rows[0][0], rows[0][0] + int(4e9)) if have_rapl else None
    res_w = energy_w(rows, wrap, idle["t0_ns"] + int(1e9), idle["t1_ns"]) if have_rapl else None
    inside = [r for r in rows if m0 <= r[0] <= m1]
    temps = [r[3] / 1e3 for r in inside if r[3] >= 0]
    freqs = [r[4] / 1e3 for r in inside if r[4] > 0]
    throttle = [r[5] for r in inside if r[5] >= 0]
    fifth = max(1, len(compute_ms) // 5)
    drift = statistics.median(compute_ms[-fifth:]) / statistics.median(compute_ms[:fifth]) - 1
    mb = head["weight_bytes"] / 1e6
    comp = statistics.mean(compute_ms)
    throttle_events = (throttle[-1] - throttle[0]) if throttle else 0
    return {
        "tag": tag, "model": meta["model"], "layers": meta["layers"], "threads": meta["threads"], "mb": mb,
        "calls": len(calls), "compute_ms": comp, "compute_p50_ms": statistics.median(compute_ms),
        "compute_p90_ms": sorted(compute_ms)[int(0.9 * (len(compute_ms) - 1))],
        "gb_per_s": mb / 1e3 / (comp / 1e3), "watts": run_w, "core_watts": core_w, "idle_w": pre_w,
        "resident_idle_w": res_w,
        "j_per_call": run_w * comp / 1e3 if run_w else None,
        "j_per_gb": run_w * comp / 1e3 / (mb / 1e3) if run_w else None,
        "net_j_per_gb": (run_w - pre_w) * comp / 1e3 / (mb / 1e3) if run_w and pre_w else None,
        "temp_start_c": temps[0] if temps else None, "temp_end_c": temps[-1] if temps else None,
        "temp_max_c": max(temps) if temps else None, "pre_temp_c": meta.get("pre_temp_mc", 0) / 1e3,
        "freq_mhz": statistics.mean(freqs) if freqs else None, "drift": drift,
        "throttle_events": throttle_events, "throttled": throttle_events > 0 or drift > 0.05,
        "cool": meta.get("cool"), "pre_load": meta.get("pre_load"),
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
    print("tag                 MB   calls  comp_ms   GB/s    W    idleW  resW   J/call    J/GB  netJ/GB  temp C pre>start>end  MHz   drift  thr")
    for r in results:
        print(f"{r['tag']:<17} {r['mb']:7.1f} {r['calls']:6d} {r['compute_ms']:8.3f} {r['gb_per_s']:6.1f} "
              f"{r['watts'] or nan:6.1f} {r['idle_w'] or nan:6.1f} {r['resident_idle_w'] or nan:5.1f} "
              f"{r['j_per_call'] or nan:8.4f} {r['j_per_gb'] or nan:7.3f} {r['net_j_per_gb'] or nan:7.3f}  "
              f"{r['pre_temp_c']:.0f}>{r['temp_start_c'] or nan:.0f}>{r['temp_end_c'] or nan:.0f} (max {r['temp_max_c'] or nan:.0f}) "
              f"{r['freq_mhz'] or nan:5.0f} {r['drift']*100:+5.1f}%  {'YES' if r['throttled'] else 'no'}")
    if args.json:
        args.json.write_text(json.dumps(results, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
