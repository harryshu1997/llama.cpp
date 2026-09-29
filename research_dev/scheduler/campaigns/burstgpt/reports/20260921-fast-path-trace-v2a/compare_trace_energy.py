#!/usr/bin/env python3
"""Compare fleet energy and duration across BurstGPT trace runs (RESULT.json or RESULT.json.gz).

    python3 compare_trace_energy.py --run label=path/to/run_dir_or_RESULT ...

Energy comes from RESULT["trace_energy"]["fleet_energy_uj_by_domain"]: cpu-package and gpu-board are
measured (RAPL, NVML), phone-system is the assumed 4.5 W active / 0.875 W idle estimate and is reported
in its own column and never summed silently into "measured host". Deltas are against the first run.
"""
import argparse
import gzip
import json
import pathlib


def load_result(path):
    p = pathlib.Path(path)
    if p.is_dir():
        for name in ("RESULT.json.gz", "RESULT.json"):
            if (p / name).exists():
                p = p / name
                break
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt") as f:
        return json.load(f)


def summarize(label, result):
    te = result.get("trace_energy") or {}
    dom = te.get("fleet_energy_uj_by_domain") or {}
    cpu = dom.get("cpu-package", 0) / 1e9
    gpu = dom.get("gpu-board", 0) / 1e9
    phone = dom.get("phone-system", 0) / 1e9
    duration = result.get("duration_us", 0) / 1e6
    counts = result.get("counts") or {}
    meta = te.get("estimation_metadata") or {}
    return {
        "label": label, "status": result.get("status"), "duration_s": duration,
        "requests": counts.get("requests"), "terminals": counts.get("terminals"), "rejected": counts.get("rejected"),
        "attempts": counts.get("execution_attempts"),
        "cpu_kj": cpu, "gpu_kj": gpu, "host_kj": cpu + gpu, "phone_kj_assumed": phone, "fleet_kj": cpu + gpu + phone,
        "host_w": (cpu + gpu) * 1e3 / duration if duration else None,
        "phone_active_s": (meta.get("phone_active_time_ns") or 0) / 1e9,
        "selection_mode": (result.get("execution_identity") or {}).get("selection_mode") or result.get("selection_mode"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="label=path")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = []
    for spec in args.run:
        label, _, path = spec.partition("=")
        rows.append(summarize(label, load_result(path)))
    base = rows[0]
    print(f"{'run':28s} {'status':6s} {'dur s':>8s} {'req':>4s} {'CPU kJ':>8s} {'GPU kJ':>8s} {'host kJ':>8s} {'host W':>7s} {'phone kJ*':>9s} {'fleet kJ':>9s} {'host vs first':>13s} {'fleet vs first':>14s}")
    for r in rows:
        dh = (r["host_kj"] / base["host_kj"] - 1) * 100 if base["host_kj"] else 0
        df = (r["fleet_kj"] / base["fleet_kj"] - 1) * 100 if base["fleet_kj"] else 0
        print(f"{r['label']:28s} {str(r['status']):6s} {r['duration_s']:8.0f} {str(r['requests']):>4s} {r['cpu_kj']:8.1f} {r['gpu_kj']:8.1f} {r['host_kj']:8.1f} {r['host_w'] or 0:7.1f} {r['phone_kj_assumed']:9.2f} {r['fleet_kj']:9.1f} {dh:+12.1f}% {df:+13.1f}%")
    print("* phone energy is assumed (4.5 W active / 0.875 W idle), not measured")
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
