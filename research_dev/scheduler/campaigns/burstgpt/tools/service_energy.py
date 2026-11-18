"""Fleet energy of a run restricted to its service window (paid start -> last request completion).

The runner's paid window ends after the post-trace drain and ``end_trace`` teardown, so on short traces the tail
(control probe, phone re-provisioning, session teardown) can be a large share of the paid energy. This tool reports,
per run, the host (RAPL package + NVML board) and phone energy inside [paid_start, last completion] next to the
full-window numbers, using the same integration helpers as ``fleet_energy``.

Usage::

    service_energy.py --run L=<run dir> [--meter L=METER.json] ... [--md OUT] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fleet_energy  # noqa: E402


def host_energy(samples, t0: float, t1: float) -> tuple[float, float, float]:
    rows = sorted((r["rapl_package"]["sample_t_ns"] / 1e9, r["rapl_package"]["energy_uj"],
                   r["rapl_package"]["max_energy_range_uj"], r["gpu"]["power_mw"] / 1000) for r in samples)
    cpu = gpu = secs = 0.0
    for (ta, ea, ma, ga), (tb, eb, _mb, gb) in zip(rows, rows[1:]):
        if tb <= ta or tb < t0 or ta > t1:
            continue
        de = eb - ea
        if de < 0:
            de += ma
        cpu += de / 1e6
        gpu += (ga + gb) / 2 * (tb - ta)
        secs += tb - ta
    return cpu, gpu, secs


def account(label: str, directory: Path, meter: Path | None) -> dict:
    result = json.loads(fleet_energy.find_result(directory).read_text())
    s0, s1 = result["paid_start_ns"] / 1e9, result["paid_end_ns"] / 1e9
    last = max(s0 + r["replay_arrival_us"] / 1e6 + r["actual_latency_us"] / 1e6 for r in result["request_results"])
    samples = fleet_energy.load_samples(directory) or []
    cpu_s, gpu_s, _ = host_energy(samples, s0, last)
    cpu_t, gpu_t, _ = host_energy(samples, last, s1)
    phones_service = phones_full = 0.0
    phones = {}
    meter_path = meter or fleet_energy.find_meter(directory)
    if meter_path and meter_path.is_file():
        data = json.loads(meter_path.read_text())
        for phone, anchor in sorted(data.items()):
            if not isinstance(anchor, dict):
                continue
            local = Path(anchor.get("local", ""))
            path = next((c for c in (local, meter_path.parent / local.name, directory / local.name) if c.is_file()), None)
            if path is None:
                continue
            rows = fleet_energy.parse_sampler(path.read_text())
            try:
                svc = fleet_energy.phone_energy(phone, anchor, rows, s0, last, 0.85)
                full = fleet_energy.phone_energy(phone, anchor, rows, s0, s1, 0.85)
            except ValueError:
                continue
            phones[phone] = {"service_j": svc.get("system_j"), "full_j": full.get("system_j")}
            phones_service += svc.get("system_j") or 0.0
            phones_full += full.get("system_j") or 0.0
    return {"label": label, "paid_s": s1 - s0, "service_s": last - s0, "tail_s": s1 - last,
            "host_service_j": cpu_s + gpu_s, "host_tail_j": cpu_t + gpu_t, "cpu_service_j": cpu_s, "gpu_service_j": gpu_s,
            "phones_service_j": phones_service, "phones_full_j": phones_full, "phones": phones,
            "fleet_service_j": cpu_s + gpu_s + phones_service, "fleet_full_j": cpu_s + gpu_s + cpu_t + gpu_t + phones_full}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", action="append", required=True, help="label=run directory")
    parser.add_argument("--meter", action="append", default=[], help="label=METER.json")
    parser.add_argument("--baseline", help="label used for the saving columns")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--md", type=Path)
    args = parser.parse_args(argv)
    meters = {k: Path(v) for k, v in fleet_energy.parse_specs(args.meter).items()}
    rows = [account(k, Path(v), meters.get(k)) for k, v in fleet_energy.parse_specs(args.run).items()]
    base = next((r for r in rows if r["label"] == args.baseline), None)
    lines = ["| run | paid s | service s | tail s | host service kJ | host tail kJ | phones service kJ | fleet service kJ | vs baseline (service) | fleet full kJ | vs baseline (full) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        sv = "-" if base is None or r is base else f"{100 * (r['fleet_service_j'] / base['fleet_service_j'] - 1):+.1f} %"
        fu = "-" if base is None or r is base else f"{100 * (r['fleet_full_j'] / base['fleet_full_j'] - 1):+.1f} %"
        lines.append(f"| {r['label']} | {r['paid_s']:.0f} | {r['service_s']:.0f} | {r['tail_s']:.0f} | {r['host_service_j'] / 1000:.1f} | {r['host_tail_j'] / 1000:.1f} | "
                     f"{r['phones_service_j'] / 1000:.1f} | {r['fleet_service_j'] / 1000:.1f} | {sv} | {r['fleet_full_j'] / 1000:.1f} | {fu} |")
    text = "\n".join(lines) + "\n"
    if args.json:
        args.json.write_text(json.dumps({"schema": "service-energy-v1", "baseline": args.baseline, "runs": rows}, indent=1, default=fleet_energy._json_default))
    if args.md:
        args.md.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
