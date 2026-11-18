#!/usr/bin/env python3
"""Per-run markdown tables: tables.py --phone "NAME=RESULTS.json" --host "NAME=RESULTS.json" ... > RUNS.md"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LABEL = {"qwen25-05b": "Qwen2.5-0.5B", "qwen3-06b": "Qwen3-0.6B", "llama32-1b": "Llama-3.2-1B",
         "qwen3-17b": "Qwen3-1.7B", "gemma4-e2b": "Gemma-4-E2B", "qwen3-4b": "Qwen3-4B", "qwen3-8b": "Qwen3-8B",
         "gemma4-12b": "Gemma-4-12B", "qwen3-14b": "Qwen3-14B", "qwen3-32b": "Qwen3-32B"}


def f(value, spec):
    return "-" if value is None else format(value, spec)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phone", action="append", default=[])
    ap.add_argument("--host", action="append", default=[])
    ap.add_argument("--gpu", action="append", default=[])
    args = ap.parse_args()
    for item in args.phone:
        name, _, path = item.partition("=")
        runs = json.loads(Path(path).read_text())
        print(f"\n### {name}\n")
        print("| Model | Layers | MB | Compute ms | GB/s | Device W | Idle W | Resident idle W | J/call | SoC C start-end | Skin C start-end | Big-core MHz | Android status | Throttled |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|")
        for r in runs:
            npu, shell = r["hot"], r["skin"]
            print(f"| {LABEL[r['model']]} | {r['layers']} | {r['mb']:.0f} | {r['compute_ms']:.3f} | {r['gb_per_s']:.1f} | "
                  f"{f(r['watts'], '.2f')} | {f(r['idle_w'], '.2f')} | {f(r['resident_idle_w'], '.2f')} | {f(r['j_per_call'], '.4f')} | "
                  f"{(str(round(npu['pre_c'])) + '-' + str(round(npu['end_c']))) if npu else '-'} | "
                  f"{(format(shell['start_c'], '.1f') + '-' + format(shell['end_c'], '.1f')) if shell else '-'} | "
                  f"{max(r.get('cpu_mhz', {}).values() or [0]):.0f} | "
                  f"{r['android_thermal_status']} | {'YES' if r['throttled'] else ('no (unsteady: sped up mid-run)' if r.get('unsteady') else 'no')} |")
    for item in args.gpu:
        name, _, path = item.partition("=")
        runs = json.loads(Path(path).read_text())
        print(f"\n### {name}\n")
        print("| Model | Layers | MB | Compute ms | GB/s | GPU board W | Idle W | Resident idle W | J/call | GPU C start-end | Clock MHz | Host CPU package W | At power limit | Throttled |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|")
        for r in runs:
            flag = "YES " + (", ".join(r["slowdown_reasons"]) or format(r["drift"] * 100, "+.0f") + " %") if r["throttled"] else "no"
            print(f"| {LABEL[r['model']]} | {r['layers']} | {r['mb']:.0f} | {r['compute_ms']:.3f} | {r['gb_per_s']:.0f} | "
                  f"{f(r['watts'], '.1f')} | {f(r['idle_w'], '.1f')} | {f(r['resident_idle_w'], '.1f')} | {f(r['j_per_call'], '.4f')} | "
                  f"{r['temp_start_c']:.0f}-{r['temp_end_c']:.0f} | {r['gr_mhz']:.0f} | {f(r['cpu_package_w'], '.1f')} | "
                  f"{'yes' if r.get('power_capped') else 'no'} | {flag} |")
    for item in args.host:
        name, _, path = item.partition("=")
        runs = json.loads(Path(path).read_text())
        print(f"\n### {name}\n")
        print("| Model | Layers | MB | Compute ms | GB/s | Package W | Idle W | Resident idle W | J/call | CPU C start-end | Slowed in run |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
        for r in runs:
            print(f"| {LABEL[r['model']]} | {r['layers']} | {r['mb']:.0f} | {r['compute_ms']:.3f} | {r['gb_per_s']:.1f} | "
                  f"{f(r['watts'], '.1f')} | {f(r['idle_w'], '.1f')} | {f(r['resident_idle_w'], '.1f')} | {f(r['j_per_call'], '.4f')} | "
                  f"{f(r['temp_start_c'], '.0f')}-{f(r['temp_end_c'], '.0f')} | {'YES ' + format(r['drift'] * 100, '+.0f') + ' %' if r['throttled'] else 'no'} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
