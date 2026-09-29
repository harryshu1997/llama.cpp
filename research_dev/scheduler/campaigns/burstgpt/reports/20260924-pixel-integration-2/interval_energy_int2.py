#!/usr/bin/env python3
"""Measured host energy (RAPL package + NVML board) inside per-model execution windows of a trace run.

    python3 interval_energy_int2.py --arm label=<run dir with RESULT.json + resource-samples.jsonl> ...

For every model, the window runs from the first execution start to the last execution end of that
model's requests (execution receipts, microseconds from paid start). It separates the phone-assisted
decode work of one model from model loads and waits, which differ between single runs (page cache).
Uses the interpolation helpers of ../20260924-coherent-policy-coalesced/phase_energy.py.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "20260924-coherent-policy-coalesced"))
from phase_energy import gpu_energy, load, rapl_at  # noqa: E402


def model_key(model_id):
    return "qwen" if "qwen" in model_id else "gemma" if "gemma" in model_id else "llama"


def windows(result):
    spans = {}
    for row in result["request_results"]:
        receipt = (row.get("completion") or {}).get("execution_receipt") or {}
        if receipt.get("started_us") and receipt.get("finished_us"):
            key = model_key(row["model_id"])
            start, end = spans.get(key, (receipt["started_us"], receipt["finished_us"]))
            spans[key] = (min(start, receipt["started_us"]), max(end, receipt["finished_us"]))
    return spans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = {}
    for spec in args.arm:
        label, _, path = spec.partition("=")
        result, rapl, gpu = load(path)
        origin = result["paid_start_ns"]
        rows = {}
        for model, (start, end) in sorted(windows(result).items()):
            a, b = origin + start * 1000, origin + end * 1000
            cpu_j = (rapl_at(rapl, b) - rapl_at(rapl, a)) / 1e6
            gpu_j = gpu_energy(gpu, a, b)
            rows[model] = {"start_s": round(start / 1e6, 1), "end_s": round(end / 1e6, 1),
                           "seconds": round((end - start) / 1e6, 1), "cpu_kj": round(cpu_j / 1e3, 3),
                           "gpu_kj": round(gpu_j / 1e3, 3), "host_kj": round((cpu_j + gpu_j) / 1e3, 3),
                           "host_w": round((cpu_j + gpu_j) / ((end - start) / 1e6), 1)}
        report[label] = rows
        print(label, json.dumps(rows))
    if args.out:
        args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
