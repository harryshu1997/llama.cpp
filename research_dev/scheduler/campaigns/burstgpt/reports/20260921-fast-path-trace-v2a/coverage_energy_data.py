#!/usr/bin/env python3
"""Extract per-model phone token coverage over time and cumulative host energy for a trace pair.

How many tokens the phone served is taken per request from `physical_execution_proof`: the proof
lists the phone calls per released layer, and an assisted decode token issues exactly one call per
layer, so the per-layer call count is that request's assisted token count. The window counter
`completed_phone_calls` is NOT used: it moves with the whole phone, so it credits a request on one
server with calls another server made (checked 2026-09-22 on the m4a5 arm, where it credited Qwen
requests although only the Gemma servers ever called the phone).

When those tokens happened comes from the adaptive decode windows, which carry a token range and a
finish time. A request's assisted tokens are laid onto its phone-policy windows in time order.
Requests without windows contribute their output tokens as unassisted at completion time.

Energy comes from the resource samples: RAPL package energy deltas with counter wrap handling, plus
the NVML board power integrated over the sample interval.

    python3 coverage_energy_data.py --treatment <run dir> --baseline <run dir> --out data.json
"""
from __future__ import annotations

import argparse
import json
import pathlib

MODEL_LABEL = {"qwen": "Qwen3 14B", "gemma": "Gemma 4 12B", "llama": "Llama 3.2 1B"}


def model_key(model_id: str) -> str:
    for prefix in ("qwen", "gemma", "llama"):
        if model_id.startswith(prefix):
            return prefix
    return model_id.split("-")[0]


def assisted_tokens_by_request(result: dict) -> dict[str, int]:
    """Assisted decode token equivalents per request.

    An assisted token issues one phone call per released layer, so calls divided by the number of
    released layers is the token count. Layers do not always carry equal counts, because the runtime
    control can change the released mask at a decode boundary, so the quotient is the work-weighted
    equivalent rather than a count of fully offloaded tokens."""
    tokens = {}
    for row in result["request_results"]:
        proof = row.get("physical_execution_proof") or {}
        by_layer = [entry for entry in proof.get("phone_calls_by_layer") or []
                    if isinstance(entry, dict) and "calls" in entry]
        total = sum(int(entry["calls"]) for entry in by_layer)
        tokens[row["request_id"]] = round(total / len(by_layer)) if by_layer else 0
    return tokens


def coverage_events(run: pathlib.Path) -> dict[str, list[tuple[float, int, int]]]:
    """(finish seconds, decode tokens, assisted tokens) per model."""
    result = json.loads((run / "RESULT.json").read_text())
    model_of = {row["request_id"]: model_key(row["model_id"]) for row in result["request_results"]}
    assisted_of = assisted_tokens_by_request(result)
    windows_of: dict[str, list[tuple[float, int, bool]]] = {}
    store = run / "ADAPTIVE_DECODE_OBSERVATIONS.json"
    if store.exists():
        for group in json.loads(store.read_text())["groups"]:
            for window in group["windows"]:
                count = int(window["token_end"]) - int(window["token_start"])
                if count <= 0:
                    continue
                phone_policy = not (window.get("policy") or {}).get("baseline", True)
                windows_of.setdefault(group["request_id"], []).append(
                    (int(window["finished_at_us"]) / 1e6, count, phone_policy))
    events: dict[str, list[tuple[float, int, int]]] = {}
    for request_id, model in model_of.items():
        windows = sorted(windows_of.get(request_id, []))
        remaining = assisted_of.get(request_id, 0)
        if not windows:
            row = next(row for row in result["request_results"] if row["request_id"] == request_id)
            end = (row.get("completion") or {}).get("actual_end_us")
            if end is None or not row["output_tokens"]:
                continue
            events.setdefault(model, []).append(
                (int(end) / 1e6, int(row["output_tokens"]), min(remaining, int(row["output_tokens"]))))
            continue
        for at_s, count, phone_policy in windows:
            share = min(count, remaining) if phone_policy else 0
            remaining -= share
            events.setdefault(model, []).append((at_s, count, share))
    for rows in events.values():
        rows.sort()
    return events


def coverage_series(run: pathlib.Path) -> list[dict]:
    series = []
    for model, rows in sorted(coverage_events(run).items()):
        points, decoded, assisted = [], 0, 0
        for at_s, tokens, assisted_tokens in rows:
            decoded += tokens
            assisted += assisted_tokens
            points.append([round(at_s, 1), round(100 * assisted / decoded, 1), assisted, decoded])
        series.append({"model": model, "label": MODEL_LABEL.get(model, model),
                       "points": points, "assisted_tokens": assisted, "decode_tokens": decoded})
    return series


def energy_series(run: pathlib.Path) -> dict:
    rows = [json.loads(line) for line in (run / "resource-samples.jsonl").read_text().splitlines() if line.strip()]
    rows.sort(key=lambda row: row["t_ns"])
    start = rows[0]["t_ns"]
    points, cpu_uj, gpu_uj = [], 0.0, 0.0
    for before, after in zip(rows, rows[1:]):
        seconds = (after["t_ns"] - before["t_ns"]) / 1e9
        if seconds <= 0:
            continue
        delta = after["rapl_package"]["energy_uj"] - before["rapl_package"]["energy_uj"]
        if delta < 0:
            delta += before["rapl_package"].get("max_energy_range_uj", 0)
        if delta >= 0:
            cpu_uj += delta
        gpu_uj += after["gpu"]["power_mw"] * 1e3 * seconds
        points.append([round((after["t_ns"] - start) / 1e9, 1),
                       round((cpu_uj + gpu_uj) / 1e9, 3), round(cpu_uj / 1e9, 3), round(gpu_uj / 1e9, 3)])
    return {"points": points[::max(1, len(points) // 400)] + points[-1:],
            "host_kj": points[-1][1], "cpu_kj": points[-1][2], "gpu_kj": points[-1][3],
            "duration_s": points[-1][0]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--treatment", type=pathlib.Path, required=True)
    parser.add_argument("--baseline", type=pathlib.Path, required=True)
    parser.add_argument("--treatment-label", default="Phone-assisted")
    parser.add_argument("--baseline-label", default="Desktop only")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    treatment_energy = energy_series(args.treatment)
    baseline_energy = energy_series(args.baseline)
    payload = {
        "coverage": coverage_series(args.treatment),
        "energy": {"treatment": {"label": args.treatment_label, **treatment_energy},
                   "baseline": {"label": args.baseline_label, **baseline_energy}},
        "saving_percent": round(100 * (baseline_energy["host_kj"] - treatment_energy["host_kj"])
                                / baseline_energy["host_kj"], 1),
        "sources": {"treatment": str(args.treatment), "baseline": str(args.baseline)},
    }
    args.out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"coverage models: {[row['model'] for row in payload['coverage']]}")
    for row in payload["coverage"]:
        print(f"  {row['label']:<14} {row['assisted_tokens']:>5} / {row['decode_tokens']:>5} tokens"
              f" = {100 * row['assisted_tokens'] / max(1, row['decode_tokens']):.0f}%")
    print(f"host energy: baseline {baseline_energy['host_kj']} kJ, treatment {treatment_energy['host_kj']} kJ"
          f", saving {payload['saving_percent']}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
