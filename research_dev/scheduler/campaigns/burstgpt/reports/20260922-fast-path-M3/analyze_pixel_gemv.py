"""Audit a Pixel kernel sweep and compare each arm with its surrounding controls."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

from tune_pixel_gemv import summarize


def read(path):
    return json.loads(path.read_text())


def analyze(root, config=None):
    result = read(root / "RESULT.json")
    if config is None:
        config = read(root / "CONFIG.json")
    cleanup = read(root / "CLEANUP.json")
    if result["status"] != "PASS" or cleanup["status"] != "PASS":
        raise ValueError("sweep or cleanup failed")
    expected_calls = 12 * config["repeats"]
    if expected_calls != result["calls_per_arm"]:
        raise ValueError("unexpected call count")
    if result["phone_calls"] != expected_calls * len(config["arms"]):
        raise ValueError("unexpected phone call total")
    cpu = [np.fromfile(root / "cpu" / f"output-{ident:03d}.f16", dtype="<f2").astype(np.float64)
           for ident in range(1, expected_calls + 1)]
    first = config["arms"][0]["name"]
    baseline = [(root / first / f"output-{ident:03d}.f16").read_bytes()
                for ident in range(1, expected_calls + 1)]
    controls = [i for i, arm in enumerate(config["arms"])
                if arm["runtime"] == "original" and arm["quantum"] == 4352
                and arm.get("worker", "original") == "original"]
    if "comparison_controls" in config:
        names = config["comparison_controls"]
        controls = [i for i, arm in enumerate(config["arms"]) if arm["name"] in names]
        if len(controls) != len(names) or len(controls) < 2:
            raise ValueError("comparison controls differ")
    if not controls:
        raise ValueError("sweep needs comparison controls")
    for index, arm in enumerate(config["arms"]):
        if (index < controls[0] or index > controls[-1]) and not arm.get("reference_only", False):
            raise ValueError("candidate lacks surrounding controls")
    summaries = {}
    for arm in config["arms"]:
        directory = root / arm["name"]
        rows = [json.loads(line) for line in (directory / "CALLS.jsonl").read_text().splitlines()]
        if [r["id"] for r in rows] != list(range(1, expected_calls + 1)):
            raise ValueError("missing or reordered calls")
        exact = 0
        for row, reference, original in zip(rows, cpu, baseline):
            output = (directory / f"output-{row['id']:03d}.f16").read_bytes()
            if hashlib.sha256(output).hexdigest() != row["output_sha256"]:
                raise ValueError("output hash differs")
            values = np.frombuffer(output, dtype="<f2").astype(np.float64)
            if values.shape != (5120,) or not np.isfinite(values).all():
                raise ValueError("output shape or finiteness differs")
            relative = float(np.linalg.norm(values - reference) / max(np.linalg.norm(reference), 1e-30))
            if abs(relative - row["relative_l2"]) > 1e-12 or relative > 0.01:
                raise ValueError("numerical comparison failed")
            exact += output == original
            if row["warm"] != (row["repeat"] >= config.get("warmup_repeats", 2)):
                raise ValueError("warmup selection differs")
        for layer in range(18, 24):
            for columns in (8704, 17408):
                selected = [r for r in rows if r["layer"] == layer and r["columns"] == columns]
                if len(selected) != config["repeats"] or len({r["output_sha256"] for r in selected}) != 1:
                    raise ValueError("missing or nondeterministic repetitions")
        summary = summarize(rows)
        recorded = read(directory / "SUMMARY.json")
        for key, value in summary.items():
            if recorded[key] != value:
                raise ValueError("timing summary differs")
        if recorded["exact_calls_vs_control"] != exact:
            raise ValueError("exact-output count differs")
        summary.update(arm=arm, calls=len(rows), exact_calls_vs_control=exact)
        summaries[arm["name"]] = summary
    comparisons = []
    for index, arm in enumerate(config["arms"]):
        if index in controls or arm.get("reference_only", False):
            continue
        before = max(c for c in controls if c < index)
        after = min(c for c in controls if c > index)
        control_names = [config["arms"][c]["name"] for c in (before, after)]
        row = {"name": arm["name"], "controls": control_names, "widths": {}}
        for columns in ("8704", "17408"):
            latency = summaries[arm["name"]][columns]["worker"]["mean_ms"]
            control_ms = [summaries[c][columns]["worker"]["mean_ms"] for c in control_names]
            row["widths"][columns] = {
                "worker_mean_ms": latency, "control_mean_ms": statistics.mean(control_ms),
                "control_ms": control_ms,
                "latency_reduction_pct": (1 - latency / statistics.mean(control_ms)) * 100,
                "reduction_pct_vs_each": [(1 - latency / ms) * 100 for ms in control_ms],
                "matrix_gflops_per_worker_second": (6 * 5120 * int(columns)) / (latency * 1e6)}
        comparisons.append(row)
    ordered = sorted(comparisons, key=lambda row: row["widths"]["17408"]["latency_reduction_pct"], reverse=True)
    return {"status": "PASS", "audit": "raw outputs, CPU comparisons, counts and timing summaries checked",
            "phone_calls": result["phone_calls"], "arms": summaries, "comparisons": comparisons,
            "best_full_width_candidate": ordered[0],
            "maximum_relative_l2": max(s["maximum_relative_l2"] for s in summaries.values()),
            "cleanup": cleanup["status"], "phone_energy_measured": False,
            "full_model_tokens_verified": False,
            "note": "Exploratory sweep. Confirm a candidate in a repeated bracketed comparison before promotion."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root), indent=2, allow_nan=False))
