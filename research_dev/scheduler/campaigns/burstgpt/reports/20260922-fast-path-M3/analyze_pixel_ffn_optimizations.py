"""Summarize finite suites against their bracketing controls without dropping outliers."""

import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def summarize(root):
    config = read(root / "CONFIG.json")
    result = read(root / "SUITE_RESULT.json")
    calls = read(root / "SUITE_CALLS.json")
    positions = {arm["name"]: index for index, arm in enumerate(config["arms"])}
    controls = config["comparison_controls"]
    rows = []
    for name, arm in result["arms"].items():
        if name in controls or name == "reference-cpu":
            continue
        before = [c for c in controls if positions[c] < positions[name]]
        after = [c for c in controls if positions[c] > positions[name]]
        if not before or not after:
            raise ValueError("candidate lacks bracketing controls")
        matched = [before[-1], after[0]]
        for key, metrics in arm["groups"].items():
            tokens, columns = (int(value[1:]) for value in key.split("-"))
            baseline = np.array([row["worker_us"] / 1000 for c in matched for row in calls[c]
                if row["warm"] and row["tokens"] == tokens and row["columns"] == columns])
            observations = [row for row in calls[name] if row["warm"] and row["tokens"] == tokens and row["columns"] == columns]
            if len(baseline) != 2 * len(observations):
                raise ValueError("different measurement composition")
            saving = 100 * (1 - metrics["mean_ms"] / float(baseline.mean()))
            rows.append(dict(arm=name, tokens=tokens, columns=columns,
                numerical_status=arm["numerical_status"], max_relative_l2=arm["max_relative_l2"],
                mean_ms=metrics["mean_ms"], median_ms=metrics["median_ms"], p99_ms=metrics["p99_ms"],
                per_row_ms=metrics["per_row_ms"], control_mean_ms=float(baseline.mean()),
                control_p99_ms=float(np.percentile(baseline, 99)), saving_percent=saving, controls=matched,
                mean_speed_status="PASS" if saving > 0 and arm["numerical_status"] == "PASS" else "FAIL",
                p99_speed_status="PASS" if metrics["p99_ms"] < float(np.percentile(baseline, 99)) else "FAIL",
                useful_matrix_gflops=6 * 5120 * columns * tokens / (metrics["mean_ms"] * 1e6),
                dual_calls=arm["dual_calls"], overlapping_calls=arm["overlapping_calls"], config=arm["arm"]))
    return dict(root=str(root.resolve()), phone_calls=result["phone_calls"], numerical_status=result["status"],
                comparison="pooled immediately preceding and following controls; same rows, layers, widths and warmup policy",
                rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [summarize(root) for root in args.roots]
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    for result in results:
        print(result["root"], result["numerical_status"], result["phone_calls"])
        for row in result["rows"]:
            if row["columns"] == 17408:
                print(row["arm"], f"B{row['tokens']}", row["numerical_status"],
                      f"{row['mean_ms']:.4f}ms {row['saving_percent']:+.3f}% p99={row['p99_ms']:.4f}")


if __name__ == "__main__":
    main()
