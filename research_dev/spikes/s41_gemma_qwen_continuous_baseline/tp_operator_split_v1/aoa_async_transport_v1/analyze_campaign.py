#!/usr/bin/env python3
"""Reduce repeated synchronous and asynchronous AOA transport runs."""

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


SCHEMA = "s41_aoa_async_transport_v1"


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1,
                       math.floor(fraction * (len(ordered) - 1)))]


def median(values):
    return statistics.median(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    runs = []
    for path in args.inputs:
        data = json.loads(path.read_text())
        if data.get("schema") != SCHEMA:
            raise RuntimeError(f"unexpected schema in {path}")
        values = data["response_ready_samples_ms"]
        if len(values) != data["iterations"] or not values:
            raise RuntimeError(f"sample count mismatch in {path}")
        derived = {
            "median": median(values),
            "p90": percentile(values, 0.90),
            "p99": percentile(values, 0.99),
        }
        for name, field in (
                ("median", "response_ready_median_ms"),
                ("p90", "response_ready_p90_ms"),
                ("p99", "response_ready_p99_ms")):
            if abs(derived[name] - data[field]) > 1e-9:
                raise RuntimeError(f"stored {name} mismatch in {path}")
        worker_log = path.with_suffix(".worker.log").read_text()
        for stage in ("read", "write"):
            match = re.search(
                rf"\[aoa-buffer\] {stage}_us median=([0-9.]+)",
                worker_log,
            )
            if match is None:
                raise RuntimeError(f"missing worker {stage} timing in {path}")
            data[f"worker_{stage}_median_us"] = float(match.group(1))
        data["path"] = str(path)
        data["derived"] = derived
        runs.append(data)

    groups = defaultdict(list)
    for run in runs:
        key = (
            run["workload"],
            run["request_bytes"],
            run["response_bytes"],
            run["phone_mode"],
            run["mode"],
            run["queue_depth"],
        )
        groups[key].append(run)

    rows = []
    for key, items in sorted(groups.items()):
        workload, request_bytes, response_bytes, phone_mode, mode, depth = key
        repetitions = [item["repetition"] for item in items]
        if len(set(repetitions)) != len(repetitions):
            raise RuntimeError(f"duplicate repetition for {key}")
        rows.append({
            "workload": workload,
            "request_bytes": request_bytes,
            "response_bytes": response_bytes,
            "phone_mode": phone_mode,
            "host_mode": mode,
            "queue_depth": depth,
            "runs": len(items),
            "repetitions": sorted(repetitions),
            "median_of_run_medians_ms": median([
                item["derived"]["median"] for item in items
            ]),
            "median_of_run_p90_ms": median([
                item["derived"]["p90"] for item in items
            ]),
            "median_of_run_p99_ms": median([
                item["derived"]["p99"] for item in items
            ]),
            "median_requests_per_second": median([
                item["requests_per_second"] for item in items
            ]),
            "median_aggregate_payload_MBps": median([
                item["aggregate_payload_MBps"] for item in items
            ]),
            "median_out_completion_ms": median([
                item["out_completion_median_ms"] for item in items
            ]),
            "median_post_out_tail_ms": median([
                item["post_out_tail_median_ms"] for item in items
            ]),
            "median_worker_read_us": median([
                item["worker_read_median_us"] for item in items
            ]),
            "median_worker_write_us": median([
                item["worker_write_median_us"] for item in items
            ]),
            "run_medians_ms": [
                item["derived"]["median"]
                for item in sorted(items, key=lambda value: value["repetition"])
            ],
            "run_requests_per_second": [
                item["requests_per_second"]
                for item in sorted(items, key=lambda value: value["repetition"])
            ],
            "paths": [item["path"] for item in items],
        })

    row_lookup = {
        (row["workload"], row["phone_mode"], row["host_mode"],
         row["queue_depth"]): row
        for row in rows
    }
    comparisons = []
    workloads = sorted({run["workload"] for run in runs})
    for workload in workloads:
        baseline = row_lookup.get((workload, "serial", "sync", 1))
        if baseline is None:
            continue
        for row in rows:
            if row["workload"] != workload or row is baseline:
                continue
            unpaired_latency_change = (
                row["median_of_run_medians_ms"] /
                baseline["median_of_run_medians_ms"] - 1.0
            )
            unpaired_throughput_change = (
                row["median_requests_per_second"] /
                baseline["median_requests_per_second"] - 1.0
            )
            baseline_items = {
                item["repetition"]: item for item in groups[(
                    baseline["workload"], baseline["request_bytes"],
                    baseline["response_bytes"], baseline["phone_mode"],
                    baseline["host_mode"], baseline["queue_depth"],
                )]
            }
            treatment_items = {
                item["repetition"]: item for item in groups[(
                    row["workload"], row["request_bytes"],
                    row["response_bytes"], row["phone_mode"],
                    row["host_mode"], row["queue_depth"],
                )]
            }
            if set(baseline_items) != set(treatment_items):
                raise RuntimeError(
                    f"unmatched repetitions for {workload}: {row}"
                )
            paired_latency_changes = []
            paired_throughput_changes = []
            for repetition in sorted(baseline_items):
                baseline_item = baseline_items[repetition]
                treatment_item = treatment_items[repetition]
                paired_latency_changes.append({
                    "repetition": repetition,
                    "change": (
                        treatment_item["derived"]["median"] /
                        baseline_item["derived"]["median"] - 1.0
                    ),
                })
                paired_throughput_changes.append({
                    "repetition": repetition,
                    "change": (
                        treatment_item["requests_per_second"] /
                        baseline_item["requests_per_second"] - 1.0
                    ),
                })
            comparisons.append({
                "workload": workload,
                "phone_mode": row["phone_mode"],
                "host_mode": row["host_mode"],
                "queue_depth": row["queue_depth"],
                "unpaired_latency_change_of_medians":
                    unpaired_latency_change,
                "unpaired_throughput_change_of_medians":
                    unpaired_throughput_change,
                "paired_latency_changes": paired_latency_changes,
                "paired_throughput_changes": paired_throughput_changes,
                "median_paired_latency_change": median([
                    value["change"] for value in paired_latency_changes
                ]),
                "median_paired_throughput_change": median([
                    value["change"] for value in paired_throughput_changes
                ]),
                "latency_wins": sum(
                    value["change"] < 0.0
                    for value in paired_latency_changes
                ),
                "throughput_wins": sum(
                    value["change"] > 0.0
                    for value in paired_throughput_changes
                ),
            })

    result = {
        "schema": "s41_aoa_async_transport_analysis_v1",
        "raw_runs": len(runs),
        "paid_samples": sum(run["iterations"] for run in runs),
        "reduction": "median across fresh-process run statistics",
        "rows": rows,
        "comparisons": comparisons,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
