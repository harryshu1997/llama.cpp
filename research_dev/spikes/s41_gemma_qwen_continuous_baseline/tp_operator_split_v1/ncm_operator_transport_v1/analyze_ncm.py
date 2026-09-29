#!/usr/bin/env python3
"""Reduce repeated NCM operator-transport latency captures."""

import argparse
import json
import statistics
from pathlib import Path


EXPECTED_SCHEMA = "s41_ncm_operator_transport_v1"


def median(values):
    return statistics.median(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if len(args.inputs) < 3:
        raise RuntimeError("at least three repetitions are required")

    captures = []
    for path in args.inputs:
        capture = json.loads(path.read_text())
        if capture.get("schema") != EXPECTED_SCHEMA:
            raise RuntimeError(f"unexpected schema in {path}")
        if not capture.get("persistent_connection"):
            raise RuntimeError(f"non-persistent connection in {path}")
        if not capture.get("tcp_nodelay"):
            raise RuntimeError(f"TCP_NODELAY missing in {path}")
        captures.append((path, capture))

    case_names = {row["name"] for row in captures[0][1]["rows"]}
    if "control" not in case_names:
        raise RuntimeError("control case is missing")

    repetitions = []
    for path, capture in captures:
        rows = {row["name"]: row for row in capture["rows"]}
        if set(rows) != case_names:
            raise RuntimeError(f"case mismatch in {path}")
        if len(rows) != len(capture["rows"]):
            raise RuntimeError(f"duplicate case in {path}")
        for row in rows.values():
            if row["iterations"] != len(row["samples_ms"]):
                raise RuntimeError(f"sample count mismatch in {path}")
        repetitions.append({"path": str(path), "rows": rows})

    control_shapes = {
        (item["rows"]["control"]["request_bytes"],
         item["rows"]["control"]["response_bytes"])
        for item in repetitions
    }
    if control_shapes != {(0, 0)}:
        raise RuntimeError("control payload must be zero bytes")

    output_rows = []
    for name in sorted(case_names):
        rows = [item["rows"][name] for item in repetitions]
        shapes = {(row["request_bytes"], row["response_bytes"])
                  for row in rows}
        if len(shapes) != 1:
            raise RuntimeError(f"shape mismatch for {name}")
        request_bytes, response_bytes = shapes.pop()
        run_medians = [row["median_ms"] for row in rows]
        run_p90s = [row["p90_ms"] for row in rows]
        run_p99s = [row["p99_ms"] for row in rows]
        run_effective_rates = [row["effective_payload_MBps_at_median"]
                               for row in rows]
        paired_control_deltas = [
            row["median_ms"] - repetition["rows"]["control"]["median_ms"]
            for row, repetition in zip(rows, repetitions)
        ]
        output_rows.append({
            "name": name,
            "request_bytes": request_bytes,
            "response_bytes": response_bytes,
            "total_payload_bytes": request_bytes + response_bytes,
            "median_of_run_medians_ms": median(run_medians),
            "median_of_run_p90_ms": median(run_p90s),
            "median_of_run_p99_ms": median(run_p99s),
            "median_of_effective_aggregate_payload_MBps": (
                median(run_effective_rates)
            ),
            "median_paired_control_delta_ms": median(paired_control_deltas),
            "run_medians_ms": run_medians,
            "run_p90_ms": run_p90s,
            "run_p99_ms": run_p99s,
            "paired_control_delta_ms": paired_control_deltas,
        })

    total_samples = sum(
        len(row["samples_ms"])
        for _, capture in captures
        for row in capture["rows"]
    )
    result = {
        "schema": "s41_ncm_operator_transport_analysis_v1",
        "repetitions": len(repetitions),
        "paid_samples": total_samples,
        "reduction": "median across fresh-process run statistics",
        "control_delta": "paired within each run, then median",
        "effective_rate_note": (
            "sum of request and response bytes divided by round-trip time; "
            "not simultaneous full-duplex goodput"
        ),
        "inputs": [str(path) for path in args.inputs],
        "rows": output_rows,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
