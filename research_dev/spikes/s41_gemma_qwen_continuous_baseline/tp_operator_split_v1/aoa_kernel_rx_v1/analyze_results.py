#!/usr/bin/env python3
"""Validate and reduce the matched OP15 AOA RX-buffer experiment."""

import argparse
import collections
import json
import math
import re
import statistics
from pathlib import Path


SCHEMA = "s41_aoa_async_transport_v1"
WORKLOADS = {
    "attention": (1308, 1384, 50, 300),
    "hidden_m1": (10268, 10344, 50, 300),
    "swiglu": (69660, 34920, 50, 300),
    "hidden_m8": (81948, 82024, 50, 300),
    "host_to_phone_1m": (1048604, 104, 20, 100),
    "phone_to_host_1m": (64, 1048680, 20, 100),
}
VARIANTS = {
    "control": ("control", "control_shapes"),
    "rx64k": ("rx64k", "rx64k_shapes"),
}
SAMPLE_FIELDS = (
    "response_ready_samples_ms",
    "out_completion_samples_ms",
    "post_out_tail_samples_ms",
)
QUEUE_RE = re.compile(
    r"\s(?P<time>[0-9]+\.[0-9]+): dwc3_ep_queue: ep1out: req "
    r"(?P<request>[0-9a-f]+) length 0/(?P<maximum>[0-9]+) .* ==> -115"
)
GIVEBACK_RE = re.compile(
    r"\s(?P<time>[0-9]+\.[0-9]+): dwc3_gadget_giveback: ep1out: req "
    r"(?P<request>[0-9a-f]+) length (?P<actual>[0-9]+)/"
    r"(?P<maximum>[0-9]+) .* ==> (?P<status>-?[0-9]+)"
)


def percentile(values, fraction):
    ordered = sorted(values)
    index = math.floor(fraction * (len(ordered) - 1))
    return ordered[index]


def check_close(actual, expected, label):
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(f"{label}: {actual} != {expected}")


def validate_run(path, variant, workload, repetition):
    data = json.loads(path.read_text())
    request_bytes, response_bytes, warmup, iterations = WORKLOADS[workload]
    expected = {
        "schema": SCHEMA,
        "case_name": f"{variant}_{workload}_r{repetition}",
        "workload": workload,
        "repetition": repetition,
        "phone_mode": "serial",
        "mode": "sync",
        "preposted_in": False,
        "request_bytes": request_bytes,
        "response_bytes": response_bytes,
        "warmup": warmup,
        "iterations": iterations,
        "queue_depth": 1,
    }
    for field, value in expected.items():
        if data.get(field) != value:
            raise RuntimeError(
                f"unexpected {field} in {path}: {data.get(field)!r}"
            )
    if path.stem != data["case_name"]:
        raise RuntimeError(f"case name does not match file name: {path}")

    for field in SAMPLE_FIELDS:
        values = data.get(field)
        if not isinstance(values, list) or len(values) != iterations:
            raise RuntimeError(f"sample count mismatch in {path}: {field}")
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise RuntimeError(f"invalid timing in {path}: {field}")

    response = data["response_ready_samples_ms"]
    check_close(
        data["response_ready_median_ms"],
        statistics.median(response),
        f"stored median in {path}",
    )
    check_close(
        data["response_ready_p90_ms"],
        percentile(response, 0.90),
        f"stored p90 in {path}",
    )
    check_close(
        data["response_ready_p99_ms"],
        percentile(response, 0.99),
        f"stored p99 in {path}",
    )
    if data.get("aoa_product_id") != "0x2d01":
        raise RuntimeError(f"unexpected AOA product in {path}")
    for field in ("campaign_seconds", "requests_per_second",
                  "aggregate_payload_MBps"):
        value = data.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value) \
                or value <= 0.0:
            raise RuntimeError(f"invalid {field} in {path}")

    log_path = path.with_suffix(".worker.log")
    log = log_path.read_text()
    configured = (
        f"[aoa-buffer] configured mode=serial request={request_bytes} "
        f"response={response_bytes} requests={warmup + iterations} "
        f"warmup={warmup} depth=1"
    )
    complete = (
        f"[aoa-buffer] complete status=0 requests={warmup + iterations}"
    )
    if configured not in log or complete not in log:
        raise RuntimeError(f"worker lifecycle mismatch in {log_path}")
    if log.count("[aoa-buffer] read_us ") != 1 or \
            log.count("[aoa-buffer] write_us ") != 1:
        raise RuntimeError(f"worker timing mismatch in {log_path}")
    lowered = log.lower()
    if any(word in lowered for word in ("failed", "invalid", "killed")):
        raise RuntimeError(f"worker error text in {log_path}")

    return data


def validate_matrix(root):
    expected_json = set()
    runs = {}
    for variant, (bulk_dir, shape_dir) in VARIANTS.items():
        for workload in WORKLOADS:
            directory = bulk_dir if workload.endswith("_1m") else shape_dir
            for repetition in (1, 2, 3):
                path = root / directory / (
                    f"{variant}_{workload}_r{repetition}.json"
                )
                expected_json.add(path)
                runs[(variant, workload, repetition)] = validate_run(
                    path, variant, workload, repetition
                )

    observed_json = set()
    for directories in VARIANTS.values():
        for directory in directories:
            observed_json.update((root / directory).glob("*.json"))
    if observed_json != expected_json:
        missing = sorted(str(path) for path in expected_json - observed_json)
        extra = sorted(str(path) for path in observed_json - expected_json)
        raise RuntimeError(f"matrix mismatch: missing={missing} extra={extra}")

    rows = []
    for workload in WORKLOADS:
        for variant in VARIANTS:
            items = [runs[(variant, workload, repetition)]
                     for repetition in (1, 2, 3)]
            rows.append({
                "variant": variant,
                "workload": workload,
                "request_bytes": items[0]["request_bytes"],
                "response_bytes": items[0]["response_bytes"],
                "run_medians_ms": [
                    item["response_ready_median_ms"] for item in items
                ],
                "run_p90_ms": [
                    item["response_ready_p90_ms"] for item in items
                ],
                "run_aggregate_payload_MBps": [
                    item["aggregate_payload_MBps"] for item in items
                ],
                "median_of_run_medians_ms": statistics.median(
                    item["response_ready_median_ms"] for item in items
                ),
                "median_of_run_p90_ms": statistics.median(
                    item["response_ready_p90_ms"] for item in items
                ),
                "median_aggregate_payload_MBps": statistics.median(
                    item["aggregate_payload_MBps"] for item in items
                ),
            })

    comparisons = []
    for workload in WORKLOADS:
        control_items = [runs[("control", workload, repetition)]
                         for repetition in (1, 2, 3)]
        rx64k_items = [runs[("rx64k", workload, repetition)]
                       for repetition in (1, 2, 3)]
        control_row = next(
            row for row in rows
            if row["variant"] == "control" and row["workload"] == workload
        )
        rx64k_row = next(
            row for row in rows
            if row["variant"] == "rx64k" and row["workload"] == workload
        )
        paired_latency = [
            treatment["response_ready_median_ms"] /
            control["response_ready_median_ms"] - 1.0
            for control, treatment in zip(control_items, rx64k_items)
        ]
        paired_throughput = [
            treatment["aggregate_payload_MBps"] /
            control["aggregate_payload_MBps"] - 1.0
            for control, treatment in zip(control_items, rx64k_items)
        ]
        comparisons.append({
            "workload": workload,
            "control_median_ms": control_row["median_of_run_medians_ms"],
            "rx64k_median_ms": rx64k_row["median_of_run_medians_ms"],
            "latency_change_of_medians": (
                rx64k_row["median_of_run_medians_ms"] /
                control_row["median_of_run_medians_ms"] - 1.0
            ),
            "paired_latency_changes": paired_latency,
            "median_paired_latency_change": statistics.median(paired_latency),
            "latency_wins": sum(change < 0.0 for change in paired_latency),
            "control_aggregate_payload_MBps":
                control_row["median_aggregate_payload_MBps"],
            "rx64k_aggregate_payload_MBps":
                rx64k_row["median_aggregate_payload_MBps"],
            "throughput_change_of_medians": (
                rx64k_row["median_aggregate_payload_MBps"] /
                control_row["median_aggregate_payload_MBps"] - 1.0
            ),
            "paired_throughput_changes": paired_throughput,
            "median_paired_throughput_change":
                statistics.median(paired_throughput),
            "throughput_wins": sum(
                change > 0.0 for change in paired_throughput
            ),
        })

    return runs, rows, comparisons


def parse_trace(path):
    queues = collections.defaultdict(list)
    givebacks = collections.defaultdict(list)
    for line in path.read_text().splitlines():
        match = QUEUE_RE.search(line)
        if match:
            queues[match.group("request")].append({
                "time": float(match.group("time")),
                "maximum": int(match.group("maximum")),
            })
        match = GIVEBACK_RE.search(line)
        if match:
            givebacks[match.group("request")].append({
                "time": float(match.group("time")),
                "actual": int(match.group("actual")),
                "maximum": int(match.group("maximum")),
                "status": int(match.group("status")),
            })
    return queues, givebacks


def find_queue_sequence(path, full_size, full_count):
    queues, givebacks = parse_trace(path)
    expected = collections.Counter({full_size: full_count, 1024: 1})
    candidates = [
        request for request, entries in queues.items()
        if collections.Counter(entry["maximum"] for entry in entries) == expected
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one {full_count} x {full_size} queue sequence in "
            f"{path}, found {candidates}"
        )
    request = candidates[0]
    entries = sorted(queues[request], key=lambda entry: entry["time"])
    return request, entries, sorted(
        givebacks[request], key=lambda entry: entry["time"]
    )


def validate_trace_case(path, case_name):
    data = json.loads(path.read_text())
    expected = {
        "schema": SCHEMA,
        "case_name": case_name,
        "workload": "host_to_phone_1m",
        "phone_mode": "serial",
        "mode": "sync",
        "preposted_in": False,
        "request_bytes": 1048604,
        "response_bytes": 104,
        "warmup": 0,
        "iterations": 1,
        "queue_depth": 1,
    }
    for field, value in expected.items():
        if data.get(field) != value:
            raise RuntimeError(f"unexpected {field} in {path}")
    if len(data.get("response_ready_samples_ms", [])) != 1:
        raise RuntimeError(f"trace timing count mismatch in {path}")
    log = path.with_suffix(".worker.log").read_text()
    if "[aoa-buffer] complete status=0 requests=1" not in log:
        raise RuntimeError(f"trace worker did not complete: {path}")
    return data


def validate_traces(root, stock_trace_root):
    stock_json = stock_trace_root / "baseline_trace_1m.json"
    stock_data = validate_trace_case(stock_json, "baseline_trace_1m")
    stock_trace = stock_trace_root / "dwc3_trace.txt"
    stock_request, stock_queues, stock_givebacks = find_queue_sequence(
        stock_trace, 16384, 64
    )
    successful_stock = [
        entry for entry in stock_givebacks if entry["status"] == 0
    ]
    expected_givebacks = collections.Counter({
        (16384, 16384): 64,
        (28, 1024): 1,
    })
    observed_givebacks = collections.Counter(
        (entry["actual"], entry["maximum"])
        for entry in successful_stock
    )
    if observed_givebacks != expected_givebacks:
        raise RuntimeError(
            f"stock completion sequence mismatch: {observed_givebacks}"
        )

    requeue_gaps_us = []
    if len(stock_queues) != len(successful_stock):
        raise RuntimeError("stock queue and completion counts differ")
    for index, (queued, completed) in enumerate(
            zip(stock_queues, successful_stock)):
        if queued["time"] >= completed["time"]:
            raise RuntimeError("stock request completed before it was queued")
        if index + 1 < len(stock_queues):
            gap = (stock_queues[index + 1]["time"] - completed["time"]) * 1e6
            if gap < 0.0:
                raise RuntimeError("stock requests overlapped unexpectedly")
            requeue_gaps_us.append(gap)

    live_json = root / "rx64k_trace_usb" / "rx64k_trace_usb_1m.json"
    live_data = validate_trace_case(live_json, "rx64k_trace_usb_1m")
    live_trace = root / "rx64k_trace_usb" / "dwc3_trace.txt"
    live_request, live_queues, live_givebacks = find_queue_sequence(
        live_trace, 65536, 16
    )

    excluded_json = root / "rx64k_trace" / "rx64k_trace_1m.json"
    excluded_data = validate_trace_case(excluded_json, "rx64k_trace_1m")
    excluded_trace = root / "rx64k_trace" / "dwc3_trace.txt"
    excluded_queues, _ = parse_trace(excluded_trace)
    excluded_matches = [
        request for request, entries in excluded_queues.items()
        if collections.Counter(entry["maximum"] for entry in entries) ==
        collections.Counter({65536: 16, 1024: 1})
    ]
    if excluded_matches:
        raise RuntimeError("excluded trace unexpectedly contains the live sequence")

    return {
        "stock": {
            "source": str(stock_trace),
            "request_pointer": stock_request,
            "queue_count": len(stock_queues),
            "full_queue_count": 64,
            "full_queue_bytes": 16384,
            "tail_queue_bytes": 1024,
            "completed_bytes": sum(
                entry["actual"] for entry in successful_stock
            ),
            "requeue_gap_median_us": statistics.median(requeue_gaps_us),
            "requeue_gap_p90_us": percentile(requeue_gaps_us, 0.90),
            "requeue_gap_max_us": max(requeue_gaps_us),
            "requeue_gap_total_us": sum(requeue_gaps_us),
            "trace_case_latency_ms": stock_data["response_ready_median_ms"],
        },
        "rx64k": {
            "source": str(live_trace),
            "request_pointer": live_request,
            "queue_count": len(live_queues),
            "full_queue_count": 16,
            "full_queue_bytes": 65536,
            "tail_queue_bytes": 1024,
            "giveback_records_for_pointer": len(live_givebacks),
            "trace_case_latency_ms": live_data["response_ready_median_ms"],
            "completion_validation":
                "exact host protocol and successful phone worker",
        },
        "excluded_wireless_attempt": {
            "source": str(excluded_trace),
            "reason": "trace enablement failed over unavailable wireless adb",
            "matching_queue_sequences": len(excluded_matches),
            "transport_case_latency_ms":
                excluded_data["response_ready_median_ms"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--stock-trace-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    runs, rows, comparisons = validate_matrix(args.root)
    traces = validate_traces(args.root, args.stock_trace_root)
    result = {
        "schema": "s41_aoa_kernel_rx64k_analysis_v1",
        "verdict": "PASS",
        "raw_runs": len(runs),
        "worker_logs": len(runs),
        "paid_samples": sum(run["iterations"] for run in runs.values()),
        "exact_response_validation": "enforced online by native host",
        "reduction": "median across three fresh-process run statistics",
        "rows": rows,
        "comparisons": comparisons,
        "traces": traces,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("PASS")


if __name__ == "__main__":
    main()
