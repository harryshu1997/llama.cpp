#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path


WORKLOADS = {
    "attention_state": 320,
    "hidden_m1": 2560,
    "hidden_m8": 20480,
}

NAME_PATTERN = re.compile(
    r"^(?P<workload>attention_state|hidden_m1|hidden_m8)\."
    r"(?P<allocator>malloc|devmem)\.r(?P<repetition>[123])$")

PHONE_PATTERN = re.compile(
    r"\[ffs-htp\] complete paid=(?P<paid>[0-9]+) "
    r"input_wait_us=(?P<input>[0-9.]+) "
    r"submit_us=(?P<submit>[0-9.]+) "
    r"sync_us=(?P<sync>[0-9.]+) "
    r"output_wait_us=(?P<output>[0-9.]+)")


def stored_percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]


def require_close(actual, expected, label, absolute=1e-9):
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=absolute):
        raise ValueError(f"{label}: {actual} != {expected}")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_names():
    return {
        f"{workload}.{allocator}.r{repetition}"
        for workload in WORKLOADS
        for allocator in ("malloc", "devmem")
        for repetition in range(1, 4)
    }


def validate_capture(root, base_name):
    match = NAME_PATTERN.fullmatch(base_name)
    if match is None:
        raise ValueError(f"invalid capture name: {base_name}")
    workload = match.group("workload")
    allocator = match.group("allocator")
    repetition = int(match.group("repetition"))
    elements = WORKLOADS[workload]
    data = json.loads((root / f"{base_name}.json").read_text(
            encoding="utf-8"))
    exact = {
        "schema": "s41_ffs_dmabuf_htp_v1",
        "operator": "sqr_f32",
        "host_allocator": allocator,
        "elements": elements,
        "request_bytes": elements * 4,
        "response_bytes": elements * 4,
        "warmup": 50,
        "iterations": 300,
        "maximum_absolute_error": 0,
    }
    for key, value in exact.items():
        if data.get(key) != value:
            raise ValueError(f"{base_name}: invalid {key}")

    response_values = data["response_samples_ms"]
    out_values = data["out_samples_ms"]
    tail_values = data["post_out_tail_samples_ms"]
    for label, values in (
            ("response", response_values),
            ("out", out_values),
            ("tail", tail_values)):
        if len(values) != 300:
            raise ValueError(f"{base_name}: invalid {label} sample count")
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(f"{base_name}: invalid {label} sample")
    for index, (out_value, tail_value, response_value) in enumerate(zip(
            out_values, tail_values, response_values)):
        require_close(out_value + tail_value, response_value,
                f"{base_name}: sample {index}", absolute=2e-6)
    require_close(data["response_median_ms"],
            statistics.median(response_values), f"{base_name}: median")
    require_close(data["response_p90_ms"],
            stored_percentile(response_values, 0.90), f"{base_name}: p90")
    require_close(data["response_p99_ms"],
            stored_percentile(response_values, 0.99), f"{base_name}: p99")
    require_close(data["out_median_ms"], statistics.median(out_values),
            f"{base_name}: out median")
    require_close(data["post_out_tail_median_ms"],
            statistics.median(tail_values), f"{base_name}: tail median")
    require_close(data["aggregate_payload_MBps"],
            data["requests_per_second"] * elements * 8 / 1e6,
            f"{base_name}: aggregate rate", absolute=1e-8)

    phone_log = (root / f"{base_name}.phone.log").read_text(encoding="utf-8")
    phone_match = PHONE_PATTERN.search(phone_log)
    if phone_match is None or int(phone_match.group("paid")) != 300:
        raise ValueError(f"{base_name}: invalid phone completion")
    session_log = (root / f"{base_name}.session.log").read_text(encoding="utf-8")
    if "worker_status=0" not in session_log:
        raise ValueError(f"{base_name}: invalid session completion")
    if (root / f"{base_name}.kernel_faults.log").stat().st_size != 0:
        raise ValueError(f"{base_name}: non-empty kernel fault log")
    terminal = (root / f"{base_name}.terminal.log").read_text(
            encoding="utf-8").replace("\r", "").splitlines()
    if terminal != ["a600000.dwc3", "", "super-speed"]:
        raise ValueError(f"{base_name}: invalid terminal state")

    return {
        "workload": workload,
        "allocator": allocator,
        "repetition": repetition,
        "median_ms": data["response_median_ms"],
        "p90_ms": data["response_p90_ms"],
        "out_ms": data["out_median_ms"],
        "tail_ms": data["post_out_tail_median_ms"],
        "aggregate_MBps": data["aggregate_payload_MBps"],
        "phone_input_wait_us": float(phone_match.group("input")),
        "phone_submit_us": float(phone_match.group("submit")),
        "phone_sync_us": float(phone_match.group("sync")),
        "phone_output_wait_us": float(phone_match.group("output")),
    }


def summarize(records):
    by_key = {
        (record["workload"], record["allocator"], record["repetition"]):
            record for record in records
    }
    summary = {}
    for workload in WORKLOADS:
        summary[workload] = {}
        malloc = [by_key[(workload, "malloc", repetition)]
                  for repetition in range(1, 4)]
        for allocator in ("malloc", "devmem"):
            items = [by_key[(workload, allocator, repetition)]
                     for repetition in range(1, 4)]
            aligned = [
                100.0 * (items[index]["median_ms"] /
                    malloc[index]["median_ms"] - 1.0)
                for index in range(3)
            ]
            summary[workload][allocator] = {
                "median_of_run_medians_ms": statistics.median(
                        item["median_ms"] for item in items),
                "median_of_run_p90_ms": statistics.median(
                        item["p90_ms"] for item in items),
                "median_out_completion_ms": statistics.median(
                        item["out_ms"] for item in items),
                "median_post_out_tail_ms": statistics.median(
                        item["tail_ms"] for item in items),
                "median_aggregate_MBps": statistics.median(
                        item["aggregate_MBps"] for item in items),
                "aligned_change_vs_malloc_percent": statistics.median(aligned),
                "wins_vs_malloc": sum(value < 0 for value in aligned),
                "phone_input_wait_us": statistics.median(
                        item["phone_input_wait_us"] for item in items),
                "phone_htp_submit_us": statistics.median(
                        item["phone_submit_us"] for item in items),
                "phone_htp_sync_us": statistics.median(
                        item["phone_sync_us"] for item in items),
                "phone_output_wait_us": statistics.median(
                        item["phone_output_wait_us"] for item in items),
            }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    expected = expected_names()
    ignored_json = {args.output.name, "ANALYSIS.json"}
    actual = {path.stem for path in args.result_root.glob("*.json")
              if path.name not in ignored_json}
    if actual != expected:
        raise ValueError(
            f"capture set mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}")
    records = [validate_capture(args.result_root, name)
               for name in sorted(expected)]
    evidence_files = sorted(path for path in args.result_root.iterdir()
            if path.is_file() and path != args.output and
            path.name not in {"ANALYSIS.json", "SHA256SUMS.txt"})
    output = {
        "schema": "s41_ffs_dmabuf_htp_analysis_v1",
        "verdict": "REAL_USB_DMABUF_HTP_DMABUF_PASS",
        "capture_count": len(records),
        "paid_request_count": len(records) * 300,
        "maximum_absolute_error": 0,
        "all_kernel_fault_logs_empty": True,
        "all_terminal_states_valid": True,
        "summary": summarize(records),
        "evidence_sha256": {
            path.name: sha256(path) for path in evidence_files
        },
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    print(json.dumps({
        "verdict": output["verdict"],
        "capture_count": output["capture_count"],
        "paid_request_count": output["paid_request_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
