#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path


WORKLOADS = {
    "attention": (1308, 1384, 50, 300),
    "hidden_m1": (10268, 10344, 50, 300),
    "swiglu": (69660, 34920, 50, 300),
    "hidden_m8": (81948, 82024, 50, 300),
    "upload_1m": (1048604, 104, 20, 100),
    "download_1m": (64, 1048680, 20, 100),
}

VARIANTS = {
    "copy_malloc": ("copy", "sync", "malloc", 1),
    "dmabuf_malloc": ("dmabuf", "sync", "malloc", 1),
    "dmabuf_devmem": ("dmabuf", "sync", "devmem", 1),
    "dmabuf_devmem_q4": ("dmabuf", "async", "devmem", 4),
}

NAME_PATTERN = re.compile(
    r"^(?P<workload>[a-z0-9_]+)\."
    r"(?P<variant>copy_malloc|dmabuf_malloc|dmabuf_devmem|"
    r"dmabuf_devmem_q4)\.r(?P<repetition>[123])$")


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
    names = set()
    for workload in WORKLOADS:
        for variant in ("copy_malloc", "dmabuf_malloc", "dmabuf_devmem"):
            for repetition in range(1, 4):
                names.add(f"{workload}.{variant}.r{repetition}")
    for workload in ("hidden_m1", "hidden_m8"):
        for repetition in range(1, 4):
            names.add(f"{workload}.dmabuf_devmem_q4.r{repetition}")
    return names


def validate_capture(root, base_name):
    match = NAME_PATTERN.fullmatch(base_name)
    if match is None:
        raise ValueError(f"invalid capture name: {base_name}")
    workload = match.group("workload")
    variant = match.group("variant")
    repetition = int(match.group("repetition"))
    if workload not in WORKLOADS:
        raise ValueError(f"unknown workload: {workload}")
    if variant == "dmabuf_devmem_q4" and workload not in {
            "hidden_m1", "hidden_m8"}:
        raise ValueError(f"unexpected queued workload: {workload}")

    result_path = root / f"{base_name}.json"
    data = json.loads(result_path.read_text(encoding="utf-8"))
    request, response, warmup, iterations = WORKLOADS[workload]
    device_mode, host_mode, allocator, depth = VARIANTS[variant]
    exact = {
        "schema": "s41_ffs_dmabuf_transport_v1",
        "case_name": base_name,
        "device_mode": device_mode,
        "host_mode": host_mode,
        "host_allocator": allocator,
        "request_bytes": request,
        "response_bytes": response,
        "warmup": warmup,
        "iterations": iterations,
        "queue_depth": depth,
    }
    for key, value in exact.items():
        if data.get(key) != value:
            raise ValueError(f"{base_name}: {key}={data.get(key)!r}, expected {value!r}")

    response_values = data["response_ready_samples_ms"]
    out_values = data["out_completion_samples_ms"]
    tail_values = data["post_out_tail_samples_ms"]
    for label, values in (
            ("response", response_values),
            ("out", out_values),
            ("tail", tail_values)):
        if len(values) != iterations:
            raise ValueError(f"{base_name}: {label} sample count")
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(f"{base_name}: invalid {label} sample")
    for index, (out_value, tail_value, response_value) in enumerate(zip(
            out_values, tail_values, response_values)):
        require_close(out_value + tail_value, response_value,
                f"{base_name}: sample {index}", absolute=2e-6)

    require_close(data["response_ready_median_ms"],
            statistics.median(response_values), f"{base_name}: median")
    require_close(data["response_ready_p90_ms"],
            stored_percentile(response_values, 0.90), f"{base_name}: p90")
    require_close(data["response_ready_p99_ms"],
            stored_percentile(response_values, 0.99), f"{base_name}: p99")
    require_close(data["out_completion_median_ms"],
            statistics.median(out_values), f"{base_name}: out median")
    require_close(data["post_out_tail_median_ms"],
            statistics.median(tail_values), f"{base_name}: tail median")
    require_close(data["aggregate_payload_MBps"],
            data["requests_per_second"] * (request + response) / 1e6,
            f"{base_name}: aggregate rate", absolute=1e-8)

    phone_log = (root / f"{base_name}.phone.log").read_text(encoding="utf-8")
    session_log = (root / f"{base_name}.session.log").read_text(encoding="utf-8")
    fault_log = root / f"{base_name}.kernel_faults.log"
    terminal = (root / f"{base_name}.terminal.log").read_text(
            encoding="utf-8").replace("\r", "").splitlines()
    if "complete status=0" not in phone_log:
        raise ValueError(f"{base_name}: phone did not complete")
    if "worker_status=0" not in session_log:
        raise ValueError(f"{base_name}: session did not complete")
    if fault_log.stat().st_size != 0:
        raise ValueError(f"{base_name}: non-empty kernel fault log")
    if terminal != ["a600000.dwc3", "", "super-speed"]:
        raise ValueError(f"{base_name}: invalid terminal state {terminal!r}")

    return {
        "workload": workload,
        "variant": variant,
        "repetition": repetition,
        "median_ms": data["response_ready_median_ms"],
        "p90_ms": data["response_ready_p90_ms"],
        "out_median_ms": data["out_completion_median_ms"],
        "tail_median_ms": data["post_out_tail_median_ms"],
        "aggregate_MBps": data["aggregate_payload_MBps"],
    }


def summarize(records):
    by_key = {}
    for record in records:
        by_key[(record["workload"], record["variant"],
                record["repetition"])] = record

    summary = {}
    for workload in WORKLOADS:
        summary[workload] = {}
        control = [by_key[(workload, "copy_malloc", repetition)]
                   for repetition in range(1, 4)]
        control_median = statistics.median(
                item["median_ms"] for item in control)
        for variant in ("copy_malloc", "dmabuf_malloc", "dmabuf_devmem"):
            items = [by_key[(workload, variant, repetition)]
                     for repetition in range(1, 4)]
            variant_median = statistics.median(
                    item["median_ms"] for item in items)
            aligned = [
                100.0 * (items[index]["median_ms"] /
                    control[index]["median_ms"] - 1.0)
                for index in range(3)
            ]
            summary[workload][variant] = {
                "median_of_run_medians_ms": variant_median,
                "median_of_run_p90_ms": statistics.median(
                        item["p90_ms"] for item in items),
                "median_out_completion_ms": statistics.median(
                        item["out_median_ms"] for item in items),
                "median_post_out_tail_ms": statistics.median(
                        item["tail_median_ms"] for item in items),
                "median_aggregate_MBps": statistics.median(
                        item["aggregate_MBps"] for item in items),
                "change_of_medians_percent":
                    100.0 * (variant_median / control_median - 1.0),
                "aligned_median_change_percent": statistics.median(aligned),
                "wins_vs_copy": sum(value < 0 for value in aligned),
            }

    queued = {}
    for workload in ("hidden_m1", "hidden_m8"):
        q1 = [by_key[(workload, "dmabuf_devmem", repetition)]
              for repetition in range(1, 4)]
        q4 = [by_key[(workload, "dmabuf_devmem_q4", repetition)]
              for repetition in range(1, 4)]
        q1_rate = statistics.median(item["aggregate_MBps"] for item in q1)
        q4_rate = statistics.median(item["aggregate_MBps"] for item in q4)
        queued[workload] = {
            "q1_median_aggregate_MBps": q1_rate,
            "q4_median_aggregate_MBps": q4_rate,
            "q4_throughput_change_percent": 100.0 * (q4_rate / q1_rate - 1.0),
            "q4_median_request_latency_ms": statistics.median(
                    item["median_ms"] for item in q4),
        }
    return summary, queued


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    expected = expected_names()
    ignored_json = {
        args.output.name,
        "ANALYSIS.json",
        "BUILD_BINDINGS.json",
    }
    actual = {path.stem for path in args.result_root.glob("*.json")
              if path.name not in ignored_json}
    if actual != expected:
        raise ValueError(
            f"capture set mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}")

    records = [validate_capture(args.result_root, name)
               for name in sorted(expected)]
    summary, queued = summarize(records)
    evidence_files = sorted(path for path in args.result_root.iterdir()
            if path.is_file() and path != args.output and
            path.name not in {"ANALYSIS.json", "SHA256SUMS.txt"})
    output = {
        "schema": "s41_ffs_dmabuf_transport_analysis_v1",
        "verdict": "REAL_FUNCTIONFS_DMABUF_PASS",
        "capture_count": len(records),
        "paid_request_count": sum(
                WORKLOADS[record["workload"]][3] for record in records),
        "all_kernel_fault_logs_empty": True,
        "all_terminal_states_valid": True,
        "summary": summary,
        "queued_throughput": queued,
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
