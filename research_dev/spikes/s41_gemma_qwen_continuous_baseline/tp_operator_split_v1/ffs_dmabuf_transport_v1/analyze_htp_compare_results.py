#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path


WORKLOADS = {
    "attention_state": {"elements": 320, "warmup": 50, "iterations": 300},
    "hidden_m1": {"elements": 2560, "warmup": 50, "iterations": 300},
    "hidden_m8": {"elements": 20480, "warmup": 50, "iterations": 300},
    "one_mib": {"elements": 262144, "warmup": 20, "iterations": 100},
}

VARIANTS = {
    "staged_malloc": {"mode": "htp-staged-sqr", "allocator": "malloc"},
    "copy_malloc": {"mode": "htp-copy-sqr", "allocator": "malloc"},
    "dmabuf_malloc": {"mode": "htp-sqr", "allocator": "malloc"},
    "dmabuf_devmem": {"mode": "htp-sqr", "allocator": "devmem"},
}

NAME_PATTERN = re.compile(
    r"^(?P<workload>attention_state|hidden_m1|hidden_m8|one_mib)\."
    r"(?P<variant>staged_malloc|copy_malloc|dmabuf_malloc|dmabuf_devmem)\."
    r"r(?P<repetition>[123])$")

PHONE_PATTERN = re.compile(
    r"\[ffs-htp\] complete mode=(?P<mode>[a-z-]+) "
    r"paid=(?P<paid>[0-9]+) "
    r"input_wait_us=(?P<input>[0-9.]+) "
    r"submit_us=(?P<submit>[0-9.]+) "
    r"sync_us=(?P<sync>[0-9.]+) "
    r"output_wait_us=(?P<output>[0-9.]+) "
    r"process_cpu_us=(?P<cpu>[0-9.]+)")


def stored_percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1,
                       int(fraction * (len(ordered) - 1)))]


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
        f"{workload}.{variant}.r{repetition}"
        for workload in WORKLOADS
        for variant in VARIANTS
        for repetition in range(1, 4)
    }


def validate_capture(root, base_name):
    match = NAME_PATTERN.fullmatch(base_name)
    if match is None:
        raise ValueError(f"invalid capture name: {base_name}")
    workload = match.group("workload")
    variant = match.group("variant")
    repetition = int(match.group("repetition"))
    workload_config = WORKLOADS[workload]
    variant_config = VARIANTS[variant]
    elements = workload_config["elements"]
    iterations = workload_config["iterations"]

    data = json.loads((root / f"{base_name}.json").read_text(
            encoding="utf-8"))
    exact = {
        "schema": "s41_ffs_dmabuf_htp_v1",
        "operator": "sqr_f32",
        "host_allocator": variant_config["allocator"],
        "elements": elements,
        "request_bytes": elements * 4,
        "response_bytes": elements * 4,
        "warmup": workload_config["warmup"],
        "iterations": iterations,
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
        if len(values) != iterations:
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

    phone_log = (root / f"{base_name}.phone.log").read_text(
            encoding="utf-8")
    phone_match = PHONE_PATTERN.search(phone_log)
    if phone_match is None:
        raise ValueError(f"{base_name}: missing phone completion")
    if phone_match.group("mode") != variant_config["mode"] or \
            int(phone_match.group("paid")) != iterations:
        raise ValueError(f"{base_name}: invalid phone completion")
    session_log = (root / f"{base_name}.session.log").read_text(
            encoding="utf-8")
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
        "variant": variant,
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
        "phone_process_cpu_us": float(phone_match.group("cpu")),
    }


def summarize(records):
    by_key = {
        (record["workload"], record["variant"], record["repetition"]):
            record for record in records
    }
    summary = {}
    for workload, workload_config in WORKLOADS.items():
        summary[workload] = {
            "bytes_each_way": workload_config["elements"] * 4,
            "variants": {},
        }
        copy_items = [by_key[(workload, "copy_malloc", repetition)]
                      for repetition in range(1, 4)]
        staged_items = [by_key[(workload, "staged_malloc", repetition)]
                        for repetition in range(1, 4)]
        for variant in VARIANTS:
            items = [by_key[(workload, variant, repetition)]
                     for repetition in range(1, 4)]
            versus_copy = [
                100.0 * (items[index]["median_ms"] /
                         copy_items[index]["median_ms"] - 1.0)
                for index in range(3)
            ]
            versus_staged = [
                100.0 * (items[index]["median_ms"] /
                         staged_items[index]["median_ms"] - 1.0)
                for index in range(3)
            ]
            summary[workload]["variants"][variant] = {
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
                "aligned_change_vs_copy_percent": statistics.median(
                        versus_copy),
                "wins_vs_copy": sum(value < 0 for value in versus_copy),
                "aligned_change_vs_staged_percent": statistics.median(
                        versus_staged),
                "wins_vs_staged": sum(value < 0 for value in versus_staged),
                "phone_input_wait_us": statistics.median(
                        item["phone_input_wait_us"] for item in items),
                "phone_htp_submit_us": statistics.median(
                        item["phone_submit_us"] for item in items),
                "phone_htp_sync_us": statistics.median(
                        item["phone_sync_us"] for item in items),
                "phone_output_wait_us": statistics.median(
                        item["phone_output_wait_us"] for item in items),
                "phone_process_cpu_us": statistics.median(
                        item["phone_process_cpu_us"] for item in items),
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
        "schema": "s41_ffs_dmabuf_htp_compare_analysis_v1",
        "verdict": "MATCHED_HTP_DMABUF_COMPARE_PASS",
        "capture_count": len(records),
        "paid_request_count": sum(
                WORKLOADS[record["workload"]]["iterations"]
                for record in records),
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
