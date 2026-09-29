#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path


USB_H2D_CEILING_MBPS = 500.0
RESPONSE_BYTES = 64

WORKLOADS = {
    "upload_1m": {
        "request_bytes": 1048576,
        "warmup": 20,
        "iterations": 200,
        "variants": (
            "copy_malloc",
            "dmabuf_malloc",
            "dmabuf_devmem",
            "dmabuf_devmem_q4",
        ),
    },
    "upload_4m": {
        "request_bytes": 4194304,
        "warmup": 10,
        "iterations": 100,
        "variants": (
            "copy_malloc",
            "dmabuf_malloc",
            "dmabuf_devmem",
            "dmabuf_devmem_q3",
        ),
    },
    "upload_15m": {
        "request_bytes": 15728640,
        "warmup": 3,
        "iterations": 30,
        "variants": (
            "copy_malloc",
            "dmabuf_malloc",
            "dmabuf_devmem",
        ),
    },
}

VARIANTS = {
    "copy_malloc": {
        "device_mode": "copy",
        "host_mode": "sync",
        "allocator": "malloc",
        "depth": 1,
    },
    "dmabuf_malloc": {
        "device_mode": "dmabuf",
        "host_mode": "sync",
        "allocator": "malloc",
        "depth": 1,
    },
    "dmabuf_devmem": {
        "device_mode": "dmabuf",
        "host_mode": "sync",
        "allocator": "devmem",
        "depth": 1,
    },
    "dmabuf_devmem_q3": {
        "device_mode": "dmabuf",
        "host_mode": "async",
        "allocator": "devmem",
        "depth": 3,
    },
    "dmabuf_devmem_q4": {
        "device_mode": "dmabuf",
        "host_mode": "async",
        "allocator": "devmem",
        "depth": 4,
    },
}

NAME_PATTERN = re.compile(
    r"^(?P<workload>upload_1m|upload_4m|upload_15m)\."
    r"(?P<variant>copy_malloc|dmabuf_malloc|dmabuf_devmem|"
    r"dmabuf_devmem_q3|dmabuf_devmem_q4)\.r(?P<repetition>[123])$")


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
        for workload, config in WORKLOADS.items()
        for variant in config["variants"]
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
    if variant not in workload_config["variants"]:
        raise ValueError(f"invalid workload variant: {base_name}")

    data = json.loads((root / f"{base_name}.json").read_text(
            encoding="utf-8"))
    exact = {
        "schema": "s41_ffs_dmabuf_transport_v1",
        "case_name": base_name,
        "device_mode": variant_config["device_mode"],
        "host_mode": variant_config["host_mode"],
        "host_allocator": variant_config["allocator"],
        "request_bytes": workload_config["request_bytes"],
        "response_bytes": RESPONSE_BYTES,
        "warmup": workload_config["warmup"],
        "iterations": workload_config["iterations"],
        "queue_depth": variant_config["depth"],
    }
    for key, value in exact.items():
        if data.get(key) != value:
            raise ValueError(f"{base_name}: invalid {key}")

    response_values = data["response_ready_samples_ms"]
    out_values = data["out_completion_samples_ms"]
    tail_values = data["post_out_tail_samples_ms"]
    iterations = workload_config["iterations"]
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
    expected_rate = (
        (workload_config["request_bytes"] + RESPONSE_BYTES) * iterations /
        data["campaign_seconds"] / 1e6
    )
    require_close(data["aggregate_payload_MBps"], expected_rate,
            f"{base_name}: aggregate rate", absolute=1e-8)

    phone_log = (root / f"{base_name}.phone.log").read_text(
            encoding="utf-8")
    if "complete status=0" not in phone_log:
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

    h2d_rate = data["aggregate_payload_MBps"] * (
        workload_config["request_bytes"] /
        (workload_config["request_bytes"] + RESPONSE_BYTES)
    )
    return {
        "workload": workload,
        "variant": variant,
        "repetition": repetition,
        "response_median_ms": data["response_ready_median_ms"],
        "response_p90_ms": data["response_ready_p90_ms"],
        "h2d_MBps": h2d_rate,
    }


def summarize(records):
    by_key = {
        (record["workload"], record["variant"], record["repetition"]):
            record for record in records
    }
    summary = {}
    for workload, workload_config in WORKLOADS.items():
        copy_items = [by_key[(workload, "copy_malloc", repetition)]
                      for repetition in range(1, 4)]
        summary[workload] = {
            "request_bytes": workload_config["request_bytes"],
            "response_bytes": RESPONSE_BYTES,
            "variants": {},
        }
        for variant in workload_config["variants"]:
            items = [by_key[(workload, variant, repetition)]
                     for repetition in range(1, 4)]
            aligned_rate_change = [
                100.0 * (items[index]["h2d_MBps"] /
                         copy_items[index]["h2d_MBps"] - 1.0)
                for index in range(3)
            ]
            median_rate = statistics.median(
                    item["h2d_MBps"] for item in items)
            summary[workload]["variants"][variant] = {
                "median_h2d_MBps": median_rate,
                "usb_symbol_ceiling_utilization_percent":
                    100.0 * median_rate / USB_H2D_CEILING_MBPS,
                "median_of_response_medians_ms": statistics.median(
                        item["response_median_ms"] for item in items),
                "median_of_response_p90_ms": statistics.median(
                        item["response_p90_ms"] for item in items),
                "aligned_rate_change_vs_copy_percent": statistics.median(
                        aligned_rate_change),
                "rate_wins_vs_copy": sum(
                        value > 0 for value in aligned_rate_change),
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
        "schema": "s41_ffs_dmabuf_h2d_bandwidth_analysis_v1",
        "verdict": "H2D_FUNCTIONFS_DMABUF_BANDWIDTH_PASS",
        "usb_link_Mbps": 5000,
        "usb_h2d_symbol_ceiling_MBps": USB_H2D_CEILING_MBPS,
        "host_usbfs_memory_mb": 16,
        "capture_count": len(records),
        "paid_request_count": sum(
                WORKLOADS[record["workload"]]["iterations"]
                for record in records),
        "all_protocol_checks_passed": True,
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
