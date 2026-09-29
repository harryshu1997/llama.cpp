#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path


MIB = 1048576
CONTROL_BYTES = 64
DIRECTION_CEILING_MBPS = 500.0

CASES = {
    "h2d_1m": {
        "request_bytes": MIB,
        "response_bytes": CONTROL_BYTES,
        "host_mode": "async",
        "warmup": 20,
        "iterations": 200,
        "depth": 4,
        "variant": "dmabuf_devmem_q4",
    },
    "d2h_1m": {
        "request_bytes": CONTROL_BYTES,
        "response_bytes": MIB,
        "host_mode": "async",
        "warmup": 20,
        "iterations": 200,
        "depth": 4,
        "variant": "dmabuf_devmem_q4",
    },
    "duplex_1m": {
        "request_bytes": MIB,
        "response_bytes": MIB,
        "host_mode": "async",
        "warmup": 20,
        "iterations": 200,
        "depth": 4,
        "variant": "dmabuf_devmem_q4",
    },
    "h2d_4m": {
        "request_bytes": 4 * MIB,
        "response_bytes": CONTROL_BYTES,
        "host_mode": "async",
        "warmup": 10,
        "iterations": 100,
        "depth": 3,
        "variant": "dmabuf_devmem_q3",
    },
    "d2h_4m": {
        "request_bytes": CONTROL_BYTES,
        "response_bytes": 4 * MIB,
        "host_mode": "async",
        "warmup": 10,
        "iterations": 100,
        "depth": 3,
        "variant": "dmabuf_devmem_q3",
    },
    "h2d_15m": {
        "request_bytes": 15 * MIB,
        "response_bytes": CONTROL_BYTES,
        "host_mode": "sync",
        "warmup": 3,
        "iterations": 30,
        "depth": 1,
        "variant": "dmabuf_devmem",
    },
    "d2h_15m": {
        "request_bytes": CONTROL_BYTES,
        "response_bytes": 15 * MIB,
        "host_mode": "sync",
        "warmup": 3,
        "iterations": 30,
        "depth": 1,
        "variant": "dmabuf_devmem",
    },
}

NAME_PATTERN = re.compile(
    r"^(?P<case>h2d_1m|d2h_1m|duplex_1m|"
    r"h2d_4m|d2h_4m|h2d_15m|d2h_15m)\."
    r"(?P<variant>dmabuf_devmem(?:_q[34])?)\."
    r"r(?P<repetition>[123])$")


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
        f"{case_name}.{config['variant']}.r{repetition}"
        for case_name, config in CASES.items()
        for repetition in range(1, 4)
    }


def validate_capture(root, base_name):
    match = NAME_PATTERN.fullmatch(base_name)
    if match is None:
        raise ValueError(f"invalid capture name: {base_name}")
    case_name = match.group("case")
    repetition = int(match.group("repetition"))
    config = CASES[case_name]
    request_bytes = config["request_bytes"]
    response_bytes = config["response_bytes"]
    if match.group("variant") != config["variant"]:
        raise ValueError(f"{base_name}: invalid variant")
    data = json.loads((root / f"{base_name}.json").read_text(
            encoding="utf-8"))

    exact = {
        "schema": "s41_ffs_dmabuf_transport_v1",
        "case_name": base_name,
        "device_mode": "dmabuf",
        "host_mode": config["host_mode"],
        "host_allocator": "devmem",
        "request_bytes": request_bytes,
        "response_bytes": response_bytes,
        "warmup": config["warmup"],
        "iterations": config["iterations"],
        "queue_depth": config["depth"],
    }
    for key, value in exact.items():
        if data.get(key) != value:
            raise ValueError(f"{base_name}: invalid {key}")

    response_values = data["response_ready_samples_ms"]
    out_values = data["out_completion_samples_ms"]
    tail_values = data["post_out_tail_samples_ms"]
    for label, values in (
            ("response", response_values),
            ("out", out_values),
            ("tail", tail_values)):
        if len(values) != config["iterations"]:
            raise ValueError(f"{base_name}: invalid {label} sample count")
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(f"{base_name}: invalid {label} sample")
    for index, (out_value, tail_value, response_value) in enumerate(zip(
            out_values, tail_values, response_values)):
        require_close(tail_value, max(response_value - out_value, 0.0),
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
        (request_bytes + response_bytes) * config["iterations"] /
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

    aggregate = data["aggregate_payload_MBps"]
    total_bytes = request_bytes + response_bytes
    return {
        "case": case_name,
        "repetition": repetition,
        "aggregate_MBps": aggregate,
        "h2d_MBps": aggregate * request_bytes / total_bytes,
        "d2h_MBps": aggregate * response_bytes / total_bytes,
        "queue_residency_median_ms": data["response_ready_median_ms"],
        "queue_residency_p90_ms": data["response_ready_p90_ms"],
    }


def summarize(records):
    by_key = {
        (record["case"], record["repetition"]): record
        for record in records
    }
    summary = {"cases": {}}
    for case_name in CASES:
        items = [by_key[(case_name, repetition)]
                 for repetition in range(1, 4)]
        median_aggregate = statistics.median(
                item["aggregate_MBps"] for item in items)
        median_h2d = statistics.median(
                item["h2d_MBps"] for item in items)
        median_d2h = statistics.median(
                item["d2h_MBps"] for item in items)
        summary["cases"][case_name] = {
            "median_aggregate_MBps": median_aggregate,
            "median_h2d_MBps": median_h2d,
            "median_d2h_MBps": median_d2h,
            "aggregate_ceiling_utilization_percent":
                100.0 * median_aggregate /
                (2.0 * DIRECTION_CEILING_MBPS
                 if case_name == "duplex_1m"
                 else DIRECTION_CEILING_MBPS),
            "median_of_queue_residency_medians_ms": statistics.median(
                    item["queue_residency_median_ms"] for item in items),
            "median_of_queue_residency_p90_ms": statistics.median(
                    item["queue_residency_p90_ms"] for item in items),
        }

    cases = summary["cases"]
    independent_h2d = cases["h2d_1m"]["median_h2d_MBps"]
    independent_d2h = cases["d2h_1m"]["median_d2h_MBps"]
    duplex_h2d = cases["duplex_1m"]["median_h2d_MBps"]
    duplex_d2h = cases["duplex_1m"]["median_d2h_MBps"]
    summary["comparison"] = {
        "d2h_change_vs_h2d_percent": {
            size: 100.0 * (
                cases[f"d2h_{size}"]["median_d2h_MBps"] /
                cases[f"h2d_{size}"]["median_h2d_MBps"] - 1.0)
            for size in ("1m", "4m", "15m")
        },
        "duplex_h2d_change_vs_independent_percent":
            100.0 * (duplex_h2d / independent_h2d - 1.0),
        "duplex_d2h_change_vs_independent_percent":
            100.0 * (duplex_d2h / independent_d2h - 1.0),
        "duplex_aggregate_change_vs_independent_sum_percent":
            100.0 * (
                cases["duplex_1m"]["median_aggregate_MBps"] /
                (independent_h2d + independent_d2h) - 1.0),
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    expected = expected_names()
    ignored_json = {args.output.name, "ANALYSIS.json"}
    actual = {
        path.stem for path in args.result_root.glob("*.json")
        if path.name not in ignored_json
    }
    if actual != expected:
        raise ValueError(
            f"capture set mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}")

    records = [validate_capture(args.result_root, name)
               for name in sorted(expected)]
    evidence_files = sorted(
        path for path in args.result_root.iterdir()
        if path.is_file() and path != args.output and
        path.name not in {"ANALYSIS.json", "SHA256SUMS.txt"}
    )
    output = {
        "schema": "s41_ffs_dmabuf_bidirectional_analysis_v1",
        "verdict": "FUNCTIONFS_DMABUF_BIDIRECTIONAL_PASS",
        "usb_link_Mbps": 5000,
        "direction_symbol_ceiling_MBps": DIRECTION_CEILING_MBPS,
        "full_duplex_symbol_ceiling_MBps":
            2.0 * DIRECTION_CEILING_MBPS,
        "capture_count": len(records),
        "paid_request_count": sum(
            CASES[record["case"]]["iterations"] for record in records),
        "all_protocol_checks_passed": True,
        "all_kernel_fault_logs_empty": True,
        "all_terminal_states_valid": True,
        "summary": summarize(records),
        "evidence_sha256": {
            path.name: sha256(path) for path in evidence_files
        },
    }
    args.output.write_text(
            json.dumps(output, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    print(json.dumps({
        "verdict": output["verdict"],
        "capture_count": output["capture_count"],
        "paid_request_count": output["paid_request_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
