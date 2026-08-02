#!/usr/bin/env python3

import argparse
import json
import re
import statistics
from pathlib import Path


LIVE_PATTERN = re.compile(
    r"m(?P<batch>\d+)_c(?P<cols>\d+)_(?P<input>f16|i8)_rep(?P<rep>\d+)"
)


def parse_fields(line):
    fields = {}
    for token in line.split()[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value
    return fields


def find_fields(path, prefix):
    matches = [
        parse_fields(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith(prefix)
    ]
    if len(matches) != 1:
        raise ValueError(f"{path}: expected one {prefix!r} line")
    return matches[0]


def as_float(fields, name):
    return float(fields[name])


def median_field(records, name):
    return statistics.median(as_float(record, name) for record in records)


def group_live(run_root):
    groups = {}
    for case_dir in sorted(run_root.glob("m*_rep*")):
        match = LIVE_PATTERN.fullmatch(case_dir.name)
        if match is None:
            continue
        fields = find_fields(case_dir / "HOST.log", "RESULT ")
        correctness = find_fields(
            case_dir / "HOST.log",
            "CORRECTNESS treatment=activation_return ",
        )
        key = int(match.group("batch"))
        groups.setdefault(
            key,
            {
                "batch": key,
                "phone_columns": int(match.group("cols")),
                "input_type": match.group("input"),
                "records": [],
                "correctness": [],
            },
        )
        group = groups[key]
        if (
            group["phone_columns"] != int(match.group("cols"))
            or group["input_type"] != match.group("input")
        ):
            raise ValueError(f"batch {key}: inconsistent live configuration")
        group["records"].append(fields)
        group["correctness"].append(correctness)
    return groups


def group_physical(run_root):
    groups = {}
    physical_root = run_root / "physical_4060"
    for path in sorted(physical_root.glob("m*_rep*.log")):
        match = LIVE_PATTERN.fullmatch(path.stem)
        if match is None:
            continue
        fields = find_fields(path, "CUDA_DUAL_PROFILE ")
        key = int(match.group("batch"))
        groups.setdefault(
            key,
            {
                "batch": key,
                "phone_columns": int(match.group("cols")),
                "input_type": match.group("input"),
                "records": [],
            },
        )
        group = groups[key]
        if (
            group["phone_columns"] != int(match.group("cols"))
            or group["input_type"] != match.group("input")
        ):
            raise ValueError(
                f"batch {key}: inconsistent physical configuration"
            )
        group["records"].append(fields)
    return groups


def summarize_live(group):
    records = group["records"]
    correctness = group["correctness"]
    cuda_median = median_field(records, "cuda_full_median_ms")
    split_median = median_field(records, "split_total_median_ms")
    cuda_p90 = median_field(records, "cuda_full_p90_ms")
    split_p90 = median_field(records, "split_total_p90_ms")
    return {
        "batch": group["batch"],
        "phone_columns": group["phone_columns"],
        "input_type": group["input_type"],
        "repetitions": len(records),
        "cuda_full_median_ms": cuda_median,
        "split_median_ms": split_median,
        "median_change_percent": 100.0 * (split_median / cuda_median - 1.0),
        "cuda_full_p90_ms": cuda_p90,
        "split_p90_ms": split_p90,
        "p90_change_percent": 100.0 * (split_p90 / cuda_p90 - 1.0),
        "phone_median_ms": median_field(records, "phone_total_median_ms"),
        "phone_p90_ms": median_field(records, "phone_total_p90_ms"),
        "hidden_fraction": median_field(records, "hidden_fraction"),
        "max_treatment_relative_l2": max(
            as_float(record, "rel_l2") for record in correctness
        ),
        "all_row_argmax_exact": all(
            record["row_argmax"].split("/")[0]
            == record["row_argmax"].split("/")[1]
            for record in correctness
        ),
        "all_finite": all(
            int(record["non_finite"]) == 0 for record in correctness
        ),
    }


def summarize_physical(group):
    records = group["records"]
    cuda_median = median_field(records, "cuda_full_median_ms")
    delayed_median = median_field(records, "delayed_dual_median_ms")
    cuda_p90 = median_field(records, "cuda_full_p90_ms")
    delayed_p90 = median_field(records, "delayed_dual_p90_ms")
    return {
        "batch": group["batch"],
        "phone_columns": group["phone_columns"],
        "input_type": group["input_type"],
        "repetitions": len(records),
        "injected_phone_delay_us": median_field(
            records, "profile_delay_us"
        ),
        "cuda_full_median_ms": cuda_median,
        "delayed_dual_median_ms": delayed_median,
        "median_change_percent": 100.0
        * (delayed_median / cuda_median - 1.0),
        "cuda_full_p90_ms": cuda_p90,
        "delayed_dual_p90_ms": delayed_p90,
        "p90_change_percent": 100.0 * (delayed_p90 / cuda_p90 - 1.0),
        "cuda_prefix_median_ms": median_field(
            records, "cuda_prefix_median_ms"
        ),
        "cuda_late_median_ms": median_field(
            records, "cuda_late_median_ms"
        ),
        "actual_launch_delay_us": 1000.0
        * median_field(records, "launch_delay_median_ms"),
        "max_relative_l2": max(
            as_float(record, "rel_l2") for record in records
        ),
        "all_row_argmax_exact": all(
            record["row_argmax"].split("/")[0]
            == record["row_argmax"].split("/")[1]
            for record in records
        ),
        "all_finite": all(
            int(record["non_finite"]) == 0 for record in records
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()

    live_groups = group_live(args.run_root)
    physical_groups = group_physical(args.run_root)
    if set(live_groups) != set(physical_groups):
        raise ValueError("live and physical batch sets differ")

    live = {
        str(batch): summarize_live(live_groups[batch])
        for batch in sorted(live_groups)
    }
    physical = {
        str(batch): summarize_physical(physical_groups[batch])
        for batch in sorted(physical_groups)
    }
    proxy_error = {}
    for batch in sorted(live_groups):
        live_cuda = live[str(batch)]["cuda_full_median_ms"]
        physical_cuda = physical[str(batch)]["cuda_full_median_ms"]
        proxy_error[str(batch)] = {
            "a6000_over_4060_latency_ratio": live_cuda / physical_cuda,
            "a6000_minus_4060_percent": 100.0
            * (live_cuda / physical_cuda - 1.0),
        }

    result = {
        "schema": "s41-causal-batch-v1",
        "run_root": str(args.run_root),
        "live_a6000_plus_op15": live,
        "physical_4060_delay_injection": physical,
        "a6000_proxy_error": proxy_error,
        "claim_limits": [
            "Live co-inference is A6000 plus OP15, not 4060 Ti plus OP15.",
            "The physical 4060 treatment injects a measured phone delay but does not carry live USB traffic.",
            "This is one synthetic Qwen3-14B FFN layer, not full-model prefill.",
            "No batched energy result is included.",
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
