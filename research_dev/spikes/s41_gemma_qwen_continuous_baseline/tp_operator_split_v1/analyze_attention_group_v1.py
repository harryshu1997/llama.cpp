#!/usr/bin/env python3
"""Aggregate the bounded S41 GQA-group attention experiment."""

import argparse
import json
import re
import statistics
from pathlib import Path


HOST_PATTERN = re.compile(r"HOST_KV([0-9]+)_G1_R([0-9]+)\.log$")
CORRECTNESS_LIMITS = {
    "local_partition": 0.0001,
    "phone": 0.03,
    "split": 0.005,
}


def fields(line):
    result = {}
    for token in line.split()[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        result[key] = value
    return result


def exact_pair(value):
    left, right = value.split("/", 1)
    return left == right


def parse_host(path):
    correctness = {}
    result = None
    verdict = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("ATTENTION_CORRECTNESS "):
            record = fields(line)
            name = record.get(
                    "control",
                    record.get("component", record.get("treatment")))
            correctness[name] = record
        elif line.startswith("ATTENTION_RESULT "):
            result = fields(line)
        elif line.startswith("ATTENTION_VERDICT "):
            verdict = fields(line)
    if result is None or verdict is None:
        raise ValueError(f"missing result or verdict in {path}")
    if set(correctness) != set(CORRECTNESS_LIMITS):
        raise ValueError(f"incomplete correctness evidence in {path}")

    checks = {}
    for name, limit in CORRECTNESS_LIMITS.items():
        record = correctness[name]
        checks[name] = (
            int(record["non_finite"]) == 0
            and float(record["rel_l2"]) <= limit
            and exact_pair(record["argmax"])
        )
    return {
        "file": path.name,
        "n_kv": int(result["n_kv"]),
        "iterations": int(result["n"]),
        "cuda_full_median_ms": float(result["cuda_full_median_ms"]),
        "cuda_full_p90_ms": float(result["cuda_full_p90_ms"]),
        "cuda_shard_median_ms": float(result["cuda_shard_median_ms"]),
        "phone_total_median_ms": float(result["phone_total_median_ms"]),
        "phone_total_p90_ms": float(result["phone_total_p90_ms"]),
        "split_total_median_ms": float(result["split_total_median_ms"]),
        "split_total_p90_ms": float(result["split_total_p90_ms"]),
        "hidden_fraction": float(result["hidden_fraction"]),
        "latency_change_percent": float(
                verdict["latency_change_percent"]),
        "correctness": checks,
        "correctness_pass": all(checks.values()),
        "cache_mode": result["cache_mode"],
    }


def aggregate(records):
    return {
        "repetitions": len(records),
        "iterations_per_repetition": sorted({
            record["iterations"] for record in records
        }),
        "cuda_full_median_ms": statistics.median(
            record["cuda_full_median_ms"] for record in records),
        "cuda_full_p90_ms": statistics.median(
            record["cuda_full_p90_ms"] for record in records),
        "phone_total_median_ms": statistics.median(
            record["phone_total_median_ms"] for record in records),
        "phone_total_p90_ms": statistics.median(
            record["phone_total_p90_ms"] for record in records),
        "split_total_median_ms": statistics.median(
            record["split_total_median_ms"] for record in records),
        "split_total_p90_ms": statistics.median(
            record["split_total_p90_ms"] for record in records),
        "latency_change_percent": statistics.median(
            record["latency_change_percent"] for record in records),
        "hidden_fraction": statistics.median(
            record["hidden_fraction"] for record in records),
        "all_correctness_pass": all(
            record["correctness_pass"] for record in records),
        "all_median_wins": all(
            record["split_total_median_ms"]
            < record["cuda_full_median_ms"]
            for record in records),
        "all_p90_wins": all(
            record["split_total_p90_ms"]
            < record["cuda_full_p90_ms"]
            for record in records),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    records = []
    for path in sorted(args.evidence_root.glob("HOST_KV*_G1_R*.log")):
        match = HOST_PATTERN.match(path.name)
        if match is None:
            continue
        record = parse_host(path)
        if record["n_kv"] != int(match.group(1)):
            raise ValueError(f"context mismatch in {path}")
        record["repetition"] = int(match.group(2))
        records.append(record)
    if not records:
        raise ValueError("no host evidence found")

    by_context = {}
    for record in records:
        by_context.setdefault(record["n_kv"], []).append(record)
    expected_repetitions = {512: 1, 2048: 1, 4096: 3, 8192: 3}
    if {
            context: len(items)
            for context, items in by_context.items()
    } != expected_repetitions:
        raise ValueError("unexpected context or repetition set")
    if any(
            record["cache_mode"] != "last_slot_update"
            for record in records):
        raise ValueError("non-stateful attention evidence included")

    aggregates = {
        str(context): aggregate(items)
        for context, items in sorted(by_context.items())
    }
    output = {
        "schema": "s41-attention-group-analysis-v1",
        "mechanism": {
            "model_shape": "qwen3_14b",
            "weight_type": "q8_0",
            "gpu_groups": 7,
            "phone_groups": 1,
            "q_heads_per_phone_group": 5,
            "kv_heads_per_phone_group": 1,
            "cache_mode": "last_slot_update",
        },
        "aggregates": aggregates,
        "raw_runs": records,
        "policy_interpretation": {
            "512": "reject_latency",
            "2048": "reject_latency",
            "4096": "diagnostic_median_only_tail_regression",
            "8192": "prototype_eligible_median_and_p90",
        },
        "claim_limits": [
            "One synthetic Qwen3-14B attention projection and KV-cache slice.",
            "No RoPE, Q/K normalization, causal mask, or full decoder layer.",
            "A6000 at 240/5001 MHz is a kernel-specific latency proxy only.",
            "No physical RTX 4060 Ti treatment, energy, or multi-phone result.",
        ],
    }
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
