#!/usr/bin/env python3
"""Guard a generic Llama FFN split table with physical shape timings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import compile_llama_ffn_split as compiler


SCHEMA = "s42-llama-ffn-physical-shape-calibration-v2"
SPLIT_UPPER_ERROR_PPM = 250_000
CPU_LOWER_ERROR_PPM = 200_000
MIN_PASSING_SHAPES = 3
MIN_PASSING_CALLS = 5


class CalibrationError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CalibrationError(message)


def canonical(value: object) -> bytes:
    return compiler.canonical(value)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    require(type(value) is dict, f"object expected: {path}")
    return value


def verified_record(
    value: dict[str, Any],
    schema: str,
    label: str,
) -> None:
    supplied = value.get("record_sha256")
    unsigned = {
        name: row for name, row in value.items()
        if name != "record_sha256"
    }
    require(
        value.get("schema") == schema
        and supplied == hashlib.sha256(canonical(unsigned)).hexdigest(),
        f"{label} record identity",
    )


def physical_shapes(content: bytes) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = None
    shapes = []
    for line in content.splitlines():
        for prefix, target in (
            (b"S41SERVERFFN ", "summary"),
            (b"S41SERVERFFNSHAPE ", "shape"),
        ):
            position = line.find(prefix)
            if position < 0:
                continue
            try:
                value = json.loads(line[position + len(prefix):])
            except json.JSONDecodeError:
                continue
            if type(value) is not dict:
                continue
            if target == "summary":
                summary = value
            else:
                shapes.append(value)
            break
    require(
        type(summary) is dict
        and summary.get("status") == "ok"
        and type(summary.get("calls")) is int
        and summary["calls"] > 0,
        "physical FFN summary",
    )
    required = ("tokens", "columns", "calls", "overlap_mean_ms")
    require(
        shapes
        and all(
            type(row.get("tokens")) is int
            and row["tokens"] > 0
            and type(row.get("columns")) is int
            and row["columns"] >= 0
            and type(row.get("calls")) is int
            and row["calls"] > 0
            and type(row.get("overlap_mean_ms")) in {int, float}
            and not isinstance(row["overlap_mean_ms"], bool)
            and row["overlap_mean_ms"] > 0
            for row in shapes
        ),
        "physical FFN shape summaries: " + ",".join(required),
    )
    return summary, sorted(shapes, key=lambda row: row["tokens"])


def cpu_baseline_points(
    estimated: dict[str, Any],
) -> list[tuple[int, int]]:
    points = []
    for decision in estimated.get("vq_decisions", []):
        shape = decision.get("shape", {})
        tokens = shape.get("m")
        latency_us = decision.get("cpu_baseline_finish_us")
        require(
            type(tokens) is int
            and tokens > 0
            and type(latency_us) is int
            and latency_us > 0,
            "estimated CPU baseline point",
        )
        points.append((tokens, latency_us))
    points.sort()
    require(
        points
        and len({tokens for tokens, _ in points}) == len(points),
        "ordered CPU baseline points",
    )
    return points


def interpolate(points: list[tuple[int, int]], tokens: int) -> int:
    if tokens <= points[0][0]:
        return points[0][1]
    for left, right in zip(points, points[1:]):
        if tokens <= right[0]:
            numerator = (
                (tokens - left[0]) * (right[1] - left[1])
            )
            denominator = right[0] - left[0]
            return left[1] + math.ceil(numerator / denominator)
    return points[-1][1]


def calibrate(
    manifest: dict[str, Any],
    estimated: dict[str, Any],
    physical_log: bytes,
    source_label: str,
    physical_log_sha256: str,
) -> dict[str, Any]:
    compiler.verify_manifest(manifest)
    verified_record(estimated, compiler.SCHEMA, "estimated policy")
    require(
        source_label
        and source_label.isascii()
        and estimated.get("qualification", {}).get("status") == "ESTIMATED"
        and estimated.get("evidence", {}).get("manifest_record_sha256")
            == manifest["record_sha256"],
        "generic policy calibration source",
    )
    summary, shapes = physical_shapes(physical_log)
    contract = compiler.model_contract(manifest)
    max_tokens = contract["max_tokens"]
    max_columns = contract["n_ff"]
    require(
        shapes[-1]["tokens"] == max_tokens
        and all(
            row["tokens"] <= max_tokens
            and row["columns"] <= max_columns
            for row in shapes
        ),
        "physical shape coverage",
    )
    points = cpu_baseline_points(estimated)
    observations = []
    for row in shapes:
        cpu_mean_us = interpolate(points, row["tokens"])
        cpu_lower_us = (
            cpu_mean_us * (1_000_000 - CPU_LOWER_ERROR_PPM)
        ) // 1_000_000
        split_mean_us = math.ceil(row["overlap_mean_ms"] * 1000)
        split_upper_us = math.ceil(
            split_mean_us * (1_000_000 + SPLIT_UPPER_ERROR_PPM)
            / 1_000_000
        )
        observations.append({
            "calls": row["calls"],
            "cpu_estimated_lower_us": cpu_lower_us,
            "cpu_estimated_mean_us": cpu_mean_us,
            "observed_phone_columns": row["columns"],
            "pass": (
                row["columns"] == max_columns
                and split_upper_us < cpu_lower_us
            ),
            "split_observed_mean_us": split_mean_us,
            "split_observed_upper_us": split_upper_us,
            "tokens": row["tokens"],
        })
    suffix = []
    for row in reversed(observations):
        if not row["pass"]:
            break
        suffix.append(row)
    suffix.reverse()
    require(
        len(suffix) >= MIN_PASSING_SHAPES
        and sum(row["calls"] for row in suffix) >= MIN_PASSING_CALLS,
        "no repeated energy-safe physical prefill suffix",
    )
    first_phone_tokens = suffix[0]["tokens"]
    cpu_cutoff = first_phone_tokens - 1
    buckets = []
    if cpu_cutoff > 0:
        buckets.append({
            "cpu_columns": max_columns,
            "max_tokens": cpu_cutoff,
            "phone_columns": 0,
        })
    buckets.append({
        "cpu_columns": 0,
        "max_tokens": max_tokens,
        "phone_columns": max_columns,
    })
    policy_text = ",".join(
        f"{row['max_tokens']}:{row['phone_columns']}" for row in buckets
    )
    result: dict[str, Any] = {
        "compiled_buckets": buckets,
        "evidence": {
            "estimated_policy_record_sha256": estimated["record_sha256"],
            "manifest_record_sha256": manifest["record_sha256"],
            "physical_log_sha256": physical_log_sha256,
            "source_label": source_label,
        },
        "model_contract": contract,
        "observations": observations,
        "placement_contract": compiler.placement_contract(manifest),
        "physical_summary": {
            "calls": summary["calls"],
            "decode_calls": summary.get("decode_calls"),
            "prefill_calls": summary.get("prefill_calls"),
        },
        "policy_text": policy_text,
        "qualification": {
            "cpu_lower_error_ppm": CPU_LOWER_ERROR_PPM,
            "minimum_passing_calls": MIN_PASSING_CALLS,
            "minimum_passing_shapes": MIN_PASSING_SHAPES,
            "route_admission": (
                "SHADOW_ONLY_UNTIL_HELDOUT_PHYSICAL_PROFILE"
            ),
            "split_upper_error_ppm": SPLIT_UPPER_ERROR_PPM,
        },
        "schema": SCHEMA,
        "source_policy_text": estimated["policy_text"],
        "status": "PASS",
    }
    result["record_sha256"] = hashlib.sha256(canonical(result)).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--estimated-policy", type=Path, required=True)
    parser.add_argument("--physical-log", type=Path, required=True)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(
        args.output.is_absolute() and not args.output.exists(),
        "new absolute output path",
    )
    value = calibrate(
        load(args.manifest),
        load(args.estimated_policy),
        args.physical_log.read_bytes(),
        args.source_label,
        digest(args.physical_log),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical(value))
    print(json.dumps({
        "output": str(args.output),
        "policy": value["policy_text"],
        "status": value["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
