#!/usr/bin/env python3
"""Validate and aggregate the optimized GPU-contention campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any


BASELINE_SHA256 = (
    "800eef0d497f4dc5018ba08c74e0ff58"
    "f518cf3e8225b620e2ac7c6bfeea4a59"
)
REQUESTS_SHA256 = (
    "ccde6e3e53dee4547e4eb80f9f090032"
    "afb04b1d3fec3fd07f0f961bed60cf8f"
)
GROUPS = (
    "resident-idle",
    "busy-baseline",
    "throughput-priority",
    "cold-priority",
)
OPTIMIZED_GROUPS = ("throughput-priority", "cold-priority")
METRICS = {
    "cold_completion_p50_s": ("cold", "completion_s", "p50"),
    "cold_decode_p50_s": ("cold", "decode_s", "p50"),
    "cold_duration_s": ("cold", "duration_s"),
    "cold_output_throughput_tokens_s": (
        "cold", "output_throughput_tokens_s"
    ),
    "cold_prefill_p50_s": ("cold", "prefill_s", "p50"),
    "cold_queue_p50_s": ("cold", "queue_s", "p50"),
    "cold_service_p50_s": ("cold", "service_s", "p50"),
    "decode_host_branch_p50_ms": (
        "offload", "ffn", "host_branch_p50_ms"
    ),
    "decode_overlap_p50_ms": (
        "offload", "ffn", "decode_overlap_p50_ms"
    ),
    "decode_phone_compute_p50_ms": (
        "offload", "ffn", "decode_phone_compute_p50_ms"
    ),
    "decode_phone_rpc_p50_ms": (
        "offload", "ffn", "decode_rpc_p50_ms"
    ),
    "decode_usb_p50_ms": ("offload", "dmabuf", "decode_usb_p50_ms"),
    "gpu_power_p50_w": ("resources", "gpu_power_w", "p50"),
    "gpu_utilization_p50_pct": (
        "resources", "gpu_utilization_pct", "p50"
    ),
    "gpu_utilization_p95_pct": (
        "resources", "gpu_utilization_pct", "p95"
    ),
    "hot_output_throughput_tokens_s": (
        "hot_load", "output_throughput_tokens_s"
    ),
    "prefill_overlap_p50_ms": (
        "offload", "ffn", "prefill_overlap_p50_ms"
    ),
    "prefill_phone_compute_p50_ms": (
        "offload", "ffn", "prefill_phone_compute_p50_ms"
    ),
    "prefill_phone_rpc_p50_ms": (
        "offload", "ffn", "prefill_rpc_p50_ms"
    ),
    "prefill_usb_p50_ms": (
        "offload", "dmabuf", "prefill_usb_p50_ms"
    ),
}
BASELINE_METRICS = {
    name for name in METRICS
    if name != "hot_output_throughput_tokens_s"
}


class AnalysisError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def nested(value: dict[str, Any], path: tuple[str, ...]) -> Any:
    result: Any = value
    for key in path:
        require(type(result) is dict and key in result, f"missing {path}")
        result = result[key]
    return result


def cpu_text(cpus: list[int]) -> str:
    return ",".join(str(cpu) for cpu in cpus)


def require_affinity(
    result: dict[str, Any],
    label: str,
    expected: list[int],
) -> None:
    record = result.get("affinity_at_paid_start", {}).get(label, {})
    sets = record.get("cpu_sets")
    require(
        type(sets) is dict
        and set(sets) == {cpu_text(expected)}
        and sum(sets.values()) == record.get("thread_count"),
        f"{label} affinity",
    )


def load_optimized_run(
    path: Path,
    group: str,
    repeat: int,
) -> dict[str, Any]:
    require(path.is_absolute() and path.is_file(), f"{group} r{repeat}: path")
    result = json.loads(path.read_text(encoding="ascii"))
    require(
        result.get("schema") == "s41-gpu-contention-result-v1"
        and result.get("status") == "PASS"
        and result.get("gpu_load") == "saturated"
        and result.get("repeat_index") == repeat,
        f"{group} r{repeat}: identity",
    )
    cold = result.get("cold", {})
    dmabuf = result.get("offload", {}).get("dmabuf", {})
    ffn = result.get("offload", {}).get("ffn", {})
    require(
        cold.get("completed") == 17
        and cold.get("output_tokens") == 505
        and dmabuf.get("status") == "ok"
        and dmabuf.get("calls") == 25_776
        and ffn.get("status") == "FFN_OVERLAP_OK"
        and ffn.get("calls") == 25_776
        and ffn.get("max_columns") == 11_136
        and ffn.get("alternate_columns") == 9_664
        and ffn.get("column_quantum") == 8_192
        and ffn.get("weight_hash") == 5_393_697_135_275_110_942,
        f"{group} r{repeat}: offload receipt",
    )
    resources = result.get("resources", {})
    require(
        resources.get("cold_swap_max_bytes") == 0
        and resources.get("hot_swap_max_bytes") == 0,
        f"{group} r{repeat}: swap",
    )
    preflight = result.get("preflight", {})
    policy = preflight.get("cpu_frequency_policy", {})
    require(
        preflight.get("requests_sha256") == REQUESTS_SHA256
        and preflight.get("offload") == {
            "decode_columns": 9_664,
            "max_columns": 11_136,
            "prefill_policy": [
                [64, 8_192],
                [128, 8_192],
                [320, 11_136],
                [512, 11_136],
            ],
        }
        and policy.get("max_perf_pct") == 100
        and policy.get("min_perf_pct") == 16
        and policy.get("no_turbo") == 0
        and result.get("cpu_frequency_policy_after") == policy,
        f"{group} r{repeat}: preflight",
    )
    cold_cpus = list(range(0, 16, 2))
    e_cpus = list(range(16, 24))
    hot_cpus = (
        list(range(24))
        if group == "throughput-priority"
        else list(range(1, 16, 2)) + e_cpus
    )
    requested = preflight.get("affinity_requested", {})
    require(
        requested.get("cold") == cold_cpus
        and requested.get("control") == e_cpus
        and requested.get("bridge") == e_cpus
        and requested.get("hot") == hot_cpus,
        f"{group} r{repeat}: requested affinity",
    )
    require_affinity(result, "cold", cold_cpus)
    require_affinity(result, "control", e_cpus)
    require_affinity(result, "bridge", e_cpus)
    require_affinity(result, "hot", hot_cpus)
    result["_result_sha256"] = digest(path)
    return result


def stats(values: list[float]) -> dict[str, Any]:
    require(
        len(values) == 3
        and all(type(value) in (int, float) for value in values),
        "numeric three-run metric",
    )
    return {
        "max": max(values),
        "median": statistics.median(values),
        "min": min(values),
        "values": values,
    }


def baseline_values(
    baseline: dict[str, Any],
    name: str,
    group: str,
) -> list[float]:
    mode = "resident-idle" if group == "resident-idle" else "saturated"
    if name == "hot_output_throughput_tokens_s":
        if mode == "resident-idle":
            return [0.0, 0.0, 0.0]
        values = baseline.get("hot_output_throughput_tokens_s", {}).get(
            "values"
        )
    else:
        require(name in BASELINE_METRICS, f"baseline metric {name}")
        values = baseline.get("aggregate", {}).get(name, {}).get(
            mode, {}
        ).get("values")
    require(type(values) is list, f"baseline values {name} {group}")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-analysis", type=Path, required=True)
    parser.add_argument(
        "--throughput-priority", type=Path, nargs=3, required=True
    )
    parser.add_argument("--cold-priority", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(
        args.baseline_analysis.is_absolute()
        and args.baseline_analysis.is_file()
        and digest(args.baseline_analysis) == BASELINE_SHA256
        and args.output.is_absolute(),
        "analysis paths",
    )
    baseline = json.loads(args.baseline_analysis.read_text(encoding="ascii"))
    require(
        baseline.get("schema") == "s41-gpu-contention-analysis-v1"
        and baseline.get("status") == "PASS"
        and baseline.get("repeat_count") == 3,
        "baseline identity",
    )

    optimized = {
        group: [
            load_optimized_run(path, group, repeat)
            for repeat, path in enumerate(getattr(args, group.replace("-", "_")), 1)
        ]
        for group in OPTIMIZED_GROUPS
    }
    token_sequences = [
        [row["tokens"] for row in run["request_results"]]
        for group in OPTIMIZED_GROUPS for run in optimized[group]
    ]
    require(
        token_sequences
        and all(tokens == token_sequences[0] for tokens in token_sequences),
        "optimized token determinism",
    )

    aggregate: dict[str, dict[str, Any]] = {}
    for name, path in METRICS.items():
        aggregate[name] = {}
        for group in GROUPS:
            values = (
                baseline_values(baseline, name, group)
                if group in ("resident-idle", "busy-baseline")
                else [nested(run, path) for run in optimized[group]]
            )
            aggregate[name][group] = stats(values)

    combined: dict[str, Any] = {}
    for group in GROUPS:
        cold_values = aggregate[
            "cold_output_throughput_tokens_s"
        ][group]["values"]
        hot_values = aggregate[
            "hot_output_throughput_tokens_s"
        ][group]["values"]
        combined[group] = stats([
            cold + hot for cold, hot in zip(cold_values, hot_values)
        ])
    aggregate["active_output_throughput_tokens_s"] = combined

    busy = "busy-baseline"
    changes = {
        group: {
            name: (
                aggregate[name][group]["median"]
                / aggregate[name][busy]["median"]
                - 1.0
            ) * 100.0
            for name in aggregate
            if aggregate[name][busy]["median"] != 0
        }
        for group in OPTIMIZED_GROUPS
    }
    analysis = {
        "aggregate": aggregate,
        "baseline_analysis_sha256": BASELINE_SHA256,
        "changes_vs_busy_baseline_pct": changes,
        "groups": list(GROUPS),
        "optimized_result_sha256": {
            group: [run["_result_sha256"] for run in optimized[group]]
            for group in OPTIMIZED_GROUPS
        },
        "repeat_count": 3,
        "schema": "s41-gpu-contention-optimization-analysis-v2",
        "status": "PASS",
        "token_sequences_identical_optimized": True,
        "token_sequences_sha256": hashlib.sha256(
            canonical(token_sequences[0])
        ).hexdigest(),
    }
    args.output.write_bytes(canonical(analysis))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
