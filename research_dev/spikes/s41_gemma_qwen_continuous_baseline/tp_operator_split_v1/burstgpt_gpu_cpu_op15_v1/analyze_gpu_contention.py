#!/usr/bin/env python3
"""Validate and aggregate the paired GPU-contention campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any


MODES = ("resident-idle", "saturated")
REPEATS = (1, 2, 3)
REQUESTS_SHA256 = (
    "ccde6e3e53dee4547e4eb80f9f090032"
    "afb04b1d3fec3fd07f0f961bed60cf8f"
)
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


def load_run(root: Path, mode: str, repeat: int) -> dict[str, Any]:
    run_root = root / f"gpu-contention-{mode}-r{repeat}"
    result_path = run_root / "RESULT.json"
    result = json.loads(result_path.read_text())
    require(
        result.get("schema") == "s41-gpu-contention-result-v1"
        and result.get("status") == "PASS"
        and result.get("gpu_load") == mode
        and result.get("repeat_index") == repeat,
        f"{mode} r{repeat}: identity",
    )
    cold = result.get("cold", {})
    require(
        cold.get("completed") == 17 and cold.get("output_tokens") == 505,
        f"{mode} r{repeat}: cold conservation",
    )
    dmabuf = result.get("offload", {}).get("dmabuf", {})
    ffn = result.get("offload", {}).get("ffn", {})
    require(
        dmabuf.get("status") == "ok"
        and dmabuf.get("calls") == 25_776
        and ffn.get("status") == "FFN_OVERLAP_OK"
        and ffn.get("calls") == 25_776,
        f"{mode} r{repeat}: offload receipt",
    )
    resources = result.get("resources", {})
    require(
        resources.get("cold_swap_max_bytes") == 0
        and resources.get("hot_swap_max_bytes") == 0,
        f"{mode} r{repeat}: swap",
    )
    preflight = result.get("preflight", {})
    policy = preflight.get("cpu_frequency_policy", {})
    require(
        preflight.get("requests_sha256") == REQUESTS_SHA256
        and policy.get("max_perf_pct") == 100
        and policy.get("min_perf_pct") == 16
        and policy.get("no_turbo") == 0
        and result.get("cpu_frequency_policy_after") == policy,
        f"{mode} r{repeat}: frequency policy",
    )
    worker = (run_root / "phone-worker.log").read_text()
    session = (run_root / "phone-session.log").read_text()
    require(
        "DMA-BUF complete requests=25776 status=0" in worker
        and "worker_status=0" in session,
        f"{mode} r{repeat}: phone receipt",
    )
    result["_result_sha256"] = digest(result_path)
    return result


def aggregate(runs: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    result = {}
    for name, path in METRICS.items():
        record = {}
        for mode in MODES:
            values = [nested(run, path) for run in runs[mode]]
            require(
                all(type(value) in (int, float) for value in values),
                f"{name}: numeric",
            )
            record[mode] = {
                "max": max(values),
                "median": statistics.median(values),
                "min": min(values),
                "values": values,
            }
        idle = record["resident-idle"]["median"]
        busy = record["saturated"]["median"]
        record["saturated_change_pct"] = (
            (busy / idle - 1.0) * 100.0 if idle else None
        )
        result[name] = record
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.root.is_absolute() and args.root.is_dir(), "root")
    require(args.output.is_absolute(), "output")

    kernel_scan = args.root / "gpu-contention-kernel-fault-scan.txt"
    require(kernel_scan.exists() and not kernel_scan.read_bytes(), "kernel scan")

    runs = {
        mode: [load_run(args.root, mode, repeat) for repeat in REPEATS]
        for mode in MODES
    }
    token_sequences = [
        [tuple(row["tokens"]) for row in run["request_results"]]
        for mode in MODES for run in runs[mode]
    ]
    require(
        all(tokens == token_sequences[0] for tokens in token_sequences),
        "token determinism",
    )

    hot_throughput = [
        run["hot_load"]["output_throughput_tokens_s"]
        for run in runs["saturated"]
    ]
    analysis = {
        "aggregate": aggregate(runs),
        "hot_output_throughput_tokens_s": {
            "max": max(hot_throughput),
            "median": statistics.median(hot_throughput),
            "min": min(hot_throughput),
            "values": hot_throughput,
        },
        "modes": list(MODES),
        "repeat_count": len(REPEATS),
        "result_sha256": {
            mode: [run["_result_sha256"] for run in runs[mode]]
            for mode in MODES
        },
        "schema": "s41-gpu-contention-analysis-v1",
        "status": "PASS",
        "token_sequences_identical": True,
    }
    args.output.write_bytes(canonical(analysis))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
