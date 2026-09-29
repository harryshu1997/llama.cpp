#!/usr/bin/env python3
"""Bind same-process Gemma GPU-weight adoption to physical receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from .analyze_prefetch_fence_run import (
        TOKEN_EMBEDDING_BYTES,
        TOKEN_EMBEDDING_OFFSET,
        canonical,
        digest,
        parse_bridge,
        parse_value,
        require,
    )
    from .analyze_prefetch_qualification import (
        gpu_utilization,
        verify_canonical_record,
    )
except ImportError:
    from analyze_prefetch_fence_run import (
        TOKEN_EMBEDDING_BYTES,
        TOKEN_EMBEDDING_OFFSET,
        canonical,
        digest,
        parse_bridge,
        parse_value,
        require,
    )
    from analyze_prefetch_qualification import (
        gpu_utilization,
        verify_canonical_record,
    )


SCHEMA = "s42-op15-same-process-weight-adoption-qualification-v1"
RUN_NAME = "adoptable-qualified-r5"
DEFAULT_RESULT_ROOT = (
    Path(__file__).resolve().parent
    / "results/OP15_SAME_PROCESS_ADOPTION_V1"
)
EXPECTED_QWEN_TOKENS = {
    31: "cdd59190ad2eb46abc7dffe3c8f4c1f6cd3587bb280a68878dd7e9057daf1c87",
    52: "1ba210c91c117d2a6a9c63ebe221556d120836a274b5f15bc93740b0d9626db2",
    53: "e2e40e0c9644caa64038ca92ded4893e28ad5333acc06a5b99400110a6e2e57b",
}
EXPECTED_GEMMA_TOKENS = (
    "b902347e9dae1ce33fe87cc507b5610b"
    "bbc0de62d93629cee9a8d7f45a73eec6"
)
PROCESS_MEMORY_SCHEMA = "s42-process-memory-monitor-v1"
CGROUP_MEMORY_SCHEMA = "s42-cgroup-memory-receipt-v1"
GEMMA_EXECUTION_SCHEMA = "s42-adopted-gemma-execution-v1"


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    require(type(value) is dict, f"object: {path}")
    return value


def parse_line(path: Path, prefix: str) -> dict[str, object]:
    rows = [
        line.removeprefix(prefix).strip()
        for line in path.read_text(encoding="ascii").splitlines()
        if line.startswith(prefix)
    ]
    require(len(rows) == 1, f"one {prefix.strip()} row")
    result: dict[str, object] = {}
    for item in rows[0].split():
        key, separator, value = item.partition("=")
        require(separator == "=" and key and key not in result,
                f"{prefix.strip()} field")
        result[key] = parse_value(value)
    return result


def adoption_windows(path: Path) -> dict[str, float | int]:
    copied_bytes = 0
    copied_chunks = 0
    copy_active_ms = 0.0
    copied_windows = 0
    total_windows = 0
    for line in path.read_text(encoding="ascii").splitlines():
        if not line.startswith("S42_FENCED_TENSOR_WINDOW "):
            continue
        total_windows += 1
        fields = dict(item.split("=", 1) for item in line.split()[1:])
        copied = int(fields["copied_bytes"])
        if copied == 0:
            continue
        copied_windows += 1
        copied_bytes += copied
        copied_chunks += int(fields["copied_chunks"])
        copy_active_ms += float(fields["copy_verify_ms"])
    return {
        "copied_bytes": copied_bytes,
        "copied_chunks": copied_chunks,
        "copy_active_ms": copy_active_ms,
        "copied_windows": copied_windows,
        "total_windows": total_windows,
    }


def command_value(command: object, option: str) -> str | None:
    require(type(command) is list, "server command")
    matches = [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == option
    ]
    require(len(matches) <= 1, f"unique command option: {option}")
    return None if not matches else matches[0]


def build(root: Path = DEFAULT_RESULT_ROOT) -> dict[str, Any]:
    raw = root / "raw"
    qwen_path = raw / RUN_NAME / "RESULT.json"
    resource_path = raw / RUN_NAME / "resource-samples.jsonl"
    phone_root = raw / f"{RUN_NAME}.phone-capture"
    adoption_root = raw / f"{RUN_NAME}.adoption"
    phone_path = phone_root / "PHONE_ENERGY_V3.json"
    bridge_path = phone_root / "bridge.stderr"
    gemma_log_path = adoption_root / "gemma.stderr"
    process_path = adoption_root / "PROCESS_MEMORY_V1.json"
    cgroup_path = adoption_root / "CGROUP_MEMORY_V1.json"
    execution_path = adoption_root / "GEMMA_EXECUTION_V1.json"

    qwen = read_object(qwen_path)
    phone = read_object(phone_path)
    process = read_object(process_path)
    cgroup = read_object(cgroup_path)
    execution = read_object(execution_path)
    ready = parse_line(gemma_log_path, "S42_FENCED_TENSOR_READY ")
    adoption = parse_line(gemma_log_path, "S42_FENCED_TENSOR_RESULT ")
    windows = adoption_windows(gemma_log_path)
    bridge = parse_bridge(bridge_path)

    verify_canonical_record(process, PROCESS_MEMORY_SCHEMA)
    verify_canonical_record(cgroup, CGROUP_MEMORY_SCHEMA)
    verify_canonical_record(execution, GEMMA_EXECUTION_SCHEMA)
    require(
        qwen.get("schema") == "s41-burstgpt-llama-server-result-v1"
        and qwen.get("status") == "PASS"
        and qwen.get("arm") == "op15"
        and qwen.get("dispatch") == "sequential"
        and qwen.get("indices") == [52, 53, 31]
        and qwen.get("metrics", {}).get("requests") == 3,
        "Qwen result",
    )
    require(
        command_value(qwen.get("server_command"), "--n-gpu-layers")
            == "15"
        and command_value(qwen.get("server_command"), "--device")
            == "CUDA0",
        "Qwen placement",
    )
    qwen_hashes = {
        row["request_index"]: hashlib.sha256(
            canonical(row["tokens"])
        ).hexdigest()
        for row in qwen["request_results"]
    }
    require(qwen_hashes == EXPECTED_QWEN_TOKENS, "exact Qwen output")
    require(
        phone.get("schema") == "s41-phone-energy-v3"
        and phone.get("status") == "PASS"
        and phone.get("boundary") == "paid_trace_interval"
        and phone.get("duration_s") == qwen["metrics"]["duration_s"],
        "phone energy receipt",
    )
    require(
        bridge.get("status") == "ok"
        and bridge.get("calls") == 1296
        and bridge.get("reset_recoveries") == 0
        and bridge.get("prefetch_fence_enabled") is True
        and bridge.get("prefetch_fence_completed") is True
        and bridge.get("prefetch_expected_total_bytes")
            == TOKEN_EMBEDDING_BYTES
        and bridge.get("prefetch_group_first_layer") == 0
        and bridge.get("prefetch_group_last_layer") == 11
        and bridge.get("prefetch_fence_calls") == 108
        and bridge.get("prefetch_copied_bytes")
            == TOKEN_EMBEDDING_BYTES
        and bridge.get("prefetch_copied_chunks") == 480,
        "bridge adoption fence",
    )
    require(
        ready.get("tensor") == "token_embd.weight"
        and ready.get("offset") == TOKEN_EMBEDDING_OFFSET
        and ready.get("bytes") == TOKEN_EMBEDDING_BYTES
        and ready.get("source_pinned") is True
        and ready.get("warmup_verified") is True
        and ready.get("free_bytes", 0) >= ready.get("reserve_bytes", 1),
        "adoption source-ready",
    )
    require(
        adoption.get("status") == "PASS"
        and adoption.get("tensor") == "token_embd.weight"
        and adoption.get("source_offset") == TOKEN_EMBEDDING_OFFSET
        and adoption.get("bytes") == TOKEN_EMBEDDING_BYTES
        and adoption.get("fence_calls") == 108
        and adoption.get("armed_calls") == 54
        and adoption.get("copy_windows") == 54
        and adoption.get("copied_chunks") == 480
        and adoption.get("verified") is True
        and adoption.get("source_pinned") is True
        and adoption.get("adoptable") is True
        and adoption.get("gpu_free_min_bytes", 0)
            >= adoption.get("gpu_reserve_bytes", 1),
        "same-process adoption result",
    )
    require(
        windows["copied_bytes"] == TOKEN_EMBEDDING_BYTES
        and windows["copied_chunks"] == 480
        and windows["copied_windows"] == 54
        and windows["total_windows"] == 108,
        "adoption window accounting",
    )
    require(
        execution.get("request_index") == 50
        and execution.get("input_tokens") == 271
        and execution.get("output_tokens") == 41
        and execution.get("tokens_sha256") == EXPECTED_GEMMA_TOKENS
        and execution.get("process_memory", {}).get("swap_bytes") == 0
        and command_value(execution.get("server_command"), "--n-gpu-layers")
            == "1"
        and command_value(execution.get("server_command"), "--device")
            == "CUDA0",
        "adopted Gemma execution",
    )
    require(
        set(process.get("processes", {})) == {"gemma", "qwen"}
        and process.get("interval_ms") == 50
        and all(
            row.get("observed_samples", 0) > 0
            and row.get("swap_bytes") == 0
            for row in process["processes"].values()
        ),
        "zero process swap",
    )
    memory = cgroup.get("memory", {})
    events = memory.get("events", {})
    require(
        memory.get("swap_max_bytes") == 0
        and memory.get("swap_current_bytes") == 0
        and all(
            events.get(name) == 0
            for name in ("oom", "oom_kill", "oom_group_kill")
        ),
        "cgroup memory safety",
    )

    gpu = gpu_utilization(
        resource_path, qwen["paid_start_ns"], qwen["paid_end_ns"]
    )
    server_j = qwen["server_energy"]["server_compute_device_energy_j"]
    phone_j = phone["whole_phone_energy_j"]
    output: dict[str, Any] = {
        "admission": (
            "SAME_PROCESS_WEIGHT_ADOPTION_QUALIFIED_"
            "ENERGY_ABBA_PENDING"
        ),
        "artifacts": {
            "bridge_log_sha256": digest(bridge_path),
            "cgroup_memory_sha256": digest(cgroup_path),
            "gemma_execution_sha256": digest(execution_path),
            "gemma_log_sha256": digest(gemma_log_path),
            "phone_energy_sha256": digest(phone_path),
            "process_memory_sha256": digest(process_path),
            "qwen_result_sha256": digest(qwen_path),
            "resource_samples_sha256": digest(resource_path),
        },
        "claim": {
            "gpu_continuously_busy": False,
            "incremental_dynamic_energy_savings_pct": None,
            "next_required_gate": (
                "MATCHED_STATIC_VS_DYNAMIC_FULL_TRACE_ABBA_WITH_"
                "LOAD_TRANSITION_AND_RESTORE_INSIDE_BOUNDARY"
            ),
            "same_process_weight_adoption_qualified": True,
        },
        "gates": {
            "adopted_gemma_exact_output": True,
            "bridge_fence_complete": True,
            "cgroup_no_swap_enforced": True,
            "cgroup_no_oom": True,
            "exact_qwen_output": True,
            "gpu_reserve_preserved": True,
            "phone_energy_synchronized": True,
            "process_swap_zero": True,
            "same_process_destination_adoptable": True,
            "tensor_readback_verified": True,
        },
        "gemma_execution": {
            "input_tokens": execution["input_tokens"],
            "output_tokens": execution["output_tokens"],
            "prompt_ms": execution["prompt_ms"],
            "request_index": execution["request_index"],
            "service_s": execution["service_s"],
            "tokens_sha256": execution["tokens_sha256"],
        },
        "gpu_utilization_during_qwen": gpu,
        "memory": {
            "cgroup_peak_bytes": memory["peak_bytes"],
            "gemma_peak_rss_bytes": process["processes"]["gemma"][
                "rss_bytes"
            ],
            "qwen_peak_rss_bytes": process["processes"]["qwen"][
                "rss_bytes"
            ],
            "swap_limit_bytes": memory["swap_max_bytes"],
        },
        "paid_qwen_diagnostic": {
            "cpu_package_energy_j": qwen["server_energy"][
                "cpu_package_energy_j"
            ],
            "duration_s": qwen["metrics"]["duration_s"],
            "fleet_energy_j": server_j + phone_j,
            "gpu_board_energy_j": qwen["server_energy"][
                "gpu_board_energy_j"
            ],
            "phone_energy_j": phone_j,
            "valid_incremental_energy_comparison": False,
        },
        "schema": SCHEMA,
        "status": "PASS",
        "weight_adoption": {
            "bytes": TOKEN_EMBEDDING_BYTES,
            "chunk_bytes": adoption["chunk_bytes"],
            "chunks_per_window": adoption["chunks_per_window"],
            "copied_chunks": adoption["copied_chunks"],
            "copy_active_ms": windows["copy_active_ms"],
            "copy_p90_ms": bridge["prefetch_copy_p90_ms"],
            "copy_windows": adoption["copy_windows"],
            "gpu_free_min_bytes": adoption["gpu_free_min_bytes"],
            "gpu_reserve_margin_bytes": (
                adoption["gpu_free_min_bytes"]
                - adoption["gpu_reserve_bytes"]
            ),
            "protected_window_overrun_max_ms": bridge[
                "prefetch_copied_window_overrun_max_ms"
            ],
            "protected_window_overrun_p90_ms": bridge[
                "prefetch_copied_window_overrun_p90_ms"
            ],
            "source_offset": TOKEN_EMBEDDING_OFFSET,
            "tensor": "token_embd.weight",
        },
    }
    require(all(output["gates"].values()), "adoption qualification gates")
    output["record_sha256"] = hashlib.sha256(canonical(output)).hexdigest()
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root", type=Path, default=DEFAULT_RESULT_ROOT
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.exists():
        parser.error("output must be an unused absolute path")
    try:
        output = build(args.result_root)
        args.output.write_bytes(canonical(output))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"adoption qualification failed: {exc}\n")
    print(json.dumps({
        "admission": output["admission"],
        "output": str(args.output),
        "status": output["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
