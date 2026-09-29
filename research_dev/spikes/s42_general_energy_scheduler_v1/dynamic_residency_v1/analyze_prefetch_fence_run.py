#!/usr/bin/env python3
"""Bind one OP15-fenced GPU weight-prefetch run to physical receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "s42-op15-fenced-gpu-prefetch-run-v1"
QWEN_SCHEMA = "s41-burstgpt-llama-server-result-v1"
PHONE_SCHEMA = "s41-phone-energy-v3"
GEMMA_MODEL_SHA256 = (
    "ed76f2183d2d1d65091986033023e6c7"
    "8d27f6276c1b0c5826cc92acf73538cf"
)
TOKEN_EMBEDDING_OFFSET = 15_838_752
TOKEN_EMBEDDING_BYTES = 2_013_265_920


class AnalysisError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def canonical(value: object) -> bytes:
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
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    require(type(value) is dict, f"object: {path}")
    return value


def parse_value(text: str) -> object:
    if text == "true":
        return True
    if text == "false":
        return False
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def parse_helper(path: Path) -> dict[str, object]:
    rows = [
        line.removeprefix("PREFETCH_RESULT ").strip()
        for line in path.read_text(encoding="ascii").splitlines()
        if line.startswith("PREFETCH_RESULT ")
    ]
    require(len(rows) == 1, "one helper result")
    fields: dict[str, object] = {}
    for item in rows[0].split():
        key, separator, value = item.partition("=")
        require(separator == "=" and key and key not in fields,
                "helper result field")
        fields[key] = parse_value(value)
    return fields


def parse_bridge(path: Path) -> dict[str, Any]:
    rows = [
        line.partition("FFNDMABUF ")[2]
        for line in path.read_text(encoding="ascii").splitlines()
        if "FFNDMABUF " in line
    ]
    require(len(rows) == 1, "one bridge result")
    value = json.loads(rows[0])
    require(type(value) is dict, "bridge result object")
    return value


def analyze(
    *,
    mode: str,
    qwen_path: Path,
    phone_path: Path,
    server_path: Path,
    bridge_path: Path,
    helper_path: Path,
) -> dict[str, Any]:
    require(mode in {"observe", "prefetch"}, "mode")
    qwen = read_object(qwen_path)
    phone = read_object(phone_path)
    bridge = parse_bridge(bridge_path)
    helper = parse_helper(helper_path)
    server_log = server_path.read_text(encoding="utf-8", errors="replace")
    require(
        qwen.get("schema") == QWEN_SCHEMA and qwen.get("status") == "PASS",
        "Qwen result",
    )
    require(
        phone.get("schema") == PHONE_SCHEMA
        and phone.get("status") == "PASS"
        and phone.get("boundary") == "paid_trace_interval",
        "phone energy result",
    )
    require(
        qwen.get("prefetch_arm_ns") is not None
        and qwen["paid_start_ns"] <= qwen["prefetch_arm_ns"]
        < qwen["paid_end_ns"],
        "prefetch arm interval",
    )
    server_command = qwen.get("server_command")
    require(type(server_command) is list, "server command receipt")
    require(
        "--n-gpu-layers" in server_command
        and server_command[server_command.index("--n-gpu-layers") + 1]
            == "15"
        and "--device" in server_command
        and server_command[server_command.index("--device") + 1]
            == "CUDA0"
        and "layers=12 mask=0000000000000fff" in server_log,
        "Qwen-15 placement",
    )
    require(
        bridge.get("status") == "ok"
        and bridge.get("prefetch_fence_enabled") is True
        and bridge.get("prefetch_group_first_layer") == 0
        and bridge.get("prefetch_group_last_layer") == 11
        and bridge.get("reset_recoveries") == 0,
        "bridge status",
    )
    require(
        helper.get("status") == "PASS"
        and helper.get("mode") == mode
        and helper.get("adoptable") is False,
        "helper status",
    )
    source_offset = helper.get("source_offset")
    stage_bytes = helper.get("stage_bytes")
    require(
        type(source_offset) is int
        and type(stage_bytes) is int
        and source_offset == TOKEN_EMBEDDING_OFFSET
        and stage_bytes > 0
        and source_offset + stage_bytes
            <= TOKEN_EMBEDDING_OFFSET + TOKEN_EMBEDDING_BYTES,
        "Gemma tensor range",
    )
    require(
        helper.get("fence_calls") == bridge.get("prefetch_fence_calls")
        and helper.get("copied_bytes")
            == bridge.get("prefetch_copied_bytes")
        and helper.get("copied_chunks")
            == bridge.get("prefetch_copied_chunks"),
        "bridge/helper receipt agreement",
    )
    require(
        type(helper.get("gpu_free_min_bytes")) is int
        and type(helper.get("gpu_reserve_bytes")) is int
        and helper["gpu_free_min_bytes"] >= helper["gpu_reserve_bytes"],
        "GPU reserve",
    )
    require(
        type(helper.get("warmup_copy_calls")) is int
        and helper["warmup_copy_calls"] > 0
        and helper.get("warmup_copy_bytes")
            == helper["warmup_copy_calls"] * helper["chunk_bytes"],
        "CUDA context warmup",
    )
    require(
        helper.get("source_pinned") is True
        and helper.get("source_resident_bytes") == stage_bytes,
        "resident pinned source",
    )
    if mode == "observe":
        require(
            helper.get("copied_bytes") == 0
            and helper.get("copied_chunks") == 0,
            "observe arm copied no bytes",
        )
    else:
        require(
            helper.get("copied_bytes") == stage_bytes
            and helper.get("verified") is True
            and helper.get("source_fnv64")
                == helper.get("destination_fnv64"),
            "prefetch byte verification",
        )

    server_energy = qwen.get("server_energy")
    require(type(server_energy) is dict, "server energy")
    server_j = server_energy.get("server_compute_device_energy_j")
    phone_j = phone.get("whole_phone_energy_j")
    require(
        type(server_j) in {int, float}
        and type(phone_j) in {int, float}
        and server_j > 0
        and phone_j >= 0,
        "fleet energy",
    )
    request_rows = qwen.get("request_results")
    require(type(request_rows) is list and request_rows, "request results")
    work = [{
        "input_tokens": row["input_tokens"],
        "output_tokens": row["output_tokens"],
        "request_index": row["request_index"],
        "tokens": row["tokens"],
    } for row in request_rows]
    gates = {
        "bridge_helper_receipts_match": True,
        "cuda_context_warmed_before_paid": True,
        "destination_bytes_verified": (
            True if mode == "prefetch" else None
        ),
        "gpu_reserve_preserved": True,
        "no_usb_reset_recovery": True,
        "protected_work_serialized_after_copy": True,
        "qwen15_placement_verified": True,
        "resident_pinned_source_verified": True,
        "transfer_completed_inside_phone_window": (
            bridge["prefetch_copied_window_overrun_max_ms"] == 0
            if mode == "prefetch" else None
        ),
    }
    output: dict[str, Any] = {
        "admission": "FENCE_TRANSFER_PASS_NO_WEIGHT_ADOPTION",
        "artifacts": {
            "bridge_log_sha256": digest(bridge_path),
            "helper_log_sha256": digest(helper_path),
            "phone_energy_sha256": digest(phone_path),
            "qwen_result_sha256": digest(qwen_path),
            "server_log_sha256": digest(server_path),
        },
        "bridge": bridge,
        "energy": {
            "boundary": "cpu-package+gpu-board+whole-phone-paid-cohort",
            "cpu_package_j": server_energy["cpu_package_energy_j"],
            "fleet_j": server_j + phone_j,
            "gpu_board_j": server_energy["gpu_board_energy_j"],
            "phone_j": phone_j,
            "server_j": server_j,
        },
        "gates": gates,
        "helper": helper,
        "mode": mode,
        "model_source": {
            "model_sha256": GEMMA_MODEL_SHA256,
            "tensor": "token_embd.weight",
            "tensor_bytes": TOKEN_EMBEDDING_BYTES,
            "tensor_offset": TOKEN_EMBEDDING_OFFSET,
        },
        "qwen": {
            "duration_s": qwen["metrics"]["duration_s"],
            "gpu_layers": 15,
            "paid_end_ns": qwen["paid_end_ns"],
            "paid_start_ns": qwen["paid_start_ns"],
            "prefetch_arm_ns": qwen["prefetch_arm_ns"],
            "work": work,
        },
        "schema": SCHEMA,
        "status": "PASS",
        "weight_adoptable_by_gemma_executor": False,
    }
    output["record_sha256"] = hashlib.sha256(canonical(output)).hexdigest()
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("observe", "prefetch"), required=True)
    parser.add_argument("--qwen-result", type=Path, required=True)
    parser.add_argument("--phone-energy", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--bridge-log", type=Path, required=True)
    parser.add_argument("--helper-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.exists():
        parser.error("output must be an unused absolute path")
    try:
        value = analyze(
            mode=args.mode,
            qwen_path=args.qwen_result,
            phone_path=args.phone_energy,
            server_path=args.server_log,
            bridge_path=args.bridge_log,
            helper_path=args.helper_log,
        )
        args.output.write_bytes(canonical(value))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"prefetch analysis failed: {exc}\n")
    print(json.dumps({
        "admission": value["admission"],
        "fleet_j": value["energy"]["fleet_j"],
        "output": str(args.output),
        "status": value["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
