#!/usr/bin/env python3
"""Bind one Qwen plus Gemma GPU-wavefront arm to physical receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


SCHEMA = "s42-fp16-burstgpt-gpu-wavefront-run-v1"


class AnalysisError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read {path}") from exc
    require(type(value) is dict, f"invalid object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def bridge_result(path: Path) -> dict[str, Any]:
    rows = [
        line.partition(marker)[2]
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        for marker in ("FFNDMABUF ", "PHONEARBITER ")
        if marker in line
    ]
    require(len(rows) == 1, "bridge result is not unique")
    value = json.loads(rows[0])
    require(type(value) is dict, "bridge result is invalid")
    return value


def router_result(path: Path) -> dict[str, Any]:
    rows = [
        line.partition("RESIDENTROUTER ")[2]
        for line in path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        if "RESIDENTROUTER " in line
    ]
    require(len(rows) == 1, "router result is not unique")
    value = json.loads(rows[0])
    require(type(value) is dict, "router result is invalid")
    return value


def ffn_split_result(path: Path) -> dict[str, Any]:
    rows = [
        line.partition("FFNSPLIT ")[2]
        for line in path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        if "FFNSPLIT " in line
    ]
    require(len(rows) == 1, "Gemma FFN split result is not unique")
    value = json.loads(rows[0])
    require(type(value) is dict, "Gemma FFN split result is invalid")
    return value


def expected_ffn_shapes(
    rows: list[dict[str, Any]],
    ubatch: int,
    prefill_enabled: bool,
    layer_count: int = 23,
) -> dict[int, int]:
    shapes: dict[int, int] = {}
    for row in rows:
        if prefill_enabled:
            remaining = row["input_tokens"]
            while remaining > 0:
                tokens = min(remaining, ubatch)
                shapes[tokens] = shapes.get(tokens, 0) + layer_count
                remaining -= tokens
        decode_calls = (row["output_tokens"] - 1) * layer_count
        shapes[1] = shapes.get(1, 0) + decode_calls
    return shapes


def placement_certificates(path: Path) -> list[dict[str, Any]]:
    rows = [
        line.partition("PLACEMENTCERT ")[2]
        for line in path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        if "PLACEMENTCERT " in line
    ]
    require(bool(rows), "Gemma placement certificate is absent")
    values = [json.loads(row) for row in rows]
    require(
        all(type(value) is dict for value in values),
        "Gemma placement certificate is invalid",
    )
    return values


def positive(value: object, name: str) -> float:
    require(
        type(value) in {int, float}
        and math.isfinite(value)
        and value > 0,
        f"invalid {name}",
    )
    return float(value)


def cpu_list(text: str) -> set[int]:
    values: set[int] = set()
    for item in text.split(","):
        fields = item.split("-", 1)
        try:
            first = int(fields[0])
            last = int(fields[-1])
        except (IndexError, ValueError) as exc:
            raise AnalysisError("invalid CPU list") from exc
        require(first >= 0 and last >= first, "invalid CPU list bounds")
        values.update(range(first, last + 1))
    require(bool(values), "empty CPU list")
    return values


def require_affinity(value: object, expected: set[int]) -> None:
    require(type(value) is dict, "Gemma affinity receipt")
    require(
        value.get("requested_cpus") == sorted(expected),
        "Gemma requested CPU affinity",
    )
    for phase in ("at_ready", "at_paid_start"):
        record = value.get(phase)
        require(type(record) is dict, f"Gemma affinity {phase}")
        sets = record.get("cpu_sets")
        require(
            type(sets) is dict
            and bool(sets)
            and all(type(count) is int and count > 0 for count in sets.values())
            and record.get("thread_count") == sum(sets.values())
            and all(cpu_list(text) <= expected for text in sets),
            f"Gemma thread affinity {phase}",
        )


def phone_residency_receipt(
    session_log: Path,
    workers_log: Path,
    expected_vmem_mib: int,
    minimum_available_kib: int,
    expected_layout: str | None = None,
) -> dict[str, object]:
    session_rows = [
        line
        for line in session_log.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        if line.startswith(
            "[resident-session] HTP0+HTP1+HTP2 WARM mem_available_kib="
        )
    ]
    require(len(session_rows) == 1, "phone memory receipt is not unique")
    try:
        available_kib = int(session_rows[0].rsplit("=", 1)[1])
    except (IndexError, ValueError) as exc:
        raise AnalysisError("phone memory receipt is invalid") from exc
    workers_text = workers_log.read_text(
        encoding="utf-8", errors="replace"
    )
    vmem_bytes = [
        int(value)
        for value in re.findall(
            r"op batching:.* vmem ([0-9]+)$",
            workers_text,
            flags=re.MULTILINE,
        )
    ]
    require(
        available_kib >= minimum_available_kib
        and len(vmem_bytes) == 3
        and set(vmem_bytes) == {expected_vmem_mib * 1024 * 1024},
        "phone resident memory geometry",
    )
    result: dict[str, object] = {
        "available_kib": available_kib,
        "minimum_available_kib": minimum_available_kib,
        "session_count": len(vmem_bytes),
        "vmem_mib_per_session": expected_vmem_mib,
    }
    if expected_layout is not None:
        marker = "RESIDENTWORKERS "
        records = [
            json.loads(line[len(marker):])
            for line in workers_text.splitlines()
            if line.startswith(marker + "{")
        ]
        require(
            len(records) == 1
            and type(records[0]) is dict
            and records[0].get("status") == "WARM"
            and records[0].get("layout") == expected_layout,
            "phone resident layout receipt",
        )
        result["workers"] = records[0]
    return result


def analyze(args: argparse.Namespace) -> dict[str, object]:
    qwen = load(args.qwen_result)
    phone = load(args.phone_energy)
    gate = load(args.gate_result)
    driver = load(args.driver_result)
    profile = load(args.profile)
    bridge = bridge_result(args.bridge_log)
    router = router_result(args.router_log)
    phone_arbiter = bridge.get("status") == "MECHANICS_ONLY"
    ffn_split = ffn_split_result(args.driver_log) if phone_arbiter else None
    phone_residency = phone_residency_receipt(
        args.phone_session_log,
        args.phone_workers_log,
        args.expected_phone_vmem_mib,
        args.expected_phone_min_available_kib,
    )
    placements = placement_certificates(args.driver_log)
    server_log = args.server_log.read_text(encoding="utf-8", errors="replace")

    require(
        qwen.get("schema") == "s41-burstgpt-llama-server-result-v1"
        and qwen.get("status") == "PASS"
        and qwen.get("arm") == "op15",
        "Qwen result identity",
    )
    paid_start_ns = qwen.get("paid_start_ns")
    qwen_end_ns = qwen.get("qwen_end_ns")
    paid_tail_ns = qwen.get("paid_tail_end_ns")
    paid_end_ns = qwen.get("paid_end_ns")
    require(
        all(type(value) is int for value in (
            paid_start_ns, qwen_end_ns, paid_tail_ns, paid_end_ns
        ))
        and paid_start_ns < qwen_end_ns <= paid_end_ns
        and paid_start_ns < paid_tail_ns <= paid_end_ns
        and paid_end_ns == max(qwen_end_ns, paid_tail_ns),
        "paid tail interval",
    )
    duration_s = positive(qwen.get("metrics", {}).get("duration_s"), "duration")
    require(
        math.isclose(duration_s, (paid_end_ns - paid_start_ns) / 1e9),
        "paid duration mismatch",
    )
    qwen_rows = qwen.get("request_results")
    require(type(qwen_rows) is list and bool(qwen_rows), "Qwen work is absent")
    for row in qwen_rows:
        require(
            type(row) is dict
            and type(row.get("tokens")) is list
            and len(row["tokens"]) == row.get("output_tokens"),
            "Qwen output is incomplete",
        )
    command = qwen.get("server_command")
    require(
        type(command) is list
        and "--n-gpu-layers" in command
        and command[command.index("--n-gpu-layers") + 1]
        == str(args.expected_qwen_gpu_layers)
        and "--device" in command
        and command[command.index("--device") + 1] == "CUDA0"
        and "layers=12 mask=0000000000000fff" in server_log,
        "Qwen physical placement",
    )

    require(
        phone.get("schema") == "s41-phone-energy-v3"
        and phone.get("status") == "PASS"
        and phone.get("boundary") == "paid_trace_interval"
        and math.isclose(
            positive(phone.get("duration_s"), "phone duration"), duration_s
        ),
        "phone energy boundary",
    )
    server_energy = qwen.get("server_energy")
    require(
        type(server_energy) is dict
        and server_energy.get("boundary") == "paid_trace_interval",
        "server energy boundary",
    )
    cpu_j = positive(server_energy.get("cpu_package_energy_j"), "CPU energy")
    gpu_j = positive(server_energy.get("gpu_board_energy_j"), "GPU energy")
    server_j = positive(
        server_energy.get("server_compute_device_energy_j"), "server energy"
    )
    phone_j = positive(phone.get("whole_phone_energy_j"), "phone energy")
    require(math.isclose(server_j, cpu_j + gpu_j), "server energy sum")

    require(
        gate.get("schema") == "s42-fp16-burstgpt-gpu-wavefront-gate-v1"
        and gate.get("status") == "PASS"
        and gate.get("mode") == args.mode
        and gate.get("qwen_complete_ns") == qwen_end_ns
        and gate.get("paid_tail_ns") == paid_tail_ns
        and gate.get("energy_claim_eligible") is False
        and gate.get("admission_disabled_reason") is None,
        "gate result identity",
    )
    backfills = gate.get("backfills")
    require(type(backfills) is int and backfills >= 0, "gate backfill count")
    phone_prefill = (
        phone_arbiter and args.expected_phone_ffn_prefill_columns != 0
    )
    if args.mode == "control":
        require(backfills == 0, "control executed a wavefront")
    elif not phone_prefill:
        require(backfills > 0, "wavefront arm executed no GPU filler")

    require(
        driver.get("schema") == "s42-gemma-wavefront-driver-v1"
        and driver.get("status") == "PASS"
        and driver.get("arm_ns") == qwen.get("prefetch_arm_ns")
        and driver.get("arm_ns") >= paid_start_ns,
        "Gemma driver identity",
    )
    gemma_rows = driver.get("request_results")
    require(type(gemma_rows) is list and bool(gemma_rows), "Gemma work is absent")
    for row in gemma_rows:
        require(
            type(row) is dict
            and type(row.get("token_ids")) is list
            and len(row["token_ids"]) == row.get("output_tokens")
            and type(row.get("input_tokens")) is int
            and row["input_tokens"] > 0
            and paid_start_ns <= row.get("dispatch_ns") < row.get("completion_ns") <= paid_tail_ns,
            "Gemma output or interval is incomplete",
        )
    driver_command = driver.get("driver_command")
    require(
        type(driver_command) is list
        and "-ngl" in driver_command
        and driver_command[driver_command.index("-ngl") + 1] == "0"
        and "-t" in driver_command
        and driver_command[driver_command.index("-t") + 1]
        == str(args.expected_gemma_threads)
        and "-tb" in driver_command
        and driver_command[driver_command.index("-tb") + 1]
        == str(args.expected_gemma_threads)
        and "--driver-ubatch" in driver_command
        and driver_command[driver_command.index("--driver-ubatch") + 1]
        == str(args.expected_gemma_ubatch)
        and driver.get("ubatch") == args.expected_gemma_ubatch
        and "--lm-head-rows" in driver_command,
        "Gemma CPU plus resident LM-head route",
    )
    if phone_arbiter:
        ffn_route = driver.get("ffn_route")
        require(
            type(ffn_route) is dict
            and ffn_route.get("host") == "127.0.0.1"
            and ffn_route.get("layers") == "0-22"
            and ffn_route.get("columns") == 6144
            and ffn_route.get("prefill_columns")
            == args.expected_phone_ffn_prefill_columns
            and ffn_route.get("timeout_ms")
            == args.expected_phone_ffn_timeout_ms
            and ffn_route.get("decode_columns") == 6144
            and ffn_route.get("f16_io") is True
            and "--ffn-layers" in driver_command
            and "--ffn-columns" in driver_command,
            "Gemma resident phone FFN route",
        )
        expected_shapes = expected_ffn_shapes(
            gemma_rows,
            args.expected_gemma_ubatch,
            args.expected_phone_ffn_prefill_columns != 0,
        )
        expected_filler_calls = sum(expected_shapes.values())
        observed_shapes: dict[int, int] = {}
        require(
            ffn_split.get("schema") == "layersplit-ffn-overlap-v2"
            and ffn_split.get("status") == "FFN_OVERLAP_OK"
            and ffn_split.get("layer_count") == 23
            and ffn_split.get("layer_mask") == 0x7FFFFF
            and ffn_split.get("max_columns") == 6144
            and ffn_split.get("calls") == expected_filler_calls
            and ffn_split.get("decode_calls") == expected_shapes.get(1, 0)
            and ffn_split.get("prefill_calls")
            == expected_filler_calls - expected_shapes.get(1, 0)
            and type(ffn_split.get("shapes")) is list,
            "Gemma FFN split summary",
        )
        for shape in ffn_split["shapes"]:
            require(
                type(shape) is dict
                and type(shape.get("tokens")) is int
                and shape["tokens"] > 0
                and shape.get("columns") == 6144
                and type(shape.get("calls")) is int
                and shape["calls"] > 0
                and shape["tokens"] not in observed_shapes,
                "Gemma FFN split shape receipt",
            )
            observed_shapes[shape["tokens"]] = shape["calls"]
        require(observed_shapes == expected_shapes, "Gemma FFN split shape coverage")
    else:
        require(driver.get("ffn_route") is None, "unexpected Gemma phone route")
    require(
        driver.get("producer_environment")
        == {"CUDA_VISIBLE_DEVICES": ""},
        "Gemma producer CUDA isolation",
    )
    expected_gemma_cpus = cpu_list(args.expected_gemma_cpus)
    require(
        args.expected_gemma_threads == len(expected_gemma_cpus),
        "Gemma thread and CPU geometry",
    )
    require_affinity(driver.get("producer_affinity"), expected_gemma_cpus)
    for placement in placements:
        compute_buffers = placement.get("compute_by_buffer_type")
        copy_buffers = placement.get("copy_by_buffer_type")
        require(
            placement.get("schema") == "layersplit-scheduled-placement-v2"
            and placement.get("role") == "overlapdriver"
            and placement.get("status") == "SCHEDULED_PLACEMENT_OK"
            and type(placement.get("compute_nodes")) is int
            and placement["compute_nodes"] > 0
            and placement.get("missing_buffer_compute_nodes") == 0
            and type(compute_buffers) is dict
            and bool(compute_buffers)
            and type(copy_buffers) is dict
            and not any(key.startswith("CUDA") for key in compute_buffers)
            and not any(key.startswith("CUDA") for key in copy_buffers),
            "Gemma producer is not CPU-only",
        )

    require(
        profile.get("schema") == "s42-fp16-burstgpt-gpu-wavefront-profile-v1"
        and gate.get("profile_sha256") == canonical_sha256(profile),
        "gate profile binding",
    )
    memory = profile.get("memory")
    candidate = profile.get("candidate")
    require(type(memory) is dict and type(candidate) is dict, "profile sections")
    require(
        memory["free_bytes"] - memory["reserve_bytes"]
        >= candidate["workspace_bytes"],
        "GPU reserve",
    )

    require(
        bridge.get("status") in {"ok", "MECHANICS_ONLY"}
        and bridge.get("prefetch_fence_enabled") is True
        and bridge.get("prefetch_group_first_layer") == 0
        and bridge.get("prefetch_group_last_layer") == 11
        and bridge.get("prefetch_fence_calls") == gate.get("fence_calls")
        and bridge.get("reset_recoveries") == 0
        and type(bridge.get("prefetch_decode_window_min_ms")) in {int, float}
        and bridge["prefetch_decode_window_min_ms"] > 0,
        "protected OP15 fence receipts",
    )
    if phone_arbiter:
        protected_first = bridge.get("prefetch_group_first_layer")
        protected_last = bridge.get("prefetch_group_last_layer")
        protected_groups = bridge.get("protected_group_ends")
        filler_before_done = bridge.get("filler_before_protected_done")
        filler_after_done = bridge.get("filler_after_protected_done")
        expected_prefill_calls = sum(expected_shapes.values()) - (
            sum((row["output_tokens"] - 1) * 23 for row in gemma_rows)
        )
        prefill_before_done = min(filler_before_done, expected_prefill_calls)
        require(
            type(bridge.get("filler_calls")) is int
            and bridge["filler_calls"] == expected_filler_calls
            and bridge.get("filler_calls") == bridge.get("filler_admitted")
            and type(filler_before_done) is int
            and filler_before_done >= 0
            and type(filler_after_done) is int
            and filler_after_done >= 0
            and filler_before_done + filler_after_done
            == expected_filler_calls
            and type(protected_groups) is int
            and protected_groups > 0
            and bridge.get("protected_group_starts") == protected_groups
            and bridge.get("protected_calls")
            == protected_groups * (protected_last - protected_first + 1)
            and bridge.get("idle_samples") == protected_groups - 1
            and bridge.get("protected_done_observed") is True
            and bridge.get("observed_idle_min_us")
            >= bridge.get("idle_lower_us")
            and bridge.get("filler_sandwich_max_us")
            <= bridge.get("filler_upper_us")
            and bridge.get("filler_upper_us") + bridge.get("guard_us")
            <= bridge.get("idle_lower_us")
            and bridge.get("filler_upper_violations") == 0
            and bridge.get("guard_violations") == 0
            and bridge.get("idle_lower_violations") == 0
            and bridge.get("protected_pending_after_filler") == 0
            and bridge.get("energy_claim_eligible") is False,
            "phone arbiter mechanics receipts",
        )
        if expected_prefill_calls > 0:
            require(
                prefill_before_done > 0,
                "Gemma prefill did not overlap protected Qwen execution",
            )
        require(
            router.get("status") == "ok"
            and router.get("terminate_requested") is True
            and type(router.get("sessions")) is int
            and router["sessions"] >= 2
            and router.get("requests")
            == bridge["protected_calls"] + bridge["filler_calls"],
            "phone arbiter router receipt",
        )
    else:
        require(
            router.get("status") == "ok"
            and router.get("requests") == bridge.get("calls"),
            "protected phone router receipt",
        )
    if args.mode != "control":
        require(
            bridge.get("prefetch_overrun_max_ms") == 0,
            "GPU filler overran protected Qwen work",
        )

    admission = (
        "PHONE_PREFILL_MECHANICS_ONLY"
        if phone_prefill
        else "PHONE_ARBITER_MECHANICS_ONLY"
    ) if phone_arbiter else {
        "control": "CONTROL_CALIBRATION_ONLY",
        "mechanics": "UNQUALIFIED_MECHANICS_PASS",
        "qualified": "MATCHED_ENERGY_QUALIFICATION_REQUIRED",
    }[args.mode]
    output: dict[str, object] = {
        "admission": admission,
        "artifacts": {
            "bridge_log_sha256": sha256(args.bridge_log),
            "driver_log_sha256": sha256(args.driver_log),
            "driver_result_sha256": sha256(args.driver_result),
            "gate_result_sha256": sha256(args.gate_result),
            "phone_energy_sha256": sha256(args.phone_energy),
            "phone_session_log_sha256": sha256(args.phone_session_log),
            "phone_workers_log_sha256": sha256(args.phone_workers_log),
            "profile_sha256": sha256(args.profile),
            "qwen_result_sha256": sha256(args.qwen_result),
            "router_log_sha256": sha256(args.router_log),
            "server_log_sha256": sha256(args.server_log),
        },
        "bridge": bridge,
        "energy": {
            "boundary": "cpu-package+gpu-board+whole-phone-paid-work",
            "cpu_package_j": cpu_j,
            "fleet_j": server_j + phone_j,
            "gpu_board_j": gpu_j,
            "phone_j": phone_j,
            "server_j": server_j,
        },
        "energy_claim_eligible": False,
        "gemma": {
            "cpu_list": sorted(expected_gemma_cpus),
            "ffn_split": ffn_split,
            "indices": driver.get("indices"),
            "requests": len(gemma_rows),
            "threads": args.expected_gemma_threads,
            "ubatch": args.expected_gemma_ubatch,
        },
        "gpu_wavefront": {
            "backfills": backfills,
            "overlap_observed": backfills > 0,
            "fence_calls": gate.get("fence_calls"),
            "rejections": gate.get("rejections"),
            "tail_calls": gate.get("tail_calls"),
        },
        "mode": args.mode,
        "paid_duration_s": duration_s,
        "profile_admission": profile.get("admission"),
        "phone_arbiter": phone_arbiter,
        "phone_residency": phone_residency,
        "phone_overlap": ({
            "filler_after_protected_done": filler_after_done,
            "filler_before_protected_done": filler_before_done,
            "prefill_before_protected_done": prefill_before_done,
            "prefill_calls": expected_prefill_calls,
        } if phone_arbiter else None),
        "router": router,
        "qwen": {
            "gpu_layers": args.expected_qwen_gpu_layers,
            "indices": qwen.get("indices"),
            "requests": len(qwen_rows),
        },
        "schema": SCHEMA,
        "status": "PASS",
    }
    output["record_sha256"] = hashlib.sha256(
        (json.dumps(output, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")
    ).hexdigest()
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("control", "mechanics", "qualified"), required=True)
    parser.add_argument("--qwen-result", type=Path, required=True)
    parser.add_argument("--phone-energy", type=Path, required=True)
    parser.add_argument("--gate-result", type=Path, required=True)
    parser.add_argument("--driver-result", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--bridge-log", type=Path, required=True)
    parser.add_argument("--router-log", type=Path, required=True)
    parser.add_argument("--phone-session-log", type=Path, required=True)
    parser.add_argument("--phone-workers-log", type=Path, required=True)
    parser.add_argument("--driver-log", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--expected-qwen-gpu-layers", type=int, required=True)
    parser.add_argument("--expected-gemma-cpus", required=True)
    parser.add_argument("--expected-gemma-threads", type=int, required=True)
    parser.add_argument("--expected-gemma-ubatch", type=int, required=True)
    parser.add_argument(
        "--expected-phone-ffn-prefill-columns", type=int, required=True
    )
    parser.add_argument(
        "--expected-phone-ffn-timeout-ms", type=int, required=True
    )
    parser.add_argument("--expected-phone-vmem-mib", type=int, required=True)
    parser.add_argument(
        "--expected-phone-min-available-kib", type=int, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.expected_qwen_gpu_layers <= 41:
        parser.error("invalid expected Qwen GPU layer count")
    if args.expected_gemma_threads <= 0:
        parser.error("invalid expected Gemma thread count")
    if not 1 <= args.expected_gemma_ubatch <= 512:
        parser.error("invalid expected Gemma ubatch")
    if args.expected_phone_ffn_prefill_columns not in {0, 6144}:
        parser.error("invalid expected phone prefill width")
    if not 1 <= args.expected_phone_ffn_timeout_ms <= 600000:
        parser.error("invalid expected phone FFN timeout")
    if not 3200 <= args.expected_phone_vmem_mib <= 3328:
        parser.error("invalid expected phone VMEM")
    if args.expected_phone_min_available_kib < 2097152:
        parser.error("invalid expected phone memory reserve")
    if not args.output.is_absolute() or args.output.exists() or not args.output.parent.is_dir():
        parser.error("output must be an unused absolute path")
    try:
        result = analyze(args)
    except (AnalysisError, OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(2, f"wavefront analysis failed: {exc}\n")
    args.output.write_text(
        json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps({
        "admission": result["admission"],
        "backfills": result["gpu_wavefront"]["backfills"],
        "fleet_j": result["energy"]["fleet_j"],
        "status": result["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
