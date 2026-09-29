#!/usr/bin/env python3
"""Validate one physical Qwen plus Gemma GPU-prefix wavefront arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from analyze_wavefront_run import (
    AnalysisError,
    bridge_result,
    canonical_sha256,
    cpu_list,
    expected_ffn_shapes,
    ffn_split_result,
    load,
    phone_residency_receipt,
    placement_certificates,
    positive,
    require,
    router_result,
    sha256,
)
from materialize_stage_wavefront_profile import (
    MaterializeError,
    parse_memory_certificate,
    validate_manifest,
)


SCHEMA = "s42-fp16-burstgpt-gpu-prefix-wavefront-run-v1"
MAX_COMPLETION_BOUNDARY_SKEW_NS = 100_000_000
MAX_AFFINITY_SWITCH_NS = 100_000_000
MAX_CONTROL_PIPELINE_OVERHEAD_US = 500_000
PHONE_LAYOUTS = {
    "gemma23-qwen12-full-v1": {
        "gemma_executed_layers": "1-22",
        "gemma_executed_layer_count": 22,
        "gemma_layer_count": 23,
        "gemma_layer_mask": 0x7FFFFF,
        "gemma_layers": "0-22",
        "qwen_layer_count": 12,
        "qwen_layer_mask": 0x0FFF,
        "qwen_layers": "0-11",
        "qwen_last_layer": 11,
    },
    "gemma46-qwen6-full-v1": {
        "gemma_executed_layers": "1-45",
        "gemma_executed_layer_count": 45,
        "gemma_layer_count": 46,
        "gemma_layer_mask": 0x3FFFFFFFFFFF,
        "gemma_layers": "0-45",
        "qwen_layer_count": 6,
        "qwen_layer_mask": 0x003F,
        "qwen_layers": "0-5",
        "qwen_last_layer": 5,
    },
}


def marker_records(path: Path, marker: str) -> list[dict[str, Any]]:
    values = [
        json.loads(line.partition(marker)[2])
        for line in path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        if marker in line
    ]
    require(
        all(type(value) is dict for value in values),
        f"invalid {marker.strip()} records",
    )
    return values


def server_ffn_result(path: Path) -> dict[str, Any]:
    marker = "S41SERVERFFN "
    values = [
        json.loads(line[len(marker):])
        for line in path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        if line.startswith(marker + "{")
    ]
    require(len(values) == 1, "Qwen FFN summary is not unique")
    require(type(values[0]) is dict, "Qwen FFN summary is invalid")
    return values[0]


def completion_boundary_skew_ms(
    completion_ns: object, paid_tail_ns: object
) -> float:
    require(
        type(completion_ns) is int and type(paid_tail_ns) is int,
        "Gemma completion boundary type",
    )
    skew_ns = completion_ns - paid_tail_ns
    # Gate STOP and driver JSON are observed by separate threads.
    require(
        abs(skew_ns) <= MAX_COMPLETION_BOUNDARY_SKEW_NS,
        "Gemma completion boundary skew",
    )
    return skew_ns / 1e6


def valid_prefill_execution(
    mode: str,
    prefill_mode: str,
    stage_us: object,
    tail_us: object,
    wall_us: object,
    overlap_us: object,
    ready_depth: object,
    chunk_count: int,
) -> bool:
    if (
        type(stage_us) is not int
        or type(tail_us) is not int
        or type(wall_us) is not int
        or type(overlap_us) is not int
        or type(ready_depth) is not int
    ):
        return False
    if prefill_mode == "sync_chunked":
        return overlap_us == 0 and ready_depth == 1
    if (
        prefill_mode != "async_stream"
        or not 1 < ready_depth <= chunk_count
    ):
        return False
    if overlap_us > 0:
        return True
    control_overhead_us = wall_us - stage_us - tail_us
    return (
        mode in {"adaptive", "control"}
        and 0 <= control_overhead_us <= MAX_CONTROL_PIPELINE_OVERHEAD_US
    )


def require_affinity_record(
    value: object, expected: set[int], phase: str
) -> None:
    require(type(value) is dict, f"Gemma affinity {phase}")
    sets = value.get("cpu_sets")
    require(
        type(sets) is dict
        and bool(sets)
        and all(type(count) is int and count > 0 for count in sets.values())
        and value.get("thread_count") == sum(sets.values())
        and all(cpu_list(text) <= expected for text in sets),
        f"Gemma thread affinity {phase}",
    )


def require_phase_affinity(
    value: object,
    protected: set[int],
    target: set[int],
    protected_done_ns: int,
    paid_end_ns: int,
) -> dict[str, Any] | None:
    require(type(value) is dict, "Gemma affinity receipt")
    require(
        value.get("requested_cpus") == sorted(target)
        and value.get("protected_requested_cpus") == sorted(protected),
        "Gemma requested CPU affinity",
    )
    require_affinity_record(value.get("at_ready"), protected, "at_ready")
    require_affinity_record(
        value.get("at_paid_start"), protected, "at_paid_start"
    )
    switch = value.get("post_protected_switch")
    if protected == target:
        require(switch is None, "unexpected Gemma affinity switch")
        return None
    require(protected.isdisjoint(target), "Gemma protected CPU isolation")
    require(type(switch) is dict, "Gemma affinity switch receipt")
    switch_started_ns = switch.get("switch_started_ns")
    switch_completed_ns = switch.get("switch_completed_ns")
    require(
        switch.get("status") == "PASS"
        and switch.get("protected_done_ns") == protected_done_ns
        and type(switch_started_ns) is int
        and type(switch_completed_ns) is int
        and protected_done_ns <= switch_started_ns <= switch_completed_ns
        and switch_completed_ns <= paid_end_ns
        and switch_completed_ns - switch_started_ns <= MAX_AFFINITY_SWITCH_NS,
        "Gemma affinity switch boundary",
    )
    require_affinity_record(switch.get("before"), protected, "before switch")
    require_affinity_record(switch.get("after"), target, "after switch")
    return switch


def require_process_affinity_receipt(
    value: object, expected: dict[str, set[int]]
) -> dict[str, dict[str, Any]]:
    require(
        type(value) is dict
        and value.get("schema") == "s42-process-affinity-receipt-v1"
        and value.get("status") == "PASS"
        and type(value.get("captured_ns")) is int
        and value["captured_ns"] > 0,
        "process affinity receipt identity",
    )
    processes = value.get("processes")
    require(
        type(processes) is dict and set(processes) == set(expected),
        "process affinity receipt membership",
    )
    for name, cpus in expected.items():
        record = processes[name]
        require(
            type(record) is dict
            and record.get("expected_cpus") == sorted(cpus)
            and type(record.get("pid")) is int
            and record["pid"] > 0
            and type(record.get("start_ticks")) is int
            and record["start_ticks"] > 0
            and type(record.get("cmdline_sha256")) is str
            and len(record["cmdline_sha256"]) == 64,
            f"process affinity identity {name}",
        )
        require_affinity_record(record, cpus, name)
    return processes


def require_cpu_tail_placements(
    placements: list[dict[str, Any]], layer_start: int, n_layer: int
) -> None:
    require(len(placements) == 2, "Gemma tail placement count")
    for placement in placements:
        compute_buffers = placement.get("compute_by_buffer_type")
        copy_buffers = placement.get("copy_by_buffer_type")
        require(
            placement.get("schema") == "layersplit-scheduled-placement-v2"
            and placement.get("role") == "host_tail"
            and placement.get("mode") == "pipedriver"
            and placement.get("layer_start") == layer_start
            and placement.get("layer_end") == n_layer
            and placement.get("n_layer") == n_layer
            and placement.get("status") == "SCHEDULED_PLACEMENT_OK"
            and type(placement.get("compute_nodes")) is int
            and placement["compute_nodes"] > 0
            and placement.get("missing_buffer_compute_nodes") == 0
            and type(compute_buffers) is dict
            and bool(compute_buffers)
            and type(copy_buffers) is dict
            and not any(key.startswith("CUDA") for key in compute_buffers)
            and not any(key.startswith("CUDA") for key in copy_buffers),
            "Gemma tail is not CPU-only",
        )


def require_stage_placement(
    path: Path, layer_start: int, layer_end: int, n_layer: int
) -> dict[str, Any]:
    placements = marker_records(path, "PLACEMENTCERT ")
    require(len(placements) == 1, "GPU-stage placement count")
    placement = placements[0]
    compute_buffers = placement.get("compute_by_buffer_type")
    require(
        placement.get("schema") == "layersplit-scheduled-placement-v2"
        and placement.get("role") == "phone_stage"
        and placement.get("mode") == "stagenet"
        and placement.get("layer_start") == layer_start
        and placement.get("layer_end") == layer_end
        and placement.get("n_layer") == n_layer
        and placement.get("status") == "SCHEDULED_PLACEMENT_OK"
        and placement.get("missing_buffer_compute_nodes") == 0
        and type(compute_buffers) is dict
        and any(key.startswith("CUDA") for key in compute_buffers),
        "resident stage did not execute on CUDA",
    )
    return placement


def require_stage_sessions(
    path: Path,
    *,
    layer_start: int,
    layer_end: int,
    n_layer: int,
    warmup_tokens: int,
    paid_tokens: int,
) -> list[dict[str, Any]]:
    sessions = marker_records(path, "SESSIONCERT ")
    require(len(sessions) == 3, "GPU-stage session count")
    identity = {
        (
            row.get("worker_pid"),
            row.get("worker_boot_nonce"),
            row.get("device_boot_id"),
        )
        for row in sessions
    }
    require(len(identity) == 1, "GPU-stage resident identity changed")
    for index, row in enumerate(sessions, start=1):
        require(
            row.get("schema") == "ls-stagenet-session-v2"
            and row.get("session_id") == index
            and row.get("expected_backend") == "CUDA0"
            and row.get("layer_start") == layer_start
            and row.get("layer_end") == layer_end
            and row.get("n_layer") == n_layer
            and row.get("missing_buffer_compute_nodes") == 0,
            "GPU-stage session identity",
        )
    require(
        sessions[0].get("session_end") == "DETACH"
        and sessions[0].get("steps_session") == 0
        and sessions[0].get("steps_total") == 0
        and sessions[0].get("reset_applied") is True
        and sessions[0].get("placement_status") == "PLACEMENT_UNOBSERVED",
        "GPU-stage materialization probe session",
    )
    require(
        sessions[1].get("session_end") == "DETACH"
        and sessions[1].get("steps_session") == warmup_tokens
        and sessions[1].get("steps_total") == warmup_tokens
        and sessions[1].get("reset_applied") is True
        and sessions[1].get("placement_status") == "SCHEDULED_PLACEMENT_OK",
        "GPU-stage warmup session",
    )
    require(
        sessions[2].get("session_end") == "STOP"
        and sessions[2].get("steps_session") == paid_tokens
        and sessions[2].get("steps_total") == warmup_tokens + paid_tokens
        and sessions[2].get("reset_applied") is False
        and sessions[2].get("placement_status") == "SCHEDULED_PLACEMENT_OK",
        "GPU-stage paid session",
    )
    return sessions


def analyze(args: argparse.Namespace) -> dict[str, object]:
    layout = PHONE_LAYOUTS[args.expected_phone_residency_layout]
    qwen = load(args.qwen_result)
    phone = load(args.phone_energy)
    gate = load(args.gate_result)
    driver = load(args.driver_result)
    profile = load(args.profile)
    manifest = load(args.stage_manifest)
    mps = load(args.mps_receipt)
    qwen_affinity = load(args.qwen_affinity_receipt)
    wavefront_affinity = load(args.wavefront_affinity_receipt)
    bridge = bridge_result(args.bridge_log)
    router = router_result(args.router_log)
    ffn_split = ffn_split_result(args.driver_log)
    phone_residency = phone_residency_receipt(
        args.phone_session_log,
        args.phone_workers_log,
        args.expected_phone_vmem_mib,
        args.expected_phone_min_available_kib,
        args.expected_phone_residency_layout,
    )
    resident_workers = phone_residency.get("workers")
    require(
        type(resident_workers) is dict
        and resident_workers.get("sessions") == ["HTP0", "HTP1", "HTP2"]
        and resident_workers.get("gemma_layers") == layout["gemma_layers"]
        and resident_workers.get("qwen_layers") == layout["qwen_layers"]
        and resident_workers.get("qwen_columns") == 17408,
        "phone resident worker geometry",
    )
    server_log = args.server_log.read_text(encoding="utf-8", errors="replace")
    qwen_ffn = server_ffn_result(args.server_log)

    require(
        qwen.get("schema") == "s41-burstgpt-llama-server-result-v1"
        and qwen.get("status") == "PASS"
        and qwen.get("arm") == "op15",
        "Qwen result identity",
    )
    paid_start_ns = qwen.get("paid_start_ns")
    paid_barrier_ready_ns = qwen.get("paid_barrier_ready_ns")
    paid_barrier_release_ns = qwen.get("paid_barrier_release_ns")
    prefetch_arm_ns = qwen.get("prefetch_arm_ns")
    qwen_end_ns = qwen.get("qwen_end_ns")
    paid_tail_ns = qwen.get("paid_tail_end_ns")
    paid_end_ns = qwen.get("paid_end_ns")
    require(
        all(
            type(value) is int
            for value in (
                paid_barrier_ready_ns,
                prefetch_arm_ns,
                paid_barrier_release_ns,
                paid_start_ns,
                qwen_end_ns,
                paid_tail_ns,
                paid_end_ns,
            )
        )
        and paid_barrier_ready_ns < prefetch_arm_ns
        <= paid_barrier_release_ns <= paid_start_ns
        and paid_start_ns < qwen_end_ns <= paid_end_ns
        and paid_start_ns < paid_tail_ns <= paid_end_ns
        and paid_end_ns == max(qwen_end_ns, paid_tail_ns),
        "paid interval",
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
    join_layer_ready = (
        args.expected_qwen_tail_fence_join_layer < 0
        or f"tail_fence_join_layer={args.expected_qwen_tail_fence_join_layer}"
        in server_log
    )
    gpu_placement_ready = args.expected_qwen_log_verbosity < 4 or (
        args.expected_qwen_gpu_layers > 0
        and "offloading output layer to GPU" in server_log
        and f"offloading {args.expected_qwen_gpu_layers - 1} repeating "
        "layers to GPU" in server_log
        and f"offloaded {args.expected_qwen_gpu_layers}/41 layers to GPU"
        in server_log
    )
    require(
        type(command) is list
        and bool(command)
        and command[0] == str(args.expected_qwen_server)
        and "--n-gpu-layers" in command
        and command[command.index("--n-gpu-layers") + 1]
        == str(args.expected_qwen_gpu_layers)
        and "--log-verbosity" in command
        and command[command.index("--log-verbosity") + 1]
        == str(args.expected_qwen_log_verbosity)
        and "--device" in command
        and command[command.index("--device") + 1] == "CUDA0"
        and (
            f"layers={layout['qwen_layer_count']} "
            f"mask={layout['qwen_layer_mask']:016x}"
        ) in server_log
        and "tail_fence=enabled" in server_log
        and f"tail_fence_layer={args.expected_qwen_tail_fence_layer}"
        in server_log
        and join_layer_ready
        and gpu_placement_ready,
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
        gate.get("schema")
        == "s42-fp16-burstgpt-gpu-stage-wavefront-gate-v1"
        and gate.get("status") == "PASS"
        and gate.get("mode") == args.mode
        and gate.get("qwen_complete_ns") == qwen_end_ns
        and gate.get("paid_tail_ns") == paid_tail_ns
        and gate.get("energy_claim_eligible") is False
        and gate.get("admission_disabled_reason") is None
        and gate.get("circuit_breaker_reason") is None
        and gate.get("circuit_breaker_at_backfill") is None
        and gate.get("fallback_executions") == 0,
        "GPU-stage gate identity",
    )
    candidate_ready_ns = gate.get("candidate_ready_ns")
    candidate_prepare_arm_ns = gate.get("candidate_prepare_arm_ns")
    candidate_prepared_ns = gate.get("candidate_prepared_ns")
    require(
        all(
            type(value) is int
            for value in (
                candidate_ready_ns,
                candidate_prepare_arm_ns,
                candidate_prepared_ns,
            )
        )
        and prefetch_arm_ns <= candidate_ready_ns
        <= candidate_prepare_arm_ns <= candidate_prepared_ns
        <= paid_barrier_release_ns,
        "GPU-stage candidate preparation interval",
    )
    worker_identity = gate.get("worker_identity")
    require(
        type(worker_identity) is dict
        and worker_identity.get("layer_start") == 0
        and worker_identity.get("layer_end") == args.expected_prefix_layers
        and worker_identity.get("n_layer") == 48
        and worker_identity.get("n_embd") == 3840
        and worker_identity.get("file_type") == 1
        and worker_identity.get("max_streams") == 1
        and worker_identity.get("n_batch") == 512
        and worker_identity.get("n_ubatch") == 512,
        "GPU-stage live identity",
    )

    require(
        driver.get("schema") == "s42-gemma-wavefront-driver-v1"
        and driver.get("status") == "PASS"
        and driver.get("route_kind") == "gpu-prefix"
        and driver.get("gpu_prefix_layers") == args.expected_prefix_layers
        and driver.get("prefill_mode") == args.expected_gemma_prefill_mode
        and driver.get("stream_prefill")
        == (args.expected_gemma_prefill_mode == "async_stream")
        and driver.get("arm_ns") == prefetch_arm_ns,
        "Gemma driver identity",
    )
    gemma_rows = driver.get("request_results")
    require(
        type(gemma_rows) is list and len(gemma_rows) == 1,
        "Gemma request count",
    )
    gemma = gemma_rows[0]
    require(type(gemma) is dict, "Gemma request record")
    completion_skew_ms = completion_boundary_skew_ms(
        gemma.get("completion_ns"), paid_tail_ns
    )
    require(
        gemma.get("request_index") == args.expected_gemma_index
        and type(gemma.get("token_ids")) is list
        and len(gemma["token_ids"]) == gemma.get("output_tokens")
        and type(gemma.get("input_tokens")) is int
        and gemma["input_tokens"] > 0
        and prefetch_arm_ns <= gemma.get("dispatch_ns")
        <= candidate_ready_ns <= paid_start_ns
        and paid_start_ns < paid_tail_ns
        and gemma.get("dispatch_ns") < gemma.get("completion_ns"),
        "Gemma output or interval",
    )
    driver_command = driver.get("driver_command")
    expected_prefill_flag = (
        "--driver-stream-prefill"
        if args.expected_gemma_prefill_mode == "async_stream"
        else "--driver-sync-prefill"
    )
    unexpected_prefill_flag = (
        "--driver-sync-prefill"
        if args.expected_gemma_prefill_mode == "async_stream"
        else "--driver-stream-prefill"
    )
    expected_gemma_cpus = cpu_list(args.expected_gemma_cpus)
    expected_gemma_protected_cpus = cpu_list(
        args.expected_gemma_protected_cpus
    )
    require(
        type(driver_command) is list
        and driver_command[:3]
        == [
            "taskset",
            "--cpu-list",
            ",".join(str(cpu) for cpu in sorted(expected_gemma_protected_cpus)),
        ]
        and "--mode" in driver_command
        and driver_command[driver_command.index("--mode") + 1] == "pipedriver"
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
        and expected_prefill_flag in driver_command
        and unexpected_prefill_flag not in driver_command
        and driver.get("ubatch") == args.expected_gemma_ubatch
        and "--lm-head-rows" not in driver_command,
        "Gemma CPU-tail route",
    )
    ffn_route = driver.get("ffn_route")
    require(
        type(ffn_route) is dict
        and ffn_route.get("host") == "127.0.0.1"
        and ffn_route.get("layers") == layout["gemma_layers"]
        and ffn_route.get("columns") == 6144
        and ffn_route.get("prefill_columns") == 6144
        and ffn_route.get("decode_columns") == 6144
        and ffn_route.get("timeout_ms") == args.expected_phone_ffn_timeout_ms
        and ffn_route.get("f16_io") is True
        and "--ffn-host" in driver_command
        and "--ffn-port" in driver_command,
        "Gemma resident phone FFN route",
    )
    require(
        driver.get("producer_environment")
        == {
            "CUDA_VISIBLE_DEVICES": "",
            "LLAMA_LAYER_START": str(args.expected_prefix_layers),
        },
        "Gemma tail CUDA isolation",
    )
    require(
        args.expected_gemma_threads == len(expected_gemma_cpus),
        "Gemma thread and CPU geometry",
    )
    require(
        args.expected_gemma_threads == len(expected_gemma_protected_cpus),
        "Gemma protected thread and CPU geometry",
    )
    affinity_switch = require_phase_affinity(
        driver.get("producer_affinity"),
        expected_gemma_protected_cpus,
        expected_gemma_cpus,
        qwen_end_ns,
        paid_end_ns,
    )
    require_cpu_tail_placements(
        placement_certificates(args.driver_log), args.expected_prefix_layers, 48
    )

    expected_shapes = expected_ffn_shapes(
        gemma_rows,
        args.expected_gemma_ubatch,
        True,
        layer_count=layout["gemma_executed_layer_count"],
    )
    expected_filler_calls = sum(expected_shapes.values())
    observed_shapes: dict[int, int] = {}
    require(
        ffn_split.get("schema") == "layersplit-ffn-overlap-v2"
        and ffn_split.get("status") == "FFN_OVERLAP_OK"
        and ffn_split.get("layer_count") == layout["gemma_layer_count"]
        and ffn_split.get("layer_mask") == layout["gemma_layer_mask"]
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

    require(
        profile.get("schema")
        == "s42-fp16-burstgpt-gpu-stage-wavefront-profile-v1"
        and gate.get("profile_sha256") == canonical_sha256(profile),
        "GPU-stage profile binding",
    )
    resource_snapshot = gate.get("resource_snapshot")
    protected_resources_reserved = args.mode in {
        "adaptive", "qualified", "staged"
    }
    expected_producer_resources = (
        ["cpu"] if args.mode == "staged" else ["cpu", "op15-htp"]
    )
    expected_fenced_resources = (
        ["op15-htp"] if args.mode == "staged" else []
    )
    require(
        profile.get("producer_resource_ids") == expected_producer_resources
        and profile.get("protected_resource_ids") == ["op15-htp"]
        and profile.get("fenced_resource_ids") == expected_fenced_resources
        and type(resource_snapshot) is dict
        and resource_snapshot.get("schema")
        == "s42-gpu-stage-resource-snapshot-v1"
        and resource_snapshot.get("producer_resource_ids")
        == profile["producer_resource_ids"]
        and resource_snapshot.get("protected_resource_ids")
        == profile["protected_resource_ids"]
        and resource_snapshot.get("fenced_resource_ids")
        == profile["fenced_resource_ids"]
        and resource_snapshot.get("protected_resources_reserved")
        is protected_resources_reserved
        and type(resource_snapshot.get("protected_resource_leases")) is list
        and len(resource_snapshot["protected_resource_leases"])
        == (1 if protected_resources_reserved else 0)
        and all(
            type(lease) is dict
            and lease.get("resource_id") == "op15-htp"
            and type(lease.get("start_us")) is int
            and type(lease.get("reserved_until_us")) is int
            and lease["start_us"] < lease["reserved_until_us"]
            for lease in resource_snapshot["protected_resource_leases"]
        ),
        "GPU-stage causal resource snapshot",
    )
    placement = validate_manifest(manifest, profile)
    memory = profile.get("memory")
    candidate = profile.get("candidate")
    model = profile.get("model")
    require(
        type(memory) is dict and type(candidate) is dict and type(model) is dict,
        "GPU-stage profile sections",
    )
    prefill_chunk_rows = candidate.get("prefill_chunk_rows")
    require(
        type(prefill_chunk_rows) is int
        and 0 < prefill_chunk_rows <= args.expected_gemma_ubatch,
        "GPU-stage prefill chunk is not bounded by the Gemma ubatch",
    )
    require(
        args.expected_gemma_ubatch % prefill_chunk_rows == 0,
        "GPU-stage chunks do not tile the host prefill microbatch",
    )
    memory_cert = parse_memory_certificate(args.worker_log)
    connected_pids = mps.get("connected_pids")
    device_uuids = mps.get("device_uuids")
    require(
        memory.get("stage_memory_certificate") == memory_cert
        and model.get("resident_bytes") == memory_cert["model_buffer_bytes"]
        and model.get("stage_weight_sha256")
        == placement["stage_weight_sha256"]
        and memory["free_bytes"] - memory["reserve_bytes"]
        >= candidate["workspace_bytes"],
        "GPU-stage memory and manifest binding",
    )
    require(
        mps.get("schema") == "s42-cuda-mps-receipt-v2"
        and mps.get("status") == "PASS"
        and type(connected_pids) is list
        and len(connected_pids) >= 2
        and len(set(connected_pids)) == len(connected_pids)
        and all(type(value) is int and value > 0 for value in connected_pids)
        and mps.get("stage_worker_pid") == memory_cert.get("pid")
        and mps["stage_worker_pid"] in connected_pids
        and type(device_uuids) is list
        and memory.get("gpu_uuid") == mps.get("gpu_uuid")
        and mps["gpu_uuid"] in device_uuids
        and type(mps.get("control_version")) is int
        and mps["control_version"] > 0
        and mps.get("qwen_active_thread_percentage")
        == args.expected_qwen_mps_active_thread_pct
        and mps.get("stage_active_thread_percentage")
        == args.expected_stage_mps_active_thread_pct
        and mps.get("qwen_client_priority")
        == args.expected_qwen_mps_client_priority
        and mps.get("stage_client_priority")
        == args.expected_stage_mps_client_priority
        and type(mps.get("qwen_client_pids")) is list
        and len(mps["qwen_client_pids"]) == 1
        and mps["qwen_client_pids"][0] in connected_pids
        and mps["qwen_client_pids"][0] != mps["stage_worker_pid"]
        and type(mps.get("client_priorities")) is dict
        and mps["client_priorities"]
        == {
            str(mps["qwen_client_pids"][0]): args.expected_qwen_mps_client_priority,
            str(mps["stage_worker_pid"]): args.expected_stage_mps_client_priority,
        }
        and all(
            type(mps.get(name)) is str and len(mps[name]) == 64
            for name in ("control_log_sha256", "server_log_sha256")
        ),
        "CUDA MPS lifecycle receipt",
    )
    qwen_affinity_processes = require_process_affinity_receipt(
        qwen_affinity,
        {
            "phone-bridge": cpu_list(args.expected_phone_bridge_cpus),
            "qwen-server": cpu_list(args.expected_qwen_cpus),
        },
    )
    wavefront_affinity_processes = require_process_affinity_receipt(
        wavefront_affinity,
        {
            "gpu-stage": cpu_list(args.expected_stage_worker_cpus),
            "wavefront-gate": cpu_list(args.expected_wavefront_gate_cpus),
        },
    )
    require(
        qwen_affinity_processes["qwen-server"]["pid"]
        == mps["qwen_client_pids"][0]
        and wavefront_affinity_processes["gpu-stage"]["pid"]
        == mps["stage_worker_pid"]
        and wavefront_affinity_processes["wavefront-gate"]["pid"]
        == gate.get("pid"),
        "process affinity PID binding",
    )
    stage_placement = require_stage_placement(
        args.worker_log, 0, args.expected_prefix_layers, 48
    )
    output_tokens = gemma["output_tokens"]
    input_tokens = gemma["input_tokens"]
    host_prefill_chunks = math.ceil(input_tokens / args.expected_gemma_ubatch)
    prefill_stage_us = gemma.get("prefill_stage_us")
    prefill_tail_us = gemma.get("prefill_tail_us")
    prefill_pipeline_wall_us = gemma.get("prefill_pipeline_wall_us")
    prefill_overlap_us = gemma.get("prefill_overlap_us")
    prefill_max_ready_depth = gemma.get("prefill_max_ready_depth")
    prefill_execution_valid = valid_prefill_execution(
        args.mode,
        args.expected_gemma_prefill_mode,
        prefill_stage_us,
        prefill_tail_us,
        prefill_pipeline_wall_us,
        prefill_overlap_us,
        prefill_max_ready_depth,
        host_prefill_chunks,
    )
    require(
        gemma.get("prefill_mode") == args.expected_gemma_prefill_mode
        and gemma.get("prefill_chunk_tokens") == args.expected_gemma_ubatch
        and gemma.get("prefill_chunks") == host_prefill_chunks
        and type(prefill_stage_us) is int
        and prefill_stage_us > 0
        and type(prefill_tail_us) is int
        and prefill_tail_us > 0
        and type(prefill_pipeline_wall_us) is int
        and prefill_pipeline_wall_us > 0
        and type(prefill_overlap_us) is int
        and prefill_overlap_us >= 0
        and prefill_stage_us + prefill_tail_us == gemma.get("prefill_us")
        and prefill_overlap_us
        == max(0, prefill_stage_us + prefill_tail_us - prefill_pipeline_wall_us)
        and prefill_execution_valid,
        "Gemma chunked prefill receipt",
    )
    paid_stage_tokens = input_tokens + output_tokens - 1
    prepare_calls = gate.get("prepare_calls")
    prepare_durations_us = gate.get("prepare_durations_us")
    require(
        type(prepare_calls) is int
        and args.expected_prepare_required_consecutive
        <= prepare_calls <= args.expected_prepare_max_replays
        and type(prepare_durations_us) is list
        and len(prepare_durations_us) == prepare_calls
        and all(type(value) is int and value > 0 for value in prepare_durations_us)
        and all(
            value <= candidate["prefill_service_latency_us"]["upper"]
            for value in prepare_durations_us[
                -args.expected_prepare_required_consecutive:
            ]
        ),
        "GPU-stage preparation stability",
    )
    sessions = require_stage_sessions(
        args.worker_log,
        layer_start=0,
        layer_end=args.expected_prefix_layers,
        n_layer=48,
        warmup_tokens=min(args.expected_gemma_ubatch, input_tokens),
        paid_tokens=paid_stage_tokens + prefill_chunk_rows * prepare_calls,
    )

    backfills = gate.get("backfills")
    prefill_chunks = math.ceil(input_tokens / prefill_chunk_rows)
    warmup_chunks = math.ceil(
        min(args.expected_gemma_ubatch, input_tokens) / prefill_chunk_rows
    )
    paid_chunk_calls = prefill_chunks + output_tokens - 1
    require(
        gate.get("calls") == output_tokens + host_prefill_chunks
        and gate.get("chunk_calls")
        == paid_chunk_calls + warmup_chunks + prepare_calls
        and gate.get("prefill_chunk_rows") == prefill_chunk_rows
        and gate.get("maximum_backfills")
        == args.expected_maximum_backfills
        and gate.get("prepare_max_replays")
        == args.expected_prepare_max_replays
        and gate.get("prepare_required_consecutive")
        == args.expected_prepare_required_consecutive
        and gate.get("warmup_calls") == warmup_chunks
        and type(backfills) is int,
        "GPU-stage call accounting",
    )
    if args.mode in {"adaptive", "control"}:
        require(
            backfills == 0
            and gate.get("tail_calls") == paid_chunk_calls,
            "deferred GPU-stage routing",
        )
    else:
        require(
            0 < backfills <= args.expected_maximum_backfills
            and backfills <= prefill_chunks
            and gate.get("tail_calls") == paid_chunk_calls - backfills,
            "treatment GPU-stage routing",
        )
    wavefront_events = [
        row for row in gate.get("events", [])
        if type(row) is dict and row.get("event") == "wavefront_execute"
    ]
    prepare_events = [
        row for row in gate.get("events", [])
        if type(row) is dict and row.get("event") == "candidate_prepare"
    ]
    prepare_output_sha256 = (
        prepare_events[0].get("prepare_output_sha256")
        if len(prepare_events) == 1
        else None
    )
    require(
        len(prepare_events) == 1
        and prepare_events[0].get("shape") == "prefill"
        and prepare_events[0].get("row_start") == 0
        and prepare_events[0].get("position_start") == 0
        and prepare_events[0].get("position_end") == prefill_chunk_rows
        and prepare_events[0].get("rows") == prefill_chunk_rows
        and prepare_events[0].get("sequence_index") == 0
        and prepare_events[0].get("candidate_prepare_arm_ns")
        == candidate_prepare_arm_ns
        and prepare_events[0].get("candidate_prepared_ns")
        == candidate_prepared_ns
        and prepare_events[0].get("prepare_replay_durations_us")
        == prepare_durations_us
        and prepare_events[0].get("prepare_required_consecutive")
        == args.expected_prepare_required_consecutive
        and prepare_events[0].get("prepare_service_upper_us")
        == candidate["prefill_service_latency_us"]["upper"]
        and type(prepare_output_sha256) is list
        and len(prepare_output_sha256) == prepare_calls
        and len(set(prepare_output_sha256)) == 1
        and all(
            type(value) is str
            and value.startswith("sha256:")
            and len(value) == 71
            for value in prepare_output_sha256
        ),
        "GPU-stage pre-boundary preparation",
    )
    rejected_events = [
        row for row in gate.get("events", [])
        if type(row) is dict and row.get("event") == "wavefront_rejected"
    ]
    if args.mode in {"adaptive", "control"}:
        require(not wavefront_events, "deferred route executed a GPU wavefront")
        prefix_overlap_us = 0
        if args.mode == "adaptive":
            require(
                gate.get("rejections") == len(rejected_events)
                and len(rejected_events) > 0
                and all(
                    type(event.get("decision")) is dict
                    and event["decision"].get("chunk_id") is None
                    and event["decision"].get("backfill", {}).get("reason")
                    == "LEAVE_GPU_IDLE"
                    and any(
                        type(row) is dict
                        and row.get("reason") == "RESOURCE_NOT_READY"
                        for row in event["decision"].get("backfill", {}).get(
                            "rejected", []
                        )
                    )
                    for event in rejected_events
                ),
                "adaptive resource-contention rejection",
            )
        else:
            require(
                gate.get("rejections") == 0 and not rejected_events,
                "control scheduled a GPU candidate",
            )
    else:
        require(len(wavefront_events) == backfills, "GPU prefix overlap count")
        prefix_overlap_us = 0
        for index, event in enumerate(wavefront_events):
            expected_start = index * prefill_chunk_rows
            expected_rows = min(
                prefill_chunk_rows, input_tokens - expected_start
            )
            duration_us = event.get("duration_us")
            require(
                event.get("shape") == "prefill"
                and event.get("rows") == expected_rows
                and event.get("position_start") == expected_start
                and event.get("position_end")
                == expected_start + expected_rows
                and event.get("sequence_index") == index
                and type(duration_us) is int
                and 0 < duration_us
                <= candidate["prefill_service_latency_us"]["upper"]
                and paid_start_ns <= event.get("at_ns") <= qwen_end_ns,
                "Gemma prefix prefill did not overlap Qwen safely",
            )
            prefix_overlap_us += duration_us
        receipts = gate.get("completed_receipts")
        require(
            type(receipts) is list
            and len(receipts) == backfills
            and len(set(receipts)) == backfills,
            "GPU prefix causal receipts",
        )

    require(
        qwen_ffn.get("status") == "ok"
        and qwen_ffn.get("calls") == qwen_ffn.get("decode_calls")
        and qwen_ffn.get("prefill_calls") == 0
        and qwen_ffn.get("tail_fence_opportunities")
        == gate.get("fence_calls")
        + qwen_ffn.get("tail_fence_skipped_complete")
        and qwen_ffn.get("tail_fence_calls") == gate.get("fence_calls")
        and type(qwen_ffn.get("phone_tail_min_ms")) in (int, float)
        and qwen_ffn["phone_tail_min_ms"] > 0
        and type(qwen_ffn.get("phone_tail_p10_ms")) in (int, float)
        and qwen_ffn["phone_tail_p10_ms"] >= qwen_ffn["phone_tail_min_ms"]
        and type(qwen_ffn.get("phone_tail_p50_ms")) in (int, float)
        and qwen_ffn["phone_tail_p50_ms"] >= qwen_ffn["phone_tail_p10_ms"]
        and type(qwen_ffn.get("tail_fence_overlap_mean_ms")) in (int, float)
        and qwen_ffn["tail_fence_overlap_mean_ms"] >= 0
        and type(qwen_ffn.get("tail_fence_overrun_max_ms")) in (int, float)
        and qwen_ffn["tail_fence_overrun_max_ms"] >= 0,
        "Qwen synchronized phone-tail fence",
    )
    if args.expected_qwen_tail_fence_join_layer >= 0:
        require(
            qwen_ffn.get("tail_fence_macro_windows")
            == qwen_ffn.get("tail_fence_calls")
            and type(qwen_ffn.get("tail_fence_window_min_ms"))
            in (int, float)
            and qwen_ffn["tail_fence_window_min_ms"] > 0
            and qwen_ffn.get("tail_fence_window_p10_ms")
            >= qwen_ffn["tail_fence_window_min_ms"]
            and qwen_ffn.get("tail_fence_window_p50_ms")
            >= qwen_ffn["tail_fence_window_p10_ms"]
            and qwen_ffn.get("tail_fence_window_p90_ms")
            >= qwen_ffn["tail_fence_window_p50_ms"]
            and qwen_ffn.get("tail_fence_window_max_ms")
            >= qwen_ffn["tail_fence_window_p90_ms"]
            and type(qwen_ffn.get("tail_fence_join_wait_max_ms"))
            in (int, float)
            and qwen_ffn["tail_fence_join_wait_max_ms"] >= 0,
            "Qwen macro CPU-phone suffix fence",
        )

    filler_before_done = bridge.get("filler_before_protected_done")
    filler_after_done = bridge.get("filler_after_protected_done")
    phone_overlap_source = "bridge-counters"
    if filler_before_done is None and filler_after_done is None:
        require(
            args.mode in {"adaptive", "control"} and backfills == 0,
            "phone overlap counters are absent",
        )
        filler_before_done = 0
        filler_after_done = expected_filler_calls
        phone_overlap_source = "causal-control"
    require(
        type(filler_before_done) is int
        and filler_before_done >= 0
        and type(filler_after_done) is int
        and filler_after_done >= 0
        and filler_before_done + filler_after_done == expected_filler_calls,
        "phone overlap accounting",
    )
    require(
        bridge.get("status") == "MECHANICS_ONLY"
        and bridge.get("defer_filler_until_protected_done")
        is (args.mode == "staged")
        and bridge.get("prefetch_fence_enabled") is False
        and bridge.get("prefetch_fence_calls") == 0
        and bridge.get("reset_recoveries") == 0
        and bridge.get("protected_done_observed") is True
        and bridge.get("filler_calls") == expected_filler_calls
        and bridge.get("filler_calls") == bridge.get("filler_admitted")
        and bridge.get("observed_idle_min_us") >= bridge.get("idle_lower_us")
        and bridge.get("filler_before_protected_done_max_us")
        <= bridge.get("filler_upper_us")
        and bridge.get("filler_upper_us") + bridge.get("guard_us")
        <= bridge.get("idle_lower_us")
        and bridge.get(
            "filler_upper_violations_before_protected_done"
        ) == 0
        and type(
            bridge.get("filler_upper_violations_after_protected_done")
        ) is int
        and bridge["filler_upper_violations_after_protected_done"] >= 0
        and bridge.get("filler_upper_violations")
        == bridge["filler_upper_violations_after_protected_done"]
        and bridge.get("guard_violations") == 0
        and bridge.get("idle_lower_violations") == 0
        and bridge.get("protected_pending_after_filler") == 0
        and bridge.get("energy_claim_eligible") is False,
        "phone protected-first mechanics",
    )
    expected_prefill_calls = (
        expected_filler_calls - expected_shapes.get(1, 0)
    )
    prefill_before_done = min(filler_before_done, expected_prefill_calls)
    if args.mode in {"adaptive", "control", "staged"}:
        require(
            prefill_before_done == 0,
            "deferred route overlapped Gemma phone prefill",
        )
    else:
        require(
            prefill_before_done > 0
            and qwen_ffn.get("tail_fence_overrun_max_ms") == 0,
            "Gemma phone prefill did not overlap Qwen safely",
        )
    protected_first = 0
    protected_last = layout["qwen_last_layer"]
    protected_groups = bridge.get("protected_group_ends")
    require(
        type(protected_groups) is int
        and protected_groups > 0
        and bridge.get("protected_group_starts") == protected_groups
        and bridge.get("protected_calls")
        == protected_groups * (protected_last - protected_first + 1)
        and qwen_ffn.get("calls") == bridge.get("protected_calls")
        and bridge.get("idle_samples") == protected_groups - 1,
        "Qwen protected phone call accounting",
    )
    require(
        router.get("status") == "ok"
        and router.get("terminate_requested") is True
        and type(router.get("sessions")) is int
        and router["sessions"] >= 2
        and router.get("requests")
        == bridge["protected_calls"] + bridge["filler_calls"],
        "phone router accounting",
    )

    output: dict[str, object] = {
        "admission": {
            "adaptive": "GPU_PREFIX_ADAPTIVE_LEAVE_IDLE",
            "control": "GPU_PREFIX_CONTROL_ONLY",
            "mechanics": "GPU_PREFIX_MECHANICS_PASS",
            "qualified": "MATCHED_ENERGY_QUALIFICATION_REQUIRED",
            "staged": "GPU_PREFIX_STAGED_MECHANICS_PASS",
        }[args.mode],
        "artifacts": {
            "bridge_log_sha256": sha256(args.bridge_log),
            "driver_log_sha256": sha256(args.driver_log),
            "driver_result_sha256": sha256(args.driver_result),
            "gate_result_sha256": sha256(args.gate_result),
            "phone_energy_sha256": sha256(args.phone_energy),
            "profile_sha256": sha256(args.profile),
            "qwen_affinity_receipt_sha256": sha256(
                args.qwen_affinity_receipt
            ),
            "qwen_result_sha256": sha256(args.qwen_result),
            "qwen_server_sha256": sha256(args.expected_qwen_server),
            "router_log_sha256": sha256(args.router_log),
            "server_log_sha256": sha256(args.server_log),
            "stage_manifest_sha256": sha256(args.stage_manifest),
            "wavefront_affinity_receipt_sha256": sha256(
                args.wavefront_affinity_receipt
            ),
            "worker_log_sha256": sha256(args.worker_log),
        },
        "energy": {
            "boundary": "cpu-package+gpu-board+whole-phone-paid-work",
            "cpu_package_j": cpu_j,
            "fleet_j": server_j + phone_j,
            "gpu_board_j": gpu_j,
            "phone_j": phone_j,
            "server_j": server_j,
        },
        "cuda_mps": mps,
        "cpu_affinity": {
            "qwen": qwen_affinity,
            "wavefront": wavefront_affinity,
        },
        "energy_claim_eligible": False,
        "gemma": {
            "affinity_switch": affinity_switch,
            "cpu_list": sorted(expected_gemma_cpus),
            "completion_boundary_skew_ms": completion_skew_ms,
            "decode_us": gemma.get("decode_us"),
            "executed_phone_ffn_layers": layout["gemma_executed_layers"],
            "ffn_split": ffn_split,
            "index": gemma.get("request_index"),
            "prefill_max_ready_depth": prefill_max_ready_depth,
            "prefill_mode": args.expected_gemma_prefill_mode,
            "prefill_overlap_us": prefill_overlap_us,
            "prefill_pipeline_wall_us": prefill_pipeline_wall_us,
            "prefill_stage_us": prefill_stage_us,
            "prefill_tail_us": prefill_tail_us,
            "prefill_us": gemma.get("prefill_us"),
            "prefill_chunks": host_prefill_chunks,
            "protected_cpu_list": sorted(expected_gemma_protected_cpus),
            "resident_phone_ffn_layers": layout["gemma_layers"],
            "stream_prefill": args.expected_gemma_prefill_mode == "async_stream",
            "threads": args.expected_gemma_threads,
            "ubatch": args.expected_gemma_ubatch,
        },
        "gpu_prefix": {
            "backfills": backfills,
            "maximum_backfills": args.expected_maximum_backfills,
            "prepare_calls": prepare_calls,
            "prepare_durations_us": prepare_durations_us,
            "prepare_required_consecutive": (
                args.expected_prepare_required_consecutive
            ),
            "prefill_chunk_rows": prefill_chunk_rows,
            "prefill_chunks": prefill_chunks,
            "layer_end": args.expected_prefix_layers,
            "layer_start": 0,
            "memory_certificate": memory_cert,
            "overlap_observed": backfills > 0,
            "placement": stage_placement,
            "prefill_overlap_us": prefix_overlap_us,
            "rejections": gate.get("rejections"),
            "resource_snapshot": gate.get("resource_snapshot"),
            "sessions": sessions,
            "stage_weight_sha256": placement["stage_weight_sha256"],
        },
        "mode": args.mode,
        "paid_duration_s": duration_s,
        "phone_overlap": {
            "accounting_source": phone_overlap_source,
            "filler_after_protected_done": filler_after_done,
            "filler_before_protected_done": filler_before_done,
            "prefill_before_protected_done": prefill_before_done,
            "prefill_calls": expected_prefill_calls,
        },
        "phone_residency": phone_residency,
        "qwen": {
            "duration_s": (qwen_end_ns - paid_start_ns) / 1e9,
            "ffn": qwen_ffn,
            "gpu_layers": args.expected_qwen_gpu_layers,
            "indices": qwen.get("indices"),
            "requests": len(qwen_rows),
            "tail_fence_join_layer": (
                args.expected_qwen_tail_fence_join_layer
            ),
        },
        "router": router,
        "schema": SCHEMA,
        "status": "PASS",
    }
    output["record_sha256"] = hashlib.sha256(
        (
            json.dumps(
                output,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    ).hexdigest()
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("adaptive", "control", "mechanics", "qualified", "staged"),
        required=True,
    )
    parser.add_argument("--qwen-result", type=Path, required=True)
    parser.add_argument("--phone-energy", type=Path, required=True)
    parser.add_argument("--gate-result", type=Path, required=True)
    parser.add_argument("--driver-result", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--bridge-log", type=Path, required=True)
    parser.add_argument("--router-log", type=Path, required=True)
    parser.add_argument("--phone-session-log", type=Path, required=True)
    parser.add_argument("--phone-workers-log", type=Path, required=True)
    parser.add_argument("--driver-log", type=Path, required=True)
    parser.add_argument("--worker-log", type=Path, required=True)
    parser.add_argument("--mps-receipt", type=Path, required=True)
    parser.add_argument("--qwen-affinity-receipt", type=Path, required=True)
    parser.add_argument("--wavefront-affinity-receipt", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--expected-qwen-server", type=Path, required=True)
    parser.add_argument("--expected-qwen-gpu-layers", type=int, required=True)
    parser.add_argument(
        "--expected-qwen-tail-fence-layer", type=int, required=True
    )
    parser.add_argument(
        "--expected-qwen-tail-fence-join-layer", type=int, default=-1
    )
    parser.add_argument("--expected-qwen-log-verbosity", type=int, default=1)
    parser.add_argument("--expected-qwen-cpus", required=True)
    parser.add_argument("--expected-phone-bridge-cpus", required=True)
    parser.add_argument("--expected-stage-worker-cpus", required=True)
    parser.add_argument("--expected-wavefront-gate-cpus", required=True)
    parser.add_argument("--expected-prefix-layers", type=int, default=1)
    parser.add_argument("--expected-gemma-index", type=int, default=50)
    parser.add_argument("--expected-gemma-cpus", required=True)
    parser.add_argument("--expected-gemma-protected-cpus", required=True)
    parser.add_argument("--expected-gemma-threads", type=int, required=True)
    parser.add_argument("--expected-gemma-ubatch", type=int, required=True)
    parser.add_argument(
        "--expected-gemma-prefill-mode",
        choices=("async_stream", "sync_chunked"),
        required=True,
    )
    parser.add_argument("--expected-maximum-backfills", type=int, required=True)
    parser.add_argument("--expected-prepare-max-replays", type=int, required=True)
    parser.add_argument(
        "--expected-prepare-required-consecutive", type=int, required=True
    )
    parser.add_argument(
        "--expected-phone-ffn-timeout-ms", type=int, required=True
    )
    parser.add_argument(
        "--expected-phone-residency-layout",
        choices=tuple(PHONE_LAYOUTS),
        required=True,
    )
    parser.add_argument("--expected-phone-vmem-mib", type=int, required=True)
    parser.add_argument(
        "--expected-phone-min-available-kib", type=int, required=True
    )
    parser.add_argument(
        "--expected-qwen-mps-active-thread-pct", type=int, required=True
    )
    parser.add_argument(
        "--expected-stage-mps-active-thread-pct", type=int, required=True
    )
    parser.add_argument(
        "--expected-qwen-mps-client-priority", type=int, required=True
    )
    parser.add_argument(
        "--expected-stage-mps-client-priority", type=int, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.expected_qwen_gpu_layers <= 41:
        parser.error("invalid expected Qwen GPU layer count")
    if not 0 <= args.expected_qwen_tail_fence_layer <= 11:
        parser.error("invalid expected Qwen tail fence layer")
    if (
        args.expected_qwen_tail_fence_layer
        > PHONE_LAYOUTS[args.expected_phone_residency_layout][
            "qwen_last_layer"
        ]
    ):
        parser.error("Qwen tail fence layer is outside phone residency")
    if not -1 <= args.expected_qwen_tail_fence_join_layer < 40:
        parser.error("invalid expected Qwen tail fence join layer")
    if (
        args.expected_qwen_tail_fence_join_layer >= 0
        and args.expected_qwen_tail_fence_join_layer
        <= args.expected_qwen_tail_fence_layer
    ):
        parser.error("Qwen tail fence join must follow its launch layer")
    if args.expected_prefix_layers != 1:
        parser.error("this qualification fixes one Gemma prefix layer")
    if args.expected_gemma_threads <= 0:
        parser.error("invalid expected Gemma thread count")
    if not 1 <= args.expected_gemma_ubatch <= 16:
        parser.error("this qualification requires a Gemma ubatch from M1 through M16")
    if not 1 <= args.expected_maximum_backfills <= 512:
        parser.error("invalid maximum GPU backfill count")
    if not 1 <= args.expected_prepare_max_replays <= 16:
        parser.error("invalid preparation replay count")
    if not (
        1
        <= args.expected_prepare_required_consecutive
        <= args.expected_prepare_max_replays
    ):
        parser.error("invalid required stable preparation count")
    if not 1 <= args.expected_phone_ffn_timeout_ms <= 600000:
        parser.error("invalid expected phone FFN timeout")
    if not 3200 <= args.expected_phone_vmem_mib <= 3328:
        parser.error("invalid expected phone VMEM")
    if args.expected_phone_min_available_kib < 2097152:
        parser.error("invalid expected phone memory reserve")
    if not 1 <= args.expected_qwen_mps_active_thread_pct <= 100:
        parser.error("invalid expected Qwen MPS active thread percentage")
    if not 1 <= args.expected_stage_mps_active_thread_pct <= 100:
        parser.error("invalid expected stage MPS active thread percentage")
    if args.expected_qwen_mps_client_priority not in {0, 1}:
        parser.error("invalid expected Qwen MPS client priority")
    if args.expected_stage_mps_client_priority not in {0, 1}:
        parser.error("invalid expected stage MPS client priority")
    required_paths = (
        args.qwen_result,
        args.phone_energy,
        args.gate_result,
        args.driver_result,
        args.profile,
        args.stage_manifest,
        args.bridge_log,
        args.router_log,
        args.phone_session_log,
        args.phone_workers_log,
        args.driver_log,
        args.worker_log,
        args.mps_receipt,
        args.qwen_affinity_receipt,
        args.wavefront_affinity_receipt,
        args.server_log,
        args.expected_qwen_server,
    )
    if any(not path.is_file() for path in required_paths):
        parser.error("analysis inputs are missing")
    if (
        not args.output.is_absolute()
        or args.output.exists()
        or not args.output.parent.is_dir()
    ):
        parser.error("output must be an unused absolute path")
    try:
        result = analyze(args)
    except (
        AnalysisError,
        MaterializeError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        parser.exit(2, f"GPU-prefix wavefront analysis failed: {exc}\n")
    args.output.write_text(
        json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps({
        "admission": result["admission"],
        "fleet_j": result["energy"]["fleet_j"],
        "paid_duration_s": result["paid_duration_s"],
        "prefill_overlap_us": result["gpu_prefix"]["prefill_overlap_us"],
        "status": result["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
