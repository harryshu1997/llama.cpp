#!/usr/bin/env python3
"""Independently validate one W9 treatment/control pair."""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
import hashlib
from pathlib import Path
from typing import Any, Sequence

import phone_cuda_delta_probe as w6
import phone_cuda_handoff_probe as w5
import phone_cuda_live_promotion_probe as live
import validate_cold_promotion as placement
import validate_phone_cuda_delta as physical
import w9_profiled_cutover as w9
from stage_v3_client import Hello, ProtocolError


SCHEMA = "s39-profiled-cutover-pair-certificate-v1"
HERE = Path(__file__).resolve().parent
W8_RUN = (
    HERE
    / "results"
    / "w8_live_promotion_r1"
    / "run_20260725T013802Z"
)
GPU_FIELDS = (
    "index",
    "name",
    "uuid",
    "pci.bus_id",
    "driver_version",
    "memory.total",
    "power.limit",
    "temperature.gpu",
    "pstate",
    "clocks.current.sm",
    "clocks.current.memory",
    "power.draw",
    "memory.used",
    "utilization.gpu",
)


def source_paths() -> dict[str, Path]:
    return {
        "async_pipeline.py": (
            HERE.parent / "s22_slo_overlap_pipeline" / "async_pipeline.py"
        ),
        "capture_phone_thermal.py": HERE / "capture_phone_thermal.py",
        "cuda_live_trace_control_probe.py": (
            HERE / "cuda_live_trace_control_probe.py"
        ),
        "cuda_profiled_trace_control_probe.py": (
            HERE / "cuda_profiled_trace_control_probe.py"
        ),
        "phone_cuda_cold_promotion_probe.py": (
            HERE / "phone_cuda_cold_promotion_probe.py"
        ),
        "phone_cuda_delta_probe.py": HERE / "phone_cuda_delta_probe.py",
        "phone_cuda_handoff_probe.py": HERE / "phone_cuda_handoff_probe.py",
        "phone_cuda_live_promotion_probe.py": (
            HERE / "phone_cuda_live_promotion_probe.py"
        ),
        "phone_cuda_profiled_cutover_probe.py": (
            HERE / "phone_cuda_profiled_cutover_probe.py"
        ),
        "qwen25_quality_probe.py": HERE / "qwen25_quality_probe.py",
        "run_w9_profiled_cutover_gate.py": (
            HERE / "run_w9_profiled_cutover_gate.py"
        ),
        "run_w9_profiled_cutover_gate.sh": (
            HERE / "run_w9_profiled_cutover_gate.sh"
        ),
        "stage_v3_client.py": (
            HERE.parent / "s22_slo_overlap_pipeline" / "stage_v3_client.py"
        ),
        "validate_cold_promotion.py": HERE / "validate_cold_promotion.py",
        "validate_phone_cuda_delta.py": (
            HERE / "validate_phone_cuda_delta.py"
        ),
        "validate_profiled_cutover_pair.py": Path(__file__).resolve(),
        "validate_profiled_cutover_series.py": (
            HERE / "validate_profiled_cutover_series.py"
        ),
        "w9_host_evidence.py": HERE / "w9_host_evidence.py",
        "w9_profiled_cutover.py": HERE / "w9_profiled_cutover.py",
    }


def read(path: Path, field: str) -> tuple[dict[str, object], bytes]:
    return w6.read_canonical(path, field)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            block = source.read(16 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def validate_w8_profile_source(contract: w9.CutoverContract) -> None:
    report_path = W8_RUN / "treatment_report.json"
    manifest_path = W8_RUN / "SHA256SUMS.txt"
    w9.require(
        report_path.is_file()
        and manifest_path.is_file()
        and file_sha256(report_path) == contract.w8_treatment_report_sha256
        and file_sha256(manifest_path) == contract.w8_manifest_sha256,
        "pair: W8 profile source identity",
    )
    seen = set()
    for index, line in enumerate(
        manifest_path.read_text(encoding="ascii").splitlines()
    ):
        fields = line.split("  ", 1)
        w9.require(len(fields) == 2, f"pair: W8 manifest line {index}")
        digest, relative = fields
        w9.checked_digest(digest, f"pair: W8 manifest digest {index}")
        path = Path(relative)
        w9.require(
            relative not in seen
            and relative
            and not path.is_absolute()
            and ".." not in path.parts,
            f"pair: W8 manifest path {index}",
        )
        seen.add(relative)
        artifact = W8_RUN / path
        w9.require(
            artifact.is_file() and file_sha256(artifact) == digest,
            f"pair: W8 manifest artifact {relative}",
        )
    w9.require("treatment_report.json" in seen, "pair: W8 report not manifested")
    report, _ = read(report_path, "w8_profile_source")
    w9.require(
        report.get("schema") == "s39-live-session-promotion-treatment-v1"
        and report.get("status") == "LIVE_SESSION_PROMOTION_PASS"
        and report.get("scope") == "MECHANICS_ONLY"
        and report.get("scheduler_eligible") is False
        and report.get("contract_sha256") == contract.w8_contract_sha256,
        "pair: W8 profile source status",
    )
    third = report["metrics"]["phone_service"]["batch_timeline"][2]
    observed = {
        "metrics.cuda_replay.elapsed_us": report["metrics"]["cuda_replay"][
            "elapsed_us"
        ],
        (
            "metrics.phone_service.batch_timeline[2]."
            "(ended_ns - started_ns) / 1000"
        ): (third["ended_ns"] - third["started_ns"] + 999) // 1000,
        "metrics.cuda_delta.elapsed_us / 2": (
            report["metrics"]["cuda_delta"]["elapsed_us"] + 1
        )
        // 2,
        "metrics.cuda_continuation.elapsed_us / 8": (
            report["metrics"]["cuda_continuation"]["elapsed_us"] + 7
        )
        // 8,
    }
    bindings = {
        item["field"]: item
        for item in contract.reference_source_bindings
    }
    w9.require(
        len(bindings) == len(contract.reference_source_bindings),
        "pair: duplicate predictor binding",
    )
    for field, value in observed.items():
        binding = bindings.get(field)
        w9.require(
            type(binding) is dict
            and binding["source_kind"] == "W8_ARTIFACT"
            and binding["artifact_sha256"]
            == contract.w8_treatment_report_sha256
            and w9.is_int(value)
            and value >= 0
            and value <= binding["value_us"],
            f"pair: W8 predictor bound {field}",
        )
    prospective = bindings.get("predicted_commit_us")
    w9.require(
        type(prospective) is dict
        and prospective["source_kind"] == "PROSPECTIVE_BOUND"
        and prospective["artifact_sha256"] == w9.ZERO_SHA256
        and prospective["value_us"] == contract.predicted_commit_us,
        "pair: prospective commit bound",
    )


def raw_integer(value: str, field: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise w9.W9Error(f"{field}: invalid integer") from exc
    w9.require(result >= 0, f"{field}: negative integer")
    return result


def raw_milli(value: str, field: str) -> int:
    try:
        result = int(Decimal(value) * 1000)
    except (InvalidOperation, ValueError) as exc:
        raise w9.W9Error(f"{field}: invalid decimal") from exc
    w9.require(result >= 0, f"{field}: negative decimal")
    return result


def parse_raw_gpu(value: object, field: str) -> dict[str, object]:
    w9.require(type(value) is str, f"{field}: expected text")
    rows = list(csv.reader(value.splitlines()))
    w9.require(
        len(rows) == 1 and len(rows[0]) == len(GPU_FIELDS),
        f"{field}: shape",
    )
    source = dict(zip(GPU_FIELDS, (item.strip() for item in rows[0])))
    return {
        "clock_memory_mhz": raw_integer(
            source["clocks.current.memory"],
            f"{field}.clock_memory",
        ),
        "clock_sm_mhz": raw_integer(
            source["clocks.current.sm"],
            f"{field}.clock_sm",
        ),
        "driver_version": source["driver_version"],
        "index": raw_integer(source["index"], f"{field}.index"),
        "memory_total_mib": raw_integer(
            source["memory.total"],
            f"{field}.memory_total",
        ),
        "memory_used_mib": raw_integer(
            source["memory.used"],
            f"{field}.memory_used",
        ),
        "name": source["name"],
        "pci_bus_id": source["pci.bus_id"],
        "power_limit_mw": raw_milli(
            source["power.limit"],
            f"{field}.power_limit",
        ),
        "power_mw": raw_milli(source["power.draw"], f"{field}.power"),
        "pstate": source["pstate"],
        "temperature_c": raw_integer(
            source["temperature.gpu"],
            f"{field}.temperature",
        ),
        "utilization_gpu_pct": raw_integer(
            source["utilization.gpu"],
            f"{field}.utilization",
        ),
        "uuid": source["uuid"],
    }


def parse_raw_processes(value: object, field: str) -> list[dict[str, object]]:
    w9.require(type(value) is str, f"{field}: expected text")
    result = []
    for index, row in enumerate(csv.reader(value.splitlines())):
        if not row:
            continue
        w9.require(len(row) == 4, f"{field}.{index}: shape")
        uuid, pid, name, used = (item.strip() for item in row)
        w9.require(bool(name), f"{field}.{index}: process name")
        result.append({
            "gpu_uuid": uuid,
            "pid": raw_integer(pid, f"{field}.{index}.pid"),
            "process_name": name,
            "used_memory_mib": raw_integer(
                used,
                f"{field}.{index}.memory",
            ),
        })
    return result


def load_dependencies(
    contract_path: Path,
    w8_contract_path: Path,
    base_contract_path: Path,
    delta_contract_path: Path,
    physical_gate_path: Path,
) -> tuple[
    w9.CutoverContract,
    live.LiveContract,
    w5.Contract,
    w6.DeltaContract,
    physical.PhysicalGate,
]:
    base = w5.load_contract(base_contract_path)
    delta = w6.load_contract(delta_contract_path, base)
    gate = physical.load_physical_gate(physical_gate_path, delta, base)
    w8 = live.load_contract(
        w8_contract_path,
        base,
        delta,
        gate.raw_sha256,
    )
    contract = w9.load_contract(contract_path)
    validate_w8_profile_source(contract)
    w9.require(
        contract.w8_contract_sha256 == w8.raw_sha256
        and contract.base_contract_sha256 == base.raw_sha256
        and contract.delta_contract_sha256 == delta.raw_sha256
        and contract.physical_gate_sha256 == gate.raw_sha256,
        "pair: dependency mismatch",
    )
    paths = source_paths()
    w9.require(set(paths) == set(contract.source_sha256), "pair: source set")
    for name, path in paths.items():
        w9.require(
            path.is_file()
            and w6.sha256(path.read_bytes()) == contract.source_sha256[name],
            f"pair: source mismatch for {name}",
        )
    return contract, w8, base, delta, gate


def validate_context(
    value: object,
    *,
    contract: w9.CutoverContract,
    base: w5.Contract,
    gate: physical.PhysicalGate,
    pair: str,
) -> dict[str, object]:
    root = w9.exact_keys(
        value,
        {
            "acquisition_unix_s",
            "artifacts",
            "base_contract_sha256",
            "base_git_commit",
            "contract_sha256",
            "cuda",
            "delta_contract_sha256",
            "model_sha256",
            "op12",
            "op15",
            "pair_ordinal",
            "physical_gate_sha256",
            "run_id",
            "schema",
            "sources",
            "w8_contract_sha256",
        },
        "pair_context",
    )
    w9.require(
        root["schema"] == "s39-profiled-cutover-pair-context-v1"
        and root["pair_ordinal"] == pair
        and pair in contract.pair_ordinals
        and w9.is_int(root["acquisition_unix_s"])
        and root["acquisition_unix_s"] > 0
        and type(root["base_git_commit"]) is str
        and len(root["base_git_commit"]) == 40
        and root["contract_sha256"] == contract.raw_sha256
        and root["w8_contract_sha256"] == contract.w8_contract_sha256
        and root["base_contract_sha256"] == base.raw_sha256
        and root["delta_contract_sha256"] == contract.delta_contract_sha256
        and root["physical_gate_sha256"] == gate.raw_sha256
        and root["model_sha256"] == base.model_sha256
        and root["sources"] == contract.source_sha256,
        "pair_context: identity",
    )
    w9.checked_digest(root["run_id"], "pair_context.run_id")
    artifacts = w9.exact_keys(
        root["artifacts"],
        {
            "host_relay_sha256",
            "host_worker_sha256",
            "model_path",
            "model_sha256",
            "op12_shard_sha256",
            "op15_relay_sha256",
            "op15_shard_sha256",
            "phone_worker_sha256",
        },
        "pair_context.artifacts",
    )
    w9.require(
        artifacts["model_sha256"] == gate.artifacts["model_sha256"]
        and artifacts["host_relay_sha256"]
        == gate.artifacts["host_relay_sha256"]
        and artifacts["host_worker_sha256"]
        == gate.artifacts["host_worker_sha256"]
        and artifacts["op12_shard_sha256"]
        == gate.artifacts["op12_shard_sha256"]
        and artifacts["op15_shard_sha256"]
        == gate.artifacts["op15_shard_sha256"]
        and artifacts["op15_relay_sha256"]
        == gate.artifacts["op15_relay_sha256"]
        and artifacts["phone_worker_sha256"]
        == gate.artifacts["phone_worker_sha256"]
        and type(artifacts["model_path"]) is str
        and artifacts["model_path"],
        "pair_context: artifacts",
    )
    cuda = w9.exact_keys(
        root["cuda"],
        {
            "boot_id",
            "cuda_device_order",
            "cuda_visible_devices",
            "driver_version",
            "logical_device",
            "memory_total_mib",
            "name",
            "pci_bus_id",
            "physical_index",
            "power_limit_mw",
            "uuid",
        },
        "pair_context.cuda",
    )
    w9.require(
        cuda["cuda_device_order"] == "PCI_BUS_ID"
        and cuda["cuda_visible_devices"] == cuda["uuid"]
        and cuda["logical_device"] == "CUDA0"
        and w9.is_int(cuda["physical_index"])
        and cuda["physical_index"] >= 0
        and cuda["uuid"] == contract.cuda_uuid
        and cuda["name"] == contract.cuda_name
        and type(cuda["uuid"]) is str
        and cuda["uuid"].startswith("GPU-")
        and type(cuda["boot_id"]) is str
        and cuda["boot_id"]
        and type(cuda["name"]) is str
        and cuda["name"]
        and type(cuda["pci_bus_id"]) is str
        and cuda["pci_bus_id"]
        and type(cuda["driver_version"]) is str
        and cuda["driver_version"]
        and w9.is_int(cuda["memory_total_mib"])
        and cuda["memory_total_mib"] > 0
        and w9.is_int(cuda["power_limit_mw"])
        and cuda["power_limit_mw"] > 0,
        "pair_context: CUDA identity",
    )
    for name, layers in (("op15", [0, 30]), ("op12", [30, 48])):
        device = w9.exact_keys(
            root[name],
            {
                "adb_target",
                "boot_id",
                "layers",
                "shard_sha256",
                "wifi",
                "worker_sha256",
            }
            | ({"relay_sha256"} if name == "op15" else set()),
            f"pair_context.{name}",
        )
        w9.require(
            device["layers"] == layers
            and device["adb_target"] == getattr(contract, f"{name}_serial")
            and device["wifi"] == getattr(contract, f"{name}_wifi")
            and type(device["adb_target"]) is str
            and device["adb_target"]
            and type(device["boot_id"]) is str
            and device["boot_id"]
            and type(device["wifi"]) is str
            and device["wifi"]
            and device["worker_sha256"]
            == gate.artifacts["phone_worker_sha256"]
            and device["shard_sha256"]
            == gate.artifacts[f"{name}_shard_sha256"],
            f"pair_context: {name}",
        )
        if name == "op15":
            w9.require(
                device["relay_sha256"]
                == gate.artifacts["op15_relay_sha256"],
                "pair_context: op15 relay",
            )
    return root


def validate_launch(
    path: Path,
    *,
    phase: str,
    pair: str,
    run_id: str,
    contract: w9.CutoverContract,
) -> tuple[dict[str, object], bytes]:
    value, raw = read(path, f"{phase.lower()}_launch")
    root = w9.exact_keys(
        value,
        {
            "cuda_launch_ns",
            "launch_deadline_ns",
            "pair_ordinal",
            "phase",
            "preexisting_route_pids",
            "request_start_ns",
            "run_id",
            "schema",
        },
        f"{phase.lower()}_launch",
    )
    delay = root["cuda_launch_ns"] - root["request_start_ns"]
    w9.require(
        root["schema"] == "s39-profiled-cutover-launch-v1"
        and root["phase"] == phase
        and root["pair_ordinal"] == pair
        and root["run_id"] == run_id
        and root["preexisting_route_pids"] == []
        and all(
            w9.is_int(root[name]) and root[name] > 0
            for name in (
                "cuda_launch_ns",
                "launch_deadline_ns",
                "request_start_ns",
            )
        )
        and root["launch_deadline_ns"]
        == root["request_start_ns"] + 100_000_000
        and root["cuda_launch_ns"] >= root["launch_deadline_ns"]
        and delay <= contract.max_launch_delay_us * 1000,
        f"{phase}: launch timing",
    )
    return root, raw


def validate_page_cache(
    path: Path,
    *,
    label: str,
    model_sha256: str,
    request_start_ns: int,
) -> dict[str, object]:
    value, _ = read(path, f"page_cache.{label}")
    root = w9.exact_keys(
        value,
        {"label", "payload", "schema", "status"},
        f"page_cache.{label}",
    )
    payload = w9.exact_keys(
        root["payload"],
        {
            "bytes_read",
            "ended_ns",
            "file_sha256",
            "path",
            "size_bytes",
            "started_ns",
        },
        f"page_cache.{label}.payload",
    )
    w9.require(
        root["schema"] == "s39-page-cache-precondition-v1"
        and root["status"] == "PASS"
        and root["label"] == label
        and payload["file_sha256"] == model_sha256
        and payload["bytes_read"] == payload["size_bytes"]
        and w9.is_int(payload["bytes_read"])
        and payload["bytes_read"] > 0
        and w9.is_int(payload["started_ns"])
        and w9.is_int(payload["ended_ns"])
        and 0 < payload["started_ns"] < payload["ended_ns"] <= request_start_ns
        and type(payload["path"]) is str
        and payload["path"],
        f"page_cache.{label}: invalid evidence",
    )
    return payload


def validate_gpu_bracket(
    path: Path,
    *,
    label: str,
    context: dict[str, object],
    contract: w9.CutoverContract,
    require_idle: bool,
) -> dict[str, object]:
    value, _ = read(path, f"gpu.{label}")
    root = w9.exact_keys(
        value,
        {"label", "payload", "schema", "status"},
        f"gpu.{label}",
    )
    payload = w9.exact_keys(
        root["payload"],
        {
            "cuda_device_order",
            "cuda_visible_devices",
            "ended_ns",
            "host_boot_id",
            "identity",
            "idle",
            "require_idle",
            "samples",
            "started_ns",
        },
        f"gpu.{label}.payload",
    )
    w9.require(
        root["schema"] == "s39-selected-gpu-bracket-v1"
        and root["status"] == "PASS"
        and root["label"] == label
        and type(payload) is dict,
        f"gpu.{label}: labels",
    )
    identity = payload.get("identity")
    w9.require(
        type(identity) is dict
        and all(
            identity.get(key) == context["cuda"][key]
            for key in (
                "driver_version",
                "memory_total_mib",
                "name",
                "pci_bus_id",
                "power_limit_mw",
                "uuid",
            )
        )
        and payload.get("host_boot_id") == context["cuda"]["boot_id"]
        and payload.get("cuda_device_order") == "PCI_BUS_ID"
        and payload.get("cuda_visible_devices") == context["cuda"]["uuid"]
        and payload.get("require_idle") is require_idle,
        f"gpu.{label}: identity",
    )
    w9.require(
        identity.get("index") == context["cuda"]["physical_index"],
        f"gpu.{label}: physical CUDA index",
    )
    samples = payload.get("samples")
    w9.require(type(samples) is list and bool(samples), f"gpu.{label}: samples")
    recomputed_idle = True
    previous_completed = 0
    for index, sample_value in enumerate(samples):
        sample = w9.exact_keys(
            sample_value,
            {
                "completed_ns",
                "compute_processes",
                "gpu",
                "raw_compute_processes_csv",
                "raw_gpu_query_csv",
                "sample_index",
                "started_ns",
            },
            f"gpu.{label}.samples.{index}",
        )
        raw_gpu = parse_raw_gpu(
            sample["raw_gpu_query_csv"],
            f"gpu.{label}.samples.{index}.raw_gpu",
        )
        raw_processes = parse_raw_processes(
            sample["raw_compute_processes_csv"],
            f"gpu.{label}.samples.{index}.raw_processes",
        )
        w9.require(
            sample["sample_index"] == index
            and w9.is_int(sample["started_ns"])
            and w9.is_int(sample["completed_ns"])
            and payload["started_ns"] <= sample["started_ns"]
            < sample["completed_ns"] <= payload["ended_ns"]
            and sample["started_ns"] >= previous_completed
            and sample["gpu"] == raw_gpu
            and sample["compute_processes"] == raw_processes
            and all(
                process["gpu_uuid"] == context["cuda"]["uuid"]
                for process in raw_processes
            )
            and all(
                sample["gpu"][key] == expected
                for key, expected in identity.items()
            ),
            f"gpu.{label}: raw sample mismatch",
        )
        previous_completed = sample["completed_ns"]
        recomputed_idle = recomputed_idle and (
            sample["gpu"]["utilization_gpu_pct"] == 0
            and not sample["compute_processes"]
        )
    w9.require(
        w9.is_int(payload["started_ns"])
        and w9.is_int(payload["ended_ns"])
        and 0 < payload["started_ns"] < payload["ended_ns"]
        and payload["idle"] is recomputed_idle,
        f"gpu.{label}: bracket timing or idle",
    )
    if require_idle:
        w9.require(
            len(samples) >= contract.idle_samples
            and payload.get("idle") is True
            and payload["ended_ns"] - payload["started_ns"]
            >= contract.idle_span_us * 1000
            and all(
                sample.get("compute_processes") == []
                and sample.get("gpu", {}).get("utilization_gpu_pct") == 0
                for sample in samples
            ),
            f"gpu.{label}: idle gate",
        )
    return payload


def validate_phone_thermal(
    path: Path,
    *,
    label: str,
    context: dict[str, object],
) -> dict[str, int]:
    value, _ = read(path, f"thermal.{label}")
    root = w9.exact_keys(
        value,
        {"captured_utc_ns", "label", "samples", "schema"},
        f"thermal.{label}",
    )
    w9.require(
        root["schema"] == "s39-phone-thermal-bracket-v1"
        and root["label"] == label
        and w9.is_int(root["captured_utc_ns"])
        and root["captured_utc_ns"] > 0,
        f"thermal.{label}: labels",
    )
    samples = w9.exact_keys(
        root["samples"],
        {"op12", "op15"},
        f"thermal.{label}.samples",
    )
    result = {}
    for name, sample in samples.items():
        item = w9.exact_keys(
            sample,
            {
                "device_boot_id",
                "gpu_max_millic",
                "gpu_zones_millic",
                "serial",
            },
            f"thermal.{label}.{name}",
        )
        zones = item["gpu_zones_millic"]
        w9.require(
            item["device_boot_id"] == context[name]["boot_id"]
            and item["serial"] == context[name]["adb_target"]
            and type(zones) is dict
            and bool(zones)
            and all(
                type(zone) is str
                and zone
                and w9.is_int(temp)
                and 0 < temp < 200000
                for zone, temp in zones.items()
            )
            and item["gpu_max_millic"] == max(zones.values()),
            f"thermal.{label}.{name}: invalid",
        )
        result[name] = item["gpu_max_millic"]
    return result


def validate_markers(
    pair_dir: Path,
    *,
    pair: str,
    run_id: str,
    contract: w9.CutoverContract,
    treatment: dict[str, object],
    control: dict[str, object],
) -> dict[str, str]:
    paths = {
        "treatment_prepaid_ready": pair_dir / "treatment/prepaid_ready.json",
        "treatment_paid_permit": pair_dir / "treatment/paid_permit.json",
        "treatment_start": pair_dir / "treatment/start.json",
        "treatment_ready": pair_dir / "treatment/ready.json",
        "control_start": pair_dir / "control/start.json",
        "control_ready": pair_dir / "control/ready.json",
    }
    loaded = {
        name: read(path, f"markers.{name}")
        for name, path in paths.items()
    }
    prepaid = w9.exact_keys(
        loaded["treatment_prepaid_ready"][0],
        {
            "pair_ordinal",
            "phone_state_count",
            "preexisting_ended_ns",
            "process_pid",
            "run_id",
            "schema",
        },
        "markers.treatment_prepaid_ready",
    )
    permit = w9.exact_keys(
        loaded["treatment_paid_permit"][0],
        {"pair_ordinal", "run_id", "schema"},
        "markers.treatment_paid_permit",
    )
    treatment_start = w9.exact_keys(
        loaded["treatment_start"][0],
        {
            "launch_deadline_ns",
            "pair_ordinal",
            "phone_state_count",
            "preexisting_ended_ns",
            "process_pid",
            "request_start_ns",
            "run_id",
            "schema",
        },
        "markers.treatment_start",
    )
    treatment_ready = w9.exact_keys(
        loaded["treatment_ready"][0],
        {
            "cuda_ready_ns",
            "f0_snapshot_ns",
            "pair_ordinal",
            "process_pid",
            "run_id",
            "schema",
        },
        "markers.treatment_ready",
    )
    control_start = w9.exact_keys(
        loaded["control_start"][0],
        {
            "launch_deadline_ns",
            "pair_ordinal",
            "request_start_ns",
            "run_id",
            "schema",
        },
        "markers.control_start",
    )
    control_ready = w9.exact_keys(
        loaded["control_ready"][0],
        {
            "cuda_ready_ns",
            "pair_ordinal",
            "process_pid",
            "run_id",
            "schema",
        },
        "markers.control_ready",
    )
    treatment_pid = treatment_start["process_pid"]
    w9.require(
        prepaid["schema"] == "s39-profiled-cutover-prepaid-ready-v1"
        and permit
        == {
            "pair_ordinal": pair,
            "run_id": run_id,
            "schema": "s39-profiled-cutover-paid-permit-v1",
        }
        and treatment_start["schema"] == "s39-profiled-cutover-start-v1"
        and treatment_ready["schema"] == "s39-profiled-cutover-cuda-ready-v1"
        and control_start["schema"] == "s39-profiled-cutover-control-start-v1"
        and control_ready["schema"]
        == "s39-profiled-cutover-control-ready-v1"
        and all(
            item["pair_ordinal"] == pair and item["run_id"] == run_id
            for item in (
                prepaid,
                treatment_start,
                treatment_ready,
                control_start,
                control_ready,
            )
        )
        and w9.is_int(treatment_pid)
        and treatment_pid > 0
        and prepaid["process_pid"] == treatment_pid
        and treatment_ready["process_pid"] == treatment_pid
        and w9.is_int(control_ready["process_pid"])
        and control_ready["process_pid"] > 0
        and prepaid["phone_state_count"] == contract.batch
        and treatment_start["phone_state_count"] == contract.batch
        and prepaid["preexisting_ended_ns"]
        == treatment["preexisting"]["ended_ns"]
        == treatment_start["preexisting_ended_ns"]
        and treatment["preexisting"]["started_ns"]
        < prepaid["preexisting_ended_ns"]
        <= treatment_start["request_start_ns"]
        and treatment_start["request_start_ns"]
        == treatment["timing"]["request_start_ns"]
        and treatment_start["launch_deadline_ns"]
        == treatment_start["request_start_ns"] + 100_000_000
        and treatment_ready["cuda_ready_ns"]
        == treatment["cuda_ready"]["cuda_ready_ns"]
        and treatment_ready["f0_snapshot_ns"]
        == treatment["cuda_ready"]["f0_snapshot_ns"]
        and control_start["request_start_ns"] == control["request_start_ns"]
        and control_start["launch_deadline_ns"]
        == control_start["request_start_ns"] + 100_000_000
        and control_ready["cuda_ready_ns"] == control["cuda_ready_ns"],
        "markers: identity or causal binding",
    )
    return {
        name: w6.sha256(raw)
        for name, (_, raw) in loaded.items()
    }


def validate_hello(value: object, base: w5.Contract, field: str) -> None:
    w9.require(type(value) is dict, f"{field}: hello")
    hello = Hello(**value)
    w5.validate_hello(field, hello, base)


def independent_decision(
    contract: w9.CutoverContract,
    report: dict[str, object],
) -> dict[str, object]:
    decision = report["decision"]
    f0_count = len(report["sequences"][0]["phone_service"])
    inflight = report["frontiers"]["d_inflight"] == 1
    elapsed = decision["inflight_elapsed_us"]
    remaining = (
        max(0, contract.phone_batch_estimate_us - elapsed)
        if inflight
        else 0
    )
    k_max = max(
        0,
        contract.output_tokens
        - f0_count
        - contract.max_inflight_tokens
        - contract.min_cuda_continuation_tokens,
    )
    candidates = []
    baseline = None
    selected = 0
    for k in range(k_max + 1):
        delta = contract.max_inflight_tokens + k
        cuda_tokens = contract.output_tokens - f0_count - delta
        phone_side = remaining + contract.phone_extra_us[k]
        cutover = (
            max(contract.cuda_replay_us, phone_side)
            + contract.delta_ingest_us[delta]
            + contract.predicted_commit_us
        )
        completion = cutover + contract.cuda_tokens_us[cuda_tokens]
        if baseline is None:
            baseline = completion
        feasible = (
            k == 0
            or (
                phone_side
                <= contract.cuda_replay_us - contract.cutover_margin_us
                and completion <= baseline
            )
        )
        if feasible:
            selected = k
        candidates.append({
            "completion_us": completion,
            "cuda_remaining_tokens": cuda_tokens,
            "cutover_us": cutover,
            "feasible": feasible,
            "k": k,
            "phone_side_us": phone_side,
            "predicted_delta_tokens": delta,
        })
    rebuilt = {
        "candidates": candidates,
        "inflight_elapsed_us": elapsed,
        "inflight_present": inflight,
        "k_extra": selected,
        "k_max": k_max,
        "phone_tokens_at_f0": f0_count,
        "predicted_inflight_remaining_us": remaining,
    }
    w9.require(decision == rebuilt and selected == 0, "pair: cutover decision")
    return rebuilt


def validate_treatment(
    value: dict[str, object],
    *,
    contract: w9.CutoverContract,
    base: w5.Contract,
    run_id: str,
    pair: str,
    ledger_dir: Path,
) -> tuple[dict[str, object], list[w9.LedgerRecord]]:
    root = w9.exact_keys(
        value,
        {
            "base_contract_sha256",
            "batch",
            "contract_sha256",
            "cuda_ready",
            "decision",
            "delta_contract_sha256",
            "frontiers",
            "hellos",
            "ledger",
            "metrics",
            "model_sha256",
            "ownership",
            "pair_ordinal",
            "physical_gate_sha256",
            "preexisting",
            "prompts",
            "run_id",
            "scheduler_eligible",
            "schema",
            "scope",
            "sequences",
            "state_counts",
            "status",
            "timing",
            "transaction_id",
            "w8_contract_sha256",
        },
        "treatment",
    )
    w9.require(
        root["schema"] == "s39-profiled-cutover-treatment-v1"
        and root["status"] == "PROFILED_ZERO_EXTRA_TREATMENT_PASS"
        and root["scope"] == "MECHANICS_ONLY"
        and root["scheduler_eligible"] is False
        and root["contract_sha256"] == contract.raw_sha256
        and root["w8_contract_sha256"] == contract.w8_contract_sha256
        and root["base_contract_sha256"] == base.raw_sha256
        and root["delta_contract_sha256"] == contract.delta_contract_sha256
        and root["physical_gate_sha256"] == contract.physical_gate_sha256
        and root["model_sha256"] == base.model_sha256
        and root["batch"] == contract.batch
        and root["prompts"] == list(base.prompt_ids)
        and root["run_id"] == run_id
        and root["pair_ordinal"] == pair,
        "treatment: identity",
    )
    validate_hello(root["hellos"]["phone"], base, "treatment.phone")
    validate_hello(root["hellos"]["cuda"], base, "treatment.cuda")
    sequences = root["sequences"]
    w9.require(
        type(sequences) is list and len(sequences) == contract.batch,
        "treatment: sequence count",
    )
    service_widths = set()
    inflight_widths = set()
    continuation_widths = set()
    f0_histories = []
    f1_histories = []
    final_histories = []
    publication_by_request: dict[int, list[tuple[int, str, int, int]]] = {}
    for index, sequence in enumerate(sequences):
        w9.require(
            type(sequence) is dict
            and sequence["sequence_index"] == index
            and sequence["prompt_id"] == base.prompt_ids[index],
            f"treatment: sequence identity {index}",
        )
        fields = (
            "prompt_tokens",
            "preexisting_tokens",
            "phone_service",
            "inflight_phone_tokens",
            "cuda_continuation",
            "cuda_continuation_control",
            "extra_phone_tokens",
            "final_published_tokens",
        )
        for field in fields:
            tokens = sequence.get(field)
            w9.require(
                type(tokens) is list
                and all(w9.is_int(token) and token >= 0 for token in tokens),
                f"treatment: {field} {index}",
            )
        w9.require(
            len(sequence["prompt_tokens"]) == base.prompt_tokens
            and len(sequence["preexisting_tokens"])
            == contract.preexisting_committed_tokens
            and sequence["extra_phone_tokens"] == []
            and sequence["cuda_continuation"]
            == sequence["cuda_continuation_control"],
            f"treatment: fixed widths {index}",
        )
        service_widths.add(len(sequence["phone_service"]))
        inflight_widths.add(len(sequence["inflight_phone_tokens"]))
        continuation_widths.add(len(sequence["cuda_continuation"]))
        post = (
            sequence["phone_service"]
            + sequence["inflight_phone_tokens"]
            + sequence["cuda_continuation"]
        )
        w9.require(
            len(post) == contract.output_tokens
            and sequence["final_published_tokens"]
            == sequence["preexisting_tokens"] + post,
            f"treatment: conservation {index}",
        )
        f0 = (
            sequence["prompt_tokens"]
            + sequence["preexisting_tokens"]
            + sequence["phone_service"]
        )
        f1 = f0 + sequence["inflight_phone_tokens"]
        f0_histories.append(f0)
        f1_histories.append(f1)
        final_histories.append(f1 + sequence["cuda_continuation"])
        phone_times = sequence["phone_service_publication_ns"]
        inflight_times = sequence["inflight_publication_ns"]
        cuda_times = sequence["cuda_publication_ns"]
        w9.require(
            type(phone_times) is list
            and type(inflight_times) is list
            and type(cuda_times) is list
            and len(phone_times) == len(sequence["phone_service"])
            and len(inflight_times) == len(sequence["inflight_phone_tokens"])
            and len(cuda_times) == len(sequence["cuda_continuation"])
            and all(
                w9.is_int(timestamp) and timestamp > 0
                for timestamp in phone_times + inflight_times + cuda_times
            ),
            f"treatment: publication times {index}",
        )
        combined = phone_times + inflight_times + cuda_times
        w9.require(
            combined == sorted(combined),
            f"treatment: publication order {index}",
        )
        boundaries = [root["timing"]["request_start_ns"]] + combined
        maximum = max(
            (right - left) // 1000
            for left, right in zip(boundaries, boundaries[1:])
        )
        w9.require(
            sequence["max_inter_token_gap_us"] == maximum,
            f"treatment: inter-token gap {index}",
        )
        first_position = (
            base.prompt_tokens + contract.preexisting_committed_tokens
        )
        publication_by_request[base.prompt_ids[index]] = [
            (timestamp, "PHONE_F0", first_position + offset, token)
            for offset, (timestamp, token) in enumerate(zip(
                phone_times,
                sequence["phone_service"],
            ))
        ] + [
            (
                timestamp,
                "PHONE_INFLIGHT",
                first_position + len(phone_times) + offset,
                token,
            )
            for offset, (timestamp, token) in enumerate(zip(
                inflight_times,
                sequence["inflight_phone_tokens"],
            ))
        ] + [
            (
                timestamp,
                "CUDA_CONTINUATION",
                first_position + len(phone_times) + len(inflight_times) + offset,
                token,
            )
            for offset, (timestamp, token) in enumerate(zip(
                cuda_times,
                sequence["cuda_continuation"],
            ))
        ]
    w9.require(
        len(service_widths) == len(inflight_widths) == len(continuation_widths) == 1,
        "treatment: nonrectangular tokens",
    )
    service = next(iter(service_widths))
    d_actual = next(iter(inflight_widths))
    continuation = next(iter(continuation_widths))
    frontiers = root["frontiers"]
    w9.require(
        d_actual in {0, 1}
        and frontiers["d_actual"] == d_actual
        and frontiers["d_inflight"] == d_actual
        and frontiers["k_extra"] == 0
        and frontiers["phone_tokens_at_f0"] == service
        and w9.is_int(frontiers["realized_inflight_remaining_us"])
        and frontiers["realized_inflight_remaining_us"] >= 0
        and frontiers["inflight_disposition"]
        == ("PUBLISHED" if d_actual else "ABSENT")
        and frontiers["f0_positions"]
        == [len(history) - 1 for history in f0_histories]
        and frontiers["f1_positions"]
        == [len(history) - 1 for history in f1_histories]
        and frontiers["f0_history_sha256"] == w5.histories_digest(f0_histories)
        and frontiers["f1_history_sha256"] == w5.histories_digest(f1_histories)
        and service + d_actual + continuation == contract.output_tokens
        and continuation >= contract.min_cuda_continuation_tokens,
        "treatment: frontier or budget",
    )
    phone_batches = root["metrics"]["phone_batches"]
    w9.require(
        type(phone_batches) is list
        and sum(
            batch.get("classification") == "F0"
            for batch in phone_batches
        )
        == service
        and sum(
            batch.get("classification") == "INFLIGHT"
            for batch in phone_batches
        )
        == d_actual,
        "treatment: phone batch classification",
    )
    w9.require(
        all(
            w9.is_int(batch.get("started_ns"))
            and w9.is_int(batch.get("ended_ns"))
            and root["timing"]["request_start_ns"]
            <= batch["started_ns"]
            < batch["ended_ns"]
            for batch in phone_batches
        ),
        "treatment: paid phone timing",
    )
    if d_actual:
        inflight_batch = next(
            batch
            for batch in phone_batches
            if batch["classification"] == "INFLIGHT"
        )
        w9.require(
            root["decision"]["inflight_elapsed_us"]
            == max(
                0,
                (
                    root["cuda_ready"]["f0_snapshot_ns"]
                    - inflight_batch["started_ns"]
                )
                // 1000,
            ),
            "treatment: in-flight elapsed accounting",
        )
        w9.require(
            frontiers["realized_inflight_remaining_us"]
            == max(
                0,
                (
                    inflight_batch["ended_ns"]
                    - root["cuda_ready"]["f0_snapshot_ns"]
                )
                // 1000,
            ),
            "treatment: realized in-flight residual",
        )
    else:
        w9.require(
            frontiers["realized_inflight_remaining_us"] == 0,
            "treatment: absent in-flight residual",
        )
    decision = independent_decision(contract, root)
    ready = root["cuda_ready"]
    ownership = root["ownership"]
    timing = root["timing"]
    all_phone_times = [
        timestamp
        for sequence in sequences
        for timestamp in (
            sequence["phone_service_publication_ns"]
            + sequence["inflight_publication_ns"]
        )
    ]
    all_cuda_times = [
        timestamp
        for sequence in sequences
        for timestamp in sequence["cuda_publication_ns"]
    ]
    w9.require(
        timing["new_request_ttft_us"] is None
        and timing["useful_pre_ready_phone_tokens"]
        == sum(
            timestamp < ready["cuda_ready_ns"]
            for timestamp in sequences[0]["phone_service_publication_ns"]
        )
        and timing["useful_pre_ready_phone_tokens"] >= 1
        and ready["request_start_ns"] == timing["request_start_ns"]
        and ready["request_start_ns"]
        < min(
            timestamp
            for sequence in sequences
            for timestamp in sequence["phone_service_publication_ns"]
        )
        < ready["cuda_ready_ns"]
        <= ready["f0_snapshot_ns"]
        <= ownership["cuda_replay_start_ns"]
        and ownership["cuda_replay_start_ns"] >= ready["cuda_ready_ns"]
        and ownership["ownership_commit_ns"] < min(all_cuda_times)
        and max(all_phone_times) < min(all_cuda_times)
        and timing["request_complete_ns"] == max(all_cuda_times)
        and timing["promotion_next_token_us"]
        == (
            min(
                timestamp
                for sequence in sequences
                for timestamp in sequence["phone_service_publication_ns"]
            )
            - timing["request_start_ns"]
        )
        // 1000
        and timing["completion_us"]
        == (timing["request_complete_ns"] - timing["request_start_ns"]) // 1000
        and timing["cuda_ready_us"]
        == (ready["cuda_ready_ns"] - timing["request_start_ns"]) // 1000
        and timing["maximum_inter_token_gap_us"]
        == max(sequence["max_inter_token_gap_us"] for sequence in sequences)
        and ownership["handoff_gap_us"]
        == (
            ownership["cuda_first_publication_ns"]
            - ownership["phone_last_publication_ns"]
        )
        // 1000
        and ownership["phone_last_publication_ns"] == max(all_phone_times)
        and ownership["cuda_first_publication_ns"] == min(all_cuda_times),
        "treatment: causal timing",
    )
    if d_actual:
        w9.require(
            ownership["inflight_overlap_ns"] > 0,
            "treatment: missing conditional overlap",
        )
    else:
        w9.require(
            ownership["inflight_overlap_ns"] is None,
            "treatment: absent overlap is not null",
        )
    w9.require(
        root["state_counts"]
        == {
            "cuda_after_completion": 0,
            "cuda_oracle_released": 0,
            "phone_after_commit": 0,
            "phone_before_promotion": contract.batch,
        },
        "treatment: terminal state",
    )

    tx_id = w9.transaction_id(contract, run_id, base.prompt_ids)
    w9.require(root["transaction_id"] == tx_id, "treatment: transaction")
    records = w9.load_ledger(
        ledger_dir,
        transaction_id=tx_id,
        run_id=run_id,
        request_ids=base.prompt_ids,
    )
    w9.require(root["ledger"] == w9.ledger_summary(records), "treatment: ledger")
    event_names = [record.value["event"] for record in records]
    for event in (
        "PAID_START",
        "CUDA_READY",
        "F0_SNAPSHOT",
        "CUDA_REPLAY_STARTED",
        "F1_ACK",
        "CUDA_CAUGHT_UP",
        "CUDA_COMMITTED",
        "CUDA_COMMIT_DURABLE",
        "PHONE_RELEASED",
        "COMPLETE",
    ):
        w9.require(event_names.count(event) == 1, f"ledger: {event}")
    order = [
        event_names.index(event)
        for event in (
            "PAID_START",
            "CUDA_READY",
            "F0_SNAPSHOT",
            "CUDA_REPLAY_STARTED",
            "F1_ACK",
            "CUDA_CAUGHT_UP",
            "CUDA_COMMITTED",
            "CUDA_COMMIT_DURABLE",
            "PHONE_RELEASED",
            "COMPLETE",
        )
    ]
    w9.require(order == sorted(order), "ledger: state order")
    commit_index = event_names.index("CUDA_COMMITTED")
    allowed_events = {
        "PAID_START",
        "TOKEN_PUBLISHED",
        "CUDA_READY",
        "F0_SNAPSHOT",
        "CUDA_REPLAY_STARTED",
        "F1_ACK",
        "CUDA_CAUGHT_UP",
        "CUDA_COMMITTED",
        "CUDA_COMMIT_DURABLE",
        "PHONE_RELEASED",
        "COMPLETE",
    }
    w9.require(
        set(event_names) <= allowed_events
        and event_names.count("TOKEN_PUBLISHED")
        == contract.batch * contract.output_tokens,
        "ledger: unexpected event or publication count",
    )
    system = {
        event: records[event_names.index(event)]
        for event in allowed_events
        if event != "TOKEN_PUBLISHED"
    }
    w9.require(
        system["PAID_START"].value["event_ns"] == timing["request_start_ns"]
        and system["PAID_START"].value["payload"]
        == {
            "owner": "PHONE",
            "owner_epoch": 1,
            "pair_ordinal": pair,
        }
        and system["CUDA_READY"].value["event_ns"]
        == ready["f0_snapshot_ns"]
        and system["CUDA_READY"].value["payload"]
        == {"cuda_ready_ns": ready["cuda_ready_ns"]}
        and system["F0_SNAPSHOT"].value["event_ns"]
        == ready["f0_snapshot_ns"]
        and system["F0_SNAPSHOT"].value["payload"]
        == {
            "history_sha256": frontiers["f0_history_sha256"],
            "positions": frontiers["f0_positions"],
            "snapshot_ns": ready["f0_snapshot_ns"],
        }
        and system["CUDA_REPLAY_STARTED"].value["event_ns"]
        == ownership["cuda_replay_start_ns"]
        and system["CUDA_REPLAY_STARTED"].value["payload"]
        == {
            "f0_history_sha256": frontiers["f0_history_sha256"],
            "replay_started_ns": ownership["cuda_replay_start_ns"],
        }
        and system["F1_ACK"].value["event_ns"] == frontiers["f1_ack_ns"]
        and system["F1_ACK"].value["payload"]
        == {
            "d_actual": d_actual,
            "disposition": frontiers["inflight_disposition"],
            "history_sha256": frontiers["f1_history_sha256"],
            "owner_epoch": 1,
            "positions": frontiers["f1_positions"],
        }
        and system["CUDA_CAUGHT_UP"].value["event_ns"]
        == ownership["cuda_catchup_ns"]
        and system["CUDA_CAUGHT_UP"].value["payload"]
        == {
            "delta_mode": "POSITIVE" if d_actual else "NOOP",
            "f1_history_sha256": frontiers["f1_history_sha256"],
            "positions": frontiers["f1_positions"],
        }
        and system["CUDA_COMMITTED"].value["payload"]
        == {
            "history_sha256": frontiers["f1_history_sha256"],
            "new_owner": "CUDA",
            "new_owner_epoch": 2,
            "old_owner_epoch": 1,
            "positions": frontiers["f1_positions"],
        }
        and system["CUDA_COMMITTED"].value["event_ns"]
        <= ownership["ownership_commit_ns"]
        and system["CUDA_COMMIT_DURABLE"].value["event_ns"]
        == ownership["ownership_commit_ns"]
        and system["CUDA_COMMIT_DURABLE"].value["payload"]
        == {
            "committed_record_sha256": system["CUDA_COMMITTED"].sha256,
            "durable_ns": ownership["ownership_commit_ns"],
        }
        and system["PHONE_RELEASED"].value["event_ns"]
        >= ownership["ownership_commit_ns"]
        and system["PHONE_RELEASED"].value["payload"]
        == {
            "old_owner_epoch": 1,
            "positions": frontiers["f1_positions"],
        }
        and system["COMPLETE"].value["event_ns"]
        >= timing["request_complete_ns"]
        and system["COMPLETE"].value["payload"]
        == {
            "final_history_sha256": w5.histories_digest(final_histories),
            "owner": "NONE",
            "post_start_tokens": contract.output_tokens,
        },
        "ledger: system event reconstruction",
    )
    ledger_tokens: dict[int, list[tuple[int, str, int, int]]] = {
        request: [] for request in base.prompt_ids
    }
    for index, record in enumerate(records):
        if record.value["event"] != "TOKEN_PUBLISHED":
            continue
        payload = record.value["payload"]
        w9.require(
            set(payload)
            == {
                "classification",
                "owner",
                "owner_epoch",
                "position",
                "request_id",
                "token",
            }
            and w9.is_int(payload.get("position"))
            and payload["position"] >= 0
            and w9.is_int(payload.get("token"))
            and payload["token"] >= 0,
            "ledger: malformed token publication",
        )
        request_id = payload.get("request_id")
        w9.require(request_id in ledger_tokens, "ledger: unknown request")
        owner = payload.get("owner")
        classification = payload.get("classification")
        w9.require(
            (
                owner == "PHONE"
                and payload.get("owner_epoch") == 1
                and classification in {"PHONE_F0", "PHONE_INFLIGHT"}
                and index < commit_index
            )
            or (
                owner == "CUDA"
                and payload.get("owner_epoch") == 2
                and classification == "CUDA_CONTINUATION"
                and index > commit_index
                and record.value["event_ns"]
                > ownership["ownership_commit_ns"]
            ),
            "ledger: ownership violation",
        )
        ledger_tokens[request_id].append(
            (
                record.value["event_ns"],
                classification,
                payload["position"],
                payload.get("token"),
            )
        )
    w9.require(
        ledger_tokens == publication_by_request,
        "ledger: publication reconstruction",
    )
    return decision, records


def validate_control(
    value: dict[str, object],
    *,
    contract: w9.CutoverContract,
    base: w5.Contract,
    treatment: dict[str, object],
    run_id: str,
    pair: str,
) -> None:
    root = value
    w9.require(
        root.get("schema") == "s39-profiled-cutover-trace-control-v1"
        and root.get("status") == "PROFILED_ZERO_EXTRA_TRACE_CONTROL_PASS"
        and root.get("scope") == "MECHANICS_ONLY"
        and root.get("scheduler_eligible") is False
        and root.get("contract_sha256") == contract.raw_sha256
        and root.get("base_contract_sha256") == base.raw_sha256
        and root.get("model_sha256") == base.model_sha256
        and root.get("run_id") == run_id
        and root.get("pair_ordinal") == pair
        and root.get("batch") == contract.batch
        and root.get("post_start_tokens") == contract.output_tokens
        and root.get("state_count") == 0,
        "control: identity",
    )
    validate_hello(root["hello"], base, "control.cuda")
    sequences = root["sequences"]
    w9.require(
        type(sequences) is list and len(sequences) == contract.batch,
        "control: sequence count",
    )
    matches = 0
    for index, (served, control) in enumerate(
        zip(treatment["sequences"], sequences)
    ):
        expected = (
            served["phone_service"]
            + served["inflight_phone_tokens"]
            + served["cuda_continuation"]
        )
        w9.require(
            control["sequence_index"] == index
            and control["prompt_id"] == base.prompt_ids[index]
            and control["prompt_tokens"] == served["prompt_tokens"]
            and control["preexisting_tokens"] == served["preexisting_tokens"]
            and control["replayed_tokens"] == expected
            and len(control["predicted_tokens"]) == contract.output_tokens,
            f"control: matched trace {index}",
        )
        matches += sum(
            left == right
            for left, right in zip(
                control["predicted_tokens"],
                control["replayed_tokens"],
            )
        )
    agreement = root["greedy_agreement"]
    w9.require(
        agreement
        == {
            "all_match": matches == contract.batch * contract.output_tokens,
            "matching_tokens": matches,
            "total_tokens": contract.batch * contract.output_tokens,
        },
        "control: agreement diagnostic",
    )
    metrics = root["metrics"]
    timeline = metrics["batch_timeline"]
    w9.require(
        metrics["history_batches"] == 5
        and metrics["continuation_batches"] == 12
        and metrics["rows"] == 176
        and len(timeline) == 17
        and len(metrics["token_ready_ns"]) == contract.output_tokens
        and timeline[0]["started_ns"] >= root["cuda_ready_ns"]
        and root["first_token_ns"] == metrics["token_ready_ns"][0]
        and root["request_complete_ns"] == metrics["token_ready_ns"][-1]
        and root["timing"]["new_request_ttft_us"] is None
        and root["timing"]["promotion_next_token_us"]
        == (root["first_token_ns"] - root["request_start_ns"]) // 1000
        and root["timing"]["completion_us"]
        == (root["request_complete_ns"] - root["request_start_ns"]) // 1000
        and root["timing"]["cuda_ready_us"]
        == (root["cuda_ready_ns"] - root["request_start_ns"]) // 1000,
        "control: timing or accounting",
    )


def validate_placements(
    pair_dir: Path,
    *,
    context: dict[str, object],
    gate: physical.PhysicalGate,
    treatment: dict[str, object],
    control: dict[str, object],
) -> dict[str, object]:
    pre_rows = treatment["metrics"]["preexisting"]["rows"]
    phone_rows = pre_rows + sum(
        batch["metrics"]["rows"]
        for batch in treatment["metrics"]["phone_batches"]
    )
    cuda_rows = (
        treatment["metrics"]["cuda_replay"]["metrics"]["rows"]
        + treatment["metrics"]["cuda_delta"]["rows"]
        + treatment["metrics"]["cuda_continuation"]["rows"]
        + treatment["metrics"]["cuda_oracle"]["rows"]
    )
    expected = {
        "op15": phone_rows,
        "op12": phone_rows,
        "treatment_cuda_head": cuda_rows,
        "treatment_cuda_tail": cuda_rows,
        "control_cuda_head": control["metrics"]["rows"],
        "control_cuda_tail": control["metrics"]["rows"],
    }
    paths = {
        "op15": pair_dir / "treatment/op15_head.log",
        "op12": pair_dir / "treatment/op12_tail.log",
        "treatment_cuda_head": pair_dir / "treatment/cuda_head.log",
        "treatment_cuda_tail": pair_dir / "treatment/cuda_tail.log",
        "control_cuda_head": pair_dir / "control/cuda_head.log",
        "control_cuda_tail": pair_dir / "control/cuda_tail.log",
    }
    result = {}
    for name, path in paths.items():
        certificates, log_sha = placement.parse_session_certificates(
            path,
            f"placement.{name}",
        )
        w9.require(
            len(certificates) == 1,
            f"placement.{name}: expected one session",
        )
        gate_name = (
            name
            if name in ("op12", "op15")
            else name.removeprefix("treatment_").removeprefix("control_")
        )
        boot = (
            context[name]["boot_id"]
            if name in ("op12", "op15")
            else context["cuda"]["boot_id"]
        )
        summary = placement.validate_session_certificate(
            certificates[0],
            field=f"placement.{name}",
            spec=gate.placements[gate_name],
            boot_id=boot,
            n_layer=gate.n_layer,
            session_id=1,
            session_end="STOP",
            reset_applied=False,
            steps_session=expected[name],
            steps_total=expected[name],
        )
        result[name] = {**summary, "log_sha256": log_sha}
    for role in ("head", "tail"):
        treatment_key = f"treatment_cuda_{role}"
        control_key = f"control_cuda_{role}"
        w9.require(
            result[treatment_key]["worker_pid"]
            != result[control_key]["worker_pid"]
            and result[treatment_key]["worker_boot_nonce"]
            != result[control_key]["worker_boot_nonce"],
            f"placement: reused CUDA {role}",
        )
    return result


def validate_pair(
    pair_dir: Path,
    *,
    contract_path: Path,
    w8_contract_path: Path,
    base_contract_path: Path,
    delta_contract_path: Path,
    physical_gate_path: Path,
) -> dict[str, object]:
    contract, _, base, _, gate = load_dependencies(
        contract_path,
        w8_contract_path,
        base_contract_path,
        delta_contract_path,
        physical_gate_path,
    )
    pair = pair_dir.name
    w9.require(pair in contract.pair_ordinals, "pair: ordinal")
    context, context_raw = read(pair_dir / "PAIR_CONTEXT.json", "pair_context")
    context = validate_context(
        context,
        contract=contract,
        base=base,
        gate=gate,
        pair=pair,
    )
    run_id = context["run_id"]
    treatment, treatment_raw = read(
        pair_dir / "treatment/report.json",
        "treatment",
    )
    control, control_raw = read(pair_dir / "control/report.json", "control")
    decision, ledger = validate_treatment(
        treatment,
        contract=contract,
        base=base,
        run_id=run_id,
        pair=pair,
        ledger_dir=pair_dir / "treatment/publication_ledger",
    )
    validate_control(
        control,
        contract=contract,
        base=base,
        treatment=treatment,
        run_id=run_id,
        pair=pair,
    )
    marker_sha256 = validate_markers(
        pair_dir,
        pair=pair,
        run_id=run_id,
        contract=contract,
        treatment=treatment,
        control=control,
    )
    treatment_launch, treatment_launch_raw = validate_launch(
        pair_dir / "treatment/launch.json",
        phase="TREATMENT",
        pair=pair,
        run_id=run_id,
        contract=contract,
    )
    control_launch, control_launch_raw = validate_launch(
        pair_dir / "control/launch.json",
        phase="CONTROL",
        pair=pair,
        run_id=run_id,
        contract=contract,
    )
    w9.require(
        treatment_launch["request_start_ns"]
        == treatment["timing"]["request_start_ns"]
        and control_launch["request_start_ns"] == control["request_start_ns"],
        "pair: launch start binding",
    )
    treatment_delay = (
        treatment_launch["cuda_launch_ns"]
        - treatment_launch["request_start_ns"]
    ) // 1000
    control_delay = (
        control_launch["cuda_launch_ns"] - control_launch["request_start_ns"]
    ) // 1000
    w9.require(
        abs(treatment_delay - control_delay)
        <= contract.paired_launch_delta_us,
        "pair: launch delay mismatch",
    )
    treatment_ready_us = treatment["timing"]["cuda_ready_us"]
    control_ready_us = control["timing"]["cuda_ready_us"]
    w9.require(
        abs(treatment_ready_us - control_ready_us)
        <= max(100_000, max(treatment_ready_us, control_ready_us) * 5 // 100),
        "pair: CUDA readiness mismatch",
    )
    validate_page_cache(
        pair_dir / "treatment/page_cache.json",
        label=f"{pair}.T",
        model_sha256=base.model_sha256,
        request_start_ns=treatment["timing"]["request_start_ns"],
    )
    validate_page_cache(
        pair_dir / "control/page_cache.json",
        label=f"{pair}.C",
        model_sha256=base.model_sha256,
        request_start_ns=control["request_start_ns"],
    )
    gpu_evidence: dict[str, dict[str, object]] = {}
    for leg in ("treatment", "control"):
        gpu_evidence[f"{leg}_pre"] = validate_gpu_bracket(
            pair_dir / leg / "gpu_pre.json",
            label=f"{pair}.{'T' if leg == 'treatment' else 'C'}.PRE",
            context=context,
            contract=contract,
            require_idle=True,
        )
        gpu_evidence[f"{leg}_ready"] = validate_gpu_bracket(
            pair_dir / leg / "gpu_ready.json",
            label=f"{pair}.{'T' if leg == 'treatment' else 'C'}.READY",
            context=context,
            contract=contract,
            require_idle=False,
        )
        gpu_evidence[f"{leg}_post"] = validate_gpu_bracket(
            pair_dir / leg / "gpu_post.json",
            label=f"{pair}.{'T' if leg == 'treatment' else 'C'}.POST",
            context=context,
            contract=contract,
            require_idle=False,
        )
    thermal_start = validate_phone_thermal(
        pair_dir / "treatment/thermal_pre.json",
        label=f"{pair}.T.PRE",
        context=context,
    )
    thermal_end = validate_phone_thermal(
        pair_dir / "treatment/thermal_post.json",
        label=f"{pair}.T.POST",
        context=context,
    )
    placement_result = validate_placements(
        pair_dir,
        context=context,
        gate=gate,
        treatment=treatment,
        control=control,
    )
    leg_times = {
        "treatment": (
            treatment["timing"]["request_start_ns"],
            treatment["cuda_ready"]["cuda_ready_ns"],
            treatment["timing"]["request_complete_ns"],
        ),
        "control": (
            control["request_start_ns"],
            control["cuda_ready_ns"],
            control["request_complete_ns"],
        ),
    }
    for leg, (request_start, ready_ns, complete_ns) in leg_times.items():
        pre = gpu_evidence[f"{leg}_pre"]
        ready_evidence = gpu_evidence[f"{leg}_ready"]
        post = gpu_evidence[f"{leg}_post"]
        expected_pids = {
            placement_result[f"{leg}_cuda_head"]["worker_pid"],
            placement_result[f"{leg}_cuda_tail"]["worker_pid"],
        }
        observed_ready_pids = {
            process["pid"]
            for sample in ready_evidence["samples"]
            for process in sample["compute_processes"]
        }
        w9.require(
            pre["ended_ns"] <= request_start
            and ready_evidence["started_ns"] >= ready_ns
            and ready_evidence["started_ns"] < complete_ns
            and observed_ready_pids == expected_pids
            and post["started_ns"] >= complete_ns
            and all(
                not sample["compute_processes"]
                for sample in post["samples"]
            ),
            f"pair: {leg} GPU process bracket",
        )
    exit_codes, exit_raw = read(pair_dir / "EXIT_CODES.json", "exit_codes")
    w9.require(
        exit_codes
        == {
            "control_probe_rc": 0,
            "host_exit_ok": True,
            "phone_exit_ok": True,
            "schema": "s39-profiled-cutover-pair-exit-v1",
            "treatment_probe_rc": 0,
        },
        "pair: exit codes",
    )
    return {
        "base_contract_sha256": base.raw_sha256,
        "contract_sha256": contract.raw_sha256,
        "control_launch_sha256": w6.sha256(control_launch_raw),
        "control_report_sha256": w6.sha256(control_raw),
        "decision": decision,
        "delta_contract_sha256": contract.delta_contract_sha256,
        "diagnostics": {
            "control_greedy_matching_tokens": (
                control["greedy_agreement"]["matching_tokens"]
            ),
            "control_greedy_total_tokens": (
                control["greedy_agreement"]["total_tokens"]
            ),
            "d_actual": treatment["frontiers"]["d_actual"],
            "gpu_name": context["cuda"]["name"],
            "inflight_overlap_ns": (
                treatment["ownership"]["inflight_overlap_ns"]
            ),
            "maximum_inter_token_gap_us": (
                treatment["timing"]["maximum_inter_token_gap_us"]
            ),
        },
        "exit_codes_sha256": w6.sha256(exit_raw),
        "ledger_final_sha256": ledger[-1].sha256,
        "marker_sha256": marker_sha256,
        "metrics": {
            "control_completion_us": control["timing"]["completion_us"],
            "control_cuda_ready_us": control["timing"]["cuda_ready_us"],
            "control_promotion_next_token_us": (
                control["timing"]["promotion_next_token_us"]
            ),
            "control_launch_delay_us": control_delay,
            "treatment_completion_us": treatment["timing"]["completion_us"],
            "treatment_cuda_ready_us": treatment["timing"]["cuda_ready_us"],
            "treatment_promotion_next_token_us": (
                treatment["timing"]["promotion_next_token_us"]
            ),
            "treatment_launch_delay_us": treatment_delay,
        },
        "pair_context_sha256": w6.sha256(context_raw),
        "pair_ordinal": pair,
        "physical_gate_sha256": gate.raw_sha256,
        "placement": placement_result,
        "run_id": run_id,
        "scheduler_eligible": False,
        "schema": SCHEMA,
        "scope": "MECHANICS_ONLY",
        "status": "PROFILED_ZERO_EXTRA_PAIR_PASS",
        "thermal_end_millic": thermal_end,
        "thermal_start_millic": thermal_start,
        "treatment_launch_sha256": w6.sha256(treatment_launch_raw),
        "treatment_report_sha256": w6.sha256(treatment_raw),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--w8-contract", type=Path, required=True)
    parser.add_argument("--base-contract", type=Path, required=True)
    parser.add_argument("--delta-contract", type=Path, required=True)
    parser.add_argument("--physical-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    certificate = validate_pair(
        args.pair_dir,
        contract_path=args.contract,
        w8_contract_path=args.w8_contract,
        base_contract_path=args.base_contract,
        delta_contract_path=args.delta_contract,
        physical_gate_path=args.physical_gate,
    )
    w9.write_atomic(args.output, certificate)
    print(w6.canonical(certificate).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        KeyError,
        OSError,
        ProtocolError,
        TypeError,
        ValueError,
        w9.W9Error,
        w5.HandoffError,
        w6.DeltaError,
        live.LivePromotionError,
    ) as exc:
        print(w6.canonical({
            "error": str(exc),
            "status": "PROFILED_ZERO_EXTRA_PAIR_VALIDATION_ERROR",
        }).decode("ascii"), end="")
        raise SystemExit(2)
