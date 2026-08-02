#!/usr/bin/python3 -I
"""Capture the V2.4 all-64 full-model CUDA route and raw memory evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import stat
import struct
import subprocess
import sys
import time
from typing import Any


SCHEMA = "s39-cp0-r1-v24-cuda-route-raw-v1"
PLAN_SCHEMA = "s39-cp0-r1-v24-cuda-route-launch-v1"
HISTORY_SCHEMA = "s39-cp0-r1-token-history-v2.4"
MODEL_ID = "qwen3-14b-q4_k_m"
PHASE = "A_ONLY"
MODEL_SHA256 = (
    "500a8806e85ee9c83f3ae084202955924"
    "51379b4f8cf2d0f41c15dffeb6b81f0"
)
MODEL_BYTES = 9001752960
MODEL_PATH = "/home/zhihao/models/Qwen3-14B-Q4_K_M.gguf"
N_LAYER = 40
N_EMBD = 5120
BATCH = 8
N_CTX_SEQ = 512
N_BATCH = 64
N_UBATCH = 64
MAX_STREAMS = 8
FILE_TYPE = 15
CAPABILITIES = 0x3F
CLOCK_NAME = "HOST_MONOTONIC_RAW"
CLOCK_ID = time.CLOCK_MONOTONIC_RAW
CUDA_NAME = "NVIDIA GeForce RTX 4060 Ti"
CUDA_UUID = "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08"
CUDA_MEMORY_TOTAL = 17175674880
CUDA_MINIMUM_FREE = 536870912
CUDA_HOST = "zhihao-Z690-C-ac"
SERVING_ENVELOPE = {
    "batch": 8,
    "kv_type_k": "f16",
    "kv_type_v": "f16",
    "max_streams": 8,
    "n_batch": 64,
    "n_ctx_seq": 512,
    "n_ubatch": 64,
    "sampler": "greedy",
}

STAGE_STOP = -1
STAGE_V3_HELLO = -8
STAGE_V3_BATCH = -9
STAGE_V3_SEQ_REMOVE = -10
STAGE_V3_STATUS = -11
STAGE_V3_IDENTITY = -13
STAGE_V3_MAGIC = 0x4C535633
STAGE_V3_VERSION = 3
STAGE_IDENTITY_MAGIC = 0x4C534944
STAGE_IDENTITY_VERSION = 1

RUNTIME_PREFIX = b"RUNTIMEPROCESS "
PLACEMENT_PREFIX = b"PLACEMENTCERT "
MEMORY_PREFIX = b"MEMORYCERT "
MAX_SMALL_FILE = 16 * 1024 * 1024
MAX_EXECUTABLE = 512 * 1024 * 1024
MAX_LOG_FILE = 128 * 1024 * 1024
MAX_COMMAND_OUTPUT = 16 * 1024 * 1024

STAT_KEYS = {
    "ctime_ns",
    "device_id",
    "inode",
    "mode",
    "mtime_ns",
    "size",
}
EXECUTABLE_KEYS = {"bytes", "path", "sha256", "stat"}
COMMAND_KEYS = {
    "argv",
    "cwd",
    "environment",
    "executable",
    "runtime_component_ids",
    "runtime_executable",
    "shutdown_timeout_ms",
    "startup_timeout_ms",
}
CODEC_KEYS = {
    "argv",
    "cwd",
    "environment",
    "executable",
    "timeout_ms",
}
NVIDIA_KEYS = {
    "device_argv",
    "executable",
    "process_argv",
    "timeout_ms",
}
MODEL_ARTIFACT_KEYS = {"bytes", "path", "sha256", "stat"}
PLAN_KEYS = {
    "codec",
    "expected_capabilities",
    "expected_file_type",
    "expected_max_streams",
    "expected_n_batch",
    "expected_n_ctx_seq",
    "expected_n_embd",
    "expected_n_layer",
    "expected_n_ubatch",
    "history_path",
    "history_sha256",
    "host",
    "io_timeout_ms",
    "mechanism_commands",
    "model_artifact",
    "model_id",
    "model_sha256",
    "nvidia_smi",
    "port",
    "quality_corpus_content_sha256",
    "route_epoch",
    "schema",
    "worker",
}
RUNTIME_KEYS = {
    "boot_id",
    "launcher_path",
    "loaded_repo_component_ids",
    "pid",
    "schema",
    "start_ticks",
    "system_dependencies",
}
DEPENDENCY_KEYS = {
    "build_id",
    "ctime_ns",
    "device_id",
    "inode",
    "mode",
    "mtime_ns",
    "path",
    "size",
}
PLACEMENT_KEYS = {
    "compute_by_buffer_type",
    "compute_by_op",
    "compute_by_op_and_buffer",
    "compute_nodes",
    "copy_by_buffer_type",
    "copy_nodes",
    "layer_end",
    "layer_start",
    "metadata_nodes",
    "missing_buffer_compute_nodes",
    "mode",
    "n_layer",
    "pid",
    "role",
    "run_rc",
    "schema",
    "status",
}
MEMORY_CERT_KEYS = {
    "compute_buffer_bytes",
    "host_compute_buffer_bytes",
    "host_context_buffer_bytes",
    "host_model_buffer_bytes",
    "kv_buffer_bytes",
    "model_buffer_bytes",
    "pid",
    "role",
    "schema",
}
CORPUS_KEYS = {
    "choices",
    "dataset",
    "dataset_revision",
    "expected_answer",
    "item_index",
    "question",
    "source_row",
    "subject",
}


class CaptureError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CaptureError(message)


def exact(value: Any, expected: Any, field: str) -> None:
    require(type(value) is type(expected) and value == expected, f"E_VALUE: {field}")


def exact_keys(value: Any, keys: set[str], field: str) -> dict[str, Any]:
    require(type(value) is dict, f"E_TYPE: {field}")
    require(set(value) == keys, f"E_KEYS: {field}")
    return value


def integer(value: Any, field: str, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, f"E_INTEGER: {field}")
    return value


def text(value: Any, field: str, maximum: int = 4096) -> str:
    require(type(value) is str and 0 < len(value) <= maximum, f"E_TEXT: {field}")
    require(
        all(0x20 <= ord(character) <= 0x7E for character in value),
        f"E_ASCII: {field}",
    )
    return value


def digest(value: Any, field: str) -> str:
    value = text(value, field, 64)
    require(
        len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"E_DIGEST: {field}",
    )
    return value


def vector(
    value: Any,
    field: str,
    length: int | None = None,
) -> list[int]:
    require(type(value) is list, f"E_VECTOR: {field}")
    if length is not None:
        exact(len(value), length, f"{field}.length")
    for index, item in enumerate(value):
        integer(item, f"{field}[{index}]")
    return value


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        require(key not in result, f"E_DUPLICATE_KEY: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise CaptureError(f"E_JSON_NUMBER: {value}")


def canonical_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, UnicodeEncodeError) as error:
        raise CaptureError("E_CANONICAL") from error


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def monotonic_ns() -> int:
    return time.clock_gettime_ns(CLOCK_ID)


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_mode,
    )


def stat_record(value: os.stat_result) -> dict[str, int]:
    return {
        "ctime_ns": value.st_ctime_ns,
        "device_id": value.st_dev,
        "inode": value.st_ino,
        "mode": value.st_mode,
        "mtime_ns": value.st_mtime_ns,
        "size": value.st_size,
    }


def validate_stat(value: Any, field: str) -> dict[str, int]:
    value = exact_keys(value, STAT_KEYS, field)
    for key in STAT_KEYS:
        integer(value[key], f"{field}.{key}")
    require(stat.S_ISREG(value["mode"]), f"E_STAT_TYPE: {field}")
    return value


def read_regular(path: Path, maximum: int = MAX_SMALL_FILE) -> bytes:
    require(path.is_absolute(), f"E_ABSOLUTE_PATH: {path}")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise CaptureError(f"E_OPEN: {path}") from error
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"E_FILE_TYPE: {path}")
        require(0 < before.st_size <= maximum, f"E_FILE_SIZE: {path}")
        raw = bytearray()
        while block := os.read(descriptor, min(maximum + 1, 1024 * 1024)):
            raw.extend(block)
            require(len(raw) <= maximum, f"E_FILE_SIZE: {path}")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    exact(_identity(after), _identity(before), f"E_FILE_CHANGED: {path}")
    exact(len(raw), before.st_size, f"E_FILE_CHANGED: {path}")
    return bytes(raw)


def parse_json(raw: bytes, field: str) -> Any:
    try:
        return json.loads(
            raw.decode("ascii"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CaptureError(f"E_JSON: {field}") from error


def read_canonical(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = read_regular(path)
    value = parse_json(raw, str(path))
    require(type(value) is dict, f"E_TYPE: {path}")
    exact(canonical_bytes(value), raw, f"E_CANONICAL: {path}")
    return value, raw


def durable_write_new(path: Path, raw: bytes, mode: int = 0o644) -> None:
    require(path.is_absolute() and not path.exists(), "E_OUTPUT_PATH")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        mode,
    )
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def artifact_record(path: Path) -> dict[str, Any]:
    raw = read_regular(path, MAX_LOG_FILE)
    return {
        "bytes": len(raw),
        "path": str(path),
        "sha256": sha256(raw),
    }


def validate_argv(value: Any, field: str) -> list[str]:
    require(
        type(value) is list
        and bool(value)
        and all(type(item) is str and 0 < len(item) <= 4096 for item in value),
        f"E_ARGV: {field}",
    )
    for index, item in enumerate(value):
        require(
            "\x00" not in item and "\n" not in item,
            f"E_ARGV: {field}[{index}]",
        )
    require(Path(value[0]).is_absolute(), f"E_LAUNCHER: {field}")
    require(
        Path(value[0]).name not in {"bash", "dash", "sh", "zsh"}
        and "-c" not in value,
        f"E_SHELL: {field}",
    )
    return value


def validate_environment(value: Any, field: str) -> dict[str, str]:
    require(type(value) is dict, f"E_ENV: {field}")
    for key, item in value.items():
        text(key, f"{field}.key", 128)
        text(item, f"{field}.{key}")
        require("=" not in key, f"E_ENV_KEY: {field}")
    return value


def validate_local_artifact(
    value: Any,
    field: str,
    maximum: int = MAX_EXECUTABLE,
) -> dict[str, Any]:
    value = exact_keys(value, EXECUTABLE_KEYS, field)
    path = Path(text(value["path"], f"{field}.path"))
    require(path.is_absolute(), f"E_ARTIFACT_PATH: {field}")
    expected_bytes = integer(value["bytes"], f"{field}.bytes", 1)
    expected_digest = digest(value["sha256"], f"{field}.sha256")
    expected_stat = validate_stat(value["stat"], f"{field}.stat")
    raw = read_regular(path, maximum)
    exact(len(raw), expected_bytes, f"{field}.bytes")
    exact(sha256(raw), expected_digest, f"{field}.sha256")
    exact(stat_record(os.stat(path, follow_symlinks=False)), expected_stat, f"{field}.stat")
    require(expected_stat["mode"] & 0o111 != 0, f"E_EXECUTABLE_MODE: {field}")
    return value


def validate_command(value: Any) -> dict[str, Any]:
    value = exact_keys(value, COMMAND_KEYS, "worker")
    argv = validate_argv(value["argv"], "worker.argv")
    executable = validate_local_artifact(value["executable"], "worker.executable")
    exact(argv[0], executable["path"], "worker.executable.path")
    runtime_executable = validate_local_artifact(
        value["runtime_executable"],
        "worker.runtime_executable",
    )
    require(
        runtime_executable["path"] in argv,
        "E_WORKER_RUNTIME_ARGV_BINDING",
    )
    cwd = Path(text(value["cwd"], "worker.cwd"))
    require(cwd.is_absolute() and cwd.is_dir(), "E_WORKER_CWD")
    environment = validate_environment(value["environment"], "worker.environment")
    exact(environment.get("LAYERSPLIT_MODEL_SHA256"), MODEL_SHA256, "worker.model")
    exact(environment.get("LAYERSPLIT_MEMORY_CERT"), "1", "worker.memory_cert")
    exact(environment.get("LAYERSPLIT_PLACEMENT_CERT"), "1", "worker.placement_cert")
    required = {
        "--mode": "monov3",
        "--backend": "CUDA0",
        "--layer-start": "0",
        "--layer-end": str(N_LAYER),
        "--model": MODEL_PATH,
    }
    for flag, expected in required.items():
        exact(argv.count(flag), 1, f"worker.argv.{flag}.count")
        index = argv.index(flag)
        require(index + 1 < len(argv), f"E_WORKER_ARG: {flag}")
        exact(argv[index + 1], expected, f"worker.argv.{flag}")
    component_ids = value["runtime_component_ids"]
    require(
        type(component_ids) is list
        and bool(component_ids)
        and component_ids == sorted(set(component_ids))
        and all(type(item) is str and 0 < len(item) <= 128 for item in component_ids),
        "E_RUNTIME_COMPONENT_IDS",
    )
    for name in ("startup_timeout_ms", "shutdown_timeout_ms"):
        require(
            integer(value[name], f"worker.{name}", 1) <= 600_000,
            f"E_TIMEOUT: worker.{name}",
        )
    return value


def validate_codec(value: Any) -> dict[str, Any]:
    value = exact_keys(value, CODEC_KEYS, "codec")
    argv = validate_argv(value["argv"], "codec.argv")
    executable = validate_local_artifact(value["executable"], "codec.executable")
    exact(argv[0], executable["path"], "codec.executable.path")
    require(
        "--model" in argv
        and MODEL_PATH in argv
        and "--model-sha256" in argv
        and MODEL_SHA256 in argv,
        "E_CODEC_MODEL_BINDING",
    )
    cwd = Path(text(value["cwd"], "codec.cwd"))
    require(cwd.is_absolute() and cwd.is_dir(), "E_CODEC_CWD")
    validate_environment(value["environment"], "codec.environment")
    require(integer(value["timeout_ms"], "codec.timeout", 1) <= 600_000, "E_CODEC_TIMEOUT")
    return value


def validate_nvidia(value: Any) -> dict[str, Any]:
    value = exact_keys(value, NVIDIA_KEYS, "nvidia_smi")
    executable = validate_local_artifact(
        value["executable"],
        "nvidia_smi.executable",
        64 * 1024 * 1024,
    )
    device = validate_argv(value["device_argv"], "nvidia_smi.device_argv")
    process = validate_argv(value["process_argv"], "nvidia_smi.process_argv")
    exact(device[0], executable["path"], "nvidia_smi.device.path")
    exact(process[0], executable["path"], "nvidia_smi.process.path")
    expected_device = [
        executable["path"],
        f"--id={CUDA_UUID}",
        "--query-gpu=name,uuid,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ]
    expected_process = [
        executable["path"],
        f"--id={CUDA_UUID}",
        "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    exact(device, expected_device, "nvidia_smi.device_argv")
    exact(process, expected_process, "nvidia_smi.process_argv")
    require(integer(value["timeout_ms"], "nvidia_smi.timeout", 1) <= 60_000, "E_NVIDIA_TIMEOUT")
    return value


def validate_model_artifact(value: Any, artifact_snapshot: dict[str, Any]) -> None:
    value = exact_keys(value, MODEL_ARTIFACT_KEYS, "model_artifact")
    exact(value["path"], MODEL_PATH, "model_artifact.path")
    exact(value["bytes"], MODEL_BYTES, "model_artifact.bytes")
    exact(value["sha256"], MODEL_SHA256, "model_artifact.sha256")
    validate_stat(value["stat"], "model_artifact.stat")
    exact(value, artifact_snapshot, "model_artifact.snapshot")
    live = os.stat(Path(MODEL_PATH), follow_symlinks=False)
    exact(stat_record(live), value["stat"], "model_artifact.live_stat")


def load_base_evidence(
    pre_dir: Path,
    phase_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    phase_raw = read_regular(pre_dir / "phase_lock.jsonl")
    lines = phase_raw.splitlines(keepends=True)
    exact(len(lines), 1, "phase_lock.rows")
    phase = parse_json(lines[0], "phase_lock")
    require(type(phase) is dict, "E_PHASE_LOCK")
    exact_keys(
        phase,
        {
            "artifact_root_sha256",
            "candidate_sha256",
            "contract_sha256",
            "device_boot_ids",
            "event_ns",
            "model_id",
            "phase",
            "phase_id",
            "preparation_sha256",
            "quality_corpus_sha256",
            "runtime_bundle_plan_sha256",
            "schema",
        },
        "phase_lock",
    )
    exact(phase["schema"], "s39-cp0-r1-phase-lock-v2.4", "phase_lock.schema")
    exact(phase.get("phase"), PHASE, "phase_lock.phase")
    exact(phase.get("phase_id"), phase_id, "phase_lock.phase_id")
    exact(phase["model_id"], MODEL_ID, "phase_lock.model_id")
    for key in (
        "artifact_root_sha256",
        "candidate_sha256",
        "contract_sha256",
        "preparation_sha256",
        "quality_corpus_sha256",
        "runtime_bundle_plan_sha256",
    ):
        digest(phase[key], f"phase_lock.{key}")
    boot_ids = exact_keys(
        phase["device_boot_ids"],
        {"cuda", "op12", "op15"},
        "phase_lock.device_boot_ids",
    )
    for endpoint, value in boot_ids.items():
        text(value, f"phase_lock.device_boot_ids.{endpoint}", 128)

    artifact, artifact_raw = read_canonical(
        pre_dir.parent / "artifact" / "artifact_root.json"
    )
    exact_keys(
        artifact,
        {
            "candidate_sha256",
            "completed_ns",
            "components",
            "contract_sha256",
            "inventories",
            "model_id",
            "phase_scope",
            "runtime_bundle_plan_sha256",
            "schema",
            "started_ns",
        },
        "artifact_root",
    )
    exact(
        artifact["schema"],
        "s39-cp0-r1-artifact-root-v2.4",
        "artifact_root.schema",
    )
    exact(artifact["phase_scope"], "PRE_REBOOT_OUTSIDE_PHASE", "artifact_root.scope")
    exact(artifact["model_id"], MODEL_ID, "artifact_root.model")
    exact(sha256(artifact_raw), phase["artifact_root_sha256"], "artifact_root.lock")
    records = artifact["components"]
    require(type(records) is list and bool(records), "E_ARTIFACT_COMPONENTS")
    model_rows = [
        record
        for record in records
        if type(record) is dict and record.get("component_id") == "model.cuda"
    ]
    exact(len(model_rows), 1, "artifact_root.model_cuda.count")
    model_row = exact_keys(
        model_rows[0],
        {
            "bytes",
            "component_id",
            "endpoint",
            "kind",
            "path",
            "sha256",
            "stat",
        },
        "artifact_root.model_cuda",
    )
    exact(model_row["endpoint"], "cuda", "artifact_root.model_cuda.endpoint")
    exact(model_row["kind"], "model_weight", "artifact_root.model_cuda.kind")
    model_artifact = {
        key: model_row[key]
        for key in ("bytes", "path", "sha256", "stat")
    }
    validate_stat(model_artifact["stat"], "artifact_root.model_cuda.stat")

    fresh, fresh_raw = read_canonical(
        pre_dir.parent / "fresh" / "fresh_snapshot.json"
    )
    exact_keys(
        fresh,
        {
            "artifact_root_sha256",
            "component_stats",
            "completed_ns",
            "devices",
            "inventories",
            "phase",
            "phase_id",
            "phase_lock_sha256",
            "preparation_sha256",
            "runtime_bundle_plan_sha256",
            "schema",
            "started_ns",
        },
        "fresh",
    )
    exact(
        fresh["schema"],
        "s39-cp0-r1-fast-fresh-readiness-v2.4",
        "fresh.schema",
    )
    exact(fresh["phase"], PHASE, "fresh.phase")
    exact(fresh["phase_id"], phase_id, "fresh.phase_id")
    exact(fresh["phase_lock_sha256"], sha256(phase_raw), "fresh.phase_lock")
    exact(fresh["artifact_root_sha256"], sha256(artifact_raw), "fresh.artifact_root")
    exact(
        fresh["preparation_sha256"],
        phase["preparation_sha256"],
        "fresh.preparation",
    )
    exact(
        fresh["runtime_bundle_plan_sha256"],
        phase["runtime_bundle_plan_sha256"],
        "fresh.runtime_plan",
    )
    require(type(fresh["component_stats"]) is list and bool(fresh["component_stats"]),
            "E_FRESH_COMPONENT_STATS")
    require(type(fresh["inventories"]) is list and bool(fresh["inventories"]),
            "E_FRESH_INVENTORIES")
    devices = exact_keys(
        fresh["devices"],
        {"cuda", "op12", "op15"},
        "fresh.devices",
    )
    fresh_cuda = exact_keys(
        devices["cuda"],
        {
            "gpu_uuid",
            "host",
            "host_boot_id",
            "pci_bus_id",
            "system_swap_used_bytes",
        },
        "fresh.devices.cuda",
    )
    exact(fresh_cuda["host"], CUDA_HOST, "fresh.cuda.host")
    exact(fresh_cuda["gpu_uuid"], CUDA_UUID, "fresh.cuda.uuid")
    exact(
        fresh_cuda["host_boot_id"],
        phase["device_boot_ids"]["cuda"],
        "fresh.cuda.boot_id",
    )
    text(fresh_cuda["host_boot_id"], "fresh.cuda.host_boot_id", 128)
    text(fresh_cuda["pci_bus_id"], "fresh.cuda.pci_bus_id", 64)
    integer(fresh_cuda["system_swap_used_bytes"], "fresh.cuda.swap")
    require(len(fresh_raw) > 0, "E_FRESH_EMPTY")
    return model_artifact, fresh_cuda, phase


def load_runtime_evidence(
    pre_dir: Path,
    phase_id: str,
    plan: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    snapshot, _ = read_canonical(pre_dir.parent / "artifact" / "artifact_root.json")
    exact_keys(
        snapshot,
        {
            "candidate_sha256",
            "completed_ns",
            "components",
            "contract_sha256",
            "inventories",
            "model_id",
            "phase_scope",
            "runtime_bundle_plan_sha256",
            "schema",
            "started_ns",
        },
        "artifact_root",
    )
    exact(
        snapshot["schema"],
        "s39-cp0-r1-artifact-root-v2.4",
        "artifact_root.schema",
    )
    exact(snapshot["phase_scope"], "PRE_REBOOT_OUTSIDE_PHASE", "artifact_root.scope")
    exact(snapshot["model_id"], MODEL_ID, "artifact_root.model")
    records = snapshot["components"]
    require(type(records) is list and bool(records), "E_ARTIFACT_COMPONENTS")
    components = {}
    for index, record in enumerate(records):
        field = f"artifact_root.components[{index}]"
        exact_keys(
            record,
            {
                "bytes",
                "component_id",
                "endpoint",
                "kind",
                "path",
                "sha256",
                "stat",
            },
            field,
        )
        if record["component_id"] not in plan["worker"]["runtime_component_ids"]:
            continue
        exact(record["endpoint"], "cuda", f"{field}.endpoint")
        component_id = text(record["component_id"], f"{field}.component_id", 128)
        require(component_id not in components, f"E_RUNTIME_COMPONENT_REUSE: {component_id}")
        exact(record["bytes"], record["stat"]["size"], f"{field}.bytes")
        digest(record["sha256"], f"{field}.sha256")
        validate_stat(record["stat"], f"{field}.stat")
        components[component_id] = record
    exact(
        sorted(components),
        plan["worker"]["runtime_component_ids"],
        "artifact_root.runtime_component_ids",
    )
    runtime_executable = plan["worker"]["runtime_executable"]
    launcher = [
        record
        for record in components.values()
        if record["path"] == runtime_executable["path"]
    ]
    exact(len(launcher), 1, "runtime_snapshot.launcher_count")
    for key in ("bytes", "path", "sha256", "stat"):
        exact(launcher[0][key], runtime_executable[key], f"runtime_snapshot.launcher.{key}")

    fresh, _ = read_canonical(pre_dir.parent / "fresh" / "fresh_snapshot.json")
    exact_keys(
        fresh,
        {
            "artifact_root_sha256",
            "component_stats",
            "completed_ns",
            "devices",
            "inventories",
            "phase",
            "phase_id",
            "phase_lock_sha256",
            "preparation_sha256",
            "runtime_bundle_plan_sha256",
            "schema",
            "started_ns",
        },
        "fresh",
    )
    exact(
        fresh["schema"],
        "s39-cp0-r1-fast-fresh-readiness-v2.4",
        "fresh.schema",
    )
    exact(fresh["phase"], PHASE, "fresh.phase")
    exact(fresh["phase_id"], phase_id, "fresh.phase_id")
    stats = {}
    for index, record in enumerate(fresh["component_stats"]):
        field = f"fresh.component_stats[{index}]"
        exact_keys(
            record,
            {"component_id", "endpoint", "path", "stat"},
            field,
        )
        if record["component_id"] not in components:
            continue
        component_id = record["component_id"]
        require(component_id in components and component_id not in stats, f"E_RUNTIME_FRESH_COMPONENT: {component_id}")
        exact(record["endpoint"], "cuda", f"{field}.endpoint")
        exact(record["path"], components[component_id]["path"], f"{field}.path")
        validate_stat(record["stat"], f"{field}.stat")
        exact(record["stat"], components[component_id]["stat"], f"{field}.stat")
        live = stat_record(os.stat(Path(record["path"]), follow_symlinks=False))
        exact(live, record["stat"], f"{field}.live")
        stats[component_id] = record
    exact(sorted(stats), sorted(components), "runtime_fresh.component_ids")
    return components


def validate_history_group(
    group: Any,
    group_index: int,
    requests: list[dict[str, Any]],
) -> None:
    field = f"histories.quality_groups[{group_index}]"
    exact_keys(
        group,
        {"decode_calls", "group_index", "item_indices", "prefill_partitions"},
        field,
    )
    item_indices = list(range(group_index * BATCH, (group_index + 1) * BATCH))
    exact(group["group_index"], group_index, f"{field}.group_index")
    exact(group["item_indices"], item_indices, f"{field}.item_indices")
    expected_rows = [
        {
            "item_index": item_index,
            "position": position,
            "request_id": requests[item_index]["request_id"],
            "seq_id": requests[item_index]["seq_id"],
            "token_id": requests[item_index]["token_ids"][position],
        }
        for position in range(
            max(len(requests[item_index]["token_ids"]) for item_index in item_indices)
        )
        for item_index in item_indices
        if position < len(requests[item_index]["token_ids"])
    ]
    observed_rows = []
    partitions = group["prefill_partitions"]
    require(type(partitions) is list and bool(partitions), f"E_PREFILL: {field}")
    for call_index, partition in enumerate(partitions):
        exact_keys(partition, {"call_index", "rows"}, f"{field}.prefill[{call_index}]")
        exact(partition["call_index"], call_index, f"{field}.prefill.call_index")
        rows = partition["rows"]
        require(type(rows) is list and 0 < len(rows) <= N_UBATCH,
                f"E_PREFILL_ROWS: {field}")
        observed_rows.extend(rows)
    exact(observed_rows, expected_rows, f"{field}.prefill_rows")
    expected_decode = [
        {
            "call_index": len(partitions) + ordinal,
            "continuation_input_ordinal": ordinal,
            "continuation_output_ordinal": ordinal + 1,
            "rows": [
                {
                    "item_index": item_index,
                    "position": len(requests[item_index]["token_ids"]) + ordinal,
                    "request_id": requests[item_index]["request_id"],
                    "seq_id": requests[item_index]["seq_id"],
                }
                for item_index in item_indices
            ],
        }
        for ordinal in range(7)
    ]
    exact(group["decode_calls"], expected_decode, f"{field}.decode_calls")


def load_histories(path: Path) -> tuple[dict[str, Any], bytes]:
    value, raw = read_canonical(path)
    exact_keys(
        value,
        {
            "batch",
            "candidate_sha256",
            "continuation_tokens_per_request",
            "corpus_sha256",
            "mechanics_b8",
            "model_id",
            "model_sha256",
            "n_batch",
            "n_ctx_seq",
            "n_ubatch",
            "prefill_chunking",
            "prefill_row_order",
            "quality_groups",
            "requests",
            "schema",
            "tokenizer",
        },
        "histories",
    )
    exact(value["schema"], HISTORY_SCHEMA, "histories.schema")
    exact(value["model_id"], MODEL_ID, "histories.model_id")
    exact(value["model_sha256"], MODEL_SHA256, "histories.model_sha256")
    exact(value["batch"], BATCH, "histories.batch")
    exact(value["continuation_tokens_per_request"], 8, "histories.continuations")
    exact(value["n_ctx_seq"], N_CTX_SEQ, "histories.n_ctx_seq")
    exact(value["n_batch"], N_BATCH, "histories.n_batch")
    exact(value["n_ubatch"], N_UBATCH, "histories.n_ubatch")
    exact(
        value["prefill_chunking"],
        "WHOLE_POSITION_WAVES_MAX_64_ROWS",
        "histories.prefill_chunking",
    )
    exact(
        value["prefill_row_order"],
        "POSITION_MAJOR_THEN_ITEM_INDEX",
        "histories.prefill_row_order",
    )
    requests = value["requests"]
    require(type(requests) is list and len(requests) == 64, "E_HISTORY_REQUESTS")
    for item_index, request in enumerate(requests):
        field = f"histories.requests[{item_index}]"
        exact_keys(
            request,
            {
                "item_index",
                "prompt_sha256",
                "prompt_utf8_base64",
                "prompt_utf8_bytes",
                "request_id",
                "seq_id",
                "token_ids",
            },
            field,
        )
        exact(request["item_index"], item_index, f"{field}.item_index")
        exact(request["request_id"], item_index % BATCH + 1, f"{field}.request_id")
        exact(request["seq_id"], item_index % BATCH, f"{field}.seq_id")
        history = request["token_ids"]
        require(
            type(history) is list
            and 0 < len(history) <= N_CTX_SEQ - 8
            and all(type(token) is int and token >= 0 for token in history),
            f"E_HISTORY_TOKENS: {field}",
        )
        digest(request["prompt_sha256"], f"{field}.prompt_sha256")
    groups = value["quality_groups"]
    require(type(groups) is list and len(groups) == 8, "E_HISTORY_GROUPS")
    for group_index, group in enumerate(groups):
        validate_history_group(group, group_index, requests)
    exact(value["mechanics_b8"], groups[0], "histories.mechanics_b8")
    return value, raw


def load_plan(
    path: Path,
    histories_path: Path,
    histories_raw: bytes,
    artifact_snapshot: dict[str, Any],
) -> tuple[dict[str, Any], bytes]:
    value, raw = read_canonical(path)
    exact_keys(value, PLAN_KEYS, "launch_plan")
    exact(value["schema"], PLAN_SCHEMA, "launch_plan.schema")
    exact(value["model_id"], MODEL_ID, "launch_plan.model_id")
    exact(value["model_sha256"], MODEL_SHA256, "launch_plan.model_sha256")
    exact(value["expected_file_type"], FILE_TYPE, "launch_plan.file_type")
    exact(value["expected_n_layer"], N_LAYER, "launch_plan.n_layer")
    exact(value["expected_n_embd"], N_EMBD, "launch_plan.n_embd")
    exact(value["expected_max_streams"], MAX_STREAMS, "launch_plan.max_streams")
    exact(value["expected_n_ctx_seq"], N_CTX_SEQ, "launch_plan.n_ctx_seq")
    exact(value["expected_n_batch"], N_BATCH, "launch_plan.n_batch")
    exact(value["expected_n_ubatch"], N_UBATCH, "launch_plan.n_ubatch")
    exact(value["expected_capabilities"], CAPABILITIES, "launch_plan.capabilities")
    exact(value["history_path"], str(histories_path), "launch_plan.history_path")
    exact(value["history_sha256"], sha256(histories_raw), "launch_plan.history_sha256")
    integer(value["route_epoch"], "launch_plan.route_epoch", 1)
    digest(
        value["quality_corpus_content_sha256"],
        "launch_plan.quality_corpus_content_sha256",
    )
    host = text(value["host"], "launch_plan.host", 255)
    require(host in {"127.0.0.1", "::1"}, "E_WORKER_HOST")
    require(integer(value["port"], "launch_plan.port", 1) <= 65535, "E_WORKER_PORT")
    require(integer(value["io_timeout_ms"], "launch_plan.io_timeout_ms", 1) <= 600_000, "E_IO_TIMEOUT")
    validate_model_artifact(value["model_artifact"], artifact_snapshot)
    validate_command(value["worker"])
    validate_codec(value["codec"])
    validate_nvidia(value["nvidia_smi"])
    return value, raw


def derive_mechanism_commands(plan: dict[str, Any]) -> dict[str, list[list[str]]]:
    matrix = exact_keys(
        plan["mechanism_commands"],
        {"desktop", "op12", "op15"},
        "mechanism_commands",
    )
    for endpoint in ("desktop", "op12", "op15"):
        commands = matrix[endpoint]
        require(type(commands) is list and bool(commands), f"E_COMMAND_MATRIX: {endpoint}")
        for index, command in enumerate(commands):
            validate_argv(command, f"mechanism_commands.{endpoint}[{index}]")
    expected_local = [
        list(plan["codec"]["argv"]),
        list(plan["worker"]["argv"]),
        list(plan["nvidia_smi"]["device_argv"]),
        list(plan["nvidia_smi"]["process_argv"]),
        list(plan["nvidia_smi"]["device_argv"]),
        list(plan["nvidia_smi"]["process_argv"]),
        list(plan["nvidia_smi"]["device_argv"]),
        list(plan["nvidia_smi"]["process_argv"]),
    ]
    exact(len(matrix["desktop"]), 9, "mechanism_commands.desktop.count")
    exact(matrix["desktop"][:8], expected_local, "mechanism_commands.desktop.local")
    require(
        matrix["desktop"][8] not in matrix["desktop"][:8],
        "E_MONOLITHIC_COMMAND_REUSE",
    )
    exact(
        sum(command == plan["codec"]["argv"] for command in matrix["desktop"]),
        1,
        "mechanism_commands.codec_count",
    )
    return matrix


def bind_mechanism_commands(plan: dict[str, Any], supplied: str) -> str:
    expected = sha256(canonical_bytes(derive_mechanism_commands(plan)))
    exact(supplied, expected, "mechanism_commands_sha256")
    return expected


def pack_i32(values: Any) -> bytes:
    values = tuple(values)
    return struct.pack(f"<{len(values)}i", *values)


def pack_i64(values: Any) -> bytes:
    values = tuple(values)
    return struct.pack(f"<{len(values)}q", *values)


class StageClient:
    def __init__(self, connection: socket.socket):
        self.connection = connection
        self.n_batch = 0
        self.n_ubatch = 0

    def recv_exact(self, size: int) -> bytes:
        integer(size, "recv.size")
        result = bytearray()
        while len(result) < size:
            block = self.connection.recv(size - len(result))
            require(bool(block), "E_UNEXPECTED_EOF")
            result.extend(block)
        return bytes(result)

    def recv_i32(self, count: int) -> tuple[int, ...]:
        return struct.unpack(f"<{count}i", self.recv_exact(count * 4))

    def hello(self, plan: dict[str, Any]) -> dict[str, Any]:
        self.connection.sendall(pack_i32([STAGE_V3_HELLO]))
        words = self.recv_i32(11)
        exact(words[0], STAGE_V3_MAGIC, "hello.magic")
        exact(words[1], STAGE_V3_VERSION, "hello.version")
        exact(words[2], 0, "hello.layer_start")
        exact(words[3], N_LAYER, "hello.layer_end")
        exact(words[4], N_LAYER, "hello.n_layer")
        exact(words[5], N_EMBD, "hello.n_embd")
        exact(words[6], MAX_STREAMS, "hello.max_streams")
        exact(words[7], N_CTX_SEQ, "hello.n_ctx_seq")
        exact(words[8], N_BATCH, "hello.n_batch")
        exact(words[9], N_UBATCH, "hello.n_ubatch")
        exact(words[10], CAPABILITIES, "hello.capabilities")
        self.n_batch = words[8]
        self.n_ubatch = words[9]
        for key, expected in (
            ("expected_n_layer", N_LAYER),
            ("expected_n_embd", N_EMBD),
            ("expected_max_streams", MAX_STREAMS),
            ("expected_n_ctx_seq", N_CTX_SEQ),
            ("expected_n_batch", N_BATCH),
            ("expected_n_ubatch", N_UBATCH),
            ("expected_capabilities", CAPABILITIES),
        ):
            exact(plan[key], expected, f"plan.{key}")
        self.connection.sendall(pack_i32([STAGE_V3_IDENTITY]))
        magic, version, file_type = self.recv_i32(3)
        model_sha256 = self.recv_exact(32).hex()
        exact(magic, STAGE_IDENTITY_MAGIC, "identity.magic")
        exact(version, STAGE_IDENTITY_VERSION, "identity.version")
        exact(file_type, FILE_TYPE, "identity.file_type")
        exact(model_sha256, MODEL_SHA256, "identity.model_sha256")
        return {
            "capabilities": words[10],
            "file_type": file_type,
            "layer_end": words[3],
            "layer_start": words[2],
            "max_streams": words[6],
            "model_sha256": model_sha256,
            "n_batch": words[8],
            "n_ctx_seq": words[7],
            "n_embd": words[5],
            "n_layer": words[4],
            "n_ubatch": words[9],
            "schema": "layersplit-stage-v3-identity-v1",
            "stage_identity_version": version,
            "stage_protocol_version": words[1],
        }

    def status(self) -> tuple[int, int, bool]:
        self.connection.sendall(pack_i32([STAGE_V3_STATUS, STAGE_V3_VERSION]))
        code, version, active, maximum, draining = self.recv_i32(5)
        exact(code, 0, "status.code")
        exact(version, STAGE_V3_VERSION, "status.version")
        require(
            0 <= active <= maximum
            and maximum == MAX_STREAMS
            and draining in (0, 1),
            "E_STATUS",
        )
        return active, maximum, bool(draining)

    def batch(self, rows: list[tuple[int, int, int, int, int]]) -> list[int]:
        require(
            bool(rows) and len(rows) <= min(self.n_batch, self.n_ubatch),
            "E_BATCH_SIZE",
        )
        payload = pack_i32([STAGE_V3_BATCH, STAGE_V3_VERSION, len(rows), 0])
        payload += pack_i64(row[0] for row in rows)
        payload += pack_i64(row[1] for row in rows)
        payload += pack_i32(row[2] for row in rows)
        payload += pack_i32(row[3] for row in rows)
        payload += pack_i32(row[4] for row in rows)
        self.connection.sendall(payload)
        exact(self.recv_i32(1)[0], 0, "batch.status")
        count, width = self.recv_i32(2)
        exact(count, len(rows), "batch.count")
        exact(width, 0, "batch.width")
        request_ids = struct.unpack(f"<{count}q", self.recv_exact(8 * count))
        epochs = struct.unpack(f"<{count}q", self.recv_exact(8 * count))
        sequences = struct.unpack(f"<{count}i", self.recv_exact(4 * count))
        positions = struct.unpack(f"<{count}i", self.recv_exact(4 * count))
        tokens = self.recv_i32(count)
        exact(request_ids, tuple(row[0] for row in rows), "batch.request_ids")
        exact(epochs, tuple(row[1] for row in rows), "batch.epochs")
        exact(sequences, tuple(row[2] for row in rows), "batch.sequences")
        exact(positions, tuple(row[3] for row in rows), "batch.positions")
        require(all(token >= 0 for token in tokens), "E_BATCH_TOKEN")
        return list(tokens)

    def remove(self, request_id: int, route_epoch: int, sequence: int) -> None:
        payload = pack_i32([STAGE_V3_SEQ_REMOVE, STAGE_V3_VERSION, sequence])
        payload += pack_i64([request_id, route_epoch])
        self.connection.sendall(payload)
        code, version, active, maximum, draining = self.recv_i32(5)
        exact(code, 0, "remove.code")
        exact(version, STAGE_V3_VERSION, "remove.version")
        require(
            0 <= active <= maximum
            and maximum == MAX_STREAMS
            and draining in (0, 1),
            "E_REMOVE_STATUS",
        )

    def stop(self) -> None:
        self.connection.sendall(pack_i32([STAGE_STOP]))


def run_history_group(
    client: StageClient,
    history: dict[str, Any],
    group: dict[str, Any],
    wire_ids: list[int],
    route_epoch: int,
    frame_offset: int,
) -> tuple[list[list[int]], list[dict[str, Any]], dict[str, Any]]:
    requests = {row["item_index"]: row for row in history["requests"]}
    item_indices = group["item_indices"]
    exact(len(item_indices), BATCH, "group.item_count")
    exact(len(wire_ids), BATCH, "group.wire_count")
    current = {}
    calls = []
    receipts = []
    for partition in group["prefill_partitions"]:
        rows = [
            (
                wire_ids[source["seq_id"]],
                route_epoch,
                source["seq_id"],
                source["position"],
                source["token_id"],
            )
            for source in partition["rows"]
        ]
        tokens = client.batch(rows)
        receipt_rows = []
        for source, wire, token in zip(partition["rows"], rows, tokens):
            request = requests[source["item_index"]]
            if source["position"] == len(request["token_ids"]) - 1:
                require(source["seq_id"] not in current, "E_PREFILL_FINAL_REUSE")
                current[source["seq_id"]] = token
            receipt_rows.append({
                "input_token": wire[4],
                "item_index": source["item_index"],
                "output_token": token,
                "position": wire[3],
                "request_id": source["request_id"],
                "route_epoch": route_epoch,
                "seq_id": wire[2],
                "wire_request_id": wire[0],
            })
        call_index = partition["call_index"]
        exact(call_index, len(calls), "group.prefill.call_index")
        calls.append({
            "call_index": call_index,
            "n_seqs": len({row[2] for row in rows}),
            "n_tokens": len(rows),
            "phase": "prefill",
            "positions": [source["position"] for source in partition["rows"]],
            "request_ids": [source["request_id"] for source in partition["rows"]],
            "seq_ids": [source["seq_id"] for source in partition["rows"]],
        })
        receipts.append({
            "call_index": call_index,
            "frame_call_index": frame_offset + len(receipts),
            "phase": "prefill",
            "rows": receipt_rows,
        })
    exact(sorted(current), list(range(BATCH)), "group.prefill.outputs")
    continuations = [[current[sequence]] for sequence in range(BATCH)]
    for ordinal, decode in enumerate(group["decode_calls"]):
        rows = [
            (
                wire_ids[source["seq_id"]],
                route_epoch,
                source["seq_id"],
                source["position"],
                current[source["seq_id"]],
            )
            for source in decode["rows"]
        ]
        tokens = client.batch(rows)
        next_tokens = {}
        receipt_rows = []
        for source, wire, token in zip(decode["rows"], rows, tokens):
            next_tokens[source["seq_id"]] = token
            continuations[source["seq_id"]].append(token)
            receipt_rows.append({
                "input_token": wire[4],
                "item_index": source["item_index"],
                "output_token": token,
                "position": wire[3],
                "request_id": source["request_id"],
                "route_epoch": route_epoch,
                "seq_id": wire[2],
                "wire_request_id": wire[0],
            })
        exact(sorted(next_tokens), list(range(BATCH)), "group.decode.outputs")
        current = next_tokens
        exact(decode["continuation_input_ordinal"], ordinal,
              "group.decode.input_ordinal")
        exact(decode["continuation_output_ordinal"], ordinal + 1,
              "group.decode.output_ordinal")
        call_index = decode["call_index"]
        exact(call_index, len(calls), "group.decode.call_index")
        calls.append({
            "call_index": call_index,
            "n_seqs": BATCH,
            "n_tokens": BATCH,
            "phase": "decode",
            "positions": [source["position"] for source in decode["rows"]],
            "request_ids": [source["request_id"] for source in decode["rows"]],
            "seq_ids": [source["seq_id"] for source in decode["rows"]],
        })
        receipts.append({
            "call_index": call_index,
            "frame_call_index": frame_offset + len(receipts),
            "phase": "decode",
            "rows": receipt_rows,
        })
    require(all(len(row) == 8 for row in continuations), "E_CONTINUATION_COUNT")
    return continuations, calls, {
        "call_receipts": receipts,
        "continuations": continuations,
        "group_index": group["group_index"],
        "item_indices": item_indices,
        "wire_request_ids": wire_ids,
    }


def remove_group(
    client: StageClient,
    request_ids: list[int],
    route_epoch: int,
) -> None:
    for sequence, request_id in enumerate(request_ids):
        client.remove(request_id, route_epoch, sequence)
    exact(client.status()[0], 0, "cleanup.state_count")


def connect(
    plan: dict[str, Any],
    process: subprocess.Popen[bytes],
) -> StageClient:
    deadline = time.monotonic() + plan["worker"]["startup_timeout_ms"] / 1000
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        require(process.poll() is None, "E_WORKER_EARLY_EXIT")
        try:
            connection = socket.create_connection(
                (plan["host"], plan["port"]),
                timeout=0.2,
            )
            connection.settimeout(plan["io_timeout_ms"] / 1000)
            return StageClient(connection)
        except OSError as error:
            last_error = error
            time.sleep(0.02)
    raise CaptureError(f"E_WORKER_CONNECT: {last_error}")


def parse_prefixed(raw: bytes, prefix: bytes, field: str) -> list[dict[str, Any]]:
    records = []
    for line in raw.splitlines():
        if line.startswith(prefix):
            value = parse_json(line[len(prefix):], field)
            require(type(value) is dict, f"E_PREFIX_TYPE: {field}")
            records.append(value)
    return records


def parse_runtime_process(
    raw: bytes,
    plan: dict[str, Any],
    host_boot_id: str,
    started_ns: int,
    completed_ns: int,
    observed_ns: int,
    process_pid: int,
    expected_components: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    records = parse_prefixed(raw, RUNTIME_PREFIX, "runtime_process")
    exact(len(records), 1, "runtime_process.count")
    value = exact_keys(records[0], RUNTIME_KEYS, "runtime_process")
    exact(value["schema"], "s39-runtime-process-source-v1", "runtime_process.schema")
    exact(value["boot_id"], host_boot_id, "runtime_process.boot_id")
    exact(value["pid"], process_pid, "runtime_process.pid")
    integer(value["start_ticks"], "runtime_process.start_ticks", 1)
    exact(
        value["launcher_path"],
        plan["worker"]["runtime_executable"]["path"],
        "runtime_process.launcher_path",
    )
    exact(
        value["loaded_repo_component_ids"],
        plan["worker"]["runtime_component_ids"],
        "runtime_process.components",
    )
    require(started_ns <= observed_ns <= completed_ns, "E_RUNTIME_PROCESS_INTERVAL")
    dependencies = value["system_dependencies"]
    require(type(dependencies) is list and bool(dependencies), "E_RUNTIME_DEPENDENCIES")
    paths = []
    for index, dependency in enumerate(dependencies):
        field = f"runtime_process.dependencies[{index}]"
        exact_keys(dependency, DEPENDENCY_KEYS, field)
        path = text(dependency["path"], f"{field}.path")
        require(Path(path).is_absolute(), f"E_DEPENDENCY_PATH: {field}")
        paths.append(path)
        for key in ("ctime_ns", "device_id", "inode", "mode", "mtime_ns"):
            integer(dependency[key], f"{field}.{key}")
        integer(dependency["size"], f"{field}.size", 1)
        build_id = dependency["build_id"]
        require(
            build_id is None or (type(build_id) is str and 0 < len(build_id) <= 256),
            f"E_DEPENDENCY_BUILD_ID: {field}",
        )
    exact(paths, sorted(set(paths)), "runtime_process.dependency_order")
    by_path = {dependency["path"]: dependency for dependency in dependencies}
    for component_id, component in expected_components.items():
        require(
            component["path"] in by_path,
            f"E_RUNTIME_DEPENDENCY_MISSING: {component_id}",
        )
        dependency = by_path[component["path"]]
        for key in STAT_KEYS:
            exact(
                dependency[key],
                component["stat"][key],
                f"runtime_process.component.{component_id}.{key}",
            )
    return {
        "boot_id": value["boot_id"],
        "bundle_id": "cuda_route",
        "endpoint": "cuda",
        "launcher_path": value["launcher_path"],
        "loaded_repo_component_ids": value["loaded_repo_component_ids"],
        "observed_ns": observed_ns,
        "pid": value["pid"],
        "start_ticks": value["start_ticks"],
        "system_dependencies": dependencies,
    }


def wait_for_prefix(
    log_path: Path,
    prefix: bytes,
    process: subprocess.Popen[bytes],
    timeout_ms: int,
) -> int:
    deadline = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < deadline:
        require(process.poll() is None, "E_WORKER_EARLY_EXIT")
        try:
            raw = read_regular(log_path, MAX_LOG_FILE)
            if len(parse_prefixed(raw, prefix, "startup")) == 1:
                return monotonic_ns()
        except (CaptureError, OSError):
            pass
        time.sleep(0.02)
    raise CaptureError(f"E_STARTUP_RECORD_TIMEOUT: {prefix!r}")


def parse_memory_cert(raw: bytes, process_pid: int) -> dict[str, int]:
    records = parse_prefixed(raw, MEMORY_PREFIX, "memory_cert")
    exact(len(records), 1, "memory_cert.count")
    value = exact_keys(records[0], MEMORY_CERT_KEYS, "memory_cert")
    exact(value["schema"], "layersplit-memory-breakdown-v1", "memory_cert.schema")
    exact(value["role"], "monov3", "memory_cert.role")
    exact(value["pid"], process_pid, "memory_cert.pid")
    for key in MEMORY_CERT_KEYS - {"schema", "role"}:
        integer(value[key], f"memory_cert.{key}")
    require(value["model_buffer_bytes"] > 0, "E_MEMORY_MODEL_BUFFER")
    require(value["kv_buffer_bytes"] > 0, "E_MEMORY_KV_BUFFER")
    require(value["compute_buffer_bytes"] > 0, "E_MEMORY_COMPUTE_BUFFER")
    return value


def parse_op_map(value: Any, field: str) -> int:
    require(type(value) is dict and bool(value), f"E_OP_MAP: {field}")
    total = 0
    for op in sorted(value):
        backends = value[op]
        require(type(op) is str and bool(op), f"E_OP: {field}")
        require(type(backends) is dict and bool(backends), f"E_BACKEND_MAP: {field}")
        for backend, count in backends.items():
            exact(backend, "CUDA0", f"{field}.{op}.backend")
            total += integer(count, f"{field}.{op}.{backend}", 1)
    return total


def parse_placement(raw: bytes, process_pid: int) -> dict[str, Any]:
    records = parse_prefixed(raw, PLACEMENT_PREFIX, "placement")
    exact(len(records), 1, "placement.count")
    value = exact_keys(records[0], PLACEMENT_KEYS, "placement")
    expected = {
        "layer_end": N_LAYER,
        "layer_start": 0,
        "missing_buffer_compute_nodes": 0,
        "mode": "monov3",
        "n_layer": N_LAYER,
        "pid": process_pid,
        "role": "monov3",
        "run_rc": 0,
        "schema": "layersplit-scheduled-placement-v2",
        "status": "SCHEDULED_PLACEMENT_OK",
    }
    for key, item in expected.items():
        exact(value[key], item, f"placement.{key}")
    compute_by_buffer = value["compute_by_buffer_type"]
    exact(set(compute_by_buffer), {"CUDA0"}, "placement.backends")
    buffer_count = integer(
        compute_by_buffer["CUDA0"],
        "placement.compute_by_buffer_type.CUDA0",
        1,
    )
    count = parse_op_map(value["compute_by_op_and_buffer"], "placement.ops")
    compute_by_op = value["compute_by_op"]
    require(type(compute_by_op) is dict and bool(compute_by_op), "E_PLACEMENT_OPS")
    op_count = sum(
        integer(item, f"placement.compute_by_op.{op}", 1)
        for op, item in compute_by_op.items()
    )
    exact(value["compute_nodes"], count, "placement.compute_nodes")
    exact(buffer_count, count, "placement.buffer_count")
    exact(op_count, count, "placement.op_count")
    integer(value["copy_nodes"], "placement.copy_nodes")
    integer(value["metadata_nodes"], "placement.metadata_nodes")
    require(type(value["copy_by_buffer_type"]) is dict, "E_PLACEMENT_COPIES")
    require(count > 0, "E_PLACEMENT_EMPTY")
    return value


def load_corpus(
    pre_dir: Path,
    phase_id: str,
    expected_content_sha256: str,
    expected_role_sha256: str,
) -> tuple[list[dict[str, Any]], str]:
    raw = read_regular(pre_dir / "quality_corpus.jsonl")
    exact(sha256(raw), expected_role_sha256, "corpus.role_sha256")
    lines = raw.splitlines(keepends=True)
    exact(len(lines), 64, "corpus.rows")
    bodies = []
    content = bytearray()
    wrapper_keys = {"acquisition_id", "phase", "phase_id", "role"}
    for index, line in enumerate(lines):
        row = parse_json(line, f"corpus[{index}]")
        require(type(row) is dict, f"E_CORPUS_ROW: {index}")
        exact(canonical_bytes(row), line, f"corpus[{index}].canonical")
        require(wrapper_keys.issubset(row), f"E_CORPUS_WRAPPER: {index}")
        exact(row["acquisition_id"], phase_id, f"corpus[{index}].acquisition")
        exact(row["phase"], PHASE, f"corpus[{index}].phase")
        exact(row["phase_id"], phase_id, f"corpus[{index}].phase_id")
        exact(row["role"], "quality.corpus", f"corpus[{index}].role")
        body = {key: item for key, item in row.items() if key not in wrapper_keys}
        exact_keys(body, CORPUS_KEYS | {"kind"}, f"corpus[{index}].body")
        exact(body["kind"], "item", f"corpus[{index}].kind")
        del body["kind"]
        exact(body["item_index"], index, f"corpus[{index}].item_index")
        choices = body["choices"]
        require(
            type(choices) is list
            and len(choices) == 4
            and all(type(choice) is str for choice in choices),
            f"E_CORPUS_CHOICES: {index}",
        )
        require(body["expected_answer"] in (0, 1, 2, 3), f"E_CORPUS_ANSWER: {index}")
        bodies.append(body)
        content.extend(canonical_bytes(body))
    exact(sha256(bytes(content)), expected_content_sha256, "corpus.content_sha256")
    return bodies, sha256(raw)


def prompt_for(item: dict[str, Any]) -> str:
    return (
        f"Question: {item['question']}\n"
        f"A. {item['choices'][0]}\n"
        f"B. {item['choices'][1]}\n"
        f"C. {item['choices'][2]}\n"
        f"D. {item['choices'][3]}\n"
        "Answer with exactly one uppercase letter: A, B, C, or D.\n"
        "Answer:"
    )


def invoke_codec(
    codec: dict[str, Any],
    requests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    raw_input = b"".join(canonical_bytes(request) for request in requests)
    require(len(raw_input) <= MAX_COMMAND_OUTPUT, "E_CODEC_INPUT")
    try:
        process = subprocess.run(
            codec["argv"],
            cwd=codec["cwd"],
            env=codec["environment"],
            input=raw_input,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=codec["timeout_ms"] / 1000,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise CaptureError("E_CODEC_TIMEOUT") from error
    exact(process.returncode, 0, "codec.returncode")
    exact(process.stderr, b"", "codec.stderr")
    require(0 < len(process.stdout) <= MAX_COMMAND_OUTPUT, "E_CODEC_OUTPUT")
    lines = process.stdout.splitlines(keepends=True)
    exact(len(lines), len(requests), "codec.output_count")
    outputs = []
    for index, (request, line) in enumerate(zip(requests, lines)):
        response = parse_json(line, f"codec[{index}]")
        require(type(response) is dict, f"E_CODEC_RESPONSE: {index}")
        exact(canonical_bytes(response), line, f"codec[{index}].canonical")
        response_key = "text" if request["op"] == "detokenize" else "tokens"
        exact_keys(
            response,
            {"model_sha256", "op", "request_id", "schema", response_key},
            f"codec[{index}]",
        )
        exact(
            response["schema"],
            "layersplit-token-codec-response-v1",
            f"codec[{index}].schema",
        )
        exact(response["model_sha256"], MODEL_SHA256, f"codec[{index}].model")
        exact(response["op"], request["op"], f"codec[{index}].op")
        exact(response["request_id"], request["request_id"], f"codec[{index}].request_id")
        outputs.append(response)
    return outputs


def tokenize_corpus(
    codec: dict[str, Any],
    corpus: list[dict[str, Any]],
) -> tuple[list[list[int]], list[str]]:
    prompts = [prompt_for(item) for item in corpus]
    requests = [
        {
            "op": "tokenize",
            "request_id": index + 1,
            "schema": "layersplit-token-codec-request-v1",
            "text": prompt,
        }
        for index, prompt in enumerate(prompts)
    ]
    responses = invoke_codec(codec, requests)
    histories = []
    for index, response in enumerate(responses):
        tokens = vector(response["tokens"], f"codec.tokens[{index}]")
        require(0 < len(tokens) <= N_CTX_SEQ - 8, f"E_QUALITY_CONTEXT: {index}")
        histories.append(tokens)
    return histories, prompts


def detokenize_outputs(
    codec: dict[str, Any],
    outputs: list[list[int]],
) -> list[str]:
    requests = [
        {
            "op": "detokenize",
            "request_id": index + 65,
            "schema": "layersplit-token-codec-request-v1",
            "tokens": tokens,
        }
        for index, tokens in enumerate(outputs)
    ]
    responses = invoke_codec(codec, requests)
    result = []
    for index, response in enumerate(responses):
        require(type(response["text"]) is str, f"E_CODEC_TEXT: {index}")
        result.append(response["text"])
    return result


def run_command(argv: list[str], timeout_ms: int) -> bytes:
    try:
        result = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_ms / 1000,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise CaptureError(f"E_PROBE_TIMEOUT: {argv[0]}") from error
    exact(result.returncode, 0, "probe.returncode")
    exact(result.stderr, b"", "probe.stderr")
    require(len(result.stdout) <= MAX_COMMAND_OUTPUT, "E_PROBE_OUTPUT")
    return result.stdout


def read_swap() -> tuple[int, bytes]:
    try:
        with open("/proc/meminfo", "rb", buffering=0) as source:
            raw = source.read(1024 * 1024 + 1)
    except OSError as error:
        raise CaptureError("E_SWAP_READ") from error
    require(0 < len(raw) <= 1024 * 1024, "E_SWAP_READ")
    values = {}
    for line in raw.decode("ascii").splitlines():
        match = re.fullmatch(r"(SwapTotal|SwapFree):\s+([0-9]+)\s+kB", line)
        if match:
            require(match.group(1) not in values, "E_SWAP_DUPLICATE")
            values[match.group(1)] = int(match.group(2)) * 1024
    exact(set(values), {"SwapFree", "SwapTotal"}, "swap.fields")
    require(0 <= values["SwapFree"] <= values["SwapTotal"], "E_SWAP_RANGE")
    return values["SwapTotal"] - values["SwapFree"], raw


def capture_model_mapping(process_pid: int, plan: dict[str, Any]) -> dict[str, Any]:
    model = plan["model_artifact"]
    model_path = Path(model["path"]).resolve(strict=True)
    model_stat = model_path.stat()
    raw = read_regular(Path(f"/proc/{process_pid}/maps"), MAX_LOG_FILE)
    mappings = []
    for line in raw.decode("utf-8").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6 or fields[5].startswith("["):
            continue
        require(not fields[5].endswith(" (deleted)"), "E_MODEL_MAP_DELETED")
        if Path(fields[5]).resolve(strict=False) != model_path:
            continue
        device_major, device_minor = (
            int(item, 16)
            for item in fields[3].split(":", 1)
        )
        inode = int(fields[4])
        require(
            device_major == os.major(model_stat.st_dev)
            and device_minor == os.minor(model_stat.st_dev)
            and inode == model_stat.st_ino,
            "E_MODEL_MAP_IDENTITY",
        )
        mappings.append({
            "address_range": fields[0],
            "device_major": device_major,
            "device_minor": device_minor,
            "inode": inode,
            "offset_bytes": int(fields[2], 16),
            "path": str(model_path),
            "permissions": fields[1],
        })
    require(bool(mappings), "E_MODEL_MAP_MISSING")
    return {
        "model_mapping_rows": mappings,
        "model_path": str(model_path),
        "model_sha256": model["sha256"],
        "other_gguf_mapping_paths": [],
        "pid": process_pid,
    }


def parse_device_probe(raw: bytes) -> tuple[int, int]:
    lines = raw.decode("ascii").splitlines()
    exact(len(lines), 1, "nvidia.device.lines")
    fields = [field.strip() for field in lines[0].split(",")]
    exact(len(fields), 4, "nvidia.device.fields")
    exact(fields[0], CUDA_NAME, "nvidia.device.name")
    exact(fields[1], CUDA_UUID, "nvidia.device.uuid")
    require(fields[2].isdigit() and fields[3].isdigit(), "E_NVIDIA_DEVICE_INTEGER")
    total = int(fields[2]) * 1024 * 1024
    used = int(fields[3]) * 1024 * 1024
    exact(total, CUDA_MEMORY_TOTAL, "nvidia.device.total")
    require(0 <= used <= total, "E_NVIDIA_DEVICE_USED")
    return total, used


def parse_process_probe(raw: bytes) -> dict[int, int]:
    result = {}
    for index, line in enumerate(raw.decode("ascii").splitlines()):
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        exact(len(fields), 2, f"nvidia.process[{index}].fields")
        require(fields[0].isdigit() and fields[1].isdigit(), "E_NVIDIA_PROCESS_INTEGER")
        pid = int(fields[0])
        used = int(fields[1]) * 1024 * 1024
        require(pid > 0 and used > 0 and pid not in result, "E_NVIDIA_PROCESS_ROW")
        result[pid] = used
    return result


def take_memory_sample(
    plan: dict[str, Any],
    kind: str,
    process_pid: int,
    evidence_dir: Path,
) -> dict[str, Any]:
    require(kind in {"before", "ready", "after"}, "E_SAMPLE_KIND")
    started_ns = monotonic_ns()
    device_raw = run_command(
        plan["nvidia_smi"]["device_argv"],
        plan["nvidia_smi"]["timeout_ms"],
    )
    process_raw = run_command(
        plan["nvidia_smi"]["process_argv"],
        plan["nvidia_smi"]["timeout_ms"],
    )
    swap_used, swap_raw = read_swap()
    timestamp_ns = monotonic_ns()
    require(started_ns <= timestamp_ns, "E_SAMPLE_INTERVAL")
    total, used = parse_device_probe(device_raw)
    processes = parse_process_probe(process_raw)
    if kind == "ready":
        exact(set(processes), {process_pid}, "nvidia.ready.processes")
    else:
        exact(process_pid, 0, f"nvidia.{kind}.expected_pid")
        exact(processes, {}, f"nvidia.{kind}.processes")
    raw_bundle = {
        "device_stdout_sha256": sha256(device_raw),
        "kind": kind,
        "process_stdout_sha256": sha256(process_raw),
        "sample_completed_ns": timestamp_ns,
        "sample_started_ns": started_ns,
        "swap_raw_sha256": sha256(swap_raw),
    }
    durable_write_new(evidence_dir / f"{kind}.device.stdout", device_raw or b"\n")
    durable_write_new(evidence_dir / f"{kind}.process.stdout", process_raw or b"\n")
    durable_write_new(evidence_dir / f"{kind}.meminfo", swap_raw)
    durable_write_new(evidence_dir / f"{kind}.sample.json", canonical_bytes(raw_bundle))
    return {
        "host_swap_used_bytes": swap_used,
        "kind": kind,
        "nvml_process_used_bytes": processes.get(process_pid, 0),
        "sample_id": sha256(canonical_bytes(raw_bundle)),
        "timestamp_ns": timestamp_ns,
        "total_bytes": total,
        "used_bytes": used,
    }


def validate_samples(
    samples: list[dict[str, Any]],
    started_ns: int,
    completed_ns: int,
) -> None:
    exact([sample["kind"] for sample in samples], ["before", "ready", "after"], "samples.kinds")
    times = [integer(sample["timestamp_ns"], "sample.timestamp", 1) for sample in samples]
    require(started_ns <= times[0] < times[1] < times[2] <= completed_ns, "E_SAMPLE_ORDER")
    exact(
        samples[2]["host_swap_used_bytes"] - samples[0]["host_swap_used_bytes"],
        0,
        "sample.swap_growth",
    )


def make_mechanics_rows(
    histories: list[list[int]],
    continuations: list[list[int]],
    calls: list[dict[str, Any]],
    program_sha256: str,
    event_ns: int,
) -> list[dict[str, Any]]:
    require(len(calls) >= 8, "E_MECHANICS_CALLS")
    require(calls[-7:] and all(call["phase"] == "decode" for call in calls[-7:]),
            "E_MECHANICS_DECODE_CALLS")
    rows = [{
        "backend": "CUDA0",
        "call_shapes": calls,
        "event_ns": event_ns,
        "kind": "meta",
        "model_id": MODEL_ID,
        "model_sha256": MODEL_SHA256,
        "program_sha256": program_sha256,
        "state_count_after": 0,
        "state_count_before": 0,
    }]
    for sequence in range(BATCH):
        rows.append({
            "continuation_tokens": vector(
                continuations[sequence],
                f"continuation[{sequence}]",
                8,
            ),
            "event_ns": event_ns,
            "input_tokens": vector(histories[sequence], f"history[{sequence}]"),
            "kind": "request",
            "model_id": MODEL_ID,
            "model_sha256": MODEL_SHA256,
            "owner_after": "RELEASED",
            "owner_before": "CUDA",
            "ownership_epoch_after": 2,
            "ownership_epoch_before": 1,
            "positions": list(range(len(histories[sequence]))),
            "request_id": sequence + 1,
        })
    return rows


def make_quality_rows(
    corpus: list[dict[str, Any]],
    outputs: list[str],
    corpus_sha256: str,
    event_ns: list[int],
    phase_id: str,
) -> list[dict[str, Any]]:
    exact(len(corpus), 64, "quality.corpus_count")
    exact(len(outputs), 64, "quality.output_count")
    exact(len(event_ns), 64, "quality.event_count")
    rows = []
    for index, (item, raw_output) in enumerate(zip(corpus, outputs)):
        wrapper = {
            "acquisition_id": phase_id,
            "kind": "item",
            "role": "quality.corpus",
            **item,
        }
        rows.append({
            "corpus_item_sha256": sha256(canonical_bytes(wrapper)),
            "corpus_sha256": corpus_sha256,
            "event_ns": event_ns[index],
            "item_index": index,
            "kind": "output",
            "model_id": MODEL_ID,
            "model_sha256": MODEL_SHA256,
            "prompt_sha256": sha256(prompt_for(item).encode("utf-8")),
            "raw_output": raw_output,
        })
    return rows


def make_memory_rows(
    samples: list[dict[str, Any]],
    memory_cert: dict[str, int],
    placement: dict[str, Any],
    sampler_sha256: str,
) -> list[dict[str, Any]]:
    model_bytes = memory_cert["model_buffer_bytes"]
    kv_bytes = memory_cert["kv_buffer_bytes"]
    accounted = model_bytes + kv_bytes
    full_device_allocation = accounted + memory_cert["compute_buffer_bytes"]
    ready_nvml = samples[1]["nvml_process_used_bytes"]
    require(full_device_allocation <= ready_nvml, "E_MEMORY_CERT_NVML")
    rows = []
    for sample in samples:
        ready = sample["kind"] == "ready"
        rows.append({
            "batch": BATCH if ready else 0,
            "clock_id": CLOCK_NAME,
            "completed_requests": BATCH if ready else 0,
            "config_sha256": (
                sha256(canonical_bytes(SERVING_ENVELOPE))
                if ready
                else "NONE"
            ),
            "device_name": CUDA_NAME,
            "device_uuid": CUDA_UUID,
            "event_ns": sample["timestamp_ns"],
            "free_bytes": sample["total_bytes"] - sample["used_bytes"],
            "host_swap_used_bytes": sample["host_swap_used_bytes"],
            "kind": sample["kind"],
            "kv_buffer_bytes": kv_bytes if ready else 0,
            "memory_total_bytes": sample["total_bytes"],
            "model_buffer_bytes": model_bytes if ready else 0,
            "model_id": MODEL_ID,
            "model_sha256": MODEL_SHA256,
            "placement_compute_nodes": placement["compute_nodes"] if ready else 0,
            "process_pid": memory_cert["pid"] if ready else 0,
            "process_used_bytes": sample["nvml_process_used_bytes"] if ready else 0,
            "sample_id": sample["sample_id"],
            "sampler_sha256": sampler_sha256,
            "state_count": BATCH if ready else 0,
            "timestamp_ns": sample["timestamp_ns"],
            "used_bytes": sample["used_bytes"],
        })
    require(rows[1]["free_bytes"] >= CUDA_MINIMUM_FREE, "E_CUDA_HEADROOM")
    exact(
        rows[2]["host_swap_used_bytes"] - rows[0]["host_swap_used_bytes"],
        0,
        "memory.swap_growth",
    )
    return rows


def normalized_memory_ready(
    ready: dict[str, Any],
    phase_id: str,
) -> dict[str, Any]:
    result = {
        "acquisition_id": phase_id,
        **ready,
        "role": f"model.{MODEL_ID}.cuda_memory",
    }
    for key in (
        "event_ns",
        "phase",
        "phase_id",
        "process_pid",
        "process_used_bytes",
        "sample_id",
        "sampler_sha256",
    ):
        result.pop(key, None)
    return result


def start_worker(
    plan: dict[str, Any],
    log_path: Path,
) -> tuple[subprocess.Popen[bytes], Any, int]:
    require(not log_path.exists(), "E_WORKER_LOG_EXISTS")
    log_file = log_path.open("xb")
    started_ns = monotonic_ns()
    try:
        process = subprocess.Popen(
            plan["worker"]["argv"],
            cwd=plan["worker"]["cwd"],
            env=plan["worker"]["environment"],
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except BaseException:
        log_file.close()
        raise
    return process, log_file, started_ns


def stop_process(
    process: subprocess.Popen[bytes],
    log_file: Any,
    timeout_ms: int,
) -> int:
    try:
        code = process.wait(timeout=timeout_ms / 1000)
    except subprocess.TimeoutExpired as error:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise CaptureError("E_WORKER_STOP_TIMEOUT") from error
    completed_ns = monotonic_ns()
    log_file.flush()
    os.fsync(log_file.fileno())
    log_file.close()
    exact(code, 0, "worker.returncode")
    return completed_ns


def kill_process(
    process: subprocess.Popen[bytes] | None,
    log_file: Any,
) -> None:
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    if log_file is not None and not log_file.closed:
        log_file.flush()
        log_file.close()


def execute(
    plan: dict[str, Any],
    history: dict[str, Any],
    route_epoch: int,
    corpus: list[dict[str, Any]],
    evidence_dir: Path,
    worker_log: Path,
    host_boot_id: str,
    expected_components: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    for item_index, (item, request) in enumerate(
        zip(corpus, history["requests"])
    ):
        exact(
            request["prompt_sha256"],
            sha256(prompt_for(item).encode("utf-8")),
            f"quality.prompt_sha256[{item_index}]",
        )
    load_start_ns = monotonic_ns()
    before = take_memory_sample(plan, "before", 0, evidence_dir)
    process: subprocess.Popen[bytes] | None = None
    log_file = None
    connection: socket.socket | None = None
    try:
        process, log_file, worker_started_ns = start_worker(plan, worker_log)
        require(load_start_ns < worker_started_ns, "E_LOAD_START_ORDER")
        client = connect(plan, process)
        connection = client.connection
        protocol_identity = client.hello(plan)
        runtime_observed_ns = wait_for_prefix(
            worker_log,
            RUNTIME_PREFIX,
            process,
            plan["worker"]["startup_timeout_ms"],
        )
        wait_for_prefix(
            worker_log,
            MEMORY_PREFIX,
            process,
            plan["worker"]["startup_timeout_ms"],
        )
        runtime_model_binding = capture_model_mapping(process.pid, plan)
        exact(client.status()[0], 0, "mechanics.state_before")

        mechanics_group = history["mechanics_b8"]
        mechanics_histories = [
            history["requests"][item_index]["token_ids"]
            for item_index in mechanics_group["item_indices"]
        ]
        request_ids = list(range(1001, 1009))
        continuations, calls, mechanics_receipt = run_history_group(
            client,
            history,
            mechanics_group,
            request_ids,
            route_epoch,
            0,
        )
        mechanics_event_ns = monotonic_ns()
        exact(client.status()[0], BATCH, "mechanics.state_ready")
        ready = take_memory_sample(plan, "ready", process.pid, evidence_dir)
        exact(client.status()[0], BATCH, "mechanics.state_ready_after_sample")
        cuda_ready_ns = monotonic_ns()
        require(ready["timestamp_ns"] <= cuda_ready_ns, "E_CUDA_READY_ORDER")
        remove_group(client, request_ids, route_epoch)

        quality_tokens = list(continuations)
        quality_events = [mechanics_event_ns] * BATCH
        execution_groups = [mechanics_receipt]
        frame_offset = len(mechanics_receipt["call_receipts"])
        for cohort in range(1, 8):
            begin = cohort * BATCH
            cohort_ids = list(range(2001 + begin, 2001 + begin + BATCH))
            cohort_outputs, _, cohort_receipt = run_history_group(
                client,
                history,
                history["quality_groups"][cohort],
                cohort_ids,
                route_epoch,
                frame_offset,
            )
            cohort_event_ns = monotonic_ns()
            quality_tokens.extend(cohort_outputs)
            quality_events.extend([cohort_event_ns] * BATCH)
            execution_groups.append(cohort_receipt)
            frame_offset += len(cohort_receipt["call_receipts"])
            remove_group(client, cohort_ids, route_epoch)
        exact(client.status()[0], 0, "quality.state_after")
        quality_outputs = detokenize_outputs(plan["codec"], quality_tokens)
        client.stop()
        connection.close()
        connection = None
        worker_completed_ns = stop_process(
            process,
            log_file,
            plan["worker"]["shutdown_timeout_ms"],
        )
        process_pid = process.pid
        process = None
        log_file = None
        after = take_memory_sample(plan, "after", 0, evidence_dir)
        completed_ns = monotonic_ns()
        require(worker_completed_ns <= after["timestamp_ns"] <= completed_ns, "E_AFTER_ORDER")

        raw_log = read_regular(worker_log, MAX_LOG_FILE)
        runtime = parse_runtime_process(
            raw_log,
            plan,
            host_boot_id,
            worker_started_ns,
            worker_completed_ns,
            runtime_observed_ns,
            process_pid,
            expected_components,
        )
        exact(runtime_model_binding["pid"], runtime["pid"],
              "runtime_model_binding.pid")
        runtime_model_binding["start_ticks"] = runtime["start_ticks"]
        memory_cert = parse_memory_cert(raw_log, process_pid)
        placement = parse_placement(raw_log, process_pid)
        samples = [before, ready, after]
        validate_samples(samples, load_start_ns, completed_ns)
        return {
            "calls": calls,
            "completed_ns": completed_ns,
            "continuations": continuations,
            "cuda_ready_ns": cuda_ready_ns,
            "execution_groups": execution_groups,
            "histories": mechanics_histories,
            "load_start_ns": load_start_ns,
            "mechanics_event_ns": mechanics_event_ns,
            "memory_cert": memory_cert,
            "placement": placement,
            "protocol_identity": protocol_identity,
            "quality_events": quality_events,
            "quality_outputs": quality_outputs,
            "runtime_process": runtime,
            "runtime_model_binding": runtime_model_binding,
            "samples": samples,
            "worker_started_ns": worker_started_ns,
        }
    finally:
        if connection is not None:
            connection.close()
        kill_process(process, log_file)


def capture(args: argparse.Namespace) -> dict[str, Any]:
    require(args.execute is True, "E_EXECUTE_GATE")
    exact(args.confirm, "RUN_V24_CUDA_ROUTE_A_ONLY", "confirm")
    started_ns = monotonic_ns()
    acquisition_started_ns = integer(args.started, "started", 1)
    require(acquisition_started_ns <= started_ns, "E_ACQUISITION_ORDER")
    phase_id = text(args.phase_id, "phase_id", 128)
    require(
        phase_id.startswith("cp0-r1-v24-a-only-")
        and all(character.isalnum() or character in ".-_" for character in phase_id),
        "E_PHASE_ID",
    )
    command_plan_sha256 = digest(args.plan, "plan")
    supplied_mechanism_sha256 = digest(
        args.mechanism_commands_sha256,
        "mechanism_commands_sha256",
    )
    exact(digest(args.model_sha256, "model_sha256"), MODEL_SHA256, "model_sha256")
    pre_dir = Path(args.pre_dir)
    require(pre_dir.is_absolute() and pre_dir.is_dir(), "E_PRE_DIR")
    output = Path(args.output)
    require(output.is_absolute() and not output.exists(), "E_OUTPUT")
    evidence_dir = output.with_suffix(output.suffix + ".evidence")
    require(not evidence_dir.exists(), "E_EVIDENCE_DIR")

    artifact, fresh_cuda, phase_lock = load_base_evidence(pre_dir, phase_id)
    histories_path = Path(args.histories)
    require(histories_path.is_absolute(), "E_HISTORY_PATH")
    history, histories_raw = load_histories(histories_path)
    plan, plan_raw = load_plan(
        Path(args.launch_plan),
        histories_path,
        histories_raw,
        artifact,
    )
    route_epoch = integer(plan["route_epoch"], "launch_plan.route_epoch", 1)
    runtime_components = load_runtime_evidence(pre_dir, phase_id, plan)
    mechanism_sha256 = bind_mechanism_commands(
        plan,
        supplied_mechanism_sha256,
    )
    corpus, corpus_raw_sha256 = load_corpus(
        pre_dir,
        phase_id,
        plan["quality_corpus_content_sha256"],
        phase_lock["quality_corpus_sha256"],
    )
    source_raw = read_regular(Path(__file__).resolve(), MAX_SMALL_FILE)
    source_sha256 = sha256(source_raw)
    program_sha256 = sha256(
        b"s39:v24:cuda-route-capture:v1\0"
        + bytes.fromhex(command_plan_sha256)
        + bytes.fromhex(source_sha256)
        + bytes.fromhex(sha256(plan_raw))
        + bytes.fromhex(sha256(histories_raw))
    )
    sampler_sha256 = sha256(canonical_bytes({
        "device_argv": plan["nvidia_smi"]["device_argv"],
        "nvidia_executable_sha256": plan["nvidia_smi"]["executable"]["sha256"],
        "process_argv": plan["nvidia_smi"]["process_argv"],
        "source_sha256": source_sha256,
    }))
    result = execute(
        plan,
        history,
        route_epoch,
        corpus,
        evidence_dir,
        output.with_suffix(output.suffix + ".worker.log"),
        fresh_cuda["host_boot_id"],
        runtime_components,
    )
    completed_ns = result["completed_ns"]
    require(started_ns < completed_ns, "E_CAPTURE_INTERVAL")
    memory_rows = make_memory_rows(
        result["samples"],
        result["memory_cert"],
        result["placement"],
        sampler_sha256,
    )
    raw_memory_samples = []
    for row in memory_rows:
        path = evidence_dir / f"{row['kind']}.memory-row.json"
        durable_write_new(path, canonical_bytes(row))
        raw_memory_samples.append({
            **artifact_record(path),
            "row": row,
        })
    cuda_route_rows = make_mechanics_rows(
        result["histories"],
        result["continuations"],
        result["calls"],
        program_sha256,
        result["mechanics_event_ns"],
    )
    quality_rows = make_quality_rows(
        corpus,
        result["quality_outputs"],
        corpus_raw_sha256,
        result["quality_events"],
        phase_id,
    )
    bridge_start = {
        "clock_id": CLOCK_NAME,
        "event_ns": result["load_start_ns"],
        "kind": "cuda_load_start",
        "model_id": MODEL_ID,
        "model_sha256": MODEL_SHA256,
        "timestamp_ns": result["load_start_ns"],
    }
    bridge_ready = {
        "clock_id": CLOCK_NAME,
        "cuda_memory_ready_sha256": sha256(
            canonical_bytes(normalized_memory_ready(memory_rows[1], phase_id))
        ),
        "event_ns": result["cuda_ready_ns"],
        "kind": "cuda_ready",
        "model_id": MODEL_ID,
        "model_sha256": MODEL_SHA256,
        "timestamp_ns": result["cuda_ready_ns"],
    }
    require(
        bridge_start["timestamp_ns"] < result["worker_started_ns"],
        "E_BRIDGE_START_ORDER",
    )
    require(
        memory_rows[1]["timestamp_ns"] <= bridge_ready["timestamp_ns"],
        "E_BRIDGE_READY_ORDER",
    )
    runtime_process = result["runtime_process"]
    gpu_runtime = {
        "artifact_path": artifact["path"],
        "gpu_uuid": CUDA_UUID,
        "host_boot_id": fresh_cuda["host_boot_id"],
        "model_id": MODEL_ID,
        "route_epoch": route_epoch,
    }
    worker_log = output.with_suffix(output.suffix + ".worker.log")
    evidence_paths = sorted(
        [
            Path(__file__).resolve(),
            Path(args.launch_plan),
            histories_path,
            worker_log,
            *[path for path in evidence_dir.iterdir() if path.is_file()],
        ],
        key=lambda path: str(path),
    )
    evidence_artifacts = [artifact_record(path) for path in evidence_paths]
    output_value = {
        "bridge_ready_row": bridge_ready,
        "bridge_start_row": bridge_start,
        "completed_ns": completed_ns,
        "cuda_memory_rows": memory_rows,
        "cuda_route_rows": cuda_route_rows,
        "evidence_artifacts": evidence_artifacts,
        "execution_groups": result["execution_groups"],
        "gpu_runtime": gpu_runtime,
        "history_sha256": sha256(histories_raw),
        "launch_plan_sha256": sha256(plan_raw),
        "mechanism_commands_sha256": mechanism_sha256,
        "memory_certificate": result["memory_cert"],
        "model_id": MODEL_ID,
        "model_sha256": MODEL_SHA256,
        "phase_id": phase_id,
        "placement_certificate": result["placement"],
        "producer_sha256": source_sha256,
        "protocol_identity": result["protocol_identity"],
        "quality_cuda_rows": quality_rows,
        "raw_memory_samples": raw_memory_samples,
        "route_epoch": route_epoch,
        "runtime_model_binding": result["runtime_model_binding"],
        "runtime_process": runtime_process,
        "schema": SCHEMA,
        "started_ns": started_ns,
    }
    durable_write_new(output, canonical_bytes(output_value))
    return output_value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase-id", required=True)
    parser.add_argument("--pre-dir", required=True)
    parser.add_argument("--started", type=int, required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--mechanism-commands-sha256", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--histories", required=True)
    parser.add_argument("--launch-plan", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    args = parser.parse_args()
    try:
        capture(args)
        return 0
    except (
        CaptureError,
        OSError,
        struct.error,
        subprocess.SubprocessError,
        ValueError,
    ) as error:
        print(
            f"A_ONLY_CUDA_ROUTE_REFUSED: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
