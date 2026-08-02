#!/usr/bin/python3 -I
"""Capture one real Qwen3-14B OP15-to-OP12 StageV3 route."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any


SCHEMA = "s39-cp0-r1-v24-phone-route-raw-v1"
PLAN_SCHEMA = "s39-cp0-r1-v24-phone-route-launch-v1"
HISTORY_SCHEMA = "s39-cp0-r1-token-history-v2.4"
PROBE_SCHEMA = "s39-cp0-r1-v24-phone-runtime-probe-v1"
MODEL_ID = "qwen3-14b-q4_k_m"
PHASE = "A_ONLY"
MODEL_SHA256 = (
    "500a8806e85ee9c83f3ae084202955924"
    "51379b4f8cf2d0f41c15dffeb6b81f0"
)
N_LAYER = 40
N_EMBD = 5120
BATCH = 8
N_CTX_SEQ = 512
N_BATCH = 64
N_UBATCH = 64
MAX_STREAMS = 8
FILE_TYPE = 15
OP15_STORED = [0, 32]
OP15_EXECUTED = [0, 30]
OP12_STORED = [24, 40]
OP12_EXECUTED = [30, 40]
OP15_SHARD_SHA256 = (
    "ba56b9c5e19b3a4512777e6a47803cc"
    "03261c2d3c2734965cd5ec96b7c6c59fb"
)
OP12_SHARD_SHA256 = (
    "72e312af745160dc33a0ba39ba94fbbc"
    "e6112950d0409d39c42ddc3b25e756ab"
)
CLOCK_NAME = "HOST_MONOTONIC_RAW"
CLOCK_ID = time.CLOCK_MONOTONIC_RAW

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
STAGE_V3_REQUIRED_CAPABILITIES = 0x3F

MAX_SMALL_FILE = 8 * 1024 * 1024
MAX_LOG_FILE = 64 * 1024 * 1024
MAX_CODEC_OUTPUT = 16 * 1024 * 1024
MAX_PROCESS_OUTPUT = 4 * 1024 * 1024

PHONE_KEYS = {
    "boot_id",
    "device",
    "direct_peer_ipv4",
    "executed_layers",
    "expected_worker_executable_path",
    "expected_worker_executable_sha256",
    "interface",
    "loaded_shard_path",
    "loaded_shard_sha256",
    "local_ipv4",
    "model",
    "product",
    "serial",
    "stored_layers",
}
COMMAND_KEYS = {
    "argv",
    "cwd",
    "environment",
    "launcher_bytes",
    "launcher_sha256",
    "runtime_component_ids",
    "runtime_executable_path",
    "runtime_executable_sha256",
    "shutdown_timeout_ms",
    "startup_timeout_ms",
}
CODEC_KEYS = {
    "argv",
    "cwd",
    "environment",
    "executable_bytes",
    "executable_sha256",
    "timeout_ms",
}
PROBE_KEYS = {
    "after_argv",
    "before_argv",
    "cwd",
    "environment",
    "launcher_bytes",
    "launcher_sha256",
    "timeout_ms",
}
PROBE_ROW_KEYS = {
    "active_sequences",
    "available_bytes",
    "boot_id",
    "device",
    "direct_peer",
    "gpu_max_millic",
    "interface",
    "loaded_shard_path",
    "loaded_shard_sha256",
    "model",
    "model_id",
    "model_sha256",
    "process_swap_bytes",
    "product",
    "schema",
    "serial",
    "system_swap_used_bytes",
    "thermal_status",
    "worker_executable_path",
    "worker_executable_sha256",
    "worker_pid",
    "worker_start_ticks",
}
PLAN_KEYS = {
    "codec",
    "expected_file_type",
    "expected_max_streams",
    "expected_n_batch",
    "expected_n_ctx_seq",
    "expected_n_embd",
    "expected_n_layer",
    "expected_n_ubatch",
    "history_path",
    "history_sha256",
    "mechanism_commands",
    "model_id",
    "model_sha256",
    "phones",
    "probes",
    "processes",
    "quality_corpus_content_sha256",
    "relay_host",
    "relay_port",
    "route_epoch",
    "schema",
}
SESSION_KEYS = {
    "compute_by_op_and_buffer",
    "device_boot_id",
    "expected_backend",
    "layer_end",
    "layer_start",
    "missing_buffer_compute_nodes",
    "n_layer",
    "placement_status",
    "proto_version",
    "reset_applied",
    "schema",
    "session_end",
    "session_id",
    "steps_session",
    "steps_total",
    "worker_boot_nonce",
    "worker_pid",
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
DIRECT_CERT_KEYS = {
    "activation_payload_bytes",
    "batches",
    "cut_layer",
    "file_type",
    "head_endpoint",
    "host_activation_payload_bytes",
    "layer_end",
    "layer_start",
    "model_sha256",
    "n_embd",
    "n_layer",
    "rows",
    "run_rc",
    "schema",
    "status",
    "tail_endpoint",
}
DIRECT_FRAME_KEYS = {
    "activation_payload_bytes",
    "call_index",
    "hidden_width",
    "payload_sha256",
    "positions",
    "request_ids",
    "route_epochs",
    "rows",
    "schema",
    "seq_ids",
}
CORPUS_BODY_KEYS = {
    "choices",
    "dataset",
    "dataset_revision",
    "expected_answer",
    "item_index",
    "question",
    "source_row",
    "subject",
}
WRAPPER_KEYS = {"acquisition_id", "phase", "phase_id", "role"}
RUNTIME_PROCESS_KEYS = {
    "boot_id",
    "bundle_id",
    "endpoint",
    "launcher_path",
    "loaded_repo_component_ids",
    "observed_ns",
    "pid",
    "start_ticks",
    "system_dependencies",
}
SYSTEM_DEPENDENCY_KEYS = {
    "build_id",
    "ctime_ns",
    "device_id",
    "inode",
    "mode",
    "mtime_ns",
    "path",
    "size",
}


class CaptureError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CaptureError(message)


def exact(value: Any, expected: Any, field: str) -> None:
    require(
        type(value) is type(expected) and value == expected,
        f"E_VALUE: {field}",
    )


def exact_keys(value: Any, keys: set[str], field: str) -> dict[str, Any]:
    require(type(value) is dict, f"E_TYPE: {field}")
    require(set(value) == keys, f"E_KEYS: {field}")
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


def integer(value: Any, field: str, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, f"E_INTEGER: {field}")
    return value


def text(value: Any, field: str, maximum: int = 4096) -> str:
    require(type(value) is str and 0 < len(value) <= maximum, f"E_TEXT: {field}")
    require("\x00" not in value and "\n" not in value, f"E_TEXT: {field}")
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
    minimum: int = 0,
) -> list[int]:
    require(type(value) is list, f"E_VECTOR: {field}")
    if length is not None:
        require(len(value) == length, f"E_VECTOR_LENGTH: {field}")
    for index, item in enumerate(value):
        integer(item, f"{field}[{index}]", minimum)
    return value


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
        while block := os.read(descriptor, min(1024 * 1024, maximum + 1)):
            raw.extend(block)
            require(len(raw) <= maximum, f"E_FILE_SIZE: {path}")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    require(_identity(before) == _identity(after), f"E_FILE_CHANGED: {path}")
    require(len(raw) == before.st_size, f"E_FILE_CHANGED: {path}")
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
    require(canonical_bytes(value) == raw, f"E_CANONICAL: {path}")
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
    launcher = Path(value[0])
    require(launcher.is_absolute(), f"E_LAUNCHER_PATH: {field}")
    require(
        launcher.name not in {"bash", "dash", "sh", "zsh"}
        and "-c" not in value,
        f"E_SHELL_COMMAND: {field}",
    )
    return value


def validate_environment(value: Any, field: str) -> dict[str, str]:
    require(type(value) is dict, f"E_ENV: {field}")
    for key, item in value.items():
        text(key, f"{field}.key", 128)
        text(item, f"{field}.{key}")
        require("=" not in key, f"E_ENV_KEY: {field}.{key}")
    return value


def verify_local_executable(
    path: Path,
    expected_bytes: int,
    expected_sha256: str,
    field: str,
) -> None:
    raw = read_regular(path, 256 * 1024 * 1024)
    exact(len(raw), expected_bytes, f"{field}.bytes")
    exact(sha256(raw), expected_sha256, f"{field}.sha256")
    require(
        os.stat(path, follow_symlinks=False).st_mode & 0o111 != 0,
        f"E_EXECUTABLE_MODE: {field}",
    )


def validate_command(value: Any, field: str) -> dict[str, Any]:
    value = exact_keys(value, COMMAND_KEYS, field)
    argv = validate_argv(value["argv"], f"{field}.argv")
    cwd = Path(text(value["cwd"], f"{field}.cwd"))
    require(cwd.is_absolute() and cwd.is_dir(), f"E_CWD: {field}")
    validate_environment(value["environment"], f"{field}.environment")
    expected_bytes = integer(value["launcher_bytes"], f"{field}.bytes", 1)
    expected_sha256 = digest(value["launcher_sha256"], f"{field}.sha256")
    verify_local_executable(
        Path(argv[0]),
        expected_bytes,
        expected_sha256,
        field,
    )
    runtime_path = text(
        value["runtime_executable_path"],
        f"{field}.runtime_executable_path",
    )
    require(Path(runtime_path).is_absolute(), f"E_RUNTIME_PATH: {field}")
    digest(
        value["runtime_executable_sha256"],
        f"{field}.runtime_executable_sha256",
    )
    component_ids = value["runtime_component_ids"]
    require(
        type(component_ids) is list
        and bool(component_ids)
        and component_ids == sorted(set(component_ids))
        and all(
            type(component_id) is str
            and 0 < len(component_id) <= 128
            for component_id in component_ids
        ),
        f"E_RUNTIME_COMPONENTS: {field}",
    )
    for name in ("shutdown_timeout_ms", "startup_timeout_ms"):
        timeout = integer(value[name], f"{field}.{name}", 1)
        require(timeout <= 600_000, f"E_TIMEOUT: {field}.{name}")
    return value


def validate_probe(value: Any, field: str) -> dict[str, Any]:
    value = exact_keys(value, PROBE_KEYS, field)
    validate_argv(value["before_argv"], f"{field}.before")
    validate_argv(value["after_argv"], f"{field}.after")
    exact(value["before_argv"][0], value["after_argv"][0], f"{field}.launcher")
    expected_bytes = integer(value["launcher_bytes"], f"{field}.bytes", 1)
    expected_sha256 = digest(value["launcher_sha256"], f"{field}.sha256")
    verify_local_executable(
        Path(value["before_argv"][0]),
        expected_bytes,
        expected_sha256,
        field,
    )
    cwd = Path(text(value["cwd"], f"{field}.cwd"))
    require(cwd.is_absolute() and cwd.is_dir(), f"E_CWD: {field}")
    validate_environment(value["environment"], f"{field}.environment")
    timeout = integer(value["timeout_ms"], f"{field}.timeout", 1)
    require(timeout <= 60_000, f"E_TIMEOUT: {field}")
    return value


def validate_codec(value: Any) -> dict[str, Any]:
    value = exact_keys(value, CODEC_KEYS, "codec")
    argv = validate_argv(value["argv"], "codec.argv")
    require(
        "--model" in argv
        and "--model-sha256" in argv
        and MODEL_SHA256 in argv,
        "E_CODEC_MODEL_BINDING",
    )
    cwd = Path(text(value["cwd"], "codec.cwd"))
    require(cwd.is_absolute() and cwd.is_dir(), "E_CODEC_CWD")
    validate_environment(value["environment"], "codec.environment")
    expected_bytes = integer(value["executable_bytes"], "codec.bytes", 1)
    expected_sha256 = digest(value["executable_sha256"], "codec.sha256")
    verify_local_executable(
        Path(argv[0]),
        expected_bytes,
        expected_sha256,
        "codec",
    )
    timeout = integer(value["timeout_ms"], "codec.timeout", 1)
    require(timeout <= 600_000, "E_CODEC_TIMEOUT")
    return value


def validate_phone(value: Any, phone: str) -> dict[str, Any]:
    value = exact_keys(value, PHONE_KEYS, f"phones.{phone}")
    expected = {
        "op15": {
            "executed_layers": OP15_EXECUTED,
            "loaded_shard_sha256": OP15_SHARD_SHA256,
            "stored_layers": OP15_STORED,
        },
        "op12": {
            "executed_layers": OP12_EXECUTED,
            "loaded_shard_sha256": OP12_SHARD_SHA256,
            "stored_layers": OP12_STORED,
        },
    }[phone]
    for key, item in expected.items():
        exact(value[key], item, f"phones.{phone}.{key}")
    for key in (
        "boot_id",
        "device",
        "direct_peer_ipv4",
        "expected_worker_executable_path",
        "interface",
        "loaded_shard_path",
        "local_ipv4",
        "model",
        "product",
        "serial",
    ):
        text(value[key], f"phones.{phone}.{key}")
    digest(
        value["expected_worker_executable_sha256"],
        f"phones.{phone}.worker_sha256",
    )
    return value


def load_plan(path: Path) -> tuple[dict[str, Any], bytes]:
    value, raw = read_canonical(path)
    exact_keys(value, PLAN_KEYS, "launch_plan")
    exact(value["schema"], PLAN_SCHEMA, "launch_plan.schema")
    exact(value["model_id"], MODEL_ID, "launch_plan.model_id")
    exact(value["model_sha256"], MODEL_SHA256, "launch_plan.model_sha256")
    exact(value["expected_file_type"], FILE_TYPE, "launch_plan.file_type")
    exact(value["expected_n_layer"], N_LAYER, "launch_plan.n_layer")
    exact(value["expected_n_embd"], N_EMBD, "launch_plan.n_embd")
    exact(value["expected_max_streams"], MAX_STREAMS, "launch_plan.streams")
    exact(value["expected_n_ctx_seq"], N_CTX_SEQ, "launch_plan.ctx")
    exact(value["expected_n_batch"], N_BATCH, "launch_plan.batch")
    exact(value["expected_n_ubatch"], N_UBATCH, "launch_plan.ubatch")
    integer(value["route_epoch"], "launch_plan.route_epoch", 1)
    history_path = Path(text(value["history_path"], "launch_plan.history_path"))
    require(history_path.is_absolute(), "E_HISTORY_PATH")
    digest(value["history_sha256"], "launch_plan.history_sha256")
    digest(
        value["quality_corpus_content_sha256"],
        "launch_plan.quality_corpus_content_sha256",
    )
    text(value["relay_host"], "launch_plan.relay_host", 255)
    port = integer(value["relay_port"], "launch_plan.relay_port", 1)
    require(port <= 65535, "E_RELAY_PORT")
    exact_keys(value["phones"], {"op12", "op15"}, "launch_plan.phones")
    for phone in ("op15", "op12"):
        validate_phone(value["phones"][phone], phone)
    exact(
        value["phones"]["op15"]["direct_peer_ipv4"],
        value["phones"]["op12"]["local_ipv4"],
        "E_DIRECT_PEER: op15",
    )
    exact(
        value["phones"]["op12"]["direct_peer_ipv4"],
        value["phones"]["op15"]["local_ipv4"],
        "E_DIRECT_PEER: op12",
    )
    processes = exact_keys(
        value["processes"],
        {"op12_stagenet", "op15_direct_relay", "op15_stagenet"},
        "launch_plan.processes",
    )
    for name, command in processes.items():
        validate_command(command, f"processes.{name}")
    probes = exact_keys(
        value["probes"],
        {"op12", "op15"},
        "launch_plan.probes",
    )
    for phone, probe in probes.items():
        validate_probe(probe, f"probes.{phone}")
    validate_codec(value["codec"])
    validate_mechanism_commands(value)
    return value, raw


def derive_mechanism_commands(plan: dict[str, Any]) -> dict[str, list[list[str]]]:
    return {
        "desktop": [
            list(plan["codec"]["argv"]),
        ],
        "op12": [
            list(plan["processes"]["op12_stagenet"]["argv"]),
            list(plan["probes"]["op12"]["before_argv"]),
            list(plan["probes"]["op12"]["after_argv"]),
        ],
        "op15": [
            list(plan["processes"]["op15_stagenet"]["argv"]),
            list(plan["processes"]["op15_direct_relay"]["argv"]),
            list(plan["probes"]["op15"]["before_argv"]),
            list(plan["probes"]["op15"]["after_argv"]),
        ],
    }


def validate_mechanism_commands(plan: dict[str, Any]) -> None:
    value = exact_keys(
        plan["mechanism_commands"],
        {"desktop", "op12", "op15"},
        "mechanism_commands",
    )
    for endpoint in ("desktop", "op12", "op15"):
        commands = value[endpoint]
        require(type(commands) is list and bool(commands),
                f"E_MECHANISM_COMMANDS: {endpoint}")
        for index, argv in enumerate(commands):
            validate_argv(argv, f"mechanism_commands.{endpoint}[{index}]")
    local = derive_mechanism_commands(plan)
    exact(value["op12"], local["op12"], "mechanism_commands.op12")
    exact(value["op15"], local["op15"], "mechanism_commands.op15")
    exact(len(value["desktop"]), 9, "mechanism_commands.desktop_count")
    exact(value["desktop"][0], local["desktop"][0],
          "mechanism_commands.desktop.codec")


def bind_mechanism_commands(
    plan: dict[str, Any],
    supplied_sha256: str,
) -> str:
    validate_mechanism_commands(plan)
    expected = sha256(canonical_bytes(plan["mechanism_commands"]))
    exact(supplied_sha256, expected, "mechanism_commands_sha256")
    return expected


def load_phase(pre_dir: Path, phase_id: str) -> tuple[dict[str, Any], bytes]:
    require(pre_dir.is_absolute() and pre_dir.is_dir(), "E_PRE_DIR")
    raw = read_regular(pre_dir / "phase_lock.jsonl")
    lines = raw.splitlines(keepends=True)
    require(len(lines) == 1, "E_PHASE_LOCK_ROWS")
    row = parse_json(lines[0], "phase_lock")
    require(type(row) is dict, "E_PHASE_LOCK")
    exact(
        set(row),
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
        "phase_lock.keys",
    )
    exact(row["schema"], "s39-cp0-r1-phase-lock-v2.4", "phase_lock.schema")
    exact(row.get("phase"), PHASE, "phase_lock.phase")
    exact(row.get("phase_id"), phase_id, "phase_lock.phase_id")
    exact(row.get("model_id"), MODEL_ID, "phase_lock.model_id")
    for key in (
        "artifact_root_sha256",
        "candidate_sha256",
        "contract_sha256",
        "preparation_sha256",
        "quality_corpus_sha256",
        "runtime_bundle_plan_sha256",
    ):
        digest(row[key], f"phase_lock.{key}")
    boot_ids = exact_keys(
        row["device_boot_ids"],
        {"cuda", "op12", "op15"},
        "phase_lock.device_boot_ids",
    )
    for endpoint, value in boot_ids.items():
        text(value, f"phase_lock.device_boot_ids.{endpoint}", 128)
    integer(row["event_ns"], "phase_lock.event_ns", 1)
    return row, raw


def validate_history_group(
    group: Any,
    group_index: int,
    requests: list[dict[str, Any]],
) -> None:
    field = f"history.quality_groups[{group_index}]"
    exact_keys(
        group,
        {"decode_calls", "group_index", "item_indices", "prefill_partitions"},
        field,
    )
    exact(group["group_index"], group_index, f"{field}.group_index")
    item_indices = list(range(group_index * BATCH, (group_index + 1) * BATCH))
    exact(group["item_indices"], item_indices, f"{field}.item_indices")

    token_rows = []
    for item_index in item_indices:
        request = requests[item_index]
        for position, token_id in enumerate(request["token_ids"]):
            token_rows.append(
                (
                    position,
                    item_index,
                    request["request_id"],
                    request["seq_id"],
                    token_id,
                )
            )
    token_rows.sort(key=lambda row: (row[0], row[1]))
    partitions = []
    current = []
    for position in sorted({row[0] for row in token_rows}):
        wave = [row for row in token_rows if row[0] == position]
        require(len(wave) <= N_UBATCH, f"E_HISTORY_WAVE: {field}")
        if current and len(current) + len(wave) > N_UBATCH:
            partitions.append(current)
            current = []
        current.extend(wave)
    if current:
        partitions.append(current)
    expected_prefill = [
        {
            "call_index": call_index,
            "rows": [
                {
                    "item_index": item_index,
                    "position": position,
                    "request_id": request_id,
                    "seq_id": seq_id,
                    "token_id": token_id,
                }
                for position, item_index, request_id, seq_id, token_id in partition
            ],
        }
        for call_index, partition in enumerate(partitions)
    ]
    exact(group["prefill_partitions"], expected_prefill, f"{field}.prefill")
    expected_decode = [
        {
            "call_index": len(expected_prefill) + ordinal,
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
    exact(group["decode_calls"], expected_decode, f"{field}.decode")


def load_histories(
    path: Path,
    expected_sha256: str,
    model_sha256: str,
) -> tuple[dict[str, Any], bytes]:
    value, raw = read_canonical(path)
    exact(sha256(raw), expected_sha256, "history.sha256")
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
    exact(value["schema"], HISTORY_SCHEMA, "history.schema")
    exact(value["model_id"], MODEL_ID, "history.model")
    exact(value["model_sha256"], model_sha256, "history.model_sha256")
    exact(value["batch"], BATCH, "history.batch")
    exact(value["continuation_tokens_per_request"], 8, "history.continuation")
    exact(value["n_ctx_seq"], N_CTX_SEQ, "history.n_ctx_seq")
    exact(value["n_batch"], N_BATCH, "history.n_batch")
    exact(value["n_ubatch"], N_UBATCH, "history.n_ubatch")
    exact(
        value["prefill_chunking"],
        "WHOLE_POSITION_WAVES_MAX_64_ROWS",
        "history.prefill_chunking",
    )
    exact(
        value["prefill_row_order"],
        "POSITION_MAJOR_THEN_ITEM_INDEX",
        "history.prefill_row_order",
    )
    requests = value["requests"]
    require(type(requests) is list and len(requests) == 64, "E_HISTORY_REQUESTS")
    for item_index, request in enumerate(requests):
        field = f"history.requests[{item_index}]"
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
        tokens = request["token_ids"]
        require(
            type(tokens) is list and 0 < len(tokens) <= N_CTX_SEQ - 8,
            f"E_HISTORY_TOKENS: {item_index}",
        )
        for token_index, token_id in enumerate(tokens):
            integer(token_id, f"{field}.token_ids[{token_index}]")
    groups = value["quality_groups"]
    require(type(groups) is list and len(groups) == 8, "E_HISTORY_GROUPS")
    exact(value["mechanics_b8"], groups[0], "E_HISTORY_MECHANICS_GROUP")
    for group_index, group in enumerate(groups):
        validate_history_group(group, group_index, requests)
    return value, raw


def load_corpus(
    pre_dir: Path,
    phase_id: str,
    expected_content_sha256: str,
    expected_role_sha256: str,
) -> tuple[list[dict[str, Any]], str]:
    path = pre_dir / "quality_corpus.jsonl"
    raw = read_regular(path)
    exact(sha256(raw), expected_role_sha256, "corpus.role_sha256")
    lines = raw.splitlines(keepends=True)
    require(len(lines) == 64, "E_CORPUS_ROWS")
    bodies = []
    body_raw = bytearray()
    for index, line in enumerate(lines):
        row = parse_json(line, f"corpus[{index}]")
        require(type(row) is dict, f"E_CORPUS_TYPE: {index}")
        require(canonical_bytes(row) == line, f"E_CORPUS_CANONICAL: {index}")
        require(WRAPPER_KEYS.issubset(row), f"E_CORPUS_WRAPPER: {index}")
        exact(row["acquisition_id"], phase_id, f"corpus[{index}].acquisition")
        exact(row["phase"], PHASE, f"corpus[{index}].phase")
        exact(row["phase_id"], phase_id, f"corpus[{index}].phase_id")
        exact(row["role"], "quality.corpus", f"corpus[{index}].role")
        body = {key: value for key, value in row.items() if key not in WRAPPER_KEYS}
        exact_keys(body, CORPUS_BODY_KEYS | {"kind"}, f"corpus[{index}].body")
        exact(body["kind"], "item", f"corpus[{index}].kind")
        del body["kind"]
        exact(body["item_index"], index, f"corpus[{index}].item_index")
        bodies.append(body)
        body_raw.extend(canonical_bytes(body))
    exact(
        sha256(bytes(body_raw)),
        expected_content_sha256,
        "corpus.content_sha256",
    )
    return bodies, sha256(raw)


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
        require(size >= 0, "E_RECV_SIZE")
        output = bytearray()
        while len(output) < size:
            block = self.connection.recv(size - len(output))
            require(bool(block), "E_UNEXPECTED_EOF")
            output.extend(block)
        return bytes(output)

    def recv_i32(self, count: int) -> tuple[int, ...]:
        return struct.unpack(f"<{count}i", self.recv_exact(4 * count))

    def hello(self, plan: dict[str, Any]) -> None:
        self.connection.sendall(pack_i32([STAGE_V3_HELLO]))
        words = self.recv_i32(11)
        exact(words[0], STAGE_V3_MAGIC, "hello.magic")
        exact(words[1], STAGE_V3_VERSION, "hello.version")
        exact(list(words[2:5]), [0, N_LAYER, N_LAYER], "hello.layers")
        exact(words[5], plan["expected_n_embd"], "hello.n_embd")
        exact(words[6], plan["expected_max_streams"], "hello.max_streams")
        exact(words[7], plan["expected_n_ctx_seq"], "hello.n_ctx_seq")
        exact(words[8], plan["expected_n_batch"], "hello.n_batch")
        exact(words[9], plan["expected_n_ubatch"], "hello.n_ubatch")
        exact(
            words[10],
            STAGE_V3_REQUIRED_CAPABILITIES,
            "hello.capabilities",
        )
        self.n_batch = words[8]
        self.n_ubatch = words[9]
        self.connection.sendall(pack_i32([STAGE_V3_IDENTITY]))
        identity = self.recv_i32(3)
        exact(
            identity,
            (
                STAGE_IDENTITY_MAGIC,
                STAGE_IDENTITY_VERSION,
                plan["expected_file_type"],
            ),
            "identity.header",
        )
        exact(self.recv_exact(32).hex(), MODEL_SHA256, "identity.model_sha256")

    def status(self) -> tuple[int, int, bool]:
        self.connection.sendall(pack_i32([STAGE_V3_STATUS, STAGE_V3_VERSION]))
        code, version, active, maximum, draining = self.recv_i32(5)
        exact(code, 0, "status.code")
        exact(version, STAGE_V3_VERSION, "status.version")
        require(
            0 <= active <= maximum == MAX_STREAMS and draining in (0, 1),
            "E_STATUS",
        )
        return active, maximum, bool(draining)

    def batch(
        self,
        rows: list[tuple[int, int, int, int, int]],
    ) -> list[int]:
        require(
            0 < len(rows) <= min(self.n_batch, self.n_ubatch),
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
        route_epochs = struct.unpack(f"<{count}q", self.recv_exact(8 * count))
        seq_ids = struct.unpack(f"<{count}i", self.recv_exact(4 * count))
        positions = struct.unpack(f"<{count}i", self.recv_exact(4 * count))
        tokens = self.recv_i32(count)
        exact(request_ids, tuple(row[0] for row in rows), "batch.request_ids")
        exact(route_epochs, tuple(row[1] for row in rows), "batch.route_epochs")
        exact(seq_ids, tuple(row[2] for row in rows), "batch.seq_ids")
        exact(positions, tuple(row[3] for row in rows), "batch.positions")
        require(all(token >= 0 for token in tokens), "E_BATCH_TOKEN")
        return list(tokens)

    def remove(self, request_id: int, route_epoch: int, seq_id: int) -> None:
        payload = pack_i32([STAGE_V3_SEQ_REMOVE, STAGE_V3_VERSION, seq_id])
        payload += pack_i64([request_id, route_epoch])
        self.connection.sendall(payload)
        code, version, active, maximum, draining = self.recv_i32(5)
        require(
            code == 0
            and version == STAGE_V3_VERSION
            and 0 <= active <= maximum == MAX_STREAMS
            and draining in (0, 1),
            "E_REMOVE",
        )

    def stop(self) -> None:
        self.connection.sendall(pack_i32([STAGE_STOP]))


def connect_route(
    host: str,
    port: int,
    timeout_ms: int,
    relay: subprocess.Popen[bytes],
) -> StageClient:
    deadline = time.monotonic() + timeout_ms / 1000
    last_error = None
    while time.monotonic() < deadline:
        require(relay.poll() is None, "E_RELAY_EARLY_EXIT")
        try:
            connection = socket.create_connection((host, port), timeout=0.25)
            connection.settimeout(timeout_ms / 1000)
            return StageClient(connection)
        except OSError as error:
            last_error = error
            time.sleep(0.025)
    raise CaptureError(f"E_ROUTE_CONNECT: {last_error}")


def _frame_shape(
    call_index: int,
    rows: list[tuple[int, int, int, int, int]],
) -> dict[str, Any]:
    return {
        "call_index": call_index,
        "hidden_width": N_EMBD,
        "positions": [row[3] for row in rows],
        "request_ids": [row[0] for row in rows],
        "route_epochs": [row[1] for row in rows],
        "rows": len(rows),
        "seq_ids": [row[2] for row in rows],
    }


def run_history_group(
    client: StageClient,
    history: dict[str, Any],
    group: dict[str, Any],
    wire_ids: list[int],
    route_epoch: int,
    frame_offset: int,
) -> tuple[
    list[list[int]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    requests = {
        request["item_index"]: request
        for request in history["requests"]
    }
    item_indices = group["item_indices"]
    require(len(item_indices) == len(wire_ids) == BATCH, "E_GROUP_BATCH")
    current: dict[int, int] = {}
    frames = []
    calls = []
    receipts = []

    for partition in group["prefill_partitions"]:
        rows = []
        receipt_rows = []
        for row in partition["rows"]:
            item_index = row["item_index"]
            request = requests[item_index]
            seq_id = item_indices.index(item_index)
            exact(row["seq_id"], seq_id, "E_GROUP_PREFILL_SEQ")
            exact(row["request_id"], seq_id + 1, "E_GROUP_PREFILL_REQUEST")
            rows.append(
                (
                    wire_ids[seq_id],
                    route_epoch,
                    seq_id,
                    row["position"],
                    row["token_id"],
                )
            )
        tokens = client.batch(rows)
        for source, wire, token in zip(partition["rows"], rows, tokens):
            item_index = source["item_index"]
            request = requests[item_index]
            if source["position"] == len(request["token_ids"]) - 1:
                require(wire[2] not in current, "E_GROUP_FINAL_REUSE")
                current[wire[2]] = token
            receipt_rows.append({
                "input_token": wire[4],
                "item_index": item_index,
                "output_token": token,
                "position": wire[3],
                "request_id": source["request_id"],
                "route_epoch": route_epoch,
                "seq_id": wire[2],
                "wire_request_id": wire[0],
            })
        exact(partition["call_index"], len(calls), "E_GROUP_PREFILL_CALL")
        calls.append({
            "call_index": partition["call_index"],
            "n_seqs": len({row[2] for row in rows}),
            "n_tokens": len(rows),
            "phase": "prefill",
            "positions": [row["position"] for row in partition["rows"]],
            "request_ids": [row["request_id"] for row in partition["rows"]],
            "seq_ids": [row["seq_id"] for row in partition["rows"]],
        })
        frames.append(_frame_shape(frame_offset + len(frames), rows))
        receipts.append({
            "call_index": partition["call_index"],
            "frame_call_index": frame_offset + len(frames) - 1,
            "phase": "prefill",
            "rows": receipt_rows,
        })

    exact(sorted(current), list(range(BATCH)), "E_GROUP_FINAL_PREDICTIONS")
    continuations = [[current[seq_id]] for seq_id in range(BATCH)]
    for ordinal, decode_call in enumerate(group["decode_calls"]):
        rows = []
        receipt_rows = []
        for row in decode_call["rows"]:
            seq_id = row["seq_id"]
            item_index = item_indices[seq_id]
            exact(row["item_index"], item_index, "E_GROUP_DECODE_ITEM")
            exact(row["request_id"], seq_id + 1, "E_GROUP_DECODE_REQUEST")
            rows.append(
                (
                    wire_ids[seq_id],
                    route_epoch,
                    seq_id,
                    row["position"],
                    current[seq_id],
                )
            )
        tokens = client.batch(rows)
        next_tokens = {}
        for source, wire, token in zip(decode_call["rows"], rows, tokens):
            next_tokens[wire[2]] = token
            continuations[wire[2]].append(token)
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
        exact(sorted(next_tokens), list(range(BATCH)), "E_GROUP_DECODE_OUTPUTS")
        current = next_tokens
        exact(
            decode_call["call_index"],
            len(calls),
            "E_GROUP_DECODE_CALL",
        )
        exact(
            decode_call["continuation_input_ordinal"],
            ordinal,
            "E_GROUP_DECODE_INPUT_ORDINAL",
        )
        exact(
            decode_call["continuation_output_ordinal"],
            ordinal + 1,
            "E_GROUP_DECODE_OUTPUT_ORDINAL",
        )
        calls.append({
            "call_index": decode_call["call_index"],
            "n_seqs": BATCH,
            "n_tokens": BATCH,
            "phase": "decode",
            "positions": [row["position"] for row in decode_call["rows"]],
            "request_ids": [row["request_id"] for row in decode_call["rows"]],
            "seq_ids": [row["seq_id"] for row in decode_call["rows"]],
        })
        frames.append(_frame_shape(frame_offset + len(frames), rows))
        receipts.append({
            "call_index": decode_call["call_index"],
            "frame_call_index": frame_offset + len(frames) - 1,
            "phase": "decode",
            "rows": receipt_rows,
        })

    require(
        all(len(tokens) == 8 for tokens in continuations),
        "E_CONTINUATION_COUNT",
    )
    return (
        continuations,
        calls,
        frames,
        {
            "call_receipts": receipts,
            "continuations": continuations,
            "group_index": group["group_index"],
            "item_indices": item_indices,
            "wire_request_ids": wire_ids,
        },
    )


def remove_group(
    client: StageClient,
    wire_ids: list[int],
    route_epoch: int,
) -> None:
    for sequence, request_id in enumerate(wire_ids):
        client.remove(request_id, route_epoch, sequence)
    exact(client.status()[0], 0, "cleanup.active_sequences")


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
    require(len(raw_input) <= MAX_CODEC_OUTPUT, "E_CODEC_INPUT_SIZE")
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
    require(0 < len(process.stdout) <= MAX_CODEC_OUTPUT, "E_CODEC_OUTPUT_SIZE")
    lines = process.stdout.splitlines(keepends=True)
    exact(len(lines), len(requests), "codec.output_count")
    outputs = []
    for index, (request, line) in enumerate(zip(requests, lines)):
        response = parse_json(line, f"codec[{index}]")
        require(type(response) is dict, f"E_CODEC_RESPONSE: {index}")
        require(canonical_bytes(response) == line, f"E_CODEC_CANONICAL: {index}")
        expected = {
            "model_sha256",
            "op",
            "request_id",
            "schema",
            "text" if request["op"] == "detokenize" else "tokens",
        }
        exact_keys(response, expected, f"codec[{index}]")
        exact(
            response["schema"],
            "layersplit-token-codec-response-v1",
            f"codec[{index}].schema",
        )
        exact(response["model_sha256"], MODEL_SHA256, f"codec[{index}].model")
        exact(response["op"], request["op"], f"codec[{index}].op")
        exact(
            response["request_id"],
            request["request_id"],
            f"codec[{index}].request_id",
        )
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
    outputs = invoke_codec(codec, requests)
    histories = []
    for index, output in enumerate(outputs):
        tokens = vector(output["tokens"], f"codec.tokens[{index}]")
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
    values = []
    for index, response in enumerate(responses):
        value = response["text"]
        require(type(value) is str, f"E_CODEC_TEXT: {index}")
        values.append(value)
    return values


class ManagedProcess:
    def __init__(self, name: str, spec: dict[str, Any], root: Path):
        self.name = name
        self.spec = spec
        self.root = root
        self.process: subprocess.Popen[bytes] | None = None
        self.log = None
        self.started_ns = 0
        self.completed_ns = 0

    @property
    def log_path(self) -> Path:
        return self.root / f"{self.name}.log"

    def start(self) -> None:
        require(self.process is None and not self.log_path.exists(), "E_PROCESS_STATE")
        self.log = self.log_path.open("xb")
        self.started_ns = monotonic_ns()
        self.process = subprocess.Popen(
            self.spec["argv"],
            cwd=self.spec["cwd"],
            env=self.spec["environment"],
            stdin=subprocess.DEVNULL,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def wait(self) -> None:
        require(self.process is not None, "E_PROCESS_STATE")
        try:
            code = self.process.wait(
                timeout=self.spec["shutdown_timeout_ms"] / 1000
            )
        except subprocess.TimeoutExpired as error:
            self.kill()
            raise CaptureError(f"E_PROCESS_TIMEOUT: {self.name}") from error
        self.completed_ns = monotonic_ns()
        exact(code, 0, f"process.{self.name}.returncode")
        self.process = None
        self._close_log()

    def kill(self) -> None:
        if self.process is not None and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.process.wait()
        self.process = None
        self.completed_ns = monotonic_ns()
        self._close_log()

    def _close_log(self) -> None:
        if self.log is not None:
            self.log.flush()
            os.fsync(self.log.fileno())
            self.log.close()
            self.log = None
        if self.log_path.exists():
            metadata = self.log_path.stat(follow_symlinks=False)
            require(
                stat.S_ISREG(metadata.st_mode)
                and metadata.st_size <= MAX_LOG_FILE,
                f"E_LOG_FILE: {self.name}",
            )


def wait_log_marker(
    process: ManagedProcess,
    marker: bytes,
) -> None:
    require(process.process is not None, "E_PROCESS_STATE")
    deadline = (
        time.monotonic()
        + process.spec["startup_timeout_ms"] / 1000
    )
    while time.monotonic() < deadline:
        require(process.process.poll() is None, f"E_PROCESS_EARLY_EXIT: {process.name}")
        if process.log is not None:
            process.log.flush()
        if process.log_path.exists():
            metadata = process.log_path.stat(follow_symlinks=False)
            require(
                stat.S_ISREG(metadata.st_mode)
                and metadata.st_size <= MAX_LOG_FILE,
                f"E_LOG_FILE: {process.name}",
            )
            if metadata.st_size > 0:
                flags = os.O_RDONLY | os.O_CLOEXEC
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                descriptor = os.open(process.log_path, flags)
                try:
                    snapshot = os.read(descriptor, metadata.st_size)
                finally:
                    os.close(descriptor)
                if marker in snapshot:
                    return
        time.sleep(0.025)
    raise CaptureError(f"E_PROCESS_STARTUP_TIMEOUT: {process.name}")


def run_probe(
    spec: dict[str, Any],
    when: str,
    output_path: Path,
) -> tuple[dict[str, Any], int]:
    argv = spec[f"{when}_argv"]
    try:
        result = subprocess.run(
            argv,
            cwd=spec["cwd"],
            env=spec["environment"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=spec["timeout_ms"] / 1000,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise CaptureError(f"E_PROBE_TIMEOUT: {when}") from error
    event_ns = monotonic_ns()
    require(
        len(result.stdout) <= MAX_PROCESS_OUTPUT
        and len(result.stderr) <= MAX_PROCESS_OUTPUT,
        f"E_PROBE_OUTPUT_SIZE: {when}",
    )
    durable_write_new(output_path, result.stdout)
    durable_write_new(output_path.with_suffix(".stderr"), result.stderr)
    exact(result.returncode, 0, f"probe.{when}.returncode")
    exact(result.stderr, b"", f"probe.{when}.stderr")
    value = parse_json(result.stdout, f"probe.{when}")
    require(type(value) is dict, f"E_PROBE_TYPE: {when}")
    require(canonical_bytes(value) == result.stdout, f"E_PROBE_CANONICAL: {when}")
    return value, event_ns


def validate_probe_row(
    value: dict[str, Any],
    phone: str,
    expected: dict[str, Any],
    field: str,
) -> dict[str, Any]:
    exact_keys(value, PROBE_ROW_KEYS, field)
    exact(value["schema"], PROBE_SCHEMA, f"{field}.schema")
    for key in ("boot_id", "device", "model", "product", "serial"):
        exact(value[key], expected[key], f"{field}.{key}")
    exact(value["model_id"], MODEL_ID, f"{field}.model_id")
    exact(value["model_sha256"], MODEL_SHA256, f"{field}.model_sha256")
    exact(
        value["worker_executable_path"],
        expected["expected_worker_executable_path"],
        f"{field}.worker_path",
    )
    exact(
        value["worker_executable_sha256"],
        expected["expected_worker_executable_sha256"],
        f"{field}.worker_sha256",
    )
    exact(value["loaded_shard_path"], expected["loaded_shard_path"], f"{field}.shard")
    exact(
        value["loaded_shard_sha256"],
        expected["loaded_shard_sha256"],
        f"{field}.shard_sha256",
    )
    integer(value["worker_pid"], f"{field}.pid", 1)
    integer(value["worker_start_ticks"], f"{field}.start_ticks", 1)
    integer(value["active_sequences"], f"{field}.active_sequences")
    integer(value["available_bytes"], f"{field}.available_bytes")
    integer(value["process_swap_bytes"], f"{field}.process_swap_bytes")
    integer(value["system_swap_used_bytes"], f"{field}.system_swap")
    exact(value["thermal_status"], 0, f"{field}.thermal_status")
    integer(value["gpu_max_millic"], f"{field}.gpu_max_millic", 1)
    interface = exact_keys(
        value["interface"],
        {"ipv4", "name", "rx_bytes", "tx_bytes"},
        f"{field}.interface",
    )
    exact(interface["name"], expected["interface"], f"{field}.interface.name")
    exact(interface["ipv4"], expected["local_ipv4"], f"{field}.interface.ipv4")
    integer(interface["rx_bytes"], f"{field}.interface.rx")
    integer(interface["tx_bytes"], f"{field}.interface.tx")
    peer = exact_keys(
        value["direct_peer"],
        {"interface", "local_ipv4", "peer_ipv4", "socket_peer_observed"},
        f"{field}.direct_peer",
    )
    exact(peer["interface"], expected["interface"], f"{field}.peer.interface")
    exact(peer["local_ipv4"], expected["local_ipv4"], f"{field}.peer.local")
    exact(peer["peer_ipv4"], expected["direct_peer_ipv4"], f"{field}.peer.remote")
    exact(peer["socket_peer_observed"], True, f"{field}.peer.observed")
    return value


def parse_prefixed(raw: bytes, prefix: bytes, field: str) -> list[dict[str, Any]]:
    records = []
    for line_index, line in enumerate(raw.splitlines(keepends=True)):
        if line.startswith(prefix):
            require(line.endswith(b"\n"), f"E_RECORD_PARTIAL: {field}[{line_index}]")
            payload = line[len(prefix):]
            value = parse_json(payload, f"{field}[{line_index}]")
            require(type(value) is dict, f"E_RECORD_TYPE: {field}")
            records.append(value)
        elif prefix.rstrip() in line:
            raise CaptureError(f"E_RECORD_PREFIX: {field}[{line_index}]")
    return records


def parse_runtime_process(
    raw: bytes,
    name: str,
    endpoint: str,
    boot_id: str,
    spec: dict[str, Any],
    started_ns: int,
    completed_ns: int,
    expected_pid: int | None = None,
    expected_start_ticks: int | None = None,
) -> dict[str, Any]:
    records = parse_prefixed(
        raw,
        b"RUNTIMEPROCESS ",
        f"{name}.runtime_process",
    )
    exact(len(records), 1, f"{name}.runtime_process_count")
    value = exact_keys(
        records[0],
        RUNTIME_PROCESS_KEYS,
        f"{name}.runtime_process",
    )
    exact(value["bundle_id"], name, f"{name}.runtime_process.bundle_id")
    exact(value["endpoint"], endpoint, f"{name}.runtime_process.endpoint")
    exact(value["boot_id"], boot_id, f"{name}.runtime_process.boot_id")
    pid = integer(value["pid"], f"{name}.runtime_process.pid", 1)
    start_ticks = integer(
        value["start_ticks"],
        f"{name}.runtime_process.start_ticks",
        1,
    )
    if expected_pid is not None:
        exact(pid, expected_pid, f"{name}.runtime_process.worker_pid")
    if expected_start_ticks is not None:
        exact(
            start_ticks,
            expected_start_ticks,
            f"{name}.runtime_process.worker_start_ticks",
        )
    observed_ns = integer(
        value["observed_ns"],
        f"{name}.runtime_process.observed_ns",
        1,
    )
    require(
        started_ns <= observed_ns <= completed_ns,
        f"E_RUNTIME_PROCESS_INTERVAL: {name}",
    )
    exact(
        value["launcher_path"],
        spec["runtime_executable_path"],
        f"{name}.runtime_process.launcher_path",
    )
    exact(
        value["loaded_repo_component_ids"],
        spec["runtime_component_ids"],
        f"{name}.runtime_process.components",
    )
    dependencies = value["system_dependencies"]
    require(
        type(dependencies) is list and bool(dependencies),
        f"E_RUNTIME_DEPENDENCIES: {name}",
    )
    paths = []
    for index, dependency in enumerate(dependencies):
        field = f"{name}.runtime_process.system_dependencies[{index}]"
        exact_keys(dependency, SYSTEM_DEPENDENCY_KEYS, field)
        path = text(dependency["path"], f"{field}.path")
        require(Path(path).is_absolute(), f"E_RUNTIME_DEPENDENCY_PATH: {field}")
        paths.append(path)
        for key in ("ctime_ns", "device_id", "inode", "mode", "mtime_ns"):
            integer(dependency[key], f"{field}.{key}")
        integer(dependency["size"], f"{field}.size", 1)
        build_id = dependency["build_id"]
        require(
            build_id is None
            or (
                type(build_id) is str
                and 0 < len(build_id) <= 256
                and "\x00" not in build_id
                and "\n" not in build_id
            ),
            f"E_RUNTIME_BUILD_ID: {field}",
        )
    exact(paths, sorted(set(paths)), f"{name}.runtime_process.dependency_order")
    return value


def validate_op_map(
    value: Any,
    field: str,
    allow_cpu_get_rows: bool,
) -> tuple[list[tuple[str, str]], int]:
    require(type(value) is dict and bool(value), f"E_OP_MAP: {field}")
    nodes = []
    total = 0
    for op in sorted(value):
        backends = value[op]
        require(type(op) is str and bool(op), f"E_OP: {field}")
        require(type(backends) is dict and bool(backends), f"E_BACKENDS: {field}.{op}")
        for backend in sorted(backends):
            count = integer(backends[backend], f"{field}.{op}.{backend}", 1)
            require(backend in {"CPU", "OpenCL"}, f"E_BACKEND: {field}.{backend}")
            if backend == "CPU":
                require(
                    allow_cpu_get_rows and op == "GET_ROWS",
                    f"E_CPU_FALLBACK: {field}.{op}",
                )
            nodes.append((op, backend))
            total += count
    return nodes, total


def parse_worker_log(
    raw: bytes,
    phone: str,
    expected: dict[str, Any],
    total_rows: int,
) -> tuple[dict[str, Any], dict[str, Any], list[tuple[str, str]]]:
    sessions = parse_prefixed(raw, b"SESSIONCERT ", f"{phone}.session")
    exact(len(sessions), 1, f"{phone}.sessions")
    session = exact_keys(sessions[0], SESSION_KEYS, f"{phone}.session")
    layer_start, layer_end = expected["executed_layers"]
    expected_session = {
        "device_boot_id": expected["boot_id"],
        "expected_backend": "GPUOpenCL",
        "layer_end": layer_end,
        "layer_start": layer_start,
        "missing_buffer_compute_nodes": 0,
        "n_layer": N_LAYER,
        "placement_status": "SCHEDULED_PLACEMENT_OK",
        "proto_version": 2,
        "reset_applied": False,
        "schema": "ls-stagenet-session-v2",
        "session_end": "STOP",
        "session_id": 1,
        "steps_session": total_rows,
        "steps_total": total_rows,
    }
    for key, value in expected_session.items():
        exact(session[key], value, f"{phone}.session.{key}")
    integer(session["worker_pid"], f"{phone}.session.pid", 1)
    nonce = text(session["worker_boot_nonce"], f"{phone}.session.nonce", 16)
    require(
        len(nonce) == 16
        and all(character in "0123456789abcdef" for character in nonce),
        f"E_NONCE: {phone}",
    )
    session_nodes, session_count = validate_op_map(
        session["compute_by_op_and_buffer"],
        f"{phone}.session.ops",
        phone == "op15",
    )
    placements = parse_prefixed(raw, b"PLACEMENTCERT ", f"{phone}.placement")
    exact(len(placements), 1, f"{phone}.placements")
    placement = exact_keys(placements[0], PLACEMENT_KEYS, f"{phone}.placement")
    expected_placement = {
        "layer_end": layer_end,
        "layer_start": layer_start,
        "missing_buffer_compute_nodes": 0,
        "mode": "stagenet" if phone == "op15" else "tailv3",
        "n_layer": N_LAYER,
        "pid": session["worker_pid"],
        "role": "phone_stage" if phone == "op15" else "host_tail_v3",
        "run_rc": 0,
        "schema": "layersplit-scheduled-placement-v2",
        "status": "SCHEDULED_PLACEMENT_OK",
    }
    for key, value in expected_placement.items():
        exact(placement[key], value, f"{phone}.placement.{key}")
    placement_nodes, placement_count = validate_op_map(
        placement["compute_by_op_and_buffer"],
        f"{phone}.placement.ops",
        phone == "op15",
    )
    exact(placement_nodes, session_nodes, f"{phone}.placement.session")
    exact(placement_count, session_count, f"{phone}.placement.session_count")
    exact(placement["compute_nodes"], placement_count, f"{phone}.compute_nodes")
    return session, placement, placement_nodes


def parse_direct_frames(
    raw: bytes,
    expected_frames: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frames = parse_prefixed(raw, b"DIRECTFRAME ", "relay.frame")
    exact(len(frames), len(expected_frames), "relay.frame_count")
    payload_total = 0
    row_total = 0
    for index, (frame, expected) in enumerate(zip(frames, expected_frames)):
        exact_keys(frame, DIRECT_FRAME_KEYS, f"relay.frame[{index}]")
        exact(frame["schema"], "ls-stage-direct-frame-v1", f"relay.frame[{index}].schema")
        for key, value in expected.items():
            exact(frame[key], value, f"relay.frame[{index}].{key}")
        expected_bytes = expected["rows"] * N_EMBD * 4
        exact(
            frame["activation_payload_bytes"],
            expected_bytes,
            f"relay.frame[{index}].bytes",
        )
        digest(frame["payload_sha256"], f"relay.frame[{index}].sha256")
        payload_total += expected_bytes
        row_total += expected["rows"]
    certificates = parse_prefixed(raw, b"DIRECTCERT ", "relay.cert")
    exact(len(certificates), 1, "relay.cert_count")
    cert = exact_keys(certificates[0], DIRECT_CERT_KEYS, "relay.cert")
    expected_cert = {
        "activation_payload_bytes": payload_total,
        "batches": len(frames),
        "cut_layer": 30,
        "file_type": FILE_TYPE,
        "host_activation_payload_bytes": 0,
        "layer_end": N_LAYER,
        "layer_start": 0,
        "model_sha256": MODEL_SHA256,
        "n_embd": N_EMBD,
        "n_layer": N_LAYER,
        "rows": row_total,
        "run_rc": 0,
        "schema": "ls-stage-direct-relay-v1",
        "status": "DIRECT_RELAY_OK",
    }
    for key, value in expected_cert.items():
        exact(cert[key], value, f"relay.cert.{key}")
    text(cert["head_endpoint"], "relay.cert.head_endpoint")
    text(cert["tail_endpoint"], "relay.cert.tail_endpoint")
    return frames, cert


def placement_rows(
    phone: str,
    expected: dict[str, Any],
    before: dict[str, Any],
    after: dict[str, Any],
    nodes: list[tuple[str, str]],
    event_ns: int,
) -> list[dict[str, Any]]:
    rows = [{
        "available_after_bytes": after["available_bytes"],
        "available_before_bytes": before["available_bytes"],
        "batch": BATCH,
        "boot_id": expected["boot_id"],
        "device": expected["device"],
        "event_ns": event_ns,
        "executed_layers": expected["executed_layers"],
        "kind": "meta",
        "model": expected["model"],
        "model_id": MODEL_ID,
        "model_sha256": MODEL_SHA256,
        "process_swap_bytes": max(
            before["process_swap_bytes"],
            after["process_swap_bytes"],
        ),
        "product": expected["product"],
        "serial": expected["serial"],
        "shard_sha256": expected["loaded_shard_sha256"],
        "stored_layers": expected["stored_layers"],
        "system_swap_after_bytes": after["system_swap_used_bytes"],
        "system_swap_before_bytes": before["system_swap_used_bytes"],
    }]
    for node_id, (op, backend) in enumerate(nodes):
        rows.append({
            "backend": "GPUOpenCL" if backend == "OpenCL" else "CPU",
            "compute": True,
            "event_ns": event_ns,
            "kind": "node",
            "missing_buffer": False,
            "node_id": node_id,
            "op": op,
        })
    return rows


def runtime_record(
    expected: dict[str, Any],
    before: dict[str, Any],
    after: dict[str, Any],
    session: dict[str, Any],
    route_epoch: int,
) -> dict[str, Any]:
    exact(before["worker_pid"], after["worker_pid"], "runtime.pid")
    exact(
        before["worker_start_ticks"],
        after["worker_start_ticks"],
        "runtime.start_ticks",
    )
    exact(session["worker_pid"], before["worker_pid"], "runtime.session_pid")
    actual_before = before["interface"]
    actual_after = after["interface"]
    require(
        actual_after["rx_bytes"] >= actual_before["rx_bytes"]
        and actual_after["tx_bytes"] >= actual_before["tx_bytes"],
        "E_INTERFACE_COUNTER_RESET",
    )
    return {
        "active_sequences_after_cleanup": after["active_sequences"],
        "available_bytes": min(
            before["available_bytes"],
            after["available_bytes"],
        ),
        "boot_id": expected["boot_id"],
        "direct_peer": after["direct_peer"],
        "gpu_max_millic": max(
            before["gpu_max_millic"],
            after["gpu_max_millic"],
        ),
        "interface_after": {
            "interface": actual_after["name"],
            "rx_bytes": actual_after["rx_bytes"],
            "tx_bytes": actual_after["tx_bytes"],
        },
        "interface_before": {
            "interface": actual_before["name"],
            "rx_bytes": actual_before["rx_bytes"],
            "tx_bytes": actual_before["tx_bytes"],
        },
        "loaded_shard_path": expected["loaded_shard_path"],
        "model_id": MODEL_ID,
        "process_swap_bytes": max(
            before["process_swap_bytes"],
            after["process_swap_bytes"],
        ),
        "route_epoch": route_epoch,
        "serial": expected["serial"],
        "session_protocol_version": session["proto_version"],
        "worker_boot_nonce": session["worker_boot_nonce"],
        "worker_executable_path": expected["expected_worker_executable_path"],
        "worker_model_sha256": MODEL_SHA256,
        "worker_pid": session["worker_pid"],
        "worker_start_ticks": before["worker_start_ticks"],
    }


def load_v24_fresh_devices(
    pre_dir: Path,
    phase_id: str,
    phase_lock: dict[str, Any],
    phase_lock_raw: bytes,
    plan: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    root, root_raw = read_canonical(
        pre_dir.parent / "artifact" / "artifact_root.json"
    )
    exact_keys(
        root,
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
    exact(root["schema"], "s39-cp0-r1-artifact-root-v2.4", "artifact_root.schema")
    exact(root["phase_scope"], "PRE_REBOOT_OUTSIDE_PHASE", "artifact_root.scope")
    exact(root["model_id"], MODEL_ID, "artifact_root.model")
    exact(
        sha256(root_raw),
        phase_lock["artifact_root_sha256"],
        "artifact_root.phase_lock",
    )
    require(type(root["components"]) is list and bool(root["components"]),
            "E_ARTIFACT_COMPONENTS")
    require(type(root["inventories"]) is list and bool(root["inventories"]),
            "E_ARTIFACT_INVENTORIES")

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
    exact(fresh["phase_lock_sha256"], sha256(phase_lock_raw), "fresh.phase_lock")
    exact(fresh["artifact_root_sha256"], sha256(root_raw), "fresh.artifact_root")
    exact(
        fresh["preparation_sha256"],
        phase_lock["preparation_sha256"],
        "fresh.preparation",
    )
    exact(
        fresh["runtime_bundle_plan_sha256"],
        phase_lock["runtime_bundle_plan_sha256"],
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
    result = {}
    for phone in ("op15", "op12"):
        value = exact_keys(
            devices[phone],
            {
                "available_bytes",
                "boot_id",
                "device",
                "model",
                "product",
                "serial",
                "system_swap_used_bytes",
                "thermal_status",
            },
            f"fresh.devices.{phone}",
        )
        expected = plan["phones"][phone]
        for key in ("boot_id", "device", "model", "product", "serial"):
            exact(value[key], expected[key], f"fresh.{phone}.{key}")
        exact(
            value["boot_id"],
            phase_lock["device_boot_ids"][phone],
            f"fresh.{phone}.phase_boot",
        )
        integer(value["available_bytes"], f"fresh.{phone}.available", 1)
        integer(value["system_swap_used_bytes"], f"fresh.{phone}.swap")
        exact(value["thermal_status"], 0, f"fresh.{phone}.thermal")
        result[phone] = value
    return result


def make_mechanics_rows(
    requests: list[dict[str, Any]],
    continuations: list[list[int]],
    calls: list[dict[str, Any]],
    program_sha256: str,
    event_ns: int,
) -> list[dict[str, Any]]:
    rows = [{
        "backend": "PHONE_COLLECTIVE",
        "call_shapes": calls,
        "event_ns": event_ns,
        "kind": "meta",
        "model_id": MODEL_ID,
        "model_sha256": MODEL_SHA256,
        "program_sha256": program_sha256,
        "state_count_after": 0,
        "state_count_before": 0,
    }]
    for sequence, request in enumerate(requests):
        request_id = request["request_id"]
        rows.append({
            "continuation_tokens": continuations[sequence],
            "event_ns": event_ns + sequence,
            "input_tokens": request["token_ids"],
            "kind": "request",
            "model_id": MODEL_ID,
            "model_sha256": MODEL_SHA256,
            "owner_after": "RELEASED",
            "owner_before": "PHONE",
            "ownership_epoch_after": 2,
            "ownership_epoch_before": 1,
            "positions": list(range(len(request["token_ids"]))),
            "request_id": request_id,
        })
    return rows


def make_bridge_rows(
    mechanics_rows: list[dict[str, Any]],
    event_ns: int,
    phase_id: str,
) -> list[dict[str, Any]]:
    exact(
        [row["request_id"] for row in mechanics_rows[1:]],
        list(range(1, 9)),
        "bridge.request_ids",
    )
    rows = []
    for request in mechanics_rows[1:]:
        request_id = request["request_id"]
        normalized_request = {
            "acquisition_id": phase_id,
            **{
                key: value
                for key, value in request.items()
                if key != "event_ns"
            },
            "role": f"model.{MODEL_ID}.mechanics.phone",
        }
        rows.append({
            "clock_id": CLOCK_NAME,
            "event_ns": event_ns + request_id,
            "kind": "phone_publication_received",
            "model_id": MODEL_ID,
            "model_sha256": MODEL_SHA256,
            "phone_request_sha256": sha256(
                canonical_bytes(normalized_request)
            ),
            "request_id": request_id,
            "timestamp_ns": event_ns + request_id,
            "token_ids": request["continuation_tokens"],
        })
    return rows


def make_transfer_rows(
    frames: list[dict[str, Any]],
    calls: list[dict[str, Any]],
    event_ns: int,
) -> list[dict[str, Any]]:
    exact(len(frames), len(calls), "mechanics.transfer_count")
    rows = [{
        "batch": BATCH,
        "cut_layer": 30,
        "event_ns": event_ns,
        "kind": "meta",
        "model_id": MODEL_ID,
        "model_sha256": MODEL_SHA256,
        "request_ids": list(range(BATCH)),
    }]
    for index, (frame, call) in enumerate(zip(frames, calls)):
        exact(frame["rows"], call["n_tokens"], f"mechanics.frame[{index}].rows")
        rows.append({
            "call_index": index,
            "event_ns": event_ns + index + 1,
            "host_payload_bytes": 0,
            "kind": "transfer",
            "path": "WIFI_TCP_DIRECT",
            "payload_bytes": frame["activation_payload_bytes"],
            "payload_sha256": frame["payload_sha256"],
            "receiver": "op12",
            "row_count": frame["rows"],
            "sender": "op15",
        })
    return rows


def make_quality_rows(
    corpus: list[dict[str, Any]],
    raw_outputs: list[str],
    corpus_role_sha256: str,
    event_ns: int,
    phase_id: str,
) -> list[dict[str, Any]]:
    exact(len(corpus), 64, "quality.corpus_count")
    exact(len(raw_outputs), 64, "quality.output_count")
    rows = []
    for index, (item, output) in enumerate(zip(corpus, raw_outputs)):
        prompt = prompt_for(item)
        normalized_item = {
            "acquisition_id": phase_id,
            "kind": "item",
            "role": "quality.corpus",
            **item,
        }
        rows.append({
            "corpus_item_sha256": sha256(canonical_bytes(normalized_item)),
            "corpus_sha256": corpus_role_sha256,
            "event_ns": event_ns + index,
            "item_index": index,
            "kind": "output",
            "model_id": MODEL_ID,
            "model_sha256": MODEL_SHA256,
            "prompt_sha256": sha256(prompt.encode("utf-8")),
            "raw_output": output,
        })
    return rows


def start_processes(
    plan: dict[str, Any],
    root: Path,
) -> dict[str, ManagedProcess]:
    processes = {
        name: ManagedProcess(name, plan["processes"][name], root)
        for name in ("op12_stagenet", "op15_stagenet", "op15_direct_relay")
    }
    try:
        processes["op12_stagenet"].start()
        wait_log_marker(
            processes["op12_stagenet"],
            b"[stagenet] listening on ",
        )
        processes["op15_stagenet"].start()
        wait_log_marker(
            processes["op15_stagenet"],
            b"[stagenet] listening on ",
        )
        processes["op15_direct_relay"].start()
        wait_log_marker(
            processes["op15_direct_relay"],
            b"[direct-relay] listening on ",
        )
        return processes
    except BaseException:
        for process in processes.values():
            process.kill()
        raise


def stop_processes(processes: dict[str, ManagedProcess]) -> None:
    first_error = None
    for name in ("op15_direct_relay", "op15_stagenet", "op12_stagenet"):
        process = processes[name]
        try:
            process.wait()
        except BaseException as error:
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error


def capture(args: argparse.Namespace) -> dict[str, Any]:
    started_ns = monotonic_ns()
    acquisition_started_ns = integer(args.started, "started", 1)
    require(acquisition_started_ns <= started_ns, "E_ACQUISITION_ORDER")
    phase_id = text(args.phase_id, "phase_id", 128)
    require(
        phase_id.startswith("cp0-r1-v24-a-only-")
        and all(character.isalnum() or character in ".-_" for character in phase_id),
        "E_PHASE_ID",
    )
    digest(args.plan, "command_plan_sha256")
    supplied_mechanism_sha256 = digest(
        args.mechanism_commands_sha256,
        "mechanism_commands_sha256",
    )
    exact(digest(args.model_sha256, "model_sha256"), MODEL_SHA256, "model_sha256")
    output = Path(args.output)
    require(output.is_absolute() and not output.exists(), "E_OUTPUT")
    pre_dir = Path(args.pre_dir)
    phase_lock, phase_lock_raw = load_phase(pre_dir, phase_id)
    launch, launch_raw = load_plan(Path(args.launch_plan))
    mechanism_sha256 = bind_mechanism_commands(
        launch,
        supplied_mechanism_sha256,
    )
    history, history_raw = load_histories(
        Path(launch["history_path"]),
        launch["history_sha256"],
        MODEL_SHA256,
    )
    route_epoch = launch["route_epoch"]
    corpus, corpus_role_sha256 = load_corpus(
        pre_dir,
        phase_id,
        launch["quality_corpus_content_sha256"],
        phase_lock["quality_corpus_sha256"],
    )
    exact(
        history["corpus_sha256"],
        launch["quality_corpus_content_sha256"],
        "history.corpus_sha256",
    )
    for item_index, (item, request) in enumerate(
        zip(corpus, history["requests"])
    ):
        exact(
            request["prompt_sha256"],
            sha256(prompt_for(item).encode("utf-8")),
            f"history.prompt_sha256[{item_index}]",
        )
    load_v24_fresh_devices(
        pre_dir,
        phase_id,
        phase_lock,
        phase_lock_raw,
        launch,
    )
    source_raw = read_regular(Path(__file__).resolve())
    program_sha256 = sha256(
        b"s39:v24:phone-route-program:v1\0"
        + bytes.fromhex(sha256(source_raw))
        + bytes.fromhex(sha256(launch_raw))
        + bytes.fromhex(sha256(history_raw))
    )

    evidence_root = output.with_suffix(output.suffix + ".evidence")
    evidence_root.mkdir(parents=True, exist_ok=False)
    processes: dict[str, ManagedProcess] = {}
    connection = None
    stopped = False
    try:
        processes = start_processes(launch, evidence_root)
        client = connect_route(
            launch["relay_host"],
            launch["relay_port"],
            launch["processes"]["op15_direct_relay"]["startup_timeout_ms"],
            processes["op15_direct_relay"].process,
        )
        connection = client.connection
        client.hello(launch)
        exact(client.status()[0], 0, "state_before")
        before = {}
        before_ns = {}
        for phone in ("op15", "op12"):
            value, event_ns = run_probe(
                launch["probes"][phone],
                "before",
                evidence_root / f"{phone}.before.json",
            )
            before[phone] = validate_probe_row(
                value,
                phone,
                launch["phones"][phone],
                f"{phone}.before",
            )
            before_ns[phone] = event_ns
            exact(before[phone]["active_sequences"], 0, f"{phone}.before.active")

        mechanics_group = history["mechanics_b8"]
        mechanics_requests = [
            history["requests"][item_index]
            for item_index in mechanics_group["item_indices"]
        ]
        wire_ids = list(range(1001, 1009))
        (
            continuations,
            calls,
            expected_frames,
            mechanics_receipt,
        ) = run_history_group(
            client,
            history,
            mechanics_group,
            wire_ids,
            route_epoch,
            0,
        )
        remove_group(client, wire_ids, route_epoch)
        mechanics_ns = monotonic_ns()
        mechanics_rows = make_mechanics_rows(
            mechanics_requests,
            continuations,
            calls,
            program_sha256,
            mechanics_ns,
        )
        bridge_rows = make_bridge_rows(
            mechanics_rows,
            monotonic_ns(),
            phase_id,
        )

        quality_tokens: list[list[int]] = list(continuations)
        execution_groups = [mechanics_receipt]
        for cohort in range(1, 8):
            first = cohort * BATCH
            cohort_group = history["quality_groups"][cohort]
            cohort_wire_ids = list(range(2001 + first, 2001 + first + BATCH))
            outputs, _, frames, receipt = run_history_group(
                client,
                history,
                cohort_group,
                cohort_wire_ids,
                route_epoch,
                len(expected_frames),
            )
            expected_frames.extend(frames)
            quality_tokens.extend(outputs)
            execution_groups.append(receipt)
            remove_group(client, cohort_wire_ids, route_epoch)
        raw_outputs = detokenize_outputs(launch["codec"], quality_tokens)
        quality_rows = make_quality_rows(
            corpus,
            raw_outputs,
            corpus_role_sha256,
            monotonic_ns(),
            phase_id,
        )

        after = {}
        after_ns = {}
        for phone in ("op15", "op12"):
            value, event_ns = run_probe(
                launch["probes"][phone],
                "after",
                evidence_root / f"{phone}.after.json",
            )
            after[phone] = validate_probe_row(
                value,
                phone,
                launch["phones"][phone],
                f"{phone}.after",
            )
            after_ns[phone] = event_ns
            exact(after[phone]["active_sequences"], 0, f"{phone}.after.active")

        client.stop()
        connection.close()
        connection = None
        stop_processes(processes)
        stopped = True

        raw_logs = {
            name: read_regular(process.log_path, MAX_LOG_FILE)
            for name, process in processes.items()
        }
        frames, direct_cert = parse_direct_frames(
            raw_logs["op15_direct_relay"],
            expected_frames,
        )
        total_rows = sum(frame["rows"] for frame in expected_frames)
        session_op15, placement_cert_op15, nodes_op15 = parse_worker_log(
            raw_logs["op15_stagenet"],
            "op15",
            launch["phones"]["op15"],
            total_rows,
        )
        session_op12, placement_cert_op12, nodes_op12 = parse_worker_log(
            raw_logs["op12_stagenet"],
            "op12",
            launch["phones"]["op12"],
            total_rows,
        )
        runtime_processes = [
            parse_runtime_process(
                raw_logs["op12_stagenet"],
                "op12_stagenet",
                "op12",
                launch["phones"]["op12"]["boot_id"],
                launch["processes"]["op12_stagenet"],
                processes["op12_stagenet"].started_ns,
                processes["op12_stagenet"].completed_ns,
                session_op12["worker_pid"],
                before["op12"]["worker_start_ticks"],
            ),
            parse_runtime_process(
                raw_logs["op15_direct_relay"],
                "op15_direct_relay",
                "op15",
                launch["phones"]["op15"]["boot_id"],
                launch["processes"]["op15_direct_relay"],
                processes["op15_direct_relay"].started_ns,
                processes["op15_direct_relay"].completed_ns,
            ),
            parse_runtime_process(
                raw_logs["op15_stagenet"],
                "op15_stagenet",
                "op15",
                launch["phones"]["op15"]["boot_id"],
                launch["processes"]["op15_stagenet"],
                processes["op15_stagenet"].started_ns,
                processes["op15_stagenet"].completed_ns,
                session_op15["worker_pid"],
                before["op15"]["worker_start_ticks"],
            ),
        ]
        exact(
            direct_cert["head_endpoint"].split(":")[0],
            "127.0.0.1",
            "relay.head_host",
        )
        exact(
            direct_cert["tail_endpoint"].split(":")[0],
            launch["phones"]["op12"]["local_ipv4"],
            "relay.tail_host",
        )
        payload_bytes = sum(frame["activation_payload_bytes"] for frame in frames)
        exact(
            direct_cert["activation_payload_bytes"],
            payload_bytes,
            "relay.payload_total",
        )
        for phone in ("op15", "op12"):
            exact(
                before[phone]["process_swap_bytes"],
                0,
                f"{phone}.before.process_swap",
            )
            exact(
                after[phone]["process_swap_bytes"],
                0,
                f"{phone}.after.process_swap",
            )
            exact(
                after[phone]["system_swap_used_bytes"],
                before[phone]["system_swap_used_bytes"],
                f"{phone}.system_swap_growth",
            )
            require(
                min(
                    before[phone]["available_bytes"],
                    after[phone]["available_bytes"],
                )
                >= 512 * 1024 * 1024,
                f"E_PHONE_HEADROOM: {phone}",
            )
        op15_tx_delta = (
            after["op15"]["interface"]["tx_bytes"]
            - before["op15"]["interface"]["tx_bytes"]
        )
        op12_rx_delta = (
            after["op12"]["interface"]["rx_bytes"]
            - before["op12"]["interface"]["rx_bytes"]
        )
        require(op15_tx_delta >= payload_bytes, "E_OP15_LINK_COUNTER")
        require(op12_rx_delta >= payload_bytes, "E_OP12_LINK_COUNTER")

        placement_event_ns = max(after_ns.values())
        placement_op15_rows = placement_rows(
            "op15",
            launch["phones"]["op15"],
            before["op15"],
            after["op15"],
            nodes_op15,
            placement_event_ns,
        )
        placement_op12_rows = placement_rows(
            "op12",
            launch["phones"]["op12"],
            before["op12"],
            after["op12"],
            nodes_op12,
            placement_event_ns,
        )
        route_transfer_rows = make_transfer_rows(
            frames[:len(calls)],
            calls,
            mechanics_ns,
        )
        op15_runtime = runtime_record(
            launch["phones"]["op15"],
            before["op15"],
            after["op15"],
            session_op15,
            route_epoch,
        )
        op12_runtime = runtime_record(
            launch["phones"]["op12"],
            before["op12"],
            after["op12"],
            session_op12,
            route_epoch,
        )
        completed_ns = monotonic_ns()
        require(
            started_ns
            <= min(before_ns.values())
            < mechanics_ns
            <= placement_event_ns
            <= completed_ns,
            "E_CAPTURE_INTERVAL",
        )
        evidence_artifacts = []
        for path in sorted(evidence_root.glob("*")):
            if not path.is_file() or path.stat().st_size == 0:
                continue
            raw = read_regular(
                path,
                MAX_LOG_FILE if path.suffix == ".log" else MAX_PROCESS_OUTPUT,
            )
            evidence_artifacts.append({
                "bytes": len(raw),
                "path": str(path),
                "sha256": sha256(raw),
            })
        result = {
            "bridge_publication_rows": bridge_rows,
            "completed_ns": completed_ns,
            "direct_certificate": direct_cert,
            "direct_frames": frames,
            "evidence_artifacts": evidence_artifacts,
            "execution_groups": execution_groups,
            "history_sha256": sha256(history_raw),
            "launch_plan_sha256": sha256(launch_raw),
            "mechanics_rows": mechanics_rows,
            "mechanism_commands_sha256": mechanism_sha256,
            "model_id": MODEL_ID,
            "model_sha256": MODEL_SHA256,
            "op12_runtime": op12_runtime,
            "op15_runtime": op15_runtime,
            "phase_id": phase_id,
            "placement_certificates": {
                "op12": placement_cert_op12,
                "op15": placement_cert_op15,
            },
            "placement_op12_rows": placement_op12_rows,
            "placement_op15_rows": placement_op15_rows,
            "producer_sha256": sha256(source_raw),
            "quality_phone_rows": quality_rows,
            "raw_probes": {
                "op12": {
                    "after": after["op12"],
                    "after_ns": after_ns["op12"],
                    "before": before["op12"],
                    "before_ns": before_ns["op12"],
                },
                "op15": {
                    "after": after["op15"],
                    "after_ns": after_ns["op15"],
                    "before": before["op15"],
                    "before_ns": before_ns["op15"],
                },
            },
            "route_epoch": route_epoch,
            "route_transfer_rows": route_transfer_rows,
            "runtime_processes": runtime_processes,
            "schema": SCHEMA,
            "session_certificates": {
                "op12": session_op12,
                "op15": session_op15,
            },
            "started_ns": started_ns,
        }
        durable_write_new(output, canonical_bytes(result))
        return result
    finally:
        if connection is not None:
            connection.close()
        if processes and not stopped:
            for process in processes.values():
                process.kill()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase-id", required=True)
    parser.add_argument("--pre-dir", required=True)
    parser.add_argument("--started", type=int, required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--mechanism-commands-sha256", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--launch-plan", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    args = parser.parse_args()
    if not args.execute or args.confirm != "RUN_V24_PHONE_ROUTE_A_ONLY":
        print(
            "A_ONLY_PHONE_ROUTE_REFUSED: E_EXECUTION_NOT_CONFIRMED",
            file=sys.stderr,
        )
        return 2
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
            f"A_ONLY_PHONE_ROUTE_REFUSED: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
