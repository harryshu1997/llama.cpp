#!/usr/bin/python3 -I
"""Capture one independent B8 monolithic CUDA execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import struct
import subprocess
import sys
import time
from typing import Any


SCHEMA = "s39-cp0-r1-a-only-cuda-monolithic-raw-v1"
LAUNCH_SCHEMA = "s39-cp0-r1-a-only-cuda-monolithic-launch-v1"
HISTORY_SCHEMA = "s39-cp0-r1-a-only-b8-histories-v1"
MODEL_ID = "qwen3-14b-q4_k_m"
MODEL_PATH = "/home/zhihao/models/Qwen3-14B-Q4_K_M.gguf"
PHASE = "A_ONLY"

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
STAGE_V3_CAP_TERMINAL = 0x10
STAGE_V3_CAP_IDENTITY = 0x20

MAX_INPUT_BYTES = 1024 * 1024
MAX_OUTPUT_BYTES = 4 * 1024 * 1024
RUNTIME_PROCESS_PREFIX = b"RUNTIMEPROCESS "
RUNTIME_PROCESS_KEYS = {
    "boot_id",
    "launcher_path",
    "loaded_repo_component_ids",
    "pid",
    "schema",
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
        raw = (
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
    require(len(raw) <= MAX_OUTPUT_BYTES, "E_OUTPUT_SIZE")
    return raw


def read_regular(path: Path, maximum: int = MAX_INPUT_BYTES) -> bytes:
    require(path.is_absolute(), f"E_ABSOLUTE_PATH: {path}")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise CaptureError(f"E_OPEN: {path}: {error}") from error
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
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_mode,
    )
    require(identity(before) == identity(after), f"E_FILE_CHANGED: {path}")
    require(len(raw) == before.st_size, f"E_FILE_CHANGED: {path}")
    return bytes(raw)


def read_canonical(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = read_regular(path)
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CaptureError(f"E_JSON: {path}") from error
    require(type(value) is dict, f"E_TYPE: {path}")
    require(canonical_bytes(value) == raw, f"E_CANONICAL: {path}")
    return value, raw


def exact_keys(value: Any, keys: set[str], field: str) -> dict[str, Any]:
    require(type(value) is dict, f"E_TYPE: {field}")
    actual = set(value)
    require(actual == keys, f"E_KEYS: {field}")
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


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_process_start_ticks(pid: int) -> int:
    require(type(pid) is int and pid > 0, "E_RUNTIME_PROCESS_PID")
    path = Path(f"/proc/{pid}/stat")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise CaptureError(f"E_RUNTIME_PROCESS_STAT: {error}") from error
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), "E_RUNTIME_PROCESS_STAT_TYPE")
        raw = bytearray()
        while block := os.read(descriptor, 4096):
            raw.extend(block)
            require(len(raw) <= 64 * 1024, "E_RUNTIME_PROCESS_STAT_SIZE")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    require(
        (before.st_dev, before.st_ino, before.st_mode)
        == (after.st_dev, after.st_ino, after.st_mode),
        "E_RUNTIME_PROCESS_STAT_CHANGED",
    )
    closing = raw.rfind(b")")
    require(closing > 0 and raw[closing + 1:closing + 2] == b" ", "E_RUNTIME_PROCESS_STAT")
    try:
        stat_pid = int(bytes(raw[:raw.find(b" ")]))
        fields = bytes(raw[closing + 2:]).split()
        start_ticks = int(fields[19])
    except (IndexError, ValueError) as error:
        raise CaptureError("E_RUNTIME_PROCESS_STAT") from error
    require(stat_pid == pid and start_ticks > 0, "E_RUNTIME_PROCESS_STAT")
    return start_ticks


def load_runtime_process(log_path: Path) -> dict[str, Any]:
    raw = read_regular(log_path, 16 * 1024 * 1024)
    records = [
        line[len(RUNTIME_PROCESS_PREFIX):]
        for line in raw.splitlines()
        if line.startswith(RUNTIME_PROCESS_PREFIX)
    ]
    require(len(records) == 1, "E_RUNTIME_PROCESS_COUNT")
    try:
        value = json.loads(
            records[0].decode("ascii"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CaptureError("E_RUNTIME_PROCESS_JSON") from error
    exact_keys(value, RUNTIME_PROCESS_KEYS, "runtime_process")
    require(
        value["schema"] == "s39-runtime-process-source-v1",
        "E_RUNTIME_PROCESS_SCHEMA",
    )
    integer(value["pid"], "runtime_process.pid", 1)
    integer(value["start_ticks"], "runtime_process.start_ticks", 1)
    text(value["boot_id"], "runtime_process.boot_id", 128)
    launcher_path = text(
        value["launcher_path"],
        "runtime_process.launcher_path",
    )
    require(Path(launcher_path).is_absolute(), "E_RUNTIME_PROCESS_LAUNCHER")
    component_ids = value["loaded_repo_component_ids"]
    require(
        type(component_ids) is list
        and bool(component_ids)
        and component_ids == sorted(set(component_ids))
        and all(
            type(component_id) is str
            and 0 < len(component_id) <= 128
            and all(
                character.isalnum() or character in "._-"
                for character in component_id
            )
            for component_id in component_ids
        ),
        "E_RUNTIME_PROCESS_COMPONENTS",
    )
    dependencies = value["system_dependencies"]
    require(
        type(dependencies) is list and bool(dependencies),
        "E_RUNTIME_PROCESS_DEPENDENCIES",
    )
    previous_path = None
    for index, dependency in enumerate(dependencies):
        field = f"runtime_process.system_dependencies[{index}]"
        exact_keys(dependency, SYSTEM_DEPENDENCY_KEYS, field)
        path = text(dependency["path"], f"{field}.path")
        require(Path(path).is_absolute(), f"E_RUNTIME_PROCESS_DEP_PATH: {field}")
        require(
            previous_path is None or previous_path < path,
            "E_RUNTIME_PROCESS_DEP_ORDER",
        )
        previous_path = path
        for key in (
            "ctime_ns",
            "device_id",
            "inode",
            "mode",
            "mtime_ns",
            "size",
        ):
            integer(dependency[key], f"{field}.{key}")
        build_id = dependency["build_id"]
        require(
            build_id is None
            or (
                type(build_id) is str
                and 0 < len(build_id) <= 256
                and all(0x20 <= ord(character) <= 0x7E for character in build_id)
            ),
            f"E_RUNTIME_PROCESS_BUILD_ID: {field}",
        )
    return value


def wait_runtime_process(
    log_path: Path,
    process: subprocess.Popen[bytes],
    timeout_ms: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_ms / 1000
    last_error: CaptureError | OSError | None = None
    while time.monotonic() < deadline:
        require(process.poll() is None, "E_WORKER_EARLY_EXIT")
        try:
            return load_runtime_process(log_path)
        except (CaptureError, OSError) as error:
            last_error = error
            time.sleep(0.02)
    raise CaptureError(f"E_RUNTIME_PROCESS_TIMEOUT: {last_error}")


def durable_write_new(path: Path, raw: bytes) -> None:
    require(path.is_absolute() and not path.exists(), "E_OUTPUT_PATH")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o644,
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


def monotonic_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def load_phase(pre_dir: Path, phase_id: str) -> None:
    require(pre_dir.is_absolute() and pre_dir.is_dir(), "E_PRE_DIR")
    raw = read_regular(pre_dir / "phase_lock.jsonl")
    lines = raw.splitlines()
    require(len(lines) == 1, "E_PHASE_LOCK_ROWS")
    try:
        row = json.loads(
            lines[0].decode("ascii"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CaptureError("E_PHASE_LOCK_JSON") from error
    require(type(row) is dict, "E_PHASE_LOCK_TYPE")
    require(row.get("phase") == PHASE, "E_PHASE_LOCK_PHASE")
    require(row.get("phase_id") == phase_id, "E_PHASE_LOCK_ID")


def load_histories(
    path: Path,
    model_sha256: str,
) -> tuple[list[list[int]], int, bytes]:
    value, raw = read_canonical(path)
    exact_keys(
        value,
        {
            "histories",
            "history_width",
            "model_id",
            "model_sha256",
            "request_ids",
            "route_epoch",
            "schema",
        },
        "histories",
    )
    require(value["schema"] == HISTORY_SCHEMA, "E_HISTORY_SCHEMA")
    require(value["model_id"] == MODEL_ID, "E_HISTORY_MODEL")
    require(value["model_sha256"] == model_sha256, "E_HISTORY_MODEL_SHA256")
    require(value["request_ids"] == list(range(8)), "E_HISTORY_REQUEST_IDS")
    width = integer(value["history_width"], "history_width", 1)
    histories = value["histories"]
    require(type(histories) is list and len(histories) == 8, "E_HISTORY_BATCH")
    for index, history in enumerate(histories):
        require(
            type(history) is list
            and len(history) == width
            and all(type(token) is int and token >= 0 for token in history),
            f"E_HISTORY_ROW: {index}",
        )
    require(8 * width <= 64, "E_HISTORY_BATCH_LIMIT")
    route_epoch = integer(value["route_epoch"], "route_epoch", 1)
    return histories, route_epoch, raw


def validate_mechanism_commands(
    value: Any,
    command: list[str],
    expected_sha256: str,
) -> dict[str, list[list[str]]]:
    value = exact_keys(
        value,
        {"desktop", "op12", "op15"},
        "launch.mechanism_commands",
    )
    for endpoint in ("desktop", "op12", "op15"):
        commands = value[endpoint]
        require(
            type(commands) is list and bool(commands),
            f"E_MECHANISM_COMMANDS: {endpoint}",
        )
        for index, argv in enumerate(commands):
            field = f"launch.mechanism_commands.{endpoint}[{index}]"
            require(
                type(argv) is list
                and bool(argv)
                and all(
                    type(item) is str
                    and 0 < len(item) <= 4096
                    and "\x00" not in item
                    and "\n" not in item
                    for item in argv
                ),
                f"E_MECHANISM_ARGV: {field}",
            )
            launcher = Path(argv[0])
            require(launcher.is_absolute(), f"E_MECHANISM_PATH: {field}")
            require(
                launcher.name not in {"bash", "dash", "sh", "zsh"}
                and "-c" not in argv,
                f"E_MECHANISM_SHELL: {field}",
            )
    require(len(value["desktop"]) == 9, "E_MECHANISM_DESKTOP_COUNT")
    require(
        value["desktop"][8] == command
        and sum(argv == command for argv in value["desktop"]) == 1,
        "E_MECHANISM_MONOLITHIC_COMMAND",
    )
    require(
        sha256(canonical_bytes(value)) == expected_sha256,
        "E_MECHANISM_COMMANDS_CONTENT",
    )
    return value


def validate_managed_command(
    command: list[str],
    environment: dict[str, str],
) -> list[str]:
    require(
        len(command) == 5
        and command[1] == "--plan-json"
        and command[3] == "--plan-sha256",
        "E_MANAGED_COMMAND_ARGV",
    )
    raw = command[2]
    require(
        command[4] == sha256(raw.encode("ascii")),
        "E_MANAGED_COMMAND_SHA256",
    )
    try:
        plan = json.loads(
            raw,
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as error:
        raise CaptureError("E_MANAGED_COMMAND_JSON") from error
    exact_keys(
        plan,
        {
            "android",
            "bundle_id",
            "components",
            "endpoint",
            "launcher_component_id",
            "mode",
            "route",
            "schema",
        },
        "managed.plan",
    )
    require(
        canonical_bytes(plan)[:-1].decode("ascii") == raw,
        "E_MANAGED_COMMAND_CANONICAL",
    )
    require(
        plan["schema"] == "s39-managed-runtime-launch-plan-v1"
        and plan["bundle_id"] == "cuda_monolithic"
        and plan["endpoint"] == "cuda"
        and plan["mode"] == "local_cuda"
        and plan["android"] is None,
        "E_MANAGED_COMMAND_IDENTITY",
    )
    route = exact_keys(
        plan["route"],
        {"argv", "cwd", "environment", "kind"},
        "managed.plan.route",
    )
    require(route["kind"] == "local_exec", "E_MANAGED_COMMAND_ROUTE")
    require(route["environment"] == environment, "E_MANAGED_COMMAND_ENVIRONMENT")
    target = route["argv"]
    require(
        type(target) is list
        and bool(target)
        and all(type(item) is str and bool(item) for item in target),
        "E_MANAGED_COMMAND_TARGET",
    )
    launchers = [
        component
        for component in plan["components"]
        if type(component) is dict
        and component.get("component_id") == plan["launcher_component_id"]
    ]
    require(
        len(launchers) == 1 and launchers[0].get("path") == target[0],
        "E_MANAGED_COMMAND_LAUNCHER",
    )
    required = {
        "--backend": "CUDA0",
        "--layer-end": "40",
        "--layer-start": "0",
        "--mode": "monov3",
        "--model": MODEL_PATH,
    }
    for flag, expected in required.items():
        require(target.count(flag) == 1, f"E_MANAGED_COMMAND_FLAG: {flag}")
        index = target.index(flag)
        require(
            index + 1 < len(target) and target[index + 1] == expected,
            f"E_MANAGED_COMMAND_VALUE: {flag}",
        )
    return target


def load_launch(
    path: Path,
    model_sha256: str,
) -> tuple[dict[str, Any], bytes]:
    value, raw = read_canonical(path)
    exact_keys(
        value,
        {
            "command",
            "cwd",
            "expected_capabilities",
            "env",
            "expected_file_type",
            "expected_max_streams",
            "expected_n_batch",
            "expected_n_ctx_seq",
            "expected_n_embd",
            "expected_n_layer",
            "expected_n_ubatch",
            "host",
            "io_timeout_ms",
            "mechanism_commands",
            "mechanism_commands_sha256",
            "model_id",
            "model_sha256",
            "port",
            "schema",
            "shutdown_timeout_ms",
            "startup_timeout_ms",
        },
        "launch",
    )
    require(value["schema"] == LAUNCH_SCHEMA, "E_LAUNCH_SCHEMA")
    require(value["model_id"] == MODEL_ID, "E_LAUNCH_MODEL")
    require(value["model_sha256"] == model_sha256, "E_LAUNCH_MODEL_SHA256")
    digest(
        value["mechanism_commands_sha256"],
        "launch.mechanism_commands_sha256",
    )
    command = value["command"]
    require(
        type(command) is list
        and bool(command)
        and all(type(item) is str and 0 < len(item) <= 4096 for item in command),
        "E_LAUNCH_COMMAND",
    )
    executable = Path(command[0])
    require(executable.is_absolute(), "E_LAUNCH_EXECUTABLE")
    validate_mechanism_commands(
        value["mechanism_commands"],
        command,
        value["mechanism_commands_sha256"],
    )
    cwd = Path(text(value["cwd"], "launch.cwd"))
    require(cwd.is_absolute() and cwd.is_dir(), "E_LAUNCH_CWD")
    environment = value["env"]
    require(type(environment) is dict and bool(environment), "E_LAUNCH_ENV")
    for key, item in environment.items():
        text(key, "launch.env.key", 128)
        text(item, f"launch.env.{key}", 4096)
        require("=" not in key, f"E_LAUNCH_ENV_KEY: {key}")
    validate_managed_command(command, environment)
    host = text(value["host"], "launch.host", 255)
    require(host in ("127.0.0.1", "::1"), "E_LAUNCH_HOST")
    port = integer(value["port"], "launch.port", 1)
    require(port <= 65535, "E_LAUNCH_PORT")
    integer(value["expected_file_type"], "launch.expected_file_type")
    integer(value["expected_n_layer"], "launch.expected_n_layer", 1)
    integer(value["expected_n_embd"], "launch.expected_n_embd", 1)
    integer(value["expected_max_streams"], "launch.expected_max_streams", 8)
    integer(value["expected_n_ctx_seq"], "launch.expected_n_ctx_seq", 1)
    integer(value["expected_n_batch"], "launch.expected_n_batch", 64)
    integer(value["expected_n_ubatch"], "launch.expected_n_ubatch", 64)
    integer(value["expected_capabilities"], "launch.expected_capabilities", 1)
    for name in ("io_timeout_ms", "shutdown_timeout_ms", "startup_timeout_ms"):
        duration = integer(value[name], f"launch.{name}", 1)
        require(duration <= 600_000, f"E_LAUNCH_TIMEOUT: {name}")
    return value, raw


def pack_i32(values) -> bytes:
    values = tuple(values)
    return struct.pack(f"<{len(values)}i", *values)


def pack_i64(values) -> bytes:
    values = tuple(values)
    return struct.pack(f"<{len(values)}q", *values)


class StageClient:
    def __init__(self, connection: socket.socket):
        self.connection = connection
        self.n_batch = 0
        self.n_ubatch = 0

    def recv_exact(self, size: int) -> bytes:
        require(size >= 0, "E_RECV_SIZE")
        result = bytearray()
        while len(result) < size:
            block = self.connection.recv(size - len(result))
            require(bool(block), "E_UNEXPECTED_EOF")
            result.extend(block)
        return bytes(result)

    def recv_i32(self, count: int) -> tuple[int, ...]:
        return struct.unpack(f"<{count}i", self.recv_exact(count * 4))

    def hello(
        self,
        expected_n_layer: int,
        expected_file_type: int,
        model_sha256: str,
        expected_n_embd: int,
        expected_max_streams: int,
        expected_n_ctx_seq: int,
        expected_n_batch: int,
        expected_n_ubatch: int,
        expected_capabilities: int,
    ) -> None:
        self.connection.sendall(pack_i32([STAGE_V3_HELLO]))
        words = self.recv_i32(11)
        require(words[0] == STAGE_V3_MAGIC, "E_HELLO_MAGIC")
        require(words[1] == STAGE_V3_VERSION, "E_HELLO_VERSION")
        layer_start, layer_end, n_layer = words[2:5]
        n_embd = words[5]
        max_streams = words[6]
        n_ctx_seq = words[7]
        self.n_batch = words[8]
        self.n_ubatch = words[9]
        capabilities = words[10]
        require(
            layer_start == 0
            and layer_end == expected_n_layer
            and n_layer == expected_n_layer,
            "E_HELLO_LAYERS",
        )
        require(n_embd == expected_n_embd, "E_HELLO_EMBD")
        require(max_streams == expected_max_streams, "E_HELLO_STREAMS")
        require(n_ctx_seq == expected_n_ctx_seq, "E_HELLO_CTX")
        require(
            self.n_batch == expected_n_batch
            and self.n_ubatch == expected_n_ubatch,
            "E_HELLO_BATCH",
        )
        require(capabilities == expected_capabilities, "E_HELLO_CAPABILITIES")
        require(capabilities & STAGE_V3_CAP_TERMINAL, "E_HELLO_TERMINAL")
        require(capabilities & STAGE_V3_CAP_IDENTITY, "E_HELLO_IDENTITY")
        self.connection.sendall(pack_i32([STAGE_V3_IDENTITY]))
        magic, version, file_type = self.recv_i32(3)
        model_digest = self.recv_exact(32).hex()
        require(
            magic == STAGE_IDENTITY_MAGIC
            and version == STAGE_IDENTITY_VERSION
            and file_type == expected_file_type
            and model_digest == model_sha256,
            "E_MODEL_IDENTITY",
        )

    def status(self) -> tuple[int, int, bool]:
        self.connection.sendall(pack_i32([STAGE_V3_STATUS, STAGE_V3_VERSION]))
        code, version, active, maximum, draining = self.recv_i32(5)
        require(code == 0 and version == STAGE_V3_VERSION, "E_STATUS")
        require(
            0 <= active <= maximum and maximum >= 8 and draining in (0, 1),
            "E_STATUS_VALUE",
        )
        return active, maximum, bool(draining)

    def batch(
        self,
        rows: list[tuple[int, int, int, int, int]],
    ) -> list[int]:
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
        require(self.recv_i32(1)[0] == 0, "E_BATCH_STATUS")
        count, width = self.recv_i32(2)
        require(count == len(rows) and width == 0, "E_BATCH_RESPONSE_SHAPE")
        request_ids = self.recv_exact(8 * count)
        route_epochs = self.recv_exact(8 * count)
        seq_ids = self.recv_exact(4 * count)
        positions = self.recv_exact(4 * count)
        tokens = self.recv_i32(count)
        require(
            struct.unpack(f"<{count}q", request_ids)
            == tuple(row[0] for row in rows)
            and struct.unpack(f"<{count}q", route_epochs)
            == tuple(row[1] for row in rows)
            and struct.unpack(f"<{count}i", seq_ids)
            == tuple(row[2] for row in rows)
            and struct.unpack(f"<{count}i", positions)
            == tuple(row[3] for row in rows),
            "E_BATCH_LINEAGE",
        )
        require(all(token >= 0 for token in tokens), "E_BATCH_TOKEN")
        return list(tokens)

    def remove(
        self,
        request_id: int,
        route_epoch: int,
        seq_id: int,
    ) -> None:
        payload = pack_i32([STAGE_V3_SEQ_REMOVE, STAGE_V3_VERSION, seq_id])
        payload += pack_i64([request_id, route_epoch])
        self.connection.sendall(payload)
        code, version, active, maximum, draining = self.recv_i32(5)
        require(
            code == 0
            and version == STAGE_V3_VERSION
            and 0 <= active <= maximum
            and draining in (0, 1),
            "E_REMOVE",
        )

    def stop(self) -> None:
        self.connection.sendall(pack_i32([STAGE_STOP]))


def connect(
    host: str,
    port: int,
    startup_timeout_ms: int,
    io_timeout_ms: int,
    process: subprocess.Popen[bytes],
) -> StageClient:
    deadline = time.monotonic() + startup_timeout_ms / 1000
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        require(process.poll() is None, "E_WORKER_EARLY_EXIT")
        try:
            connection = socket.create_connection((host, port), timeout=0.2)
            connection.settimeout(io_timeout_ms / 1000)
            return StageClient(connection)
        except OSError as error:
            last_error = error
            time.sleep(0.02)
    raise CaptureError(f"E_WORKER_CONNECT: {last_error}")


def select_predictions(
    tokens: list[int],
    rows: list[tuple[int, int, int, int, int]],
    width: int,
) -> list[int]:
    require(len(tokens) == len(rows) == 8 * width, "E_PREDICTION_SHAPE")
    return [tokens[(sequence + 1) * width - 1] for sequence in range(8)]


def execute(
    launch: dict[str, Any],
    histories: list[list[int]],
    route_epoch: int,
    model_sha256: str,
    log_path: Path,
) -> tuple[
    list[list[int]],
    list[dict[str, Any]],
    int,
    int,
    dict[str, Any],
]:
    log_file = None
    process: subprocess.Popen[bytes] | None = None
    connection: socket.socket | None = None
    state_before = -1
    state_after = -1
    try:
        require(not log_path.exists(), "E_LOG_EXISTS")
        log_file = log_path.open("xb")
        process = subprocess.Popen(
            launch["command"],
            cwd=launch["cwd"],
            env=launch["env"],
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        client = connect(
            launch["host"],
            launch["port"],
            launch["startup_timeout_ms"],
            launch["io_timeout_ms"],
            process,
        )
        connection = client.connection
        client.hello(
            launch["expected_n_layer"],
            launch["expected_file_type"],
            model_sha256,
            launch["expected_n_embd"],
            launch["expected_max_streams"],
            launch["expected_n_ctx_seq"],
            launch["expected_n_batch"],
            launch["expected_n_ubatch"],
            launch["expected_capabilities"],
        )
        runtime_process_source = wait_runtime_process(
            log_path,
            process,
            launch["startup_timeout_ms"],
        )
        process_pid = process.pid
        require(
            runtime_process_source["pid"] == process_pid,
            "E_RUNTIME_PROCESS_PID",
        )
        live_start_ticks = read_process_start_ticks(process_pid)
        require(
            runtime_process_source["start_ticks"] == live_start_ticks,
            "E_RUNTIME_PROCESS_START_TICKS",
        )
        identity_probe_raw = canonical_bytes({
            "pid": process_pid,
            "schema": "s39-cp0-r1-local-process-probe-v1",
            "start_ticks": live_start_ticks,
        })
        durable_write_new(
            log_path.with_name(log_path.name + ".process.json"),
            identity_probe_raw,
        )
        runtime_process_observed_ns = monotonic_ns()
        state_before = client.status()[0]
        require(state_before == 0, "E_STATE_BEFORE")

        width = len(histories[0])
        wire_request_ids = [1001 + sequence for sequence in range(8)]
        prefill_rows = [
            (
                wire_request_ids[sequence],
                route_epoch,
                sequence,
                position,
                histories[sequence][position],
            )
            for sequence in range(8)
            for position in range(width)
        ]
        predictions = select_predictions(
            client.batch(prefill_rows),
            prefill_rows,
            width,
        )
        outputs = [[token] for token in predictions]
        calls = [{
            "call_index": 0,
            "n_seqs": 8,
            "n_tokens": len(prefill_rows),
            "phase": "prefill",
        }]
        for offset in range(1, 9):
            position = width + offset - 1
            rows = [
                (
                    wire_request_ids[sequence],
                    route_epoch,
                    sequence,
                    position,
                    predictions[sequence],
                )
                for sequence in range(8)
            ]
            predictions = client.batch(rows)
            if offset < 8:
                for sequence, token in enumerate(predictions):
                    outputs[sequence].append(token)
            calls.append({
                "call_index": offset,
                "n_seqs": 8,
                "n_tokens": 8,
                "phase": "decode",
            })
        require(all(len(row) == 8 for row in outputs), "E_CONTINUATION_COUNT")
        for sequence in range(8):
            client.remove(
                wire_request_ids[sequence],
                route_epoch,
                sequence,
            )
        state_after = client.status()[0]
        require(state_after == 0, "E_STATE_AFTER")
        client.stop()
        connection.close()
        connection = None
        try:
            return_code = process.wait(
                timeout=launch["shutdown_timeout_ms"] / 1000
            )
        except subprocess.TimeoutExpired as error:
            raise CaptureError("E_WORKER_STOP_TIMEOUT") from error
        require(return_code == 0, f"E_WORKER_EXIT: {return_code}")
        process = None
        require(
            load_runtime_process(log_path) == runtime_process_source,
            "E_RUNTIME_PROCESS_CHANGED",
        )
        runtime_process = {
            "boot_id": runtime_process_source["boot_id"],
            "bundle_id": "cuda_monolithic",
            "endpoint": "cuda",
            "identity_probe_sha256": sha256(identity_probe_raw),
            "launcher_path": runtime_process_source["launcher_path"],
            "loaded_repo_component_ids": runtime_process_source[
                "loaded_repo_component_ids"
            ],
            "observed_ns": runtime_process_observed_ns,
            "pid": process_pid,
            "start_ticks": live_start_ticks,
            "system_dependencies": runtime_process_source[
                "system_dependencies"
            ],
        }
        return outputs, calls, state_before, state_after, runtime_process
    finally:
        if connection is not None:
            connection.close()
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if log_file is not None:
            log_file.flush()
            os.fsync(log_file.fileno())
            log_file.close()
        if log_path.exists():
            metadata = log_path.stat(follow_symlinks=False)
            require(
                stat.S_ISREG(metadata.st_mode)
                and metadata.st_size <= 16 * 1024 * 1024,
                "E_LOG_FILE",
            )


def make_rows(
    histories: list[list[int]],
    continuations: list[list[int]],
    calls: list[dict[str, Any]],
    model_sha256: str,
    program_sha256: str,
    state_before: int,
    state_after: int,
    event_ns: int,
) -> list[dict[str, Any]]:
    rows = [{
        "backend": "CUDA0",
        "call_shapes": calls,
        "event_ns": event_ns,
        "kind": "meta",
        "model_id": MODEL_ID,
        "model_sha256": model_sha256,
        "program_sha256": program_sha256,
        "state_count_after": state_after,
        "state_count_before": state_before,
    }]
    for request_id in range(8):
        rows.append({
            "continuation_tokens": continuations[request_id],
            "event_ns": event_ns,
            "input_tokens": histories[request_id],
            "kind": "request",
            "model_id": MODEL_ID,
            "model_sha256": model_sha256,
            "owner_after": "RELEASED",
            "owner_before": "CUDA",
            "ownership_epoch_after": 2,
            "ownership_epoch_before": 1,
            "positions": list(range(len(histories[request_id]))),
            "request_id": request_id,
        })
    return rows


def capture(args: argparse.Namespace) -> dict[str, Any]:
    started_ns = monotonic_ns()
    acquisition_started_ns = integer(args.started, "started", 1)
    require(acquisition_started_ns <= started_ns, "E_ACQUISITION_ORDER")
    phase_id = text(args.phase_id, "phase_id", 128)
    require(
        phase_id.startswith("cp0-r1-v23-a-only-")
        and all(character.isalnum() or character in ".-_" for character in phase_id),
        "E_PHASE_ID",
    )
    command_plan_sha256 = digest(args.plan, "plan")
    mechanism_sha256 = digest(
        args.mechanism_commands_sha256,
        "mechanism_commands_sha256",
    )
    model_sha256 = digest(args.model_sha256, "model_sha256")
    output = Path(args.output)
    require(output.is_absolute() and not output.exists(), "E_OUTPUT")
    load_phase(Path(args.pre_dir), phase_id)
    histories, route_epoch, history_raw = load_histories(
        Path(args.histories),
        model_sha256,
    )
    launch, launch_raw = load_launch(Path(args.launch_plan), model_sha256)
    require(
        mechanism_sha256 == launch["mechanism_commands_sha256"],
        "E_MECHANISM_COMMANDS_SHA256",
    )
    source_raw = read_regular(Path(__file__).resolve())
    program_sha256 = sha256(
        b"s39:cuda-monolithic-program:v1\0"
        + bytes.fromhex(command_plan_sha256)
        + bytes.fromhex(sha256(source_raw))
        + bytes.fromhex(sha256(launch_raw))
        + bytes.fromhex(sha256(history_raw))
    )
    (
        continuations,
        calls,
        state_before,
        state_after,
        runtime_process,
    ) = execute(
        launch,
        histories,
        route_epoch,
        model_sha256,
        output.with_suffix(output.suffix + ".worker.log"),
    )
    event_ns = monotonic_ns()
    rows = make_rows(
        histories,
        continuations,
        calls,
        model_sha256,
        program_sha256,
        state_before,
        state_after,
        event_ns,
    )
    completed_ns = monotonic_ns()
    require(started_ns < event_ns <= completed_ns, "E_CAPTURE_INTERVAL")
    result = {
        "completed_ns": completed_ns,
        "mechanism_commands_sha256": mechanism_sha256,
        "model_id": MODEL_ID,
        "model_sha256": model_sha256,
        "oracle_cuda_monolithic_rows": rows,
        "phase_id": phase_id,
        "runtime_process": runtime_process,
        "schema": SCHEMA,
        "started_ns": started_ns,
    }
    durable_write_new(output, canonical_bytes(result))
    return result


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
            f"A_ONLY_CUDA_MONOLITHIC_REFUSED: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
