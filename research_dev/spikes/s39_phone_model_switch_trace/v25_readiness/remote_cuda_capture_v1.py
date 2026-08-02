#!/usr/bin/env python3
"""Run a frozen V2.4 CUDA producer on the pinned RTX host."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import sys
import time
import types
from typing import Any


PLAN_SCHEMA = "s39-v25-remote-cuda-wrapper-plan-v1"
RECEIPT_SCHEMA = "s39-v25-remote-cuda-wrapper-receipt-v1"
FETCH_SCHEMA = "s39-v25-remote-cuda-fetch-v1"
FETCH_TRANSPORT_SCHEMA = "s39-v25-remote-cuda-fetch-transport-v1"
TRANSPORT_SCHEMA = "s39-managed-remote-transport-process-v1"
RUNTIME_SCHEMA = "s39-runtime-process-source-v1"
MANAGED_CLEANUP_SCHEMA = "s39-managed-remote-cleanup-v1"
MANAGED_CLEANUP_PREFIX = b"REMOTECLEANUP "
PHASE = "A_ONLY"
CONFIRMATION = "RUN-S39-V25-REMOTE-CUDA"
MAX_JSON = 8 * 1024 * 1024
MAX_REMOTE_OUTPUT = 256 * 1024 * 1024
MAX_INT = (1 << 63) - 1

ROLE_CONFIG = {
    "cuda_monolithic": {
        "producer_name": "cuda_monolithic_v1.py",
        "result_schema": "s39-cp0-r1-v24-cuda-monolithic-raw-v1",
        "sequence_index": 1,
    },
    "joint_phone_cuda": {
        "producer_name": "joint_phone_cuda_v1.py",
        "result_schema": "s39-cp0-r1-v24-joint-phone-cuda-raw-v1",
        "sequence_index": 2,
    },
}

ARTIFACT_KEYS = {"bytes", "path", "sha256", "stat"}
JOINT_BINDING_KEYS = {
    "adb",
    "adb_server_port",
    "adb_server_process",
    "capture_plan",
    "cuda_launch_plan",
    "op12_selector",
    "op15_selector",
    "phone_launch_plan",
}
ADB_SERVER_KEYS = {
    "argv",
    "boot_id",
    "executable_path",
    "listen_host",
    "listen_port",
    "pid",
    "start_ticks",
}
STAT_KEYS = {
    "build_id",
    "ctime_ns",
    "device_id",
    "inode",
    "mode",
    "mtime_ns",
    "size",
}
RUNTIME_DEPENDENCY_KEYS = STAT_KEYS | {"path", "sha256"}
PLAN_KEYS = {
    "frozen_producer",
    "joint_bindings",
    "local_python",
    "managed_launcher",
    "managed_plan_sha256",
    "phase",
    "phase_id",
    "producer_argv",
    "remote_output_path",
    "role",
    "schema",
    "sequence_index",
    "timeout_seconds",
    "v24_phase_id",
}
TRANSPORT_KEYS = {
    "argv",
    "bundle_id",
    "endpoint",
    "host_boot_id",
    "managed_launcher_pid",
    "managed_launcher_start_ticks",
    "observed_ns",
    "pid",
    "plan_sha256",
    "remote_boot_id",
    "schema",
    "start_ticks",
}
RUNTIME_KEYS = {
    "boot_id",
    "bundle_id",
    "endpoint",
    "launcher_path",
    "loaded_repo_component_ids",
    "observed_ns",
    "pid",
    "schema",
    "start_ticks",
    "system_dependencies",
}
REMOTE_RUNTIME_KEYS = {
    "boot_id",
    "bundle_id",
    "controller_clock",
    "controller_observed_ns",
    "endpoint",
    "launch_token",
    "launcher_path",
    "loaded_repo_component_ids",
    "pgid",
    "pid",
    "remote_observed_ns",
    "remote_clock",
    "schema",
    "start_ticks",
    "system_dependencies",
}
MANAGED_CLEANUP_KEYS = {
    "absent",
    "boot_id",
    "clock",
    "gpu_uuid",
    "launch_token",
    "matching_nvml_pids",
    "matching_process_groups",
    "matching_processes",
    "observed_ns",
    "pgid",
    "pid",
    "schema",
    "start_ticks",
}
FETCH_TRANSPORT_KEYS = {
    "argv",
    "completed_ns",
    "pid",
    "schema",
    "start_ticks",
    "started_ns",
}
RECEIPT_KEYS = {
    "cleanup",
    "completed_ns",
    "execution_transport_process",
    "fetch_transport_process",
    "frozen_producer",
    "gpu_uuid",
    "adb_server_process",
    "joint_bindings",
    "local_python",
    "local_evidence_artifacts",
    "local_result_artifact",
    "managed_remote_cleanup",
    "managed_plan_sha256",
    "phase",
    "phase_id",
    "remote_boot_id",
    "remote_evidence_artifacts",
    "remote_execution_interval",
    "remote_producer_process",
    "remote_result_artifact",
    "role",
    "schema",
    "sequence_index",
    "started_ns",
    "system_swap_used_bytes",
    "phone_processes_absent",
    "v24_phase_id",
    "wrapper_plan_sha256",
}
CLEANUP_KEYS = {
    "execution_transport_absent",
    "fetch_transport_absent",
    "local_forward_listener_absent",
    "observed_ns",
    "remote_cuda_processes_absent",
    "remote_cleanup_observed_ns",
}

REMOTE_FETCH_SOURCE = r'''
import base64
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

MAX_BYTES = 256 * 1024 * 1024

def fail(message):
    raise RuntimeError(message)

def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")

def decode(index):
    raw = base64.b64decode(sys.argv[index], validate=True)
    if not 0 < len(raw) <= 8 * 1024 * 1024:
        fail("E_ARGUMENT_SIZE")
    value = json.loads(raw.decode("ascii"))
    if type(value) is not dict or canonical(value) != raw:
        fail("E_ARGUMENT_CANONICAL")
    return value

def file_stat(metadata):
    return {
        "build_id": None,
        "ctime_ns": metadata.st_ctime_ns,
        "device_id": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": metadata.st_mode,
        "mtime_ns": metadata.st_mtime_ns,
        "size": metadata.st_size,
    }

def snapshot(path_text, include_bytes):
    path = Path(path_text)
    if not path.is_absolute():
        fail("E_PATH")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_BYTES:
            fail("E_FILE")
        digest = hashlib.sha256()
        raw = bytearray()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            if include_bytes:
                raw.extend(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
        item.st_mode,
    )
    if identity(before) != identity(after):
        fail("E_FILE_CHANGED")
    result = {
        "bytes": before.st_size,
        "path": str(path),
        "sha256": digest.hexdigest(),
        "stat": file_stat(before),
    }
    if include_bytes:
        if len(raw) != before.st_size:
            fail("E_FILE_CHANGED")
        result["content_base64"] = base64.b64encode(raw).decode("ascii")
    return result

def start_ticks(pid):
    raw = (Path("/proc") / str(pid) / "stat").read_bytes()
    closing = raw.rfind(b")")
    if closing <= 0 or raw[closing + 1:closing + 2] != b" ":
        fail("E_PROCESS_STAT")
    fields = raw[closing + 2:].split()
    if int(raw[:raw.find(b" ")]) != pid:
        fail("E_PROCESS_STAT")
    return int(fields[19])

def process_absent(pid, ticks):
    try:
        observed = start_ticks(pid)
    except FileNotFoundError:
        return True
    if observed != ticks:
        fail("E_PROCESS_REUSED")
    return False

def swap_used():
    values = {}
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        key, separator, tail = line.partition(":")
        if separator and key in {"SwapTotal", "SwapFree"}:
            fields = tail.split()
            if len(fields) != 2 or fields[1] != "kB":
                fail("E_SWAP")
            values[key] = int(fields[0]) * 1024
    if set(values) != {"SwapTotal", "SwapFree"}:
        fail("E_SWAP")
    return values["SwapTotal"] - values["SwapFree"]

def nvml_pids(executable, gpu_uuid):
    completed = subprocess.run(
        [
            executable,
            "--id",
            gpu_uuid,
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=15,
    )
    if completed.returncode != 0 or completed.stderr:
        fail("E_NVML")
    rows = []
    for line in completed.stdout.decode("ascii").splitlines():
        value = line.strip()
        if value:
            rows.append(int(value))
    return sorted(set(rows))

def process_snapshot(expected):
    pid = expected["pid"]
    if start_ticks(pid) != expected["start_ticks"]:
        fail("E_PROCESS_TICKS")
    root = Path("/proc") / str(pid)
    executable = os.path.realpath(os.readlink(root / "exe"))
    raw = (root / "cmdline").read_bytes()
    if not raw.endswith(b"\0"):
        fail("E_PROCESS_CMDLINE")
    argv = [
        item.decode("ascii")
        for item in raw[:-1].split(b"\0")
    ]
    if executable != expected["executable_path"] or argv != expected["argv"]:
        fail("E_PROCESS_IDENTITY")
    return {
        "argv": argv,
        "boot_id": expected["boot_id"],
        "executable_path": executable,
        "listen_host": expected["listen_host"],
        "listen_port": expected["listen_port"],
        "observed_ns": __import__("time").clock_gettime_ns(
            __import__("time").CLOCK_MONOTONIC_RAW
        ),
        "pid": pid,
        "start_ticks": expected["start_ticks"],
    }

def listener_inode(host, port):
    if host != "127.0.0.1":
        fail("E_LISTEN_HOST")
    expected_address = "0100007F"
    expected_port = f"{port:04X}"
    matches = []
    for name in ("/proc/net/tcp",):
        for line in Path(name).read_text(encoding="ascii").splitlines()[1:]:
            fields = line.split()
            address, separator, observed_port = fields[1].partition(":")
            if (
                separator
                and address == expected_address
                and observed_port == expected_port
                and fields[3] == "0A"
            ):
                matches.append(fields[9])
    if len(matches) != 1:
        fail("E_LISTENER")
    return matches[0]

def process_owns_socket(pid, inode):
    root = Path("/proc") / str(pid) / "fd"
    target = f"socket:[{inode}]"
    for item in root.iterdir():
        try:
            if os.readlink(item) == target:
                return True
        except FileNotFoundError:
            continue
    return False

def adb_run(binding, selector, command):
    completed = subprocess.run(
        [
            binding["adb"]["path"],
            "-P",
            str(binding["adb_server_port"]),
            "-s",
            selector,
            *command,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    if completed.returncode != 0 or completed.stderr:
        fail("E_ADB")
    return completed.stdout

def phone_cleanup(result, binding):
    rows = []
    endpoints = {
        "op12": binding["op12_selector"],
        "op15": binding["op15_selector"],
    }
    boots = {
        "op12": result["op12_runtime"]["boot_id"],
        "op15": result["op15_runtime"]["boot_id"],
    }
    for endpoint, selector in sorted(endpoints.items()):
        observed_boot = adb_run(
            binding,
            selector,
            ["shell", "cat", "/proc/sys/kernel/random/boot_id"],
        ).decode("ascii").strip()
        if observed_boot != boots[endpoint]:
            fail("E_PHONE_BOOT")
    for process in result["runtime_processes"]:
        endpoint = process["endpoint"]
        if endpoint not in endpoints:
            continue
        pid = process["pid"]
        ticks = process["start_ticks"]
        raw = adb_run(
            binding,
            endpoints[endpoint],
            [
                "shell",
                "sh",
                "-c",
                f"cat /proc/{pid}/stat 2>/dev/null || true",
            ],
        )
        if not raw.strip():
            state = "absent"
        else:
            closing = raw.rfind(b")")
            fields = raw[closing + 2:].split()
            observed_ticks = int(fields[19])
            if observed_ticks == ticks:
                fail("E_PHONE_PROCESS_LIVE")
            state = "reused"
        rows.append({
            "endpoint": endpoint,
            "pid": pid,
            "start_ticks": ticks,
            "state": state,
        })
    if len(rows) != 3:
        fail("E_PHONE_PROCESS_COUNT")
    return rows

plan = decode(1)
interpreter = plan["interpreter"]
if (
    os.path.realpath(sys.executable) != interpreter["path"]
    or snapshot(interpreter["path"], False) != interpreter
):
    fail("E_INTERPRETER")
if Path("/proc/sys/kernel/random/boot_id").read_text(
    encoding="ascii"
).strip() != plan["boot_id"]:
    fail("E_BOOT")

primary = snapshot(plan["output_path"], True)
raw = base64.b64decode(primary["content_base64"], validate=True)
try:
    result = json.loads(raw.decode("ascii"))
except Exception:
    fail("E_RESULT_JSON")
if canonical(result) + b"\n" != raw:
    fail("E_RESULT_CANONICAL")
if result.get("schema") != plan["result_schema"]:
    fail("E_RESULT_SCHEMA")

paths = {}
if plan["role"] == "cuda_monolithic":
    paths[result["worker_log_path"]] = {
        "bytes": result["worker_log_bytes"],
        "sha256": result["worker_log_sha256"],
    }
else:
    direct = [
        result["capture_plan_artifact"],
        result["joint_producer_artifact"],
        result["cuda_evidence"]["fragment"],
        result["cuda_evidence"]["receipt_artifact"],
        result["phone_evidence"]["fragment"],
        result["phone_evidence"]["receipt_artifact"],
    ]
    for row in direct:
        paths[row["path"]] = {
            "bytes": row["bytes"],
            "sha256": row["sha256"],
        }
artifacts = []
for path, expected in sorted(paths.items()):
    observed = snapshot(path, True)
    if (
        observed["bytes"] != expected["bytes"]
        or observed["sha256"] != expected["sha256"]
    ):
        fail("E_REFERENCED_ARTIFACT")
    artifacts.append(observed)

if plan["role"] == "joint_phone_cuda":
    fragments = [
        item
        for item in artifacts
        if item["path"] in {
            result["cuda_evidence"]["fragment"]["path"],
            result["phone_evidence"]["fragment"]["path"],
        }
    ]
    nested = {}
    for item in fragments:
        raw = base64.b64decode(item["content_base64"], validate=True)
        fragment = json.loads(raw.decode("ascii"))
        if canonical(fragment) + b"\n" != raw:
            fail("E_FRAGMENT_CANONICAL")
        for row in fragment["evidence_artifacts"]:
            nested[row["path"]] = {
                "bytes": row["bytes"],
                "sha256": row["sha256"],
            }
    for path, expected in sorted(nested.items()):
        if path in paths:
            continue
        observed = snapshot(path, True)
        if (
            observed["bytes"] != expected["bytes"]
            or observed["sha256"] != expected["sha256"]
        ):
            fail("E_NESTED_ARTIFACT")
        artifacts.append(observed)

pairs = [
    (plan["producer_pid"], plan["producer_start_ticks"]),
]
runtime = (
    result.get("runtime_process")
    if plan["role"] == "cuda_monolithic"
    else result["cuda_evidence"]["runtime_process"]
)
pairs.append((runtime["pid"], runtime["start_ticks"]))
for pid, ticks in pairs:
    if not process_absent(pid, ticks):
        fail("E_PROCESS_LIVE")
nvml = nvml_pids(plan["nvidia_smi_path"], plan["gpu_uuid"])
if any(pid in nvml for pid, unused in pairs):
    fail("E_NVML_PROCESS_LIVE")

adb_server = None
phone_processes = []
if plan["role"] == "joint_phone_cuda":
    binding = plan["joint_bindings"]
    if snapshot(binding["adb"]["path"], False) != binding["adb"]:
        fail("E_ADB_ARTIFACT")
    expected_server = binding["adb_server_process"]
    if expected_server["boot_id"] != plan["boot_id"]:
        fail("E_ADB_SERVER_BOOT")
    adb_server = process_snapshot(expected_server)
    inode = listener_inode(
        expected_server["listen_host"],
        expected_server["listen_port"],
    )
    if not process_owns_socket(expected_server["pid"], inode):
        fail("E_ADB_SERVER_LISTENER")
    adb_server["listener_inode"] = int(inode)
    phone_processes = phone_cleanup(result, binding)

response = {
    "adb_server_process": adb_server,
    "artifacts": artifacts,
    "boot_id": plan["boot_id"],
    "gpu_uuid": plan["gpu_uuid"],
    "nvml_compute_pids": nvml,
    "primary": primary,
    "phone_processes_absent": phone_processes,
    "processes_absent": [
        {"pid": pid, "start_ticks": ticks}
        for pid, ticks in pairs
    ],
    "remote_cleanup_observed_ns": __import__("time").clock_gettime_ns(
        __import__("time").CLOCK_MONOTONIC_RAW
    ),
    "schema": "s39-v25-remote-cuda-fetch-v1",
    "system_swap_used_bytes": swap_used(),
}
print(base64.b64encode(canonical(response) + b"\n").decode("ascii"))
'''.strip()


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


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"E_DUPLICATE_KEY: {key}")
        result[key] = value
    return result


def canonical_compact(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, UnicodeEncodeError) as error:
        raise CaptureError("E_CANONICAL") from error


def canonical_bytes(value: Any) -> bytes:
    return canonical_compact(value) + b"\n"


def parse_json(raw: bytes, field: str) -> Any:
    require(0 < len(raw) <= MAX_REMOTE_OUTPUT, f"E_SIZE: {field}")
    try:
        return json.loads(
            raw.decode("ascii"),
            object_pairs_hook=strict_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                CaptureError(f"E_NUMBER: {field}: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CaptureError(f"E_JSON: {field}") from error


def exact_keys(value: Any, expected: set[str], field: str) -> dict[str, Any]:
    require(type(value) is dict and set(value) == expected, f"E_KEYS: {field}")
    return value


def integer(value: Any, field: str, minimum: int = 0) -> int:
    require(
        type(value) is int and minimum <= value <= MAX_INT,
        f"E_INTEGER: {field}",
    )
    return value


def text(value: Any, field: str, maximum: int = 32768) -> str:
    require(
        type(value) is str
        and 0 < len(value) <= maximum
        and value.isascii()
        and "\x00" not in value
        and "\n" not in value,
        f"E_TEXT: {field}",
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


def absolute_path(value: Any, field: str) -> str:
    value = text(value, field)
    path = Path(value)
    require(path.is_absolute() and ".." not in path.parts, f"E_PATH: {field}")
    return value


def validate_stat(value: Any, field: str) -> dict[str, Any]:
    value = exact_keys(value, STAT_KEYS, field)
    exact(value["build_id"], None, f"{field}.build_id")
    for key in STAT_KEYS - {"build_id"}:
        integer(value[key], f"{field}.{key}")
    require(
        value["inode"] > 0
        and value["size"] > 0
        and stat.S_ISREG(value["mode"]),
        f"E_STAT: {field}",
    )
    return value


def validate_artifact(value: Any, field: str) -> dict[str, Any]:
    value = exact_keys(value, ARTIFACT_KEYS, field)
    size = integer(value["bytes"], f"{field}.bytes", 1)
    absolute_path(value["path"], f"{field}.path")
    digest(value["sha256"], f"{field}.sha256")
    metadata = validate_stat(value["stat"], f"{field}.stat")
    exact(metadata["size"], size, f"{field}.stat.size")
    return value


def validate_argv(value: Any, field: str) -> list[str]:
    require(
        type(value) is list and 0 < len(value) <= 256,
        f"E_ARGV: {field}",
    )
    for index, item in enumerate(value):
        text(item, f"{field}[{index}]", 32768)
    return value


def read_regular(path: Path, maximum: int = MAX_REMOTE_OUTPUT) -> bytes:
    require(path.is_absolute(), f"E_PATH: {path}")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        require(
            stat.S_ISREG(before.st_mode) and 0 < before.st_size <= maximum,
            f"E_FILE: {path}",
        )
        raw = bytearray()
        while block := os.read(descriptor, 1024 * 1024):
            raw.extend(block)
            require(len(raw) <= maximum, f"E_FILE_SIZE: {path}")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
        item.st_mode,
    )
    require(identity(before) == identity(after), f"E_FILE_CHANGED: {path}")
    require(len(raw) == before.st_size, f"E_FILE_CHANGED: {path}")
    return bytes(raw)


def read_canonical(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = read_regular(path, MAX_JSON)
    value = parse_json(raw, str(path))
    require(type(value) is dict, f"E_TYPE: {path}")
    require(canonical_bytes(value) == raw, f"E_CANONICAL: {path}")
    return value, raw


def artifact_from_raw(path: Path, raw: bytes) -> dict[str, Any]:
    metadata = path.stat()
    return {
        "bytes": len(raw),
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "stat": {
            "build_id": None,
            "ctime_ns": metadata.st_ctime_ns,
            "device_id": metadata.st_dev,
            "inode": metadata.st_ino,
            "mode": metadata.st_mode,
            "mtime_ns": metadata.st_mtime_ns,
            "size": metadata.st_size,
        },
    }


def write_new(path: Path, raw: bytes) -> dict[str, Any]:
    require(path.is_absolute() and not path.exists(), f"E_OUTPUT: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o644,
    )
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            require(written > 0, f"E_WRITE: {path}")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    return artifact_from_raw(path, raw)


def validate_plan(value: Any) -> dict[str, Any]:
    value = exact_keys(value, PLAN_KEYS, "plan")
    exact(value["schema"], PLAN_SCHEMA, "plan.schema")
    exact(value["phase"], PHASE, "plan.phase")
    phase_id = text(value["phase_id"], "plan.phase_id", 128)
    phase_prefix = "cp0-r1-v25-a-only-"
    require(
        phase_id.startswith(phase_prefix)
        and all(character.isalnum() or character in ".-_" for character in phase_id),
        "E_PHASE_ID",
    )
    v24_phase_id = text(value["v24_phase_id"], "plan.v24_phase_id", 128)
    v24_prefix = "cp0-r1-v24-a-only-"
    require(
        v24_phase_id.startswith(v24_prefix)
        and v24_phase_id[len(v24_prefix):] == phase_id[len(phase_prefix):],
        "E_V24_PHASE_ID",
    )
    role = text(value["role"], "plan.role", 64)
    require(role in ROLE_CONFIG, "E_ROLE")
    exact(
        integer(value["sequence_index"], "plan.sequence_index", 1),
        ROLE_CONFIG[role]["sequence_index"],
        "plan.sequence_index",
    )
    digest(value["managed_plan_sha256"], "plan.managed_plan_sha256")
    validate_artifact(value["managed_launcher"], "plan.managed_launcher")
    validate_artifact(value["local_python"], "plan.local_python")
    producer = validate_artifact(
        value["frozen_producer"],
        "plan.frozen_producer",
    )
    exact(
        Path(producer["path"]).name,
        ROLE_CONFIG[role]["producer_name"],
        "plan.frozen_producer.name",
    )
    joint = value["joint_bindings"]
    if role == "cuda_monolithic":
        exact(joint, None, "plan.joint_bindings")
    else:
        joint = exact_keys(joint, JOINT_BINDING_KEYS, "plan.joint_bindings")
        for key in (
            "adb",
            "capture_plan",
            "cuda_launch_plan",
            "phone_launch_plan",
        ):
            validate_artifact(joint[key], f"plan.joint_bindings.{key}")
        port = integer(
            joint["adb_server_port"],
            "plan.joint_bindings.adb_server_port",
            1,
        )
        exact(port, 5038, "plan.joint_bindings.adb_server_port")
        server = exact_keys(
            joint["adb_server_process"],
            ADB_SERVER_KEYS,
            "plan.joint_bindings.adb_server_process",
        )
        validate_argv(
            server["argv"],
            "plan.joint_bindings.adb_server_process.argv",
        )
        text(
            server["boot_id"],
            "plan.joint_bindings.adb_server_process.boot_id",
            128,
        )
        exact(
            server["executable_path"],
            joint["adb"]["path"],
            "plan.joint_bindings.adb_server_process.executable",
        )
        exact(
            server["listen_host"],
            "127.0.0.1",
            "plan.joint_bindings.adb_server_process.listen_host",
        )
        exact(
            integer(
                server["listen_port"],
                "plan.joint_bindings.adb_server_process.listen_port",
                1,
            ),
            port,
            "plan.joint_bindings.adb_server_process.listen_port",
        )
        integer(
            server["pid"],
            "plan.joint_bindings.adb_server_process.pid",
            1,
        )
        integer(
            server["start_ticks"],
            "plan.joint_bindings.adb_server_process.start_ticks",
            1,
        )
        for endpoint in ("op12", "op15"):
            selector = text(
                joint[f"{endpoint}_selector"],
                f"plan.joint_bindings.{endpoint}_selector",
                255,
            )
            require(":" in selector, f"E_ADB_SELECTOR: {endpoint}")
        require(
            joint["op12_selector"] != joint["op15_selector"],
            "E_ADB_SELECTOR_REUSE",
        )
    absolute_path(value["remote_output_path"], "plan.remote_output_path")
    validate_argv(value["producer_argv"], "plan.producer_argv")
    timeout = integer(value["timeout_seconds"], "plan.timeout_seconds", 1)
    require(timeout <= 7200, "E_TIMEOUT")
    return value


def parse_plan(path: Path, expected_sha256: str) -> tuple[dict[str, Any], bytes]:
    digest(expected_sha256, "plan_sha256")
    value, raw = read_canonical(path)
    exact(hashlib.sha256(raw).hexdigest(), expected_sha256, "plan.sha256")
    return validate_plan(value), raw


def load_managed_launcher(
    artifact: dict[str, Any],
) -> types.ModuleType:
    path = Path(artifact["path"])
    raw = read_regular(path, MAX_JSON)
    exact(len(raw), artifact["bytes"], "managed_launcher.bytes")
    exact(
        hashlib.sha256(raw).hexdigest(),
        artifact["sha256"],
        "managed_launcher.sha256",
    )
    module = types.ModuleType("s39_v25_managed_runtime_launcher")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def verify_local_artifact(value: dict[str, Any], field: str) -> None:
    path = Path(value["path"])
    raw = read_regular(path, 512 * 1024 * 1024)
    exact(len(raw), value["bytes"], f"{field}.bytes")
    exact(hashlib.sha256(raw).hexdigest(), value["sha256"], f"{field}.sha256")
    metadata = path.stat()
    observed = {
        "build_id": None,
        "ctime_ns": metadata.st_ctime_ns,
        "device_id": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": metadata.st_mode,
        "mtime_ns": metadata.st_mtime_ns,
        "size": metadata.st_size,
    }
    exact(observed, value["stat"], f"{field}.stat")


def _component_artifact(value: Any, field: str) -> dict[str, Any]:
    require(type(value) is dict, f"E_TYPE: {field}")
    keys = set(value)
    require(
        keys in (
            {"bytes", "path", "sha256", "stat"},
            {"bytes", "component_id", "path", "sha256", "stat"},
        ),
        f"E_KEYS: {field}",
    )
    if "component_id" in value:
        text(value["component_id"], f"{field}.component_id", 128)
    return {
        key: value[key]
        for key in ("bytes", "path", "sha256", "stat")
    }


def validate_managed_plan(
    launcher: Any,
    raw: bytes,
    expected_sha256: str,
    wrapper_plan: dict[str, Any],
) -> dict[str, Any]:
    require(not raw.endswith(b"\n"), "E_MANAGED_PLAN_FRAMING")
    plan = launcher.parse_plan_json(raw.decode("ascii"), expected_sha256)
    exact(plan["mode"], "remote_cuda", "managed.mode")
    exact(plan["endpoint"], "cuda", "managed.endpoint")
    exact(plan["route"]["kind"], "remote_exec", "managed.route.kind")
    argv = plan["_normalized"]["argv"]
    exact(argv, wrapper_plan["producer_argv"], "managed.argv")
    require(len(argv) >= 4, "E_MANAGED_ARGV")
    exact(argv[0], "/usr/bin/python3.14", "managed.argv.python")
    exact(
        plan["ssh"]["remote_python_path"],
        "/usr/bin/python3.14",
        "managed.ssh.remote_python",
    )
    exact(argv[1], "-I", "managed.argv.isolated")
    exact(argv[2], wrapper_plan["frozen_producer"]["path"],
          "managed.argv.producer")
    require(argv.count("--output") == 1, "E_MANAGED_OUTPUT_ARG")
    output_index = argv.index("--output")
    require(output_index + 1 < len(argv), "E_MANAGED_OUTPUT_ARG")
    exact(
        argv[output_index + 1],
        wrapper_plan["remote_output_path"],
        "managed.argv.output",
    )
    require(argv.count("--phase-id") == 1, "E_MANAGED_PHASE_ARG")
    phase_index = argv.index("--phase-id")
    require(phase_index + 1 < len(argv), "E_MANAGED_PHASE_ARG")
    exact(
        argv[phase_index + 1],
        wrapper_plan["v24_phase_id"],
        "managed.argv.phase",
    )
    require("--execute" in argv and "--confirm" in argv, "E_MANAGED_CONFIRM")
    components = {}
    for index, item in enumerate(plan["components"]):
        artifact = _component_artifact(item, f"managed.components[{index}]")
        require(artifact["path"] not in components, "E_COMPONENT_PATH_REUSE")
        components[artifact["path"]] = artifact
    require(
        wrapper_plan["frozen_producer"]["path"] in components,
        "E_PRODUCER_COMPONENT",
    )
    exact(
        components[wrapper_plan["frozen_producer"]["path"]],
        wrapper_plan["frozen_producer"],
        "managed.producer",
    )
    if wrapper_plan["role"] == "joint_phone_cuda":
        joint = wrapper_plan["joint_bindings"]
        for key in (
            "adb",
            "capture_plan",
            "cuda_launch_plan",
            "phone_launch_plan",
        ):
            expected = joint[key]
            require(expected["path"] in components, f"E_JOINT_COMPONENT: {key}")
            exact(components[expected["path"]], expected, f"managed.joint.{key}")
        require(argv.count("--capture-plan") == 1, "E_CAPTURE_PLAN_ARG")
        capture_index = argv.index("--capture-plan")
        exact(
            argv[capture_index + 1],
            joint["capture_plan"]["path"],
            "managed.capture_plan.path",
        )
        require(
            argv.count("--capture-plan-sha256") == 1,
            "E_CAPTURE_PLAN_DIGEST_ARG",
        )
        digest_index = argv.index("--capture-plan-sha256")
        exact(
            argv[digest_index + 1],
            joint["capture_plan"]["sha256"],
            "managed.capture_plan.sha256",
        )
    return plan


def _parse_prefixed(
    raw: bytes,
    prefix: bytes,
    field: str,
) -> list[dict[str, Any]]:
    rows = []
    for line in raw.splitlines(keepends=True):
        if line.startswith(prefix):
            payload = line[len(prefix):]
            value = parse_json(payload, field)
            require(canonical_bytes(value) == payload, f"E_CANONICAL: {field}")
            require(type(value) is dict, f"E_TYPE: {field}")
            rows.append(value)
    return rows


def parse_launcher_output(
    raw: bytes,
    managed_plan: dict[str, Any],
    plan_sha256: str,
    remote_boot_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    transport_rows = _parse_prefixed(raw, b"TRANSPORTPROCESS ", "transport")
    runtime_rows = _parse_prefixed(raw, b"RUNTIMEPROCESS ", "runtime")
    cleanup_rows = _parse_prefixed(raw, MANAGED_CLEANUP_PREFIX, "cleanup")
    exact(len(transport_rows), 1, "transport.count")
    exact(len(runtime_rows), 1, "runtime.count")
    exact(len(cleanup_rows), 1, "cleanup.count")
    permitted = (
        b"TRANSPORTPROCESS ",
        b"RUNTIMEPROCESS ",
        MANAGED_CLEANUP_PREFIX,
    )
    for line in raw.splitlines(keepends=True):
        require(line.startswith(permitted), "E_LAUNCHER_STDOUT")

    transport = exact_keys(transport_rows[0], TRANSPORT_KEYS, "transport")
    exact(transport["schema"], TRANSPORT_SCHEMA, "transport.schema")
    exact(transport["endpoint"], "cuda", "transport.endpoint")
    exact(transport["bundle_id"], managed_plan["bundle_id"], "transport.bundle")
    exact(transport["plan_sha256"], plan_sha256, "transport.plan")
    exact(transport["remote_boot_id"], remote_boot_id, "transport.remote_boot")
    for key in (
        "managed_launcher_pid",
        "managed_launcher_start_ticks",
        "observed_ns",
        "pid",
        "start_ticks",
    ):
        integer(transport[key], f"transport.{key}", 1)
    require(type(transport["argv"]) is list and transport["argv"],
            "E_TRANSPORT_ARGV")

    runtime = exact_keys(runtime_rows[0], REMOTE_RUNTIME_KEYS, "runtime")
    exact(runtime["schema"], RUNTIME_SCHEMA, "runtime.schema")
    exact(runtime["endpoint"], "cuda", "runtime.endpoint")
    exact(runtime["bundle_id"], managed_plan["bundle_id"], "runtime.bundle")
    exact(runtime["boot_id"], remote_boot_id, "runtime.boot")
    exact(runtime["controller_clock"], "CONTROLLER_MONOTONIC", "runtime.clock")
    exact(
        runtime["remote_clock"],
        "RTX_CLOCK_MONOTONIC_RAW",
        "runtime.remote_clock",
    )
    for key in (
        "controller_observed_ns",
        "remote_observed_ns",
        "pid",
        "pgid",
        "start_ticks",
    ):
        integer(runtime[key], f"runtime.{key}", 1)
    exact(runtime["pgid"], runtime["pid"], "runtime.pgid")
    require(runtime["pid"] != transport["pid"], "E_PID_SUBSTITUTION")
    token = text(runtime["launch_token"], "runtime.launch_token", 32)
    require(
        len(token) == 32
        and all(character in "0123456789abcdef" for character in token),
        "E_RUNTIME_TOKEN",
    )
    exact(
        runtime["launcher_path"],
        managed_plan["_normalized"]["launcher_path"],
        "runtime.launcher",
    )
    exact(
        runtime["loaded_repo_component_ids"],
        sorted(managed_plan["_normalized"]["component_map"]),
        "runtime.components",
    )
    expected_dependencies = sorted(
        [
            {
                **component["stat"],
                "path": component["path"],
                "sha256": component["sha256"],
            }
            for component in managed_plan["components"]
        ],
        key=lambda item: item["path"],
    )
    exact(
        runtime["system_dependencies"],
        expected_dependencies,
        "runtime.dependencies",
    )
    cleanup = exact_keys(
        cleanup_rows[0],
        MANAGED_CLEANUP_KEYS,
        "managed_cleanup",
    )
    exact(cleanup["schema"], MANAGED_CLEANUP_SCHEMA, "managed_cleanup.schema")
    exact(cleanup["boot_id"], remote_boot_id, "managed_cleanup.boot")
    exact(
        cleanup["gpu_uuid"],
        managed_plan["ssh"]["gpu_uuid"],
        "managed_cleanup.gpu",
    )
    exact(cleanup["launch_token"], token, "managed_cleanup.token")
    for key in ("pid", "pgid", "start_ticks"):
        exact(cleanup[key], runtime[key], f"managed_cleanup.{key}")
    exact(cleanup["clock"], "RTX_CLOCK_MONOTONIC_RAW", "managed_cleanup.clock")
    cleanup_observed = integer(
        cleanup["observed_ns"],
        "managed_cleanup.observed_ns",
        runtime["remote_observed_ns"],
    )
    del cleanup_observed
    for key in (
        "matching_nvml_pids",
        "matching_process_groups",
        "matching_processes",
    ):
        exact(cleanup[key], [], f"managed_cleanup.{key}")
    absent = cleanup["absent"]
    require(type(absent) is list, "E_MANAGED_CLEANUP_ABSENT")
    exact(
        absent,
        sorted(
            absent,
            key=lambda item: (item.get("pid", 0), item.get("start_ticks", 0)),
        ),
        "managed_cleanup.absent.order",
    )
    pairs = set()
    for index, item in enumerate(absent):
        item = exact_keys(
            item,
            {"pid", "start_ticks"},
            f"managed_cleanup.absent[{index}]",
        )
        pair = (
            integer(item["pid"], f"managed_cleanup.absent[{index}].pid", 1),
            integer(
                item["start_ticks"],
                f"managed_cleanup.absent[{index}].start_ticks",
                1,
            ),
        )
        require(pair not in pairs, "E_MANAGED_CLEANUP_DUPLICATE")
        pairs.add(pair)
    require(
        (runtime["pid"], runtime["start_ticks"]) in pairs,
        "E_MANAGED_CLEANUP_PRODUCER",
    )
    return transport, runtime, cleanup


def process_start_ticks(pid: int) -> int:
    raw = Path(f"/proc/{pid}/stat").read_bytes()
    closing = raw.rfind(b")")
    require(
        closing > 0 and raw[closing + 1:closing + 2] == b" ",
        "E_PROCESS_STAT",
    )
    fields = raw[closing + 2:].split()
    require(int(raw[:raw.find(b" ")]) == pid, "E_PROCESS_STAT")
    return int(fields[19])


def process_absent(pid: int, start_ticks: int) -> bool:
    try:
        observed = process_start_ticks(pid)
    except FileNotFoundError:
        return True
    require(observed == start_ticks, "E_PROCESS_REUSED")
    return False


def listener_absent(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(0.2)
        return connection.connect_ex((host, port)) != 0


def terminate_process(process: subprocess.Popen, timeout_seconds: float) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_managed(
    argv: list[str],
    timeout_seconds: int,
) -> tuple[bytes, int, int]:
    process = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        env={
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    pid = process.pid
    ticks = process_start_ticks(pid)
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        terminate_process(process, 30)
        require(process_absent(pid, ticks), "E_TIMEOUT_CLEANUP")
        raise CaptureError("E_MANAGED_TIMEOUT") from error
    require(process.returncode == 0, f"E_MANAGED_EXIT: {process.returncode}")
    require(stderr == b"", "E_MANAGED_STDERR")
    require(process_absent(pid, ticks), "E_MANAGED_CLEANUP")
    return stdout, pid, ticks


def build_fetch_argv(
    launcher: Any,
    managed_plan: dict[str, Any],
    runtime: dict[str, Any],
    wrapper_plan: dict[str, Any],
) -> list[str]:
    ssh = managed_plan["ssh"]
    interpreter = {
        "bytes": ssh["remote_python_stat"]["size"],
        "path": ssh["remote_python_path"],
        "sha256": ssh["remote_python_sha256"],
        "stat": ssh["remote_python_stat"],
    }
    payload = {
        "boot_id": runtime["boot_id"],
        "gpu_uuid": ssh["gpu_uuid"],
        "interpreter": interpreter,
        "joint_bindings": wrapper_plan["joint_bindings"],
        "nvidia_smi_path": ssh["nvidia_smi_path"],
        "output_path": wrapper_plan["remote_output_path"],
        "producer_pid": runtime["pid"],
        "producer_start_ticks": runtime["start_ticks"],
        "result_schema": ROLE_CONFIG[wrapper_plan["role"]]["result_schema"],
        "role": wrapper_plan["role"],
    }
    encoded = base64.b64encode(canonical_compact(payload)).decode("ascii")
    import shlex
    command = shlex.join([
        ssh["remote_python_path"],
        "-I",
        "-c",
        REMOTE_FETCH_SOURCE,
        encoded,
    ])
    return launcher.ssh_prefix(ssh) + [command]


def run_fetch(
    argv: list[str],
    timeout_seconds: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    process = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        env={
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
        },
    )
    pid = process.pid
    ticks = process_start_ticks(pid)
    started_ns = time.monotonic_ns()
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        terminate_process(process, 30)
        require(process_absent(pid, ticks), "E_FETCH_TIMEOUT_CLEANUP")
        raise CaptureError("E_FETCH_TIMEOUT") from error
    completed_ns = time.monotonic_ns()
    require(process.returncode == 0, f"E_FETCH_EXIT: {process.returncode}")
    require(stderr == b"", "E_FETCH_STDERR")
    require(process_absent(pid, ticks), "E_FETCH_CLEANUP")
    try:
        decoded = base64.b64decode(stdout.strip(), validate=True)
    except ValueError as error:
        raise CaptureError("E_FETCH_BASE64") from error
    value = parse_json(decoded, "fetch")
    require(canonical_bytes(value) == decoded, "E_FETCH_CANONICAL")
    transport = {
        "argv": argv,
        "completed_ns": completed_ns,
        "pid": pid,
        "schema": FETCH_TRANSPORT_SCHEMA,
        "start_ticks": ticks,
        "started_ns": started_ns,
    }
    return value, transport


def validate_fetch(
    value: dict[str, Any],
    wrapper_plan: dict[str, Any],
    runtime: dict[str, Any],
    managed_plan: dict[str, Any],
) -> tuple[bytes, list[tuple[str, bytes]], dict[str, Any]]:
    value = exact_keys(
        value,
        {
            "adb_server_process",
            "artifacts",
            "boot_id",
            "gpu_uuid",
            "nvml_compute_pids",
            "primary",
            "phone_processes_absent",
            "processes_absent",
            "remote_cleanup_observed_ns",
            "schema",
            "system_swap_used_bytes",
        },
        "fetch",
    )
    exact(value["schema"], FETCH_SCHEMA, "fetch.schema")
    exact(value["boot_id"], runtime["boot_id"], "fetch.boot")
    exact(value["gpu_uuid"], managed_plan["ssh"]["gpu_uuid"], "fetch.gpu")
    integer(
        value["remote_cleanup_observed_ns"],
        "fetch.remote_cleanup_observed_ns",
        1,
    )
    integer(value["system_swap_used_bytes"], "fetch.swap")
    require(
        type(value["nvml_compute_pids"]) is list
        and all(type(item) is int and item > 0 for item in value["nvml_compute_pids"]),
        "E_FETCH_NVML",
    )
    require(
        value["nvml_compute_pids"] == [],
        "E_FETCH_NVML_NOT_EMPTY",
    )
    absent = value["processes_absent"]
    require(type(absent) is list and bool(absent), "E_FETCH_ABSENT")
    require(
        {"pid": runtime["pid"], "start_ticks": runtime["start_ticks"]} in absent,
        "E_FETCH_PRODUCER_ABSENT",
    )
    primary = dict(value["primary"])
    encoded = text(primary.pop("content_base64"), "fetch.primary.content",
                   MAX_REMOTE_OUTPUT * 2)
    validate_artifact(primary, "fetch.primary")
    exact(primary["path"], wrapper_plan["remote_output_path"],
          "fetch.primary.path")
    raw = base64.b64decode(encoded, validate=True)
    exact(len(raw), primary["bytes"], "fetch.primary.bytes")
    exact(hashlib.sha256(raw).hexdigest(), primary["sha256"],
          "fetch.primary.sha256")
    result = parse_json(raw, "remote_result")
    exact(canonical_bytes(result), raw, "remote_result.canonical")
    exact(
        result["schema"],
        ROLE_CONFIG[wrapper_plan["role"]]["result_schema"],
        "remote_result.schema",
    )
    exact(
        result["phase_id"],
        wrapper_plan["v24_phase_id"],
        "remote_result.phase_id",
    )
    remote_started = integer(result["started_ns"], "remote_result.started_ns", 1)
    remote_completed = integer(
        result["completed_ns"],
        "remote_result.completed_ns",
        1,
    )
    require(
        remote_started
        < remote_completed
        <= value["remote_cleanup_observed_ns"],
        "E_REMOTE_INTERVAL",
    )
    producer_digest = (
        result["producer_sha256"]
        if wrapper_plan["role"] == "cuda_monolithic"
        else result["joint_producer_sha256"]
    )
    exact(
        producer_digest,
        wrapper_plan["frozen_producer"]["sha256"],
        "remote_result.producer",
    )
    if wrapper_plan["role"] == "joint_phone_cuda":
        joint = wrapper_plan["joint_bindings"]
        exact(
            result["capture_plan_sha256"],
            joint["capture_plan"]["sha256"],
            "remote_result.capture_plan",
        )
        for name in ("cuda", "phone"):
            exact(
                result["subproducer_bindings"][name]["launch_plan_sha256"],
                joint[f"{name}_launch_plan"]["sha256"],
                f"remote_result.{name}_launch_plan",
            )
    cuda_runtime = exact_keys(
        (
            result["runtime_process"]
            if wrapper_plan["role"] == "cuda_monolithic"
            else result["cuda_evidence"]["runtime_process"]
        ),
        RUNTIME_KEYS,
        "remote_result.runtime_process",
    )
    exact(
        cuda_runtime["boot_id"],
        runtime["boot_id"],
        "remote_result.runtime_process.boot",
    )
    require(
        cuda_runtime["pid"] != runtime["pid"],
        "E_REMOTE_PROCESS_ALIAS",
    )
    cuda_pair = {
        "pid": integer(
            cuda_runtime["pid"],
            "remote_result.runtime_process.pid",
            1,
        ),
        "start_ticks": integer(
            cuda_runtime["start_ticks"],
            "remote_result.runtime_process.start_ticks",
            1,
        ),
    }
    require(cuda_pair in absent, "E_FETCH_CUDA_RUNTIME_ABSENT")
    if wrapper_plan["role"] == "cuda_monolithic":
        exact(value["adb_server_process"], None, "fetch.adb_server_process")
        exact(value["phone_processes_absent"], [], "fetch.phone_processes")
        receipt = result["producer_process_receipt"]
        exact(receipt["pid"], runtime["pid"], "remote_result.producer.pid")
        exact(
            receipt["start_ticks"],
            runtime["start_ticks"],
            "remote_result.producer.start_ticks",
        )
        exact(receipt["boot_id"], runtime["boot_id"], "remote_result.producer.boot")
    else:
        server = exact_keys(
            value["adb_server_process"],
            ADB_SERVER_KEYS | {"listener_inode", "observed_ns"},
            "fetch.adb_server_process",
        )
        exact(
            server["boot_id"],
            runtime["boot_id"],
            "fetch.adb_server_process.boot",
        )
        expected_server = wrapper_plan["joint_bindings"]["adb_server_process"]
        for key in ADB_SERVER_KEYS:
            exact(
                server[key],
                expected_server[key],
                f"fetch.adb_server_process.{key}",
            )
        integer(server["listener_inode"], "fetch.adb_server_process.listener", 1)
        observed = integer(
            server["observed_ns"],
            "fetch.adb_server_process.observed_ns",
            1,
        )
        require(
            remote_completed <= observed <= value["remote_cleanup_observed_ns"],
            "E_ADB_SERVER_INTERVAL",
        )
        phone_rows = value["phone_processes_absent"]
        require(
            type(phone_rows) is list and len(phone_rows) == 3,
            "E_PHONE_CLEANUP_COUNT",
        )
        for index, row in enumerate(phone_rows):
            row = exact_keys(
                row,
                {"endpoint", "pid", "start_ticks", "state"},
                f"fetch.phone_processes[{index}]",
            )
            require(row["endpoint"] in {"op12", "op15"}, "E_PHONE_ENDPOINT")
            integer(row["pid"], f"fetch.phone_processes[{index}].pid", 1)
            integer(
                row["start_ticks"],
                f"fetch.phone_processes[{index}].start_ticks",
                1,
            )
            require(row["state"] in {"absent", "reused"}, "E_PHONE_STATE")
        expected_phone = {
            (
                row["endpoint"],
                row["pid"],
                row["start_ticks"],
            )
            for row in result["runtime_processes"]
            if row["endpoint"] in {"op12", "op15"}
        }
        observed_phone = {
            (row["endpoint"], row["pid"], row["start_ticks"])
            for row in phone_rows
        }
        exact(observed_phone, expected_phone, "fetch.phone_processes")

    attachments = []
    for index, artifact in enumerate(value["artifacts"]):
        artifact = dict(artifact)
        content = text(
            artifact.pop("content_base64"),
            f"fetch.artifacts[{index}].content",
            MAX_REMOTE_OUTPUT * 2,
        )
        validate_artifact(artifact, f"fetch.artifacts[{index}]")
        attachment_raw = base64.b64decode(content, validate=True)
        exact(len(attachment_raw), artifact["bytes"],
              f"fetch.artifacts[{index}].bytes")
        exact(hashlib.sha256(attachment_raw).hexdigest(), artifact["sha256"],
              f"fetch.artifacts[{index}].sha256")
        attachments.append((artifact["path"], attachment_raw))
    return raw, attachments, primary


def local_forward_absent(managed_plan: dict[str, Any]) -> bool:
    forward = managed_plan["route"]["local_forward"]
    return listener_absent(forward["local_host"], forward["local_port"])


def capture(
    wrapper_plan: dict[str, Any],
    wrapper_plan_sha256: str,
    wrapper_plan_raw: bytes,
    managed_plan: dict[str, Any],
    managed_plan_raw: bytes,
    launcher: Any,
    output: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    require(output.is_absolute() and receipt_path.is_absolute(), "E_OUTPUT_PATH")
    require(not output.exists() and not receipt_path.exists(), "E_OUTPUT_EXISTS")
    started_ns = time.monotonic_ns()
    launcher_argv = [
        wrapper_plan["local_python"]["path"],
        "-I",
        wrapper_plan["managed_launcher"]["path"],
        "--plan-json",
        managed_plan_raw.decode("ascii"),
        "--plan-sha256",
        wrapper_plan["managed_plan_sha256"],
        "--boot-id",
        managed_plan["ssh"]["_expected_boot_id"],
    ]
    stdout, unused_pid, unused_ticks = run_managed(
        launcher_argv,
        wrapper_plan["timeout_seconds"],
    )
    del unused_pid, unused_ticks
    transport, runtime, managed_cleanup = parse_launcher_output(
        stdout,
        managed_plan,
        wrapper_plan["managed_plan_sha256"],
        managed_plan["ssh"]["_expected_boot_id"],
    )
    require(process_absent(transport["pid"], transport["start_ticks"]),
            "E_EXECUTION_TRANSPORT_LIVE")
    require(local_forward_absent(managed_plan), "E_FORWARD_LIVE")

    fetch_argv = build_fetch_argv(
        launcher,
        managed_plan,
        runtime,
        wrapper_plan,
    )
    fetch, fetch_transport = run_fetch(
        fetch_argv,
        wrapper_plan["timeout_seconds"],
    )
    raw, attachments, remote_artifact = validate_fetch(
        fetch,
        wrapper_plan,
        runtime,
        managed_plan,
    )
    remote_result = parse_json(raw, "remote_result")
    local_result = write_new(output, raw)
    local_attachments = []
    attachment_root = output.with_suffix(output.suffix + ".artifacts")
    for index, (remote_path, attachment_raw) in enumerate(attachments):
        local_path = attachment_root / f"{index:03d}-{Path(remote_path).name}"
        artifact = write_new(local_path.resolve(), attachment_raw)
        local_attachments.append({
            "local": artifact,
            "remote_path": remote_path,
        })
    completed_ns = time.monotonic_ns()
    cleanup = {
        "execution_transport_absent": process_absent(
            transport["pid"],
            transport["start_ticks"],
        ),
        "fetch_transport_absent": process_absent(
            fetch_transport["pid"],
            fetch_transport["start_ticks"],
        ),
        "local_forward_listener_absent": local_forward_absent(managed_plan),
        "observed_ns": completed_ns,
        "remote_cuda_processes_absent": fetch["processes_absent"],
        "remote_cleanup_observed_ns": fetch["remote_cleanup_observed_ns"],
    }
    require(all(
        cleanup[key]
        for key in (
            "execution_transport_absent",
            "fetch_transport_absent",
            "local_forward_listener_absent",
        )
    ), "E_CLEANUP")
    result = {
        "adb_server_process": fetch["adb_server_process"],
        "cleanup": cleanup,
        "completed_ns": completed_ns,
        "execution_transport_process": transport,
        "fetch_transport_process": fetch_transport,
        "frozen_producer": wrapper_plan["frozen_producer"],
        "gpu_uuid": fetch["gpu_uuid"],
        "joint_bindings": wrapper_plan["joint_bindings"],
        "local_python": wrapper_plan["local_python"],
        "local_evidence_artifacts": local_attachments,
        "local_result_artifact": local_result,
        "managed_remote_cleanup": managed_cleanup,
        "managed_plan_sha256": wrapper_plan["managed_plan_sha256"],
        "phase": PHASE,
        "phase_id": wrapper_plan["phase_id"],
        "phone_processes_absent": fetch["phone_processes_absent"],
        "remote_boot_id": fetch["boot_id"],
        "remote_evidence_artifacts": [
            {
                key: value
                for key, value in artifact.items()
                if key != "content_base64"
            }
            for artifact in fetch["artifacts"]
        ],
        "remote_execution_interval": {
            "clock": "RTX_CLOCK_MONOTONIC_RAW",
            "completed_ns": remote_result["completed_ns"],
            "started_ns": remote_result["started_ns"],
        },
        "remote_producer_process": runtime,
        "remote_result_artifact": remote_artifact,
        "role": wrapper_plan["role"],
        "schema": RECEIPT_SCHEMA,
        "sequence_index": wrapper_plan["sequence_index"],
        "started_ns": started_ns,
        "system_swap_used_bytes": fetch["system_swap_used_bytes"],
        "v24_phase_id": wrapper_plan["v24_phase_id"],
        "wrapper_plan_sha256": wrapper_plan_sha256,
    }
    validate_receipt(
        result,
        wrapper_plan,
        wrapper_plan_raw,
        managed_plan,
        managed_plan_raw,
    )
    write_new(receipt_path, canonical_bytes(result))
    return result


def _managed_public_plan(value: dict[str, Any]) -> dict[str, Any]:
    result = {
        key: item
        for key, item in value.items()
        if key != "_normalized"
    }
    if type(result.get("ssh")) is dict and "_expected_boot_id" in result["ssh"]:
        result["ssh"] = {
            key: item
            for key, item in result["ssh"].items()
            if key != "_expected_boot_id"
        }
    return result


def _bind_plan_contents(
    wrapper_plan: dict[str, Any],
    wrapper_plan_raw: bytes,
    managed_plan: dict[str, Any],
    managed_plan_raw: bytes,
) -> tuple[str, str]:
    require(
        type(wrapper_plan_raw) is bytes
        and 0 < len(wrapper_plan_raw) <= MAX_JSON,
        "E_WRAPPER_PLAN_CONTENT",
    )
    parsed_wrapper = parse_json(wrapper_plan_raw, "wrapper_plan")
    exact(canonical_bytes(parsed_wrapper), wrapper_plan_raw, "wrapper_plan.canonical")
    exact(validate_plan(parsed_wrapper), wrapper_plan, "wrapper_plan.content")
    require(
        type(managed_plan_raw) is bytes
        and 0 < len(managed_plan_raw) <= MAX_JSON,
        "E_MANAGED_PLAN_CONTENT",
    )
    parsed_managed = parse_json(managed_plan_raw, "managed_plan")
    exact(
        canonical_compact(parsed_managed),
        managed_plan_raw,
        "managed_plan.canonical",
    )
    exact(
        parsed_managed,
        _managed_public_plan(managed_plan),
        "managed_plan.content",
    )
    wrapper_sha256 = hashlib.sha256(wrapper_plan_raw).hexdigest()
    managed_sha256 = hashlib.sha256(managed_plan_raw).hexdigest()
    exact(
        wrapper_plan["managed_plan_sha256"],
        managed_sha256,
        "wrapper_plan.managed_plan_sha256",
    )
    return wrapper_sha256, managed_sha256


def reopen_artifact(value: Any, field: str) -> bytes:
    artifact = validate_artifact(value, field)
    try:
        raw = read_regular(Path(artifact["path"]))
    except OSError as error:
        raise CaptureError(f"E_LOCAL_ARTIFACT: {field}") from error
    exact(len(raw), artifact["bytes"], f"{field}.bytes")
    exact(hashlib.sha256(raw).hexdigest(), artifact["sha256"], f"{field}.sha256")
    exact(
        artifact_from_raw(Path(artifact["path"]), raw),
        artifact,
        f"{field}.stat",
    )
    return raw


def _validate_local_result(
    raw: bytes,
    wrapper_plan: dict[str, Any],
    remote_interval: dict[str, Any],
) -> None:
    result = parse_json(raw, "receipt.local_result.content")
    exact(canonical_bytes(result), raw, "receipt.local_result.canonical")
    role = wrapper_plan["role"]
    exact(
        result["schema"],
        ROLE_CONFIG[role]["result_schema"],
        "receipt.local_result.schema",
    )
    exact(
        result["phase_id"],
        wrapper_plan["v24_phase_id"],
        "receipt.local_result.phase_id",
    )
    exact(
        result["started_ns"],
        remote_interval["started_ns"],
        "receipt.local_result.started_ns",
    )
    exact(
        result["completed_ns"],
        remote_interval["completed_ns"],
        "receipt.local_result.completed_ns",
    )
    producer_sha256 = (
        result["producer_sha256"]
        if role == "cuda_monolithic"
        else result["joint_producer_sha256"]
    )
    exact(
        producer_sha256,
        wrapper_plan["frozen_producer"]["sha256"],
        "receipt.local_result.producer_sha256",
    )
    if role == "joint_phone_cuda":
        joint = wrapper_plan["joint_bindings"]
        exact(
            result["capture_plan_sha256"],
            joint["capture_plan"]["sha256"],
            "receipt.local_result.capture_plan_sha256",
        )
        for name in ("cuda", "phone"):
            exact(
                result["subproducer_bindings"][name]["launch_plan_sha256"],
                joint[f"{name}_launch_plan"]["sha256"],
                f"receipt.local_result.{name}_launch_plan_sha256",
            )


def validate_sequence(
    receipts: list[dict[str, Any]],
    bindings: list[tuple[
        dict[str, Any],
        bytes,
        dict[str, Any],
        bytes,
    ]],
) -> None:
    require(len(receipts) == 2 and len(bindings) == 2, "E_SEQUENCE_COUNT")
    roles = ["cuda_monolithic", "joint_phone_cuda"]
    previous_completed = None
    previous_remote_cleanup = None
    phase_id = None
    v24_phase_id = None
    remote_boot_id = None
    gpu_uuid = None
    wrapper_digests = set()
    managed_digests = set()
    for index, (receipt, binding, role) in enumerate(
        zip(receipts, bindings, roles),
        1,
    ):
        require(
            type(binding) in (list, tuple) and len(binding) == 4,
            f"E_SEQUENCE_BINDING: {index}",
        )
        receipt = validate_receipt(receipt, *binding)
        exact(receipt["role"], role, f"sequence[{index}].role")
        exact(receipt["sequence_index"], index, f"sequence[{index}].index")
        if phase_id is None:
            phase_id = receipt["phase_id"]
            v24_phase_id = receipt["v24_phase_id"]
            remote_boot_id = receipt["remote_boot_id"]
            gpu_uuid = receipt["gpu_uuid"]
        exact(receipt["phase_id"], phase_id, f"sequence[{index}].phase_id")
        exact(
            receipt["v24_phase_id"],
            v24_phase_id,
            f"sequence[{index}].v24_phase_id",
        )
        exact(
            receipt["remote_boot_id"],
            remote_boot_id,
            f"sequence[{index}].remote_boot_id",
        )
        exact(receipt["gpu_uuid"], gpu_uuid, f"sequence[{index}].gpu_uuid")
        require(
            receipt["wrapper_plan_sha256"] not in wrapper_digests,
            "E_SEQUENCE_WRAPPER_PLAN_REUSE",
        )
        require(
            receipt["managed_plan_sha256"] not in managed_digests,
            "E_SEQUENCE_MANAGED_PLAN_REUSE",
        )
        wrapper_digests.add(receipt["wrapper_plan_sha256"])
        managed_digests.add(receipt["managed_plan_sha256"])
        started = integer(receipt["started_ns"], f"sequence[{index}].started", 1)
        completed = integer(
            receipt["completed_ns"],
            f"sequence[{index}].completed",
            1,
        )
        require(started < completed, f"E_SEQUENCE_INTERVAL: {index}")
        if previous_completed is not None:
            require(previous_completed <= started, "E_SEQUENCE_OVERLAP")
        remote_started = receipt["remote_execution_interval"]["started_ns"]
        if previous_remote_cleanup is not None:
            require(
                previous_remote_cleanup <= remote_started,
                "E_SEQUENCE_REMOTE_OVERLAP",
            )
        previous_completed = completed
        previous_remote_cleanup = receipt["cleanup"][
            "remote_cleanup_observed_ns"
        ]


def validate_receipt(
    value: Any,
    wrapper_plan: dict[str, Any],
    wrapper_plan_raw: bytes,
    managed_plan: dict[str, Any],
    managed_plan_raw: bytes,
) -> dict[str, Any]:
    wrapper_sha256, managed_sha256 = _bind_plan_contents(
        wrapper_plan,
        wrapper_plan_raw,
        managed_plan,
        managed_plan_raw,
    )
    value = exact_keys(value, RECEIPT_KEYS, "receipt")
    exact(value["schema"], RECEIPT_SCHEMA, "receipt.schema")
    exact(value["phase"], PHASE, "receipt.phase")
    role = text(value["role"], "receipt.role", 64)
    require(role in ROLE_CONFIG, "E_RECEIPT_ROLE")
    exact(role, wrapper_plan["role"], "receipt.plan.role")
    exact(
        integer(value["sequence_index"], "receipt.sequence_index", 1),
        ROLE_CONFIG[role]["sequence_index"],
        "receipt.sequence_index",
    )
    exact(
        value["sequence_index"],
        wrapper_plan["sequence_index"],
        "receipt.plan.sequence_index",
    )
    phase_id = text(value["phase_id"], "receipt.phase_id", 128)
    v24_phase_id = text(value["v24_phase_id"], "receipt.v24_phase_id", 128)
    v25_prefix = "cp0-r1-v25-a-only-"
    v24_prefix = "cp0-r1-v24-a-only-"
    require(
        phase_id.startswith(v25_prefix)
        and v24_phase_id.startswith(v24_prefix)
        and phase_id[len(v25_prefix):] == v24_phase_id[len(v24_prefix):],
        "E_RECEIPT_PHASE_LINK",
    )
    started = integer(value["started_ns"], "receipt.started_ns", 1)
    completed = integer(value["completed_ns"], "receipt.completed_ns", 1)
    require(started < completed, "E_RECEIPT_INTERVAL")
    exact(value["phase_id"], wrapper_plan["phase_id"], "receipt.plan.phase_id")
    exact(
        value["v24_phase_id"],
        wrapper_plan["v24_phase_id"],
        "receipt.plan.v24_phase_id",
    )
    exact(
        digest(value["managed_plan_sha256"], "receipt.managed_plan_sha256"),
        managed_sha256,
        "receipt.managed_plan_sha256",
    )
    exact(
        digest(value["wrapper_plan_sha256"], "receipt.wrapper_plan_sha256"),
        wrapper_sha256,
        "receipt.wrapper_plan_sha256",
    )
    validate_artifact(value["frozen_producer"], "receipt.frozen_producer")
    validate_artifact(value["local_python"], "receipt.local_python")
    exact(
        value["frozen_producer"],
        wrapper_plan["frozen_producer"],
        "receipt.plan.frozen_producer",
    )
    exact(
        value["local_python"],
        wrapper_plan["local_python"],
        "receipt.plan.local_python",
    )
    exact(
        value["joint_bindings"],
        wrapper_plan["joint_bindings"],
        "receipt.plan.joint_bindings",
    )
    if role == "cuda_monolithic":
        exact(value["joint_bindings"], None, "receipt.joint_bindings")
        exact(value["adb_server_process"], None, "receipt.adb_server_process")
        exact(value["phone_processes_absent"], [], "receipt.phone_processes")
    else:
        joint = exact_keys(
            value["joint_bindings"],
            JOINT_BINDING_KEYS,
            "receipt.joint_bindings",
        )
        for key in (
            "adb",
            "capture_plan",
            "cuda_launch_plan",
            "phone_launch_plan",
        ):
            validate_artifact(joint[key], f"receipt.joint_bindings.{key}")
        integer(
            joint["adb_server_port"],
            "receipt.joint_bindings.adb_server_port",
            1,
        )
        text(joint["op12_selector"], "receipt.joint_bindings.op12_selector", 255)
        text(joint["op15_selector"], "receipt.joint_bindings.op15_selector", 255)
        server = exact_keys(
            value["adb_server_process"],
            ADB_SERVER_KEYS | {"listener_inode", "observed_ns"},
            "receipt.adb_server_process",
        )
        exact(
            server["boot_id"],
            value["remote_boot_id"],
            "receipt.adb_server_process.boot",
        )
        expected_server = joint["adb_server_process"]
        for key in ADB_SERVER_KEYS:
            exact(
                server[key],
                expected_server[key],
                f"receipt.adb_server_process.{key}",
            )
        integer(server["listener_inode"], "receipt.adb_server_process.listener", 1)
        integer(server["observed_ns"], "receipt.adb_server_process.observed", 1)
        phone_rows = value["phone_processes_absent"]
        require(
            type(phone_rows) is list and len(phone_rows) == 3,
            "E_RECEIPT_PHONE_CLEANUP",
        )
        for index, row in enumerate(phone_rows):
            row = exact_keys(
                row,
                {"endpoint", "pid", "start_ticks", "state"},
                f"receipt.phone_processes[{index}]",
            )
            require(row["endpoint"] in {"op12", "op15"}, "E_RECEIPT_PHONE")
            integer(row["pid"], f"receipt.phone_processes[{index}].pid", 1)
            integer(
                row["start_ticks"],
                f"receipt.phone_processes[{index}].start_ticks",
                1,
            )
            require(row["state"] in {"absent", "reused"}, "E_RECEIPT_PHONE")
    text(value["gpu_uuid"], "receipt.gpu_uuid", 128)
    text(value["remote_boot_id"], "receipt.remote_boot_id", 128)
    exact(
        value["gpu_uuid"],
        managed_plan["ssh"]["gpu_uuid"],
        "receipt.plan.gpu_uuid",
    )
    integer(value["system_swap_used_bytes"], "receipt.system_swap_used_bytes")

    transport = exact_keys(
        value["execution_transport_process"],
        TRANSPORT_KEYS,
        "receipt.execution_transport",
    )
    exact(transport["schema"], TRANSPORT_SCHEMA, "receipt.execution.schema")
    exact(
        transport["bundle_id"],
        managed_plan["bundle_id"],
        "receipt.execution.bundle_id",
    )
    exact(transport["endpoint"], "cuda", "receipt.execution.endpoint")
    exact(
        transport["plan_sha256"],
        value["managed_plan_sha256"],
        "receipt.execution.plan",
    )
    exact(
        transport["remote_boot_id"],
        value["remote_boot_id"],
        "receipt.execution.boot",
    )
    for key in (
        "managed_launcher_pid",
        "managed_launcher_start_ticks",
        "observed_ns",
        "pid",
        "start_ticks",
    ):
        integer(transport[key], f"receipt.execution.{key}", 1)
    require(
        started <= transport["observed_ns"] <= completed,
        "E_RECEIPT_EXECUTION_INTERVAL",
    )

    runtime = exact_keys(
        value["remote_producer_process"],
        REMOTE_RUNTIME_KEYS,
        "receipt.remote_producer",
    )
    exact(runtime["schema"], RUNTIME_SCHEMA, "receipt.remote_producer.schema")
    exact(
        runtime["controller_clock"],
        "CONTROLLER_MONOTONIC",
        "receipt.remote_producer.controller_clock",
    )
    exact(
        runtime["remote_clock"],
        "RTX_CLOCK_MONOTONIC_RAW",
        "receipt.remote_producer.remote_clock",
    )
    exact(runtime["boot_id"], value["remote_boot_id"], "receipt.remote_producer.boot")
    exact(
        runtime["bundle_id"],
        managed_plan["bundle_id"],
        "receipt.remote_producer.bundle_id",
    )
    exact(runtime["endpoint"], "cuda", "receipt.remote_producer.endpoint")
    for key in (
        "controller_observed_ns",
        "remote_observed_ns",
        "pgid",
        "pid",
        "start_ticks",
    ):
        integer(runtime[key], f"receipt.remote_producer.{key}", 1)
    exact(runtime["pgid"], runtime["pid"], "receipt.remote_producer.pgid")
    require(
        started <= runtime["controller_observed_ns"] <= completed,
        "E_RECEIPT_REMOTE_INTERVAL",
    )
    require(runtime["pid"] != transport["pid"], "E_PID_SUBSTITUTION")
    exact(
        runtime["launcher_path"],
        managed_plan["_normalized"]["launcher_path"],
        "receipt.remote_producer.launcher",
    )
    exact(
        runtime["loaded_repo_component_ids"],
        sorted(managed_plan["_normalized"]["component_map"]),
        "receipt.remote_producer.components",
    )
    expected_dependencies = sorted(
        [
            {
                **component["stat"],
                "path": component["path"],
                "sha256": component["sha256"],
            }
            for component in managed_plan["components"]
        ],
        key=lambda item: item["path"],
    )
    exact(
        runtime["system_dependencies"],
        expected_dependencies,
        "receipt.remote_producer.dependencies",
    )
    managed_cleanup = exact_keys(
        value["managed_remote_cleanup"],
        MANAGED_CLEANUP_KEYS,
        "receipt.managed_remote_cleanup",
    )
    exact(
        managed_cleanup["schema"],
        MANAGED_CLEANUP_SCHEMA,
        "receipt.managed_remote_cleanup.schema",
    )
    exact(
        managed_cleanup["boot_id"],
        value["remote_boot_id"],
        "receipt.managed_remote_cleanup.boot",
    )
    exact(
        managed_cleanup["gpu_uuid"],
        value["gpu_uuid"],
        "receipt.managed_remote_cleanup.gpu",
    )
    exact(
        managed_cleanup["launch_token"],
        runtime["launch_token"],
        "receipt.managed_remote_cleanup.token",
    )
    for key in ("pid", "pgid", "start_ticks"):
        exact(
            managed_cleanup[key],
            runtime[key],
            f"receipt.managed_remote_cleanup.{key}",
        )
    exact(
        managed_cleanup["clock"],
        "RTX_CLOCK_MONOTONIC_RAW",
        "receipt.managed_remote_cleanup.clock",
    )
    managed_cleanup_observed = integer(
        managed_cleanup["observed_ns"],
        "receipt.managed_remote_cleanup.observed_ns",
        runtime["remote_observed_ns"],
    )
    for key in (
        "matching_nvml_pids",
        "matching_process_groups",
        "matching_processes",
    ):
        exact(
            managed_cleanup[key],
            [],
            f"receipt.managed_remote_cleanup.{key}",
        )
    managed_absent = managed_cleanup["absent"]
    require(type(managed_absent) is list, "E_RECEIPT_MANAGED_CLEANUP")
    managed_pairs = set()
    for index, row in enumerate(managed_absent):
        row = exact_keys(
            row,
            {"pid", "start_ticks"},
            f"receipt.managed_remote_cleanup.absent[{index}]",
        )
        pair = (
            integer(
                row["pid"],
                f"receipt.managed_remote_cleanup.absent[{index}].pid",
                1,
            ),
            integer(
                row["start_ticks"],
                (
                    "receipt.managed_remote_cleanup."
                    f"absent[{index}].start_ticks"
                ),
                1,
            ),
        )
        require(pair not in managed_pairs, "E_RECEIPT_MANAGED_CLEANUP_DUPLICATE")
        managed_pairs.add(pair)
    require(
        (runtime["pid"], runtime["start_ticks"]) in managed_pairs,
        "E_RECEIPT_MANAGED_PRODUCER_CLEANUP",
    )

    fetch_transport = exact_keys(
        value["fetch_transport_process"],
        FETCH_TRANSPORT_KEYS,
        "receipt.fetch_transport",
    )
    exact(
        fetch_transport["schema"],
        FETCH_TRANSPORT_SCHEMA,
        "receipt.fetch_transport.schema",
    )
    fetch_started = integer(
        fetch_transport["started_ns"],
        "receipt.fetch_transport.started_ns",
        1,
    )
    fetch_completed = integer(
        fetch_transport["completed_ns"],
        "receipt.fetch_transport.completed_ns",
        1,
    )
    require(
        started <= fetch_started < fetch_completed <= completed,
        "E_RECEIPT_FETCH_INTERVAL",
    )
    integer(fetch_transport["pid"], "receipt.fetch_transport.pid", 1)
    integer(
        fetch_transport["start_ticks"],
        "receipt.fetch_transport.start_ticks",
        1,
    )

    remote_result = validate_artifact(
        value["remote_result_artifact"],
        "receipt.remote_result",
    )
    local_result = validate_artifact(
        value["local_result_artifact"],
        "receipt.local_result",
    )
    exact(local_result["bytes"], remote_result["bytes"], "receipt.result.bytes")
    exact(local_result["sha256"], remote_result["sha256"], "receipt.result.sha256")
    exact(
        remote_result["path"],
        wrapper_plan["remote_output_path"],
        "receipt.result.remote_path",
    )
    local_result_raw = reopen_artifact(
        local_result,
        "receipt.local_result.disk",
    )

    remote_evidence = value["remote_evidence_artifacts"]
    local_evidence = value["local_evidence_artifacts"]
    require(
        type(remote_evidence) is list
        and type(local_evidence) is list
        and len(remote_evidence) == len(local_evidence),
        "E_RECEIPT_EVIDENCE_COUNT",
    )
    for index, (remote, local) in enumerate(
        zip(remote_evidence, local_evidence)
    ):
        remote = validate_artifact(remote, f"receipt.remote_evidence[{index}]")
        local = exact_keys(
            local,
            {"local", "remote_path"},
            f"receipt.local_evidence[{index}]",
        )
        exact(
            local["remote_path"],
            remote["path"],
            f"receipt.local_evidence[{index}].remote_path",
        )
        copied = validate_artifact(
            local["local"],
            f"receipt.local_evidence[{index}].local",
        )
        exact(copied["bytes"], remote["bytes"], f"receipt.evidence[{index}].bytes")
        exact(
            copied["sha256"],
            remote["sha256"],
            f"receipt.evidence[{index}].sha256",
        )
        reopen_artifact(
            copied,
            f"receipt.local_evidence[{index}].disk",
        )

    cleanup = exact_keys(value["cleanup"], CLEANUP_KEYS, "receipt.cleanup")
    for key in (
        "execution_transport_absent",
        "fetch_transport_absent",
        "local_forward_listener_absent",
    ):
        exact(cleanup[key], True, f"receipt.cleanup.{key}")
    exact(
        integer(cleanup["observed_ns"], "receipt.cleanup.observed_ns", 1),
        completed,
        "receipt.cleanup.observed_ns",
    )
    remote_interval = exact_keys(
        value["remote_execution_interval"],
        {"clock", "completed_ns", "started_ns"},
        "receipt.remote_execution_interval",
    )
    exact(
        remote_interval["clock"],
        "RTX_CLOCK_MONOTONIC_RAW",
        "receipt.remote_execution_interval.clock",
    )
    remote_started = integer(
        remote_interval["started_ns"],
        "receipt.remote_execution_interval.started_ns",
        1,
    )
    remote_completed = integer(
        remote_interval["completed_ns"],
        "receipt.remote_execution_interval.completed_ns",
        1,
    )
    remote_cleanup = integer(
        cleanup["remote_cleanup_observed_ns"],
        "receipt.cleanup.remote_cleanup_observed_ns",
        1,
    )
    require(
        remote_started < remote_completed <= remote_cleanup,
        "E_RECEIPT_REMOTE_INTERVAL",
    )
    require(
        runtime["remote_observed_ns"]
        <= remote_started
        < remote_completed
        <= managed_cleanup_observed
        <= remote_cleanup,
        "E_RECEIPT_REMOTE_CLOCK_ORDER",
    )
    _validate_local_result(
        local_result_raw,
        wrapper_plan,
        remote_interval,
    )
    if role == "joint_phone_cuda":
        require(
            remote_completed
            <= value["adb_server_process"]["observed_ns"]
            <= remote_cleanup,
            "E_RECEIPT_ADB_INTERVAL",
        )
    absent = cleanup["remote_cuda_processes_absent"]
    require(
        type(absent) is list and len(absent) >= 2,
        "E_RECEIPT_REMOTE_CLEANUP",
    )
    pairs = set()
    for index, row in enumerate(absent):
        row = exact_keys(
            row,
            {"pid", "start_ticks"},
            f"receipt.cleanup.remote[{index}]",
        )
        pair = (
            integer(row["pid"], f"receipt.cleanup.remote[{index}].pid", 1),
            integer(
                row["start_ticks"],
                f"receipt.cleanup.remote[{index}].start_ticks",
                1,
            ),
        )
        require(pair not in pairs, "E_RECEIPT_REMOTE_CLEANUP_DUPLICATE")
        pairs.add(pair)
    require(
        (runtime["pid"], runtime["start_ticks"]) in pairs,
        "E_RECEIPT_REMOTE_PRODUCER_CLEANUP",
    )
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--managed-plan", required=True)
    parser.add_argument("--managed-plan-sha256", required=True)
    parser.add_argument("--remote-boot-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    args = parser.parse_args(argv)
    require(args.execute and args.confirm == CONFIRMATION, "E_CONFIRM")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        wrapper_plan, wrapper_raw = parse_plan(
            Path(args.plan),
            args.plan_sha256,
        )
        exact(
            args.managed_plan_sha256,
            wrapper_plan["managed_plan_sha256"],
            "managed_plan.sha256",
        )
        managed_raw = read_regular(Path(args.managed_plan), MAX_JSON)
        exact(
            hashlib.sha256(managed_raw).hexdigest(),
            args.managed_plan_sha256,
            "managed_plan.sha256",
        )
        launcher = load_managed_launcher(wrapper_plan["managed_launcher"])
        verify_local_artifact(wrapper_plan["local_python"], "local_python")
        exact(
            os.path.realpath(sys.executable),
            wrapper_plan["local_python"]["path"],
            "local_python.executable",
        )
        managed = validate_managed_plan(
            launcher,
            managed_raw,
            args.managed_plan_sha256,
            wrapper_plan,
        )
        managed["ssh"]["_expected_boot_id"] = text(
            args.remote_boot_id,
            "remote_boot_id",
            128,
        )
        capture(
            wrapper_plan,
            args.plan_sha256,
            wrapper_raw,
            managed,
            managed_raw,
            launcher,
            Path(args.output),
            Path(args.receipt),
        )
        return 0
    except (
        CaptureError,
        OSError,
        subprocess.SubprocessError,
        ValueError,
    ) as error:
        print(f"S39_V25_REMOTE_CUDA_REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
