#!/usr/bin/env python3
"""Execute and materialize the frozen V2.4 fan-in on the pinned RTX host."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import stat
import subprocess
import sys
import time
import types
from typing import Any


CONFIRMATION = "RUN_CP0_R1_V25_REMOTE_FAN_IN"
FETCH_SCHEMA = "s39-v25-remote-fan-in-fetch-v1"
TRANSPORT_PREFIX = b"TRANSPORTPROCESS "
RUNTIME_PREFIX = b"RUNTIMEPROCESS "
CLEANUP_PREFIX = b"REMOTECLEANUP "
MAX_PLAN_BYTES = 64 * 1024 * 1024
MAX_LAUNCHER_OUTPUT = 16 * 1024 * 1024
MAX_BUNDLE_BYTES = 512 * 1024 * 1024
MAX_FETCH_OUTPUT = 768 * 1024 * 1024
MAX_FILES = 4096
common: types.ModuleType | None = None
contract: types.ModuleType | None = None

REMOTE_FETCH_SOURCE = r'''
import base64
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

MAX_FILE_BYTES = 512 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024
MAX_FILES = 4096

def fail(message):
    raise RuntimeError(message)

def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")

def decode():
    raw = base64.b64decode(sys.argv[1], validate=True)
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

def identity(metadata):
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_mode,
    )

def canonical_path(path_text):
    path = Path(path_text)
    if not path.is_absolute() or str(path) != path_text or ".." in path.parts:
        fail("E_PATH")
    return path

def open_dir(path):
    path = canonical_path(str(path))
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for part in path.parts[1:]:
            child = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            fail("E_DIRECTORY")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise

def open_parent(path):
    path = canonical_path(str(path))
    if path == Path("/"):
        fail("E_FILE_PATH")
    return open_dir(path.parent), path.name

def snapshot(path_text, include_content=True):
    path = canonical_path(path_text)
    parent, name = open_parent(path)
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, dir_fd=parent)
        before_link = os.stat(name, dir_fd=parent, follow_symlinks=False)
    finally:
        os.close(parent)
    try:
        before = os.fstat(descriptor)
        if (
            identity(before_link) != identity(before)
            or not stat.S_ISREG(before.st_mode)
            or not 0 < before.st_size <= MAX_FILE_BYTES
        ):
            fail("E_FILE")
        digest = hashlib.sha256()
        raw = bytearray()
        consumed = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            consumed += len(block)
            if include_content:
                raw.extend(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if identity(before) != identity(after) or consumed != before.st_size:
        fail("E_FILE_CHANGED")
    result = {
        "bytes": consumed,
        "path": str(path),
        "sha256": digest.hexdigest(),
        "stat": file_stat(before),
    }
    if include_content:
        result["content_base64"] = base64.b64encode(raw).decode("ascii")
    return result

def verified_fd(expected):
    path = canonical_path(expected["path"])
    parent, name = open_parent(path)
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent,
        )
    finally:
        os.close(parent)
    try:
        before = os.fstat(descriptor)
        digest = hashlib.sha256()
        consumed = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            consumed += len(block)
            digest.update(block)
        after = os.fstat(descriptor)
        observed = {
            "bytes": consumed,
            "path": str(path),
            "sha256": digest.hexdigest(),
            "stat": file_stat(before),
        }
        if identity(before) != identity(after) or observed != expected:
            fail("E_VERIFIED_FILE")
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise

def walk_directory(root_text):
    root = canonical_path(root_text)
    root_fd = open_dir(root)
    rows = []
    directory_count = 0
    total = 0

    def walk(descriptor, relative):
        nonlocal directory_count, total
        directory_count += 1
        if directory_count > MAX_FILES:
            fail("E_DIRECTORY_COUNT")
        before = os.fstat(descriptor)
        names = sorted(os.listdir(descriptor))
        if len(names) != len(set(names)):
            fail("E_DIRECTORY_NAMES")
        if relative and not names:
            fail("E_EMPTY_DIRECTORY")
        for name in names:
            if (
                not name
                or name in {".", ".."}
                or "/" in name
                or "\0" in name
            ):
                fail("E_DIRECTORY_NAME")
            metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            child_relative = name if not relative else relative + "/" + name
            if stat.S_ISLNK(metadata.st_mode):
                fail("E_SYMLINK")
            if stat.S_ISDIR(metadata.st_mode):
                child = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
                try:
                    if identity(os.fstat(child)) != identity(metadata):
                        fail("E_DIRECTORY_CHANGED")
                    walk(child, child_relative)
                finally:
                    os.close(child)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                fail("E_FILE_TYPE")
            artifact = snapshot(str(root / child_relative), True)
            if identity(metadata) != (
                artifact["stat"]["device_id"],
                artifact["stat"]["inode"],
                artifact["stat"]["size"],
                artifact["stat"]["mtime_ns"],
                artifact["stat"]["ctime_ns"],
                artifact["stat"]["mode"],
            ):
                fail("E_FILE_CHANGED")
            total += artifact["bytes"]
            if total > MAX_TOTAL_BYTES or len(rows) >= MAX_FILES:
                fail("E_BUNDLE_SIZE")
            rows.append({
                "artifact": artifact,
                "materialized_path": child_relative,
            })
        after = os.fstat(descriptor)
        if identity(before) != identity(after):
            fail("E_DIRECTORY_CHANGED")

    try:
        walk(root_fd, "")
    finally:
        os.close(root_fd)
    if not rows:
        fail("E_EMPTY_BUNDLE")
    return rows

def start_ticks(pid):
    raw = (Path("/proc") / str(pid) / "stat").read_bytes()
    closing = raw.rfind(b")")
    if closing <= 0 or raw[closing + 1:closing + 2] != b" ":
        fail("E_PROCESS_STAT")
    fields = raw[closing + 2:].split()
    if int(raw[:raw.find(b" ")]) != pid:
        fail("E_PROCESS_STAT")
    return int(fields[19])

def require_process_absent(pid, ticks):
    try:
        observed = start_ticks(pid)
    except FileNotFoundError:
        return
    if observed == ticks:
        fail("E_PROCESS_LIVE")

def checked(executable_fd, arguments):
    completed = subprocess.run(
        ["/proc/self/fd/" + str(executable_fd), *arguments],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        pass_fds=(executable_fd,),
        check=False,
        timeout=15,
    )
    if completed.returncode != 0 or completed.stderr:
        fail("E_NVML")
    return completed.stdout.decode("ascii")

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

plan = decode()
if set(plan) != {
    "boot_id",
    "capture_input_paths",
    "gpu_uuid",
    "nvidia_smi",
    "producer_pid",
    "producer_start_ticks",
    "remote_acquisition_output",
    "remote_bundle_root",
    "remote_python",
    "remote_runtime_output",
}:
    fail("E_PLAN_KEYS")
if (
    os.path.realpath(sys.executable) != plan["remote_python"]["path"]
    or snapshot(plan["remote_python"]["path"], False) != plan["remote_python"]
):
    fail("E_PYTHON")
boot_path = Path("/proc/sys/kernel/random/boot_id")
if boot_path.read_text(encoding="ascii").strip() != plan["boot_id"]:
    fail("E_BOOT")
nvidia_fd = verified_fd(plan["nvidia_smi"])
try:
    observed_gpu = checked(
        nvidia_fd,
        [
            "--id",
            plan["gpu_uuid"],
            "--query-gpu=uuid",
            "--format=csv,noheader,nounits",
        ],
    ).strip()
    if observed_gpu != plan["gpu_uuid"]:
        fail("E_GPU")
    nvml = []
    for line in checked(
        nvidia_fd,
        [
            "--id",
            plan["gpu_uuid"],
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
    ).splitlines():
        value = line.strip()
        if value and value != "[N/A]":
            nvml.append(int(value))
    nvml = sorted(set(nvml))
finally:
    os.close(nvidia_fd)
if nvml:
    fail("E_NVML_NOT_EMPTY")
require_process_absent(plan["producer_pid"], plan["producer_start_ticks"])
captures = {
    role: snapshot(path, True)
    for role, path in sorted(plan["capture_input_paths"].items())
}
files = walk_directory(plan["remote_bundle_root"])
source_paths = {row["artifact"]["path"] for row in files}
if len(source_paths) != len(files):
    fail("E_SOURCE_PATH_REUSE")
v24_manifest = str(
    Path(plan["remote_bundle_root"]) / "EVIDENCE_BUNDLE_V2_4.json"
)
v25_manifest = str(
    Path(plan["remote_bundle_root"]) / "EVIDENCE_BUNDLE_V2_5_RAW.json"
)
manifest_rows = [
    row for row in files if row["artifact"]["path"] == v24_manifest
]
if len(manifest_rows) != 1 or v25_manifest in source_paths:
    fail("E_RAW_MANIFEST")
manifest_rows[0]["materialized_path"] = "EVIDENCE_BUNDLE_V2_5_RAW.json"
for path, relative in (
    (plan["remote_acquisition_output"], "acquisition.json"),
    (plan["remote_runtime_output"], "runtime.json"),
):
    if path in source_paths:
        fail("E_OUTPUT_ALIAS")
    files.append({
        "artifact": snapshot(path, True),
        "materialized_path": relative,
    })
logical_paths = [row["materialized_path"] for row in files]
if len(logical_paths) != len(set(logical_paths)):
    fail("E_LOGICAL_PATH_REUSE")
files.sort(key=lambda row: row["materialized_path"])
by_logical = {row["materialized_path"]: row["artifact"] for row in files}
for role, relative in (
    ("cuda_monolithic", "raw/cuda-monolithic.json"),
    ("joint_phone_cuda", "raw/joint-phone-cuda.json"),
):
    if relative not in by_logical:
        fail("E_CAPTURE_BUNDLE_FILE")
    for key in ("bytes", "sha256"):
        if captures[role][key] != by_logical[relative][key]:
            fail("E_CAPTURE_CONTENT")
if boot_path.read_text(encoding="ascii").strip() != plan["boot_id"]:
    fail("E_BOOT_CHANGED")
system_swap_used_bytes = swap_used()
require_process_absent(plan["producer_pid"], plan["producer_start_ticks"])
nvidia_fd = verified_fd(plan["nvidia_smi"])
try:
    observed_gpu = checked(
        nvidia_fd,
        [
            "--id",
            plan["gpu_uuid"],
            "--query-gpu=uuid",
            "--format=csv,noheader,nounits",
        ],
    ).strip()
    if observed_gpu != plan["gpu_uuid"]:
        fail("E_GPU_CHANGED")
    final_nvml = []
    for line in checked(
        nvidia_fd,
        [
            "--id",
            plan["gpu_uuid"],
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
    ).splitlines():
        value = line.strip()
        if value and value != "[N/A]":
            final_nvml.append(int(value))
    final_nvml = sorted(set(final_nvml))
finally:
    os.close(nvidia_fd)
if final_nvml:
    fail("E_NVML_NOT_EMPTY_FINAL")
observed_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
response = {
    "capture_input_artifacts": captures,
    "files": files,
    "remote_cleanup": {
        "boot_id": plan["boot_id"],
        "clock": "RTX_CLOCK_MONOTONIC_RAW",
        "gpu_uuid": plan["gpu_uuid"],
        "nvml_compute_pids": final_nvml,
        "observed_ns": observed_ns,
        "producer_absent": {
            "pid": plan["producer_pid"],
            "start_ticks": plan["producer_start_ticks"],
        },
        "schema": "s39-v25-remote-fan-in-post-cleanup-v1",
    },
    "schema": "s39-v25-remote-fan-in-fetch-v1",
    "system_swap_used_bytes": system_swap_used_bytes,
}
print(base64.b64encode(canonical(response) + b"\n").decode("ascii"))
'''.strip()


class ExecuteError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExecuteError(message)


def exact(value: Any, expected: Any, field: str) -> None:
    require(
        type(value) is type(expected) and value == expected,
        f"E_VALUE: {field}",
    )


def read_regular(path: Path, maximum: int = MAX_BUNDLE_BYTES) -> bytes:
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


def artifact_from_path(path: Path, raw: bytes) -> dict[str, Any]:
    metadata = os.stat(path, follow_symlinks=False)
    require(stat.S_ISREG(metadata.st_mode), f"E_FILE: {path}")
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


def _bootstrap_json(raw: bytes) -> dict[str, Any]:
    def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value = {}
        for key, item in pairs:
            require(key not in value, f"E_DUPLICATE_KEY: {key}")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise ExecuteError(f"E_JSON_NUMBER: {value}")

    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExecuteError("E_BOOTSTRAP_PLAN_JSON") from error
    require(type(value) is dict, "E_BOOTSTRAP_PLAN_TYPE")
    try:
        canonical = (
            json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, UnicodeEncodeError) as error:
        raise ExecuteError("E_BOOTSTRAP_PLAN_CANONICAL") from error
    exact(canonical, raw, "bootstrap.plan.canonical")
    return value


def _bootstrap_reopen_artifact(value: Any, field: str) -> bytes:
    require(
        type(value) is dict
        and set(value) == {"bytes", "path", "sha256", "stat"},
        f"E_BOOTSTRAP_ARTIFACT: {field}",
    )
    path_text = value["path"]
    require(
        type(path_text) is str
        and path_text.isascii()
        and "\x00" not in path_text
        and "\n" not in path_text,
        f"E_BOOTSTRAP_PATH: {field}",
    )
    path = Path(path_text)
    require(
        path.is_absolute()
        and str(path) == path_text
        and ".." not in path.parts,
        f"E_BOOTSTRAP_PATH: {field}",
    )
    size = value["bytes"]
    digest = value["sha256"]
    require(
        type(size) is int
        and 0 < size <= MAX_PLAN_BYTES
        and type(digest) is str
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest),
        f"E_BOOTSTRAP_CONTENT: {field}",
    )
    raw = read_regular(path, MAX_PLAN_BYTES)
    exact(len(raw), size, f"{field}.bytes")
    exact(hashlib.sha256(raw).hexdigest(), digest, f"{field}.sha256")
    exact(artifact_from_path(path, raw), value, f"{field}.stat")
    return raw


def _source_module(name: str, path: Path, raw: bytes) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def load_pinned_plan(
    path: Path,
    expected_sha256: str,
) -> tuple[
    dict[str, Any],
    bytes,
    types.ModuleType,
    types.ModuleType,
]:
    require(
        type(expected_sha256) is str
        and len(expected_sha256) == 64
        and all(
            character in "0123456789abcdef"
            for character in expected_sha256
        ),
        "E_BOOTSTRAP_PLAN_DIGEST",
    )
    raw = read_regular(path, MAX_PLAN_BYTES)
    exact(
        hashlib.sha256(raw).hexdigest(),
        expected_sha256,
        "bootstrap.plan.sha256",
    )
    value = _bootstrap_json(raw)
    common_value = value.get("local_common")
    validator_value = value.get("contract_validator")
    common_raw = _bootstrap_reopen_artifact(
        common_value,
        "bootstrap.local_common",
    )
    validator_raw = _bootstrap_reopen_artifact(
        validator_value,
        "bootstrap.contract_validator",
    )
    common_path = Path(common_value["path"])
    validator_path = Path(validator_value["path"])
    common_module = _source_module(
        "s39_v25_remote_fan_in_common",
        common_path,
        common_raw,
    )
    validator_module = types.ModuleType(
        "s39_v25_remote_fan_in_contract"
    )
    validator_module.__file__ = str(validator_path)
    validator_module.__package__ = ""
    previous = sys.modules.get("v25_common")
    try:
        sys.modules["v25_common"] = common_module
        exec(
            compile(validator_raw, str(validator_path), "exec"),
            validator_module.__dict__,
        )
    finally:
        if previous is None:
            sys.modules.pop("v25_common", None)
        else:
            sys.modules["v25_common"] = previous
    plan, reopened = validator_module.parse_plan(path, expected_sha256)
    exact(reopened, raw, "bootstrap.plan.reopen")
    return plan, raw, common_module, validator_module


def reopen_artifact(value: Any, field: str) -> bytes:
    require(contract is not None, "E_CONTRACT_NOT_LOADED")
    artifact = contract._artifact(value, field)
    try:
        raw = read_regular(Path(artifact["path"]))
    except OSError as error:
        raise ExecuteError(f"E_LOCAL_ARTIFACT: {field}") from error
    exact(len(raw), artifact["bytes"], f"{field}.bytes")
    exact(hashlib.sha256(raw).hexdigest(), artifact["sha256"], f"{field}.sha256")
    exact(
        artifact_from_path(Path(artifact["path"]), raw),
        artifact,
        f"{field}.stat",
    )
    return raw


def load_managed_launcher(artifact: dict[str, Any]) -> types.ModuleType:
    raw = reopen_artifact(artifact, "managed_launcher")
    path = Path(artifact["path"])
    module = types.ModuleType("s39_v25_fan_in_managed_launcher")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def _component_artifact(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key != "component_id"
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
    exact(
        plan["_normalized"]["environment"],
        {
            "CUDA_VISIBLE_DEVICES": plan["ssh"]["gpu_uuid"],
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        "managed.environment",
    )
    exact(
        plan["_normalized"]["argv"],
        wrapper_plan["producer_argv"],
        "managed.argv",
    )
    exact(
        plan["_normalized"]["launcher_path"],
        wrapper_plan["remote_python"]["path"],
        "managed.launcher",
    )
    ssh = plan["ssh"]
    exact(
        ssh["remote_python_path"],
        wrapper_plan["remote_python"]["path"],
        "managed.remote_python.path",
    )
    exact(
        ssh["remote_python_sha256"],
        wrapper_plan["remote_python"]["sha256"],
        "managed.remote_python.sha256",
    )
    exact(
        ssh["remote_python_stat"],
        wrapper_plan["remote_python"]["stat"],
        "managed.remote_python.stat",
    )
    components = {
        component["path"]: component
        for component in plan["components"]
    }
    expected = {
        wrapper_plan["remote_python"]["path"]: wrapper_plan["remote_python"],
        **{
            artifact["path"]: artifact
            for artifact in wrapper_plan["source_artifacts"].values()
        },
        **{
            artifact["path"]: artifact
            for artifact in wrapper_plan["input_artifacts"].values()
        },
    }
    require(ssh["nvidia_smi_path"] in components, "E_NVIDIA_COMPONENT")
    exact(
        set(components),
        set(expected) | {ssh["nvidia_smi_path"]},
        "managed.component_paths",
    )
    for path, artifact in expected.items():
        exact(
            _component_artifact(components[path]),
            artifact,
            f"managed.component[{path}]",
        )
    exact(
        sorted(plan["_normalized"]["component_map"]),
        [
            component["component_id"]
            for component in plan["components"]
        ],
        "managed.component_ids",
    )
    return plan


def _parse_prefixed(
    raw: bytes,
    prefix: bytes,
    field: str,
) -> list[dict[str, Any]]:
    rows = []
    for line in raw.splitlines(keepends=True):
        if not line.startswith(prefix):
            continue
        payload = line[len(prefix):]
        value = common.parse_json(payload, field)
        require(type(value) is dict, f"E_TYPE: {field}")
        exact(common.canonical_bytes(value), payload, f"{field}.canonical")
        rows.append(value)
    return rows


def parse_launcher_output(
    raw: bytes,
    managed_plan: dict[str, Any],
    managed_sha256: str,
    remote_boot_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    require(0 < len(raw) <= MAX_LAUNCHER_OUTPUT, "E_LAUNCHER_OUTPUT_SIZE")
    rows = (
        _parse_prefixed(raw, TRANSPORT_PREFIX, "managed_transport"),
        _parse_prefixed(raw, RUNTIME_PREFIX, "remote_runtime"),
        _parse_prefixed(raw, CLEANUP_PREFIX, "managed_cleanup"),
    )
    for group, name in zip(rows, ("transport", "runtime", "cleanup")):
        exact(len(group), 1, f"{name}.count")
    for line in raw.splitlines(keepends=True):
        require(
            line.startswith((TRANSPORT_PREFIX, RUNTIME_PREFIX, CLEANUP_PREFIX)),
            "E_LAUNCHER_STDOUT",
        )
    transport = contract._managed_transport(rows[0][0], "managed_transport")
    runtime = contract._remote_process(rows[1][0], "remote_runtime")
    cleanup = contract._managed_cleanup(rows[2][0], "managed_cleanup")
    exact(transport["bundle_id"], managed_plan["bundle_id"], "transport.bundle")
    exact(transport["plan_sha256"], managed_sha256, "transport.plan")
    exact(transport["remote_boot_id"], remote_boot_id, "transport.boot")
    exact(runtime["bundle_id"], managed_plan["bundle_id"], "runtime.bundle")
    exact(runtime["boot_id"], remote_boot_id, "runtime.boot")
    exact(cleanup["boot_id"], remote_boot_id, "cleanup.boot")
    exact(cleanup["gpu_uuid"], managed_plan["ssh"]["gpu_uuid"], "cleanup.gpu")
    for key in ("launch_token", "pid", "pgid", "start_ticks"):
        exact(cleanup[key], runtime[key], f"cleanup.{key}")
    require(
        {"pid": runtime["pid"], "start_ticks": runtime["start_ticks"]}
        in cleanup["absent"],
        "E_CLEANUP_PRODUCER",
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
    process.wait(timeout=timeout_seconds)


def controller_boot_id() -> str:
    value = Path("/proc/sys/kernel/random/boot_id").read_text(
        encoding="ascii"
    ).strip()
    common.uuid(value, "controller_boot_id")
    return value


def run_controller_process(
    argv: list[str],
    timeout_seconds: int,
    maximum_output: int,
    field: str,
) -> tuple[bytes, dict[str, Any]]:
    boot_id = controller_boot_id()
    started_ns = time.monotonic_ns()
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
        process.communicate()
        require(process_absent(pid, ticks), f"E_{field}_TIMEOUT_ORPHAN")
        raise ExecuteError(f"E_{field}_TIMEOUT") from error
    completed_ns = time.monotonic_ns()
    require(len(stdout) <= maximum_output, f"E_{field}_OUTPUT_SIZE")
    require(process.returncode == 0, f"E_{field}_EXIT: {process.returncode}")
    require(stderr == b"", f"E_{field}_STDERR")
    require(process_absent(pid, ticks), f"E_{field}_ORPHAN")
    return stdout, {
        "argv": argv,
        "clock": "CONTROLLER_MONOTONIC",
        "completed_ns": completed_ns,
        "controller_boot_id": boot_id,
        "exit_code": process.returncode,
        "pid": pid,
        "schema": contract.CONTROLLER_TRANSPORT_SCHEMA,
        "start_ticks": ticks,
        "started_ns": started_ns,
    }


def execution_argv(
    wrapper_plan: dict[str, Any],
    managed_raw: bytes,
    remote_boot_id: str,
) -> list[str]:
    return [
        wrapper_plan["local_python"]["path"],
        "-I",
        wrapper_plan["managed_launcher"]["path"],
        "--plan-json",
        managed_raw.decode("ascii"),
        "--plan-sha256",
        wrapper_plan["managed_plan_sha256"],
        "--boot-id",
        remote_boot_id,
    ]


def _nvidia_component(managed_plan: dict[str, Any]) -> dict[str, Any]:
    path = managed_plan["ssh"]["nvidia_smi_path"]
    rows = [
        component
        for component in managed_plan["components"]
        if component["path"] == path
    ]
    exact(len(rows), 1, "nvidia_component.count")
    return _component_artifact(rows[0])


def fetch_argv(
    launcher: Any,
    wrapper_plan: dict[str, Any],
    managed_plan: dict[str, Any],
    runtime: dict[str, Any],
) -> list[str]:
    payload = {
        "boot_id": runtime["boot_id"],
        "capture_input_paths": wrapper_plan["capture_input_paths"],
        "gpu_uuid": managed_plan["ssh"]["gpu_uuid"],
        "nvidia_smi": _nvidia_component(managed_plan),
        "producer_pid": runtime["pid"],
        "producer_start_ticks": runtime["start_ticks"],
        "remote_acquisition_output": wrapper_plan["remote_acquisition_output"],
        "remote_bundle_root": wrapper_plan["remote_bundle_root"],
        "remote_python": wrapper_plan["remote_python"],
        "remote_runtime_output": wrapper_plan["remote_runtime_output"],
    }
    encoded = base64.b64encode(common.canonical_compact(payload)).decode("ascii")
    command = shlex.join([
        managed_plan["ssh"]["remote_python_path"],
        "-I",
        "-c",
        REMOTE_FETCH_SOURCE,
        encoded,
    ])
    return launcher.ssh_prefix(managed_plan["ssh"]) + [command]


def _content_artifact(
    value: Any,
    field: str,
) -> tuple[dict[str, Any], bytes]:
    require(type(value) is dict, f"E_TYPE: {field}")
    row = dict(value)
    encoded = row.pop("content_base64", None)
    artifact = contract._artifact(row, field)
    require(type(encoded) is str and encoded.isascii(), f"E_CONTENT: {field}")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError as error:
        raise ExecuteError(f"E_BASE64: {field}") from error
    exact(len(raw), artifact["bytes"], f"{field}.bytes")
    exact(hashlib.sha256(raw).hexdigest(), artifact["sha256"], f"{field}.sha256")
    return artifact, raw


def validate_fetch(
    value: Any,
    wrapper_plan: dict[str, Any],
    managed_plan: dict[str, Any],
    runtime: dict[str, Any],
    managed_cleanup: dict[str, Any],
) -> tuple[
    dict[str, bytes],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
    int,
]:
    value = common.exact_keys(
        value,
        {
            "capture_input_artifacts",
            "files",
            "remote_cleanup",
            "schema",
            "system_swap_used_bytes",
        },
        "fetch",
    )
    exact(value["schema"], FETCH_SCHEMA, "fetch.schema")
    cleanup = common.exact_keys(
        value["remote_cleanup"],
        {
            "boot_id",
            "clock",
            "gpu_uuid",
            "nvml_compute_pids",
            "observed_ns",
            "producer_absent",
            "schema",
        },
        "fetch.remote_cleanup",
    )
    exact(cleanup["schema"], contract.REMOTE_CLEANUP_SCHEMA, "fetch.cleanup.schema")
    exact(cleanup["boot_id"], runtime["boot_id"], "fetch.cleanup.boot")
    exact(cleanup["gpu_uuid"], managed_plan["ssh"]["gpu_uuid"],
          "fetch.cleanup.gpu")
    exact(cleanup["clock"], "RTX_CLOCK_MONOTONIC_RAW", "fetch.cleanup.clock")
    common.integer(
        cleanup["observed_ns"],
        "fetch.cleanup.observed_ns",
        managed_cleanup["observed_ns"],
    )
    exact(cleanup["nvml_compute_pids"], [], "fetch.cleanup.nvml")
    exact(
        cleanup["producer_absent"],
        {"pid": runtime["pid"], "start_ticks": runtime["start_ticks"]},
        "fetch.cleanup.producer",
    )
    swap = common.integer(value["system_swap_used_bytes"], "fetch.swap")
    capture_values = common.exact_keys(
        value["capture_input_artifacts"],
        {"cuda_monolithic", "joint_phone_cuda"},
        "fetch.capture_inputs",
    )
    captures = {}
    for role in sorted(capture_values):
        artifact, unused_raw = _content_artifact(
            capture_values[role],
            f"fetch.capture_inputs.{role}",
        )
        del unused_raw
        exact(
            artifact["path"],
            wrapper_plan["capture_input_paths"][role],
            f"fetch.capture_inputs.{role}.path",
        )
        captures[role] = artifact
    rows = value["files"]
    require(type(rows) is list and 1 <= len(rows) <= MAX_FILES, "E_FETCH_FILES")
    contents = {}
    snapshots = []
    logical_paths = []
    source_paths = []
    total = 0
    for index, row in enumerate(rows):
        row = common.exact_keys(
            row,
            {"artifact", "materialized_path"},
            f"fetch.files[{index}]",
        )
        relative = common.relative_path(
            row["materialized_path"],
            f"fetch.files[{index}].materialized_path",
        )
        require(str(Path(relative)) == relative, "E_FETCH_PATH_CANONICAL")
        artifact, raw = _content_artifact(
            row["artifact"],
            f"fetch.files[{index}].artifact",
        )
        logical_paths.append(relative)
        source_paths.append(artifact["path"])
        total += len(raw)
        require(total <= MAX_BUNDLE_BYTES, "E_FETCH_TOTAL")
        contents[relative] = raw
        snapshots.append({
            "artifact": artifact,
            "materialized_path": relative,
        })
    exact(logical_paths, sorted(set(logical_paths)), "fetch.file_order")
    require(len(source_paths) == len(set(source_paths)), "E_FETCH_SOURCE_REUSE")
    required = {
        contract.RAW_MANIFEST_NAME,
        contract.RUNTIME_IDENTITY_NAME,
        contract.ACQUISITION_NAME,
        "raw/cuda-monolithic.json",
        "raw/joint-phone-cuda.json",
    }
    require(required.issubset(contents), "E_FETCH_REQUIRED_FILES")
    for relative, expected_source in (
        (
            contract.RAW_MANIFEST_NAME,
            str(
                Path(wrapper_plan["remote_bundle_root"])
                / contract.V24_RAW_MANIFEST_NAME
            ),
        ),
        (
            contract.RUNTIME_IDENTITY_NAME,
            wrapper_plan["remote_runtime_output"],
        ),
        (
            contract.ACQUISITION_NAME,
            wrapper_plan["remote_acquisition_output"],
        ),
    ):
        row = next(
            item for item in snapshots if item["materialized_path"] == relative
        )
        exact(row["artifact"]["path"], expected_source, f"fetch.source.{relative}")
    root = Path(wrapper_plan["remote_bundle_root"])
    for row in snapshots:
        relative = row["materialized_path"]
        if relative in {
            contract.RAW_MANIFEST_NAME,
            contract.RUNTIME_IDENTITY_NAME,
            contract.ACQUISITION_NAME,
        }:
            continue
        exact(
            row["artifact"]["path"],
            str(root / relative),
            f"fetch.source.{relative}",
        )
    for role, relative in (
        ("cuda_monolithic", "raw/cuda-monolithic.json"),
        ("joint_phone_cuda", "raw/joint-phone-cuda.json"),
    ):
        exact(
            {
                "bytes": captures[role]["bytes"],
                "sha256": captures[role]["sha256"],
            },
            {
                "bytes": len(contents[relative]),
                "sha256": hashlib.sha256(contents[relative]).hexdigest(),
            },
            f"fetch.capture_content.{role}",
        )
    manifest_raw = contents[contract.RAW_MANIFEST_NAME]
    manifest = common.parse_json(manifest_raw, "remote_raw_manifest")
    require(type(manifest) is dict, "E_RAW_MANIFEST_TYPE")
    exact(common.canonical_bytes(manifest), manifest_raw, "remote_raw_manifest")
    exact(manifest.get("schema"), "s39-cp0-r1-evidence-bundle-v2.1",
          "remote_raw_manifest.schema")
    exact(manifest.get("phase"), contract.PHASE, "remote_raw_manifest.phase")
    exact(manifest.get("phase_id"), wrapper_plan["v24_phase_id"],
          "remote_raw_manifest.phase_id")
    manifest_artifacts = manifest.get("artifacts")
    require(
        type(manifest_artifacts) is list and bool(manifest_artifacts),
        "E_RAW_MANIFEST_ARTIFACTS",
    )
    expected_files = {
        contract.RAW_MANIFEST_NAME,
        contract.RUNTIME_IDENTITY_NAME,
        contract.ACQUISITION_NAME,
        "raw/cuda-monolithic.json",
        "raw/joint-phone-cuda.json",
    }
    for index, row in enumerate(manifest_artifacts):
        require(type(row) is dict, f"E_RAW_MANIFEST_ROW: {index}")
        path = common.relative_path(
            row.get("path"),
            f"remote_raw_manifest.artifacts[{index}].path",
        )
        require(str(Path(path)) == path, "E_RAW_MANIFEST_PATH")
        expected_files.add(path)
    exact(set(contents), expected_files, "fetch.file_set")
    phase_closed = common.integer(
        manifest.get("phase_closed_ns"),
        "remote_raw_manifest.phase_closed_ns",
        runtime["remote_observed_ns"],
    )
    require(
        phase_closed <= managed_cleanup["observed_ns"],
        "E_RAW_MANIFEST_CLEANUP_ORDER",
    )
    files = [
        {
            "bytes": len(raw),
            "path": relative,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        for relative, raw in sorted(contents.items())
    ]
    remote_bundle = contract.bundle_manifest(
        wrapper_plan["remote_bundle_root"],
        files,
    )
    contract._bundle_manifest(remote_bundle, "fetch.remote_bundle")
    return contents, remote_bundle, snapshots, captures, cleanup, swap


def _canonical_local_path(path: Path, field: str) -> Path:
    require(path.is_absolute(), f"E_PATH: {field}")
    require(str(path) == str(Path(str(path))) and ".." not in path.parts,
            f"E_PATH_CANONICAL: {field}")
    return path


def _reject_symlink_chain(path: Path) -> None:
    current = Path("/")
    for part in path.parts[1:]:
        current /= part
        metadata = current.lstat()
        require(not stat.S_ISLNK(metadata.st_mode), f"E_PARENT_SYMLINK: {current}")
        require(stat.S_ISDIR(metadata.st_mode), f"E_PARENT_DIRECTORY: {current}")


def _write_file_new(path: Path, raw: bytes) -> None:
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
    exact(read_regular(path), raw, f"reopen.{path}")


def scan_bundle(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"E_LOCAL_SYMLINK: {path}")
        if path.is_dir():
            continue
        raw = read_regular(path)
        rows.append({
            "bytes": len(raw),
            "path": str(path.relative_to(root)),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
    return rows


def materialize_bundle(
    root: Path,
    contents: dict[str, bytes],
) -> dict[str, Any]:
    root = _canonical_local_path(root, "bundle_root")
    require(not root.exists() and not root.is_symlink(), "E_BUNDLE_ROOT_EXISTS")
    _reject_symlink_chain(root.parent)
    os.mkdir(root, 0o755)
    directories = {root}
    try:
        for relative, raw in sorted(contents.items()):
            common.relative_path(relative, f"materialize.{relative}")
            target = root / relative
            current = root
            for part in Path(relative).parts[:-1]:
                current /= part
                if current not in directories:
                    os.mkdir(current, 0o755)
                    directories.add(current)
            _write_file_new(target, raw)
        for directory in sorted(
            directories,
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            descriptor = os.open(
                directory,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        parent = os.open(
            root.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
        )
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except BaseException:
        raise
    files = scan_bundle(root)
    exact([row["path"] for row in files], sorted(contents), "local.files")
    return contract.bundle_manifest(str(root), files)


def listener_absent(managed_plan: dict[str, Any]) -> bool:
    forward = managed_plan["route"]["local_forward"]
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(0.2)
        return (
            connection.connect_ex(
                (forward["local_host"], forward["local_port"])
            )
            != 0
        )


def _public_managed_plan(value: dict[str, Any]) -> dict[str, Any]:
    return contract._managed_public_plan(value)


def validate_receipt(
    value: Any,
    wrapper_plan: dict[str, Any],
    wrapper_plan_raw: bytes,
    managed_plan: dict[str, Any],
    managed_plan_raw: bytes,
) -> dict[str, Any]:
    require(common is not None and contract is not None, "E_PINNED_MODULES")
    contract.validate_plan(wrapper_plan)
    launcher = load_managed_launcher(wrapper_plan["managed_launcher"])
    parsed_managed = validate_managed_plan(
        launcher,
        managed_plan_raw,
        wrapper_plan["managed_plan_sha256"],
        wrapper_plan,
    )
    remote_boot_id = common.uuid(
        value.get("remote_boot_id") if type(value) is dict else None,
        "receipt.remote_boot_id",
    )
    parsed_managed["ssh"]["_expected_boot_id"] = remote_boot_id
    exact(
        _public_managed_plan(managed_plan),
        _public_managed_plan(parsed_managed),
        "receipt.managed_plan.parsed",
    )
    for key, module_path in (
        ("local_common", Path(common.__file__)),
        ("contract_validator", Path(contract.__file__)),
        ("executor", Path(__file__)),
    ):
        reopen_artifact(wrapper_plan[key], f"wrapper_plan.{key}")
        exact(
            module_path.resolve(),
            Path(wrapper_plan[key]["path"]).resolve(),
            f"receipt.{key}.path",
        )
    value = contract.validate_receipt(
        value,
        wrapper_plan,
        wrapper_plan_raw,
        parsed_managed,
        managed_plan_raw,
    )
    runtime = value["remote_producer_process"]
    exact(
        value["execution_transport"]["argv"],
        execution_argv(wrapper_plan, managed_plan_raw, value["remote_boot_id"]),
        "receipt.execution_transport.argv",
    )
    exact(
        value["fetch_transport"]["argv"],
        fetch_argv(launcher, wrapper_plan, parsed_managed, runtime),
        "receipt.fetch_transport.argv",
    )
    exact(
        value["managed_transport_process"]["argv"],
        launcher.ssh_prefix(
            parsed_managed["ssh"],
            parsed_managed["route"]["local_forward"],
        )
        + value["managed_transport_process"]["argv"][
            len(
                launcher.ssh_prefix(
                    parsed_managed["ssh"],
                    parsed_managed["route"]["local_forward"],
                )
            ):
        ],
        "receipt.managed_transport.argv",
    )
    return value


def _write_receipt(path: Path, value: dict[str, Any]) -> None:
    path = _canonical_local_path(path, "receipt")
    require(not path.exists() and not path.is_symlink(), "E_RECEIPT_EXISTS")
    _reject_symlink_chain(path.parent)
    raw = common.canonical_bytes(value)
    _write_file_new(path, raw)
    parsed = common.parse_json(read_regular(path, MAX_PLAN_BYTES), "receipt")
    exact(parsed, value, "receipt.reopen")


def execute(
    *,
    wrapper_plan: dict[str, Any],
    wrapper_plan_raw: bytes,
    wrapper_plan_sha256: str,
    managed_plan: dict[str, Any],
    managed_plan_raw: bytes,
    launcher: Any,
    remote_boot_id: str,
    bundle_root: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    bundle_root = _canonical_local_path(bundle_root, "bundle_root")
    receipt_path = _canonical_local_path(receipt_path, "receipt")
    require(
        not receipt_path.is_relative_to(bundle_root),
        "E_RECEIPT_INSIDE_BUNDLE",
    )
    require(
        not bundle_root.exists()
        and not receipt_path.exists(),
        "E_OUTPUT_EXISTS",
    )
    started_ns = time.monotonic_ns()
    stdout, execution_transport = run_controller_process(
        execution_argv(wrapper_plan, managed_plan_raw, remote_boot_id),
        wrapper_plan["timeout_seconds"],
        MAX_LAUNCHER_OUTPUT,
        "EXECUTION",
    )
    managed_transport, runtime, managed_cleanup = parse_launcher_output(
        stdout,
        managed_plan,
        wrapper_plan["managed_plan_sha256"],
        remote_boot_id,
    )
    require(
        process_absent(
            managed_transport["pid"],
            managed_transport["start_ticks"],
        ),
        "E_MANAGED_TRANSPORT_LIVE",
    )
    require(listener_absent(managed_plan), "E_LOCAL_FORWARD_LIVE")
    launcher.verify_ssh(managed_plan["ssh"])
    fetch_command = fetch_argv(
        launcher,
        wrapper_plan,
        managed_plan,
        runtime,
    )
    fetch_stdout, fetch_transport = run_controller_process(
        fetch_command,
        wrapper_plan["timeout_seconds"],
        MAX_FETCH_OUTPUT,
        "FETCH",
    )
    try:
        decoded = base64.b64decode(fetch_stdout.strip(), validate=True)
    except ValueError as error:
        raise ExecuteError("E_FETCH_BASE64") from error
    fetch_value = common.parse_json(decoded, "fetch")
    exact(common.canonical_bytes(fetch_value), decoded, "fetch.canonical")
    (
        contents,
        remote_bundle,
        snapshots,
        captures,
        remote_cleanup,
        swap,
    ) = validate_fetch(
        fetch_value,
        wrapper_plan,
        managed_plan,
        runtime,
        managed_cleanup,
    )
    local_bundle = materialize_bundle(bundle_root, contents)
    exact(local_bundle["files"], remote_bundle["files"], "bundle.files")
    exact(
        local_bundle["root_sha256"],
        remote_bundle["root_sha256"],
        "bundle.root_sha256",
    )
    controller_cleanup_ns = time.monotonic_ns()
    controller_cleanup = {
        "clock": "CONTROLLER_MONOTONIC",
        "execution_transport_absent": process_absent(
            execution_transport["pid"],
            execution_transport["start_ticks"],
        ),
        "fetch_transport_absent": process_absent(
            fetch_transport["pid"],
            fetch_transport["start_ticks"],
        ),
        "local_forward_listener_absent": listener_absent(managed_plan),
        "managed_transport_absent": process_absent(
            managed_transport["pid"],
            managed_transport["start_ticks"],
        ),
        "observed_ns": controller_cleanup_ns,
    }
    require(
        all(
            value is True
            for key, value in controller_cleanup.items()
            if key.endswith("_absent")
        ),
        "E_CONTROLLER_CLEANUP",
    )
    file_map = {
        row["path"]: row
        for row in local_bundle["files"]
    }
    completed_ns = time.monotonic_ns()
    result = {
        "acquisition_artifact": file_map[contract.ACQUISITION_NAME],
        "capture_input_artifacts": captures,
        "completed_ns": completed_ns,
        "contract_validator": wrapper_plan["contract_validator"],
        "controller_cleanup": controller_cleanup,
        "controller_clock": "CONTROLLER_MONOTONIC",
        "execution_transport": execution_transport,
        "executor": wrapper_plan["executor"],
        "fetch_transport": fetch_transport,
        "gpu_uuid": managed_plan["ssh"]["gpu_uuid"],
        "local_bundle": local_bundle,
        "local_common": wrapper_plan["local_common"],
        "managed_plan_artifact": wrapper_plan["managed_plan"],
        "managed_plan_sha256": wrapper_plan["managed_plan_sha256"],
        "managed_remote_cleanup": managed_cleanup,
        "managed_transport_process": managed_transport,
        "outer_phase_id": wrapper_plan["outer_phase_id"],
        "phase": contract.PHASE,
        "remote_boot_id": remote_boot_id,
        "remote_bundle": remote_bundle,
        "remote_cleanup": remote_cleanup,
        "remote_execution_interval": {
            "clock": "RTX_CLOCK_MONOTONIC_RAW",
            "completed_ns": managed_cleanup["observed_ns"],
            "phase_closed_ns": common.parse_json(
                contents[contract.RAW_MANIFEST_NAME],
                "raw_manifest",
            )["phase_closed_ns"],
            "started_ns": runtime["remote_observed_ns"],
        },
        "remote_producer_process": runtime,
        "remote_snapshot_artifacts": snapshots,
        "role": contract.ROLE,
        "runtime_identity_artifact": file_map[contract.RUNTIME_IDENTITY_NAME],
        "schema": contract.RECEIPT_SCHEMA,
        "started_ns": started_ns,
        "system_swap_used_bytes": swap,
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
    contract.validate_materialized_bundle(
        result,
        bundle_root,
        wrapper_plan,
        wrapper_plan_raw,
        managed_plan,
        managed_plan_raw,
    )
    _write_receipt(receipt_path, result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--boot-id", required=True)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    args = parser.parse_args(argv)
    require(args.execute and args.confirm == CONFIRMATION, "E_CONFIRM")
    return args


def main(argv: list[str] | None = None) -> int:
    global common, contract
    try:
        args = parse_args(argv)
        (
            wrapper_plan,
            wrapper_raw,
            common,
            contract,
        ) = load_pinned_plan(
            Path(args.plan),
            args.plan_sha256,
        )
        for key in (
            "contract_validator",
            "executor",
            "local_common",
            "local_python",
            "managed_launcher",
            "managed_plan",
        ):
            reopen_artifact(wrapper_plan[key], f"wrapper_plan.{key}")
        exact(
            Path(wrapper_plan["executor"]["path"]).resolve(),
            Path(__file__).resolve(),
            "executor.path",
        )
        exact(
            os.path.realpath(sys.executable),
            wrapper_plan["local_python"]["path"],
            "local_python.executable",
        )
        managed_raw = reopen_artifact(
            wrapper_plan["managed_plan"],
            "managed_plan",
        )
        exact(
            hashlib.sha256(managed_raw).hexdigest(),
            wrapper_plan["managed_plan_sha256"],
            "managed_plan.sha256",
        )
        launcher = load_managed_launcher(wrapper_plan["managed_launcher"])
        managed = validate_managed_plan(
            launcher,
            managed_raw,
            wrapper_plan["managed_plan_sha256"],
            wrapper_plan,
        )
        remote_boot_id = common.uuid(args.boot_id, "remote_boot_id")
        managed["ssh"]["_expected_boot_id"] = remote_boot_id
        execute(
            wrapper_plan=wrapper_plan,
            wrapper_plan_raw=wrapper_raw,
            wrapper_plan_sha256=args.plan_sha256,
            managed_plan=managed,
            managed_plan_raw=managed_raw,
            launcher=launcher,
            remote_boot_id=remote_boot_id,
            bundle_root=Path(args.bundle_root),
            receipt_path=Path(args.receipt),
        )
        return 0
    except (
        ExecuteError,
        OSError,
        RuntimeError,
        ValueError,
        subprocess.SubprocessError,
    ) as error:
        print(f"S39_V25_REMOTE_FAN_IN_REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
