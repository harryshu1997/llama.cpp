#!/usr/bin/env python3
"""Bind one gateway process to its finalized warm-tier runtime record."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import socket
import stat
import struct
import threading
import time
from typing import Any


MAX_RUNTIME_BYTES = 16 * 1024 * 1024
MAX_EXECUTABLE_BYTES = 2 * 1024 * 1024 * 1024
ROOT_KEYS = {
    "c3_profile_lock_sha256",
    "configuration",
    "evidence_root_sha256",
    "event_log_path",
    "executors",
    "initial_models",
    "promotion_enabled",
    "run_id",
    "runtime_plan_sha256",
    "schema",
}
EXECUTOR_KEYS = {
    "credits",
    "execute_concurrency",
    "executor_id",
    "executor_instance_id",
    "expected_peer_pid",
    "expected_peer_start_time_ticks",
    "order",
    "output_limit_bytes",
    "queue_capacity",
    "role",
    "socket_path",
    "timeout_ms",
    "transport",
}
CONTROLLER_IDENTITY_KEYS = {
    "controller_executable_path",
    "controller_executable_sha256",
    "controller_gid",
    "controller_pid",
    "controller_start_time_ticks",
    "controller_uid",
    "host_boot_id",
    "run_id",
    "runtime_config_device",
    "runtime_config_inode",
    "runtime_config_path",
    "runtime_config_sha256",
    "schema",
}
CONTROLLER_BINDING_EVIDENCE_KEYS = {
    "authenticated_ns",
    "controller_executable_path",
    "controller_executable_sha256",
    "controller_gid",
    "controller_identity_device",
    "controller_identity_inode",
    "controller_identity_path",
    "controller_identity_sha256",
    "controller_pid",
    "controller_start_time_ticks",
    "controller_uid",
    "executor_id",
    "executor_instance_id",
    "gateway_pid",
    "gateway_start_time_ticks",
    "host_boot_id",
    "peer_gid",
    "peer_pid",
    "peer_uid",
    "run_id",
    "runtime_config_device",
    "runtime_config_inode",
    "runtime_config_path",
    "runtime_config_sha256",
    "schema",
}


class RuntimeBindingError(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeBinding:
    executor_id: str
    executor_instance_id: str
    gateway_pid: int
    gateway_start_time_ticks: int
    runtime_config_path: str
    runtime_config_sha256: str
    runtime_config_device: int
    runtime_config_inode: int


@dataclass(frozen=True)
class ControllerBinding:
    controller_executable_ctime_ns: int
    controller_executable_device: int
    controller_executable_inode: int
    controller_executable_mtime_ns: int
    controller_executable_path: str
    controller_executable_sha256: str
    controller_executable_size: int
    controller_gid: int
    controller_identity_device: int
    controller_identity_inode: int
    controller_identity_path: str
    controller_identity_sha256: str
    controller_pid: int
    controller_start_time_ticks: int
    controller_uid: int
    host_boot_id: str
    runtime_config_device: int
    runtime_config_inode: int
    runtime_config_path: str
    runtime_config_sha256: str


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeBindingError(message)


def _strict_json(raw: bytes) -> Any:
    def reject_constant(value: str) -> None:
        raise RuntimeBindingError(f"runtime config constant: {value}")

    def reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, "runtime config duplicate key")
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=reject_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeBindingError(f"runtime config JSON: {error}") from error


def _canonical_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise RuntimeBindingError(
            f"runtime config canonical JSON: {error}"
        ) from error


def _integer(value: Any, field: str, minimum: int = 0) -> int:
    require(
        type(value) is int and value >= minimum,
        f"{field}: expected integer >= {minimum}",
    )
    return value


def _string(value: Any, field: str, maximum: int) -> str:
    require(
        type(value) is str
        and 0 < len(value) <= maximum
        and value.isascii(),
        f"{field}: expected nonempty ASCII string",
    )
    return value


def _sha256(value: Any, field: str) -> str:
    result = _string(value, field, 64)
    require(
        len(result) == 64
        and all(character in "0123456789abcdef" for character in result),
        f"{field}: expected lowercase SHA-256",
    )
    return result


def _absolute_path(value: Any, field: str) -> str:
    result = _string(value, field, 4096)
    require(Path(result).is_absolute(), f"{field}: expected absolute path")
    return result


def process_start_time_ticks(pid: int | None = None) -> int:
    expected_pid = os.getpid() if pid is None else _integer(pid, "pid", 1)
    raw = Path(f"/proc/{expected_pid}/stat").read_text(encoding="ascii")
    close = raw.rfind(")")
    require(close > 0, "process stat: malformed comm field")
    prefix = raw[:close]
    open_index = prefix.find("(")
    require(open_index > 0, "process stat: malformed pid field")
    observed_pid = int(prefix[:open_index].strip())
    fields = raw[close + 2:].split()
    require(
        observed_pid == expected_pid and len(fields) >= 20,
        "process stat: identity mismatch",
    )
    return _integer(int(fields[19]), "process start ticks", 1)


def _directory_chain(path: Path) -> tuple[tuple[str, int, int], ...]:
    require(path.is_absolute(), "runtime config path must be absolute")
    current = Path(path.anchor)
    rows: list[tuple[str, int, int]] = []
    for part in path.parts[1:]:
        current /= part
        metadata = os.lstat(current)
        require(
            stat.S_ISDIR(metadata.st_mode)
            and not stat.S_ISLNK(metadata.st_mode),
            f"runtime config parent is not a real directory: {current}",
        )
        rows.append((str(current), metadata.st_dev, metadata.st_ino))
    return tuple(rows)


def _open_directory_chain(
    path: Path,
) -> tuple[int, tuple[tuple[str, int, int], ...]]:
    require(path.is_absolute(), "runtime config path must be absolute")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path.anchor, flags)
    rows: list[tuple[str, int, int]] = []
    current = Path(path.anchor)
    try:
        for part in path.parts[1:]:
            child = os.open(part, flags, dir_fd=descriptor)
            metadata = os.fstat(child)
            named = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
            os.close(descriptor)
            descriptor = child
            current /= part
            require(
                stat.S_ISDIR(metadata.st_mode)
                and stat.S_ISDIR(named.st_mode)
                and (metadata.st_dev, metadata.st_ino)
                == (named.st_dev, named.st_ino),
                f"runtime config parent changed while opening: {current}",
            )
            rows.append((str(current), metadata.st_dev, metadata.st_ino))
        return descriptor, tuple(rows)
    except BaseException:
        os.close(descriptor)
        raise


def read_stable_file(path: Path) -> tuple[bytes, os.stat_result]:
    parent_fd, parent_before = _open_directory_chain(path.parent)
    try:
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path.name, flags, dir_fd=parent_fd)
        try:
            before = os.fstat(descriptor)
            require(
                stat.S_ISREG(before.st_mode)
                and 0 < before.st_size <= MAX_RUNTIME_BYTES,
                "runtime config must be a bounded regular file",
            )
            blocks = []
            remaining = before.st_size
            while remaining:
                block = os.read(descriptor, min(1024 * 1024, remaining))
                require(block, "runtime config truncated during read")
                blocks.append(block)
                remaining -= len(block)
            require(
                os.read(descriptor, 1) == b"",
                "runtime config grew during read",
            )
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
    finally:
        os.close(parent_fd)
    require(
        (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ),
        "runtime config changed during read",
    )
    require(
        stat.S_ISREG(named.st_mode)
        and (named.st_dev, named.st_ino) == (after.st_dev, after.st_ino),
        "runtime config path was replaced during read",
    )
    require(
        _directory_chain(path.parent) == parent_before,
        "runtime config parent was replaced during read",
    )
    return b"".join(blocks), after


def _digest_file_descriptor(path: Path) -> tuple[str, os.stat_result]:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        before = os.fstat(descriptor)
        require(
            stat.S_ISREG(before.st_mode)
            and 0 < before.st_size <= MAX_EXECUTABLE_BYTES,
            "controller executable must be a bounded regular file",
        )
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            require(block, "controller executable truncated during read")
            digest.update(block)
            remaining -= len(block)
        require(
            os.read(descriptor, 1) == b"",
            "controller executable grew during read",
        )
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    require(
        (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ),
        "controller executable changed during read",
    )
    return digest.hexdigest(), after


def _file_descriptor_metadata(path: Path) -> os.stat_result:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        metadata = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    require(
        stat.S_ISREG(metadata.st_mode)
        and 0 < metadata.st_size <= MAX_EXECUTABLE_BYTES,
        "controller executable must be a bounded regular file",
    )
    return metadata


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def read_controller_binding_evidence_capture(
    path: Path,
) -> tuple[dict[str, Any], bytes, os.stat_result]:
    raw, metadata = read_stable_file(path)
    value = _strict_json(raw)
    require(
        _canonical_bytes(value) == raw,
        "controller binding evidence is not canonical",
    )
    require(
        type(value) is dict
        and set(value) == CONTROLLER_BINDING_EVIDENCE_KEYS,
        "controller binding evidence fields",
    )
    require(
        value["schema"] == "s40-executor-controller-binding-v1",
        "controller binding evidence schema",
    )
    for field in (
        "authenticated_ns",
        "controller_gid",
        "controller_identity_device",
        "controller_identity_inode",
        "controller_pid",
        "controller_start_time_ticks",
        "controller_uid",
        "gateway_pid",
        "gateway_start_time_ticks",
        "peer_gid",
        "peer_pid",
        "peer_uid",
        "runtime_config_device",
        "runtime_config_inode",
    ):
        _integer(
            value[field],
            f"controller binding evidence {field}",
            1
            if field
            in {
                "authenticated_ns",
                "controller_identity_inode",
                "controller_pid",
                "controller_start_time_ticks",
                "gateway_pid",
                "gateway_start_time_ticks",
                "peer_pid",
                "runtime_config_inode",
            }
            else 0,
        )
    for field in (
        "controller_executable_sha256",
        "controller_identity_sha256",
        "runtime_config_sha256",
    ):
        _sha256(value[field], f"controller binding evidence {field}")
    for field in (
        "controller_executable_path",
        "controller_identity_path",
        "runtime_config_path",
    ):
        _absolute_path(value[field], f"controller binding evidence {field}")
    for field in (
        "executor_id",
        "executor_instance_id",
        "host_boot_id",
        "run_id",
    ):
        _string(value[field], f"controller binding evidence {field}", 256)
    require(
        (
            value["peer_pid"],
            value["peer_uid"],
            value["peer_gid"],
        )
        == (
            value["controller_pid"],
            value["controller_uid"],
            value["controller_gid"],
        ),
        "controller binding evidence peer mismatch",
    )
    return value, raw, metadata


def read_controller_binding_evidence(path: Path) -> dict[str, Any]:
    value, _, _ = read_controller_binding_evidence_capture(path)
    return value


def _write_new_durable(path: Path, raw: bytes) -> os.stat_result:
    require(path.is_absolute(), "controller binding path must be absolute")
    parent_fd, parent_before = _open_directory_chain(path.parent)
    descriptor = -1
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path.name, flags, 0o600, dir_fd=parent_fd)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            require(written > 0, "controller binding write failed")
            offset += written
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        require(
            stat.S_ISREG(metadata.st_mode)
            and metadata.st_size == len(raw),
            "controller binding publish mismatch",
        )
        os.close(descriptor)
        descriptor = -1
        os.fsync(parent_fd)
        named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)
    require(
        stat.S_ISREG(named.st_mode)
        and (named.st_dev, named.st_ino)
        == (metadata.st_dev, metadata.st_ino),
        "controller binding path was replaced during publish",
    )
    require(
        _directory_chain(path.parent) == parent_before,
        "controller binding parent was replaced during publish",
    )
    return metadata


class ControllerAuthenticator:
    def __init__(
        self,
        identity_path: Path,
        binding_evidence_path: Path,
        *,
        run_id: str,
        runtime_binding: RuntimeBinding,
        timeout_s: float,
    ):
        require(
            hasattr(socket, "SO_PEERCRED"),
            "controller authentication requires SO_PEERCRED",
        )
        require(
            identity_path.is_absolute(),
            "controller identity path must be absolute",
        )
        require(
            binding_evidence_path.is_absolute(),
            "controller binding evidence path must be absolute",
        )
        require(
            type(timeout_s) in (int, float)
            and math.isfinite(timeout_s)
            and 0 < timeout_s <= 3600,
            "controller identity timeout",
        )
        self.identity_path = identity_path
        self.binding_evidence_path = binding_evidence_path
        self.run_id = _string(run_id, "controller identity run ID", 128)
        self.runtime_binding = runtime_binding
        self.timeout_s = float(timeout_s)
        self._lock = threading.Lock()
        self._bound: ControllerBinding | None = None
        self._binding_evidence_identity: tuple[int, int] | None = None
        self._binding_evidence_raw: bytes | None = None

    def _read_identity(
        self,
    ) -> tuple[dict[str, Any], bytes, os.stat_result]:
        deadline = time.monotonic() + self.timeout_s
        while True:
            try:
                raw, metadata = read_stable_file(self.identity_path)
                break
            except FileNotFoundError:
                require(
                    time.monotonic() < deadline,
                    "controller identity wait timed out",
                )
                time.sleep(
                    min(0.01, max(0.0, deadline - time.monotonic()))
                )
        value = _strict_json(raw)
        require(
            _canonical_bytes(value) == raw,
            "controller identity is not canonical",
        )
        require(
            type(value) is dict and set(value) == CONTROLLER_IDENTITY_KEYS,
            "controller identity fields",
        )
        require(
            value["schema"] == "s40-controller-identity-lock-v1",
            "controller identity schema",
        )
        return value, raw, metadata

    def _validate_identity(
        self,
        connection: socket.socket,
        expected: ControllerBinding | None,
    ) -> tuple[ControllerBinding, tuple[int, int, int]]:
        try:
            peer_raw = connection.getsockopt(
                socket.SOL_SOCKET,
                socket.SO_PEERCRED,
                struct.calcsize("3i"),
            )
            peer_pid, peer_uid, peer_gid = struct.unpack("3i", peer_raw)
        except (OSError, struct.error) as error:
            raise RuntimeBindingError(
                f"controller peer credentials: {error}"
            ) from error
        require(peer_pid >= 2, "controller peer PID")
        identity, identity_raw, identity_metadata = self._read_identity()
        controller_pid = _integer(
            identity["controller_pid"],
            "controller identity PID",
            2,
        )
        controller_uid = _integer(
            identity["controller_uid"],
            "controller identity UID",
        )
        controller_gid = _integer(
            identity["controller_gid"],
            "controller identity GID",
        )
        controller_start_ticks = _integer(
            identity["controller_start_time_ticks"],
            "controller identity start ticks",
            1,
        )
        require(
            (peer_pid, peer_uid, peer_gid)
            == (controller_pid, controller_uid, controller_gid),
            "controller peer identity mismatch",
        )
        require(
            process_start_time_ticks(controller_pid)
            == controller_start_ticks,
            "controller process start ticks mismatch",
        )
        proc_root = Path(f"/proc/{controller_pid}")
        proc_metadata = proc_root.stat()
        require(
            (proc_metadata.st_uid, proc_metadata.st_gid)
            == (controller_uid, controller_gid),
            "controller process ownership mismatch",
        )
        host_boot_id = _string(
            identity["host_boot_id"],
            "controller identity boot ID",
            128,
        )
        observed_boot_id = Path(
            "/proc/sys/kernel/random/boot_id"
        ).read_text(encoding="ascii").strip()
        require(
            observed_boot_id == host_boot_id,
            "controller host boot identity mismatch",
        )
        executable_path = _absolute_path(
            identity["controller_executable_path"],
            "controller executable path",
        )
        observed_executable = Path(
            os.readlink(proc_root / "exe")
        )
        require(
            observed_executable.is_absolute()
            and " (deleted)" not in str(observed_executable)
            and observed_executable.resolve() == Path(executable_path).resolve(),
            "controller executable path mismatch",
        )
        executable_sha256 = _sha256(
            identity["controller_executable_sha256"],
            "controller executable SHA-256",
        )
        if expected is None:
            observed_executable_sha256, executable_metadata = (
                _digest_file_descriptor(proc_root / "exe")
            )
            require(
                observed_executable_sha256 == executable_sha256,
                "controller executable digest mismatch",
            )
        else:
            executable_metadata = _file_descriptor_metadata(
                proc_root / "exe"
            )
            require(
                executable_sha256
                == expected.controller_executable_sha256
                and _file_identity(executable_metadata)
                == (
                    expected.controller_executable_device,
                    expected.controller_executable_inode,
                    expected.controller_executable_size,
                    expected.controller_executable_mtime_ns,
                    expected.controller_executable_ctime_ns,
                ),
                "controller executable identity changed",
            )
        runtime_path = _absolute_path(
            identity["runtime_config_path"],
            "controller runtime config path",
        )
        runtime_sha256 = _sha256(
            identity["runtime_config_sha256"],
            "controller runtime config SHA-256",
        )
        runtime_device = _integer(
            identity["runtime_config_device"],
            "controller runtime config device",
        )
        runtime_inode = _integer(
            identity["runtime_config_inode"],
            "controller runtime config inode",
            1,
        )
        runtime_raw, runtime_metadata = read_stable_file(Path(runtime_path))
        require(
            runtime_path == self.runtime_binding.runtime_config_path
            and runtime_sha256
            == self.runtime_binding.runtime_config_sha256
            == hashlib.sha256(runtime_raw).hexdigest()
            and runtime_device
            == self.runtime_binding.runtime_config_device
            == runtime_metadata.st_dev
            and runtime_inode
            == self.runtime_binding.runtime_config_inode
            == runtime_metadata.st_ino,
            "controller runtime config identity mismatch",
        )
        require(
            identity["run_id"] == self.run_id,
            "controller identity run ID mismatch",
        )
        result = ControllerBinding(
            controller_executable_ctime_ns=executable_metadata.st_ctime_ns,
            controller_executable_device=executable_metadata.st_dev,
            controller_executable_inode=executable_metadata.st_ino,
            controller_executable_mtime_ns=executable_metadata.st_mtime_ns,
            controller_executable_path=executable_path,
            controller_executable_sha256=executable_sha256,
            controller_executable_size=executable_metadata.st_size,
            controller_gid=controller_gid,
            controller_identity_device=identity_metadata.st_dev,
            controller_identity_inode=identity_metadata.st_ino,
            controller_identity_path=str(self.identity_path),
            controller_identity_sha256=hashlib.sha256(
                identity_raw
            ).hexdigest(),
            controller_pid=controller_pid,
            controller_start_time_ticks=controller_start_ticks,
            controller_uid=controller_uid,
            host_boot_id=host_boot_id,
            runtime_config_device=runtime_device,
            runtime_config_inode=runtime_inode,
            runtime_config_path=runtime_path,
            runtime_config_sha256=runtime_sha256,
        )
        return result, (peer_pid, peer_uid, peer_gid)

    def _binding_evidence(
        self,
        binding: ControllerBinding,
        peer: tuple[int, int, int],
    ) -> bytes:
        return _canonical_bytes({
            "authenticated_ns": time.monotonic_ns(),
            "controller_executable_path":
                binding.controller_executable_path,
            "controller_executable_sha256":
                binding.controller_executable_sha256,
            "controller_gid": binding.controller_gid,
            "controller_identity_device":
                binding.controller_identity_device,
            "controller_identity_inode":
                binding.controller_identity_inode,
            "controller_identity_path":
                binding.controller_identity_path,
            "controller_identity_sha256":
                binding.controller_identity_sha256,
            "controller_pid": binding.controller_pid,
            "controller_start_time_ticks":
                binding.controller_start_time_ticks,
            "controller_uid": binding.controller_uid,
            "executor_id": self.runtime_binding.executor_id,
            "executor_instance_id":
                self.runtime_binding.executor_instance_id,
            "gateway_pid": self.runtime_binding.gateway_pid,
            "gateway_start_time_ticks":
                self.runtime_binding.gateway_start_time_ticks,
            "host_boot_id": binding.host_boot_id,
            "peer_gid": peer[2],
            "peer_pid": peer[0],
            "peer_uid": peer[1],
            "run_id": self.run_id,
            "runtime_config_device": binding.runtime_config_device,
            "runtime_config_inode": binding.runtime_config_inode,
            "runtime_config_path": binding.runtime_config_path,
            "runtime_config_sha256": binding.runtime_config_sha256,
            "schema": "s40-executor-controller-binding-v1",
        })

    def authenticate(self, connection: socket.socket) -> ControllerBinding:
        with self._lock:
            bound = self._bound
            binding_evidence_raw = self._binding_evidence_raw
            if bound is None:
                binding, peer = self._validate_identity(connection, None)
                raw = self._binding_evidence(binding, peer)
                metadata = _write_new_durable(
                    self.binding_evidence_path,
                    raw,
                )
                rebound_raw, rebound_metadata = read_stable_file(
                    self.binding_evidence_path
                )
                require(
                    rebound_raw == raw
                    and (rebound_metadata.st_dev, rebound_metadata.st_ino)
                    == (metadata.st_dev, metadata.st_ino),
                    "controller binding evidence changed after publish",
                )
                read_controller_binding_evidence(
                    self.binding_evidence_path
                )
                final_raw, final_metadata = read_stable_file(
                    self.binding_evidence_path
                )
                require(
                    final_raw == raw
                    and (final_metadata.st_dev, final_metadata.st_ino)
                    == (metadata.st_dev, metadata.st_ino),
                    "controller binding evidence changed during validation",
                )
                self._bound = binding
                self._binding_evidence_identity = (
                    metadata.st_dev,
                    metadata.st_ino,
                )
                self._binding_evidence_raw = raw
                return binding
            binding_evidence_identity = self._binding_evidence_identity
        require(
            bound is not None
            and binding_evidence_identity is not None
            and binding_evidence_raw is not None,
            "controller binding publication state",
        )
        binding, _ = self._validate_identity(connection, bound)
        require(
            binding == bound,
            "controller identity lock changed",
        )
        raw, metadata = read_stable_file(self.binding_evidence_path)
        require(
            raw == binding_evidence_raw
            and (metadata.st_dev, metadata.st_ino)
            == binding_evidence_identity,
            "controller binding evidence changed",
        )
        return binding


def await_runtime_binding(
    path: Path,
    *,
    executor_id: str,
    executor_instance_id: str,
    run_id: str,
    socket_path: Path,
    timeout_s: float,
) -> RuntimeBinding:
    require(path.is_absolute(), "runtime config path must be absolute")
    require(socket_path.is_absolute(), "gateway socket path must be absolute")
    require(
        type(timeout_s) in (int, float)
        and math.isfinite(timeout_s)
        and 0 < timeout_s <= 3600,
        "startup timeout",
    )
    expected_executor = _string(executor_id, "executor ID", 128)
    expected_instance = _string(
        executor_instance_id,
        "executor instance ID",
        256,
    )
    expected_run = _string(run_id, "runtime run ID", 128)
    gateway_pid = os.getpid()
    gateway_start_ticks = process_start_time_ticks()
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            raw, metadata = read_stable_file(path)
            break
        except FileNotFoundError:
            require(
                time.monotonic() < deadline,
                "runtime config wait timed out",
            )
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
    value = _strict_json(raw)
    require(_canonical_bytes(value) == raw, "runtime config is not canonical")
    require(
        type(value) is dict and set(value) == ROOT_KEYS,
        "runtime config root fields",
    )
    require(
        value["schema"] == "llama-server-warm-tier-runtime-v4",
        "runtime config schema",
    )
    require(value["run_id"] == expected_run, "runtime config run ID")
    executors = value["executors"]
    require(
        type(executors) is list and 0 < len(executors) <= 16,
        "runtime config executors",
    )
    matching = []
    seen_ids = set()
    seen_instances = set()
    for index, record in enumerate(executors):
        field = f"runtime config executor[{index}]"
        require(
            type(record) is dict and set(record) == EXECUTOR_KEYS,
            f"{field}: fields",
        )
        record_id = _string(record["executor_id"], f"{field}.executor_id", 128)
        instance = _string(
            record["executor_instance_id"],
            f"{field}.executor_instance_id",
            256,
        )
        require(
            record_id not in seen_ids and instance not in seen_instances,
            f"{field}: duplicate identity",
        )
        seen_ids.add(record_id)
        seen_instances.add(instance)
        if record_id == expected_executor:
            matching.append(record)
    require(len(matching) == 1, "runtime config executor identity")
    record = matching[0]
    require(
        record["executor_instance_id"] == expected_instance,
        "runtime config executor instance",
    )
    require(
        _integer(record["expected_peer_pid"], "runtime config peer PID", 1)
        == gateway_pid,
        "runtime config peer PID mismatch",
    )
    require(
        _integer(
            record["expected_peer_start_time_ticks"],
            "runtime config peer start ticks",
            1,
        )
        == gateway_start_ticks,
        "runtime config peer start ticks mismatch",
    )
    require(
        record["transport"] == "UNIX_SOCKET"
        and record["socket_path"] == str(socket_path),
        "runtime config gateway socket mismatch",
    )
    return RuntimeBinding(
        executor_id=expected_executor,
        executor_instance_id=expected_instance,
        gateway_pid=gateway_pid,
        gateway_start_time_ticks=gateway_start_ticks,
        runtime_config_path=str(path),
        runtime_config_sha256=hashlib.sha256(raw).hexdigest(),
        runtime_config_device=metadata.st_dev,
        runtime_config_inode=metadata.st_ino,
    )
