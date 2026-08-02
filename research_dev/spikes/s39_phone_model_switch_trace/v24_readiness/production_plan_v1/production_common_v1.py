#!/usr/bin/env python3
"""Strict helpers for the V2.4 A_ONLY production plan."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import time
from typing import Any, Protocol


CUDA_SSH_TARGET = "zhihao@172.20.74.85"
PHONE_ADB_PORT = 5038
PHASE = "A_ONLY"
MODEL_ID = "qwen3-14b-q4_k_m"
UNBOUND_BOOT_IDS = {
    "cuda": "00000000-0000-0000-0000-000000000000",
    "op12": "00000000-0000-0000-0000-000000000001",
    "op15": "00000000-0000-0000-0000-000000000002",
}
UNBOUND_PHONE_NETWORK = {
    "op12": {
        "interface": "UNBOUND_AFTER_REBOOT",
        "local_ipv4": "0.0.0.12",
    },
    "op15": {
        "interface": "UNBOUND_AFTER_REBOOT",
        "local_ipv4": "0.0.0.15",
    },
}
UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)
DIGEST_RE = re.compile(r"[0-9a-f]{64}")
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_INT = (1 << 63) - 1


class ProductionError(ValueError):
    pass


class Runner(Protocol):
    def run(
        self,
        argv: list[str],
        *,
        timeout: float,
    ) -> subprocess.CompletedProcess[bytes]:
        ...


class SubprocessRunner:
    def run(
        self,
        argv: list[str],
        *,
        timeout: float,
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            argv,
            capture_output=True,
            check=False,
            timeout=timeout,
        )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProductionError(message)


def exact(value: Any, expected: Any, field: str) -> None:
    require(
        type(value) is type(expected) and value == expected,
        f"E_VALUE: {field}: expected {expected!r}, got {value!r}",
    )


def exact_keys(value: Any, keys: set[str], field: str) -> dict[str, Any]:
    require(type(value) is dict, f"E_TYPE: {field}")
    actual = set(value)
    require(
        actual == keys,
        f"E_KEYS: {field}: missing={sorted(keys - actual)}, "
        f"unknown={sorted(actual - keys)}",
    )
    return value


def integer(value: Any, field: str, minimum: int = 0) -> int:
    require(
        type(value) is int and minimum <= value <= MAX_INT,
        f"E_INTEGER: {field}",
    )
    return value


def text(value: Any, field: str) -> str:
    require(
        type(value) is str and bool(value) and value.isascii(),
        f"E_TEXT: {field}",
    )
    return value


def digest(value: Any, field: str) -> str:
    value = text(value, field)
    require(DIGEST_RE.fullmatch(value) is not None, f"E_DIGEST: {field}")
    return value


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        require(key not in result, f"E_DUPLICATE_KEY: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ProductionError(f"E_JSON_NUMBER: {value}")


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
        raise ProductionError("E_CANONICAL") from error


def parse_json(raw: bytes, field: str) -> Any:
    try:
        return json.loads(
            raw.decode("ascii"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProductionError(f"E_JSON: {field}: {error}") from error


def read_regular(path: Path, field: str, maximum: int = MAX_FILE_BYTES) -> bytes:
    require(path.is_absolute(), f"E_ABSOLUTE_PATH: {field}")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ProductionError(f"E_READ: {field}: {path}: {error}") from error
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"E_REGULAR: {field}")
        require(0 < before.st_size <= maximum, f"E_SIZE: {field}")
        raw = bytearray()
        while block := os.read(descriptor, min(1024 * 1024, maximum + 1)):
            raw.extend(block)
            require(len(raw) <= maximum, f"E_SIZE: {field}")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    exact(identity(after), identity(before), f"E_TOCTOU: {field}")
    exact(len(raw), before.st_size, f"E_READ_SIZE: {field}")
    return bytes(raw)


def read_canonical(path: Path, field: str) -> tuple[dict[str, Any], bytes]:
    raw = read_regular(path, field)
    value = parse_json(raw, field)
    require(type(value) is dict, f"E_TYPE: {field}")
    exact(canonical_bytes(value), raw, f"E_CANONICAL: {field}")
    return value, raw


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_record(path: Path, field: str) -> dict[str, Any]:
    raw = read_regular(path, field)
    return {
        "bytes": len(raw),
        "path": str(path),
        "sha256": sha256_bytes(raw),
    }


def write_new(path: Path, value: Any) -> bytes:
    require(path.is_absolute(), "E_OUTPUT_ABSOLUTE")
    raw = canonical_bytes(value)
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
    return raw


def write_raw_new(path: Path, raw: bytes) -> None:
    require(path.is_absolute() and bool(raw), "E_RAW_OUTPUT")
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


def run_probe(
    runner: Runner,
    argv: list[str],
    timeout: float,
    field: str,
    *,
    empty_stdout: bool = False,
) -> bytes:
    try:
        result = runner.run(argv, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        raise ProductionError(f"E_TIMEOUT: {field}") from error
    exact(result.returncode, 0, f"E_EXIT: {field}")
    exact(result.stderr, b"", f"E_STDERR: {field}")
    require(type(result.stdout) is bytes, f"E_STDOUT_TYPE: {field}")
    if empty_stdout:
        exact(result.stdout, b"", f"E_STDOUT: {field}")
    return result.stdout


def parse_assignments(raw: bytes, field: str) -> dict[str, str]:
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ProductionError(f"E_ASCII: {field}") from error
    result = {}
    for line in lines:
        key, separator, value = line.partition("=")
        require(separator == "=" and key and key not in result, f"E_FIELD: {field}")
        result[key] = value
    return result


def adb(serial: str, *argv: str, port: int = PHONE_ADB_PORT) -> list[str]:
    return ["adb", "-P", str(port), "-s", serial, *argv]


def preflight_commands(
    contract: dict[str, Any],
    artifact_root: dict[str, Any],
) -> dict[str, list[str]]:
    model_component = next(
        value
        for value in artifact_root["components"]
        if value["component_id"] == "model.cuda"
    )
    geometry = contract["model_geometry"][MODEL_ID]
    route = contract["incumbent_route_lock"]
    cuda = contract["devices"]["cuda"]
    cuda_script = (
        "set -eu; hostname; "
        f"nvidia-smi --id={shlex.quote(cuda['uuid'])} "
        "--query-gpu=name,uuid,memory.total --format=csv,noheader,nounits; "
        f"printf 'MODEL_BYTES='; stat -c %s {shlex.quote(model_component['path'])}; "
        f"printf 'MODEL_SHA256='; sha256sum {shlex.quote(model_component['path'])} "
        "| cut -d' ' -f1"
    )
    commands = {
        "adb_5037": ["adb", "-P", "5037", "devices", "-l"],
        "adb_5038": ["adb", "-P", "5038", "devices", "-l"],
        "cuda_A": [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            CUDA_SSH_TARGET,
            cuda_script,
        ],
    }
    for phone in ("op15", "op12"):
        expected = contract["devices"][phone]
        shard = geometry["known_shards"][phone]
        exact(
            shard["sha256"],
            route[f"{phone}_shard_sha256"],
            f"preflight.{phone}.shard",
        )
        script = (
            "set -eu; getprop ro.product.model; getprop ro.product.name; "
            "getprop ro.product.device; cat /proc/sys/kernel/random/boot_id; "
            f"printf 'SHARD_BYTES='; stat -c %s {shlex.quote(shard['path'])}; "
            f"printf 'SHARD_SHA256='; sha256sum {shlex.quote(shard['path'])} "
            "| cut -d' ' -f1"
        )
        commands[f"{phone}_A"] = adb(expected["serial"], "shell", script)
    return commands


def capture_preflight(
    runner: Runner,
    contract: dict[str, Any],
    artifact_root: dict[str, Any],
    timeout: float,
    clock_ns,
) -> dict[str, Any]:
    probes = {}
    for label, argv in sorted(preflight_commands(contract, artifact_root).items()):
        started_ns = clock_ns()
        raw = run_probe(runner, argv, timeout, f"preflight.{label}")
        completed_ns = clock_ns()
        require(started_ns <= completed_ns, f"E_PREFLIGHT_INTERVAL: {label}")
        try:
            stdout = raw.decode("ascii")
        except UnicodeDecodeError as error:
            raise ProductionError(f"E_PREFLIGHT_ASCII: {label}") from error
        probes[label] = {
            "argv": argv,
            "completed_ns": completed_ns,
            "returncode": 0,
            "started_ns": started_ns,
            "stderr": "",
            "stdout": stdout,
        }
    return {
        "probes": probes,
        "schema": "s39-cp0-r1-v24-preflight-capture-v1",
    }


def validate_phase_id(value: Any) -> str:
    value = text(value, "phase_id")
    require(
        value.startswith("cp0-r1-v24-a-only-")
        and len(value) <= 128
        and all(character.isalnum() or character in ".-_" for character in value),
        "E_PHASE_ID",
    )
    return value
