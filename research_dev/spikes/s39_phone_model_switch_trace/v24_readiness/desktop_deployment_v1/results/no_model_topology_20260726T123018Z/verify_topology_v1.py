#!/usr/bin/env python3
"""Verify the no-model V2.4 RTX desktop deployment inputs and topology."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Protocol


TOPOLOGY_SCHEMA = "s39-cp0-r1-v24-desktop-topology-v1"
OPERATOR_INPUT_SCHEMA = "s39-cp0-r1-v24-desktop-operator-input-v1"
CONTRACT_SHA256 = (
    "20eea0fd1fb3aba6be9265a1cf84aaddeb867a9c0277db1109273faa4731b455"
)
CONTROLLER_HOST = "zhihao-Z690-C-ac"
CONTROLLER_WIFI_INTERFACE = "wlp3s0"
CONTROLLER_WIFI_IPV4 = "172.20.74.85"
ADB_HOST = "127.0.0.1"
ADB_PORT = 5038
ADB_PATH = "/usr/lib/android-sdk/platform-tools/adb"
NVIDIA_SMI_PATH = "/usr/bin/nvidia-smi"
PYTHON_PATH = "/usr/bin/python3.14"
SSH_PATH = "/usr/bin/ssh"
CUDA_SSH_TARGET = "zhihao@172.20.74.85"
CUDA_NAME = "NVIDIA GeForce RTX 4060 Ti"
CUDA_UUID = "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08"
CUDA_MEMORY_TOTAL_BYTES = 17_175_674_880
CUDA_MONOLITHIC_PORT = 39124
PHASE = "A_ONLY"
MODEL_ID = "qwen3-14b-q4_k_m"
MAX_FILE_BYTES = 16 * 1024 * 1024 * 1024
MAX_RETAINED_FILE_BYTES = 64 * 1024 * 1024
MAX_STDOUT_BYTES = 1024 * 1024
TOPOLOGY_MAXIMUM_AGE_NS = 5_000_000_000
TOPOLOGY_STATUS = "NO_MODEL_TOPOLOGY_PASS"
OPERATOR_INPUT_STATUS = "NO_MODEL_INPUT_PASS_REMOTE_EVIDENCE_PENDING"
UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)
DIGEST_RE = re.compile(r"[0-9a-f]{64}")

EXPECTED_PHONES = {
    "op12": {
        "device": "OP595DL1",
        "model": "CPH2583",
        "product": "CPH2583",
        "serial": "5ae7a43d",
    },
    "op15": {
        "device": "OP611FL1",
        "model": "CPH2749",
        "product": "CPH2749",
        "serial": "3C15AU002CL00000",
    },
}

FILE_KEYS = {
    "adb",
    "candidate",
    "codec",
    "contract",
    "cuda_launcher",
    "cuda_monolithic_launch",
    "cuda_runtime",
    "model",
    "monolithic_launcher",
    "monolithic_runtime",
    "nvidia_smi",
    "python",
    "quality_corpus",
    "runtime_bundle_inventory",
    "ssh",
    "token_history",
    "tokenizer_plan",
    "topology_receipt",
}
EXECUTABLE_FILE_KEYS = {
    "adb",
    "codec",
    "cuda_launcher",
    "cuda_runtime",
    "monolithic_launcher",
    "monolithic_runtime",
    "nvidia_smi",
    "python",
    "ssh",
}
DIRECTORY_KEYS = {"cuda_bundle_root", "joint_cwd"}
PORT_KEYS = {
    "adb_server",
    "cuda_monolithic",
    "cuda_route",
    "op12_stage",
    "op15_stage",
    "relay",
    "relay_tail_source",
}
RUNTIME_BUNDLE_IDS = {
    "cuda_monolithic",
    "cuda_route",
    "op12_stagenet",
    "op15_direct_relay",
    "op15_stagenet",
}
JSON_SCHEMAS = {
    "candidate": "s39-cp0-r1-candidate-v1",
    "contract": "s39-cp0-r1-evidence-contract-v2.4",
    "cuda_monolithic_launch": "s39-cp0-r1-v24-cuda-monolithic-launch-v1",
    "runtime_bundle_inventory": "s39-cp0-r1-runtime-bundle-closure-input-v1",
    "token_history": "s39-cp0-r1-token-history-v2.4",
    "tokenizer_plan": "s39-cp0-r1-a-only-tokenizer-plan-v2",
    "topology_receipt": TOPOLOGY_SCHEMA,
}
RETAIN_FILE_KEYS = set(JSON_SCHEMAS) | {"quality_corpus"}

CONTROLLER_COMMAND = [
    "/bin/sh",
    "-c",
    (
        "set -eu; printf 'HOST='; hostname; "
        "printf 'BOOT_ID='; cat /proc/sys/kernel/random/boot_id"
    ),
]
PATH_RESOLUTION_COMMAND = [
    "/bin/sh",
    "-c",
    (
        "set -eu; PATH=/usr/local/cuda/bin:/usr/bin:/bin; "
        "printf 'ADB='; command -v adb; "
        "printf 'ADB_REAL='; readlink -f \"$(command -v adb)\"; "
        "printf 'NVIDIA_SMI='; command -v nvidia-smi; "
        "printf 'NVIDIA_SMI_REAL='; readlink -f \"$(command -v nvidia-smi)\"; "
        "printf 'PYTHON='; command -v python3; "
        "printf 'PYTHON_REAL='; readlink -f \"$(command -v python3)\"; "
        "printf 'SSH='; command -v ssh"
    ),
]
SELF_SSH_COMMAND = [
    SSH_PATH,
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=10",
    "-o",
    "StrictHostKeyChecking=accept-new",
    "-o",
    "UserKnownHostsFile=/dev/null",
    "-o",
    "LogLevel=ERROR",
    CUDA_SSH_TARGET,
    (
        "set -eu; printf 'HOST='; hostname; "
        "printf 'BOOT_ID='; cat /proc/sys/kernel/random/boot_id; "
        "/usr/bin/nvidia-smi --query-gpu=name,uuid,memory.total "
        "--format=csv,noheader,nounits"
    ),
]
GPU_COMMAND = [
    NVIDIA_SMI_PATH,
    "--query-gpu=name,uuid,memory.total",
    "--format=csv,noheader,nounits",
]
ADB_DEVICES_COMMAND = [ADB_PATH, "-P", str(ADB_PORT), "devices", "-l"]
PHONE_STATUS = """\
set -eu
printf 'PHYSICAL_SERIAL='; getprop ro.serialno
printf 'PRODUCT='; getprop ro.product.name
printf 'MODEL='; getprop ro.product.model
printf 'DEVICE='; getprop ro.product.device
printf 'BOOT_ID='; cat /proc/sys/kernel/random/boot_id
printf 'INTERFACE=wlan0\n'
ip -4 -o addr show dev wlan0 scope global | awk 'NR==1{split($4,a,"/");print "WIFI_IPV4="a[1]}END{if(NR!=1)exit 43}'
"""

PING_OPTIONS = ["-n", "-q", "-c", "3", "-s", "0", "-W", "1", "-w", "5"]


class DeploymentError(ValueError):
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
        raise DeploymentError(message)


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


def text(value: Any, field: str) -> str:
    require(
        type(value) is str
        and bool(value)
        and value.isascii()
        and all(0x20 <= ord(character) <= 0x7E for character in value),
        f"E_TEXT: {field}",
    )
    return value


def integer(value: Any, field: str, minimum: int = 0) -> int:
    require(
        type(value) is int and minimum <= value <= (1 << 63) - 1,
        f"E_INTEGER: {field}",
    )
    return value


def digest(value: Any, field: str) -> str:
    value = text(value, field)
    require(DIGEST_RE.fullmatch(value) is not None, f"E_DIGEST: {field}")
    return value


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise DeploymentError(f"E_DUPLICATE_KEY: {key}")
        result[key] = value
    return result


def parse_json(raw: bytes, field: str) -> Any:
    def reject_constant(value: str) -> None:
        raise DeploymentError(f"E_JSON_NUMBER: {field}: {value}")

    def reject_float(value: str) -> None:
        raise DeploymentError(f"E_JSON_FLOAT: {field}: {value}")

    try:
        return json.loads(
            raw.decode("ascii"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
            parse_float=reject_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DeploymentError(f"E_JSON: {field}") from error


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
        raise DeploymentError("E_CANONICAL") from error


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def stat_identity(value: os.stat_result) -> tuple[int, ...]:
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


def no_symlink_chain(path: Path, field: str) -> None:
    require(path.is_absolute(), f"E_PATH_ABSOLUTE: {field}")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        metadata = os.lstat(current)
        require(not stat.S_ISLNK(metadata.st_mode), f"E_SYMLINK: {field}")


def secure_read(
    path: Path,
    field: str,
    *,
    after_read: Callable[[], None] | None = None,
) -> tuple[bytes, os.stat_result]:
    no_symlink_chain(path, field)
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        require(
            stat.S_ISREG(before.st_mode)
            and 0 < before.st_size <= MAX_FILE_BYTES,
            f"E_FILE: {field}",
        )
        raw = bytearray()
        while block := os.read(descriptor, 4 * 1024 * 1024):
            raw.extend(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    require(
        stat_identity(before) == stat_identity(after)
        and len(raw) == before.st_size,
        f"E_FILE_MUTATION: {field}",
    )
    if after_read is not None:
        after_read()
    no_symlink_chain(path, field)
    descriptor = os.open(path, flags)
    try:
        reopened = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    require(
        stat_identity(before) == stat_identity(reopened),
        f"E_FILE_REOPEN_MUTATION: {field}",
    )
    return bytes(raw), before


def secure_hash(
    path: Path,
    field: str,
    *,
    retain: bool,
    after_read: Callable[[], None] | None = None,
) -> tuple[str, int, os.stat_result, bytes | None]:
    no_symlink_chain(path, field)
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        require(
            stat.S_ISREG(before.st_mode)
            and 0 < before.st_size <= MAX_FILE_BYTES,
            f"E_FILE: {field}",
        )
        if retain:
            require(
                before.st_size <= MAX_RETAINED_FILE_BYTES,
                f"E_RETAIN_SIZE: {field}",
            )
        hasher = hashlib.sha256()
        retained = bytearray() if retain else None
        count = 0
        while block := os.read(descriptor, 4 * 1024 * 1024):
            hasher.update(block)
            count += len(block)
            if retained is not None:
                retained.extend(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    require(
        stat_identity(before) == stat_identity(after)
        and count == before.st_size,
        f"E_FILE_MUTATION: {field}",
    )
    if after_read is not None:
        after_read()
    no_symlink_chain(path, field)
    descriptor = os.open(path, flags)
    try:
        reopened = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    require(
        stat_identity(before) == stat_identity(reopened),
        f"E_FILE_REOPEN_MUTATION: {field}",
    )
    return (
        hasher.hexdigest(),
        count,
        before,
        bytes(retained) if retained is not None else None,
    )


def snapshot_file(path: Path, field: str) -> dict[str, Any]:
    raw, metadata = secure_read(path, field)
    return {
        "bytes": len(raw),
        "path": str(path),
        "sha256": sha256_bytes(raw),
        "stat": stat_record(metadata),
    }


def read_canonical(path: Path, field: str) -> tuple[dict[str, Any], bytes]:
    raw, _ = secure_read(path, field)
    value = parse_json(raw, field)
    require(type(value) is dict, f"E_TYPE: {field}")
    exact(canonical_bytes(value), raw, f"E_CANONICAL: {field}")
    return value, raw


def write_new(path: Path, value: Any) -> None:
    require(path.is_absolute() and not path.exists(), "E_OUTPUT")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_bytes(value)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o644,
    )
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            require(written > 0, "E_OUTPUT_WRITE")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def parse_assignments(raw: bytes, field: str) -> dict[str, str]:
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise DeploymentError(f"E_ASCII: {field}") from error
    values = {}
    for line in lines:
        key, separator, value = line.partition("=")
        require(separator == "=" and key and key not in values, f"E_ASSIGNMENT: {field}")
        values[key] = value
    return values


def phone_command(selector: str) -> list[str]:
    return [
        ADB_PATH,
        "-P",
        str(ADB_PORT),
        "-s",
        selector,
        "shell",
        PHONE_STATUS,
    ]


def desktop_ping_command(target: str) -> list[str]:
    return [
        "/usr/bin/ping",
        *PING_OPTIONS,
        "-I",
        CONTROLLER_WIFI_INTERFACE,
        target,
    ]


def phone_ping_command(selector: str, target: str) -> list[str]:
    return [
        ADB_PATH,
        "-P",
        str(ADB_PORT),
        "-s",
        selector,
        "shell",
        "/system/bin/ping",
        *PING_OPTIONS,
        "-I",
        "wlan0",
        target,
    ]


def connectivity_specs(
    phones: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    op12 = phones["op12"]
    op15 = phones["op15"]
    return [
        {
            "argv": desktop_ping_command(op12["wifi_ipv4"]),
            "edge_id": "desktop_to_op12",
            "interface": CONTROLLER_WIFI_INTERFACE,
            "source_ipv4": CONTROLLER_WIFI_IPV4,
            "target_ipv4": op12["wifi_ipv4"],
        },
        {
            "argv": desktop_ping_command(op15["wifi_ipv4"]),
            "edge_id": "desktop_to_op15",
            "interface": CONTROLLER_WIFI_INTERFACE,
            "source_ipv4": CONTROLLER_WIFI_IPV4,
            "target_ipv4": op15["wifi_ipv4"],
        },
        {
            "argv": phone_ping_command(
                op12["usb_selector"],
                CONTROLLER_WIFI_IPV4,
            ),
            "edge_id": "op12_to_desktop",
            "interface": "wlan0",
            "source_ipv4": op12["wifi_ipv4"],
            "target_ipv4": CONTROLLER_WIFI_IPV4,
        },
        {
            "argv": phone_ping_command(
                op12["usb_selector"],
                op15["wifi_ipv4"],
            ),
            "edge_id": "op12_to_op15",
            "interface": "wlan0",
            "source_ipv4": op12["wifi_ipv4"],
            "target_ipv4": op15["wifi_ipv4"],
        },
        {
            "argv": phone_ping_command(
                op15["usb_selector"],
                CONTROLLER_WIFI_IPV4,
            ),
            "edge_id": "op15_to_desktop",
            "interface": "wlan0",
            "source_ipv4": op15["wifi_ipv4"],
            "target_ipv4": CONTROLLER_WIFI_IPV4,
        },
        {
            "argv": phone_ping_command(
                op15["usb_selector"],
                op12["wifi_ipv4"],
            ),
            "edge_id": "op15_to_op12",
            "interface": "wlan0",
            "source_ipv4": op15["wifi_ipv4"],
            "target_ipv4": op12["wifi_ipv4"],
        },
    ]


def _parse_ping(
    raw: str,
    *,
    interface: str,
    source_ipv4: str,
    target_ipv4: str,
    field: str,
) -> None:
    pattern = re.compile(
        rf"\APING {re.escape(target_ipv4)} "
        rf"\({re.escape(target_ipv4)}\) from "
        rf"{re.escape(source_ipv4)} {re.escape(interface)}: "
        rf"0\(28\) bytes of data\.\n\n"
        rf"--- {re.escape(target_ipv4)} ping statistics ---\n"
        rf"3 packets transmitted, 3 received, 0% packet loss, "
        rf"time [0-9]+ms\n\n\Z"
    )
    require(pattern.fullmatch(raw) is not None, f"E_CONNECTIVITY_OUTPUT: {field}")


def capture_connectivity(
    runner: Runner,
    phones: dict[str, dict[str, Any]],
    *,
    timeout: float,
    clock_ns: Callable[[], int],
) -> dict[str, Any]:
    specs = connectivity_specs(phones)
    started_ns = clock_ns()

    def capture(spec: dict[str, Any]) -> dict[str, Any]:
        return {
            "capture": run_capture(
                runner,
                spec["argv"],
                timeout,
                clock_ns,
                f"connectivity.{spec['edge_id']}",
            ),
            "edge_id": spec["edge_id"],
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(specs)) as pool:
        rows = list(pool.map(capture, specs))
    completed_ns = clock_ns()
    value = {
        "completed_ns": completed_ns,
        "edges": sorted(rows, key=lambda row: row["edge_id"]),
        "started_ns": started_ns,
    }
    validate_connectivity(value, phones)
    return value


def validate_connectivity(
    value: Any,
    phones: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    value = exact_keys(
        value,
        {"completed_ns", "edges", "started_ns"},
        "connectivity",
    )
    started_ns = integer(value["started_ns"], "connectivity.started_ns", 1)
    completed_ns = integer(
        value["completed_ns"],
        "connectivity.completed_ns",
        started_ns,
    )
    specs = {
        spec["edge_id"]: spec for spec in connectivity_specs(phones)
    }
    rows = value["edges"]
    require(
        type(rows) is list and len(rows) == len(specs),
        "E_CONNECTIVITY_EDGES",
    )
    edge_ids = []
    result = []
    for index, row in enumerate(rows):
        field = f"connectivity.edges[{index}]"
        row = exact_keys(row, {"capture", "edge_id"}, field)
        edge_id = text(row["edge_id"], f"{field}.edge_id")
        require(edge_id in specs, f"E_CONNECTIVITY_EDGE: {edge_id}")
        edge_ids.append(edge_id)
        spec = specs[edge_id]
        capture = _capture_record(
            row["capture"],
            spec["argv"],
            f"{field}.capture",
        )
        require(
            started_ns
            <= capture["started_ns"]
            <= capture["completed_ns"]
            <= completed_ns,
            f"E_CONNECTIVITY_INTERVAL: {edge_id}",
        )
        _parse_ping(
            capture["stdout"],
            interface=spec["interface"],
            source_ipv4=spec["source_ipv4"],
            target_ipv4=spec["target_ipv4"],
            field=edge_id,
        )
        result.append(
            {
                "edge_id": edge_id,
                "interface": spec["interface"],
                "source_ipv4": spec["source_ipv4"],
                "target_ipv4": spec["target_ipv4"],
            }
        )
    exact(edge_ids, sorted(specs), "connectivity.edge_order")
    return result


def run_capture(
    runner: Runner,
    argv: list[str],
    timeout: float,
    clock_ns: Callable[[], int],
    field: str,
) -> dict[str, Any]:
    started_ns = clock_ns()
    try:
        completed = runner.run(argv, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        raise DeploymentError(f"E_TIMEOUT: {field}") from error
    completed_ns = clock_ns()
    exact(completed.returncode, 0, f"E_EXIT: {field}")
    exact(completed.stderr, b"", f"E_STDERR: {field}")
    require(
        type(completed.stdout) is bytes
        and len(completed.stdout) <= MAX_STDOUT_BYTES,
        f"E_STDOUT: {field}",
    )
    try:
        stdout = completed.stdout.decode("ascii")
    except UnicodeDecodeError as error:
        raise DeploymentError(f"E_ASCII: {field}") from error
    require(started_ns <= completed_ns, f"E_INTERVAL: {field}")
    return {
        "argv": argv,
        "completed_ns": completed_ns,
        "started_ns": started_ns,
        "stdout": stdout,
    }


def monotonic_raw_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def validate_contract(value: Any) -> dict[str, Any]:
    require(type(value) is dict, "E_CONTRACT_TYPE")
    exact(
        sha256_bytes(canonical_bytes(value)),
        CONTRACT_SHA256,
        "contract.sha256",
    )
    exact(value.get("schema"), "s39-cp0-r1-evidence-contract-v2.4", "contract.schema")
    devices = exact_keys(value.get("devices"), {"cuda", "op12", "op15"}, "contract.devices")
    exact(
        devices["cuda"],
        {
            "host": CONTROLLER_HOST,
            "memory_total_bytes": CUDA_MEMORY_TOTAL_BYTES,
            "name": CUDA_NAME,
            "uuid": CUDA_UUID,
        },
        "contract.devices.cuda",
    )
    for phone, expected in EXPECTED_PHONES.items():
        exact(devices[phone], expected, f"contract.devices.{phone}")
    candidate_lock = value.get("candidate_lock")
    require(type(candidate_lock) is dict, "E_CONTRACT_CANDIDATE_LOCK")
    for key in ("bytes", "model_id", "sha256", "slot"):
        require(key in candidate_lock, f"E_CONTRACT_CANDIDATE_LOCK: {key}")
    quality_corpus = value.get("quality_corpus")
    require(type(quality_corpus) is dict, "E_CONTRACT_QUALITY_CORPUS")
    for key in ("bytes", "sha256"):
        require(key in quality_corpus, f"E_CONTRACT_QUALITY_CORPUS: {key}")
    protocol = exact_keys(
        value.get("token_history_protocol"),
        {
            "batch",
            "continuation_tokens_per_request",
            "decode_calls_after_prefill",
            "mechanics_item_indices",
            "n_batch",
            "n_ctx_seq",
            "n_ubatch",
            "prefill_chunking",
            "prefill_row_order",
            "prompt_hash_encoding",
            "quality_group_count",
            "quality_group_size",
            "quality_items",
            "request_id_mapping",
            "seq_id_mapping",
            "vocab_size",
        },
        "contract.token_history_protocol",
    )
    exact(
        protocol,
        {
            "batch": 8,
            "continuation_tokens_per_request": 8,
            "decode_calls_after_prefill": 7,
            "mechanics_item_indices": list(range(8)),
            "n_batch": 64,
            "n_ctx_seq": 512,
            "n_ubatch": 64,
            "prefill_chunking": "WHOLE_POSITION_WAVES_MAX_64_ROWS",
            "prefill_row_order": "POSITION_MAJOR_THEN_ITEM_INDEX",
            "prompt_hash_encoding": "UTF-8",
            "quality_group_count": 8,
            "quality_group_size": 8,
            "quality_items": 64,
            "request_id_mapping": "GROUP_LOCAL_ONE_BASED",
            "seq_id_mapping": "GROUP_LOCAL_ZERO_BASED",
            "vocab_size": 151936,
        },
        "contract.token_history_protocol",
    )
    return value


def _parse_controller(raw: str, field: str) -> dict[str, str]:
    values = parse_assignments(raw.encode("ascii"), field)
    exact(set(values), {"BOOT_ID", "HOST"}, f"{field}.keys")
    exact(values["HOST"], CONTROLLER_HOST, f"{field}.host")
    require(UUID_RE.fullmatch(values["BOOT_ID"]) is not None, f"E_BOOT_ID: {field}")
    return {"boot_id": values["BOOT_ID"], "host": values["HOST"]}


def _parse_gpu(raw: str) -> dict[str, Any]:
    lines = raw.splitlines()
    require(len(lines) == 1, "E_GPU_COUNT")
    fields = [item.strip() for item in lines[0].split(",")]
    require(len(fields) == 3, "E_GPU_FIELDS")
    name, uuid_value, memory_mib_text = fields
    exact(name, CUDA_NAME, "gpu.name")
    exact(uuid_value, CUDA_UUID, "gpu.uuid")
    try:
        memory_mib = int(memory_mib_text)
    except ValueError as error:
        raise DeploymentError("E_GPU_MEMORY") from error
    memory_bytes = memory_mib * 1024 * 1024
    exact(memory_bytes, CUDA_MEMORY_TOTAL_BYTES, "gpu.memory_total_bytes")
    return {
        "memory_total_bytes": memory_bytes,
        "name": name,
        "uuid": uuid_value,
    }


def _parse_self_ssh(raw: str) -> tuple[dict[str, str], dict[str, Any]]:
    lines = raw.splitlines()
    require(len(lines) == 3, "E_SELF_SSH_LINES")
    controller = _parse_controller("\n".join(lines[:2]) + "\n", "self_ssh")
    gpu = _parse_gpu(lines[2] + "\n")
    return controller, gpu


def _parse_path_resolution(raw: str) -> dict[str, str]:
    values = parse_assignments(raw.encode("ascii"), "path_resolution")
    exact(
        set(values),
        {
            "ADB",
            "ADB_REAL",
            "NVIDIA_SMI",
            "NVIDIA_SMI_REAL",
            "PYTHON",
            "PYTHON_REAL",
            "SSH",
        },
        "path_resolution.keys",
    )
    exact(values["ADB"], "/usr/bin/adb", "path_resolution.adb")
    exact(values["ADB_REAL"], ADB_PATH, "path_resolution.adb_real")
    exact(
        values["NVIDIA_SMI"],
        NVIDIA_SMI_PATH,
        "path_resolution.nvidia_smi",
    )
    exact(
        values["NVIDIA_SMI_REAL"],
        NVIDIA_SMI_PATH,
        "path_resolution.nvidia_smi_real",
    )
    exact(values["PYTHON"], "/usr/bin/python3", "path_resolution.python")
    exact(values["PYTHON_REAL"], PYTHON_PATH, "path_resolution.python_real")
    exact(values["SSH"], SSH_PATH, "path_resolution.ssh")
    return values


def _parse_adb_devices(raw: str) -> dict[str, dict[str, Any]]:
    lines = raw.splitlines()
    require(bool(lines) and lines[0] == "List of devices attached", "E_ADB_HEADER")
    result = {}
    for line in lines[1:]:
        if not line:
            continue
        fields = line.split()
        require(len(fields) >= 3, "E_ADB_ROW")
        selector, status, *attribute_fields = fields
        require(selector not in result, f"E_ADB_SELECTOR_REUSE: {selector}")
        exact(status, "device", f"adb.{selector}.status")
        attributes = {}
        for item in attribute_fields:
            key, separator, value = item.partition(":")
            require(
                separator == ":" and key and value and key not in attributes,
                f"E_ADB_ATTRIBUTE: {selector}",
            )
            attributes[key] = value
        result[selector] = {
            "attributes": attributes,
            "selector": selector,
            "status": status,
        }
    expected_serials = {
        value["serial"] for value in EXPECTED_PHONES.values()
    }
    usb_selectors = {
        selector for selector, row in result.items()
        if "usb" in row["attributes"]
    }
    exact(usb_selectors, expected_serials, "adb.usb_selectors")
    expected_by_serial = {
        value["serial"]: value
        for value in EXPECTED_PHONES.values()
    }
    for selector in sorted(usb_selectors):
        attributes = result[selector]["attributes"]
        exact(
            set(attributes),
            {"device", "model", "product", "transport_id", "usb"},
            f"adb.{selector}.attributes",
        )
        for key in ("device", "model", "product"):
            exact(
                attributes[key],
                expected_by_serial[selector][key],
                f"adb.{selector}.{key}",
            )
    wifi_selectors = set(result) - usb_selectors
    require(len(wifi_selectors) in {0, 2}, "E_ADB_WIFI_COUNT")
    for selector in wifi_selectors:
        attributes = result[selector]["attributes"]
        exact(
            set(attributes),
            {"device", "model", "product", "transport_id"},
            f"adb.{selector}.attributes",
        )
        host, separator, port = selector.rpartition(":")
        require(separator == ":" and port == "5555", f"E_ADB_WIFI_SELECTOR: {selector}")
        try:
            address = ipaddress.IPv4Address(host)
        except ipaddress.AddressValueError as error:
            raise DeploymentError(f"E_ADB_WIFI_SELECTOR: {selector}") from error
        require(
            not (
                address.is_unspecified
                or address.is_loopback
                or address.is_multicast
            ),
            f"E_ADB_WIFI_SELECTOR: {selector}",
        )
    exact(set(result), usb_selectors | wifi_selectors, "adb.selectors")
    return result


def _parse_phone(raw: str, selector: str) -> dict[str, Any]:
    values = parse_assignments(raw.encode("ascii"), f"phone.{selector}")
    exact(
        set(values),
        {
            "BOOT_ID",
            "DEVICE",
            "INTERFACE",
            "MODEL",
            "PHYSICAL_SERIAL",
            "PRODUCT",
            "WIFI_IPV4",
        },
        f"phone.{selector}.keys",
    )
    serial = values["PHYSICAL_SERIAL"]
    expected = next(
        (
            value for value in EXPECTED_PHONES.values()
            if value["serial"] == serial
        ),
        None,
    )
    require(expected is not None, f"E_PHONE_SERIAL: {selector}")
    for source, key in (
        ("DEVICE", "device"),
        ("MODEL", "model"),
        ("PRODUCT", "product"),
    ):
        exact(values[source], expected[key], f"phone.{selector}.{key}")
    require(UUID_RE.fullmatch(values["BOOT_ID"]) is not None, f"E_PHONE_BOOT: {selector}")
    exact(values["INTERFACE"], "wlan0", f"phone.{selector}.interface")
    try:
        address = ipaddress.IPv4Address(values["WIFI_IPV4"])
    except ipaddress.AddressValueError as error:
        raise DeploymentError(f"E_PHONE_IPV4: {selector}") from error
    require(
        not (
            address.is_unspecified
            or address.is_loopback
            or address.is_multicast
        ),
        f"E_PHONE_IPV4: {selector}",
    )
    return {
        "boot_id": values["BOOT_ID"],
        "device": values["DEVICE"],
        "interface": values["INTERFACE"],
        "model": values["MODEL"],
        "physical_serial": serial,
        "product": values["PRODUCT"],
        "wifi_ipv4": str(address),
    }


def _capture_record(value: Any, argv: list[str], field: str) -> dict[str, Any]:
    value = exact_keys(
        value,
        {"argv", "completed_ns", "started_ns", "stdout"},
        field,
    )
    exact(value["argv"], argv, f"{field}.argv")
    started = integer(value["started_ns"], f"{field}.started_ns", 1)
    completed = integer(value["completed_ns"], f"{field}.completed_ns", started)
    require(
        type(value["stdout"]) is str
        and bool(value["stdout"])
        and value["stdout"].isascii()
        and "\x00" not in value["stdout"]
        and len(value["stdout"].encode("ascii")) <= MAX_STDOUT_BYTES,
        f"E_STDOUT: {field}",
    )
    return value


def derive_topology(captures: Any) -> dict[str, Any]:
    captures = exact_keys(
        captures,
        {
            "adb_devices",
            "connectivity",
            "controller_after",
            "controller_before",
            "gpu",
            "path_resolution",
            "phones",
            "self_ssh",
        },
        "captures",
    )
    before = _capture_record(
        captures["controller_before"],
        CONTROLLER_COMMAND,
        "captures.controller_before",
    )
    after = _capture_record(
        captures["controller_after"],
        CONTROLLER_COMMAND,
        "captures.controller_after",
    )
    gpu_capture = _capture_record(captures["gpu"], GPU_COMMAND, "captures.gpu")
    path_capture = _capture_record(
        captures["path_resolution"],
        PATH_RESOLUTION_COMMAND,
        "captures.path_resolution",
    )
    ssh_capture = _capture_record(
        captures["self_ssh"],
        SELF_SSH_COMMAND,
        "captures.self_ssh",
    )
    adb_capture = _capture_record(
        captures["adb_devices"],
        ADB_DEVICES_COMMAND,
        "captures.adb_devices",
    )
    controller_before = _parse_controller(
        before["stdout"],
        "controller_before",
    )
    controller_after = _parse_controller(
        after["stdout"],
        "controller_after",
    )
    exact(controller_after, controller_before, "E_CONTROLLER_IDENTITY_CHANGED")
    gpu = _parse_gpu(gpu_capture["stdout"])
    path_resolution = _parse_path_resolution(path_capture["stdout"])
    ssh_controller, ssh_gpu = _parse_self_ssh(ssh_capture["stdout"])
    exact(ssh_controller, controller_before, "E_SELF_SSH_CONTROLLER")
    exact(ssh_gpu, gpu, "E_SELF_SSH_GPU")
    adb_rows = _parse_adb_devices(adb_capture["stdout"])
    phone_captures = captures["phones"]
    require(
        type(phone_captures) is list
        and len(phone_captures) == len(adb_rows),
        "E_PHONE_CAPTURE_COUNT",
    )
    captured_selectors = []
    identities = {}
    for index, item in enumerate(phone_captures):
        item = exact_keys(item, {"capture", "selector"}, f"captures.phones[{index}]")
        selector = text(item["selector"], f"captures.phones[{index}].selector")
        require(selector in adb_rows, f"E_PHONE_SELECTOR: {selector}")
        require(selector not in captured_selectors, f"E_PHONE_CAPTURE_REUSE: {selector}")
        capture = _capture_record(
            item["capture"],
            phone_command(selector),
            f"captures.phones[{index}].capture",
        )
        identities[selector] = _parse_phone(capture["stdout"], selector)
        captured_selectors.append(selector)
    exact(captured_selectors, sorted(adb_rows), "E_PHONE_CAPTURE_ORDER")

    for phone, expected in sorted(EXPECTED_PHONES.items()):
        exact(
            identities[expected["serial"]]["physical_serial"],
            expected["serial"],
            f"E_PHONE_USB_SERIAL: {phone}",
        )
    for selector, identity in identities.items():
        for key in ("device", "model", "product"):
            exact(
                adb_rows[selector]["attributes"][key],
                identity[key],
                f"E_ADB_SHELL_IDENTITY: {selector}.{key}",
            )

    phones = {}
    wifi_seen = set()
    for phone, expected in sorted(EXPECTED_PHONES.items()):
        serial = expected["serial"]
        usb = identities[serial]
        wifi_matches = [
            (selector, identity)
            for selector, identity in identities.items()
            if selector != serial and identity["physical_serial"] == serial
        ]
        require(len(wifi_matches) in {0, 1}, f"E_PHONE_WIFI_MATCH: {phone}")
        wifi_selector = None
        if wifi_matches:
            wifi_selector, wifi = wifi_matches[0]
            exact(wifi, usb, f"E_PHONE_USB_WIFI_IDENTITY: {phone}")
            exact(
                wifi_selector,
                f"{usb['wifi_ipv4']}:5555",
                f"E_PHONE_WIFI_SELECTOR: {phone}",
            )
            wifi_seen.add(wifi_selector)
        phones[phone] = {
            **usb,
            "usb_selector": serial,
            "wifi_selector": wifi_selector,
        }
    expected_wifi = set(adb_rows) - {
        value["serial"] for value in EXPECTED_PHONES.values()
    }
    exact(wifi_seen, expected_wifi, "E_ADB_WIFI_IDENTITY")
    require(
        len({value["boot_id"] for value in phones.values()}) == 2,
        "E_PHONE_BOOT_REUSE",
    )
    require(
        len({value["wifi_ipv4"] for value in phones.values()}) == 2,
        "E_PHONE_IPV4_REUSE",
    )
    connectivity = validate_connectivity(captures["connectivity"], phones)
    return {
        "adb": {
            "host": ADB_HOST,
            "port": ADB_PORT,
            "selectors": [
                adb_rows[selector] for selector in sorted(adb_rows)
            ],
        },
        "controller": controller_before,
        "cuda": {
            "boot_id": controller_before["boot_id"],
            "host": controller_before["host"],
            **gpu,
            "self_ssh_target": CUDA_SSH_TARGET,
        },
        "path_resolution": path_resolution,
        "phones": phones,
        "connectivity": connectivity,
    }


def validate_topology(
    value: Any,
    contract: dict[str, Any],
    *,
    now_ns: int | None = None,
    require_fresh: bool = False,
) -> dict[str, Any]:
    validate_contract(contract)
    value = exact_keys(
        value,
        {
            "captures",
            "completed_ns",
            "observed",
            "schema",
            "started_ns",
            "status",
        },
        "topology",
    )
    exact(value["schema"], TOPOLOGY_SCHEMA, "topology.schema")
    exact(value["status"], TOPOLOGY_STATUS, "topology.status")
    started = integer(value["started_ns"], "topology.started_ns", 1)
    completed = integer(value["completed_ns"], "topology.completed_ns", started)
    observed = derive_topology(value["captures"])
    exact(value["observed"], observed, "topology.observed")
    before_connectivity = [
        value["captures"]["controller_before"],
        value["captures"]["gpu"],
        value["captures"]["path_resolution"],
        value["captures"]["self_ssh"],
        value["captures"]["adb_devices"],
        *[item["capture"] for item in value["captures"]["phones"]],
    ]
    cursor = started
    for index, item in enumerate(before_connectivity):
        require(
            cursor <= item["started_ns"] <= item["completed_ns"] <= completed,
            f"E_TOPOLOGY_INTERVAL: {index}",
        )
        cursor = item["completed_ns"]
    connectivity = value["captures"]["connectivity"]
    require(
        cursor
        <= connectivity["started_ns"]
        <= connectivity["completed_ns"]
        <= completed,
        "E_TOPOLOGY_CONNECTIVITY_INTERVAL",
    )
    after = value["captures"]["controller_after"]
    require(
        connectivity["completed_ns"]
        <= after["started_ns"]
        <= after["completed_ns"]
        <= completed,
        "E_TOPOLOGY_AFTER_INTERVAL",
    )
    if require_fresh:
        now = integer(
            monotonic_raw_ns() if now_ns is None else now_ns,
            "topology.now_ns",
            completed,
        )
        require(
            now - completed <= TOPOLOGY_MAXIMUM_AGE_NS,
            "E_TOPOLOGY_STALE",
        )
    return observed


def capture_topology(
    contract: dict[str, Any],
    *,
    runner: Runner | None = None,
    clock_ns: Callable[[], int] = monotonic_raw_ns,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    validate_contract(contract)
    require(
        type(timeout_seconds) is int and 1 <= timeout_seconds <= 120,
        "E_TIMEOUT_SECONDS",
    )
    runner = runner or SubprocessRunner()
    started_ns = clock_ns()
    captures = {
        "controller_before": run_capture(
            runner,
            CONTROLLER_COMMAND,
            timeout_seconds,
            clock_ns,
            "controller_before",
        ),
        "gpu": run_capture(
            runner,
            GPU_COMMAND,
            timeout_seconds,
            clock_ns,
            "gpu",
        ),
        "path_resolution": run_capture(
            runner,
            PATH_RESOLUTION_COMMAND,
            timeout_seconds,
            clock_ns,
            "path_resolution",
        ),
        "self_ssh": run_capture(
            runner,
            SELF_SSH_COMMAND,
            timeout_seconds,
            clock_ns,
            "self_ssh",
        ),
        "adb_devices": run_capture(
            runner,
            ADB_DEVICES_COMMAND,
            timeout_seconds,
            clock_ns,
            "adb_devices",
        ),
    }
    adb_rows = _parse_adb_devices(captures["adb_devices"]["stdout"])
    captures["phones"] = [
        {
            "capture": run_capture(
                runner,
                phone_command(selector),
                timeout_seconds,
                clock_ns,
                f"phone.{selector}",
            ),
            "selector": selector,
        }
        for selector in sorted(adb_rows)
    ]
    probe_identities = {
        item["selector"]: _parse_phone(
            item["capture"]["stdout"],
            item["selector"],
        )
        for item in captures["phones"]
    }
    for phone, expected in sorted(EXPECTED_PHONES.items()):
        exact(
            probe_identities[expected["serial"]]["physical_serial"],
            expected["serial"],
            f"E_PHONE_USB_SERIAL: {phone}",
        )
    probe_phones = {
        phone: {
            **probe_identities[expected["serial"]],
            "usb_selector": expected["serial"],
        }
        for phone, expected in sorted(EXPECTED_PHONES.items())
    }
    captures["connectivity"] = capture_connectivity(
        runner,
        probe_phones,
        timeout=timeout_seconds,
        clock_ns=clock_ns,
    )
    captures["controller_after"] = run_capture(
        runner,
        CONTROLLER_COMMAND,
        timeout_seconds,
        clock_ns,
        "controller_after",
    )
    completed_ns = clock_ns()
    value = {
        "captures": captures,
        "completed_ns": completed_ns,
        "observed": derive_topology(captures),
        "schema": TOPOLOGY_SCHEMA,
        "started_ns": started_ns,
        "status": TOPOLOGY_STATUS,
    }
    validate_topology(
        value,
        contract,
        now_ns=clock_ns(),
        require_fresh=True,
    )
    return value


def _validate_pin(
    value: Any,
    name: str,
    after_file_read: Callable[[str], None] | None,
) -> tuple[
    Path,
    os.stat_result,
    str,
    dict[str, Any] | None,
    bytes | None,
]:
    value = exact_keys(
        value,
        {"bytes", "path", "sha256", "stat"},
        f"files.{name}",
    )
    path = Path(text(value["path"], f"files.{name}.path"))
    require(path.is_absolute(), f"E_PATH_ABSOLUTE: files.{name}")
    retain = name in RETAIN_FILE_KEYS
    actual_sha256, actual_bytes, metadata, raw = secure_hash(
        path,
        f"files.{name}",
        retain=retain,
        after_read=(
            (lambda: after_file_read(name))
            if after_file_read is not None
            else None
        ),
    )
    exact(
        integer(value["bytes"], f"files.{name}.bytes", 1),
        actual_bytes,
        f"files.{name}.bytes",
    )
    exact(
        digest(value["sha256"], f"files.{name}.sha256"),
        actual_sha256,
        f"files.{name}.sha256",
    )
    exact_keys(
        value["stat"],
        {"ctime_ns", "device_id", "inode", "mode", "mtime_ns", "size"},
        f"files.{name}.stat",
    )
    exact(value["stat"], stat_record(metadata), f"files.{name}.stat")
    if name in EXECUTABLE_FILE_KEYS:
        require(
            stat.S_IMODE(metadata.st_mode) & 0o111 != 0,
            f"E_EXECUTABLE: files.{name}",
        )
    expected_schema = JSON_SCHEMAS.get(name)
    parsed = None
    if expected_schema is not None:
        require(raw is not None, f"E_INTERNAL_RETAIN: files.{name}")
        parsed = parse_json(raw, f"files.{name}")
        require(type(parsed) is dict, f"E_TYPE: files.{name}")
        exact(canonical_bytes(parsed), raw, f"E_CANONICAL: files.{name}")
        exact(parsed.get("schema"), expected_schema, f"files.{name}.schema")
        if name == "contract":
            validate_contract(parsed)
    return path, metadata, actual_sha256, parsed, raw


def _absolute_path(value: Any, field: str) -> Path:
    path = Path(text(value, field))
    require(path.is_absolute() and ".." not in path.parts, f"E_PATH: {field}")
    return path


def _component_stat(value: Any, field: str) -> dict[str, int]:
    value = exact_keys(
        value,
        {"ctime_ns", "device_id", "inode", "mode", "mtime_ns", "size"},
        field,
    )
    for key, item in value.items():
        integer(item, f"{field}.{key}")
    require(stat.S_ISREG(value["mode"]), f"E_COMPONENT_MODE: {field}")
    return value


def validate_runtime_inventory(
    value: Any,
    *,
    cuda_launchers: dict[str, str],
) -> dict[str, Any]:
    value = exact_keys(
        value,
        {
            "bundle_roots",
            "bundles",
            "closure_complete",
            "components",
            "schema",
        },
        "runtime_inventory",
    )
    exact(
        value["schema"],
        "s39-cp0-r1-runtime-bundle-closure-input-v1",
        "runtime_inventory.schema",
    )
    exact(value["closure_complete"], True, "runtime_inventory.closure_complete")
    roots = exact_keys(
        value["bundle_roots"],
        RUNTIME_BUNDLE_IDS,
        "runtime_inventory.bundle_roots",
    )
    root_paths = {
        name: _absolute_path(path, f"runtime_inventory.bundle_roots.{name}")
        for name, path in roots.items()
    }
    for left, left_path in root_paths.items():
        for right, right_path in root_paths.items():
            if left != right:
                require(
                    left_path not in right_path.parents,
                    f"E_RUNTIME_ROOT_OVERLAP: {left}:{right}",
                )

    require(type(value["bundles"]) is list, "E_RUNTIME_BUNDLES")
    bundles = {}
    for index, item in enumerate(value["bundles"]):
        field = f"runtime_inventory.bundles[{index}]"
        item = exact_keys(
            item,
            {
                "bundle_id",
                "endpoint",
                "launcher_component_id",
                "process_role",
                "required_component_ids",
            },
            field,
        )
        bundle_id = text(item["bundle_id"], f"{field}.bundle_id")
        require(
            bundle_id in RUNTIME_BUNDLE_IDS and bundle_id not in bundles,
            f"E_RUNTIME_BUNDLE_ID: {bundle_id}",
        )
        expected_endpoint = (
            "cuda"
            if bundle_id.startswith("cuda_")
            else "op12"
            if bundle_id.startswith("op12_")
            else "op15"
        )
        exact(item["endpoint"], expected_endpoint, f"{field}.endpoint")
        exact(item["process_role"], bundle_id, f"{field}.process_role")
        required = item["required_component_ids"]
        require(
            type(required) is list
            and len(required) >= 2,
            f"E_RUNTIME_REQUIRED: {bundle_id}",
        )
        required_ids = [
            text(component_id, f"{field}.required_component_ids[{index}]")
            for index, component_id in enumerate(required)
        ]
        require(
            required_ids == sorted(set(required_ids))
            and item["launcher_component_id"] in required_ids,
            f"E_RUNTIME_REQUIRED: {bundle_id}",
        )
        bundles[bundle_id] = item
    exact(set(bundles), RUNTIME_BUNDLE_IDS, "runtime_inventory.bundle_ids")

    require(
        type(value["components"]) is list and bool(value["components"]),
        "E_RUNTIME_COMPONENTS",
    )
    components = {}
    endpoint_paths = {"cuda": set(), "op12": set(), "op15": set()}
    for index, item in enumerate(value["components"]):
        field = f"runtime_inventory.components[{index}]"
        item = exact_keys(
            item,
            {
                "bundle_id",
                "bytes",
                "component_id",
                "endpoint",
                "path",
                "role",
                "sha256",
                "stat",
            },
            field,
        )
        component_id = text(item["component_id"], f"{field}.component_id")
        require(
            component_id not in components,
            f"E_RUNTIME_COMPONENT_REUSE: {component_id}",
        )
        bundle_id = text(item["bundle_id"], f"{field}.bundle_id")
        require(bundle_id in bundles, f"E_RUNTIME_COMPONENT_BUNDLE: {component_id}")
        endpoint = text(item["endpoint"], f"{field}.endpoint")
        exact(
            endpoint,
            bundles[bundle_id]["endpoint"],
            f"{field}.endpoint",
        )
        path = _absolute_path(item["path"], f"{field}.path")
        try:
            path.relative_to(root_paths[bundle_id])
        except ValueError as error:
            raise DeploymentError(
                f"E_RUNTIME_ROOT_ESCAPE: {component_id}"
            ) from error
        require(
            str(path) not in endpoint_paths[endpoint],
            f"E_RUNTIME_PATH_REUSE: {path}",
        )
        endpoint_paths[endpoint].add(str(path))
        byte_count = integer(item["bytes"], f"{field}.bytes", 1)
        digest(item["sha256"], f"{field}.sha256")
        metadata = _component_stat(item["stat"], f"{field}.stat")
        exact(metadata["size"], byte_count, f"{field}.stat.size")
        role = text(item["role"], f"{field}.role")
        require(
            role in {
                "backend_library",
                "executable",
                "shared_library",
            },
            f"E_RUNTIME_ROLE: {component_id}",
        )
        if endpoint == "cuda":
            actual_sha256, actual_bytes, actual_stat, _ = secure_hash(
                path,
                field,
                retain=False,
            )
            exact(actual_bytes, byte_count, f"{field}.bytes")
            exact(actual_sha256, item["sha256"], f"{field}.sha256")
            exact(stat_record(actual_stat), metadata, f"{field}.stat")
        components[component_id] = item

    referenced = []
    for bundle_id, bundle in bundles.items():
        required = bundle["required_component_ids"]
        require(
            all(component_id in components for component_id in required),
            f"E_RUNTIME_DEPENDENCY: {bundle_id}",
        )
        require(
            all(
                components[component_id]["bundle_id"] == bundle_id
                for component_id in required
            ),
            f"E_RUNTIME_OWNER: {bundle_id}",
        )
        launcher = components[bundle["launcher_component_id"]]
        exact(launcher["role"], "executable", f"{bundle_id}.launcher.role")
        if bundle_id in cuda_launchers:
            exact(
                launcher["path"],
                cuda_launchers[bundle_id],
                f"{bundle_id}.launcher.path",
            )
        require(
            any(
                components[component_id]["role"]
                in {"backend_library", "shared_library"}
                for component_id in required
            ),
            f"E_RUNTIME_LIBRARY: {bundle_id}",
        )
        referenced.extend(required)
    exact(sorted(referenced), sorted(components), "runtime_inventory.ownership")
    return {
        "bundle_roots": {
            name: str(root_paths[name]) for name in sorted(root_paths)
        },
        "bundles": [bundles[name] for name in sorted(bundles)],
        "components": [
            components[name] for name in sorted(components)
        ],
    }


def _option(argv: Any, option: str, expected: str, field: str) -> None:
    require(
        type(argv) is list
        and argv.count(option) == 1
        and argv.index(option) + 1 < len(argv),
        f"E_OPTION: {field}.{option}",
    )
    exact(argv[argv.index(option) + 1], expected, f"{field}.{option}")


def validate_monolithic_launch(
    value: Any,
    *,
    model_pin: dict[str, Any],
    runtime_path: str,
    route_epoch: int,
    port: int,
) -> dict[str, Any]:
    value = exact_keys(
        value,
        {
            "allowed_system_roots",
            "bundle_id",
            "bundle_root",
            "bundle_sha256",
            "command",
            "cwd",
            "endpoint",
            "env",
            "expected_capabilities",
            "expected_file_type",
            "expected_max_streams",
            "expected_n_batch",
            "expected_n_ctx_seq",
            "expected_n_embd",
            "expected_n_layer",
            "expected_n_ubatch",
            "host",
            "io_timeout_ms",
            "launcher_component_id",
            "model_artifact",
            "model_id",
            "model_sha256",
            "port",
            "required_components",
            "route_epoch",
            "schema",
            "shutdown_timeout_ms",
            "startup_timeout_ms",
        },
        "cuda_monolithic_launch",
    )
    for key, expected in (
        ("bundle_id", "cuda_monolithic"),
        ("endpoint", "cuda"),
        ("expected_capabilities", 0x3F),
        ("expected_file_type", 15),
        ("expected_max_streams", 8),
        ("expected_n_batch", 64),
        ("expected_n_ctx_seq", 512),
        ("expected_n_embd", 5120),
        ("expected_n_layer", 40),
        ("expected_n_ubatch", 64),
        ("host", "127.0.0.1"),
        ("model_id", MODEL_ID),
        ("model_sha256", model_pin["sha256"]),
        ("port", port),
        ("route_epoch", route_epoch),
        ("schema", "s39-cp0-r1-v24-cuda-monolithic-launch-v1"),
        ("io_timeout_ms", 300000),
        ("shutdown_timeout_ms", 30000),
        ("startup_timeout_ms", 300000),
    ):
        exact(value[key], expected, f"cuda_monolithic_launch.{key}")
    bundle_root = _absolute_path(
        value["bundle_root"],
        "cuda_monolithic_launch.bundle_root",
    )
    no_symlink_chain(bundle_root, "cuda_monolithic_launch.bundle_root")
    require(bundle_root.is_dir(), "E_MONOLITHIC_BUNDLE_ROOT")
    cwd = _absolute_path(value["cwd"], "cuda_monolithic_launch.cwd")
    no_symlink_chain(cwd, "cuda_monolithic_launch.cwd")
    require(cwd.is_dir(), "E_MONOLITHIC_CWD")
    roots = value["allowed_system_roots"]
    require(
        type(roots) is list
        and bool(roots)
        and roots == sorted(set(roots))
        and all(
            type(root) is str
            and root.endswith("/")
            and Path(root).is_absolute()
            and ".." not in Path(root).parts
            for root in roots
        ),
        "E_MONOLITHIC_SYSTEM_ROOTS",
    )
    command = value["command"]
    exact(
        command,
        [
            runtime_path,
            "-m",
            model_pin["path"],
            "--mode",
            "monov3",
            "--port",
            str(port),
            "--devices",
            "CUDA0",
            "--driver-batch",
            "8",
            "--driver-context",
            "512",
            "--driver-max-prefill",
            "8",
        ],
        "cuda_monolithic_launch.command",
    )
    model = exact_keys(
        value["model_artifact"],
        {"path", "sha256", "stat"},
        "cuda_monolithic_launch.model_artifact",
    )
    exact(model["path"], model_pin["path"], "cuda_monolithic_launch.model.path")
    exact(
        model["sha256"],
        model_pin["sha256"],
        "cuda_monolithic_launch.model.sha256",
    )
    exact(
        model["stat"],
        model_pin["stat"],
        "cuda_monolithic_launch.model.stat",
    )
    exact(
        value["env"],
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "HOME": "/home/zhihao",
            "LAYERSPLIT_MEMORY_CERT": "1",
            "LAYERSPLIT_MODEL_SHA256": model_pin["sha256"],
            "LAYERSPLIT_PLACEMENT_CERT": "1",
            "LC_ALL": "C",
            "LD_LIBRARY_PATH": str(bundle_root),
        },
        "cuda_monolithic_launch.env",
    )
    components = value["required_components"]
    require(type(components) is list and bool(components), "E_MONOLITHIC_COMPONENTS")
    by_id = {}
    for index, component in enumerate(components):
        field = f"cuda_monolithic_launch.required_components[{index}]"
        component = exact_keys(
            component,
            {"component_id", "path", "sha256", "stat"},
            field,
        )
        component_id = text(component["component_id"], f"{field}.component_id")
        require(component_id not in by_id, f"E_MONOLITHIC_COMPONENT_REUSE: {component_id}")
        path = _absolute_path(component["path"], f"{field}.path")
        try:
            path.relative_to(bundle_root)
        except ValueError as error:
            raise DeploymentError(
                f"E_MONOLITHIC_ROOT_ESCAPE: {component_id}"
            ) from error
        actual_sha256, actual_bytes, actual_stat, _ = secure_hash(
            path,
            field,
            retain=False,
        )
        exact(actual_sha256, component["sha256"], f"{field}.sha256")
        exact(stat_record(actual_stat), component["stat"], f"{field}.stat")
        exact(actual_bytes, component["stat"]["size"], f"{field}.bytes")
        by_id[component_id] = component
    exact(
        list(by_id),
        sorted(by_id),
        "cuda_monolithic_launch.component_order",
    )
    launcher_id = text(
        value["launcher_component_id"],
        "cuda_monolithic_launch.launcher_component_id",
    )
    require(launcher_id in by_id, "E_MONOLITHIC_LAUNCHER")
    exact(
        by_id[launcher_id]["path"],
        runtime_path,
        "cuda_monolithic_launch.launcher.path",
    )
    exact(
        value["bundle_sha256"],
        sha256_bytes(
            canonical_bytes(
                {
                    "bundle_id": "cuda_monolithic",
                    "components": components,
                    "endpoint": "cuda",
                    "launcher_component_id": launcher_id,
                    "process_role": "cuda_monolithic",
                    "schema": "s39-cp0-r1-runtime-bundle-root-identity-v2.4",
                }
            )
        ),
        "cuda_monolithic_launch.bundle_sha256",
    )
    return {
        "bundle_root": value["bundle_root"],
        "components": by_id,
        "launcher_component_id": launcher_id,
    }


def validate_token_history(
    value: Any,
    *,
    contract: dict[str, Any],
    candidate_sha256: str,
    model_sha256: str,
    corpus_sha256: str,
    tokenizer_plan_sha256: str,
    tokenizer_sha256: str,
    candidate: dict[str, Any],
    corpus_raw: bytes,
    history_raw: bytes,
    tokenizer_plan: dict[str, Any],
) -> None:
    value = exact_keys(
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
        "token_history",
    )
    exact(value["schema"], "s39-cp0-r1-token-history-v2.4", "token_history.schema")
    exact(value["model_id"], MODEL_ID, "token_history.model_id")
    exact(
        value["candidate_sha256"],
        candidate_sha256,
        "token_history.candidate_sha256",
    )
    exact(
        value["model_sha256"],
        model_sha256,
        "token_history.model_sha256",
    )
    exact(
        value["corpus_sha256"],
        corpus_sha256,
        "token_history.corpus_sha256",
    )
    protocol = contract["token_history_protocol"]
    for key in (
        "batch",
        "continuation_tokens_per_request",
        "n_batch",
        "n_ctx_seq",
        "n_ubatch",
        "prefill_chunking",
        "prefill_row_order",
    ):
        exact(value[key], protocol[key], f"token_history.{key}")
    plan = exact_keys(
        tokenizer_plan,
        {
            "command_template",
            "component_id",
            "cwd",
            "environment",
            "executable",
            "model",
            "protocol",
            "schema",
            "timeout_seconds",
        },
        "tokenizer_plan",
    )
    exact(
        plan["schema"],
        "s39-cp0-r1-a-only-tokenizer-plan-v2",
        "tokenizer_plan.schema",
    )
    exact(plan["component_id"], "cuda-tokenize", "tokenizer_plan.component_id")
    plan_executable = exact_keys(
        plan["executable"],
        {"bytes", "path", "sha256"},
        "tokenizer_plan.executable",
    )
    plan_model = exact_keys(
        plan["model"],
        {"bytes", "model_id", "path", "sha256", "vocab_size"},
        "tokenizer_plan.model",
    )
    exact(
        plan["command_template"],
        [
            plan_executable["path"],
            "-m",
            plan_model["path"],
            "--ids",
            "-f",
            "{PROMPT_FILE}",
            "--log-disable",
        ],
        "tokenizer_plan.command_template",
    )
    exact(
        plan["cwd"],
        str(Path(plan_executable["path"]).parent),
        "tokenizer_plan.cwd",
    )
    exact(
        plan["environment"],
        {"LC_ALL": "C", "LD_LIBRARY_PATH": plan["cwd"]},
        "tokenizer_plan.environment",
    )
    exact(
        plan["protocol"],
        {
            "add_bos": "MODEL_DEFAULT",
            "escape": True,
            "output_format": "BRACKETED_DECIMAL_IDS",
            "parse_special": True,
            "prompt_file_placeholder": "{PROMPT_FILE}",
        },
        "tokenizer_plan.protocol",
    )
    exact(plan["timeout_seconds"], 300, "tokenizer_plan.timeout_seconds")

    candidate = exact_keys(
        candidate,
        {
            "candidate_attempt",
            "candidate_attempt_limit",
            "contract_sha256",
            "historical_routes",
            "models",
            "schema",
            "status",
            "task_suite",
        },
        "candidate",
    )
    task_suite = candidate["task_suite"]
    require(type(task_suite) is dict, "E_CANDIDATE_TASK_SUITE")
    prompt_format = task_suite.get("prompt_format")
    require(
        type(prompt_format) is str
        and bool(prompt_format)
        and prompt_format.isascii()
        and "\x00" not in prompt_format,
        "E_CANDIDATE_PROMPT_FORMAT",
    )

    corpus_lines = corpus_raw.splitlines(keepends=True)
    require(
        len(corpus_lines) == protocol["quality_items"]
        and all(line.endswith(b"\n") for line in corpus_lines),
        "E_HISTORY_CORPUS_ROWS",
    )
    corpus = []
    for index, line in enumerate(corpus_lines):
        item = parse_json(line, f"quality_corpus[{index}]")
        item = exact_keys(
            item,
            {
                "choices",
                "dataset",
                "dataset_revision",
                "expected_answer",
                "item_index",
                "question",
                "source_row",
                "subject",
            },
            f"quality_corpus[{index}]",
        )
        exact(canonical_bytes(item), line, f"quality_corpus[{index}].canonical")
        exact(item["item_index"], index, f"quality_corpus[{index}].item_index")
        require(
            type(item["choices"]) is list
            and len(item["choices"]) == 4
            and all(type(choice) is str for choice in item["choices"]),
            f"E_HISTORY_CHOICES: {index}",
        )
        require(type(item["question"]) is str, f"E_HISTORY_QUESTION: {index}")
        corpus.append(item)

    requests_value = value["requests"]
    require(
        type(requests_value) is list
        and len(requests_value) == protocol["quality_items"],
        "E_HISTORY_REQUESTS",
    )
    requests = {}
    for index, request in enumerate(requests_value):
        field = f"token_history.requests[{index}]"
        request = exact_keys(
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
        exact(request["item_index"], index, f"{field}.item_index")
        exact(
            request["request_id"],
            index % protocol["batch"] + 1,
            f"{field}.request_id",
        )
        exact(
            request["seq_id"],
            index % protocol["batch"],
            f"{field}.seq_id",
        )
        item = corpus[index]
        try:
            prompt = prompt_format.format(
                question=item["question"],
                choice0=item["choices"][0],
                choice1=item["choices"][1],
                choice2=item["choices"][2],
                choice3=item["choices"][3],
            )
        except (IndexError, KeyError, ValueError) as error:
            raise DeploymentError(f"E_HISTORY_PROMPT: {index}") from error
        prompt_raw = prompt.encode(
            protocol["prompt_hash_encoding"].lower()
        )
        exact(
            request["prompt_sha256"],
            sha256_bytes(prompt_raw),
            f"{field}.prompt_sha256",
        )
        exact(
            request["prompt_utf8_bytes"],
            len(prompt_raw),
            f"{field}.prompt_utf8_bytes",
        )
        exact(
            request["prompt_utf8_base64"],
            base64.b64encode(prompt_raw).decode("ascii"),
            f"{field}.prompt_utf8_base64",
        )
        tokens = request["token_ids"]
        require(
            type(tokens) is list
            and 0 < len(tokens)
            <= protocol["n_ctx_seq"] - protocol["continuation_tokens_per_request"],
            f"E_HISTORY_TOKENS: {index}",
        )
        for position, token_id in enumerate(tokens):
            require(
                integer(token_id, f"{field}.token_ids[{position}]")
                < protocol["vocab_size"],
                f"E_HISTORY_TOKEN_RANGE: {index}:{position}",
            )
        requests[index] = request

    def expected_group(group_index: int) -> dict[str, Any]:
        item_indices = list(
            range(
                group_index * protocol["quality_group_size"],
                (group_index + 1) * protocol["quality_group_size"],
            )
        )
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
        waves = [
            [row for row in token_rows if row[0] == position]
            for position in sorted({row[0] for row in token_rows})
        ]
        partitions = []
        current = []
        for wave in waves:
            require(
                len(wave) <= protocol["n_ubatch"],
                "E_HISTORY_POSITION_WAVE",
            )
            if current and len(current) + len(wave) > protocol["n_ubatch"]:
                partitions.append(current)
                current = []
            current.extend(wave)
        if current:
            partitions.append(current)
        prefill = [
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
                    for position, item_index, request_id, seq_id, token_id
                    in partition
                ],
            }
            for call_index, partition in enumerate(partitions)
        ]
        decode = [
            {
                "call_index": len(prefill) + call_index,
                "continuation_input_ordinal": call_index,
                "continuation_output_ordinal": call_index + 1,
                "rows": [
                    {
                        "item_index": item_index,
                        "position": (
                            len(requests[item_index]["token_ids"]) + call_index
                        ),
                        "request_id": requests[item_index]["request_id"],
                        "seq_id": requests[item_index]["seq_id"],
                    }
                    for item_index in item_indices
                ],
            }
            for call_index in range(protocol["decode_calls_after_prefill"])
        ]
        return {
            "decode_calls": decode,
            "group_index": group_index,
            "item_indices": item_indices,
            "prefill_partitions": prefill,
        }

    expected_groups = [
        expected_group(index)
        for index in range(protocol["quality_group_count"])
    ]
    exact(
        value["quality_groups"],
        expected_groups,
        "token_history.quality_groups",
    )
    exact(
        value["mechanics_b8"],
        expected_groups[0],
        "token_history.mechanics_b8",
    )
    tokenizer = exact_keys(
        value["tokenizer"],
        {"component_id", "path", "plan_sha256", "sha256"},
        "token_history.tokenizer",
    )
    exact(
        tokenizer["plan_sha256"],
        tokenizer_plan_sha256,
        "token_history.tokenizer.plan_sha256",
    )
    exact(
        tokenizer["sha256"],
        tokenizer_sha256,
        "token_history.tokenizer.sha256",
    )
    exact(
        tokenizer["component_id"],
        plan["component_id"],
        "token_history.tokenizer.component_id",
    )
    exact(
        tokenizer["path"],
        plan_executable["path"],
        "token_history.tokenizer.path",
    )
    exact(canonical_bytes(value), history_raw, "token_history.canonical")


def validate_operator_input(
    value: Any,
    *,
    after_file_read: Callable[[str], None] | None = None,
    port_available: Callable[[int], bool] | None = None,
) -> dict[str, Any]:
    value = exact_keys(
        value,
        {
            "controller_host",
            "directories",
            "files",
            "model_id",
            "phase",
            "ports",
            "route_epoch",
            "schema",
            "topology",
        },
        "operator_input",
    )
    exact(value["schema"], OPERATOR_INPUT_SCHEMA, "operator_input.schema")
    exact(value["controller_host"], CONTROLLER_HOST, "operator_input.controller_host")
    exact(value["phase"], PHASE, "operator_input.phase")
    exact(value["model_id"], MODEL_ID, "operator_input.model_id")
    integer(value["route_epoch"], "operator_input.route_epoch", 1)
    exact(
        value["topology"],
        {
            "adb_host": ADB_HOST,
            "adb_port": ADB_PORT,
            "cuda_uuid": CUDA_UUID,
            "physical_serials": {
                phone: expected["serial"]
                for phone, expected in sorted(EXPECTED_PHONES.items())
            },
        },
        "operator_input.topology",
    )
    ports = exact_keys(value["ports"], PORT_KEYS, "operator_input.ports")
    for name, port in ports.items():
        integer(port, f"operator_input.ports.{name}", 1)
        require(port <= 65535, f"E_PORT_RANGE: {name}")
    exact(ports["adb_server"], ADB_PORT, "operator_input.ports.adb_server")
    exact(
        ports["cuda_monolithic"],
        CUDA_MONOLITHIC_PORT,
        "operator_input.ports.cuda_monolithic",
    )
    require(len(set(ports.values())) == len(ports), "E_PORT_REUSE")
    if port_available is None:
        def port_available(port: int) -> bool:
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
                probe.bind(("127.0.0.1", port))
                return True
            except OSError:
                return False
            finally:
                probe.close()
    for name in ("cuda_monolithic", "cuda_route"):
        require(
            port_available(ports[name]),
            f"E_LOCAL_PORT_OCCUPIED: {name}",
        )

    files = exact_keys(value["files"], FILE_KEYS, "operator_input.files")
    paths = {}
    identities = {}
    hashes = {}
    parsed_files = {}
    retained_files = {}
    for name in sorted(FILE_KEYS):
        path, metadata, actual_sha256, parsed, retained = _validate_pin(
            files[name],
            name,
            after_file_read,
        )
        require(str(path) not in paths, f"E_PATH_REUSE: {name}")
        identity = (metadata.st_dev, metadata.st_ino)
        require(identity not in identities, f"E_FILE_IDENTITY_REUSE: {name}")
        paths[str(path)] = name
        identities[identity] = name
        hashes[name] = actual_sha256
        if parsed is not None:
            parsed_files[name] = parsed
        if retained is not None:
            retained_files[name] = retained

    contract = parsed_files["contract"]
    candidate = parsed_files["candidate"]
    history = parsed_files["token_history"]
    tokenizer = parsed_files["tokenizer_plan"]
    monolithic = parsed_files["cuda_monolithic_launch"]
    topology_receipt = parsed_files["topology_receipt"]
    corpus_pin = files["quality_corpus"]
    model_pin = files["model"]
    models = candidate.get("models")
    require(type(models) is list, "E_CANDIDATE_MODELS")
    candidate_model = None
    for index, item in enumerate(models):
        require(type(item) is dict, f"E_CANDIDATE_MODEL_TYPE: {index}")
        if item.get("slot") == "A" and item.get("model_id") == MODEL_ID:
            require(candidate_model is None, "E_CANDIDATE_MODEL_REUSE")
            candidate_model = item
    require(type(candidate_model) is dict, "E_CANDIDATE_MODEL")
    candidate_artifact = candidate_model.get("artifact")
    require(type(candidate_artifact) is dict, "E_CANDIDATE_ARTIFACT")
    require(
        {"bytes", "sha256"}.issubset(candidate_artifact),
        "E_CANDIDATE_ARTIFACT_FIELDS",
    )
    tokenizer = exact_keys(
        tokenizer,
        {
            "command_template",
            "component_id",
            "cwd",
            "environment",
            "executable",
            "model",
            "protocol",
            "schema",
            "timeout_seconds",
        },
        "tokenizer_plan",
    )
    tokenizer_model = exact_keys(
        tokenizer["model"],
        {"bytes", "model_id", "path", "sha256", "vocab_size"},
        "tokenizer_plan.model",
    )
    tokenizer_executable = exact_keys(
        tokenizer["executable"],
        {"bytes", "path", "sha256"},
        "tokenizer_plan.executable",
    )
    exact(
        hashes["candidate"],
        contract["candidate_lock"]["sha256"],
        "candidate.sha256",
    )
    exact(
        files["candidate"]["bytes"],
        contract["candidate_lock"]["bytes"],
        "candidate.bytes",
    )
    exact(
        candidate_artifact["sha256"],
        hashes["model"],
        "model.sha256",
    )
    exact(
        candidate_artifact["bytes"],
        model_pin["bytes"],
        "model.bytes",
    )
    exact(
        tokenizer_model,
        {
            "bytes": model_pin["bytes"],
            "model_id": MODEL_ID,
            "path": model_pin["path"],
            "sha256": model_pin["sha256"],
            "vocab_size": contract["token_history_protocol"]["vocab_size"],
        },
        "tokenizer.model",
    )
    exact(
        tokenizer_executable["path"],
        files["codec"]["path"],
        "tokenizer.executable.path",
    )
    exact(
        tokenizer_executable["sha256"],
        files["codec"]["sha256"],
        "tokenizer.executable.sha256",
    )
    exact(
        tokenizer_executable["bytes"],
        files["codec"]["bytes"],
        "tokenizer.executable.bytes",
    )
    exact(
        hashes["quality_corpus"],
        contract["quality_corpus"]["sha256"],
        "quality_corpus.sha256",
    )
    exact(
        corpus_pin["bytes"],
        contract["quality_corpus"]["bytes"],
        "quality_corpus.bytes",
    )
    validate_token_history(
        history,
        contract=contract,
        candidate_sha256=hashes["candidate"],
        model_sha256=hashes["model"],
        corpus_sha256=hashes["quality_corpus"],
        tokenizer_plan_sha256=hashes["tokenizer_plan"],
        tokenizer_sha256=hashes["codec"],
        candidate=candidate,
        corpus_raw=retained_files["quality_corpus"],
        history_raw=retained_files["token_history"],
        tokenizer_plan=tokenizer,
    )
    monolithic_closure = validate_monolithic_launch(
        monolithic,
        model_pin=model_pin,
        runtime_path=files["monolithic_runtime"]["path"],
        route_epoch=value["route_epoch"],
        port=ports["cuda_monolithic"],
    )
    runtime_inventory = validate_runtime_inventory(
        parsed_files["runtime_bundle_inventory"],
        cuda_launchers={
            "cuda_monolithic": files["monolithic_runtime"]["path"],
            "cuda_route": files["cuda_runtime"]["path"],
        },
    )
    inventory_monolithic = {
        item["component_id"]: item
        for item in runtime_inventory["components"]
        if item["bundle_id"] == "cuda_monolithic"
    }
    exact(
        set(inventory_monolithic),
        set(monolithic_closure["components"]),
        "cuda_monolithic.component_ids",
    )
    for component_id, component in inventory_monolithic.items():
        launch_component = monolithic_closure["components"][component_id]
        for key in ("path", "sha256", "stat"):
            exact(
                component[key],
                launch_component[key],
                f"cuda_monolithic.{component_id}.{key}",
            )
    exact(
        runtime_inventory["bundle_roots"]["cuda_monolithic"],
        monolithic_closure["bundle_root"],
        "cuda_monolithic.bundle_root",
    )
    exact(files["adb"]["path"], ADB_PATH, "files.adb.path")
    exact(
        files["nvidia_smi"]["path"],
        NVIDIA_SMI_PATH,
        "files.nvidia_smi.path",
    )
    exact(files["python"]["path"], PYTHON_PATH, "files.python.path")
    observed = validate_topology(topology_receipt, contract)
    exact(
        observed["cuda"]["uuid"],
        value["topology"]["cuda_uuid"],
        "topology_receipt.cuda.uuid",
    )
    for phone, serial in value["topology"]["physical_serials"].items():
        exact(
            observed["phones"][phone]["usb_selector"],
            serial,
            f"topology_receipt.{phone}.usb_selector",
        )

    directories = exact_keys(
        value["directories"],
        DIRECTORY_KEYS,
        "operator_input.directories",
    )
    resolved_directories = {}
    for name in sorted(DIRECTORY_KEYS):
        path = Path(text(directories[name], f"directories.{name}"))
        no_symlink_chain(path, f"directories.{name}")
        require(path.is_dir(), f"E_DIRECTORY: {name}")
        resolved = str(path.resolve(strict=True))
        require(resolved not in resolved_directories, f"E_DIRECTORY_REUSE: {name}")
        require(str(path) not in paths, f"E_PATH_REUSE: directories.{name}")
        resolved_directories[resolved] = name
    exact(
        directories["cuda_bundle_root"],
        runtime_inventory["bundle_roots"]["cuda_route"],
        "directories.cuda_bundle_root",
    )
    return {
        "directories": {
            name: directories[name] for name in sorted(directories)
        },
        "file_sha256": {
            name: hashes[name] for name in sorted(hashes)
        },
        "topology_receipt_sha256": hashes["topology_receipt"],
        "ports": {name: ports[name] for name in sorted(ports)},
        "route_epoch": value["route_epoch"],
        "schema": OPERATOR_INPUT_SCHEMA,
        "status": OPERATOR_INPUT_STATUS,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture-topology")
    capture.add_argument("--contract", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--timeout-seconds", type=int, default=30)
    validate_capture = subparsers.add_parser("validate-topology")
    validate_capture.add_argument("--contract", type=Path, required=True)
    validate_capture.add_argument("--input", type=Path, required=True)
    validate_replay = subparsers.add_parser("validate-topology-replay")
    validate_replay.add_argument("--contract", type=Path, required=True)
    validate_replay.add_argument("--input", type=Path, required=True)
    validate_operator = subparsers.add_parser("validate-operator-input")
    validate_operator.add_argument("--input", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if args.command == "capture-topology":
            contract, _ = read_canonical(args.contract, "contract")
            value = capture_topology(
                contract,
                timeout_seconds=args.timeout_seconds,
            )
            write_new(args.output, value)
            print("V24_NO_MODEL_TOPOLOGY_PASS")
        elif args.command in {
            "validate-topology",
            "validate-topology-replay",
        }:
            contract, _ = read_canonical(args.contract, "contract")
            value, _ = read_canonical(args.input, "topology")
            validate_topology(
                value,
                contract,
                require_fresh=args.command == "validate-topology",
            )
            print(
                "V24_NO_MODEL_TOPOLOGY_PASS"
                if args.command == "validate-topology"
                else "V24_TOPOLOGY_REPLAY_VALID"
            )
        else:
            value, _ = read_canonical(args.input, "operator_input")
            validate_operator_input(value)
            print("V24_NO_MODEL_INPUT_PASS_REMOTE_EVIDENCE_PENDING")
        return 0
    except (
        DeploymentError,
        AttributeError,
        KeyError,
        OSError,
        subprocess.SubprocessError,
        TypeError,
        ValueError,
    ) as error:
        print(f"V24_DESKTOP_DEPLOYMENT_REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
