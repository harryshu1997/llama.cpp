#!/usr/bin/python3 -I
"""Capture one fail-closed phone runtime snapshot."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys
import time
from typing import Any


PLAN_SCHEMA = "s39-phone-runtime-probe-plan-v1"
OUTPUT_SCHEMA = "s39-phone-runtime-probe-v1"
CAPTURE_SCHEMA = "s39-cp0-r1-phone-runtime-probe-v1"
CAPTURE_SCHEMA_V24 = "s39-cp0-r1-v24-phone-runtime-probe-v1"
CAPTURE_SCHEMAS = {
    CAPTURE_SCHEMA,
    CAPTURE_SCHEMA_V24,
}
ADB_PORT = 5038
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
BOOT_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}$"
)
ANDROID_TIME_RE = re.compile(
    r"^(?P<base>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
    r"\.(?P<fraction>\d{1,9}) (?P<zone>[+-]\d{4})$"
)
THERMAL_STATUS_RE = re.compile(r"^\s*Thermal Status:\s*(\d+)\s*$")
MAX_JSON = 8 * 1024 * 1024
MAX_REMOTE_OUTPUT = 32 * 1024 * 1024

PLAN_KEYS = {
    "android",
    "capture_schema",
    "model_id",
    "model_sha256",
    "network_process",
    "process",
    "schema",
    "shard_artifact",
    "stage_v3",
    "telemetry",
    "worker_artifact",
}
ANDROID_KEYS = {
    "adb_path",
    "adb_port",
    "adb_selector",
    "adb_sha256",
    "boot_id_source",
    "device",
    "model",
    "physical_serial",
    "product",
}
PROCESS_KEYS = {
    "argv",
    "executable_path",
}
NETWORK_PROCESS_KEYS = {
    "artifact",
    "argv",
    "executable_path",
    "role",
}
ARTIFACT_KEYS = {
    "bytes",
    "path",
    "sha256",
    "stat",
}
STAT_KEYS = {
    "ctime_ns",
    "device_id",
    "inode",
    "mode",
    "mtime_ns",
    "size",
}
STAGE_KEYS = {
    "expected_active_sequences",
    "source",
}
TELEMETRY_KEYS = {
    "direct_peer_ipv4",
    "direct_peer_local_port",
    "direct_peer_port",
    "interface",
    "local_ipv4",
    "max_gpu_millic",
    "min_available_bytes",
}

REMOTE_SNAPSHOT_SCRIPT = r'''
hex_file() { od -An -tx1 -v "$1" 2>/dev/null | tr -d ' \n'; }
hex_cmd() { "$@" 2>/dev/null | od -An -tx1 -v | tr -d ' \n'; }
pid="$1"
network_pid="$2"
worker="$3"
shard="$4"
interface="$5"
network_executable="$6"
printf 'SERIAL %s\n' "$(getprop ro.serialno | tr -d '\r\n')"
printf 'PRODUCT %s\n' "$(getprop ro.product.name | tr -d '\r\n')"
printf 'MODEL %s\n' "$(getprop ro.product.model | tr -d '\r\n')"
printf 'DEVICE %s\n' "$(getprop ro.product.device | tr -d '\r\n')"
printf 'BOOT %s\n' "$(cat /proc/sys/kernel/random/boot_id | tr -d '\r\n')"
printf 'EXE %s\n' "$(readlink "/proc/$pid/exe" | od -An -tx1 -v | tr -d ' \n')"
printf 'CMD %s\n' "$(hex_file "/proc/$pid/cmdline")"
printf 'STAT %s\n' "$(hex_file "/proc/$pid/stat")"
printf 'NEXE %s\n' "$(readlink "/proc/$network_pid/exe" | od -An -tx1 -v | tr -d ' \n')"
printf 'NCMD %s\n' "$(hex_file "/proc/$network_pid/cmdline")"
printf 'NSTAT %s\n' "$(hex_file "/proc/$network_pid/stat")"
printf 'PSTATUS %s\n' "$(hex_file "/proc/$pid/status")"
printf 'MEMINFO %s\n' "$(hex_file /proc/meminfo)"
printf 'THERMAL %s\n' "$(hex_cmd dumpsys thermalservice)"
printf 'ZONES %s\n' "$(
  for z in /sys/class/thermal/thermal_zone*; do
    [ -r "$z/type" ] && [ -r "$z/temp" ] || continue
    printf '%s=%s\n' "$(cat "$z/type")" "$(cat "$z/temp")"
  done | od -An -tx1 -v | tr -d ' \n'
)"
printf 'IPV4 %s\n' "$(hex_cmd ip -o -4 addr show dev "$interface")"
printf 'RX %s\n' "$(cat "/sys/class/net/$interface/statistics/rx_bytes")"
printf 'TX %s\n' "$(cat "/sys/class/net/$interface/statistics/tx_bytes")"
printf 'TCP %s\n' "$(hex_file /proc/net/tcp)"
printf 'FDS %s\n' "$(
  for f in /proc/"$network_pid"/fd/*; do readlink "$f" 2>/dev/null || true; done |
    od -An -tx1 -v | tr -d ' \n'
)"
stat_format='DEV=%d|INO=%i|SIZE=%s|MODE=%f|MTIME_S=%Y|MTIME=%y|CTIME_S=%Z|CTIME=%z'
printf 'WORKERSTAT %s\n' "$(stat -c "$stat_format" -- "$worker")"
printf 'SHARDSTAT %s\n' "$(stat -c "$stat_format" -- "$shard")"
printf 'NETWORKSTAT %s\n' "$(stat -c "$stat_format" -- "$network_executable")"
printf 'STAT2 %s\n' "$(hex_file "/proc/$pid/stat")"
printf 'NSTAT2 %s\n' "$(hex_file "/proc/$network_pid/stat")"
'''.strip()


class ProbeError(RuntimeError):
    pass


class SubprocessRunner:
    def run(self, argv: list[str], *, timeout: float) -> subprocess.CompletedProcess:
        return subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProbeError(message)


def exact(value: Any, expected: Any, field: str) -> None:
    require(
        type(value) is type(expected) and value == expected,
        f"E_VALUE: {field}",
    )


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        require(key not in result, f"E_DUPLICATE_KEY: {key}")
        result[key] = value
    return result


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
        raise ProbeError("E_CANONICAL") from error
    require(len(raw) <= MAX_JSON, "E_JSON_SIZE")
    return raw


def canonical_compact(value: Any) -> bytes:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, UnicodeEncodeError) as error:
        raise ProbeError("E_CANONICAL") from error
    require(0 < len(raw) <= MAX_JSON, "E_JSON_SIZE")
    return raw


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
        "\x00" not in value
        and "\n" not in value
        and all(0x20 <= ord(character) <= 0x7E for character in value),
        f"E_ASCII: {field}",
    )
    return value


def digest(value: Any, field: str) -> str:
    value = text(value, field, 64)
    require(DIGEST_RE.fullmatch(value) is not None, f"E_DIGEST: {field}")
    return value


def absolute_path(value: Any, field: str) -> str:
    value = text(value, field)
    path = Path(value)
    require(path.is_absolute() and ".." not in path.parts, f"E_PATH: {field}")
    return value


def validate_argv(value: Any, field: str) -> list[str]:
    require(type(value) is list and bool(value) and len(value) <= 256, f"E_ARGV: {field}")
    for index, item in enumerate(value):
        text(item, f"{field}[{index}]", 32 * 1024)
    return value


def validate_artifact(value: Any, field: str) -> dict[str, Any]:
    value = exact_keys(value, ARTIFACT_KEYS, field)
    absolute_path(value["path"], f"{field}.path")
    size = integer(value["bytes"], f"{field}.bytes", 1)
    digest(value["sha256"], f"{field}.sha256")
    record = exact_keys(value["stat"], STAT_KEYS, f"{field}.stat")
    for key in record:
        integer(record[key], f"{field}.stat.{key}")
    require(
        record["inode"] > 0
        and record["size"] == size
        and stat.S_ISREG(record["mode"]),
        f"E_STAT: {field}",
    )
    return value


def validate_plan(value: Any) -> dict[str, Any]:
    value = exact_keys(value, PLAN_KEYS, "plan")
    exact(value["schema"], PLAN_SCHEMA, "plan.schema")
    android = exact_keys(value["android"], ANDROID_KEYS, "android")
    absolute_path(android["adb_path"], "android.adb_path")
    digest(android["adb_sha256"], "android.adb_sha256")
    exact(integer(android["adb_port"], "android.adb_port", 1), ADB_PORT, "adb.port")
    selector = text(android["adb_selector"], "android.adb_selector", 255)
    serial = text(android["physical_serial"], "android.physical_serial", 255)
    require(":" in selector and selector != serial, "E_ADB_SELECTOR")
    exact(
        android["boot_id_source"],
        "phase_fresh_snapshot",
        "android.boot_id_source",
    )
    for key in ("device", "model", "product"):
        text(android[key], f"android.{key}", 255)
    text(value["model_id"], "plan.model_id", 255)
    digest(value["model_sha256"], "plan.model_sha256")
    capture_schema = text(value["capture_schema"], "plan.capture_schema", 128)
    require(capture_schema in CAPTURE_SCHEMAS, "E_CAPTURE_SCHEMA")

    process = exact_keys(value["process"], PROCESS_KEYS, "process")
    executable = absolute_path(process["executable_path"], "process.executable")
    argv = validate_argv(process["argv"], "process.argv")
    exact(argv[0], executable, "process.argv[0]")

    worker = validate_artifact(value["worker_artifact"], "worker_artifact")
    shard = validate_artifact(value["shard_artifact"], "shard_artifact")
    exact(worker["path"], executable, "worker.path")

    network = exact_keys(
        value["network_process"],
        NETWORK_PROCESS_KEYS,
        "network_process",
    )
    network_executable = absolute_path(
        network["executable_path"],
        "network_process.executable_path",
    )
    network_argv = validate_argv(network["argv"], "network_process.argv")
    exact(network_argv[0], network_executable, "network_process.argv[0]")
    network_artifact = validate_artifact(
        network["artifact"],
        "network_process.artifact",
    )
    exact(
        network_artifact["path"],
        network_executable,
        "network_process.artifact.path",
    )
    role = text(network["role"], "network_process.role", 32)
    require(role in ("direct_relay", "stagenet_worker"), "E_NETWORK_ROLE")
    if role == "stagenet_worker":
        exact(network_executable, executable, "network_process.executable_path")
        exact(network_argv, argv, "network_process.argv")
        exact(network_artifact, worker, "network_process.artifact")

    stage = exact_keys(value["stage_v3"], STAGE_KEYS, "stage_v3")
    exact(
        integer(
            stage["expected_active_sequences"],
            "stage_v3.expected_active_sequences",
        ),
        0,
        "stage_v3.expected_active_sequences",
    )
    exact(stage["source"], "relay_owned_status", "stage_v3.source")

    telemetry = exact_keys(value["telemetry"], TELEMETRY_KEYS, "telemetry")
    interface = text(telemetry["interface"], "telemetry.interface", 64)
    require(
        all(character.isalnum() or character in "._-" for character in interface),
        "E_INTERFACE",
    )
    for key in ("local_ipv4", "direct_peer_ipv4"):
        item = text(telemetry[key], f"telemetry.{key}", 64)
        try:
            parsed = ipaddress.ip_address(item)
        except ValueError as error:
            raise ProbeError(f"E_IPV4: telemetry.{key}") from error
        require(type(parsed) is ipaddress.IPv4Address, f"E_IPV4: telemetry.{key}")
    for key in ("direct_peer_local_port", "direct_peer_port"):
        item = integer(telemetry[key], f"telemetry.{key}", 1)
        require(item <= 65535, f"E_PORT: telemetry.{key}")
    integer(telemetry["min_available_bytes"], "telemetry.min_available_bytes", 1)
    max_gpu = integer(telemetry["max_gpu_millic"], "telemetry.max_gpu_millic", 1)
    require(max_gpu <= 200_000, "E_GPU_THERMAL_BOUND")
    return value


def parse_plan_json(raw_text: str, expected_sha256: str) -> dict[str, Any]:
    text(raw_text, "plan_json", MAX_JSON)
    digest(expected_sha256, "plan_sha256")
    raw = raw_text.encode("ascii")
    exact(hashlib.sha256(raw).hexdigest(), expected_sha256, "plan.sha256")
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=strict_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ProbeError(f"E_JSON_NUMBER: {item}")
            ),
        )
    except json.JSONDecodeError as error:
        raise ProbeError("E_PLAN_JSON") from error
    exact(canonical_compact(value), raw, "plan.canonical")
    return validate_plan(value)


def validate_boot_id(value: Any, field: str = "boot_id") -> str:
    value = text(value, field, 64)
    require(BOOT_RE.fullmatch(value) is not None, f"E_BOOT_ID: {field}")
    return value


def read_sealed_adb(android: dict[str, Any]) -> None:
    path = Path(android["adb_path"])
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ProbeError(f"E_ADB_OPEN: {error}") from error
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), "E_ADB_TYPE")
        hasher = hashlib.sha256()
        while block := os.read(descriptor, 1024 * 1024):
            hasher.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "E_ADB_CHANGED",
    )
    exact(hasher.hexdigest(), android["adb_sha256"], "adb.sha256")
    require(after.st_mode & 0o111, "E_ADB_EXECUTABLE")


def adb_prefix(android: dict[str, Any]) -> list[str]:
    exact(android["adb_port"], ADB_PORT, "adb.port")
    require(
        ":" in android["adb_selector"]
        and android["adb_selector"] != android["physical_serial"],
        "E_ADB_SELECTOR",
    )
    return [
        android["adb_path"],
        "-P",
        str(ADB_PORT),
        "-s",
        android["adb_selector"],
    ]


def remote_argv(
    plan: dict[str, Any],
    pid: int,
    network_pid: int,
) -> list[str]:
    android = plan["android"]
    integer(pid, "runtime.pid", 1)
    integer(network_pid, "network_runtime.pid", 1)
    command = " ".join([
        "sh -c",
        shlex.quote(REMOTE_SNAPSHOT_SCRIPT),
        "s39-probe",
        str(pid),
        str(network_pid),
        shlex.quote(plan["worker_artifact"]["path"]),
        shlex.quote(plan["shard_artifact"]["path"]),
        shlex.quote(plan["telemetry"]["interface"]),
        shlex.quote(plan["network_process"]["executable_path"]),
    ])
    return (
        adb_prefix(android)
        + [
            "shell",
            command,
        ]
    )


def decode_hex(value: str, field: str) -> bytes:
    require(
        len(value) % 2 == 0
        and all(character in "0123456789abcdef" for character in value),
        f"E_HEX: {field}",
    )
    try:
        return bytes.fromhex(value)
    except ValueError as error:
        raise ProbeError(f"E_HEX: {field}") from error


def parse_android_time(value: str, seconds: int, field: str) -> int:
    match = ANDROID_TIME_RE.fullmatch(value)
    require(match is not None, f"E_ANDROID_TIME: {field}")
    base = datetime.datetime.strptime(
        f"{match.group('base')} {match.group('zone')}",
        "%Y-%m-%d %H:%M:%S %z",
    )
    fraction = match.group("fraction").ljust(9, "0")
    result = int(base.timestamp()) * 1_000_000_000 + int(fraction)
    exact(result // 1_000_000_000, seconds, f"{field}.seconds")
    return result


def parse_android_stat(value: str, field: str) -> dict[str, int]:
    expected = (
        "DEV",
        "INO",
        "SIZE",
        "MODE",
        "MTIME_S",
        "MTIME",
        "CTIME_S",
        "CTIME",
    )
    parts = value.split("|")
    require(len(parts) == len(expected), f"E_ANDROID_STAT_FIELDS: {field}")
    fields = {}
    for item, key in zip(parts, expected):
        prefix = f"{key}="
        require(item.startswith(prefix), f"E_ANDROID_STAT_FIELD: {field}.{key}")
        fields[key] = item[len(prefix):]
    try:
        device = int(fields["DEV"])
        inode = int(fields["INO"])
        size = int(fields["SIZE"])
        mode = int(fields["MODE"], 16)
        mtime_s = int(fields["MTIME_S"])
        ctime_s = int(fields["CTIME_S"])
    except ValueError as error:
        raise ProbeError(f"E_ANDROID_STAT_INTEGER: {field}") from error
    result = {
        "ctime_ns": parse_android_time(fields["CTIME"], ctime_s, f"{field}.ctime"),
        "device_id": device,
        "inode": inode,
        "mode": mode,
        "mtime_ns": parse_android_time(fields["MTIME"], mtime_s, f"{field}.mtime"),
        "size": size,
    }
    exact_keys(result, STAT_KEYS, field)
    require(
        result["inode"] > 0
        and result["size"] > 0
        and stat.S_ISREG(result["mode"]),
        f"E_STAT: {field}",
    )
    return result


def parse_start_ticks(raw: bytes, expected_pid: int) -> int:
    closing = raw.rfind(b")")
    require(
        closing > 0 and raw[closing + 1:closing + 2] == b" ",
        "E_PROCESS_STAT",
    )
    try:
        pid = int(raw[:raw.find(b" ")])
        fields = raw[closing + 2:].split()
        ticks = int(fields[19])
    except (IndexError, ValueError) as error:
        raise ProbeError("E_PROCESS_STAT") from error
    require(pid == expected_pid and ticks > 0, "E_PROCESS_STAT")
    return ticks


def parse_key_value_bytes(raw: bytes) -> dict[str, int]:
    result = {}
    for line in raw.decode("ascii").splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0].endswith(":"):
            try:
                scale = 1024 if len(fields) >= 3 and fields[2] == "kB" else 1
                result[fields[0][:-1]] = int(fields[1]) * scale
            except ValueError:
                continue
    return result


def parse_thermal_status(raw: bytes) -> int:
    try:
        text_value = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise ProbeError("E_THERMAL_ASCII") from error
    values = [
        int(match.group(1))
        for line in text_value.splitlines()
        if (match := THERMAL_STATUS_RE.fullmatch(line)) is not None
    ]
    require(len(values) == 1, "E_THERMAL_STATUS")
    return values[0]


def parse_gpu_temperature(raw: bytes) -> tuple[int, list[dict[str, Any]]]:
    try:
        text_value = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise ProbeError("E_ZONE_ASCII") from error
    rows = []
    names = set()
    for line in text_value.splitlines():
        name, separator, value = line.rpartition("=")
        require(separator and name and name not in names, "E_ZONE_ROW")
        names.add(name)
        try:
            temperature = int(value)
        except ValueError as error:
            raise ProbeError("E_ZONE_VALUE") from error
        require(0 < temperature <= 250_000, "E_ZONE_RANGE")
        rows.append({"name": name, "temp_millic": temperature})
    rows.sort(key=lambda item: item["name"])
    gpu = [
        row["temp_millic"]
        for row in rows
        if "gpu" in row["name"].lower() or "adreno" in row["name"].lower()
    ]
    require(gpu, "E_GPU_ZONE")
    return max(gpu), rows


def parse_ipv4(raw: bytes, interface: str) -> str:
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ProbeError("E_INTERFACE_ASCII") from error
    addresses = []
    for line in lines:
        fields = line.split()
        if len(fields) >= 4 and fields[1] == interface and fields[2] == "inet":
            try:
                addresses.append(str(ipaddress.IPv4Interface(fields[3]).ip))
            except ValueError as error:
                raise ProbeError("E_INTERFACE_IPV4") from error
    require(len(set(addresses)) == 1, "E_INTERFACE_IPV4")
    return addresses[0]


def parse_tcp(raw: bytes) -> list[dict[str, Any]]:
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ProbeError("E_TCP_ASCII") from error
    result = []
    for line in lines[1:]:
        fields = line.split()
        if len(fields) < 10 or fields[3] != "01":
            continue
        try:
            local_hex, local_port = fields[1].split(":")
            remote_hex, remote_port = fields[2].split(":")
            result.append({
                "local_ipv4": str(
                    ipaddress.IPv4Address(bytes.fromhex(local_hex)[::-1])
                ),
                "local_port": int(local_port, 16),
                "peer_ipv4": str(
                    ipaddress.IPv4Address(bytes.fromhex(remote_hex)[::-1])
                ),
                "peer_port": int(remote_port, 16),
                "socket_inode": int(fields[9]),
            })
        except (ValueError, ipaddress.AddressValueError) as error:
            raise ProbeError("E_TCP_ROW") from error
    return result


def parse_socket_inodes(raw: bytes) -> set[int]:
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ProbeError("E_FD_ASCII") from error
    result = set()
    for line in lines:
        match = re.fullmatch(r"socket:\[(\d+)\]", line)
        if match is not None:
            result.add(int(match.group(1)))
    return result


def parse_remote_snapshot(
    raw: bytes,
    plan: dict[str, Any],
    pid: int,
    expected_start_ticks: int,
    network_pid: int,
    expected_network_start_ticks: int,
    expected_boot_id: str,
) -> dict[str, Any]:
    integer(pid, "runtime.pid", 1)
    integer(expected_start_ticks, "runtime.start_ticks", 1)
    integer(network_pid, "network_runtime.pid", 1)
    integer(
        expected_network_start_ticks,
        "network_runtime.start_ticks",
        1,
    )
    expected_boot_id = validate_boot_id(expected_boot_id)
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ProbeError("E_REMOTE_ASCII") from error
    expected_keys = {
        "BOOT",
        "CMD",
        "DEVICE",
        "EXE",
        "FDS",
        "IPV4",
        "MEMINFO",
        "MODEL",
        "NCMD",
        "NEXE",
        "NSTAT",
        "NSTAT2",
        "NETWORKSTAT",
        "PSTATUS",
        "PRODUCT",
        "RX",
        "SERIAL",
        "SHARDSTAT",
        "STAT",
        "STAT2",
        "TCP",
        "THERMAL",
        "TX",
        "WORKERSTAT",
        "ZONES",
    }
    values = {}
    for index, line in enumerate(lines):
        key, separator, value = line.partition(" ")
        require(separator and key not in values, f"E_REMOTE_RECORD: {index}")
        values[key] = value
    exact(set(values), expected_keys, "remote.keys")
    android = plan["android"]
    process = plan["process"]
    exact(values["SERIAL"], android["physical_serial"], "remote.serial")
    exact(values["BOOT"], expected_boot_id, "remote.boot_id")
    exact(values["DEVICE"], android["device"], "remote.device")
    exact(values["MODEL"], android["model"], "remote.model")
    exact(values["PRODUCT"], android["product"], "remote.product")
    try:
        executable = decode_hex(values["EXE"], "remote.exe").decode("ascii")
        cmdline = decode_hex(values["CMD"], "remote.cmd")
    except UnicodeDecodeError as error:
        raise ProbeError("E_REMOTE_PROCESS_TEXT") from error
    exact(executable, process["executable_path"], "remote.executable")
    require(cmdline.endswith(b"\x00"), "E_REMOTE_ARGV_END")
    try:
        argv = [item.decode("ascii") for item in cmdline[:-1].split(b"\x00")]
    except UnicodeDecodeError as error:
        raise ProbeError("E_REMOTE_ARGV") from error
    exact(argv, process["argv"], "remote.argv")
    start_ticks = parse_start_ticks(
        decode_hex(values["STAT"], "remote.stat"),
        pid,
    )
    exact(start_ticks, expected_start_ticks, "remote.start_ticks")
    exact(values["STAT2"], values["STAT"], "remote.process_changed")
    network = plan["network_process"]
    try:
        network_executable = decode_hex(
            values["NEXE"],
            "remote.network_exe",
        ).decode("ascii")
        network_cmdline = decode_hex(values["NCMD"], "remote.network_cmd")
    except UnicodeDecodeError as error:
        raise ProbeError("E_REMOTE_NETWORK_PROCESS_TEXT") from error
    exact(
        network_executable,
        network["executable_path"],
        "remote.network_executable",
    )
    require(network_cmdline.endswith(b"\x00"), "E_REMOTE_NETWORK_ARGV_END")
    try:
        network_argv = [
            item.decode("ascii")
            for item in network_cmdline[:-1].split(b"\x00")
        ]
    except UnicodeDecodeError as error:
        raise ProbeError("E_REMOTE_NETWORK_ARGV") from error
    exact(network_argv, network["argv"], "remote.network_argv")
    network_start_ticks = parse_start_ticks(
        decode_hex(values["NSTAT"], "remote.network_stat"),
        network_pid,
    )
    exact(
        network_start_ticks,
        expected_network_start_ticks,
        "remote.network_start_ticks",
    )
    exact(values["NSTAT2"], values["NSTAT"], "remote.network_process_changed")
    if network["role"] == "stagenet_worker":
        exact(network_pid, pid, "remote.network_pid")
        exact(network_start_ticks, start_ticks, "remote.network_start_ticks")
    else:
        require(network_pid != pid, "E_NETWORK_PROCESS_ALIAS")

    worker = plan["worker_artifact"]
    shard = plan["shard_artifact"]
    worker_stat = parse_android_stat(values["WORKERSTAT"], "remote.worker_stat")
    shard_stat = parse_android_stat(values["SHARDSTAT"], "remote.shard_stat")
    network_stat = parse_android_stat(
        values["NETWORKSTAT"],
        "remote.network_stat",
    )
    exact(worker_stat, worker["stat"], "remote.worker_stat")
    exact(shard_stat, shard["stat"], "remote.shard_stat")
    exact(
        network_stat,
        network["artifact"]["stat"],
        "remote.network_stat",
    )
    try:
        rx_bytes = int(values["RX"].strip())
        tx_bytes = int(values["TX"].strip())
    except ValueError as error:
        raise ProbeError("E_REMOTE_INTEGER") from error
    require(rx_bytes >= 0 and tx_bytes >= 0, "E_INTERFACE_COUNTER")

    process_status = parse_key_value_bytes(
        decode_hex(values["PSTATUS"], "remote.process_status")
    )
    memory = parse_key_value_bytes(
        decode_hex(values["MEMINFO"], "remote.meminfo")
    )
    require(
        all(key in memory for key in ("MemAvailable", "SwapFree", "SwapTotal")),
        "E_MEMORY_FIELDS",
    )
    require(memory["SwapFree"] <= memory["SwapTotal"], "E_SWAP_COUNTERS")
    process_swap = process_status.get("VmSwap")
    require(process_swap is not None, "E_PROCESS_SWAP_FIELD")
    system_swap = memory["SwapTotal"] - memory["SwapFree"]
    exact(process_swap, 0, "remote.process_swap_bytes")
    require(
        memory["MemAvailable"] >= plan["telemetry"]["min_available_bytes"],
        "E_MEMORY_HEADROOM",
    )

    thermal_status = parse_thermal_status(
        decode_hex(values["THERMAL"], "remote.thermal")
    )
    gpu_max, zones = parse_gpu_temperature(
        decode_hex(values["ZONES"], "remote.zones")
    )
    exact(thermal_status, 0, "remote.thermal_status")
    require(
        gpu_max <= plan["telemetry"]["max_gpu_millic"],
        "E_GPU_THERMAL",
    )

    telemetry = plan["telemetry"]
    ipv4 = parse_ipv4(
        decode_hex(values["IPV4"], "remote.ipv4"),
        telemetry["interface"],
    )
    exact(ipv4, telemetry["local_ipv4"], "remote.local_ipv4")
    tcp = parse_tcp(decode_hex(values["TCP"], "remote.tcp"))
    socket_inodes = parse_socket_inodes(
        decode_hex(values["FDS"], "remote.fds")
    )
    matches = [
        row
        for row in tcp
        if row["local_ipv4"] == telemetry["local_ipv4"]
        and row["local_port"] == telemetry["direct_peer_local_port"]
        and row["peer_ipv4"] == telemetry["direct_peer_ipv4"]
        and row["peer_port"] == telemetry["direct_peer_port"]
        and row["socket_inode"] in socket_inodes
    ]
    require(len(matches) == 1, "E_DIRECT_PEER")
    return {
        "available_bytes": memory["MemAvailable"],
        "boot_id": expected_boot_id,
        "direct_peer": matches[0],
        "gpu_max_millic": gpu_max,
        "interface": {
            "ipv4": ipv4,
            "name": telemetry["interface"],
            "rx_bytes": rx_bytes,
            "tx_bytes": tx_bytes,
        },
        "physical_serial": android["physical_serial"],
        "network_process": {
            "argv": network_argv,
            "executable_path": network_executable,
            "executable_sha256": network["artifact"]["sha256"],
            "pid": network_pid,
            "role": network["role"],
            "start_ticks": network_start_ticks,
            "observed_stat": network_stat,
        },
        "process": {
            "argv": argv,
            "executable_path": executable,
            "pid": pid,
            "start_ticks": start_ticks,
        },
        "process_swap_bytes": process_swap,
        "shard_artifact": {**shard, "observed_stat": shard_stat},
        "system_swap_used_bytes": system_swap,
        "thermal_status": thermal_status,
        "thermal_zones": zones,
        "worker_artifact": {**worker, "observed_stat": worker_stat},
    }


def capture_row(
    plan: dict[str, Any],
    remote: dict[str, Any],
) -> dict[str, Any]:
    android = plan["android"]
    process = remote["process"]
    interface = remote["interface"]
    peer = remote["direct_peer"]
    return {
        "active_sequences": 0,
        "available_bytes": remote["available_bytes"],
        "boot_id": remote["boot_id"],
        "device": android["device"],
        "direct_peer": {
            "interface": interface["name"],
            "local_ipv4": interface["ipv4"],
            "peer_ipv4": peer["peer_ipv4"],
            "socket_peer_observed": True,
        },
        "gpu_max_millic": remote["gpu_max_millic"],
        "interface": interface,
        "loaded_shard_path": plan["shard_artifact"]["path"],
        "loaded_shard_sha256": plan["shard_artifact"]["sha256"],
        "model": android["model"],
        "model_id": plan["model_id"],
        "model_sha256": plan["model_sha256"],
        "network_executable_path": remote["network_process"]["executable_path"],
        "network_executable_sha256": remote["network_process"][
            "executable_sha256"
        ],
        "network_pid": remote["network_process"]["pid"],
        "network_process_role": remote["network_process"]["role"],
        "network_start_ticks": remote["network_process"]["start_ticks"],
        "process_swap_bytes": remote["process_swap_bytes"],
        "product": android["product"],
        "schema": plan["capture_schema"],
        "serial": android["physical_serial"],
        "system_swap_used_bytes": remote["system_swap_used_bytes"],
        "worker_executable_path": process["executable_path"],
        "worker_executable_sha256": plan["worker_artifact"]["sha256"],
        "worker_pid": process["pid"],
        "worker_start_ticks": process["start_ticks"],
    }


def run_probe(
    plan: dict[str, Any],
    runner: Any,
    pid: int,
    start_ticks: int,
    network_pid: int,
    network_start_ticks: int,
    boot_id: str,
    capture_compatible: bool,
) -> dict[str, Any]:
    integer(pid, "runtime.pid", 1)
    integer(start_ticks, "runtime.start_ticks", 1)
    integer(network_pid, "network_runtime.pid", 1)
    integer(network_start_ticks, "network_runtime.start_ticks", 1)
    boot_id = validate_boot_id(boot_id)
    read_sealed_adb(plan["android"])
    started_ns = time.monotonic_ns()
    try:
        completed = runner.run(
            remote_argv(plan, pid, network_pid),
            timeout=360,
        )
    except subprocess.TimeoutExpired as error:
        raise ProbeError("E_ADB_TIMEOUT") from error
    require(
        type(completed.stdout) is bytes
        and type(completed.stderr) is bytes
        and len(completed.stdout) <= MAX_REMOTE_OUTPUT
        and len(completed.stderr) <= MAX_REMOTE_OUTPUT,
        "E_ADB_OUTPUT",
    )
    exact(completed.returncode, 0, "adb.returncode")
    exact(completed.stderr, b"", "adb.stderr")
    remote = parse_remote_snapshot(
        completed.stdout,
        plan,
        pid,
        start_ticks,
        network_pid,
        network_start_ticks,
        boot_id,
    )
    completed_ns = time.monotonic_ns()
    if capture_compatible:
        return capture_row(plan, remote)
    return {
        "adb_path": plan["android"]["adb_path"],
        "adb_port": ADB_PORT,
        "adb_selector": plan["android"]["adb_selector"],
        "adb_sha256": plan["android"]["adb_sha256"],
        "completed_ns": completed_ns,
        "remote": remote,
        "schema": OUTPUT_SCHEMA,
        "stage_status_source": "relay_owned_status",
        "started_ns": started_ns,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-json", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--start-ticks", type=int, required=True)
    parser.add_argument("--network-pid", type=int, required=True)
    parser.add_argument("--network-start-ticks", type=int, required=True)
    parser.add_argument("--boot-id", required=True)
    parser.add_argument("--capture-compatible", action="store_true")
    args = parser.parse_args(argv)
    try:
        plan = parse_plan_json(args.plan_json, args.plan_sha256)
        result = run_probe(
            plan,
            SubprocessRunner(),
            args.pid,
            args.start_ticks,
            args.network_pid,
            args.network_start_ticks,
            args.boot_id,
            args.capture_compatible,
        )
        sys.stdout.buffer.write(canonical_bytes(result))
        return 0
    except (
        OSError,
        ProbeError,
        subprocess.SubprocessError,
        ValueError,
    ) as error:
        print(f"PHONE_RUNTIME_PROBE_REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
