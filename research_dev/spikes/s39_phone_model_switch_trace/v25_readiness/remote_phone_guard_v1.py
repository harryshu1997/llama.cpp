#!/usr/bin/env python3
"""Validate phase-bound remote phone quiescence guard evidence."""

from __future__ import annotations

import argparse
import base64
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
from typing import Any


PLAN_SCHEMA = "s39-v25-remote-phone-guard-plan-v1"
RECEIPT_SCHEMA = "s39-v25-remote-phone-guard-receipt-v1"
REMOTE_OUTPUT_SCHEMA = "s39-v25-remote-phone-guard-output-v1"
DESKTOP_SNAPSHOT_SCHEMA = "s39-v25-remote-phone-guard-desktop-snapshot-v1"
PHONE_SNAPSHOT_SCHEMA = "s39-v25-remote-phone-guard-phone-snapshot-v1"
SSH_PROCESS_SCHEMA = "s39-v25-remote-phone-guard-ssh-process-v1"
SSH_CLEANUP_SCHEMA = "s39-v25-remote-phone-guard-ssh-cleanup-v1"
PHASE = "A_ONLY"
CONFIRMATION = "RUN_CP0_R1_V25_PHONE_GUARD"
RTX_CLOCK_NAME = "RTX_CLOCK_MONOTONIC_RAW"
CONTROLLER_CLOCK_NAME = "CONTROLLER_MONOTONIC"
GPU_UUID = "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08"
SSH_TARGET = "zhihao@172.20.74.85"
ADB_SERVER_PORT = 5038
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_COMMAND_BYTES = 8 * 1024 * 1024
MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
MAX_INT = (1 << 63) - 1

PHONE_SERIALS = {
    "op12": "5ae7a43d",
    "op15": "3C15AU002CL00000",
}
ARTIFACT_KEYS = {"bytes", "path", "sha256", "stat"}
STAT_KEYS = {
    "build_id",
    "ctime_ns",
    "device_id",
    "inode",
    "mode",
    "mtime_ns",
    "size",
}
ARTIFACT_SET_KEYS = {"adb", "helper", "python"}
PROCESS_POLICY_KEYS = {"executable_path", "sha256"}
ADB_SERVER_PLAN_KEYS = {
    "argv",
    "boot_id",
    "executable_path",
    "listen_host",
    "listen_port",
    "pid",
    "start_ticks",
}
ADB_SERVER_RECEIPT_KEYS = ADB_SERVER_PLAN_KEYS | {
    "executable_sha256",
    "listener_inode",
    "observed_ns",
}
PLAN_PHONE_KEYS = {
    "boot_id",
    "forbidden_listen_ports",
    "forbidden_processes",
    "interface",
    "physical_serial",
    "wifi_ipv4",
    "wifi_selector",
}
PLAN_KEYS = {
    "adb_server_port",
    "adb_server_process",
    "desktop_forbidden_listen_ports",
    "desktop_forbidden_processes",
    "forbid_adb_forward_for_selectors",
    "gpu_uuid",
    "inner_phase_id",
    "local_artifacts",
    "local_policy_artifact",
    "outer_phase_id",
    "phase",
    "phones",
    "remote_artifacts",
    "remote_policy",
    "remote_policy_artifact",
    "rtx_boot_id",
    "schema",
    "ssh_transport",
    "timeout_seconds",
}
SSH_TRANSPORT_KEYS = {
    "argv_by_moment",
    "connect_timeout_seconds",
    "host_key_alias",
    "identity_file",
    "known_hosts",
    "remote_python",
    "ssh",
    "ssh_port",
    "ssh_target",
}
REMOTE_POLICY_KEYS = {
    "adb_server_port",
    "adb_server_process",
    "desktop_forbidden_listen_ports",
    "desktop_forbidden_processes",
    "forbid_adb_forward_for_selectors",
    "gpu_uuid",
    "inner_phase_id",
    "outer_phase_id",
    "phase",
    "phones",
    "remote_artifacts",
    "rtx_boot_id",
    "schema",
}

_REMOTE_BOOTSTRAP_SOURCE = r"""
import hashlib,json,os,stat,sys
def die(message):
    raise SystemExit(message)
def stable(path):
    flags=os.O_RDONLY|os.O_CLOEXEC
    if hasattr(os,"O_NOFOLLOW"):
        flags|=os.O_NOFOLLOW
    fd=os.open(path,flags)
    try:
        a=os.fstat(fd); data=bytearray()
        while True:
            block=os.read(fd,1048576)
            if not block: break
            data.extend(block)
        b=os.fstat(fd)
    finally:
        os.close(fd)
    ident=lambda x:(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns,x.st_mode)
    if ident(a)!=ident(b) or not stat.S_ISREG(a.st_mode) or not 0<a.st_size<=536870912 or len(data)!=a.st_size:
        die("E_BOOTSTRAP_FILE")
    return bytes(data),a
policy_path,policy_sha,moment=sys.argv[1:4]
policy_raw,unused=stable(policy_path)
if hashlib.sha256(policy_raw).hexdigest()!=policy_sha:
    die("E_BOOTSTRAP_POLICY_SHA")
try:
    policy=json.loads(policy_raw.decode("ascii"))
except Exception:
    die("E_BOOTSTRAP_POLICY_JSON")
canonical=(json.dumps(policy,ensure_ascii=True,sort_keys=True,separators=(",",":"))+"\\n").encode("ascii")
if canonical!=policy_raw:
    die("E_BOOTSTRAP_POLICY_CANONICAL")
helper=policy["remote_artifacts"]["helper"]
helper_raw,helper_stat=stable(helper["path"])
observed={
    "build_id":None,
    "ctime_ns":helper_stat.st_ctime_ns,
    "device_id":helper_stat.st_dev,
    "inode":helper_stat.st_ino,
    "mode":helper_stat.st_mode,
    "mtime_ns":helper_stat.st_mtime_ns,
    "size":helper_stat.st_size,
}
if len(helper_raw)!=helper["bytes"] or hashlib.sha256(helper_raw).hexdigest()!=helper["sha256"] or observed!=helper["stat"]:
    die("E_BOOTSTRAP_HELPER")
sys.argv=[helper["path"],"--remote","--policy",policy_path,"--policy-sha256",policy_sha,"--moment",moment]
scope={"__name__":"__main__","__file__":helper["path"],"__package__":None}
exec(compile(helper_raw,helper["path"],"exec"),scope)
""".strip()
REMOTE_BOOTSTRAP = (
    "import base64;"
    "exec(compile(base64.b64decode("
    + repr(base64.b64encode(_REMOTE_BOOTSTRAP_SOURCE.encode("ascii")))
    + "),'<s39-v25-phone-guard-bootstrap>','exec'))"
)

_PHONE_PROC_SCAN_SOURCE = r"""
set -eu
target="$1"
expected_sha="$2"
self_uid="$(id -u)"
found=0
for proc in /proc/[0-9]*; do
    uid_line="$(grep '^Uid:' "$proc/status")"
    set -- $uid_line
    [ "$2" = "$self_uid" ] || continue
    if ! executable="$(readlink "$proc/exe")"; then
        printf 'ERROR\t%s\treadlink\n' "${proc##*/}"
        exit 17
    fi
    normalized="$executable"
    case "$normalized" in
        *" (deleted)") normalized="${normalized% (deleted)}" ;;
    esac
    [ "$normalized" = "$target" ] || continue
    pid="${proc##*/}"
    actual_sha="$(sha256sum "$proc/exe" | awk '{print $1}')"
    stat_hex="$(od -An -tx1 -v "$proc/stat" | tr -d ' \n')"
    cmd_hex="$(od -An -tx1 -v "$proc/cmdline" | tr -d ' \n')"
    [ -n "$stat_hex" ] && [ -n "$cmd_hex" ] || exit 18
    if [ "$actual_sha" != "$expected_sha" ]; then
        printf 'MISMATCH\t%s\t%s\t%s\n' "$pid" "$executable" "$actual_sha"
        exit 19
    fi
    printf 'MATCH\t%s\t%s\t%s\t%s\t%s\n' \
        "$pid" "$executable" "$actual_sha" "$stat_hex" "$cmd_hex"
    found=1
done
[ "$found" -eq 1 ] || printf 'OK\n'
""".strip()
PHONE_PROC_SCAN_SCRIPT = (
    "eval \"$(printf '%s' '"
    + base64.b64encode(_PHONE_PROC_SCAN_SOURCE.encode("ascii")).decode("ascii")
    + "' | toybox base64 -d)\""
)
RECEIPT_PHONE_KEYS = {
    "adb_forwards",
    "adb_state",
    "boot_id",
    "interface",
    "matching_listeners",
    "matching_processes",
    "physical_serial",
    "raw_snapshot_artifact",
    "wifi_ipv4",
    "wifi_selector",
}
SSH_PROCESS_KEYS = {
    "argv",
    "clock_name",
    "controller_boot_id",
    "observed_ns",
    "pid",
    "plan_sha256",
    "remote_boot_id",
    "schema",
    "start_ticks",
}
SSH_CLEANUP_KEYS = {
    "clock_name",
    "controller_boot_id",
    "observed_ns",
    "pid",
    "process_absent",
    "schema",
    "start_ticks",
}
RECEIPT_KEYS = {
    "adb_server_process",
    "clock_name",
    "completed_ns",
    "desktop_raw_snapshot_artifact",
    "desktop_matching_listeners",
    "desktop_matching_processes",
    "gpu_uuid",
    "inner_phase_id",
    "moment",
    "outer_phase_id",
    "phase",
    "phones",
    "plan_sha256",
    "remote_output_artifact",
    "rtx_boot_id",
    "schema",
    "ssh_transport_cleanup",
    "ssh_transport_process",
    "started_ns",
}

DIGEST_RE = re.compile(r"[0-9a-f]{64}")
UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}"
)
INTERFACE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
PHASE_SUFFIX_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}")


class GuardError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GuardError(message)


def exact(value: Any, expected: Any, field: str) -> None:
    require(
        type(value) is type(expected) and value == expected,
        f"E_VALUE: {field}",
    )


def exact_keys(value: Any, keys: set[str], field: str) -> dict[str, Any]:
    require(type(value) is dict, f"E_TYPE: {field}")
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual)
        unknown = sorted(repr(key) for key in actual - keys)
        raise GuardError(
            f"E_KEYS: {field}: missing={missing}, unknown={unknown}"
        )
    return value


def integer(
    value: Any,
    field: str,
    minimum: int = 0,
    maximum: int = MAX_INT,
) -> int:
    require(type(value) is int, f"E_TYPE: {field}")
    require(minimum <= value <= maximum, f"E_RANGE: {field}")
    return value


def text(value: Any, field: str, maximum: int = 32768) -> str:
    require(
        type(value) is str and 0 < len(value) <= maximum,
        f"E_TYPE: {field}",
    )
    require(
        value.isascii() and "\x00" not in value and "\n" not in value,
        f"E_ASCII: {field}",
    )
    return value


def digest(value: Any, field: str) -> str:
    value = text(value, field, 64)
    require(DIGEST_RE.fullmatch(value) is not None, f"E_DIGEST: {field}")
    return value


def uuid(value: Any, field: str) -> str:
    value = text(value, field, 64)
    require(UUID_RE.fullmatch(value) is not None, f"E_UUID: {field}")
    return value


def absolute_path(value: Any, field: str) -> str:
    value = text(value, field)
    path = Path(value)
    require(path.is_absolute() and ".." not in path.parts, f"E_PATH: {field}")
    return value


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
    except (TypeError, UnicodeEncodeError, ValueError) as error:
        raise GuardError("E_CANONICAL") from error


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"E_DUPLICATE_KEY: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise GuardError(f"E_JSON_NUMBER: {value}")


def parse_json(raw: bytes, field: str) -> Any:
    require(0 < len(raw) <= MAX_JSON_BYTES, f"E_SIZE: {field}")
    try:
        return json.loads(
            raw.decode("ascii"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GuardError(f"E_JSON: {field}") from error


def read_regular(
    path: Path,
    maximum_bytes: int = MAX_JSON_BYTES,
) -> bytes:
    require(path.is_absolute(), f"E_PATH: {path}")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise GuardError(f"E_READ: {path}: {error}") from error
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"E_FILE_TYPE: {path}")
        require(0 < before.st_size <= maximum_bytes, f"E_FILE_SIZE: {path}")
        raw = bytearray()
        while block := os.read(descriptor, 1024 * 1024):
            raw.extend(block)
            require(len(raw) <= maximum_bytes, f"E_FILE_SIZE: {path}")
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


def validate_artifact(
    value: Any,
    field: str,
    *,
    executable: bool = False,
) -> dict[str, Any]:
    value = exact_keys(value, ARTIFACT_KEYS, field)
    size = integer(value["bytes"], f"{field}.bytes", 1)
    absolute_path(value["path"], f"{field}.path")
    digest(value["sha256"], f"{field}.sha256")
    metadata = validate_stat(value["stat"], f"{field}.stat")
    exact(metadata["size"], size, f"{field}.stat.size")
    if executable:
        require(metadata["mode"] & 0o111 != 0, f"E_EXECUTABLE: {field}")
    return value


def validate_artifact_set(value: Any, field: str) -> dict[str, Any]:
    value = exact_keys(value, ARTIFACT_SET_KEYS, field)
    for name in sorted(ARTIFACT_SET_KEYS):
        validate_artifact(
            value[name],
            f"{field}.{name}",
            executable=name in {"adb", "python"},
        )
    paths = [value[name]["path"] for name in sorted(ARTIFACT_SET_KEYS)]
    require(len(paths) == len(set(paths)), f"E_ARTIFACT_PATH_REUSE: {field}")
    return value


def verify_local_artifact(
    value: dict[str, Any],
    field: str,
) -> None:
    value = validate_artifact(value, field)
    path = Path(value["path"])
    raw = read_regular(path, MAX_ARTIFACT_BYTES)
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


def expected_ssh_argv(
    plan: dict[str, Any],
    moment: str,
) -> list[str]:
    require(moment in {"before", "after"}, "E_MOMENT")
    transport = plan["ssh_transport"]
    return [
        transport["ssh"]["path"],
        "-T",
        "-F",
        "/dev/null",
        "-p",
        str(transport["ssh_port"]),
        "-i",
        transport["identity_file"]["path"],
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={transport['connect_timeout_seconds']}",
        "-o",
        f"HostKeyAlias={transport['host_key_alias']}",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={transport['known_hosts']['path']}",
        "-o",
        "GlobalKnownHostsFile=/dev/null",
        "-o",
        "PasswordAuthentication=no",
        "-o",
        "KbdInteractiveAuthentication=no",
        "-o",
        "IdentityAgent=none",
        "-o",
        "LogLevel=ERROR",
        transport["ssh_target"],
        transport["remote_python"]["path"],
        "-I",
        "-c",
        REMOTE_BOOTSTRAP,
        plan["remote_policy_artifact"]["path"],
        plan["remote_policy_artifact"]["sha256"],
        moment,
    ]


def expected_remote_policy(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "adb_server_port": plan["adb_server_port"],
        "adb_server_process": plan["adb_server_process"],
        "desktop_forbidden_listen_ports": plan[
            "desktop_forbidden_listen_ports"
        ],
        "desktop_forbidden_processes": plan[
            "desktop_forbidden_processes"
        ],
        "forbid_adb_forward_for_selectors": plan[
            "forbid_adb_forward_for_selectors"
        ],
        "gpu_uuid": plan["gpu_uuid"],
        "inner_phase_id": plan["inner_phase_id"],
        "outer_phase_id": plan["outer_phase_id"],
        "phase": plan["phase"],
        "phones": plan["phones"],
        "remote_artifacts": plan["remote_artifacts"],
        "rtx_boot_id": plan["rtx_boot_id"],
        "schema": "s39-v25-remote-phone-guard-policy-v1",
    }


def validate_adb_server_plan(
    value: Any,
    adb_artifact: dict[str, Any],
    boot_id: str,
    field: str,
) -> dict[str, Any]:
    value = exact_keys(value, ADB_SERVER_PLAN_KEYS, field)
    argv = value["argv"]
    require(
        type(argv) is list and 1 <= len(argv) <= 64,
        f"E_ARGV: {field}",
    )
    for index, item in enumerate(argv):
        text(item, f"{field}.argv[{index}]")
    require(len(argv) == 7, f"E_ADB_SERVER_ARGV: {field}")
    exact(argv[0], Path(adb_artifact["path"]).name, f"{field}.argv[0]")
    exact(argv[1:6], [
        "-L",
        f"tcp:{ADB_SERVER_PORT}",
        "fork-server",
        "server",
        "--reply-fd",
    ], f"{field}.argv")
    reply_fd = argv[6]
    require(
        reply_fd.isdigit()
        and str(int(reply_fd)) == reply_fd
        and 0 <= int(reply_fd) <= 1_048_575,
        f"E_ADB_SERVER_REPLY_FD: {field}",
    )
    exact(value["boot_id"], boot_id, f"{field}.boot_id")
    exact(
        value["executable_path"],
        adb_artifact["path"],
        f"{field}.executable_path",
    )
    exact(value["listen_host"], "127.0.0.1", f"{field}.listen_host")
    exact(value["listen_port"], ADB_SERVER_PORT, f"{field}.listen_port")
    integer(value["pid"], f"{field}.pid", 1)
    integer(value["start_ticks"], f"{field}.start_ticks", 1)
    return value


def validate_phase_ids(
    outer_phase_id: Any,
    inner_phase_id: Any,
    field: str,
) -> tuple[str, str]:
    outer = text(outer_phase_id, f"{field}.outer_phase_id", 128)
    inner = text(inner_phase_id, f"{field}.inner_phase_id", 128)
    outer_prefix = "cp0-r1-v25-a-only-"
    inner_prefix = "cp0-r1-v24-a-only-"
    require(outer.startswith(outer_prefix), f"E_OUTER_PHASE_ID: {field}")
    require(inner.startswith(inner_prefix), f"E_INNER_PHASE_ID: {field}")
    outer_suffix = outer[len(outer_prefix):]
    inner_suffix = inner[len(inner_prefix):]
    require(
        PHASE_SUFFIX_RE.fullmatch(outer_suffix) is not None
        and outer_suffix == inner_suffix,
        f"E_PHASE_ID_LINK: {field}",
    )
    return outer, inner


def validate_ipv4(value: Any, field: str) -> str:
    value = text(value, field, 15)
    try:
        address = ipaddress.IPv4Address(value)
    except ipaddress.AddressValueError as error:
        raise GuardError(f"E_IPV4: {field}") from error
    require(
        str(address) == value
        and not address.is_unspecified
        and not address.is_loopback
        and not address.is_link_local
        and not address.is_multicast,
        f"E_IPV4: {field}",
    )
    return value


def validate_phone_identity(
    value: dict[str, Any],
    phone: str,
    field: str,
) -> None:
    exact(
        value["physical_serial"],
        PHONE_SERIALS[phone],
        f"{field}.physical_serial",
    )
    uuid(value["boot_id"], f"{field}.boot_id")
    ipv4 = validate_ipv4(value["wifi_ipv4"], f"{field}.wifi_ipv4")
    exact(value["wifi_selector"], f"{ipv4}:5555", f"{field}.wifi_selector")
    interface = text(value["interface"], f"{field}.interface", 64)
    require(
        INTERFACE_RE.fullmatch(interface) is not None,
        f"E_INTERFACE: {field}",
    )


def validate_forbidden_processes(value: Any, field: str) -> None:
    require(
        type(value) is list and 0 < len(value) <= 128,
        f"E_PROCESS_POLICY: {field}",
    )
    rows: list[tuple[str, str]] = []
    for index, item in enumerate(value):
        item_field = f"{field}[{index}]"
        item = exact_keys(item, PROCESS_POLICY_KEYS, item_field)
        rows.append((
            absolute_path(item["executable_path"], f"{item_field}.executable_path"),
            digest(item["sha256"], f"{item_field}.sha256"),
        ))
    require(rows == sorted(rows), f"E_PROCESS_POLICY_ORDER: {field}")
    require(
        len({path for path, _sha256 in rows}) == len(rows),
        f"E_PROCESS_POLICY_REUSE: {field}",
    )


def validate_forbidden_ports(value: Any, field: str) -> None:
    require(
        type(value) is list and 0 < len(value) <= 128,
        f"E_PORT_POLICY: {field}",
    )
    ports = [
        integer(item, f"{field}[{index}]", 1, 65535)
        for index, item in enumerate(value)
    ]
    require(
        ports == sorted(set(ports)),
        f"E_PORT_POLICY_ORDER: {field}",
    )


def validate_plan(value: Any) -> dict[str, Any]:
    value = exact_keys(value, PLAN_KEYS, "plan")
    exact(value["schema"], PLAN_SCHEMA, "plan.schema")
    exact(value["phase"], PHASE, "plan.phase")
    validate_phase_ids(
        value["outer_phase_id"],
        value["inner_phase_id"],
        "plan",
    )
    rtx_boot_id = uuid(value["rtx_boot_id"], "plan.rtx_boot_id")
    exact(value["gpu_uuid"], GPU_UUID, "plan.gpu_uuid")
    exact(value["adb_server_port"], ADB_SERVER_PORT, "plan.adb_server_port")
    exact(
        value["forbid_adb_forward_for_selectors"],
        True,
        "plan.forbid_adb_forward_for_selectors",
    )
    timeout = integer(value["timeout_seconds"], "plan.timeout_seconds", 1)
    require(timeout <= 600, "E_TIMEOUT")
    validate_forbidden_processes(
        value["desktop_forbidden_processes"],
        "plan.desktop_forbidden_processes",
    )
    validate_forbidden_ports(
        value["desktop_forbidden_listen_ports"],
        "plan.desktop_forbidden_listen_ports",
    )

    local = validate_artifact_set(value["local_artifacts"], "plan.local_artifacts")
    remote = validate_artifact_set(
        value["remote_artifacts"],
        "plan.remote_artifacts",
    )
    exact(
        (remote["helper"]["bytes"], remote["helper"]["sha256"]),
        (local["helper"]["bytes"], local["helper"]["sha256"]),
        "plan.helper_content",
    )
    validate_adb_server_plan(
        value["adb_server_process"],
        remote["adb"],
        rtx_boot_id,
        "plan.adb_server_process",
    )
    transport = exact_keys(
        value["ssh_transport"],
        SSH_TRANSPORT_KEYS,
        "plan.ssh_transport",
    )
    ssh = validate_artifact(
        transport["ssh"],
        "plan.ssh_transport.ssh",
        executable=True,
    )
    identity = validate_artifact(
        transport["identity_file"],
        "plan.ssh_transport.identity_file",
    )
    known_hosts = validate_artifact(
        transport["known_hosts"],
        "plan.ssh_transport.known_hosts",
    )
    remote_python = validate_artifact(
        transport["remote_python"],
        "plan.ssh_transport.remote_python",
        executable=True,
    )
    exact(Path(ssh["path"]).name, "ssh", "plan.ssh_transport.ssh.name")
    require(
        stat.S_IMODE(identity["stat"]["mode"]) == 0o600,
        "E_IDENTITY_MODE",
    )
    require(
        known_hosts["stat"]["mode"] & 0o222 == 0,
        "E_KNOWN_HOSTS_WRITABLE",
    )
    exact(
        transport["ssh_target"],
        SSH_TARGET,
        "plan.ssh_transport.ssh_target",
    )
    exact(
        transport["host_key_alias"],
        "172.20.74.85",
        "plan.ssh_transport.host_key_alias",
    )
    exact(transport["ssh_port"], 22, "plan.ssh_transport.ssh_port")
    timeout = integer(
        transport["connect_timeout_seconds"],
        "plan.ssh_transport.connect_timeout_seconds",
        1,
    )
    require(timeout <= 60, "E_SSH_TIMEOUT")
    exact(
        remote_python,
        remote["python"],
        "plan.ssh_transport.remote_python",
    )
    argv_by_moment = exact_keys(
        transport["argv_by_moment"],
        {"after", "before"},
        "plan.ssh_transport.argv_by_moment",
    )
    for moment in ("before", "after"):
        validate_argv(
            argv_by_moment[moment],
            f"plan.ssh_transport.argv_by_moment.{moment}",
        )
        exact(
            argv_by_moment[moment],
            expected_ssh_argv(value, moment),
            f"plan.ssh_transport.argv_by_moment.{moment}",
        )

    phones = exact_keys(value["phones"], set(PHONE_SERIALS), "plan.phones")
    selectors: set[str] = set()
    ipv4s: set[str] = set()
    boot_ids: set[str] = {rtx_boot_id}
    for phone in sorted(PHONE_SERIALS):
        field = f"plan.phones.{phone}"
        row = exact_keys(phones[phone], PLAN_PHONE_KEYS, field)
        validate_phone_identity(row, phone, field)
        validate_forbidden_processes(
            row["forbidden_processes"],
            f"{field}.forbidden_processes",
        )
        validate_forbidden_ports(
            row["forbidden_listen_ports"],
            f"{field}.forbidden_listen_ports",
        )
        require(row["wifi_selector"] not in selectors, "E_SELECTOR_REUSE")
        require(row["wifi_ipv4"] not in ipv4s, "E_IPV4_REUSE")
        require(row["boot_id"] not in boot_ids, "E_BOOT_ID_REUSE")
        selectors.add(row["wifi_selector"])
        ipv4s.add(row["wifi_ipv4"])
        boot_ids.add(row["boot_id"])
    policy = exact_keys(
        value["remote_policy"],
        REMOTE_POLICY_KEYS,
        "plan.remote_policy",
    )
    exact(
        policy,
        expected_remote_policy(value),
        "plan.remote_policy",
    )
    policy_raw = canonical_bytes(policy)
    local_policy = validate_artifact(
        value["local_policy_artifact"],
        "plan.local_policy_artifact",
    )
    remote_policy = validate_artifact(
        value["remote_policy_artifact"],
        "plan.remote_policy_artifact",
    )
    exact(local_policy["bytes"], len(policy_raw), "plan.local_policy.bytes")
    exact(
        local_policy["sha256"],
        hashlib.sha256(policy_raw).hexdigest(),
        "plan.local_policy.sha256",
    )
    exact(
        (remote_policy["bytes"], remote_policy["sha256"]),
        (local_policy["bytes"], local_policy["sha256"]),
        "plan.remote_policy_artifact.content",
    )
    require(
        local_policy["path"] != remote_policy["path"],
        "E_POLICY_PATH_REUSE",
    )
    return value


def parse_plan(
    path: Path | str,
    expected_sha256: str,
) -> tuple[dict[str, Any], bytes]:
    digest(expected_sha256, "plan_sha256")
    plan_path = Path(path)
    raw = read_regular(plan_path)
    value = parse_json(raw, str(plan_path))
    require(type(value) is dict, f"E_TYPE: {plan_path}")
    require(canonical_bytes(value) == raw, f"E_CANONICAL: {plan_path}")
    exact(
        hashlib.sha256(raw).hexdigest(),
        expected_sha256,
        "plan.sha256",
    )
    return validate_plan(value), raw


def validate_remote_policy(value: Any) -> dict[str, Any]:
    value = exact_keys(value, REMOTE_POLICY_KEYS, "remote_policy")
    exact(
        value["schema"],
        "s39-v25-remote-phone-guard-policy-v1",
        "remote_policy.schema",
    )
    exact(value["phase"], PHASE, "remote_policy.phase")
    validate_phase_ids(
        value["outer_phase_id"],
        value["inner_phase_id"],
        "remote_policy",
    )
    uuid(value["rtx_boot_id"], "remote_policy.rtx_boot_id")
    exact(value["gpu_uuid"], GPU_UUID, "remote_policy.gpu_uuid")
    exact(
        value["adb_server_port"],
        ADB_SERVER_PORT,
        "remote_policy.adb_server_port",
    )
    exact(
        value["forbid_adb_forward_for_selectors"],
        True,
        "remote_policy.forbid_adb_forward_for_selectors",
    )
    validate_forbidden_processes(
        value["desktop_forbidden_processes"],
        "remote_policy.desktop_forbidden_processes",
    )
    validate_forbidden_ports(
        value["desktop_forbidden_listen_ports"],
        "remote_policy.desktop_forbidden_listen_ports",
    )
    remote_artifacts = validate_artifact_set(
        value["remote_artifacts"],
        "remote_policy.remote_artifacts",
    )
    validate_adb_server_plan(
        value["adb_server_process"],
        remote_artifacts["adb"],
        value["rtx_boot_id"],
        "remote_policy.adb_server_process",
    )
    phones = exact_keys(
        value["phones"],
        set(PHONE_SERIALS),
        "remote_policy.phones",
    )
    selectors: set[str] = set()
    boot_ids: set[str] = {value["rtx_boot_id"]}
    for phone in sorted(PHONE_SERIALS):
        field = f"remote_policy.phones.{phone}"
        row = exact_keys(phones[phone], PLAN_PHONE_KEYS, field)
        validate_phone_identity(row, phone, field)
        validate_forbidden_processes(
            row["forbidden_processes"],
            f"{field}.forbidden_processes",
        )
        validate_forbidden_ports(
            row["forbidden_listen_ports"],
            f"{field}.forbidden_listen_ports",
        )
        require(row["wifi_selector"] not in selectors, "E_SELECTOR_REUSE")
        require(row["boot_id"] not in boot_ids, "E_BOOT_ID_REUSE")
        selectors.add(row["wifi_selector"])
        boot_ids.add(row["boot_id"])
    return value


def read_remote_policy(
    path: Path,
    expected_sha256: str,
) -> tuple[dict[str, Any], bytes]:
    digest(expected_sha256, "remote_policy.sha256")
    raw = read_regular(path)
    exact(
        hashlib.sha256(raw).hexdigest(),
        expected_sha256,
        "remote_policy.sha256",
    )
    value = parse_json(raw, "remote_policy")
    require(type(value) is dict, "E_TYPE: remote_policy")
    require(canonical_bytes(value) == raw, "E_CANONICAL: remote_policy")
    return validate_remote_policy(value), raw


def artifact_from_path(path: Path, field: str) -> dict[str, Any]:
    raw = read_regular(path)
    metadata = path.stat()
    value = {
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
    return validate_artifact(value, field)


def write_exclusive(path: Path, raw: bytes) -> dict[str, Any]:
    require(path.is_absolute(), f"E_PATH: {path}")
    require(0 < len(raw) <= MAX_JSON_BYTES, f"E_FILE_SIZE: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            require(written > 0, f"E_WRITE: {path}")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | os.O_CLOEXEC)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    observed = read_regular(path)
    exact(observed, raw, f"E_REOPEN: {path}")
    return artifact_from_path(path, str(path))


def raw_clock_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def process_start_ticks(pid: int) -> int:
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    closing = raw.rfind(")")
    require(closing > 0, "E_PROCESS_STAT")
    fields = raw[closing + 2:].split()
    require(len(fields) > 19, "E_PROCESS_STAT")
    return integer(int(fields[19]), "process.start_ticks", 1)


def process_absent(pid: int, start_ticks: int) -> bool:
    try:
        observed = process_start_ticks(pid)
    except (FileNotFoundError, ProcessLookupError):
        return True
    return observed != start_ticks


def process_group_absent(pgid: int) -> bool:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text(encoding="ascii")
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as error:
            raise GuardError("E_PROCESS_STAT_PERMISSION") from error
        closing = raw.rfind(")")
        require(closing > 0, "E_PROCESS_STAT")
        fields = raw[closing + 2:].split()
        require(len(fields) > 2, "E_PROCESS_STAT")
        try:
            observed_pgid = int(fields[2])
        except ValueError as error:
            raise GuardError("E_PROCESS_STAT") from error
        if observed_pgid == pgid:
            return False
    return True


def read_process_executable(pid: int, expected_ticks: int) -> bytes:
    exact(process_start_ticks(pid), expected_ticks, "process.start_ticks.before")
    descriptor = os.open(
        f"/proc/{pid}/exe",
        os.O_RDONLY | os.O_CLOEXEC,
    )
    try:
        before = os.fstat(descriptor)
        require(
            stat.S_ISREG(before.st_mode)
            and 0 < before.st_size <= 512 * 1024 * 1024,
            "E_PROCESS_EXECUTABLE",
        )
        raw = bytearray()
        while block := os.read(descriptor, 1024 * 1024):
            raw.extend(block)
            require(
                len(raw) <= 512 * 1024 * 1024,
                "E_PROCESS_EXECUTABLE_SIZE",
            )
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
    require(identity(before) == identity(after), "E_PROCESS_EXECUTABLE_CHANGED")
    exact(process_start_ticks(pid), expected_ticks, "process.start_ticks.after")
    return bytes(raw)


def terminate_process_group(
    process: subprocess.Popen,
    timeout_seconds: int = 5,
) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        if process.poll() is None:
            process.wait(timeout=timeout_seconds)
        return
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process_group_absent(process.pid):
            if process.poll() is None:
                process.wait(timeout=timeout_seconds)
            return
        time.sleep(0.01)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if process.poll() is None:
        process.wait(timeout=timeout_seconds)


def default_command_runner(
    argv: list[str],
    timeout_seconds: int,
) -> dict[str, Any]:
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
    ticks = process_start_ticks(process.pid)
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        terminate_process_group(process)
        require(
            process_absent(process.pid, ticks),
            "E_COMMAND_TIMEOUT_ORPHAN",
        )
        raise GuardError("E_COMMAND_TIMEOUT") from error
    except Exception:
        terminate_process_group(process)
        raise
    command_absent = process_absent(process.pid, ticks)
    group_absent = process_group_absent(process.pid)
    if not command_absent or not group_absent:
        terminate_process_group(process)
    require(
        process_absent(process.pid, ticks),
        "E_COMMAND_PROCESS_LIVE",
    )
    require(process_group_absent(process.pid), "E_COMMAND_PROCESS_GROUP_LIVE")
    require(
        len(stdout) <= MAX_COMMAND_BYTES
        and len(stderr) <= MAX_COMMAND_BYTES,
        "E_COMMAND_OUTPUT_SIZE",
    )
    return {
        "argv": argv,
        "returncode": process.returncode,
        "stderr": stderr,
        "stdout": stdout,
    }


def command_record(
    command_runner,
    argv: list[str],
    timeout_seconds: int,
    field: str,
    *,
    require_empty_stderr: bool = True,
) -> tuple[dict[str, Any], bytes]:
    result = command_runner(argv, timeout_seconds)
    result = exact_keys(
        result,
        {"argv", "returncode", "stderr", "stdout"},
        field,
    )
    exact(result["argv"], argv, f"{field}.argv")
    exact(result["returncode"], 0, f"{field}.returncode")
    require(type(result["stdout"]) is bytes, f"E_TYPE: {field}.stdout")
    require(type(result["stderr"]) is bytes, f"E_TYPE: {field}.stderr")
    require(
        len(result["stdout"]) <= MAX_COMMAND_BYTES
        and len(result["stderr"]) <= MAX_COMMAND_BYTES,
        f"E_COMMAND_OUTPUT_SIZE: {field}",
    )
    if require_empty_stderr:
        exact(result["stderr"], b"", f"{field}.stderr")
    record = {
        "argv": argv,
        "returncode": 0,
        "stderr_base64": base64.b64encode(result["stderr"]).decode("ascii"),
        "stdout_base64": base64.b64encode(result["stdout"]).decode("ascii"),
    }
    return record, result["stdout"]


def read_boot_id() -> str:
    return uuid(
        Path("/proc/sys/kernel/random/boot_id")
        .read_text(encoding="ascii")
        .strip(),
        "runtime.boot_id",
    )


def read_gpu_uuids() -> list[str]:
    result = []
    root = Path("/proc/driver/nvidia/gpus")
    for path in sorted(root.glob("*/information")):
        for line in path.read_text(encoding="ascii").splitlines():
            if line.startswith("GPU UUID:"):
                result.append(line.split(":", 1)[1].strip())
    return sorted(set(result))


def scan_matching_processes(
    policies: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expected = {
        row["executable_path"]: row["sha256"]
        for row in policies
    }
    result = []
    for entry in sorted(Path("/proc").iterdir(), key=lambda path: path.name):
        if not entry.name.isdigit():
            continue
        try:
            executable = os.readlink(entry / "exe")
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as error:
            raise GuardError("E_PROCESS_EXE_PERMISSION") from error
        except OSError as error:
            raise GuardError("E_PROCESS_EXE") from error
        normalized = (
            executable[:-10]
            if executable.endswith(" (deleted)")
            else executable
        )
        if normalized not in expected:
            continue
        try:
            ticks = process_start_ticks(int(entry.name))
            raw = read_process_executable(int(entry.name), ticks)
            observed_sha256 = hashlib.sha256(raw).hexdigest()
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (OSError, GuardError, ValueError):
            observed_sha256 = None
            try:
                ticks = process_start_ticks(int(entry.name))
            except (FileNotFoundError, ProcessLookupError):
                continue
            except (OSError, GuardError, ValueError) as error:
                raise GuardError("E_PROCESS_IDENTITY") from error
        result.append({
            "executable_path": executable,
            "expected_sha256": expected[normalized],
            "observed_sha256": observed_sha256,
            "pid": int(entry.name),
            "start_ticks": ticks,
        })
    return result


def parse_listener_table(
    raw: bytes,
    forbidden_ports: list[int],
) -> list[dict[str, Any]]:
    try:
        text_value = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise GuardError("E_LISTENER_ASCII") from error
    forbidden = set(forbidden_ports)
    result = []
    headers = 0
    for line in text_value.splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "sl":
            require(
                len(fields) >= 10
                and fields[1:4] == ["local_address", "rem_address", "st"],
                "E_LISTENER_HEADER",
            )
            headers += 1
            continue
        require(len(fields) >= 10, "E_LISTENER_TRUNCATED")
        local = fields[1]
        state = fields[3]
        require(
            re.fullmatch(r"[0-9A-Fa-f]{8}(?:[0-9A-Fa-f]{24})?:[0-9A-Fa-f]{4}",
                         local) is not None
            and re.fullmatch(r"[0-9A-Fa-f]{2}", state) is not None,
            "E_LISTENER_PARSE",
        )
        if state != "0A":
            continue
        try:
            port = int(local.rsplit(":", 1)[1], 16)
            inode = int(fields[9])
        except ValueError as error:
            raise GuardError("E_LISTENER_PARSE") from error
        if port in forbidden:
            result.append({
                "inode": inode,
                "local_address_hex": local,
                "port": port,
            })
    require(headers >= 1, "E_LISTENER_HEADER_MISSING")
    return sorted(
        result,
        key=lambda row: (row["port"], row["inode"], row["local_address_hex"]),
    )


def parse_proc_net_ip(local_address_hex: str) -> str:
    value = text(local_address_hex, "listener.local_address_hex", 32)
    try:
        raw = bytes.fromhex(value)
    except ValueError as error:
        raise GuardError("E_LISTENER_ADDRESS") from error
    if len(raw) == 4:
        return str(ipaddress.IPv4Address(raw[::-1]))
    if len(raw) == 16:
        normalized = b"".join(
            raw[offset:offset + 4][::-1]
            for offset in range(0, len(raw), 4)
        )
        return str(ipaddress.IPv6Address(normalized))
    raise GuardError("E_LISTENER_ADDRESS")


def scan_matching_listeners(
    forbidden_ports: list[int],
) -> list[dict[str, Any]]:
    raw = bytearray()
    for path in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        try:
            raw.extend(path.read_bytes())
            raw.extend(b"\n")
        except FileNotFoundError:
            continue
    return parse_listener_table(bytes(raw), forbidden_ports)


def listener_owners(inode: int) -> list[dict[str, Any]]:
    result = []
    target = f"socket:[{inode}]"
    for entry in sorted(Path("/proc").iterdir(), key=lambda path: path.name):
        if not entry.name.isdigit():
            continue
        fd_root = entry / "fd"
        try:
            descriptors = list(fd_root.iterdir())
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as error:
            raise GuardError("E_LISTENER_OWNER_PERMISSION") from error
        except OSError as error:
            raise GuardError("E_LISTENER_OWNER") from error
        owns = False
        for descriptor in descriptors:
            try:
                if os.readlink(descriptor) == target:
                    owns = True
                    break
            except (FileNotFoundError, ProcessLookupError):
                continue
            except PermissionError as error:
                raise GuardError("E_LISTENER_FD_PERMISSION") from error
            except OSError as error:
                raise GuardError("E_LISTENER_FD") from error
        if not owns:
            continue
        try:
            executable = os.readlink(entry / "exe")
            ticks = process_start_ticks(int(entry.name))
            raw = read_process_executable(int(entry.name), ticks)
            cmdline_raw = Path(f"/proc/{entry.name}/cmdline").read_bytes()
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (OSError, GuardError, ValueError) as error:
            raise GuardError("E_LISTENER_OWNER_IDENTITY") from error
        require(
            bool(cmdline_raw) and cmdline_raw.endswith(b"\0"),
            "E_PROCESS_CMDLINE",
        )
        try:
            argv = [
                item.decode("ascii")
                for item in cmdline_raw[:-1].split(b"\0")
            ]
        except UnicodeDecodeError as error:
            raise GuardError("E_PROCESS_CMDLINE_ASCII") from error
        result.append({
            "argv": argv,
            "executable_path": executable,
            "executable_sha256": hashlib.sha256(raw).hexdigest(),
            "pid": int(entry.name),
            "start_ticks": ticks,
        })
    return result


def observe_adb_server(
    adb_artifact: dict[str, Any],
    expected: dict[str, Any],
    clock=raw_clock_ns,
) -> dict[str, Any]:
    port = expected["listen_port"]
    listeners = scan_matching_listeners([port])
    require(len(listeners) == 1, "E_ADB_SERVER_LISTENER")
    local_address_hex = listeners[0]["local_address_hex"].rsplit(":", 1)[0]
    exact(
        parse_proc_net_ip(local_address_hex),
        expected["listen_host"],
        "adb_server.listen_host",
    )
    owners = listener_owners(listeners[0]["inode"])
    require(len(owners) == 1, "E_ADB_SERVER_OWNER")
    owner = owners[0]
    normalized = (
        owner["executable_path"][:-10]
        if owner["executable_path"].endswith(" (deleted)")
        else owner["executable_path"]
    )
    exact(normalized, adb_artifact["path"], "adb_server.executable_path")
    exact(
        owner["executable_sha256"],
        adb_artifact["sha256"],
        "adb_server.executable_sha256",
    )
    exact(owner["pid"], expected["pid"], "adb_server.pid")
    exact(owner["start_ticks"], expected["start_ticks"], "adb_server.start_ticks")
    exact(owner["argv"], expected["argv"], "adb_server.argv")
    exact(expected["boot_id"], read_boot_id(), "adb_server.boot_id")
    return {
        "argv": owner["argv"],
        "boot_id": expected["boot_id"],
        "executable_path": normalized,
        "executable_sha256": owner["executable_sha256"],
        "listen_host": expected["listen_host"],
        "listen_port": port,
        "listener_inode": listeners[0]["inode"],
        "observed_ns": clock(),
        "pid": owner["pid"],
        "start_ticks": owner["start_ticks"],
    }


def decode_ascii_line(raw: bytes, field: str) -> str:
    try:
        value = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise GuardError(f"E_ASCII: {field}") from error
    require(value.endswith("\n"), f"E_LINE_END: {field}")
    lines = value.splitlines()
    require(len(lines) == 1 and bool(lines[0]), f"E_LINE: {field}")
    return lines[0]


def parse_interface_ipv4(
    raw: bytes,
    interface: str,
    field: str,
) -> str:
    try:
        value = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise GuardError(f"E_ASCII: {field}") from error
    matches = re.findall(r"(?:^|\s)inet ([0-9.]+)/[0-9]+(?:\s|$)", value)
    require(len(matches) == 1, f"E_INTERFACE_IPV4: {field}")
    observed = validate_ipv4(matches[0], field)
    require(
        re.search(rf"\b{re.escape(interface)}\b", value) is not None,
        f"E_INTERFACE_NAME: {field}",
    )
    return observed


def parse_adb_forwards(raw: bytes) -> list[dict[str, str]]:
    try:
        value = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise GuardError("E_ADB_FORWARD_ASCII") from error
    result = []
    for index, line in enumerate(value.splitlines()):
        fields = line.split()
        require(len(fields) == 3, f"E_ADB_FORWARD_ROW: {index}")
        result.append({
            "local": fields[1],
            "remote": fields[2],
            "serial": fields[0],
        })
    return sorted(
        result,
        key=lambda row: (row["serial"], row["local"], row["remote"]),
    )


def parse_phone_proc_scan(
    raw: bytes,
    policy: dict[str, Any],
    field: str,
) -> list[dict[str, Any]]:
    try:
        value = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise GuardError(f"E_PHONE_PROC_ASCII: {field}") from error
    lines = value.splitlines()
    require(bool(lines), f"E_PHONE_PROC_EMPTY: {field}")
    if lines == ["OK"]:
        return []
    result = []
    for index, line in enumerate(lines):
        fields = line.split("\t")
        require(len(fields) == 6, f"E_PHONE_PROC_ROW: {field}:{index}")
        exact(fields[0], "MATCH", f"{field}[{index}].kind")
        try:
            pid = int(fields[1])
            stat_raw = bytes.fromhex(fields[4])
            cmdline_raw = bytes.fromhex(fields[5])
        except ValueError as error:
            raise GuardError(f"E_PHONE_PROC_ENCODING: {field}:{index}") from error
        require(pid > 0, f"E_PHONE_PROC_PID: {field}:{index}")
        executable = fields[2]
        normalized = (
            executable[:-10]
            if executable.endswith(" (deleted)")
            else executable
        )
        exact(
            normalized,
            policy["executable_path"],
            f"{field}[{index}].executable",
        )
        exact(
            fields[3],
            policy["sha256"],
            f"{field}[{index}].sha256",
        )
        try:
            stat_text = stat_raw.decode("ascii")
        except UnicodeDecodeError as error:
            raise GuardError(f"E_PHONE_PROC_STAT: {field}:{index}") from error
        closing = stat_text.rfind(")")
        require(closing > 0, f"E_PHONE_PROC_STAT: {field}:{index}")
        stat_fields = stat_text[closing + 2:].split()
        require(
            len(stat_fields) > 19,
            f"E_PHONE_PROC_STAT: {field}:{index}",
        )
        try:
            start_ticks = int(stat_fields[19])
        except ValueError as error:
            raise GuardError(
                f"E_PHONE_PROC_START_TICKS: {field}:{index}"
            ) from error
        require(
            bool(cmdline_raw) and cmdline_raw.endswith(b"\0"),
            f"E_PHONE_PROC_CMDLINE: {field}:{index}",
        )
        result.append({
            "cmdline_base64": base64.b64encode(cmdline_raw).decode("ascii"),
            "executable_path": executable,
            "executable_sha256": fields[3],
            "pid": pid,
            "start_ticks": start_ticks,
        })
    return sorted(
        result,
        key=lambda row: (row["executable_path"], row["pid"]),
    )


def expected_phone_commands(
    policy: dict[str, Any],
    phone: str,
) -> list[tuple[str, list[str]]]:
    expected = policy["phones"][phone]
    prefix = [
        policy["remote_artifacts"]["adb"]["path"],
        "-P",
        str(policy["adb_server_port"]),
        "-s",
        expected["wifi_selector"],
    ]
    commands = [
        ("state", prefix + ["get-state"]),
        (
            "boot_id",
            prefix
            + ["shell", "cat", "/proc/sys/kernel/random/boot_id"],
        ),
        (
            "serial",
            prefix + ["shell", "getprop", "ro.serialno"],
        ),
        (
            "interface",
            prefix
            + [
                "shell",
                "ip",
                "-o",
                "-4",
                "addr",
                "show",
                "dev",
                expected["interface"],
            ],
        ),
    ]
    for index, process_policy in enumerate(expected["forbidden_processes"]):
        commands.append((
            f"processes_{index}",
            prefix
            + [
                "shell",
                "sh",
                "-c",
                PHONE_PROC_SCAN_SCRIPT,
                "sh",
                process_policy["executable_path"],
                process_policy["sha256"],
            ],
        ))
    commands.append((
        "listeners",
        prefix
        + [
            "shell",
            "cat",
            "/proc/net/tcp",
            "/proc/net/tcp6",
        ],
    ))
    return commands


def phone_probe(
    policy: dict[str, Any],
    phone: str,
    moment: str,
    command_runner,
    timeout_seconds: int,
    forwards: list[dict[str, str]],
    clock,
) -> dict[str, Any]:
    expected = policy["phones"][phone]
    adb = policy["remote_artifacts"]["adb"]["path"]
    prefix = [
        adb,
        "-P",
        str(policy["adb_server_port"]),
        "-s",
        expected["wifi_selector"],
    ]
    commands = []

    def run(name: str, suffix: list[str]) -> bytes:
        record, raw = command_record(
            command_runner,
            prefix + suffix,
            timeout_seconds,
            f"phone.{phone}.{name}",
        )
        commands.append({"name": name, **record})
        return raw

    state = decode_ascii_line(
        run("state", ["get-state"]),
        f"phone.{phone}.state",
    )
    exact(state, "device", f"phone.{phone}.state")
    boot_id = decode_ascii_line(
        run(
            "boot_id",
            ["shell", "cat", "/proc/sys/kernel/random/boot_id"],
        ),
        f"phone.{phone}.boot_id",
    )
    uuid(boot_id, f"phone.{phone}.boot_id")
    serial = decode_ascii_line(
        run("serial", ["shell", "getprop", "ro.serialno"]),
        f"phone.{phone}.serial",
    )
    interface_raw = run(
        "interface",
        [
            "shell",
            "ip",
            "-o",
            "-4",
            "addr",
            "show",
            "dev",
            expected["interface"],
        ],
    )
    ipv4 = parse_interface_ipv4(
        interface_raw,
        expected["interface"],
        f"phone.{phone}.ipv4",
    )
    processes = []
    for index, process_policy in enumerate(expected["forbidden_processes"]):
        processes.extend(parse_phone_proc_scan(
            run(
                f"processes_{index}",
                [
                    "shell",
                    "sh",
                    "-c",
                    PHONE_PROC_SCAN_SCRIPT,
                    "sh",
                    process_policy["executable_path"],
                    process_policy["sha256"],
                ],
            ),
            process_policy,
            f"phone.{phone}.processes[{index}]",
        ))
    listeners = parse_listener_table(
        run(
            "listeners",
            [
                "shell",
                "cat",
                "/proc/net/tcp",
                "/proc/net/tcp6",
            ],
        ),
        expected["forbidden_listen_ports"],
    )
    phone_forwards = [
        row for row in forwards if row["serial"] == expected["wifi_selector"]
    ]
    snapshot = {
        "adb_forwards": phone_forwards,
        "adb_state": state,
        "boot_id": boot_id,
        "clock_name": RTX_CLOCK_NAME,
        "commands": commands,
        "interface": expected["interface"],
        "matching_listeners": listeners,
        "matching_processes": processes,
        "moment": moment,
        "observed_ns": clock(),
        "outer_phase_id": policy["outer_phase_id"],
        "phase": PHASE,
        "physical_serial": serial,
        "schema": PHONE_SNAPSHOT_SCHEMA,
        "wifi_ipv4": ipv4,
        "wifi_selector": expected["wifi_selector"],
    }
    return snapshot


def remote_output_paths(
    policy: dict[str, Any],
    moment: str,
    root: Path | None = None,
) -> dict[str, Path]:
    require(moment in {"before", "after"}, "E_MOMENT")
    if root is None:
        root = (
            Path("/tmp/s39-v25-phone-guard")
            / policy["outer_phase_id"]
            / moment
        )
    require(root.is_absolute(), "E_REMOTE_OUTPUT_ROOT")
    return {
        "desktop": root / "desktop-snapshot.json",
        "op12": root / "op12-snapshot.json",
        "op15": root / "op15-snapshot.json",
        "result": root / "remote-result.json",
    }


def snapshot_payload(
    path: Path,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    raw = canonical_bytes(snapshot)
    artifact = write_exclusive(path, raw)
    return {
        "artifact": artifact,
        "content_base64": base64.b64encode(raw).decode("ascii"),
    }


def observe_verified_artifacts(
    artifacts: dict[str, dict[str, Any]],
    field: str,
) -> dict[str, dict[str, Any]]:
    result = {}
    for name in sorted(artifacts):
        verify_local_artifact(artifacts[name], f"{field}.{name}")
        observed = artifact_from_path(
            Path(artifacts[name]["path"]),
            f"{field}.{name}.observed",
        )
        exact(observed, artifacts[name], f"{field}.{name}")
        result[name] = observed
    return result


def execute_remote(
    policy: dict[str, Any],
    policy_sha256: str,
    moment: str,
    *,
    command_runner=default_command_runner,
    clock=raw_clock_ns,
    boot_reader=read_boot_id,
    gpu_reader=read_gpu_uuids,
    process_scanner=scan_matching_processes,
    listener_scanner=scan_matching_listeners,
    adb_server_observer=observe_adb_server,
    output_root: Path | None = None,
) -> tuple[dict[str, Any], bytes]:
    policy = validate_remote_policy(policy)
    digest(policy_sha256, "remote.policy_sha256")
    exact(
        hashlib.sha256(canonical_bytes(policy)).hexdigest(),
        policy_sha256,
        "remote.policy_sha256",
    )
    started_ns = clock()
    before_artifacts = observe_verified_artifacts(
        policy["remote_artifacts"],
        "remote.artifacts.before",
    )
    boot_id = boot_reader()
    exact(boot_id, policy["rtx_boot_id"], "remote.rtx_boot_id")
    gpu_uuids = gpu_reader()
    require(
        type(gpu_uuids) is list
        and gpu_uuids == sorted(set(gpu_uuids))
        and policy["gpu_uuid"] in gpu_uuids,
        "E_REMOTE_GPU_UUID",
    )
    desktop_processes_before = process_scanner(
        policy["desktop_forbidden_processes"]
    )
    desktop_listeners_before = listener_scanner(
        policy["desktop_forbidden_listen_ports"]
    )
    adb = policy["remote_artifacts"]["adb"]["path"]
    port = policy["adb_server_port"]
    timeout = 60
    adb_server_before = adb_server_observer(
        policy["remote_artifacts"]["adb"],
        policy["adb_server_process"],
        clock,
    )
    forward_record, forward_raw = command_record(
        command_runner,
        [adb, "-P", str(port), "forward", "--list"],
        timeout,
        "remote.adb_forwards",
    )
    forwards = parse_adb_forwards(forward_raw)
    phones = {
        phone: phone_probe(
            policy,
            phone,
            moment,
            command_runner,
            timeout,
            forwards,
            clock,
        )
        for phone in ("op12", "op15")
    }
    adb_server_after = adb_server_observer(
        policy["remote_artifacts"]["adb"],
        policy["adb_server_process"],
        clock,
    )
    exact(adb_server_after, {
        **adb_server_before,
        "observed_ns": adb_server_after["observed_ns"],
    }, "E_ADB_SERVER_CHANGED")
    desktop_processes_after = process_scanner(
        policy["desktop_forbidden_processes"]
    )
    desktop_listeners_after = listener_scanner(
        policy["desktop_forbidden_listen_ports"]
    )
    after_artifacts = observe_verified_artifacts(
        policy["remote_artifacts"],
        "remote.artifacts.after",
    )
    desktop_snapshot = {
        "adb_forward_command": forward_record,
        "adb_server_after": adb_server_after,
        "adb_server_before": adb_server_before,
        "artifact_checks": {
            "after": after_artifacts,
            "before": before_artifacts,
        },
        "boot_id": boot_id,
        "clock_name": RTX_CLOCK_NAME,
        "gpu_uuids": gpu_uuids,
        "matching_listeners_after": desktop_listeners_after,
        "matching_listeners_before": desktop_listeners_before,
        "matching_processes_after": desktop_processes_after,
        "matching_processes_before": desktop_processes_before,
        "moment": moment,
        "observed_ns": clock(),
        "outer_phase_id": policy["outer_phase_id"],
        "phase": PHASE,
        "schema": DESKTOP_SNAPSHOT_SCHEMA,
    }
    paths = remote_output_paths(policy, moment, output_root)
    snapshots = {
        "desktop": snapshot_payload(paths["desktop"], desktop_snapshot),
        "op12": snapshot_payload(paths["op12"], phones["op12"]),
        "op15": snapshot_payload(paths["op15"], phones["op15"]),
    }
    exact(desktop_processes_before, [], "remote.desktop.processes_before")
    exact(desktop_processes_after, [], "remote.desktop.processes_after")
    exact(desktop_listeners_before, [], "remote.desktop.listeners_before")
    exact(desktop_listeners_after, [], "remote.desktop.listeners_after")
    for phone in ("op12", "op15"):
        expected = policy["phones"][phone]
        observed = phones[phone]
        exact(observed["adb_state"], "device", f"remote.{phone}.state")
        exact(observed["boot_id"], expected["boot_id"], f"remote.{phone}.boot")
        exact(
            observed["physical_serial"],
            expected["physical_serial"],
            f"remote.{phone}.serial",
        )
        exact(
            observed["interface"],
            expected["interface"],
            f"remote.{phone}.interface",
        )
        exact(
            observed["wifi_ipv4"],
            expected["wifi_ipv4"],
            f"remote.{phone}.ipv4",
        )
        exact(
            observed["matching_processes"],
            [],
            f"remote.{phone}.processes",
        )
        exact(
            observed["matching_listeners"],
            [],
            f"remote.{phone}.listeners",
        )
        exact(observed["adb_forwards"], [], f"remote.{phone}.forwards")
    completed_ns = clock()
    require(started_ns < completed_ns, "E_REMOTE_INTERVAL")
    output = {
        "clock_name": RTX_CLOCK_NAME,
        "completed_ns": completed_ns,
        "gpu_uuid": policy["gpu_uuid"],
        "moment": moment,
        "outer_phase_id": policy["outer_phase_id"],
        "phase": PHASE,
        "policy_sha256": policy_sha256,
        "rtx_boot_id": boot_id,
        "schema": REMOTE_OUTPUT_SCHEMA,
        "snapshots": snapshots,
        "started_ns": started_ns,
        "v24_phase_id": policy["inner_phase_id"],
    }
    raw = canonical_bytes(output)
    write_exclusive(paths["result"], raw)
    return output, raw


def validate_adb_server_receipt(
    value: Any,
    expected: dict[str, Any],
    adb_artifact: dict[str, Any],
    field: str,
) -> dict[str, Any]:
    value = exact_keys(value, ADB_SERVER_RECEIPT_KEYS, field)
    for key in ADB_SERVER_PLAN_KEYS:
        exact(value[key], expected[key], f"{field}.{key}")
    exact(
        value["executable_sha256"],
        adb_artifact["sha256"],
        f"{field}.executable_sha256",
    )
    integer(value["listener_inode"], f"{field}.listener_inode", 1)
    integer(value["observed_ns"], f"{field}.observed_ns", 1)
    return value


def validate_command_snapshot(value: Any, field: str) -> dict[str, Any]:
    value = exact_keys(
        value,
        {
            "argv",
            "name",
            "returncode",
            "stderr_base64",
            "stdout_base64",
        },
        field,
    )
    require(
        type(value["argv"]) is list and bool(value["argv"]),
        f"E_ARGV: {field}",
    )
    for index, item in enumerate(value["argv"]):
        text(item, f"{field}.argv[{index}]")
    text(value["name"], f"{field}.name", 128)
    exact(value["returncode"], 0, f"{field}.returncode")
    for key in ("stderr_base64", "stdout_base64"):
        encoded = value[key]
        require(type(encoded) is str and encoded.isascii(), f"E_BASE64: {field}")
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except ValueError as error:
            raise GuardError(f"E_BASE64: {field}.{key}") from error
        require(len(decoded) <= MAX_COMMAND_BYTES, f"E_SIZE: {field}.{key}")
    exact(value["stderr_base64"], "", f"{field}.stderr_base64")
    return value


def command_stdout(value: dict[str, Any], field: str) -> bytes:
    validate_command_snapshot(value, field)
    try:
        return base64.b64decode(value["stdout_base64"], validate=True)
    except ValueError as error:
        raise GuardError(f"E_BASE64: {field}.stdout_base64") from error


def validate_snapshot_payload(
    value: Any,
    field: str,
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    value = exact_keys(value, {"artifact", "content_base64"}, field)
    artifact = validate_artifact(value["artifact"], f"{field}.artifact")
    try:
        raw = base64.b64decode(value["content_base64"], validate=True)
    except ValueError as error:
        raise GuardError(f"E_BASE64: {field}") from error
    exact(len(raw), artifact["bytes"], f"{field}.bytes")
    exact(
        hashlib.sha256(raw).hexdigest(),
        artifact["sha256"],
        f"{field}.sha256",
    )
    parsed = parse_json(raw, field)
    require(type(parsed) is dict, f"E_TYPE: {field}")
    exact(canonical_bytes(parsed), raw, f"{field}.canonical")
    return parsed, raw, artifact


def validate_remote_output(
    value: Any,
    policy: dict[str, Any],
    policy_sha256: str,
    moment: str,
) -> tuple[dict[str, Any], dict[str, tuple[dict[str, Any], bytes, dict[str, Any]]]]:
    policy = validate_remote_policy(policy)
    value = exact_keys(
        value,
        {
            "clock_name",
            "completed_ns",
            "gpu_uuid",
            "moment",
            "outer_phase_id",
            "phase",
            "policy_sha256",
            "rtx_boot_id",
            "schema",
            "snapshots",
            "started_ns",
            "v24_phase_id",
        },
        "remote_output",
    )
    exact(value["schema"], REMOTE_OUTPUT_SCHEMA, "remote_output.schema")
    exact(value["phase"], PHASE, "remote_output.phase")
    exact(value["clock_name"], RTX_CLOCK_NAME, "remote_output.clock")
    exact(value["moment"], moment, "remote_output.moment")
    exact(
        value["outer_phase_id"],
        policy["outer_phase_id"],
        "remote_output.outer_phase_id",
    )
    exact(
        value["v24_phase_id"],
        policy["inner_phase_id"],
        "remote_output.v24_phase_id",
    )
    exact(
        value["policy_sha256"],
        policy_sha256,
        "remote_output.policy_sha256",
    )
    exact(
        value["rtx_boot_id"],
        policy["rtx_boot_id"],
        "remote_output.rtx_boot_id",
    )
    exact(value["gpu_uuid"], policy["gpu_uuid"], "remote_output.gpu_uuid")
    started = integer(value["started_ns"], "remote_output.started_ns", 1)
    completed = integer(value["completed_ns"], "remote_output.completed_ns", 1)
    require(started < completed, "E_REMOTE_INTERVAL")
    payloads = exact_keys(
        value["snapshots"],
        {"desktop", "op12", "op15"},
        "remote_output.snapshots",
    )
    decoded = {
        name: validate_snapshot_payload(
            payloads[name],
            f"remote_output.snapshots.{name}",
        )
        for name in ("desktop", "op12", "op15")
    }
    remote_paths = [
        decoded[name][2]["path"]
        for name in ("desktop", "op12", "op15")
    ]
    require(
        len(remote_paths) == len(set(remote_paths)),
        "E_REMOTE_SNAPSHOT_PATH_REUSE",
    )
    desktop = exact_keys(
        decoded["desktop"][0],
        {
            "adb_forward_command",
            "adb_server_after",
            "adb_server_before",
            "artifact_checks",
            "boot_id",
            "clock_name",
            "gpu_uuids",
            "matching_listeners_after",
            "matching_listeners_before",
            "matching_processes_after",
            "matching_processes_before",
            "moment",
            "observed_ns",
            "outer_phase_id",
            "phase",
            "schema",
        },
        "remote_output.desktop",
    )
    exact(desktop["schema"], DESKTOP_SNAPSHOT_SCHEMA, "desktop.schema")
    exact(desktop["phase"], PHASE, "desktop.phase")
    exact(desktop["clock_name"], RTX_CLOCK_NAME, "desktop.clock")
    exact(desktop["moment"], moment, "desktop.moment")
    exact(
        desktop["outer_phase_id"],
        policy["outer_phase_id"],
        "desktop.outer_phase_id",
    )
    exact(desktop["boot_id"], policy["rtx_boot_id"], "desktop.boot_id")
    require(
        type(desktop["gpu_uuids"]) is list
        and policy["gpu_uuid"] in desktop["gpu_uuids"],
        "E_DESKTOP_GPU_UUID",
    )
    observed_ns = integer(
        desktop["observed_ns"],
        "desktop.observed_ns",
        started,
    )
    require(observed_ns <= completed, "E_DESKTOP_INTERVAL")
    checks = exact_keys(
        desktop["artifact_checks"],
        {"after", "before"},
        "desktop.artifact_checks",
    )
    for when in ("before", "after"):
        exact(
            checks[when],
            policy["remote_artifacts"],
            f"desktop.artifact_checks.{when}",
        )
    for field in (
        "matching_listeners_after",
        "matching_listeners_before",
        "matching_processes_after",
        "matching_processes_before",
    ):
        exact(desktop[field], [], f"desktop.{field}")
    adb_server_before = validate_adb_server_receipt(
        desktop["adb_server_before"],
        policy["adb_server_process"],
        policy["remote_artifacts"]["adb"],
        "desktop.adb_server_before",
    )
    adb_server_after = validate_adb_server_receipt(
        desktop["adb_server_after"],
        policy["adb_server_process"],
        policy["remote_artifacts"]["adb"],
        "desktop.adb_server_after",
    )
    require(
        started
        <= adb_server_before["observed_ns"]
        <= adb_server_after["observed_ns"]
        <= completed,
        "E_ADB_SERVER_INTERVAL",
    )
    exact(
        adb_server_after,
        {
            **adb_server_before,
            "observed_ns": adb_server_after["observed_ns"],
        },
        "desktop.adb_server_identity",
    )
    forward = {"name": "forwards", **desktop["adb_forward_command"]}
    validate_command_snapshot(forward, "desktop.adb_forward_command")
    exact(
        forward["argv"],
        [
            policy["remote_artifacts"]["adb"]["path"],
            "-P",
            str(policy["adb_server_port"]),
            "forward",
            "--list",
        ],
        "desktop.adb_forward_command.argv",
    )
    forwards = parse_adb_forwards(
        command_stdout(forward, "desktop.adb_forward_command")
    )

    for phone in ("op12", "op15"):
        snapshot = exact_keys(
            decoded[phone][0],
            {
                "adb_forwards",
                "adb_state",
                "boot_id",
                "clock_name",
                "commands",
                "interface",
                "matching_listeners",
                "matching_processes",
                "moment",
                "observed_ns",
                "outer_phase_id",
                "phase",
                "physical_serial",
                "schema",
                "wifi_ipv4",
                "wifi_selector",
            },
            f"remote_output.{phone}",
        )
        expected = policy["phones"][phone]
        exact(snapshot["schema"], PHONE_SNAPSHOT_SCHEMA, f"{phone}.schema")
        exact(snapshot["phase"], PHASE, f"{phone}.phase")
        exact(snapshot["clock_name"], RTX_CLOCK_NAME, f"{phone}.clock")
        exact(snapshot["moment"], moment, f"{phone}.moment")
        exact(
            snapshot["outer_phase_id"],
            policy["outer_phase_id"],
            f"{phone}.outer_phase_id",
        )
        phone_observed = integer(
            snapshot["observed_ns"],
            f"{phone}.observed_ns",
            started,
        )
        require(phone_observed <= completed, f"E_PHONE_INTERVAL: {phone}")
        for key, expected_value in (
            ("adb_state", "device"),
            ("boot_id", expected["boot_id"]),
            ("interface", expected["interface"]),
            ("physical_serial", expected["physical_serial"]),
            ("wifi_ipv4", expected["wifi_ipv4"]),
            ("wifi_selector", expected["wifi_selector"]),
            ("adb_forwards", []),
            ("matching_listeners", []),
            ("matching_processes", []),
        ):
            exact(snapshot[key], expected_value, f"{phone}.{key}")
        commands = snapshot["commands"]
        specs = expected_phone_commands(policy, phone)
        require(
            type(commands) is list and len(commands) == len(specs),
            f"E_COMMANDS: {phone}",
        )
        outputs = {}
        for index, (command, spec) in enumerate(zip(commands, specs)):
            field = f"{phone}.commands[{index}]"
            validate_command_snapshot(command, field)
            exact(command["name"], spec[0], f"{field}.name")
            exact(command["argv"], spec[1], f"{field}.argv")
            outputs[spec[0]] = command_stdout(command, field)
        exact(
            decode_ascii_line(outputs["state"], f"{phone}.state"),
            snapshot["adb_state"],
            f"{phone}.state.summary",
        )
        exact(
            decode_ascii_line(outputs["boot_id"], f"{phone}.boot_id"),
            snapshot["boot_id"],
            f"{phone}.boot_id.summary",
        )
        exact(
            decode_ascii_line(outputs["serial"], f"{phone}.serial"),
            snapshot["physical_serial"],
            f"{phone}.serial.summary",
        )
        exact(
            parse_interface_ipv4(
                outputs["interface"],
                expected["interface"],
                f"{phone}.interface",
            ),
            snapshot["wifi_ipv4"],
            f"{phone}.interface.summary",
        )
        processes = []
        for index, process_policy in enumerate(
            expected["forbidden_processes"]
        ):
            processes.extend(parse_phone_proc_scan(
                outputs[f"processes_{index}"],
                process_policy,
                f"{phone}.processes[{index}]",
            ))
        exact(
            sorted(
                processes,
                key=lambda row: (row["executable_path"], row["pid"]),
            ),
            snapshot["matching_processes"],
            f"{phone}.processes.summary",
        )
        exact(
            parse_listener_table(
                outputs["listeners"],
                expected["forbidden_listen_ports"],
            ),
            snapshot["matching_listeners"],
            f"{phone}.listeners.summary",
        )
        exact(
            [
                row
                for row in forwards
                if row["serial"] == expected["wifi_selector"]
            ],
            snapshot["adb_forwards"],
            f"{phone}.forwards.summary",
        )
    return value, decoded


def validate_argv(value: Any, field: str) -> list[str]:
    require(
        type(value) is list and 0 < len(value) <= 256,
        f"E_ARGV: {field}",
    )
    for index, item in enumerate(value):
        text(item, f"{field}[{index}]", 32768)
    absolute_path(value[0], f"{field}[0]")
    require(Path(value[0]).name == "ssh", f"E_SSH_ARGV: {field}")
    return value


def validate_receipt(
    value: Any,
    plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if plan is not None:
        plan = validate_plan(plan)
    value = exact_keys(value, RECEIPT_KEYS, "receipt")
    exact(value["schema"], RECEIPT_SCHEMA, "receipt.schema")
    exact(value["phase"], PHASE, "receipt.phase")
    validate_phase_ids(
        value["outer_phase_id"],
        value["inner_phase_id"],
        "receipt",
    )
    plan_sha256 = digest(value["plan_sha256"], "receipt.plan_sha256")
    rtx_boot_id = uuid(value["rtx_boot_id"], "receipt.rtx_boot_id")
    exact(value["gpu_uuid"], GPU_UUID, "receipt.gpu_uuid")
    exact(value["clock_name"], RTX_CLOCK_NAME, "receipt.clock_name")
    adb_server = exact_keys(
        value["adb_server_process"],
        ADB_SERVER_RECEIPT_KEYS,
        "receipt.adb_server_process",
    )
    require(
        type(adb_server["argv"]) is list and bool(adb_server["argv"]),
        "E_ADB_SERVER_ARGV",
    )
    for index, item in enumerate(adb_server["argv"]):
        text(item, f"receipt.adb_server_process.argv[{index}]")
    uuid(adb_server["boot_id"], "receipt.adb_server_process.boot_id")
    absolute_path(
        adb_server["executable_path"],
        "receipt.adb_server_process.executable_path",
    )
    digest(
        adb_server["executable_sha256"],
        "receipt.adb_server_process.executable_sha256",
    )
    exact(
        adb_server["listen_host"],
        "127.0.0.1",
        "receipt.adb_server_process.listen_host",
    )
    exact(
        adb_server["listen_port"],
        ADB_SERVER_PORT,
        "receipt.adb_server_process.listen_port",
    )
    for key in ("listener_inode", "observed_ns", "pid", "start_ticks"):
        integer(
            adb_server[key],
            f"receipt.adb_server_process.{key}",
            1,
        )
    moment = text(value["moment"], "receipt.moment", 6)
    require(moment in {"before", "after"}, "E_MOMENT")
    started = integer(value["started_ns"], "receipt.started_ns", 1)
    completed = integer(value["completed_ns"], "receipt.completed_ns", 1)
    require(started < completed, "E_RECEIPT_INTERVAL")
    require(
        started <= adb_server["observed_ns"] <= completed,
        "E_ADB_SERVER_REMOTE_INTERVAL",
    )
    exact(
        value["desktop_matching_processes"],
        [],
        "receipt.desktop_matching_processes",
    )
    exact(
        value["desktop_matching_listeners"],
        [],
        "receipt.desktop_matching_listeners",
    )
    validate_artifact(
        value["desktop_raw_snapshot_artifact"],
        "receipt.desktop_raw_snapshot_artifact",
    )
    validate_artifact(
        value["remote_output_artifact"],
        "receipt.remote_output_artifact",
    )

    phones = exact_keys(value["phones"], set(PHONE_SERIALS), "receipt.phones")
    selectors: set[str] = set()
    ipv4s: set[str] = set()
    boot_ids: set[str] = {rtx_boot_id}
    snapshot_paths: set[str] = set()
    for phone in sorted(PHONE_SERIALS):
        field = f"receipt.phones.{phone}"
        row = exact_keys(phones[phone], RECEIPT_PHONE_KEYS, field)
        validate_phone_identity(row, phone, field)
        exact(row["adb_state"], "device", f"{field}.adb_state")
        exact(row["matching_processes"], [], f"{field}.matching_processes")
        exact(row["matching_listeners"], [], f"{field}.matching_listeners")
        exact(row["adb_forwards"], [], f"{field}.adb_forwards")
        snapshot = validate_artifact(
            row["raw_snapshot_artifact"],
            f"{field}.raw_snapshot_artifact",
        )
        require(snapshot["path"] not in snapshot_paths, "E_SNAPSHOT_PATH_REUSE")
        snapshot_paths.add(snapshot["path"])
        require(row["wifi_selector"] not in selectors, "E_SELECTOR_REUSE")
        require(row["wifi_ipv4"] not in ipv4s, "E_IPV4_REUSE")
        require(row["boot_id"] not in boot_ids, "E_BOOT_ID_REUSE")
        selectors.add(row["wifi_selector"])
        ipv4s.add(row["wifi_ipv4"])
        boot_ids.add(row["boot_id"])
    receipt_paths = [
        value["desktop_raw_snapshot_artifact"]["path"],
        value["remote_output_artifact"]["path"],
        *[
            phones[phone]["raw_snapshot_artifact"]["path"]
            for phone in sorted(PHONE_SERIALS)
        ],
    ]
    require(
        len(receipt_paths) == len(set(receipt_paths)),
        "E_RECEIPT_ARTIFACT_PATH_REUSE",
    )

    process = exact_keys(
        value["ssh_transport_process"],
        SSH_PROCESS_KEYS,
        "receipt.ssh_transport_process",
    )
    exact(
        process["schema"],
        SSH_PROCESS_SCHEMA,
        "receipt.ssh_transport_process.schema",
    )
    exact(
        process["clock_name"],
        CONTROLLER_CLOCK_NAME,
        "receipt.ssh_transport_process.clock_name",
    )
    controller_boot_id = uuid(
        process["controller_boot_id"],
        "receipt.ssh_transport_process.controller_boot_id",
    )
    require(controller_boot_id != rtx_boot_id, "E_CONTROLLER_BOOT_ALIAS")
    exact(
        process["remote_boot_id"],
        rtx_boot_id,
        "receipt.ssh_transport_process.remote_boot_id",
    )
    exact(
        process["plan_sha256"],
        plan_sha256,
        "receipt.ssh_transport_process.plan_sha256",
    )
    validate_argv(process["argv"], "receipt.ssh_transport_process.argv")
    process_observed = integer(
        process["observed_ns"],
        "receipt.ssh_transport_process.observed_ns",
        1,
    )
    process_pid = integer(
        process["pid"],
        "receipt.ssh_transport_process.pid",
        1,
    )
    process_ticks = integer(
        process["start_ticks"],
        "receipt.ssh_transport_process.start_ticks",
        1,
    )

    cleanup = exact_keys(
        value["ssh_transport_cleanup"],
        SSH_CLEANUP_KEYS,
        "receipt.ssh_transport_cleanup",
    )
    exact(
        cleanup["schema"],
        SSH_CLEANUP_SCHEMA,
        "receipt.ssh_transport_cleanup.schema",
    )
    exact(
        cleanup["clock_name"],
        CONTROLLER_CLOCK_NAME,
        "receipt.ssh_transport_cleanup.clock_name",
    )
    exact(
        cleanup["controller_boot_id"],
        controller_boot_id,
        "receipt.ssh_transport_cleanup.controller_boot_id",
    )
    exact(cleanup["pid"], process_pid, "receipt.ssh_transport_cleanup.pid")
    exact(
        cleanup["start_ticks"],
        process_ticks,
        "receipt.ssh_transport_cleanup.start_ticks",
    )
    exact(
        cleanup["process_absent"],
        True,
        "receipt.ssh_transport_cleanup.process_absent",
    )
    cleanup_observed = integer(
        cleanup["observed_ns"],
        "receipt.ssh_transport_cleanup.observed_ns",
        1,
    )
    require(
        process_observed <= cleanup_observed,
        "E_SSH_TRANSPORT_INTERVAL",
    )
    if plan is not None:
        validate_adb_server_receipt(
            adb_server,
            plan["adb_server_process"],
            plan["remote_artifacts"]["adb"],
            "receipt.adb_server_process",
        )
        exact(value["plan_sha256"], hashlib.sha256(canonical_bytes(plan)).hexdigest(),
              "receipt.plan_sha256")
        exact(
            process["argv"],
            expected_ssh_argv(plan, moment),
            "receipt.ssh_transport_process.argv",
        )
        for phone in sorted(PHONE_SERIALS):
            for key in (
                "boot_id",
                "interface",
                "physical_serial",
                "wifi_ipv4",
                "wifi_selector",
            ):
                exact(
                    value["phones"][phone][key],
                    plan["phones"][phone][key],
                    f"receipt.phones.{phone}.{key}",
                )
    return value


def validate_receipt_evidence(
    value: Any,
    plan: dict[str, Any],
) -> dict[str, Any]:
    plan = validate_plan(plan)
    receipt = validate_receipt(value, plan)
    result_artifact = receipt["remote_output_artifact"]
    verify_local_artifact(result_artifact, "receipt.remote_output_artifact")
    remote_raw = read_regular(Path(result_artifact["path"]))
    remote_value = parse_json(remote_raw, "receipt.remote_output")
    require(type(remote_value) is dict, "E_TYPE: receipt.remote_output")
    exact(
        canonical_bytes(remote_value),
        remote_raw,
        "receipt.remote_output.canonical",
    )
    remote_value, decoded = validate_remote_output(
        remote_value,
        plan["remote_policy"],
        plan["remote_policy_artifact"]["sha256"],
        receipt["moment"],
    )

    local_artifacts = {
        "desktop": receipt["desktop_raw_snapshot_artifact"],
        "op12": receipt["phones"]["op12"]["raw_snapshot_artifact"],
        "op15": receipt["phones"]["op15"]["raw_snapshot_artifact"],
    }
    for name in ("desktop", "op12", "op15"):
        artifact = local_artifacts[name]
        verify_local_artifact(
            artifact,
            f"receipt.{name}.raw_snapshot_artifact",
        )
        exact(
            read_regular(Path(artifact["path"])),
            decoded[name][1],
            f"receipt.{name}.raw_snapshot",
        )

    for receipt_key, remote_key in (
        ("clock_name", "clock_name"),
        ("completed_ns", "completed_ns"),
        ("gpu_uuid", "gpu_uuid"),
        ("moment", "moment"),
        ("outer_phase_id", "outer_phase_id"),
        ("phase", "phase"),
        ("rtx_boot_id", "rtx_boot_id"),
        ("started_ns", "started_ns"),
    ):
        exact(
            receipt[receipt_key],
            remote_value[remote_key],
            f"receipt.remote_link.{receipt_key}",
        )
    exact(
        receipt["inner_phase_id"],
        remote_value["v24_phase_id"],
        "receipt.remote_link.inner_phase_id",
    )
    exact(
        remote_value["policy_sha256"],
        plan["remote_policy_artifact"]["sha256"],
        "receipt.remote_link.policy_sha256",
    )

    desktop = decoded["desktop"][0]
    exact(
        receipt["adb_server_process"],
        desktop["adb_server_after"],
        "receipt.remote_link.adb_server_process",
    )
    exact(
        receipt["desktop_matching_listeners"],
        [
            *desktop["matching_listeners_before"],
            *desktop["matching_listeners_after"],
        ],
        "receipt.remote_link.desktop_matching_listeners",
    )
    exact(
        receipt["desktop_matching_processes"],
        [
            *desktop["matching_processes_before"],
            *desktop["matching_processes_after"],
        ],
        "receipt.remote_link.desktop_matching_processes",
    )
    for phone in ("op12", "op15"):
        snapshot = decoded[phone][0]
        for key in (
            "adb_forwards",
            "adb_state",
            "boot_id",
            "interface",
            "matching_listeners",
            "matching_processes",
            "physical_serial",
            "wifi_ipv4",
            "wifi_selector",
        ):
            exact(
                receipt["phones"][phone][key],
                snapshot[key],
                f"receipt.remote_link.{phone}.{key}",
            )
    return receipt


def validate_pair(
    before: Any,
    after: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    before = validate_receipt(before)
    after = validate_receipt(after)
    exact(before["moment"], "before", "pair.before.moment")
    exact(after["moment"], "after", "pair.after.moment")
    for key in (
        "clock_name",
        "gpu_uuid",
        "inner_phase_id",
        "outer_phase_id",
        "phase",
        "plan_sha256",
        "rtx_boot_id",
    ):
        exact(after[key], before[key], f"pair.{key}")
    for key in ADB_SERVER_RECEIPT_KEYS - {"observed_ns", "listener_inode"}:
        exact(
            after["adb_server_process"][key],
            before["adb_server_process"][key],
            f"pair.adb_server_process.{key}",
        )
    require(
        before["completed_ns"] <= after["started_ns"],
        "E_PAIR_INTERVAL",
    )
    for phone in sorted(PHONE_SERIALS):
        for key in (
            "adb_state",
            "boot_id",
            "interface",
            "physical_serial",
            "wifi_ipv4",
            "wifi_selector",
        ):
            exact(
                after["phones"][phone][key],
                before["phones"][phone][key],
                f"pair.phones.{phone}.{key}",
            )
    exact(
        after["ssh_transport_process"]["controller_boot_id"],
        before["ssh_transport_process"]["controller_boot_id"],
        "pair.controller_boot_id",
    )
    snapshot_paths = [
        receipt["remote_output_artifact"]["path"]
        for receipt in (before, after)
    ]
    snapshot_paths.extend(
        receipt["desktop_raw_snapshot_artifact"]["path"]
        for receipt in (before, after)
    )
    snapshot_paths.extend(
        receipt["phones"][phone]["raw_snapshot_artifact"]["path"]
        for receipt in (before, after)
        for phone in sorted(PHONE_SERIALS)
    )
    require(
        len(snapshot_paths) == len(set(snapshot_paths)),
        "E_PAIR_SNAPSHOT_PATH_REUSE",
    )
    return before, after


def verify_controller_inputs(
    plan: dict[str, Any],
    *,
    executable_path: str | None = None,
) -> None:
    plan = validate_plan(plan)
    local_artifacts = [
        *plan["local_artifacts"].values(),
        plan["local_policy_artifact"],
        plan["ssh_transport"]["ssh"],
        plan["ssh_transport"]["identity_file"],
        plan["ssh_transport"]["known_hosts"],
    ]
    for index, artifact in enumerate(local_artifacts):
        verify_local_artifact(artifact, f"controller.local[{index}]")
    identity_stat = Path(
        plan["ssh_transport"]["identity_file"]["path"]
    ).stat()
    exact(identity_stat.st_uid, os.geteuid(), "controller.identity.uid")
    exact(
        stat.S_IMODE(identity_stat.st_mode),
        0o600,
        "controller.identity.mode",
    )
    policy_raw = read_regular(Path(plan["local_policy_artifact"]["path"]))
    exact(
        policy_raw,
        canonical_bytes(plan["remote_policy"]),
        "controller.policy.bytes",
    )
    exact(
        (
            plan["local_policy_artifact"]["bytes"],
            plan["local_policy_artifact"]["sha256"],
        ),
        (
            plan["remote_policy_artifact"]["bytes"],
            plan["remote_policy_artifact"]["sha256"],
        ),
        "controller.policy.remote_content",
    )
    if executable_path is None:
        executable_path = os.path.realpath(sys.executable)
    exact(
        os.path.realpath(executable_path),
        os.path.realpath(plan["local_artifacts"]["python"]["path"]),
        "controller.python",
    )


def controller_evidence_paths(
    receipt_path: Path,
    moment: str,
) -> dict[str, Path]:
    require(receipt_path.is_absolute(), "E_RECEIPT_PATH")
    root = receipt_path.parent / "raw-phone-guard"
    return {
        "desktop": root / f"{moment}-desktop-snapshot.json",
        "op12": root / f"{moment}-op12-snapshot.json",
        "op15": root / f"{moment}-op15-snapshot.json",
        "result": root / f"{moment}-remote-output.json",
    }


def execute_controller(
    plan: dict[str, Any],
    plan_sha256: str,
    moment: str,
    receipt_path: Path,
    *,
    popen_factory=subprocess.Popen,
    clock=time.monotonic_ns,
    boot_reader=read_boot_id,
    ticks_reader=process_start_ticks,
    process_executable_reader=read_process_executable,
    absent_checker=process_absent,
    group_absent_checker=process_group_absent,
    terminator=terminate_process_group,
    executable_path: str | None = None,
) -> dict[str, Any]:
    plan = validate_plan(plan)
    digest(plan_sha256, "controller.plan_sha256")
    exact(
        hashlib.sha256(canonical_bytes(plan)).hexdigest(),
        plan_sha256,
        "controller.plan_sha256",
    )
    require(moment in {"before", "after"}, "E_MOMENT")
    verify_controller_inputs(plan, executable_path=executable_path)
    controller_boot_id = boot_reader()
    argv = expected_ssh_argv(plan, moment)
    controller_started = clock()
    process = popen_factory(
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
    pid = integer(process.pid, "controller.ssh.pid", 1)
    ticks = ticks_reader(pid)
    process_executable = process_executable_reader(pid, ticks)
    require(
        type(process_executable) is bytes
        and 0 < len(process_executable) <= MAX_ARTIFACT_BYTES,
        "E_SSH_PROCESS_EXECUTABLE",
    )
    exact(
        hashlib.sha256(process_executable).hexdigest(),
        plan["ssh_transport"]["ssh"]["sha256"],
        "controller.ssh.executable_sha256",
    )
    process_observed = clock()
    try:
        stdout, stderr = process.communicate(
            timeout=plan["timeout_seconds"]
        )
    except subprocess.TimeoutExpired as error:
        terminator(process)
        require(absent_checker(pid, ticks), "E_SSH_TIMEOUT_PROCESS")
        require(group_absent_checker(pid), "E_SSH_TIMEOUT_PROCESS_GROUP")
        raise GuardError("E_SSH_TIMEOUT") from error
    except Exception:
        terminator(process)
        require(absent_checker(pid, ticks), "E_SSH_ERROR_PROCESS")
        require(group_absent_checker(pid), "E_SSH_ERROR_PROCESS_GROUP")
        raise
    require(type(stdout) is bytes and type(stderr) is bytes, "E_SSH_OUTPUT")
    require(
        len(stdout) <= MAX_JSON_BYTES * 2
        and len(stderr) <= MAX_COMMAND_BYTES,
        "E_SSH_OUTPUT_SIZE",
    )
    command_absent = absent_checker(pid, ticks)
    group_absent = group_absent_checker(pid)
    if not command_absent or not group_absent:
        terminator(process)
    require(absent_checker(pid, ticks), "E_SSH_PROCESS_LIVE")
    require(group_absent_checker(pid), "E_SSH_PROCESS_GROUP_LIVE")
    exact(process.returncode, 0, "controller.ssh.returncode")
    exact(stderr, b"", "controller.ssh.stderr")
    cleanup_observed = clock()
    lines = stdout.splitlines()
    require(
        len(lines) == 1 and stdout.endswith(b"\n"),
        "E_REMOTE_PACKET_FRAMING",
    )
    try:
        remote_raw = base64.b64decode(lines[0], validate=True)
    except ValueError as error:
        raise GuardError("E_REMOTE_PACKET_BASE64") from error
    remote_value = parse_json(remote_raw, "remote_output")
    require(type(remote_value) is dict, "E_TYPE: remote_output")
    exact(canonical_bytes(remote_value), remote_raw, "remote_output.canonical")
    remote_value, decoded = validate_remote_output(
        remote_value,
        plan["remote_policy"],
        plan["remote_policy_artifact"]["sha256"],
        moment,
    )
    verify_controller_inputs(plan, executable_path=executable_path)

    paths = controller_evidence_paths(receipt_path, moment)
    local_result = write_exclusive(paths["result"], remote_raw)
    local_snapshots = {
        name: write_exclusive(paths[name], decoded[name][1])
        for name in ("desktop", "op12", "op15")
    }
    reopened_result = read_regular(paths["result"])
    exact(reopened_result, remote_raw, "controller.result.reopen")
    for name in ("desktop", "op12", "op15"):
        exact(
            read_regular(paths[name]),
            decoded[name][1],
            f"controller.snapshot.{name}.reopen",
        )
    desktop = decoded["desktop"][0]
    phones = {}
    for phone in ("op12", "op15"):
        snapshot = decoded[phone][0]
        phones[phone] = {
            "adb_forwards": snapshot["adb_forwards"],
            "adb_state": snapshot["adb_state"],
            "boot_id": snapshot["boot_id"],
            "interface": snapshot["interface"],
            "matching_listeners": snapshot["matching_listeners"],
            "matching_processes": snapshot["matching_processes"],
            "physical_serial": snapshot["physical_serial"],
            "raw_snapshot_artifact": local_snapshots[phone],
            "wifi_ipv4": snapshot["wifi_ipv4"],
            "wifi_selector": snapshot["wifi_selector"],
        }
    receipt = {
        "adb_server_process": desktop["adb_server_after"],
        "clock_name": RTX_CLOCK_NAME,
        "completed_ns": remote_value["completed_ns"],
        "desktop_matching_listeners": [
            *desktop["matching_listeners_before"],
            *desktop["matching_listeners_after"],
        ],
        "desktop_matching_processes": [
            *desktop["matching_processes_before"],
            *desktop["matching_processes_after"],
        ],
        "desktop_raw_snapshot_artifact": local_snapshots["desktop"],
        "gpu_uuid": remote_value["gpu_uuid"],
        "inner_phase_id": remote_value["v24_phase_id"],
        "moment": moment,
        "outer_phase_id": remote_value["outer_phase_id"],
        "phase": PHASE,
        "phones": phones,
        "plan_sha256": plan_sha256,
        "remote_output_artifact": local_result,
        "rtx_boot_id": remote_value["rtx_boot_id"],
        "schema": RECEIPT_SCHEMA,
        "ssh_transport_cleanup": {
            "clock_name": CONTROLLER_CLOCK_NAME,
            "controller_boot_id": controller_boot_id,
            "observed_ns": cleanup_observed,
            "pid": pid,
            "process_absent": True,
            "schema": SSH_CLEANUP_SCHEMA,
            "start_ticks": ticks,
        },
        "ssh_transport_process": {
            "argv": argv,
            "clock_name": CONTROLLER_CLOCK_NAME,
            "controller_boot_id": controller_boot_id,
            "observed_ns": process_observed,
            "pid": pid,
            "plan_sha256": plan_sha256,
            "remote_boot_id": remote_value["rtx_boot_id"],
            "schema": SSH_PROCESS_SCHEMA,
            "start_ticks": ticks,
        },
        "started_ns": remote_value["started_ns"],
    }
    validate_receipt_evidence(receipt, plan)
    receipt_raw = canonical_bytes(receipt)
    write_exclusive(receipt_path, receipt_raw)
    reopened, reopened_raw = parse_receipt(
        receipt_path,
        hashlib.sha256(receipt_raw).hexdigest(),
    )
    exact(reopened, receipt, "controller.receipt.reopen")
    exact(reopened_raw, receipt_raw, "controller.receipt.raw")
    validate_receipt_evidence(reopened, plan)
    require(
        controller_started <= process_observed <= cleanup_observed,
        "E_CONTROLLER_INTERVAL",
    )
    return receipt


def parse_receipt(
    path: Path,
    expected_sha256: str,
) -> tuple[dict[str, Any], bytes]:
    digest(expected_sha256, "receipt.sha256")
    raw = read_regular(path)
    exact(
        hashlib.sha256(raw).hexdigest(),
        expected_sha256,
        "receipt.sha256",
    )
    value = parse_json(raw, "receipt")
    require(type(value) is dict, "E_TYPE: receipt")
    exact(canonical_bytes(value), raw, "receipt.canonical")
    return validate_receipt(value), raw


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--policy")
    parser.add_argument("--policy-sha256")
    parser.add_argument("--plan")
    parser.add_argument("--plan-sha256")
    parser.add_argument("--moment", choices=("before", "after"), required=True)
    parser.add_argument("--receipt")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    args = parser.parse_args(argv)
    if args.remote:
        require(
            args.policy is not None
            and args.policy_sha256 is not None
            and args.plan is None
            and args.plan_sha256 is None
            and args.receipt is None
            and not args.execute
            and args.confirm is None,
            "E_REMOTE_ARGS",
        )
    else:
        require(
            args.plan is not None
            and args.plan_sha256 is not None
            and args.receipt is not None
            and args.policy is None
            and args.policy_sha256 is None
            and args.execute
            and args.confirm == CONFIRMATION,
            "E_CONTROLLER_ARGS",
        )
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if args.remote:
            policy, unused_raw = read_remote_policy(
                Path(args.policy),
                args.policy_sha256,
            )
            del unused_raw
            unused_value, raw = execute_remote(
                policy,
                args.policy_sha256,
                args.moment,
            )
            del unused_value
            packet = base64.b64encode(raw) + b"\n"
            sys.stdout.buffer.write(packet)
            sys.stdout.buffer.flush()
        else:
            plan_value, unused_raw = parse_plan(
                Path(args.plan),
                args.plan_sha256,
            )
            del unused_raw
            execute_controller(
                plan_value,
                args.plan_sha256,
                args.moment,
                Path(args.receipt),
            )
        return 0
    except (
        GuardError,
        OSError,
        subprocess.SubprocessError,
        ValueError,
    ) as error:
        print(f"S39_V25_PHONE_GUARD_REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
