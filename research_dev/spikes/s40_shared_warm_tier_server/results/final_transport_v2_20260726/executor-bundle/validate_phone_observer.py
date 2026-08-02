#!/usr/bin/env python3
"""Fail-closed validator for S40 phone identity and telemetry evidence."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
from pathlib import Path
import re
import sys
from typing import Any

from phone_gateway import (
    MAX_COMMAND_BYTES,
    canonical_bytes,
    exact_keys,
    integer,
    require,
    sha256_text,
    strict_json_loads,
    string,
)


BOOT_ID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}$"
)
MIN_AVAILABLE_BYTES = 512 * 1024 * 1024
REMOTE_DEPENDENCY_ROLES = {
    "a6000_phone_observer",
    "a6000_phone_route_control",
    "adb",
    "phone_gateway",
    "python",
    "readiness_v23",
    "remote_config",
}
COMMON_RUNTIME_FILES = {
    "libcxx_shared": "libc++_shared.so",
    "libggml": "libggml.so",
    "libggml_base": "libggml-base.so",
    "libggml_cpu": "libggml-cpu.so",
    "libggml_hexagon": "libggml-hexagon.so",
    "libggml_opencl": "libggml-opencl.so",
    "libllama": "libllama.so",
    "libllama_common": "libllama-common.so",
    "worker": "llama-layersplit",
}


def read_json(path: Path, field: str) -> dict[str, Any]:
    require(path.is_absolute() and path.is_file(), f"{field} path")
    raw = path.read_bytes()
    require(
        0 < len(raw) <= MAX_COMMAND_BYTES and raw.endswith(b"\n"),
        f"{field} framing",
    )
    value = strict_json_loads(raw, field)
    require(
        type(value) is dict and canonical_bytes(value) == raw,
        f"{field} canonical JSON",
    )
    return value


def read_jsonl(path: Path, field: str) -> list[dict[str, Any]]:
    require(path.is_absolute() and path.is_file(), f"{field} path")
    result = []
    with path.open("rb") as source:
        for index, raw in enumerate(source):
            require(
                0 < len(raw) <= MAX_COMMAND_BYTES and raw.endswith(b"\n"),
                f"{field}[{index}] framing",
            )
            value = strict_json_loads(raw, f"{field}[{index}]")
            require(
                type(value) is dict and canonical_bytes(value) == raw,
                f"{field}[{index}] canonical JSON",
            )
            result.append(value)
    require(result, f"{field} empty")
    return result


def interval(value: dict[str, Any], field: str) -> tuple[int, int]:
    started = integer(value["started_ns"], f"{field}.started_ns", 1)
    completed = integer(value["completed_ns"], f"{field}.completed_ns", 1)
    require(started <= completed, f"{field} interval")
    return started, completed


def ipv4(value: Any, field: str) -> str:
    address = string(value, field)
    try:
        normalized = str(ipaddress.IPv4Address(address))
    except ipaddress.AddressValueError as error:
        raise RuntimeError(f"{field}: invalid IPv4 address") from error
    require(address == normalized, f"{field}: non-canonical IPv4 address")
    return address


def argv_sha256(argv: list[str]) -> str:
    return hashlib.sha256(b"".join(
        argument.encode("utf-8") + b"\0"
        for argument in argv
    )).hexdigest()


def validate_file_rows(
    value: Any,
    expected_roles: set[str],
    field: str,
) -> list[dict[str, Any]]:
    require(
        type(value) is list and len(value) == len(expected_roles),
        f"{field} files",
    )
    seen = set()
    for index, row in enumerate(value):
        row = exact_keys(
            row,
            {"bytes", "path", "role", "sha256"},
            f"{field}[{index}]",
        )
        role = string(row["role"], f"{field} role")
        require(
            role in expected_roles and role not in seen,
            f"{field} role",
        )
        seen.add(role)
        require(Path(string(row["path"], f"{field} path")).is_absolute(),
                f"{field} absolute path")
        integer(row["bytes"], f"{field} bytes", 1)
        sha256_text(row["sha256"], f"{field} SHA-256")
    require(seen == expected_roles, f"{field} role set")
    require(
        value == sorted(value, key=lambda row: row["role"]),
        f"{field} order",
    )
    return value


def validate_remote_package(
    value: Any,
    expected_sha256: str,
    field: str,
) -> dict[str, Any]:
    value = exact_keys(
        value,
        {"a6000_identity", "files", "host_boot_id", "schema"},
        field,
    )
    require(
        value["schema"] == "s40-a6000-identity-package-v1",
        f"{field} schema",
    )
    sha256_text(value["a6000_identity"], f"{field} A6000 identity")
    boot_id = string(value["host_boot_id"], f"{field} boot ID")
    require(BOOT_ID.fullmatch(boot_id) is not None, f"{field} boot ID")
    validate_file_rows(value["files"], REMOTE_DEPENDENCY_ROLES, f"{field}.files")
    require(
        hashlib.sha256(canonical_bytes(value)).hexdigest()
        == sha256_text(expected_sha256, f"{field} digest"),
        f"{field} digest mismatch",
    )
    return value


def validate_local_package(
    value: Any,
    expected_sha256: str,
    field: str,
) -> dict[str, Any]:
    value = exact_keys(
        value,
        {
            "files",
            "identity_file_path",
            "identity_public_key_fingerprint",
            "schema",
            "ssh_argv",
            "ssh_env",
        },
        field,
    )
    require(
        value["schema"] == "s40-local-ssh-identity-package-v1",
        f"{field} schema",
    )
    validate_file_rows(
        value["files"],
        {
            "identity_public_key",
            "known_hosts",
            "ssh_executable",
            "ssh_keygen",
        },
        f"{field}.files",
    )
    require(
        Path(string(
            value["identity_file_path"],
            f"{field} identity file",
        )).is_absolute(),
        f"{field} identity path",
    )
    fingerprint = string(
        value["identity_public_key_fingerprint"],
        f"{field} public key fingerprint",
    )
    require(fingerprint.startswith("SHA256:"), f"{field} fingerprint")
    argv = value["ssh_argv"]
    require(
        type(argv) is list
        and argv
        and all(type(item) is str and item for item in argv),
        f"{field} argv",
    )
    require(
        value["ssh_env"]
        == {"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        f"{field} environment",
    )
    require(
        hashlib.sha256(canonical_bytes(value)).hexdigest()
        == sha256_text(expected_sha256, f"{field} digest"),
        f"{field} digest mismatch",
    )
    return value


def validate_snapshot(
    value: Any,
    field: str,
    expected_boot_id: str | None = None,
    expected_phone: str | None = None,
) -> dict[str, Any]:
    value = exact_keys(
        value,
        {
            "available_bytes",
            "boot_id",
            "completed_ns",
            "device",
            "interfaces",
            "model",
            "product",
            "processes",
            "runtime_files",
            "schema",
            "serial",
            "started_ns",
            "swap_total_bytes",
            "swap_used_bytes",
            "tcp_established",
            "temperatures",
            "thermal_status",
        },
        field,
    )
    require(
        value["schema"] == "s40-phone-runtime-snapshot-v1",
        f"{field} schema",
    )
    interval(value, field)
    for key in ("device", "model", "product", "serial"):
        string(value[key], f"{field}.{key}")
    boot_id = string(value["boot_id"], f"{field}.boot_id")
    require(BOOT_ID.fullmatch(boot_id) is not None, f"{field} boot ID")
    if expected_boot_id is not None:
        require(boot_id == expected_boot_id, f"{field} boot changed")
    require(
        integer(value["available_bytes"], f"{field}.available_bytes")
        >= MIN_AVAILABLE_BYTES,
        f"{field} memory headroom",
    )
    swap_total = integer(
        value["swap_total_bytes"],
        f"{field}.swap_total_bytes",
    )
    swap_used = integer(
        value["swap_used_bytes"],
        f"{field}.swap_used_bytes",
    )
    require(
        swap_used <= swap_total and swap_used == 0,
        f"{field} swap usage",
    )
    require(
        integer(value["thermal_status"], f"{field}.thermal_status") == 0,
        f"{field} thermal status",
    )
    temperatures = value["temperatures"]
    require(
        type(temperatures) is list and temperatures,
        f"{field} temperatures",
    )
    names = set()
    for index, row in enumerate(temperatures):
        row = exact_keys(
            row,
            {"name", "temp_millic"},
            f"{field}.temperatures[{index}]",
        )
        name = string(row["name"], f"{field}.temperature name")
        require(name not in names, f"{field} duplicate temperature")
        names.add(name)
        temperature = integer(
            row["temp_millic"],
            f"{field}.temperature",
            -100_000,
        )
        require(temperature <= 300_000, f"{field} temperature range")
    interfaces = value["interfaces"]
    require(type(interfaces) is dict and interfaces, f"{field} interfaces")
    for name, row in interfaces.items():
        string(name, f"{field} interface name")
        row = exact_keys(
            row,
            {"ipv4", "rx_bytes", "tx_bytes"},
            f"{field}.interface.{name}",
        )
        require(
            type(row["ipv4"]) is list
            and row["ipv4"]
            and len(row["ipv4"]) == len(set(row["ipv4"])),
            f"{field} interface addresses",
        )
        for address in row["ipv4"]:
            ipv4(address, f"{field} IPv4")
        integer(row["rx_bytes"], f"{field} rx bytes")
        integer(row["tx_bytes"], f"{field} tx bytes")
    tcp = value["tcp_established"]
    require(type(tcp) is list, f"{field} TCP sockets")
    seen = set()
    for index, row in enumerate(tcp):
        row = exact_keys(
            row,
            {
                "local_address",
                "local_port",
                "remote_address",
                "remote_port",
                "socket_inode",
            },
            f"{field}.tcp[{index}]",
        )
        key = (
            ipv4(row["local_address"], f"{field} local address"),
            integer(row["local_port"], f"{field} local port", 1),
            ipv4(row["remote_address"], f"{field} remote address"),
            integer(row["remote_port"], f"{field} remote port", 1),
            integer(row["socket_inode"], f"{field} socket inode", 1),
        )
        require(
            key[1] <= 65535
            and key[3] <= 65535
            and key not in seen,
            f"{field} TCP socket",
        )
        seen.add(key)
    processes = value["processes"]
    require(type(processes) is list, f"{field} processes")
    process_names = set()
    process_pids = set()
    for index, row in enumerate(processes):
        row = exact_keys(
            row,
            {
                "argv",
                "artifact_path",
                "artifact_role",
                "artifact_sha256",
                "backend",
                "cmdline_sha256",
                "env",
                "head_host",
                "head_port",
                "kind",
                "layer_end",
                "layer_start",
                "listen_port",
                "name",
                "pid",
                "process_start_ticks",
                "socket_inodes",
                "tail_host",
                "tail_port",
                "tail_source_port",
            },
            f"{field}.processes[{index}]",
        )
        name = string(row["name"], f"{field} process name")
        pid = integer(row["pid"], f"{field} process PID", 1)
        require(
            name not in process_names and pid not in process_pids,
            f"{field} duplicate process",
        )
        process_names.add(name)
        process_pids.add(pid)
        require(
            row["kind"] in ("STAGE_HEAD", "STAGE_TAIL", "DIRECT_RELAY"),
            f"{field} process kind",
        )
        argv = row["argv"]
        require(
            type(argv) is list
            and argv
            and all(type(item) is str and item for item in argv),
            f"{field} process argv",
        )
        artifact_path = string(
            row["artifact_path"],
            f"{field} process artifact path",
        )
        require(
            Path(artifact_path).is_absolute() and argv[0] == artifact_path,
            f"{field} process executable",
        )
        require(
            sha256_text(
                row["cmdline_sha256"],
                f"{field} process cmdline",
            )
            == argv_sha256(argv),
            f"{field} process argv digest",
        )
        sha256_text(
            row["artifact_sha256"],
            f"{field} process artifact SHA-256",
        )
        env = row["env"]
        require(
            type(env) is dict
            and all(
                type(key) is str
                and key
                and type(item) is str
                and item
                for key, item in env.items()
            ),
            f"{field} process environment",
        )
        listen_port = integer(
            row["listen_port"],
            f"{field} process listen port",
            1,
        )
        require(listen_port <= 65535, f"{field} process listen port")
        if row["kind"] in ("STAGE_HEAD", "STAGE_TAIL"):
            layer_start = integer(
                row["layer_start"],
                f"{field} process layer start",
            )
            layer_end = integer(
                row["layer_end"],
                f"{field} process layer end",
                1,
            )
            require(layer_start < layer_end, f"{field} process layer range")
            backend = string(row["backend"], f"{field} process backend")
            require(
                backend in ("GPUOpenCL", "HTP0")
                and row["artifact_role"] == "worker"
                and row["name"] == "stage"
                and row["head_host"] is None
                and row["head_port"] is None
                and row["tail_host"] is None
                and row["tail_port"] is None
                and row["tail_source_port"] is None,
                f"{field} stage identity",
            )
            mode = (
                "stagenet"
                if row["kind"] == "STAGE_HEAD"
                else "tailv3"
            )
            require(
                len(argv) == 17
                and argv[1] == "-m"
                and Path(argv[2]).is_absolute()
                and argv[3:] == [
                    "--mode",
                    mode,
                    "--port",
                    str(listen_port),
                    "--driver-batch",
                    "8",
                    "--driver-context",
                    argv[10],
                    "--driver-max-prefill",
                    argv[12],
                    "--devices",
                    backend,
                    "-ngl",
                    "99",
                ]
                and argv[10].isdigit()
                and int(argv[10]) >= 8
                and argv[12].isdigit()
                and 1 <= int(argv[12]) <= 8,
                f"{field} stage argv",
            )
            runtime_root = str(Path(artifact_path).parent)
            require(
                set(env)
                == {
                    "ADSP_LIBRARY_PATH",
                    "LAYERSPLIT_MODEL_SHA256",
                    "LAYERSPLIT_PLACEMENT_CERT",
                    "LD_LIBRARY_PATH",
                    "LLAMA_LAYER_END",
                    "LLAMA_LAYER_START",
                    "PATH",
                }
                and env["ADSP_LIBRARY_PATH"] == runtime_root
                and env["LD_LIBRARY_PATH"] == runtime_root
                and env["LAYERSPLIT_PLACEMENT_CERT"] == "1"
                and env["LLAMA_LAYER_END"] == str(layer_end)
                and env["LLAMA_LAYER_START"] == str(layer_start)
                and env["PATH"] == "/system/bin:/system/xbin",
                f"{field} stage environment",
            )
            sha256_text(
                env["LAYERSPLIT_MODEL_SHA256"],
                f"{field} stage model SHA-256",
            )
        else:
            require(
                row["artifact_role"] == "relay"
                and row["backend"] == "NETWORK"
                and row["name"] == "relay"
                and row["layer_start"] is None
                and row["layer_end"] is None,
                f"{field} relay identity",
            )
            head_host = string(row["head_host"], f"{field} relay head")
            head_port = integer(
                row["head_port"],
                f"{field} relay head port",
                1,
            )
            tail_host = ipv4(row["tail_host"], f"{field} relay tail")
            tail_port = integer(
                row["tail_port"],
                f"{field} relay tail port",
                1,
            )
            source_port = integer(
                row["tail_source_port"],
                f"{field} relay source port",
                1,
            )
            require(
                max(head_port, tail_port, source_port) <= 65535
                and argv == [
                    artifact_path,
                    "--listen",
                    str(listen_port),
                    "--head",
                    f"{head_host}:{head_port}",
                    "--tail",
                    f"{tail_host}:{tail_port}",
                    "--tail-source-port",
                    str(source_port),
                ]
                and env
                == {
                    "LD_LIBRARY_PATH": str(Path(artifact_path).parent),
                    "PATH": "/system/bin:/system/xbin",
                },
                f"{field} relay argv/environment",
            )
        integer(
            row["process_start_ticks"],
            f"{field} process start ticks",
            1,
        )
        require(
            type(row["socket_inodes"]) is list
            and row["socket_inodes"]
            == sorted(set(row["socket_inodes"]))
            and all(type(item) is int and item > 0
                    for item in row["socket_inodes"]),
            f"{field} process socket inodes",
        )
    runtime_files = value["runtime_files"]
    require(
        type(runtime_files) is list and runtime_files,
        f"{field} runtime files",
    )
    runtime_roles = set()
    runtime_by_role = {}
    for index, row in enumerate(runtime_files):
        row = exact_keys(
            row,
            {"path", "role", "sha256", "stat"},
            f"{field}.runtime_files[{index}]",
        )
        role = string(row["role"], f"{field} runtime role")
        require(role not in runtime_roles, f"{field} runtime role")
        runtime_roles.add(role)
        runtime_by_role[role] = row
        require(
            Path(string(row["path"], f"{field} runtime path")).is_absolute(),
            f"{field} runtime absolute path",
        )
        sha256_text(row["sha256"], f"{field} runtime SHA-256")
        stat = exact_keys(
            row["stat"],
            {"ctime_ns", "device_id", "inode", "mode", "mtime_ns", "size"},
            f"{field} runtime stat",
        )
        for key, item in stat.items():
            integer(item, f"{field} runtime stat {key}")
    if expected_phone is not None:
        require(
            expected_phone in ("op12", "op15"),
            f"{field} expected phone",
        )
        expected_runtime = dict(COMMON_RUNTIME_FILES)
        if expected_phone == "op15":
            expected_runtime["htp_skel_v81"] = "libggml-htp-v81.so"
            expected_runtime["relay"] = "llama-stage-direct-relay"
            expected_kinds = {"STAGE_HEAD", "DIRECT_RELAY"}
        else:
            expected_runtime["htp_skel_v75"] = "libggml-htp-v75.so"
            expected_kinds = {"STAGE_TAIL"}
        require(
            runtime_roles == set(expected_runtime)
            and all(
                Path(runtime_by_role[role]["path"]).name == filename
                for role, filename in expected_runtime.items()
            ),
            f"{field} runtime closure",
        )
        if processes:
            require(
                {row["kind"] for row in processes} == expected_kinds,
                f"{field} process roles",
            )
            for row in processes:
                runtime = runtime_by_role[row["artifact_role"]]
                require(
                    row["artifact_path"] == runtime["path"]
                    and row["artifact_sha256"] == runtime["sha256"],
                    f"{field} process runtime binding",
                )
    return value


def validate_direct_peer(
    value: Any,
    phones: dict[str, dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    value = exact_keys(
        value,
        {
            "op12_address",
            "op12_port",
            "op15_address",
            "op15_port",
            "op12_socket_inode",
            "op15_socket_inode",
            "schema",
        },
        field,
    )
    require(value["schema"] == "s40-phone-direct-peer-v2", f"{field} schema")
    forward = (
        ipv4(value["op12_address"], f"{field}.op12_address"),
        integer(value["op12_port"], f"{field}.op12_port", 1),
        ipv4(value["op15_address"], f"{field}.op15_address"),
        integer(value["op15_port"], f"{field}.op15_port", 1),
    )
    require(
        forward[1] <= 65535 and forward[3] <= 65535,
        f"{field} ports",
    )
    op12_addresses = {
        address
        for row in phones["op12"]["interfaces"].values()
        for address in row["ipv4"]
    }
    op15_addresses = {
        address
        for row in phones["op15"]["interfaces"].values()
        for address in row["ipv4"]
    }
    op12_sockets = {
        (
            row["local_address"],
            row["local_port"],
            row["remote_address"],
            row["remote_port"],
        )
        for row in phones["op12"]["tcp_established"]
    }
    op15_sockets = {
        (
            row["local_address"],
            row["local_port"],
            row["remote_address"],
            row["remote_port"],
        )
        for row in phones["op15"]["tcp_established"]
    }
    require(
        forward[0] in op12_addresses
        and forward[2] in op15_addresses
        and forward in op12_sockets
        and (forward[2], forward[3], forward[0], forward[1])
        in op15_sockets,
        f"{field} is not reciprocal",
    )
    op12_inode = integer(
        value["op12_socket_inode"],
        f"{field}.op12_socket_inode",
        1,
    )
    op15_inode = integer(
        value["op15_socket_inode"],
        f"{field}.op15_socket_inode",
        1,
    )
    op12_tail = [
        row for row in phones["op12"]["processes"]
        if row["kind"] == "STAGE_TAIL"
    ]
    op15_relay = [
        row for row in phones["op15"]["processes"]
        if row["kind"] == "DIRECT_RELAY"
    ]
    op15_head = [
        row for row in phones["op15"]["processes"]
        if row["kind"] == "STAGE_HEAD"
    ]
    require(
        len(op12_tail) == 1
        and len(op15_relay) == 1
        and len(op15_head) == 1
        and op12_inode in op12_tail[0]["socket_inodes"]
        and op15_inode in op15_relay[0]["socket_inodes"],
        f"{field} socket process attribution",
    )
    head = op15_head[0]
    tail = op12_tail[0]
    relay = op15_relay[0]
    require(
        head["layer_start"] == 0
        and head["layer_end"] == tail["layer_start"]
        and head["env"]["LAYERSPLIT_MODEL_SHA256"]
        == tail["env"]["LAYERSPLIT_MODEL_SHA256"]
        and relay["head_host"] == "127.0.0.1"
        and relay["head_port"] == head["listen_port"]
        and relay["tail_host"] == forward[0]
        and relay["tail_port"] == tail["listen_port"] == forward[1]
        and relay["tail_source_port"] == forward[3],
        f"{field} process chain binding",
    )
    return value


def validate_remote_identity(
    value: Any,
    field: str,
) -> dict[str, Any]:
    value = exact_keys(
        value,
        {
            "a6000_identity",
            "completed_ns",
            "direct_peer",
            "model_id",
            "phones",
            "remote_identity_package",
            "remote_identity_package_sha256",
            "route_instance_id",
            "schema",
            "started_ns",
        },
        field,
    )
    require(value["schema"] == "s40-phone-identity-v2", f"{field} schema")
    parent_started, parent_completed = interval(value, field)
    sha256_text(value["a6000_identity"], f"{field} A6000 identity")
    package = validate_remote_package(
        value["remote_identity_package"],
        value["remote_identity_package_sha256"],
        f"{field}.remote package",
    )
    require(
        package["a6000_identity"] == value["a6000_identity"],
        f"{field} A6000 package identity",
    )
    string(value["model_id"], f"{field} model")
    string(value["route_instance_id"], f"{field} route instance")
    phones = value["phones"]
    require(
        type(phones) is dict and set(phones) == {"op12", "op15"},
        f"{field} phones",
    )
    normalized = {
        name: validate_snapshot(
            phones[name],
            f"{field}.{name}",
            expected_phone=name,
        )
        for name in ("op12", "op15")
    }
    for name, snapshot in normalized.items():
        started, completed = interval(snapshot, f"{field}.{name}")
        require(
            parent_started <= started <= completed <= parent_completed,
            f"{field}.{name} interval containment",
        )
    validate_direct_peer(value["direct_peer"], normalized, f"{field}.peer")
    return value


def validate_active_route(
    value: Any,
    phones: dict[str, dict[str, Any]],
    field: str,
) -> tuple[str, str]:
    value = exact_keys(
        value,
        {
            "direct_peer",
            "model_id",
            "route_instance_id",
            "schema",
        },
        field,
    )
    require(
        value["schema"] == "s40-phone-route-observation-v1",
        f"{field} schema",
    )
    model_id = string(value["model_id"], f"{field} model")
    route_instance_id = string(
        value["route_instance_id"],
        f"{field} route instance",
    )
    validate_direct_peer(value["direct_peer"], phones, f"{field}.peer")
    return model_id, route_instance_id


def validate_phone_observer(
    identity_path: Path,
    telemetry_path: Path,
    telemetry_stderr_path: Path,
    expected_model_id: str,
    expected_run_id: str,
) -> dict[str, Any]:
    require(
        telemetry_stderr_path.is_absolute()
        and telemetry_stderr_path.is_file()
        and telemetry_stderr_path.stat().st_size == 0,
        "phone telemetry stderr",
    )
    identity = exact_keys(
        read_json(identity_path, "phone identity"),
        {
            "argv",
            "completed_ns",
            "exit_code",
            "local_identity_package",
            "local_identity_package_sha256",
            "remote",
            "remote_identity_package_sha256",
            "run_id",
            "schema",
            "ssh_config_sha256",
            "started_ns",
            "success",
        },
        "phone identity bridge",
    )
    require(
        identity["schema"] == "s40-phone-identity-bridge-v2"
        and integer(identity["exit_code"], "phone identity exit code") == 0
        and identity["run_id"] == expected_run_id
        and identity["success"] is True,
        "phone identity bridge result",
    )
    identity_interval = interval(identity, "phone identity bridge")
    sha256_text(identity["ssh_config_sha256"], "phone SSH config SHA-256")
    local_package = validate_local_package(
        identity["local_identity_package"],
        identity["local_identity_package_sha256"],
        "phone identity local package",
    )
    remote_package_sha256 = sha256_text(
        identity["remote_identity_package_sha256"],
        "phone identity remote package SHA-256",
    )
    argv = identity["argv"]
    require(
        type(argv) is list
        and argv
        and all(type(item) is str and item for item in argv),
        "phone identity argv",
    )
    require(
        len(argv) >= 7
        and argv[-6:] == [
            "--action",
            "identity",
            "--config",
            argv[-3],
            "--model",
            expected_model_id,
        ]
        and Path(argv[-3]).is_absolute(),
        "phone identity argv binding",
    )
    remote_identity = validate_remote_identity(
        identity["remote"],
        "phone identity remote",
    )
    require(
        remote_identity["model_id"] == expected_model_id,
        "phone identity model",
    )
    require(
        remote_identity["remote_identity_package_sha256"]
        == remote_package_sha256,
        "phone identity remote package binding",
    )
    rows = read_jsonl(telemetry_path, "phone telemetry")
    require(len(rows) >= 3, "phone telemetry row count")
    received_ns = []
    remote_rows = []
    telemetry_argv = None
    for index, row in enumerate(rows):
        row = exact_keys(
            row,
            {
                "local_identity_package",
                "local_identity_package_sha256",
                "local_received_ns",
                "remote",
                "remote_argv",
                "remote_identity_package_sha256",
                "run_id",
                "schema",
                "ssh_config_sha256",
            },
            f"phone telemetry[{index}]",
        )
        require(
            row["schema"] == "s40-phone-telemetry-bridge-v2"
            and row["run_id"] == expected_run_id
            and row["ssh_config_sha256"]
            == identity["ssh_config_sha256"],
            "phone telemetry bridge schema",
        )
        require(
            row["local_identity_package"] == local_package
            and row["local_identity_package_sha256"]
            == identity["local_identity_package_sha256"]
            and row["remote_identity_package_sha256"]
            == remote_package_sha256,
            "phone telemetry dependency package changed",
        )
        current_argv = row["remote_argv"]
        require(
            type(current_argv) is list
            and all(type(item) is str and item for item in current_argv)
            and len(current_argv) >= 11
            and current_argv[-10:-8] == ["--action", "telemetry"]
            and current_argv[-8] == "--config"
            and Path(current_argv[-7]).is_absolute()
            and current_argv[-6:] == [
                "--model",
                expected_model_id,
                "--run-id",
                expected_run_id,
                "--interval-ms",
                current_argv[-1],
            ]
            and current_argv[-1].isdigit(),
            "phone telemetry argv binding",
        )
        if telemetry_argv is None:
            telemetry_argv = current_argv
        else:
            require(
                current_argv == telemetry_argv,
                "phone telemetry argv changed",
            )
        received_ns.append(integer(
            row["local_received_ns"],
            "phone telemetry receive time",
            1,
        ))
        require(
            index == 0 or received_ns[-1] >= received_ns[-2],
            "phone telemetry local time order",
        )
        remote_rows.append(row["remote"])
    header = exact_keys(
        remote_rows[0],
        {"identity", "interval_ms", "run_id", "schema"},
        "phone telemetry header",
    )
    require(
        header["schema"] == "s40-phone-telemetry-header-v1"
        and header["run_id"] == expected_run_id
        and 100
        <= integer(header["interval_ms"], "telemetry interval", 100)
        <= 10_000,
        "phone telemetry header identity",
    )
    require(
        telemetry_argv is not None
        and int(telemetry_argv[-1]) == header["interval_ms"],
        "phone telemetry interval binding",
    )
    header_identity = validate_remote_identity(
        header["identity"],
        "phone telemetry header identity",
    )
    require(
        header_identity["model_id"] == expected_model_id
        and header_identity["route_instance_id"]
        == remote_identity["route_instance_id"]
        and header_identity["a6000_identity"]
        == remote_identity["a6000_identity"],
        "phone telemetry identity changed",
    )
    require(
        header_identity["remote_identity_package"]
        == remote_identity["remote_identity_package"]
        and header_identity["remote_identity_package_sha256"]
        == remote_package_sha256,
        "phone telemetry remote dependency package changed",
    )
    require(
        identity_interval[1] <= received_ns[0]
        and remote_identity["completed_ns"]
        <= header_identity["started_ns"],
        "phone identity precedes telemetry",
    )
    footer = exact_keys(
        remote_rows[-1],
        {
            "active_route",
            "completed_ns",
            "phones",
            "run_id",
            "sample_count",
            "schema",
            "started_ns",
        },
        "phone telemetry footer",
    )
    require(
        footer["schema"] == "s40-phone-telemetry-footer-v2"
        and footer["run_id"] == expected_run_id
        and footer["active_route"] is None
        and integer(footer["sample_count"], "telemetry sample count", 1)
        == len(remote_rows) - 2,
        "phone telemetry footer identity",
    )
    footer_started, footer_completed = interval(
        footer,
        "phone telemetry footer",
    )
    samples = remote_rows[1:-1]
    require(samples, "phone telemetry has no samples")
    boot_ids = {
        name: remote_identity["phones"][name]["boot_id"]
        for name in ("op12", "op15")
    }
    all_snapshots = {
        name: [
            remote_identity["phones"][name],
            header_identity["phones"][name],
        ]
        for name in ("op12", "op15")
    }
    observed_routes = {
        remote_identity["route_instance_id"]: {
            "model_id": remote_identity["model_id"],
            "sample_count": 2,
        },
    }
    previous_remote_completed = header_identity["completed_ns"]
    for index, sample in enumerate(samples, 1):
        sample = exact_keys(
            sample,
            {
                "active_route",
                "completed_ns",
                "phones",
                "run_id",
                "sample_index",
                "schema",
                "started_ns",
            },
            f"phone telemetry sample[{index}]",
        )
        require(
            sample["schema"] == "s40-phone-telemetry-sample-v2"
            and sample["run_id"] == expected_run_id
            and integer(
                sample["sample_index"],
                "phone telemetry sample index",
                1,
            )
            == index,
            "phone telemetry sample identity",
        )
        sample_started, sample_completed = interval(
            sample,
            f"phone telemetry sample[{index}]",
        )
        require(
            previous_remote_completed <= sample_started,
            "phone telemetry remote time order",
        )
        previous_remote_completed = sample_completed
        require(
            type(sample["phones"]) is dict
            and set(sample["phones"]) == {"op12", "op15"},
            "phone telemetry sample phones",
        )
        for name in ("op12", "op15"):
            snapshot = validate_snapshot(
                sample["phones"][name],
                f"phone telemetry sample[{index}].{name}",
                boot_ids[name],
                name,
            )
            started, completed = interval(
                snapshot,
                f"phone telemetry sample[{index}].{name}",
            )
            require(
                sample_started <= started <= completed <= sample_completed,
                "phone telemetry sample interval containment",
            )
            all_snapshots[name].append(snapshot)
        active_route = sample["active_route"]
        if active_route is not None:
            model, route_instance = validate_active_route(
                active_route,
                sample["phones"],
                f"phone telemetry sample[{index}].active_route",
            )
            prior = observed_routes.setdefault(
                route_instance,
                {"model_id": model, "sample_count": 0},
            )
            require(
                prior["model_id"] == model,
                "phone route instance changed model",
            )
            prior["sample_count"] += 1
    require(
        type(footer["phones"]) is dict
        and set(footer["phones"]) == {"op12", "op15"},
        "phone telemetry footer phones",
    )
    require(
        previous_remote_completed <= footer_started,
        "phone telemetry footer time order",
    )
    for name in ("op12", "op15"):
        snapshot = validate_snapshot(
            footer["phones"][name],
            f"phone telemetry footer.{name}",
            boot_ids[name],
            name,
        )
        started, completed = interval(
            snapshot,
            f"phone telemetry footer.{name}",
        )
        require(
            footer_started <= started <= completed <= footer_completed,
            "phone telemetry footer interval containment",
        )
        all_snapshots[name].append(snapshot)
    memory_min = {}
    swap_growth = {}
    thermal_max = {}
    interface_deltas = {}
    for name in ("op12", "op15"):
        snapshots = all_snapshots[name]
        stable = ("serial", "model", "product", "device")
        for key in stable:
            require(
                len({row[key] for row in snapshots}) == 1,
                f"{name} {key} changed",
            )
        memory_min[name] = min(row["available_bytes"] for row in snapshots)
        swap_values = [row["swap_used_bytes"] for row in snapshots]
        swap_growth[name] = max(swap_values) - swap_values[0]
        require(swap_growth[name] == 0, f"{name} swap grew")
        thermal_max[name] = max(
            row["temp_millic"]
            for snapshot in snapshots
            for row in snapshot["temperatures"]
        )
        first = snapshots[0]["interfaces"]
        last = snapshots[-1]["interfaces"]
        require(
            set(first).issubset(last),
            f"{name} interface disappeared",
        )
        interface_deltas[name] = {}
        for interface, before in first.items():
            after = last[interface]
            require(
                after["rx_bytes"] >= before["rx_bytes"]
                and after["tx_bytes"] >= before["tx_bytes"],
                f"{name} interface counter reset",
            )
            interface_deltas[name][interface] = {
                "rx_bytes": after["rx_bytes"] - before["rx_bytes"],
                "tx_bytes": after["tx_bytes"] - before["tx_bytes"],
            }
    return {
        "a6000_identity": remote_identity["a6000_identity"],
        "boot_ids": boot_ids,
        "direct_peer": remote_identity["direct_peer"],
        "identity_interval_ns": list(identity_interval),
        "interface_byte_deltas": interface_deltas,
        "memory_min_bytes": memory_min,
        "model_id": expected_model_id,
        "route_instance_id": remote_identity["route_instance_id"],
        "observed_routes": [
            {
                "model_id": row["model_id"],
                "route_instance_id": route_instance,
                "sample_count": row["sample_count"],
            }
            for route_instance, row in sorted(observed_routes.items())
        ],
        "local_identity_package_sha256":
            identity["local_identity_package_sha256"],
        "remote_identity_package_sha256": remote_package_sha256,
        "run_id": expected_run_id,
        "schema": "s40-phone-observer-summary-v1",
        "stable_ids": {
            name: remote_identity["phones"][name]["serial"]
            for name in ("op12", "op15")
        },
        "status": "PASS",
        "swap_growth_bytes": swap_growth,
        "telemetry_interval_ns": [received_ns[0], received_ns[-1]],
        "thermal_max_millic": thermal_max,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--telemetry-stderr", type=Path, required=True)
    args = parser.parse_args()
    result = validate_phone_observer(
        args.identity,
        args.telemetry,
        args.telemetry_stderr,
        args.model,
        args.run_id,
    )
    print(canonical_bytes(result).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"phone observer validation failed: {error}", file=sys.stderr)
        raise SystemExit(2)
