#!/usr/bin/env python3
"""Fail-closed validator for S40 executor evidence streams."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from desktop_gateway import SERVING_ENVELOPE, parse_desktop_config
from a6000_phone_route_control import parse_stage_certificates
from validate_phone_observer import validate_active_route, validate_snapshot
from executor_bundle import (
    ISOLATED_LAUNCHER,
    validate_executor_bundle as validate_python_executor_bundle,
)
from phone_gateway import (
    COMMAND_EXECUTE,
    COMMAND_LOAD,
    COMMAND_REPLAY,
    COMMAND_UNLOAD,
    MAX_COMMAND_BYTES,
    canonical_bytes,
    exact_keys,
    integer,
    parse_command,
    parse_route_config,
    require,
    sha256_text,
    strict_json_loads,
    string,
)
from runtime_binding import (
    CONTROLLER_IDENTITY_KEYS,
    EXECUTOR_KEYS,
    ROOT_KEYS,
    RuntimeBindingError,
    read_controller_binding_evidence_capture,
    read_stable_file,
)

EXECUTED_FILE_KEYS = {
    "argv_index",
    "bytes",
    "captured_path",
    "executed_path",
    "sha256",
    "source_ctime_ns",
    "source_device",
    "source_inode",
    "source_mtime_ns",
    "source_size",
}
GATEWAY_PROCESS_IDENTITY_KEYS = {
    "cmdline_base64",
    "cmdline_sha256",
    "executable_ctime_ns",
    "executable_device",
    "executable_inode",
    "executable_mtime_ns",
    "executable_path",
    "executable_sha256",
    "executable_size",
    "gateway_pid",
    "gateway_start_time_ticks",
    "observed_ns",
}
PROHIBITED_LAUNCH_ENV = {
    "BASH_ENV",
    "ENV",
    "GCONV_PATH",
    "LD_AUDIT",
    "LD_DEBUG",
    "LD_PRELOAD",
    "NODE_OPTIONS",
    "PERL5OPT",
    "PERL5LIB",
    "PYTHONBREAKPOINT",
    "PYTHONHOME",
    "PYTHONINSPECT",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "RUBYOPT",
}
BASE_LAUNCH_ENV_KEYS = {
    "CUDA_CACHE_PATH",
    "CUDA_VISIBLE_DEVICES",
    "HOME",
    "LANG",
    "LC_ALL",
    "LD_LIBRARY_PATH",
    "LLAMA_SERVER_WARM_TIER_CONFIG",
    "NVIDIA_VISIBLE_DEVICES",
    "PATH",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONHASHSEED",
    "PYTHONNOUSERSITE",
    "S40_EVIDENCE_BUNDLE",
    "S40_EVIDENCE_BUNDLE_MANIFEST",
    "S40_EVIDENCE_BUNDLE_SHA256",
    "S40_EXECUTOR_BUNDLE",
    "S40_EXECUTOR_BUNDLE_MANIFEST",
    "S40_EXECUTOR_BUNDLE_SHA256",
    "S40_NVIDIA_SMI_PATH",
    "S40_NVIDIA_SMI_SHA256",
    "TMPDIR",
    "TZ",
}
PRIVILEGED_LAUNCH_ENV_KEYS = (
    BASE_LAUNCH_ENV_KEYS
    | {"LLAMA_SERVER_WARM_TIER_INTERNAL_TOKEN_FILE"}
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl(path: Path, field: str) -> list[dict[str, Any]]:
    require(path.is_absolute() and path.is_file(), f"{field} path")
    result = []
    with path.open("rb") as source:
        for index, raw in enumerate(source):
            require(
                raw.endswith(b"\n") and 0 < len(raw) <= MAX_COMMAND_BYTES,
                f"{field}[{index}] framing",
            )
            value = strict_json_loads(raw, f"{field}[{index}]")
            require(
                canonical_bytes(value) == raw,
                f"{field}[{index}] is not canonical",
            )
            require(type(value) is dict, f"{field}[{index}] object")
            result.append(value)
    require(result, f"{field} is empty")
    return result


def interval(row: dict[str, Any], field: str) -> None:
    started = integer(row["started_ns"], f"{field}.started_ns", 1)
    completed = integer(row["completed_ns"], f"{field}.completed_ns", 1)
    require(started <= completed, f"{field} interval")


def validate_result(
    value: Any,
    row: dict[str, Any],
    command: dict[str, Any],
    field: str,
) -> list[dict[str, Any]]:
    value = exact_keys(
        value,
        {
            "command_id",
            "controller_epoch",
            "detail",
            "executor_id",
            "executor_instance_id",
            "has_replay_snapshot",
            "kind",
            "model_id",
            "publications",
            "replay_snapshot",
            "request_complete",
            "request_id",
            "schema",
            "success",
        },
        field,
    )
    require(
        value["schema"] == "llama-server-warm-tier-result-v2"
        and value["command_id"] == row["command_id"]
        and value["controller_epoch"] == row["controller_epoch"]
        and value["executor_id"] == row["executor_id"]
        and value["executor_instance_id"]
        == row["executor_instance_id"]
        == command["executor_instance_id"]
        and value["kind"] == row["kind"]
        and value["model_id"] == row["model_id"]
        and value["request_id"] == (row["request_id"] or "")
        and value["success"] is row["success"],
        f"{field} command binding",
    )
    string(value["detail"], f"{field}.detail")
    require(
        type(value["has_replay_snapshot"]) is bool
        and type(value["request_complete"]) is bool
        and type(value["success"]) is bool,
        f"{field} booleans",
    )
    require(
        value["has_replay_snapshot"]
        is (value["replay_snapshot"] is not None),
        f"{field} replay marker",
    )
    publications = value["publications"]
    require(type(publications) is list, f"{field}.publications")
    normalized = []
    for index, publication in enumerate(publications):
        publication = exact_keys(
            publication,
            {
                "owner_id",
                "ownership_epoch",
                "position",
                "publication_index",
                "token",
            },
            f"{field}.publications[{index}]",
        )
        require(
            publication["owner_id"] == row["executor_id"],
            f"{field} publication owner",
        )
        normalized.append({
            "owner_id": publication["owner_id"],
            "ownership_epoch": integer(
                publication["ownership_epoch"],
                f"{field} ownership epoch",
                1,
            ),
            "position": integer(
                publication["position"],
                f"{field} position",
            ),
            "publication_index": integer(
                publication["publication_index"],
                f"{field} publication index",
            ),
            "token": integer(
                publication["token"],
                f"{field} token",
            ),
        })
    require(
        (
            row["kind"] == COMMAND_EXECUTE
            and value["success"] is True
            and len(normalized) == 1
        )
        or (
            row["kind"] != COMMAND_EXECUTE
            and not normalized
            and value["request_complete"] is False
        ),
        f"{field} publication cardinality",
    )
    if row["kind"] == COMMAND_EXECUTE:
        request = command["request"]
        publication = normalized[0]
        require(
            publication["ownership_epoch"] == request["ownership_epoch"]
            and publication["position"] == request["position"]
            and publication["publication_index"]
            == request["publication_index"],
            f"{field} publication frontier",
        )
        require(
            value["request_complete"]
            is (
                len(request["committed_output_tokens"]) + 1
                == command["total_output_tokens"]
            )
            and value["replay_snapshot"] is None,
            f"{field} completion frontier",
        )
    elif row["kind"] == COMMAND_REPLAY:
        require(
            value["replay_snapshot"] == command["request"]
            and value["request_complete"] is False,
            f"{field} replay frontier",
        )
    else:
        require(
            value["replay_snapshot"] is None
            and value["request_complete"] is False,
            f"{field} lifecycle frontier",
        )
    return normalized


def argv_flag(argv: list[str], flag: str, field: str) -> str:
    require(
        not any(value.startswith(flag + "=") for value in argv),
        f"{field} assignment form",
    )
    positions = [
        index
        for index, value in enumerate(argv[:-1])
        if value == flag
    ]
    require(len(positions) == 1, f"{field} flag")
    return string(argv[positions[0] + 1], f"{field} value")


def expected_nul_cmdline(argv: list[str]) -> bytes:
    require(
        argv
        and all(
            type(argument) is str
            and argument.isascii()
            and "\x00" not in argument
            for argument in argv
        ),
        "gateway argv encoding",
    )
    return b"".join(
        argument.encode("ascii") + b"\x00" for argument in argv)


def validate_gateway_process_identity(
    value: Any,
    argv: list[str],
    field: str,
    expected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    value = exact_keys(value, GATEWAY_PROCESS_IDENTITY_KEYS, field)
    for key in (
        "executable_ctime_ns",
        "executable_device",
        "executable_inode",
        "executable_mtime_ns",
        "executable_size",
        "gateway_pid",
        "gateway_start_time_ticks",
        "observed_ns",
    ):
        integer(
            value[key],
            f"{field}.{key}",
            0 if key == "executable_device" else 1,
        )
    executable = Path(string(
        value["executable_path"], f"{field}.executable_path"))
    require(
        executable.is_absolute()
        and executable.resolve() == Path(argv[0]).resolve(),
        f"{field} executable path",
    )
    sha256_text(
        value["executable_sha256"], f"{field}.executable_sha256")
    sha256_text(value["cmdline_sha256"], f"{field}.cmdline_sha256")
    try:
        cmdline = base64.b64decode(
            string(value["cmdline_base64"], f"{field}.cmdline_base64"),
            validate=True,
        )
    except (ValueError, binascii.Error) as error:
        raise RuntimeError(f"{field} cmdline encoding") from error
    require(
        cmdline == expected_nul_cmdline(argv)
        and hashlib.sha256(cmdline).hexdigest()
        == value["cmdline_sha256"],
        f"{field} cmdline binding",
    )
    if expected is not None:
        require(
            {
                key: value[key]
                for key in GATEWAY_PROCESS_IDENTITY_KEYS - {"observed_ns"}
            }
            == {
                key: expected[key]
                for key in GATEWAY_PROCESS_IDENTITY_KEYS - {"observed_ns"}
            }
            and value["observed_ns"] >= expected["observed_ns"],
            f"{field} process replacement",
        )
    return value


def validate_gateway_environment(
    value: Any,
    descriptor_root: Path,
    runtime_config_path: Path,
    executor_bundle_manifest_path: Path,
    kind: str,
) -> dict[str, str]:
    expected_keys = (
        PRIVILEGED_LAUNCH_ENV_KEYS
        if kind == "desktop"
        else BASE_LAUNCH_ENV_KEYS
    )
    require(
        type(value) is dict
        and set(value) == expected_keys
        and not (set(value) & PROHIBITED_LAUNCH_ENV)
        and all(
            type(key) is str
            and type(item) is str
            and key
            and item
            and key.isascii()
            and item.isascii()
            and "\x00" not in key
            and "\x00" not in item
            for key, item in value.items()
        ),
        "gateway environment allowlist",
    )
    evidence_manifest = Path(value["S40_EVIDENCE_BUNDLE_MANIFEST"])
    nvidia_smi = Path(value["S40_NVIDIA_SMI_PATH"])
    require(
        value["LANG"] == "C"
        and value["LC_ALL"] == "C"
        and value["PYTHONDONTWRITEBYTECODE"] == "1"
        and value["PYTHONHASHSEED"] == "0"
        and value["PYTHONNOUSERSITE"] == "1"
        and value["S40_EVIDENCE_BUNDLE"] == "1"
        and value["S40_EXECUTOR_BUNDLE"] == "1"
        and value["TZ"] == "UTC"
        and value["CUDA_VISIBLE_DEVICES"]
        == value["NVIDIA_VISIBLE_DEVICES"]
        and value["LLAMA_SERVER_WARM_TIER_CONFIG"]
        == str(runtime_config_path)
        and value["PATH"]
        == f"{descriptor_root / 'captured-runtime' / 'bin'}:/usr/bin:/bin"
        and value["HOME"] == str(descriptor_root / "run-home")
        and value["TMPDIR"] == str(descriptor_root / "run-tmp")
        and value["CUDA_CACHE_PATH"]
        == str(descriptor_root / "cuda-cache")
        and value["LD_LIBRARY_PATH"]
        == str(descriptor_root / "captured-runtime" / "lib")
        and value["S40_EXECUTOR_BUNDLE_MANIFEST"]
        == str(executor_bundle_manifest_path)
        and value["S40_EXECUTOR_BUNDLE_SHA256"]
        == file_sha256(executor_bundle_manifest_path)
        and evidence_manifest.is_absolute()
        and evidence_manifest.is_file()
        and not evidence_manifest.is_symlink()
        and value["S40_EVIDENCE_BUNDLE_SHA256"]
        == file_sha256(evidence_manifest)
        and nvidia_smi.is_absolute()
        and nvidia_smi.is_file()
        and not nvidia_smi.is_symlink()
        and value["S40_NVIDIA_SMI_SHA256"] == file_sha256(nvidia_smi),
        "gateway environment binding",
    )
    return value


def validate_gateway_executed_files(
    value: Any,
    argv: list[str],
    descriptor_root: Path,
) -> list[dict[str, Any]]:
    require(type(value) is list and value, "gateway executed files")
    indexes = set()
    for index, record in enumerate(value):
        field = f"gateway executed files[{index}]"
        record = exact_keys(record, EXECUTED_FILE_KEYS, field)
        argv_index = integer(record["argv_index"], f"{field}.argv_index")
        require(
            argv_index < len(argv) and argv_index not in indexes,
            f"{field} argv index",
        )
        indexes.add(argv_index)
        executed = Path(string(
            record["executed_path"], f"{field}.executed_path"))
        relative = Path(string(
            record["captured_path"], f"{field}.captured_path"))
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            f"{field} captured path",
        )
        captured = (descriptor_root / relative).resolve()
        try:
            captured.relative_to(descriptor_root.resolve())
        except ValueError as error:
            raise RuntimeError(f"{field} captured path") from error
        require(
            record["executed_path"] == argv[argv_index]
            and executed.is_absolute()
            and executed.is_file()
            and not executed.is_symlink()
            and captured.is_file()
            and not captured.is_symlink(),
            f"{field} paths",
        )
        executed_stat = executed.stat(follow_symlinks=False)
        captured_stat = captured.stat(follow_symlinks=False)
        digest = sha256_text(record["sha256"], f"{field}.sha256")
        require(
            integer(record["bytes"], f"{field}.bytes", 1)
            == captured_stat.st_size
            and integer(
                record["source_device"], f"{field}.source_device")
            == executed_stat.st_dev
            and integer(
                record["source_inode"], f"{field}.source_inode", 1)
            == executed_stat.st_ino
            and integer(
                record["source_size"], f"{field}.source_size", 1)
            == executed_stat.st_size
            and integer(
                record["source_mtime_ns"], f"{field}.source_mtime_ns", 1)
            == executed_stat.st_mtime_ns
            and integer(
                record["source_ctime_ns"], f"{field}.source_ctime_ns", 1)
            == executed_stat.st_ctime_ns
            and file_sha256(executed) == digest
            and file_sha256(captured) == digest,
            f"{field} identity",
        )
    separately_bound = {
        index + 1
        for index, argument in enumerate(argv[:-1])
        if argument in {
            "--controller-binding-evidence",
            "--controller-identity",
            "--evidence",
            "--route-evidence",
            "--runtime-config",
            "--wire-evidence",
        }
    }
    required = {
        index
        for index, argument in enumerate(argv)
        if index not in separately_bound
        and Path(argument).is_absolute()
        and Path(argument).is_file()
    }
    require(indexes == required and 0 in indexes,
            "gateway executed file set")
    return value


def stable_json_record(
    path: Path,
    keys: set[str],
    schema: str,
    field: str,
) -> tuple[dict[str, Any], bytes, Any]:
    require(path.is_absolute(), f"{field} path")
    try:
        raw, metadata = read_stable_file(path)
    except (OSError, RuntimeBindingError) as error:
        raise RuntimeError(f"{field}: {error}") from error
    value = strict_json_loads(raw, field)
    require(canonical_bytes(value) == raw, f"{field} is not canonical")
    value = exact_keys(value, keys, field)
    require(value["schema"] == schema, f"{field} schema")
    return value, raw, metadata


def validate_runtime_executor(
    runtime_path: Path,
    runtime_sha256: str,
    runtime_device: int,
    runtime_inode: int,
    executor_id: str,
    executor_instance_id: str,
    gateway_pid: int,
    gateway_start_time_ticks: int,
    run_id: str,
    socket_path: Path,
) -> None:
    value, raw, metadata = stable_json_record(
        runtime_path,
        ROOT_KEYS,
        "llama-server-warm-tier-runtime-v4",
        "executor runtime config",
    )
    require(
        hashlib.sha256(raw).hexdigest() == runtime_sha256
        and metadata.st_dev == runtime_device
        and metadata.st_ino == runtime_inode,
        "executor runtime file identity",
    )
    require(value["run_id"] == run_id, "executor runtime run ID")
    records = value["executors"]
    require(type(records) is list and 0 < len(records) <= 16,
            "executor runtime executors")
    seen_ids = set()
    seen_instances = set()
    matching = []
    for index, record in enumerate(records):
        field = f"executor runtime executors[{index}]"
        record = exact_keys(record, EXECUTOR_KEYS, field)
        record_id = string(record["executor_id"], f"{field}.executor_id")
        instance_id = string(
            record["executor_instance_id"],
            f"{field}.executor_instance_id",
        )
        require(
            record_id not in seen_ids and instance_id not in seen_instances,
            f"{field} duplicate identity",
        )
        seen_ids.add(record_id)
        seen_instances.add(instance_id)
        if record_id == executor_id:
            matching.append(record)
    require(len(matching) == 1, "executor runtime executor identity")
    record = matching[0]
    require(
        record["executor_instance_id"] == executor_instance_id
        and integer(
            record["expected_peer_pid"],
            "executor runtime gateway PID",
            2,
        ) == gateway_pid
        and integer(
            record["expected_peer_start_time_ticks"],
            "executor runtime gateway start time",
            1,
        ) == gateway_start_time_ticks
        and record["transport"] == "UNIX_SOCKET"
        and record["socket_path"] == str(socket_path),
        "executor runtime gateway binding",
    )


def validate_transport_descriptor(
    descriptor_path: Path,
    gateway_argv_path: Path,
    gateway_source_path: Path,
    executor_bundle_manifest_path: Path,
    config_path: Path,
    socket_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    command_path: Path,
    executor_id: str,
    kind: str,
    wire_path: Path | None,
    route_path: Path | None,
) -> dict[str, Any]:
    raw = descriptor_path.read_bytes()
    value = strict_json_loads(raw, "executor transport descriptor")
    require(
        canonical_bytes(value) == raw,
        "executor transport descriptor is not canonical",
    )
    value = exact_keys(
        value,
        {
            "controller_binding_device",
            "controller_binding_inode",
            "controller_binding_path",
            "controller_binding_sha256",
            "controller_executable_path",
            "controller_executable_sha256",
            "controller_gid",
            "controller_identity_device",
            "controller_identity_inode",
            "controller_identity_path",
            "controller_identity_published_ns",
            "controller_identity_sha256",
            "controller_pid",
            "controller_start_time_ticks",
            "controller_uid",
            "executor_bundle_manifest_sha256",
            "executor_id",
            "executor_instance_id",
            "gateway_argv_sha256",
            "gateway_config_sha256",
            "gateway_environment",
            "gateway_executed_files",
            "gateway_pid",
            "gateway_post_auth_identity",
            "gateway_prepublication_identity",
            "gateway_source_sha256",
            "gateway_start_time_ticks",
            "host_boot_id",
            "identity_captured_ns",
            "runtime_config_device",
            "runtime_config_inode",
            "runtime_config_path",
            "runtime_config_published_ns",
            "runtime_config_sha256",
            "schema",
            "socket_path",
            "transport",
        },
        "executor transport descriptor",
    )
    require(
        value["schema"] == "s40-executor-transport-descriptor-v4"
        and value["transport"] == "UNIX_SOCKET"
        and value["executor_id"] == executor_id
        and value["socket_path"] == str(socket_path),
        "executor transport descriptor identity",
    )
    executor_instance_id = string(
        value["executor_instance_id"],
        "executor transport instance ID",
    )
    gateway_pid = integer(
        value["gateway_pid"], "executor transport gateway PID", 2)
    gateway_start_time_ticks = integer(
        value["gateway_start_time_ticks"],
        "executor transport gateway start time",
        1,
    )
    host_boot_id = string(
        value["host_boot_id"], "executor transport host boot ID")
    identity_captured_ns = integer(
        value["identity_captured_ns"],
        "executor transport identity capture time",
        1,
    )
    runtime_config_published_ns = integer(
        value["runtime_config_published_ns"],
        "executor transport runtime publication time",
        identity_captured_ns,
    )
    runtime_config_sha256 = sha256_text(
        value["runtime_config_sha256"],
        "executor transport runtime config SHA-256",
    )
    runtime_config_path = Path(string(
        value["runtime_config_path"],
        "executor transport runtime config path",
    ))
    require(
        runtime_config_path.is_absolute()
        and runtime_config_path.is_file()
        and not runtime_config_path.is_symlink(),
        "executor transport runtime config path",
    )
    runtime_config_device = integer(
        value["runtime_config_device"],
        "executor transport runtime config device",
        0,
    )
    runtime_config_inode = integer(
        value["runtime_config_inode"],
        "executor transport runtime config inode",
        1,
    )
    runtime_stat = runtime_config_path.stat(follow_symlinks=False)
    require(
        runtime_stat.st_dev == runtime_config_device
        and runtime_stat.st_ino == runtime_config_inode
        and file_sha256(runtime_config_path) == runtime_config_sha256,
        "executor transport runtime config identity",
    )
    for key in (
        "executor_bundle_manifest_sha256",
        "gateway_argv_sha256",
        "gateway_config_sha256",
        "gateway_source_sha256",
    ):
        sha256_text(value[key], f"executor transport {key}")
    require(
        file_sha256(config_path) == value["gateway_config_sha256"],
        "executor transport gateway config digest",
    )
    require(
        file_sha256(gateway_source_path) == value["gateway_source_sha256"],
        "executor transport gateway source digest",
    )
    require(
        file_sha256(executor_bundle_manifest_path)
        == value["executor_bundle_manifest_sha256"],
        "executor transport bundle manifest digest",
    )
    require(
        gateway_source_path.parent
        == executor_bundle_manifest_path.parent
        and executor_bundle_manifest_path.name == "MANIFEST.json",
        "executor transport bundle paths",
    )
    validate_python_executor_bundle(
        executor_bundle_manifest_path.parent,
        executor_bundle_manifest_path,
        value["executor_bundle_manifest_sha256"],
    )
    argv_raw = gateway_argv_path.read_bytes()
    require(
        file_sha256(gateway_argv_path) == value["gateway_argv_sha256"],
        "executor transport gateway argv digest",
    )
    argv_record = strict_json_loads(argv_raw, "gateway argv")
    require(
        canonical_bytes(argv_record) == argv_raw,
        "gateway argv is not canonical",
    )
    argv_record = exact_keys(
        argv_record,
        {"argv", "schema"},
        "gateway argv",
    )
    require(
        argv_record["schema"] == "s40-gateway-argv-v4",
        "gateway argv schema",
    )
    argv = argv_record["argv"]
    require(type(argv) is list and len(argv) >= 16, "gateway argv")
    for index, argument in enumerate(argv):
        string(argument, f"gateway argv[{index}]")
    require(
        Path(argv[0]).is_absolute()
        and Path(argv[0]).is_file()
        and argv[1:4] == ["-I", "-S", "-B"]
        and argv[4] == "-c"
        and argv[5] == ISOLATED_LAUNCHER
        and Path(argv[6]).resolve()
        == executor_bundle_manifest_path.parent.resolve()
        and argv[7] == str(gateway_source_path),
        "gateway executable binding",
    )
    gateway_environment = validate_gateway_environment(
        value["gateway_environment"],
        descriptor_path.parent,
        runtime_config_path,
        executor_bundle_manifest_path,
        kind,
    )
    require(
        kind == "phone"
        or gateway_environment[
            "LLAMA_SERVER_WARM_TIER_INTERNAL_TOKEN_FILE"
        ] == str(descriptor_path.parent / "warm-tier-internal.token"),
        "gateway internal capability path",
    )
    gateway_executed_files = validate_gateway_executed_files(
        value["gateway_executed_files"],
        argv,
        descriptor_path.parent,
    )
    gateway_prepublication_identity = (
        validate_gateway_process_identity(
            value["gateway_prepublication_identity"],
            argv,
            "gateway prepublication identity",
        )
    )
    gateway_post_auth_identity = validate_gateway_process_identity(
        value["gateway_post_auth_identity"],
        argv,
        "gateway post-auth identity",
        gateway_prepublication_identity,
    )
    require(
        gateway_prepublication_identity["gateway_pid"] == gateway_pid
        and gateway_prepublication_identity["gateway_start_time_ticks"]
        == gateway_start_time_ticks
        and gateway_prepublication_identity["observed_ns"]
        == identity_captured_ns,
        "gateway prepublication identity binding",
    )
    expected_source = (
        "desktop_gateway.py" if kind == "desktop" else "phone_gateway.py"
    )
    require(
        gateway_source_path.name == expected_source
        and gateway_source_path.name != "gateway_bridge.py",
        "gateway source role",
    )
    config_flag = "--config" if kind == "desktop" else "--route-config"
    require(
        argv_flag(argv, "--socket", "gateway socket") == str(socket_path)
        and argv_flag(argv, config_flag, "gateway config")
        == str(config_path)
        and argv_flag(argv, "--evidence", "gateway command evidence")
        == str(command_path)
        and argv_flag(
            argv,
            "--runtime-config",
            "gateway runtime config",
        )
        == str(runtime_config_path)
        and argv_flag(
            argv,
            "--executor-instance-id",
            "gateway executor instance",
        )
        == executor_instance_id
        and "--runtime-config-sha256" not in argv,
        "gateway argv binding",
    )
    if kind == "desktop":
        require(
            "--wire-evidence" not in argv
            and "--route-evidence" not in argv,
            "desktop gateway has phone evidence paths",
        )
    else:
        require(
            wire_path is not None
            and route_path is not None
            and argv_flag(argv, "--wire-evidence", "gateway wire evidence")
            == str(wire_path)
            and argv_flag(argv, "--route-evidence", "gateway route evidence")
            == str(route_path),
            "phone gateway evidence binding",
        )
    run_id = argv_flag(argv, "--run-id", "gateway run ID")
    controller_identity_path = Path(argv_flag(
        argv,
        "--controller-identity",
        "gateway controller identity",
    ))
    controller_binding_path = Path(argv_flag(
        argv,
        "--controller-binding-evidence",
        "gateway controller binding evidence",
    ))
    require(
        controller_identity_path.is_absolute()
        and controller_binding_path.is_absolute(),
        "gateway controller evidence paths",
    )
    require(
        not any(
            value == "--initial-model"
            or value.startswith("--initial-model=")
            for value in argv
        ),
        "gateway initial model is forbidden",
    )
    require(stdout_path.is_file() and stderr_path.is_file(), "gateway log path")
    require(stdout_path.stat().st_size == 0, "gateway stdout is not empty")
    require(stderr_path.stat().st_size == 0, "gateway stderr is not empty")

    validate_runtime_executor(
        runtime_config_path,
        runtime_config_sha256,
        runtime_config_device,
        runtime_config_inode,
        executor_id,
        executor_instance_id,
        gateway_pid,
        gateway_start_time_ticks,
        run_id,
        socket_path,
    )

    identity, identity_raw, identity_stat = stable_json_record(
        controller_identity_path,
        CONTROLLER_IDENTITY_KEYS,
        "s40-controller-identity-lock-v1",
        "controller identity lock",
    )
    controller_pid = integer(
        identity["controller_pid"], "controller identity PID", 2)
    controller_start_time_ticks = integer(
        identity["controller_start_time_ticks"],
        "controller identity start time",
        1,
    )
    controller_uid = integer(
        identity["controller_uid"], "controller identity UID")
    controller_gid = integer(
        identity["controller_gid"], "controller identity GID")
    controller_executable_path = Path(string(
        identity["controller_executable_path"],
        "controller executable path",
    ))
    controller_executable_sha256 = sha256_text(
        identity["controller_executable_sha256"],
        "controller executable SHA-256",
    )
    require(
        controller_executable_path.is_absolute()
        and controller_executable_path.is_file()
        and not controller_executable_path.is_symlink()
        and file_sha256(controller_executable_path)
        == controller_executable_sha256,
        "controller executable identity",
    )
    identity_sha256 = hashlib.sha256(identity_raw).hexdigest()
    identity_published_ns = integer(
        value["controller_identity_published_ns"],
        "controller identity publication time",
        runtime_config_published_ns,
    )
    descriptor_controller_pid = integer(
        value["controller_pid"],
        "transport controller PID",
        2,
    )
    descriptor_controller_start_time_ticks = integer(
        value["controller_start_time_ticks"],
        "transport controller start time",
        1,
    )
    descriptor_controller_uid = integer(
        value["controller_uid"],
        "transport controller UID",
    )
    descriptor_controller_gid = integer(
        value["controller_gid"],
        "transport controller GID",
    )
    descriptor_controller_executable_path = string(
        value["controller_executable_path"],
        "transport controller executable path",
    )
    descriptor_controller_executable_sha256 = sha256_text(
        value["controller_executable_sha256"],
        "transport controller executable SHA-256",
    )
    require(
        Path(string(
            value["controller_identity_path"],
            "transport controller identity path",
        )) == controller_identity_path
        and integer(
            value["controller_identity_device"],
            "transport controller identity device",
        ) == identity_stat.st_dev
        and integer(
            value["controller_identity_inode"],
            "transport controller identity inode",
            1,
        ) == identity_stat.st_ino
        and sha256_text(
            value["controller_identity_sha256"],
            "transport controller identity SHA-256",
        ) == identity_sha256
        and descriptor_controller_pid == controller_pid
        and descriptor_controller_start_time_ticks
        == controller_start_time_ticks
        and descriptor_controller_uid == controller_uid
        and descriptor_controller_gid == controller_gid
        and descriptor_controller_executable_path
        == str(controller_executable_path)
        and descriptor_controller_executable_sha256
        == controller_executable_sha256
        and identity["run_id"] == run_id
        and identity["host_boot_id"] == host_boot_id
        and identity["runtime_config_path"] == str(runtime_config_path)
        and identity["runtime_config_device"] == runtime_config_device
        and identity["runtime_config_inode"] == runtime_config_inode
        and identity["runtime_config_sha256"] == runtime_config_sha256,
        "controller identity binding",
    )
    for field, minimum in (
        ("controller_pid", 2),
        ("controller_start_time_ticks", 1),
        ("controller_uid", 0),
        ("controller_gid", 0),
        ("runtime_config_device", 0),
        ("runtime_config_inode", 1),
    ):
        integer(identity[field], f"controller identity {field}", minimum)
    for field in ("controller_executable_sha256", "runtime_config_sha256"):
        sha256_text(identity[field], f"controller identity {field}")

    try:
        binding, binding_raw, binding_stat = (
            read_controller_binding_evidence_capture(
                controller_binding_path
            )
        )
    except (OSError, RuntimeBindingError) as error:
        raise RuntimeError(f"controller binding evidence: {error}") from error
    binding_sha256 = hashlib.sha256(binding_raw).hexdigest()
    authenticated_ns = integer(
        binding["authenticated_ns"],
        "controller binding authentication time",
        identity_published_ns,
    )
    command_rows = jsonl(command_path, "executor command ordering")
    command_started_ns = [
        integer(
            row["started_ns"],
            f"executor command ordering[{index}].started_ns",
            1,
        )
        for index, row in enumerate(command_rows)
        if "started_ns" in row
    ]
    require(
        command_started_ns and authenticated_ns <= min(command_started_ns),
        "executor command predates controller authentication",
    )
    require(
        Path(string(
            value["controller_binding_path"],
            "transport controller binding path",
        )) == controller_binding_path
        and integer(
            value["controller_binding_device"],
            "transport controller binding device",
        ) == binding_stat.st_dev
        and integer(
            value["controller_binding_inode"],
            "transport controller binding inode",
            1,
        ) == binding_stat.st_ino
        and sha256_text(
            value["controller_binding_sha256"],
            "transport controller binding SHA-256",
        ) == binding_sha256
        and binding["run_id"] == run_id
        and binding["host_boot_id"] == host_boot_id
        and binding["executor_id"] == executor_id
        and binding["executor_instance_id"] == executor_instance_id
        and binding["gateway_pid"] == gateway_pid
        and binding["gateway_start_time_ticks"]
        == gateway_start_time_ticks
        and binding["peer_pid"] == controller_pid
        and binding["peer_uid"] == controller_uid
        and binding["peer_gid"] == controller_gid
        and binding["controller_pid"] == controller_pid
        and binding["controller_start_time_ticks"]
        == controller_start_time_ticks
        and binding["controller_uid"] == controller_uid
        and binding["controller_gid"] == controller_gid
        and binding["controller_executable_path"]
        == str(controller_executable_path)
        and binding["controller_executable_sha256"]
        == controller_executable_sha256
        and binding["controller_identity_path"]
        == str(controller_identity_path)
        and binding["controller_identity_device"] == identity_stat.st_dev
        and binding["controller_identity_inode"] == identity_stat.st_ino
        and binding["controller_identity_sha256"] == identity_sha256
        and binding["runtime_config_path"] == str(runtime_config_path)
        and binding["runtime_config_device"] == runtime_config_device
        and binding["runtime_config_inode"] == runtime_config_inode
        and binding["runtime_config_sha256"] == runtime_config_sha256,
        "controller binding evidence identity",
    )
    require(
        identity_captured_ns <= runtime_config_published_ns
        <= identity_published_ns <= authenticated_ns
        <= gateway_post_auth_identity["observed_ns"],
        "executor authentication ordering",
    )
    return {
        "controller_authenticated_ns": authenticated_ns,
        "controller_binding_sha256": binding_sha256,
        "controller_identity_sha256": identity_sha256,
        "descriptor_sha256": hashlib.sha256(raw).hexdigest(),
        "executor_instance_id": executor_instance_id,
        "gateway_pid": gateway_pid,
        "gateway_start_time_ticks": gateway_start_time_ticks,
        "gateway_argv_sha256": hashlib.sha256(argv_raw).hexdigest(),
        "gateway_environment": gateway_environment,
        "gateway_executed_files": gateway_executed_files,
        "gateway_post_auth_identity": gateway_post_auth_identity,
        "gateway_prepublication_identity":
            gateway_prepublication_identity,
        "host_boot_id": host_boot_id,
        "identity_captured_ns": identity_captured_ns,
        "run_id": run_id,
        "runtime_config_device": runtime_config_device,
        "runtime_config_inode": runtime_config_inode,
        "runtime_config_path": str(runtime_config_path),
        "runtime_config_published_ns": runtime_config_published_ns,
        "runtime_config_sha256": runtime_config_sha256,
    }


def validate_runtime_probe(
    probe: Any,
    spec: Any,
    expected_nvidia_smi: Any,
) -> None:
    probe = exact_keys(
        probe,
        {
            "argv",
            "artifact_certificate_sha256",
            "backend",
            "child_argv_sha256",
            "child_executable",
            "device_memory_free_mib",
            "device_memory_total_mib",
            "device_name",
            "device_uuid",
            "host_boot_id",
            "minimum_free_device_memory_mib",
            "model_path",
            "model_sha256",
            "native_model_id",
            "nvidia_smi",
            "n_gpu_layers",
            "process_id",
            "process_start_ticks",
            "port",
            "qualification",
            "readiness_lock_sha256",
            "readiness_phase_id",
            "schema",
            "serving_envelope",
            "slot_save_path",
        },
        "runtime probe",
    )
    require(
        probe["schema"] == "s40-desktop-runtime-probe-v3",
        "runtime probe schema",
    )
    nvidia_smi = exact_keys(
        probe["nvidia_smi"],
        {"bytes", "path", "sha256"},
        "runtime probe nvidia-smi",
    )
    require(
        nvidia_smi
        == {
            "bytes": expected_nvidia_smi.bytes,
            "path": expected_nvidia_smi.path,
            "sha256": expected_nvidia_smi.sha256,
        },
        "runtime probe nvidia-smi identity",
    )
    port = integer(probe["port"], "runtime child port", 1)
    require(port <= 65535, "runtime child port")
    expected_argv = [
        str(port) if value == "{PORT}" else value
        for value in spec.child_argv_template
    ]
    require(probe["argv"] == expected_argv, "runtime child argv")
    raw_cmdline = (
        "\0".join(expected_argv) + "\0"
    ).encode("utf-8")
    require(
        probe["child_argv_sha256"]
        == hashlib.sha256(raw_cmdline).hexdigest(),
        "runtime child argv digest",
    )
    require(
        probe["child_executable"]
        == {
            "bytes": spec.child_executable.bytes,
            "path": spec.child_executable.path,
            "sha256": spec.child_executable.sha256,
        },
        "runtime child executable identity",
    )
    require(
        probe["serving_envelope"] == SERVING_ENVELOPE,
        "runtime serving envelope",
    )
    require(
        probe["qualification"] == spec.qualification,
        "runtime route qualification",
    )
    require(
        probe["native_model_id"] == spec.native_model_id,
        "native model mismatch",
    )
    require(probe["n_gpu_layers"] == spec.n_gpu_layers, "GPU layer mismatch")
    require(
        probe["slot_save_path"] == spec.slot_save_path,
        "slot save path mismatch",
    )
    integer(probe["process_id"], "runtime process ID", 1)
    integer(probe["process_start_ticks"], "runtime process start ticks", 1)
    require(
        probe["artifact_certificate_sha256"]
        == spec.artifact_certificate_sha256
        and probe["model_path"] == spec.model_path
        and probe["model_sha256"] == spec.model_sha256
        and probe["readiness_lock_sha256"]
        == spec.readiness_lock_sha256
        and probe["readiness_phase_id"] == spec.readiness_phase_id
        and probe["backend"] == spec.backend
        and probe["host_boot_id"] == spec.host_boot_id,
        "runtime readiness identity",
    )
    require(
        probe["minimum_free_device_memory_mib"]
        == spec.minimum_free_device_memory_mib,
        "runtime memory headroom policy",
    )
    if spec.backend == "CPU":
        require(
            probe["device_uuid"] is None
            and probe["device_name"] is None
            and probe["device_memory_total_mib"] is None
            and probe["device_memory_free_mib"] is None,
            "CPU runtime device identity",
        )
    else:
        require(
            probe["device_uuid"] == spec.device_uuid
            and probe["device_name"] == spec.device_name
            and integer(
                probe["device_memory_total_mib"],
                "runtime device memory total",
                1,
            )
            == spec.device_memory_total_mib
            and integer(
                probe["device_memory_free_mib"],
                "runtime device memory free",
            )
            >= spec.minimum_free_device_memory_mib,
            "CUDA runtime device identity",
        )


def validate_cache_control(
    evidence: Any,
    spec: Any,
    expected_regime: str,
) -> dict[str, Any]:
    evidence = exact_keys(
        evidence,
        {
            "argv",
            "completed_ns",
            "exit_code",
            "output",
            "schema",
            "started_ns",
            "stderr",
            "success",
        },
        "cache control evidence",
    )
    require(
        evidence["schema"] == "s40-cache-control-evidence-v1"
        and evidence["exit_code"] == 0
        and evidence["stderr"] == ""
        and evidence["success"] is True,
        "cache control process result",
    )
    interval(evidence, "cache control evidence")
    argv = evidence["argv"]
    require(
        type(argv) is list
        and argv
        in (
            [
                sys.executable,
                "-B",
                str(Path(argv[-5]).resolve()),
                "--regime",
                expected_regime,
                "--model",
                spec.model_path,
            ],
            [
                sys.executable,
                "-B",
                "-s",
                "-P",
                str(Path(argv[-5]).resolve()),
                "--regime",
                expected_regime,
                "--model",
                spec.model_path,
            ],
        )
        and Path(argv[-5]).name == "cache_control_runner.py",
        "cache control argv",
    )
    output = exact_keys(
        evidence["output"],
        {
            "after",
            "before",
            "completed_ns",
            "comparison",
            "limit_ppm",
            "model_path",
            "model_stat",
            "regime",
            "schema",
            "started_ns",
            "success",
        },
        "cache control output",
    )
    require(
        output["schema"] == "s40-cache-control-result-v1"
        and output["success"] is True
        and output["regime"] == expected_regime
        and output["model_path"] == spec.model_path
        and output["model_stat"] == spec.model_stat,
        "cache control output identity",
    )
    interval(output, "cache control output")
    require(
        evidence["started_ns"] <= output["started_ns"]
        <= output["completed_ns"] <= evidence["completed_ns"],
        "cache control process bracket",
    )
    before = exact_keys(
        output["before"],
        {"bytes", "page_size", "pages", "resident_pages", "resident_ppm"},
        "cache control before",
    )
    after = exact_keys(
        output["after"],
        {
            "bytes",
            "elapsed_ns",
            "method",
            "page_size",
            "pages",
            "resident_pages",
            "resident_ppm",
        },
        "cache control after",
    )
    for row, field in ((before, "before"), (after, "after")):
        require(
            integer(row["bytes"], f"cache {field} bytes", 1)
            == spec.model_stat["size"]
            and integer(row["pages"], f"cache {field} pages", 1) > 0
            and 0
            <= integer(
                row["resident_pages"],
                f"cache {field} resident pages",
            )
            <= row["pages"]
            and 0
            <= integer(
                row["resident_ppm"],
                f"cache {field} resident ppm",
            )
            <= 1_000_000,
            f"cache control {field} counters",
        )
    if expected_regime == "WARM_CACHE":
        require(
            output["comparison"] == "at_least"
            and output["limit_ppm"] == 950_000
            and after["resident_ppm"] >= 950_000
            and after["method"] == "COMPLETE_SEQUENTIAL_READ",
            "warm cache gate",
        )
    else:
        require(
            output["comparison"] == "at_most"
            and output["limit_ppm"] == 50_000
            and after["resident_ppm"] <= 50_000
            and after["method"] == "POSIX_FADV_DONTNEED_COMPLETE_FILE",
            "cold cache gate",
        )
    return evidence


def validate_desktop(
    config_path: Path,
    command_path: Path,
    expected_executor: str,
    executor_bundle_manifest_path: Path,
    executor_bundle_manifest_sha256: str,
) -> dict[str, Any]:
    (
        executor_id,
        role,
        mode,
        _,
        cache_regime,
        routes,
        profile_lock,
        nvidia_smi,
        config_sha256,
    ) = parse_desktop_config(
        config_path,
        executor_bundle_manifest_path=executor_bundle_manifest_path,
        executor_bundle_manifest_sha256=executor_bundle_manifest_sha256,
    )
    require(executor_id == expected_executor, "desktop executor mismatch")
    rows = jsonl(command_path, "desktop evidence")
    startup = exact_keys(
        rows[0],
        {
            "cache_regime",
            "executor_config_sha256",
            "executor_id",
            "executor_instance_id",
            "gateway_pid",
            "gateway_start_time_ticks",
            "mode",
            "profile_lock_sha256",
            "role",
            "routes",
            "run_id",
            "runtime_config_device",
            "runtime_config_inode",
            "runtime_config_path",
            "runtime_config_sha256",
            "schema",
        },
        "desktop startup",
    )
    require(
        startup["schema"] == "s40-desktop-startup-evidence-v2",
        "desktop startup schema",
    )
    require(
        startup["executor_config_sha256"] == config_sha256
        and startup["executor_id"] == executor_id
        and startup["mode"] == mode
        and startup["cache_regime"] == cache_regime
        and startup["role"] == role
        and startup["profile_lock_sha256"] == profile_lock,
        "desktop startup identity",
    )
    run_id = string(startup["run_id"], "desktop startup run ID")
    executor_instance_id = string(
        startup["executor_instance_id"],
        "desktop startup executor instance ID",
    )
    gateway_pid = integer(
        startup["gateway_pid"],
        "desktop startup gateway PID",
        1,
    )
    gateway_start_time_ticks = integer(
        startup["gateway_start_time_ticks"],
        "desktop startup gateway start ticks",
        1,
    )
    runtime_config_path = string(
        startup["runtime_config_path"],
        "desktop startup runtime config path",
    )
    require(
        Path(runtime_config_path).is_absolute(),
        "desktop startup runtime config path is not absolute",
    )
    runtime_config_device = integer(
        startup["runtime_config_device"],
        "desktop startup runtime config device",
        1,
    )
    runtime_config_inode = integer(
        startup["runtime_config_inode"],
        "desktop startup runtime config inode",
        1,
    )
    runtime_config_sha256 = sha256_text(
        startup["runtime_config_sha256"],
        "desktop startup runtime config",
    )
    require(len(rows) >= 2, "desktop cleanup evidence missing")
    cleanup = exact_keys(
        rows[-1],
        {
            "completed_ns",
            "executor_id",
            "executor_instance_id",
            "initial_active_models",
            "initial_busy_requests",
            "initial_request_sessions",
            "problems",
            "remaining_active_models",
            "remaining_request_sessions",
            "run_id",
            "runtime_config_sha256",
            "schema",
            "started_ns",
            "success",
            "unloaded",
        },
        "desktop cleanup",
    )
    require(
        cleanup["schema"] == "s40-desktop-cleanup-evidence-v2"
        and cleanup["executor_id"] == executor_id
        and cleanup["executor_instance_id"] == executor_instance_id
        and cleanup["run_id"] == run_id
        and cleanup["runtime_config_sha256"] == runtime_config_sha256
        and cleanup["success"] is True
        and cleanup["problems"] == []
        and cleanup["initial_busy_requests"] == []
        and cleanup["initial_request_sessions"] == []
        and cleanup["remaining_active_models"] == []
        and cleanup["remaining_request_sessions"] == [],
        "desktop cleanup failed",
    )
    interval(cleanup, "desktop cleanup")
    inventory = startup["routes"]
    require(
        type(inventory) is list and not inventory,
        "desktop executor did not start empty",
    )
    process_ids = set()
    instances = set()
    inventory_models = set()
    runtime_identities = {}
    runtime_processes: dict[tuple[int, int], dict[str, Any]] = {}

    def remember_runtime_process(probe: dict[str, Any]) -> None:
        key = (
            probe["process_id"],
            probe["process_start_ticks"],
        )
        record = {
            "argv": list(probe["argv"]),
            "pid": probe["process_id"],
            "start_ticks": probe["process_start_ticks"],
        }
        require(
            key not in runtime_processes
            or runtime_processes[key] == record,
            "desktop runtime process identity changed",
        )
        runtime_processes[key] = record

    allocations = {
        model_id: {
            "backend": spec.backend,
            "device_uuid": spec.device_uuid,
            "minimum_free_device_memory_mib":
                spec.minimum_free_device_memory_mib,
            "n_gpu_layers": spec.n_gpu_layers,
        }
        for model_id, spec in routes.items()
    }
    for row in inventory:
        row = exact_keys(
            row,
            {
                "instance_id",
                "logical_model_id",
                "native_model_id",
                "n_gpu_layers",
                "runtime_probe",
            },
            "desktop inventory row",
        )
        model_id = row["logical_model_id"]
        require(
            model_id in routes and model_id not in inventory_models,
            "desktop inventory model",
        )
        inventory_models.add(model_id)
        spec = routes[model_id]
        require(
            row["native_model_id"] == spec.native_model_id
            and row["n_gpu_layers"] == spec.n_gpu_layers,
            "desktop inventory allocation",
        )
        validate_runtime_probe(
            row["runtime_probe"],
            spec,
            nvidia_smi,
        )
        remember_runtime_process(row["runtime_probe"])
        process_ids.add(row["runtime_probe"]["process_id"])
        instances.add(row["instance_id"])
        runtime_identities[model_id] = (
            row["instance_id"],
            row["runtime_probe"]["process_id"],
            row["runtime_probe"]["process_start_ticks"],
        )
    command_ids = set()
    execute_count = 0
    execute_durations_ns = []
    command_lineage = []
    active_at_end = set(inventory_models)
    for index, row in enumerate(rows[1:-1], 1):
        row = exact_keys(
            row,
            {
                "command",
                "command_id",
                "completed_ns",
                "controller_epoch",
                "durability",
                "execute",
                "executor_id",
                "executor_instance_id",
                "kind",
                "lifecycle",
            "model_id",
                "request_id",
                "result",
                "role",
                "run_id",
                "runtime_config_sha256",
                "schema",
                "started_ns",
                "success",
            },
            f"desktop command[{index}]",
        )
        require(
            row["schema"] == "s40-desktop-command-evidence-v4"
            and row["executor_id"] == executor_id
            and row["executor_instance_id"] == executor_instance_id
            and row["role"] == role
            and row["run_id"] == run_id
            and row["runtime_config_sha256"] == runtime_config_sha256
            and row["durability"] == "fsync_each_record",
            "desktop command identity",
        )
        command = parse_command(
            canonical_bytes(row["command"]),
            executor_id,
            executor_instance_id,
        )
        require(
            command["command_id"] == row["command_id"]
            and command["controller_epoch"] == row["controller_epoch"]
            and command["executor_id"] == row["executor_id"]
            and command["kind"] == row["kind"]
            and command["model_id"] == row["model_id"]
            and (command["request_id"] or None) == row["request_id"],
            "desktop command frontier identity",
        )
        require(row["success"] is True, "desktop command failed")
        require(row["model_id"] in routes, "desktop command model")
        interval(row, f"desktop command[{index}]")
        command_id = integer(row["command_id"], "desktop command ID", 1)
        require(command_id not in command_ids, "duplicate desktop command")
        command_ids.add(command_id)
        publications = validate_result(
            row["result"],
            row,
            command,
            f"desktop command[{index}].result",
        )
        command_lineage.append({
            "command": command,
            "command_id": command_id,
            "controller_epoch": row["controller_epoch"],
            "executor_id": row["executor_id"],
            "kind": row["kind"],
            "model_id": row["model_id"],
            "publications": publications,
            "request_complete": row["result"]["request_complete"],
            "request_id": row["request_id"],
            "success": row["success"],
        })
        if row["kind"] == COMMAND_EXECUTE and row["success"] is True:
            execute_count += 1
            execute_durations_ns.append(
                row["completed_ns"] - row["started_ns"]
            )
            execute = exact_keys(
                row["execute"],
                {
                    "execute_quantum_tokens",
                    "full_history_per_token_reprefill",
                    "initial_history_replay",
                    "instance_id",
                    "publication_count",
                    "resident_session_reused",
                    "runtime_probe",
                    "sampler",
                    "slot_id",
                    "tokens_cached",
                    "tokens_evaluated",
                },
                "desktop execute",
            )
            require(
                execute["execute_quantum_tokens"] == 1
                and execute["publication_count"] == 1
                and execute["full_history_per_token_reprefill"] is False,
                "desktop execute quantum",
            )
            require(
                len(publications) == execute["publication_count"],
                "desktop publication count",
            )
            require(
                execute["sampler"]
                == {"seed": 0, "temperature": 0.0, "type": "greedy"},
                "desktop sampler",
            )
            spec = routes[row["model_id"]]
            validate_runtime_probe(
                execute["runtime_probe"],
                spec,
                nvidia_smi,
            )
            remember_runtime_process(execute["runtime_probe"])
            identity = (
                execute["instance_id"],
                execute["runtime_probe"]["process_id"],
                execute["runtime_probe"]["process_start_ticks"],
            )
            require(
                row["model_id"] not in runtime_identities
                or runtime_identities[row["model_id"]] == identity,
                "desktop model runtime identity changed",
            )
            runtime_identities[row["model_id"]] = identity
            if execute["resident_session_reused"] is True:
                require(
                    execute["tokens_cached"] > 0
                    and execute["tokens_evaluated"]
                    - execute["tokens_cached"] <= 1,
                    "desktop resident cache",
                )
        else:
            require(row["execute"] is None, "non-execute desktop evidence")
        lifecycle = row["lifecycle"]
        if row["kind"] == COMMAND_LOAD:
            lifecycle = exact_keys(
                lifecycle,
                {
                    "cache_control",
                    "instance_id",
                    "operation",
                    "preflight",
                    "runtime_probe",
                },
                "desktop load lifecycle",
            )
            require(
                lifecycle["operation"] == "LOAD",
                "desktop load lifecycle operation",
            )
            spec = routes[row["model_id"]]
            validate_cache_control(
                lifecycle["cache_control"],
                spec,
                cache_regime,
            )
            preflight = exact_keys(
                lifecycle["preflight"],
                {"models", "schema"},
                "desktop load preflight",
            )
            require(
                preflight["schema"]
                == "s40-desktop-empty-router-preflight-v1"
                and {
                    row["logical_model_id"]
                    for row in preflight["models"]
                }
                == set(routes)
                and all(
                    row["status"] == "unloaded"
                    for row in preflight["models"]
                ),
                "desktop load preflight identity",
            )
            validate_runtime_probe(
                lifecycle["runtime_probe"],
                spec,
                nvidia_smi,
            )
            remember_runtime_process(lifecycle["runtime_probe"])
            runtime_identities[row["model_id"]] = (
                lifecycle["instance_id"],
                lifecycle["runtime_probe"]["process_id"],
                lifecycle["runtime_probe"]["process_start_ticks"],
            )
            require(
                row["model_id"] not in active_at_end,
                "desktop duplicate model load",
            )
            active_at_end.add(row["model_id"])
        elif row["kind"] == COMMAND_UNLOAD:
            lifecycle = exact_keys(
                lifecycle,
                {
                    "instance_id",
                    "operation",
                    "process_exited",
                    "router_status",
                    "runtime_probe",
                },
                "desktop unload lifecycle",
            )
            require(
                lifecycle["operation"] == "UNLOAD"
                and lifecycle["router_status"] == "unloaded"
                and lifecycle["process_exited"] is True,
                "desktop unload lifecycle operation",
            )
            spec = routes[row["model_id"]]
            validate_runtime_probe(
                lifecycle["runtime_probe"],
                spec,
                nvidia_smi,
            )
            remember_runtime_process(lifecycle["runtime_probe"])
            require(
                runtime_identities.get(row["model_id"])
                == (
                    lifecycle["instance_id"],
                    lifecycle["runtime_probe"]["process_id"],
                    lifecycle["runtime_probe"]["process_start_ticks"],
                ),
                "desktop unload child identity",
            )
            require(
                row["model_id"] in active_at_end,
                "desktop unload of inactive model",
            )
            active_at_end.remove(row["model_id"])
        else:
            require(lifecycle is None, "unexpected desktop lifecycle evidence")
    if mode == "DUAL_STATIC_PARTIAL":
        require(
            set(runtime_identities) == set(routes)
            and len({row[0] for row in runtime_identities.values()}) == 2
            and len({(row[1], row[2]) for row in runtime_identities.values()})
            == 2,
            "C3 native child identity",
        )
    require(
        cleanup["initial_active_models"] == sorted(active_at_end),
        "desktop cleanup active-model state mismatch",
    )
    unloaded_models = set()
    for index, record in enumerate(cleanup["unloaded"]):
        record = exact_keys(
            record,
            {
                "instance_id",
                "logical_model_id",
                "native_model_id",
                "process_exited",
                "process_id",
                "process_start_ticks",
            },
            f"desktop cleanup unloaded[{index}]",
        )
        model_id = record["logical_model_id"]
        require(
            model_id in active_at_end
            and model_id not in unloaded_models
            and record["native_model_id"] == routes[model_id].native_model_id
            and record["process_exited"] is True
            and runtime_identities.get(model_id)
            == (
                record["instance_id"],
                record["process_id"],
                record["process_start_ticks"],
            ),
            "desktop cleanup child identity",
        )
        unloaded_models.add(model_id)
    require(
        unloaded_models == active_at_end,
        "desktop cleanup omitted an active model",
    )
    return {
        "allocations": allocations,
        "command_lineage": command_lineage,
        "configured_models": sorted(routes),
        "initial_active_models": sorted(
            row["logical_model_id"] for row in inventory
        ),
        "cleanup": cleanup,
        "command_records": len(rows) - 2,
        "cache_regime": cache_regime,
        "config_sha256": config_sha256,
        "execute_records": execute_count,
        "executor_instance_id": executor_instance_id,
        "fastest_execute_duration_ns": (
            min(execute_durations_ns) if execute_durations_ns else None
        ),
        "mode": mode,
        "route_qualifications": [
            routes[model_id].qualification
            for model_id in sorted(routes)
        ],
        "role": role,
        "run_id": run_id,
        "gateway_pid": gateway_pid,
        "gateway_start_time_ticks": gateway_start_time_ticks,
        "runtime_config_path": runtime_config_path,
        "runtime_config_device": runtime_config_device,
        "runtime_config_inode": runtime_config_inode,
        "runtime_config_sha256": runtime_config_sha256,
        "runtime_processes": [
            runtime_processes[key]
            for key in sorted(runtime_processes)
        ],
    }


def validate_phone(
    config_path: Path,
    command_path: Path,
    wire_path: Path,
    route_path: Path,
    expected_executor: str,
) -> dict[str, Any]:
    executor_id, specs, config_sha256 = parse_route_config(config_path)
    require(executor_id == expected_executor, "phone executor mismatch")
    route_rows = jsonl(route_path, "phone route evidence")
    startup = exact_keys(
        route_rows[0],
        {"executor_id", "route_config_sha256", "routes", "schema"},
        "phone route startup",
    )
    require(
        startup["schema"] == "s40-phone-route-config-evidence-v1"
        and startup["executor_id"] == executor_id
        and startup["route_config_sha256"] == config_sha256,
        "phone route startup identity",
    )
    startup_routes = startup["routes"]
    require(
        type(startup_routes) is list
        and len(startup_routes) == len(specs),
        "phone startup route count",
    )
    for index, row in enumerate(startup_routes):
        row = exact_keys(
            row,
            {
                "artifact_certificate_sha256",
                "batch_knee",
                "model_id",
                "n_batch",
                "n_ubatch",
                "phase_lock_sha256",
                "qualification",
                "readiness_lock_sha256",
                "readiness_phase_id",
            },
            f"phone startup route[{index}]",
        )
        model_id = row["model_id"]
        require(model_id in specs, "phone startup route model")
        spec = specs[model_id]
        require(
            row
            == {
                "artifact_certificate_sha256":
                    spec.artifact_certificate_sha256,
                "batch_knee": spec.batch_knee,
                "model_id": spec.model_id,
                "n_batch": spec.n_batch,
                "n_ubatch": spec.n_ubatch,
                "phase_lock_sha256": spec.phase_lock_sha256,
                "qualification": spec.qualification,
                "readiness_lock_sha256": spec.readiness_lock_sha256,
                "readiness_phase_id": spec.readiness_phase_id,
            },
            "phone startup route binding",
        )
    require(
        [row["model_id"] for row in startup_routes] == sorted(specs),
        "phone startup route order",
    )
    active = None
    known_instances: dict[str, str] = {}
    placement_by_instance: dict[str, dict[str, Any]] = {}
    observation_by_instance: dict[str, dict[str, Any]] = {}
    observation_placement_by_instance: dict[str, dict[str, Any]] = {}
    load_order: list[str] = []
    for index, row in enumerate(route_rows[1:], 1):
        schema = row.get("schema")
        if schema == "s40-phone-route-load-v3":
            row = exact_keys(
                row,
                {
                    "a6000_identity",
                    "artifact_certificate_sha256",
                    "model_id",
                    "model_sha256",
                    "op12_boot_id",
                    "op12_shard_sha256",
                    "op15_boot_id",
                    "op15_shard_sha256",
                    "qualification_sha256",
                    "readiness_lock_sha256",
                    "readiness_phase_id",
                    "route_observation",
                    "route_instance_id",
                    "schema",
                    "success",
                    "worker_sha256",
                },
                f"phone route[{index}]",
            )
            require(active is None and row["success"] is True, "phone double load")
            model_id = row["model_id"]
            require(model_id in specs, "phone load model")
            spec = specs[model_id]
            require(
                row["a6000_identity"] == spec.a6000_identity
                and row["model_sha256"] == spec.model_sha256
                and row["op12_boot_id"] == spec.op12_boot_id
                and row["op12_shard_sha256"] == spec.op12_shard_sha256
                and row["op15_boot_id"] == spec.op15_boot_id
                and row["op15_shard_sha256"] == spec.op15_shard_sha256
                and row["worker_sha256"] == spec.worker_sha256
                and row["qualification_sha256"]
                == spec.qualification_sha256
                and row["artifact_certificate_sha256"]
                == spec.artifact_certificate_sha256
                and row["readiness_lock_sha256"]
                == spec.readiness_lock_sha256
                and row["readiness_phase_id"] == spec.readiness_phase_id,
                "phone load readiness binding",
            )
            active = row["route_instance_id"]
            require(active not in known_instances, "phone route instance reused")
            known_instances[active] = model_id
            load_order.append(active)
            observation = exact_keys(
                row["route_observation"],
                {
                    "direct_peer",
                    "model_id",
                    "phones",
                    "route_instance_id",
                    "schema",
                },
                "phone load route observation",
            )
            require(
                observation["schema"]
                == "s40-phone-route-observation-v1"
                and observation["model_id"] == model_id
                and observation["route_instance_id"] == active
                and type(observation["phones"]) is dict
                and set(observation["phones"]) == {"op12", "op15"},
                "phone load route observation identity",
            )
            observed_placements = {}
            for phone_name, boot_id in (
                ("op12", spec.op12_boot_id),
                ("op15", spec.op15_boot_id),
            ):
                snapshot = validate_snapshot(
                    observation["phones"][phone_name],
                    f"phone load observation {phone_name}",
                    boot_id,
                    phone_name,
                )
                expected_kind = (
                    "STAGE_HEAD"
                    if phone_name == "op15"
                    else "STAGE_TAIL"
                )
                stages = [
                    process
                    for process in snapshot["processes"]
                    if process["kind"] == expected_kind
                ]
                require(
                    len(stages) == 1 and stages[0]["backend"] == "HTP0",
                    f"phone load observation {phone_name} HTP placement",
                )
                observed_placements[phone_name] = {
                    "backend": stages[0]["backend"],
                    "layer_end": stages[0]["layer_end"],
                    "layer_start": stages[0]["layer_start"],
                }
            observed_model, observed_instance = validate_active_route(
                {
                    key: value
                    for key, value in observation.items()
                    if key != "phones"
                },
                observation["phones"],
                "phone load active route",
            )
            require(
                observed_model == model_id and observed_instance == active,
                "phone load active route binding",
            )
            observation_by_instance[active] = {
                "model_id": model_id,
                "route_instance_id": active,
                "schema": observation["schema"],
            }
            observation_placement_by_instance[active] = observed_placements
        elif schema == "s40-phone-route-unload-v2":
            row = exact_keys(
                row,
                {
                    "model_id",
                    "placements",
                    "route_instance_id",
                    "schema",
                    "success",
                },
                f"phone route[{index}]",
            )
            require(
                active is not None
                and row["route_instance_id"] == active
                and row["model_id"] == known_instances[active]
                and row["success"] is True,
                "phone unload identity",
            )
            spec = specs[row["model_id"]]
            placements = row["placements"]
            require(
                type(placements) is dict
                and set(placements) == {"op12", "op15"},
                "phone unload placement roles",
            )
            normalized = {}
            for phone_name in ("op12", "op15"):
                evidence = exact_keys(
                    placements[phone_name],
                    {
                        "certificate_lines",
                        "certificate_lines_sha256",
                        "placement",
                        "session",
                    },
                    f"phone unload {phone_name}",
                )
                lines = evidence["certificate_lines"]
                require(
                    type(lines) is list and len(lines) == 2,
                    f"phone unload {phone_name} certificate lines",
                )
                for line_index, line in enumerate(lines):
                    string(
                        line,
                        f"phone unload {phone_name} line[{line_index}]",
                    )
                require(
                    hashlib.sha256(
                        "".join(line + "\n" for line in lines).encode("ascii")
                    ).hexdigest()
                    == evidence["certificate_lines_sha256"],
                    f"phone unload {phone_name} certificate lines changed",
                )
                session = evidence["session"]
                require(type(session) is dict, "phone unload session")
                process = {
                    "backend": session.get("expected_backend"),
                    "kind": (
                        "STAGE_HEAD"
                        if phone_name == "op15"
                        else "STAGE_TAIL"
                    ),
                    "layer_end": session.get("layer_end"),
                    "layer_start": session.get("layer_start"),
                    "pid": session.get("worker_pid"),
                }
                parsed = parse_stage_certificates(
                    "\n".join(lines) + "\n",
                    process,
                    (
                        spec.op15_boot_id
                        if phone_name == "op15"
                        else spec.op12_boot_id
                    ),
                    spec.n_layer,
                )
                require(
                    parsed == evidence
                    and parsed["session"]["session_end"] == "STOP"
                    and parsed["session"]["expected_backend"] == "HTP0",
                    f"phone unload {phone_name} terminal placement",
                )
                normalized[phone_name] = {
                    "backend": parsed["session"]["expected_backend"],
                    "layer_end": parsed["session"]["layer_end"],
                    "layer_start": parsed["session"]["layer_start"],
                    "steps": parsed["session"]["steps_session"],
                }
            require(
                normalized["op15"]["layer_start"] == 0
                and normalized["op15"]["layer_end"]
                == normalized["op12"]["layer_start"]
                and normalized["op12"]["layer_end"] == spec.n_layer,
                "phone unload placement coverage",
            )
            require(
                {
                    phone_name: {
                        key: value
                        for key, value in normalized[phone_name].items()
                        if key != "steps"
                    }
                    for phone_name in ("op12", "op15")
                }
                == observation_placement_by_instance[active],
                "phone terminal placement differs from load observation",
            )
            placement_by_instance[active] = normalized
            active = None
        else:
            raise RuntimeError("unknown phone route evidence schema")
    require(active is None, "phone route was not unloaded")

    wire_rows = jsonl(wire_path, "phone wire evidence")
    wire_sessions = {}
    for index, row in enumerate(wire_rows):
        row = exact_keys(
            row,
            {
                "batch_index",
                "batch_size",
                "completed_ns",
                "executor_id",
                "input_rows",
                "model_id",
                "output_rows",
                "route_epoch",
                "route_instance_id",
                "schema",
                "started_ns",
            },
            f"wire[{index}]",
        )
        require(
            row["schema"] == "s40-phone-wire-batch-v1"
            and row["executor_id"] == executor_id
            and row["route_instance_id"] in known_instances,
            "wire identity",
        )
        require(
            row["model_id"] == known_instances[row["route_instance_id"]],
            "wire route model",
        )
        interval(row, f"wire[{index}]")
        require(
            integer(row["batch_index"], "wire batch index", 1) == index + 1,
            "wire batch order",
        )
        route_epoch = integer(row["route_epoch"], "wire route epoch", 1)
        batch_size = integer(row["batch_size"], "wire batch size", 1)
        spec = specs[row["model_id"]]
        require(
            batch_size <= min(spec.n_batch, spec.n_ubatch),
            "wire batch exceeds route capacity",
        )
        inputs = row["input_rows"]
        outputs = row["output_rows"]
        require(
            type(inputs) is list
            and type(outputs) is list
            and len(inputs) == len(outputs) == batch_size,
            "wire batch cardinality",
        )
        for row_index, (before, after) in enumerate(zip(inputs, outputs)):
            before = exact_keys(
                before,
                {"position", "request_id", "route_epoch", "seq_id", "token"},
                f"wire[{index}].input[{row_index}]",
            )
            after = exact_keys(
                after,
                {"position", "request_id", "route_epoch", "seq_id", "token"},
                f"wire[{index}].output[{row_index}]",
            )
            for key in ("position", "request_id", "route_epoch", "seq_id"):
                require(before[key] == after[key], "wire row lineage")
            require(
                integer(before["route_epoch"], "wire row route epoch", 1)
                == route_epoch,
                "wire row batch epoch",
            )
            for value, field in (
                (before["position"], "wire input position"),
                (before["request_id"], "wire input request ID"),
                (before["seq_id"], "wire input sequence ID"),
                (before["token"], "wire input token"),
                (after["token"], "wire output token"),
            ):
                integer(value, field)
            key = (
                row["route_instance_id"],
                route_epoch,
                before["request_id"],
                before["seq_id"],
                before["position"],
            )
            require(key not in wire_sessions, "duplicate phone wire row")
            wire_sessions[key] = {
                "input_token": before["token"],
                "position": after["position"],
                "token": after["token"],
            }

    command_rows = jsonl(command_path, "phone command evidence")
    startup_command = exact_keys(
        command_rows[0],
        {
            "configured_models",
            "executor_id",
            "executor_instance_id",
            "gateway_pid",
            "gateway_start_time_ticks",
            "initial_active_models",
            "route_config_sha256",
            "run_id",
            "runtime_config_device",
            "runtime_config_inode",
            "runtime_config_path",
            "runtime_config_sha256",
            "schema",
        },
        "phone startup",
    )
    require(
        startup_command["schema"] == "s40-phone-startup-evidence-v2"
        and startup_command["executor_id"] == executor_id
        and startup_command["route_config_sha256"] == config_sha256
        and startup_command["configured_models"] == sorted(specs)
        and startup_command["initial_active_models"] == [],
        "phone startup identity",
    )
    run_id = string(startup_command["run_id"], "phone startup run ID")
    executor_instance_id = string(
        startup_command["executor_instance_id"],
        "phone startup executor instance ID",
    )
    gateway_pid = integer(
        startup_command["gateway_pid"],
        "phone startup gateway PID",
        1,
    )
    gateway_start_time_ticks = integer(
        startup_command["gateway_start_time_ticks"],
        "phone startup gateway start ticks",
        1,
    )
    runtime_config_path = string(
        startup_command["runtime_config_path"],
        "phone startup runtime config path",
    )
    require(
        Path(runtime_config_path).is_absolute(),
        "phone startup runtime config path is not absolute",
    )
    runtime_config_device = integer(
        startup_command["runtime_config_device"],
        "phone startup runtime config device",
        1,
    )
    runtime_config_inode = integer(
        startup_command["runtime_config_inode"],
        "phone startup runtime config inode",
        1,
    )
    runtime_config_sha256 = sha256_text(
        startup_command["runtime_config_sha256"],
        "phone startup runtime config",
    )
    command_rows = command_rows[1:]
    require(command_rows, "phone command evidence has no commands")
    command_ids = set()
    execute_count = 0
    execute_durations_ns = []
    command_lineage = []
    for index, row in enumerate(command_rows):
        row = exact_keys(
            row,
            {
                "command",
                "command_id",
                "completed_ns",
                "controller_epoch",
                "durability",
                "execute_quantum_tokens",
                "executor_id",
                "executor_instance_id",
                "full_history_per_token_reprefill",
                "initial_history_replay",
                "internal_request_id",
                "kind",
                "model_id",
                "publication_count",
                "request_id",
                "resident_session_reused",
                "result",
                "role",
                "run_id",
                "runtime_config_sha256",
                "route_epoch",
                "route_instance_id",
                "sampler",
                "schema",
                "seq_id",
                "started_ns",
                "success",
            },
            f"phone command[{index}]",
        )
        require(
            row["schema"] == "s40-phone-command-evidence-v4"
            and row["executor_id"] == executor_id
            and row["executor_instance_id"] == executor_instance_id
            and row["role"] == "PHONE"
            and row["run_id"] == run_id
            and row["runtime_config_sha256"] == runtime_config_sha256
            and row["durability"] == "fsync_each_record",
            "phone command identity",
        )
        command = parse_command(
            canonical_bytes(row["command"]),
            executor_id,
            executor_instance_id,
        )
        require(
            command["command_id"] == row["command_id"]
            and command["controller_epoch"] == row["controller_epoch"]
            and command["executor_id"] == row["executor_id"]
            and command["kind"] == row["kind"]
            and command["model_id"] == row["model_id"]
            and (command["request_id"] or None) == row["request_id"],
            "phone command frontier identity",
        )
        require(row["success"] is True, "phone command failed")
        require(row["model_id"] in specs, "phone command model")
        interval(row, f"phone command[{index}]")
        command_id = integer(row["command_id"], "phone command ID", 1)
        require(command_id not in command_ids, "duplicate phone command")
        command_ids.add(command_id)
        publications = validate_result(
            row["result"],
            row,
            command,
            f"phone command[{index}].result",
        )
        command_lineage.append({
            "command": command,
            "command_id": command_id,
            "controller_epoch": row["controller_epoch"],
            "executor_id": row["executor_id"],
            "kind": row["kind"],
            "model_id": row["model_id"],
            "publications": publications,
            "request_complete": row["result"]["request_complete"],
            "request_id": row["request_id"],
            "success": row["success"],
        })
        if row["kind"] == COMMAND_EXECUTE and row["success"] is True:
            execute_count += 1
            execute_durations_ns.append(
                row["completed_ns"] - row["started_ns"]
            )
            require(
                row["execute_quantum_tokens"] == 1
                and row["publication_count"] == 1
                and row["full_history_per_token_reprefill"] is False
                and row["sampler"]
                == {"temperature": 0.0, "type": "greedy_argmax"},
                "phone execute quantum",
            )
            require(
                len(publications) == row["publication_count"],
                "phone publication count",
            )
            require(
                row["resident_session_reused"] in (True, False)
                and row["initial_history_replay"]
                is (not row["resident_session_reused"]),
                "phone execute session evidence",
            )
            key = (
                row["route_instance_id"],
                integer(row["route_epoch"], "phone command route epoch", 1),
                integer(
                    row["internal_request_id"],
                    "phone command internal request ID",
                    1,
                ),
                integer(row["seq_id"], "phone command sequence ID"),
                publications[0]["position"] - 1,
            )
            require(
                known_instances.get(row["route_instance_id"])
                == row["model_id"],
                "phone command route model",
            )
            require(key in wire_sessions, "phone execute lacks wire lineage")
            require(
                publications[0]["position"] - 1
                == wire_sessions[key]["position"]
                and publications[0]["token"] == wire_sessions[key]["token"],
                "phone result differs from wire output",
            )
            history = (
                command["request"]["prompt_tokens"]
                + command["request"]["committed_output_tokens"]
            )
            require(
                history
                and wire_sessions[key]["input_token"] == history[-1],
                "phone wire input differs from command frontier",
            )
        else:
            require(
                row["execute_quantum_tokens"] is None
                and row["initial_history_replay"] is None
                and row["internal_request_id"] is None
                and row["publication_count"] == 0
                and row["resident_session_reused"] is None
                and row["route_epoch"] is None
                and row["route_instance_id"] is None
                and row["sampler"] is None
                and row["seq_id"] is None,
                "non-execute phone evidence",
            )
    require(load_order, "phone route was never loaded")
    boot_ids = {
        "op12": sorted({spec.op12_boot_id for spec in specs.values()}),
        "op15": sorted({spec.op15_boot_id for spec in specs.values()}),
    }
    require(
        len(boot_ids["op12"]) == 1 and len(boot_ids["op15"]) == 1,
        "phone route config spans multiple boot identities",
    )
    return {
        "boot_ids": {
            "op12": boot_ids["op12"][0],
            "op15": boot_ids["op15"][0],
        },
        "command_lineage": command_lineage,
        "command_records": len(command_rows),
        "configured_models": sorted(specs),
        "config_sha256": config_sha256,
        "execute_records": execute_count,
        "executor_instance_id": executor_instance_id,
        "fastest_execute_duration_ns": (
            min(execute_durations_ns) if execute_durations_ns else None
        ),
        "initial_active_models": startup_command["initial_active_models"],
        "route_records": len(route_rows),
        "route_qualifications": [
            specs[model_id].qualification
            for model_id in sorted(specs)
        ],
        "route_instances": [
            {
                "model_id": known_instances[route_instance],
                "placements": placement_by_instance[route_instance],
                "route_instance_id": route_instance,
            }
            for route_instance in load_order
            if route_instance in placement_by_instance
        ],
        "synchronous_route_observations": [
            observation_by_instance[route_instance]
            for route_instance in load_order
        ],
        "run_id": run_id,
        "gateway_pid": gateway_pid,
        "gateway_start_time_ticks": gateway_start_time_ticks,
        "runtime_config_path": runtime_config_path,
        "runtime_config_device": runtime_config_device,
        "runtime_config_inode": runtime_config_inode,
        "runtime_config_sha256": runtime_config_sha256,
        "runtime_processes": [],
        "wire_batches": len(wire_rows),
    }


def validate_executor_bundle(
    *,
    kind: str,
    config_path: Path,
    command_path: Path,
    executor_id: str,
    transport_descriptor_path: Path,
    gateway_argv_path: Path,
    gateway_source_path: Path,
    executor_bundle_manifest_path: Path,
    socket_path: Path,
    gateway_stdout_path: Path,
    gateway_stderr_path: Path,
    wire_path: Path | None = None,
    route_path: Path | None = None,
) -> dict[str, Any]:
    require(kind in ("desktop", "phone"), "executor kind")
    for path in (
        command_path,
        config_path,
        executor_bundle_manifest_path,
        gateway_argv_path,
        gateway_source_path,
        gateway_stderr_path,
        gateway_stdout_path,
        socket_path,
        transport_descriptor_path,
    ):
        require(path.is_absolute(), "validator paths must be absolute")
    transport = validate_transport_descriptor(
        transport_descriptor_path,
        gateway_argv_path,
        gateway_source_path,
        executor_bundle_manifest_path,
        config_path,
        socket_path,
        gateway_stdout_path,
        gateway_stderr_path,
        command_path,
        executor_id,
        kind,
        wire_path,
        route_path,
    )
    if kind == "desktop":
        require(route_path is None and wire_path is None, "desktop evidence paths")
        result = validate_desktop(
            config_path,
            command_path,
            executor_id,
            executor_bundle_manifest_path,
            file_sha256(executor_bundle_manifest_path),
        )
    else:
        require(
            route_path is not None
            and wire_path is not None
            and route_path.is_absolute()
            and wire_path.is_absolute(),
            "phone evidence paths",
        )
        result = validate_phone(
            config_path,
            command_path,
            wire_path,
            route_path,
            executor_id,
        )
    require(
        result["run_id"] == transport["run_id"]
        and result["executor_instance_id"]
        == transport["executor_instance_id"]
        and result["gateway_pid"] == transport["gateway_pid"]
        and result["gateway_start_time_ticks"]
        == transport["gateway_start_time_ticks"]
        and result["runtime_config_path"]
        == transport["runtime_config_path"]
        and result["runtime_config_device"]
        == transport["runtime_config_device"]
        and result["runtime_config_inode"]
        == transport["runtime_config_inode"]
        and result["runtime_config_sha256"]
        == transport["runtime_config_sha256"],
        "executor transport runtime binding",
    )
    return {
        **result,
        "command_sha256": file_sha256(command_path),
        "controller_authenticated_ns":
            transport["controller_authenticated_ns"],
        "controller_binding_sha256":
            transport["controller_binding_sha256"],
        "controller_identity_sha256":
            transport["controller_identity_sha256"],
        "executor_id": executor_id,
        "executor_bundle_manifest_sha256":
            file_sha256(executor_bundle_manifest_path),
        "gateway_argv_sha256": transport["gateway_argv_sha256"],
        "host_boot_id": transport["host_boot_id"],
        "identity_captured_ns": transport["identity_captured_ns"],
        "gateway_source_sha256": file_sha256(gateway_source_path),
        "kind": kind,
        "schema": "s40-executor-evidence-validation-v3",
        "status": "PASS",
        "transport_descriptor_sha256": transport["descriptor_sha256"],
        "runtime_config_published_ns":
            transport["runtime_config_published_ns"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--executor-bundle-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--executor-id", required=True)
    parser.add_argument("--gateway-argv", type=Path, required=True)
    parser.add_argument("--gateway-source", type=Path, required=True)
    parser.add_argument("--gateway-stderr", type=Path, required=True)
    parser.add_argument("--gateway-stdout", type=Path, required=True)
    parser.add_argument("--kind", choices=("desktop", "phone"), required=True)
    parser.add_argument("--route", type=Path)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument(
        "--transport-descriptor",
        type=Path,
        required=True,
    )
    parser.add_argument("--wire", type=Path)
    args = parser.parse_args()
    output = validate_executor_bundle(
        kind=args.kind,
        config_path=args.config,
        command_path=args.command,
        executor_id=args.executor_id,
        transport_descriptor_path=args.transport_descriptor,
        gateway_argv_path=args.gateway_argv,
        gateway_source_path=args.gateway_source,
        executor_bundle_manifest_path=args.executor_bundle_manifest,
        socket_path=args.socket,
        gateway_stdout_path=args.gateway_stdout,
        gateway_stderr_path=args.gateway_stderr,
        wire_path=args.wire,
        route_path=args.route,
    )
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"executor evidence rejected: {error}", file=__import__("sys").stderr)
        raise SystemExit(2)
