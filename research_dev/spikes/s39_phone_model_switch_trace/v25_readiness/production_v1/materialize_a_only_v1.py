#!/usr/bin/env python3
"""Materialize phase-bound V2.5 A_ONLY plans without executing hardware."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time
import types
from typing import Any


HERE = Path(__file__).resolve().parent
V25 = HERE.parent
S39 = V25.parent
V24 = S39 / "v24_readiness"
V23 = S39 / "v23_readiness"

INVENTORY_SCHEMA = "s39-v25-a-only-production-inventory-v1"
IDENTITY_SCHEMA = "s39-v25-a-only-fresh-artifact-identity-v1"
ROOT_SCHEMA = "s39-v25-a-only-materialization-v1"
PHASE = "A_ONLY"
GPU_UUID = "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08"
SSH_TARGET = "zhihao@172.20.74.85"
CONTROLLER_HOST = "FCHLLX01"
ADB_PORT = 5038
MAX_INPUT_BYTES = 512 * 1024 * 1024

V24_INPUT_SCHEMAS = {
    "artifact_root": "s39-cp0-r1-artifact-root-v2.4",
    "bound_root": "s39-cp0-r1-v24-bound-runtime-root-v1",
    "candidate": "s39-cp0-r1-candidate-v1",
    "contract": "s39-cp0-r1-evidence-contract-v2.4",
    "cuda_monolithic_launch": "s39-cp0-r1-v24-cuda-monolithic-launch-v1",
    "cuda_route_launch": "s39-cp0-r1-v24-cuda-route-launch-v1",
    "fresh_readiness": "s39-cp0-r1-fast-fresh-readiness-v2.4",
    "identity_binding_receipt": (
        "s39-cp0-r1-v24-identity-binding-receipt-v1"
    ),
    "identity_binding_attestation": (
        "s39-cp0-r1-v24-identity-binding-attestation-v1"
    ),
    "identity_binding_stage_receipt": (
        "s39-cp0-r1-v24-stage-receipt-v1"
    ),
    "joint_capture_plan": "s39-cp0-r1-v24-joint-capture-plan-v1",
    "orchestration_plan": "s39-cp0-r1-v24-a-only-orchestration-plan-v2",
    "phase_lock": "s39-cp0-r1-phase-lock-v2.4",
    "phone_route_launch": "s39-cp0-r1-v24-phone-route-launch-v1",
    "preparation": "s39-cp0-r1-reboot-preparation-v2.4",
    "prospective_root": "s39-cp0-r1-v24-prospective-runtime-root-v1",
    "runtime_plan": "s39-cp0-r1-runtime-bundle-plan-v2.4",
    "token_history": "s39-cp0-r1-token-history-v2.4",
    "tokenizer_plan": "s39-cp0-r1-a-only-tokenizer-plan-v2",
}
V25_INPUT_SCHEMAS = {
    **V24_INPUT_SCHEMAS,
    "contract_v25": "s39-cp0-r1-evidence-contract-v2.5",
}
PRE_INPUT_NAMES = {
    "phase_lock": "phase-lock.jsonl",
    "phase_preflight": "phase-preflight.jsonl",
    "quality_corpus": "quality-corpus.jsonl",
    "route_lock": "route-lock.jsonl",
}
LOCAL_ARTIFACT_ROLES = {
    "adb",
    "identity_file",
    "identity_public_key",
    "known_hosts",
    "managed_launcher",
    "phone_guard",
    "python",
    "remote_cuda_capture",
    "remote_fan_in_contract",
    "remote_fan_in_execute",
    "remote_history",
    "ssh",
    "ssh_keygen",
    "v25_common",
}
FAN_IN_INPUT_ROLES = {
    "artifact_root",
    "candidate",
    "contract",
    "fresh_readiness",
    "orchestration_plan",
    "phase_lock",
    "pre.phase_lock",
    "pre.phase_preflight",
    "pre.quality_corpus",
    "pre.route_lock",
    "preparation",
    "runtime_plan",
}
REMOTE_ARTIFACT_ROLES = {
    "adb",
    "cuda_monolithic_producer",
    "fan_in_authority",
    "fan_in_producer",
    "joint_phone_cuda_producer",
    "nvidia_smi",
    "phone_guard",
    "phone_guard_policy",
    "production_common",
    "python",
    "v24_common",
    "v24_contract_builder",
}
LOCAL_SOURCE_BINDINGS = {
    "managed_launcher": ("v23", "managed_runtime_launcher"),
    "phone_guard": ("v25", "remote_phone_guard"),
    "remote_cuda_capture": ("v25", "remote_cuda_capture"),
    "remote_fan_in_contract": ("v25", "remote_fan_in_contract"),
    "remote_fan_in_execute": ("v25", "remote_fan_in_execute"),
    "remote_history": ("v25", "remote_history_driver"),
    "v25_common": ("v25", "common"),
}
REMOTE_SOURCE_BINDINGS = {
    "cuda_monolithic_producer": ("v24", "cuda_monolithic_producer"),
    "fan_in_authority": ("v24", "authority"),
    "fan_in_producer": ("v24", "remote_fan_in"),
    "joint_phone_cuda_producer": ("v24", "joint_phone_cuda_producer"),
    "phone_guard": ("v25", "remote_phone_guard"),
    "production_common": ("v24", "production_common"),
    "v24_common": ("v24", "common"),
    "v24_contract_builder": ("v24", "contract_builder"),
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
ARTIFACT_KEYS = {"bytes", "path", "sha256", "stat"}


class MaterializeError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MaterializeError(message)


def exact(value: Any, expected: Any, field: str) -> None:
    require(
        type(value) is type(expected) and value == expected,
        f"E_VALUE: {field}",
    )


def exact_keys(value: Any, expected: set[str], field: str) -> dict[str, Any]:
    require(type(value) is dict, f"E_TYPE: {field}")
    actual = set(value)
    require(
        actual == expected,
        f"E_KEYS: {field}: missing={sorted(expected - actual)}, "
        f"unknown={sorted(actual - expected)}",
    )
    return value


def integer(value: Any, field: str, minimum: int = 0) -> int:
    require(
        type(value) is int and minimum <= value <= (1 << 63) - 1,
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
        raise MaterializeError("E_CANONICAL") from error


def canonical_compact(value: Any) -> bytes:
    return canonical_bytes(value)[:-1]


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        require(key not in result, f"E_DUPLICATE_KEY: {key}")
        result[key] = value
    return result


def parse_json(raw: bytes, field: str) -> Any:
    try:
        return json.loads(
            raw.decode("ascii"),
            object_pairs_hook=strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                MaterializeError(f"E_JSON_NUMBER: {value}")
            ),
            parse_float=lambda value: (_ for _ in ()).throw(
                MaterializeError(f"E_JSON_FLOAT: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MaterializeError(f"E_JSON: {field}") from error


def read_regular(path: Path, field: str, maximum: int = MAX_INPUT_BYTES) -> bytes:
    require(path.is_absolute(), f"E_ABSOLUTE: {field}")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        require(
            stat.S_ISREG(before.st_mode) and 0 < before.st_size <= maximum,
            f"E_FILE: {field}",
        )
        raw = bytearray()
        while block := os.read(descriptor, 1024 * 1024):
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
    require(identity(before) == identity(after), f"E_CHANGED: {field}")
    require(len(raw) == before.st_size, f"E_CHANGED: {field}")
    return bytes(raw)


def read_canonical(path: Path, field: str) -> tuple[dict[str, Any], bytes]:
    raw = read_regular(path, field)
    value = parse_json(raw, field)
    require(type(value) is dict, f"E_TYPE: {field}")
    exact(canonical_bytes(value), raw, f"{field}.canonical")
    return value, raw


def stat_record(metadata: os.stat_result) -> dict[str, Any]:
    return {
        "build_id": None,
        "ctime_ns": metadata.st_ctime_ns,
        "device_id": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": metadata.st_mode,
        "mtime_ns": metadata.st_mtime_ns,
        "size": metadata.st_size,
    }


def artifact_from_path(path: Path, field: str) -> dict[str, Any]:
    raw = read_regular(path, field)
    return {
        "bytes": len(raw),
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "stat": stat_record(path.stat(follow_symlinks=False)),
    }


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
        require(bool(metadata["mode"] & 0o111), f"E_EXECUTABLE: {field}")
    return value


def verify_local_artifact(value: Any, field: str) -> dict[str, Any]:
    value = validate_artifact(value, field)
    observed = artifact_from_path(Path(value["path"]), f"{field}.observed")
    exact(observed, value, field)
    return value


def validate_composition_bindings(
    contract: dict[str, Any],
    local: dict[str, dict[str, Any]],
    remote: dict[str, dict[str, Any]],
) -> None:
    composition = contract["composition"]
    for role, (section, name) in sorted(LOCAL_SOURCE_BINDINGS.items()):
        source = composition[section][name]
        expected_path = S39 / source["path"]
        exact(local[role]["path"], str(expected_path), f"identity.local.{role}.path")
        exact(
            (local[role]["bytes"], local[role]["sha256"]),
            (source["bytes"], source["sha256"]),
            f"identity.local.{role}.content",
        )
    remote_s39 = Path("/home/zhihao/llama.cpp-s40/research_dev/spikes/"
                      "s39_phone_model_switch_trace")
    for role, (section, name) in sorted(REMOTE_SOURCE_BINDINGS.items()):
        source = composition[section][name]
        exact(
            remote[role]["path"],
            str(remote_s39 / source["path"]),
            f"identity.remote.{role}.path",
        )
        exact(
            (remote[role]["bytes"], remote[role]["sha256"]),
            (source["bytes"], source["sha256"]),
            f"identity.remote.{role}.content",
        )
    exact(
        (
            remote["python"]["bytes"],
            remote["python"]["path"],
            remote["python"]["sha256"],
        ),
        (
            contract["topology"]["cuda_python_bytes"],
            contract["topology"]["cuda_python_path"],
            contract["topology"]["cuda_python_sha256"],
        ),
        "identity.remote.python",
    )


def validate_remote_input_bindings(
    identity: dict[str, Any],
    v24_inputs: dict[str, dict[str, Any]],
    pre_inputs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    expected = {
        role: v24_inputs[role]
        for role in FAN_IN_INPUT_ROLES
        if not role.startswith("pre.")
    }
    expected.update({
        f"pre.{role}": pre_inputs[role]
        for role in PRE_INPUT_NAMES
    })
    exact(set(expected), FAN_IN_INPUT_ROLES, "remote_inputs.expected_roles")
    observed = identity["remote_inputs"]
    for role in sorted(FAN_IN_INPUT_ROLES):
        exact(
            (observed[role]["bytes"], observed[role]["sha256"]),
            (expected[role]["bytes"], expected[role]["sha256"]),
            f"identity.remote_inputs.{role}.content",
        )
    pre_root = Path(observed["pre.phase_lock"]["path"]).parents[1]
    for role, filename in sorted(PRE_INPUT_NAMES.items()):
        exact(
            observed[f"pre.{role}"]["path"],
            str(pre_root / "raw" / filename),
            f"identity.remote_inputs.pre.{role}.path",
        )
    return observed


def load_source(name: str, path: Path) -> types.ModuleType:
    raw = read_regular(path, f"source.{name}")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def phase_suffix(outer: str, inner: str) -> str:
    outer = text(outer, "phase.outer", 128)
    inner = text(inner, "phase.inner", 128)
    outer_prefix = "cp0-r1-v25-a-only-"
    inner_prefix = "cp0-r1-v24-a-only-"
    require(
        outer.startswith(outer_prefix)
        and inner.startswith(inner_prefix)
        and outer[len(outer_prefix):] == inner[len(inner_prefix):],
        "E_PHASE_LINK",
    )
    return outer[len(outer_prefix):]


def validate_identity(
    value: Any,
    raw: bytes,
    discovery: dict[str, Any],
    discovery_raw: bytes,
    contract: dict[str, Any],
) -> dict[str, Any]:
    value = exact_keys(
        value,
        {
            "adb_server_process",
            "completed_ns",
            "controller_boot_id",
            "discovery_sha256",
            "gpu_uuid",
            "local_artifacts",
            "outer_phase_id",
            "phase",
            "remote_artifacts",
            "remote_inputs",
            "rtx_boot_id",
            "schema",
            "started_ns",
            "v24_phase_id",
        },
        "identity",
    )
    del raw
    exact(value["schema"], IDENTITY_SCHEMA, "identity.schema")
    exact(value["phase"], PHASE, "identity.phase")
    phase_suffix(value["outer_phase_id"], value["v24_phase_id"])
    exact(
        value["outer_phase_id"],
        discovery["phase_id"],
        "identity.outer_phase_id",
    )
    exact(
        value["discovery_sha256"],
        hashlib.sha256(discovery_raw).hexdigest(),
        "identity.discovery",
    )
    started = integer(value["started_ns"], "identity.started_ns", 1)
    completed = integer(value["completed_ns"], "identity.completed_ns", started + 1)
    require(discovery["completed_ns"] <= started < completed, "E_IDENTITY_ORDER")
    exact(
        value["controller_boot_id"],
        discovery["controller"]["boot_id"],
        "identity.controller_boot",
    )
    exact(value["rtx_boot_id"], discovery["cuda"]["boot_id"], "identity.rtx_boot")
    exact(value["gpu_uuid"], GPU_UUID, "identity.gpu_uuid")
    local = exact_keys(
        value["local_artifacts"],
        LOCAL_ARTIFACT_ROLES,
        "identity.local_artifacts",
    )
    remote = exact_keys(
        value["remote_artifacts"],
        REMOTE_ARTIFACT_ROLES,
        "identity.remote_artifacts",
    )
    for role in sorted(local):
        verify_local_artifact(local[role], f"identity.local.{role}")
    for role in sorted(remote):
        validate_artifact(remote[role], f"identity.remote.{role}")
    validate_composition_bindings(contract, local, remote)
    remote_inputs = exact_keys(
        value["remote_inputs"],
        FAN_IN_INPUT_ROLES,
        "identity.remote_inputs",
    )
    remote_paths = []
    for role in sorted(remote_inputs):
        artifact = validate_artifact(
            remote_inputs[role],
            f"identity.remote_inputs.{role}",
        )
        remote_paths.append(artifact["path"])
    require(
        len(remote_paths) == len(set(remote_paths)),
        "E_REMOTE_INPUT_PATH_REUSE",
    )
    exact(
        (local["phone_guard"]["bytes"], local["phone_guard"]["sha256"]),
        (remote["phone_guard"]["bytes"], remote["phone_guard"]["sha256"]),
        "identity.phone_guard_content",
    )
    server = exact_keys(
        value["adb_server_process"],
        {
            "argv",
            "boot_id",
            "executable_path",
            "listen_host",
            "listen_port",
            "pid",
            "start_ticks",
        },
        "identity.adb_server_process",
    )
    exact(server["boot_id"], value["rtx_boot_id"], "identity.adb_server.boot")
    exact(
        server["executable_path"],
        remote["adb"]["path"],
        "identity.adb_server.executable",
    )
    exact(server["listen_host"], "127.0.0.1", "identity.adb_server.host")
    exact(server["listen_port"], ADB_PORT, "identity.adb_server.port")
    argv = server["argv"]
    require(
        type(argv) is list
        and len(argv) == 7
        and argv[:6]
        == [
            "adb",
            "-L",
            f"tcp:{ADB_PORT}",
            "fork-server",
            "server",
            "--reply-fd",
        ]
        and type(argv[6]) is str
        and argv[6].isascii()
        and argv[6].isdigit()
        and str(int(argv[6])) == argv[6]
        and int(argv[6]) >= 0,
        "E_ADB_SERVER_ARGV",
    )
    integer(server["pid"], "identity.adb_server.pid", 1)
    integer(server["start_ticks"], "identity.adb_server.start_ticks", 1)
    return value


def validate_inventory(
    value: Any,
    contract: dict[str, Any],
) -> dict[str, Any]:
    value = exact_keys(
        value,
        {
            "acquisition_started_ns",
            "desktop_forbidden_listen_ports",
            "desktop_forbidden_processes",
            "local_forward_ports",
            "output_paths",
            "phase",
            "phase_id",
            "phone_forbidden_listen_ports",
            "phone_forbidden_processes",
            "pre_inputs",
            "schema",
            "ssh",
            "v24_inputs",
            "v24_phase_id",
        },
        "inventory",
    )
    exact(value["schema"], INVENTORY_SCHEMA, "inventory.schema")
    exact(value["phase"], PHASE, "inventory.phase")
    phase_suffix(value["phase_id"], value["v24_phase_id"])
    integer(
        value["acquisition_started_ns"],
        "inventory.acquisition_started_ns",
        1,
    )
    inputs = exact_keys(
        value["v24_inputs"],
        set(V25_INPUT_SCHEMAS),
        "inventory.v24_inputs",
    )
    for name in sorted(inputs):
        validate_artifact(inputs[name], f"inventory.v24_inputs.{name}")
    require(
        len({value["path"] for value in inputs.values()}) == len(inputs),
        "E_V24_INPUT_PATH_REUSE",
    )
    pre = exact_keys(
        value["pre_inputs"],
        set(PRE_INPUT_NAMES),
        "inventory.pre_inputs",
    )
    for name, filename in sorted(PRE_INPUT_NAMES.items()):
        record = validate_artifact(pre[name], f"inventory.pre_inputs.{name}")
        exact(Path(record["path"]).name, filename, f"inventory.pre_inputs.{name}.name")
    require(
        len({value["path"] for value in pre.values()}) == len(pre)
        and not (
            {value["path"] for value in pre.values()}
            & {value["path"] for value in inputs.values()}
        ),
        "E_PRE_INPUT_PATH_REUSE",
    )
    output_paths = exact_keys(
        value["output_paths"],
        {
            "cuda_monolithic",
            "fan_in_acquisition",
            "fan_in_bundle_root",
            "fan_in_runtime",
            "joint_phone_cuda",
            "remote_root",
        },
        "inventory.output_paths",
    )
    for name, path in output_paths.items():
        absolute_path(path, f"inventory.output_paths.{name}")
    require(
        len(set(output_paths.values())) == len(output_paths),
        "E_OUTPUT_PATH_REUSE",
    )
    ports = exact_keys(
        value["local_forward_ports"],
        {"cuda_monolithic", "joint_phone_cuda", "remote_fan_in"},
        "inventory.local_forward_ports",
    )
    normalized_ports = []
    for name, pair in ports.items():
        pair = exact_keys(pair, {"local", "remote"}, f"inventory.ports.{name}")
        for side in ("local", "remote"):
            normalized_ports.append(integer(pair[side], f"inventory.ports.{name}.{side}", 1))
    require(
        all(port <= 65535 for port in normalized_ports)
        and len(normalized_ports) == len(set(normalized_ports)),
        "E_FORWARD_PORTS",
    )
    for name in (
        "desktop_forbidden_listen_ports",
        "desktop_forbidden_processes",
        "phone_forbidden_listen_ports",
        "phone_forbidden_processes",
    ):
        require(type(value[name]) is dict or type(value[name]) is list, f"E_TYPE: {name}")
    ssh = exact_keys(
        value["ssh"],
        {
            "connect_timeout_s",
            "identity_public_key_fingerprint",
            "shutdown_timeout_ms",
            "startup_timeout_ms",
        },
        "inventory.ssh",
    )
    timeout = integer(ssh["connect_timeout_s"], "inventory.ssh.timeout", 1)
    require(timeout <= 60, "E_SSH_TIMEOUT")
    for name in ("shutdown_timeout_ms", "startup_timeout_ms"):
        timeout = integer(ssh[name], f"inventory.ssh.{name}", 1)
        require(timeout <= 600_000, f"E_TIMEOUT: {name}")
    fingerprint = text(
        ssh["identity_public_key_fingerprint"],
        "inventory.ssh.fingerprint",
        128,
    )
    require(fingerprint.startswith("SHA256:"), "E_SSH_FINGERPRINT")
    exact(contract["phase"], PHASE, "contract.phase")
    exact(contract["topology"]["cuda_gpu_uuid"], GPU_UUID, "contract.gpu_uuid")
    exact(contract["topology"]["cuda_ssh_target"], SSH_TARGET, "contract.ssh_target")
    return value


def managed_component(
    component_id: str,
    artifact: dict[str, Any],
) -> dict[str, Any]:
    return {
        "bytes": artifact["bytes"],
        "component_id": component_id,
        "path": artifact["path"],
        "sha256": artifact["sha256"],
        "stat": artifact["stat"],
    }


def artifact_only(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in ("bytes", "path", "sha256", "stat")
    }


def ssh_binding(
    inventory: dict[str, Any],
    identity: dict[str, Any],
) -> dict[str, Any]:
    local = identity["local_artifacts"]
    remote = identity["remote_artifacts"]
    config = inventory["ssh"]
    return {
        "_expected_boot_id": identity["rtx_boot_id"],
        "boot_id_source": "phase_fresh_snapshot",
        "connect_timeout_s": config["connect_timeout_s"],
        "gpu_uuid": GPU_UUID,
        "host_key_alias": "172.20.74.85",
        "identity_file_path": local["identity_file"]["path"],
        "identity_file_sha256": local["identity_file"]["sha256"],
        "identity_file_stat": local["identity_file"]["stat"],
        "identity_public_key_fingerprint": config[
            "identity_public_key_fingerprint"
        ],
        "identity_public_key_path": local["identity_public_key"]["path"],
        "identity_public_key_sha256": local["identity_public_key"]["sha256"],
        "identity_public_key_stat": local["identity_public_key"]["stat"],
        "known_hosts_path": local["known_hosts"]["path"],
        "known_hosts_sha256": local["known_hosts"]["sha256"],
        "known_hosts_stat": local["known_hosts"]["stat"],
        "nvidia_smi_path": remote["nvidia_smi"]["path"],
        "remote_python_path": remote["python"]["path"],
        "remote_python_sha256": remote["python"]["sha256"],
        "remote_python_stat": remote["python"]["stat"],
        "shutdown_timeout_ms": config["shutdown_timeout_ms"],
        "ssh_path": local["ssh"]["path"],
        "ssh_port": 22,
        "ssh_sha256": local["ssh"]["sha256"],
        "ssh_stat": local["ssh"]["stat"],
        "ssh_target": SSH_TARGET,
        "ssh_keygen_path": local["ssh_keygen"]["path"],
        "ssh_keygen_sha256": local["ssh_keygen"]["sha256"],
        "ssh_keygen_stat": local["ssh_keygen"]["stat"],
        "startup_timeout_ms": config["startup_timeout_ms"],
    }


def public_ssh(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key != "_expected_boot_id"
    }


def build_history_plan(
    contract: dict[str, Any],
    inventory: dict[str, Any],
    identity: dict[str, Any],
) -> dict[str, Any]:
    remote = identity["remote_artifacts"]
    history_sources = contract["composition"]["history"]
    inputs = contract["quality"]["remote_inputs"]
    support = {
        "history_common": {
            **history_sources["history_common"]["source"],
            "path": history_sources["history_common"]["remote_path"],
        },
        "python": {
            "bytes": remote["python"]["bytes"],
            "path": remote["python"]["path"],
            "sha256": remote["python"]["sha256"],
        },
        "validator": {
            **history_sources["validator"]["source"],
            "path": history_sources["validator"]["remote_path"],
        },
    }
    validator_wrapper = (
        "import runpy,sys;"
        "sys.path.insert(0,sys.argv.pop(1));"
        "runpy.run_path(sys.argv.pop(1),run_name='__main__')"
    )
    command = [
        support["python"]["path"],
        "-I",
        "-c",
        validator_wrapper,
        str(Path(support["validator"]["path"]).parent),
        support["validator"]["path"],
    ]
    for flag, name in (
        ("--candidate", "candidate"),
        ("--corpus", "corpus"),
        ("--history", "history"),
        ("--tokenizer-plan", "tokenizer_plan"),
    ):
        command.extend([flag, inputs[name]["path"]])
    return {
        "command_argv": command,
        "inputs": copy.deepcopy(inputs),
        "remote_cwd": str(Path(inputs["candidate"]["path"]).parent),
        "schema": "s39-v25-remote-history-validation-plan-v1",
        "ssh": public_ssh(ssh_binding(inventory, identity)),
        "support": support,
    }


def build_managed_plan(
    *,
    bundle_id: str,
    producer_argv: list[str],
    components: dict[str, dict[str, Any]],
    forward: dict[str, int],
    inventory: dict[str, Any],
    identity: dict[str, Any],
) -> dict[str, Any]:
    ssh = ssh_binding(inventory, identity)
    result = {
        "android": None,
        "bundle_id": bundle_id,
        "components": [
            managed_component(name, artifact)
            for name, artifact in sorted(components.items())
        ],
        "endpoint": "cuda",
        "launcher_component_id": "python",
        "mode": "remote_cuda",
        "route": {
            "argv": producer_argv,
            "cwd": inventory["output_paths"]["remote_root"],
            "environment": {
                "CUDA_VISIBLE_DEVICES": GPU_UUID,
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": "/usr/bin:/bin",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            "kind": "remote_exec",
            "local_forward": {
                "local_host": "127.0.0.1",
                "local_port": forward["local"],
                "remote_host": "127.0.0.1",
                "remote_port": forward["remote"],
            },
        },
        "schema": "s39-managed-runtime-launch-plan-v1",
        "ssh": ssh,
    }
    return result


def build_cuda_pair(
    *,
    role: str,
    contract: dict[str, Any],
    inventory: dict[str, Any],
    identity: dict[str, Any],
    materialized_inputs: dict[str, dict[str, Any]],
    phones: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    config = {
        "cuda_monolithic": {
            "confirm": "RUN_V24_CUDA_MONOLITHIC_A_ONLY",
            "producer": "cuda_monolithic_producer",
            "sequence": 1,
        },
        "joint_phone_cuda": {
            "confirm": "RUN_V24_JOINT_PHONE_CUDA_A_ONLY",
            "producer": "joint_phone_cuda_producer",
            "sequence": 2,
        },
    }[role]
    remote = identity["remote_artifacts"]
    output = inventory["output_paths"][role]
    producer = remote[config["producer"]]
    if role == "cuda_monolithic":
        mono = materialized_inputs["cuda_monolithic_launch"]
        histories = materialized_inputs["token_history"]
        mechanism = materialized_inputs["phone_route_launch"]["value"][
            "mechanism_commands"
        ]
        argv = [
            remote["python"]["path"],
            "-I",
            producer["path"],
            "--output",
            output,
            "--phase-id",
            inventory["v24_phase_id"],
            "--pre-dir",
            str(Path(inventory["pre_inputs"]["phase_lock"]["path"]).parents[1]),
            "--started",
            str(inventory["acquisition_started_ns"]),
            "--plan",
            materialized_inputs["orchestration_plan"]["sha256"],
            "--mechanism-commands-sha256",
            hashlib.sha256(canonical_bytes(mechanism)).hexdigest(),
            "--model-sha256",
            materialized_inputs["cuda_monolithic_launch"]["value"]["model_sha256"],
            "--histories",
            histories["path"],
            "--histories-sha256",
            histories["sha256"],
            "--launch-plan",
            mono["path"],
            "--launch-plan-sha256",
            mono["sha256"],
            "--execute",
            "--confirm",
            config["confirm"],
        ]
        joint_bindings = None
        components = {
            "python": remote["python"],
            "nvidia_smi": remote["nvidia_smi"],
            "producer": producer,
            "cuda_monolithic_launch": mono,
            "token_history": histories,
        }
    else:
        capture = materialized_inputs["joint_capture_plan"]
        argv = [
            remote["python"]["path"],
            "-I",
            producer["path"],
            "--capture-plan",
            capture["path"],
            "--capture-plan-sha256",
            capture["sha256"],
            "--output",
            output,
            "--phase-id",
            inventory["v24_phase_id"],
            "--pre-dir",
            str(Path(inventory["pre_inputs"]["phase_lock"]["path"]).parents[1]),
            "--acquisition-started-ns",
            str(inventory["acquisition_started_ns"]),
            "--command-plan-sha256",
            materialized_inputs["orchestration_plan"]["sha256"],
            "--execute",
            "--confirm",
            config["confirm"],
        ]
        joint_bindings = {
            "adb": remote["adb"],
            "adb_server_port": ADB_PORT,
            "adb_server_process": identity["adb_server_process"],
            "capture_plan": artifact_only(capture),
            "cuda_launch_plan": artifact_only(
                materialized_inputs["cuda_route_launch"]
            ),
            "op12_selector": phones["op12"]["wifi_selector"],
            "op15_selector": phones["op15"]["wifi_selector"],
            "phone_launch_plan": artifact_only(
                materialized_inputs["phone_route_launch"]
            ),
        }
        components = {
            "python": remote["python"],
            "nvidia_smi": remote["nvidia_smi"],
            "adb": remote["adb"],
            "producer": producer,
            "joint_capture_plan": capture,
            "cuda_route_launch": materialized_inputs["cuda_route_launch"],
            "phone_route_launch": materialized_inputs["phone_route_launch"],
        }
    managed = build_managed_plan(
        bundle_id=f"v25_{role}_producer",
        producer_argv=argv,
        components=components,
        forward=inventory["local_forward_ports"][role],
        inventory=inventory,
        identity=identity,
    )
    managed_public = copy.deepcopy(managed)
    managed_public["ssh"].pop("_expected_boot_id")
    managed_raw = canonical_compact(managed_public)
    wrapper = {
        "frozen_producer": producer,
        "joint_bindings": joint_bindings,
        "local_python": identity["local_artifacts"]["python"],
        "managed_launcher": identity["local_artifacts"]["managed_launcher"],
        "managed_plan_sha256": hashlib.sha256(managed_raw).hexdigest(),
        "phase": PHASE,
        "phase_id": inventory["phase_id"],
        "producer_argv": argv,
        "remote_output_path": output,
        "role": role,
        "schema": "s39-v25-remote-cuda-wrapper-plan-v1",
        "sequence_index": config["sequence"],
        "timeout_seconds": 7200,
        "v24_phase_id": inventory["v24_phase_id"],
    }
    return managed_public, managed_raw, wrapper


def build_guard_plan(
    *,
    inventory: dict[str, Any],
    identity: dict[str, Any],
    discovery: dict[str, Any],
    local_policy: dict[str, Any],
    local_policy_artifact: dict[str, Any],
    remote_policy_artifact: dict[str, Any],
    guard: types.ModuleType,
) -> dict[str, Any]:
    local = identity["local_artifacts"]
    remote = identity["remote_artifacts"]
    phones = {}
    for phone in ("op12", "op15"):
        observed = discovery["phones"][phone]
        phones[phone] = {
            "boot_id": observed["boot_id"],
            "forbidden_listen_ports": inventory[
                "phone_forbidden_listen_ports"
            ][phone],
            "forbidden_processes": inventory["phone_forbidden_processes"][phone],
            "interface": observed["interface"],
            "physical_serial": observed["physical_serial"],
            "wifi_ipv4": observed["wifi_ipv4"],
            "wifi_selector": observed["wifi_selector"],
        }
    plan = {
        "adb_server_port": ADB_PORT,
        "adb_server_process": identity["adb_server_process"],
        "desktop_forbidden_listen_ports": inventory[
            "desktop_forbidden_listen_ports"
        ],
        "desktop_forbidden_processes": inventory["desktop_forbidden_processes"],
        "forbid_adb_forward_for_selectors": True,
        "gpu_uuid": GPU_UUID,
        "inner_phase_id": inventory["v24_phase_id"],
        "local_artifacts": {
            "adb": local["adb"],
            "helper": local["phone_guard"],
            "python": local["python"],
        },
        "local_policy_artifact": local_policy_artifact,
        "outer_phase_id": inventory["phase_id"],
        "phase": PHASE,
        "phones": phones,
        "remote_artifacts": {
            "adb": remote["adb"],
            "helper": remote["phone_guard"],
            "python": remote["python"],
        },
        "remote_policy": local_policy,
        "remote_policy_artifact": remote_policy_artifact,
        "rtx_boot_id": identity["rtx_boot_id"],
        "schema": guard.PLAN_SCHEMA,
        "ssh_transport": {
            "argv_by_moment": {"after": [], "before": []},
            "connect_timeout_seconds": inventory["ssh"]["connect_timeout_s"],
            "host_key_alias": "172.20.74.85",
            "identity_file": local["identity_file"],
            "known_hosts": local["known_hosts"],
            "remote_python": remote["python"],
            "ssh": local["ssh"],
            "ssh_port": 22,
            "ssh_target": SSH_TARGET,
        },
        "timeout_seconds": 600,
    }
    expected = guard.expected_remote_policy(plan)
    exact(local_policy, expected, "guard.remote_policy")
    for moment in ("before", "after"):
        plan["ssh_transport"]["argv_by_moment"][moment] = guard.expected_ssh_argv(
            plan,
            moment,
        )
    return guard.validate_plan(plan)


def load_fan_modules(
    common: types.ModuleType,
    local: dict[str, dict[str, Any]],
) -> tuple[types.ModuleType, types.ModuleType]:
    previous_common = sys.modules.get("v25_common")
    previous_contract = sys.modules.get("remote_fan_in_contract_v1")
    try:
        sys.modules["v25_common"] = common
        contract = load_source(
            "remote_fan_in_contract_v1",
            Path(local["remote_fan_in_contract"]["path"]),
        )
        sys.modules["remote_fan_in_contract_v1"] = contract
        executor = load_source(
            "remote_fan_in_execute_v1",
            Path(local["remote_fan_in_execute"]["path"]),
        )
    finally:
        if previous_common is None:
            sys.modules.pop("v25_common", None)
        else:
            sys.modules["v25_common"] = previous_common
        if previous_contract is None:
            sys.modules.pop("remote_fan_in_contract_v1", None)
        else:
            sys.modules["remote_fan_in_contract_v1"] = previous_contract
    return contract, executor


def build_fan_pair(
    *,
    inventory: dict[str, Any],
    identity: dict[str, Any],
    remote_inputs: dict[str, dict[str, Any]],
    managed_path: Path,
    fan_contract: types.ModuleType,
    fan_executor: types.ModuleType,
    managed_launcher: types.ModuleType,
) -> tuple[bytes, dict[str, Any]]:
    local = identity["local_artifacts"]
    remote = identity["remote_artifacts"]
    sources = {
        "authority": remote["fan_in_authority"],
        "fan_in": remote["fan_in_producer"],
        "production_common": remote["production_common"],
        "v24_common": remote["v24_common"],
        "v24_contract_builder": remote["v24_contract_builder"],
    }
    wrapper = {
        "acquisition_started_ns": inventory["acquisition_started_ns"],
        "capture_input_paths": {
            "cuda_monolithic": inventory["output_paths"]["cuda_monolithic"],
            "joint_phone_cuda": inventory["output_paths"]["joint_phone_cuda"],
        },
        "contract_validator": local["remote_fan_in_contract"],
        "executor": local["remote_fan_in_execute"],
        "input_artifacts": copy.deepcopy(remote_inputs),
        "local_common": local["v25_common"],
        "local_python": local["python"],
        "managed_launcher": local["managed_launcher"],
        "managed_plan": None,
        "managed_plan_sha256": "0" * 64,
        "outer_phase_id": inventory["phase_id"],
        "phase": PHASE,
        "producer_argv": [],
        "remote_acquisition_output": inventory["output_paths"][
            "fan_in_acquisition"
        ],
        "remote_bundle_root": inventory["output_paths"]["fan_in_bundle_root"],
        "remote_python": remote["python"],
        "remote_runtime_output": inventory["output_paths"]["fan_in_runtime"],
        "role": "remote_fan_in",
        "schema": "s39-v25-remote-fan-in-plan-v1",
        "source_artifacts": sources,
        "timeout_seconds": 7200,
        "v24_phase_id": inventory["v24_phase_id"],
    }
    wrapper["producer_argv"] = fan_contract.expected_producer_argv(wrapper)
    components = {
        "nvidia_smi": remote["nvidia_smi"],
        "python": remote["python"],
    }
    components.update({
        "source_" + role: artifact
        for role, artifact in sources.items()
    })
    components.update({
        "input_" + role.replace(".", "_"): artifact
        for role, artifact in remote_inputs.items()
    })
    managed = build_managed_plan(
        bundle_id="v25_remote_fan_in",
        producer_argv=wrapper["producer_argv"],
        components=components,
        forward=inventory["local_forward_ports"]["remote_fan_in"],
        inventory=inventory,
        identity=identity,
    )
    managed.pop("_normalized", None)
    managed["ssh"].pop("_expected_boot_id")
    managed_raw = canonical_compact(managed)
    write_raw_exclusive(managed_path, managed_raw)
    wrapper["managed_plan"] = artifact_from_path(managed_path, "fan.managed_plan")
    wrapper["managed_plan_sha256"] = hashlib.sha256(managed_raw).hexdigest()
    fan_contract.validate_plan(wrapper)
    fan_executor.validate_managed_plan(
        managed_launcher,
        managed_raw,
        wrapper["managed_plan_sha256"],
        wrapper,
    )
    return managed_raw, wrapper


def stage_descriptor(
    *,
    argv: list[str],
    cwd: str,
    entrypoint: dict[str, Any],
    expected_output: str,
    support: dict[str, dict[str, Any]],
    timeout_seconds: int,
) -> dict[str, Any]:
    return {
        "argv": argv,
        "cwd": cwd,
        "entrypoint": entrypoint,
        "environment": {
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        "expected_output": expected_output,
        "support": support,
        "timeout_seconds": timeout_seconds,
    }


def write_raw_exclusive(path: Path, raw: bytes) -> None:
    require(path.is_absolute() and not path.exists() and bool(raw), "E_OUTPUT")
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
            require(written > 0, "E_WRITE")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def materialize(
    *,
    inventory_path: Path,
    inventory_sha256: str,
    preparation_path: Path,
    discovery_path: Path,
    identity_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    started_ns = time.monotonic_ns()
    require(
        all(path.is_absolute() for path in (
            inventory_path,
            preparation_path,
            discovery_path,
            identity_path,
            output_root,
        )),
        "E_ABSOLUTE_INPUT",
    )
    require(not output_root.exists(), "E_OUTPUT_EXISTS")
    inventory, inventory_raw = read_canonical(inventory_path, "inventory")
    exact(
        hashlib.sha256(inventory_raw).hexdigest(),
        digest(inventory_sha256, "inventory_sha256"),
        "inventory.sha256",
    )

    common = load_source("v25_common_materializer", V25 / "v25_common.py")
    previous = sys.modules.get("v25_common")
    try:
        sys.modules["v25_common"] = common
        authority = load_source(
            "v25_authority_materializer",
            V25 / "cp0_r1_evidence_v25.py",
        )
        contract_path = Path(inventory["v24_inputs"]["contract_v25"]["path"])
        contract, _ = common.read_canonical(contract_path)
    finally:
        if previous is None:
            sys.modules.pop("v25_common", None)
        else:
            sys.modules["v25_common"] = previous
    validate_inventory(inventory, contract)

    preparation_value, preparation_raw = read_canonical(
        preparation_path,
        "preparation",
    )
    discovery_value, discovery_raw = read_canonical(discovery_path, "discovery")
    preparation = authority.validate_preparation(preparation_value, contract)
    discovery = authority.validate_discovery(
        discovery_value,
        contract,
        preparation,
        preparation_raw,
    )
    identity_value, identity_raw = read_canonical(identity_path, "identity")
    identity = validate_identity(
        identity_value,
        identity_raw,
        discovery_value,
        discovery_raw,
        contract,
    )
    exact(
        identity["local_artifacts"]["python"],
        preparation["local_python"],
        "identity.local.python.preparation",
    )
    require(identity["completed_ns"] <= started_ns, "E_MATERIALIZATION_ORDER")
    exact(inventory["phase_id"], discovery["phase_id"], "inventory.phase_id")
    exact(identity["outer_phase_id"], inventory["phase_id"], "identity.phase_id")

    input_records: dict[str, dict[str, Any]] = {}
    input_raws: dict[str, bytes] = {}
    input_values: dict[str, dict[str, Any]] = {}
    for name, expected_schema in sorted(V25_INPUT_SCHEMAS.items()):
        source = validate_artifact(
            inventory["v24_inputs"][name],
            f"inventory.v24_inputs.{name}",
        )
        path = Path(source["path"])
        value, raw = read_canonical(path, f"v24.{name}")
        exact(value.get("schema"), expected_schema, f"v24.{name}.schema")
        exact(len(raw), source["bytes"], f"v24.{name}.bytes")
        exact(hashlib.sha256(raw).hexdigest(), source["sha256"], f"v24.{name}.sha256")
        exact(
            artifact_from_path(path, f"v24.{name}.identity"),
            source,
            f"v24.{name}.identity",
        )
        input_values[name] = value
        input_raws[name] = raw
        input_records[name] = source
    remote_inputs = validate_remote_input_bindings(
        identity,
        input_records,
        inventory["pre_inputs"],
    )
    for name, artifact in sorted(inventory["pre_inputs"].items()):
        verify_local_artifact(artifact, f"pre.{name}")
    history = load_source("v25_history_materializer", V25 / "remote_history_validate_v1.py")
    capture = load_source("v25_capture_materializer", V25 / "remote_cuda_capture_v1.py")
    guard = load_source("v25_guard_materializer", V25 / "remote_phone_guard_v1.py")
    managed_launcher = load_source(
        "v25_managed_launcher_materializer",
        V23
        / "a_only_acquisition_driver_v1"
        / "producers_v1"
        / "managed_runtime_launcher_v1.py",
    )
    fan_contract, fan_executor = load_fan_modules(
        common,
        identity["local_artifacts"],
    )

    history_plan = build_history_plan(contract, inventory, identity)
    history.validate_plan(history_plan)

    materialized_inputs = {
        name: {
            **input_records[name],
            "value": input_values[name],
        }
        for name in input_records
        if name != "contract_v25"
    }
    pairs = {}
    for role in ("cuda_monolithic", "joint_phone_cuda"):
        managed, managed_raw, wrapper = build_cuda_pair(
            role=role,
            contract=contract,
            inventory=inventory,
            identity=identity,
            materialized_inputs=materialized_inputs,
            phones=discovery["phones"],
        )
        parsed = managed_launcher.parse_plan_json(
            managed_raw.decode("ascii"),
            hashlib.sha256(managed_raw).hexdigest(),
        )
        capture.validate_plan(wrapper)
        capture.validate_managed_plan(
            managed_launcher,
            managed_raw,
            hashlib.sha256(managed_raw).hexdigest(),
            wrapper,
        )
        exact(
            {
                key: value
                for key, value in parsed.items()
                if key != "_normalized"
            },
            managed,
            f"managed.{role}",
        )
        pairs[role] = (managed_raw, wrapper)

    local_policy = {
        "adb_server_port": ADB_PORT,
        "adb_server_process": identity["adb_server_process"],
        "desktop_forbidden_listen_ports": inventory[
            "desktop_forbidden_listen_ports"
        ],
        "desktop_forbidden_processes": inventory["desktop_forbidden_processes"],
        "forbid_adb_forward_for_selectors": True,
        "gpu_uuid": GPU_UUID,
        "inner_phase_id": inventory["v24_phase_id"],
        "outer_phase_id": inventory["phase_id"],
        "phase": PHASE,
        "phones": {
            phone: {
                "boot_id": discovery["phones"][phone]["boot_id"],
                "forbidden_listen_ports": inventory[
                    "phone_forbidden_listen_ports"
                ][phone],
                "forbidden_processes": inventory[
                    "phone_forbidden_processes"
                ][phone],
                "interface": discovery["phones"][phone]["interface"],
                "physical_serial": discovery["phones"][phone]["physical_serial"],
                "wifi_ipv4": discovery["phones"][phone]["wifi_ipv4"],
                "wifi_selector": discovery["phones"][phone]["wifi_selector"],
            }
            for phone in ("op12", "op15")
        },
        "remote_artifacts": {
            "adb": identity["remote_artifacts"]["adb"],
            "helper": identity["remote_artifacts"]["phone_guard"],
            "python": identity["remote_artifacts"]["python"],
        },
        "rtx_boot_id": identity["rtx_boot_id"],
        "schema": "s39-v25-remote-phone-guard-policy-v1",
    }
    guard.validate_remote_policy(local_policy)

    output_root.mkdir(parents=True, exist_ok=False)
    try:
        prewritten_paths: set[Path] = set()
        files: dict[str, tuple[Path, bytes]] = {}
        files["phase.preparation"] = (
            output_root / "phase-preparation.json",
            preparation_raw,
        )
        files["phase.discovery"] = (
            output_root / "phase-discovery.json",
            discovery_raw,
        )
        for name in (
            "artifact_root",
            "bound_root",
            "cuda_route_launch",
            "fresh_readiness",
            "identity_binding_attestation",
            "identity_binding_receipt",
            "identity_binding_stage_receipt",
            "joint_capture_plan",
            "orchestration_plan",
            "phase_lock",
            "phone_route_launch",
            "preparation",
            "prospective_root",
            "runtime_plan",
        ):
            files[f"inner.{name}"] = (
                output_root / f"inner-{name.replace('_', '-')}.json",
                input_raws[name],
            )
        files["history.remote_plan"] = (
            output_root / "history-remote-plan.json",
            canonical_compact(history_plan),
        )
        for role, (managed_raw, wrapper) in pairs.items():
            files[f"plan.managed.{role}"] = (
                output_root / f"managed-{role}.json",
                managed_raw,
            )
            files[f"plan.wrapper.{role}"] = (
                output_root / f"wrapper-{role}.json",
                canonical_bytes(wrapper),
            )
        policy_path = output_root / "remote-phone-policy.json"
        policy_raw = canonical_bytes(local_policy)
        write_raw_exclusive(policy_path, policy_raw)
        prewritten_paths.add(policy_path)
        files["plan.remote_phone_guard_policy"] = (
            policy_path,
            policy_raw,
        )
        local_policy_artifact = artifact_from_path(policy_path, "local_policy")
        remote_policy_artifact = identity["remote_artifacts"].get(
            "phone_guard_policy"
        )
        require(
            remote_policy_artifact is not None,
            "E_REMOTE_POLICY_IDENTITY_MISSING",
        )
        exact(
            (
                remote_policy_artifact["bytes"],
                remote_policy_artifact["sha256"],
            ),
            (len(policy_raw), hashlib.sha256(policy_raw).hexdigest()),
            "remote_policy.content",
        )
        guard_plan = build_guard_plan(
            inventory=inventory,
            identity=identity,
            discovery=discovery,
            local_policy=local_policy,
            local_policy_artifact=local_policy_artifact,
            remote_policy_artifact=remote_policy_artifact,
            guard=guard,
        )
        files["plan.remote_phone_guard"] = (
            output_root / "remote-phone-guard-plan.json",
            canonical_bytes(guard_plan),
        )
        fan_managed_path = output_root / "managed-remote_fan_in.json"
        fan_managed_raw, fan_wrapper = build_fan_pair(
            inventory=inventory,
            identity=identity,
            remote_inputs=remote_inputs,
            managed_path=fan_managed_path,
            fan_contract=fan_contract,
            fan_executor=fan_executor,
            managed_launcher=managed_launcher,
        )
        prewritten_paths.add(fan_managed_path)
        files["plan.managed.remote_fan_in"] = (
            fan_managed_path,
            fan_managed_raw,
        )
        files["plan.remote_fan_in"] = (
            output_root / "remote-fan-in-plan.json",
            canonical_bytes(fan_wrapper),
        )
        for role, (path, raw) in sorted(files.items()):
            if path in prewritten_paths:
                exact(read_regular(path, role), raw, f"{role}.prewritten")
            else:
                write_raw_exclusive(path, raw)

        stage_outputs = {
            "history.remote_receipt": str(
                output_root / "history-remote-receipt.json"
            ),
            "runtime.remote_phone_guard.before": str(
                output_root / "phone-guard-before.json"
            ),
            "capture.cuda_monolithic.wrapper": str(
                output_root / "cuda-monolithic-receipt.json"
            ),
            "capture.cuda_monolithic.result": str(
                output_root / "cuda-monolithic.json"
            ),
            "capture.joint_phone_cuda.wrapper": str(
                output_root / "joint-phone-cuda-receipt.json"
            ),
            "capture.joint_phone_cuda.result": str(
                output_root / "joint-phone-cuda.json"
            ),
            "runtime.remote_phone_guard.after": str(
                output_root / "phone-guard-after.json"
            ),
            "capture.remote_fan_in.wrapper": str(
                output_root / "remote-fan-in-receipt.json"
            ),
        }
        stages = {
            "remote_history": stage_descriptor(
                argv=[
                    identity["local_artifacts"]["python"]["path"],
                    identity["local_artifacts"]["remote_history"]["path"],
                    "--plan-json",
                    canonical_compact(history_plan).decode("ascii"),
                    "--plan-sha256",
                    hashlib.sha256(canonical_compact(history_plan)).hexdigest(),
                    "--boot-id",
                    identity["rtx_boot_id"],
                    "--output",
                    stage_outputs["history.remote_receipt"],
                ],
                cwd=str(S39.parents[2]),
                entrypoint=identity["local_artifacts"]["remote_history"],
                expected_output=stage_outputs["history.remote_receipt"],
                support={},
                timeout_seconds=720,
            )
        }
        guard_plan_path = files["plan.remote_phone_guard"][0]
        guard_plan_raw = files["plan.remote_phone_guard"][1]
        for moment in ("before", "after"):
            role = f"runtime.remote_phone_guard.{moment}"
            stages[f"phone_guard_{moment}"] = stage_descriptor(
                argv=[
                    identity["local_artifacts"]["python"]["path"],
                    identity["local_artifacts"]["phone_guard"]["path"],
                    "--plan",
                    str(guard_plan_path),
                    "--plan-sha256",
                    hashlib.sha256(guard_plan_raw).hexdigest(),
                    "--moment",
                    moment,
                    "--receipt",
                    stage_outputs[role],
                    "--execute",
                    "--confirm",
                    "RUN_CP0_R1_V25_PHONE_GUARD",
                ],
                cwd=str(S39.parents[2]),
                entrypoint=identity["local_artifacts"]["phone_guard"],
                expected_output=stage_outputs[role],
                support={},
                timeout_seconds=600,
            )
        for role in ("cuda_monolithic", "joint_phone_cuda"):
            managed_path = files[f"plan.managed.{role}"][0]
            managed_raw = files[f"plan.managed.{role}"][1]
            wrapper_path = files[f"plan.wrapper.{role}"][0]
            wrapper_raw = files[f"plan.wrapper.{role}"][1]
            stages[role] = stage_descriptor(
                argv=[
                    identity["local_artifacts"]["python"]["path"],
                    identity["local_artifacts"]["remote_cuda_capture"]["path"],
                    "--plan",
                    str(wrapper_path),
                    "--plan-sha256",
                    hashlib.sha256(wrapper_raw).hexdigest(),
                    "--managed-plan",
                    str(managed_path),
                    "--managed-plan-sha256",
                    hashlib.sha256(managed_raw).hexdigest(),
                    "--remote-boot-id",
                    identity["rtx_boot_id"],
                    "--output",
                    stage_outputs[f"capture.{role}.result"],
                    "--receipt",
                    stage_outputs[f"capture.{role}.wrapper"],
                    "--execute",
                    "--confirm",
                    "RUN-S39-V25-REMOTE-CUDA",
                ],
                cwd=str(S39.parents[2]),
                entrypoint=identity["local_artifacts"]["remote_cuda_capture"],
                expected_output=stage_outputs[f"capture.{role}.wrapper"],
                support={
                    "managed_launcher": identity["local_artifacts"][
                        "managed_launcher"
                    ]
                },
                timeout_seconds=7200,
            )
        fan_wrapper_path = files["plan.remote_fan_in"][0]
        fan_wrapper_raw = files["plan.remote_fan_in"][1]
        stages["remote_fan_in"] = stage_descriptor(
            argv=[
                identity["local_artifacts"]["python"]["path"],
                identity["local_artifacts"]["remote_fan_in_execute"]["path"],
                "--plan",
                str(fan_wrapper_path),
                "--plan-sha256",
                hashlib.sha256(fan_wrapper_raw).hexdigest(),
                "--boot-id",
                identity["rtx_boot_id"],
                "--bundle-root",
                str(output_root / "remote-fan-in-bundle"),
                "--receipt",
                stage_outputs["capture.remote_fan_in.wrapper"],
                "--execute",
                "--confirm",
                "RUN_CP0_R1_V25_REMOTE_FAN_IN",
            ],
            cwd=str(S39.parents[2]),
            entrypoint=identity["local_artifacts"]["remote_fan_in_execute"],
            expected_output=stage_outputs["capture.remote_fan_in.wrapper"],
            support={
                "contract": identity["local_artifacts"][
                    "remote_fan_in_contract"
                ],
                "managed_launcher": identity["local_artifacts"][
                    "managed_launcher"
                ],
                "v25_common": identity["local_artifacts"]["v25_common"],
            },
            timeout_seconds=7200,
        )

        artifact_rows = []
        for role, (path, raw) in sorted(files.items()):
            artifact_rows.append(
                {
                    "bytes": len(raw),
                    "path": path.name,
                    "role": role,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        completed_ns = time.monotonic_ns()
        require(started_ns < completed_ns, "E_MATERIALIZATION_INTERVAL")
        root = {
            "artifacts": artifact_rows,
            "completed_ns": completed_ns,
            "discovery_sha256": hashlib.sha256(discovery_raw).hexdigest(),
            "fan_in_materialized": True,
            "fresh_identity_sha256": hashlib.sha256(identity_raw).hexdigest(),
            "inventory_sha256": hashlib.sha256(inventory_raw).hexdigest(),
            "outer_phase_id": inventory["phase_id"],
            "phase": PHASE,
            "schema": ROOT_SCHEMA,
            "started_ns": started_ns,
            "stages": stages,
            "status": "A_ONLY_PLANS_MATERIALIZED_NO_HARDWARE_RUN",
            "v24_phase_id": inventory["v24_phase_id"],
        }
        root_raw = canonical_bytes(root)
        write_raw_exclusive(output_root / "MATERIALIZATION.json", root_raw)
        directory = os.open(output_root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return root
    except Exception:
        if output_root.exists():
            for path in sorted(output_root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            output_root.rmdir()
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--fresh-identity", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        root = materialize(
            inventory_path=args.inventory.resolve(),
            inventory_sha256=args.inventory_sha256,
            preparation_path=args.preparation.resolve(),
            discovery_path=args.discovery.resolve(),
            identity_path=args.fresh_identity.resolve(),
            output_root=args.output_root.resolve(),
        )
    except (MaterializeError, OSError, RuntimeError, ValueError) as error:
        print(f"V25_MATERIALIZATION_REFUSED: {error}", file=sys.stderr)
        return 2
    print(canonical_bytes(root).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
