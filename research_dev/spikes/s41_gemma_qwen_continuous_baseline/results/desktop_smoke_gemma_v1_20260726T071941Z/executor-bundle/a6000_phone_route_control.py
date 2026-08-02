#!/usr/bin/env python3
"""A6000-side, ADB-only lifecycle control for versioned phone routes."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from phone_gateway import (
    MAX_COMMAND_BYTES,
    GatewayError,
    canonical_bytes,
    command_argv,
    exact_keys,
    integer,
    require,
    sha256_text,
    strict_json_loads,
    string,
)
from readiness_v23 import (
    absolute_path,
    parse_artifact_certificate,
    parse_readiness_lock,
    stat_time_ns,
)

ADB_ENV = {
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/system/bin:/system/xbin",
}
COMMON_PHONE_RUNTIME_ROLES = {
    "libcxx_shared",
    "libggml",
    "libggml_base",
    "libggml_cpu",
    "libggml_hexagon",
    "libggml_opencl",
    "libllama",
    "libllama_common",
    "worker",
}
PHONE_RUNTIME_FILENAMES = {
    "libcxx_shared": "libc++_shared.so",
    "libggml": "libggml.so",
    "libggml_base": "libggml-base.so",
    "libggml_cpu": "libggml-cpu.so",
    "libggml_hexagon": "libggml-hexagon.so",
    "libggml_opencl": "libggml-opencl.so",
    "libllama": "libllama.so",
    "libllama_common": "libllama-common.so",
    "relay": "llama-stage-direct-relay",
    "worker": "llama-layersplit",
}
SESSION_PREFIX = "SESSIONCERT "
PLACEMENT_PREFIX = "PLACEMENTCERT "


def local_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_count_map(value: Any, field: str) -> dict[str, int]:
    require(type(value) is dict and value, f"{field} map")
    result = {}
    for key, count in value.items():
        name = string(key, f"{field} key")
        require(name not in result, f"{field} duplicate key")
        result[name] = integer(count, f"{field}.{name}", 1)
    return result


def _placement_map(
    value: Any,
    expected_backend: str,
    field: str,
) -> dict[str, dict[str, int]]:
    require(type(value) is dict and value, f"{field} map")
    result = {}
    for op, buffers in value.items():
        op_name = string(op, f"{field} operation")
        require(type(buffers) is dict and buffers, f"{field}.{op_name}")
        normalized = _positive_count_map(buffers, f"{field}.{op_name}")
        if op_name == "GET_ROWS":
            require(
                set(normalized).issubset({expected_backend, "CPU"}),
                f"{field}.{op_name} backend placement",
            )
        else:
            require(
                set(normalized) == {expected_backend},
                f"{field}.{op_name} backend placement",
            )
        result[op_name] = normalized
    return result


def parse_stage_certificates(
    raw: str,
    process: dict[str, Any],
    boot_id: str,
    n_layer: int,
) -> dict[str, Any]:
    require(
        0 < len(raw.encode("ascii")) <= MAX_COMMAND_BYTES,
        "phone stage log size",
    )
    sessions = []
    placements = []
    certificate_lines = []
    for line in raw.splitlines():
        if line.startswith(SESSION_PREFIX):
            certificate_lines.append(line)
            sessions.append(strict_json_loads(
                line[len(SESSION_PREFIX):].encode("ascii"),
                "phone SESSIONCERT",
            ))
        elif line.startswith(PLACEMENT_PREFIX):
            certificate_lines.append(line)
            placements.append(strict_json_loads(
                line[len(PLACEMENT_PREFIX):].encode("ascii"),
                "phone PLACEMENTCERT",
            ))
    require(
        len(sessions) == 1 and len(placements) == 1,
        "phone stage certificate cardinality",
    )
    require(
        certificate_lines[0].startswith(SESSION_PREFIX)
        and certificate_lines[1].startswith(PLACEMENT_PREFIX),
        "phone stage certificate order",
    )
    session = exact_keys(
        sessions[0],
        {
            "compute_by_op_and_buffer",
            "device_boot_id",
            "expected_backend",
            "layer_end",
            "layer_start",
            "missing_buffer_compute_nodes",
            "n_layer",
            "placement_status",
            "proto_version",
            "reset_applied",
            "schema",
            "session_end",
            "session_id",
            "steps_session",
            "steps_total",
            "worker_boot_nonce",
            "worker_pid",
        },
        "phone SESSIONCERT",
    )
    placement = exact_keys(
        placements[0],
        {
            "compute_by_buffer_type",
            "compute_by_op",
            "compute_by_op_and_buffer",
            "compute_nodes",
            "copy_by_buffer_type",
            "copy_nodes",
            "layer_end",
            "layer_start",
            "metadata_nodes",
            "missing_buffer_compute_nodes",
            "mode",
            "n_layer",
            "pid",
            "role",
            "run_rc",
            "schema",
            "status",
        },
        "phone PLACEMENTCERT",
    )
    expected_backend = string(process["backend"], "stage backend")
    compute = _placement_map(
        session["compute_by_op_and_buffer"],
        expected_backend,
        "phone SESSIONCERT compute",
    )
    placement_compute = _placement_map(
        placement["compute_by_op_and_buffer"],
        expected_backend,
        "phone PLACEMENTCERT compute",
    )
    require(compute == placement_compute, "phone placement tally changed")
    compute_by_op = _positive_count_map(
        placement["compute_by_op"],
        "phone PLACEMENTCERT operations",
    )
    compute_by_buffer = _positive_count_map(
        placement["compute_by_buffer_type"],
        "phone PLACEMENTCERT buffers",
    )
    derived_by_op = {
        op: sum(buffers.values())
        for op, buffers in placement_compute.items()
    }
    derived_by_buffer: dict[str, int] = {}
    for buffers in placement_compute.values():
        for backend, count in buffers.items():
            derived_by_buffer[backend] = (
                derived_by_buffer.get(backend, 0) + count
            )
    require(
        compute_by_op == derived_by_op
        and compute_by_buffer == derived_by_buffer,
        "phone placement aggregate tally",
    )
    copy_nodes = integer(
        placement["copy_nodes"],
        "placement copy nodes",
    )
    copy_by_buffer = placement["copy_by_buffer_type"]
    require(type(copy_by_buffer) is dict, "placement copy buffers")
    if copy_nodes == 0:
        require(not copy_by_buffer, "placement zero-copy tally")
    else:
        copy_by_buffer = _positive_count_map(
            copy_by_buffer,
            "phone PLACEMENTCERT copy buffers",
        )
        require(
            sum(copy_by_buffer.values()) == copy_nodes,
            "phone placement copy tally",
        )
    require(
        session["schema"] == "ls-stagenet-session-v2"
        and integer(session["proto_version"], "session protocol", 1) == 2
        and integer(session["session_id"], "session ID", 1) == 1
        and session["session_end"] in ("EOF", "STOP")
        and session["expected_backend"] == expected_backend
        and session["device_boot_id"] == boot_id
        and integer(session["worker_pid"], "session worker PID", 1)
        == process["pid"]
        and string(session["worker_boot_nonce"], "worker boot nonce")
        and session["layer_start"] == process["layer_start"]
        and session["layer_end"] == process["layer_end"]
        and session["n_layer"] == n_layer
        and integer(session["steps_session"], "session steps", 1)
        == integer(session["steps_total"], "total steps", 1)
        and session["reset_applied"] is False
        and integer(
            session["missing_buffer_compute_nodes"],
            "session missing buffers",
        ) == 0
        and session["placement_status"] == "SCHEDULED_PLACEMENT_OK",
        "phone session placement certificate",
    )
    require(
        placement["schema"] == "layersplit-scheduled-placement-v2"
        and placement["role"] == "phone_stage"
        and placement["mode"]
        == ("stagenet" if process["kind"] == "STAGE_HEAD" else "tailv3")
        and placement["layer_start"] == process["layer_start"]
        and placement["layer_end"] == process["layer_end"]
        and placement["n_layer"] == n_layer
        and integer(placement["pid"], "placement PID", 1) == process["pid"]
        and integer(placement["run_rc"], "placement return code") == 0
        and integer(placement["compute_nodes"], "placement compute nodes", 1)
        == sum(compute_by_op.values())
        == sum(compute_by_buffer.values())
        and copy_nodes >= 0
        and integer(placement["metadata_nodes"], "placement metadata nodes") >= 0
        and integer(
            placement["missing_buffer_compute_nodes"],
            "placement missing buffers",
        ) == 0
        and placement["status"] == "SCHEDULED_PLACEMENT_OK",
        "phone terminal placement certificate",
    )
    return {
        "certificate_lines": certificate_lines,
        "certificate_lines_sha256": hashlib.sha256(
            "".join(line + "\n" for line in certificate_lines).encode("ascii")
        ).hexdigest(),
        "placement": placement,
        "session": session,
    }


def locked_file_identity(
    path: Path,
    role: str,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    require(path.is_absolute() and path.is_file(), f"{role} dependency file")
    require(not path.is_symlink(), f"{role} dependency symlink")
    size = path.stat().st_size
    digest = local_sha256(path)
    if expected_bytes is not None:
        require(size == expected_bytes, f"{role} dependency size changed")
    if expected_sha256 is not None:
        require(digest == expected_sha256, f"{role} dependency changed")
    return {
        "bytes": size,
        "path": str(path),
        "role": role,
        "sha256": digest,
    }


def canonical_ipv4(value: Any, field: str) -> str:
    address = string(value, field)
    try:
        normalized = str(ipaddress.IPv4Address(address))
    except ipaddress.AddressValueError as error:
        raise GatewayError(f"{field}: {error}") from error
    require(address == normalized, f"{field} IPv4")
    return address


def dependency_package_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def validate_dependency_package(config: dict[str, Any]) -> dict[str, Any]:
    package = config["dependency_package"]
    expected = config["dependency_package_sha256"]
    require(
        type(package) is dict
        and type(expected) is str
        and dependency_package_sha256(package) == expected,
        "A6000 dependency package identity",
    )
    current_boot_id = Path(
        "/proc/sys/kernel/random/boot_id"
    ).read_text(encoding="ascii").strip()
    require(
        current_boot_id == package["host_boot_id"],
        "A6000 host boot changed",
    )
    for row in package["files"]:
        locked_file_identity(
            Path(row["path"]),
            row["role"],
            row["bytes"],
            row["sha256"],
        )
    return package


def argv_sha256(argv: list[str]) -> str:
    payload = b"".join(
        argument.encode("utf-8") + b"\0"
        for argument in argv
    )
    return hashlib.sha256(payload).hexdigest()


def safe_remote_path(value: Any, field: str) -> str:
    return absolute_path(value, field)


def stat_identity(value: Any, field: str) -> dict[str, int]:
    value = exact_keys(
        value,
        {"ctime_ns", "device_id", "inode", "mode", "mtime_ns", "size"},
        field,
    )
    result = {
        key: integer(value[key], f"{field}.{key}")
        for key in value
    }
    require(
        result["device_id"] >= 0
        and result["inode"] > 0
        and result["size"] > 0
        and result["mode"] > 0,
        f"{field} values",
    )
    return result


def exact_stage_process(
    process: Any,
    phone_name: str,
    worker_path: str,
    shard_path: str,
    model_sha256: str,
    runtime_root: str,
) -> dict[str, Any]:
    process = exact_keys(
        process,
        {
            "argv",
            "backend",
            "driver_batch",
            "driver_context",
            "driver_max_prefill",
            "env",
            "kind",
            "layer_end",
            "layer_start",
            "listen_port",
            "name",
            "ready_marker",
        },
        f"{phone_name} stage process",
    )
    kind = string(process["kind"], f"{phone_name} process kind")
    require(
        kind in ("STAGE_HEAD", "STAGE_TAIL"),
        f"{phone_name} process kind",
    )
    layer_start = integer(
        process["layer_start"],
        f"{phone_name} process layer start",
    )
    layer_end = integer(
        process["layer_end"],
        f"{phone_name} process layer end",
        1,
    )
    require(layer_start < layer_end, f"{phone_name} process layer range")
    backend = string(process["backend"], f"{phone_name} process backend")
    require(
        backend in ("GPUOpenCL", "HTP0"),
        f"{phone_name} process backend",
    )
    batch = integer(
        process["driver_batch"],
        f"{phone_name} process driver batch",
        1,
    )
    context = integer(
        process["driver_context"],
        f"{phone_name} process driver context",
        1,
    )
    prefill = integer(
        process["driver_max_prefill"],
        f"{phone_name} process max prefill",
        1,
    )
    require(
        batch == 8 and prefill <= batch and context >= batch,
        f"{phone_name} process serving geometry",
    )
    listen_port = integer(
        process["listen_port"],
        f"{phone_name} process listen port",
        1,
    )
    require(listen_port <= 65535, f"{phone_name} process listen port")
    mode = "stagenet" if kind == "STAGE_HEAD" else "tailv3"
    expected_argv = (
        worker_path,
        "-m",
        shard_path,
        "--mode",
        mode,
        "--port",
        str(listen_port),
        "--driver-batch",
        str(batch),
        "--driver-context",
        str(context),
        "--driver-max-prefill",
        str(prefill),
        "--devices",
        backend,
        "-ngl",
        "99",
    )
    argv = command_argv(process["argv"], f"{phone_name} process argv")
    require(argv == expected_argv, f"{phone_name} stage argv")
    env = process["env"]
    expected_env = {
        "ADSP_LIBRARY_PATH": runtime_root,
        "LAYERSPLIT_MODEL_SHA256": model_sha256,
        "LAYERSPLIT_PLACEMENT_CERT": "1",
        "LD_LIBRARY_PATH": runtime_root,
        "LLAMA_LAYER_END": str(layer_end),
        "LLAMA_LAYER_START": str(layer_start),
        "PATH": "/system/bin:/system/xbin",
    }
    require(env == expected_env, f"{phone_name} stage environment")
    require(
        process["name"] == "stage"
        and process["ready_marker"]
        == f"[stagenet] listening on 0.0.0.0:{listen_port} (",
        f"{phone_name} stage readiness identity",
    )
    return {
        "argv": argv,
        "artifact_path": worker_path,
        "artifact_role": "worker",
        "backend": backend,
        "env": expected_env,
        "kind": kind,
        "layer_end": layer_end,
        "layer_start": layer_start,
        "listen_port": listen_port,
        "name": "stage",
        "ready_marker":
            f"[stagenet] listening on 0.0.0.0:{listen_port} (",
    }


def exact_relay_process(
    process: Any,
    phone_name: str,
    relay_path: str,
    runtime_root: str,
    expected_peer: dict[str, Any],
) -> dict[str, Any]:
    process = exact_keys(
        process,
        {
            "argv",
            "env",
            "head_host",
            "head_port",
            "kind",
            "listen_port",
            "name",
            "ready_marker",
            "tail_host",
            "tail_port",
            "tail_source_port",
        },
        f"{phone_name} relay process",
    )
    require(
        phone_name == "op15" and process["kind"] == "DIRECT_RELAY",
        f"{phone_name} relay kind",
    )
    listen_port = integer(
        process["listen_port"],
        f"{phone_name} relay listen port",
        1,
    )
    head_port = integer(
        process["head_port"],
        f"{phone_name} relay head port",
        1,
    )
    tail_port = integer(
        process["tail_port"],
        f"{phone_name} relay tail port",
        1,
    )
    source_port = integer(
        process["tail_source_port"],
        f"{phone_name} relay source port",
        1,
    )
    require(
        max(listen_port, head_port, tail_port, source_port) <= 65535,
        f"{phone_name} relay ports",
    )
    head_host = string(process["head_host"], f"{phone_name} relay head")
    tail_host = canonical_ipv4(
        process["tail_host"],
        f"{phone_name} relay tail",
    )
    require(
        head_host == "127.0.0.1"
        and tail_host == expected_peer["op12_address"]
        and tail_port == expected_peer["op12_port"]
        and source_port == expected_peer["op15_port"],
        f"{phone_name} relay endpoint binding",
    )
    expected_argv = (
        relay_path,
        "--listen",
        str(listen_port),
        "--head",
        f"{head_host}:{head_port}",
        "--tail",
        f"{tail_host}:{tail_port}",
        "--tail-source-port",
        str(source_port),
    )
    argv = command_argv(process["argv"], f"{phone_name} relay argv")
    require(argv == expected_argv, f"{phone_name} relay argv")
    expected_env = {
        "LD_LIBRARY_PATH": runtime_root,
        "PATH": "/system/bin:/system/xbin",
    }
    require(process["env"] == expected_env, f"{phone_name} relay environment")
    expected_marker = (
        f"[direct-relay] listening on 0.0.0.0:{listen_port} "
        f"head={head_host}:{head_port} tail={tail_host}:{tail_port}"
    )
    require(
        process["name"] == "relay"
        and process["ready_marker"] == expected_marker,
        f"{phone_name} relay readiness identity",
    )
    return {
        "argv": argv,
        "artifact_path": relay_path,
        "artifact_role": "relay",
        "backend": "NETWORK",
        "env": expected_env,
        "head_host": head_host,
        "head_port": head_port,
        "kind": "DIRECT_RELAY",
        "layer_end": None,
        "layer_start": None,
        "listen_port": listen_port,
        "name": "relay",
        "ready_marker": expected_marker,
        "tail_host": tail_host,
        "tail_port": tail_port,
        "tail_source_port": source_port,
    }


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_durable_new(path: Path, value: dict[str, Any]) -> None:
    raw = canonical_bytes(value)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp"
    )
    try:
        with temporary.open("xb", buffering=0) as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def replace_durable(path: Path, value: dict[str, Any]) -> None:
    raw = canonical_bytes(value)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp"
    )
    try:
        with temporary.open("xb", buffering=0) as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def remove_durable(path: Path) -> None:
    path.unlink()
    fsync_directory(path.parent)


def qualification_binding(
    value: Any,
    *,
    model_id: str,
    phase: str,
    slot: str,
    artifact_certificate_sha256: str,
    readiness_lock_sha256: str,
) -> str:
    value = exact_keys(
        value,
        {"a_chain", "current", "model_id", "phase", "schema", "slot"},
        "A6000 route qualification",
    )
    require(
        value["schema"] == "s40-route-qualification-authority-v1"
        and value["model_id"] == model_id
        and value["phase"] == phase
        and value["slot"] == slot,
        "A6000 route qualification identity",
    )

    def root(item: Any, field: str) -> dict[str, Any]:
        item = exact_keys(
            item,
            {
                "artifact_snapshot",
                "bundle_manifest_sha256",
                "bundle_root",
                "fresh_snapshot",
                "readiness_lock",
                "runtime_identity",
            },
            field,
        )
        string(item["bundle_root"], f"{field} bundle root")
        sha256_text(
            item["bundle_manifest_sha256"],
            f"{field} bundle manifest SHA-256",
        )
        for name in (
            "artifact_snapshot",
            "fresh_snapshot",
            "readiness_lock",
            "runtime_identity",
        ):
            record = exact_keys(
                item[name],
                {"path", "sha256"},
                f"{field} {name}",
            )
            string(record["path"], f"{field} {name} path")
            sha256_text(
                record["sha256"],
                f"{field} {name} SHA-256",
            )
        return item

    current = root(value["current"], "A6000 current qualification")
    require(
        current["artifact_snapshot"]["sha256"]
        == artifact_certificate_sha256
        and current["readiness_lock"]["sha256"]
        == readiness_lock_sha256,
        "A6000 qualification readiness roots",
    )
    if phase == "A_ONLY":
        require(value["a_chain"] is None, "A6000 A_ONLY qualification chain")
    else:
        require(phase == "B_ONLY", "A6000 qualification phase")
        root(value["a_chain"], "A6000 A qualification chain")
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    require(0 < len(raw) <= MAX_COMMAND_BYTES, "A6000 config size")
    value = strict_json_loads(raw, "A6000 config JSON")
    require(canonical_bytes(value) == raw, "A6000 config is not canonical JSON")
    schema = value.get("schema") if type(value) is dict else None
    base_keys = {
        "a6000_identity",
        "a6000_identity_path",
        "adb_path",
        "adb_port",
        "routes",
        "schema",
        "state_dir",
    }
    require(
        schema == "s40-a6000-phone-routes-v3",
        "A6000 config schema",
    )
    value = exact_keys(
        value,
        base_keys | {"dependency_files", "host_boot_id"},
        "A6000 config",
    )
    identity_path = Path(
        safe_remote_path(value["a6000_identity_path"], "A6000 identity path")
    )
    require(identity_path.is_file(), "A6000 identity file")
    identity = sha256_text(value["a6000_identity"], "A6000 identity")
    require(local_sha256(identity_path) == identity, "A6000 identity changed")
    adb_path = Path(safe_remote_path(value["adb_path"], "ADB path"))
    require(adb_path.is_file(), "ADB executable missing")
    adb_port = integer(value["adb_port"], "ADB port", 1)
    require(adb_port == 5038, "ADB port must be 5038")
    state_dir = Path(safe_remote_path(value["state_dir"], "state directory"))
    routes = value["routes"]
    require(type(routes) is dict and 1 <= len(routes) <= 8, "A6000 routes")
    normalized = {}
    for model_id, route in routes.items():
        string(model_id, "A6000 model id")
        route_keys = {
            "artifact_certificate_path",
            "artifact_certificate_sha256",
            "model_sha256",
            "op12",
            "op15",
            "phase",
            "phase_lock_sha256",
            "qualification",
            "readiness_lock_path",
            "readiness_lock_sha256",
            "route_lock_sha256",
            "slot",
            "worker_sha256",
        }
        route_keys.add("direct_peer")
        route = exact_keys(
            route,
            route_keys,
            "A6000 route",
        )
        certificate_path = Path(
            safe_remote_path(
                route["artifact_certificate_path"],
                "artifact certificate path",
            )
        )
        certificate_sha256 = sha256_text(
            route["artifact_certificate_sha256"],
            "artifact certificate SHA-256",
        )
        require(certificate_path.is_file(), "artifact certificate missing")
        phase = string(route["phase"], "route phase")
        require(phase in ("A_ONLY", "B_ONLY"), "route phase")
        slot = string(route["slot"], "route slot")
        require(slot in ("A", "B"), "route slot")
        route_lock_sha256 = sha256_text(
            route["route_lock_sha256"],
            "route lock SHA-256",
        )
        phase_lock_sha256 = sha256_text(
            route["phase_lock_sha256"],
            "phase lock SHA-256",
        )
        readiness_lock_path = Path(
            safe_remote_path(
                route["readiness_lock_path"],
                "readiness lock path",
            )
        )
        readiness_lock_sha256 = sha256_text(
            route["readiness_lock_sha256"],
            "readiness lock SHA-256",
        )
        require(readiness_lock_path.is_file(), "readiness lock missing")
        artifact_certificate = parse_artifact_certificate(
            certificate_path,
            certificate_sha256,
            model_id,
            phase,
            slot,
            route_lock_sha256,
        )
        normalized_route = {
            "artifact_certificate": artifact_certificate,
            "readiness_lock": parse_readiness_lock(
                readiness_lock_path,
                readiness_lock_sha256,
                phase,
                artifact_certificate,
                phase_lock_sha256,
            ),
            "model_sha256": sha256_text(
                route["model_sha256"],
                "A6000 model SHA-256",
            ),
            "worker_sha256": sha256_text(
                route["worker_sha256"],
                "A6000 worker SHA-256",
            ),
        }
        normalized_route["qualification_sha256"] = qualification_binding(
            route["qualification"],
            model_id=model_id,
            phase=phase,
            slot=slot,
            artifact_certificate_sha256=certificate_sha256,
            readiness_lock_sha256=readiness_lock_sha256,
        )
        normalized_route["phase_lock_sha256"] = phase_lock_sha256
        peer = exact_keys(
            route["direct_peer"],
            {
                "op12_address",
                "op12_port",
                "op15_address",
                "op15_port",
                "schema",
            },
            "A6000 route direct peer",
        )
        require(
            peer["schema"] == "s40-phone-direct-peer-v1",
            "A6000 route direct peer schema",
        )
        normalized_route["direct_peer"] = {
            "op12_address": canonical_ipv4(
                peer["op12_address"],
                "A6000 route OP12 address",
            ),
            "op12_port": integer(
                peer["op12_port"],
                "A6000 route OP12 port",
                1,
            ),
            "op15_address": canonical_ipv4(
                peer["op15_address"],
                "A6000 route OP15 address",
            ),
            "op15_port": integer(
                peer["op15_port"],
                "A6000 route OP15 port",
                1,
            ),
            "schema": "s40-phone-direct-peer-v1",
        }
        require(
            normalized_route["direct_peer"]["op12_port"] <= 65535
            and normalized_route["direct_peer"]["op15_port"] <= 65535,
            "A6000 route direct peer ports",
        )
        for phone_name in ("op15", "op12"):
            phone = exact_keys(
                route[phone_name],
                {
                    "boot_id",
                    "processes",
                    "runtime_files",
                    "runtime_root",
                    "serial",
                    "shard_path",
                    "shard_sha256",
                    "shard_size",
                    "worker_path",
                },
                f"A6000 {phone_name}",
            )
            runtime_root = safe_remote_path(
                phone["runtime_root"],
                f"{phone_name} runtime root",
            )
            worker_path = safe_remote_path(
                phone["worker_path"],
                f"{phone_name} worker",
            )
            shard_path = safe_remote_path(
                phone["shard_path"],
                f"{phone_name} shard",
            )
            runtime_rows = phone["runtime_files"]
            require(
                type(runtime_rows) is list,
                f"{phone_name} runtime files",
            )
            expected_runtime_roles = (
                COMMON_PHONE_RUNTIME_ROLES
                | {"htp_skel_v81", "relay"}
                if phone_name == "op15"
                else COMMON_PHONE_RUNTIME_ROLES | {"htp_skel_v75"}
            )
            normalized_runtime = {}
            for runtime_index, runtime_file in enumerate(runtime_rows):
                runtime_file = exact_keys(
                    runtime_file,
                    {"bytes", "path", "role", "sha256", "stat"},
                    f"{phone_name} runtime file[{runtime_index}]",
                )
                role = string(
                    runtime_file["role"],
                    f"{phone_name} runtime role",
                )
                require(
                    role in expected_runtime_roles
                    and role not in normalized_runtime,
                    f"{phone_name} runtime role",
                )
                runtime_path = safe_remote_path(
                    runtime_file["path"],
                    f"{phone_name} runtime path",
                )
                expected_name = (
                    "libggml-htp-v81.so"
                    if role == "htp_skel_v81"
                    else (
                        "libggml-htp-v75.so"
                        if role == "htp_skel_v75"
                        else PHONE_RUNTIME_FILENAMES[role]
                    )
                )
                require(
                    Path(runtime_path).name == expected_name
                    and str(Path(runtime_path).parent) == runtime_root,
                    f"{phone_name} runtime role path",
                )
                runtime_stat = stat_identity(
                    runtime_file["stat"],
                    f"{phone_name} runtime stat",
                )
                runtime_bytes = integer(
                    runtime_file["bytes"],
                    f"{phone_name} runtime bytes",
                    1,
                )
                require(
                    runtime_stat["size"] == runtime_bytes,
                    f"{phone_name} runtime size",
                )
                normalized_runtime[role] = {
                    "bytes": runtime_bytes,
                    "path": runtime_path,
                    "role": role,
                    "sha256": sha256_text(
                        runtime_file["sha256"],
                        f"{phone_name} runtime SHA-256",
                    ),
                    "stat": runtime_stat,
                }
            require(
                set(normalized_runtime) == expected_runtime_roles,
                f"{phone_name} runtime role set",
            )
            require(
                normalized_runtime["worker"]["path"] == worker_path,
                f"{phone_name} worker runtime path",
            )
            processes = phone["processes"]
            require(
                type(processes) is list and 1 <= len(processes) <= 2,
                f"{phone_name} processes",
            )
            normalized_processes = []
            names = set()
            for process in processes:
                kind = (
                    process.get("kind")
                    if type(process) is dict
                    else None
                )
                if kind in ("STAGE_HEAD", "STAGE_TAIL"):
                    normalized_process = exact_stage_process(
                        process,
                        phone_name,
                        worker_path,
                        shard_path,
                        normalized_route["model_sha256"],
                        runtime_root,
                    )
                elif kind == "DIRECT_RELAY":
                    normalized_process = exact_relay_process(
                        process,
                        phone_name,
                        normalized_runtime["relay"]["path"],
                        runtime_root,
                        normalized_route["direct_peer"],
                    )
                else:
                    raise GatewayError(f"{phone_name} process kind")
                name = normalized_process["name"]
                require(name not in names, "duplicate phone process name")
                names.add(name)
                runtime_file = normalized_runtime[
                    normalized_process["artifact_role"]
                ]
                normalized_process["artifact_sha256"] = runtime_file["sha256"]
                normalized_processes.append(normalized_process)
            require(
                (
                    phone_name == "op15"
                    and {row["kind"] for row in normalized_processes}
                    == {"STAGE_HEAD", "DIRECT_RELAY"}
                )
                or (
                    phone_name == "op12"
                    and [row["kind"] for row in normalized_processes]
                    == ["STAGE_TAIL"]
                ),
                f"{phone_name} process role set",
            )
            if phone_name == "op12":
                require(
                    normalized_processes[0]["listen_port"]
                    == normalized_route["direct_peer"]["op12_port"],
                    "OP12 tail port differs from the direct peer",
                )
            else:
                head = next(
                    row for row in normalized_processes
                    if row["kind"] == "STAGE_HEAD"
                )
                relay = next(
                    row for row in normalized_processes
                    if row["kind"] == "DIRECT_RELAY"
                )
                require(
                    relay["head_port"] == head["listen_port"],
                    "OP15 relay does not target the configured head",
                )
            normalized_route[phone_name] = {
                "boot_id": string(phone["boot_id"], f"{phone_name} boot id"),
                "processes": normalized_processes,
                "runtime_files": normalized_runtime,
                "runtime_root": runtime_root,
                "serial": string(phone["serial"], f"{phone_name} serial"),
                "shard_path": shard_path,
                "shard_sha256": sha256_text(
                    phone["shard_sha256"],
                    f"{phone_name} shard SHA-256",
                ),
                "shard_size": integer(
                    phone["shard_size"],
                    f"{phone_name} shard size",
                    1,
                ),
                "worker_path": worker_path,
            }
            certificate = normalized_route["artifact_certificate"]["artifacts"]
            shard_key = (
                phone_name,
                normalized_route[phone_name]["shard_path"],
            )
            require(shard_key in certificate, f"{phone_name} shard certificate")
            shard_row = certificate[shard_key]
            require(
                shard_row["sha256"]
                == normalized_route[phone_name]["shard_sha256"]
                and shard_row["bytes"]
                == normalized_route[phone_name]["shard_size"],
                f"{phone_name} shard certificate mismatch",
            )
            worker_key = (
                f"{phone_name}_worker",
                normalized_route[phone_name]["worker_path"],
            )
            require(worker_key in certificate, f"{phone_name} worker certificate")
            worker_row = certificate[worker_key]
            require(
                worker_row["sha256"] == normalized_route["worker_sha256"],
                f"{phone_name} worker certificate mismatch",
            )
            require(
                normalized_route[phone_name]["runtime_files"]["worker"][
                    "sha256"
                ]
                == normalized_route["worker_sha256"]
                and normalized_route[phone_name]["runtime_files"]["worker"][
                    "stat"
                ]
                == worker_row["stat"],
                f"{phone_name} runtime worker certificate mismatch",
            )
            for process in normalized_processes:
                require(
                    process["artifact_path"]
                    == normalized_route[phone_name]["runtime_files"][
                        process["artifact_role"]
                    ]["path"]
                    and process["artifact_sha256"]
                    == normalized_route[phone_name]["runtime_files"][
                        process["artifact_role"]
                    ]["sha256"],
                    f"{phone_name} process is not bound to runtime",
                )
        op15_stage = next(
            row for row in normalized_route["op15"]["processes"]
            if row["kind"] == "STAGE_HEAD"
        )
        op12_stage = normalized_route["op12"]["processes"][0]
        require(
            op15_stage["layer_start"] == 0
            and op15_stage["layer_end"] == op12_stage["layer_start"]
            and op12_stage["layer_end"] > op12_stage["layer_start"],
            "phone stage chain is not contiguous",
        )
        normalized[model_id] = normalized_route
    dependency_package = None
    dependency_package_digest = None
    if schema == "s40-a6000-phone-routes-v3":
        host_boot_id = string(value["host_boot_id"], "A6000 host boot ID")
        current_boot_id = Path(
            "/proc/sys/kernel/random/boot_id"
        ).read_text(encoding="ascii").strip()
        require(current_boot_id == host_boot_id, "A6000 host boot changed")
        dependency_files = value["dependency_files"]
        require(
            type(dependency_files) is list
            and len(dependency_files) == 6,
            "A6000 dependency files",
        )
        expected_roles = {
            "a6000_phone_observer",
            "a6000_phone_route_control",
            "adb",
            "phone_gateway",
            "python",
            "readiness_v23",
        }
        locked = []
        seen_roles = set()
        for index, row in enumerate(dependency_files):
            row = exact_keys(
                row,
                {"bytes", "path", "role", "sha256"},
                f"A6000 dependency[{index}]",
            )
            role = string(row["role"], "A6000 dependency role")
            require(
                role in expected_roles and role not in seen_roles,
                "A6000 dependency role",
            )
            seen_roles.add(role)
            locked.append(locked_file_identity(
                Path(safe_remote_path(
                    row["path"],
                    f"A6000 dependency {role} path",
                )),
                role,
                integer(row["bytes"], f"A6000 dependency {role} bytes", 1),
                sha256_text(
                    row["sha256"],
                    f"A6000 dependency {role} SHA-256",
                ),
            ))
            if role in (
                "a6000_phone_observer",
                "a6000_phone_route_control",
                "adb",
                "python",
            ):
                require(
                    Path(locked[-1]["path"]).stat().st_mode & 0o111 != 0,
                    f"A6000 dependency {role} executable mode",
                )
        require(seen_roles == expected_roles, "A6000 dependency role set")
        locked.sort(key=lambda row: row["role"])
        adb_rows = [row for row in locked if row["role"] == "adb"]
        require(
            len(adb_rows) == 1 and adb_rows[0]["path"] == str(adb_path),
            "A6000 dependency ADB path",
        )
        locked.append(locked_file_identity(
            path,
            "remote_config",
            len(raw),
            hashlib.sha256(raw).hexdigest(),
        ))
        locked.sort(key=lambda row: row["role"])
        dependency_package = {
            "a6000_identity": identity,
            "files": locked,
            "host_boot_id": host_boot_id,
            "schema": "s40-a6000-identity-package-v1",
        }
        dependency_package_digest = dependency_package_sha256(
            dependency_package
        )
    return {
        "a6000_identity": identity,
        "adb_path": str(adb_path),
        "adb_port": adb_port,
        "dependency_package": dependency_package,
        "dependency_package_sha256": dependency_package_digest,
        "host_boot_id": (
            dependency_package["host_boot_id"]
            if dependency_package is not None
            else None
        ),
        "routes": normalized,
        "state_dir": state_dir,
    }


class PhoneControl:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        require(
            config["dependency_package"] is not None,
            "versioned A6000 dependency package is required",
        )
        validate_dependency_package(config)

    def checkpoint(self, name: str) -> None:
        del name

    def adb(
        self,
        serial: str,
        *arguments: str,
        check: bool = True,
        timeout: float = 60.0,
    ) -> subprocess.CompletedProcess:
        validate_dependency_package(self.config)
        completed = subprocess.run(
            [
                self.config["adb_path"],
                "-P",
                str(self.config["adb_port"]),
                "-s",
                serial,
                *arguments,
            ],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            env=ADB_ENV,
        )
        validate_dependency_package(self.config)
        if check:
            require(completed.returncode == 0, f"ADB failed for {serial}")
            require(not completed.stderr, f"ADB wrote stderr for {serial}")
        return completed

    def text(self, serial: str, command: str) -> str:
        return self.adb(serial, "shell", command).stdout.decode("ascii").strip()

    def remote_stat(self, serial: str, path: str) -> dict[str, int]:
        raw = self.text(
            serial,
            "stat -c "
            + shlex.quote("%d|%i|%s|%f|%y|%z")
            + " "
            + shlex.quote(path),
        )
        fields = raw.split("|")
        require(len(fields) == 6, "remote stat fields")
        try:
            result = {
                "device_id": int(fields[0]),
                "inode": int(fields[1]),
                "size": int(fields[2]),
                "mode": int(fields[3], 16),
                "mtime_ns": stat_time_ns(fields[4]),
                "ctime_ns": stat_time_ns(fields[5]),
            }
        except ValueError as error:
            raise GatewayError(f"remote stat value: {error}") from error
        require(
            result["device_id"] >= 0
            and result["inode"] > 0
            and result["size"] > 0
            and result["mode"] > 0,
            "remote stat values",
        )
        return result

    def verify_phone_artifacts(
        self,
        phone_name: str,
        route: dict[str, Any],
    ) -> dict[str, Any]:
        phone = route[phone_name]
        serial = phone["serial"]
        started_ns = time.monotonic_ns()
        require(
            self.adb(serial, "get-state").stdout.decode("ascii").strip()
            == "device",
            f"{serial} is not ready",
        )
        certificate = route["artifact_certificate"]["artifacts"]
        rows = [
            certificate[(phone_name, phone["shard_path"])],
            certificate[(f"{phone_name}_worker", phone["worker_path"])],
        ]
        artifacts = []
        for row in rows:
            current = self.remote_stat(serial, row["path"])
            require(
                current == row["stat"],
                f"{serial} certified artifact stat changed",
            )
            artifacts.append({
                "artifact_path": row["path"],
                "artifact_sha256": row["sha256"],
                "stat": current,
            })
        runtime_files = []
        for role, row in sorted(phone["runtime_files"].items()):
            current = self.remote_stat(serial, row["path"])
            require(
                current == row["stat"],
                f"{serial} runtime stat changed: {role}",
            )
            require(
                self.remote_sha256(serial, row["path"]) == row["sha256"],
                f"{serial} runtime digest changed: {role}",
            )
            runtime_files.append({
                "artifact_path": row["path"],
                "artifact_sha256": row["sha256"],
                "role": role,
                "stat": current,
            })
        return {
            "artifacts": artifacts,
            "artifact_certificate_sha256":
                route["artifact_certificate"]["sha256"],
            "completed_ns": time.monotonic_ns(),
            "shard_sha256": phone["shard_sha256"],
            "shard_size": phone["shard_size"],
            "started_ns": started_ns,
            "runtime_files": runtime_files,
            "worker_sha256": route["worker_sha256"],
        }

    def snapshot_phone_identity(self, phone: dict[str, Any]) -> dict[str, Any]:
        serial = phone["serial"]
        started_ns = time.monotonic_ns()
        require(
            self.adb(serial, "get-state").stdout.decode("ascii").strip()
            == "device",
            f"{serial} is not ready",
        )
        boot_id = self.text(serial, "cat /proc/sys/kernel/random/boot_id")
        require(boot_id == phone["boot_id"], f"{serial} boot identity changed")
        return {
            "boot_id": boot_id,
            "completed_ns": time.monotonic_ns(),
            "serial": serial,
            "started_ns": started_ns,
        }

    def process_cmdline_sha256(self, serial: str, pid: int) -> str:
        value = self.text(serial, f"sha256sum /proc/{pid}/cmdline")
        fields = value.split()
        require(len(fields) >= 1, "phone process command line digest")
        return sha256_text(fields[0], "phone process command line digest")

    def remote_sha256(self, serial: str, path: str) -> str:
        value = self.text(serial, f"sha256sum {shlex.quote(path)}")
        fields = value.split()
        require(len(fields) >= 1, "phone artifact digest")
        return sha256_text(fields[0], "phone artifact digest")

    def process_start_ticks(self, serial: str, pid: int) -> int:
        raw = self.text(serial, f"cat /proc/{pid}/stat")
        closing = raw.rfind(")")
        require(closing > 0, "phone process stat framing")
        fields = raw[closing + 1:].split()
        require(len(fields) > 19, "phone process stat fields")
        try:
            value = int(fields[19])
        except ValueError as error:
            raise GatewayError(f"phone process start ticks: {error}") from error
        require(value > 0, "phone process start ticks")
        return value

    @staticmethod
    def _journal_path(config: dict[str, Any]) -> Path:
        return config["state_dir"] / "active.json"

    def _read_journal(self) -> dict[str, Any]:
        path = self._journal_path(self.config)
        raw = path.read_bytes()
        value = strict_json_loads(raw, "route journal JSON")
        require(canonical_bytes(value) == raw, "route journal is not canonical")
        value = exact_keys(
            value,
            {
                "artifact_checks",
                "identity_snapshots",
                "model_id",
                "model_sha256",
                "processes",
                "readiness_lock",
                "release_completed_ns",
                "release_kind",
                "release_snapshots",
                "route_instance_id",
                "schema",
                "started_ns",
                "state",
                "unload_started_ns",
                "worker_sha256",
            },
            "route journal",
        )
        require(
            value["schema"] == "s40-a6000-route-journal-v3",
            "route journal schema",
        )
        require(
            value["state"] in ("PREPARING", "ACTIVE", "RELEASING"),
            "route journal state",
        )
        require(
            value["release_kind"] in (None, "ROLLBACK", "UNLOAD"),
            "route journal release kind",
        )
        require(
            (
                value["state"] != "RELEASING"
                and value["release_kind"] is None
                and value["release_completed_ns"] is None
                and value["release_snapshots"] is None
                and value["unload_started_ns"] is None
            )
            or (
                value["state"] == "RELEASING"
                and value["release_kind"] in ("ROLLBACK", "UNLOAD")
                and (
                    value["release_completed_ns"] is None
                    or type(value["release_completed_ns"]) is int
                )
                and type(value["release_snapshots"]) is dict
                and type(value["unload_started_ns"]) is int
            ),
            "route journal release fields",
        )
        route = self.config["routes"].get(value["model_id"])
        require(route is not None, "route journal model")
        require(
            value["model_sha256"] == route["model_sha256"]
            and value["worker_sha256"] == route["worker_sha256"]
            and value["readiness_lock"] == route["readiness_lock"],
            "route journal artifact identity",
        )
        processes = value["processes"]
        require(
            type(processes) is dict and set(processes) == {"op12", "op15"},
            "route journal process map",
        )
        for phone_name in ("op12", "op15"):
            records = processes[phone_name]
            require(type(records) is list, "route journal process list")
            seen = set()
            for record in records:
                record = exact_keys(
                    record,
                    {
                        "argv",
                        "artifact_path",
                        "artifact_role",
                        "artifact_sha256",
                        "backend",
                        "cmdline_sha256",
                        "env",
                        "graceful_exit",
                        "kind",
                        "layer_end",
                        "layer_start",
                        "lifecycle",
                        "listen_port",
                        "log",
                        "name",
                        "pid",
                        "pid_file",
                        "ready",
                        "runtime",
                        "start_ticks",
                        "tail_source_port",
                    },
                    "route journal process",
                )
                name = string(record["name"], "route journal process name")
                require(name not in seen, "duplicate route journal process")
                seen.add(name)
                require(
                    record["lifecycle"]
                    in ("LAUNCHING", "LIVE", "RELEASING", "RELEASED"),
                    "route journal process lifecycle",
                )
                require(type(record["ready"]) is bool, "route journal ready")
                require(
                    record["graceful_exit"] is None
                    or type(record["graceful_exit"]) is bool,
                    "route journal graceful exit",
                )
                require(
                    record["pid"] is None
                    or type(record["pid"]) is int and record["pid"] > 0,
                    "route journal PID",
                )
                require(
                    record["start_ticks"] is None
                    or type(record["start_ticks"]) is int
                    and record["start_ticks"] > 0,
                    "route journal process start ticks",
                )
                sha256_text(
                    record["cmdline_sha256"],
                    "route journal command line",
                )
                for field in (
                    "artifact_path",
                    "artifact_sha256",
                    "log",
                    "pid_file",
                    "runtime",
                ):
                    string(record[field], f"route journal {field}")
                require(
                    type(record["env"]) is dict
                    and record["kind"]
                    in ("STAGE_HEAD", "STAGE_TAIL", "DIRECT_RELAY")
                    and type(record["listen_port"]) is int
                    and record["listen_port"] > 0,
                    "route journal process semantics",
                )
        return value

    def _write_journal(self, state: dict[str, Any]) -> None:
        replace_durable(self._journal_path(self.config), state)

    def _new_process_record(
        self,
        phone: dict[str, Any],
        process: dict[str, Any],
        route_instance_id: str,
    ) -> dict[str, Any]:
        runtime = f"{phone['runtime_root']}/{route_instance_id}"
        return {
            "argv": list(process["argv"]),
            "artifact_path": process["artifact_path"],
            "artifact_role": process["artifact_role"],
            "artifact_sha256": process["artifact_sha256"],
            "backend": process["backend"],
            "cmdline_sha256": argv_sha256(process["argv"]),
            "env": process["env"],
            "graceful_exit": None,
            "kind": process["kind"],
            "layer_end": process["layer_end"],
            "layer_start": process["layer_start"],
            "lifecycle": "LAUNCHING",
            "listen_port": process["listen_port"],
            "log": f"{runtime}/{process['name']}.log",
            "name": process["name"],
            "pid": None,
            "pid_file": f"{runtime}/{process['name']}.pid",
            "ready": False,
            "runtime": runtime,
            "start_ticks": None,
            "tail_source_port": process.get("tail_source_port"),
        }

    def launch(
        self,
        phone_name: str,
        phone: dict[str, Any],
        state: dict[str, Any],
    ) -> None:
        serial = phone["serial"]
        runtime = f"{phone['runtime_root']}/{state['route_instance_id']}"
        self.adb(serial, "shell", f"mkdir {shlex.quote(runtime)}")
        for process in phone["processes"]:
            record = self._new_process_record(
                phone,
                process,
                state["route_instance_id"],
            )
            state["processes"][phone_name].append(record)
            self._write_journal(state)
            self.checkpoint(f"launch-planned:{phone_name}:{process['name']}")
            wrapper = (
                f"echo $$ >{shlex.quote(record['pid_file'])}; "
                "exec /system/bin/env -i "
                + " ".join(
                    f"{key}={shlex.quote(value)}"
                    for key, value in sorted(process["env"].items())
                )
                + " "
                + shlex.join(process["argv"])
            )
            command = (
                f"nohup sh -c {shlex.quote(wrapper)} "
                f">{shlex.quote(record['log'])} 2>&1 </dev/null &"
            )
            self.adb(
                serial,
                "shell",
                "sh -c " + shlex.quote(command),
            )
            pid_text = self.text(
                serial,
                f"cat {shlex.quote(record['pid_file'])}",
            )
            require(pid_text.isdigit() and int(pid_text) > 0, "phone PID")
            record["pid"] = int(pid_text)
            require(
                self.process_cmdline_sha256(serial, record["pid"])
                == record["cmdline_sha256"],
                "phone process command line differs from frozen argv",
            )
            record["start_ticks"] = self.process_start_ticks(
                serial,
                record["pid"],
            )
            record["lifecycle"] = "LIVE"
            self._write_journal(state)
            self.checkpoint(f"launch-pid:{phone_name}:{process['name']}")
            deadline = time.monotonic() + 360.0
            while True:
                ready = self.adb(
                    serial,
                    "shell",
                    (
                        f"grep -Fq {shlex.quote(process['ready_marker'])} "
                        f"{shlex.quote(record['log'])}"
                    ),
                    check=False,
                ).returncode == 0
                if ready:
                    break
                alive = self.adb(
                    serial,
                    "shell",
                    f"kill -0 {record['pid']}",
                    check=False,
                ).returncode == 0
                require(alive, f"{phone_name} process exited before ready")
                require(time.monotonic() < deadline, "phone readiness timed out")
                time.sleep(0.1)
            record["ready"] = True
            self._write_journal(state)
            self.checkpoint(f"launch-ready:{phone_name}:{process['name']}")

    def _read_planned_pid(
        self,
        serial: str,
        process: dict[str, Any],
    ) -> int | None:
        completed = self.adb(
            serial,
            "shell",
            f"cat {shlex.quote(process['pid_file'])}",
            check=False,
        )
        if completed.returncode != 0:
            return None
        value = completed.stdout.decode("ascii").strip()
        require(value.isdigit() and int(value) > 0, "recovery PID file")
        return int(value)

    @staticmethod
    def _expected_process(
        route: dict[str, Any],
        phone_name: str,
        process: dict[str, Any],
    ) -> dict[str, Any]:
        matches = [
            row
            for row in route[phone_name]["processes"]
            if row["name"] == process["name"]
        ]
        require(len(matches) == 1, "journal process is not configured")
        expected = matches[0]
        require(
            process["argv"] == list(expected["argv"])
            and process["artifact_path"] == expected["artifact_path"]
            and process["artifact_role"] == expected["artifact_role"]
            and process["artifact_sha256"] == expected["artifact_sha256"]
            and process["backend"] == expected["backend"]
            and process["env"] == expected["env"]
            and process["kind"] == expected["kind"]
            and process["layer_start"] == expected["layer_start"]
            and process["layer_end"] == expected["layer_end"]
            and process["listen_port"] == expected["listen_port"]
            and process["tail_source_port"]
            == expected.get("tail_source_port")
            and process["cmdline_sha256"] == argv_sha256(expected["argv"]),
            "journal process identity changed",
        )
        return expected

    def _begin_release(
        self,
        state: dict[str, Any],
        release_kind: str,
    ) -> None:
        route = self.config["routes"][state["model_id"]]
        if state["state"] == "RELEASING":
            require(
                state["release_kind"] == release_kind,
                "route release kind changed",
            )
            snapshots = {
                name: self.snapshot_phone_identity(route[name])
                for name in ("op12", "op15")
            }
            for phone_name in ("op12", "op15"):
                require(
                    snapshots[phone_name]["boot_id"]
                    == state["identity_snapshots"][phone_name]["boot_id"]
                    == state["release_snapshots"][phone_name]["boot_id"],
                    "phone rebooted during route release",
                )
            return
        require(
            state["state"] in ("PREPARING", "ACTIVE"),
            "route cannot begin release",
        )
        snapshots = {
            name: self.snapshot_phone_identity(route[name])
            for name in ("op12", "op15")
        }
        for phone_name in ("op12", "op15"):
            require(
                snapshots[phone_name]["boot_id"]
                == state["identity_snapshots"][phone_name]["boot_id"],
                "phone rebooted while route was resident",
            )
        state["release_kind"] = release_kind
        state["release_snapshots"] = snapshots
        state["state"] = "RELEASING"
        state["unload_started_ns"] = time.monotonic_ns()
        self._write_journal(state)
        self.checkpoint("release-begin")

    def _release_one(
        self,
        state: dict[str, Any],
        route: dict[str, Any],
        phone_name: str,
        process: dict[str, Any],
    ) -> None:
        self._expected_process(route, phone_name, process)
        serial = route[phone_name]["serial"]
        lifecycle = process["lifecycle"]
        if lifecycle == "RELEASED":
            return
        if lifecycle == "LIVE":
            pid = integer(process["pid"], "live journal PID", 1)
            alive = self.adb(
                serial,
                "shell",
                f"kill -0 {pid}",
                check=False,
            ).returncode == 0
            require(
                alive or state["release_kind"] == "UNLOAD",
                "live journal process exited before rollback",
            )
            if alive:
                require(
                    self.process_cmdline_sha256(serial, pid)
                    == process["cmdline_sha256"],
                    "refusing to stop a reused phone PID",
                )
                require(
                    self.process_start_ticks(serial, pid)
                    == process["start_ticks"],
                    "refusing to stop a reused phone process",
                )
        elif lifecycle == "LAUNCHING":
            require(process["pid"] is None, "launching process has a PID")
        else:
            require(
                lifecycle == "RELEASING",
                "invalid release process state",
            )
        if lifecycle != "RELEASING":
            process["lifecycle"] = "RELEASING"
            self._write_journal(state)
            self.checkpoint(
                f"release-mark:{phone_name}:{process['name']}"
            )
        pid = process["pid"]
        if pid is None:
            pid = self._read_planned_pid(serial, process)
            if pid is not None:
                require(
                    self.process_cmdline_sha256(serial, pid)
                    == process["cmdline_sha256"],
                    "refusing to adopt an unexpected phone process",
                )
                process["pid"] = pid
                process["start_ticks"] = self.process_start_ticks(serial, pid)
                self._write_journal(state)
                self.checkpoint(
                    f"release-adopt:{phone_name}:{process['name']}"
                )
        if pid is not None:
            alive = self.adb(
                serial,
                "shell",
                f"kill -0 {pid}",
                check=False,
            ).returncode == 0
            if alive and state["release_kind"] == "UNLOAD":
                alive = not self.wait_for_process_exit(serial, pid, 30.0)
            if alive:
                require(
                    self.process_cmdline_sha256(serial, pid)
                    == process["cmdline_sha256"],
                    "refusing to stop a reused phone PID",
                )
                require(
                    self.process_start_ticks(serial, pid)
                    == process["start_ticks"],
                    "refusing to stop a reused phone process",
                )
                killed = self.adb(
                    serial,
                    "shell",
                    f"kill {pid}",
                    check=False,
                )
                require(killed.returncode == 0, "phone process kill failed")
                deadline = time.monotonic() + 30.0
                while self.adb(
                    serial,
                    "shell",
                    f"kill -0 {pid}",
                    check=False,
                ).returncode == 0:
                    require(
                        time.monotonic() < deadline,
                        "phone process did not exit",
                    )
                    time.sleep(0.1)
            process["graceful_exit"] = (
                state["release_kind"] == "UNLOAD" and not alive
            )
        process["lifecycle"] = "RELEASED"
        self._write_journal(state)
        self.checkpoint(f"release-done:{phone_name}:{process['name']}")

    def _finish_release(
        self,
        state: dict[str, Any],
        release_kind: str,
    ) -> None:
        self._begin_release(state, release_kind)
        route = self.config["routes"][state["model_id"]]
        for phone_name in ("op15", "op12"):
            for process in reversed(state["processes"][phone_name]):
                self._release_one(state, route, phone_name, process)
        require(
            all(
                process["lifecycle"] == "RELEASED"
                for records in state["processes"].values()
                for process in records
            ),
            "route release is incomplete",
        )
        graceful = all(
            process["graceful_exit"] is True
            for records in state["processes"].values()
            for process in records
        )
        if state["release_completed_ns"] is None:
            state["release_completed_ns"] = time.monotonic_ns()
            self._write_journal(state)
            self.checkpoint("release-all-done")
        prefix = "completed" if release_kind == "UNLOAD" else "aborted"
        completed_path = (
            self.config["state_dir"]
            / f"{prefix}-{state['route_instance_id']}.json"
        )
        completed = {
            **state,
            "schema": "s40-a6000-completed-route-v3",
        }
        if completed_path.exists():
            raw = completed_path.read_bytes()
            require(
                strict_json_loads(raw, "completed route") == completed
                and canonical_bytes(completed) == raw,
                "completed route record changed",
            )
        else:
            write_durable_new(completed_path, completed)
        self.checkpoint("release-recorded")
        remove_durable(self._journal_path(self.config))
        require(
            release_kind != "UNLOAD" or graceful,
            "phone route required forced termination",
        )

    def wait_for_process_exit(
        self,
        serial: str,
        pid: int,
        timeout_s: float,
    ) -> bool:
        deadline = time.monotonic() + timeout_s
        while self.adb(
            serial,
            "shell",
            f"kill -0 {pid}",
            check=False,
        ).returncode == 0:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.1)
        return True

    def _recover_incomplete(self) -> None:
        active_path = self._journal_path(self.config)
        if not active_path.exists():
            return
        state = self._read_journal()
        if state["state"] == "ACTIVE":
            raise GatewayError("an active phone route already exists")
        release_kind = state["release_kind"] or "ROLLBACK"
        self._finish_release(state, release_kind)

    def synchronous_observation(
        self,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        from a6000_phone_observer import PhoneObserver

        return PhoneObserver(self).observe_route(state)

    def load(self, model_id: str) -> dict[str, Any]:
        route = self.config["routes"].get(model_id)
        require(route is not None, "unknown A6000 route")
        state_dir: Path = self.config["state_dir"]
        state_dir.mkdir(parents=True, exist_ok=True)
        fsync_directory(state_dir.parent)
        self._recover_incomplete()
        active_path = self._journal_path(self.config)
        started_ns = time.monotonic_ns()
        route_instance_id = (
            f"{model_id}-{os.getpid()}-{started_ns}"
        )
        artifact_checks = {
            name: self.verify_phone_artifacts(name, route)
            for name in ("op12", "op15")
        }
        identity_snapshots = {
            name: self.snapshot_phone_identity(route[name])
            for name in ("op12", "op15")
        }
        require(
            artifact_checks["op12"]["worker_sha256"] == route["worker_sha256"]
            and artifact_checks["op15"]["worker_sha256"]
            == route["worker_sha256"],
            "phone worker identity changed",
        )
        state = {
            "artifact_checks": artifact_checks,
            "identity_snapshots": identity_snapshots,
            "model_id": model_id,
            "model_sha256": route["model_sha256"],
            "processes": {"op12": [], "op15": []},
            "readiness_lock": route["readiness_lock"],
            "release_completed_ns": None,
            "release_kind": None,
            "release_snapshots": None,
            "route_instance_id": route_instance_id,
            "schema": "s40-a6000-route-journal-v3",
            "started_ns": started_ns,
            "state": "PREPARING",
            "unload_started_ns": None,
            "worker_sha256": route["worker_sha256"],
        }
        write_durable_new(active_path, state)
        self.checkpoint("prepare-created")
        try:
            self.launch("op12", route["op12"], state)
            self.launch("op15", route["op15"], state)
            for phone_name in ("op12", "op15"):
                require(
                    len(state["processes"][phone_name])
                    == len(route[phone_name]["processes"])
                    and all(
                        row["lifecycle"] == "LIVE" and row["ready"]
                        for row in state["processes"][phone_name]
                    ),
                    "phone route did not become ready",
                )
            state["state"] = "ACTIVE"
            self._write_journal(state)
            self.checkpoint("active")
            route_observation = self.synchronous_observation(state)
        except BaseException:
            if active_path.exists():
                rollback = self._read_journal()
                self._finish_release(rollback, "ROLLBACK")
            raise
        return {
            "a6000_identity": self.config["a6000_identity"],
            "artifact_certificate_sha256":
                route["artifact_certificate"]["sha256"],
            "model_id": model_id,
            "model_sha256": route["model_sha256"],
            "op12_boot_id": identity_snapshots["op12"]["boot_id"],
            "op12_shard_sha256": artifact_checks["op12"]["shard_sha256"],
            "op15_boot_id": identity_snapshots["op15"]["boot_id"],
            "op15_shard_sha256": artifact_checks["op15"]["shard_sha256"],
            "qualification_sha256": route["qualification_sha256"],
            "readiness_lock_sha256":
                route["readiness_lock"]["sha256"],
            "readiness_phase_id":
                route["readiness_lock"]["phase_id"],
            "route_observation": route_observation,
            "route_instance_id": route_instance_id,
            "schema": "s40-phone-route-load-v3",
            "success": True,
            "worker_sha256": route["worker_sha256"],
        }

    def unload(self, model_id: str) -> dict[str, Any]:
        state = self._read_journal()
        require(state["model_id"] == model_id, "active route model mismatch")
        if state["state"] == "PREPARING":
            self._finish_release(state, "ROLLBACK")
            raise GatewayError("phone route was incomplete and was rolled back")
        require(
            state["state"] == "ACTIVE"
            or (
                state["state"] == "RELEASING"
                and state["release_kind"] == "UNLOAD"
            ),
            "phone route is not unloadable",
        )
        route_instance_id = state["route_instance_id"]
        self._finish_release(state, "UNLOAD")
        route = self.config["routes"][model_id]
        tail_processes = [
            process
            for process in state["processes"]["op12"]
            if process["kind"] == "STAGE_TAIL"
        ]
        require(len(tail_processes) == 1, "phone tail process count")
        n_layer = tail_processes[0]["layer_end"]
        placements = {}
        for phone_name in ("op12", "op15"):
            stage_processes = [
                process
                for process in state["processes"][phone_name]
                if process["kind"] in ("STAGE_HEAD", "STAGE_TAIL")
            ]
            require(
                len(stage_processes) == 1,
                f"{phone_name} stage process count",
            )
            process = stage_processes[0]
            log_tail = self.text(
                route[phone_name]["serial"],
                f"tail -n 64 {shlex.quote(process['log'])}",
            )
            placements[phone_name] = parse_stage_certificates(
                log_tail,
                process,
                route[phone_name]["boot_id"],
                n_layer,
            )
        return {
            "model_id": model_id,
            "placements": placements,
            "route_instance_id": route_instance_id,
            "schema": "s40-phone-route-unload-v2",
            "success": True,
        }

    def rollback(self, model_id: str) -> dict[str, Any]:
        state = self._read_journal()
        require(state["model_id"] == model_id, "active route model mismatch")
        route_instance_id = state["route_instance_id"]
        self._finish_release(state, "ROLLBACK")
        return {
            "model_id": model_id,
            "route_instance_id": route_instance_id,
            "schema": "s40-phone-route-rollback-v1",
            "success": True,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        choices=("load", "rollback", "unload"),
        required=True,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    require(args.config.is_absolute(), "A6000 config path")
    controller = PhoneControl(load_config(args.config))
    if args.action == "load":
        result = controller.load(args.model)
    elif args.action == "unload":
        result = controller.unload(args.model)
    else:
        result = controller.rollback(args.model)
    __import__("sys").stdout.buffer.write(canonical_bytes(result))
    __import__("sys").stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (GatewayError, OSError, subprocess.SubprocessError) as error:
        print(f"A6000 phone control failed: {error}", file=__import__("sys").stderr)
        raise SystemExit(2)
