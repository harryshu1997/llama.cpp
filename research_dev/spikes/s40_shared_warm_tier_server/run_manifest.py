#!/usr/bin/env python3
"""Validate an S40 physical-run manifest and every bound artifact."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
from typing import Any

EXECUTOR_DIR = Path(__file__).resolve().parent / "executors"
if str(EXECUTOR_DIR) not in sys.path:
    sys.path.insert(0, str(EXECUTOR_DIR))

from build_runtime_config import build_runtime_config
from campaign_plan import validate_campaign, validate_campaign_tool_lock
from evidence_common import (
    EvidenceError,
    canonical_bytes,
    digest_file,
    parse_json,
    read_json,
    read_jsonl,
    require,
    require_int,
    require_string,
    validate_digest,
)
from event_evidence import reduce_paths
from gpu_isolation import (
    canonical_lock_path,
    validate_lock_record,
    validate_observation,
)
from bridge_overhead import (
    validate_combined_measurement,
    validate_combined_transport_gate,
)
from validate_http_evidence import validate_http_evidence
from validate_inputs import (
    DEFAULT_CONTRACT,
    validate_contract,
    validate_serving_argv,
)
from validate_executor_evidence import validate_executor_bundle
from validate_phone_observer import validate_phone_observer
from executor_bundle import (  # noqa: E402
    SOURCES as EXECUTOR_BUNDLE_SOURCES,
    validate_executor_bundle as validate_python_executor_bundle,
)
from evidence_bundle import (  # noqa: E402
    SOURCES as EVIDENCE_BUNDLE_SOURCES,
    validate_bundle as validate_evidence_bundle,
)


MANIFEST_KEYS = {
    "artifacts",
    "cache_regime",
    "campaign_binding",
    "command_argv",
    "controller_binary",
    "development",
    "devices",
    "executor_bindings",
    "experiment_contract_sha256",
    "mode",
    "policy_id",
    "repeat_index",
    "run_id",
    "schema",
    "schema_version",
}
MANIFEST_CAMPAIGN_BINDING_KEYS = {
    "campaign_id",
    "campaign_sha256",
    "order",
    "phase",
}
SOURCE_CAMPAIGN_BINDING_KEYS = {
    "campaign_path",
    "campaign_sha256",
    "order",
    "phase",
}
ARTIFACT_KEYS = {
    "bytes",
    "format",
    "path",
    "record_count",
    "role",
    "sha256",
}
DEVICE_KEYS = {
    "boot_id",
    "device_role",
    "stable_id",
}
BINARY_KEYS = {
    "bytes",
    "captured_path",
    "executed_path",
    "sha256",
}
EXECUTOR_BINDING_KEYS = {
    "command_role",
    "controller_binding_role",
    "executor_id",
    "gateway_argv",
    "gateway_argv_role",
    "gateway_config_role",
    "gateway_executed_files",
    "gateway_final_identity",
    "gateway_launch_environment",
    "gateway_post_auth_identity",
    "gateway_prepublication_identity",
    "gateway_source_role",
    "gateway_stderr_role",
    "gateway_stdout_role",
    "transport_descriptor_role",
}
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
FORMATS = {"BINARY", "JSON", "JSONL", "TEXT"}
BASE_ARTIFACT_ROLES = {
    "activation_evidence",
    "controller_identity",
    "controller_launch",
    "controller_events",
    "evidence_root",
    "evidence_bundle_manifest",
    "executor_bundle_manifest",
    "gpu_lock_record",
    "gpu_observer",
    "gpu_observer_argv",
    "gpu_observer_stderr",
    "gpu_observer_stdout",
    "http_evidence",
    "native_bench_binary",
    "orchestrator_evidence",
    "resource_samples",
    "resource_sampler_stderr",
    "resource_sampler_stdout",
    "resource_sampler_argv",
    "runtime_dependency_manifest",
    "runtime_config",
    "runtime_plan",
    "server_stderr",
    "server_readiness",
    "server_stdout",
    "source_preflight",
    "transport_overhead",
    "transport_overhead_stderr",
    "transport_overhead_stdout",
    "trace_driver_stderr",
    "trace_driver_stdout",
    "trace_acquisition_result",
    "trace_start",
}
ACTIVATION_KEYS = {
    "http_status",
    "method",
    "path",
    "request_body_base64",
    "request_body_sha256",
    "response_body_base64",
    "response_body_sha256",
    "run_id",
    "schema",
    "sequence",
    "t_end_ns",
    "t_start_ns",
}
READINESS_KEYS = {
    "activation_evidence_sha256",
    "activation_ready",
    "base_readiness",
    "controller_bindings",
    "controller_identity",
    "controller_started_ns",
    "gateway_final",
    "gateway_post_auth",
    "gateway_ready",
    "host_boot_id",
    "launch_environment",
    "phone_observer",
    "run_id",
    "runtime_config_device",
    "runtime_config_inode",
    "runtime_config_path",
    "runtime_config_published_ns",
    "runtime_config_sha256",
    "schema",
}
LAUNCH_KEYS = {
    "binary_sha256",
    "command_argv",
    "controller_identity_device",
    "controller_identity_inode",
    "controller_identity_path",
    "controller_identity_sha256",
    "controller_pid",
    "controller_start_ticks",
    "experiment_contract_sha256",
    "exit_code",
    "host_boot_id",
    "launch_environment",
    "library_path",
    "physical_plan_sha256",
    "run_id",
    "runtime_config_sha256",
    "schema",
    "selected_gpu_environment",
    "source_preflight_sha256",
    "started_ns",
    "stopped_ns",
}
SOURCE_PREFLIGHT_KEYS = {
    "captured_ns",
    "rows",
    "schema",
}
SOURCE_PREFLIGHT_ROW_KEYS = {
    "bytes",
    "captured_path",
    "role",
    "sha256",
    "source_path",
}
SOURCE_PHYSICAL_PLAN_KEYS = {
    "cache_regime",
    "campaign_binding",
    "contract_path",
    "development",
    "devices",
    "gateway_processes",
    "gpu_index",
    "gpu_lock_path",
    "gpu_pci_bus_id",
    "ldd_path",
    "mode",
    "nvidia_smi_path",
    "output_dir",
    "phone_identity_argv",
    "phone_telemetry_argv",
    "repeat_index",
    "run_id",
    "runtime_plan_template",
    "schema",
    "server_argv",
    "server_base_url",
}
ORCHESTRATOR_KEYS = {
    "acquisition_returncode",
    "cleanup",
    "completed_ns",
    "run_id",
    "schema",
}
ORCHESTRATOR_CLEANUP_KEYS = {
    "exit_code",
    "name",
}
PHONE_ARTIFACT_ROLES = {
    "phone_identity",
    "phone_identity_argv",
    "phone_telemetry",
    "phone_telemetry_argv",
    "phone_telemetry_stderr",
}
LDD_ADDRESS = re.compile(r"\s+\(0x[0-9a-fA-F]+\)$")
MODE_DEVICE_ROLES = {
    "C1_GPU_ONLY_OPTIMIZED": {"GPU"},
    "C2_GPU_PLUS_CPU_WARM_EXECUTOR": {"CPU", "GPU"},
    "C3_DUAL_PARTIAL_OFFLOAD": {"CPU", "GPU"},
    "C4_TWO_GPU_ORACLE": {"GPU", "GPU1"},
    "T1_PHONE_WARM_TIER": {"GPU", "OP12", "OP15"},
    "T2_PHONE_NO_PROMOTION": {"GPU", "OP12", "OP15"},
}
QUALIFICATION_ROOT_KEYS = {
    "artifact_snapshot",
    "bundle_manifest_sha256",
    "bundle_root",
    "fresh_snapshot",
    "readiness_lock",
    "runtime_identity",
}
QUALIFICATION_FILE_KEYS = {"path", "sha256"}
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
CONTROLLER_PUBLICATION_KEYS = {
    "captured_ns",
    "device",
    "inode",
    "path",
    "published_ns",
    "sha256",
}
CONTROLLER_BINDING_KEYS = {
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
CONTROLLER_BINDING_READINESS_KEYS = {
    "device",
    "executor_id",
    "inode",
    "path",
    "sha256",
}
TRANSPORT_DESCRIPTOR_V3_KEYS = {
    "controller_executable_path",
    "controller_executable_sha256",
    "controller_gid",
    "controller_binding_device",
    "controller_binding_inode",
    "controller_binding_path",
    "controller_binding_sha256",
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
    "gateway_pid",
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
}
TRANSPORT_DESCRIPTOR_V4_KEYS = TRANSPORT_DESCRIPTOR_V3_KEYS | {
    "gateway_environment",
    "gateway_executed_files",
    "gateway_post_auth_identity",
    "gateway_prepublication_identity",
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
ISOLATED_LAUNCHER = (
    "import runpy,sys;"
    "root=sys.argv[1];script=sys.argv[2];"
    "sys.path.insert(0,root);"
    "sys.argv=[script]+sys.argv[3:];"
    "runpy.run_path(script,run_name='__main__')"
)


def _bound_path(root: Path, value: Any, field: str) -> Path:
    text = require_string(value, field)
    relative = Path(text)
    require(
        not relative.is_absolute()
        and ".." not in relative.parts,
        f"{field}: expected contained relative path",
    )
    path = (root / relative).resolve()
    require(path.is_file(), f"{field}: missing file {path}")
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise EvidenceError(f"{field}: path escapes run directory") from error
    return path


def _isolated_python_argv(
        python: Path,
        bundle_root: Path,
        script: Path,
        arguments: list[str]) -> list[str]:
    return [
        str(python),
        "-I",
        "-S",
        "-B",
        "-c",
        ISOLATED_LAUNCHER,
        str(bundle_root),
        str(script),
        *arguments,
    ]


def qualification_source_records(
        value: Any,
        executor_id: str,
        expected_model_id: str) -> list[dict[str, str]]:
    field = f"qualification::{executor_id}::{expected_model_id}"
    require(
        isinstance(value, dict)
        and set(value)
        == {"a_chain", "current", "model_id", "phase", "schema", "slot"}
        and value["schema"] == "s40-route-qualification-authority-v1"
        and value["model_id"] == expected_model_id,
        f"{field}: authority identity mismatch",
    )
    phase = value["phase"]
    slot = value["slot"]
    require(
        (phase, slot) in {("A_ONLY", "A"), ("B_ONLY", "B")},
        f"{field}: phase or slot mismatch",
    )
    require(
        (phase == "A_ONLY" and value["a_chain"] is None)
        or (phase == "B_ONLY" and isinstance(value["a_chain"], dict)),
        f"{field}: A-chain mismatch",
    )

    roots = [("current", value["current"])]
    if value["a_chain"] is not None:
        roots.append(("a_chain", value["a_chain"]))
    records = []
    seen_paths = set()
    for root_name, root_value in roots:
        root_field = f"{field}::{root_name}"
        require(
            isinstance(root_value, dict)
            and set(root_value) == QUALIFICATION_ROOT_KEYS,
            f"{root_field}: root key set mismatch",
        )
        root = Path(require_string(
            root_value["bundle_root"], f"{root_field}.bundle_root"))
        require(
            root.is_absolute()
            and root.is_dir()
            and not root.is_symlink()
            and root.resolve() == root,
            f"{root_field}: invalid bundle root",
        )
        manifest_digest = validate_digest(
            root_value["bundle_manifest_sha256"],
            f"{root_field}.bundle_manifest_sha256",
        )
        manifest_path = root / "EVIDENCE_BUNDLE.json"

        def add(role_suffix: str, source: Path, digest: str) -> None:
            require(
                source.is_absolute()
                and source.is_file()
                and not source.is_symlink()
                and source.resolve() == source
                and source.is_relative_to(root)
                and digest_file(source) == digest
                and source not in seen_paths,
                f"{root_field}: invalid or duplicate {role_suffix}",
            )
            seen_paths.add(source)
            records.append({
                "path": str(source),
                "role": f"{field}::{root_name}::{role_suffix}",
                "sha256": digest,
            })

        add("bundle_manifest", manifest_path, manifest_digest)
        manifest = read_json(
            manifest_path, f"{root_field}.bundle_manifest")
        require(
            manifest_path.read_bytes() == canonical_bytes(manifest)
            and isinstance(manifest, dict)
            and isinstance(manifest.get("artifacts"), list)
            and manifest["artifacts"],
            f"{root_field}: invalid bundle manifest",
        )
        for index, artifact in enumerate(manifest["artifacts"]):
            artifact_field = f"{root_field}.artifacts[{index}]"
            require(isinstance(artifact, dict),
                    f"{artifact_field}: expected object")
            relative_text = require_string(
                artifact.get("path"), f"{artifact_field}.path")
            relative = PurePosixPath(relative_text)
            require(
                not relative.is_absolute()
                and ".." not in relative.parts
                and "." not in relative.parts,
                f"{artifact_field}: invalid relative path",
            )
            digest = validate_digest(
                artifact.get("sha256"), f"{artifact_field}.sha256")
            add(
                f"artifact::{index:04d}",
                root.joinpath(*relative.parts),
                digest,
            )
        for name in (
                "artifact_snapshot",
                "readiness_lock",
                "fresh_snapshot",
                "runtime_identity"):
            record = root_value[name]
            record_field = f"{root_field}.{name}"
            require(
                isinstance(record, dict)
                and set(record) == QUALIFICATION_FILE_KEYS,
                f"{record_field}: key set mismatch",
            )
            source = Path(require_string(
                record["path"], f"{record_field}.path"))
            digest = validate_digest(
                record["sha256"], f"{record_field}.sha256")
            add(name, source, digest)
    return records


def validate_qualification_route_identity(
        value: Any,
        route: dict[str, Any],
        executor_id: str) -> None:
    model_id = require_string(
        route.get("model_id"), "qualification route model_id")
    qualification_source_records(value, executor_id, model_id)
    current = value["current"]
    artifact_path = Path(current["artifact_snapshot"]["path"])
    readiness_path = Path(current["readiness_lock"]["path"])
    artifact = read_json(
        artifact_path, "qualification artifact snapshot")
    readiness = read_json(
        readiness_path, "qualification readiness lock")
    require(
        artifact_path.read_bytes() == canonical_bytes(artifact)
        and isinstance(artifact, dict)
        and set(artifact)
        == {
            "artifacts",
            "completed_ns",
            "model_id",
            "phase",
            "route_lock_sha256",
            "schema",
            "slot",
            "started_ns",
        }
        and artifact["schema"] == "s39-cp0-r1-artifact-snapshot-v2.3",
        "qualification route: artifact snapshot mismatch",
    )
    require(
        isinstance(artifact["artifacts"], list)
        and artifact["artifacts"],
        "qualification route: artifact list is empty",
    )
    require(
        readiness_path.read_bytes() == canonical_bytes(readiness)
        and isinstance(readiness, dict)
        and set(readiness)
        == {
            "artifact_snapshot_sha256",
            "event_ns",
            "phase",
            "phase_id",
            "schema",
            "v2_2_phase_lock_sha256",
        }
        and readiness["schema"] == "s39-cp0-r1-readiness-lock-v2.3",
        "qualification route: readiness lock mismatch",
    )
    artifact_sha256 = validate_digest(
        route.get("artifact_certificate_sha256"),
        "qualification route artifact_certificate_sha256",
    )
    readiness_sha256 = validate_digest(
        route.get("readiness_lock_sha256"),
        "qualification route readiness_lock_sha256",
    )
    route_lock_sha256 = validate_digest(
        route.get("route_lock_sha256"),
        "qualification route route_lock_sha256",
    )
    phase_lock_sha256 = validate_digest(
        route.get("phase_lock_sha256"),
        "qualification route phase_lock_sha256",
    )
    phase = require_string(route.get("phase"), "qualification route phase")
    slot = require_string(route.get("slot"), "qualification route slot")
    require(
        value["phase"] == phase
        and value["slot"] == slot
        and current["artifact_snapshot"]["sha256"] == artifact_sha256
        == digest_file(artifact_path)
        and current["readiness_lock"]["sha256"] == readiness_sha256
        == digest_file(readiness_path)
        and artifact["model_id"] == model_id
        and artifact["phase"] == phase
        and artifact["slot"] == slot
        and artifact["route_lock_sha256"] == route_lock_sha256
        and readiness["artifact_snapshot_sha256"] == artifact_sha256
        and readiness["phase"] == phase
        and readiness["phase_id"] == route.get("readiness_phase_id")
        and readiness["v2_2_phase_lock_sha256"] == phase_lock_sha256
        and route_lock_sha256 != phase_lock_sha256,
        "qualification route: route identity mismatch",
    )
    if "artifact_certificate_path" in route:
        require(
            Path(route["artifact_certificate_path"]).resolve()
            == artifact_path.resolve()
            and Path(route["readiness_lock_path"]).resolve()
            == readiness_path.resolve(),
            "qualification route: local evidence path mismatch",
        )


def _flag_value(argv: list[str], flag: str, field: str) -> str:
    positions = [index for index, value in enumerate(argv) if value == flag]
    require(
        len(positions) == 1 and positions[0] + 1 < len(argv),
        f"{field}: {flag} must occur exactly once",
    )
    return argv[positions[0] + 1]


def _validate_file_binding(
        root: Path,
        record: dict[str, Any],
        field: str,
        path_key: str = "path",
        minimum_bytes: int = 0) -> Path:
    path = _bound_path(root, record.get(path_key), f"{field}.{path_key}")
    expected_bytes = require_int(
        record.get("bytes"), f"{field}.bytes", minimum_bytes)
    require(path.stat().st_size == expected_bytes,
            f"{field}: byte count mismatch")
    expected_digest = validate_digest(record.get("sha256"), f"{field}.sha256")
    require(digest_file(path) == expected_digest,
            f"{field}: SHA-256 mismatch")
    return path


def _validate_artifact(
        root: Path,
        record: Any,
        index: int) -> tuple[str, Path]:
    field = f"artifact[{index}]"
    require(isinstance(record, dict), f"{field}: expected object")
    require(set(record) == ARTIFACT_KEYS, f"{field}: key set mismatch")
    role = require_string(record["role"], f"{field}.role")
    file_format = require_string(record["format"], f"{field}.format")
    require(file_format in FORMATS, f"{field}: unsupported format")
    path = _validate_file_binding(root, record, field)
    record_count = record["record_count"]
    if file_format == "JSONL":
        require_int(record_count, f"{field}.record_count", 1)
        require(
            len(read_jsonl(path, field)) == record_count,
            f"{field}: record count mismatch",
        )
    else:
        require(record_count is None, f"{field}: unexpected record count")
        if file_format == "JSON":
            value = read_json(path, field)
            require(path.read_bytes() == canonical_bytes(value),
                    f"{field}: noncanonical JSON")
        elif file_format == "TEXT":
            try:
                path.read_text(encoding="ascii")
            except (OSError, UnicodeDecodeError) as error:
                raise EvidenceError(f"{field}: invalid ASCII text") from error
    return role, path


def _require_exact_http_threads(argv: list[str], required: int) -> None:
    positions = [
        index for index, value in enumerate(argv)
        if value == "--threads-http"
    ]
    require(len(positions) == 1, "command: --threads-http must occur once")
    index = positions[0]
    require(
        index + 1 < len(argv) and argv[index + 1] == str(required),
        f"command: --threads-http must equal {required}",
    )


def _expected_executor_roles(runtime: dict[str, Any]) -> set[str]:
    roles: set[str] = set()
    for executor in runtime["executors"]:
        executor_id = executor["executor_id"]
        roles.update({
            f"executor_command::{executor_id}",
            f"executor_controller_binding::{executor_id}",
            f"executor_gateway_argv::{executor_id}",
            f"executor_transport::{executor_id}",
            f"gateway_config::{executor_id}",
            f"gateway_stderr::{executor_id}",
            f"gateway_stdout::{executor_id}",
        })
        if executor["role"] == "PHONE":
            roles.update({
                f"executor_route::{executor_id}",
                f"executor_wire::{executor_id}",
            })
    return roles


def _validate_launch_environment(
        value: Any,
        field: str,
        *,
        privileged: bool = True) -> dict[str, str]:
    expected_keys = (
        PRIVILEGED_LAUNCH_ENV_KEYS
        if privileged
        else BASE_LAUNCH_ENV_KEYS
    )
    require(
        isinstance(value, dict)
        and set(value) == expected_keys
        and not (set(value) & PROHIBITED_LAUNCH_ENV)
        and all(
            isinstance(key, str)
            and isinstance(item, str)
            and key
            and item
            and key.isascii()
            and item.isascii()
            and "\x00" not in key
            and "\x00" not in item
            for key, item in value.items()
        ),
        f"{field}: invalid deterministic environment",
    )
    return value


def _expected_nul_cmdline(argv: list[str]) -> bytes:
    require(
        argv
        and all(
            isinstance(argument, str)
            and argument.isascii()
            and "\x00" not in argument
            for argument in argv
        ),
        "gateway argv: invalid command array",
    )
    return b"".join(argument.encode("ascii") + b"\x00" for argument in argv)


def _validate_gateway_process_identity(
        value: Any,
        argv: list[str],
        field: str,
        expected: dict[str, Any] | None = None) -> dict[str, Any]:
    require(
        isinstance(value, dict)
        and set(value) == GATEWAY_PROCESS_IDENTITY_KEYS,
        f"{field}: key set mismatch",
    )
    for key in (
            "executable_ctime_ns",
            "executable_device",
            "executable_inode",
            "executable_mtime_ns",
            "executable_size",
            "gateway_pid",
            "gateway_start_time_ticks",
            "observed_ns"):
        require_int(
            value[key],
            f"{field}.{key}",
            1 if key not in {"executable_device"} else 0,
        )
    executable_path = Path(require_string(
        value["executable_path"], f"{field}.executable_path"))
    require(
        executable_path.is_absolute()
        and executable_path.resolve() == Path(argv[0]).resolve(),
        f"{field}: executable path mismatch",
    )
    validate_digest(
        value["executable_sha256"], f"{field}.executable_sha256")
    validate_digest(value["cmdline_sha256"], f"{field}.cmdline_sha256")
    try:
        cmdline = base64.b64decode(
            require_string(
                value["cmdline_base64"], f"{field}.cmdline_base64"),
            validate=True,
        )
    except (ValueError, binascii.Error) as error:
        raise EvidenceError(f"{field}: invalid cmdline encoding") from error
    require(
        cmdline == _expected_nul_cmdline(argv)
        and hashlib.sha256(cmdline).hexdigest()
        == value["cmdline_sha256"],
        f"{field}: cmdline mismatch",
    )
    if expected is not None:
        require(
            isinstance(expected, dict)
            and set(expected) == GATEWAY_PROCESS_IDENTITY_KEYS
            and {
                key: value[key]
                for key in GATEWAY_PROCESS_IDENTITY_KEYS - {"observed_ns"}
            }
            == {
                key: expected[key]
                for key in GATEWAY_PROCESS_IDENTITY_KEYS - {"observed_ns"}
            }
            and value["observed_ns"] >= expected["observed_ns"],
            f"{field}: process replacement detected",
        )
    return value


def _validate_gateway_identity_rows(
        value: Any,
        field: str,
        binding_by_id: dict[str, dict[str, Any]],
        identity_key: str) -> list[dict[str, Any]]:
    require(
        isinstance(value, list)
        and [
            row.get("executor_id")
            for row in value
            if isinstance(row, dict)
        ] == sorted(binding_by_id),
        f"{field}: executor order mismatch",
    )
    result = []
    for index, row in enumerate(value):
        row_field = f"{field}[{index}]"
        require(
            isinstance(row, dict)
            and set(row) == {"executor_id", "identity"},
            f"{row_field}: key set mismatch",
        )
        executor_id = require_string(
            row["executor_id"], f"{row_field}.executor_id")
        require(
            executor_id in binding_by_id,
            f"{row_field}: unknown executor",
        )
        binding = binding_by_id[executor_id]
        expected = binding[identity_key]
        observed = _validate_gateway_process_identity(
            row["identity"],
            binding["gateway_argv"],
            f"{row_field}.identity",
        )
        require(
            observed == expected,
            f"{row_field}: manifest identity mismatch",
        )
        result.append(row)
    return result


def _validate_executed_files(
        root: Path,
        argv: list[str],
        binding: dict[str, Any],
        field: str,
        binding_key: str) -> None:
    values = binding[binding_key]
    require(isinstance(values, list) and values,
            f"{field}.{binding_key}: expected nonempty array")
    indexes: set[int] = set()
    for index, record in enumerate(values):
        item_field = f"{field}.{binding_key}[{index}]"
        require(
            isinstance(record, dict) and set(record) == EXECUTED_FILE_KEYS,
            f"{item_field}: key set mismatch",
        )
        argv_index = require_int(
            record["argv_index"], f"{item_field}.argv_index")
        require(argv_index < len(argv), f"{item_field}: argv index out of range")
        require(argv_index not in indexes, f"{item_field}: duplicate argv index")
        indexes.add(argv_index)
        executed_path = require_string(
            record["executed_path"], f"{item_field}.executed_path")
        require(executed_path == argv[argv_index],
                f"{item_field}: executed path mismatch")
        require(Path(executed_path).is_absolute(),
                f"{item_field}: executed path is not absolute")
        captured = _validate_file_binding(
            root, record, item_field, "captured_path", 1)
        current = Path(executed_path)
        current_stat = current.stat(follow_symlinks=False)
        require(
            current.is_file() and not current.is_symlink(),
            f"{item_field}: executed file is missing")
        require(
            (
                current_stat.st_dev,
                current_stat.st_ino,
                current_stat.st_size,
                current_stat.st_mtime_ns,
                current_stat.st_ctime_ns,
            ) == (
                require_int(
                    record["source_device"],
                    f"{item_field}.source_device",
                ),
                require_int(
                    record["source_inode"],
                    f"{item_field}.source_inode",
                    1,
                ),
                require_int(
                    record["source_size"],
                    f"{item_field}.source_size",
                    1,
                ),
                require_int(
                    record["source_mtime_ns"],
                    f"{item_field}.source_mtime_ns",
                    1,
                ),
                require_int(
                    record["source_ctime_ns"],
                    f"{item_field}.source_ctime_ns",
                    1,
                ),
            )
            and current_stat.st_size == captured.stat().st_size
            and digest_file(current) == digest_file(captured),
            f"{item_field}: executed file differs from captured bytes",
        )
    separately_bound_indexes = {
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
    required_indexes = {
        index for index, argument in enumerate(argv)
        if index not in separately_bound_indexes
        and Path(argument).is_absolute()
        and Path(argument).is_file()
    }
    require(0 in indexes, f"{field}: bridge executable is not bound")
    require(indexes == required_indexes,
            f"{field}: executed file binding set mismatch")


def _validate_executor_bindings(
        root: Path,
        values: Any,
        runtime: dict[str, Any],
        artifact_roles: dict[str, Path]) -> dict[str, dict[str, Any]]:
    require(isinstance(values, list), "executor_bindings: expected array")
    runtime_by_id = {
        executor["executor_id"]: executor
        for executor in runtime["executors"]
    }
    require(len(values) == len(runtime_by_id),
            "executor_bindings: count mismatch")
    seen: set[str] = set()
    summaries = {}
    for index, binding in enumerate(values):
        field = f"executor_binding[{index}]"
        require(
            isinstance(binding, dict)
            and set(binding) == EXECUTOR_BINDING_KEYS,
            f"{field}: key set mismatch",
        )
        executor_id = require_string(
            binding["executor_id"], f"{field}.executor_id")
        require(
            executor_id in runtime_by_id and executor_id not in seen,
            f"{field}: unknown or duplicate executor",
        )
        seen.add(executor_id)
        expected_roles = {
            "command_role": f"executor_command::{executor_id}",
            "controller_binding_role":
                f"executor_controller_binding::{executor_id}",
            "gateway_argv_role":
                f"executor_gateway_argv::{executor_id}",
            "gateway_config_role": f"gateway_config::{executor_id}",
            "gateway_stderr_role": f"gateway_stderr::{executor_id}",
            "gateway_stdout_role": f"gateway_stdout::{executor_id}",
            "transport_descriptor_role":
                f"executor_transport::{executor_id}",
        }
        for name, expected in expected_roles.items():
            require(binding[name] == expected,
                    f"{field}.{name}: role mismatch")
            require(expected in artifact_roles,
                    f"{field}.{name}: missing artifact")
        gateway_argv = binding["gateway_argv"]
        require(
            isinstance(gateway_argv, list)
            and gateway_argv
            and all(isinstance(item, str) and item for item in gateway_argv),
            f"{field}.gateway_argv: invalid command array",
        )
        _validate_executed_files(
            root,
            gateway_argv,
            binding,
            field,
            "gateway_executed_files",
        )
        gateway_environment = _validate_launch_environment(
            binding["gateway_launch_environment"],
            f"{field}.gateway_launch_environment",
            privileged=runtime_by_id[executor_id]["role"] != "PHONE",
        )
        prepublication_identity = _validate_gateway_process_identity(
            binding["gateway_prepublication_identity"],
            gateway_argv,
            f"{field}.gateway_prepublication_identity",
        )
        post_auth_identity = _validate_gateway_process_identity(
            binding["gateway_post_auth_identity"],
            gateway_argv,
            f"{field}.gateway_post_auth_identity",
            prepublication_identity,
        )
        final_identity = _validate_gateway_process_identity(
            binding["gateway_final_identity"],
            gateway_argv,
            f"{field}.gateway_final_identity",
            post_auth_identity,
        )
        executable_rows = [
            row for row in binding["gateway_executed_files"]
            if row["argv_index"] == 0
        ]
        require(
            len(executable_rows) == 1
            and prepublication_identity["executable_sha256"]
            == executable_rows[0]["sha256"]
            and (
                prepublication_identity["executable_device"],
                prepublication_identity["executable_inode"],
                prepublication_identity["executable_size"],
                prepublication_identity["executable_mtime_ns"],
                prepublication_identity["executable_ctime_ns"],
            ) == (
                executable_rows[0]["source_device"],
                executable_rows[0]["source_inode"],
                executable_rows[0]["source_size"],
                executable_rows[0]["source_mtime_ns"],
                executable_rows[0]["source_ctime_ns"],
            ),
            f"{field}: live executable is not the prelaunch file",
        )
        gateway_argv_record = read_json(
            artifact_roles[binding["gateway_argv_role"]],
            f"{field}.gateway_argv",
        )
        require(
            gateway_argv_record == {
                "argv": gateway_argv,
                "schema": "s40-gateway-argv-v4",
            },
            f"{field}: bound gateway argv mismatch",
        )
        require(
            Path(_flag_value(
                gateway_argv,
                "--controller-identity",
                f"{field}.gateway_argv",
            )).resolve() == artifact_roles["controller_identity"].resolve()
            and Path(_flag_value(
                gateway_argv,
                "--controller-binding-evidence",
                f"{field}.gateway_argv",
            )).resolve()
            == artifact_roles[
                binding["controller_binding_role"]].resolve(),
            f"{field}: controller authentication argv mismatch",
        )
        config_role = binding["gateway_config_role"]
        config_path = artifact_roles[config_role]
        config_positions = []
        for flag in ("--config", "--route-config"):
            config_positions.extend(
                index + 1 for index, item in enumerate(gateway_argv[:-1])
                if item == flag)
        require(
            len(config_positions) == 1
            and Path(gateway_argv[config_positions[0]]).resolve()
            == config_path.resolve(),
            f"{field}: gateway config argv mismatch",
        )
        executor = runtime_by_id[executor_id]
        require(
            executor["transport"] == "UNIX_SOCKET",
            f"{field}: native Unix transport is required",
        )
        socket_path = Path(executor["socket_path"])
        kind = "phone" if executor["role"] == "PHONE" else "desktop"
        expected_source_role = (
            "executor_bundle::phone_gateway.py"
            if kind == "phone"
            else "executor_bundle::desktop_gateway.py"
        )
        require(
            binding["gateway_source_role"] == expected_source_role
            and expected_source_role in artifact_roles,
            f"{field}: gateway source role mismatch",
        )
        descriptor = read_json(
            artifact_roles[binding["transport_descriptor_role"]],
            f"{field}.transport_descriptor",
        )
        require(
            isinstance(descriptor, dict)
            and set(descriptor) == TRANSPORT_DESCRIPTOR_V4_KEYS
            and descriptor["schema"]
            == "s40-executor-transport-descriptor-v4"
            and descriptor["gateway_environment"] == gateway_environment
            and descriptor["gateway_executed_files"]
            == binding["gateway_executed_files"]
            and descriptor["gateway_prepublication_identity"]
            == prepublication_identity
            and descriptor["gateway_post_auth_identity"]
            == post_auth_identity,
            f"{field}: transport gateway provenance mismatch",
        )
        summary = validate_executor_bundle(
            kind=kind,
            config_path=config_path,
            command_path=artifact_roles[binding["command_role"]],
            executor_id=executor_id,
            transport_descriptor_path=artifact_roles[
                binding["transport_descriptor_role"]],
            gateway_argv_path=artifact_roles[
                binding["gateway_argv_role"]],
            gateway_source_path=artifact_roles[
                binding["gateway_source_role"]],
            executor_bundle_manifest_path=artifact_roles[
                "executor_bundle_manifest"],
            socket_path=socket_path,
            gateway_stdout_path=artifact_roles[
                binding["gateway_stdout_role"]],
            gateway_stderr_path=artifact_roles[
                binding["gateway_stderr_role"]],
            wire_path=(
                artifact_roles[f"executor_wire::{executor_id}"]
                if kind == "phone" else None
            ),
            route_path=(
                artifact_roles[f"executor_route::{executor_id}"]
                if kind == "phone" else None
            ),
        )
        require(
            summary["executor_instance_id"]
            == executor["executor_instance_id"]
            and summary["gateway_pid"] == executor["expected_peer_pid"]
            and summary["gateway_start_time_ticks"]
            == executor["expected_peer_start_time_ticks"],
            f"{field}: runtime peer identity mismatch",
        )
        summary["gateway_final_identity"] = final_identity
        summary["gateway_launch_environment"] = gateway_environment
        summary["gateway_post_auth_identity"] = post_auth_identity
        summary["gateway_prepublication_identity"] = prepublication_identity
        summaries[executor_id] = summary
    require(seen == set(runtime_by_id),
            "executor_bindings: executor set mismatch")
    return summaries


def _validate_command_bijection(
        summaries: dict[str, dict[str, Any]],
        controller_commands: Any,
        run_id: str,
        runtime_config_sha256: str) -> None:
    require(
        isinstance(controller_commands, list),
        "controller commands: expected array",
    )
    normalized_controller: list[dict[str, Any]] = []
    for index, row in enumerate(controller_commands):
        field = f"controller command[{index}]"
        require(isinstance(row, dict), f"{field}: expected object")
        require(
            set(row) == {
                "command_id",
                "command_kind",
                "controller_epoch",
                "disposition",
                "executor_id",
                "model_id",
                "publications",
                "request_complete",
                "request_id",
                "success",
            },
            f"{field}: key set mismatch",
        )
        require(
            row["disposition"] == "RECEIVED",
            f"{field}: result was not received on the live frontier",
        )
        normalized_controller.append({
            "command_id": row["command_id"],
            "command_kind": row["command_kind"],
            "controller_epoch": row["controller_epoch"],
            "executor_id": row["executor_id"],
            "model_id": row["model_id"],
            "publications": row["publications"],
            "request_complete": row["request_complete"],
            "request_id": row["request_id"],
            "success": row["success"],
        })
    executor_commands: list[dict[str, Any]] = []
    seen: set[int] = set()
    for executor_id, summary in summaries.items():
        require(
            summary.get("run_id") == run_id
            and summary.get("runtime_config_sha256")
            == runtime_config_sha256,
            f"executor {executor_id}: run identity mismatch",
        )
        lineage = summary.get("command_lineage")
        require(
            isinstance(lineage, list),
            f"executor {executor_id}: missing command lineage",
        )
        for index, row in enumerate(lineage):
            field = f"executor {executor_id} command[{index}]"
            require(isinstance(row, dict), f"{field}: expected object")
            require(
                set(row) == {
                    "command",
                    "command_id",
                    "controller_epoch",
                    "executor_id",
                    "kind",
                    "model_id",
                    "publications",
                    "request_complete",
                    "request_id",
                    "success",
                },
                f"{field}: key set mismatch",
            )
            command_id = require_int(row["command_id"], f"{field}.command_id", 1)
            require(command_id not in seen, f"{field}: duplicate command")
            seen.add(command_id)
            require(
                row["executor_id"] == executor_id
                and isinstance(row["command"], dict)
                and row["command"].get("executor_instance_id")
                == summary["executor_instance_id"],
                f"{field}: executor mismatch",
            )
            executor_commands.append({
                "command_id": command_id,
                "command_kind": require_int(
                    row["kind"], f"{field}.kind"),
                "controller_epoch": require_int(
                    row["controller_epoch"],
                    f"{field}.controller_epoch",
                ),
                "executor_id": executor_id,
                "model_id": row["model_id"],
                "publications": row["publications"],
                "request_complete": row["request_complete"],
                "request_id": row["request_id"],
                "success": row["success"],
            })
    require(
        sorted(executor_commands, key=lambda row: row["command_id"])
        == normalized_controller,
        "executor/controller command lineage mismatch",
    )


def _decode_http_body(value: Any, digest: Any, field: str) -> bytes:
    require(isinstance(value, str), f"{field}.base64: expected string")
    text = value
    try:
        raw = base64.b64decode(text, validate=True)
    except (ValueError, TypeError) as error:
        raise EvidenceError(f"{field}: invalid base64") from error
    require(
        hashlib.sha256(raw).hexdigest()
        == validate_digest(digest, f"{field}.sha256"),
        f"{field}: digest mismatch",
    )
    return raw


def _validate_activation_evidence(
        path: Path,
        run_id: str) -> dict[str, Any]:
    rows = read_jsonl(path, "activation_evidence")
    require(len(rows) >= 3, "activation_evidence: incomplete exchange")
    post_count = 0
    post_started_ns = None
    seen_post = False
    previous_end = -1
    previous_epoch = 0
    responses = []
    for index, row in enumerate(rows):
        field = f"activation_evidence[{index}]"
        require(
            isinstance(row, dict) and set(row) == ACTIVATION_KEYS,
            f"{field}: key set mismatch",
        )
        require(
            row["schema"] == "s40-activation-http-evidence-v1"
            and row["run_id"] == run_id
            and row["sequence"] == index
            and row["path"] == "/experimental/warm-tier/activate"
            and row["http_status"] == 200,
            f"{field}: identity mismatch",
        )
        started_ns = require_int(row["t_start_ns"], f"{field}.t_start_ns")
        ended_ns = require_int(row["t_end_ns"], f"{field}.t_end_ns")
        require(
            previous_end <= started_ns <= ended_ns,
            f"{field}: invalid interval",
        )
        previous_end = ended_ns
        request_raw = _decode_http_body(
            row["request_body_base64"],
            row["request_body_sha256"],
            f"{field}.request",
        )
        response_raw = _decode_http_body(
            row["response_body_base64"],
            row["response_body_sha256"],
            f"{field}.response",
        )
        response = parse_json(response_raw, f"{field}.response")
        require(
            isinstance(response, dict)
            and set(response) == {"controller_epoch", "schema", "state"},
            f"{field}: response key set mismatch",
        )
        epoch = require_int(
            response["controller_epoch"], f"{field}.controller_epoch")
        require(epoch >= previous_epoch, f"{field}: epoch regressed")
        previous_epoch = epoch
        method = row["method"]
        if method == "POST":
            post_count += 1
            seen_post = True
            post_started_ns = started_ns
            require(
                request_raw == canonical_bytes({
                    "schema": "llama-server-warm-tier-activate-v1",
                })
                and response["schema"]
                == "llama-server-warm-tier-activate-result-v1"
                and response["state"] in {"PREPARING", "READY"},
                f"{field}: invalid activation request/result",
            )
        else:
            require(
                method == "GET"
                and request_raw == b""
                and response["schema"]
                == "llama-server-warm-tier-activate-status-v1",
                f"{field}: invalid status exchange",
            )
            if not seen_post:
                require(
                    index == 0 and response["state"] == "WAITING",
                    f"{field}: initial state is not WAITING",
                )
            else:
                require(
                    response["state"] in {"PREPARING", "READY"},
                    f"{field}: invalid activation state",
                )
        responses.append(response)
    require(
        post_count == 1 and post_started_ns is not None,
        "activation_evidence: POST count mismatch",
    )
    require(
        rows[-1]["method"] == "GET"
        and responses[-1]["state"] == "READY",
        "activation_evidence: missing final READY",
    )
    return {
        "controller_epoch": responses[-1]["controller_epoch"],
        "post_started_ns": post_started_ns,
        "ready_ns": rows[-1]["t_end_ns"],
        "started_ns": rows[0]["t_start_ns"],
    }


def _validate_orchestrator_evidence(
        path: Path,
        run_id: str,
        expected_cleanup: list[str],
        launch_stopped_ns: int) -> None:
    value = read_json(path, "orchestrator_evidence")
    require(
        path.read_bytes() == canonical_bytes(value),
        "orchestrator_evidence: not canonical JSON",
    )
    require(
        set(value) == ORCHESTRATOR_KEYS
        and value["schema"] == "s40-orchestrator-evidence-v2"
        and value["run_id"] == run_id,
        "orchestrator_evidence: identity mismatch",
    )
    require(
        require_int(
            value["acquisition_returncode"],
            "orchestrator_evidence.acquisition_returncode",
        ) == 0,
        "orchestrator_evidence: acquisition failed",
    )
    completed_ns = require_int(
        value["completed_ns"],
        "orchestrator_evidence.completed_ns",
    )
    require(
        completed_ns >= launch_stopped_ns,
        "orchestrator_evidence: cleanup completed before launch stop",
    )
    cleanup = value["cleanup"]
    require(
        isinstance(cleanup, list) and len(cleanup) == len(expected_cleanup),
        "orchestrator_evidence: cleanup count mismatch",
    )
    observed = []
    for index, row in enumerate(cleanup):
        field = f"orchestrator_evidence.cleanup[{index}]"
        require(
            isinstance(row, dict) and set(row) == ORCHESTRATOR_CLEANUP_KEYS,
            f"{field}: key set mismatch",
        )
        observed.append(require_string(row["name"], f"{field}.name"))
        require(
            require_int(row["exit_code"], f"{field}.exit_code") == 0,
            f"{field}: process did not exit cleanly",
        )
    require(
        observed == expected_cleanup,
        "orchestrator_evidence: cleanup order or process set mismatch",
    )


def _validate_source_preflight(
        path: Path,
        root: Path,
        contract_path: Path,
        manifest: dict[str, Any]) -> dict[str, Any]:
    value = read_json(path, "source_preflight")
    require(
        path.read_bytes() == canonical_bytes(value),
        "source_preflight: not canonical JSON",
    )
    require(
        set(value) == SOURCE_PREFLIGHT_KEYS
        and value["schema"] == "s40-source-preflight-v1",
        "source_preflight: identity mismatch",
    )
    captured_ns = require_int(
        value["captured_ns"], "source_preflight.captured_ns")
    rows = value["rows"]
    require(isinstance(rows, list) and rows,
            "source_preflight.rows: expected nonempty array")
    by_role: dict[str, tuple[dict[str, Any], Path]] = {}
    for index, row in enumerate(rows):
        field = f"source_preflight.rows[{index}]"
        require(
            isinstance(row, dict)
            and set(row) == SOURCE_PREFLIGHT_ROW_KEYS,
            f"{field}: key set mismatch",
        )
        role = require_string(row["role"], f"{field}.role")
        require(role not in by_role, f"{field}: duplicate role")
        captured = _validate_file_binding(
            root,
            row,
            field,
            path_key="captured_path",
            minimum_bytes=1,
        )
        source = Path(require_string(
            row["source_path"], f"{field}.source_path"))
        require(source.is_absolute() and source.is_file(),
                f"{field}: source is missing")
        require(
            source.stat().st_size == row["bytes"]
            and digest_file(source) == row["sha256"],
            f"{field}: source changed after preflight",
        )
        by_role[role] = (row, captured)

    required = {
        "physical_plan",
        "experiment_contract",
        "runtime_plan_template",
        "evidence_root",
        "runtime_binary::controller",
        "runtime_binary::native_bench",
        "runtime_binary::nvidia_smi",
        "runtime_binary::python",
        "runtime_tool::ldd",
    }
    require(required <= set(by_role),
            "source_preflight: required source is missing")
    contract_row, contract_capture = by_role["experiment_contract"]
    require(
        Path(contract_row["source_path"]).resolve()
        == contract_path.resolve()
        and contract_row["sha256"] == digest_file(contract_path)
        == manifest["experiment_contract_sha256"],
        "source_preflight: experiment contract mismatch",
    )
    physical_row, physical_capture = by_role["physical_plan"]
    physical = read_json(physical_capture, "source_preflight.physical_plan")
    require(
        physical_capture.read_bytes() == canonical_bytes(physical)
        and set(physical) == SOURCE_PHYSICAL_PLAN_KEYS
        and physical["schema"] == "s40-physical-run-plan-v2",
        "source_preflight: physical plan mismatch",
    )
    require(
        physical["run_id"] == manifest["run_id"]
        and physical["mode"] == manifest["mode"]
        and physical["cache_regime"] == manifest["cache_regime"]
        and physical["development"] == manifest["development"]
        and physical["repeat_index"] == manifest["repeat_index"]
        and len(physical["server_argv"]) == len(manifest["command_argv"])
        and physical["server_argv"][1:] == manifest["command_argv"][1:]
        and physical["devices"] == manifest["devices"]
        and Path(physical["contract_path"]).resolve()
        == contract_path.resolve()
        and Path(physical["runtime_plan_template"]).resolve()
        == Path(by_role["runtime_plan_template"][0]["source_path"]).resolve(),
        "source_preflight: physical plan identity mismatch",
    )
    campaign_binding = manifest["campaign_binding"]
    source_campaign_binding = physical["campaign_binding"]
    campaign = None
    if manifest["development"]:
        require(
            campaign_binding is None
            and source_campaign_binding is None
            and "campaign_plan" not in by_role,
            "source_preflight: development run has campaign evidence",
        )
    else:
        require(
            isinstance(campaign_binding, dict)
            and set(campaign_binding) == MANIFEST_CAMPAIGN_BINDING_KEYS
            and isinstance(source_campaign_binding, dict)
            and set(source_campaign_binding)
            == SOURCE_CAMPAIGN_BINDING_KEYS
            and "campaign_plan" in by_role,
            "source_preflight: primary campaign binding is missing",
        )
        campaign_row, campaign_capture = by_role["campaign_plan"]
        campaign = read_json(
            campaign_capture, "source_preflight.campaign_plan")
        validate_campaign(campaign)
        validate_campaign_tool_lock(campaign["software_lock"])
        require(
            campaign_capture.read_bytes() == canonical_bytes(campaign)
            and campaign_row["sha256"]
            == source_campaign_binding["campaign_sha256"]
            == campaign_binding["campaign_sha256"]
            and Path(campaign_row["source_path"]).resolve()
            == Path(source_campaign_binding["campaign_path"]).resolve()
            and campaign_binding["campaign_id"] == campaign["campaign_id"]
            and campaign["experiment_contract_sha256"]
            == manifest["experiment_contract_sha256"],
            "source_preflight: campaign identity mismatch",
        )
        order = require_int(
            campaign_binding["order"], "campaign_binding.order")
        require(
            source_campaign_binding["order"] == order
            and source_campaign_binding["phase"]
            == campaign_binding["phase"],
            "source_preflight: campaign binding mismatch",
        )
        rows = [
            row for row in campaign["primary"]
            if row["order"] == order
        ]
        require(
            len(rows) == 1
            and rows[0]["run_id"] == manifest["run_id"]
            and rows[0]["mode"] == manifest["mode"]
            and rows[0]["cache_regime"] == manifest["cache_regime"]
            and rows[0]["repeat_index"] == manifest["repeat_index"]
            and rows[0]["phase"] == campaign_binding["phase"],
            "source_preflight: campaign row mismatch",
        )
    controller_source = by_role["runtime_binary::controller"][0]
    native_source = by_role["runtime_binary::native_bench"][0]
    require(
        Path(controller_source["source_path"]).resolve()
        == Path(physical["server_argv"][0]).resolve()
        and Path(native_source["source_path"]).resolve()
        == (
            Path(physical["server_argv"][0]).resolve().parent
            / "test-warm-tier-executors"
        ).resolve(),
        "source_preflight: runtime binary source mismatch",
    )
    nvidia_source = by_role["runtime_binary::nvidia_smi"][0]
    require(
        Path(nvidia_source["source_path"]).resolve()
        == Path(physical["nvidia_smi_path"]).resolve(),
        "source_preflight: nvidia-smi source mismatch",
    )
    ldd_source = by_role["runtime_tool::ldd"][0]
    require(
        Path(ldd_source["source_path"]).resolve()
        == Path(physical["ldd_path"]).resolve(),
        "source_preflight: ldd source mismatch",
    )
    runtime_row, runtime_capture = by_role["runtime_plan_template"]
    source_plan = read_json(
        runtime_capture, "source_preflight.runtime_plan_template")
    require(
        runtime_capture.read_bytes() == canonical_bytes(source_plan)
        and source_plan.get("run_id") == manifest["run_id"]
        and source_plan.get("mode") == manifest["mode"],
        "source_preflight: runtime plan identity mismatch",
    )
    root_row, root_capture = by_role["evidence_root"]
    source_root = read_json(root_capture, "source_preflight.evidence_root")
    require(
        root_capture.read_bytes() == canonical_bytes(source_root)
        and Path(source_plan["evidence_root_path"]).resolve()
        == Path(root_row["source_path"]).resolve()
        and source_plan["evidence_root_sha256"] == root_row["sha256"],
        "source_preflight: evidence root mismatch",
    )
    expected = set(required)
    if campaign is not None:
        expected.add("campaign_plan")
    profile_path = source_plan.get("c3_profile_lock_path")
    profile_sha256 = source_plan.get("c3_profile_lock_sha256")
    if profile_path is not None:
        expected.add("c3_profile_lock")
        require(
            "c3_profile_lock" in by_role
            and Path(by_role["c3_profile_lock"][0]["source_path"]).resolve()
            == Path(profile_path).resolve()
            and by_role["c3_profile_lock"][0]["sha256"] == profile_sha256,
            "source_preflight: C3 profile lock mismatch",
        )
    else:
        require(profile_sha256 is None,
                "source_preflight: unexpected C3 profile digest")
    placements = source_root.get("c3_placement_artifacts")
    require(isinstance(placements, dict),
            "source_preflight: invalid C3 placement map")
    for model_id, record in placements.items():
        role = f"c3_placement::{model_id}"
        expected.add(role)
        require(
            role in by_role
            and Path(by_role[role][0]["source_path"]).resolve()
            == Path(record["path"]).resolve()
            and by_role[role][0]["sha256"] == record["sha256"],
            "source_preflight: C3 placement mismatch",
        )
    configs = source_root.get("executor_configs")
    require(isinstance(configs, list),
            "source_preflight: invalid executor config list")
    for record in configs:
        executor_id = require_string(
            record.get("executor_id"),
            "source_preflight.executor_id",
        )
        config_role = f"gateway_config::{executor_id}"
        expected.add(config_role)
        require(
            config_role in by_role
            and Path(by_role[config_role][0]["source_path"]).resolve()
            == Path(record["gateway_config_path"]).resolve()
            and by_role[config_role][0]["sha256"]
            == record["gateway_config_sha256"],
            "source_preflight: gateway config mismatch",
        )
        config = read_json(
            by_role[config_role][1],
            f"source_preflight.{config_role}",
        )
        schema = config.get("schema")
        require(
            schema in {
                "s40-desktop-executor-config-v4",
                "s40-phone-route-config-v3",
            },
            "source_preflight: acquisition config schema mismatch",
        )
        routes = config.get("routes", [])
        require(isinstance(routes, list),
                "source_preflight: executor routes must be an array")
        for route in routes:
            require(isinstance(route, dict),
                    "source_preflight: executor route must be an object")
            model_id = require_string(
                route.get("model_id"),
                "source_preflight.executor.model_id",
            )
            if schema.startswith("s40-desktop-executor-config-"):
                for prefix, path_key, digest_key in (
                        (
                            "artifact_certificate",
                            "artifact_certificate_path",
                            "artifact_certificate_sha256",
                        ),
                        (
                            "readiness_lock",
                            "readiness_lock_path",
                            "readiness_lock_sha256",
                        )):
                    role = f"{prefix}::{executor_id}::{model_id}"
                    expected.add(role)
                    require(
                        role in by_role
                        and Path(
                            by_role[role][0]["source_path"]).resolve()
                        == Path(route[path_key]).resolve()
                        and by_role[role][0]["sha256"]
                        == route[digest_key],
                        f"source_preflight: {prefix} mismatch",
                    )
            if schema in {
                    "s40-desktop-executor-config-v4",
                    "s40-phone-route-config-v3"}:
                qualification = route.get("qualification")
                validate_qualification_route_identity(
                    qualification,
                    route,
                    executor_id,
                )
                bindings = qualification_source_records(
                    qualification, executor_id, model_id)
                for binding in bindings:
                    role = binding["role"]
                    require(
                        role not in expected,
                        "source_preflight: duplicate qualification role",
                    )
                    expected.add(role)
                    require(
                        role in by_role
                        and Path(
                            by_role[role][0]["source_path"]).resolve()
                        == Path(binding["path"]).resolve()
                        and by_role[role][0]["sha256"]
                        == binding["sha256"],
                        "source_preflight: qualification binding mismatch",
                    )
    if physical["phone_identity_argv"] is not None:
        expected.add("phone_ssh_config")
        require(
            "phone_ssh_config" in by_role,
            "source_preflight: phone SSH config is missing",
        )
        identity_argv = physical["phone_identity_argv"]
        telemetry_argv = physical["phone_telemetry_argv"]
        identity_positions = [
            index for index, item in enumerate(identity_argv)
            if item == "--ssh-config"
        ]
        telemetry_positions = [
            index for index, item in enumerate(telemetry_argv)
            if item == "--ssh-config"
        ]
        require(
            len(identity_positions) == len(telemetry_positions) == 1
            and identity_positions[0] + 1 < len(identity_argv)
            and telemetry_positions[0] + 1 < len(telemetry_argv),
            "source_preflight: phone SSH config argument mismatch",
        )
        identity_config = Path(
            identity_argv[identity_positions[0] + 1]).resolve()
        telemetry_config = Path(
            telemetry_argv[telemetry_positions[0] + 1]).resolve()
        require(
            identity_config == telemetry_config
            == Path(
                by_role["phone_ssh_config"][0]["source_path"]).resolve(),
            "source_preflight: phone SSH config mismatch",
        )
    for role in by_role:
        if role.startswith("runtime_dependency::"):
            expected.add(role)
    require(
        set(by_role) == expected,
        "source_preflight: source role set mismatch",
    )
    return {
        "captured_ns": captured_ns,
        "contract_sha256": contract_row["sha256"],
        "campaign": campaign,
        "physical_plan": physical,
        "physical_plan_sha256": physical_row["sha256"],
        "rows": by_role,
        "sha256": digest_file(path),
    }


def _is_system_library(path: Path) -> bool:
    resolved = path.resolve()
    for root in (Path("/lib").resolve(), Path("/usr/lib").resolve()):
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            pass
    return False


def _resolve_binary_dependencies(
        binary: Path,
        label: str,
        ldd_path: Path,
        ldd_sha256: str) -> list[dict[str, Any]]:
    raw = binary.read_bytes()
    if not raw.startswith(b"\x7fELF"):
        return []
    require(
        ldd_path.is_absolute()
        and ldd_path.is_file()
        and os.access(ldd_path, os.X_OK)
        and digest_file(ldd_path) == ldd_sha256,
        "runtime_dependencies: ldd changed before execution",
    )
    result = subprocess.run(
        [str(ldd_path), str(binary.resolve())],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        },
        timeout=30,
    )
    require(
        digest_file(ldd_path) == ldd_sha256,
        "runtime_dependencies: ldd changed during execution",
    )
    require(
        result.returncode == 0 and not result.stderr,
        f"runtime_dependencies: ldd failed for {label}",
    )
    try:
        text = result.stdout.decode("ascii")
    except UnicodeDecodeError as error:
        raise EvidenceError(
            f"runtime_dependencies: ldd output is not ASCII for {label}"
        ) from error
    rows = []
    seen = set()
    for line in text.splitlines():
        value = line.strip()
        if not value or value.startswith("linux-vdso"):
            continue
        if "=>" in value:
            needed_name, resolution = (
                part.strip() for part in value.split("=>", 1))
            require(
                resolution != "not found",
                f"runtime_dependencies: unresolved {needed_name}",
            )
            resolved_text = LDD_ADDRESS.sub("", resolution)
        else:
            resolved_text = LDD_ADDRESS.sub("", value)
            needed_name = Path(resolved_text).name
        resolved = Path(resolved_text).resolve()
        require(
            needed_name
            and Path(needed_name).name == needed_name
            and resolved.is_file(),
            f"runtime_dependencies: invalid dependency for {label}",
        )
        key = (needed_name, str(resolved))
        require(
            key not in seen,
            f"runtime_dependencies: duplicate dependency for {label}",
        )
        seen.add(key)
        rows.append({
            "bytes": resolved.stat().st_size,
            "consumer": label,
            "needed_name": needed_name,
            "sha256": digest_file(resolved),
            "source_path": str(resolved),
            "system": _is_system_library(resolved),
        })
    return rows


def _validate_runtime_dependencies(
        path: Path,
        root: Path,
        source_preflight: dict[str, Any],
        artifact_roles: dict[str, Path]) -> dict[str, Any]:
    value = read_json(path, "runtime_dependencies")
    require(
        path.read_bytes() == canonical_bytes(value)
        and isinstance(value, dict)
        and set(value)
        == {"binaries", "dependencies", "library_path", "schema"}
        and value["schema"] == "s40-captured-runtime-v1",
        "runtime_dependencies: schema mismatch",
    )
    library_path = Path(require_string(
        value["library_path"], "runtime_dependencies.library_path"))
    require(
        library_path.is_absolute()
        and library_path.is_dir(),
        "runtime_dependencies: private library path is missing",
    )
    try:
        library_path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise EvidenceError(
            "runtime_dependencies: private library path escapes run"
        ) from error
    binaries = value["binaries"]
    require(
        isinstance(binaries, list) and len(binaries) == 4,
        "runtime_dependencies: binary set mismatch",
    )
    binary_paths = {}
    source_rows = source_preflight["rows"]
    ldd_row = source_rows["runtime_tool::ldd"][0]
    ldd_path = Path(ldd_row["source_path"])
    for index, record in enumerate(binaries):
        field = f"runtime_dependencies.binaries[{index}]"
        require(
            isinstance(record, dict)
            and set(record)
            == {
                "bytes",
                "captured_path",
                "label",
                "sha256",
                "source_path",
            },
            f"{field}: key set mismatch",
        )
        label = require_string(record["label"], f"{field}.label")
        require(
            label in {"controller", "native_bench", "nvidia_smi", "python"}
            and label not in binary_paths,
            f"{field}: unknown or duplicate label",
        )
        captured = _validate_file_binding(
            root, record, field, path_key="captured_path", minimum_bytes=1)
        source_role = f"runtime_binary::{label}"
        require(source_role in source_rows, f"{field}: source lock missing")
        source_row = source_rows[source_role][0]
        require(
            Path(record["source_path"]).resolve()
            == Path(source_row["source_path"]).resolve()
            and record["sha256"] == source_row["sha256"],
            f"{field}: source lock mismatch",
        )
        binary_paths[label] = captured
    require(
        set(binary_paths)
        == {"controller", "native_bench", "nvidia_smi", "python"},
        "runtime_dependencies: binary role set mismatch",
    )

    observed: dict[tuple[str, str], dict[str, Any]] = {}
    for label in sorted(binary_paths):
        source = Path(
            source_rows[f"runtime_binary::{label}"][0]["source_path"])
        for record in _resolve_binary_dependencies(
                source, label, ldd_path, ldd_row["sha256"]):
            key = (record["needed_name"], record["source_path"])
            current = observed.get(key)
            if current is None:
                observed[key] = {
                    "bytes": record["bytes"],
                    "consumers": [label],
                    "needed_name": record["needed_name"],
                    "sha256": record["sha256"],
                    "source_path": record["source_path"],
                    "system": record["system"],
                }
            else:
                current["consumers"].append(label)
    for record in observed.values():
        record["consumers"].sort()
    by_name = {}
    for record in observed.values():
        previous = by_name.get(record["needed_name"])
        require(
            previous is None
            or (
                previous["source_path"] == record["source_path"]
                and previous["sha256"] == record["sha256"]
            ),
            "runtime_dependencies: ambiguous dependency name",
        )
        by_name[record["needed_name"]] = record
    expected = [
        by_name[name] for name in sorted(by_name)
    ]
    dependencies = value["dependencies"]
    require(
        isinstance(dependencies, list)
        and len(dependencies) == len(expected),
        "runtime_dependencies: dependency count mismatch",
    )
    dependency_roles = set()
    private_names = set()
    normalized = []
    for index, record in enumerate(dependencies):
        field = f"runtime_dependencies.dependencies[{index}]"
        require(
            isinstance(record, dict)
            and set(record)
            == {
                "bytes",
                "captured_path",
                "consumers",
                "needed_name",
                "sha256",
                "source_path",
                "system",
            },
            f"{field}: key set mismatch",
        )
        base = {key: value for key, value in record.items()
                if key != "captured_path"}
        normalized.append(base)
        name = require_string(record["needed_name"], f"{field}.needed_name")
        require(
            type(record["system"]) is bool,
            f"{field}.system: expected bool",
        )
        if record["system"]:
            require(
                record["captured_path"] is None,
                f"{field}: system dependency was privately captured",
            )
            source = Path(record["source_path"])
            require(
                source.is_file()
                and source.stat().st_size == record["bytes"]
                and digest_file(source) == record["sha256"],
                f"{field}: system dependency changed",
            )
            continue
        role = f"runtime_dependency::{name}"
        dependency_roles.add(role)
        private_names.add(name)
        require(
            role in source_rows and role in artifact_roles,
            f"{field}: private dependency binding missing",
        )
        captured = _validate_file_binding(
            root, record, field, path_key="captured_path", minimum_bytes=1)
        require(
            captured.resolve() == artifact_roles[role].resolve()
            and captured.parent.resolve() == library_path.resolve()
            and captured.name == name
            and record["sha256"] == source_rows[role][0]["sha256"],
            f"{field}: private dependency mismatch",
        )
    require(
        normalized == expected,
        "runtime_dependencies: resolved dependency set mismatch",
    )
    require(
        {item.name for item in library_path.iterdir() if item.is_file()}
        == private_names,
        "runtime_dependencies: private library directory mismatch",
    )
    return {
        "binary_paths": binary_paths,
        "dependency_roles": dependency_roles,
        "library_path": library_path,
    }


def _validate_phone_route_coverage(
        executor_summaries: dict[str, dict[str, Any]],
        runtime: dict[str, Any],
        phone_summary: dict[str, Any] | None) -> list[dict[str, str]]:
    phone_ids = {
        executor["executor_id"]
        for executor in runtime["executors"]
        if executor["role"] == "PHONE"
    }
    if not phone_ids:
        require(
            phone_summary is None
            and all(
                summary.get("role") != "PHONE"
                for summary in executor_summaries.values()
            ),
            "phone routes: unexpected phone evidence",
        )
        return []

    require(
        isinstance(phone_summary, dict)
        and phone_summary.get("schema")
        == "s40-phone-observer-summary-v1",
        "phone routes: periodic observer summary is missing",
    )
    require(
        phone_ids == {
            executor_id
            for executor_id, summary in executor_summaries.items()
            if summary.get("role") == "PHONE"
        },
        "phone routes: executor role set mismatch",
    )

    routes: dict[str, dict[str, str]] = {}
    synchronous: dict[str, dict[str, str]] = {}
    for executor_id in sorted(phone_ids):
        summary = executor_summaries[executor_id]
        route_rows = summary.get("route_instances")
        observation_rows = summary.get("synchronous_route_observations")
        require(
            isinstance(route_rows, list)
            and route_rows
            and isinstance(observation_rows, list)
            and observation_rows,
            f"phone routes: {executor_id} has no synchronous route evidence",
        )
        for index, row in enumerate(route_rows):
            field = f"phone routes: {executor_id}.route_instances[{index}]"
            require(
                isinstance(row, dict)
                and set(row)
                == {"model_id", "placements", "route_instance_id"},
                f"{field}: key set mismatch",
            )
            model_id = require_string(row["model_id"], f"{field}.model_id")
            route_id = require_string(
                row["route_instance_id"],
                f"{field}.route_instance_id",
            )
            require(
                isinstance(row["placements"], dict)
                and set(row["placements"]) == {"op12", "op15"},
                f"{field}: terminal placement is missing",
            )
            require(
                route_id not in routes,
                "phone routes: duplicate route instance",
            )
            routes[route_id] = {
                "model_id": model_id,
                "route_instance_id": route_id,
            }
        for index, row in enumerate(observation_rows):
            field = (
                f"phone routes: {executor_id}."
                f"synchronous_route_observations[{index}]"
            )
            require(
                isinstance(row, dict)
                and set(row)
                == {"model_id", "route_instance_id", "schema"}
                and row["schema"] == "s40-phone-route-observation-v1",
                f"{field}: identity mismatch",
            )
            model_id = require_string(row["model_id"], f"{field}.model_id")
            route_id = require_string(
                row["route_instance_id"],
                f"{field}.route_instance_id",
            )
            require(
                route_id not in synchronous,
                "phone routes: duplicate synchronous observation",
            )
            synchronous[route_id] = {
                "model_id": model_id,
                "route_instance_id": route_id,
            }
    require(
        routes == synchronous,
        "phone routes: synchronous observation set mismatch",
    )

    periodic_rows = phone_summary.get("observed_routes")
    require(
        isinstance(periodic_rows, list),
        "phone routes: periodic route set is missing",
    )
    periodic_ids = set()
    for index, row in enumerate(periodic_rows):
        field = f"phone routes: observed_routes[{index}]"
        require(
            isinstance(row, dict)
            and set(row)
            == {"model_id", "route_instance_id", "sample_count"},
            f"{field}: key set mismatch",
        )
        model_id = require_string(row["model_id"], f"{field}.model_id")
        route_id = require_string(
            row["route_instance_id"],
            f"{field}.route_instance_id",
        )
        require(
            require_int(row["sample_count"], f"{field}.sample_count", 1) >= 1
            and route_id not in periodic_ids,
            "phone routes: invalid periodic observation",
        )
        periodic_ids.add(route_id)
        require(
            routes.get(route_id)
            == {
                "model_id": model_id,
                "route_instance_id": route_id,
            },
            "phone routes: periodic observation is not a loaded route",
        )
    return [routes[route_id] for route_id in sorted(routes)]


def _validate_route_qualification_summaries(
        executor_summaries: dict[str, dict[str, Any]],
        expected_models: list[str]) -> list[dict[str, Any]]:
    expected_model_set = set(expected_models)
    common = None
    for executor_id, summary in sorted(executor_summaries.items()):
        rows = summary.get("route_qualifications")
        require(
            isinstance(rows, list) and len(rows) == len(expected_models),
            f"executor {executor_id}: qualification set mismatch",
        )
        by_model = {}
        for index, row in enumerate(rows):
            field = f"executor {executor_id}.qualification[{index}]"
            require(
                isinstance(row, dict)
                and set(row)
                == {
                    "a_chain_phase_id",
                    "bundle_manifest_sha256",
                    "model_id",
                    "phase",
                    "phase_id",
                    "phase_lock_sha256",
                    "schema",
                    "scope",
                    "status",
                    "v2_2_result_sha256",
                }
                and row["schema"]
                == "s40-route-qualification-derived-v1"
                and row["scope"] == "QUALIFIED_ROUTE",
                f"{field}: non-qualifying authority",
            )
            model_id = require_string(row["model_id"], f"{field}.model_id")
            require(
                model_id in expected_model_set and model_id not in by_model,
                f"{field}: unknown or duplicate model",
            )
            phase = row["phase"]
            phase_id = require_string(row["phase_id"], f"{field}.phase_id")
            validate_digest(
                row["bundle_manifest_sha256"],
                f"{field}.bundle_manifest_sha256",
            )
            validate_digest(
                row["phase_lock_sha256"],
                f"{field}.phase_lock_sha256",
            )
            validate_digest(
                row["v2_2_result_sha256"],
                f"{field}.v2_2_result_sha256",
            )
            require(
                (
                    phase == "A_ONLY"
                    and row["status"] == "MODEL_A_QUALIFICATION_PASS"
                    and row["a_chain_phase_id"] is None
                )
                or (
                    phase == "B_ONLY"
                    and row["status"] == "MODEL_B_QUALIFICATION_PASS"
                    and isinstance(row["a_chain_phase_id"], str)
                    and row["a_chain_phase_id"]
                ),
                f"{field}: phase chain mismatch",
            )
            by_model[model_id] = {**row, "phase_id": phase_id}
        require(
            set(by_model) == expected_model_set,
            f"executor {executor_id}: incomplete qualification set",
        )
        a_rows = [
            row for row in by_model.values()
            if row["phase"] == "A_ONLY"
        ]
        b_rows = [
            row for row in by_model.values()
            if row["phase"] == "B_ONLY"
        ]
        require(
            len(a_rows) == len(b_rows) == 1
            and b_rows[0]["a_chain_phase_id"] == a_rows[0]["phase_id"]
            and b_rows[0]["phase_id"] != a_rows[0]["phase_id"],
            f"executor {executor_id}: invalid A/B qualification chain",
        )
        normalized = [by_model[model_id] for model_id in sorted(by_model)]
        require(
            common is None or normalized == common,
            "executor qualification roots differ",
        )
        common = normalized
    require(common is not None, "executor qualification set is empty")
    return common


def _validate_controller_authentication(
        artifact_roles: dict[str, Path],
        readiness: dict[str, Any],
        launch: dict[str, Any],
        runtime: dict[str, Any],
        runtime_path: Path,
        runtime_stat: os.stat_result,
        binary_path: Path,
        runtime_config_published_ns: int,
        controller_started_ns: int,
        base_started_ns: int,
        activation_post_started_ns: int,
        activation_ready_ns: int,
        run_id: str) -> dict[str, Any]:
    identity_path = artifact_roles["controller_identity"]
    identity_stat = identity_path.stat(follow_symlinks=False)
    identity = read_json(identity_path, "controller_identity")
    require(
        identity_path.read_bytes() == canonical_bytes(identity)
        and isinstance(identity, dict)
        and set(identity) == CONTROLLER_IDENTITY_KEYS
        and identity["schema"] == "s40-controller-identity-lock-v1",
        "controller_identity: identity mismatch",
    )
    controller_pid = require_int(
        identity["controller_pid"], "controller_identity.controller_pid", 2)
    controller_start_ticks = require_int(
        identity["controller_start_time_ticks"],
        "controller_identity.controller_start_time_ticks",
        1,
    )
    controller_uid = require_int(
        identity["controller_uid"], "controller_identity.controller_uid")
    controller_gid = require_int(
        identity["controller_gid"], "controller_identity.controller_gid")
    identity_executable_path = Path(require_string(
        identity["controller_executable_path"],
        "controller_identity.controller_executable_path",
    ))
    identity_executable_sha256 = validate_digest(
        identity["controller_executable_sha256"],
        "controller_identity.controller_executable_sha256",
    )
    identity_runtime_path = Path(require_string(
        identity["runtime_config_path"],
        "controller_identity.runtime_config_path",
    ))
    identity_runtime_sha256 = validate_digest(
        identity["runtime_config_sha256"],
        "controller_identity.runtime_config_sha256",
    )
    identity_runtime_device = require_int(
        identity["runtime_config_device"],
        "controller_identity.runtime_config_device",
    )
    identity_runtime_inode = require_int(
        identity["runtime_config_inode"],
        "controller_identity.runtime_config_inode",
        1,
    )
    require(
        identity["run_id"] == run_id
        and identity["host_boot_id"] == readiness["host_boot_id"]
        and identity_executable_path.resolve()
        == binary_path.resolve()
        and identity_executable_sha256 == digest_file(binary_path)
        and identity_runtime_path.resolve()
        == runtime_path.resolve()
        and identity_runtime_sha256 == digest_file(runtime_path)
        and identity_runtime_device == runtime_stat.st_dev
        and identity_runtime_inode == runtime_stat.st_ino,
        "controller_identity: bound identity mismatch",
    )

    publication = readiness["controller_identity"]
    require(
        isinstance(publication, dict)
        and set(publication) == CONTROLLER_PUBLICATION_KEYS,
        "server_readiness.controller_identity: key set mismatch",
    )
    captured_ns = require_int(
        publication["captured_ns"],
        "server_readiness.controller_identity.captured_ns",
        controller_started_ns,
    )
    published_ns = require_int(
        publication["published_ns"],
        "server_readiness.controller_identity.published_ns",
        captured_ns,
    )
    publication_path = Path(require_string(
        publication["path"],
        "server_readiness.controller_identity.path",
    ))
    publication_sha256 = validate_digest(
        publication["sha256"],
        "server_readiness.controller_identity.sha256",
    )
    publication_device = require_int(
        publication["device"],
        "server_readiness.controller_identity.device",
    )
    publication_inode = require_int(
        publication["inode"],
        "server_readiness.controller_identity.inode",
        1,
    )
    require(
        publication_path.resolve() == identity_path.resolve()
        and publication_sha256 == digest_file(identity_path)
        and publication_device == identity_stat.st_dev
        and publication_inode == identity_stat.st_ino
        and runtime_config_published_ns <= controller_started_ns
        <= captured_ns <= published_ns <= base_started_ns,
        "server_readiness.controller_identity: publication mismatch",
    )
    require(
        launch["controller_pid"] == controller_pid
        and launch["controller_start_ticks"] == controller_start_ticks
        and launch["host_boot_id"] == identity["host_boot_id"]
        and Path(require_string(
            launch["controller_identity_path"],
            "controller_launch.controller_identity_path",
        )).resolve()
        == identity_path.resolve()
        and validate_digest(
            launch["controller_identity_sha256"],
            "controller_launch.controller_identity_sha256",
        )
        == digest_file(identity_path)
        and require_int(
            launch["controller_identity_device"],
            "controller_launch.controller_identity_device",
        ) == identity_stat.st_dev
        and require_int(
            launch["controller_identity_inode"],
            "controller_launch.controller_identity_inode",
            1,
        ) == identity_stat.st_ino,
        "controller_launch: controller identity mismatch",
    )

    runtime_by_id = {
        executor["executor_id"]: executor
        for executor in runtime["executors"]
    }
    privileged_environment = _validate_launch_environment(
        launch["launch_environment"],
        "controller_launch.launch_environment",
    )
    base_environment = dict(privileged_environment)
    base_environment.pop(
        "LLAMA_SERVER_WARM_TIER_INTERNAL_TOKEN_FILE",
    )
    _validate_launch_environment(
        base_environment,
        "controller_launch.base_environment",
        privileged=False,
    )
    readiness_rows = readiness["controller_bindings"]
    require(
        isinstance(readiness_rows, list)
        and [
            row.get("executor_id")
            for row in readiness_rows
            if isinstance(row, dict)
        ] == sorted(runtime_by_id),
        "server_readiness.controller_bindings: executor set mismatch",
    )
    readiness_by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(readiness_rows):
        field = f"server_readiness.controller_bindings[{index}]"
        require(
            isinstance(row, dict)
            and set(row) == CONTROLLER_BINDING_READINESS_KEYS,
            f"{field}: key set mismatch",
        )
        executor_id = require_string(
            row["executor_id"], f"{field}.executor_id")
        require(
            executor_id in runtime_by_id
            and executor_id not in readiness_by_id,
            f"{field}: unknown or duplicate executor",
        )
        readiness_by_id[executor_id] = row
    gateway_ready_by_id = {
        row["executor_id"]: row
        for row in readiness["gateway_ready"]
        if isinstance(row, dict) and "executor_id" in row
    }
    gateway_post_auth_by_id = {
        row["executor_id"]: row["identity"]
        for row in readiness["gateway_post_auth"]
        if isinstance(row, dict)
        and set(row) == {"executor_id", "identity"}
    }
    require(
        set(gateway_ready_by_id) == set(runtime_by_id)
        and set(gateway_post_auth_by_id) == set(runtime_by_id),
        "server_readiness: gateway provenance set mismatch",
    )

    bindings = []
    for executor_id in sorted(runtime_by_id):
        executor = runtime_by_id[executor_id]
        role = f"executor_controller_binding::{executor_id}"
        binding_path = artifact_roles[role]
        binding_stat = binding_path.stat(follow_symlinks=False)
        binding = read_json(binding_path, role)
        require(
            binding_path.read_bytes() == canonical_bytes(binding)
            and isinstance(binding, dict)
            and set(binding) == CONTROLLER_BINDING_KEYS
            and binding["schema"] == "s40-executor-controller-binding-v1",
            f"{role}: identity mismatch",
        )
        authenticated_ns = require_int(
            binding["authenticated_ns"],
            f"{role}.authenticated_ns",
            activation_post_started_ns,
        )
        require(
            activation_post_started_ns
            <= authenticated_ns <= activation_ready_ns,
            f"{role}: authentication is outside activation",
        )
        require(
            require_int(
                gateway_post_auth_by_id[executor_id]["observed_ns"],
                f"{role}.gateway_post_auth.observed_ns",
                authenticated_ns,
            ) >= authenticated_ns,
            f"{role}: gateway was not rechecked after authentication",
        )
        command_rows = read_jsonl(
            artifact_roles[f"executor_command::{executor_id}"],
            f"executor commands {executor_id}",
        )
        command_started_ns = [
            require_int(
                row["started_ns"],
                f"executor commands {executor_id}[{index}].started_ns",
                1,
            )
            for index, row in enumerate(command_rows)
            if isinstance(row, dict) and "started_ns" in row
        ]
        require(
            command_started_ns
            and authenticated_ns <= min(command_started_ns),
            f"{role}: command predates controller authentication",
        )
        require(
            require_string(binding["run_id"], f"{role}.run_id") == run_id
            and require_string(
                binding["host_boot_id"], f"{role}.host_boot_id")
            == identity["host_boot_id"]
            and require_string(
                binding["executor_id"], f"{role}.executor_id")
            == executor_id
            and require_string(
                binding["executor_instance_id"],
                f"{role}.executor_instance_id",
            )
            == executor["executor_instance_id"]
            and require_int(
                binding["gateway_pid"], f"{role}.gateway_pid", 2)
            == executor["expected_peer_pid"]
            and require_int(
                binding["gateway_start_time_ticks"],
                f"{role}.gateway_start_time_ticks",
                1,
            ) == executor["expected_peer_start_time_ticks"]
            and require_int(
                binding["peer_pid"], f"{role}.peer_pid", 2)
            == controller_pid
            and require_int(
                binding["peer_uid"], f"{role}.peer_uid")
            == controller_uid
            and require_int(
                binding["peer_gid"], f"{role}.peer_gid")
            == controller_gid
            and require_int(
                binding["controller_pid"], f"{role}.controller_pid", 2)
            == controller_pid
            and require_int(
                binding["controller_start_time_ticks"],
                f"{role}.controller_start_time_ticks",
                1,
            ) == controller_start_ticks
            and require_int(
                binding["controller_uid"], f"{role}.controller_uid")
            == controller_uid
            and require_int(
                binding["controller_gid"], f"{role}.controller_gid")
            == controller_gid
            and require_string(
                binding["controller_executable_path"],
                f"{role}.controller_executable_path",
            ) == identity["controller_executable_path"]
            and validate_digest(
                binding["controller_executable_sha256"],
                f"{role}.controller_executable_sha256",
            ) == identity_executable_sha256
            and Path(require_string(
                binding["runtime_config_path"],
                f"{role}.runtime_config_path",
            )).resolve() == runtime_path.resolve()
            and validate_digest(
                binding["runtime_config_sha256"],
                f"{role}.runtime_config_sha256",
            ) == digest_file(runtime_path)
            and require_int(
                binding["runtime_config_device"],
                f"{role}.runtime_config_device",
            ) == runtime_stat.st_dev
            and require_int(
                binding["runtime_config_inode"],
                f"{role}.runtime_config_inode",
                1,
            ) == runtime_stat.st_ino
            and Path(require_string(
                binding["controller_identity_path"],
                f"{role}.controller_identity_path",
            )).resolve() == identity_path.resolve()
            and validate_digest(
                binding["controller_identity_sha256"],
                f"{role}.controller_identity_sha256",
            ) == digest_file(identity_path)
            and require_int(
                binding["controller_identity_device"],
                f"{role}.controller_identity_device",
            ) == identity_stat.st_dev
            and require_int(
                binding["controller_identity_inode"],
                f"{role}.controller_identity_inode",
                1,
            ) == identity_stat.st_ino,
            f"{role}: bound identity mismatch",
        )
        readiness_row = readiness_by_id[executor_id]
        readiness_path = Path(require_string(
            readiness_row["path"], f"{role}.readiness.path"))
        readiness_sha256 = validate_digest(
            readiness_row["sha256"], f"{role}.readiness.sha256")
        readiness_device = require_int(
            readiness_row["device"], f"{role}.readiness.device")
        readiness_inode = require_int(
            readiness_row["inode"], f"{role}.readiness.inode", 1)
        require(
            readiness_path.resolve() == binding_path.resolve()
            and readiness_sha256 == digest_file(binding_path)
            and readiness_device == binding_stat.st_dev
            and readiness_inode == binding_stat.st_ino,
            f"{role}: readiness binding mismatch",
        )

        descriptor_role = f"executor_transport::{executor_id}"
        descriptor_path = artifact_roles[descriptor_role]
        descriptor = read_json(descriptor_path, descriptor_role)
        require(
            descriptor_path.read_bytes() == canonical_bytes(descriptor)
            and isinstance(descriptor, dict)
            and set(descriptor) == TRANSPORT_DESCRIPTOR_V4_KEYS
            and descriptor["schema"]
            == "s40-executor-transport-descriptor-v4",
            f"{descriptor_role}: identity mismatch",
        )
        require(
            descriptor["gateway_environment"]
            == (
                base_environment
                if executor["role"] == "PHONE"
                else privileged_environment
            )
            and descriptor["gateway_prepublication_identity"]
            == gateway_ready_by_id[executor_id]["process_identity"]
            and descriptor["gateway_post_auth_identity"]
            == gateway_post_auth_by_id[executor_id],
            f"{descriptor_role}: gateway provenance mismatch",
        )
        require(
            require_int(
                descriptor["controller_pid"],
                f"{descriptor_role}.controller_pid",
                2,
            ) == controller_pid
            and require_int(
                descriptor["controller_start_time_ticks"],
                f"{descriptor_role}.controller_start_time_ticks",
                1,
            ) == controller_start_ticks
            and require_int(
                descriptor["controller_uid"],
                f"{descriptor_role}.controller_uid",
            ) == controller_uid
            and require_int(
                descriptor["controller_gid"],
                f"{descriptor_role}.controller_gid",
            ) == controller_gid
            and require_string(
                descriptor["controller_executable_path"],
                f"{descriptor_role}.controller_executable_path",
            )
            == identity["controller_executable_path"]
            and validate_digest(
                descriptor["controller_executable_sha256"],
                f"{descriptor_role}.controller_executable_sha256",
            ) == identity_executable_sha256
            and Path(require_string(
                descriptor["controller_identity_path"],
                f"{descriptor_role}.controller_identity_path",
            )).resolve()
            == identity_path.resolve()
            and validate_digest(
                descriptor["controller_identity_sha256"],
                f"{descriptor_role}.controller_identity_sha256",
            )
            == digest_file(identity_path)
            and require_int(
                descriptor["controller_identity_device"],
                f"{descriptor_role}.controller_identity_device",
            )
            == identity_stat.st_dev
            and require_int(
                descriptor["controller_identity_inode"],
                f"{descriptor_role}.controller_identity_inode",
                1,
            )
            == identity_stat.st_ino
            and require_int(
                descriptor["controller_identity_published_ns"],
                f"{descriptor_role}.controller_identity_published_ns",
                1,
            )
            == published_ns
            and Path(require_string(
                descriptor["controller_binding_path"],
                f"{descriptor_role}.controller_binding_path",
            )).resolve()
            == binding_path.resolve()
            and validate_digest(
                descriptor["controller_binding_sha256"],
                f"{descriptor_role}.controller_binding_sha256",
            )
            == digest_file(binding_path)
            and require_int(
                descriptor["controller_binding_device"],
                f"{descriptor_role}.controller_binding_device",
            )
            == binding_stat.st_dev
            and require_int(
                descriptor["controller_binding_inode"],
                f"{descriptor_role}.controller_binding_inode",
                1,
            )
            == binding_stat.st_ino
            and require_string(
                descriptor["host_boot_id"],
                f"{descriptor_role}.host_boot_id",
            ) == identity["host_boot_id"]
            and require_string(
                descriptor["executor_id"],
                f"{descriptor_role}.executor_id",
            ) == executor_id
            and require_string(
                descriptor["executor_instance_id"],
                f"{descriptor_role}.executor_instance_id",
            )
            == executor["executor_instance_id"]
            and require_int(
                descriptor["gateway_pid"],
                f"{descriptor_role}.gateway_pid",
                2,
            ) == executor["expected_peer_pid"]
            and require_int(
                descriptor["gateway_start_time_ticks"],
                f"{descriptor_role}.gateway_start_time_ticks",
                1,
            )
            == executor["expected_peer_start_time_ticks"]
            and Path(require_string(
                descriptor["runtime_config_path"],
                f"{descriptor_role}.runtime_config_path",
            )).resolve()
            == runtime_path.resolve()
            and validate_digest(
                descriptor["runtime_config_sha256"],
                f"{descriptor_role}.runtime_config_sha256",
            )
            == digest_file(runtime_path)
            and require_int(
                descriptor["runtime_config_device"],
                f"{descriptor_role}.runtime_config_device",
            ) == runtime_stat.st_dev
            and require_int(
                descriptor["runtime_config_inode"],
                f"{descriptor_role}.runtime_config_inode",
                1,
            ) == runtime_stat.st_ino
            and require_int(
                descriptor["runtime_config_published_ns"],
                f"{descriptor_role}.runtime_config_published_ns",
                1,
            )
            == runtime_config_published_ns,
            f"{descriptor_role}: controller binding mismatch",
        )
        bindings.append({
            "authenticated_ns": authenticated_ns,
            "executor_id": executor_id,
            "sha256": digest_file(binding_path),
        })
    require(
        set(readiness_by_id) == set(runtime_by_id),
        "server_readiness.controller_bindings: executor set mismatch",
    )
    return {
        "bindings": bindings,
        "controller_identity_sha256": digest_file(identity_path),
        "controller_pid": controller_pid,
        "controller_start_time_ticks": controller_start_ticks,
        "published_ns": published_ns,
    }


def validate_run_manifest(
        path: Path,
        contract_path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    require(
        not os.path.lexists(path.parent / "warm-tier-internal.token"),
        "run_manifest: warm-tier internal token was not removed",
    )
    contract_result = validate_contract(contract_path)
    contract = read_json(contract_path, "contract")
    manifest = read_json(path, "run_manifest")
    require(
        path.read_bytes() == canonical_bytes(manifest),
        "run_manifest: not canonical JSON",
    )
    require(set(manifest) == MANIFEST_KEYS,
            "run_manifest: key set mismatch")
    require(
        manifest["schema"] == "s40-physical-run-manifest-v6"
        and manifest["schema_version"] == 6,
        "run_manifest: unsupported identity",
    )
    run_id = require_string(manifest["run_id"], "run_manifest.run_id")
    mode = require_string(manifest["mode"], "run_manifest.mode")
    matrix = contract["matrix"]
    require(mode in matrix and mode != "C4_TWO_GPU_ORACLE",
            "run_manifest: unsupported physical mode")
    cache_regime = require_string(
        manifest["cache_regime"], "run_manifest.cache_regime")
    require(
        cache_regime in matrix[mode]["cache_regimes"],
        "run_manifest: cache regime is not allowed for mode",
    )
    require(isinstance(manifest["development"], bool),
            "run_manifest.development: expected bool")
    require_int(manifest["repeat_index"], "run_manifest.repeat_index")
    require(
        manifest["policy_id"] == contract["policy"]["id"],
        "run_manifest: policy mismatch",
    )
    require(
        manifest["experiment_contract_sha256"] == digest_file(contract_path),
        "run_manifest: experiment contract digest mismatch",
    )
    require(contract_result["request_count"] == 74,
            "run_manifest: frozen workload is not validated")

    binary = manifest["controller_binary"]
    require(isinstance(binary, dict) and set(binary) == BINARY_KEYS,
            "run_manifest.controller_binary: key set mismatch")
    binary_path = _validate_file_binding(
        path.parent, binary, "run_manifest.controller_binary",
        path_key="captured_path", minimum_bytes=1)
    executed_binary = Path(require_string(
        binary["executed_path"],
        "run_manifest.controller_binary.executed_path",
    ))
    require(executed_binary.is_absolute() and executed_binary.is_file(),
            "run_manifest.controller_binary: executed binary is missing")
    require(
        executed_binary.stat().st_size == binary_path.stat().st_size
        and digest_file(executed_binary) == digest_file(binary_path),
        "run_manifest.controller_binary: executed bytes differ",
    )
    argv = manifest["command_argv"]
    require(
        isinstance(argv, list)
        and argv
        and all(isinstance(value, str) and value for value in argv),
        "run_manifest.command_argv: expected nonempty string array",
    )
    require(argv[0] == str(executed_binary),
            "command: executable does not match bound binary")
    validate_serving_argv(
        argv,
        contract["runtime_source"]["serving"],
        "run_manifest.command_argv",
    )
    _require_exact_http_threads(
        argv,
        contract["run_manifest_requirements"]["required_http_threads"],
    )

    artifacts = manifest["artifacts"]
    require(isinstance(artifacts, list) and artifacts,
            "run_manifest.artifacts: expected nonempty array")
    artifact_roles: dict[str, Path] = {}
    for index, record in enumerate(artifacts):
        role, artifact_path = _validate_artifact(path.parent, record, index)
        require(role not in artifact_roles,
                f"run_manifest: duplicate artifact role {role}")
        artifact_roles[role] = artifact_path
    source_preflight_path = artifact_roles.get("source_preflight")
    require(source_preflight_path is not None,
            "run_manifest: missing source preflight")
    source_preflight = _validate_source_preflight(
        source_preflight_path,
        path.parent,
        contract_path,
        manifest,
    )
    runtime_dependency_path = artifact_roles.get(
        "runtime_dependency_manifest")
    require(
        runtime_dependency_path is not None,
        "run_manifest: missing runtime dependency manifest",
    )
    runtime_dependencies = _validate_runtime_dependencies(
        runtime_dependency_path,
        path.parent,
        source_preflight,
        artifact_roles,
    )
    require(
        "native_bench_binary" in artifact_roles,
        "run_manifest: missing native benchmark binary",
    )
    campaign = source_preflight["campaign"]
    if campaign is not None:
        software = campaign["software_lock"]
        require(
            digest_file(binary_path)
            == software["controller_binary_sha256"]
            and digest_file(artifact_roles["native_bench_binary"])
            == software["native_bench_binary_sha256"]
            and digest_file(runtime_dependencies[
                "binary_paths"]["nvidia_smi"])
            == software["nvidia_smi_sha256"]
            and source_preflight["rows"]["runtime_tool::ldd"][0]["sha256"]
            == software["ldd_sha256"]
            and digest_file(runtime_dependencies["binary_paths"]["python"])
            == software["python_sha256"]
            and digest_file(artifact_roles["executor_bundle_manifest"])
            == software["executor_bundle_manifest_sha256"]
            and digest_file(artifact_roles["evidence_bundle_manifest"])
            == software["evidence_bundle_manifest_sha256"],
            "run_manifest: campaign software lock mismatch",
        )
    require(
        runtime_dependencies["binary_paths"]["controller"].resolve()
        == binary_path.resolve()
        and runtime_dependencies["binary_paths"]["native_bench"].resolve()
        == artifact_roles["native_bench_binary"].resolve(),
        "run_manifest: runtime binary binding mismatch",
    )

    runtime_path = artifact_roles.get("runtime_config")
    plan_path = artifact_roles.get("runtime_plan")
    require(runtime_path is not None and plan_path is not None,
            "run_manifest: missing runtime configuration")
    runtime = read_json(runtime_path, "runtime_config")
    require(runtime_path.read_bytes() == canonical_bytes(runtime),
            "runtime_config: not canonical JSON")
    runtime_stat = runtime_path.stat(follow_symlinks=False)
    require(
        stat.S_ISREG(runtime_stat.st_mode)
        and not runtime_path.is_symlink()
        and runtime.get("schema") == "llama-server-warm-tier-runtime-v4"
        and runtime.get("run_id") == run_id,
        "runtime_config: identity mismatch",
    )
    built = build_runtime_config(plan_path, contract_path)
    require(
        canonical_bytes(built["runtime"]) == runtime_path.read_bytes(),
        "runtime_config: does not match bound runtime plan",
    )
    require(built["mode"] == mode, "runtime_config: mode mismatch")
    evidence_root_path = artifact_roles.get("evidence_root")
    require(evidence_root_path is not None,
            "run_manifest: missing evidence root")
    require(
        evidence_root_path.resolve()
        == Path(built["evidence_root_path"]).resolve()
        and digest_file(evidence_root_path)
        == built["evidence_root_sha256"]
        == runtime["evidence_root_sha256"],
        "run_manifest: evidence root binding mismatch",
    )

    required_roles = set(BASE_ARTIFACT_ROLES)
    required_roles.update(
        f"executor_bundle::{name}"
        for name in EXECUTOR_BUNDLE_SOURCES
    )
    required_roles.update(
        f"evidence_bundle::{name}"
        for name in EVIDENCE_BUNDLE_SOURCES
    )
    required_roles.update(_expected_executor_roles(runtime))
    required_roles.update(runtime_dependencies["dependency_roles"])
    if mode.startswith("T"):
        required_roles.update(PHONE_ARTIFACT_ROLES)
    if mode == "C3_DUAL_PARTIAL_OFFLOAD":
        required_roles.add("c3_profile_lock")
        for model_id in contract["workload"]["models"]:
            required_roles.add(f"c3_placement::{model_id}")
        require(
            artifact_roles["c3_profile_lock"].resolve()
            == Path(built["c3_profile_lock_path"]).resolve()
            and digest_file(artifact_roles["c3_profile_lock"])
            == built["c3_profile_lock_sha256"],
            "run_manifest: C3 profile lock binding mismatch",
        )
    require(set(artifact_roles) == required_roles,
            "run_manifest: artifact role set mismatch")
    bundle_manifest = artifact_roles["executor_bundle_manifest"]
    bundle_directory = bundle_manifest.parent
    bundle_result = validate_python_executor_bundle(
        bundle_directory,
        bundle_manifest,
        digest_file(bundle_manifest),
    )
    for name in EXECUTOR_BUNDLE_SOURCES:
        require(
            artifact_roles[f"executor_bundle::{name}"].resolve()
            == (bundle_directory / name).resolve(),
            "run_manifest: executor bundle artifact mismatch",
        )
    require(
        bundle_result["status"] == "PASS"
        and read_json(
            bundle_manifest, "executor_bundle_manifest")["schema"]
        == "s40-executor-bundle-v2"
        and read_json(
            bundle_manifest, "executor_bundle_manifest")["python_flags"]
        == ["-I", "-S", "-B"],
        "run_manifest: executor bundle rejected",
    )
    evidence_bundle_manifest = artifact_roles["evidence_bundle_manifest"]
    evidence_bundle_directory = evidence_bundle_manifest.parent
    evidence_bundle_result = validate_evidence_bundle(
        evidence_bundle_directory,
        evidence_bundle_manifest,
        digest_file(evidence_bundle_manifest),
    )
    for name in EVIDENCE_BUNDLE_SOURCES:
        require(
            artifact_roles[f"evidence_bundle::{name}"].resolve()
            == (evidence_bundle_directory / name).resolve(),
            "run_manifest: evidence bundle artifact mismatch",
        )
    require(
        evidence_bundle_result["status"] == "PASS"
        and read_json(
            evidence_bundle_manifest, "evidence_bundle_manifest")["schema"]
        == "s40-evidence-bundle-v2"
        and read_json(
            evidence_bundle_manifest,
            "evidence_bundle_manifest",
        )["python_flags"] == ["-I", "-S", "-B"],
        "run_manifest: evidence bundle rejected",
    )
    sampler_record = read_json(
        artifact_roles["resource_sampler_argv"],
        "resource_sampler_argv",
    )
    require(
        isinstance(sampler_record, dict)
        and set(sampler_record) == {"argv", "schema"}
        and sampler_record["schema"] == "s40-resource-sampler-argv-v1",
        "resource_sampler_argv: schema mismatch",
    )
    sampler_argv = sampler_record["argv"]
    require(
        isinstance(sampler_argv, list)
        and all(isinstance(item, str) and item for item in sampler_argv),
        "resource_sampler_argv: invalid argv",
    )
    sampler_controller_pid_text = _flag_value(
        sampler_argv,
        "--controller-pid",
        "resource_sampler_argv",
    )
    require(
        sampler_controller_pid_text.isascii()
        and sampler_controller_pid_text.isdigit(),
        "resource_sampler_argv: invalid controller PID",
    )
    sampler_controller_pid = require_int(
        int(sampler_controller_pid_text),
        "resource_sampler_argv.controller_pid",
        2,
    )
    expected_sampler_argv = _isolated_python_argv(
        runtime_dependencies["binary_paths"]["python"],
        evidence_bundle_directory,
        artifact_roles["evidence_bundle::resource_sampler.py"],
        [
            "--run-id",
            run_id,
            "--gpu-uuid",
            contract["runtime_source"]["expected_gpu_uuid"],
            "--controller-pid",
            str(sampler_controller_pid),
            "--output",
            str(artifact_roles["resource_samples"]),
            "--stop-file",
            str(path.parent / "resource-sampler.stop"),
            "--interval-ms",
            "200",
            "--nvidia-smi-path",
            str(runtime_dependencies["binary_paths"]["nvidia_smi"]),
            "--nvidia-smi-sha256",
            digest_file(runtime_dependencies["binary_paths"]["nvidia_smi"]),
        ],
    )
    require(
        sampler_argv == expected_sampler_argv,
        "resource_sampler_argv: command mismatch",
    )
    executor_summaries = _validate_executor_bindings(
        path.parent,
        manifest["executor_bindings"],
        runtime,
        artifact_roles,
    )
    binding_by_id = {
        binding["executor_id"]: binding
        for binding in manifest["executor_bindings"]
    }
    expected_base_environment = {
        "CUDA_CACHE_PATH": str(path.parent / "cuda-cache"),
        "CUDA_VISIBLE_DEVICES":
            contract["runtime_source"]["expected_gpu_uuid"],
        "HOME": str(path.parent / "run-home"),
        "LANG": "C",
        "LC_ALL": "C",
        "LD_LIBRARY_PATH": str(runtime_dependencies["library_path"]),
        "LLAMA_SERVER_WARM_TIER_CONFIG": str(runtime_path),
        "NVIDIA_VISIBLE_DEVICES":
            contract["runtime_source"]["expected_gpu_uuid"],
        "PATH":
            f"{path.parent / 'captured-runtime' / 'bin'}:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "S40_EVIDENCE_BUNDLE": "1",
        "S40_EVIDENCE_BUNDLE_MANIFEST":
            str(evidence_bundle_manifest),
        "S40_EVIDENCE_BUNDLE_SHA256":
            digest_file(evidence_bundle_manifest),
        "S40_EXECUTOR_BUNDLE": "1",
        "S40_EXECUTOR_BUNDLE_MANIFEST": str(bundle_manifest),
        "S40_EXECUTOR_BUNDLE_SHA256": digest_file(bundle_manifest),
        "S40_NVIDIA_SMI_PATH":
            str(runtime_dependencies["binary_paths"]["nvidia_smi"]),
        "S40_NVIDIA_SMI_SHA256":
            digest_file(runtime_dependencies["binary_paths"]["nvidia_smi"]),
        "TMPDIR": str(path.parent / "run-tmp"),
        "TZ": "UTC",
    }
    expected_privileged_environment = dict(expected_base_environment)
    expected_privileged_environment[
        "LLAMA_SERVER_WARM_TIER_INTERNAL_TOKEN_FILE"
    ] = str(path.parent / "warm-tier-internal.token")
    _validate_launch_environment(
        expected_base_environment,
        "expected_base_environment",
        privileged=False,
    )
    _validate_launch_environment(
        expected_privileged_environment,
        "expected_privileged_environment",
    )
    runtime_by_id = {
        executor["executor_id"]: executor
        for executor in runtime["executors"]
    }
    expected_models = sorted(contract["workload"]["models"])
    for executor_id, summary in executor_summaries.items():
        require(
            summary.get("status") == "PASS"
            and summary.get("configured_models") == expected_models,
            f"executor {executor_id}: configured model set mismatch",
        )
        require(
            summary.get("initial_active_models") == [],
            f"executor {executor_id}: startup must be empty before activation",
        )
        require(
            Path(summary["runtime_config_path"]).resolve()
            == runtime_path.resolve()
            and summary["runtime_config_sha256"] == digest_file(runtime_path)
            and summary["runtime_config_device"] == runtime_stat.st_dev
            and summary["runtime_config_inode"] == runtime_stat.st_ino,
            f"executor {executor_id}: final runtime config identity mismatch",
        )
        expected_gateway_environment = (
            expected_base_environment
            if runtime_by_id[executor_id]["role"] == "PHONE"
            else expected_privileged_environment
        )
        require(
            summary["gateway_launch_environment"]
            == expected_gateway_environment,
            f"executor {executor_id}: launch environment mismatch",
        )
    route_qualifications = _validate_route_qualification_summaries(
        executor_summaries,
        expected_models,
    )
    transport_value = read_json(
        artifact_roles["transport_overhead"], "transport_overhead")
    transport_measurement = validate_combined_measurement(
        transport_value, 50)
    require(
        transport_value["executor_bundle_manifest_sha256"]
        == digest_file(artifact_roles["executor_bundle_manifest"]),
        "transport_overhead: executor bundle mismatch",
    )
    require(
        transport_value["host_boot_id"]
        == next(
            device["boot_id"] for device in manifest["devices"]
            if device["device_role"] == "GPU"
        ),
        "transport_overhead: host boot mismatch",
    )
    native_binary = artifact_roles["native_bench_binary"]
    require(
        Path(transport_value["native_bench_binary"]["path"]).resolve()
        == native_binary.resolve()
        and transport_measurement["native_bench_binary_bytes"]
        == native_binary.stat().st_size
        and transport_measurement["native_bench_binary_sha256"]
        == digest_file(native_binary),
        "transport_overhead: native binary mismatch",
    )
    desktop_ids = {
        executor["executor_id"]
        for executor in runtime["executors"]
        if executor["role"] in {"GPU", "CPU"}
    }
    fastest_execute = [
        summary.get("fastest_execute_duration_ns")
        for executor_id, summary in executor_summaries.items()
        if executor_id in desktop_ids
        and summary.get("fastest_execute_duration_ns") is not None
    ]
    require(fastest_execute,
            "transport_overhead: no desktop execute quantum")
    fastest_execute_ns = min(
        require_int(
            value, "transport_overhead.fastest_execute_duration_ns", 1)
        for value in fastest_execute
    )
    transport_gate = validate_combined_transport_gate(
        transport_value,
        fastest_execute_ns,
        50,
    )
    evidence_root = read_json(evidence_root_path, "evidence_root")
    for record in evidence_root["executor_configs"]:
        executor_id = record["executor_id"]
        config_path = artifact_roles[f"gateway_config::{executor_id}"]
        require(
            config_path.resolve()
            == Path(record["gateway_config_path"]).resolve()
            and digest_file(config_path) == record["gateway_config_sha256"],
            "run_manifest: gateway config binding mismatch",
        )
        executor = next(
            item for item in runtime["executors"]
            if item["executor_id"] == executor_id
        )
        if executor["role"] in {"GPU", "CPU"}:
            config = read_json(
                config_path, f"gateway config {executor_id}")
            nvidia_smi = config.get("nvidia_smi")
            captured_nvidia_smi = runtime_dependencies[
                "binary_paths"]["nvidia_smi"]
            require(
                config.get("schema") == "s40-desktop-executor-config-v4"
                and nvidia_smi == {
                    "bytes": captured_nvidia_smi.stat().st_size,
                    "path": str(captured_nvidia_smi),
                    "sha256": digest_file(captured_nvidia_smi),
                },
                "run_manifest: desktop nvidia-smi binding mismatch",
            )
    if mode == "C3_DUAL_PARTIAL_OFFLOAD":
        for model_id, record in evidence_root[
                "c3_placement_artifacts"].items():
            placement = artifact_roles[f"c3_placement::{model_id}"]
            require(
                placement.resolve() == Path(record["path"]).resolve()
                and digest_file(placement) == record["sha256"],
                "run_manifest: C3 placement binding mismatch",
            )

    trace_start = read_json(artifact_roles["trace_start"], "trace_start")
    require(trace_start.get("run_id") == run_id,
            "trace_start: run ID mismatch")
    require(
        trace_start.get("runtime_config_sha256") == digest_file(runtime_path),
        "trace_start: runtime config digest mismatch",
    )
    require(
        trace_start.get("experiment_contract_sha256")
        == digest_file(contract_path),
        "trace_start: contract digest mismatch",
    )
    require(
        trace_start.get("requests_sha256")
        == contract["workload"]["requests_sha256"],
        "trace_start: request digest mismatch",
    )
    activation = _validate_activation_evidence(
        artifact_roles["activation_evidence"], run_id)
    readiness = read_json(
        artifact_roles["server_readiness"], "server_readiness")
    require(
        set(readiness) == READINESS_KEYS
        and readiness["schema"] == "s40-server-readiness-v6"
        and readiness["run_id"] == run_id
        and readiness["activation_evidence_sha256"]
        == digest_file(artifact_roles["activation_evidence"])
        and Path(readiness["runtime_config_path"]).resolve()
        == runtime_path.resolve()
        and readiness["runtime_config_sha256"] == digest_file(runtime_path)
        and readiness["runtime_config_device"] == runtime_stat.st_dev
        and readiness["runtime_config_inode"] == runtime_stat.st_ino,
        "server_readiness: identity mismatch",
    )
    require(
        _validate_launch_environment(
            readiness["launch_environment"],
            "server_readiness.launch_environment",
        ) == expected_privileged_environment,
        "server_readiness: launch environment mismatch",
    )
    _validate_gateway_identity_rows(
        readiness["gateway_post_auth"],
        "server_readiness.gateway_post_auth",
        binding_by_id,
        "gateway_post_auth_identity",
    )
    _validate_gateway_identity_rows(
        readiness["gateway_final"],
        "server_readiness.gateway_final",
        binding_by_id,
        "gateway_final_identity",
    )
    runtime_config_published_ns = require_int(
        readiness["runtime_config_published_ns"],
        "server_readiness.runtime_config_published_ns",
        1,
    )
    controller_started_ns = require_int(
        readiness["controller_started_ns"],
        "server_readiness.controller_started_ns",
        runtime_config_published_ns,
    )
    require(
        all(
            summary["runtime_config_published_ns"]
            == runtime_config_published_ns
            and summary["host_boot_id"] == readiness["host_boot_id"]
            for summary in executor_summaries.values()
        ),
        "server_readiness: executor publication identity mismatch",
    )
    if mode in {"T1_PHONE_WARM_TIER", "T2_PHONE_NO_PROMOTION"}:
        phone_readiness = readiness["phone_observer"]
        require(
            isinstance(phone_readiness, dict)
            and set(phone_readiness)
            == {
                "identity_sha256",
                "telemetry_pid",
                "telemetry_started_ns",
            }
            and phone_readiness["identity_sha256"]
            == digest_file(artifact_roles["phone_identity"]),
            "server_readiness: phone observer identity mismatch",
        )
        require_int(
            phone_readiness["telemetry_pid"],
            "server_readiness.phone_observer.telemetry_pid",
            2,
        )
        phone_telemetry_started_ns = require_int(
            phone_readiness["telemetry_started_ns"],
            "server_readiness.phone_observer.telemetry_started_ns",
            1,
        )
    else:
        require(
            readiness["phone_observer"] is None,
            "server_readiness: unexpected phone observer",
        )
        phone_telemetry_started_ns = None
    base = readiness["base_readiness"]
    require(
        isinstance(base, dict)
        and set(base) == {
            "completed_ns",
            "http_status",
            "response",
            "response_sha256",
            "schema",
            "started_ns",
        }
        and base["schema"] == "s40-server-readiness-v1"
        and base["http_status"] == 200
        and base["response"] == {
            "controller_epoch": 0,
            "schema": "llama-server-warm-tier-activate-status-v1",
            "state": "WAITING",
        },
        "server_readiness: invalid pre-gateway state",
    )
    gateway_ready = readiness["gateway_ready"]
    require(
        isinstance(gateway_ready, list)
        and len(gateway_ready) == len(runtime["executors"]),
        "server_readiness: gateway count mismatch",
    )
    gateway_ids = set()
    gateway_times = []
    gateway_identity_times = []
    runtime_by_id = {
        executor["executor_id"]: executor
        for executor in runtime["executors"]
    }
    require(
        [
            row.get("executor_id")
            for row in gateway_ready
            if isinstance(row, dict)
        ] == sorted(runtime_by_id),
        "server_readiness: gateway order is not deterministic",
    )
    for index, row in enumerate(gateway_ready):
        field = f"server_readiness.gateway_ready[{index}]"
        require(
            isinstance(row, dict)
            and set(row) == {
                "executor_id",
                "executor_instance_id",
                "identity_captured_ns",
                "pid",
                "process_identity",
                "process_start_time_ticks",
                "socket_path",
                "t_ns",
            },
            f"{field}: key set mismatch",
        )
        executor_id = require_string(
            row["executor_id"], f"{field}.executor_id")
        require(
            executor_id not in gateway_ids,
            f"{field}: duplicate executor",
        )
        gateway_ids.add(executor_id)
        require(executor_id in runtime_by_id, f"{field}: unknown executor")
        executor = runtime_by_id[executor_id]
        summary = executor_summaries[executor_id]
        identity_captured_ns = require_int(
            row["identity_captured_ns"],
            f"{field}.identity_captured_ns",
            1,
        )
        require(
            row["executor_instance_id"] == executor["executor_instance_id"]
            == summary["executor_instance_id"]
            and row["pid"] == executor["expected_peer_pid"]
            == summary["gateway_pid"]
            and row["process_start_time_ticks"]
            == executor["expected_peer_start_time_ticks"]
            == summary["gateway_start_time_ticks"]
            and row["socket_path"] == executor["socket_path"]
            and identity_captured_ns == summary["identity_captured_ns"]
            and _validate_gateway_process_identity(
                row["process_identity"],
                binding_by_id[executor_id]["gateway_argv"],
                f"{field}.process_identity",
            )
            == summary["gateway_prepublication_identity"],
            f"{field}: executor identity mismatch",
        )
        gateway_identity_times.append(identity_captured_ns)
        gateway_times.append(require_int(row["t_ns"], f"{field}.t_ns"))
    require(
        gateway_ids == {
            executor["executor_id"] for executor in runtime["executors"]},
        "server_readiness: gateway set mismatch",
    )
    require(
        max(gateway_identity_times)
        <= runtime_config_published_ns
        <= min(gateway_times)
        <= max(gateway_times)
        <= controller_started_ns
        <= require_int(base["started_ns"], "server_readiness.started_ns")
        <= require_int(base["completed_ns"], "server_readiness.completed_ns")
        <= activation["started_ns"]
        <= activation["ready_ns"]
        < require_int(trace_start["created_ns"], "trace_start.created_ns"),
        "server_readiness: startup ordering mismatch",
    )
    if phone_telemetry_started_ns is not None:
        require(
            activation["ready_ns"] <= phone_telemetry_started_ns,
            "server_readiness: phone observer started before activation",
        )
    require(
        readiness["activation_ready"] == {
            "controller_epoch": activation["controller_epoch"],
            "schema": "llama-server-warm-tier-activate-status-v1",
            "state": "READY",
        },
        "server_readiness: final activation mismatch",
    )

    requests_path = (
        contract_path.parent
        / contract["workload"]["requests_path"]
    ).resolve()
    validate_http_evidence(
        artifact_roles["http_evidence"],
        artifact_roles["trace_start"],
        artifact_roles["trace_acquisition_result"],
        requests_path,
        run_id,
    )
    reduced = reduce_paths(
        artifact_roles["controller_events"],
        requests_path,
        artifact_roles["resource_samples"],
        artifact_roles["trace_start"],
    )
    require(reduced["runtime_config_sha256"] == digest_file(runtime_path),
            "controller events: runtime config mismatch")
    require(reduced["verdict"] == "PASS",
            "controller events: fail-closed verdict")
    _validate_command_bijection(
        executor_summaries,
        reduced["command_ledger"],
        run_id,
        digest_file(runtime_path),
    )
    phone_summary = None
    if mode in {"T1_PHONE_WARM_TIER", "T2_PHONE_NO_PROMOTION"}:
        physical = source_preflight["physical_plan"]
        source_identity_argv = physical["phone_identity_argv"]
        source_telemetry_argv = physical["phone_telemetry_argv"]
        captured_python = runtime_dependencies["binary_paths"]["python"]
        observer_source = (
            artifact_roles["executor_bundle::phone_observer_bridge.py"])
        expected_identity_argv = _isolated_python_argv(
            captured_python,
            bundle_directory,
            observer_source,
            source_identity_argv[2:],
        )
        expected_telemetry_argv = _isolated_python_argv(
            captured_python,
            bundle_directory,
            observer_source,
            source_telemetry_argv[2:],
        )
        for observer_argv in (
                expected_identity_argv, expected_telemetry_argv):
            config_index = observer_argv.index("--ssh-config") + 1
            observer_argv[config_index] = str(
                artifact_roles["phone_ssh_config"])
        identity_argv_record = read_json(
            artifact_roles["phone_identity_argv"],
            "phone_identity_argv",
        )
        telemetry_argv_record = read_json(
            artifact_roles["phone_telemetry_argv"],
            "phone_telemetry_argv",
        )
        require(
            identity_argv_record == {
                "argv": expected_identity_argv,
                "schema": "s40-phone-observer-argv-v1",
            }
            and telemetry_argv_record == {
                "argv": expected_telemetry_argv,
                "schema": "s40-phone-observer-argv-v1",
            },
            "phone observer: executed argv mismatch",
        )
        expected_model_id = _flag_value(
            source_identity_argv, "--model", "phone_identity_argv")
        require(
            _flag_value(
                source_telemetry_argv,
                "--model",
                "phone_telemetry_argv",
            ) == expected_model_id
            and _flag_value(
                source_identity_argv,
                "--run-id",
                "phone_identity_argv",
            ) == run_id
            and _flag_value(
                source_telemetry_argv,
                "--run-id",
                "phone_telemetry_argv",
            ) == run_id,
            "phone observer: source command identity mismatch",
        )
        identity_value = read_json(
            artifact_roles["phone_identity"], "phone_identity")
        require(
            digest_file(artifact_roles["phone_ssh_config"])
            == source_preflight["rows"]["phone_ssh_config"][0]["sha256"]
            == identity_value.get("ssh_config_sha256"),
            "phone observer: SSH config digest mismatch",
        )
        phone_summary = validate_phone_observer(
            artifact_roles["phone_identity"],
            artifact_roles["phone_telemetry"],
            artifact_roles["phone_telemetry_stderr"],
            expected_model_id,
            run_id,
        )
        run_end_ns = (
            reduced["trace_origin_ns"] + reduced["run_duration_ns"])
        require(
            activation["ready_ns"]
            <= phone_summary["identity_interval_ns"][0]
            <= phone_summary["identity_interval_ns"][1]
            <= phone_telemetry_started_ns
            <= phone_summary["telemetry_interval_ns"][0]
            < require_int(
                trace_start["created_ns"], "trace_start.created_ns")
            and phone_summary["telemetry_interval_ns"][1] >= run_end_ns,
            "phone observer: interval does not cover the measured run",
        )
    phone_route_instances = _validate_phone_route_coverage(
        executor_summaries,
        runtime,
        phone_summary,
    )
    launch = read_json(artifact_roles["controller_launch"], "controller_launch")
    require(set(launch) == LAUNCH_KEYS,
            "controller_launch: key set mismatch")
    launch_library_path = Path(require_string(
        launch["library_path"], "controller_launch.library_path"))
    require(
        launch["schema"] == "s40-controller-launch-v4"
        and launch["run_id"] == run_id
        and launch["command_argv"] == argv
        and launch["binary_sha256"] == digest_file(binary_path)
        and launch["runtime_config_sha256"] == digest_file(runtime_path)
        and launch_library_path.resolve()
        == runtime_dependencies["library_path"].resolve()
        and launch["experiment_contract_sha256"]
        == source_preflight["contract_sha256"]
        and launch["physical_plan_sha256"]
        == source_preflight["physical_plan_sha256"]
        and launch["selected_gpu_environment"]
        == {
            "CUDA_VISIBLE_DEVICES":
                contract["runtime_source"]["expected_gpu_uuid"],
            "NVIDIA_VISIBLE_DEVICES":
                contract["runtime_source"]["expected_gpu_uuid"],
        }
        and _validate_launch_environment(
            launch["launch_environment"],
            "controller_launch.launch_environment",
        ) == expected_privileged_environment
        and launch["source_preflight_sha256"]
        == source_preflight["sha256"],
        "controller_launch: identity mismatch",
    )
    launch_pid = require_int(
        launch["controller_pid"], "controller_launch.controller_pid", 2)
    require(
        sampler_controller_pid == launch_pid,
        "resource_sampler_argv: controller PID mismatch",
    )
    launch_start_ticks = require_int(
        launch["controller_start_ticks"],
        "controller_launch.controller_start_ticks",
        1,
    )
    require_string(launch["host_boot_id"], "controller_launch.host_boot_id")
    launch_started_ns = require_int(
        launch["started_ns"], "controller_launch.started_ns")
    launch_stopped_ns = require_int(
        launch["stopped_ns"], "controller_launch.stopped_ns")
    require(
        launch_started_ns == controller_started_ns
        and launch["host_boot_id"] == readiness["host_boot_id"]
        and launch_stopped_ns >= launch_started_ns,
            "controller_launch: invalid interval")
    measured_run_end_ns = (
        reduced["trace_origin_ns"] + reduced["run_duration_ns"])
    require(
        all(
            measured_run_end_ns
            <= require_int(
                binding["gateway_final_identity"]["observed_ns"],
                "executor_binding.gateway_final_identity.observed_ns",
                measured_run_end_ns,
            )
            <= launch_stopped_ns
            for binding in manifest["executor_bindings"]
        ),
        "gateway final identity does not bracket shutdown",
    )
    require(
        source_preflight["captured_ns"] <= launch_started_ns,
        "controller_launch: source preflight completed after launch",
    )
    require(
        require_int(launch["exit_code"], "controller_launch.exit_code") == 0,
        "controller_launch: unclean exit",
    )
    controller_authentication = _validate_controller_authentication(
        artifact_roles,
        readiness,
        launch,
        runtime,
        runtime_path,
        runtime_stat,
        binary_path,
        runtime_config_published_ns,
        controller_started_ns,
        require_int(base["started_ns"], "server_readiness.started_ns"),
        activation["post_started_ns"],
        activation["ready_ns"],
        run_id,
    )
    expected_cleanup = [
        f"gateway::{executor['executor_id']}"
        for executor in runtime["executors"]
    ]
    if mode in {"T1_PHONE_WARM_TIER", "T2_PHONE_NO_PROMOTION"}:
        expected_cleanup.append("phone_telemetry")
    expected_cleanup.append("controller")
    expected_cleanup.append("gpu_observer")
    _validate_orchestrator_evidence(
        artifact_roles["orchestrator_evidence"],
        run_id,
        expected_cleanup,
        launch_stopped_ns,
    )
    energy = reduced["energy"]
    require(
        energy["gpu_uuid"]
        == contract["runtime_source"]["expected_gpu_uuid"]
        and energy["host_boot_id"] == launch["host_boot_id"]
        and energy["controller_pid"] == launch_pid
        and energy["controller_start_ticks"] == launch_start_ticks,
        "controller_launch: resource identity mismatch",
    )
    require(
        artifact_roles["gpu_observer_stdout"].stat().st_size == 0
        and artifact_roles["gpu_observer_stderr"].stat().st_size == 0,
        "GPU observer: stdout or stderr is not empty",
    )
    physical = source_preflight["physical_plan"]
    require(
        Path(physical["gpu_lock_path"])
        == canonical_lock_path(
            contract["runtime_source"]["expected_gpu_uuid"]),
        "GPU lock: noncanonical source lock path",
    )
    observer_argv_record = read_json(
        artifact_roles["gpu_observer_argv"], "gpu_observer_argv")
    expected_observer_argv = _isolated_python_argv(
        runtime_dependencies["binary_paths"]["python"],
        evidence_bundle_directory,
        artifact_roles["evidence_bundle::gpu_isolation.py"],
        [
            "--output",
            str(artifact_roles["gpu_observer"]),
            "--stop-file",
            str(path.parent / "gpu-observer.stop"),
            "--lock-path",
            physical["gpu_lock_path"],
            "--lock-output",
            str(artifact_roles["gpu_lock_record"]),
            "--run-id",
            run_id,
            "--gpu-uuid",
            contract["runtime_source"]["expected_gpu_uuid"],
            "--gpu-name",
            contract["runtime_source"]["expected_gpu_name"],
            "--nvidia-smi",
            str(runtime_dependencies["binary_paths"]["nvidia_smi"]),
            "--nvidia-smi-sha256",
            digest_file(runtime_dependencies["binary_paths"]["nvidia_smi"]),
            "--interval-ms",
            "200",
        ],
    )
    require(
        observer_argv_record
        == {
            "argv": expected_observer_argv,
            "schema": "s40-gpu-observer-argv-v1",
        },
        "GPU observer: executed argv mismatch",
    )
    allowed_gpu_processes: dict[tuple[int, int], list[str]] = {}
    for summary in executor_summaries.values():
        if summary.get("role") != "GPU":
            continue
        processes = summary.get("runtime_processes")
        require(
            isinstance(processes, list),
            "GPU observer: executor process evidence is missing",
        )
        for index, process in enumerate(processes):
            require(
                isinstance(process, dict)
                and set(process) == {"argv", "pid", "start_ticks"},
                f"GPU observer: process[{index}] key set mismatch",
            )
            identity = (
                require_int(
                    process["pid"], "GPU observer allowed PID", 1),
                require_int(
                    process["start_ticks"],
                    "GPU observer allowed start ticks",
                    1,
                ),
            )
            argv_value = process["argv"]
            require(
                isinstance(argv_value, list)
                and argv_value
                and all(
                    isinstance(item, str) and item
                    for item in argv_value
                ),
                "GPU observer: invalid allowed command",
            )
            require(
                identity not in allowed_gpu_processes
                or allowed_gpu_processes[identity] == argv_value,
                "GPU observer: conflicting process identity",
            )
            allowed_gpu_processes[identity] = argv_value
    require(
        allowed_gpu_processes,
        "GPU observer: no validated GPU process identity",
    )
    gpu_observer = validate_observation(
        artifact_roles["gpu_observer"],
        expected_run_id=run_id,
        expected_uuid=contract["runtime_source"]["expected_gpu_uuid"],
        expected_name=contract["runtime_source"]["expected_gpu_name"],
        expected_boot_id=launch["host_boot_id"],
        trace_start_ns=launch_started_ns,
        trace_end_ns=launch_stopped_ns,
        allowed_processes=allowed_gpu_processes,
        max_gap_ns=1_000_000_000,
        max_probe_duration_ns=1_000_000_000,
    )
    captured_nvidia_smi = runtime_dependencies["binary_paths"]["nvidia_smi"]
    require(
        Path(gpu_observer["nvidia_smi_path"]).resolve()
        == captured_nvidia_smi.resolve()
        and gpu_observer["nvidia_smi_bytes"]
        == captured_nvidia_smi.stat().st_size
        and gpu_observer["nvidia_smi_sha256"]
        == digest_file(captured_nvidia_smi)
        and gpu_observer["gpu_index"] == physical["gpu_index"]
        and gpu_observer["gpu_pci_bus_id"] == physical["gpu_pci_bus_id"],
        "GPU observer: selected GPU mapping or binary mismatch",
    )
    lock_value = read_json(
        artifact_roles["gpu_lock_record"], "gpu_lock_record")
    lock_interval = validate_lock_record(
        lock_value,
        contract["runtime_source"]["expected_gpu_uuid"],
        run_id,
        launch["host_boot_id"],
    )
    require(
        lock_value["lock_path"] == physical["gpu_lock_path"]
        and lock_interval["acquired_ns"] <= gpu_observer["started_ns"]
        <= launch_started_ns
        and lock_interval["released_ns"] >= gpu_observer["completed_ns"]
        >= launch_stopped_ns,
        "GPU lock: interval does not cover the selected-GPU run",
    )

    devices = manifest["devices"]
    require(isinstance(devices, list) and devices,
            "run_manifest.devices: expected nonempty array")
    device_roles: set[str] = set()
    stable_ids: set[str] = set()
    for index, device in enumerate(devices):
        field = f"device[{index}]"
        require(isinstance(device, dict) and set(device) == DEVICE_KEYS,
                f"{field}: key set mismatch")
        role = require_string(device["device_role"], f"{field}.device_role")
        stable_id = require_string(device["stable_id"], f"{field}.stable_id")
        require(role not in device_roles, f"{field}: duplicate role")
        require(stable_id not in stable_ids, f"{field}: duplicate stable ID")
        device_roles.add(role)
        stable_ids.add(stable_id)
        require_string(device["boot_id"], f"{field}.boot_id")
    require(device_roles == MODE_DEVICE_ROLES[mode],
            "run_manifest: device role set mismatch")
    selected_gpu = next(
        device for device in devices if device["device_role"] == "GPU")
    require(
        selected_gpu["stable_id"]
        == contract["runtime_source"]["expected_gpu_uuid"],
        "run_manifest: selected GPU identity mismatch",
    )
    require(
        selected_gpu["boot_id"] == trace_start["host_boot_id"],
        "run_manifest: trace host boot identity mismatch",
    )
    require(
        selected_gpu["stable_id"] == reduced["energy"]["gpu_uuid"]
        and selected_gpu["boot_id"] == reduced["energy"]["host_boot_id"]
        == launch["host_boot_id"],
        "run_manifest: selected resource identity mismatch",
    )
    if phone_summary is not None:
        by_device_role = {
            device["device_role"]: device for device in devices}
        require(
            phone_summary["stable_ids"]
            == {
                "op12": by_device_role["OP12"]["stable_id"],
                "op15": by_device_role["OP15"]["stable_id"],
            }
            and phone_summary["boot_ids"]
            == {
                "op12": by_device_role["OP12"]["boot_id"],
                "op15": by_device_role["OP15"]["boot_id"],
            },
            "run_manifest: phone observer device identity mismatch",
        )
    return {
        "artifact_roles": sorted(artifact_roles),
        "cache_regime": cache_regime,
        "campaign_binding": manifest["campaign_binding"],
        "device_roles": sorted(device_roles),
        "executor_ids": sorted(
            executor["executor_id"] for executor in runtime["executors"]),
        "mode": mode,
        "native_transport_cells": transport_gate["cells"],
        "performance_claim_authorized":
            transport_gate["performance_claim_authorized"]
            and gpu_observer["status"] == "PASS",
        "phone_observer_summary": phone_summary,
        "phone_route_instances": phone_route_instances,
        "route_qualifications": route_qualifications,
        "repeat_index": manifest["repeat_index"],
        "transport_threshold_ns": transport_gate["threshold_ns"],
        "run_id": run_id,
        "controller_authentication": controller_authentication,
        "status": "S40_RUN_MANIFEST_V6_VALID",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    try:
        result = validate_run_manifest(args.manifest, args.contract)
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}")
        return 2
    print(canonical_bytes(result).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
