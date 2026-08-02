#!/usr/bin/env python3
"""Run one isolated S40 physical acquisition and build its evidence root."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import http.client
import os
from pathlib import Path
import re
import secrets
import shutil
import socket
import stat
import subprocess
import sys
import time
from typing import Any, Callable
from urllib.parse import urlsplit

from acquire_trace import write_new_durable
from build_runtime_config import (
    build_runtime_config,
    validate_runtime_plan_template,
)
from campaign_plan import read_campaign
from evidence_common import (
    EvidenceError,
    canonical_bytes,
    digest_bytes,
    digest_file,
    parse_json,
    read_json,
    require,
    require_int,
    require_string,
)
from run_manifest import (
    qualification_source_records,
    validate_qualification_route_identity,
    validate_run_manifest,
)
from bridge_overhead import validate_combined_measurement
from gpu_isolation import (
    canonical_lock_path,
    observation_commands,
    parse_gpu_identity,
)
from resource_sampler import parse_process_stat
from validate_inputs import (
    DEFAULT_CONTRACT,
    validate_contract,
    validate_serving_argv,
)

EXECUTOR_DIR = Path(__file__).resolve().parent / "executors"
if str(EXECUTOR_DIR) not in sys.path:
    sys.path.insert(0, str(EXECUTOR_DIR))

from executor_bundle import (  # noqa: E402
    build_executor_bundle,
    validate_executor_bundle as validate_python_executor_bundle,
)
from evidence_bundle import (  # noqa: E402
    build_bundle as build_evidence_bundle,
    validate_bundle as validate_evidence_bundle,
)
from runtime_binding import (  # noqa: E402
    RuntimeBindingError,
    read_controller_binding_evidence_capture,
)


HERE = Path(__file__).resolve().parent
PLAN_KEYS = {
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
PLAN_CAMPAIGN_BINDING_KEYS = {
    "campaign_path",
    "campaign_sha256",
    "order",
    "phase",
}
MANIFEST_CAMPAIGN_BINDING_KEYS = {
    "campaign_id",
    "campaign_sha256",
    "order",
    "phase",
}
GATEWAY_KEYS = {
    "argv",
    "executor_id",
    "socket_name",
}
DEVICE_KEYS = {
    "boot_id",
    "device_role",
    "stable_id",
}
PHONE_MODES = {"T1_PHONE_WARM_TIER", "T2_PHONE_NO_PROMOTION"}
CACHE_REGIME_TO_EXECUTOR = {
    "WARM_HOST_CACHE": "WARM_CACHE",
    "COLD_NVME": "COLD_NVME",
}
MODE_DEVICE_ROLES = {
    "C1_GPU_ONLY_OPTIMIZED": {"GPU"},
    "C2_GPU_PLUS_CPU_WARM_EXECUTOR": {"CPU", "GPU"},
    "C3_DUAL_PARTIAL_OFFLOAD": {"CPU", "GPU"},
    "T1_PHONE_WARM_TIER": {"GPU", "OP12", "OP15"},
    "T2_PHONE_NO_PROMOTION": {"GPU", "OP12", "OP15"},
}
SOURCE_LOCK_KEY = "_validated_source_lock"
BINARY_LOCK_KEY = "_validated_binary_lock"
SOURCE_PREFLIGHT_SCHEMA = "s40-source-preflight-v1"
RUNTIME_DEPENDENCY_SCHEMA = "s40-captured-runtime-v1"
LDD_ADDRESS = re.compile(r"\s+\(0x[0-9a-fA-F]+\)$")
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


def command_array(value: Any, field: str) -> list[str]:
    require(
        isinstance(value, list)
        and value
        and len(value) <= 128
        and all(
            isinstance(argument, str)
            and argument
            and argument.isascii()
            and len(argument) <= 16 * 1024
            for argument in value
        ),
        f"{field}: invalid command array",
    )
    result = list(value)
    executable = Path(result[0])
    require(executable.is_absolute() and executable.is_file(),
            f"{field}: executable is missing")
    return result


def deterministic_launch_environment(
        output: Path,
        private_library_path: Path,
        runtime_config_path: Path,
        captured_nvidia_smi: Path,
        selected_gpu_uuid: str,
        executor_environment: dict[str, str],
        evidence_environment: dict[str, str]) -> dict[str, str]:
    home = output / "run-home"
    temporary = output / "run-tmp"
    cuda_cache = output / "cuda-cache"
    for path in (home, temporary, cuda_cache):
        if not path.exists():
            mkdir_new(path)
    environment = {
        "CUDA_CACHE_PATH": str(cuda_cache),
        "CUDA_VISIBLE_DEVICES": selected_gpu_uuid,
        "HOME": str(home),
        "LANG": "C",
        "LC_ALL": "C",
        "LD_LIBRARY_PATH": str(private_library_path),
        "LLAMA_SERVER_WARM_TIER_CONFIG": str(runtime_config_path),
        "NVIDIA_VISIBLE_DEVICES": selected_gpu_uuid,
        "PATH": f"{output / 'captured-runtime' / 'bin'}:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "S40_EVIDENCE_BUNDLE": "1",
        "S40_EVIDENCE_BUNDLE_MANIFEST":
            str(output / "evidence-bundle" / "MANIFEST.json"),
        "S40_EVIDENCE_BUNDLE_SHA256":
            digest_file(output / "evidence-bundle" / "MANIFEST.json"),
        "S40_NVIDIA_SMI_PATH": str(captured_nvidia_smi),
        "S40_NVIDIA_SMI_SHA256": digest_file(captured_nvidia_smi),
        "TMPDIR": str(temporary),
        "TZ": "UTC",
    }
    environment.update(executor_environment)
    environment.update(evidence_environment)
    require(
        set(environment) <= BASE_LAUNCH_ENV_KEYS
        and not (set(environment) & PROHIBITED_LAUNCH_ENV)
        and all(
            isinstance(key, str)
            and isinstance(value, str)
            and key
            and value
            and key.isascii()
            and value.isascii()
            and "\x00" not in key
            and "\x00" not in value
            for key, value in environment.items()
        ),
        "launch environment: non-allowlisted entry",
    )
    require(
        environment["LD_LIBRARY_PATH"]
        == str(private_library_path.resolve())
        and environment["S40_EXECUTOR_BUNDLE_MANIFEST"]
        == str((output / "executor-bundle" / "MANIFEST.json").resolve())
        and environment["S40_EVIDENCE_BUNDLE_MANIFEST"]
        == str((output / "evidence-bundle" / "MANIFEST.json").resolve()),
        "launch environment: code path is not captured",
    )
    return dict(sorted(environment.items()))


def privileged_launch_environment(
        environment: dict[str, str],
        internal_token_path: Path) -> dict[str, str]:
    require(
        set(environment) == BASE_LAUNCH_ENV_KEYS
        and internal_token_path.is_absolute(),
        "warm-tier privileged launch environment",
    )
    result = dict(environment)
    result["LLAMA_SERVER_WARM_TIER_INTERNAL_TOKEN_FILE"] = str(
        internal_token_path)
    return dict(sorted(result.items()))


def create_internal_token_file(path: Path) -> None:
    require(
        path.is_absolute() and not os.path.lexists(path),
        "warm-tier internal token path",
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            flags,
            0o600,
        )
        os.fchmod(descriptor, 0o600)
        raw = (secrets.token_hex(32) + "\n").encode("ascii")
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        require(
            stat.S_ISREG(metadata.st_mode)
            and metadata.st_uid == os.geteuid()
            and metadata.st_nlink == 1
            and metadata.st_mode & 0o777 == 0o600
            and metadata.st_size == 65,
            "warm-tier internal token metadata",
        )
        os.close(descriptor)
        descriptor = -1
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def remove_internal_token_file(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    safe_metadata = (
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_uid == os.geteuid()
        and metadata.st_nlink == 1
        and metadata.st_mode & 0o077 == 0
    )
    path.unlink()
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    require(
        safe_metadata,
        "warm-tier internal token cleanup metadata",
    )


def run_cleanup_actions(
        actions: list[tuple[str, Callable[[], None]]]) -> None:
    failures: list[tuple[str, BaseException]] = []
    for name, action in actions:
        try:
            action()
        except BaseException as error:
            failures.append((name, error))
    if failures:
        detail = "; ".join(
            f"{name}: {type(error).__name__}: {error}"
            for name, error in failures
        )
        raise EvidenceError(f"cleanup failed: {detail}") from failures[0][1]


def remove_temporary_publication(record: dict[str, Any]) -> None:
    temporary_path = Path(record["temporary_path"])
    if not temporary_path.exists():
        return
    temporary_path.unlink()
    directory_fd = os.open(
        temporary_path.parent,
        os.O_RDONLY | os.O_DIRECTORY,
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def sync_close_stream(stream: Any) -> None:
    try:
        stream.flush()
        os.fsync(stream.fileno())
    finally:
        stream.close()


def stop_gpu_observer_process(
        process: subprocess.Popen,
        stop_path: Path) -> None:
    if not stop_path.exists():
        write_new_durable(
            stop_path,
            b"stop\n",
            "GPU observer stop",
        )
    try:
        if process.wait(timeout=30) != 0:
            stop_process(process, "GPU observer")
    except subprocess.TimeoutExpired:
        stop_process(process, "GPU observer")


def isolated_python_argv(
        captured_python: Path,
        bundle: dict[str, Any],
        script: Path,
        arguments: list[str]) -> list[str]:
    prefix = list(bundle["python_argv_prefix"])
    require(
        len(prefix) == 7
        and prefix[1:4] == ["-I", "-S", "-B"]
        and prefix[4] == "-c"
        and Path(prefix[6]).resolve() == script.parent.resolve(),
        "isolated Python bundle prefix mismatch",
    )
    prefix[0] = str(captured_python)
    return prefix + [str(script), *arguments]


def optional_command(value: Any, field: str) -> list[str] | None:
    if value is None:
        return None
    return command_array(value, field)


def flag_value(argv: list[str], flag: str, field: str) -> str:
    positions = [index for index, value in enumerate(argv) if value == flag]
    require(
        len(positions) == 1 and positions[0] + 1 < len(argv),
        f"{field}: {flag} must occur exactly once",
    )
    return argv[positions[0] + 1]


def _snapshot_source(
        rows: list[dict[str, Any]],
        role: str,
        path: Path,
        expected_sha256: str | None = None) -> None:
    require(
        role.isascii() and 0 < len(role) <= 256,
        f"source lock {role}: invalid role",
    )
    require(
        all(row["role"] != role for row in rows),
        f"source lock {role}: duplicate role",
    )
    resolved = path.resolve()
    require(resolved.is_file(), f"source lock {role}: source is missing")
    raw = resolved.read_bytes()
    require(raw, f"source lock {role}: source is empty")
    sha256 = digest_bytes(raw)
    if expected_sha256 is not None:
        require(
            sha256 == expected_sha256,
            f"source lock {role}: referenced digest mismatch",
        )
    rows.append({
        "bytes": len(raw),
        "path": str(resolved),
        "raw": raw,
        "role": role,
        "sha256": sha256,
    })


def _source_row(
        rows: list[dict[str, Any]],
        role: str) -> dict[str, Any]:
    matches = [row for row in rows if row["role"] == role]
    require(len(matches) == 1, f"source lock {role}: missing row")
    return matches[0]


def _snapshot_json(
        rows: list[dict[str, Any]],
        role: str,
        path: Path,
        expected_sha256: str | None = None) -> dict[str, Any]:
    _snapshot_source(rows, role, path, expected_sha256)
    value = parse_json(_source_row(rows, role)["raw"], f"source lock {role}")
    require(isinstance(value, dict), f"source lock {role}: expected object")
    return value


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
        "dependency resolver changed before execution",
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
            "PATH": "/usr/bin:/bin",
        },
        timeout=30,
    )
    require(
        digest_file(ldd_path) == ldd_sha256,
        "dependency resolver changed during execution",
    )
    require(
        result.returncode == 0 and not result.stderr,
        f"binary dependency resolution failed: {label}",
    )
    try:
        text = result.stdout.decode("ascii")
    except UnicodeDecodeError as error:
        raise EvidenceError(
            f"binary dependency resolution is not ASCII: {label}") from error
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
                f"binary dependency is unresolved: {needed_name}",
            )
            resolved_text = LDD_ADDRESS.sub("", resolution)
        else:
            resolved_text = LDD_ADDRESS.sub("", value)
            needed_name = Path(resolved_text).name
        resolved = Path(resolved_text).resolve()
        require(
            needed_name
            and Path(needed_name).name == needed_name
            and needed_name.isascii()
            and resolved.is_absolute()
            and resolved.is_file(),
            f"binary dependency is invalid: {label}",
        )
        key = (needed_name, str(resolved))
        require(key not in seen, f"binary dependency is duplicated: {label}")
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


def _capture_binary_lock(
        plan: dict[str, Any],
        rows: list[dict[str, Any]]) -> dict[str, Any]:
    ldd_path = Path(plan["ldd_path"]).resolve()
    _snapshot_source(rows, "runtime_tool::ldd", ldd_path)
    ldd_sha256 = _source_row(rows, "runtime_tool::ldd")["sha256"]
    binaries = {
        "controller": Path(plan["server_argv"][0]).resolve(),
        "native_bench": (
            Path(plan["server_argv"][0]).resolve().parent
            / "test-warm-tier-executors"
        ),
        "nvidia_smi": Path(plan["nvidia_smi_path"]).resolve(),
        "python": Path(sys.executable).resolve(),
    }
    for label, source in binaries.items():
        require(
            source.is_file() and os.access(source, os.X_OK),
            f"runtime binary is missing: {label}",
        )
        _snapshot_source(rows, f"runtime_binary::{label}", source)

    dependencies: dict[tuple[str, str], dict[str, Any]] = {}
    for label, source in binaries.items():
        for record in _resolve_binary_dependencies(
                source, label, ldd_path, ldd_sha256):
            key = (record["needed_name"], record["source_path"])
            existing = dependencies.get(key)
            if existing is None:
                dependencies[key] = {
                    "bytes": record["bytes"],
                    "consumers": [label],
                    "needed_name": record["needed_name"],
                    "sha256": record["sha256"],
                    "source_path": record["source_path"],
                    "system": record["system"],
                }
            else:
                existing["consumers"].append(label)
    by_name: dict[str, dict[str, Any]] = {}
    for record in dependencies.values():
        name = record["needed_name"]
        previous = by_name.get(name)
        require(
            previous is None
            or (
                previous["source_path"] == record["source_path"]
                and previous["sha256"] == record["sha256"]
            ),
            f"runtime dependency resolves ambiguously: {name}",
        )
        by_name[name] = record
    dependency_rows = []
    for name, record in sorted(by_name.items()):
        record["consumers"].sort()
        dependency_rows.append(record)
        if not record["system"]:
            _snapshot_source(
                rows,
                f"runtime_dependency::{name}",
                Path(record["source_path"]),
                record["sha256"],
            )
    return {
        "binaries": {
            label: {
                "bytes": _source_row(
                    rows, f"runtime_binary::{label}")["bytes"],
                "sha256": _source_row(
                    rows, f"runtime_binary::{label}")["sha256"],
                "source_path": _source_row(
                    rows, f"runtime_binary::{label}")["path"],
            }
            for label in sorted(binaries)
        },
        "dependencies": dependency_rows,
        "ldd": {
            "bytes": _source_row(rows, "runtime_tool::ldd")["bytes"],
            "sha256": ldd_sha256,
            "source_path": str(ldd_path),
        },
    }


def _capture_source_lock(
        plan_path: Path,
        plan: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    _snapshot_source(rows, "physical_plan", plan_path)
    if plan["campaign_binding"] is not None:
        _snapshot_source(
            rows,
            "campaign_plan",
            Path(plan["campaign_binding"]["campaign_path"]),
            plan["campaign_binding"]["campaign_sha256"],
        )
    contract_path = Path(plan["contract_path"])
    _snapshot_source(rows, "experiment_contract", contract_path)
    source_plan = _snapshot_json(
        rows,
        "runtime_plan_template",
        Path(plan["runtime_plan_template"]),
    )
    source_root = _snapshot_json(
        rows,
        "evidence_root",
        Path(source_plan["evidence_root_path"]),
        source_plan["evidence_root_sha256"],
    )
    if source_plan["c3_profile_lock_path"] is not None:
        _snapshot_source(
            rows,
            "c3_profile_lock",
            Path(source_plan["c3_profile_lock_path"]),
            source_plan["c3_profile_lock_sha256"],
        )
    for model_id, record in sorted(
            source_root["c3_placement_artifacts"].items()):
        _snapshot_source(
            rows,
            f"c3_placement::{model_id}",
            Path(record["path"]),
            record["sha256"],
        )
    for record in sorted(
            source_root["executor_configs"],
            key=lambda item: item["executor_id"]):
        executor_id = record["executor_id"]
        role = f"gateway_config::{executor_id}"
        config = _snapshot_json(
            rows,
            role,
            Path(record["gateway_config_path"]),
            record["gateway_config_sha256"],
        )
        schema = config.get("schema")
        require(
            schema in {
                "s40-desktop-executor-config-v4",
                "s40-phone-route-config-v3",
            },
            f"source lock {role}: acquisition config schema mismatch",
        )
        routes = config.get("routes", [])
        require(isinstance(routes, list),
                f"source lock {role}: routes must be an array")
        for route in routes:
            require(isinstance(route, dict),
                    f"source lock {role}: route must be an object")
            model_id = require_string(
                route.get("model_id"), f"source lock {role}.model_id")
            if schema.startswith("s40-desktop-executor-config-"):
                _snapshot_source(
                    rows,
                    f"artifact_certificate::{executor_id}::{model_id}",
                    Path(route["artifact_certificate_path"]),
                    route["artifact_certificate_sha256"],
                )
                _snapshot_source(
                    rows,
                    f"readiness_lock::{executor_id}::{model_id}",
                    Path(route["readiness_lock_path"]),
                    route["readiness_lock_sha256"],
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
                for binding in qualification_source_records(
                        qualification, executor_id, model_id):
                    _snapshot_source(
                        rows,
                        binding["role"],
                        Path(binding["path"]),
                        binding["sha256"],
                    )
    if plan["phone_identity_argv"] is not None:
        identity = plan["phone_identity_argv"]
        telemetry = plan["phone_telemetry_argv"]
        identity_config = Path(flag_value(
            identity, "--ssh-config", "phone_identity_argv"))
        telemetry_config = Path(flag_value(
            telemetry, "--ssh-config", "phone_telemetry_argv"))
        require(
            identity_config.resolve() == telemetry_config.resolve(),
            "phone observer: SSH config mismatch",
        )
        _snapshot_source(rows, "phone_ssh_config", identity_config)
    plan[BINARY_LOCK_KEY] = _capture_binary_lock(plan, rows)
    return rows


def verify_source_lock(rows: list[dict[str, Any]]) -> None:
    require(isinstance(rows, list) and rows, "source lock: missing rows")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        field = f"source lock[{index}]"
        require(
            isinstance(row, dict)
            and set(row) == {"bytes", "path", "raw", "role", "sha256"},
            f"{field}: key set mismatch",
        )
        role = require_string(row["role"], f"{field}.role")
        require(role not in seen, f"{field}: duplicate role")
        seen.add(role)
        raw = row["raw"]
        require(isinstance(raw, bytes) and raw, f"{field}: invalid bytes")
        source = Path(require_string(row["path"], f"{field}.path"))
        require(
            source.is_absolute()
            and source.is_file()
            and source.stat().st_size == row["bytes"]
            and digest_bytes(raw) == row["sha256"]
            and source.read_bytes() == raw,
            f"source lock changed: {role}",
        )


def verify_binary_lock(plan: dict[str, Any]) -> None:
    rows = plan.get(SOURCE_LOCK_KEY)
    lock = plan.get(BINARY_LOCK_KEY)
    require(
        isinstance(rows, list) and isinstance(lock, dict),
        "binary lock: plan was not validated",
    )
    verify_source_lock(rows)
    ldd = lock.get("ldd")
    require(
        isinstance(ldd, dict)
        and set(ldd) == {"bytes", "sha256", "source_path"},
        "binary lock: dependency resolver lock is missing",
    )
    ldd_path = Path(ldd["source_path"])
    require(
        ldd_path.resolve() == Path(plan["ldd_path"]).resolve()
        and ldd_path.stat().st_size == ldd["bytes"]
        and digest_file(ldd_path) == ldd["sha256"],
        "binary lock changed: ldd",
    )
    observed = {}
    for label, record in lock["binaries"].items():
        source = Path(record["source_path"])
        require(
            source.is_file()
            and source.stat().st_size == record["bytes"]
            and digest_file(source) == record["sha256"],
            f"binary lock changed: {label}",
        )
        observed[label] = _resolve_binary_dependencies(
            source, label, ldd_path, ldd["sha256"])
    normalized: dict[tuple[str, str], dict[str, Any]] = {}
    for records in observed.values():
        for record in records:
            key = (record["needed_name"], record["source_path"])
            existing = normalized.get(key)
            if existing is None:
                normalized[key] = {
                    "bytes": record["bytes"],
                    "consumers": [record["consumer"]],
                    "needed_name": record["needed_name"],
                    "sha256": record["sha256"],
                    "source_path": record["source_path"],
                    "system": record["system"],
                }
            else:
                existing["consumers"].append(record["consumer"])
    for record in normalized.values():
        record["consumers"].sort()
    require(
        sorted(
            normalized.values(),
            key=lambda item: (item["needed_name"], item["source_path"]),
        )
        == lock["dependencies"],
        "binary dependency resolution changed",
    )


def materialize_source_preflight(
        rows: list[dict[str, Any]],
        output: Path) -> Path:
    verify_source_lock(rows)
    directory = output / "source-locks"
    mkdir_new(directory)
    records = []
    for index, row in enumerate(rows):
        safe_role = "".join(
            character if character.isalnum() else "-"
            for character in row["role"]
        )
        target = directory / f"{index:03d}-{safe_role}.artifact"
        write_new_durable(
            target,
            row["raw"],
            f"source lock {row['role']}",
        )
        records.append({
            "bytes": row["bytes"],
            "captured_path": str(target.relative_to(output)),
            "role": row["role"],
            "sha256": row["sha256"],
            "source_path": row["path"],
        })
    path = output / "source-preflight.json"
    write_new_durable(
        path,
        canonical_bytes({
            "captured_ns": time.monotonic_ns(),
            "rows": records,
            "schema": SOURCE_PREFLIGHT_SCHEMA,
        }),
        "source_preflight",
    )
    return path


def validate_physical_plan(path: Path) -> dict[str, Any]:
    plan = read_json(path, "physical_plan")
    require(path.read_bytes() == canonical_bytes(plan),
            "physical_plan: not canonical JSON")
    require(set(plan) == PLAN_KEYS, "physical_plan: key set mismatch")
    require(plan["schema"] == "s40-physical-run-plan-v2",
            "physical_plan: schema mismatch")
    contract_path = Path(require_string(
        plan["contract_path"], "physical_plan.contract_path"))
    require(contract_path.is_absolute() and contract_path.is_file(),
            "physical_plan: contract path is missing")
    contract = read_json(contract_path, "contract")
    validate_contract(contract_path)
    ldd_path = Path(require_string(
        plan["ldd_path"], "physical_plan.ldd_path"))
    require(
        ldd_path.is_absolute()
        and ldd_path.is_file()
        and os.access(ldd_path, os.X_OK),
        "physical_plan: ldd executable is missing",
    )
    mode = require_string(plan["mode"], "physical_plan.mode")
    require(
        mode in contract["matrix"] and mode != "C4_TWO_GPU_ORACLE",
        "physical_plan: unsupported mode",
    )
    require(
        plan["cache_regime"] in contract["matrix"][mode]["cache_regimes"],
        "physical_plan: invalid cache regime",
    )
    require(isinstance(plan["development"], bool),
            "physical_plan.development: expected bool")
    require_int(plan["repeat_index"], "physical_plan.repeat_index")
    run_id = require_string(plan["run_id"], "physical_plan.run_id")
    require(run_id.isascii() and len(run_id) <= 128,
            "physical_plan.run_id: invalid value")
    campaign_binding = plan["campaign_binding"]
    if plan["development"]:
        require(
            campaign_binding is None,
            "physical_plan: development run cannot bind a primary campaign",
        )
    else:
        require(
            isinstance(campaign_binding, dict)
            and set(campaign_binding) == PLAN_CAMPAIGN_BINDING_KEYS,
            "physical_plan: primary campaign binding is required",
        )
        campaign_path = Path(require_string(
            campaign_binding["campaign_path"],
            "physical_plan.campaign_binding.campaign_path",
        ))
        require(
            campaign_path.is_absolute() and campaign_path.is_file(),
            "physical_plan: campaign path is missing",
        )
        campaign = read_campaign(campaign_path)
        require(
            digest_file(campaign_path) == campaign_binding["campaign_sha256"]
            and campaign["experiment_contract_sha256"]
            == digest_file(contract_path),
            "physical_plan: campaign digest or contract mismatch",
        )
        order = require_int(
            campaign_binding["order"],
            "physical_plan.campaign_binding.order",
        )
        phase = require_string(
            campaign_binding["phase"],
            "physical_plan.campaign_binding.phase",
        )
        rows = [
            row for row in campaign["primary"]
            if row["order"] == order
        ]
        require(
            len(rows) == 1
            and rows[0]["phase"] == phase
            and rows[0]["run_id"] == run_id
            and rows[0]["mode"] == mode
            and rows[0]["cache_regime"] == plan["cache_regime"]
            and rows[0]["repeat_index"] == plan["repeat_index"],
            "physical_plan: campaign row mismatch",
        )
        software = campaign["software_lock"]
        require(
            digest_file(Path(plan["server_argv"][0]))
            == software["controller_binary_sha256"]
            and digest_file(
                Path(plan["server_argv"][0]).resolve().parent
                / "test-warm-tier-executors"
            ) == software["native_bench_binary_sha256"],
            "physical_plan: campaign software lock mismatch",
        )
        require(
            digest_file(Path(plan["nvidia_smi_path"]))
            == software["nvidia_smi_sha256"],
            "physical_plan: campaign nvidia-smi lock mismatch",
        )
        require(
            digest_file(ldd_path) == software["ldd_sha256"]
            and digest_file(Path(sys.executable))
            == software["python_sha256"],
            "physical_plan: campaign tool lock mismatch",
        )
    output = Path(require_string(
        plan["output_dir"], "physical_plan.output_dir"))
    require(output.is_absolute() and not output.exists(),
            "physical_plan: output directory already exists")
    runtime_template = Path(require_string(
        plan["runtime_plan_template"],
        "physical_plan.runtime_plan_template",
    ))
    require(runtime_template.is_absolute() and runtime_template.is_file(),
            "physical_plan: runtime plan template is missing")
    source_runtime = validate_runtime_plan_template(
        runtime_template, contract_path)
    require(source_runtime["mode"] == mode,
            "physical_plan: runtime mode mismatch")
    require(
        source_runtime["run_id"] == run_id,
        "physical_plan: runtime template run ID mismatch",
    )

    server_argv = command_array(
        plan["server_argv"], "physical_plan.server_argv")
    validate_serving_argv(
        server_argv,
        contract["runtime_source"]["serving"],
        "physical_plan.server_argv",
    )
    nvidia_smi = Path(require_string(
        plan["nvidia_smi_path"], "physical_plan.nvidia_smi_path"))
    require(
        nvidia_smi.is_absolute()
        and nvidia_smi.is_file()
        and os.access(nvidia_smi, os.X_OK),
        "physical_plan: nvidia-smi executable is missing",
    )
    required_threads = str(
        contract["run_manifest_requirements"]["required_http_threads"])
    require(
        flag_value(
            server_argv, "--threads-http", "physical_plan.server_argv")
        == required_threads,
        "physical_plan: server HTTP thread count mismatch",
    )
    parsed = urlsplit(require_string(
        plan["server_base_url"], "physical_plan.server_base_url"))
    require(
        parsed.scheme == "http"
        and parsed.hostname == "127.0.0.1"
        and parsed.port is not None
        and 1 <= parsed.port <= 65535
        and parsed.path in {"", "/"}
        and not parsed.query
        and not parsed.fragment,
        "physical_plan: server base URL must be loopback HTTP",
    )
    require(
        flag_value(server_argv, "--host", "physical_plan.server_argv")
        == "127.0.0.1"
        and flag_value(server_argv, "--port", "physical_plan.server_argv")
        == str(parsed.port),
        "physical_plan: server command and base URL differ",
    )
    for index, executor in enumerate(source_runtime["executors"]):
        field = f"runtime.executors[{index}]"
        require(
            executor["transport"] == "UNIX_SOCKET"
            and Path(executor["socket_path"]).is_absolute(),
            f"{field}: native Unix transport is required",
        )

    gateways = plan["gateway_processes"]
    require(isinstance(gateways, list), "physical_plan.gateways: expected array")
    expected_ids = {
        item["executor_id"]
        for item in source_runtime["executors"]
    }
    seen = set()
    instance_ids = set()
    expected_runtime_path = output / "runtime-config.json"
    expected_controller_identity_path = (
        output / "controller-identity-lock.json")
    for index, gateway in enumerate(gateways):
        field = f"physical_plan.gateway_processes[{index}]"
        require(
            isinstance(gateway, dict) and set(gateway) == GATEWAY_KEYS,
            f"{field}: key set mismatch",
        )
        executor_id = require_string(
            gateway["executor_id"], f"{field}.executor_id")
        require(
            executor_id in expected_ids and executor_id not in seen,
            f"{field}: unknown or duplicate executor",
        )
        seen.add(executor_id)
        gateway_argv = command_array(gateway["argv"], f"{field}.argv")
        require(
            flag_value(gateway_argv, "--run-id", f"{field}.argv")
            == run_id,
            f"{field}: run ID mismatch",
        )
        require(
            Path(flag_value(
                gateway_argv,
                "--runtime-config",
                f"{field}.argv",
            )).resolve() == expected_runtime_path.resolve(),
            f"{field}: runtime config path mismatch",
        )
        require(
            Path(flag_value(
                gateway_argv,
                "--controller-identity",
                f"{field}.argv",
            )).resolve() == expected_controller_identity_path.resolve(),
            f"{field}: controller identity path mismatch",
        )
        require(
            Path(flag_value(
                gateway_argv,
                "--controller-binding-evidence",
                f"{field}.argv",
            )).resolve()
            == (
                output
                / f"executor-controller-binding-{executor_id}.json"
            ).resolve(),
            f"{field}: controller binding path mismatch",
        )
        require(
            "--runtime-config-sha256" not in gateway_argv,
            f"{field}: legacy runtime digest flag is forbidden",
        )
        instance_id = flag_value(
            gateway_argv, "--executor-instance-id", f"{field}.argv")
        require(
            instance_id.isascii()
            and len(instance_id) <= 256
            and all(0x21 <= ord(character) <= 0x7e
                    for character in instance_id)
            and instance_id not in instance_ids,
            f"{field}: invalid or duplicate executor instance ID",
        )
        instance_ids.add(instance_id)
        socket_name = require_string(
            gateway["socket_name"], f"{field}.socket_name")
        require(
            Path(socket_name).name == socket_name
            and socket_name not in {".", ".."},
            f"{field}: invalid socket name",
        )
    require(seen == expected_ids, "physical_plan: gateway set mismatch")

    identity = optional_command(
        plan["phone_identity_argv"],
        "physical_plan.phone_identity_argv",
    )
    telemetry = optional_command(
        plan["phone_telemetry_argv"],
        "physical_plan.phone_telemetry_argv",
    )
    if mode in PHONE_MODES:
        require(identity is not None and telemetry is not None,
                "physical_plan: phone evidence commands are required")
        require(
            Path(identity[0]).resolve() == Path(sys.executable).resolve()
            and Path(telemetry[0]).resolve() == Path(sys.executable).resolve()
            and Path(identity[1]).name == "phone_observer_bridge.py"
            and Path(telemetry[1]).name == "phone_observer_bridge.py"
            and flag_value(
                identity, "--action", "physical_plan.phone_identity_argv")
            == "identity"
            and flag_value(
                telemetry, "--action", "physical_plan.phone_telemetry_argv")
            == "telemetry"
            and flag_value(
                identity, "--run-id", "physical_plan.phone_identity_argv")
            == run_id
            and flag_value(
                telemetry, "--run-id", "physical_plan.phone_telemetry_argv")
            == run_id
            and flag_value(
                identity, "--model", "physical_plan.phone_identity_argv")
            == flag_value(
                telemetry, "--model", "physical_plan.phone_telemetry_argv")
            and Path(flag_value(
                identity,
                "--ssh-config",
                "physical_plan.phone_identity_argv",
            )).resolve()
            == Path(flag_value(
                telemetry,
                "--ssh-config",
                "physical_plan.phone_telemetry_argv",
            )).resolve(),
            "physical_plan: phone observer commands mismatch",
        )
    else:
        require(identity is None and telemetry is None,
                "physical_plan: unexpected phone evidence commands")
    devices = plan["devices"]
    require(isinstance(devices, list) and devices,
            "physical_plan.devices: expected array")
    roles = set()
    stable_ids = set()
    for index, device in enumerate(devices):
        field = f"physical_plan.devices[{index}]"
        require(
            isinstance(device, dict) and set(device) == DEVICE_KEYS,
            f"{field}: key set mismatch",
        )
        role = require_string(device["device_role"], f"{field}.device_role")
        require(role not in roles, f"{field}: duplicate role")
        roles.add(role)
        stable_id = require_string(
            device["stable_id"], f"{field}.stable_id")
        require(stable_id not in stable_ids, f"{field}: duplicate stable ID")
        stable_ids.add(stable_id)
        require_string(device["boot_id"], f"{field}.boot_id")
    require(
        roles == MODE_DEVICE_ROLES[mode],
        "physical_plan: device role set mismatch",
    )
    selected_gpu = next(
        device for device in devices if device["device_role"] == "GPU")
    require(
        selected_gpu["stable_id"]
        == contract["runtime_source"]["expected_gpu_uuid"],
        "physical_plan: selected GPU mismatch",
    )
    gpu_lock_path = Path(require_string(
        plan["gpu_lock_path"], "physical_plan.gpu_lock_path"))
    require(
        gpu_lock_path == canonical_lock_path(selected_gpu["stable_id"]),
        "physical_plan: noncanonical GPU lock path",
    )
    require(
        gpu_lock_path.parent.is_dir() and not gpu_lock_path.is_symlink(),
        "physical_plan: invalid canonical GPU lock path",
    )
    gpu_index = require_int(
        plan["gpu_index"], "physical_plan.gpu_index", 0)
    gpu_pci_bus_id = require_string(
        plan["gpu_pci_bus_id"], "physical_plan.gpu_pci_bus_id")
    identity_argv, _ = observation_commands(
        nvidia_smi, selected_gpu["stable_id"])
    identity_probe = subprocess.run(
        identity_argv,
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
    )
    require(
        identity_probe.returncode == 0 and not identity_probe.stderr,
        "physical_plan: selected GPU identity query failed",
    )
    identity_value = parse_gpu_identity(
        identity_probe.stdout,
        selected_gpu["stable_id"],
        contract["runtime_source"]["expected_gpu_name"],
    )
    require(
        identity_value["gpu_index"] == gpu_index
        and identity_value["gpu_pci_bus_id"] == gpu_pci_bus_id,
        "physical_plan: selected GPU mapping mismatch",
    )
    plan[SOURCE_LOCK_KEY] = _capture_source_lock(path, plan)
    return plan


def require_controller_port_available(base_url: str) -> None:
    parsed = urlsplit(base_url)
    require(parsed.hostname == "127.0.0.1" and parsed.port is not None,
            "controller port: invalid base URL")
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind((parsed.hostname, parsed.port))
    except OSError as error:
        raise EvidenceError(
            f"controller port: pre-existing listener on {parsed.port}") from error
    finally:
        listener.close()


def mkdir_new(path: Path) -> None:
    path.mkdir(mode=0o700, parents=False, exist_ok=False)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def copy_new(source: Path, target: Path) -> None:
    require(source.is_file(), f"copy: source is missing {source}")
    require(not target.exists(), f"copy: target exists {target}")
    with source.open("rb") as inp, target.open("xb", buffering=0) as out:
        shutil.copyfileobj(inp, out, 8 * 1024 * 1024)
        out.flush()
        os.fsync(out.fileno())


def write_locked_runtime_file(
        target: Path,
        raw: bytes,
        mode: int) -> None:
    write_new_durable(target, raw, f"runtime file {target.name}")
    target.chmod(mode)
    with target.open("rb") as source:
        os.fsync(source.fileno())
    fsync_directory(target.parent)


def materialize_runtime_binaries(
        physical_plan: dict[str, Any],
        output: Path) -> tuple[dict[str, Path], Path]:
    verify_binary_lock(physical_plan)
    source_rows = physical_plan[SOURCE_LOCK_KEY]
    lock = physical_plan[BINARY_LOCK_KEY]
    runtime_root = output / "captured-runtime"
    binary_root = runtime_root / "bin"
    library_root = runtime_root / "lib"
    mkdir_new(runtime_root)
    mkdir_new(binary_root)
    mkdir_new(library_root)

    binary_names = {
        "controller": "llama-server",
        "native_bench": "test-warm-tier-executors",
        "nvidia_smi": "nvidia-smi",
        "python": "python3",
    }
    binaries = {}
    binary_rows = []
    for label, name in sorted(binary_names.items()):
        source = _source_row(
            source_rows, f"runtime_binary::{label}")
        target = binary_root / name
        write_locked_runtime_file(target, source["raw"], 0o500)
        binaries[label] = target
        binary_rows.append({
            "bytes": target.stat().st_size,
            "captured_path": str(target.relative_to(output)),
            "label": label,
            "sha256": digest_file(target),
            "source_path": source["path"],
        })

    dependency_rows = []
    for record in lock["dependencies"]:
        captured_path = None
        if not record["system"]:
            source = _source_row(
                source_rows,
                f"runtime_dependency::{record['needed_name']}",
            )
            target = library_root / record["needed_name"]
            write_locked_runtime_file(target, source["raw"], 0o400)
            captured_path = str(target.relative_to(output))
        dependency_rows.append({
            **record,
            "captured_path": captured_path,
        })
    path = output / "runtime-dependencies.json"
    write_new_durable(
        path,
        canonical_bytes({
            "binaries": binary_rows,
            "dependencies": dependency_rows,
            "library_path": str(library_root),
            "schema": RUNTIME_DEPENDENCY_SCHEMA,
        }),
        "runtime_dependencies",
    )
    verify_binary_lock(physical_plan)
    return binaries, path


def replace_flag(argv: list[str], flag: str, value: str) -> list[str]:
    positions = [index for index, item in enumerate(argv) if item == flag]
    require(len(positions) == 1, f"command: {flag} must occur once")
    index = positions[0]
    require(index + 1 < len(argv), f"command: missing value for {flag}")
    result = list(argv)
    result[index + 1] = value
    return result


def materialize_inputs(
        physical_plan: dict[str, Any],
        output: Path) -> tuple[
            Path,
            Path,
            dict[str, list[str]],
            dict[str, Path],
            dict[str, Path],
            dict[str, Path],
            dict[str, Any],
            dict[str, Any],
            Path,
            dict[str, Path],
            Path,
            dict[str, list[str]],
            dict[str, Path],
            dict[str, Any],
        ]:
    source_rows = physical_plan.get(SOURCE_LOCK_KEY)
    require(isinstance(source_rows, list), "source lock: plan was not validated")
    verify_source_lock(source_rows)
    contract_path = Path(physical_plan["contract_path"])
    source_plan = parse_json(
        _source_row(source_rows, "runtime_plan_template")["raw"],
        "runtime_plan_template",
    )
    source_root = parse_json(
        _source_row(source_rows, "evidence_root")["raw"],
        "runtime_evidence_root",
    )
    require(isinstance(source_plan, dict),
            "runtime_plan_template: expected object")
    require(isinstance(source_root, dict),
            "runtime_evidence_root: expected object")
    inputs = output / "inputs"
    sockets = output / "sockets"
    mkdir_new(inputs)
    mkdir_new(sockets)
    source_preflight_path = materialize_source_preflight(source_rows, output)
    runtime_binaries, runtime_dependencies_path = (
        materialize_runtime_binaries(physical_plan, output))
    executor_bundle = build_executor_bundle(output / "executor-bundle")
    evidence_bundle = build_evidence_bundle(output / "evidence-bundle")
    def bundle_argv(argv: list[str], field: str) -> list[str]:
        require(
            len(argv) >= 2
            and Path(argv[1]).suffix == ".py",
            f"{field}: expected Python script argv",
        )
        entrypoint = output / "executor-bundle" / Path(argv[1]).name
        require(entrypoint.is_file(), f"{field}: unknown bundle entrypoint")
        return isolated_python_argv(
            runtime_binaries["python"],
            executor_bundle,
            entrypoint,
            argv[2:],
        )

    config_paths: dict[str, Path] = {}
    root_value = copy.deepcopy(source_root)
    executor_kinds = {
        record["executor_id"]: record["kind"]
        for record in source_plan["executors"]
    }
    for record in root_value["executor_configs"]:
        executor_id = record["executor_id"]
        target = inputs / f"gateway-config-{executor_id}.json"
        source_config_row = _source_row(
            source_rows, f"gateway_config::{executor_id}")
        if executor_kinds[executor_id] == "PHONE_WARM":
            write_new_durable(
                target,
                source_config_row["raw"],
                f"gateway_config.{executor_id}",
            )
        else:
            config = parse_json(
                source_config_row["raw"],
                f"gateway_config.{executor_id}",
            )
            require(isinstance(config, dict),
                    f"gateway_config.{executor_id}: expected object")
            require(
                config.get("schema") == "s40-desktop-executor-config-v4"
                and config.get("cache_regime")
                in {"WARM_CACHE", "COLD_NVME"},
                f"gateway_config.{executor_id}: desktop cache contract missing",
            )
            config["cache_regime"] = CACHE_REGIME_TO_EXECUTOR[
                physical_plan["cache_regime"]]
            config["nvidia_smi"] = {
                "bytes": runtime_binaries["nvidia_smi"].stat().st_size,
                "path": str(runtime_binaries["nvidia_smi"]),
                "sha256": digest_file(runtime_binaries["nvidia_smi"]),
            }
            write_new_durable(
                target,
                canonical_bytes(config),
                f"gateway_config.{executor_id}",
            )
        record["gateway_config_path"] = str(target)
        record["gateway_config_sha256"] = digest_file(target)
        config_paths[executor_id] = target

    for model_id, record in root_value["c3_placement_artifacts"].items():
        target = inputs / f"c3-placement-{model_id}.artifact"
        write_new_durable(
            target,
            _source_row(
                source_rows, f"c3_placement::{model_id}")["raw"],
            f"c3_placement.{model_id}",
        )
        record["path"] = str(target)
        record["sha256"] = digest_file(target)
    evidence_root_path = output / "evidence-root.json"
    write_new_durable(
        evidence_root_path,
        canonical_bytes(root_value),
        "evidence_root",
    )

    plan_value = copy.deepcopy(source_plan)
    plan_value["run_id"] = physical_plan["run_id"]
    plan_value["evidence_root_path"] = str(evidence_root_path)
    plan_value["evidence_root_sha256"] = digest_file(evidence_root_path)
    plan_value["event_log_path"] = str(output / "controller-events.jsonl")
    if plan_value["c3_profile_lock_path"] is not None:
        target = inputs / "c3-profile-lock.json"
        write_new_durable(
            target,
            _source_row(source_rows, "c3_profile_lock")["raw"],
            "c3_profile_lock",
        )
        plan_value["c3_profile_lock_path"] = str(target)
        plan_value["c3_profile_lock_sha256"] = digest_file(target)

    gateways = {
        record["executor_id"]: record
        for record in physical_plan["gateway_processes"]
    }
    gateway_argv: dict[str, list[str]] = {}
    for executor in plan_value["executors"]:
        executor_id = executor["executor_id"]
        socket_path = sockets / gateways[executor_id]["socket_name"]
        require(
            executor["transport"] == "UNIX_SOCKET",
            f"runtime.executors.{executor_id}: non-native transport",
        )
        executor["socket_path"] = str(socket_path)
        argv = bundle_argv(
            list(gateways[executor_id]["argv"]),
            f"gateway.{executor_id}.argv",
        )
        argv = replace_flag(
            argv,
            "--socket",
            str(socket_path),
        )
        config_flag = (
            "--route-config"
            if executor["kind"] == "PHONE_WARM" else "--config"
        )
        argv = replace_flag(
            argv, config_flag, str(config_paths[executor_id]))
        argv = replace_flag(
            argv,
            "--evidence",
            str(output / f"executor-{executor_id}.jsonl"),
        )
        if executor["kind"] == "PHONE_WARM":
            argv = replace_flag(
                argv,
                "--wire-evidence",
                str(output / f"executor-wire-{executor_id}.jsonl"),
            )
            argv = replace_flag(
                argv,
                "--route-evidence",
                str(output / f"executor-route-{executor_id}.jsonl"),
            )
        gateway_argv[executor_id] = argv

    runtime_plan_path = output / "runtime-plan.json"
    runtime_config_path = output / "runtime-config.json"
    for executor_id, argv in list(gateway_argv.items()):
        argv = replace_flag(
            argv, "--run-id", physical_plan["run_id"])
        argv = replace_flag(
            argv,
            "--runtime-config",
            str(runtime_config_path),
        )
        argv = replace_flag(
            argv,
            "--controller-identity",
            str(output / "controller-identity-lock.json"),
        )
        argv = replace_flag(
            argv,
            "--controller-binding-evidence",
            str(output / f"executor-controller-binding-{executor_id}.json"),
        )
        gateway_argv[executor_id] = argv
    gateway_argv_paths = {}
    transport_descriptor_paths = {}
    for executor in plan_value["executors"]:
        executor_id = executor["executor_id"]
        gateway_argv_path = output / f"gateway-argv-{executor_id}.json"
        write_new_durable(
            gateway_argv_path,
            canonical_bytes({
                "argv": gateway_argv[executor_id],
                "schema": "s40-gateway-argv-v4",
            }),
            "gateway_argv",
        )
        gateway_argv_paths[executor_id] = gateway_argv_path
        source_name = (
            "phone_gateway.py"
            if executor["kind"] == "PHONE_WARM" else "desktop_gateway.py"
        )
        gateway_source = output / "executor-bundle" / source_name
        require(gateway_source.is_file(),
                f"gateway {executor_id}: bundled source is missing")
        descriptor_path = output / (
            f"executor-transport-{executor_id}.json")
        transport_descriptor_paths[executor_id] = descriptor_path
    phone_argv: dict[str, list[str]] = {}
    phone_argv_paths: dict[str, Path] = {}
    phone_ssh_config_path = None
    if physical_plan["phone_identity_argv"] is not None:
        phone_ssh_config_path = inputs / "phone-ssh-config.json"
        write_new_durable(
            phone_ssh_config_path,
            _source_row(source_rows, "phone_ssh_config")["raw"],
            "phone_ssh_config",
        )
    for action, source_argv in (
            ("identity", physical_plan["phone_identity_argv"]),
            ("telemetry", physical_plan["phone_telemetry_argv"])):
        if source_argv is None:
            continue
        argv = bundle_argv(
            list(source_argv), f"phone_observer.{action}.argv")
        argv = replace_flag(
            argv, "--ssh-config", str(phone_ssh_config_path))
        phone_argv[action] = argv
        path = output / f"phone-{action}-argv.json"
        write_new_durable(
            path,
            canonical_bytes({
                "argv": argv,
                "schema": "s40-phone-observer-argv-v1",
            }),
            f"phone_{action}_argv",
        )
        phone_argv_paths[action] = path
    return (
        runtime_plan_path,
        runtime_config_path,
        gateway_argv,
        config_paths,
        gateway_argv_paths,
        transport_descriptor_paths,
        executor_bundle,
        evidence_bundle,
        source_preflight_path,
        runtime_binaries,
        runtime_dependencies_path,
        phone_argv,
        phone_argv_paths,
        plan_value,
    )


def finalize_runtime_identity(
        plan_value: dict[str, Any],
        runtime_plan_path: Path,
        runtime_config_path: Path,
        gateway_processes: dict[str, subprocess.Popen],
        gateway_argv: dict[str, list[str]],
        gateway_file_bindings: dict[str, list[dict[str, Any]]],
        output: Path,
        gateway_argv_paths: dict[str, Path],
        config_paths: dict[str, Path],
        transport_descriptor_paths: dict[str, Path],
        executor_bundle: dict[str, Any],
        executor_bundle_root: Path,
        contract_path: Path) -> tuple[
            dict[str, Any],
            dict[str, dict[str, Any]],
            dict[str, Any],
        ]:
    require(
        not runtime_plan_path.exists() and not runtime_config_path.exists(),
        "runtime identity: config was published before gateway identity",
    )
    value = copy.deepcopy(plan_value)
    require(
        value.get("schema") == "s40-runtime-config-plan-v2",
        "runtime identity: source template schema mismatch",
    )
    value["schema"] = "s40-runtime-config-plan-v3"
    by_id = {
        executor["executor_id"]: executor
        for executor in value["executors"]
    }
    require(
        set(by_id) == set(gateway_processes) == set(gateway_argv)
        == set(gateway_file_bindings),
        "runtime identity: executor process set mismatch",
    )
    identities: dict[str, dict[str, Any]] = {}
    for executor_id, executor in by_id.items():
        process = gateway_processes[executor_id]
        verify_argument_bindings(
            gateway_file_bindings[executor_id], output)
        process_identity = gateway_process_identity(
            process, gateway_argv[executor_id], executor_id)
        start_ticks = process_identity["gateway_start_time_ticks"]
        require(
            process_identity["executable_path"]
            == str(Path(gateway_argv[executor_id][0]).resolve())
            and process_identity["executable_sha256"]
            == digest_file(Path(gateway_argv[executor_id][0])),
            f"runtime identity: gateway {executor_id} executable mismatch",
        )
        instance_id = flag_value(
            gateway_argv[executor_id],
            "--executor-instance-id",
            f"gateway {executor_id}",
        )
        executor["executor_instance_id"] = instance_id
        executor["expected_peer_pid"] = process.pid
        executor["expected_peer_start_time_ticks"] = start_ticks
        identities[executor_id] = {
            "executor_id": executor_id,
            "executor_instance_id": instance_id,
            "gateway_pid": process.pid,
            "gateway_start_time_ticks": start_ticks,
            "identity_captured_ns": process_identity["observed_ns"],
            "process_identity": process_identity,
        }

    write_new_durable(
        runtime_plan_path,
        canonical_bytes(value),
        "runtime_plan",
    )
    runtime = build_runtime_config(
        runtime_plan_path, contract_path)["runtime"]
    runtime_raw = canonical_bytes(runtime)
    temporary = runtime_config_path.with_name(
        f".{runtime_config_path.name}.{os.getpid()}.tmp")
    require(
        not temporary.exists(),
        "runtime identity: stale temporary config",
    )
    try:
        with temporary.open("xb", buffering=0) as sink:
            sink.write(runtime_raw)
            sink.flush()
            os.fsync(sink.fileno())
        for executor_id, identity in identities.items():
            process = gateway_processes[executor_id]
            require(
                process.poll() is None
                and process.pid == identity["gateway_pid"],
                f"runtime identity: gateway {executor_id} exited before publish",
            )
            current = gateway_process_identity(
                process,
                gateway_argv[executor_id],
                executor_id,
                identity["process_identity"],
            )
            require(
                current["gateway_start_time_ticks"]
                == identity["gateway_start_time_ticks"],
                f"runtime identity: gateway {executor_id} changed before publish",
            )
        os.link(temporary, runtime_config_path)
        directory_fd = os.open(
            runtime_config_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    runtime_config_published_ns = time.monotonic_ns()
    runtime_config_sha256 = digest_file(runtime_config_path)
    runtime_config_stat = runtime_config_path.stat(follow_symlinks=False)
    require(
        stat.S_ISREG(runtime_config_stat.st_mode)
        and not runtime_config_path.is_symlink(),
        "runtime identity: published config is not a regular file",
    )
    runtime_by_id = {
        executor["executor_id"]: executor
        for executor in runtime["executors"]
    }
    for executor_id, identity in identities.items():
        executor = runtime_by_id[executor_id]
        require(
            executor["executor_instance_id"]
            == identity["executor_instance_id"]
            and executor["expected_peer_pid"] == identity["gateway_pid"]
            and executor["expected_peer_start_time_ticks"]
            == identity["gateway_start_time_ticks"],
            f"runtime identity: executor {executor_id} mismatch",
        )
    return runtime, identities, {
        "device": runtime_config_stat.st_dev,
        "inode": runtime_config_stat.st_ino,
        "path": str(runtime_config_path),
        "published_ns": runtime_config_published_ns,
        "sha256": runtime_config_sha256,
        "temporary_path": str(temporary),
    }


def publish_controller_identity(
        path: Path,
        process: subprocess.Popen,
        controller_started_ns: int,
        controller_executable: Path,
        run_id: str,
        host_boot_id: str,
        runtime_publication: dict[str, Any]) -> tuple[
            dict[str, Any],
            dict[str, Any],
        ]:
    require(
        not path.exists() and process.poll() is None and process.pid >= 2,
        "controller identity: invalid prepublication state",
    )
    pid = process.pid
    proc_root = Path(f"/proc/{pid}")
    stat_path = proc_root / "stat"
    require(stat_path.is_file(), "controller identity: process disappeared")
    _, start_ticks = parse_process_stat(
        stat_path.read_text(encoding="ascii"))
    proc_stat = proc_root.stat()
    executable_path = Path(os.readlink(proc_root / "exe")).resolve()
    require(
        executable_path == controller_executable.resolve()
        and digest_file(executable_path) == digest_file(controller_executable),
        "controller identity: executable mismatch",
    )
    identity_captured_ns = time.monotonic_ns()
    require(
        controller_started_ns <= identity_captured_ns,
        "controller identity: invalid capture interval",
    )
    value = {
        "controller_executable_path": str(executable_path),
        "controller_executable_sha256": digest_file(executable_path),
        "controller_gid": proc_stat.st_gid,
        "controller_pid": pid,
        "controller_start_time_ticks": start_ticks,
        "controller_uid": proc_stat.st_uid,
        "host_boot_id": host_boot_id,
        "run_id": run_id,
        "runtime_config_device": runtime_publication["device"],
        "runtime_config_inode": runtime_publication["inode"],
        "runtime_config_path": runtime_publication["path"],
        "runtime_config_sha256": runtime_publication["sha256"],
        "schema": "s40-controller-identity-lock-v1",
    }
    raw = canonical_bytes(value)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    require(
        not temporary.exists(),
        "controller identity: stale temporary lock",
    )
    try:
        with temporary.open("xb", buffering=0) as sink:
            sink.write(raw)
            sink.flush()
            os.fsync(sink.fileno())
        require(
            process.poll() is None and process.pid == pid,
            "controller identity: process exited before publish",
        )
        _, current_start_ticks = parse_process_stat(
            stat_path.read_text(encoding="ascii"))
        current_proc_stat = proc_root.stat()
        current_executable = Path(os.readlink(proc_root / "exe")).resolve()
        require(
            current_start_ticks == start_ticks
            and current_proc_stat.st_uid == proc_stat.st_uid
            and current_proc_stat.st_gid == proc_stat.st_gid
            and current_executable == executable_path
            and digest_file(current_executable)
            == value["controller_executable_sha256"],
            "controller identity: process changed before publish",
        )
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    published_ns = time.monotonic_ns()
    lock_stat = path.stat(follow_symlinks=False)
    require(
        stat.S_ISREG(lock_stat.st_mode)
        and not path.is_symlink()
        and digest_file(path) == hashlib.sha256(raw).hexdigest(),
        "controller identity: published lock mismatch",
    )
    return value, {
        "captured_ns": identity_captured_ns,
        "device": lock_stat.st_dev,
        "inode": lock_stat.st_ino,
        "path": str(path),
        "published_ns": published_ns,
        "sha256": digest_file(path),
        "temporary_path": str(temporary),
    }


def write_transport_descriptors(
        runtime: dict[str, Any],
        identities: dict[str, dict[str, Any]],
        post_auth_identities: list[dict[str, Any]],
        gateway_file_bindings: dict[str, list[dict[str, Any]]],
        gateway_environments: dict[str, dict[str, str]],
        runtime_publication: dict[str, Any],
        controller_identity: dict[str, Any],
        controller_publication: dict[str, Any],
        gateway_argv_paths: dict[str, Path],
        config_paths: dict[str, Path],
        controller_binding_paths: dict[str, Path],
        controller_binding_readiness: list[dict[str, Any]],
        transport_descriptor_paths: dict[str, Path],
        executor_bundle: dict[str, Any],
        executor_bundle_root: Path) -> None:
    runtime_by_id = {
        executor["executor_id"]: executor
        for executor in runtime["executors"]
    }
    require(
        set(runtime_by_id) == set(identities)
        == set(gateway_file_bindings)
        == set(gateway_argv_paths) == set(config_paths)
        == set(controller_binding_paths)
        == set(transport_descriptor_paths) == set(gateway_environments),
        "executor transport: identity set mismatch",
    )
    readiness_by_id = {
        row["executor_id"]: row
        for row in controller_binding_readiness
        if isinstance(row, dict) and "executor_id" in row
    }
    require(
        len(readiness_by_id) == len(controller_binding_readiness)
        and set(readiness_by_id) == set(runtime_by_id),
        "executor transport: controller binding readiness mismatch",
    )
    post_auth_by_id = {
        row["executor_id"]: row["identity"]
        for row in post_auth_identities
        if isinstance(row, dict)
        and set(row) == {"executor_id", "identity"}
    }
    require(
        len(post_auth_by_id) == len(post_auth_identities)
        and set(post_auth_by_id) == set(runtime_by_id),
        "executor transport: post-auth identity mismatch",
    )
    for executor_id, executor in runtime_by_id.items():
        identity = identities[executor_id]
        source_name = (
            "phone_gateway.py"
            if executor["role"] == "PHONE" else "desktop_gateway.py"
        )
        gateway_source = executor_bundle_root / source_name
        binding_path = controller_binding_paths[executor_id]
        try:
            _, binding_raw, binding_stat = (
                read_controller_binding_evidence_capture(binding_path)
            )
        except RuntimeBindingError as error:
            raise EvidenceError(
                f"executor transport: controller binding "
                f"{executor_id}: {error}"
            ) from error
        readiness = readiness_by_id[executor_id]
        require(
            readiness == {
                "device": binding_stat.st_dev,
                "executor_id": executor_id,
                "inode": binding_stat.st_ino,
                "path": str(binding_path),
                "sha256": hashlib.sha256(binding_raw).hexdigest(),
            },
            f"executor transport: controller binding "
            f"{executor_id} changed after readiness",
        )
        write_new_durable(
            transport_descriptor_paths[executor_id],
            canonical_bytes({
                "controller_executable_path":
                    controller_identity["controller_executable_path"],
                "controller_executable_sha256":
                    controller_identity["controller_executable_sha256"],
                "controller_gid": controller_identity["controller_gid"],
                "controller_binding_device": binding_stat.st_dev,
                "controller_binding_inode": binding_stat.st_ino,
                "controller_binding_path": str(binding_path),
                "controller_binding_sha256": digest_file(binding_path),
                "controller_identity_device":
                    controller_publication["device"],
                "controller_identity_inode":
                    controller_publication["inode"],
                "controller_identity_path":
                    controller_publication["path"],
                "controller_identity_published_ns":
                    controller_publication["published_ns"],
                "controller_identity_sha256":
                    controller_publication["sha256"],
                "controller_pid": controller_identity["controller_pid"],
                "controller_start_time_ticks":
                    controller_identity["controller_start_time_ticks"],
                "controller_uid": controller_identity["controller_uid"],
                "executor_bundle_manifest_sha256":
                    executor_bundle["manifest_sha256"],
                "executor_id": executor_id,
                "executor_instance_id":
                    identity["executor_instance_id"],
                "gateway_argv_sha256":
                    digest_file(gateway_argv_paths[executor_id]),
                "gateway_config_sha256":
                    digest_file(config_paths[executor_id]),
                "gateway_pid": identity["gateway_pid"],
                "gateway_environment":
                    gateway_environments[executor_id],
                "gateway_executed_files":
                    gateway_file_bindings[executor_id],
                "gateway_post_auth_identity":
                    post_auth_by_id[executor_id],
                "gateway_prepublication_identity":
                    identity["process_identity"],
                "gateway_source_sha256": digest_file(gateway_source),
                "gateway_start_time_ticks":
                    identity["gateway_start_time_ticks"],
                "host_boot_id": controller_identity["host_boot_id"],
                "identity_captured_ns":
                    identity["identity_captured_ns"],
                "runtime_config_path": runtime_publication["path"],
                "runtime_config_device": runtime_publication["device"],
                "runtime_config_inode": runtime_publication["inode"],
                "runtime_config_published_ns":
                    runtime_publication["published_ns"],
                "runtime_config_sha256": runtime_publication["sha256"],
                "schema": "s40-executor-transport-descriptor-v4",
                "socket_path": executor["socket_path"],
                "transport": "UNIX_SOCKET",
            }),
            "executor_transport_descriptor",
        )


def wait_for_controller_bindings(
        processes: dict[str, subprocess.Popen],
        paths: dict[str, Path],
        runtime: dict[str, Any],
        identities: dict[str, dict[str, Any]],
        runtime_publication: dict[str, Any],
        controller_identity: dict[str, Any],
        controller_publication: dict[str, Any],
        timeout_s: int = 300) -> list[dict[str, Any]]:
    require(
        set(processes) == set(paths) == set(identities)
        == {
            executor["executor_id"]
            for executor in runtime["executors"]
        },
        "controller binding readiness: executor set mismatch",
    )
    runtime_by_id = {
        executor["executor_id"]: executor
        for executor in runtime["executors"]
    }
    deadline = time.monotonic() + timeout_s
    pending = set(processes)
    ready = []
    while pending:
        for executor_id in list(pending):
            process = processes[executor_id]
            require(
                process.poll() is None,
                f"controller binding {executor_id}: gateway exited",
            )
            path = paths[executor_id]
            if path.exists():
                try:
                    value, raw, binding_stat = (
                        read_controller_binding_evidence_capture(path)
                    )
                except RuntimeBindingError as error:
                    raise EvidenceError(
                        f"controller binding {executor_id}: {error}"
                    ) from error
                identity = identities[executor_id]
                executor = runtime_by_id[executor_id]
                authenticated_ns = require_int(
                    value["authenticated_ns"],
                    f"controller binding {executor_id}.authenticated_ns",
                    controller_publication["published_ns"],
                )
                require(
                    value["executor_id"] == executor_id
                    and value["executor_instance_id"]
                    == executor["executor_instance_id"]
                    == identity["executor_instance_id"]
                    and value["gateway_pid"] == process.pid
                    == executor["expected_peer_pid"]
                    == identity["gateway_pid"]
                    and value["gateway_start_time_ticks"]
                    == executor["expected_peer_start_time_ticks"]
                    == identity["gateway_start_time_ticks"]
                    and value["run_id"] == controller_identity["run_id"]
                    and value["host_boot_id"]
                    == controller_identity["host_boot_id"]
                    and value["controller_pid"]
                    == value["peer_pid"]
                    == controller_identity["controller_pid"]
                    and value["controller_start_time_ticks"]
                    == controller_identity["controller_start_time_ticks"]
                    and value["controller_uid"]
                    == value["peer_uid"]
                    == controller_identity["controller_uid"]
                    and value["controller_gid"]
                    == value["peer_gid"]
                    == controller_identity["controller_gid"]
                    and value["controller_executable_path"]
                    == controller_identity["controller_executable_path"]
                    and value["controller_executable_sha256"]
                    == controller_identity["controller_executable_sha256"]
                    and value["controller_identity_path"]
                    == controller_publication["path"]
                    and value["controller_identity_sha256"]
                    == controller_publication["sha256"]
                    and value["controller_identity_device"]
                    == controller_publication["device"]
                    and value["controller_identity_inode"]
                    == controller_publication["inode"]
                    and value["runtime_config_path"]
                    == runtime_publication["path"]
                    and value["runtime_config_sha256"]
                    == runtime_publication["sha256"]
                    and value["runtime_config_device"]
                    == runtime_publication["device"]
                    and value["runtime_config_inode"]
                    == runtime_publication["inode"],
                    f"controller binding {executor_id}: identity mismatch",
                )
                ready.append({
                    "device": binding_stat.st_dev,
                    "executor_id": executor_id,
                    "inode": binding_stat.st_ino,
                    "path": str(path),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                })
                pending.remove(executor_id)
        require(
            time.monotonic() < deadline,
            "controller binding readiness timed out",
        )
        if pending:
            time.sleep(0.05)
    return sorted(ready, key=lambda row: row["executor_id"])


def release_runtime_publication_link(publication: dict[str, Any]) -> None:
    temporary = Path(publication["temporary_path"])
    require(
        temporary.is_file() and not temporary.is_symlink(),
        "runtime identity: temporary publication link is missing",
    )
    temporary.unlink()
    directory_fd = os.open(
        temporary.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def release_controller_publication_link(publication: dict[str, Any]) -> None:
    temporary = Path(publication["temporary_path"])
    require(
        temporary.is_file() and not temporary.is_symlink(),
        "controller identity: temporary publication link is missing",
    )
    temporary.unlink()
    directory_fd = os.open(
        temporary.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def wait_for_sockets(
        processes: dict[str, subprocess.Popen],
        socket_paths: dict[str, Path],
        identities: dict[str, dict[str, Any]],
        timeout_s: int = 300) -> list[dict[str, Any]]:
    require(
        set(processes) == set(socket_paths) == set(identities),
        "gateway readiness: identity set mismatch",
    )
    deadline = time.monotonic() + timeout_s
    pending = set(processes)
    ready = []
    while pending:
        for executor_id in list(pending):
            process = processes[executor_id]
            require(process.poll() is None,
                    f"gateway {executor_id}: exited before readiness")
            if socket_paths[executor_id].is_socket():
                identity = identities[executor_id]
                stat_path = Path(f"/proc/{process.pid}/stat")
                require(
                    stat_path.is_file(),
                    f"gateway {executor_id}: process identity disappeared",
                )
                _, start_ticks = parse_process_stat(
                    stat_path.read_text(encoding="ascii"))
                require(
                    process.pid == identity["gateway_pid"]
                    and start_ticks == identity["gateway_start_time_ticks"],
                    f"gateway {executor_id}: process identity changed",
                )
                ready.append({
                    "executor_id": executor_id,
                    "executor_instance_id":
                        identity["executor_instance_id"],
                    "identity_captured_ns":
                        identity["identity_captured_ns"],
                    "pid": process.pid,
                    "process_identity": identity["process_identity"],
                    "process_start_time_ticks": start_ticks,
                    "socket_path": str(socket_paths[executor_id]),
                    "t_ns": time.monotonic_ns(),
                })
                pending.remove(executor_id)
        require(time.monotonic() < deadline, "gateway readiness timed out")
        if pending:
            time.sleep(0.05)
    return sorted(ready, key=lambda row: row["executor_id"])


def wait_for_gpu_observer(
        process: subprocess.Popen,
        output: Path,
        lock_path: Path,
        run_id: str,
        timeout_s: int = 30) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        require(
            process.poll() is None,
            "GPU observer exited before readiness",
        )
        if output.is_file() and lock_path.is_file():
            raw = output.read_bytes()
            if raw.endswith(b"\n") and len(raw.splitlines()) >= 2:
                header = parse_json(
                    raw.splitlines()[0] + b"\n", "GPU observer header")
                sample = parse_json(
                    raw.splitlines()[1] + b"\n",
                    "GPU observer first sample",
                )
                lock = read_json(lock_path, "GPU lock active record")
                require(
                    header.get("type") == "START"
                    and header.get("run_id") == run_id
                    and sample.get("type") == "SAMPLE"
                    and sample.get("sequence") == 0
                    and sample.get("process_observations") == []
                    and sample.get("process_stdout_base64") == ""
                    and lock.get("schema") == "s40-selected-gpu-lock-v1"
                    and lock.get("run_id") == run_id
                    and lock.get("released_ns") is None,
                    "GPU observer did not certify pre-launch idle",
                )
                return
        time.sleep(0.05)
    raise EvidenceError("GPU observer readiness timed out")


def wait_for_server(base_url: str, timeout_s: int = 300) -> dict[str, Any]:
    parsed = urlsplit(base_url)
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        started_ns = time.monotonic_ns()
        connection = http.client.HTTPConnection(
            parsed.hostname, parsed.port or 80, timeout=2)
        try:
            connection.request("GET", "/experimental/warm-tier/activate")
            response = connection.getresponse()
            raw = response.read()
            completed_ns = time.monotonic_ns()
            if response.status == 200:
                value = parse_json(raw, "server readiness")
                require(
                    value.get("schema")
                    == "llama-server-warm-tier-activate-status-v1"
                    and value.get("state") == "WAITING",
                    "server readiness: controller is not WAITING",
                )
                return {
                    "completed_ns": completed_ns,
                    "http_status": response.status,
                    "response_sha256": __import__("hashlib").sha256(
                        raw).hexdigest(),
                    "response": value,
                    "schema": "s40-server-readiness-v1",
                    "started_ns": started_ns,
                }
            last_error = f"HTTP {response.status}"
        except (OSError, http.client.HTTPException, EvidenceError) as error:
            last_error = str(error)
        finally:
            connection.close()
        time.sleep(0.05)
    raise EvidenceError(f"server readiness timed out: {last_error}")


def activation_request(
        base_url: str,
        method: str,
        body: dict[str, Any] | None,
        run_id: str,
        sequence: int) -> tuple[dict[str, Any], dict[str, Any]]:
    parsed = urlsplit(base_url)
    request_raw = canonical_bytes(body) if body is not None else b""
    started_ns = time.monotonic_ns()
    connection = http.client.HTTPConnection(
        parsed.hostname, parsed.port or 80, timeout=10)
    try:
        connection.request(
            method,
            "/experimental/warm-tier/activate",
            body=request_raw if body is not None else None,
            headers=(
                {"Content-Type": "application/json"}
                if body is not None else {}
            ),
        )
        response = connection.getresponse()
        response_raw = response.read()
        completed_ns = time.monotonic_ns()
    finally:
        connection.close()
    require(response.status == 200, "activation: non-200 response")
    value = parse_json(response_raw, "activation response")
    require(isinstance(value, dict), "activation: response is not an object")
    row = {
        "http_status": response.status,
        "method": method,
        "path": "/experimental/warm-tier/activate",
        "request_body_base64": base64.b64encode(
            request_raw).decode("ascii"),
        "request_body_sha256": hashlib.sha256(request_raw).hexdigest(),
        "response_body_base64": base64.b64encode(
            response_raw).decode("ascii"),
        "response_body_sha256": hashlib.sha256(response_raw).hexdigest(),
        "run_id": run_id,
        "schema": "s40-activation-http-evidence-v1",
        "sequence": sequence,
        "t_end_ns": completed_ns,
        "t_start_ns": started_ns,
    }
    return row, value


def activate_server(
        base_url: str,
        run_id: str,
        timeout_s: int = 300) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    row, waiting = activation_request(
        base_url, "GET", None, run_id, len(rows))
    rows.append(row)
    require(
        set(waiting) == {"controller_epoch", "schema", "state"}
        and waiting["schema"]
        == "llama-server-warm-tier-activate-status-v1"
        and waiting["state"] == "WAITING",
        "activation: initial controller state is not WAITING",
    )
    row, started = activation_request(
        base_url,
        "POST",
        {"schema": "llama-server-warm-tier-activate-v1"},
        run_id,
        len(rows),
    )
    rows.append(row)
    require(
        set(started) == {"controller_epoch", "schema", "state"}
        and started["schema"]
        == "llama-server-warm-tier-activate-result-v1"
        and started["state"] in {"PREPARING", "READY"},
        "activation: invalid POST result",
    )
    deadline = time.monotonic() + timeout_s
    while True:
        row, status = activation_request(
            base_url, "GET", None, run_id, len(rows))
        rows.append(row)
        require(
            set(status) == {"controller_epoch", "schema", "state"}
            and status["schema"]
            == "llama-server-warm-tier-activate-status-v1"
            and status["state"]
            in {"PREPARING", "READY", "FAILED"},
            "activation: invalid status result",
        )
        require(status["state"] != "FAILED", "activation: controller failed")
        if status["state"] == "READY":
            return rows, status
        require(time.monotonic() < deadline, "activation: timed out")
        time.sleep(0.05)


def normalize_exit(returncode: int) -> int:
    return returncode if returncode >= 0 else 128 - returncode


def stop_process(
        process: subprocess.Popen,
        name: str,
        timeout_s: int = 60) -> int:
    if process.poll() is None:
        process.terminate()
    try:
        returncode = process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired as error:
        process.kill()
        process.wait(timeout=10)
        raise EvidenceError(f"{name}: did not terminate cleanly") from error
    return normalize_exit(returncode)


def expected_nul_cmdline(argv: list[str]) -> bytes:
    require(
        argv
        and all(
            isinstance(argument, str)
            and argument.isascii()
            and "\x00" not in argument
            for argument in argv
        ),
        "gateway process: invalid expected argv",
    )
    return b"".join(argument.encode("ascii") + b"\x00" for argument in argv)


def gateway_process_identity(
        process: subprocess.Popen,
        argv: list[str],
        executor_id: str,
        expected: dict[str, Any] | None = None) -> dict[str, Any]:
    require(
        process.poll() is None and process.pid >= 2,
        f"gateway {executor_id}: process is not live",
    )
    proc_root = Path(f"/proc/{process.pid}")
    stat_path = proc_root / "stat"
    executable_link = proc_root / "exe"
    cmdline_path = proc_root / "cmdline"
    require(
        stat_path.is_file() and executable_link.exists()
        and cmdline_path.is_file(),
        f"gateway {executor_id}: process identity disappeared",
    )
    _, start_before = parse_process_stat(
        stat_path.read_text(encoding="ascii"))
    executable_target = os.readlink(executable_link)
    require(
        executable_target.startswith("/")
        and " (deleted)" not in executable_target,
        f"gateway {executor_id}: executable path is not stable",
    )
    executable_path = Path(executable_target).resolve()
    executable_stat_before = executable_link.stat()
    executable_sha256 = digest_file(executable_link)
    executable_stat_after = executable_link.stat()
    cmdline = cmdline_path.read_bytes()
    _, start_after = parse_process_stat(
        stat_path.read_text(encoding="ascii"))
    require(
        start_before == start_after
        and (
            executable_stat_before.st_dev,
            executable_stat_before.st_ino,
            executable_stat_before.st_size,
            executable_stat_before.st_mtime_ns,
            executable_stat_before.st_ctime_ns,
        )
        == (
            executable_stat_after.st_dev,
            executable_stat_after.st_ino,
            executable_stat_after.st_size,
            executable_stat_after.st_mtime_ns,
            executable_stat_after.st_ctime_ns,
        )
        and cmdline == expected_nul_cmdline(argv),
        f"gateway {executor_id}: live executable or cmdline changed",
    )
    value = {
        "cmdline_base64": base64.b64encode(cmdline).decode("ascii"),
        "cmdline_sha256": hashlib.sha256(cmdline).hexdigest(),
        "executable_ctime_ns": executable_stat_after.st_ctime_ns,
        "executable_device": executable_stat_after.st_dev,
        "executable_inode": executable_stat_after.st_ino,
        "executable_mtime_ns": executable_stat_after.st_mtime_ns,
        "executable_path": str(executable_path),
        "executable_sha256": executable_sha256,
        "executable_size": executable_stat_after.st_size,
        "gateway_pid": process.pid,
        "gateway_start_time_ticks": start_after,
        "observed_ns": time.monotonic_ns(),
    }
    require(
        set(value) == GATEWAY_PROCESS_IDENTITY_KEYS,
        f"gateway {executor_id}: identity key set mismatch",
    )
    if expected is not None:
        require(
            set(expected) == GATEWAY_PROCESS_IDENTITY_KEYS
            and {
                key: value[key]
                for key in GATEWAY_PROCESS_IDENTITY_KEYS - {"observed_ns"}
            }
            == {
                key: expected[key]
                for key in GATEWAY_PROCESS_IDENTITY_KEYS - {"observed_ns"}
            }
            and require_int(
                value["observed_ns"],
                f"gateway {executor_id}.observed_ns",
                require_int(
                    expected["observed_ns"],
                    f"gateway {executor_id}.expected.observed_ns",
                    1,
                ),
            ) >= expected["observed_ns"],
            f"gateway {executor_id}: live process identity changed",
        )
    return value


def recheck_gateway_processes(
        processes: dict[str, subprocess.Popen],
        argv: dict[str, list[str]],
        identities: dict[str, dict[str, Any]],
        file_bindings: dict[str, list[dict[str, Any]]],
        output: Path) -> list[dict[str, Any]]:
    require(
        set(processes) == set(argv) == set(identities) == set(file_bindings),
        "gateway recheck: executor set mismatch",
    )
    rows = []
    for executor_id in sorted(processes):
        verify_argument_bindings(file_bindings[executor_id], output)
        rows.append({
            "executor_id": executor_id,
            "identity": gateway_process_identity(
                processes[executor_id],
                argv[executor_id],
                executor_id,
                identities[executor_id],
            ),
        })
    return rows


def file_argument_bindings(
        argv: list[str],
        output: Path,
        prefix: str) -> list[dict[str, Any]]:
    captured = output / "captured"
    if not captured.exists():
        mkdir_new(captured)
    rows = []
    output_indexes = {
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
    for index, argument in enumerate(argv):
        if index in output_indexes:
            continue
        source = Path(argument)
        if not source.is_absolute() or not source.is_file():
            continue
        require(not source.is_symlink(),
                f"{prefix}: symlinked launch input is forbidden")
        source_stat = source.stat(follow_symlinks=False)
        target = captured / f"{prefix}-{index}-{source.name}"
        copy_new(source, target)
        rows.append({
            "argv_index": index,
            "bytes": target.stat().st_size,
            "captured_path": str(target.relative_to(output)),
            "executed_path": str(source),
            "sha256": digest_file(target),
            "source_ctime_ns": source_stat.st_ctime_ns,
            "source_device": source_stat.st_dev,
            "source_inode": source_stat.st_ino,
            "source_mtime_ns": source_stat.st_mtime_ns,
            "source_size": source_stat.st_size,
        })
    require(rows and rows[0]["argv_index"] == 0,
            f"{prefix}: executable binding is missing")
    return rows


def verify_argument_bindings(rows: list[dict[str, Any]], output: Path) -> None:
    for row in rows:
        source = Path(row["executed_path"])
        captured = output / row["captured_path"]
        source_stat = source.stat(follow_symlinks=False)
        require(
            source.is_file()
            and not source.is_symlink()
            and source.stat().st_size == row["bytes"]
            and (
                source_stat.st_dev,
                source_stat.st_ino,
                source_stat.st_size,
                source_stat.st_mtime_ns,
                source_stat.st_ctime_ns,
            ) == (
                row["source_device"],
                row["source_inode"],
                row["source_size"],
                row["source_mtime_ns"],
                row["source_ctime_ns"],
            )
            and digest_file(source) == row["sha256"]
            and digest_file(captured) == row["sha256"],
            "launch input changed after capture",
        )


def artifact(
        output: Path,
        role: str,
        path: Path,
        file_format: str) -> dict[str, Any]:
    require(path.is_file(), f"artifact {role}: missing {path}")
    count = None
    if file_format == "JSONL":
        raw = path.read_bytes()
        require(raw.endswith(b"\n"), f"artifact {role}: incomplete JSONL")
        count = len(raw.splitlines())
        require(count > 0, f"artifact {role}: empty JSONL")
    return {
        "bytes": path.stat().st_size,
        "format": file_format,
        "path": str(path.relative_to(output)),
        "record_count": count,
        "role": role,
        "sha256": digest_file(path),
    }


def run_physical(plan_path: Path) -> dict[str, Any]:
    plan = validate_physical_plan(plan_path)
    require_controller_port_available(plan["server_base_url"])
    host_boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
        encoding="ascii").strip()
    selected_gpu = next(
        device for device in plan["devices"]
        if device["device_role"] == "GPU")
    require(
        selected_gpu["boot_id"] == host_boot_id,
        "physical_plan: host boot identity changed",
    )
    output = Path(plan["output_dir"])
    mkdir_new(output)
    (
        runtime_plan_path,
        runtime_config_path,
        gateway_argv,
        config_paths,
        gateway_argv_paths,
        transport_descriptor_paths,
        executor_bundle,
        evidence_bundle,
        source_preflight_path,
        runtime_binaries,
        runtime_dependencies_path,
        phone_argv,
        phone_argv_paths,
        plan_value,
    ) = materialize_inputs(plan, output)
    validate_python_executor_bundle(
        output / "executor-bundle",
        output / "executor-bundle" / "MANIFEST.json",
        executor_bundle["manifest_sha256"],
    )
    validate_evidence_bundle(
        output / "evidence-bundle",
        output / "evidence-bundle" / "MANIFEST.json",
        evidence_bundle["manifest_sha256"],
    )
    require(
        read_json(
            output / "executor-bundle" / "MANIFEST.json",
            "executor_bundle_manifest",
        )["schema"] == "s40-executor-bundle-v2"
        and read_json(
            output / "evidence-bundle" / "MANIFEST.json",
            "evidence_bundle_manifest",
        )["schema"] == "s40-evidence-bundle-v2",
        "physical acquisition requires isolated Python bundles",
    )
    native_capture = runtime_binaries["native_bench"]
    controller_capture = runtime_binaries["controller"]
    captured_python = runtime_binaries["python"]
    captured_nvidia_smi = runtime_binaries["nvidia_smi"]
    private_library_path = (
        output / "captured-runtime" / "lib")
    internal_token_path = output / "warm-tier-internal.token"
    launch_environment = deterministic_launch_environment(
        output,
        private_library_path,
        runtime_config_path,
        captured_nvidia_smi,
        selected_gpu["stable_id"],
        executor_bundle["environment"],
        evidence_bundle["environment"],
    )
    privileged_environment = privileged_launch_environment(
        launch_environment, internal_token_path)
    gateway_environments = {
        executor["executor_id"]: (
            launch_environment
            if executor["role"] == "PHONE"
            else privileged_environment
        )
        for executor in plan_value["executors"]
    }
    transport_stdout_path = output / "transport-overhead.stdout"
    transport_stderr_path = output / "transport-overhead.stderr"
    transport_argv = isolated_python_argv(
        captured_python,
        evidence_bundle,
        output / "evidence-bundle" / "bridge_overhead.py",
        [
            "--executor-bundle",
            str(output / "executor-bundle"),
            "--native-bench",
            str(native_capture),
            "--output",
            str(output / "transport-overhead.json"),
            "--samples-per-cell",
            "50",
        ],
    )
    with transport_stdout_path.open("xb", buffering=0) as transport_stdout, \
            transport_stderr_path.open("xb", buffering=0) as transport_stderr:
        transport = subprocess.run(
            transport_argv,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=transport_stdout,
            stderr=transport_stderr,
            env=launch_environment,
            timeout=300,
        )
    require(
        transport.returncode == 0
        and transport_stderr_path.stat().st_size == 0,
        "native Unix transport diagnostic failed",
    )
    validate_combined_measurement(
        read_json(output / "transport-overhead.json",
                  "transport_overhead"),
        50,
    )
    contract_path = Path(plan["contract_path"])
    server_argv = list(plan["server_argv"])
    server_argv[0] = str(controller_capture)
    gateway_file_bindings = {}
    for executor in plan_value["executors"]:
        executor_id = executor["executor_id"]
        gateway_file_bindings[executor_id] = file_argument_bindings(
            gateway_argv[executor_id],
            output,
            f"gateway-{executor_id}",
        )

    gateway_processes: dict[str, subprocess.Popen] = {}
    gateway_streams = []
    server = None
    sampler = None
    telemetry = None
    gpu_observer = None
    server_streams = []
    sampler_streams = []
    telemetry_streams = []
    gpu_observer_streams = []
    acquisition_returncode = None
    cleanup_rows = []
    server_started_ns = None
    server_stopped_ns = None
    server_pid = None
    server_start_ticks = None
    runtime_publication = None
    controller_identity = None
    controller_publication = None
    controller_binding_ready = None
    gateway_post_auth = None
    gateway_final = None
    readiness_value = None
    controller_identity_path = output / "controller-identity-lock.json"
    controller_binding_paths = {
        executor["executor_id"]:
            output
            / f"executor-controller-binding-{executor['executor_id']}.json"
        for executor in plan_value["executors"]
    }
    gpu_observer_stop = output / "gpu-observer.stop"
    gpu_observer_output = output / "gpu-observer.jsonl"
    gpu_lock_output = output / "gpu-lock-record.json"
    gpu_lock_path = Path(plan["gpu_lock_path"])
    gpu_observer_argv = isolated_python_argv(
        captured_python,
        evidence_bundle,
        output / "evidence-bundle" / "gpu_isolation.py",
        [
            "--output", str(gpu_observer_output),
            "--stop-file", str(gpu_observer_stop),
            "--lock-path", str(gpu_lock_path),
            "--lock-output", str(gpu_lock_output),
            "--run-id", plan["run_id"],
            "--gpu-uuid", selected_gpu["stable_id"],
            "--gpu-name",
            read_json(contract_path, "contract")["runtime_source"][
                "expected_gpu_name"],
            "--nvidia-smi", str(captured_nvidia_smi),
            "--nvidia-smi-sha256", digest_file(captured_nvidia_smi),
            "--interval-ms", "200",
        ],
    )
    write_new_durable(
        output / "gpu-observer-argv.json",
        canonical_bytes({
            "argv": gpu_observer_argv,
            "schema": "s40-gpu-observer-argv-v1",
        }),
        "gpu_observer_argv",
    )
    try:
        create_internal_token_file(internal_token_path)
        gpu_observer_stdout = (
            output / "gpu-observer.stdout").open("xb", buffering=0)
        gpu_observer_stderr = (
            output / "gpu-observer.stderr").open("xb", buffering=0)
        gpu_observer_streams.extend(
            (gpu_observer_stdout, gpu_observer_stderr))
        gpu_observer = subprocess.Popen(
            gpu_observer_argv,
            stdin=subprocess.DEVNULL,
            stdout=gpu_observer_stdout,
            stderr=gpu_observer_stderr,
            env=launch_environment,
            start_new_session=True,
        )
        wait_for_gpu_observer(
            gpu_observer,
            gpu_observer_output,
            gpu_lock_path,
            plan["run_id"],
        )
        verify_binary_lock(plan)

        sockets = {}
        for executor in plan_value["executors"]:
            executor_id = executor["executor_id"]
            verify_argument_bindings(
                gateway_file_bindings[executor_id], output)
            require(
                executor["transport"] == "UNIX_SOCKET",
                f"gateway {executor_id}: non-native transport",
            )
            sockets[executor_id] = Path(executor["socket_path"])
            stdout_path = output / f"gateway-{executor_id}.stdout"
            stderr_path = output / f"gateway-{executor_id}.stderr"
            stdout = stdout_path.open("xb", buffering=0)
            stderr = stderr_path.open("xb", buffering=0)
            gateway_streams.extend((stdout, stderr))
            gateway_processes[executor_id] = subprocess.Popen(
                gateway_argv[executor_id],
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                env=gateway_environments[executor_id],
                start_new_session=True,
            )
        (
            runtime,
            gateway_identities,
            runtime_publication,
        ) = finalize_runtime_identity(
            plan_value,
            runtime_plan_path,
            runtime_config_path,
            gateway_processes,
            gateway_argv,
            gateway_file_bindings,
            output,
            gateway_argv_paths,
            config_paths,
            transport_descriptor_paths,
            executor_bundle,
            output / "executor-bundle",
            contract_path,
        )
        gateway_ready = wait_for_sockets(
            gateway_processes, sockets, gateway_identities)
        release_runtime_publication_link(runtime_publication)

        server_stdout = (output / "server.stdout").open("xb", buffering=0)
        server_stderr = (output / "server.stderr").open("xb", buffering=0)
        server_streams.extend((server_stdout, server_stderr))
        server_started_ns = time.monotonic_ns()
        server = subprocess.Popen(
            server_argv,
            stdin=subprocess.DEVNULL,
            stdout=server_stdout,
            stderr=server_stderr,
            env=privileged_environment,
            start_new_session=True,
        )
        server_pid = server.pid
        _, server_start_ticks = parse_process_stat(
            Path(f"/proc/{server_pid}/stat").read_text(encoding="ascii"))
        (
            controller_identity,
            controller_publication,
        ) = publish_controller_identity(
            controller_identity_path,
            server,
            server_started_ns,
            controller_capture,
            plan["run_id"],
            host_boot_id,
            runtime_publication,
        )
        require(
            controller_identity["controller_start_time_ticks"]
            == server_start_ticks,
            "controller identity: launch start time mismatch",
        )
        base_readiness = wait_for_server(plan["server_base_url"])
        require(server.poll() is None,
                "controller exited before activation")

        activation_rows, activation_ready = activate_server(
            plan["server_base_url"], plan["run_id"])
        write_new_durable(
            output / "activation-evidence.jsonl",
            b"".join(canonical_bytes(row) for row in activation_rows),
            "activation_evidence",
        )
        controller_binding_ready = wait_for_controller_bindings(
            gateway_processes,
            controller_binding_paths,
            runtime,
            gateway_identities,
            runtime_publication,
            controller_identity,
            controller_publication,
        )
        gateway_post_auth = recheck_gateway_processes(
            gateway_processes,
            gateway_argv,
            {
                executor_id: identity["process_identity"]
                for executor_id, identity in gateway_identities.items()
            },
            gateway_file_bindings,
            output,
        )
        write_transport_descriptors(
            runtime,
            gateway_identities,
            gateway_post_auth,
            gateway_file_bindings,
            gateway_environments,
            runtime_publication,
            controller_identity,
            controller_publication,
            gateway_argv_paths,
            config_paths,
            controller_binding_paths,
            controller_binding_ready,
            transport_descriptor_paths,
            executor_bundle,
            output / "executor-bundle",
        )
        release_controller_publication_link(controller_publication)
        phone_observer_readiness = None
        if phone_argv:
            identity = subprocess.run(
                phone_argv["identity"],
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=launch_environment,
                timeout=300,
            )
            require(identity.returncode == 0 and not identity.stderr,
                    "phone identity command failed")
            value = parse_json(identity.stdout, "phone identity")
            write_new_durable(
                output / "phone-identity.json",
                canonical_bytes(value),
                "phone_identity",
            )
            telemetry_stdout = (
                output / "phone-telemetry.jsonl").open("xb", buffering=0)
            telemetry_stderr = (
                output / "phone-telemetry.stderr").open("xb", buffering=0)
            telemetry_streams.extend((telemetry_stdout, telemetry_stderr))
            telemetry_started_ns = time.monotonic_ns()
            telemetry = subprocess.Popen(
                phone_argv["telemetry"],
                stdin=subprocess.DEVNULL,
                stdout=telemetry_stdout,
                stderr=telemetry_stderr,
                env=launch_environment,
                start_new_session=True,
            )
            time.sleep(0.1)
            require(telemetry.poll() is None,
                    "phone telemetry exited before acquisition")
            phone_observer_readiness = {
                "identity_sha256": digest_file(
                    output / "phone-identity.json"),
                "telemetry_pid": telemetry.pid,
                "telemetry_started_ns": telemetry_started_ns,
            }

        readiness_value = {
                "activation_evidence_sha256": digest_file(
                    output / "activation-evidence.jsonl"),
                "activation_ready": activation_ready,
                "base_readiness": base_readiness,
                "controller_bindings": controller_binding_ready,
                "controller_identity": {
                    key: controller_publication[key]
                    for key in (
                        "captured_ns",
                        "device",
                        "inode",
                        "path",
                        "published_ns",
                        "sha256",
                    )
                },
                "controller_started_ns": server_started_ns,
                "gateway_final": None,
                "gateway_post_auth": gateway_post_auth,
                "gateway_ready": gateway_ready,
                "host_boot_id": host_boot_id,
                "launch_environment": privileged_environment,
                "phone_observer": phone_observer_readiness,
                "run_id": plan["run_id"],
                "runtime_config_path": runtime_publication["path"],
                "runtime_config_device": runtime_publication["device"],
                "runtime_config_inode": runtime_publication["inode"],
                "runtime_config_published_ns":
                    runtime_publication["published_ns"],
                "runtime_config_sha256": runtime_publication["sha256"],
                "schema": "s40-server-readiness-v6",
        }

        sampler_stdout = (
            output / "resource-sampler.stdout").open("xb", buffering=0)
        sampler_stderr = (
            output / "resource-sampler.stderr").open("xb", buffering=0)
        sampler_streams.extend((sampler_stdout, sampler_stderr))
        stop_file = output / "resource-sampler.stop"
        sampler_argv = isolated_python_argv(
            captured_python,
            evidence_bundle,
            output / "evidence-bundle" / "resource_sampler.py",
            [
                "--run-id", plan["run_id"],
                "--gpu-uuid",
                read_json(contract_path, "contract")["runtime_source"][
                    "expected_gpu_uuid"],
                "--controller-pid", str(server_pid),
                "--output", str(output / "resource-samples.jsonl"),
                "--stop-file", str(stop_file),
                "--interval-ms", "200",
                "--nvidia-smi-path", str(captured_nvidia_smi),
                "--nvidia-smi-sha256", digest_file(captured_nvidia_smi),
            ],
        )
        write_new_durable(
            output / "resource-sampler-argv.json",
            canonical_bytes({
                "argv": sampler_argv,
                "schema": "s40-resource-sampler-argv-v1",
            }),
            "resource_sampler_argv",
        )
        sampler = subprocess.Popen(
            sampler_argv,
            stdin=subprocess.DEVNULL,
            stdout=sampler_stdout,
            stderr=sampler_stderr,
            env=launch_environment,
            start_new_session=True,
        )
        time.sleep(0.25)
        require(sampler.poll() is None,
                "resource sampler exited before acquisition")

        driver_stdout_path = output / "trace-driver.stdout"
        driver_stderr_path = output / "trace-driver.stderr"
        driver_argv = isolated_python_argv(
            captured_python,
            evidence_bundle,
            output / "evidence-bundle" / "acquire_trace.py",
            [
                "--base-url", plan["server_base_url"],
                "--event-log", str(output / "controller-events.jsonl"),
                "--runtime-config", str(runtime_config_path),
                "--http-evidence", str(output / "http-evidence.jsonl"),
                "--trace-start", str(output / "trace-start.json"),
                "--result", str(output / "trace-result.json"),
                "--run-id", plan["run_id"],
                "--contract", str(contract_path),
            ],
        )
        with driver_stdout_path.open("xb", buffering=0) as driver_stdout, \
                driver_stderr_path.open("xb", buffering=0) as driver_stderr:
            driver = subprocess.run(
                driver_argv,
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=driver_stdout,
                stderr=driver_stderr,
                env=launch_environment,
            )
        acquisition_returncode = driver.returncode
        require(acquisition_returncode == 0,
                "trace acquisition failed")

        write_new_durable(stop_file, b"stop\n", "resource stop")
        require(sampler.wait(timeout=30) == 0,
                "resource sampler did not stop cleanly")
        sampler = None
        gateway_final = recheck_gateway_processes(
            gateway_processes,
            gateway_argv,
            {
                executor_id: identity["process_identity"]
                for executor_id, identity in gateway_identities.items()
            },
            gateway_file_bindings,
            output,
        )
        require(
            readiness_value is not None,
            "server readiness: missing pre-shutdown evidence",
        )
        readiness_value["gateway_final"] = gateway_final
        write_new_durable(
            output / "server-readiness.json",
            canonical_bytes(readiness_value),
            "server_readiness",
        )
        for executor_id, process in gateway_processes.items():
            exit_code = stop_process(
                process, f"gateway {executor_id}")
            require(exit_code == 0,
                    f"gateway {executor_id}: unclean shutdown")
            cleanup_rows.append({
                "exit_code": exit_code,
                "name": f"gateway::{executor_id}",
            })
        gateway_processes.clear()
        if telemetry is not None:
            cleanup_rows.append({
                "exit_code": stop_process(telemetry, "phone telemetry"),
                "name": "phone_telemetry",
            })
            telemetry = None

        server_returncode = stop_process(server, "controller")
        require(server_returncode == 0, "controller: unclean shutdown")
        cleanup_rows.append({
            "exit_code": server_returncode,
            "name": "controller",
        })
        server_stopped_ns = time.monotonic_ns()
        server = None
        write_new_durable(
            gpu_observer_stop, b"stop\n", "GPU observer stop")
        require(
            gpu_observer.wait(timeout=30) == 0,
            "GPU observer did not stop cleanly",
        )
        gpu_observer = None
        require(
            (output / "gpu-observer.stderr").stat().st_size == 0,
            "GPU observer wrote stderr",
        )
        cleanup_rows.append({
            "exit_code": 0,
            "name": "gpu_observer",
        })
    finally:
        cleanup_actions: list[tuple[str, Callable[[], None]]] = []
        if controller_publication is not None:
            cleanup_actions.append((
                "controller publication",
                lambda record=controller_publication:
                    remove_temporary_publication(record),
            ))
        if runtime_publication is not None:
            cleanup_actions.append((
                "runtime publication",
                lambda record=runtime_publication:
                    remove_temporary_publication(record),
            ))
        if sampler is not None:
            cleanup_actions.append((
                "resource sampler",
                lambda process=sampler:
                    stop_process(process, "resource sampler"),
            ))
        for executor_id, process in gateway_processes.items():
            cleanup_actions.append((
                f"gateway {executor_id}",
                lambda process=process, executor_id=executor_id:
                    stop_process(process, f"gateway {executor_id}"),
            ))
        if telemetry is not None:
            cleanup_actions.append((
                "phone telemetry",
                lambda process=telemetry:
                    stop_process(process, "phone telemetry"),
            ))
        if server is not None:
            cleanup_actions.append((
                "controller",
                lambda process=server:
                    stop_process(process, "controller"),
            ))
        if gpu_observer is not None:
            cleanup_actions.append((
                "GPU observer",
                lambda process=gpu_observer:
                    stop_gpu_observer_process(process, gpu_observer_stop),
            ))
        for index, stream in enumerate(
                gateway_streams + server_streams
                + sampler_streams + telemetry_streams
                + gpu_observer_streams):
            cleanup_actions.append((
                f"stream {index}",
                lambda stream=stream: sync_close_stream(stream),
            ))
        cleanup_actions.append((
            "warm-tier internal token",
            lambda: remove_internal_token_file(internal_token_path),
        ))
        run_cleanup_actions(cleanup_actions)

    require(
        not os.path.lexists(internal_token_path),
        "warm-tier internal token remains after teardown",
    )
    require(
        server_started_ns is not None
        and server_pid is not None
        and server_start_ticks is not None
        and server_stopped_ns is not None
        and controller_identity is not None
        and controller_publication is not None
        and controller_binding_ready is not None,
            "controller launch identity was not captured")
    verify_binary_lock(plan)
    orchestrator_completed_ns = time.monotonic_ns()
    launch = {
        "binary_sha256": digest_file(controller_capture),
        "command_argv": server_argv,
        "controller_identity_device": controller_publication["device"],
        "controller_identity_inode": controller_publication["inode"],
        "controller_identity_path": controller_publication["path"],
        "controller_identity_sha256": controller_publication["sha256"],
        "controller_pid": server_pid,
        "controller_start_ticks": server_start_ticks,
        "experiment_contract_sha256":
            _source_row(
                plan[SOURCE_LOCK_KEY],
                "experiment_contract",
            )["sha256"],
        "exit_code": server_returncode,
        "host_boot_id": host_boot_id,
        "launch_environment": privileged_environment,
        "library_path": str(private_library_path),
        "physical_plan_sha256":
            _source_row(plan[SOURCE_LOCK_KEY], "physical_plan")["sha256"],
        "run_id": plan["run_id"],
        "runtime_config_sha256": digest_file(runtime_config_path),
        "schema": "s40-controller-launch-v4",
        "selected_gpu_environment": {
            "CUDA_VISIBLE_DEVICES": selected_gpu["stable_id"],
            "NVIDIA_VISIBLE_DEVICES": selected_gpu["stable_id"],
        },
        "source_preflight_sha256": digest_file(source_preflight_path),
        "started_ns": server_started_ns,
        "stopped_ns": server_stopped_ns,
    }
    write_new_durable(
        output / "controller-launch.json",
        canonical_bytes(launch),
        "controller_launch",
    )
    write_new_durable(
        output / "orchestrator-evidence.json",
        canonical_bytes({
            "acquisition_returncode": acquisition_returncode,
            "cleanup": cleanup_rows,
            "completed_ns": orchestrator_completed_ns,
            "run_id": plan["run_id"],
            "schema": "s40-orchestrator-evidence-v2",
        }),
        "orchestrator_evidence",
    )
    validate_python_executor_bundle(
        output / "executor-bundle",
        output / "executor-bundle" / "MANIFEST.json",
        executor_bundle["manifest_sha256"],
    )
    validate_evidence_bundle(
        output / "evidence-bundle",
        output / "evidence-bundle" / "MANIFEST.json",
        evidence_bundle["manifest_sha256"],
    )

    artifacts = [
        artifact(output, "activation_evidence",
                 output / "activation-evidence.jsonl", "JSONL"),
        artifact(output, "native_bench_binary",
                 native_capture, "BINARY"),
        artifact(output, "runtime_dependency_manifest",
                 runtime_dependencies_path, "JSON"),
        artifact(output, "transport_overhead",
                 output / "transport-overhead.json", "JSON"),
        artifact(output, "transport_overhead_stderr",
                 transport_stderr_path, "TEXT"),
        artifact(output, "transport_overhead_stdout",
                 transport_stdout_path, "TEXT"),
        artifact(output, "controller_identity",
                 controller_identity_path, "JSON"),
        artifact(output, "controller_launch",
                 output / "controller-launch.json", "JSON"),
        artifact(output, "controller_events",
                 output / "controller-events.jsonl", "JSONL"),
        artifact(output, "evidence_root",
                 output / "evidence-root.json", "JSON"),
        artifact(output, "evidence_bundle_manifest",
                 output / "evidence-bundle" / "MANIFEST.json", "JSON"),
        artifact(output, "executor_bundle_manifest",
                 output / "executor-bundle" / "MANIFEST.json", "JSON"),
        artifact(output, "http_evidence",
                 output / "http-evidence.jsonl", "JSONL"),
        artifact(output, "gpu_lock_record",
                 gpu_lock_output, "JSON"),
        artifact(output, "gpu_observer",
                 gpu_observer_output, "JSONL"),
        artifact(output, "gpu_observer_argv",
                 output / "gpu-observer-argv.json", "JSON"),
        artifact(output, "gpu_observer_stderr",
                 output / "gpu-observer.stderr", "TEXT"),
        artifact(output, "gpu_observer_stdout",
                 output / "gpu-observer.stdout", "TEXT"),
        artifact(output, "orchestrator_evidence",
                 output / "orchestrator-evidence.json", "JSON"),
        artifact(output, "resource_samples",
                 output / "resource-samples.jsonl", "JSONL"),
        artifact(output, "resource_sampler_stderr",
                 output / "resource-sampler.stderr", "TEXT"),
        artifact(output, "resource_sampler_stdout",
                 output / "resource-sampler.stdout", "TEXT"),
        artifact(output, "resource_sampler_argv",
                 output / "resource-sampler-argv.json", "JSON"),
        artifact(output, "runtime_config", runtime_config_path, "JSON"),
        artifact(output, "runtime_plan", runtime_plan_path, "JSON"),
        artifact(output, "server_readiness",
                 output / "server-readiness.json", "JSON"),
        artifact(output, "server_stderr",
                 output / "server.stderr", "TEXT"),
        artifact(output, "server_stdout",
                 output / "server.stdout", "TEXT"),
        artifact(output, "source_preflight",
                 source_preflight_path, "JSON"),
        artifact(output, "trace_acquisition_result",
                 output / "trace-result.json", "JSON"),
        artifact(output, "trace_driver_stderr",
                 output / "trace-driver.stderr", "TEXT"),
        artifact(output, "trace_driver_stdout",
                 output / "trace-driver.stdout", "TEXT"),
        artifact(output, "trace_start",
                 output / "trace-start.json", "JSON"),
    ]
    for row in executor_bundle["files"]:
        artifacts.append(artifact(
            output,
            f"executor_bundle::{row['name']}",
            output / "executor-bundle" / row["name"],
            "TEXT",
        ))
    for row in evidence_bundle["files"]:
        artifacts.append(artifact(
            output,
            f"evidence_bundle::{row['name']}",
            output / "evidence-bundle" / row["name"],
            "TEXT",
        ))
    runtime_dependencies = read_json(
        runtime_dependencies_path, "runtime_dependencies")
    for record in runtime_dependencies["dependencies"]:
        if record["captured_path"] is None:
            continue
        artifacts.append(artifact(
            output,
            f"runtime_dependency::{record['needed_name']}",
            output / record["captured_path"],
            "BINARY",
        ))
    executor_bindings = []
    require(
        gateway_post_auth is not None and gateway_final is not None,
        "gateway lifecycle evidence is incomplete",
    )
    post_auth_by_id = {
        row["executor_id"]: row["identity"]
        for row in gateway_post_auth
    }
    final_by_id = {
        row["executor_id"]: row["identity"]
        for row in gateway_final
    }
    runtime_plan = read_json(runtime_plan_path, "runtime_plan")
    for executor in runtime["executors"]:
        executor_id = executor["executor_id"]
        artifacts.extend([
            artifact(
                output,
                f"executor_gateway_argv::{executor_id}",
                gateway_argv_paths[executor_id],
                "JSON",
            ),
            artifact(
                output,
                f"executor_command::{executor_id}",
                output / f"executor-{executor_id}.jsonl",
                "JSONL",
            ),
            artifact(
                output,
                f"executor_controller_binding::{executor_id}",
                controller_binding_paths[executor_id],
                "JSON",
            ),
            artifact(
                output,
                f"executor_transport::{executor_id}",
                transport_descriptor_paths[executor_id],
                "JSON",
            ),
            artifact(
                output,
                f"gateway_config::{executor_id}",
                config_paths[executor_id],
                "JSON",
            ),
            artifact(
                output,
                f"gateway_stderr::{executor_id}",
                output / f"gateway-{executor_id}.stderr",
                "TEXT",
            ),
            artifact(
                output,
                f"gateway_stdout::{executor_id}",
                output / f"gateway-{executor_id}.stdout",
                "TEXT",
            ),
        ])
        if executor["role"] == "PHONE":
            artifacts.extend([
                artifact(
                    output,
                    f"executor_route::{executor_id}",
                    output / f"executor-route-{executor_id}.jsonl",
                    "JSONL",
                ),
                artifact(
                    output,
                    f"executor_wire::{executor_id}",
                    output / f"executor-wire-{executor_id}.jsonl",
                    "JSONL",
                ),
            ])
        executor_bindings.append({
            "command_role": f"executor_command::{executor_id}",
            "controller_binding_role":
                f"executor_controller_binding::{executor_id}",
            "executor_id": executor_id,
            "gateway_argv": gateway_argv[executor_id],
            "gateway_argv_role":
                f"executor_gateway_argv::{executor_id}",
            "gateway_config_role": f"gateway_config::{executor_id}",
            "gateway_executed_files":
                gateway_file_bindings[executor_id],
            "gateway_final_identity": final_by_id[executor_id],
            "gateway_launch_environment":
                gateway_environments[executor_id],
            "gateway_post_auth_identity": post_auth_by_id[executor_id],
            "gateway_prepublication_identity":
                gateway_identities[executor_id]["process_identity"],
            "gateway_source_role": (
                "executor_bundle::phone_gateway.py"
                if executor["role"] == "PHONE"
                else "executor_bundle::desktop_gateway.py"
            ),
            "gateway_stderr_role": f"gateway_stderr::{executor_id}",
            "gateway_stdout_role": f"gateway_stdout::{executor_id}",
            "transport_descriptor_role":
                f"executor_transport::{executor_id}",
        })
    evidence_root = read_json(output / "evidence-root.json", "evidence_root")
    if plan["mode"] == "C3_DUAL_PARTIAL_OFFLOAD":
        profile = Path(runtime_plan["c3_profile_lock_path"])
        artifacts.append(artifact(
            output, "c3_profile_lock", profile, "JSON"))
        for model_id, record in evidence_root[
                "c3_placement_artifacts"].items():
            artifacts.append(artifact(
                output,
                f"c3_placement::{model_id}",
                Path(record["path"]),
                "JSON",
            ))
    if plan["mode"] in PHONE_MODES:
        artifacts.extend([
            artifact(
                output, "phone_identity",
                output / "phone-identity.json", "JSON"),
            artifact(
                output, "phone_telemetry",
                output / "phone-telemetry.jsonl", "JSONL"),
            artifact(
                output, "phone_telemetry_stderr",
                output / "phone-telemetry.stderr", "TEXT"),
            artifact(
                output, "phone_identity_argv",
                phone_argv_paths["identity"], "JSON"),
            artifact(
                output, "phone_ssh_config",
                output / "inputs" / "phone-ssh-config.json", "JSON"),
            artifact(
                output, "phone_telemetry_argv",
                phone_argv_paths["telemetry"], "JSON"),
        ])
    manifest = {
        "artifacts": artifacts,
        "cache_regime": plan["cache_regime"],
        "campaign_binding": (
            None
            if plan["campaign_binding"] is None
            else {
                "campaign_id": read_campaign(
                    Path(plan["campaign_binding"]["campaign_path"])
                )["campaign_id"],
                "campaign_sha256":
                    plan["campaign_binding"]["campaign_sha256"],
                "order": plan["campaign_binding"]["order"],
                "phase": plan["campaign_binding"]["phase"],
            }
        ),
        "command_argv": server_argv,
        "controller_binary": {
            "bytes": controller_capture.stat().st_size,
            "captured_path": str(controller_capture.relative_to(output)),
            "executed_path": str(controller_capture),
            "sha256": digest_file(controller_capture),
        },
        "development": plan["development"],
        "devices": plan["devices"],
        "executor_bindings": executor_bindings,
        "experiment_contract_sha256": digest_file(contract_path),
        "mode": plan["mode"],
        "policy_id": read_json(contract_path, "contract")["policy"]["id"],
        "repeat_index": plan["repeat_index"],
        "run_id": plan["run_id"],
        "schema": "s40-physical-run-manifest-v6",
        "schema_version": 6,
    }
    manifest_path = output / "manifest.json"
    write_new_durable(
        manifest_path, canonical_bytes(manifest), "run_manifest")
    result = validate_run_manifest(manifest_path, contract_path)
    result["manifest_path"] = str(manifest_path)
    result["manifest_sha256"] = digest_file(manifest_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run_physical(args.plan)
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}")
        return 2
    print(canonical_bytes(result).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
