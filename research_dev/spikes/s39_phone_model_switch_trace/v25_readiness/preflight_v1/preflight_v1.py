#!/usr/bin/env python3
"""Capture and finalize the no-model V2.5 production preflight."""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Protocol


PHASE = "A_ONLY"
PLAN_SCHEMA = "s39-v25-preflight-plan-v1"
PREPARATION_SCHEMA = "s39-cp0-r1-v25-reboot-preparation-v1"
DISCOVERY_SCHEMA = "s39-cp0-r1-v25-post-reboot-discovery-v1"
IDENTITY_SCHEMA = "s39-v25-a-only-fresh-artifact-identity-v1"
INVENTORY_SCHEMA = "s39-v25-a-only-production-inventory-v1"
CAPTURE_EVIDENCE_SCHEMA = "s39-v25-preflight-capture-evidence-v1"
FINAL_EVIDENCE_SCHEMA = "s39-v25-preflight-final-evidence-v1"
PROJECTION_SCHEMA = "s39-v25-v24-preparation-projection-v1"
CAPTURE_MANIFEST_SCHEMA = "s39-v25-preflight-capture-manifest-v1"
FINAL_MANIFEST_SCHEMA = "s39-v25-preflight-final-manifest-v1"
COMMAND_RECEIPT_SCHEMA = "s39-v25-preflight-command-receipt-v1"
REMOTE_PROBE_SCHEMA = "s39-v25-preflight-remote-probe-v1"
LOCAL_ADB_PROBE_SCHEMA = "s39-v25-preflight-local-adb-probe-v1"
CONFIRM_CAPTURE = "RUN_CP0_R1_V25_PREFLIGHT_REBOOT"
CONFIRM_FINALIZE = "RUN_CP0_R1_V25_PREFLIGHT_FINALIZE"
ADB_PORT = 5038
MAX_FILE_BYTES = 512 * 1024 * 1024
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_RECEIPTS = 256
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
LOCAL_ARTIFACT_ROLES = {
    "adb",
    "a_only_runner",
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
V24_INPUT_SCHEMAS = {
    "artifact_root": "s39-cp0-r1-artifact-root-v2.4",
    "bound_root": "s39-cp0-r1-v24-bound-runtime-root-v1",
    "candidate": "s39-cp0-r1-candidate-v1",
    "contract": "s39-cp0-r1-evidence-contract-v2.4",
    "contract_v25": "s39-cp0-r1-evidence-contract-v2.5",
    "cuda_monolithic_launch": "s39-cp0-r1-v24-cuda-monolithic-launch-v1",
    "cuda_route_launch": "s39-cp0-r1-v24-cuda-route-launch-v1",
    "fresh_readiness": "s39-cp0-r1-fast-fresh-readiness-v2.4",
    "identity_binding_attestation": (
        "s39-cp0-r1-v24-identity-binding-attestation-v1"
    ),
    "identity_binding_receipt": (
        "s39-cp0-r1-v24-identity-binding-receipt-v1"
    ),
    "identity_binding_stage_receipt": "s39-cp0-r1-v24-stage-receipt-v1",
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
PRE_INPUT_NAMES = {
    "phase_lock": "phase-lock.jsonl",
    "phase_preflight": "phase-preflight.jsonl",
    "quality_corpus": "quality-corpus.jsonl",
    "route_lock": "route-lock.jsonl",
}
LOCAL_SOURCE_BINDINGS = {
    "a_only_runner": ("v25", "a_only_runner"),
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


class PreflightError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightError(message)


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


def uuid(value: Any, field: str) -> str:
    value = text(value, field, 64)
    pieces = value.split("-")
    require(
        [len(piece) for piece in pieces] == [8, 4, 4, 4, 12]
        and all(
            character in "0123456789abcdef"
            for piece in pieces
            for character in piece
        ),
        f"E_UUID: {field}",
    )
    return value


def absolute_path(value: Any, field: str) -> str:
    value = text(value, field)
    path = Path(value)
    require(
        path.is_absolute()
        and str(path) == value
        and ".." not in path.parts,
        f"E_PATH: {field}",
    )
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
        raise PreflightError("E_CANONICAL") from error


def canonical_compact(value: Any) -> bytes:
    return canonical_bytes(value)[:-1]


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value = {}
    for key, item in pairs:
        require(key not in value, f"E_DUPLICATE_KEY: {key}")
        value[key] = item
    return value


def parse_json(raw: bytes, field: str) -> Any:
    require(0 < len(raw) <= MAX_FILE_BYTES, f"E_SIZE: {field}")
    try:
        return json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                PreflightError(f"E_JSON_NUMBER: {field}: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreflightError(f"E_JSON: {field}") from error


def _stat_row(metadata: os.stat_result) -> dict[str, Any]:
    return {
        "build_id": None,
        "ctime_ns": metadata.st_ctime_ns,
        "device_id": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": metadata.st_mode,
        "mtime_ns": metadata.st_mtime_ns,
        "size": metadata.st_size,
    }


def read_regular(
    path: Path,
    field: str,
    maximum: int = MAX_FILE_BYTES,
) -> tuple[bytes, os.stat_result]:
    require(path.is_absolute(), f"E_PATH: {field}")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise PreflightError(f"E_READ: {field}: {error}") from error
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"E_FILE_TYPE: {field}")
        require(0 < before.st_size <= maximum, f"E_FILE_SIZE: {field}")
        raw = bytearray()
        while block := os.read(descriptor, 1024 * 1024):
            raw.extend(block)
            require(len(raw) <= maximum, f"E_FILE_SIZE: {field}")
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
    require(identity(before) == identity(after), f"E_FILE_CHANGED: {field}")
    require(len(raw) == before.st_size, f"E_FILE_CHANGED: {field}")
    return bytes(raw), before


def read_canonical(path: Path, field: str) -> tuple[dict[str, Any], bytes]:
    raw, unused_stat = read_regular(path, field)
    value = parse_json(raw, field)
    require(type(value) is dict, f"E_TYPE: {field}")
    exact(canonical_bytes(value), raw, f"{field}.canonical")
    return value, raw


def artifact_from_path(path: Path, field: str) -> dict[str, Any]:
    raw, metadata = read_regular(path, field)
    return {
        "bytes": len(raw),
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "stat": _stat_row(metadata),
    }


def validate_artifact(value: Any, field: str) -> dict[str, Any]:
    value = exact_keys(value, ARTIFACT_KEYS, field)
    size = integer(value["bytes"], f"{field}.bytes", 1)
    absolute_path(value["path"], f"{field}.path")
    digest(value["sha256"], f"{field}.sha256")
    metadata = exact_keys(value["stat"], STAT_KEYS, f"{field}.stat")
    exact(metadata["build_id"], None, f"{field}.stat.build_id")
    for key in STAT_KEYS - {"build_id"}:
        integer(metadata[key], f"{field}.stat.{key}")
    require(
        metadata["inode"] > 0
        and metadata["size"] == size
        and stat.S_ISREG(metadata["mode"]),
        f"E_STAT: {field}",
    )
    return value


def reopen_artifact(value: Any, field: str) -> bytes:
    value = validate_artifact(value, field)
    path = Path(value["path"])
    raw, metadata = read_regular(path, field)
    observed = {
        "bytes": len(raw),
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "stat": _stat_row(metadata),
    }
    exact(observed, value, field)
    return raw


def _reject_symlink_chain(path: Path) -> None:
    current = path
    while True:
        require(not current.is_symlink(), f"E_SYMLINK: {current}")
        if current.parent == current:
            break
        current = current.parent


def _write_file_new(path: Path, raw: bytes) -> None:
    require(path.is_absolute() and bool(raw), f"E_OUTPUT_PATH: {path}")
    require(not path.exists() and not path.is_symlink(), f"E_OUTPUT_EXISTS: {path}")
    _reject_symlink_chain(path.parent)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o644,
    )
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            require(written > 0, f"E_WRITE: {path}")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    reopened, unused_stat = read_regular(path, f"reopen.{path.name}")
    exact(reopened, raw, f"reopen.{path.name}")


def write_bundle_new(
    root: Path,
    values: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    require(root.is_absolute(), "E_OUTPUT_ROOT_ABSOLUTE")
    require(not root.exists() and not root.is_symlink(), "E_OUTPUT_ROOT_EXISTS")
    _reject_symlink_chain(root.parent)
    root.mkdir(mode=0o755)
    artifacts = {}
    for name, value in sorted(values.items()):
        require(Path(name).name == name, f"E_OUTPUT_NAME: {name}")
        raw = canonical_bytes(value)
        path = root / name
        _write_file_new(path, raw)
        artifacts[name] = artifact_from_path(path, f"output.{name}")
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent = os.open(
        root.parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
    )
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    return artifacts


def _phase_ids(outer: Any, inner: Any) -> tuple[str, str]:
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
    return outer, inner


@dataclass(frozen=True)
class CommandOutcome:
    argv: list[str]
    completed_ns: int
    pgid: int
    pid: int
    process_group_absent: bool
    returncode: int
    started_ns: int
    start_ticks: int
    stderr: bytes
    stdout: bytes
    timed_out: bool = False


class Runner(Protocol):
    def run(
        self,
        argv: list[str],
        timeout_seconds: float,
        environment: dict[str, str],
    ) -> CommandOutcome:
        ...


def _process_start_ticks(pid: int) -> int:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        close = raw.rfind(")")
        require(close >= 0, "E_PROC_STAT")
        return int(raw[close + 2:].split()[19])
    except (FileNotFoundError, PermissionError, ValueError) as error:
        raise PreflightError(f"E_PROC_STAT: {pid}") from error


def _process_absent(pid: int, start_ticks: int) -> bool:
    try:
        return _process_start_ticks(pid) != start_ticks
    except PreflightError:
        return True


class SubprocessRunner:
    def __init__(self, clock_ns: Callable[[], int] = time.monotonic_ns):
        self.clock_ns = clock_ns

    def run(
        self,
        argv: list[str],
        timeout_seconds: float,
        environment: dict[str, str],
    ) -> CommandOutcome:
        require(
            type(argv) is list
            and bool(argv)
            and all(type(value) is str and value for value in argv),
            "E_RUN_ARGV",
        )
        require(
            type(timeout_seconds) in (int, float)
            and 0 < timeout_seconds <= 1800,
            "E_RUN_TIMEOUT",
        )
        started_ns = self.clock_ns()
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            env=environment,
        )
        start_ticks = _process_start_ticks(process.pid)
        pgid = os.getpgid(process.pid)
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                stdout, stderr = process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = process.communicate()
        completed_ns = self.clock_ns()
        require(
            len(stdout) <= MAX_OUTPUT_BYTES
            and len(stderr) <= MAX_OUTPUT_BYTES,
            "E_RUN_OUTPUT_SIZE",
        )
        return CommandOutcome(
            argv=list(argv),
            completed_ns=completed_ns,
            pgid=pgid,
            pid=process.pid,
            process_group_absent=_process_absent(process.pid, start_ticks),
            returncode=process.returncode,
            started_ns=started_ns,
            start_ticks=start_ticks,
            stderr=stderr,
            stdout=stdout,
            timed_out=timed_out,
        )


def _blob(raw: bytes) -> dict[str, Any]:
    return {
        "bytes": len(raw),
        "content_base64": base64.b64encode(raw).decode("ascii"),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


class EvidenceRunner:
    def __init__(
        self,
        runner: Runner,
        environment: dict[str, str] | None = None,
    ):
        self.runner = runner
        self.environment = environment or {
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        self.receipts: list[dict[str, Any]] = []

    def run(
        self,
        receipt_id: str,
        argv: list[str],
        timeout_seconds: float,
        allowed_returncodes: set[int] = {0},
    ) -> CommandOutcome:
        require(
            len(self.receipts) < MAX_RECEIPTS
            and receipt_id
            not in {row["receipt_id"] for row in self.receipts},
            "E_RECEIPT_ID",
        )
        outcome = self.runner.run(argv, timeout_seconds, self.environment)
        exact(outcome.argv, argv, f"command.{receipt_id}.argv")
        require(
            not outcome.timed_out
            and outcome.process_group_absent
            and outcome.returncode in allowed_returncodes,
            f"E_COMMAND: {receipt_id}",
        )
        started = integer(
            outcome.started_ns,
            f"command.{receipt_id}.started_ns",
            1,
        )
        completed = integer(
            outcome.completed_ns,
            f"command.{receipt_id}.completed_ns",
            started,
        )
        row = {
            "argv": argv,
            "clock": "CONTROLLER_MONOTONIC",
            "completed_ns": completed,
            "pgid": integer(outcome.pgid, f"command.{receipt_id}.pgid", 1),
            "pid": integer(outcome.pid, f"command.{receipt_id}.pid", 1),
            "process_group_absent": True,
            "receipt_id": receipt_id,
            "returncode": outcome.returncode,
            "schema": COMMAND_RECEIPT_SCHEMA,
            "started_ns": started,
            "start_ticks": integer(
                outcome.start_ticks,
                f"command.{receipt_id}.start_ticks",
                1,
            ),
            "stderr": _blob(outcome.stderr),
            "stdout": _blob(outcome.stdout),
            "timed_out": False,
        }
        self.receipts.append(row)
        return outcome


def _path_map(value: Any, expected: set[str], field: str) -> dict[str, str]:
    value = exact_keys(value, expected, field)
    result = {}
    for role in sorted(expected):
        result[role] = absolute_path(value[role], f"{field}.{role}")
    require(len(set(result.values())) == len(result), f"E_PATH_REUSE: {field}")
    return result


def validate_plan(value: Any) -> dict[str, Any]:
    value = exact_keys(
        value,
        {
            "contract_v25",
            "inventory_config",
            "local_artifact_paths",
            "outer_phase_id",
            "phase",
            "pre_input_paths",
            "remote_artifact_paths",
            "remote_input_paths",
            "schema",
            "timeouts",
            "v24_input_paths",
            "v24_phase_id",
        },
        "plan",
    )
    exact(value["schema"], PLAN_SCHEMA, "plan.schema")
    exact(value["phase"], PHASE, "plan.phase")
    _phase_ids(value["outer_phase_id"], value["v24_phase_id"])
    validate_artifact(value["contract_v25"], "plan.contract_v25")
    local = _path_map(
        value["local_artifact_paths"],
        LOCAL_ARTIFACT_ROLES,
        "plan.local_artifact_paths",
    )
    remote = _path_map(
        value["remote_artifact_paths"],
        REMOTE_ARTIFACT_ROLES,
        "plan.remote_artifact_paths",
    )
    inputs = _path_map(
        value["remote_input_paths"],
        FAN_IN_INPUT_ROLES,
        "plan.remote_input_paths",
    )
    v24 = _path_map(
        value["v24_input_paths"],
        set(V24_INPUT_SCHEMAS),
        "plan.v24_input_paths",
    )
    pre = _path_map(
        value["pre_input_paths"],
        set(PRE_INPUT_NAMES),
        "plan.pre_input_paths",
    )
    exact(
        v24["contract_v25"],
        value["contract_v25"]["path"],
        "plan.contract_v25.path",
    )
    for role, name in PRE_INPUT_NAMES.items():
        exact(Path(pre[role]).name, name, f"plan.pre_input_paths.{role}")
    remote_pre_root = Path(inputs["pre.phase_lock"]).parents[1]
    for role, name in PRE_INPUT_NAMES.items():
        exact(
            inputs[f"pre.{role}"],
            str(remote_pre_root / "raw" / name),
            f"plan.remote_input_paths.pre.{role}",
        )
    require(
        not (set(local.values()) & set(v24.values()))
        and not (set(v24.values()) & set(pre.values())),
        "E_LOCAL_PATH_ALIAS",
    )
    require(
        not (set(remote.values()) & set(inputs.values())),
        "E_REMOTE_PATH_ALIAS",
    )
    config = exact_keys(
        value["inventory_config"],
        {
            "acquisition_started_ns",
            "desktop_forbidden_listen_ports",
            "desktop_forbidden_processes",
            "local_forward_ports",
            "output_paths",
            "phone_forbidden_listen_ports",
            "phone_forbidden_processes",
            "ssh",
        },
        "plan.inventory_config",
    )
    integer(
        config["acquisition_started_ns"],
        "plan.inventory_config.acquisition_started_ns",
        1,
    )
    exact_keys(
        config["local_forward_ports"],
        {"cuda_monolithic", "joint_phone_cuda", "remote_fan_in"},
        "plan.inventory_config.local_forward_ports",
    )
    ports = []
    for role, row in config["local_forward_ports"].items():
        row = exact_keys(
            row,
            {"local", "remote"},
            f"plan.inventory_config.local_forward_ports.{role}",
        )
        for side in ("local", "remote"):
            ports.append(integer(row[side], f"plan.port.{role}.{side}", 1))
    require(
        all(value <= 65535 for value in ports)
        and len(ports) == len(set(ports)),
        "E_PORTS",
    )
    outputs = _path_map(
        config["output_paths"],
        {
            "cuda_monolithic",
            "fan_in_acquisition",
            "fan_in_bundle_root",
            "fan_in_runtime",
            "joint_phone_cuda",
            "remote_root",
        },
        "plan.inventory_config.output_paths",
    )
    del outputs
    ssh = exact_keys(
        config["ssh"],
        {
            "connect_timeout_s",
            "identity_public_key_fingerprint",
            "shutdown_timeout_ms",
            "startup_timeout_ms",
        },
        "plan.inventory_config.ssh",
    )
    timeout = integer(ssh["connect_timeout_s"], "plan.ssh.connect_timeout_s", 1)
    require(timeout <= 60, "E_SSH_TIMEOUT")
    for key in ("shutdown_timeout_ms", "startup_timeout_ms"):
        require(
            integer(ssh[key], f"plan.ssh.{key}", 1) <= 600_000,
            f"E_TIMEOUT: {key}",
        )
    require(
        text(
            ssh["identity_public_key_fingerprint"],
            "plan.ssh.fingerprint",
            128,
        ).startswith("SHA256:"),
        "E_SSH_FINGERPRINT",
    )
    timeouts = exact_keys(
        value["timeouts"],
        {"boot_seconds", "command_seconds", "disconnect_seconds", "remote_seconds"},
        "plan.timeouts",
    )
    for key, maximum in (
        ("boot_seconds", 1800),
        ("command_seconds", 300),
        ("disconnect_seconds", 300),
        ("remote_seconds", 600),
    ):
        require(
            integer(timeouts[key], f"plan.timeouts.{key}", 1) <= maximum,
            f"E_TIMEOUT: {key}",
        )
    return value


def parse_plan(path: Path, expected_sha256: str) -> tuple[dict[str, Any], bytes]:
    digest(expected_sha256, "plan.sha256")
    value, raw = read_canonical(path, "plan")
    exact(hashlib.sha256(raw).hexdigest(), expected_sha256, "plan.sha256")
    return validate_plan(value), raw


_PROCESS_PROBE_COMMON = r'''
def fail(message):
    raise RuntimeError(message)
def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",",":"),
    ).encode("ascii")
def blob(raw):
    return {
        "bytes":len(raw),
        "content_base64":base64.b64encode(raw).decode("ascii"),
        "sha256":hashlib.sha256(raw).hexdigest(),
    }
def start_ticks(pid):
    raw=Path("/proc/%d/stat"%pid).read_text(encoding="ascii")
    close=raw.rfind(")")
    if close<0:
        fail("E_PROC_STAT")
    return int(raw[close+2:].split()[19])
def listening_inodes(port):
    result=set()
    for name in ("/proc/net/tcp","/proc/net/tcp6"):
        try:
            lines=Path(name).read_text(encoding="ascii").splitlines()[1:]
        except FileNotFoundError:
            continue
        for line in lines:
            fields=line.split()
            if len(fields)>=10 and fields[3]=="0A":
                try:
                    observed=int(fields[1].rsplit(":",1)[1],16)
                except ValueError:
                    continue
                if observed==port:
                    result.add(fields[9])
    return result
def server_process(adb_path,port):
    listeners=listening_inodes(port)
    matches=[]
    expected=[
        "adb","-L","tcp:%d"%port,
        "fork-server","server","--reply-fd",
    ]
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            executable=os.readlink(entry/"exe")
            raw=(entry/"cmdline").read_bytes()
            argv=[
                value.decode("ascii")
                for value in raw.split(b"\0")
                if value
            ]
            sockets={
                target[8:-1]
                for fd in (entry/"fd").iterdir()
                if (target:=os.readlink(fd)).startswith("socket:[")
            }
        except (FileNotFoundError,PermissionError,UnicodeDecodeError,OSError):
            continue
        if (
            os.path.realpath(executable)==os.path.realpath(adb_path)
            and len(argv)==7
            and os.path.basename(argv[0])=="adb"
            and [os.path.basename(argv[0]),*argv[1:6]]==expected
            and argv[6].isdigit()
            and sockets&listeners
        ):
            matches.append({
                "argv":[os.path.basename(argv[0]),*argv[1:]],
                "executable_path":adb_path,
                "listen_host":"127.0.0.1",
                "listen_port":port,
                "pid":int(entry.name),
                "start_ticks":start_ticks(int(entry.name)),
            })
    if len(matches)!=1:
        fail("E_ADB_SERVER_COUNT")
    return matches[0]
'''.strip()


LOCAL_ADB_PROBE_SOURCE = (
    "import base64,hashlib,json,os\n"
    "from pathlib import Path\n"
    "import time\n"
    + _PROCESS_PROBE_COMMON
    + r'''
payload=json.loads(base64.b64decode(__import__("sys").argv[1]).decode("ascii"))
boot=Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
server=server_process(payload["adb_path"],payload["port"])
server["boot_id"]=boot
result={
    "boot_id":boot,
    "clock":"CONTROLLER_MONOTONIC_RAW",
    "observed_ns":time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
    "schema":"s39-v25-preflight-local-adb-probe-v1",
    "server":server,
}
print(base64.b64encode(canonical(result)).decode("ascii"))
'''
)


REMOTE_PROBE_SOURCE = (
    "import base64,hashlib,json,os,stat,subprocess,time\n"
    "from pathlib import Path\n"
    + _PROCESS_PROBE_COMMON
    + r'''
def decode():
    raw=base64.b64decode(__import__("sys").argv[1],validate=True)
    value=json.loads(raw.decode("ascii"))
    if canonical(value)!=raw:
        fail("E_PAYLOAD_CANONICAL")
    return value
def identity(value):
    return (
        value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,
        value.st_ctime_ns,value.st_mode
    )
def snapshot(path_text):
    path=Path(path_text)
    if not path.is_absolute() or str(path)!=path_text or ".." in path.parts:
        fail("E_PATH")
    flags=os.O_RDONLY|os.O_CLOEXEC
    if hasattr(os,"O_NOFOLLOW"):
        flags|=os.O_NOFOLLOW
    descriptor=os.open(path,flags)
    try:
        before=os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0<before.st_size<=536870912:
            fail("E_FILE")
        digest_value=hashlib.sha256()
        consumed=0
        while True:
            block=os.read(descriptor,1048576)
            if not block:
                break
            consumed+=len(block)
            digest_value.update(block)
        after=os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if identity(before)!=identity(after) or consumed!=before.st_size:
        fail("E_FILE_CHANGED")
    return {
        "bytes":consumed,
        "path":path_text,
        "sha256":digest_value.hexdigest(),
        "stat":{
            "build_id":None,
            "ctime_ns":before.st_ctime_ns,
            "device_id":before.st_dev,
            "inode":before.st_ino,
            "mode":before.st_mode,
            "mtime_ns":before.st_mtime_ns,
            "size":before.st_size,
        },
    }
def child(argv):
    started=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    result=subprocess.run(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
        env={
            "LANG":"C","LC_ALL":"C","PATH":"/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE":"1",
        },
    )
    completed=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    return {
        "argv":argv,
        "clock":"RTX_MONOTONIC_RAW",
        "completed_ns":completed,
        "returncode":result.returncode,
        "started_ns":started,
        "stderr":blob(result.stderr),
        "stdout":blob(result.stdout),
    },result
def swap_used():
    values={}
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        fields=line.split()
        if fields and fields[0] in ("SwapTotal:","SwapFree:"):
            values[fields[0]]=int(fields[1])*1024
    if set(values)!={"SwapTotal:","SwapFree:"}:
        fail("E_SWAP")
    value=values["SwapTotal:"]-values["SwapFree:"]
    if value<0:
        fail("E_SWAP")
    return value
payload=decode()
boot_path=Path("/proc/sys/kernel/random/boot_id")
boot_before=boot_path.read_text(encoding="ascii").strip()
if (
    payload["expected_boot_id"] is not None
    and boot_before!=payload["expected_boot_id"]
):
    fail("E_BOOT")
host=os.uname().nodename
if host!=payload["expected_host"]:
    fail("E_HOST")
paths=[*payload["artifacts"].values(),*payload["inputs"].values()]
if len(paths)!=len(set(paths)):
    fail("E_PATH_ALIAS")
artifacts={
    role:snapshot(path)
    for role,path in sorted(payload["artifacts"].items())
}
inputs={
    role:snapshot(path)
    for role,path in sorted(payload["inputs"].items())
}
adb_receipt,adb_result=child([
    payload["artifacts"]["adb"],"-L","tcp:%d"%payload["adb_port"],
    "start-server",
])
if adb_result.returncode!=0:
    fail("E_ADB_START")
server=server_process(payload["artifacts"]["adb"],payload["adb_port"])
server["boot_id"]=boot_before
gpu_argv=[
    payload["artifacts"]["nvidia_smi"],
    "--id",payload["expected_gpu_uuid"],
    "--query-gpu=uuid,name,memory.total,pci.bus_id",
    "--format=csv,noheader,nounits",
]
gpu_receipt,gpu_result=child(gpu_argv)
if gpu_result.returncode!=0:
    fail("E_GPU_COMMAND")
lines=gpu_result.stdout.decode("ascii").strip().splitlines()
if len(lines)!=1:
    fail("E_GPU_ROWS")
fields=[value.strip() for value in lines[0].split(",")]
if len(fields)!=4:
    fail("E_GPU_FIELDS")
gpu_uuid,name,memory_mib,pci_bus_id=fields
if gpu_uuid!=payload["expected_gpu_uuid"]:
    fail("E_GPU_UUID")
try:
    memory_total_bytes=int(memory_mib)*1024*1024
except ValueError:
    fail("E_GPU_MEMORY")
boot_after=boot_path.read_text(encoding="ascii").strip()
if boot_after!=boot_before:
    fail("E_BOOT_CHANGED")
result={
    "artifacts":artifacts,
    "boot_id":boot_after,
    "child_receipts":{
        "adb_start":adb_receipt,
        "gpu_query":gpu_receipt,
    },
    "clock":"RTX_MONOTONIC_RAW",
    "gpu":{
        "memory_total_bytes":memory_total_bytes,
        "name":name,
        "pci_bus_id":pci_bus_id,
        "uuid":gpu_uuid,
    },
    "host":host,
    "inputs":inputs,
    "observed_ns":time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
    "schema":"s39-v25-preflight-remote-probe-v1",
    "server":server,
    "system_swap_used_bytes":swap_used(),
}
print(base64.b64encode(canonical(result)).decode("ascii"))
'''
)


PHONE_STATUS_SOURCE = r'''set -eu
printf 'BOOT_ID='
cat /proc/sys/kernel/random/boot_id
printf 'BOOT_COMPLETED='
getprop sys.boot_completed
printf 'PRODUCT='
getprop ro.product.name
printf 'MODEL='
getprop ro.product.model
printf 'DEVICE='
getprop ro.product.device
printf 'INTERFACE=wlan0\n'
ip -4 -o addr show dev wlan0 scope global | awk 'NR==1{split($4,a,"/");print "LOCAL_IPV4="a[1]}END{if(NR!=1)exit 43}'
awk '/^MemAvailable:/{print "MEM_AVAILABLE_KB="$2}' /proc/meminfo
awk '/^SwapTotal:/{total=$2}/^SwapFree:/{free=$2}END{print "SWAP_USED_KB="total-free}' /proc/meminfo
dumpsys thermalservice | awk -F: '/Thermal Status:/{gsub(/[[:space:]]/,"",$2);print "THERMAL_STATUS="$2;found=1;exit}END{if(!found)exit 42}'
'''.strip()


CONTROLLER_PROBE_SOURCE = r'''import base64,json,os,time
from pathlib import Path
value={
    "boot_id":Path("/proc/sys/kernel/random/boot_id").read_text(
        encoding="ascii"
    ).strip(),
    "clock":"CONTROLLER_MONOTONIC_RAW",
    "host":os.uname().nodename,
    "observed_ns":time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
    "schema":"s39-v25-preflight-controller-probe-v1",
}
raw=json.dumps(
    value,ensure_ascii=True,sort_keys=True,separators=(",",":")
).encode("ascii")
print(base64.b64encode(raw).decode("ascii"))
'''


def _decode_probe(raw: bytes, field: str) -> dict[str, Any]:
    try:
        decoded = base64.b64decode(raw.strip(), validate=True)
    except ValueError as error:
        raise PreflightError(f"E_PROBE_BASE64: {field}") from error
    value = parse_json(decoded, field)
    require(type(value) is dict, f"E_PROBE_TYPE: {field}")
    exact(canonical_compact(value), decoded, f"{field}.canonical")
    return value


def _controller_probe(
    evidence: EvidenceRunner,
    receipt_id: str,
    plan: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    argv = [
        plan["local_artifact_paths"]["python"],
        "-I",
        "-c",
        CONTROLLER_PROBE_SOURCE,
    ]
    outcome = evidence.run(
        receipt_id,
        argv,
        plan["timeouts"]["command_seconds"],
    )
    value = _decode_probe(outcome.stdout, receipt_id)
    exact_keys(
        value,
        {"boot_id", "clock", "host", "observed_ns", "schema"},
        receipt_id,
    )
    exact(
        value["schema"],
        "s39-v25-preflight-controller-probe-v1",
        f"{receipt_id}.schema",
    )
    exact(
        value["clock"],
        "CONTROLLER_MONOTONIC_RAW",
        f"{receipt_id}.clock",
    )
    exact(
        value["host"],
        contract["topology"]["controller_host"],
        f"{receipt_id}.host",
    )
    uuid(value["boot_id"], f"{receipt_id}.boot_id")
    integer(value["observed_ns"], f"{receipt_id}.observed_ns", 1)
    return value


def _snapshot_local_artifacts(
    plan: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    result = {
        role: artifact_from_path(Path(path), f"local_artifact.{role}")
        for role, path in sorted(plan["local_artifact_paths"].items())
    }
    composition = contract["composition"]
    for role, (section, name) in sorted(LOCAL_SOURCE_BINDINGS.items()):
        source = composition[section][name]
        exact(
            (result[role]["bytes"], result[role]["sha256"]),
            (source["bytes"], source["sha256"]),
            f"local_artifact.{role}.content",
        )
    for role in ("adb", "python", "ssh", "ssh_keygen"):
        require(
            bool(result[role]["stat"]["mode"] & 0o111),
            f"E_EXECUTABLE: local_artifact.{role}",
        )
    exact(
        os.path.realpath(result["python"]["path"]),
        os.path.realpath(sys.executable),
        "local_artifact.python.executable",
    )
    return result


def _validate_contract(
    contract: dict[str, Any],
    plan: dict[str, Any],
) -> None:
    exact(
        contract.get("schema"),
        "s39-cp0-r1-evidence-contract-v2.5",
        "contract.schema",
    )
    exact(contract.get("phase"), PHASE, "contract.phase")
    topology = contract["topology"]
    exact(
        topology["cuda_ssh_target"],
        "zhihao@172.20.74.85",
        "contract.cuda_ssh_target",
    )
    exact(
        topology["cuda_gpu_uuid"],
        contract["devices"]["cuda"]["uuid"],
        "contract.cuda_gpu_uuid",
    )
    exact(
        contract["devices"]["cuda"]["host"],
        topology["cuda_host"],
        "contract.cuda_host",
    )
    exact(
        plan["remote_artifact_paths"]["python"],
        topology["cuda_python_path"],
        "plan.remote_python",
    )
    for phone in ("op12", "op15"):
        expected = contract["devices"][phone]
        for key in ("serial", "device", "model", "product"):
            text(expected[key], f"contract.devices.{phone}.{key}", 256)


def _validate_remote_artifact_bindings(
    artifacts: dict[str, dict[str, Any]],
    local: dict[str, dict[str, Any]],
    contract: dict[str, Any],
) -> None:
    for role, (section, name) in sorted(REMOTE_SOURCE_BINDINGS.items()):
        source = contract["composition"][section][name]
        exact(
            (artifacts[role]["bytes"], artifacts[role]["sha256"]),
            (source["bytes"], source["sha256"]),
            f"remote_artifact.{role}.content",
        )
    topology = contract["topology"]
    exact(
        (
            artifacts["python"]["bytes"],
            artifacts["python"]["path"],
            artifacts["python"]["sha256"],
        ),
        (
            topology["cuda_python_bytes"],
            topology["cuda_python_path"],
            topology["cuda_python_sha256"],
        ),
        "remote_artifact.python",
    )
    exact(
        (
            artifacts["phone_guard"]["bytes"],
            artifacts["phone_guard"]["sha256"],
        ),
        (
            local["phone_guard"]["bytes"],
            local["phone_guard"]["sha256"],
        ),
        "remote_artifact.phone_guard",
    )


def _verify_ssh_fingerprint(
    evidence: EvidenceRunner,
    plan: dict[str, Any],
) -> None:
    outcome = evidence.run(
        "local.ssh_fingerprint",
        [
            plan["local_artifact_paths"]["ssh_keygen"],
            "-lf",
            plan["local_artifact_paths"]["identity_public_key"],
        ],
        plan["timeouts"]["command_seconds"],
    )
    try:
        fields = outcome.stdout.decode("ascii").strip().split()
    except UnicodeDecodeError as error:
        raise PreflightError("E_SSH_FINGERPRINT_ASCII") from error
    require(len(fields) >= 2, "E_SSH_FINGERPRINT_OUTPUT")
    exact(
        fields[1],
        plan["inventory_config"]["ssh"]["identity_public_key_fingerprint"],
        "ssh.fingerprint",
    )


def _start_local_adb(
    evidence: EvidenceRunner,
    plan: dict[str, Any],
) -> None:
    evidence.run(
        "local.adb_start",
        [
            plan["local_artifact_paths"]["adb"],
            "-P",
            str(ADB_PORT),
            "start-server",
        ],
        plan["timeouts"]["command_seconds"],
    )


def _receipt(
    evidence: EvidenceRunner,
    receipt_id: str,
) -> dict[str, Any]:
    rows = [
        row
        for row in evidence.receipts
        if row["receipt_id"] == receipt_id
    ]
    require(len(rows) == 1, f"E_RECEIPT_LOOKUP: {receipt_id}")
    return rows[0]


def _capture_phone_boot(
    evidence: EvidenceRunner,
    plan: dict[str, Any],
    contract: dict[str, Any],
    phone: str,
    receipt_id: str,
) -> str:
    serial = contract["devices"][phone]["serial"]
    outcome = evidence.run(
        receipt_id,
        _adb_argv(
            plan["local_artifact_paths"]["adb"],
            serial,
            "shell",
            "cat",
            "/proc/sys/kernel/random/boot_id",
        ),
        plan["timeouts"]["command_seconds"],
    )
    try:
        value = outcome.stdout.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise PreflightError(f"E_BOOT_ASCII: {phone}") from error
    return uuid(value, f"{receipt_id}.boot_id")


def _wait_for_disconnect(
    evidence: EvidenceRunner,
    plan: dict[str, Any],
    contract: dict[str, Any],
    phone: str,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> tuple[str, int]:
    deadline = monotonic() + plan["timeouts"]["disconnect_seconds"]
    serial = contract["devices"][phone]["serial"]
    attempt = 0
    while monotonic() < deadline:
        receipt_id = f"phone.{phone}.disconnect.{attempt:03d}"
        outcome = evidence.run(
            receipt_id,
            _adb_argv(
                plan["local_artifact_paths"]["adb"],
                serial,
                "get-state",
            ),
            plan["timeouts"]["command_seconds"],
            set(range(256)),
        )
        if outcome.returncode != 0:
            return receipt_id, outcome.completed_ns
        attempt += 1
        sleep(0.1)
    raise PreflightError(f"E_PHONE_NO_DISCONNECT: {phone}")


def _wait_for_phone_status(
    evidence: EvidenceRunner,
    plan: dict[str, Any],
    contract: dict[str, Any],
    phone: str,
    before_boot_id: str,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> tuple[dict[str, Any], str]:
    deadline = monotonic() + plan["timeouts"]["boot_seconds"]
    serial = contract["devices"][phone]["serial"]
    attempt = 0
    minimum = contract["gates"]["phone_minimum_available_bytes"]
    while monotonic() < deadline:
        receipt_id = f"phone.{phone}.status.{attempt:03d}"
        outcome = evidence.run(
            receipt_id,
            _adb_argv(
                plan["local_artifact_paths"]["adb"],
                serial,
                "shell",
                "sh",
                "-c",
                PHONE_STATUS_SOURCE,
            ),
            plan["timeouts"]["command_seconds"],
            set(range(256)),
        )
        if outcome.returncode == 0:
            try:
                observed = _phone_status(outcome.stdout, contract, phone)
            except PreflightError:
                observed = None
            if (
                observed is not None
                and observed["boot_completed"]
                and observed["boot_id"] != before_boot_id
                and observed["available_bytes"] >= minimum
                and observed["thermal_status"] == 0
            ):
                return observed, receipt_id
        attempt += 1
        sleep(0.1)
    raise PreflightError(f"E_PHONE_BOOT_TIMEOUT: {phone}")


def _manifest(
    schema: str,
    plan_sha256: str,
    artifacts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows = [
        {
            "bytes": row["bytes"],
            "path": name,
            "role": role,
            "sha256": row["sha256"],
        }
        for role, (name, row) in sorted(artifacts.items())
    ]
    return {
        "artifacts": rows,
        "phase": PHASE,
        "plan_sha256": plan_sha256,
        "schema": schema,
    }


def _publish_manifest(
    root: Path,
    name: str,
    value: dict[str, Any],
) -> dict[str, Any]:
    path = root / name
    raw = canonical_bytes(value)
    _write_file_new(path, raw)
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return artifact_from_path(path, f"output.{name}")


def capture_preflight(
    *,
    plan: dict[str, Any],
    plan_raw: bytes,
    output_root: Path,
    confirmation: str,
    runner: Runner | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    exact(confirmation, CONFIRM_CAPTURE, "capture.confirmation")
    plan = validate_plan(plan)
    exact(canonical_bytes(plan), plan_raw, "capture.plan_raw")
    plan_sha256 = hashlib.sha256(plan_raw).hexdigest()
    contract_raw = reopen_artifact(plan["contract_v25"], "contract_v25")
    contract = parse_json(contract_raw, "contract_v25")
    require(type(contract) is dict, "E_CONTRACT_TYPE")
    exact(canonical_bytes(contract), contract_raw, "contract_v25.canonical")
    _validate_contract(contract, plan)
    local_artifacts = _snapshot_local_artifacts(plan, contract)
    evidence = EvidenceRunner(runner or SubprocessRunner())

    controller_before = _controller_probe(
        evidence,
        "controller.before",
        plan,
        contract,
    )
    preparation_started_ns = _receipt(
        evidence,
        "controller.before",
    )["started_ns"]
    _verify_ssh_fingerprint(evidence, plan)
    _start_local_adb(evidence, plan)
    local_adb_before = _local_adb_probe(
        evidence,
        "local.adb_server.before",
        plan,
        controller_before["boot_id"],
    )

    adb = plan["local_artifact_paths"]["adb"]
    devices_before_outcome = evidence.run(
        "phones.usb.before",
        [adb, "-P", str(ADB_PORT), "devices", "-l"],
        plan["timeouts"]["command_seconds"],
    )
    usb_before = _parse_devices(
        devices_before_outcome.stdout,
        contract,
        "phones.usb.before",
    )
    before_boot_ids = {
        phone: _capture_phone_boot(
            evidence,
            plan,
            contract,
            phone,
            f"phone.{phone}.boot.before",
        )
        for phone in ("op12", "op15")
    }
    require(
        len(set(before_boot_ids.values())) == 2,
        "E_BEFORE_BOOT_ALIAS",
    )

    reboot_receipts = {}
    for phone in ("op12", "op15"):
        serial = contract["devices"][phone]["serial"]
        receipt_id = f"phone.{phone}.reboot"
        evidence.run(
            receipt_id,
            _adb_argv(adb, serial, "reboot"),
            plan["timeouts"]["command_seconds"],
        )
        reboot_receipts[phone] = receipt_id
    reboot_started_ns = min(
        _receipt(evidence, receipt_id)["started_ns"]
        for receipt_id in reboot_receipts.values()
    )

    disconnect_receipts = {}
    disconnected_ns = {}
    for phone in ("op12", "op15"):
        receipt_id, observed_ns = _wait_for_disconnect(
            evidence,
            plan,
            contract,
            phone,
            monotonic,
            sleep,
        )
        disconnect_receipts[phone] = receipt_id
        disconnected_ns[phone] = observed_ns

    wait_receipts = {}
    for phone in ("op12", "op15"):
        serial = contract["devices"][phone]["serial"]
        receipt_id = f"phone.{phone}.wait"
        evidence.run(
            receipt_id,
            _adb_argv(adb, serial, "wait-for-device"),
            plan["timeouts"]["boot_seconds"],
        )
        wait_receipts[phone] = receipt_id
    preparation_completed_ns = max(
        _receipt(evidence, receipt_id)["completed_ns"]
        for receipt_id in wait_receipts.values()
    )

    devices_after_outcome = evidence.run(
        "phones.usb.after",
        [adb, "-P", str(ADB_PORT), "devices", "-l"],
        plan["timeouts"]["command_seconds"],
    )
    discovery_started_ns = devices_after_outcome.started_ns
    require(
        preparation_completed_ns <= discovery_started_ns,
        "E_DISCOVERY_BEFORE_PREPARATION",
    )
    usb_after = _parse_devices(
        devices_after_outcome.stdout,
        contract,
        "phones.usb.after",
    )
    phones = {}
    status_receipts = {}
    for phone in ("op12", "op15"):
        observed, receipt_id = _wait_for_phone_status(
            evidence,
            plan,
            contract,
            phone,
            before_boot_ids[phone],
            monotonic,
            sleep,
        )
        for key in ("device", "model", "product", "serial"):
            usb_key = "physical_serial" if key == "serial" else key
            exact(
                observed[key],
                usb_after[phone][usb_key],
                f"E_USB_STATUS_SPLICE: {phone}.{key}",
            )
        phones[phone] = observed
        status_receipts[phone] = receipt_id
    require(
        phones["op12"]["local_ipv4"] != phones["op15"]["local_ipv4"],
        "E_SELECTOR_ALIAS",
    )

    remote = _remote_probe(
        evidence,
        "rtx.capture",
        plan,
        contract,
        None,
        False,
    )
    _validate_remote_artifact_bindings(
        remote["artifacts"],
        local_artifacts,
        contract,
    )
    controller_after = _controller_probe(
        evidence,
        "controller.after",
        plan,
        contract,
    )
    exact(
        controller_after["boot_id"],
        controller_before["boot_id"],
        "E_CONTROLLER_BOOT_CHANGED",
    )
    local_adb_after = _local_adb_probe(
        evidence,
        "local.adb_server.after",
        plan,
        controller_before["boot_id"],
    )
    exact(
        local_adb_after["server"],
        local_adb_before["server"],
        "E_LOCAL_ADB_SERVER_CHANGED",
    )
    discovery_completed_ns = _receipt(
        evidence,
        "local.adb_server.after",
    )["completed_ns"]

    preparation = {
        "completed_ns": preparation_completed_ns,
        "controller": {
            "boot_id": controller_before["boot_id"],
            "host": controller_before["host"],
        },
        "local_python": local_artifacts["python"],
        "phase": PHASE,
        "phase_id": plan["outer_phase_id"],
        "phones": {
            phone: {
                "adb_path": local_artifacts["adb"]["path"],
                "adb_port": ADB_PORT,
                "adb_sha256": local_artifacts["adb"]["sha256"],
                "boot_id_before": before_boot_ids[phone],
                "disconnected_ns": disconnected_ns[phone],
                "physical_serial": contract["devices"][phone]["serial"],
                "reboot_argv": _receipt(
                    evidence,
                    reboot_receipts[phone],
                )["argv"],
                "reboot_returncode": _receipt(
                    evidence,
                    reboot_receipts[phone],
                )["returncode"],
                "requested_ns": _receipt(
                    evidence,
                    reboot_receipts[phone],
                )["started_ns"],
            }
            for phone in ("op12", "op15")
        },
        "schema": PREPARATION_SCHEMA,
        "started_ns": preparation_started_ns,
    }
    discovery = {
        "completed_ns": discovery_completed_ns,
        "controller": {
            "boot_id": controller_after["boot_id"],
            "host": controller_after["host"],
        },
        "cuda": {
            "boot_id": remote["boot_id"],
            "gpu_uuid": remote["gpu"]["uuid"],
            "host": remote["host"],
            "memory_total_bytes": remote["gpu"]["memory_total_bytes"],
            "name": remote["gpu"]["name"],
            "ssh_target": contract["topology"]["cuda_ssh_target"],
            "system_swap_used_bytes": remote["system_swap_used_bytes"],
        },
        "phase": PHASE,
        "phase_id": plan["outer_phase_id"],
        "phones": {
            phone: {
                "adb_port": ADB_PORT,
                "attested_ns": _receipt(
                    evidence,
                    status_receipts[phone],
                )["completed_ns"],
                "boot_id": phones[phone]["boot_id"],
                "device": phones[phone]["device"],
                "interface": phones[phone]["interface"],
                "model": phones[phone]["model"],
                "physical_serial": phones[phone]["serial"],
                "product": phones[phone]["product"],
                "system_swap_used_bytes": phones[phone][
                    "system_swap_used_bytes"
                ],
                "usb_observed_ns": _receipt(
                    evidence,
                    "phones.usb.after",
                )["completed_ns"],
                "wifi_ipv4": phones[phone]["local_ipv4"],
                "wifi_selector": f"{phones[phone]['local_ipv4']}:5555",
            }
            for phone in ("op12", "op15")
        },
        "preparation_completed_ns": preparation_completed_ns,
        "preparation_sha256": hashlib.sha256(
            canonical_bytes(preparation)
        ).hexdigest(),
        "schema": DISCOVERY_SCHEMA,
        "started_ns": discovery_started_ns,
    }
    projection = {
        "before_boot_ids": before_boot_ids,
        "completed_ns": discovery_completed_ns,
        "devices": {
            "cuda": {
                "gpu_uuid": remote["gpu"]["uuid"],
                "host": remote["host"],
                "host_boot_id": remote["boot_id"],
                "pci_bus_id": remote["gpu"]["pci_bus_id"],
                "system_swap_used_bytes": remote[
                    "system_swap_used_bytes"
                ],
            },
            **{
                phone: {
                    "available_bytes": phones[phone]["available_bytes"],
                    "boot_id": phones[phone]["boot_id"],
                    "device": phones[phone]["device"],
                    "interface": phones[phone]["interface"],
                    "local_ipv4": phones[phone]["local_ipv4"],
                    "model": phones[phone]["model"],
                    "product": phones[phone]["product"],
                    "serial": phones[phone]["serial"],
                    "system_swap_used_bytes": phones[phone][
                        "system_swap_used_bytes"
                    ],
                    "thermal_status": phones[phone]["thermal_status"],
                }
                for phone in ("op12", "op15")
            },
        },
        "outer_phase_id": plan["outer_phase_id"],
        "phase": PHASE,
        "receipt_ids": {
            "disconnect": disconnect_receipts,
            "reboot": reboot_receipts,
            "status": status_receipts,
            "wait": wait_receipts,
        },
        "reboot_started_ns": reboot_started_ns,
        "schema": PROJECTION_SCHEMA,
        "started_ns": preparation_started_ns,
        "v24_phase_id": plan["v24_phase_id"],
    }
    evidence_value = {
        "command_receipts": evidence.receipts,
        "completed_ns": discovery_completed_ns,
        "controller": {
            "after": controller_after,
            "before": controller_before,
        },
        "discovery_sha256": hashlib.sha256(
            canonical_bytes(discovery)
        ).hexdigest(),
        "local_adb_server": {
            "after": local_adb_after,
            "before": local_adb_before,
        },
        "local_artifacts": local_artifacts,
        "outer_phase_id": plan["outer_phase_id"],
        "phase": PHASE,
        "plan_sha256": plan_sha256,
        "post_reboot_devices": phones,
        "preparation_sha256": hashlib.sha256(
            canonical_bytes(preparation)
        ).hexdigest(),
        "projection_sha256": hashlib.sha256(
            canonical_bytes(projection)
        ).hexdigest(),
        "remote_probe": remote,
        "schema": CAPTURE_EVIDENCE_SCHEMA,
        "started_ns": preparation_started_ns,
        "usb": {"after": usb_after, "before": usb_before},
        "v24_phase_id": plan["v24_phase_id"],
    }
    output_values = {
        "phase-discovery.json": discovery,
        "phase-preparation.json": preparation,
        "preflight-capture-evidence.json": evidence_value,
        "v24-preparation-projection.json": projection,
    }
    artifacts = write_bundle_new(output_root, output_values)
    manifest = _manifest(
        CAPTURE_MANIFEST_SCHEMA,
        plan_sha256,
        {
            "phase.discovery": (
                "phase-discovery.json",
                artifacts["phase-discovery.json"],
            ),
            "phase.preparation": (
                "phase-preparation.json",
                artifacts["phase-preparation.json"],
            ),
            "preflight.capture_evidence": (
                "preflight-capture-evidence.json",
                artifacts["preflight-capture-evidence.json"],
            ),
            "preflight.v24_projection": (
                "v24-preparation-projection.json",
                artifacts["v24-preparation-projection.json"],
            ),
        },
    )
    _publish_manifest(
        output_root,
        "CAPTURE_MANIFEST.json",
        manifest,
    )
    return {
        "discovery": discovery,
        "evidence": evidence_value,
        "manifest": manifest,
        "preparation": preparation,
        "projection": projection,
    }


def _parse_assignments(raw: bytes, expected: set[str], field: str) -> dict[str, str]:
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise PreflightError(f"E_ASCII: {field}") from error
    values = {}
    for line in lines:
        require("=" in line, f"E_ASSIGNMENT: {field}")
        key, value = line.split("=", 1)
        require(key and key not in values, f"E_ASSIGNMENT: {field}.{key}")
        values[key] = value
    exact(set(values), expected, f"{field}.keys")
    return values


def _parse_devices(
    raw: bytes,
    contract: dict[str, Any],
    field: str,
) -> dict[str, dict[str, str]]:
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise PreflightError(f"E_ASCII: {field}") from error
    require(lines and lines[0] == "List of devices attached", f"E_ADB_HEADER: {field}")
    rows = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        fields = line.split()
        require(len(fields) >= 2 and fields[1] == "device", f"E_ADB_STATE: {field}")
        serial = fields[0]
        metadata = {}
        for item in fields[2:]:
            if ":" in item:
                key, value = item.split(":", 1)
                metadata[key] = value
        rows[serial] = metadata
    expected_serials = {
        contract["devices"][phone]["serial"]
        for phone in ("op12", "op15")
    }
    exact(set(rows), expected_serials, f"{field}.serials")
    result = {}
    for phone in ("op12", "op15"):
        expected = contract["devices"][phone]
        metadata = rows[expected["serial"]]
        for key in ("device", "model", "product"):
            exact(metadata.get(key), expected[key], f"{field}.{phone}.{key}")
        result[phone] = {
            "device": metadata["device"],
            "model": metadata["model"],
            "physical_serial": expected["serial"],
            "product": metadata["product"],
        }
    return result


def _positive_kib(value: str, field: str, allow_zero: bool = False) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise PreflightError(f"E_INTEGER: {field}") from error
    require(parsed >= (0 if allow_zero else 1), f"E_INTEGER: {field}")
    return parsed * 1024


def _phone_status(
    raw: bytes,
    contract: dict[str, Any],
    phone: str,
) -> dict[str, Any]:
    field = f"phone_status.{phone}"
    values = _parse_assignments(
        raw,
        {
            "BOOT_COMPLETED",
            "BOOT_ID",
            "DEVICE",
            "INTERFACE",
            "LOCAL_IPV4",
            "MEM_AVAILABLE_KB",
            "MODEL",
            "PRODUCT",
            "SWAP_USED_KB",
            "THERMAL_STATUS",
        },
        field,
    )
    expected = contract["devices"][phone]
    uuid(values["BOOT_ID"], f"{field}.boot_id")
    for source, key in (
        ("DEVICE", "device"),
        ("MODEL", "model"),
        ("PRODUCT", "product"),
    ):
        exact(values[source], expected[key], f"{field}.{key}")
    exact(values["INTERFACE"], "wlan0", f"{field}.interface")
    try:
        address = ipaddress.IPv4Address(values["LOCAL_IPV4"])
    except ipaddress.AddressValueError as error:
        raise PreflightError(f"E_IPV4: {phone}") from error
    require(
        not (
            address.is_unspecified
            or address.is_loopback
            or address.is_multicast
        ),
        f"E_IPV4: {phone}",
    )
    try:
        thermal = int(values["THERMAL_STATUS"])
    except ValueError as error:
        raise PreflightError(f"E_THERMAL: {phone}") from error
    return {
        "available_bytes": _positive_kib(
            values["MEM_AVAILABLE_KB"],
            f"{field}.available",
        ),
        "boot_completed": values["BOOT_COMPLETED"] == "1",
        "boot_id": values["BOOT_ID"],
        "device": expected["device"],
        "interface": "wlan0",
        "local_ipv4": str(address),
        "model": expected["model"],
        "product": expected["product"],
        "serial": expected["serial"],
        "system_swap_used_bytes": _positive_kib(
            values["SWAP_USED_KB"],
            f"{field}.swap",
            allow_zero=True,
        ),
        "thermal_status": thermal,
    }


def _adb_argv(adb: str, serial: str, *arguments: str) -> list[str]:
    return [adb, "-P", str(ADB_PORT), "-s", serial, *arguments]


def _ssh_argv(
    plan: dict[str, Any],
    contract: dict[str, Any],
    payload: dict[str, Any],
) -> list[str]:
    local = plan["local_artifact_paths"]
    ssh = plan["inventory_config"]["ssh"]
    target = contract["topology"]["cuda_ssh_target"]
    host_alias = target.rsplit("@", 1)[-1]
    encoded = base64.b64encode(canonical_compact(payload)).decode("ascii")
    return [
        local["ssh"],
        "-F",
        "/dev/null",
        "-i",
        local["identity_file"],
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={ssh['connect_timeout_s']}",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={local['known_hosts']}",
        "-o",
        f"HostKeyAlias={host_alias}",
        "-p",
        "22",
        target,
        plan["remote_artifact_paths"]["python"],
        "-I",
        "-c",
        REMOTE_PROBE_SOURCE,
        encoded,
    ]


def _remote_probe(
    evidence: EvidenceRunner,
    receipt_id: str,
    plan: dict[str, Any],
    contract: dict[str, Any],
    expected_boot_id: str | None,
    include_inputs: bool,
) -> dict[str, Any]:
    payload = {
        "adb_port": ADB_PORT,
        "artifacts": plan["remote_artifact_paths"],
        "expected_boot_id": expected_boot_id,
        "expected_gpu_uuid": contract["topology"]["cuda_gpu_uuid"],
        "expected_host": contract["topology"]["cuda_host"],
        "inputs": plan["remote_input_paths"] if include_inputs else {},
    }
    outcome = evidence.run(
        receipt_id,
        _ssh_argv(plan, contract, payload),
        plan["timeouts"]["remote_seconds"],
    )
    return validate_remote_probe(
        _decode_probe(outcome.stdout, receipt_id),
        plan,
        contract,
        include_inputs,
    )


def _local_adb_probe(
    evidence: EvidenceRunner,
    receipt_id: str,
    plan: dict[str, Any],
    expected_boot_id: str,
) -> dict[str, Any]:
    payload = {
        "adb_path": plan["local_artifact_paths"]["adb"],
        "port": ADB_PORT,
    }
    argv = [
        plan["local_artifact_paths"]["python"],
        "-I",
        "-c",
        LOCAL_ADB_PROBE_SOURCE,
        base64.b64encode(canonical_compact(payload)).decode("ascii"),
    ]
    outcome = evidence.run(
        receipt_id,
        argv,
        plan["timeouts"]["command_seconds"],
    )
    value = _decode_probe(outcome.stdout, receipt_id)
    exact(value.get("schema"), LOCAL_ADB_PROBE_SCHEMA, f"{receipt_id}.schema")
    exact(value.get("boot_id"), expected_boot_id, f"{receipt_id}.boot_id")
    exact(value.get("clock"), "CONTROLLER_MONOTONIC_RAW", f"{receipt_id}.clock")
    integer(value.get("observed_ns"), f"{receipt_id}.observed_ns", 1)
    server = _validate_adb_server(
        value.get("server"),
        plan["local_artifact_paths"]["adb"],
        expected_boot_id,
        f"{receipt_id}.server",
    )
    return {**value, "server": server}


def _validate_adb_server(
    value: Any,
    expected_path: str,
    expected_boot_id: str,
    field: str,
) -> dict[str, Any]:
    value = exact_keys(
        value,
        {
            "argv",
            "boot_id",
            "executable_path",
            "listen_host",
            "listen_port",
            "pid",
            "start_ticks",
        },
        field,
    )
    exact(value["boot_id"], expected_boot_id, f"{field}.boot_id")
    exact(value["executable_path"], expected_path, f"{field}.executable_path")
    exact(value["listen_host"], "127.0.0.1", f"{field}.listen_host")
    exact(value["listen_port"], ADB_PORT, f"{field}.listen_port")
    argv = value["argv"]
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
        and argv[6].isdigit()
        and str(int(argv[6])) == argv[6],
        f"E_ADB_SERVER_ARGV: {field}",
    )
    integer(value["pid"], f"{field}.pid", 1)
    integer(value["start_ticks"], f"{field}.start_ticks", 1)
    return value


def validate_remote_probe(
    value: Any,
    plan: dict[str, Any],
    contract: dict[str, Any],
    include_inputs: bool,
) -> dict[str, Any]:
    value = exact_keys(
        value,
        {
            "artifacts",
            "boot_id",
            "child_receipts",
            "clock",
            "gpu",
            "host",
            "inputs",
            "observed_ns",
            "schema",
            "server",
            "system_swap_used_bytes",
        },
        "remote_probe",
    )
    exact(value["schema"], REMOTE_PROBE_SCHEMA, "remote_probe.schema")
    exact(value["clock"], "RTX_MONOTONIC_RAW", "remote_probe.clock")
    exact(value["host"], contract["topology"]["cuda_host"], "remote_probe.host")
    uuid(value["boot_id"], "remote_probe.boot_id")
    integer(value["observed_ns"], "remote_probe.observed_ns", 1)
    integer(
        value["system_swap_used_bytes"],
        "remote_probe.system_swap_used_bytes",
    )
    gpu = exact_keys(
        value["gpu"],
        {"memory_total_bytes", "name", "pci_bus_id", "uuid"},
        "remote_probe.gpu",
    )
    exact(
        gpu["uuid"],
        contract["topology"]["cuda_gpu_uuid"],
        "remote_probe.gpu.uuid",
    )
    exact(
        gpu["name"],
        contract["devices"]["cuda"]["name"],
        "remote_probe.gpu.name",
    )
    exact(
        gpu["memory_total_bytes"],
        contract["devices"]["cuda"]["memory_total_bytes"],
        "remote_probe.gpu.memory",
    )
    text(gpu["pci_bus_id"], "remote_probe.gpu.pci_bus_id", 64)
    artifacts = exact_keys(
        value["artifacts"],
        REMOTE_ARTIFACT_ROLES,
        "remote_probe.artifacts",
    )
    for role in sorted(artifacts):
        artifact = validate_artifact(
            artifacts[role],
            f"remote_probe.artifacts.{role}",
        )
        exact(
            artifact["path"],
            plan["remote_artifact_paths"][role],
            f"remote_probe.artifacts.{role}.path",
        )
    expected_inputs = FAN_IN_INPUT_ROLES if include_inputs else set()
    inputs = exact_keys(value["inputs"], expected_inputs, "remote_probe.inputs")
    for role in sorted(inputs):
        artifact = validate_artifact(inputs[role], f"remote_probe.inputs.{role}")
        exact(
            artifact["path"],
            plan["remote_input_paths"][role],
            f"remote_probe.inputs.{role}.path",
        )
    _validate_adb_server(
        value["server"],
        artifacts["adb"]["path"],
        value["boot_id"],
        "remote_probe.server",
    )
    child = exact_keys(
        value["child_receipts"],
        {"adb_start", "gpu_query"},
        "remote_probe.child_receipts",
    )
    for name, receipt in child.items():
        receipt = exact_keys(
            receipt,
            {
                "argv",
                "clock",
                "completed_ns",
                "returncode",
                "started_ns",
                "stderr",
                "stdout",
            },
            f"remote_probe.child_receipts.{name}",
        )
        exact(
            receipt["clock"],
            "RTX_MONOTONIC_RAW",
            f"remote_probe.child_receipts.{name}.clock",
        )
        started = integer(
            receipt["started_ns"],
            f"remote_probe.child_receipts.{name}.started_ns",
            1,
        )
        integer(
            receipt["completed_ns"],
            f"remote_probe.child_receipts.{name}.completed_ns",
            started,
        )
        exact(receipt["returncode"], 0, f"remote_probe.child_receipts.{name}.rc")
        for stream in ("stdout", "stderr"):
            _validate_blob(
                receipt[stream],
                f"remote_probe.child_receipts.{name}.{stream}",
            )
    return value


def _validate_blob(value: Any, field: str) -> bytes:
    value = exact_keys(
        value,
        {"bytes", "content_base64", "sha256"},
        field,
    )
    size = integer(value["bytes"], f"{field}.bytes")
    digest(value["sha256"], f"{field}.sha256")
    try:
        raw = base64.b64decode(value["content_base64"], validate=True)
    except (TypeError, ValueError) as error:
        raise PreflightError(f"E_BASE64: {field}") from error
    exact(len(raw), size, f"{field}.bytes")
    exact(hashlib.sha256(raw).hexdigest(), value["sha256"], f"{field}.sha256")
    return raw
