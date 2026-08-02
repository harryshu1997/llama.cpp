#!/usr/bin/env python3
"""Resident adapter from warm-tier commands to one StageNet v3 route."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Protocol


HERE = Path(__file__).resolve().parent
from executor_bundle import BundleError, validate_runtime_environment
from runtime_binding import (
    ControllerAuthenticator,
    RuntimeBinding,
    RuntimeBindingError,
    await_runtime_binding,
    process_start_time_ticks,
)


try:
    validate_runtime_environment(Path(__file__).resolve())
except (BundleError, OSError) as error:
    print(f"executor bundle failed: {error}", file=sys.stderr)
    raise SystemExit(2)
S39 = HERE.parents[1] / "s39_phone_model_switch_trace"
S22 = HERE.parents[1] / "s22_slo_overlap_pipeline"
if os.environ.get("S40_EXECUTOR_BUNDLE") != "1":
    for directory in (S39, S22):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))

from mixed_phase_batcher import (
    PHASE_DECODE,
    PHASE_PREFILL,
    MixedPhaseBatcher,
    PhaseRow,
)
from stage_v3_client import (
    BatchResult,
    BatchRow,
    Hello,
    ProtocolError,
    STAGE_V3_CAP_TERMINAL,
    StageV3Client,
)


COMMAND_EXECUTE = 0
COMMAND_DRAIN = 1
COMMAND_UNLOAD = 2
COMMAND_LOAD = 3
COMMAND_REPLAY = 4
COMMAND_DISCARD = 5
COMMAND_CLEANUP = 6
MAX_COMMAND_BYTES = 4 * 1024 * 1024
MAX_TOKENS = 1 << 20


class GatewayError(RuntimeError):
    pass


class TerminalBatchClient(Protocol):
    def batch(self, rows: list[BatchRow]) -> tuple[BatchResult, ...]:
        ...

    def remove(
        self,
        seq_id: int,
        request_id: int,
        route_epoch: int,
    ) -> Any:
        ...

    def drain(self) -> Any:
        ...

    def stop(self) -> None:
        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True)
class RouteSpec:
    model_id: str
    model_sha256: str
    artifact_certificate_sha256: str
    readiness_lock_sha256: str
    readiness_phase_id: str
    relay_host: str
    relay_port: int
    file_type: int
    layer_start: int
    layer_end: int
    n_layer: int
    n_embd: int
    max_streams: int
    n_batch: int
    n_ubatch: int
    batch_knee: int
    gather_us: int
    queue_depth: int
    prefill_chunk: int
    phase: str
    phase_lock_sha256: str
    qualification: dict[str, Any]
    qualification_sha256: str
    slot: str
    load_argv: tuple[str, ...]
    unload_argv: tuple[str, ...]
    a6000_identity: str
    op15_boot_id: str
    op12_boot_id: str
    op15_shard_sha256: str
    op12_shard_sha256: str
    worker_sha256: str
    control_files: tuple[tuple[str, str, int, str], ...] = ()
    control_env: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class RouteConnection:
    client: TerminalBatchClient
    hello: Hello
    route_instance_id: str = ""


class RouteSupervisor(Protocol):
    def open(self, spec: RouteSpec) -> RouteConnection:
        ...

    def close(self, spec: RouteSpec, client: TerminalBatchClient) -> None:
        ...


class _RecordingBatchClient:
    def __init__(
        self,
        client: TerminalBatchClient,
        record: Callable[[list[BatchRow], tuple[BatchResult, ...], int, int], None],
    ):
        self.client = client
        self.record = record

    def batch(self, rows: list[BatchRow]) -> tuple[BatchResult, ...]:
        rows = list(rows)
        started_ns = time.monotonic_ns()
        results = tuple(self.client.batch(rows))
        completed_ns = time.monotonic_ns()
        self.record(rows, results, started_ns, completed_ns)
        return results


@dataclass
class Session:
    internal_request_id: int
    seq_id: int
    route_epoch: int
    ownership_epoch: int
    logical_history: list[int]
    prediction: int | None = None


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GatewayError(message)


def canonical_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise GatewayError(f"canonical JSON: {error}") from error
    return (encoded + "\n").encode("ascii")


def strict_json_loads(raw: bytes, field: str, encoding: str = "ascii") -> Any:
    def object_hook(pairs):
        result = {}
        for key, value in pairs:
            require(type(key) is str and key not in result, f"{field}: duplicate key")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise GatewayError(f"{field}: invalid numeric constant {value}")

    try:
        return json.loads(
            raw.decode(encoding),
            object_pairs_hook=object_hook,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GatewayError(f"{field}: {error}") from error


def exact_keys(value: Any, keys: set[str], field: str) -> dict[str, Any]:
    require(type(value) is dict, f"{field}: expected object")
    require(set(value) == keys, f"{field}: fields do not match schema")
    return value


def integer(value: Any, field: str, minimum: int = 0) -> int:
    require(type(value) is int and minimum <= value < (1 << 63), f"{field}: integer")
    return value


def string(value: Any, field: str) -> str:
    require(type(value) is str and bool(value), f"{field}: string")
    require(
        all(0x20 <= ord(character) <= 0x7E for character in value),
        f"{field}: ASCII",
    )
    return value


def tokens(value: Any, field: str) -> list[int]:
    require(type(value) is list and len(value) <= MAX_TOKENS, f"{field}: token list")
    result = []
    for index, token in enumerate(value):
        require(
            type(token) is int and 0 <= token < (1 << 31),
            f"{field}[{index}]: token",
        )
        result.append(token)
    return result


def parse_empty_request(value: Any) -> dict[str, Any]:
    value = exact_keys(
        value,
        {
            "committed_output_tokens",
            "model_id",
            "owner_id",
            "ownership_epoch",
            "position",
            "prompt_tokens",
            "publication_index",
            "request_id",
            "state",
        },
        "request",
    )
    require(
        value
        == {
            "committed_output_tokens": [],
            "model_id": "",
            "owner_id": "",
            "ownership_epoch": 0,
            "position": 0,
            "prompt_tokens": [],
            "publication_index": 0,
            "request_id": "",
            "state": 0,
        },
        "model lifecycle request is not empty",
    )
    return value


def parse_request(value: Any) -> dict[str, Any]:
    value = exact_keys(
        value,
        {
            "committed_output_tokens",
            "model_id",
            "owner_id",
            "ownership_epoch",
            "position",
            "prompt_tokens",
            "publication_index",
            "request_id",
            "state",
        },
        "request",
    )
    result = {
        "committed_output_tokens": tokens(
            value["committed_output_tokens"],
            "request.committed_output_tokens",
        ),
        "model_id": string(value["model_id"], "request.model_id"),
        "owner_id": string(value["owner_id"], "request.owner_id"),
        "ownership_epoch": integer(
            value["ownership_epoch"],
            "request.ownership_epoch",
            1,
        ),
        "position": integer(value["position"], "request.position"),
        "prompt_tokens": tokens(value["prompt_tokens"], "request.prompt_tokens"),
        "publication_index": integer(
            value["publication_index"],
            "request.publication_index",
        ),
        "request_id": string(value["request_id"], "request.request_id"),
        "state": integer(value["state"], "request.state"),
    }
    require(result["prompt_tokens"], "request.prompt_tokens: empty")
    require(
        result["position"]
        == len(result["prompt_tokens"]) + len(result["committed_output_tokens"]),
        "request.position: history mismatch",
    )
    return result


def parse_command(
    raw: bytes,
    executor_id: str,
    executor_instance_id: str,
) -> dict[str, Any]:
    require(0 < len(raw) <= MAX_COMMAND_BYTES, "command size")
    value = strict_json_loads(raw, "command JSON")
    require(canonical_bytes(value) == raw, "command is not canonical JSON")
    value = exact_keys(
        value,
        {
            "command_id",
            "controller_epoch",
            "executor_id",
            "executor_instance_id",
            "kind",
            "max_output_tokens",
            "model_id",
            "request",
            "request_id",
            "schema",
            "total_output_tokens",
        },
        "command",
    )
    require(
        value["schema"] == "llama-server-warm-tier-command-v3",
        "command schema",
    )
    kind = integer(value["kind"], "command.kind")
    request = (
        parse_request(value["request"])
        if kind in (COMMAND_EXECUTE, COMMAND_REPLAY, COMMAND_CLEANUP)
        else parse_empty_request(value["request"])
    )
    result = {
        "command_id": integer(value["command_id"], "command.command_id", 1),
        "controller_epoch": integer(
            value["controller_epoch"],
            "command.controller_epoch",
        ),
        "executor_id": string(value["executor_id"], "command.executor_id"),
        "executor_instance_id": string(
            value["executor_instance_id"],
            "command.executor_instance_id",
        ),
        "kind": kind,
        "max_output_tokens": integer(
            value["max_output_tokens"],
            "command.max_output_tokens",
        ),
        "model_id": string(value["model_id"], "command.model_id"),
        "request_id": value["request_id"],
        "request": request,
        "schema": value["schema"],
        "total_output_tokens": integer(
            value["total_output_tokens"],
            "command.total_output_tokens",
        ),
    }
    require(result["executor_id"] == executor_id, "command executor mismatch")
    require(
        result["executor_instance_id"] == executor_instance_id,
        "command executor instance mismatch",
    )
    require(result["kind"] <= COMMAND_CLEANUP, "command kind")
    if result["kind"] == COMMAND_EXECUTE:
        require(
            type(result["request_id"]) is str and bool(result["request_id"]),
            "execute request id",
        )
        require(
            result["request_id"] == result["request"]["request_id"],
            "execute request identity",
        )
        require(
            result["model_id"] == result["request"]["model_id"],
            "execute model identity",
        )
        require(result["max_output_tokens"] == 1, "execute token quantum")
        require(
            result["total_output_tokens"]
            > len(result["request"]["committed_output_tokens"]),
            "execute total output budget",
        )
    elif result["kind"] in (COMMAND_REPLAY, COMMAND_CLEANUP):
        require(
            type(result["request_id"]) is str and bool(result["request_id"]),
            "lifecycle request id",
        )
        require(
            result["request_id"] == result["request"]["request_id"],
            "lifecycle request identity",
        )
        require(result["max_output_tokens"] == 0, "lifecycle output budget")
        require(result["total_output_tokens"] == 0, "lifecycle total budget")
    else:
        require(result["request_id"] == "", "model lifecycle request id")
        require(result["max_output_tokens"] == 0, "model lifecycle output budget")
        require(result["total_output_tokens"] == 0, "model lifecycle total budget")
    return result


def make_result(
    command: dict[str, Any],
    *,
    success: bool,
    detail: str,
    publications: list[dict[str, Any]] | None = None,
    request_complete: bool = False,
    replay_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "command_id": command["command_id"],
        "controller_epoch": command["controller_epoch"],
        "detail": detail,
        "executor_id": command["executor_id"],
        "executor_instance_id": command["executor_instance_id"],
        "has_replay_snapshot": replay_snapshot is not None,
        "kind": command["kind"],
        "model_id": command["model_id"],
        "publications": publications or [],
        "replay_snapshot": replay_snapshot,
        "request_complete": request_complete,
        "request_id": command["request_id"],
        "schema": "llama-server-warm-tier-result-v2",
        "success": success,
    }


def sha256_text(value: Any, field: str) -> str:
    result = string(value, field)
    require(
        len(result) == 64
        and result == result.lower()
        and all(character in "0123456789abcdef" for character in result),
        f"{field}: SHA-256",
    )
    return result


def command_argv(value: Any, field: str) -> tuple[str, ...]:
    require(type(value) is list and 0 < len(value) <= 64, f"{field}: argv")
    return tuple(
        string(argument, f"{field}[{index}]")
        for index, argument in enumerate(value)
    )


CONTROL_FILE_ROLES = {
    "a6000_ssh_control",
    "executor_bundle_manifest",
    "python",
    "ssh_config",
}
CONTROL_ENV_KEYS = {
    "LANG",
    "LC_ALL",
    "PATH",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONNOUSERSITE",
    "PYTHONPATH",
    "S40_EXECUTOR_BUNDLE",
    "S40_EXECUTOR_BUNDLE_MANIFEST",
    "S40_EXECUTOR_BUNDLE_SHA256",
}


def local_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def validate_control_files(
    files: tuple[tuple[str, str, int, str], ...],
) -> dict[str, tuple[str, int, str]]:
    require(
        {role for role, _, _, _ in files} == CONTROL_FILE_ROLES
        and len(files) == len(CONTROL_FILE_ROLES),
        "route control file roles",
    )
    result = {}
    for role, path_text, expected_bytes, expected_sha256 in files:
        path = Path(path_text)
        require(
            path.is_absolute() and path.is_file() and not path.is_symlink(),
            f"route control file {role}",
        )
        require(
            type(expected_bytes) is int
            and expected_bytes > 0
            and path.stat().st_size == expected_bytes
            and local_file_sha256(path) == expected_sha256,
            f"route control file changed: {role}",
        )
        if role in ("a6000_ssh_control", "python"):
            require(
                path.stat().st_mode & 0o111 != 0,
                f"route control executable mode: {role}",
            )
        result[role] = (path_text, expected_bytes, expected_sha256)
    return result


def control_argv(
    spec: RouteSpec,
    action: str,
) -> tuple[str, ...]:
    require(action in ("load", "rollback", "unload"), "route control action")
    files = validate_control_files(spec.control_files)
    return (
        files["python"][0],
        "-B",
        "-s",
        "-P",
        files["a6000_ssh_control"][0],
        "--action",
        action,
        "--config",
        files["ssh_config"][0],
        "--model",
        spec.model_id,
    )


def derive_route_qualification(
    value: Any,
    *,
    model_id: str,
    phase: str,
    slot: str,
    artifact_certificate_sha256: str,
    readiness_lock_sha256: str,
    environment: dict[str, str],
) -> dict[str, Any]:
    from qualification_authority import validate_route_authority

    manifest = Path(environment["S40_EXECUTOR_BUNDLE_MANIFEST"])
    expected = environment["S40_EXECUTOR_BUNDLE_SHA256"]
    require(
        manifest.is_absolute()
        and manifest.name == "MANIFEST.json"
        and manifest.is_file(),
        "phone executor bundle manifest",
    )
    expected = sha256_text(expected, "phone executor bundle SHA-256")
    require(
        hashlib.sha256(manifest.read_bytes()).hexdigest() == expected,
        "phone executor bundle manifest changed",
    )
    return validate_route_authority(
        value,
        expected_model_id=model_id,
        expected_phase=phase,
        expected_slot=slot,
        expected_artifact_certificate_sha256=
            artifact_certificate_sha256,
        expected_readiness_lock_sha256=readiness_lock_sha256,
        executor_bundle=manifest.parent,
        executor_bundle_manifest_sha256=expected,
    )


def parse_route_config(path: Path) -> tuple[str, dict[str, RouteSpec], str]:
    raw = path.read_bytes()
    require(0 < len(raw) <= MAX_COMMAND_BYTES, "route config size")
    value = strict_json_loads(raw, "route config JSON")
    require(canonical_bytes(value) == raw, "route config is not canonical JSON")
    value = exact_keys(
        value,
        {"control", "executor_id", "routes", "schema"},
        "route_config",
    )
    require(value["schema"] == "s40-phone-route-config-v3", "route config schema")
    executor_id = string(value["executor_id"], "route_config.executor_id")
    control = exact_keys(
        value["control"],
        {"environment", "files"},
        "route_config.control",
    )
    environment = control["environment"]
    require(
        type(environment) is dict
        and set(environment) == CONTROL_ENV_KEYS
        and all(type(key) is str and type(item) is str and item
                for key, item in environment.items())
        and environment["LANG"] == "C"
        and environment["LC_ALL"] == "C"
        and environment["PATH"] == "/usr/bin:/bin"
        and environment["PYTHONDONTWRITEBYTECODE"] == "1"
        and environment["PYTHONNOUSERSITE"] == "1"
        and environment["S40_EXECUTOR_BUNDLE"] == "1",
        "route control environment",
    )
    control_rows = control["files"]
    require(
        type(control_rows) is list
        and len(control_rows) == len(CONTROL_FILE_ROLES),
        "route control files",
    )
    normalized_control_files = []
    seen_control_roles = set()
    for index, row in enumerate(control_rows):
        row = exact_keys(
            row,
            {"bytes", "path", "role", "sha256"},
            f"route_config.control.files[{index}]",
        )
        role = string(row["role"], "route control file role")
        require(
            role in CONTROL_FILE_ROLES and role not in seen_control_roles,
            "route control file role",
        )
        seen_control_roles.add(role)
        path_text = string(row["path"], f"route control {role} path")
        require(Path(path_text).is_absolute(), f"route control {role} path")
        normalized_control_files.append((
            role,
            path_text,
            integer(row["bytes"], f"route control {role} bytes", 1),
            sha256_text(row["sha256"], f"route control {role} SHA-256"),
        ))
    normalized_control_files.sort()
    control_files = tuple(normalized_control_files)
    locked_control = validate_control_files(control_files)
    require(
        environment["PYTHONPATH"]
        == str(Path(locked_control["a6000_ssh_control"][0]).parent)
        and environment["S40_EXECUTOR_BUNDLE_MANIFEST"]
        == locked_control["executor_bundle_manifest"][0]
        and environment["S40_EXECUTOR_BUNDLE_SHA256"]
        == locked_control["executor_bundle_manifest"][2],
        "route control bundle binding",
    )
    control_env = tuple(sorted(environment.items()))
    routes = value["routes"]
    require(type(routes) is list and 1 <= len(routes) <= 8, "route config routes")
    result: dict[str, RouteSpec] = {}
    keys = {
        "a6000_identity",
        "artifact_certificate_sha256",
        "batch_knee",
        "file_type",
        "gather_us",
        "layer_end",
        "layer_start",
        "max_streams",
        "model_id",
        "model_sha256",
        "n_batch",
        "n_embd",
        "n_layer",
        "n_ubatch",
        "op12_boot_id",
        "op12_shard_sha256",
        "op15_boot_id",
        "op15_shard_sha256",
        "prefill_chunk",
        "phase",
        "phase_lock_sha256",
        "qualification",
        "queue_depth",
        "readiness_lock_sha256",
        "readiness_phase_id",
        "relay_host",
        "relay_port",
        "slot",
        "worker_sha256",
    }
    for index, item in enumerate(routes):
        item = exact_keys(item, keys, f"route_config.routes[{index}]")
        model_id = string(item["model_id"], f"routes[{index}].model_id")
        require(model_id not in result, "duplicate route model")
        layer_start = integer(item["layer_start"], "route.layer_start")
        layer_end = integer(item["layer_end"], "route.layer_end", 1)
        n_layer = integer(item["n_layer"], "route.n_layer", 1)
        require(0 <= layer_start < layer_end <= n_layer, "route layer range")
        n_batch = integer(item["n_batch"], "route.n_batch", 1)
        n_ubatch = integer(item["n_ubatch"], "route.n_ubatch", 1)
        maximum_rows = min(n_batch, n_ubatch)
        batch_knee = integer(item["batch_knee"], "route.batch_knee", 1)
        prefill_chunk = integer(item["prefill_chunk"], "route.prefill_chunk", 1)
        require(
            batch_knee <= maximum_rows and prefill_chunk <= maximum_rows,
            "route batch geometry",
        )
        phase = string(item["phase"], "route.phase")
        require(phase in ("A_ONLY", "B_ONLY"), "route phase")
        slot = string(item["slot"], "route.slot")
        require(
            (phase == "A_ONLY" and slot == "A")
            or (phase == "B_ONLY" and slot == "B"),
            "route phase slot",
        )
        artifact_certificate_sha256 = sha256_text(
            item["artifact_certificate_sha256"],
            "route.artifact_certificate_sha256",
        )
        readiness_lock_sha256 = sha256_text(
            item["readiness_lock_sha256"],
            "route.readiness_lock_sha256",
        )
        qualification = derive_route_qualification(
            item["qualification"],
            model_id=model_id,
            phase=phase,
            slot=slot,
            artifact_certificate_sha256=artifact_certificate_sha256,
            readiness_lock_sha256=readiness_lock_sha256,
            environment=environment,
        )
        phase_lock_sha256 = sha256_text(
            item["phase_lock_sha256"],
            "route.phase_lock_sha256",
        )
        require(
            qualification["phase_id"] == item["readiness_phase_id"]
            and qualification["phase_lock_sha256"] == phase_lock_sha256,
            "phone qualification readiness roots",
        )
        result[model_id] = RouteSpec(
            model_id=model_id,
            model_sha256=sha256_text(item["model_sha256"], "route.model_sha256"),
            artifact_certificate_sha256=artifact_certificate_sha256,
            readiness_lock_sha256=readiness_lock_sha256,
            readiness_phase_id=string(
                item["readiness_phase_id"],
                "route.readiness_phase_id",
            ),
            relay_host=string(item["relay_host"], "route.relay_host"),
            relay_port=integer(item["relay_port"], "route.relay_port", 1),
            file_type=integer(item["file_type"], "route.file_type"),
            layer_start=layer_start,
            layer_end=layer_end,
            n_layer=n_layer,
            n_embd=integer(item["n_embd"], "route.n_embd", 1),
            max_streams=integer(item["max_streams"], "route.max_streams", 1),
            n_batch=n_batch,
            n_ubatch=n_ubatch,
            batch_knee=batch_knee,
            gather_us=integer(item["gather_us"], "route.gather_us"),
            queue_depth=integer(item["queue_depth"], "route.queue_depth", 1),
            prefill_chunk=prefill_chunk,
            phase=phase,
            phase_lock_sha256=phase_lock_sha256,
            qualification=qualification,
            qualification_sha256=hashlib.sha256(
                canonical_bytes(item["qualification"])
            ).hexdigest(),
            slot=slot,
            load_argv=(),
            unload_argv=(),
            a6000_identity=string(item["a6000_identity"], "route.a6000_identity"),
            op15_boot_id=string(item["op15_boot_id"], "route.op15_boot_id"),
            op12_boot_id=string(item["op12_boot_id"], "route.op12_boot_id"),
            op15_shard_sha256=sha256_text(
                item["op15_shard_sha256"],
                "route.op15_shard_sha256",
            ),
            op12_shard_sha256=sha256_text(
                item["op12_shard_sha256"],
                "route.op12_shard_sha256",
            ),
            worker_sha256=sha256_text(item["worker_sha256"], "route.worker_sha256"),
            control_files=control_files,
            control_env=control_env,
        )
        result[model_id] = replace(
            result[model_id],
            load_argv=control_argv(result[model_id], "load"),
            unload_argv=control_argv(result[model_id], "unload"),
        )
    return executor_id, result, hashlib.sha256(raw).hexdigest()


class SubprocessRouteSupervisor:
    def __init__(
        self,
        timeout_s: float,
        evidence_path: Path,
        *,
        executor_id: str | None = None,
        route_config_sha256: str | None = None,
        route_specs: dict[str, RouteSpec] | None = None,
    ):
        require(timeout_s > 0, "supervisor timeout")
        require(evidence_path.is_absolute(), "route evidence must be absolute")
        self.timeout_s = timeout_s
        self._evidence = evidence_path.open("x", encoding="ascii")
        self._lock = threading.Lock()
        self._instances: dict[int, str] = {}
        if executor_id is not None:
            require(
                route_config_sha256 is not None and route_specs is not None,
                "route config evidence is incomplete",
            )
            self._record({
                "executor_id": string(executor_id, "route executor ID"),
                "route_config_sha256": sha256_text(
                    route_config_sha256,
                    "route config SHA-256",
                ),
                "routes": [
                    {
                        "artifact_certificate_sha256":
                            spec.artifact_certificate_sha256,
                        "batch_knee": spec.batch_knee,
                        "model_id": spec.model_id,
                        "n_batch": spec.n_batch,
                        "n_ubatch": spec.n_ubatch,
                        "phase_lock_sha256": spec.phase_lock_sha256,
                        "qualification": spec.qualification,
                        "readiness_lock_sha256":
                            spec.readiness_lock_sha256,
                        "readiness_phase_id": spec.readiness_phase_id,
                    }
                    for spec in sorted(
                        route_specs.values(),
                        key=lambda item: item.model_id,
                    )
                ],
                "schema": "s40-phone-route-config-evidence-v1",
            })

    def _run(self, spec: RouteSpec, action: str) -> dict[str, Any]:
        argv = control_argv(spec, action)
        environment = dict(spec.control_env)
        require(
            set(environment) == CONTROL_ENV_KEYS,
            "route control environment changed",
        )
        completed = subprocess.run(
            argv,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout_s,
            env=environment,
        )
        require(
            control_argv(spec, action) == argv,
            "route control files changed during execution",
        )
        require(completed.returncode == 0, "route control command failed")
        require(not completed.stderr, "route control command wrote stderr")
        require(
            0 < len(completed.stdout) <= MAX_COMMAND_BYTES,
            "route control output size",
        )
        value = strict_json_loads(completed.stdout, "route control JSON")
        require(
            canonical_bytes(value) == completed.stdout,
            "route control output is not canonical JSON",
        )
        require(type(value) is dict, "route control output")
        return value

    @staticmethod
    def _validate_load(value: dict[str, Any], spec: RouteSpec) -> str:
        value = exact_keys(
            value,
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
            "load_result",
        )
        require(value["schema"] == "s40-phone-route-load-v3", "load schema")
        require(value["success"] is True, "load failed")
        expected = {
            "a6000_identity": spec.a6000_identity,
            "artifact_certificate_sha256":
                spec.artifact_certificate_sha256,
            "model_id": spec.model_id,
            "model_sha256": spec.model_sha256,
            "op12_boot_id": spec.op12_boot_id,
            "op12_shard_sha256": spec.op12_shard_sha256,
            "op15_boot_id": spec.op15_boot_id,
            "op15_shard_sha256": spec.op15_shard_sha256,
            "qualification_sha256": spec.qualification_sha256,
            "readiness_lock_sha256": spec.readiness_lock_sha256,
            "readiness_phase_id": spec.readiness_phase_id,
            "worker_sha256": spec.worker_sha256,
        }
        for key, expected_value in expected.items():
            require(value[key] == expected_value, f"load {key} mismatch")
        route_instance_id = string(
            value["route_instance_id"],
            "load.route_instance_id",
        )
        observation = exact_keys(
            value["route_observation"],
            {
                "direct_peer",
                "model_id",
                "phones",
                "route_instance_id",
                "schema",
            },
            "load.route_observation",
        )
        require(
            observation["schema"] == "s40-phone-route-observation-v1"
            and observation["model_id"] == spec.model_id
            and observation["route_instance_id"] == route_instance_id
            and type(observation["phones"]) is dict
            and set(observation["phones"]) == {"op12", "op15"}
            and type(observation["direct_peer"]) is dict,
            "load synchronous route observation",
        )
        return route_instance_id

    def _record(self, value: dict[str, Any]) -> None:
        raw = canonical_bytes(value).decode("ascii")
        with self._lock:
            self._evidence.write(raw)
            self._evidence.flush()
            os.fsync(self._evidence.fileno())

    def _unload(self, spec: RouteSpec, route_instance_id: str) -> dict[str, Any]:
        result = self._run(spec, "unload")
        result = exact_keys(
            result,
            {
                "model_id",
                "placements",
                "route_instance_id",
                "schema",
                "success",
            },
            "unload_result",
        )
        require(result["schema"] == "s40-phone-route-unload-v2", "unload schema")
        require(result["success"] is True, "unload failed")
        require(result["model_id"] == spec.model_id, "unload model mismatch")
        require(
            result["route_instance_id"] == route_instance_id,
            "unload route instance mismatch",
        )
        placements = result["placements"]
        require(
            type(placements) is dict
            and set(placements) == {"op12", "op15"},
            "unload placement roles",
        )
        for phone, evidence in placements.items():
            evidence = exact_keys(
                evidence,
                {
                    "certificate_lines",
                    "certificate_lines_sha256",
                    "placement",
                    "session",
                },
                f"unload {phone} placement",
            )
            sha256_text(
                evidence["certificate_lines_sha256"],
                f"unload {phone} placement lines SHA-256",
            )
            require(
                type(evidence["certificate_lines"]) is list
                and len(evidence["certificate_lines"]) == 2
                and evidence["session"].get("placement_status")
                == "SCHEDULED_PLACEMENT_OK"
                and evidence["placement"].get("status")
                == "SCHEDULED_PLACEMENT_OK",
                f"unload {phone} placement status",
            )
        return result

    def _rollback(self, spec: RouteSpec, route_instance_id: str) -> dict[str, Any]:
        result = self._run(spec, "rollback")
        result = exact_keys(
            result,
            {
                "model_id",
                "route_instance_id",
                "schema",
                "success",
            },
            "rollback_result",
        )
        require(
            result["schema"] == "s40-phone-route-rollback-v1"
            and result["success"] is True
            and result["model_id"] == spec.model_id
            and result["route_instance_id"] == route_instance_id,
            "rollback result mismatch",
        )
        return result

    def open(self, spec: RouteSpec) -> RouteConnection:
        load = self._run(spec, "load")
        route_instance_id = self._validate_load(load, spec)
        self._record(load)
        deadline = time.monotonic() + self.timeout_s
        client: StageV3Client | None = None
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            try:
                client = StageV3Client.connect(
                    spec.relay_host,
                    spec.relay_port,
                    min(5.0, max(0.1, deadline - time.monotonic())),
                )
                hello = client.hello()
                self._instances[id(client)] = route_instance_id
                return RouteConnection(client, hello, route_instance_id)
            except BaseException as error:
                last_error = error
                if client is not None:
                    client.close()
                    client = None
                time.sleep(0.1)
        try:
            rollback = self._rollback(spec, route_instance_id)
            self._record(rollback)
        except BaseException as rollback_error:
            raise GatewayError(
                "route relay did not become ready and rollback failed: "
                f"{last_error}; {rollback_error}"
            ) from rollback_error
        raise GatewayError(
            f"route relay did not become ready; route was unloaded: {last_error}"
        )

    def close(self, spec: RouteSpec, client: TerminalBatchClient) -> None:
        route_instance_id = self._instances.pop(id(client), "")
        stop_error: BaseException | None = None
        try:
            client.stop()
        except BaseException as error:
            stop_error = error
        finally:
            client.close()
        if stop_error is not None:
            try:
                rollback = self._rollback(spec, route_instance_id)
                self._record(rollback)
            except BaseException as rollback_error:
                raise GatewayError(
                    "route stop failed and rollback failed: "
                    f"{stop_error}; {rollback_error}"
                ) from rollback_error
            raise GatewayError(
                f"route stop failed; route was rolled back: {stop_error}"
            ) from stop_error
        result = self._unload(spec, route_instance_id)
        self._record(result)


class PhoneRouteExecutor:
    def __init__(
        self,
        executor_id: str,
        route_specs: dict[str, RouteSpec],
        supervisor: RouteSupervisor,
        initial_model_id: str | None,
        *,
        timeout_s: float,
        wire_evidence_path: Path | None = None,
    ):
        require(executor_id and route_specs, "executor and routes are required")
        require(timeout_s > 0, "route timeout")
        require(set(route_specs) == {spec.model_id for spec in route_specs.values()},
                "route spec identity")
        self.executor_id = executor_id
        self.route_specs = dict(route_specs)
        self.supervisor = supervisor
        self.timeout_s = timeout_s
        self._condition = threading.Condition()
        self._free_sequences: list[int] = []
        self._sessions: dict[str, Session] = {}
        self._removing_sessions: set[str] = set()
        self._busy: set[str] = set()
        self._next_internal_request_id = 1
        self._route_epoch = 0
        self._draining = False
        self._closed = False
        self._active_spec: RouteSpec | None = None
        self._route_instance_id = ""
        self.client: TerminalBatchClient | None = None
        self.batcher: MixedPhaseBatcher | None = None
        self.prefill_chunk = 0
        self._execute_evidence: dict[int, dict[str, Any]] = {}
        self._wire_lock = threading.Lock()
        self._wire_batch_index = 0
        self._wire_evidence = (
            None
            if wire_evidence_path is None
            else wire_evidence_path.open("x", encoding="ascii")
        )
        if initial_model_id is not None:
            self._activate(initial_model_id)

    @property
    def model_id(self) -> str:
        return "" if self._active_spec is None else self._active_spec.model_id

    @property
    def route_epoch(self) -> int:
        return self._route_epoch

    def _activate(self, model_id: str) -> None:
        with self._condition:
            require(not self._closed, "route executor is closed")
            require(self._active_spec is None, "route already loaded")
            spec = self.route_specs.get(model_id)
            require(spec is not None, "unknown route model")
        connection = self.supervisor.open(spec)
        hello = connection.hello
        maximum_rows = min(hello.n_batch, hello.n_ubatch)
        try:
            require(
                (
                    hello.layer_start,
                    hello.layer_end,
                    hello.n_layer,
                    hello.n_embd,
                    hello.max_streams,
                    hello.n_batch,
                    hello.n_ubatch,
                    hello.file_type,
                    hello.model_sha256,
                )
                == (
                    spec.layer_start,
                    spec.layer_end,
                    spec.n_layer,
                    spec.n_embd,
                    spec.max_streams,
                    spec.n_batch,
                    spec.n_ubatch,
                    spec.file_type,
                    spec.model_sha256,
                ),
                "route hello does not match frozen spec",
            )
            require(
                hello.layer_start == 0
                and hello.layer_end == hello.n_layer
                and bool(hello.capabilities & STAGE_V3_CAP_TERMINAL),
                "direct relay is not terminal",
            )
            require(
                0 < spec.prefill_chunk <= maximum_rows
                and 0 < spec.batch_knee <= maximum_rows,
                "route batching bounds",
            )
            route_epoch = self._route_epoch + 1
            recording_client = _RecordingBatchClient(
                connection.client,
                lambda rows, results, started, completed: self._record_wire_batch(
                    spec,
                    connection.route_instance_id,
                    route_epoch,
                    rows,
                    results,
                    started,
                    completed,
                ),
            )
            batcher = MixedPhaseBatcher(
                f"{self.executor_id}-{model_id}-{self._route_epoch + 1}",
                recording_client,
                maximum_rows,
                spec.batch_knee,
                spec.gather_us,
                spec.queue_depth,
            )
        except BaseException:
            self.supervisor.close(spec, connection.client)
            raise
        with self._condition:
            self._route_epoch += 1
            self._active_spec = spec
            self._route_instance_id = connection.route_instance_id
            self.client = connection.client
            self.batcher = batcher
            self.prefill_chunk = spec.prefill_chunk
            self._free_sequences = list(range(hello.max_streams))
            self._draining = False

    def _record_wire_batch(
        self,
        spec: RouteSpec,
        route_instance_id: str,
        route_epoch: int,
        rows: list[BatchRow],
        results: tuple[BatchResult, ...],
        started_ns: int,
        completed_ns: int,
    ) -> None:
        if self._wire_evidence is None:
            return
        with self._wire_lock:
            self._wire_batch_index += 1
            record = {
                "batch_index": self._wire_batch_index,
                "batch_size": len(rows),
                "completed_ns": completed_ns,
                "executor_id": self.executor_id,
                "input_rows": [
                    {
                        "position": row.position,
                        "request_id": row.request_id,
                        "route_epoch": row.route_epoch,
                        "seq_id": row.seq_id,
                        "token": row.token,
                    }
                    for row in rows
                ],
                "model_id": spec.model_id,
                "output_rows": [
                    {
                        "position": result.position,
                        "request_id": result.request_id,
                        "route_epoch": result.route_epoch,
                        "seq_id": result.seq_id,
                        "token": result.token,
                    }
                    for result in results
                ],
                "route_epoch": route_epoch,
                "route_instance_id": route_instance_id,
                "schema": "s40-phone-wire-batch-v1",
                "started_ns": started_ns,
            }
            self._wire_evidence.write(
                canonical_bytes(record).decode("ascii")
            )
            self._wire_evidence.flush()
            os.fsync(self._wire_evidence.fileno())

    def take_execute_evidence(self, command_id: int) -> dict[str, Any] | None:
        with self._condition:
            return self._execute_evidence.pop(command_id, None)

    def _require_active(self, model_id: str) -> tuple[RouteSpec, TerminalBatchClient, MixedPhaseBatcher]:
        with self._condition:
            require(self._active_spec is not None, "route is not loaded")
            require(self._active_spec.model_id == model_id, "model is not active")
            require(self.client is not None and self.batcher is not None, "route incomplete")
            return self._active_spec, self.client, self.batcher

    def _allocate_session(self, request: dict[str, Any]) -> Session:
        with self._condition:
            require(not self._draining and not self._closed, "route is not accepting")
            require(request["request_id"] not in self._sessions, "session exists")
            require(self._free_sequences, "route has no sequence credit")
            session = Session(
                internal_request_id=self._next_internal_request_id,
                seq_id=self._free_sequences.pop(0),
                route_epoch=self._route_epoch,
                ownership_epoch=request["ownership_epoch"],
                logical_history=[],
            )
            self._next_internal_request_id += 1
            self._sessions[request["request_id"]] = session
            return session

    def _remove_session(self, request_id: str) -> None:
        with self._condition:
            session = self._sessions.get(request_id)
            require(
                request_id not in self._removing_sessions,
                "session removal already in progress",
            )
            if session is not None:
                self._removing_sessions.add(request_id)
        if session is None:
            return
        try:
            require(self.client is not None, "route client missing")
            self.client.remove(
                session.seq_id,
                session.internal_request_id,
                session.route_epoch,
            )
        except BaseException:
            with self._condition:
                self._removing_sessions.discard(request_id)
                self._condition.notify_all()
            raise
        with self._condition:
            require(
                self._sessions.get(request_id) is session,
                "session changed during removal",
            )
            self._sessions.pop(request_id)
            self._removing_sessions.discard(request_id)
            self._free_sequences.append(session.seq_id)
            self._free_sequences.sort()
            self._condition.notify_all()

    def _submit_rows(
        self,
        session: Session,
        values: list[int],
        start_position: int,
        phase: str,
    ) -> list[BatchResult]:
        require(self.batcher is not None, "route batcher missing")
        results: list[BatchResult] = []
        for start in range(0, len(values), self.prefill_chunk):
            chunk = values[start:start + self.prefill_chunk]
            entries = tuple(
                PhaseRow(
                    BatchRow(
                        session.internal_request_id,
                        session.route_epoch,
                        session.seq_id,
                        start_position + start + index,
                        token,
                    ),
                    phase,
                    0,
                )
                for index, token in enumerate(chunk)
            )
            futures = self.batcher.submit_many(entries, self.timeout_s)
            chunk_results = [
                future.result(timeout=self.timeout_s) for future in futures
            ]
            for entry, result in zip(entries, chunk_results):
                row = entry.row
                require(
                    (
                        result.request_id,
                        result.route_epoch,
                        result.seq_id,
                        result.position,
                    )
                    == (
                        row.request_id,
                        row.route_epoch,
                        row.seq_id,
                        row.position,
                    ),
                    "stale or mismatched batch result",
                )
            results.extend(chunk_results)
        return results

    def _restore(self, request: dict[str, Any]) -> Session:
        self._remove_session(request["request_id"])
        session = self._allocate_session(request)
        history = request["prompt_tokens"] + request["committed_output_tokens"]
        try:
            results = self._submit_rows(
                session,
                history,
                0,
                PHASE_PREFILL,
            )
            require(results and results[-1].token is not None, "restore prediction")
            session.logical_history = list(history)
            session.prediction = results[-1].token
            return session
        except BaseException:
            self._remove_session(request["request_id"])
            raise

    def _session_for_execute(self, request: dict[str, Any]) -> Session:
        with self._condition:
            session = self._sessions.get(request["request_id"])
        if session is None:
            return self._restore(request)
        require(
            session.ownership_epoch == request["ownership_epoch"],
            "stale ownership epoch",
        )
        require(
            session.logical_history
            == request["prompt_tokens"] + request["committed_output_tokens"],
            "request frontier mismatch",
        )
        if session.prediction is None:
            last_position = len(session.logical_history) - 1
            result = self._submit_rows(
                session,
                [session.logical_history[-1]],
                last_position,
                PHASE_DECODE,
            )
            require(result[-1].token is not None, "decode prediction")
            session.prediction = result[-1].token
        return session

    def _execute(self, command: dict[str, Any]) -> dict[str, Any]:
        request = command["request"]
        with self._condition:
            require(
                not self._draining and not self._closed,
                "route is not accepting",
            )
            require(request["request_id"] not in self._busy, "request already executing")
            reused = request["request_id"] in self._sessions
            self._busy.add(request["request_id"])
        try:
            session = self._session_for_execute(request)
            publications = []
            for index in range(command["max_output_tokens"]):
                require(session.prediction is not None, "prediction missing")
                token = session.prediction
                session.prediction = None
                publications.append({
                    "owner_id": command["executor_id"],
                    "ownership_epoch": request["ownership_epoch"],
                    "position": request["position"] + index,
                    "publication_index": request["publication_index"] + index,
                    "token": token,
                })
                session.logical_history.append(token)
                if index + 1 < command["max_output_tokens"]:
                    result = self._submit_rows(
                        session,
                        [token],
                        request["position"] + index,
                        PHASE_DECODE,
                    )
                    require(result[-1].token is not None, "decode prediction")
                    session.prediction = result[-1].token
            result = make_result(
                command,
                success=True,
                detail="phone route executed",
                publications=publications,
                request_complete=(
                    len(request["committed_output_tokens"])
                    + len(publications)
                    == command["total_output_tokens"]
                ),
            )
            with self._condition:
                self._execute_evidence[command["command_id"]] = {
                    "execute_quantum_tokens": 1,
                    "full_history_per_token_reprefill": False,
                    "initial_history_replay": not reused,
                    "internal_request_id": session.internal_request_id,
                    "publication_count": len(publications),
                    "resident_session_reused": reused,
                    "route_epoch": session.route_epoch,
                    "route_instance_id": self._route_instance_id,
                    "sampler": {
                        "temperature": 0.0,
                        "type": "greedy_argmax",
                    },
                    "seq_id": session.seq_id,
                }
            return result
        finally:
            with self._condition:
                self._busy.discard(request["request_id"])
                self._condition.notify_all()

    def handle(self, command: dict[str, Any]) -> dict[str, Any]:
        require(command["executor_id"] == self.executor_id, "executor mismatch")
        kind = command["kind"]
        if kind == COMMAND_EXECUTE:
            self._require_active(command["model_id"])
            return self._execute(command)
        if kind == COMMAND_REPLAY:
            self._require_active(command["model_id"])
            with self._condition:
                require(
                    not self._draining and not self._closed,
                    "route is not accepting",
                )
            self._restore(command["request"])
            return make_result(
                command,
                success=True,
                detail="phone route replayed",
                replay_snapshot=command["request"],
            )
        if kind in (COMMAND_DISCARD, COMMAND_CLEANUP):
            self._require_active(command["model_id"])
            if kind == COMMAND_DISCARD:
                with self._condition:
                    request_ids = list(self._sessions)
                for request_id in request_ids:
                    self._remove_session(request_id)
            else:
                self._remove_session(command["request_id"])
            return make_result(
                command,
                success=True,
                detail=(
                    "phone route state discarded"
                    if kind == COMMAND_DISCARD
                    else "phone sequence removed"
                ),
            )
        if kind == COMMAND_DRAIN:
            _, client, _ = self._require_active(command["model_id"])
            with self._condition:
                require(not self._busy, "route still has executing requests")
                self._draining = True
            status = client.drain()
            require(status.active_sequences == len(self._sessions), "drain status")
            return make_result(command, success=True, detail="phone route drained")
        if kind == COMMAND_UNLOAD:
            spec, client, batcher = self._require_active(command["model_id"])
            with self._condition:
                require(self._draining and not self._sessions, "route not clean")
            batcher.stop(self.timeout_s)
            self.supervisor.close(spec, client)
            with self._condition:
                self._active_spec = None
                self._route_instance_id = ""
                self.client = None
                self.batcher = None
                self.prefill_chunk = 0
                self._free_sequences.clear()
                self._draining = False
            return make_result(command, success=True, detail="phone route unloaded")
        if kind == COMMAND_LOAD:
            self._activate(command["model_id"])
            return make_result(command, success=True, detail="phone route loaded")
        raise GatewayError("unsupported command")

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._draining = True
            request_ids = list(self._sessions)
        errors = []
        for request_id in request_ids:
            try:
                self._remove_session(request_id)
            except BaseException as error:
                errors.append(f"remove {request_id}: {error}")
        with self._condition:
            spec = self._active_spec
            client = self.client
            batcher = self.batcher
        if spec is not None and client is not None and batcher is not None:
            try:
                batcher.stop(self.timeout_s)
            except BaseException as error:
                errors.append(f"batcher stop: {error}")
            try:
                self.supervisor.close(spec, client)
            except BaseException as error:
                errors.append(f"route close: {error}")
        with self._condition:
            self._active_spec = None
            self._route_instance_id = ""
            self.client = None
            self.batcher = None
            self.prefill_chunk = 0
            self._sessions.clear()
            self._removing_sessions.clear()
            self._busy.clear()
            self._free_sequences.clear()
            self._condition.notify_all()
        if self._wire_evidence is not None:
            try:
                self._wire_evidence.close()
            except BaseException as error:
                errors.append(f"wire evidence close: {error}")
            self._wire_evidence = None
        require(not errors, "phone cleanup failed: " + "; ".join(errors))


class GatewayServer:
    def __init__(
        self,
        socket_path: Path,
        executor: PhoneRouteExecutor,
        evidence_path: Path,
        *,
        controller_authenticator: ControllerAuthenticator,
        route_config_sha256: str,
        run_id: str,
        runtime_binding: RuntimeBinding,
    ):
        require(socket_path.is_absolute(), "gateway socket must be absolute")
        require(evidence_path.is_absolute(), "gateway evidence must be absolute")
        self.socket_path = socket_path
        self.executor = executor
        self.run_id = string(run_id, "phone gateway run ID")
        require(
            runtime_binding.executor_id == executor.executor_id
            and runtime_binding.gateway_pid == os.getpid()
            and runtime_binding.gateway_start_time_ticks
            == process_start_time_ticks(),
            "phone gateway runtime executor identity",
        )
        self.runtime_binding = runtime_binding
        self.controller_authenticator = controller_authenticator
        self.executor_instance_id = runtime_binding.executor_instance_id
        self.runtime_config_sha256 = runtime_binding.runtime_config_sha256
        self._evidence = evidence_path.open("x", encoding="ascii")
        self._evidence_lock = threading.Lock()
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._stopping = threading.Event()
        self._threads: set[threading.Thread] = set()
        self._threads_lock = threading.Lock()
        self._fatal_error: BaseException | None = None
        startup = {
            "configured_models": sorted(executor.route_specs),
            "executor_id": executor.executor_id,
            "executor_instance_id": self.executor_instance_id,
            "gateway_pid": runtime_binding.gateway_pid,
            "gateway_start_time_ticks":
                runtime_binding.gateway_start_time_ticks,
            "initial_active_models": [],
            "route_config_sha256": sha256_text(
                route_config_sha256,
                "phone route config SHA-256",
            ),
            "run_id": self.run_id,
            "runtime_config_device": runtime_binding.runtime_config_device,
            "runtime_config_inode": runtime_binding.runtime_config_inode,
            "runtime_config_path": runtime_binding.runtime_config_path,
            "runtime_config_sha256": self.runtime_config_sha256,
            "schema": "s40-phone-startup-evidence-v2",
        }
        self._evidence.write(canonical_bytes(startup).decode("ascii"))
        self._evidence.flush()
        os.fsync(self._evidence.fileno())

    def _record(
        self,
        command: dict[str, Any],
        result: dict[str, Any],
        started_ns: int,
        completed_ns: int,
    ) -> None:
        execute = self.executor.take_execute_evidence(command["command_id"])
        row = {
            "command": command,
            "command_id": command["command_id"],
            "completed_ns": completed_ns,
            "controller_epoch": command["controller_epoch"],
            "durability": "fsync_each_record",
            "execute_quantum_tokens": (
                None if execute is None else execute["execute_quantum_tokens"]
            ),
            "executor_id": command["executor_id"],
            "executor_instance_id": command["executor_instance_id"],
            "full_history_per_token_reprefill": (
                None
                if execute is None
                else execute["full_history_per_token_reprefill"]
            ),
            "initial_history_replay": (
                None if execute is None else execute["initial_history_replay"]
            ),
            "internal_request_id": (
                None if execute is None else execute["internal_request_id"]
            ),
            "kind": command["kind"],
            "model_id": command["model_id"],
            "publication_count": (
                0 if execute is None else execute["publication_count"]
            ),
            "request_id": command["request_id"] or None,
            "resident_session_reused": (
                None if execute is None else execute["resident_session_reused"]
            ),
            "result": result,
            "role": "PHONE",
            "run_id": self.run_id,
            "runtime_config_sha256": self.runtime_config_sha256,
            "route_epoch": (
                None if execute is None else execute["route_epoch"]
            ),
            "route_instance_id": (
                None if execute is None else execute["route_instance_id"]
            ),
            "sampler": None if execute is None else execute["sampler"],
            "schema": "s40-phone-command-evidence-v4",
            "seq_id": None if execute is None else execute["seq_id"],
            "started_ns": started_ns,
            "success": result["success"],
        }
        raw = canonical_bytes(row).decode("ascii")
        with self._evidence_lock:
            self._evidence.write(raw)
            self._evidence.flush()
            os.fsync(self._evidence.fileno())

    @staticmethod
    def _read_one(connection: socket.socket) -> bytes:
        data = bytearray()
        while len(data) <= MAX_COMMAND_BYTES:
            chunk = connection.recv(min(65536, MAX_COMMAND_BYTES + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
            if data.endswith(b"\n"):
                break
        require(data.endswith(b"\n"), "gateway command framing")
        require(len(data) <= MAX_COMMAND_BYTES, "gateway command too large")
        require(connection.recv(1) == b"", "gateway command has trailing bytes")
        return bytes(data)

    def _serve_one(self, connection: socket.socket) -> None:
        try:
            with connection:
                connection.settimeout(self.executor.timeout_s)
                self.controller_authenticator.authenticate(connection)
                raw = self._read_one(connection)
                command = parse_command(
                    raw,
                    self.executor.executor_id,
                    self.executor_instance_id,
                )
                started_ns = time.monotonic_ns()
                try:
                    result = self.executor.handle(command)
                except BaseException as error:
                    result = make_result(
                        command,
                        success=False,
                        detail=f"{type(error).__name__}: {error}",
                    )
                completed_ns = time.monotonic_ns()
                self._record(command, result, started_ns, completed_ns)
                connection.sendall(canonical_bytes(result))
        except BaseException as error:
            with self._threads_lock:
                if self._fatal_error is None:
                    self._fatal_error = error
            self._stopping.set()
        finally:
            with self._threads_lock:
                self._threads.discard(threading.current_thread())

    def request_stop(self) -> None:
        self._stopping.set()

    def serve_forever(self) -> None:
        try:
            self._server.bind(str(self.socket_path))
            os.chmod(self.socket_path, 0o600)
            self._server.listen(64)
            self._server.settimeout(0.2)
            while not self._stopping.is_set():
                try:
                    connection, _ = self._server.accept()
                except socket.timeout:
                    continue
                thread = threading.Thread(
                    target=self._serve_one,
                    args=(connection,),
                    daemon=False,
                )
                with self._threads_lock:
                    self._threads.add(thread)
                thread.start()
        finally:
            self._stopping.set()
            self._server.close()
            while True:
                with self._threads_lock:
                    threads = list(self._threads)
                if not threads:
                    break
                for thread in threads:
                    thread.join(self.executor.timeout_s)
                    require(
                        not thread.is_alive(),
                        "phone gateway request did not drain",
                    )
            cleanup_error = None
            try:
                self.executor.close()
            except BaseException as error:
                cleanup_error = error
            try:
                self._evidence.close()
            finally:
                self.socket_path.unlink(missing_ok=True)
            if cleanup_error is not None:
                raise cleanup_error
            with self._threads_lock:
                fatal_error = self._fatal_error
            if fatal_error is not None:
                raise fatal_error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--route-evidence", type=Path, required=True)
    parser.add_argument("--wire-evidence", type=Path, required=True)
    parser.add_argument("--route-config", type=Path, required=True)
    parser.add_argument("--initial-model", action="append", default=[])
    parser.add_argument("--executor-instance-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--controller-identity", type=Path, required=True)
    parser.add_argument(
        "--controller-binding-evidence",
        type=Path,
        required=True,
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    require(
        args.socket.is_absolute()
        and args.evidence.is_absolute()
        and args.route_evidence.is_absolute()
        and args.wire_evidence.is_absolute()
        and args.route_config.is_absolute()
        and args.runtime_config.is_absolute()
        and args.controller_identity.is_absolute()
        and args.controller_binding_evidence.is_absolute(),
        "gateway paths must be absolute",
    )
    executor_id, route_specs, route_config_sha256 = parse_route_config(
        args.route_config
    )
    require(
        not args.initial_model,
        "phone gateway must bootstrap through controller LOAD",
    )
    runtime_binding = await_runtime_binding(
        args.runtime_config,
        executor_id=executor_id,
        executor_instance_id=args.executor_instance_id,
        run_id=args.run_id,
        socket_path=args.socket,
        timeout_s=args.timeout,
    )
    controller_authenticator = ControllerAuthenticator(
        args.controller_identity,
        args.controller_binding_evidence,
        run_id=args.run_id,
        runtime_binding=runtime_binding,
        timeout_s=args.timeout,
    )
    supervisor = SubprocessRouteSupervisor(
        args.timeout,
        args.route_evidence,
        executor_id=executor_id,
        route_config_sha256=route_config_sha256,
        route_specs=route_specs,
    )
    executor = PhoneRouteExecutor(
        executor_id,
        route_specs,
        supervisor,
        None,
        timeout_s=args.timeout,
        wire_evidence_path=args.wire_evidence,
    )
    server = GatewayServer(
        args.socket,
        executor,
        args.evidence,
        controller_authenticator=controller_authenticator,
        route_config_sha256=route_config_sha256,
        run_id=args.run_id,
        runtime_binding=runtime_binding,
    )
    previous_term = signal.signal(
        signal.SIGTERM,
        lambda _signum, _frame: server.request_stop(),
    )
    previous_int = signal.signal(
        signal.SIGINT,
        lambda _signum, _frame: server.request_stop(),
    )
    try:
        server.serve_forever()
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        GatewayError,
        OSError,
        ProtocolError,
        RuntimeBindingError,
        RuntimeError,
        TimeoutError,
    ) as error:
        print(f"phone gateway failed: {error}", file=sys.stderr)
        raise SystemExit(2)
