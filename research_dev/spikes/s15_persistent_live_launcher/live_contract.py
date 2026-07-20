#!/usr/bin/env python3
"""Frozen JSONL v1 contracts for the S15 persistent live-launcher bridge."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path

from physical_executor import PhysicalExecutorError


EXECUTE_SCHEMA = "s15-persistent-execute-v1"
RESULT_SCHEMA = "s15-persistent-result-v1"
CHILD_COMMAND_SCHEMA = "layersplit-persistent-command-v1"
CHILD_RESULT_SCHEMA = "layersplit-persistent-result-v1"
HOST_PLACEMENT_SCHEMA = "layersplit-scheduled-placement-v2"
HOST_PLACEMENT_PREFIX = b"PLACEMENTCERT "
PROTOCOL_VERSION = 1
TERMINALS = ("DETACHED", "STOPPED", "ERROR")
SESSION_ENDS = ("DETACH", "STOP")
MAX_LINE_BYTES = 4 * 1024 * 1024
MAX_PROMPT_BYTES = 16 * 1024
MAX_BATCH = 4096
MAX_N_GEN = 4096


class LiveContractError(PhysicalExecutorError):
    pass


def integer(name: str, value: object, minimum: int = 0, maximum: int | None = None) -> int:
    if type(value) is not int or value < minimum \
            or (maximum is not None and value > maximum):
        suffix = "" if maximum is None else f" and <= {maximum}"
        raise LiveContractError(f"{name} must be an integer >= {minimum}{suffix}")
    return value


def text(name: str, value: object, maximum_bytes: int | None = None) -> str:
    if type(value) is not str or not value:
        raise LiveContractError(f"{name} must be a non-empty string")
    if maximum_bytes is not None and len(value.encode("utf-8")) > maximum_bytes:
        raise LiveContractError(f"{name} exceeds its byte bound")
    return value


def sha256_text(name: str, value: object) -> str:
    result = text(name, value)
    if re.fullmatch(r"sha256:[0-9a-f]{64}", result) is None:
        raise LiveContractError(f"{name} must be a lowercase sha256 digest")
    return result


def digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def canonical(value: object) -> bytes:
    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise LiveContractError(f"value is not canonical JSON: {exc}") from exc


def strict_line(payload: bytes, label: str) -> dict:
    if type(payload) is not bytes or not payload or len(payload) > MAX_LINE_BYTES \
            or not payload.endswith(b"\n") or payload.count(b"\n") != 1:
        raise LiveContractError(f"{label} is not one bounded JSON line")

    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise LiveContractError(f"duplicate key {key!r} in {label}")
            result[key] = value
        return result

    def no_constants(value):
        raise LiveContractError(f"invalid constant {value!r} in {label}")

    try:
        value = json.loads(
            payload, object_pairs_hook=no_duplicates, parse_constant=no_constants,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LiveContractError(f"invalid {label}: {exc}") from exc
    if type(value) is not dict:
        raise LiveContractError(f"{label} must be an object")
    if canonical(value) != payload:
        raise LiveContractError(f"{label} is not canonical JSON")
    return value


def strict_wire_line(payload: bytes, label: str) -> dict:
    """Parse a minified JSON object while preserving the producer's key order."""
    if type(payload) is not bytes or not payload or len(payload) > MAX_LINE_BYTES \
            or not payload.endswith(b"\n") or payload.count(b"\n") != 1:
        raise LiveContractError(f"{label} is not one bounded JSON line")

    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise LiveContractError(f"duplicate key {key!r} in {label}")
            result[key] = value
        return result

    def no_constants(value):
        raise LiveContractError(f"invalid constant {value!r} in {label}")

    try:
        value = json.loads(
            payload, object_pairs_hook=no_duplicates, parse_constant=no_constants,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LiveContractError(f"invalid {label}: {exc}") from exc
    if type(value) is not dict:
        raise LiveContractError(f"{label} must be an object")
    encoded = (
        json.dumps(value, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("ascii")
    if encoded != payload:
        raise LiveContractError(f"{label} is not the exact minified JSON wire")
    return value


@dataclass(frozen=True)
class FrozenRoute:
    route_id: str
    profile_id: str
    evidence_sha256: str
    device_id: str
    device_boot_id: str
    worker_binary_sha256: str
    layer_range: tuple[int, int]
    host_tail_range: tuple[int, int]
    route_epoch: int
    residency_epoch: int
    device_boot_epoch: int
    registry_generation: int
    batch_size: int
    max_n_gen: int

    def validate(self) -> None:
        text("route_id", self.route_id)
        sha256_text("profile_id", self.profile_id)
        sha256_text("evidence_sha256", self.evidence_sha256)
        text("device_id", self.device_id)
        text("device_boot_id", self.device_boot_id)
        sha256_text("worker_binary_sha256", self.worker_binary_sha256)
        if type(self.layer_range) is not tuple or len(self.layer_range) != 2 \
                or type(self.layer_range[0]) is not int \
                or type(self.layer_range[1]) is not int \
                or self.layer_range[0] < 0 or self.layer_range[1] <= self.layer_range[0]:
            raise LiveContractError("layer_range must be a valid integer tuple")
        if type(self.host_tail_range) is not tuple or len(self.host_tail_range) != 2 \
                or type(self.host_tail_range[0]) is not int \
                or type(self.host_tail_range[1]) is not int \
                or self.host_tail_range[0] != self.layer_range[1] \
                or self.host_tail_range[1] <= self.host_tail_range[0]:
            raise LiveContractError("host_tail_range must continue the phone layer range")
        integer("route_epoch", self.route_epoch, 1)
        integer("residency_epoch", self.residency_epoch, 1)
        integer("device_boot_epoch", self.device_boot_epoch, 1)
        integer("registry_generation", self.registry_generation, 1)
        integer("batch_size", self.batch_size, 1, MAX_BATCH)
        integer("max_n_gen", self.max_n_gen, 1, MAX_N_GEN)


@dataclass(frozen=True)
class LaunchSpec:
    launch_id: int
    prompt: str
    n_gen: int
    request_ids: tuple[str, ...]
    session_end: str
    deadline_us: int
    expected_token_sha256: str

    def validate(self, route: FrozenRoute) -> None:
        integer("launch_id", self.launch_id, 1)
        text("prompt", self.prompt, MAX_PROMPT_BYTES)
        integer("n_gen", self.n_gen, 1, route.max_n_gen)
        if type(self.request_ids) is not tuple \
                or len(self.request_ids) != route.batch_size:
            raise LiveContractError("request_ids must match the frozen batch size")
        for request_id in self.request_ids:
            text("request_id", request_id)
        if len(set(self.request_ids)) != len(self.request_ids):
            raise LiveContractError("request_ids must be unique")
        if self.session_end not in SESSION_ENDS:
            raise LiveContractError("session_end must be DETACH or STOP")
        integer("deadline_us", self.deadline_us, 1)
        sha256_text("expected_token_sha256", self.expected_token_sha256)


def execute_record(route: FrozenRoute, spec: LaunchSpec, lease_epoch: int,
                   cohort_sha256: str, input_manifest_sha256: str) -> dict:
    route.validate()
    spec.validate(route)
    integer("lease_epoch", lease_epoch, 1)
    sha256_text("cohort_sha256", cohort_sha256)
    sha256_text("input_manifest_sha256", input_manifest_sha256)
    return {
        "schema": EXECUTE_SCHEMA,
        "command": "EXECUTE",
        "protocol_version": PROTOCOL_VERSION,
        "launch_id": spec.launch_id,
        "prompt": spec.prompt,
        "n_gen": spec.n_gen,
        "batch_size": route.batch_size,
        "request_ids": list(spec.request_ids),
        "session_end": spec.session_end,
        "deadline_us": spec.deadline_us,
        "expected_route_id": route.route_id,
        "expected_profile_id": route.profile_id,
        "expected_evidence_sha256": route.evidence_sha256,
        "expected_device_id": route.device_id,
        "expected_device_boot_id": route.device_boot_id,
        "expected_worker_binary_sha256": route.worker_binary_sha256,
        "expected_layer_range": list(route.layer_range),
        "expected_host_tail_range": list(route.host_tail_range),
        "route_epoch": route.route_epoch,
        "residency_epoch": route.residency_epoch,
        "lease_epoch": lease_epoch,
        "device_boot_epoch": route.device_boot_epoch,
        "registry_generation": route.registry_generation,
        "cohort_sha256": cohort_sha256,
        "input_manifest_sha256": input_manifest_sha256,
    }


def child_command(value: dict) -> dict:
    return {
        "schema": CHILD_COMMAND_SCHEMA,
        "launch_id": value["launch_id"],
        "prompt": value["prompt"],
        "n_gen": value["n_gen"],
        "request_count": value["batch_size"],
        "session_end": value["session_end"],
    }


def parse_host_placement(payload: bytes, host_tail_range: tuple[int, int],
                         host_pid: int) -> dict:
    if type(payload) is not bytes or not payload.startswith(HOST_PLACEMENT_PREFIX):
        raise LiveContractError("host PLACEMENTCERT prefix is missing")
    value = strict_wire_line(payload[len(HOST_PLACEMENT_PREFIX):], "host PLACEMENTCERT")
    required = {
        "schema", "role", "mode", "layer_start", "layer_end", "n_layer",
        "pid", "run_rc", "compute_nodes", "copy_nodes", "metadata_nodes",
        "missing_buffer_compute_nodes", "compute_by_buffer_type",
        "compute_by_op", "compute_by_op_and_buffer", "copy_by_buffer_type",
        "status",
    }
    if set(value) != required:
        raise LiveContractError("host PLACEMENTCERT has missing or unknown fields")
    expected = {
        "schema": HOST_PLACEMENT_SCHEMA,
        "role": "host_tail",
        "mode": "pipedriver",
        "layer_start": host_tail_range[0],
        "layer_end": host_tail_range[1],
        "n_layer": host_tail_range[1],
        "pid": host_pid,
        "run_rc": 0,
        "missing_buffer_compute_nodes": 0,
        "status": "SCHEDULED_PLACEMENT_OK",
    }
    for key, wanted in expected.items():
        if type(value.get(key)) is not type(wanted) or value.get(key) != wanted:
            raise LiveContractError(f"host PLACEMENTCERT identity mismatch: {key}")
    compute_nodes = integer("host compute_nodes", value["compute_nodes"], 1)
    copy_nodes = integer("host copy_nodes", value["copy_nodes"])
    integer("host metadata_nodes", value["metadata_nodes"])

    by_buffer = value["compute_by_buffer_type"]
    if type(by_buffer) is not dict or set(by_buffer) != {"CUDA0"} \
            or integer("host CUDA compute count", by_buffer.get("CUDA0"), 1) != compute_nodes:
        raise LiveContractError("host placement is not exclusively CUDA0")
    by_op = value["compute_by_op"]
    by_op_buffer = value["compute_by_op_and_buffer"]
    if type(by_op) is not dict or not by_op \
            or type(by_op_buffer) is not dict or set(by_op_buffer) != set(by_op):
        raise LiveContractError("host placement compute maps are incomplete")
    op_total = 0
    op_buffer_total = 0
    for op, count in by_op.items():
        text("host placement op", op)
        op_total += integer("host placement op count", count, 1)
        buffers = by_op_buffer[op]
        if type(buffers) is not dict or set(buffers) != {"CUDA0"}:
            raise LiveContractError("host placement op used a non-CUDA backend")
        op_buffer_total += integer(
            "host placement op CUDA count", buffers["CUDA0"], 1,
        )
    if op_total != compute_nodes or op_buffer_total != compute_nodes:
        raise LiveContractError("host placement compute totals are inconsistent")

    copy_map = value["copy_by_buffer_type"]
    if type(copy_map) is not dict:
        raise LiveContractError("host placement copy map is invalid")
    copy_total = 0
    for backend, count in copy_map.items():
        text("host placement copy backend", backend)
        copy_total += integer("host placement copy count", count, 1)
    if copy_total != copy_nodes:
        raise LiveContractError("host placement copy total is inconsistent")
    return value


def safe_artifact(root: Path, relative: object) -> Path:
    name = text("artifact path", relative)
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise LiveContractError("artifact path escapes its root")
    resolved_root = root.resolve()
    candidate = root / path
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise LiveContractError("artifact path contains a symlink")
    resolved = candidate.resolve()
    if resolved == resolved_root or resolved_root not in resolved.parents \
            or not resolved.is_file():
        raise LiveContractError("artifact path is not a regular file under its root")
    return resolved


def read_artifact(root: Path, relative: object) -> tuple[Path, bytes]:
    path = safe_artifact(root, relative)
    candidate = root / Path(text("artifact path", relative))
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(candidate, flags)
    except OSError as exc:
        raise LiveContractError(f"artifact could not be opened safely: {exc}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise LiveContractError("artifact descriptor is not a regular file")
        descriptor_path = Path(f"/proc/self/fd/{fd}")
        if descriptor_path.exists():
            opened = descriptor_path.resolve()
            resolved_root = root.resolve()
            if opened != path or resolved_root not in opened.parents:
                raise LiveContractError("artifact descriptor identity changed during open")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            payload = stream.read(MAX_LINE_BYTES + 1)
        if len(payload) > MAX_LINE_BYTES:
            raise LiveContractError("artifact exceeds its byte bound")
        return path, payload
    finally:
        os.close(fd)
