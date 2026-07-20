#!/usr/bin/env python3
"""Fail-closed physical executor for the S15 dispatch boundary.

The executor invokes a configured launcher command. The launcher owns ADB or
socket setup and must return one strict JSON session record on stdout. Transport
bytes are persisted before any completion is admitted.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from executor_contract import (
    BOUNDARY_SCHEMA,
    ExecutionRequest,
    ExecutionResult,
    Executor,
    ExecutorError,
)
from power_frontier_policy import BoundaryCertificate


REQUEST_SCHEMA = "s15-physical-request-v1"
SESSION_SCHEMA = "s15-physical-session-v1"
TRANSPORT_STATES = ("reply", "timed_out", "error")
SESSION_OUTCOMES = ("completed", "error")
EXPECTED_BACKENDS = ("HTP0", "CUDA0")
MAX_REPLY_BYTES = 4 * 1024 * 1024


class PhysicalExecutorError(ExecutorError):
    pass


def _int(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PhysicalExecutorError(f"{name} must be an integer >= {minimum}")
    return value


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise PhysicalExecutorError(f"{name} must be a non-empty string")
    return value


def _sha256(name: str, value: object) -> str:
    text = _text(name, value)
    prefix = "sha256:"
    if not text.startswith(prefix) or len(text) != len(prefix) + 64:
        raise PhysicalExecutorError(f"{name} must be a sha256 digest")
    suffix = text[len(prefix):]
    if any(char not in "0123456789abcdef" for char in suffix):
        raise PhysicalExecutorError(f"{name} must be a lowercase sha256 digest")
    return text


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")


def _load_strict(payload: bytes) -> dict:
    if type(payload) is not bytes or not payload or len(payload) > MAX_REPLY_BYTES:
        raise PhysicalExecutorError("session record has an invalid byte length")

    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise PhysicalExecutorError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PhysicalExecutorError(f"invalid session JSON: {exc}") from exc
    if type(value) is not dict:
        raise PhysicalExecutorError("session record must be an object")
    return value


@dataclass(frozen=True)
class PhysicalRouteBinding:
    route_id: str
    profile_id: str
    device_id: str
    protocol_version: int
    worker_binary_sha256: str
    worker_generation: int
    first_session_id: int
    device_boot_id: str
    cohort_sha256: str
    input_manifest_sha256: str
    layer_range: tuple[int, int] | None
    expected_backend: str
    allowed_cpu_ops: tuple[str, ...] = ("GET_ROWS",)

    def validate(self) -> None:
        _text("route_id", self.route_id)
        _sha256("profile_id", self.profile_id)
        _text("device_id", self.device_id)
        _int("protocol_version", self.protocol_version, 1)
        _sha256("worker_binary_sha256", self.worker_binary_sha256)
        _int("worker_generation", self.worker_generation, 1)
        _int("first_session_id", self.first_session_id, 1)
        _text("device_boot_id", self.device_boot_id)
        _sha256("cohort_sha256", self.cohort_sha256)
        _sha256("input_manifest_sha256", self.input_manifest_sha256)
        if self.expected_backend not in EXPECTED_BACKENDS:
            raise PhysicalExecutorError("unsupported expected backend")
        if self.layer_range is not None:
            if type(self.layer_range) is not tuple or len(self.layer_range) != 2 \
                    or type(self.layer_range[0]) is not int or type(self.layer_range[1]) is not int \
                    or self.layer_range[0] < 0 or self.layer_range[1] <= self.layer_range[0]:
                raise PhysicalExecutorError("invalid layer range")
        if self.expected_backend == "HTP0" and self.layer_range is None:
            raise PhysicalExecutorError("HTP0 route requires a layer range")
        if type(self.allowed_cpu_ops) is not tuple or len(set(self.allowed_cpu_ops)) != len(self.allowed_cpu_ops):
            raise PhysicalExecutorError("allowed_cpu_ops must be a unique tuple")
        for op in self.allowed_cpu_ops:
            _text("allowed CPU op", op)


@dataclass(frozen=True)
class TransportReply:
    state: str
    elapsed_us: int
    payload: bytes = b""

    def validate(self) -> None:
        if self.state not in TRANSPORT_STATES:
            raise PhysicalExecutorError(f"unknown transport state {self.state!r}")
        _int("elapsed_us", self.elapsed_us)
        if type(self.payload) is not bytes:
            raise PhysicalExecutorError("transport payload must be bytes")
        if self.state == "reply" and not self.payload:
            raise PhysicalExecutorError("reply transport has no payload")
        if self.state != "reply" and self.payload:
            raise PhysicalExecutorError("failed transport must not carry a session record")


class SessionTransport(ABC):
    @abstractmethod
    def exchange(self, request: ExecutionRequest, payload: bytes) -> TransportReply:
        raise NotImplementedError


class SubprocessSessionTransport(SessionTransport):
    """Run one configured launcher command and retain the exact transport bytes."""

    def __init__(self, commands: Mapping[str, tuple[str, ...]], artifact_dir: Path) -> None:
        if not isinstance(commands, Mapping) or not commands:
            raise PhysicalExecutorError("commands must be a non-empty mapping")
        self._commands = {}
        for route_id, command in commands.items():
            _text("command route_id", route_id)
            if type(command) is not tuple or not command:
                raise PhysicalExecutorError("each launcher command must be a non-empty tuple")
            for argument in command:
                _text("launcher argument", argument)
            self._commands[route_id] = command
        if not isinstance(artifact_dir, Path):
            raise PhysicalExecutorError("artifact_dir must be a Path")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        self._artifact_dir = artifact_dir

    def exchange(self, request: ExecutionRequest, payload: bytes) -> TransportReply:
        command = self._commands.get(request.route_id)
        if command is None:
            return TransportReply("error", 0)
        route_tag = hashlib.sha256(request.route_id.encode("ascii")).hexdigest()[:12]
        run_dir = self._artifact_dir / f"{route_tag}-launch-{request.launch_id}"
        try:
            run_dir.mkdir()
        except FileExistsError as exc:
            raise PhysicalExecutorError("launch artifact directory already exists") from exc
        (run_dir / "request.json").write_bytes(payload)
        start_ns = time.monotonic_ns()
        try:
            process = subprocess.run(
                command,
                input=payload,
                capture_output=True,
                timeout=request.timeout_us / 1_000_000,
                check=False,
            )
            elapsed_us = max(0, (time.monotonic_ns() - start_ns) // 1000)
            stdout = process.stdout
            stderr = process.stderr
            state = "reply" if process.returncode == 0 and 0 < len(stdout) <= MAX_REPLY_BYTES else "error"
        except subprocess.TimeoutExpired as exc:
            elapsed_us = max(0, (time.monotonic_ns() - start_ns) // 1000)
            stdout = exc.stdout or b""
            stderr = exc.stderr or b""
            process = None
            state = "timed_out"
        except OSError as exc:
            elapsed_us = max(0, (time.monotonic_ns() - start_ns) // 1000)
            stdout = b""
            stderr = str(exc).encode("utf-8", errors="replace")
            process = None
            state = "error"
        (run_dir / "stdout.bin").write_bytes(stdout)
        (run_dir / "stderr.bin").write_bytes(stderr)
        metadata = {
            "state": state,
            "elapsed_us": elapsed_us,
            "returncode": None if process is None else process.returncode,
            "request_sha256": _digest(payload),
            "stdout_sha256": _digest(stdout),
            "stderr_sha256": _digest(stderr),
        }
        (run_dir / "transport.json").write_bytes(_canonical(metadata))
        return TransportReply(state, elapsed_us, stdout if state == "reply" else b"")


class PhysicalExecutor(Executor):
    def __init__(self, bindings: Sequence[PhysicalRouteBinding], transport: SessionTransport) -> None:
        if type(bindings) not in (list, tuple) or not bindings:
            raise PhysicalExecutorError("bindings must be a non-empty list or tuple")
        if not isinstance(transport, SessionTransport):
            raise PhysicalExecutorError("transport must implement SessionTransport")
        self._bindings = {}
        self._last_session_id = {}
        for binding in bindings:
            if type(binding) is not PhysicalRouteBinding:
                raise PhysicalExecutorError("bindings must contain PhysicalRouteBinding values")
            binding.validate()
            if binding.route_id in self._bindings:
                raise PhysicalExecutorError("duplicate physical route binding")
            self._bindings[binding.route_id] = binding
            self._last_session_id[binding.route_id] = binding.first_session_id - 1
        self._transport = transport

    def launch(self, request: ExecutionRequest, now_us: int) -> ExecutionResult:
        request.validate()
        _int("now_us", now_us)
        binding = self._bindings.get(request.route_id)
        if binding is None:
            return self._result(request, "error", now_us, ())
        self._validate_binding(request, binding)
        payload = self._request_payload(request, binding)
        reply = self._transport.exchange(request, payload)
        if type(reply) is not TransportReply:
            raise PhysicalExecutorError("transport returned the wrong type")
        reply.validate()
        if reply.state == "timed_out" or reply.elapsed_us > request.timeout_us:
            return self._result(request, "timed_out", now_us + request.timeout_us, ())
        finish_us = now_us + reply.elapsed_us
        if reply.state == "error":
            return self._result(request, "error", finish_us, ())
        record = _load_strict(reply.payload)
        certificates = self._validate_record(record, request, binding)
        session_id = record["session_id"]
        if session_id != self._last_session_id[request.route_id] + 1:
            raise PhysicalExecutorError("session id is stale, duplicated, or has a gap")
        self._last_session_id[request.route_id] = session_id
        if record["outcome"] == "error":
            return self._result(request, "error", finish_us, ())
        return self._result(request, "completed", finish_us, certificates)

    @staticmethod
    def _result(request, outcome, finish_us, certificates) -> ExecutionResult:
        result = ExecutionResult(
            request.launch_id,
            outcome,
            finish_us,
            request.expected_boundary_schema,
            tuple(certificates),
        )
        result.validate(request)
        return result

    @staticmethod
    def _validate_binding(request: ExecutionRequest, binding: PhysicalRouteBinding) -> None:
        if request.profile_id != binding.profile_id or request.device_id != binding.device_id:
            raise PhysicalExecutorError("physical route binding does not match the request")
        if request.input_manifest_sha256 != binding.input_manifest_sha256:
            raise PhysicalExecutorError("physical input manifest does not match the request")
        if request.cohort_sha256 != binding.cohort_sha256:
            raise PhysicalExecutorError("physical cohort does not match the request")

    @staticmethod
    def _request_payload(request: ExecutionRequest, binding: PhysicalRouteBinding) -> bytes:
        return _canonical({
            "schema": REQUEST_SCHEMA,
            "command": "EXECUTE",
            "protocol_version": binding.protocol_version,
            "launch_id": request.launch_id,
            "route_id": request.route_id,
            "profile_id": request.profile_id,
            "device_id": request.device_id,
            "route_epoch": request.route_epoch,
            "residency_epoch": request.residency_epoch,
            "lease_epoch": request.lease_epoch,
            "device_boot_epoch": request.device_boot_epoch,
            "registry_generation": request.registry_generation,
            "compatibility_key": request.compatibility_key,
            "request_ids": list(request.request_ids),
            "cohort_sha256": request.cohort_sha256,
            "input_manifest_sha256": request.input_manifest_sha256,
            "timeout_us": request.timeout_us,
            "expected_boundary_schema": request.expected_boundary_schema,
            "worker_binary_sha256": binding.worker_binary_sha256,
            "worker_generation": binding.worker_generation,
            "device_boot_id": binding.device_boot_id,
            "layer_range": None if binding.layer_range is None else list(binding.layer_range),
        })

    def _validate_record(self, record, request, binding):
        required = {
            "schema", "protocol_version", "launch_id", "route_id", "profile_id",
            "device_id", "route_epoch", "residency_epoch", "lease_epoch",
            "device_boot_epoch", "registry_generation", "compatibility_key",
            "request_ids", "worker_binary_sha256", "worker_generation",
            "device_boot_id", "layer_range", "session_id", "outcome",
            "boundary_schema", "cohort_sha256", "input_manifest_sha256",
            "placement", "boundaries",
        }
        if set(record) != required:
            raise PhysicalExecutorError("session record has missing or unknown fields")
        expected = {
            "schema": SESSION_SCHEMA,
            "protocol_version": binding.protocol_version,
            "launch_id": request.launch_id,
            "route_id": request.route_id,
            "profile_id": request.profile_id,
            "device_id": request.device_id,
            "route_epoch": request.route_epoch,
            "residency_epoch": request.residency_epoch,
            "lease_epoch": request.lease_epoch,
            "device_boot_epoch": request.device_boot_epoch,
            "registry_generation": request.registry_generation,
            "compatibility_key": request.compatibility_key,
            "request_ids": list(request.request_ids),
            "cohort_sha256": request.cohort_sha256,
            "input_manifest_sha256": request.input_manifest_sha256,
            "worker_binary_sha256": binding.worker_binary_sha256,
            "worker_generation": binding.worker_generation,
            "device_boot_id": binding.device_boot_id,
            "layer_range": None if binding.layer_range is None else list(binding.layer_range),
            "boundary_schema": request.expected_boundary_schema,
        }
        for key, value in expected.items():
            actual = record.get(key)
            if type(actual) is not type(value) or actual != value:
                raise PhysicalExecutorError(f"session identity mismatch: {key}")
        _int("session_id", record["session_id"], 1)
        if record["outcome"] not in SESSION_OUTCOMES:
            raise PhysicalExecutorError("invalid session outcome")
        if record["outcome"] == "error":
            if record["placement"] is not None or record["boundaries"] != []:
                raise PhysicalExecutorError("error session carries completion evidence")
            return ()
        self._validate_placement(record["placement"], binding)
        return self._boundaries(record["boundaries"], request)

    @staticmethod
    def _validate_placement(placement, binding):
        if type(placement) is not dict:
            raise PhysicalExecutorError("completed session has no placement record")
        required = {
            "status", "layer_start", "layer_end", "missing_buffer_compute_nodes",
            "compute_by_op_and_buffer",
        }
        if set(placement) != required or placement["status"] != "SCHEDULED_PLACEMENT_OK":
            raise PhysicalExecutorError("placement gate failed")
        _int("placement layer_start", placement["layer_start"])
        _int("placement layer_end", placement["layer_end"], 1)
        if placement["layer_end"] <= placement["layer_start"]:
            raise PhysicalExecutorError("placement layer range is empty")
        if _int("missing_buffer_compute_nodes", placement["missing_buffer_compute_nodes"]) != 0:
            raise PhysicalExecutorError("placement gate failed")
        if binding.layer_range is not None \
                and [placement["layer_start"], placement["layer_end"]] != list(binding.layer_range):
            raise PhysicalExecutorError("placement layer range mismatch")
        mapping = placement["compute_by_op_and_buffer"]
        if type(mapping) is not dict or not mapping:
            raise PhysicalExecutorError("placement compute map missing")
        expected_nodes = 0
        for op, buffers in mapping.items():
            _text("placement op", op)
            if type(buffers) is not dict or not buffers:
                raise PhysicalExecutorError("placement buffer map missing")
            for backend, count in buffers.items():
                _int("placement node count", count, 1)
                if backend == binding.expected_backend:
                    expected_nodes += count
                elif backend != "CPU" or op not in binding.allowed_cpu_ops:
                    raise PhysicalExecutorError("undeclared backend fallback")
        if expected_nodes == 0:
            raise PhysicalExecutorError("expected backend has no compute nodes")

    @staticmethod
    def _boundaries(values, request):
        if type(values) is not list:
            raise PhysicalExecutorError("boundaries must be a list")
        certificates = []
        for value in values:
            if type(value) is not dict or set(value) != {
                "request_id", "identity_ok", "epoch_ok", "correctness_ok", "d2h_complete"
            }:
                raise PhysicalExecutorError("invalid boundary record")
            certificate = BoundaryCertificate(
                value["request_id"], value["identity_ok"], value["epoch_ok"],
                value["correctness_ok"], value["d2h_complete"],
            )
            certificate.admitted()
            certificates.append(certificate)
        result = tuple(certificates)
        probe = ExecutionResult(request.launch_id, "completed", 0, BOUNDARY_SCHEMA, result)
        try:
            probe.validate(request)
        except ExecutorError as exc:
            raise PhysicalExecutorError(str(exc)) from exc
        return result
