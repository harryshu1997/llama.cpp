#!/usr/bin/env python3
"""Fail-closed adapter from LayerSplit SESSIONCERT to the S15 session record."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from executor_contract import ExecutionRequest
from physical_executor import (
    PhysicalExecutorError,
    PhysicalRouteBinding,
    SESSION_SCHEMA,
)


CERT_SCHEMA = "ls-stagenet-session-v2"
CERT_PREFIX = b"SESSIONCERT "
CERT_ENDS = ("DETACH", "STOP")
ADAPTER_STATES = ("IDLE", "ACTIVE", "DETACHED", "STOPPED", "POISONED")
MAX_CERT_BYTES = 4 * 1024 * 1024


class SessionAdapterError(PhysicalExecutorError):
    pass


def _int(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise SessionAdapterError(f"{name} must be an integer >= {minimum}")
    return value


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise SessionAdapterError(f"{name} must be a non-empty string")
    return value


def _sha256(name: str, value: object) -> str:
    text = _text(name, value)
    if re.fullmatch(r"sha256:[0-9a-f]{64}", text) is None:
        raise SessionAdapterError(f"{name} must be a lowercase sha256 digest")
    return text


def parse_session_cert(payload: bytes) -> dict:
    if type(payload) is not bytes or not payload or len(payload) > MAX_CERT_BYTES \
            or not payload.endswith(b"\n") or payload.count(b"\n") != 1 \
            or not payload.startswith(CERT_PREFIX):
        raise SessionAdapterError("SESSIONCERT is not one bounded prefixed JSON line")

    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise SessionAdapterError(f"duplicate SESSIONCERT key {key!r}")
            result[key] = value
        return result

    def no_constants(value):
        raise SessionAdapterError(f"invalid SESSIONCERT constant {value!r}")

    try:
        value = json.loads(
            payload[len(CERT_PREFIX):],
            object_pairs_hook=no_duplicates,
            parse_constant=no_constants,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SessionAdapterError(f"invalid SESSIONCERT JSON: {exc}") from exc
    if type(value) is not dict:
        raise SessionAdapterError("SESSIONCERT must be an object")
    return value


@dataclass(frozen=True)
class PersistentWorkerCapability:
    worker_binary_sha256: str
    protocol_version: int
    detach_supported: bool

    def validate(self) -> None:
        _sha256("capability worker_binary_sha256", self.worker_binary_sha256)
        if type(self.protocol_version) is not int or self.protocol_version != 2:
            raise SessionAdapterError("persistent capability protocol_version must be 2")
        if type(self.detach_supported) is not bool:
            raise SessionAdapterError("detach_supported must be bool")


@dataclass(frozen=True)
class StageNetSessionBinding:
    worker_binary_sha256: str
    first_session_id: int
    device_boot_id: str
    layer_range: tuple[int, int]
    n_layer: int
    allowed_cpu_ops: tuple[str, ...] = ("GET_ROWS",)

    def validate(self) -> None:
        _sha256("worker_binary_sha256", self.worker_binary_sha256)
        _int("first_session_id", self.first_session_id, 1)
        _text("device_boot_id", self.device_boot_id)
        if type(self.layer_range) is not tuple or len(self.layer_range) != 2 \
                or type(self.layer_range[0]) is not int \
                or type(self.layer_range[1]) is not int \
                or self.layer_range[0] < 0 \
                or self.layer_range[1] <= self.layer_range[0]:
            raise SessionAdapterError("layer_range must be a valid integer tuple")
        _int("n_layer", self.n_layer, 1)
        if self.layer_range[1] > self.n_layer:
            raise SessionAdapterError("layer_range exceeds n_layer")
        if type(self.allowed_cpu_ops) is not tuple \
                or len(set(self.allowed_cpu_ops)) != len(self.allowed_cpu_ops):
            raise SessionAdapterError("allowed_cpu_ops must be a unique tuple")
        for op in self.allowed_cpu_ops:
            _text("allowed CPU op", op)


class StageNetSessionAdapter:
    """Validate session closure before a resident-worker lease is reusable."""

    def __init__(self, binding: StageNetSessionBinding,
                 capabilities: tuple[PersistentWorkerCapability, ...]) -> None:
        if type(binding) is not StageNetSessionBinding:
            raise SessionAdapterError("binding must be StageNetSessionBinding")
        binding.validate()
        if type(capabilities) is not tuple:
            raise SessionAdapterError("capabilities must be a tuple")
        self._capabilities = {}
        for capability in capabilities:
            if type(capability) is not PersistentWorkerCapability:
                raise SessionAdapterError("invalid persistent capability value")
            capability.validate()
            if capability.worker_binary_sha256 in self._capabilities:
                raise SessionAdapterError("duplicate persistent worker capability")
            self._capabilities[capability.worker_binary_sha256] = capability
        self._binding = binding
        self._state = "IDLE"
        self._expected_end: str | None = None
        self._next_session_id = binding.first_session_id
        self._worker_pid: int | None = None
        self._worker_boot_nonce: str | None = None
        self._steps_total: int | None = None
        self._poison_reason: str | None = None

    @property
    def state(self) -> str:
        return self._state

    @property
    def poison_reason(self) -> str | None:
        return self._poison_reason

    @property
    def reuse_permitted(self) -> bool:
        return self._state == "DETACHED" and self._poison_reason is None

    def begin(self, expected_end: str) -> int:
        if expected_end not in CERT_ENDS:
            raise SessionAdapterError("expected_end must be DETACH or STOP")
        if self._state != "IDLE" or self._poison_reason is not None:
            raise SessionAdapterError(f"cannot begin in adapter state {self._state}")
        if expected_end == "DETACH":
            capability = self._capabilities.get(self._binding.worker_binary_sha256)
            if capability is None or not capability.detach_supported:
                raise SessionAdapterError(
                    "worker binary has no explicit persistent DETACH capability"
                )
        self._state = "ACTIVE"
        self._expected_end = expected_end
        return self._next_session_id

    def abort(self, reason: str) -> None:
        self._poison(_text("abort reason", reason))

    def accept(self, cert_payload: bytes | None, detach_ack: int | None,
               worker_terminated: bool,
               request: ExecutionRequest, physical_binding: PhysicalRouteBinding,
               boundaries: list[dict]) -> bytes:
        if self._state != "ACTIVE" or self._expected_end is None:
            self._poison("certificate arrived outside an active session")
            raise SessionAdapterError(self._poison_reason)
        try:
            if cert_payload is None:
                raise SessionAdapterError("session ended without SESSIONCERT")
            cert = parse_session_cert(cert_payload)
            self._validate_cert(cert, detach_ack, worker_terminated)
            record = self._physical_record(cert, request, physical_binding, boundaries)
        except Exception as exc:
            reason = str(exc) if isinstance(exc, SessionAdapterError) else f"adapter failure: {exc}"
            self._poison(reason)
            if isinstance(exc, SessionAdapterError):
                raise
            raise SessionAdapterError(reason) from exc

        self._steps_total = cert["steps_total"]
        self._next_session_id += 1
        if cert["session_end"] == "DETACH":
            self._state = "DETACHED"
        else:
            self._state = "STOPPED"
        self._expected_end = None
        return (
            json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        ).encode("ascii")

    def release_after_detach(self) -> None:
        if not self.reuse_permitted:
            raise SessionAdapterError("worker lease is not reusable")
        self._state = "IDLE"

    def _poison(self, reason: str) -> None:
        if self._poison_reason is None:
            self._poison_reason = reason
        self._state = "POISONED"
        self._expected_end = None

    def _validate_cert(self, cert: dict, detach_ack: int | None,
                       worker_terminated: bool) -> None:
        required = {
            "schema", "proto_version", "session_id", "session_end",
            "expected_backend", "worker_pid", "worker_boot_nonce",
            "device_boot_id", "layer_start", "layer_end", "n_layer",
            "steps_session", "steps_total", "reset_applied",
            "missing_buffer_compute_nodes", "compute_by_op_and_buffer",
            "placement_status",
        }
        if set(cert) != required:
            raise SessionAdapterError("SESSIONCERT has missing or unknown fields")
        expected_scalars = {
            "schema": CERT_SCHEMA,
            "proto_version": 2,
            "session_id": self._next_session_id,
            "session_end": self._expected_end,
            "expected_backend": "HTP0",
            "device_boot_id": self._binding.device_boot_id,
            "layer_start": self._binding.layer_range[0],
            "layer_end": self._binding.layer_range[1],
            "n_layer": self._binding.n_layer,
            "missing_buffer_compute_nodes": 0,
            "placement_status": "SCHEDULED_PLACEMENT_OK",
        }
        for key, expected in expected_scalars.items():
            if type(cert.get(key)) is not type(expected) or cert.get(key) != expected:
                raise SessionAdapterError(f"SESSIONCERT identity mismatch: {key}")
        worker_pid = _int("worker_pid", cert["worker_pid"], 1)
        worker_nonce = _text("worker_boot_nonce", cert["worker_boot_nonce"])
        if self._worker_pid is None:
            self._worker_pid = worker_pid
            self._worker_boot_nonce = worker_nonce
        elif worker_pid != self._worker_pid or worker_nonce != self._worker_boot_nonce:
            raise SessionAdapterError("resident worker identity changed across sessions")

        steps_session = _int("steps_session", cert["steps_session"])
        steps_total = _int("steps_total", cert["steps_total"])
        if steps_session > steps_total:
            raise SessionAdapterError("steps_session exceeds steps_total")
        if self._steps_total is not None and steps_total != self._steps_total + steps_session:
            raise SessionAdapterError("steps_total is not contiguous across sessions")
        if type(cert["reset_applied"]) is not bool:
            raise SessionAdapterError("reset_applied must be bool")
        if type(worker_terminated) is not bool:
            raise SessionAdapterError("worker_terminated must be bool")
        if self._expected_end == "DETACH":
            if cert["reset_applied"] is not True:
                raise SessionAdapterError("DETACH did not reset request-local state")
            if type(detach_ack) is not int or type(detach_ack) is bool or detach_ack != 0:
                raise SessionAdapterError("DETACH is missing its zero ACK")
            if worker_terminated:
                raise SessionAdapterError("DETACH unexpectedly terminated the resident worker")
        else:
            if detach_ack is not None:
                raise SessionAdapterError("STOP must not carry a DETACH ACK")
            if not worker_terminated:
                raise SessionAdapterError("STOP did not terminate the resident worker")
        self._validate_placement(cert["compute_by_op_and_buffer"])

    def _validate_placement(self, mapping: object) -> None:
        if type(mapping) is not dict or not mapping:
            raise SessionAdapterError("SESSIONCERT has no placement tally")
        htp_nodes = 0
        for op, buffers in mapping.items():
            _text("placement op", op)
            if type(buffers) is not dict or not buffers:
                raise SessionAdapterError("placement backend map is empty")
            for backend, count in buffers.items():
                _text("placement backend", backend)
                _int("placement count", count, 1)
                if backend == "HTP0":
                    htp_nodes += count
                elif backend != "CPU" or op not in self._binding.allowed_cpu_ops:
                    raise SessionAdapterError("SESSIONCERT contains undeclared backend fallback")
        if htp_nodes == 0:
            raise SessionAdapterError("SESSIONCERT observed no HTP0 compute")

    def _physical_record(self, cert: dict, request: ExecutionRequest,
                         physical_binding: PhysicalRouteBinding,
                         boundaries: list[dict]) -> dict:
        if type(request) is not ExecutionRequest \
                or type(physical_binding) is not PhysicalRouteBinding:
            raise SessionAdapterError("physical adapter inputs have invalid types")
        request.validate()
        physical_binding.validate()
        if physical_binding.worker_binary_sha256 != self._binding.worker_binary_sha256 \
                or physical_binding.device_boot_id != self._binding.device_boot_id \
                or physical_binding.layer_range != self._binding.layer_range \
                or physical_binding.expected_backend != "HTP0":
            raise SessionAdapterError("physical route binding differs from SESSIONCERT binding")
        if type(boundaries) is not list or len(boundaries) != len(request.request_ids):
            raise SessionAdapterError("completion boundaries do not match request count")
        expected_boundary_keys = {
            "request_id", "identity_ok", "epoch_ok", "correctness_ok", "d2h_complete",
        }
        seen = []
        for boundary in boundaries:
            if type(boundary) is not dict or set(boundary) != expected_boundary_keys:
                raise SessionAdapterError("completion boundary has invalid fields")
            seen.append(boundary["request_id"])
            if any(boundary[key] is not True for key in expected_boundary_keys - {"request_id"}):
                raise SessionAdapterError("completion boundary did not pass")
        if seen != list(request.request_ids):
            raise SessionAdapterError("completion boundary ownership is not exact")

        placement = {
            "status": cert["placement_status"],
            "layer_start": cert["layer_start"],
            "layer_end": cert["layer_end"],
            "missing_buffer_compute_nodes": cert["missing_buffer_compute_nodes"],
            "compute_by_op_and_buffer": cert["compute_by_op_and_buffer"],
        }
        return {
            "schema": SESSION_SCHEMA,
            "protocol_version": physical_binding.protocol_version,
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
            "worker_binary_sha256": physical_binding.worker_binary_sha256,
            "worker_generation": physical_binding.worker_generation,
            "device_boot_id": physical_binding.device_boot_id,
            "layer_range": list(self._binding.layer_range),
            "session_id": cert["session_id"],
            "outcome": "completed",
            "boundary_schema": request.expected_boundary_schema,
            "placement": placement,
            "boundaries": boundaries,
        }
