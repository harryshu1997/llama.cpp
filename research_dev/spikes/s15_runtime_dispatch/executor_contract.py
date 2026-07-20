#!/usr/bin/env python3
"""Typed executor contract for later physical S15 integration.

This defines the request/result boundary between the host dispatch plane and a
device executor. The only executor implemented here is a deterministic recorded
replay for tests. There is no ADB, socket, or device I/O in this module, and it
never fabricates a physical completion: a launch with no recorded entry returns
a timed_out result.
"""

from __future__ import annotations

import sys
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

_S14 = Path(__file__).resolve().parent.parent / "s14_mixed_streaming_scheduler"
if str(_S14) not in sys.path:
    sys.path.insert(0, str(_S14))

from power_frontier_policy import BoundaryCertificate  # noqa: E402


BOUNDARY_SCHEMA = "s15-boundary-certificate-v1"
OUTCOMES = ("completed", "timed_out", "error")


class ExecutorError(ValueError):
    pass


def _int(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ExecutorError(f"{name} must be an integer >= {minimum}")
    return value


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise ExecutorError(f"{name} must be a non-empty string")
    return value


def _sha256(name: str, value: object) -> str:
    text = _text(name, value)
    if re.fullmatch(r"sha256:[0-9a-f]{64}", text) is None:
        raise ExecutorError(f"{name} must be a lowercase sha256 digest")
    return text


@dataclass(frozen=True)
class ExecutionRequest:
    launch_id: int
    route_id: str
    profile_id: str
    device_id: str
    route_epoch: int
    residency_epoch: int
    lease_epoch: int
    device_boot_epoch: int
    registry_generation: int
    compatibility_key: str
    request_ids: tuple[str, ...]
    cohort_sha256: str
    input_manifest_sha256: str
    timeout_us: int
    expected_boundary_schema: str

    def validate(self) -> None:
        _int("launch_id", self.launch_id, 1)
        _text("route_id", self.route_id)
        _text("profile_id", self.profile_id)
        _text("device_id", self.device_id)
        _int("route_epoch", self.route_epoch, 1)
        _int("residency_epoch", self.residency_epoch, 1)
        _int("lease_epoch", self.lease_epoch, 1)
        _int("device_boot_epoch", self.device_boot_epoch, 1)
        _int("registry_generation", self.registry_generation, 1)
        _text("compatibility_key", self.compatibility_key)
        _sha256("cohort_sha256", self.cohort_sha256)
        _sha256("input_manifest_sha256", self.input_manifest_sha256)
        _text("expected_boundary_schema", self.expected_boundary_schema)
        _int("timeout_us", self.timeout_us, 1)
        if type(self.request_ids) is not tuple or not self.request_ids:
            raise ExecutorError("request_ids must be a non-empty tuple")
        for request_id in self.request_ids:
            _text("request_id", request_id)
        if len(set(self.request_ids)) != len(self.request_ids):
            raise ExecutorError("request_ids must be unique")


@dataclass(frozen=True)
class ExecutionResult:
    launch_id: int
    outcome: str
    finish_us: int
    boundary_schema: str
    certificates: tuple[BoundaryCertificate, ...]

    def validate(self, request: ExecutionRequest) -> None:
        if type(request) is not ExecutionRequest:
            raise ExecutorError("result must be validated against an ExecutionRequest")
        request.validate()
        _int("launch_id", self.launch_id, 1)
        if self.launch_id != request.launch_id:
            raise ExecutorError("result launch id does not match the request")
        if self.outcome not in OUTCOMES:
            raise ExecutorError(f"unknown outcome {self.outcome!r}")
        _int("finish_us", self.finish_us, 0)
        if self.outcome == "completed":
            if self.boundary_schema != request.expected_boundary_schema:
                raise ExecutorError("completed result has the wrong boundary schema")
            if type(self.certificates) is not tuple:
                raise ExecutorError("certificates must be a tuple")
            seen = []
            for cert in self.certificates:
                if type(cert) is not BoundaryCertificate:
                    raise ExecutorError("certificates must be BoundaryCertificate values")
                seen.append(cert.request_id)
            if sorted(seen) != sorted(request.request_ids) or len(seen) != len(set(seen)):
                raise ExecutorError("completed result must carry one certificate per request")
        else:
            if self.certificates != ():
                raise ExecutorError("a non-completed result must carry no certificate")


class Executor(ABC):
    """Typed launch boundary. A physical executor implements launch() later."""

    @abstractmethod
    def launch(self, request: ExecutionRequest, now_us: int) -> ExecutionResult:
        raise NotImplementedError


@dataclass(frozen=True)
class RecordedOutcome:
    outcome: str
    finish_delay_us: int
    profile_id: str
    route_epoch: int
    residency_epoch: int
    device_boot_epoch: int
    cohort_sha256: str
    input_manifest_sha256: str
    lease_epoch: int = 1
    registry_generation: int = 1
    admit: Mapping[str, bool] = field(default_factory=dict)

    def validate(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ExecutorError(f"unknown recorded outcome {self.outcome!r}")
        _int("finish_delay_us", self.finish_delay_us, 0)
        _text("profile_id", self.profile_id)
        _int("route_epoch", self.route_epoch, 1)
        _int("residency_epoch", self.residency_epoch, 1)
        _int("device_boot_epoch", self.device_boot_epoch, 1)
        _sha256("cohort_sha256", self.cohort_sha256)
        _sha256("input_manifest_sha256", self.input_manifest_sha256)
        _int("lease_epoch", self.lease_epoch, 1)
        _int("registry_generation", self.registry_generation, 1)
        if not isinstance(self.admit, Mapping):
            raise ExecutorError("admit must be a mapping")
        for key, value in self.admit.items():
            _text("admit key", key)
            if type(value) is not bool:
                raise ExecutorError("admit values must be bool")


def recorded_key(route_id: str, request_ids: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
    return (route_id, tuple(sorted(request_ids)))


class RecordedExecutor(Executor):
    """Replay recorded outcomes. Never contacts a device, never invents success."""

    def __init__(self, script: Mapping[tuple[str, tuple[str, ...]], RecordedOutcome]) -> None:
        if not isinstance(script, Mapping):
            raise ExecutorError("script must be a mapping")
        self._script: dict[tuple[str, tuple[str, ...]], RecordedOutcome] = {}
        for key, outcome in script.items():
            if type(key) is not tuple or len(key) != 2 or type(key[0]) is not str \
                    or type(key[1]) is not tuple:
                raise ExecutorError("script keys must be (route_id, request_ids) tuples")
            if type(outcome) is not RecordedOutcome:
                raise ExecutorError("script values must be RecordedOutcome")
            outcome.validate()
            self._script[(key[0], tuple(sorted(key[1])))] = outcome

    def launch(self, request: ExecutionRequest, now_us: int) -> ExecutionResult:
        request.validate()
        _int("now_us", now_us)
        key = recorded_key(request.route_id, request.request_ids)
        recorded = self._script.get(key)
        if recorded is None:
            # No recorded evidence: the plane must treat this as a timeout, not a win.
            result = ExecutionResult(
                request.launch_id, "timed_out", now_us + request.timeout_us,
                request.expected_boundary_schema, (),
            )
            result.validate(request)
            return result
        if (
            recorded.profile_id != request.profile_id
            or recorded.route_epoch != request.route_epoch
            or recorded.residency_epoch != request.residency_epoch
            or recorded.lease_epoch != request.lease_epoch
            or recorded.device_boot_epoch != request.device_boot_epoch
            or recorded.registry_generation != request.registry_generation
            or recorded.cohort_sha256 != request.cohort_sha256
            or recorded.input_manifest_sha256 != request.input_manifest_sha256
        ):
            raise ExecutorError("recorded outcome identity does not match the launch")
        if recorded.outcome != "completed":
            result = ExecutionResult(
                request.launch_id, recorded.outcome,
                now_us + recorded.finish_delay_us,
                request.expected_boundary_schema, (),
            )
            result.validate(request)
            return result
        certificates = tuple(
            self._certificate(request_id, recorded)
            for request_id in request.request_ids
        )
        result = ExecutionResult(
            request.launch_id, "completed",
            now_us + recorded.finish_delay_us,
            request.expected_boundary_schema, certificates,
        )
        result.validate(request)
        return result

    @staticmethod
    def _certificate(request_id: str, recorded: RecordedOutcome) -> BoundaryCertificate:
        admitted = recorded.admit.get(request_id, True)
        return BoundaryCertificate(request_id, admitted, admitted, admitted, admitted)
