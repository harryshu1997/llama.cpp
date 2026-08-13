#!/usr/bin/env python3
"""Fail-closed runtime, semantic, and failure gates for certified routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


__all__ = [
    "CONTRACT_SCHEMA",
    "FAILURE_MODES",
    "SNAPSHOT_SCHEMA",
    "RequestSemantics",
    "ResourceRequirement",
    "RouteRuntimeContract",
    "RuntimeGateError",
    "RuntimeGateReceipt",
    "RuntimeResourceState",
    "RuntimeSnapshot",
    "SemanticCapabilities",
    "evaluate_runtime_gate",
    "reject",
]


SNAPSHOT_SCHEMA = "s42-runtime-snapshot-v1"
CONTRACT_SCHEMA = "s42-route-runtime-contract-v1"
FAILURE_MODES = {"fallback_before_dispatch"}


class RuntimeGateError(ValueError):
    pass


def _object(name: str, value: object) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise RuntimeGateError(f"{name} must be an object")
    return value


def _list(name: str, value: object, *, nonempty: bool = False) -> list[Any]:
    if type(value) is not list or (nonempty and not value):
        qualifier = "non-empty " if nonempty else ""
        raise RuntimeGateError(f"{name} must be a {qualifier}list")
    return value


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise RuntimeGateError(f"{name} must be a non-empty string")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimeGateError(f"{name} must be ASCII") from exc
    return value


def _integer(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RuntimeGateError(f"{name} must be an integer >= {minimum}")
    return value


def _boolean(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise RuntimeGateError(f"{name} must be bool")
    return value


def _optional_integer(name: str, value: object, minimum: int = 0) -> int | None:
    if value is None:
        return None
    return _integer(name, value, minimum)


def _unique_text_list(name: str, value: object, *, nonempty: bool = False) -> tuple[str, ...]:
    rows = tuple(_text(name, item) for item in _list(name, value, nonempty=nonempty))
    if len(rows) != len(set(rows)):
        raise RuntimeGateError(f"{name} must not contain duplicates")
    return rows


def _sha(name: str, value: object) -> str:
    result = _text(name, value)
    digest = result.removeprefix("sha256:")
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise RuntimeGateError(f"{name} must be a lowercase SHA-256")
    return "sha256:" + digest


@dataclass(frozen=True)
class RequestSemantics:
    cancelled: bool = False
    cancellation_required: bool = False
    kv_owner: str = "llama_context"
    kv_migration_required: bool = False
    context_shift_required: bool = False
    full_logits_required: bool = True
    grammar_required: bool = False
    sampler_location: str = "desktop"
    speculative_decode: bool = False

    def validate(self) -> None:
        _boolean("request cancelled", self.cancelled)
        _boolean("request cancellation_required", self.cancellation_required)
        _text("request kv_owner", self.kv_owner)
        _boolean("request kv_migration_required", self.kv_migration_required)
        _boolean("request context_shift_required", self.context_shift_required)
        _boolean("request full_logits_required", self.full_logits_required)
        _boolean("request grammar_required", self.grammar_required)
        _text("request sampler_location", self.sampler_location)
        _boolean("request speculative_decode", self.speculative_decode)


@dataclass(frozen=True)
class RuntimeResourceState:
    ready: bool
    generation: int
    heartbeat_age_us: int | None
    temperature_millic: int | None
    thermal_bucket: str
    contention_bucket: str
    slowdown_ppm: int
    failure_count: int
    circuit_open: bool
    reset_generation: int
    residency_ids: tuple[str, ...]

    @classmethod
    def from_json(cls, name: str, value: object) -> "RuntimeResourceState":
        row = _object(f"runtime resource {name}", value)
        return cls(
            ready=_boolean(f"runtime resource {name} ready", row.get("ready")),
            generation=_integer(
                f"runtime resource {name} generation", row.get("generation")
            ),
            heartbeat_age_us=_optional_integer(
                f"runtime resource {name} heartbeat_age_us",
                row.get("heartbeat_age_us"),
            ),
            temperature_millic=_optional_integer(
                f"runtime resource {name} temperature_millic",
                row.get("temperature_millic"),
            ),
            thermal_bucket=_text(
                f"runtime resource {name} thermal_bucket",
                row.get("thermal_bucket"),
            ),
            contention_bucket=_text(
                f"runtime resource {name} contention_bucket",
                row.get("contention_bucket"),
            ),
            slowdown_ppm=_integer(
                f"runtime resource {name} slowdown_ppm",
                row.get("slowdown_ppm"),
                1_000_000,
            ),
            failure_count=_integer(
                f"runtime resource {name} failure_count",
                row.get("failure_count"),
            ),
            circuit_open=_boolean(
                f"runtime resource {name} circuit_open",
                row.get("circuit_open"),
            ),
            reset_generation=_integer(
                f"runtime resource {name} reset_generation",
                row.get("reset_generation"),
            ),
            residency_ids=_unique_text_list(
                f"runtime resource {name} residency_ids",
                row.get("residency_ids", []),
            ),
        )


@dataclass(frozen=True)
class RuntimeSnapshot:
    snapshot_id: str
    generation: int
    epoch_key: str
    captured_at_us: int
    valid_until_us: int
    cancellation_generation: int
    resources: Mapping[str, RuntimeResourceState]

    @classmethod
    def from_json(cls, value: object) -> "RuntimeSnapshot":
        row = _object("runtime snapshot", value)
        if row.get("schema") != SNAPSHOT_SCHEMA:
            raise RuntimeGateError("runtime snapshot schema mismatch")
        raw_resources = _object("runtime snapshot resources", row.get("resources"))
        if not raw_resources:
            raise RuntimeGateError("runtime snapshot resources cannot be empty")
        resources = {
            _text("runtime resource id", name): RuntimeResourceState.from_json(
                name, state
            )
            for name, state in raw_resources.items()
        }
        captured = _integer(
            "runtime snapshot captured_at_us", row.get("captured_at_us")
        )
        valid_until = _integer(
            "runtime snapshot valid_until_us", row.get("valid_until_us"), 1
        )
        if valid_until <= captured:
            raise RuntimeGateError("runtime snapshot validity interval is empty")
        return cls(
            snapshot_id=_text("runtime snapshot id", row.get("snapshot_id")),
            generation=_integer(
                "runtime snapshot generation", row.get("generation"), 1
            ),
            epoch_key=_sha("runtime snapshot epoch_key", row.get("epoch_key")),
            captured_at_us=captured,
            valid_until_us=valid_until,
            cancellation_generation=_integer(
                "runtime snapshot cancellation_generation",
                row.get("cancellation_generation"),
            ),
            resources=resources,
        )


@dataclass(frozen=True)
class ResourceRequirement:
    heartbeat_max_age_us: int | None
    max_temperature_millic: int | None
    allowed_thermal_buckets: tuple[str, ...]
    allowed_contention_buckets: tuple[str, ...]
    max_slowdown_ppm: int
    max_failure_count: int
    reset_generation: int
    required_residency_ids: tuple[str, ...]

    @classmethod
    def from_json(cls, name: str, value: object) -> "ResourceRequirement":
        row = _object(f"resource requirement {name}", value)
        return cls(
            heartbeat_max_age_us=_optional_integer(
                f"resource requirement {name} heartbeat_max_age_us",
                row.get("heartbeat_max_age_us"),
            ),
            max_temperature_millic=_optional_integer(
                f"resource requirement {name} max_temperature_millic",
                row.get("max_temperature_millic"),
            ),
            allowed_thermal_buckets=_unique_text_list(
                f"resource requirement {name} allowed_thermal_buckets",
                row.get("allowed_thermal_buckets"),
                nonempty=True,
            ),
            allowed_contention_buckets=_unique_text_list(
                f"resource requirement {name} allowed_contention_buckets",
                row.get("allowed_contention_buckets"),
                nonempty=True,
            ),
            max_slowdown_ppm=_integer(
                f"resource requirement {name} max_slowdown_ppm",
                row.get("max_slowdown_ppm"),
                1_000_000,
            ),
            max_failure_count=_integer(
                f"resource requirement {name} max_failure_count",
                row.get("max_failure_count"),
            ),
            reset_generation=_integer(
                f"resource requirement {name} reset_generation",
                row.get("reset_generation"),
            ),
            required_residency_ids=_unique_text_list(
                f"resource requirement {name} required_residency_ids",
                row.get("required_residency_ids", []),
            ),
        )


@dataclass(frozen=True)
class SemanticCapabilities:
    kv_owner: str
    kv_migration: bool
    context_shift: bool
    full_logits: bool
    grammar: bool
    sampler_locations: tuple[str, ...]
    speculative_decode: bool
    cancellation_modes: tuple[str, ...]

    @classmethod
    def from_json(cls, value: object) -> "SemanticCapabilities":
        row = _object("semantic capabilities", value)
        return cls(
            kv_owner=_text("semantic kv_owner", row.get("kv_owner")),
            kv_migration=_boolean(
                "semantic kv_migration", row.get("kv_migration")
            ),
            context_shift=_boolean(
                "semantic context_shift", row.get("context_shift")
            ),
            full_logits=_boolean("semantic full_logits", row.get("full_logits")),
            grammar=_boolean("semantic grammar", row.get("grammar")),
            sampler_locations=_unique_text_list(
                "semantic sampler_locations",
                row.get("sampler_locations"),
                nonempty=True,
            ),
            speculative_decode=_boolean(
                "semantic speculative_decode", row.get("speculative_decode")
            ),
            cancellation_modes=_unique_text_list(
                "semantic cancellation_modes",
                row.get("cancellation_modes"),
                nonempty=True,
            ),
        )


@dataclass(frozen=True)
class RouteRuntimeContract:
    epoch_key: str
    failure_mode: str
    resources: Mapping[str, ResourceRequirement]
    semantics: SemanticCapabilities

    @classmethod
    def from_json(cls, value: object) -> "RouteRuntimeContract":
        row = _object("route runtime contract", value)
        if row.get("schema") != CONTRACT_SCHEMA:
            raise RuntimeGateError("route runtime contract schema mismatch")
        failure_mode = _text(
            "route runtime failure_mode", row.get("failure_mode")
        )
        if failure_mode not in FAILURE_MODES:
            raise RuntimeGateError("unsupported route failure mode")
        raw_resources = _object(
            "route runtime resource requirements", row.get("resources")
        )
        if not raw_resources:
            raise RuntimeGateError(
                "route runtime resource requirements cannot be empty"
            )
        resources = {
            _text("route runtime resource id", name): ResourceRequirement.from_json(
                name, requirement
            )
            for name, requirement in raw_resources.items()
        }
        return cls(
            epoch_key=_sha("route runtime epoch_key", row.get("epoch_key")),
            failure_mode=failure_mode,
            resources=resources,
            semantics=SemanticCapabilities.from_json(row.get("semantics")),
        )


@dataclass(frozen=True)
class RuntimeGateReceipt:
    admitted: bool
    reason: str
    snapshot_id: str | None
    snapshot_generation: int | None
    epoch_key: str | None
    checked_resources: tuple[str, ...]


def reject(
    reason: str,
    snapshot: RuntimeSnapshot | None,
    checked_resources: tuple[str, ...] = (),
) -> RuntimeGateReceipt:
    return RuntimeGateReceipt(
        admitted=False,
        reason=reason,
        snapshot_id=None if snapshot is None else snapshot.snapshot_id,
        snapshot_generation=None if snapshot is None else snapshot.generation,
        epoch_key=None if snapshot is None else snapshot.epoch_key,
        checked_resources=checked_resources,
    )


def evaluate_runtime_gate(
    contract: RouteRuntimeContract,
    request: RequestSemantics,
    snapshot: RuntimeSnapshot | None,
    now_us: int | None,
) -> RuntimeGateReceipt:
    request.validate()
    if request.cancelled:
        return reject("REQUEST_CANCELLED", snapshot)
    if snapshot is None:
        return reject("RUNTIME_SNAPSHOT_MISSING", None)
    if now_us is None:
        return reject("RUNTIME_TIME_MISSING", snapshot)
    _integer("runtime gate now_us", now_us)
    if now_us < snapshot.captured_at_us or now_us > snapshot.valid_until_us:
        return reject("RUNTIME_SNAPSHOT_EXPIRED", snapshot)
    if snapshot.epoch_key != contract.epoch_key:
        return reject("RUNTIME_EPOCH_MISMATCH", snapshot)

    checked: list[str] = []
    for resource_id in sorted(contract.resources):
        requirement = contract.resources[resource_id]
        state = snapshot.resources.get(resource_id)
        if state is None:
            return reject("RUNTIME_RESOURCE_MISSING", snapshot, tuple(checked))
        checked.append(resource_id)
        if not state.ready:
            return reject("RUNTIME_RESOURCE_NOT_READY", snapshot, tuple(checked))
        if state.circuit_open:
            return reject("RUNTIME_CIRCUIT_OPEN", snapshot, tuple(checked))
        if state.failure_count > requirement.max_failure_count:
            return reject("RUNTIME_FAILURE_LIMIT", snapshot, tuple(checked))
        if state.reset_generation != requirement.reset_generation:
            return reject("RUNTIME_RESET_GENERATION", snapshot, tuple(checked))
        if requirement.heartbeat_max_age_us is not None:
            if state.heartbeat_age_us is None:
                return reject("RUNTIME_HEARTBEAT_MISSING", snapshot, tuple(checked))
            if state.heartbeat_age_us > requirement.heartbeat_max_age_us:
                return reject("RUNTIME_HEARTBEAT_STALE", snapshot, tuple(checked))
        if state.thermal_bucket not in requirement.allowed_thermal_buckets:
            return reject("RUNTIME_THERMAL_BUCKET", snapshot, tuple(checked))
        if requirement.max_temperature_millic is not None:
            if state.temperature_millic is None:
                return reject("RUNTIME_TEMPERATURE_MISSING", snapshot, tuple(checked))
            if state.temperature_millic > requirement.max_temperature_millic:
                return reject("RUNTIME_THERMAL_LIMIT", snapshot, tuple(checked))
        if state.contention_bucket not in requirement.allowed_contention_buckets:
            return reject("RUNTIME_CONTENTION_BUCKET", snapshot, tuple(checked))
        if state.slowdown_ppm > requirement.max_slowdown_ppm:
            return reject("RUNTIME_CONTENTION_LIMIT", snapshot, tuple(checked))
        if not set(requirement.required_residency_ids).issubset(
            state.residency_ids
        ):
            return reject("RUNTIME_RESIDENCY_MISSING", snapshot, tuple(checked))

    semantics = contract.semantics
    if request.kv_owner != semantics.kv_owner:
        return reject("KV_OWNER_UNSUPPORTED", snapshot, tuple(checked))
    if request.kv_migration_required and not semantics.kv_migration:
        return reject("KV_MIGRATION_UNSUPPORTED", snapshot, tuple(checked))
    if request.context_shift_required and not semantics.context_shift:
        return reject("CONTEXT_SHIFT_UNSUPPORTED", snapshot, tuple(checked))
    if request.full_logits_required and not semantics.full_logits:
        return reject("FULL_LOGITS_UNSUPPORTED", snapshot, tuple(checked))
    if request.grammar_required and not semantics.grammar:
        return reject("GRAMMAR_UNSUPPORTED", snapshot, tuple(checked))
    if request.sampler_location not in semantics.sampler_locations:
        return reject("SAMPLER_LOCATION_UNSUPPORTED", snapshot, tuple(checked))
    if request.speculative_decode and not semantics.speculative_decode:
        return reject("SPECULATIVE_DECODE_UNSUPPORTED", snapshot, tuple(checked))
    if request.cancellation_required and "cooperative" not in semantics.cancellation_modes:
        return reject("CANCELLATION_UNSUPPORTED", snapshot, tuple(checked))

    return RuntimeGateReceipt(
        admitted=True,
        reason="RUNTIME_GATES_PASS",
        snapshot_id=snapshot.snapshot_id,
        snapshot_generation=snapshot.generation,
        epoch_key=snapshot.epoch_key,
        checked_resources=tuple(checked),
    )
