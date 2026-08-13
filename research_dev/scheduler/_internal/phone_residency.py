"""Multi-session phone residency and energy-positive arm decisions.

FastRPC HTP sessions provide separate address spaces and queues, but they do
not provide independent DSP compute engines. This module therefore models
separate per-session mapping limits and one shared execution resource.

Weights are loaded, verified, and warmed before a session becomes eligible.
An online arm signal selects an already-warm slice; it never performs weight
loading or repacking on the request path.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from .types import MetricEstimate


PHONE_RESIDENCY_SCHEMA = "research-scheduler-phone-residency-v1"
PHONE_OFFLOAD_EXECUTION_MODES = frozenset({
    "parallel_split",
    "full_replacement",
})
SESSION_STATES = frozenset({
    "UNLOADED",
    "LOADING",
    "HASHED",
    "WARM",
    "ARMED",
    "EXECUTING",
    "FAILED",
})

_SESSION_TRANSITIONS = {
    "UNLOADED": frozenset({"LOADING", "FAILED"}),
    "LOADING": frozenset({"HASHED", "FAILED"}),
    "HASHED": frozenset({"WARM", "FAILED"}),
    "WARM": frozenset({"ARMED", "FAILED", "UNLOADED"}),
    "ARMED": frozenset({"EXECUTING", "WARM", "FAILED"}),
    "EXECUTING": frozenset({"WARM", "FAILED"}),
    "FAILED": frozenset({"UNLOADED"}),
}

__all__ = [
    "PHONE_RESIDENCY_SCHEMA",
    "PHONE_OFFLOAD_EXECUTION_MODES",
    "SESSION_STATES",
    "PhoneArmGroup",
    "PhoneArmSignal",
    "PhoneOffloadCandidate",
    "PhoneOffloadDecision",
    "PhoneResidencyError",
    "PhoneResidencyPlan",
    "PhoneResidencySnapshot",
    "PhoneSessionPlan",
    "PhoneSessionReceipt",
    "ResidentPhoneSlice",
    "build_arm_group",
    "build_arm_signal",
    "select_energy_positive_offload",
    "validate_session_transition",
]


class PhoneResidencyError(ValueError):
    pass


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise PhoneResidencyError(f"{name} must be a non-empty string")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise PhoneResidencyError(f"{name} must be ASCII") from exc
    return value


def _integer(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PhoneResidencyError(
            f"{name} must be an integer >= {minimum}"
        )
    return value


def _sha256(name: str, value: object) -> str:
    digest = _text(name, value).removeprefix("sha256:")
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise PhoneResidencyError(f"{name} must be a lowercase SHA-256")
    return "sha256:" + digest


def _evidence(name: str, values: Sequence[str]) -> tuple[str, ...]:
    rows = tuple(_text(name, value) for value in values)
    if not rows or len(rows) != len(set(rows)):
        raise PhoneResidencyError(f"{name} must be non-empty and unique")
    return rows


@dataclass(frozen=True)
class ResidentPhoneSlice:
    slice_id: str
    model_id: str
    model_hash: str
    operator_family: str
    weight_hash: str
    resident_bytes: int
    physical_m_min: int
    physical_m_max: int
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _text("slice_id", self.slice_id)
        _text("slice model_id", self.model_id)
        _text("slice operator_family", self.operator_family)
        object.__setattr__(
            self, "model_hash", _sha256("slice model_hash", self.model_hash)
        )
        object.__setattr__(
            self, "weight_hash", _sha256("slice weight_hash", self.weight_hash)
        )
        _integer("slice resident_bytes", self.resident_bytes, 1)
        _integer("slice physical_m_min", self.physical_m_min, 1)
        _integer("slice physical_m_max", self.physical_m_max, 1)
        if self.physical_m_max < self.physical_m_min:
            raise PhoneResidencyError("slice physical M range is empty")
        object.__setattr__(
            self,
            "evidence_ids",
            _evidence("slice evidence_ids", self.evidence_ids),
        )

    def supports(self, physical_m: int) -> bool:
        _integer("physical_m", physical_m, 1)
        return self.physical_m_min <= physical_m <= self.physical_m_max

    @classmethod
    def from_json(cls, value: object) -> "ResidentPhoneSlice":
        if type(value) is not dict:
            raise PhoneResidencyError("resident phone slice must be an object")
        evidence = value.get("evidence_ids")
        if type(evidence) is not list:
            raise PhoneResidencyError("slice evidence_ids must be a list")
        return cls(
            slice_id=value.get("slice_id"),
            model_id=value.get("model_id"),
            model_hash=value.get("model_hash"),
            operator_family=value.get("operator_family"),
            weight_hash=value.get("weight_hash"),
            resident_bytes=value.get("resident_bytes"),
            physical_m_min=value.get("physical_m_min"),
            physical_m_max=value.get("physical_m_max"),
            evidence_ids=tuple(evidence),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "evidence_ids": list(self.evidence_ids),
            "model_hash": self.model_hash,
            "model_id": self.model_id,
            "operator_family": self.operator_family,
            "physical_m_max": self.physical_m_max,
            "physical_m_min": self.physical_m_min,
            "resident_bytes": self.resident_bytes,
            "slice_id": self.slice_id,
            "weight_hash": self.weight_hash,
        }


@dataclass(frozen=True)
class PhoneSessionPlan:
    session_id: str
    compute_backend: str
    mapping_limit_bytes: int
    slices: tuple[ResidentPhoneSlice, ...]

    def __post_init__(self) -> None:
        _text("phone session_id", self.session_id)
        _text("phone compute_backend", self.compute_backend)
        _integer("phone mapping_limit_bytes", self.mapping_limit_bytes, 1)
        slices = tuple(self.slices)
        if (
            not slices
            or any(not isinstance(row, ResidentPhoneSlice) for row in slices)
        ):
            raise PhoneResidencyError("phone session slices are invalid")
        slice_ids = [row.slice_id for row in slices]
        if len(slice_ids) != len(set(slice_ids)):
            raise PhoneResidencyError("phone session has duplicate slice ids")
        if sum(row.resident_bytes for row in slices) > self.mapping_limit_bytes:
            raise PhoneResidencyError("phone session exceeds its mapping limit")
        object.__setattr__(self, "slices", slices)

    @property
    def resident_bytes(self) -> int:
        return sum(row.resident_bytes for row in self.slices)

    @classmethod
    def from_json(cls, value: object) -> "PhoneSessionPlan":
        if type(value) is not dict:
            raise PhoneResidencyError("phone session plan must be an object")
        slices = value.get("slices")
        if type(slices) is not list:
            raise PhoneResidencyError("phone session slices must be a list")
        return cls(
            session_id=value.get("session_id"),
            compute_backend=value.get("compute_backend"),
            mapping_limit_bytes=value.get("mapping_limit_bytes"),
            slices=tuple(ResidentPhoneSlice.from_json(row) for row in slices),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "compute_backend": self.compute_backend,
            "mapping_limit_bytes": self.mapping_limit_bytes,
            "session_id": self.session_id,
            "slices": [row.to_json() for row in self.slices],
        }


@dataclass(frozen=True)
class PhoneResidencyPlan:
    plan_id: str
    phone_serial: str
    memory_resource_id: str
    shared_compute_resource_id: str
    transport_resource_ids: tuple[str, ...]
    memory_capacity_bytes: int
    minimum_available_bytes: int
    reset_generation: int
    sessions: tuple[PhoneSessionPlan, ...]

    def __post_init__(self) -> None:
        for name in (
            "plan_id",
            "phone_serial",
            "memory_resource_id",
            "shared_compute_resource_id",
        ):
            _text(name, getattr(self, name))
        transport_resource_ids = tuple(
            _text("phone transport_resource_id", resource_id)
            for resource_id in self.transport_resource_ids
        )
        if (
            not transport_resource_ids
            or len(transport_resource_ids) != len(set(transport_resource_ids))
            or self.memory_resource_id == self.shared_compute_resource_id
            or self.shared_compute_resource_id in transport_resource_ids
            or self.memory_resource_id in transport_resource_ids
        ):
            raise PhoneResidencyError("phone transport resources are invalid")
        object.__setattr__(
            self, "transport_resource_ids", transport_resource_ids
        )
        _integer("phone memory_capacity_bytes", self.memory_capacity_bytes, 1)
        _integer("phone minimum_available_bytes", self.minimum_available_bytes, 1)
        _integer("phone reset_generation", self.reset_generation)
        if self.minimum_available_bytes >= self.memory_capacity_bytes:
            raise PhoneResidencyError("phone memory reserve consumes capacity")
        sessions = tuple(self.sessions)
        if (
            not sessions
            or any(not isinstance(row, PhoneSessionPlan) for row in sessions)
        ):
            raise PhoneResidencyError("phone residency sessions are invalid")
        session_ids = [row.session_id for row in sessions]
        backends = [row.compute_backend for row in sessions]
        slice_ids = [row.slice_id for session in sessions for row in session.slices]
        if len(session_ids) != len(set(session_ids)):
            raise PhoneResidencyError("phone session ids must be unique")
        if len(backends) != len(set(backends)):
            raise PhoneResidencyError("phone compute backends must be unique")
        if len(slice_ids) != len(set(slice_ids)):
            raise PhoneResidencyError("phone slice ids must be globally unique")
        if self.resident_bytes + self.minimum_available_bytes > self.memory_capacity_bytes:
            raise PhoneResidencyError("phone resident set violates memory reserve")
        object.__setattr__(self, "sessions", sessions)

    @property
    def resident_bytes(self) -> int:
        return sum(row.resident_bytes for row in self.sessions)

    @property
    def execution_resource_ids(self) -> tuple[str, ...]:
        return (
            self.shared_compute_resource_id,
            *self.transport_resource_ids,
        )

    def slice_location(
        self, slice_id: str
    ) -> tuple[PhoneSessionPlan, ResidentPhoneSlice]:
        _text("slice_id", slice_id)
        for session in self.sessions:
            for row in session.slices:
                if row.slice_id == slice_id:
                    return session, row
        raise PhoneResidencyError(f"unknown resident phone slice: {slice_id}")

    @classmethod
    def from_json(cls, value: object) -> "PhoneResidencyPlan":
        if type(value) is not dict:
            raise PhoneResidencyError("phone residency plan must be an object")
        if value.get("schema") != PHONE_RESIDENCY_SCHEMA:
            raise PhoneResidencyError("phone residency plan schema mismatch")
        sessions = value.get("sessions")
        if type(sessions) is not list:
            raise PhoneResidencyError("phone residency sessions must be a list")
        transport_resources = value.get("transport_resource_ids")
        if type(transport_resources) is not list:
            raise PhoneResidencyError(
                "phone transport_resource_ids must be a list"
            )
        return cls(
            plan_id=value.get("plan_id"),
            phone_serial=value.get("phone_serial"),
            memory_resource_id=value.get("memory_resource_id"),
            shared_compute_resource_id=value.get("shared_compute_resource_id"),
            transport_resource_ids=tuple(transport_resources),
            memory_capacity_bytes=value.get("memory_capacity_bytes"),
            minimum_available_bytes=value.get("minimum_available_bytes"),
            reset_generation=value.get("reset_generation"),
            sessions=tuple(PhoneSessionPlan.from_json(row) for row in sessions),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "memory_capacity_bytes": self.memory_capacity_bytes,
            "memory_resource_id": self.memory_resource_id,
            "minimum_available_bytes": self.minimum_available_bytes,
            "phone_serial": self.phone_serial,
            "plan_id": self.plan_id,
            "reset_generation": self.reset_generation,
            "schema": PHONE_RESIDENCY_SCHEMA,
            "sessions": [row.to_json() for row in self.sessions],
            "shared_compute_resource_id": self.shared_compute_resource_id,
            "transport_resource_ids": list(self.transport_resource_ids),
        }


@dataclass(frozen=True)
class PhoneSessionReceipt:
    session_id: str
    compute_backend: str
    state: str
    generation: int
    reset_generation: int
    worker_hash: str
    allocated_bytes: int
    slice_weight_hashes: Mapping[str, str]
    last_transition_us: int

    def __post_init__(self) -> None:
        _text("receipt session_id", self.session_id)
        _text("receipt compute_backend", self.compute_backend)
        if self.state not in SESSION_STATES:
            raise PhoneResidencyError("unknown phone session state")
        _integer("receipt generation", self.generation, 1)
        _integer("receipt reset_generation", self.reset_generation)
        object.__setattr__(
            self, "worker_hash", _sha256("receipt worker_hash", self.worker_hash)
        )
        _integer("receipt allocated_bytes", self.allocated_bytes)
        _integer("receipt last_transition_us", self.last_transition_us)
        hashes = {
            _text("receipt slice_id", key): _sha256(
                "receipt slice weight_hash", value
            )
            for key, value in self.slice_weight_hashes.items()
        }
        object.__setattr__(
            self, "slice_weight_hashes", MappingProxyType(dict(sorted(hashes.items())))
        )

    def validate_against(
        self,
        plan: PhoneSessionPlan,
        reset_generation: int,
    ) -> None:
        if self.session_id != plan.session_id:
            raise PhoneResidencyError("phone session receipt id mismatch")
        if self.compute_backend != plan.compute_backend:
            raise PhoneResidencyError("phone session receipt backend mismatch")
        if self.reset_generation != reset_generation:
            raise PhoneResidencyError("phone session reset generation mismatch")
        if self.allocated_bytes > plan.mapping_limit_bytes:
            raise PhoneResidencyError("phone session receipt exceeds mapping limit")
        if self.state in {"HASHED", "WARM", "ARMED", "EXECUTING"}:
            expected = {row.slice_id: row.weight_hash for row in plan.slices}
            if dict(self.slice_weight_hashes) != expected:
                raise PhoneResidencyError("phone session slice identity mismatch")
            if self.allocated_bytes < plan.resident_bytes:
                raise PhoneResidencyError("phone session allocation is incomplete")


@dataclass(frozen=True)
class PhoneResidencySnapshot:
    snapshot_id: str
    plan_id: str
    captured_at_us: int
    mem_available_bytes: int
    sessions: Mapping[str, PhoneSessionReceipt]

    def __post_init__(self) -> None:
        _text("phone snapshot_id", self.snapshot_id)
        _text("phone snapshot plan_id", self.plan_id)
        _integer("phone snapshot captured_at_us", self.captured_at_us)
        _integer("phone snapshot mem_available_bytes", self.mem_available_bytes)
        sessions = dict(self.sessions)
        if (
            not sessions
            or any(
                not isinstance(receipt, PhoneSessionReceipt)
                or key != receipt.session_id
                for key, receipt in sessions.items()
            )
        ):
            raise PhoneResidencyError("phone snapshot sessions are invalid")
        object.__setattr__(
            self, "sessions", MappingProxyType(dict(sorted(sessions.items())))
        )

    def validate_against(self, plan: PhoneResidencyPlan) -> None:
        if self.plan_id != plan.plan_id:
            raise PhoneResidencyError("phone snapshot plan mismatch")
        if self.mem_available_bytes < plan.minimum_available_bytes:
            raise PhoneResidencyError("phone runtime memory reserve is unavailable")
        expected_ids = {row.session_id for row in plan.sessions}
        if set(self.sessions) != expected_ids:
            raise PhoneResidencyError("phone snapshot session set mismatch")
        for session in plan.sessions:
            self.sessions[session.session_id].validate_against(
                session, plan.reset_generation
            )

    def to_json(self) -> dict[str, object]:
        return {
            "captured_at_us": self.captured_at_us,
            "mem_available_bytes": self.mem_available_bytes,
            "plan_id": self.plan_id,
            "schema": PHONE_RESIDENCY_SCHEMA,
            "sessions": {
                session_id: {
                    "allocated_bytes": receipt.allocated_bytes,
                    "compute_backend": receipt.compute_backend,
                    "generation": receipt.generation,
                    "last_transition_us": receipt.last_transition_us,
                    "reset_generation": receipt.reset_generation,
                    "session_id": receipt.session_id,
                    "slice_weight_hashes": dict(
                        receipt.slice_weight_hashes
                    ),
                    "state": receipt.state,
                    "worker_hash": receipt.worker_hash,
                }
                for session_id, receipt in self.sessions.items()
            },
            "snapshot_id": self.snapshot_id,
        }


def validate_session_transition(source: str, target: str) -> None:
    if source not in SESSION_STATES or target not in SESSION_STATES:
        raise PhoneResidencyError("unknown phone session state")
    if target not in _SESSION_TRANSITIONS[source]:
        raise PhoneResidencyError(
            f"invalid phone session transition: {source} -> {target}"
        )


@dataclass(frozen=True)
class PhoneArmSignal:
    request_id: str
    route_id: str
    session_id: str
    compute_backend: str
    slice_id: str
    model_hash: str
    weight_hash: str
    session_generation: int
    reset_generation: int
    physical_m: int
    armed_at_us: int
    execute_not_before_us: int
    deadline_us: int
    shared_compute_resource_id: str

    def __post_init__(self) -> None:
        for name in (
            "request_id",
            "route_id",
            "session_id",
            "compute_backend",
            "slice_id",
            "shared_compute_resource_id",
        ):
            _text(name, getattr(self, name))
        object.__setattr__(
            self, "model_hash", _sha256("arm model_hash", self.model_hash)
        )
        object.__setattr__(
            self, "weight_hash", _sha256("arm weight_hash", self.weight_hash)
        )
        _integer("arm session_generation", self.session_generation, 1)
        _integer("arm reset_generation", self.reset_generation)
        _integer("arm physical_m", self.physical_m, 1)
        _integer("arm armed_at_us", self.armed_at_us)
        _integer("arm execute_not_before_us", self.execute_not_before_us)
        _integer("arm deadline_us", self.deadline_us, 1)
        if self.execute_not_before_us < self.armed_at_us:
            raise PhoneResidencyError("phone execution precedes its arm signal")
        if self.deadline_us <= self.execute_not_before_us:
            raise PhoneResidencyError("phone arm deadline is empty")

    def to_json(self) -> dict[str, object]:
        return {
            "armed_at_us": self.armed_at_us,
            "compute_backend": self.compute_backend,
            "deadline_us": self.deadline_us,
            "execute_not_before_us": self.execute_not_before_us,
            "model_hash": self.model_hash,
            "physical_m": self.physical_m,
            "request_id": self.request_id,
            "reset_generation": self.reset_generation,
            "route_id": self.route_id,
            "session_generation": self.session_generation,
            "session_id": self.session_id,
            "shared_compute_resource_id": self.shared_compute_resource_id,
            "slice_id": self.slice_id,
            "weight_hash": self.weight_hash,
        }


@dataclass(frozen=True)
class PhoneArmGroup:
    signals: tuple[PhoneArmSignal, ...]

    def __post_init__(self) -> None:
        signals = tuple(self.signals)
        if len(signals) < 2 or any(
            not isinstance(signal, PhoneArmSignal) for signal in signals
        ):
            raise PhoneResidencyError(
                "phone arm group requires at least two signals"
            )
        first = signals[0]
        shared_identity = (
            first.request_id,
            first.route_id,
            first.model_hash,
            first.reset_generation,
            first.physical_m,
            first.armed_at_us,
            first.execute_not_before_us,
            first.deadline_us,
            first.shared_compute_resource_id,
        )
        if any(
            (
                signal.request_id,
                signal.route_id,
                signal.model_hash,
                signal.reset_generation,
                signal.physical_m,
                signal.armed_at_us,
                signal.execute_not_before_us,
                signal.deadline_us,
                signal.shared_compute_resource_id,
            ) != shared_identity
            for signal in signals[1:]
        ):
            raise PhoneResidencyError("phone arm group identity mismatch")
        session_ids = [signal.session_id for signal in signals]
        slice_ids = [signal.slice_id for signal in signals]
        if (
            len(session_ids) != len(set(session_ids))
            or len(slice_ids) != len(set(slice_ids))
        ):
            raise PhoneResidencyError(
                "phone arm group requires distinct sessions and slices"
            )
        object.__setattr__(self, "signals", signals)

    def to_json(self) -> dict[str, object]:
        return {
            "kind": "composite_phone_arm",
            "signals": [signal.to_json() for signal in self.signals],
        }


def build_arm_signal(
    plan: PhoneResidencyPlan,
    snapshot: PhoneResidencySnapshot,
    *,
    request_id: str,
    route_id: str,
    slice_id: str,
    physical_m: int,
    armed_at_us: int,
    execute_not_before_us: int,
    deadline_us: int,
) -> PhoneArmSignal:
    snapshot.validate_against(plan)
    session, resident_slice = plan.slice_location(slice_id)
    if not resident_slice.supports(physical_m):
        raise PhoneResidencyError("resident slice does not support physical M")
    receipt = snapshot.sessions[session.session_id]
    if receipt.state != "WARM":
        raise PhoneResidencyError("phone session is not warm and idle")
    return PhoneArmSignal(
        request_id=request_id,
        route_id=route_id,
        session_id=session.session_id,
        compute_backend=session.compute_backend,
        slice_id=resident_slice.slice_id,
        model_hash=resident_slice.model_hash,
        weight_hash=resident_slice.weight_hash,
        session_generation=receipt.generation,
        reset_generation=plan.reset_generation,
        physical_m=physical_m,
        armed_at_us=armed_at_us,
        execute_not_before_us=execute_not_before_us,
        deadline_us=deadline_us,
        shared_compute_resource_id=plan.shared_compute_resource_id,
    )


def build_arm_group(
    plan: PhoneResidencyPlan,
    snapshot: PhoneResidencySnapshot,
    *,
    request_id: str,
    route_id: str,
    slice_ids: Sequence[str],
    physical_m: int,
    armed_at_us: int,
    execute_not_before_us: int,
    deadline_us: int,
) -> PhoneArmSignal | PhoneArmGroup:
    identifiers = tuple(_text("arm slice_id", value) for value in slice_ids)
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise PhoneResidencyError("phone arm slice ids must be non-empty and unique")
    signals = tuple(
        build_arm_signal(
            plan,
            snapshot,
            request_id=request_id,
            route_id=route_id,
            slice_id=slice_id,
            physical_m=physical_m,
            armed_at_us=armed_at_us,
            execute_not_before_us=execute_not_before_us,
            deadline_us=deadline_us,
        )
        for slice_id in identifiers
    )
    return signals[0] if len(signals) == 1 else PhoneArmGroup(signals)


@dataclass(frozen=True)
class PhoneOffloadCandidate:
    candidate_id: str
    slice_id: str
    offload_units: int
    energy_boundary_id: str
    accounting_scope: str
    baseline_latency_us: MetricEstimate
    host_remainder_us: MetricEstimate
    phone_path_us: MetricEstimate
    baseline_energy_uj: MetricEstimate
    split_energy_uj: MetricEstimate
    evidence_ids: tuple[str, ...]
    additional_slice_ids: tuple[str, ...] = ()
    execution_mode: str = "parallel_split"

    def __post_init__(self) -> None:
        for name in (
            "candidate_id",
            "slice_id",
            "energy_boundary_id",
            "accounting_scope",
            "execution_mode",
        ):
            _text(name, getattr(self, name))
        if self.execution_mode not in PHONE_OFFLOAD_EXECUTION_MODES:
            raise PhoneResidencyError("unknown phone offload execution mode")
        _integer("phone offload_units", self.offload_units, 1)
        for name in (
            "baseline_latency_us",
            "host_remainder_us",
            "phone_path_us",
            "baseline_energy_uj",
            "split_energy_uj",
        ):
            if not isinstance(getattr(self, name), MetricEstimate):
                raise PhoneResidencyError(f"{name} must be MetricEstimate")
        object.__setattr__(
            self,
            "evidence_ids",
            _evidence("phone candidate evidence_ids", self.evidence_ids),
        )
        additional_slice_ids = tuple(
            _text("phone candidate additional_slice_id", slice_id)
            for slice_id in self.additional_slice_ids
        )
        if (
            self.slice_id in additional_slice_ids
            or len(additional_slice_ids) != len(set(additional_slice_ids))
        ):
            raise PhoneResidencyError(
                "phone candidate slice ids must be unique"
            )
        object.__setattr__(
            self, "additional_slice_ids", additional_slice_ids
        )

    @property
    def slice_ids(self) -> tuple[str, ...]:
        return (self.slice_id, *self.additional_slice_ids)


@dataclass(frozen=True)
class PhoneOffloadDecision:
    candidate_id: str | None
    reason: str
    arm_signal: PhoneArmSignal | PhoneArmGroup | None
    offload_units: int
    split_latency_upper_us: int | None
    exposed_join_wait_upper_us: int | None
    energy_saving_ppm: int | None
    rejected: tuple[tuple[str, str], ...]


def select_energy_positive_offload(
    plan: PhoneResidencyPlan,
    snapshot: PhoneResidencySnapshot,
    candidates: Sequence[PhoneOffloadCandidate],
    *,
    request_id: str,
    route_id: str,
    physical_m: int,
    now_us: int,
    phone_resource_ready_us: int | Mapping[str, int | None],
    deadline_us: int,
    minimum_energy_saving_ppm: int = 50_000,
    latency_limit_ppm: int = 1_000_000,
    maximum_join_wait_ppm: int = 50_000,
    require_measured: bool = True,
) -> PhoneOffloadDecision:
    _integer("phone decision physical_m", physical_m, 1)
    _integer("phone decision now_us", now_us)
    ready_by_candidate: Mapping[str, int | None] | None = None
    if isinstance(phone_resource_ready_us, Mapping):
        ready_by_candidate = phone_resource_ready_us
        for candidate_id, ready_us in ready_by_candidate.items():
            _text("phone ready candidate_id", candidate_id)
            if ready_us is not None:
                _integer("phone candidate resource_ready_us", ready_us)
    else:
        _integer("phone resource_ready_us", phone_resource_ready_us)
    _integer("phone decision deadline_us", deadline_us, 1)
    _integer("minimum_energy_saving_ppm", minimum_energy_saving_ppm)
    _integer("latency_limit_ppm", latency_limit_ppm, 1)
    _integer("maximum_join_wait_ppm", maximum_join_wait_ppm)
    if minimum_energy_saving_ppm >= 1_000_000:
        raise PhoneResidencyError("energy saving gate reaches 100 percent")
    if latency_limit_ppm < 1_000_000:
        raise PhoneResidencyError("latency limit is below baseline")
    if maximum_join_wait_ppm > 1_000_000:
        raise PhoneResidencyError("join wait gate exceeds 100 percent")
    if deadline_us <= now_us:
        raise PhoneResidencyError("phone decision deadline has passed")

    snapshot.validate_against(plan)
    rejected: list[tuple[str, str]] = []
    feasible: list[tuple[PhoneOffloadCandidate, int, int, int, int]] = []
    seen_ids: set[str] = set()
    boundary_scope: tuple[str, str] | None = None

    for candidate in candidates:
        if not isinstance(candidate, PhoneOffloadCandidate):
            raise PhoneResidencyError("phone offload candidate is invalid")
        if candidate.candidate_id in seen_ids:
            raise PhoneResidencyError("duplicate phone candidate id")
        seen_ids.add(candidate.candidate_id)
        current_boundary = (
            candidate.energy_boundary_id,
            candidate.accounting_scope,
        )
        if boundary_scope is None:
            boundary_scope = current_boundary
        elif current_boundary != boundary_scope:
            raise PhoneResidencyError(
                "phone candidates use different energy boundaries or scopes"
            )

        try:
            locations = tuple(
                plan.slice_location(slice_id)
                for slice_id in candidate.slice_ids
            )
        except PhoneResidencyError:
            rejected.append((candidate.candidate_id, "RESIDENCY_UNKNOWN"))
            continue
        sessions = [session for session, _ in locations]
        resident_slices = [resident_slice for _, resident_slice in locations]
        if len({session.session_id for session in sessions}) != len(sessions):
            rejected.append((candidate.candidate_id, "COMPOSITE_SESSION_CONFLICT"))
            continue
        if len({row.model_hash for row in resident_slices}) != 1:
            rejected.append((candidate.candidate_id, "COMPOSITE_MODEL_MISMATCH"))
            continue
        if any(
            snapshot.sessions[session.session_id].state != "WARM"
            for session in sessions
        ):
            rejected.append((candidate.candidate_id, "SESSION_NOT_WARM"))
            continue
        if any(not row.supports(physical_m) for row in resident_slices):
            rejected.append((candidate.candidate_id, "SHAPE_UNSUPPORTED"))
            continue
        candidate_ready_us = (
            ready_by_candidate.get(candidate.candidate_id)
            if ready_by_candidate is not None
            else phone_resource_ready_us
        )
        if candidate_ready_us is None:
            rejected.append((candidate.candidate_id, "RESOURCE_NOT_READY"))
            continue
        execute_not_before_us = max(now_us, candidate_ready_us)
        queue_delay_us = execute_not_before_us - now_us

        metrics = (
            candidate.baseline_latency_us,
            candidate.host_remainder_us,
            candidate.phone_path_us,
            candidate.baseline_energy_uj,
            candidate.split_energy_uj,
        )
        if require_measured and not all(row.measured for row in metrics):
            rejected.append((candidate.candidate_id, "MEASUREMENT_REQUIRED"))
            continue
        if candidate.baseline_energy_uj.lower is None:
            rejected.append((candidate.candidate_id, "BASELINE_ENERGY_LCB_MISSING"))
            continue

        phone_upper_us = queue_delay_us + candidate.phone_path_us.upper
        if candidate.execution_mode == "full_replacement":
            split_upper_us = phone_upper_us
            join_wait_upper_us = 0
        else:
            split_upper_us = max(
                candidate.host_remainder_us.upper,
                phone_upper_us,
            )
            host_lower_us = (
                candidate.host_remainder_us.lower
                if candidate.host_remainder_us.lower is not None
                else candidate.host_remainder_us.mean
            )
            join_wait_upper_us = max(0, phone_upper_us - host_lower_us)
        join_wait_ppm = (
            join_wait_upper_us * 1_000_000 + split_upper_us - 1
        ) // split_upper_us
        if now_us + split_upper_us > deadline_us:
            rejected.append((candidate.candidate_id, "DEADLINE"))
            continue
        if (
            split_upper_us * 1_000_000
            > candidate.baseline_latency_us.upper * latency_limit_ppm
        ):
            rejected.append((candidate.candidate_id, "LATENCY_REGRESSION"))
            continue
        if join_wait_ppm > maximum_join_wait_ppm:
            rejected.append((candidate.candidate_id, "PHONE_EXPOSED"))
            continue
        energy_threshold = (
            candidate.baseline_energy_uj.lower
            * (1_000_000 - minimum_energy_saving_ppm)
        ) // 1_000_000
        if candidate.split_energy_uj.upper > energy_threshold:
            rejected.append((candidate.candidate_id, "ENERGY_MARGIN"))
            continue
        energy_saving_ppm = (
            (candidate.baseline_energy_uj.lower - candidate.split_energy_uj.upper)
            * 1_000_000
        ) // candidate.baseline_energy_uj.lower
        feasible.append(
            (
                candidate,
                split_upper_us,
                join_wait_upper_us,
                energy_saving_ppm,
                execute_not_before_us,
            )
        )

    if not feasible:
        return PhoneOffloadDecision(
            candidate_id=None,
            reason="BASELINE_NO_ENERGY_POSITIVE_PHONE_ROUTE",
            arm_signal=None,
            offload_units=0,
            split_latency_upper_us=None,
            exposed_join_wait_upper_us=None,
            energy_saving_ppm=None,
            rejected=tuple(rejected),
        )

    (
        candidate,
        split_upper_us,
        join_wait_upper_us,
        energy_saving_ppm,
        execute_not_before_us,
    ) = min(
        feasible,
        key=lambda row: (
            row[0].split_energy_uj.upper,
            row[1],
            -row[0].offload_units,
            row[0].candidate_id,
        ),
    )
    signal = build_arm_group(
        plan,
        snapshot,
        request_id=request_id,
        route_id=route_id,
        slice_ids=candidate.slice_ids,
        physical_m=physical_m,
        armed_at_us=now_us,
        execute_not_before_us=execute_not_before_us,
        deadline_us=deadline_us,
    )
    return PhoneOffloadDecision(
        candidate_id=candidate.candidate_id,
        reason="ENERGY_POSITIVE_RESIDENT_PHONE_ROUTE",
        arm_signal=signal,
        offload_units=candidate.offload_units,
        split_latency_upper_us=split_upper_us,
        exposed_join_wait_upper_us=join_wait_upper_us,
        energy_saving_ppm=energy_saving_ppm,
        rejected=tuple(rejected),
    )
