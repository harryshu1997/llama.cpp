"""Conservative dynamic weight residency and atomic epoch transitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .capacity import DeviceMemoryCapacity
from .types import MetricEstimate, canonical_sha256


DYNAMIC_RESIDENCY_SCHEMA = "research-scheduler-dynamic-residency-v1"
DYNAMIC_RESIDENCY_RECEIPT_SCHEMA = (
    "research-scheduler-dynamic-residency-receipt-v1"
)
DYNAMIC_RESIDENCY_ACTIONS = frozenset({
    "FALLBACK_ACQUIRE",
    "FALLBACK_RELEASE",
    "PREFETCH",
    "VERIFY",
    "DRAIN",
    "EVICT",
    "PUBLISH",
})
DYNAMIC_RESIDENCY_RECEIPT_STATUSES = frozenset({"READY", "FAILED"})
DYNAMIC_RESIDENCY_TRANSITION_MODES = frozenset({
    "ATOMIC_STAGE_BEFORE_EVICT",
    "DRAIN_OR_EVICT_BEFORE_STAGE_WITH_READY_FALLBACK",
})
DYNAMIC_FALLBACK_CONTRACT_SCHEMA = (
    "research-scheduler-dynamic-fallback-contract-v1"
)
DYNAMIC_FALLBACK_RECEIPT_SCHEMA = (
    "research-scheduler-dynamic-fallback-receipt-v1"
)

__all__ = [
    "DYNAMIC_RESIDENCY_ACTIONS",
    "DYNAMIC_RESIDENCY_RECEIPT_SCHEMA",
    "DYNAMIC_RESIDENCY_RECEIPT_STATUSES",
    "DYNAMIC_RESIDENCY_SCHEMA",
    "DYNAMIC_RESIDENCY_TRANSITION_MODES",
    "DYNAMIC_FALLBACK_CONTRACT_SCHEMA",
    "DYNAMIC_FALLBACK_RECEIPT_SCHEMA",
    "DynamicFallbackServiceContract",
    "DynamicFallbackServiceReceipt",
    "DynamicResidencyCandidate",
    "DynamicResidencyDecision",
    "DynamicResidencyError",
    "DynamicResidencyReceipt",
    "DynamicResidencySnapshot",
    "DynamicWeightPlacement",
    "DynamicWeightPlacementSpec",
    "ResidencyTransitionAction",
    "apply_dynamic_residency_receipt",
    "select_dynamic_residency_transition",
]


class DynamicResidencyError(ValueError):
    pass


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise DynamicResidencyError(f"{name} must be a non-empty string")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise DynamicResidencyError(f"{name} must be ASCII") from exc
    return value


def _integer(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise DynamicResidencyError(
            f"{name} must be an integer >= {minimum}"
        )
    return value


def _sha256(name: str, value: object) -> str:
    digest = _text(name, value).removeprefix("sha256:")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise DynamicResidencyError(f"{name} must be a lowercase SHA-256")
    return "sha256:" + digest


def _object(name: str, value: object) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise DynamicResidencyError(f"{name} must be an object")
    return value


def _unique_text(
    name: str,
    values: Sequence[str],
    *,
    nonempty: bool = False,
) -> tuple[str, ...]:
    rows = tuple(_text(name, value) for value in values)
    if (nonempty and not rows) or len(rows) != len(set(rows)):
        qualifier = "non-empty and " if nonempty else ""
        raise DynamicResidencyError(
            f"{name} must be {qualifier}unique"
        )
    return rows


def _evidence(values: Sequence[str]) -> tuple[str, ...]:
    return _unique_text("dynamic residency evidence id", values, nonempty=True)


def _metric(name: str, value: object) -> MetricEstimate:
    row = _object(name, value)
    try:
        return MetricEstimate(
            mean=row.get("mean"),
            upper=row.get("upper"),
            lower=row.get("lower"),
            sample_count=row.get("sample_count"),
            measured=row.get("measured"),
        )
    except (TypeError, ValueError) as exc:
        raise DynamicResidencyError(f"{name} is invalid") from exc


def _metric_json(value: MetricEstimate) -> dict[str, object]:
    return {
        "lower": value.lower,
        "mean": value.mean,
        "measured": value.measured,
        "sample_count": value.sample_count,
        "upper": value.upper,
    }


@dataclass(frozen=True)
class DynamicWeightPlacementSpec:
    placement_id: str
    slice_id: str
    model_id: str
    model_hash: str
    weight_hash: str
    resource_id: str
    resident_bytes: int
    execution_resource_ids: tuple[str, ...]
    runtime_binding_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("placement_id", "slice_id", "model_id", "resource_id"):
            _text(f"dynamic placement {name}", getattr(self, name))
        object.__setattr__(
            self,
            "model_hash",
            _sha256("dynamic placement model_hash", self.model_hash),
        )
        object.__setattr__(
            self,
            "weight_hash",
            _sha256("dynamic placement weight_hash", self.weight_hash),
        )
        _integer("dynamic placement resident_bytes", self.resident_bytes, 1)
        object.__setattr__(
            self,
            "execution_resource_ids",
            _unique_text(
                "dynamic placement execution resource",
                self.execution_resource_ids,
                nonempty=True,
            ),
        )
        object.__setattr__(
            self,
            "runtime_binding_ids",
            _unique_text(
                "dynamic placement runtime binding",
                self.runtime_binding_ids,
                nonempty=True,
            ),
        )
        object.__setattr__(self, "evidence_ids", _evidence(self.evidence_ids))

    @classmethod
    def from_json(cls, value: object) -> "DynamicWeightPlacementSpec":
        row = _object("dynamic placement spec", value)
        evidence = row.get("evidence_ids")
        execution_resources = row.get("execution_resource_ids")
        runtime_bindings = row.get("runtime_binding_ids")
        if not all(
            type(value) is list
            for value in (evidence, execution_resources, runtime_bindings)
        ):
            raise DynamicResidencyError(
                "dynamic placement collections must be lists"
            )
        return cls(
            placement_id=row.get("placement_id"),
            slice_id=row.get("slice_id"),
            model_id=row.get("model_id"),
            model_hash=row.get("model_hash"),
            weight_hash=row.get("weight_hash"),
            resource_id=row.get("resource_id"),
            resident_bytes=row.get("resident_bytes"),
            execution_resource_ids=tuple(execution_resources),
            runtime_binding_ids=tuple(runtime_bindings),
            evidence_ids=tuple(evidence),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "evidence_ids": list(self.evidence_ids),
            "execution_resource_ids": list(self.execution_resource_ids),
            "model_hash": self.model_hash,
            "model_id": self.model_id,
            "placement_id": self.placement_id,
            "resident_bytes": self.resident_bytes,
            "resource_id": self.resource_id,
            "runtime_binding_ids": list(self.runtime_binding_ids),
            "slice_id": self.slice_id,
            "weight_hash": self.weight_hash,
        }


@dataclass(frozen=True)
class DynamicWeightPlacement:
    spec: DynamicWeightPlacementSpec
    generation: int
    resident_since_us: int
    minimum_resident_until_us: int
    active_leases: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.spec, DynamicWeightPlacementSpec):
            raise DynamicResidencyError("dynamic placement spec is invalid")
        _integer("dynamic placement generation", self.generation, 1)
        _integer("dynamic placement resident_since_us", self.resident_since_us)
        _integer(
            "dynamic placement minimum_resident_until_us",
            self.minimum_resident_until_us,
        )
        _integer("dynamic placement active_leases", self.active_leases)
        if self.minimum_resident_until_us < self.resident_since_us:
            raise DynamicResidencyError(
                "dynamic placement minimum residency precedes publication"
            )

    @property
    def placement_id(self) -> str:
        return self.spec.placement_id

    @classmethod
    def from_json(cls, value: object) -> "DynamicWeightPlacement":
        row = _object("dynamic weight placement", value)
        if row.get("state") != "READY":
            raise DynamicResidencyError(
                "published dynamic placement is not READY"
            )
        return cls(
            spec=DynamicWeightPlacementSpec.from_json(row.get("spec")),
            generation=row.get("generation"),
            resident_since_us=row.get("resident_since_us"),
            minimum_resident_until_us=row.get(
                "minimum_resident_until_us"
            ),
            active_leases=row.get("active_leases", 0),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "active_leases": self.active_leases,
            "generation": self.generation,
            "minimum_resident_until_us": self.minimum_resident_until_us,
            "resident_since_us": self.resident_since_us,
            "spec": self.spec.to_json(),
            "state": "READY",
        }


def _epoch_key(
    source_epoch_key: str,
    generation: int,
    placements: Mapping[str, DynamicWeightPlacement],
) -> str:
    return canonical_sha256({
        "generation": generation,
        "placements": [
            {
                "generation": placement.generation,
                **placement.spec.to_json(),
            }
            for placement in sorted(
                placements.values(), key=lambda row: row.placement_id
            )
        ],
        "schema": "research-scheduler-residency-epoch-v1",
        "source_epoch_key": source_epoch_key,
    })


@dataclass(frozen=True)
class DynamicResidencySnapshot:
    snapshot_id: str
    epoch_key: str
    generation: int
    captured_at_us: int
    valid_until_us: int
    memory: Mapping[str, DeviceMemoryCapacity]
    placements: Mapping[str, DynamicWeightPlacement]

    def __post_init__(self) -> None:
        _text("dynamic snapshot_id", self.snapshot_id)
        object.__setattr__(
            self,
            "epoch_key",
            _sha256("dynamic snapshot epoch_key", self.epoch_key),
        )
        _integer("dynamic snapshot generation", self.generation, 1)
        _integer("dynamic snapshot captured_at_us", self.captured_at_us)
        _integer("dynamic snapshot valid_until_us", self.valid_until_us, 1)
        if self.valid_until_us <= self.captured_at_us:
            raise DynamicResidencyError(
                "dynamic snapshot validity interval is empty"
            )
        memory = dict(self.memory)
        if (
            not memory
            or any(
                not isinstance(row, DeviceMemoryCapacity)
                or resource_id != row.resource_id
                for resource_id, row in memory.items()
            )
        ):
            raise DynamicResidencyError(
                "dynamic snapshot memory resources are invalid"
            )
        placements = dict(self.placements)
        if any(
            not isinstance(row, DynamicWeightPlacement)
            or placement_id != row.placement_id
            for placement_id, row in placements.items()
        ):
            raise DynamicResidencyError(
                "dynamic snapshot placements are invalid"
            )
        identities: set[tuple[str, str]] = set()
        placed_bytes = {resource_id: 0 for resource_id in memory}
        for placement in placements.values():
            if placement.generation > self.generation:
                raise DynamicResidencyError(
                    "placement generation exceeds snapshot generation"
                )
            resource_id = placement.spec.resource_id
            if resource_id not in memory:
                raise DynamicResidencyError(
                    "dynamic placement memory resource is absent"
                )
            identity = (placement.spec.slice_id, resource_id)
            if identity in identities:
                raise DynamicResidencyError(
                    "duplicate slice placement on one resource"
                )
            identities.add(identity)
            placed_bytes[resource_id] += placement.spec.resident_bytes
        if any(
            placed_bytes[resource_id] > capacity.occupied_bytes
            for resource_id, capacity in memory.items()
        ):
            raise DynamicResidencyError(
                "resident weights exceed observed occupied memory"
            )
        object.__setattr__(
            self, "memory", MappingProxyType(dict(sorted(memory.items())))
        )
        object.__setattr__(
            self,
            "placements",
            MappingProxyType(dict(sorted(placements.items()))),
        )

    @classmethod
    def from_json(cls, value: object) -> "DynamicResidencySnapshot":
        row = _object("dynamic residency snapshot", value)
        if row.get("schema") != DYNAMIC_RESIDENCY_SCHEMA:
            raise DynamicResidencyError(
                "dynamic residency snapshot schema mismatch"
            )
        raw_memory = row.get("memory")
        raw_placements = row.get("placements")
        if type(raw_memory) is not list or type(raw_placements) is not list:
            raise DynamicResidencyError(
                "dynamic snapshot collections must be lists"
            )
        memory_rows = tuple(
            DeviceMemoryCapacity.from_json(item) for item in raw_memory
        )
        placement_rows = tuple(
            DynamicWeightPlacement.from_json(item) for item in raw_placements
        )
        return cls(
            snapshot_id=row.get("snapshot_id"),
            epoch_key=row.get("epoch_key"),
            generation=row.get("generation"),
            captured_at_us=row.get("captured_at_us"),
            valid_until_us=row.get("valid_until_us"),
            memory={item.resource_id: item for item in memory_rows},
            placements={item.placement_id: item for item in placement_rows},
        )

    def to_json(self) -> dict[str, object]:
        return {
            "captured_at_us": self.captured_at_us,
            "epoch_key": self.epoch_key,
            "generation": self.generation,
            "memory": [row.to_json() for row in self.memory.values()],
            "placements": [
                row.to_json() for row in self.placements.values()
            ],
            "schema": DYNAMIC_RESIDENCY_SCHEMA,
            "snapshot_id": self.snapshot_id,
            "valid_until_us": self.valid_until_us,
        }


@dataclass(frozen=True)
class DynamicFallbackServiceContract:
    fallback_id: str
    route_id: str
    route_hash: str
    placement_ids: tuple[str, ...]
    execution_resource_ids: tuple[str, ...]
    model_hashes: Mapping[str, str]
    ready_at_us: int
    valid_until_us: int
    service_latency_us: MetricEstimate
    service_energy_uj: MetricEstimate
    restore_latency_us: MetricEstimate
    restore_energy_uj: MetricEstimate
    energy_boundary_id: str
    accounting_scope: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "fallback_id",
            "route_id",
            "energy_boundary_id",
            "accounting_scope",
        ):
            _text(f"dynamic fallback {name}", getattr(self, name))
        object.__setattr__(
            self,
            "route_hash",
            _sha256("dynamic fallback route_hash", self.route_hash),
        )
        object.__setattr__(
            self,
            "placement_ids",
            _unique_text(
                "dynamic fallback placement id",
                self.placement_ids,
                nonempty=True,
            ),
        )
        object.__setattr__(
            self,
            "execution_resource_ids",
            _unique_text(
                "dynamic fallback execution resource",
                self.execution_resource_ids,
                nonempty=True,
            ),
        )
        model_hashes = {
            _text("dynamic fallback model id", model_id): _sha256(
                "dynamic fallback model hash", model_hash
            )
            for model_id, model_hash in dict(self.model_hashes).items()
        }
        if not model_hashes:
            raise DynamicResidencyError(
                "dynamic fallback model hashes must be non-empty"
            )
        object.__setattr__(
            self,
            "model_hashes",
            MappingProxyType(dict(sorted(model_hashes.items()))),
        )
        _integer("dynamic fallback ready_at_us", self.ready_at_us)
        _integer("dynamic fallback valid_until_us", self.valid_until_us, 1)
        if self.valid_until_us <= self.ready_at_us:
            raise DynamicResidencyError(
                "dynamic fallback validity interval is empty"
            )
        for name in (
            "service_latency_us",
            "service_energy_uj",
            "restore_latency_us",
            "restore_energy_uj",
        ):
            metric = getattr(self, name)
            if not isinstance(metric, MetricEstimate):
                raise DynamicResidencyError(
                    f"dynamic fallback {name} must be MetricEstimate"
                )
        object.__setattr__(self, "evidence_ids", _evidence(self.evidence_ids))

    @property
    def contract_sha256(self) -> str:
        return canonical_sha256(self.to_json())

    @classmethod
    def from_json(cls, value: object) -> "DynamicFallbackServiceContract":
        row = _object("dynamic fallback contract", value)
        if row.get("schema") != DYNAMIC_FALLBACK_CONTRACT_SCHEMA:
            raise DynamicResidencyError(
                "dynamic fallback contract schema mismatch"
            )
        placements = row.get("placement_ids")
        resources = row.get("execution_resource_ids")
        evidence = row.get("evidence_ids")
        model_hashes = row.get("model_hashes")
        if (
            type(placements) is not list
            or type(resources) is not list
            or type(evidence) is not list
            or type(model_hashes) is not dict
        ):
            raise DynamicResidencyError(
                "dynamic fallback collections are invalid"
            )
        return cls(
            fallback_id=row.get("fallback_id"),
            route_id=row.get("route_id"),
            route_hash=row.get("route_hash"),
            placement_ids=tuple(placements),
            execution_resource_ids=tuple(resources),
            model_hashes=model_hashes,
            ready_at_us=row.get("ready_at_us"),
            valid_until_us=row.get("valid_until_us"),
            service_latency_us=_metric(
                "dynamic fallback service_latency_us",
                row.get("service_latency_us"),
            ),
            service_energy_uj=_metric(
                "dynamic fallback service_energy_uj",
                row.get("service_energy_uj"),
            ),
            restore_latency_us=_metric(
                "dynamic fallback restore_latency_us",
                row.get("restore_latency_us"),
            ),
            restore_energy_uj=_metric(
                "dynamic fallback restore_energy_uj",
                row.get("restore_energy_uj"),
            ),
            energy_boundary_id=row.get("energy_boundary_id"),
            accounting_scope=row.get("accounting_scope"),
            evidence_ids=tuple(evidence),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "accounting_scope": self.accounting_scope,
            "energy_boundary_id": self.energy_boundary_id,
            "evidence_ids": list(self.evidence_ids),
            "execution_resource_ids": list(self.execution_resource_ids),
            "fallback_id": self.fallback_id,
            "model_hashes": dict(self.model_hashes),
            "placement_ids": list(self.placement_ids),
            "ready_at_us": self.ready_at_us,
            "restore_energy_uj": _metric_json(self.restore_energy_uj),
            "restore_latency_us": _metric_json(self.restore_latency_us),
            "route_hash": self.route_hash,
            "route_id": self.route_id,
            "schema": DYNAMIC_FALLBACK_CONTRACT_SCHEMA,
            "service_energy_uj": _metric_json(self.service_energy_uj),
            "service_latency_us": _metric_json(self.service_latency_us),
            "valid_until_us": self.valid_until_us,
        }


@dataclass(frozen=True)
class DynamicResidencyCandidate:
    candidate_id: str
    source_snapshot_id: str
    source_snapshot_sha256: str
    source_generation: int
    source_epoch_key: str
    target: DynamicWeightPlacementSpec
    evict_placement_ids: tuple[str, ...]
    transition_resource_ids: tuple[str, ...]
    expected_reuse_count: int
    minimum_reuse_count: int
    minimum_residency_us: int
    latest_ready_us: int
    energy_boundary_id: str
    accounting_scope: str
    baseline_latency_us: MetricEstimate
    resident_latency_us: MetricEstimate
    load_latency_us: MetricEstimate
    eviction_latency_us: MetricEstimate
    baseline_energy_uj: MetricEstimate
    resident_energy_uj: MetricEstimate
    load_energy_uj: MetricEstimate
    eviction_energy_uj: MetricEstimate
    evidence_ids: tuple[str, ...]
    transition_mode: str = "ATOMIC_STAGE_BEFORE_EVICT"
    fallback_contract: DynamicFallbackServiceContract | None = None
    transition_workspace_bytes: Mapping[str, int] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for name in (
            "candidate_id",
            "source_snapshot_id",
            "energy_boundary_id",
            "accounting_scope",
        ):
            _text(f"dynamic candidate {name}", getattr(self, name))
        _integer("dynamic candidate source_generation", self.source_generation, 1)
        object.__setattr__(
            self,
            "source_snapshot_sha256",
            _sha256(
                "dynamic candidate source_snapshot_sha256",
                self.source_snapshot_sha256,
            ),
        )
        object.__setattr__(
            self,
            "source_epoch_key",
            _sha256("dynamic candidate source_epoch_key", self.source_epoch_key),
        )
        if not isinstance(self.target, DynamicWeightPlacementSpec):
            raise DynamicResidencyError("dynamic candidate target is invalid")
        evictions = _unique_text(
            "dynamic candidate eviction id", self.evict_placement_ids
        )
        if self.target.placement_id in evictions:
            raise DynamicResidencyError(
                "dynamic candidate evicts its target placement id"
            )
        object.__setattr__(self, "evict_placement_ids", evictions)
        object.__setattr__(
            self,
            "transition_resource_ids",
            _unique_text(
                "dynamic candidate transition resource",
                self.transition_resource_ids,
                nonempty=True,
            ),
        )
        _integer(
            "dynamic candidate expected_reuse_count",
            self.expected_reuse_count,
        )
        _integer(
            "dynamic candidate minimum_reuse_count",
            self.minimum_reuse_count,
            1,
        )
        _integer(
            "dynamic candidate minimum_residency_us",
            self.minimum_residency_us,
        )
        _integer("dynamic candidate latest_ready_us", self.latest_ready_us, 1)
        for name in (
            "baseline_latency_us",
            "resident_latency_us",
            "load_latency_us",
            "eviction_latency_us",
            "baseline_energy_uj",
            "resident_energy_uj",
            "load_energy_uj",
            "eviction_energy_uj",
        ):
            metric = getattr(self, name)
            if not isinstance(metric, MetricEstimate):
                raise DynamicResidencyError(
                    f"dynamic candidate {name} must be MetricEstimate"
                )
            if type(metric.measured) is not bool:
                raise DynamicResidencyError(
                    f"dynamic candidate {name} measured must be bool"
                )
        object.__setattr__(self, "evidence_ids", _evidence(self.evidence_ids))
        if self.transition_mode not in DYNAMIC_RESIDENCY_TRANSITION_MODES:
            raise DynamicResidencyError(
                "unknown dynamic residency transition mode"
            )
        if self.transition_mode == "ATOMIC_STAGE_BEFORE_EVICT":
            if self.fallback_contract is not None:
                raise DynamicResidencyError(
                    "atomic dynamic transition carries a fallback contract"
                )
        elif not isinstance(
            self.fallback_contract, DynamicFallbackServiceContract
        ):
            raise DynamicResidencyError(
                "drain-first dynamic transition requires a fallback contract"
            )
        elif (
            self.fallback_contract.energy_boundary_id
                != self.energy_boundary_id
            or self.fallback_contract.accounting_scope
                != self.accounting_scope
        ):
            raise DynamicResidencyError(
                "dynamic fallback boundary or scope mismatch"
            )
        workspace: dict[str, int] = {}
        try:
            workspace_rows = dict(self.transition_workspace_bytes)
        except (TypeError, ValueError) as exc:
            raise DynamicResidencyError(
                "dynamic transition workspace must be a mapping"
            ) from exc
        for resource_id, workspace_bytes in workspace_rows.items():
            resource_id = _text(
                "dynamic transition workspace resource", resource_id
            )
            workspace[resource_id] = _integer(
                "dynamic transition workspace bytes", workspace_bytes, 1
            )
        object.__setattr__(
            self,
            "transition_workspace_bytes",
            MappingProxyType(dict(sorted(workspace.items()))),
        )

    @property
    def transition_latency_mean_us(self) -> int:
        return self.load_latency_us.mean + self.eviction_latency_us.mean

    @property
    def transition_latency_upper_us(self) -> int:
        return self.load_latency_us.upper + self.eviction_latency_us.upper

    @property
    def protected_transition_latency_mean_us(self) -> int:
        restore = (
            0
            if self.fallback_contract is None
            else self.fallback_contract.restore_latency_us.mean
        )
        return self.transition_latency_mean_us + restore

    @property
    def protected_transition_latency_upper_us(self) -> int:
        restore = (
            0
            if self.fallback_contract is None
            else self.fallback_contract.restore_latency_us.upper
        )
        return self.transition_latency_upper_us + restore

    @classmethod
    def from_json(cls, value: object) -> "DynamicResidencyCandidate":
        row = _object("dynamic residency candidate", value)
        evictions = row.get("evict_placement_ids")
        resources = row.get("transition_resource_ids")
        evidence = row.get("evidence_ids")
        workspace = row.get("transition_workspace_bytes", {})
        if not all(type(item) is list for item in (evictions, resources, evidence)):
            raise DynamicResidencyError(
                "dynamic candidate collections must be lists"
            )
        if type(workspace) is not dict:
            raise DynamicResidencyError(
                "dynamic transition workspace must be an object"
            )
        return cls(
            candidate_id=row.get("candidate_id"),
            source_snapshot_id=row.get("source_snapshot_id"),
            source_snapshot_sha256=row.get("source_snapshot_sha256"),
            source_generation=row.get("source_generation"),
            source_epoch_key=row.get("source_epoch_key"),
            target=DynamicWeightPlacementSpec.from_json(row.get("target")),
            evict_placement_ids=tuple(evictions),
            transition_resource_ids=tuple(resources),
            expected_reuse_count=row.get("expected_reuse_count"),
            minimum_reuse_count=row.get("minimum_reuse_count"),
            minimum_residency_us=row.get("minimum_residency_us"),
            latest_ready_us=row.get("latest_ready_us"),
            energy_boundary_id=row.get("energy_boundary_id"),
            accounting_scope=row.get("accounting_scope"),
            baseline_latency_us=_metric(
                "dynamic baseline_latency_us", row.get("baseline_latency_us")
            ),
            resident_latency_us=_metric(
                "dynamic resident_latency_us", row.get("resident_latency_us")
            ),
            load_latency_us=_metric(
                "dynamic load_latency_us", row.get("load_latency_us")
            ),
            eviction_latency_us=_metric(
                "dynamic eviction_latency_us", row.get("eviction_latency_us")
            ),
            baseline_energy_uj=_metric(
                "dynamic baseline_energy_uj", row.get("baseline_energy_uj")
            ),
            resident_energy_uj=_metric(
                "dynamic resident_energy_uj", row.get("resident_energy_uj")
            ),
            load_energy_uj=_metric(
                "dynamic load_energy_uj", row.get("load_energy_uj")
            ),
            eviction_energy_uj=_metric(
                "dynamic eviction_energy_uj", row.get("eviction_energy_uj")
            ),
            evidence_ids=tuple(evidence),
            transition_mode=row.get(
                "transition_mode", "ATOMIC_STAGE_BEFORE_EVICT"
            ),
            fallback_contract=(
                None
                if row.get("fallback_contract") is None
                else DynamicFallbackServiceContract.from_json(
                    row.get("fallback_contract")
                )
            ),
            transition_workspace_bytes=workspace,
        )

    def to_json(self) -> dict[str, object]:
        result: dict[str, object] = {
            "accounting_scope": self.accounting_scope,
            "baseline_energy_uj": _metric_json(self.baseline_energy_uj),
            "baseline_latency_us": _metric_json(self.baseline_latency_us),
            "candidate_id": self.candidate_id,
            "energy_boundary_id": self.energy_boundary_id,
            "evict_placement_ids": list(self.evict_placement_ids),
            "eviction_energy_uj": _metric_json(self.eviction_energy_uj),
            "eviction_latency_us": _metric_json(self.eviction_latency_us),
            "evidence_ids": list(self.evidence_ids),
            "expected_reuse_count": self.expected_reuse_count,
            "latest_ready_us": self.latest_ready_us,
            "load_energy_uj": _metric_json(self.load_energy_uj),
            "load_latency_us": _metric_json(self.load_latency_us),
            "minimum_residency_us": self.minimum_residency_us,
            "minimum_reuse_count": self.minimum_reuse_count,
            "resident_energy_uj": _metric_json(self.resident_energy_uj),
            "resident_latency_us": _metric_json(self.resident_latency_us),
            "source_epoch_key": self.source_epoch_key,
            "source_generation": self.source_generation,
            "source_snapshot_id": self.source_snapshot_id,
            "source_snapshot_sha256": self.source_snapshot_sha256,
            "target": self.target.to_json(),
            "transition_mode": self.transition_mode,
            "fallback_contract": (
                None
                if self.fallback_contract is None
                else self.fallback_contract.to_json()
            ),
            "transition_resource_ids": list(self.transition_resource_ids),
        }
        if self.transition_workspace_bytes:
            result["transition_workspace_bytes"] = dict(
                self.transition_workspace_bytes
            )
        return result


@dataclass(frozen=True)
class ResidencyTransitionAction:
    kind: str
    placement_id: str
    resource_id: str
    generation: int

    def __post_init__(self) -> None:
        if self.kind not in DYNAMIC_RESIDENCY_ACTIONS:
            raise DynamicResidencyError(
                "unknown dynamic residency transition action"
            )
        _text("dynamic action placement_id", self.placement_id)
        _text("dynamic action resource_id", self.resource_id)
        _integer("dynamic action generation", self.generation, 1)

    def to_json(self) -> dict[str, object]:
        return {
            "generation": self.generation,
            "kind": self.kind,
            "placement_id": self.placement_id,
            "resource_id": self.resource_id,
        }


@dataclass(frozen=True)
class DynamicResidencyDecision:
    candidate_id: str | None
    reason: str
    transition_id: str | None
    source_snapshot_id: str
    source_snapshot_sha256: str
    source_generation: int
    source_epoch_key: str
    target_generation: int
    target_epoch_key: str
    target: DynamicWeightPlacementSpec | None
    transition_mode: str | None
    fallback_contract: DynamicFallbackServiceContract | None
    target_minimum_residency_us: int
    evict_placement_ids: tuple[str, ...]
    actions: tuple[ResidencyTransitionAction, ...]
    transition_start_us: int | None
    ready_upper_us: int | None
    recovery_upper_us: int | None
    total_latency_upper_us: int | None
    energy_saving_lower_uj: int | None
    energy_saving_ppm: int | None
    occupied_bytes_after: Mapping[str, int]
    transition_workspace_bytes: Mapping[str, int]
    rejected: tuple[tuple[str, str], ...]

    @property
    def decision_sha256(self) -> str:
        return canonical_sha256(self.to_json())

    def to_json(self) -> dict[str, object]:
        result: dict[str, object] = {
            "actions": [row.to_json() for row in self.actions],
            "candidate_id": self.candidate_id,
            "energy_saving_lower_uj": self.energy_saving_lower_uj,
            "energy_saving_ppm": self.energy_saving_ppm,
            "evict_placement_ids": list(self.evict_placement_ids),
            "fallback_contract": (
                None
                if self.fallback_contract is None
                else self.fallback_contract.to_json()
            ),
            "occupied_bytes_after": dict(self.occupied_bytes_after),
            "ready_upper_us": self.ready_upper_us,
            "recovery_upper_us": self.recovery_upper_us,
            "reason": self.reason,
            "rejected": [
                {"candidate_id": candidate_id, "reason": reason}
                for candidate_id, reason in self.rejected
            ],
            "source_epoch_key": self.source_epoch_key,
            "source_generation": self.source_generation,
            "source_snapshot_id": self.source_snapshot_id,
            "source_snapshot_sha256": self.source_snapshot_sha256,
            "target": None if self.target is None else self.target.to_json(),
            "target_epoch_key": self.target_epoch_key,
            "target_generation": self.target_generation,
            "target_minimum_residency_us": (
                self.target_minimum_residency_us
            ),
            "total_latency_upper_us": self.total_latency_upper_us,
            "transition_mode": self.transition_mode,
            "transition_id": self.transition_id,
            "transition_start_us": self.transition_start_us,
        }
        if self.transition_workspace_bytes:
            result["transition_workspace_bytes"] = dict(
                self.transition_workspace_bytes
            )
        return result


def _no_change(
    snapshot: DynamicResidencySnapshot,
    rejected: list[tuple[str, str]],
) -> DynamicResidencyDecision:
    return DynamicResidencyDecision(
        candidate_id=None,
        reason="KEEP_CURRENT_RESIDENCY",
        transition_id=None,
        source_snapshot_id=snapshot.snapshot_id,
        source_snapshot_sha256=canonical_sha256(snapshot.to_json()),
        source_generation=snapshot.generation,
        source_epoch_key=snapshot.epoch_key,
        target_generation=snapshot.generation,
        target_epoch_key=snapshot.epoch_key,
        target=None,
        transition_mode=None,
        fallback_contract=None,
        target_minimum_residency_us=0,
        evict_placement_ids=(),
        actions=(),
        transition_start_us=None,
        ready_upper_us=None,
        recovery_upper_us=None,
        total_latency_upper_us=None,
        energy_saving_lower_uj=None,
        energy_saving_ppm=None,
        occupied_bytes_after=MappingProxyType({
            resource_id: row.occupied_bytes
            for resource_id, row in snapshot.memory.items()
        }),
        transition_workspace_bytes=MappingProxyType({}),
        rejected=tuple(rejected),
    )


def _fallback_invalid_reason(
    snapshot: DynamicResidencySnapshot,
    candidate: DynamicResidencyCandidate,
    evictions: Sequence[DynamicWeightPlacement],
) -> str | None:
    contract = candidate.fallback_contract
    if candidate.transition_mode == "ATOMIC_STAGE_BEFORE_EVICT":
        return None
    assert contract is not None
    placements: list[DynamicWeightPlacement] = []
    for placement_id in contract.placement_ids:
        placement = snapshot.placements.get(placement_id)
        if placement is None:
            return "FALLBACK_PLACEMENT_MISSING"
        if placement_id in candidate.evict_placement_ids:
            return "FALLBACK_PLACEMENT_EVICTED"
        if placement.spec.resource_id == candidate.target.resource_id:
            return "FALLBACK_USES_TARGET_MEMORY"
        if placement.active_leases:
            return "FALLBACK_PLACEMENT_LEASED"
        placements.append(placement)
    required_models: dict[str, str] = {}
    for placement in evictions:
        previous = required_models.get(placement.spec.model_id)
        if previous is not None and previous != placement.spec.model_hash:
            return "FALLBACK_MODEL_IDENTITY_CONFLICT"
        required_models[placement.spec.model_id] = placement.spec.model_hash
    previous = required_models.get(candidate.target.model_id)
    if previous is not None and previous != candidate.target.model_hash:
        return "FALLBACK_MODEL_IDENTITY_CONFLICT"
    required_models[candidate.target.model_id] = candidate.target.model_hash
    if any(
        contract.model_hashes.get(model_id) != model_hash
        for model_id, model_hash in required_models.items()
    ):
        return "FALLBACK_MODEL_COVERAGE"
    if (
        any(
            contract.model_hashes.get(placement.spec.model_id)
                != placement.spec.model_hash
            for placement in placements
        )
        or any(
            not any(
                placement.spec.model_id == model_id
                and placement.spec.model_hash == model_hash
                for placement in placements
            )
            for model_id, model_hash in contract.model_hashes.items()
        )
    ):
        return "FALLBACK_PLACEMENT_MODEL_COVERAGE"
    placement_execution_resources = {
        resource_id
        for placement in placements
        for resource_id in placement.spec.execution_resource_ids
    }
    protected_gpu_resources = {
        resource_id
        for placement in evictions
        for resource_id in placement.spec.execution_resource_ids
    } | set(candidate.target.execution_resource_ids)
    if (
        set(contract.execution_resource_ids) != placement_execution_resources
        or not set(contract.execution_resource_ids)
            <= set(candidate.transition_resource_ids)
    ):
        return "FALLBACK_RESOURCE_NOT_LEASED"
    if set(contract.execution_resource_ids) & protected_gpu_resources:
        return "FALLBACK_RESOURCE_OVERLAP"
    return None


def select_dynamic_residency_transition(
    snapshot: DynamicResidencySnapshot,
    candidates: Sequence[DynamicResidencyCandidate],
    *,
    now_us: int,
    transition_resource_ready_us: int | Mapping[str, int | None],
    minimum_energy_saving_ppm: int = 50_000,
    latency_limit_ppm: int = 1_000_000,
    require_measured: bool = True,
) -> DynamicResidencyDecision:
    if not isinstance(snapshot, DynamicResidencySnapshot):
        raise DynamicResidencyError("dynamic residency snapshot is invalid")
    _integer("dynamic residency now_us", now_us)
    _integer(
        "dynamic minimum_energy_saving_ppm",
        minimum_energy_saving_ppm,
    )
    _integer("dynamic latency_limit_ppm", latency_limit_ppm, 1_000_000)
    if minimum_energy_saving_ppm >= 1_000_000:
        raise DynamicResidencyError(
            "dynamic energy saving gate reaches 100 percent"
        )
    if now_us < snapshot.captured_at_us or now_us >= snapshot.valid_until_us:
        raise DynamicResidencyError("dynamic residency snapshot is expired")

    ready_by_candidate: Mapping[str, int | None] | None = None
    if isinstance(transition_resource_ready_us, Mapping):
        ready_by_candidate = transition_resource_ready_us
        for candidate_id, ready_us in ready_by_candidate.items():
            _text("dynamic ready candidate_id", candidate_id)
            if ready_us is not None:
                _integer("dynamic transition resource_ready_us", ready_us)
    else:
        _integer(
            "dynamic transition resource_ready_us",
            transition_resource_ready_us,
        )

    candidate_rows = tuple(candidates)
    if any(
        not isinstance(candidate, DynamicResidencyCandidate)
        for candidate in candidate_rows
    ):
        raise DynamicResidencyError("dynamic residency candidate is invalid")
    candidate_ids = [candidate.candidate_id for candidate in candidate_rows]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise DynamicResidencyError("duplicate dynamic residency candidate id")
    boundary_scope = {
        (candidate.energy_boundary_id, candidate.accounting_scope)
        for candidate in candidate_rows
    }
    if len(boundary_scope) > 1:
        raise DynamicResidencyError(
            "dynamic candidates use different energy boundaries or scopes"
        )

    rejected: list[tuple[str, str]] = []
    feasible: list[
        tuple[
            int,
            int,
            str,
            DynamicResidencyCandidate,
            int,
            int,
            int,
            int,
            int,
            Mapping[str, int],
            tuple[ResidencyTransitionAction, ...],
            str,
        ]
    ] = []
    for candidate in candidate_rows:
        if (
            candidate.source_snapshot_id != snapshot.snapshot_id
            or candidate.source_snapshot_sha256
                != canonical_sha256(snapshot.to_json())
            or candidate.source_generation != snapshot.generation
            or candidate.source_epoch_key != snapshot.epoch_key
        ):
            rejected.append((candidate.candidate_id, "SOURCE_EPOCH_MISMATCH"))
            continue
        current_target = snapshot.placements.get(candidate.target.placement_id)
        if current_target is not None:
            reason = (
                "ALREADY_RESIDENT"
                if current_target.spec == candidate.target
                else "PLACEMENT_IDENTITY_MISMATCH"
            )
            rejected.append((candidate.candidate_id, reason))
            continue
        if any(
            placement.placement_id not in candidate.evict_placement_ids
            and
            placement.spec.slice_id == candidate.target.slice_id
            and placement.spec.resource_id == candidate.target.resource_id
            for placement in snapshot.placements.values()
        ):
            rejected.append(
                (candidate.candidate_id, "SLICE_ALREADY_ON_RESOURCE")
            )
            continue
        if candidate.expected_reuse_count < candidate.minimum_reuse_count:
            rejected.append((candidate.candidate_id, "REUSE_NOT_AMORTIZED"))
            continue
        metrics = (
            candidate.baseline_latency_us,
            candidate.resident_latency_us,
            candidate.load_latency_us,
            candidate.eviction_latency_us,
            candidate.baseline_energy_uj,
            candidate.resident_energy_uj,
            candidate.load_energy_uj,
            candidate.eviction_energy_uj,
        )
        if candidate.fallback_contract is not None:
            metrics = (
                *metrics,
                candidate.fallback_contract.service_latency_us,
                candidate.fallback_contract.service_energy_uj,
                candidate.fallback_contract.restore_latency_us,
                candidate.fallback_contract.restore_energy_uj,
            )
        if require_measured and not all(
            metric.measured and metric.sample_count > 0 for metric in metrics
        ):
            rejected.append((candidate.candidate_id, "MEASUREMENT_REQUIRED"))
            continue
        if candidate.baseline_energy_uj.lower is None:
            rejected.append(
                (candidate.candidate_id, "BASELINE_ENERGY_LCB_MISSING")
            )
            continue
        evictions: list[DynamicWeightPlacement] = []
        eviction_invalid = None
        for placement_id in candidate.evict_placement_ids:
            placement = snapshot.placements.get(placement_id)
            if placement is None:
                eviction_invalid = "EVICTION_UNKNOWN"
                break
            if placement.active_leases:
                eviction_invalid = "EVICTION_LEASED"
                break
            if now_us < placement.minimum_resident_until_us:
                eviction_invalid = "EVICTION_HYSTERESIS"
                break
            evictions.append(placement)
        if eviction_invalid is not None:
            rejected.append((candidate.candidate_id, eviction_invalid))
            continue
        fallback_invalid = _fallback_invalid_reason(
            snapshot, candidate, evictions
        )
        if fallback_invalid is not None:
            rejected.append((candidate.candidate_id, fallback_invalid))
            continue
        target_memory = snapshot.memory.get(candidate.target.resource_id)
        if target_memory is None:
            rejected.append((candidate.candidate_id, "MEMORY_RESOURCE_MISSING"))
            continue
        if not set(candidate.transition_workspace_bytes) <= set(
            snapshot.memory
        ):
            rejected.append((
                candidate.candidate_id,
                "TRANSITION_MEMORY_RESOURCE_MISSING",
            ))
            continue

        occupied_after = {
            resource_id: row.occupied_bytes
            for resource_id, row in snapshot.memory.items()
        }
        for placement in evictions:
            resource_id = placement.spec.resource_id
            occupied_after[resource_id] -= placement.spec.resident_bytes
        occupied_after[candidate.target.resource_id] += (
            candidate.target.resident_bytes
        )
        final_memory_invalid = any(
            occupied_after[resource_id] + capacity.reserve_bytes
            > capacity.capacity_bytes
            for resource_id, capacity in snapshot.memory.items()
        )
        if final_memory_invalid:
            rejected.append((candidate.candidate_id, "MEMORY_CAPACITY"))
            continue
        peak_occupied = {
            resource_id: row.occupied_bytes
            for resource_id, row in snapshot.memory.items()
        }
        peak_occupied[candidate.target.resource_id] += (
            candidate.target.resident_bytes
        )
        if (
            candidate.transition_mode == "ATOMIC_STAGE_BEFORE_EVICT"
            and any(
                peak_occupied[resource_id] + capacity.reserve_bytes
                > capacity.capacity_bytes
                for resource_id, capacity in snapshot.memory.items()
            )
        ):
            rejected.append((candidate.candidate_id, "ATOMIC_STAGING_MEMORY"))
            continue
        workspace_peak = dict(
            peak_occupied
            if candidate.transition_mode == "ATOMIC_STAGE_BEFORE_EVICT"
            else occupied_after
        )
        for resource_id, workspace_bytes in (
            candidate.transition_workspace_bytes.items()
        ):
            workspace_peak[resource_id] += workspace_bytes
        if any(
            workspace_peak[resource_id] + capacity.reserve_bytes
            > capacity.capacity_bytes
            for resource_id, capacity in snapshot.memory.items()
        ):
            rejected.append((
                candidate.candidate_id,
                "TRANSITION_WORKSPACE_MEMORY",
            ))
            continue

        ready_us = (
            ready_by_candidate.get(candidate.candidate_id)
            if ready_by_candidate is not None
            else transition_resource_ready_us
        )
        if ready_us is None:
            rejected.append((candidate.candidate_id, "RESOURCE_NOT_READY"))
            continue
        fallback = candidate.fallback_contract
        transition_start_us = max(
            now_us,
            ready_us,
            now_us if fallback is None else fallback.ready_at_us,
        )
        ready_upper_us = (
            transition_start_us + candidate.transition_latency_upper_us
        )
        recovery_upper_us = (
            transition_start_us
            + candidate.protected_transition_latency_upper_us
        )
        if recovery_upper_us >= snapshot.valid_until_us:
            rejected.append(
                (candidate.candidate_id, "SNAPSHOT_EXPIRES_BEFORE_READY")
            )
            continue
        if (
            fallback is not None
            and fallback.valid_until_us <= recovery_upper_us
        ):
            rejected.append(
                (candidate.candidate_id, "FALLBACK_EXPIRES_BEFORE_RECOVERY")
            )
            continue
        if ready_upper_us > candidate.latest_ready_us:
            rejected.append((candidate.candidate_id, "READY_DEADLINE"))
            continue
        total_latency_upper_us = (
            ready_upper_us - now_us + candidate.resident_latency_us.upper
            + (
                0
                if fallback is None
                else fallback.service_latency_us.upper
            )
        )
        if (
            total_latency_upper_us * 1_000_000
            > candidate.baseline_latency_us.upper * latency_limit_ppm
        ):
            rejected.append((candidate.candidate_id, "LATENCY_REGRESSION"))
            continue
        transition_energy_upper = (
            candidate.load_energy_uj.upper
            + candidate.eviction_energy_uj.upper
            + (
                0
                if fallback is None
                else fallback.service_energy_uj.upper
                    + fallback.restore_energy_uj.upper
            )
        )
        total_energy_upper = (
            candidate.resident_energy_uj.upper + transition_energy_upper
        )
        saving_lower = candidate.baseline_energy_uj.lower - total_energy_upper
        if saving_lower <= 0:
            rejected.append((candidate.candidate_id, "ENERGY_REGRESSION"))
            continue
        saving_ppm = (
            saving_lower * 1_000_000 // candidate.baseline_energy_uj.lower
        )
        if saving_ppm < minimum_energy_saving_ppm:
            rejected.append((candidate.candidate_id, "ENERGY_MARGIN"))
            continue

        target_generation = snapshot.generation + 1
        eviction_actions = tuple(
            action
            for placement in evictions
            for action in (
                ResidencyTransitionAction(
                    "DRAIN",
                    placement.placement_id,
                    placement.spec.resource_id,
                    target_generation,
                ),
                ResidencyTransitionAction(
                    "EVICT",
                    placement.placement_id,
                    placement.spec.resource_id,
                    target_generation,
                ),
            )
        )
        target_actions = (
            ResidencyTransitionAction(
                "PREFETCH",
                candidate.target.placement_id,
                candidate.target.resource_id,
                target_generation,
            ),
            ResidencyTransitionAction(
                "VERIFY",
                candidate.target.placement_id,
                candidate.target.resource_id,
                target_generation,
            ),
            ResidencyTransitionAction(
                "PUBLISH",
                candidate.target.placement_id,
                candidate.target.resource_id,
                target_generation,
            ),
        )
        if fallback is None:
            actions = (
                *target_actions[:2],
                *eviction_actions,
                target_actions[2],
            )
        else:
            fallback_placements = [
                snapshot.placements[placement_id]
                for placement_id in fallback.placement_ids
            ]
            actions = (
                *tuple(
                    ResidencyTransitionAction(
                        "FALLBACK_ACQUIRE",
                        placement.placement_id,
                        placement.spec.resource_id,
                        target_generation,
                    )
                    for placement in fallback_placements
                ),
                *eviction_actions,
                *target_actions,
                *tuple(
                    ResidencyTransitionAction(
                        "FALLBACK_RELEASE",
                        placement.placement_id,
                        placement.spec.resource_id,
                        target_generation,
                    )
                    for placement in reversed(fallback_placements)
                ),
            )
        planned = {
            placement_id: placement
            for placement_id, placement in snapshot.placements.items()
            if placement_id not in candidate.evict_placement_ids
        }
        planned[candidate.target.placement_id] = DynamicWeightPlacement(
            spec=candidate.target,
            generation=target_generation,
            resident_since_us=ready_upper_us,
            minimum_resident_until_us=(
                ready_upper_us + candidate.minimum_residency_us
            ),
        )
        target_epoch_key = _epoch_key(
            snapshot.epoch_key, target_generation, planned
        )
        feasible.append((
            -saving_lower,
            ready_upper_us,
            candidate.candidate_id,
            candidate,
            transition_start_us,
            total_latency_upper_us,
            recovery_upper_us,
            saving_lower,
            saving_ppm,
            MappingProxyType(dict(sorted(occupied_after.items()))),
            actions,
            target_epoch_key,
        ))

    if not feasible:
        return _no_change(snapshot, rejected)
    (
        _,
        ready_upper_us,
        _,
        selected,
        transition_start_us,
        total_latency_upper_us,
        recovery_upper_us,
        saving_lower,
        saving_ppm,
        occupied_after,
        actions,
        target_epoch_key,
    ) = min(feasible)
    target_generation = snapshot.generation + 1
    return DynamicResidencyDecision(
        candidate_id=selected.candidate_id,
        reason=(
            "ENERGY_POSITIVE_RESIDENCY_TRANSITION"
            if selected.fallback_contract is None
            else "ENERGY_POSITIVE_FALLBACK_BACKED_RESIDENCY_TRANSITION"
        ),
        transition_id=(
            f"residency:{snapshot.generation}:{selected.candidate_id}"
        ),
        source_snapshot_id=snapshot.snapshot_id,
        source_snapshot_sha256=canonical_sha256(snapshot.to_json()),
        source_generation=snapshot.generation,
        source_epoch_key=snapshot.epoch_key,
        target_generation=target_generation,
        target_epoch_key=target_epoch_key,
        target=selected.target,
        transition_mode=selected.transition_mode,
        fallback_contract=selected.fallback_contract,
        target_minimum_residency_us=selected.minimum_residency_us,
        evict_placement_ids=selected.evict_placement_ids,
        actions=actions,
        transition_start_us=transition_start_us,
        ready_upper_us=ready_upper_us,
        recovery_upper_us=recovery_upper_us,
        total_latency_upper_us=total_latency_upper_us,
        energy_saving_lower_uj=saving_lower,
        energy_saving_ppm=saving_ppm,
        occupied_bytes_after=occupied_after,
        transition_workspace_bytes=selected.transition_workspace_bytes,
        rejected=tuple(rejected),
    )


@dataclass(frozen=True)
class DynamicFallbackServiceReceipt:
    fallback_id: str
    contract_sha256: str
    route_hash: str
    placement_ids: tuple[str, ...]
    acquired_at_us: int
    ready_at_us: int
    released_at_us: int
    served_request_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _text("dynamic fallback receipt fallback_id", self.fallback_id)
        object.__setattr__(
            self,
            "contract_sha256",
            _sha256(
                "dynamic fallback receipt contract_sha256",
                self.contract_sha256,
            ),
        )
        object.__setattr__(
            self,
            "route_hash",
            _sha256("dynamic fallback receipt route_hash", self.route_hash),
        )
        object.__setattr__(
            self,
            "placement_ids",
            _unique_text(
                "dynamic fallback receipt placement id",
                self.placement_ids,
                nonempty=True,
            ),
        )
        _integer(
            "dynamic fallback receipt acquired_at_us", self.acquired_at_us
        )
        _integer("dynamic fallback receipt ready_at_us", self.ready_at_us)
        _integer(
            "dynamic fallback receipt released_at_us", self.released_at_us
        )
        if (
            self.ready_at_us > self.acquired_at_us
            or self.released_at_us < self.acquired_at_us
        ):
            raise DynamicResidencyError(
                "dynamic fallback receipt interval is invalid"
            )
        object.__setattr__(
            self,
            "served_request_ids",
            _unique_text(
                "dynamic fallback served request id",
                self.served_request_ids,
            ),
        )
        object.__setattr__(self, "evidence_ids", _evidence(self.evidence_ids))

    @classmethod
    def from_json(cls, value: object) -> "DynamicFallbackServiceReceipt":
        row = _object("dynamic fallback receipt", value)
        if row.get("schema") != DYNAMIC_FALLBACK_RECEIPT_SCHEMA:
            raise DynamicResidencyError(
                "dynamic fallback receipt schema mismatch"
            )
        placements = row.get("placement_ids")
        requests = row.get("served_request_ids")
        evidence = row.get("evidence_ids")
        if not all(
            type(item) is list for item in (placements, requests, evidence)
        ):
            raise DynamicResidencyError(
                "dynamic fallback receipt collections are invalid"
            )
        return cls(
            fallback_id=row.get("fallback_id"),
            contract_sha256=row.get("contract_sha256"),
            route_hash=row.get("route_hash"),
            placement_ids=tuple(placements),
            acquired_at_us=row.get("acquired_at_us"),
            ready_at_us=row.get("ready_at_us"),
            released_at_us=row.get("released_at_us"),
            served_request_ids=tuple(requests),
            evidence_ids=tuple(evidence),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "acquired_at_us": self.acquired_at_us,
            "contract_sha256": self.contract_sha256,
            "evidence_ids": list(self.evidence_ids),
            "fallback_id": self.fallback_id,
            "placement_ids": list(self.placement_ids),
            "ready_at_us": self.ready_at_us,
            "released_at_us": self.released_at_us,
            "route_hash": self.route_hash,
            "schema": DYNAMIC_FALLBACK_RECEIPT_SCHEMA,
            "served_request_ids": list(self.served_request_ids),
        }


@dataclass(frozen=True)
class DynamicResidencyReceipt:
    transition_id: str
    decision_sha256: str
    status: str
    source_snapshot_id: str
    source_generation: int
    source_epoch_key: str
    started_at_us: int
    completed_at_us: int
    result_snapshot: DynamicResidencySnapshot
    evidence_ids: tuple[str, ...]
    failure_reason: str | None = None
    fallback_receipt: DynamicFallbackServiceReceipt | None = None

    def __post_init__(self) -> None:
        _text("dynamic receipt transition_id", self.transition_id)
        object.__setattr__(
            self,
            "decision_sha256",
            _sha256("dynamic receipt decision_sha256", self.decision_sha256),
        )
        if self.status not in DYNAMIC_RESIDENCY_RECEIPT_STATUSES:
            raise DynamicResidencyError(
                "unknown dynamic residency receipt status"
            )
        _text("dynamic receipt source_snapshot_id", self.source_snapshot_id)
        _integer("dynamic receipt source_generation", self.source_generation, 1)
        object.__setattr__(
            self,
            "source_epoch_key",
            _sha256("dynamic receipt source_epoch_key", self.source_epoch_key),
        )
        _integer("dynamic receipt started_at_us", self.started_at_us)
        _integer("dynamic receipt completed_at_us", self.completed_at_us)
        if self.completed_at_us < self.started_at_us:
            raise DynamicResidencyError(
                "dynamic receipt completes before it starts"
            )
        if not isinstance(self.result_snapshot, DynamicResidencySnapshot):
            raise DynamicResidencyError(
                "dynamic receipt result snapshot is invalid"
            )
        if self.status == "FAILED":
            _text("dynamic receipt failure_reason", self.failure_reason)
        elif self.failure_reason is not None:
            raise DynamicResidencyError(
                "successful dynamic receipt carries a failure reason"
            )
        if (
            self.fallback_receipt is not None
            and not isinstance(
                self.fallback_receipt, DynamicFallbackServiceReceipt
            )
        ):
            raise DynamicResidencyError(
                "dynamic fallback receipt is invalid"
            )
        object.__setattr__(self, "evidence_ids", _evidence(self.evidence_ids))

    @classmethod
    def from_json(cls, value: object) -> "DynamicResidencyReceipt":
        row = _object("dynamic residency receipt", value)
        if row.get("schema") != DYNAMIC_RESIDENCY_RECEIPT_SCHEMA:
            raise DynamicResidencyError(
                "dynamic residency receipt schema mismatch"
            )
        evidence = row.get("evidence_ids")
        if type(evidence) is not list:
            raise DynamicResidencyError(
                "dynamic receipt evidence_ids must be a list"
            )
        return cls(
            transition_id=row.get("transition_id"),
            decision_sha256=row.get("decision_sha256"),
            status=row.get("status"),
            source_snapshot_id=row.get("source_snapshot_id"),
            source_generation=row.get("source_generation"),
            source_epoch_key=row.get("source_epoch_key"),
            started_at_us=row.get("started_at_us"),
            completed_at_us=row.get("completed_at_us"),
            result_snapshot=DynamicResidencySnapshot.from_json(
                row.get("result_snapshot")
            ),
            evidence_ids=tuple(evidence),
            failure_reason=row.get("failure_reason"),
            fallback_receipt=(
                None
                if row.get("fallback_receipt") is None
                else DynamicFallbackServiceReceipt.from_json(
                    row.get("fallback_receipt")
                )
            ),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "completed_at_us": self.completed_at_us,
            "decision_sha256": self.decision_sha256,
            "evidence_ids": list(self.evidence_ids),
            "failure_reason": self.failure_reason,
            "fallback_receipt": (
                None
                if self.fallback_receipt is None
                else self.fallback_receipt.to_json()
            ),
            "result_snapshot": self.result_snapshot.to_json(),
            "schema": DYNAMIC_RESIDENCY_RECEIPT_SCHEMA,
            "source_epoch_key": self.source_epoch_key,
            "source_generation": self.source_generation,
            "source_snapshot_id": self.source_snapshot_id,
            "started_at_us": self.started_at_us,
            "status": self.status,
            "transition_id": self.transition_id,
        }


def apply_dynamic_residency_receipt(
    snapshot: DynamicResidencySnapshot,
    decision: DynamicResidencyDecision,
    receipt: DynamicResidencyReceipt,
) -> DynamicResidencySnapshot:
    if decision.candidate_id is None or decision.transition_id is None:
        raise DynamicResidencyError(
            "no dynamic residency transition is pending"
        )
    source_identity = (
        snapshot.snapshot_id,
        snapshot.generation,
        snapshot.epoch_key,
    )
    if (
        canonical_sha256(snapshot.to_json())
        != decision.source_snapshot_sha256
    ):
        raise DynamicResidencyError("dynamic residency decision is stale")
    if source_identity != (
        decision.source_snapshot_id,
        decision.source_generation,
        decision.source_epoch_key,
    ):
        raise DynamicResidencyError("dynamic residency decision is stale")
    if source_identity != (
        receipt.source_snapshot_id,
        receipt.source_generation,
        receipt.source_epoch_key,
    ):
        raise DynamicResidencyError("dynamic residency receipt is stale")
    if receipt.transition_id != decision.transition_id:
        raise DynamicResidencyError("dynamic transition identity mismatch")
    if receipt.decision_sha256 != decision.decision_sha256:
        raise DynamicResidencyError("dynamic transition decision hash mismatch")
    fallback = decision.fallback_contract
    if (
        decision.transition_start_us is None
        or decision.ready_upper_us is None
        or decision.recovery_upper_us is None
        or receipt.started_at_us < decision.transition_start_us
        or receipt.completed_at_us > (
            decision.ready_upper_us
            if receipt.status == "READY"
            else decision.recovery_upper_us
        )
    ):
        raise DynamicResidencyError(
            "dynamic transition timing exceeds its decision envelope"
        )
    if fallback is None:
        if receipt.fallback_receipt is not None:
            raise DynamicResidencyError(
                "atomic transition carries a fallback receipt"
            )
    else:
        fallback_receipt = receipt.fallback_receipt
        if fallback_receipt is None:
            raise DynamicResidencyError(
                "fallback-backed transition omitted its fallback receipt"
            )
        if (
            fallback_receipt.fallback_id != fallback.fallback_id
            or fallback_receipt.contract_sha256
                != fallback.contract_sha256
            or fallback_receipt.route_hash != fallback.route_hash
            or fallback_receipt.placement_ids != fallback.placement_ids
        ):
            raise DynamicResidencyError(
                "dynamic fallback receipt identity mismatch"
            )
        if (
            fallback_receipt.ready_at_us != fallback.ready_at_us
            or fallback_receipt.acquired_at_us
                != decision.transition_start_us
            or fallback_receipt.released_at_us
                != receipt.completed_at_us
            or fallback_receipt.released_at_us >= fallback.valid_until_us
        ):
            raise DynamicResidencyError(
                "dynamic fallback receipt does not cover the transition"
            )
    if receipt.status == "FAILED":
        if receipt.result_snapshot != snapshot:
            raise DynamicResidencyError(
                "failed transition did not restore the source residency"
            )
        return snapshot

    result = receipt.result_snapshot
    if (
        result.generation != decision.target_generation
        or result.epoch_key != decision.target_epoch_key
        or result.captured_at_us != receipt.completed_at_us
        or result.valid_until_us <= receipt.completed_at_us
    ):
        raise DynamicResidencyError(
            "dynamic transition result epoch is invalid"
        )
    expected_ids = (
        set(snapshot.placements)
        - set(decision.evict_placement_ids)
    )
    assert decision.target is not None
    expected_ids.add(decision.target.placement_id)
    if set(result.placements) != expected_ids:
        raise DynamicResidencyError(
            "dynamic transition result placement set mismatch"
        )
    for placement_id, placement in snapshot.placements.items():
        if placement_id in decision.evict_placement_ids:
            continue
        if result.placements[placement_id] != placement:
            raise DynamicResidencyError(
                "dynamic transition changed a retained placement"
            )
    target = result.placements[decision.target.placement_id]
    if (
        target.spec != decision.target
        or target.generation != decision.target_generation
        or target.resident_since_us != receipt.completed_at_us
        or target.minimum_resident_until_us
            != receipt.completed_at_us
                + decision.target_minimum_residency_us
        or target.active_leases != 0
    ):
        raise DynamicResidencyError(
            "dynamic transition target receipt is invalid"
        )
    expected_epoch = _epoch_key(
        snapshot.epoch_key, result.generation, result.placements
    )
    if result.epoch_key != expected_epoch:
        raise DynamicResidencyError(
            "dynamic transition result epoch hash mismatch"
        )
    if set(result.memory) != set(snapshot.memory):
        raise DynamicResidencyError(
            "dynamic transition memory resource set changed"
        )
    for resource_id, capacity in result.memory.items():
        source = snapshot.memory[resource_id]
        if (
            capacity.capacity_bytes != source.capacity_bytes
            or capacity.reserve_bytes != source.reserve_bytes
            or capacity.occupied_bytes
                != decision.occupied_bytes_after[resource_id]
        ):
            raise DynamicResidencyError(
                "dynamic transition memory receipt mismatch"
            )
    return result
