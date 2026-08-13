"""Canonical scheduling-unit and evaluated-route contracts.

The online engine consumes RouteAlternative objects. Model-specific compilers
may generate them from measurements or conservative performance predictions.
The contracts keep placement granularity independent from energy-accounting
scope so request totals and non-additive cohort totals cannot be mixed.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping, Sequence


CONTRACT_SCHEMA = "s42-energy-scheduler-contract-v1"


class SchedulerContractError(ValueError):
    pass


class UnitKind(str, Enum):
    REQUEST = "request"
    PHYSICAL_UBATCH = "physical_ubatch"
    COHORT = "cohort"


class PlacementGranularity(str, Enum):
    TASK = "task"
    LAYER = "layer"
    OPERATOR = "operator"


class AccountingScope(str, Enum):
    REQUEST = "request"
    PHYSICAL_UBATCH = "physical_ubatch"
    COHORT = "cohort"
    EPOCH = "epoch"


class AccountingKind(str, Enum):
    ADDITIVE_MARGINAL = "additive_marginal"
    EXCLUSIVE_UNIT_TOTAL = "exclusive_unit_total"
    NON_ADDITIVE_COHORT_TOTAL = "non_additive_cohort_total"
    EPOCH_OBLIGATION = "epoch_obligation"


class RouteMaturity(str, Enum):
    PREDICTED = "predicted"
    MEASURED = "measured"
    STABLE = "stable"


class QualityClass(str, Enum):
    UNVERIFIED = "unverified"
    SEMANTIC = "semantic"
    BOUNDED_NUMERIC = "bounded_numeric"
    EXACT = "exact"


QUALITY_RANK = {
    QualityClass.UNVERIFIED: 0,
    QualityClass.SEMANTIC: 1,
    QualityClass.BOUNDED_NUMERIC: 2,
    QualityClass.EXACT: 3,
}


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise SchedulerContractError(f"{name} must be a non-empty string")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise SchedulerContractError(f"{name} must be ASCII") from exc
    return value


def _integer(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise SchedulerContractError(
            f"{name} must be an integer >= {minimum}"
        )
    return value


def _sha256(name: str, value: object) -> str:
    result = _text(name, value)
    digest = result.removeprefix("sha256:")
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise SchedulerContractError(f"{name} must be a lowercase SHA-256")
    return "sha256:" + digest


def _text_tuple(name: str, values: Sequence[str], *, nonempty: bool = False) -> tuple[str, ...]:
    result = tuple(_text(name, value) for value in values)
    if nonempty and not result:
        raise SchedulerContractError(f"{name} must not be empty")
    if len(result) != len(set(result)):
        raise SchedulerContractError(f"{name} must not contain duplicates")
    return result


def _immutable_int_mapping(name: str, values: Mapping[str, int]) -> Mapping[str, int]:
    result = {
        _text(f"{name} key", key): _integer(f"{name} {key}", value)
        for key, value in values.items()
    }
    return MappingProxyType(dict(sorted(result.items())))


def _canonical_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            item.name: _canonical_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if value is None or type(value) in {bool, int, str}:
        return value
    raise SchedulerContractError(
        f"unsupported canonical value type: {type(value).__name__}"
    )


def canonical_json(value: object) -> str:
    return json.dumps(
        _canonical_value(value),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_sha256(value: object) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()
    return "sha256:" + digest


def make_work_set_hash(member_request_ids: Sequence[str]) -> str:
    members = _text_tuple(
        "member request id", member_request_ids, nonempty=True
    )
    return canonical_sha256({
        "kind": "s42-work-set-v1",
        "member_request_ids": sorted(members),
    })


@dataclass(frozen=True)
class ModelIdentity:
    model_id: str
    model_hash: str
    architecture: str
    weight_format: str
    weight_bytes: int

    def __post_init__(self) -> None:
        _text("model_id", self.model_id)
        _sha256("model_hash", self.model_hash)
        _text("model architecture", self.architecture)
        _text("model weight_format", self.weight_format)
        _integer("model weight_bytes", self.weight_bytes, 1)


@dataclass(frozen=True)
class SchedulingUnit:
    unit_id: str
    kind: UnitKind
    member_request_ids: tuple[str, ...]
    work_set_hash: str
    workload_id: str
    model: ModelIdentity
    arrival_us: int
    deadline_us: int
    input_tokens: int
    output_tokens: int
    features: Mapping[str, int]
    quality_requirement: QualityClass
    semantics: Mapping[str, bool | str]
    epoch_id: str | None = None

    def __post_init__(self) -> None:
        _text("unit_id", self.unit_id)
        if not isinstance(self.kind, UnitKind):
            raise SchedulerContractError("unit kind must be UnitKind")
        if not isinstance(self.model, ModelIdentity):
            raise SchedulerContractError("unit model must be ModelIdentity")
        if not isinstance(self.quality_requirement, QualityClass):
            raise SchedulerContractError(
                "quality requirement must be QualityClass"
            )
        members = _text_tuple(
            "member request id", self.member_request_ids, nonempty=True
        )
        object.__setattr__(self, "member_request_ids", members)
        if self.kind == UnitKind.REQUEST and len(members) != 1:
            raise SchedulerContractError(
                "request scheduling unit must contain exactly one request"
            )
        expected_hash = make_work_set_hash(members)
        if _sha256("work_set_hash", self.work_set_hash) != expected_hash:
            raise SchedulerContractError(
                "work_set_hash does not match member_request_ids"
            )
        _text("workload_id", self.workload_id)
        _integer("arrival_us", self.arrival_us)
        _integer("deadline_us", self.deadline_us, 1)
        if self.deadline_us <= self.arrival_us:
            raise SchedulerContractError("deadline_us must follow arrival_us")
        _integer("input_tokens", self.input_tokens, 1)
        _integer("output_tokens", self.output_tokens, 1)
        reserved_features = {"input_tokens", "output_tokens", "member_requests"}
        if reserved_features & set(self.features):
            raise SchedulerContractError(
                "unit features must not redefine built-in features"
            )
        object.__setattr__(self, "features", _immutable_int_mapping(
            "unit feature", self.features
        ))
        semantic_values: dict[str, bool | str] = {}
        for key, value in self.semantics.items():
            name = _text("semantic key", key)
            if type(value) is str:
                semantic_values[name] = _text(f"semantic {name}", value)
            elif type(value) is bool:
                semantic_values[name] = value
            else:
                raise SchedulerContractError(
                    f"semantic {name} must be bool or string"
                )
        object.__setattr__(
            self,
            "semantics",
            MappingProxyType(dict(sorted(semantic_values.items()))),
        )
        if self.epoch_id is not None:
            _text("epoch_id", self.epoch_id)

    def feature(self, name: str) -> int:
        if name == "input_tokens":
            return self.input_tokens
        if name == "output_tokens":
            return self.output_tokens
        if name == "member_requests":
            return len(self.member_request_ids)
        try:
            return self.features[name]
        except KeyError as exc:
            raise SchedulerContractError(
                f"scheduling unit lacks feature: {name}"
            ) from exc


@dataclass(frozen=True)
class FeatureRange:
    minimum: int | None = None
    maximum: int | None = None

    def __post_init__(self) -> None:
        if self.minimum is None and self.maximum is None:
            raise SchedulerContractError("feature range must have a bound")
        if self.minimum is not None:
            _integer("feature minimum", self.minimum)
        if self.maximum is not None:
            _integer("feature maximum", self.maximum)
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.maximum < self.minimum
        ):
            raise SchedulerContractError("feature range is empty")

    def contains(self, value: int) -> bool:
        return not (
            (self.minimum is not None and value < self.minimum)
            or (self.maximum is not None and value > self.maximum)
        )


@dataclass(frozen=True)
class ApplicabilityContract:
    workload_ids: tuple[str, ...]
    unit_kinds: tuple[UnitKind, ...]
    model_ids: tuple[str, ...]
    model_hashes: tuple[str, ...]
    architectures: tuple[str, ...]
    weight_formats: tuple[str, ...]
    feature_ranges: Mapping[str, FeatureRange]
    work_set_hashes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "workload_ids",
            _text_tuple("applicable workload", self.workload_ids, nonempty=True),
        )
        if (
            not self.unit_kinds
            or any(not isinstance(item, UnitKind) for item in self.unit_kinds)
            or len(self.unit_kinds) != len(set(self.unit_kinds))
        ):
            raise SchedulerContractError(
                "applicable unit kinds must be non-empty and unique"
            )
        object.__setattr__(
            self, "model_ids", _text_tuple("applicable model", self.model_ids)
        )
        model_hashes = tuple(
            _sha256("applicable model hash", item) for item in self.model_hashes
        )
        if len(model_hashes) != len(set(model_hashes)):
            raise SchedulerContractError(
                "applicable model hashes must not contain duplicates"
            )
        object.__setattr__(self, "model_hashes", model_hashes)
        object.__setattr__(
            self,
            "architectures",
            _text_tuple("applicable architecture", self.architectures),
        )
        object.__setattr__(
            self,
            "weight_formats",
            _text_tuple("applicable weight format", self.weight_formats),
        )
        ranges = {
            _text("feature range name", name): value
            for name, value in self.feature_ranges.items()
        }
        if any(not isinstance(value, FeatureRange) for value in ranges.values()):
            raise SchedulerContractError(
                "feature_ranges values must be FeatureRange objects"
            )
        object.__setattr__(
            self,
            "feature_ranges",
            MappingProxyType(dict(sorted(ranges.items()))),
        )
        hashes = tuple(
            _sha256("applicable work set hash", item)
            for item in self.work_set_hashes
        )
        if len(hashes) != len(set(hashes)):
            raise SchedulerContractError(
                "applicable work set hashes must not contain duplicates"
            )
        object.__setattr__(self, "work_set_hashes", hashes)

    @classmethod
    def exact_for(cls, unit: SchedulingUnit) -> "ApplicabilityContract":
        feature_ranges = {
            **{
                name: FeatureRange(value, value)
                for name, value in unit.features.items()
            },
            "input_tokens": FeatureRange(unit.input_tokens, unit.input_tokens),
            "output_tokens": FeatureRange(unit.output_tokens, unit.output_tokens),
        }
        return cls(
            workload_ids=(unit.workload_id,),
            unit_kinds=(unit.kind,),
            model_ids=(unit.model.model_id,),
            model_hashes=(unit.model.model_hash,),
            architectures=(unit.model.architecture,),
            weight_formats=(unit.model.weight_format,),
            feature_ranges=feature_ranges,
            work_set_hashes=(unit.work_set_hash,),
        )

    def rejection_reason(self, unit: SchedulingUnit) -> str | None:
        if unit.workload_id not in self.workload_ids:
            return "WORKLOAD_OUT_OF_DOMAIN"
        if unit.kind not in self.unit_kinds:
            return "UNIT_KIND_OUT_OF_DOMAIN"
        if self.model_ids and unit.model.model_id not in self.model_ids:
            return "MODEL_OUT_OF_DOMAIN"
        if self.model_hashes and unit.model.model_hash not in self.model_hashes:
            return "MODEL_HASH_OUT_OF_DOMAIN"
        if self.architectures and unit.model.architecture not in self.architectures:
            return "ARCHITECTURE_OUT_OF_DOMAIN"
        if self.weight_formats and unit.model.weight_format not in self.weight_formats:
            return "WEIGHT_FORMAT_OUT_OF_DOMAIN"
        if self.work_set_hashes and unit.work_set_hash not in self.work_set_hashes:
            return "WORK_SET_OUT_OF_DOMAIN"
        for name, limits in self.feature_ranges.items():
            try:
                value = unit.feature(name)
            except SchedulerContractError:
                return "FEATURE_MISSING"
            if not limits.contains(value):
                return "FEATURE_OUT_OF_DOMAIN"
        return None


@dataclass(frozen=True)
class AccountingContract:
    scope: AccountingScope
    kind: AccountingKind
    boundary_id: str
    work_set_hash: str | None = None
    epoch_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scope, AccountingScope):
            raise SchedulerContractError(
                "accounting scope must be AccountingScope"
            )
        if not isinstance(self.kind, AccountingKind):
            raise SchedulerContractError(
                "accounting kind must be AccountingKind"
            )
        _text("accounting boundary_id", self.boundary_id)
        if self.scope == AccountingScope.EPOCH:
            if self.epoch_id is None or self.work_set_hash is not None:
                raise SchedulerContractError(
                    "epoch accounting requires epoch_id and no work_set_hash"
                )
            _text("accounting epoch_id", self.epoch_id)
        else:
            if self.work_set_hash is None or self.epoch_id is not None:
                raise SchedulerContractError(
                    "unit accounting requires work_set_hash and no epoch_id"
                )
            _sha256("accounting work_set_hash", self.work_set_hash)
        if (
            self.kind == AccountingKind.NON_ADDITIVE_COHORT_TOTAL
            and self.scope != AccountingScope.COHORT
        ):
            raise SchedulerContractError(
                "non-additive cohort energy requires cohort scope"
            )
        if (
            self.kind == AccountingKind.EPOCH_OBLIGATION
            and self.scope != AccountingScope.EPOCH
        ):
            raise SchedulerContractError(
                "epoch obligation requires epoch scope"
            )
        if (
            self.scope == AccountingScope.EPOCH
            and self.kind != AccountingKind.EPOCH_OBLIGATION
        ):
            raise SchedulerContractError(
                "epoch scope requires epoch-obligation accounting"
            )

    def rejection_reason(self, unit: SchedulingUnit) -> str | None:
        expected = {
            UnitKind.REQUEST: AccountingScope.REQUEST,
            UnitKind.PHYSICAL_UBATCH: AccountingScope.PHYSICAL_UBATCH,
            UnitKind.COHORT: AccountingScope.COHORT,
        }[unit.kind]
        if self.scope == AccountingScope.EPOCH:
            if unit.epoch_id != self.epoch_id:
                return "EPOCH_ACCOUNTING_MISMATCH"
            return None
        if self.scope != expected:
            return "ACCOUNTING_SCOPE_MISMATCH"
        if self.work_set_hash != unit.work_set_hash:
            return "ACCOUNTING_WORK_SET_MISMATCH"
        return None


@dataclass(frozen=True)
class MetricEstimate:
    mean: int
    upper: int
    sample_count: int
    measured: bool
    lower: int | None = None

    def __post_init__(self) -> None:
        _integer("metric mean", self.mean, 1)
        _integer("metric upper", self.upper, self.mean)
        _integer("metric sample_count", self.sample_count)
        if self.lower is not None:
            _integer("metric lower", self.lower, 1)
            if self.lower > self.mean:
                raise SchedulerContractError("metric lower exceeds mean")


@dataclass(frozen=True)
class EnergyComponent:
    estimate_uj: MetricEstimate
    accounting: AccountingContract

    def __post_init__(self) -> None:
        if not isinstance(self.estimate_uj, MetricEstimate):
            raise SchedulerContractError(
                "energy estimate must be MetricEstimate"
            )
        if not isinstance(self.accounting, AccountingContract):
            raise SchedulerContractError(
                "energy accounting must be AccountingContract"
            )

    def rejection_reason(self, unit: SchedulingUnit) -> str | None:
        return self.accounting.rejection_reason(unit)


@dataclass(frozen=True)
class OverlapEstimate:
    status: str
    exposed_join_wait_ppm: int | None
    upper_join_wait_ppm: int | None
    sample_count: int

    def __post_init__(self) -> None:
        if self.status not in {
            "unknown",
            "predicted",
            "measured",
            "not_applicable",
        }:
            raise SchedulerContractError("unknown overlap status")
        _integer("overlap sample_count", self.sample_count)
        if self.status in {"unknown", "not_applicable"}:
            if (
                self.exposed_join_wait_ppm is not None
                or self.upper_join_wait_ppm is not None
            ):
                raise SchedulerContractError(
                    "unmeasured overlap cannot have wait measurements"
                )
            return
        if (
            self.exposed_join_wait_ppm is None
            or self.upper_join_wait_ppm is None
        ):
            raise SchedulerContractError(
                "measured or predicted overlap requires wait bounds"
            )
        _integer("exposed join wait", self.exposed_join_wait_ppm)
        _integer(
            "upper exposed join wait",
            self.upper_join_wait_ppm,
            self.exposed_join_wait_ppm,
        )
        if self.upper_join_wait_ppm > 1_000_000:
            raise SchedulerContractError("join wait cannot exceed 100 percent")


@dataclass(frozen=True)
class QualityContract:
    quality_class: QualityClass
    finite_output_required: bool = True
    nonempty_output_required: bool = True
    validation_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.quality_class, QualityClass):
            raise SchedulerContractError(
                "quality class must be QualityClass"
            )
        if type(self.finite_output_required) is not bool:
            raise SchedulerContractError("finite_output_required must be bool")
        if type(self.nonempty_output_required) is not bool:
            raise SchedulerContractError("nonempty_output_required must be bool")
        if self.validation_id is not None:
            _text("quality validation_id", self.validation_id)

    def satisfies(self, requirement: QualityClass) -> bool:
        return QUALITY_RANK[self.quality_class] >= QUALITY_RANK[requirement]


@dataclass(frozen=True)
class ResourceRequirement:
    resource_id: str
    kind: str
    role: str
    slots: int
    identity: str | None = None

    def __post_init__(self) -> None:
        _text("resource_id", self.resource_id)
        _text("resource kind", self.kind)
        _text("resource role", self.role)
        _integer("resource slots", self.slots, 1)
        if self.identity is not None:
            _text("resource identity", self.identity)


@dataclass(frozen=True)
class MemoryRequirement:
    resource_id: str
    pool_id: str
    bytes: int
    persistent: bool

    def __post_init__(self) -> None:
        _text("memory resource_id", self.resource_id)
        _text("memory pool_id", self.pool_id)
        _integer("memory bytes", self.bytes, 1)
        if type(self.persistent) is not bool:
            raise SchedulerContractError("memory persistent must be bool")


@dataclass(frozen=True)
class ResidencyRequirement:
    resource_id: str
    residency_id: str
    role: str

    def __post_init__(self) -> None:
        _text("residency resource_id", self.resource_id)
        _text("residency_id", self.residency_id)
        _text("residency role", self.role)


@dataclass(frozen=True)
class PhaseLease:
    lease_id: str
    resource_id: str
    slots: int
    start_offset_us: int
    duration: MetricEstimate

    def __post_init__(self) -> None:
        _text("lease_id", self.lease_id)
        _text("lease resource_id", self.resource_id)
        _integer("lease slots", self.slots, 1)
        _integer("lease start_offset_us", self.start_offset_us)
        if not isinstance(self.duration, MetricEstimate):
            raise SchedulerContractError(
                "lease duration must be MetricEstimate"
            )


@dataclass(frozen=True)
class RouteAlternative:
    route_id: str
    workload_id: str
    baseline: bool
    placement_granularity: PlacementGranularity
    maturity: RouteMaturity
    applicability: ApplicabilityContract
    latency_us: MetricEstimate
    energy: tuple[EnergyComponent, ...]
    overlap: OverlapEstimate
    quality: QualityContract
    resources: tuple[ResourceRequirement, ...]
    phase_leases: tuple[PhaseLease, ...]
    memory: tuple[MemoryRequirement, ...]
    residency: tuple[ResidencyRequirement, ...]
    placement_verified: bool
    evidence_ids: tuple[str, ...]
    source_profile_id: str

    def __post_init__(self) -> None:
        _text("route_id", self.route_id)
        _text("route workload_id", self.workload_id)
        if not isinstance(self.placement_granularity, PlacementGranularity):
            raise SchedulerContractError(
                "placement granularity must be PlacementGranularity"
            )
        if not isinstance(self.maturity, RouteMaturity):
            raise SchedulerContractError("route maturity must be RouteMaturity")
        if not isinstance(self.applicability, ApplicabilityContract):
            raise SchedulerContractError(
                "route applicability must be ApplicabilityContract"
            )
        if not isinstance(self.latency_us, MetricEstimate):
            raise SchedulerContractError(
                "route latency must be MetricEstimate"
            )
        if not isinstance(self.overlap, OverlapEstimate):
            raise SchedulerContractError(
                "route overlap must be OverlapEstimate"
            )
        if not isinstance(self.quality, QualityContract):
            raise SchedulerContractError(
                "route quality must be QualityContract"
            )
        if type(self.baseline) is not bool:
            raise SchedulerContractError("route baseline must be bool")
        if self.workload_id not in self.applicability.workload_ids:
            raise SchedulerContractError(
                "route workload is absent from its applicability contract"
            )
        object.__setattr__(self, "resources", tuple(self.resources))
        object.__setattr__(self, "phase_leases", tuple(self.phase_leases))
        object.__setattr__(self, "memory", tuple(self.memory))
        object.__setattr__(self, "residency", tuple(self.residency))
        object.__setattr__(self, "energy", tuple(self.energy))
        if any(
            not isinstance(resource, ResourceRequirement)
            for resource in self.resources
        ):
            raise SchedulerContractError(
                "route resources must be ResourceRequirement objects"
            )
        if any(not isinstance(lease, PhaseLease) for lease in self.phase_leases):
            raise SchedulerContractError(
                "route phase leases must be PhaseLease objects"
            )
        if any(
            not isinstance(requirement, MemoryRequirement)
            for requirement in self.memory
        ):
            raise SchedulerContractError(
                "route memory entries must be MemoryRequirement objects"
            )
        if any(
            not isinstance(requirement, ResidencyRequirement)
            for requirement in self.residency
        ):
            raise SchedulerContractError(
                "route residency entries must be ResidencyRequirement objects"
            )
        if any(
            not isinstance(component, EnergyComponent)
            for component in self.energy
        ):
            raise SchedulerContractError(
                "route energy entries must be EnergyComponent objects"
            )
        resource_ids = [resource.resource_id for resource in self.resources]
        if not resource_ids or len(resource_ids) != len(set(resource_ids)):
            raise SchedulerContractError(
                "route resources must be non-empty and unique"
            )
        resource_map = {
            resource.resource_id: resource for resource in self.resources
        }
        lease_ids: set[str] = set()
        leased_resources: set[str] = set()
        for lease in self.phase_leases:
            if lease.lease_id in lease_ids:
                raise SchedulerContractError("duplicate phase lease id")
            resource = resource_map.get(lease.resource_id)
            if resource is None:
                raise SchedulerContractError(
                    "phase lease references an undeclared resource"
                )
            if lease.slots > resource.slots:
                raise SchedulerContractError(
                    "phase lease exceeds declared resource slots"
                )
            if lease.start_offset_us + lease.duration.mean > self.latency_us.mean:
                raise SchedulerContractError(
                    "phase lease exceeds mean service time"
                )
            if lease.start_offset_us + lease.duration.upper > self.latency_us.upper:
                raise SchedulerContractError(
                    "phase lease exceeds upper service time"
                )
            lease_ids.add(lease.lease_id)
            leased_resources.add(lease.resource_id)
        if self.phase_leases and leased_resources != set(resource_ids):
            raise SchedulerContractError(
                "phase leases must cover every declared resource"
            )
        if type(self.placement_verified) is not bool:
            raise SchedulerContractError("placement_verified must be bool")
        boundary_ids = {
            component.accounting.boundary_id for component in self.energy
        }
        if len(boundary_ids) > 1:
            raise SchedulerContractError(
                "route energy components must use one fleet boundary"
            )
        evidence = _text_tuple(
            "route evidence id", self.evidence_ids, nonempty=True
        )
        object.__setattr__(self, "evidence_ids", evidence)
        _text("source_profile_id", self.source_profile_id)
        if self.maturity == RouteMaturity.STABLE:
            if not self.placement_verified or not self.latency_us.measured:
                raise SchedulerContractError(
                    "stable route requires verified placement and measured latency"
                )
            if not self.energy or any(
                not component.estimate_uj.measured for component in self.energy
            ):
                raise SchedulerContractError(
                    "stable route requires measured energy"
                )
            if not self.applicability.model_hashes:
                raise SchedulerContractError(
                    "stable route requires a model-hash applicability gate"
                )

    def rejection_reason(self, unit: SchedulingUnit) -> str | None:
        applicability = self.applicability.rejection_reason(unit)
        if applicability is not None:
            return applicability
        for component in self.energy:
            accounting = component.rejection_reason(unit)
            if accounting is not None:
                return accounting
        if not self.quality.satisfies(unit.quality_requirement):
            return "QUALITY_INSUFFICIENT"
        return None

    @property
    def energy_uj(self) -> MetricEstimate | None:
        if not self.energy:
            return None
        return MetricEstimate(
            mean=sum(component.estimate_uj.mean for component in self.energy),
            upper=sum(component.estimate_uj.upper for component in self.energy),
            sample_count=min(
                component.estimate_uj.sample_count for component in self.energy
            ),
            measured=all(
                component.estimate_uj.measured for component in self.energy
            ),
            lower=(
                sum(component.estimate_uj.lower for component in self.energy)
                if all(
                    component.estimate_uj.lower is not None
                    for component in self.energy
                )
                else None
            ),
        )


@dataclass(frozen=True)
class CandidateSet:
    profile_id: str
    unit: SchedulingUnit
    routes: tuple[RouteAlternative, ...]
    schema: str = CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        _text("candidate profile_id", self.profile_id)
        if not isinstance(self.unit, SchedulingUnit):
            raise SchedulerContractError(
                "candidate unit must be SchedulingUnit"
            )
        if self.schema != CONTRACT_SCHEMA:
            raise SchedulerContractError("candidate schema mismatch")
        object.__setattr__(self, "routes", tuple(self.routes))
        if any(not isinstance(route, RouteAlternative) for route in self.routes):
            raise SchedulerContractError(
                "candidate routes must be RouteAlternative objects"
            )
        route_ids = [route.route_id for route in self.routes]
        if not route_ids or len(route_ids) != len(set(route_ids)):
            raise SchedulerContractError(
                "candidate routes must be non-empty and unique"
            )
        if sum(route.baseline for route in self.routes) != 1:
            raise SchedulerContractError(
                "candidate set must contain exactly one baseline"
            )
        for route in self.routes:
            reason = route.rejection_reason(self.unit)
            if reason is not None:
                raise SchedulerContractError(
                    f"candidate route {route.route_id} rejects unit: {reason}"
                )
        boundaries = {
            component.accounting.boundary_id
            for route in self.routes
            for component in route.energy
        }
        if len(boundaries) > 1:
            raise SchedulerContractError(
                "candidate routes use different fleet-energy boundaries"
            )
