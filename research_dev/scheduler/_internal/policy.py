#!/usr/bin/env python3
"""Data-driven heterogeneous inference scheduling primitives.

The scheduler consumes pre-certified route alternatives. It does not perform
device I/O and it does not create new tensor partitions at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .runtime_gates import (
    RequestSemantics,
    RouteRuntimeContract,
    RuntimeGateError,
    RuntimeGateReceipt,
    RuntimeSnapshot,
    evaluate_runtime_gate,
)
from .runtime_system_cost import (
    RuntimeSystemCostError,
    calculate_marginal_system_cost,
)

from .policy_common import (
    _mapping as _mapping,
    SchedulerError as SchedulerError,
    _strict_bool as _strict_bool,
    _strict_int as _strict_int,
    _text as _text,
    PROFILE_SCHEMA as PROFILE_SCHEMA,
    QUALITY_RANK as QUALITY_RANK,
    POLICY_MODES as POLICY_MODES,
    GRANULARITIES as GRANULARITIES,
    ENERGY_STATUSES as ENERGY_STATUSES,
    OVERLAP_STATUSES as OVERLAP_STATUSES,
    ResourceProfile as ResourceProfile,
    LeaseDemand as LeaseDemand,
)

from .resource_timeline import (
    LeasePlan as LeasePlan,
    LeasePreview as LeasePreview,
    LeaseRecord as LeaseRecord,
    _CalendarReservation as _CalendarReservation,
    _ResourceTimelineCheckpoint as _ResourceTimelineCheckpoint,
    ResourceTimeline as ResourceTimeline,
)


__all__ = [
    'AffineCost',
    'Candidate',
    'Decision',
    'ENERGY_STATUSES',
    'EnergyDomain',
    'EnergyProfile',
    'GRANULARITIES',
    'KernelEnergyEstimate',
    'KernelEnergyProfile',
    'LatencyProfile',
    'LatencyVariant',
    'LeaseDemand',
    'LeasePlan',
    'LeasePreview',
    'LeaseRecord',
    'MarginalSystemCostContext',
    'OVERLAP_STATUSES',
    'OperatorEnergyCost',
    'OperatorWork',
    'OverlapProfile',
    'POLICY_MODES',
    'PROFILE_SCHEMA',
    'PolicyConfig',
    'ProfileBundle',
    'QUALITY_RANK',
    'Request',
    'RequestSemantics',
    'ResourceLeaseProfile',
    'ResourceProfile',
    'ResourceTimeline',
    'RoutePolicy',
    'RouteProfile',
    'RouteRuntimeContract',
    'RuntimeGateError',
    'RuntimeGateReceipt',
    'RuntimeSnapshot',
    'RuntimeSystemCostError',
    'SchedulerError',
    '_CalendarReservation',
    '_ResourceTimelineCheckpoint',
    '_mapping',
    '_strict_bool',
    '_strict_int',
    '_text',
    'calculate_marginal_system_cost',
    'decision_to_json',
    'estimate_kernel_energy',
    'evaluate_runtime_gate',
]


@dataclass(frozen=True)
class Request:
    request_id: str
    workload_id: str
    arrival_us: int
    deadline_us: int
    input_tokens: int
    output_tokens: int
    quality_requirement: str
    features: Mapping[str, int] = field(default_factory=dict)
    semantics: RequestSemantics = field(default_factory=RequestSemantics)

    def validate(self) -> None:
        _text("request_id", self.request_id)
        _text("workload_id", self.workload_id)
        _strict_int("arrival_us", self.arrival_us)
        _strict_int("deadline_us", self.deadline_us, 1)
        _strict_int("input_tokens", self.input_tokens, 1)
        _strict_int("output_tokens", self.output_tokens, 1)
        if self.deadline_us <= self.arrival_us:
            raise SchedulerError("deadline_us must follow arrival_us")
        if self.quality_requirement not in QUALITY_RANK:
            raise SchedulerError("unknown quality requirement")
        for name, value in self.features.items():
            _text("request feature name", name)
            _strict_int(f"request feature {name}", value)
        try:
            self.semantics.validate()
        except RuntimeGateError as exc:
            raise SchedulerError(str(exc)) from exc

    def feature(self, name: str) -> int:
        if name == "input_tokens":
            return self.input_tokens
        if name == "output_tokens":
            return self.output_tokens
        if name not in self.features:
            raise SchedulerError(f"request lacks cost feature: {name}")
        return self.features[name]



@dataclass(frozen=True)
class AffineCost:
    fixed: int
    coefficients: Mapping[str, int]

    @classmethod
    def from_json(cls, name: str, value: object) -> "AffineCost":
        row = _mapping(name, value)
        kind = row.get("kind")
        if kind == "affine_tokens_v1":
            coefficients = {
                "input_tokens": _strict_int(
                    f"{name}.input_token", row.get("input_token")
                ),
                "output_tokens": _strict_int(
                    f"{name}.output_token", row.get("output_token")
                ),
            }
        elif kind == "affine_features_v1":
            raw_coefficients = _mapping(
                f"{name}.coefficients", row.get("coefficients")
            )
            if not raw_coefficients:
                raise SchedulerError(f"{name}.coefficients must not be empty")
            coefficients = {
                _text(f"{name} feature", feature): _strict_int(
                    f"{name} coefficient {feature}", coefficient
                )
                for feature, coefficient in raw_coefficients.items()
            }
        else:
            raise SchedulerError(f"{name}.kind is unsupported")
        return cls(
            fixed=_strict_int(f"{name}.fixed", row.get("fixed")),
            coefficients=coefficients,
        )

    def predict(self, request: Request) -> int:
        return self.fixed + sum(
            coefficient * request.feature(name)
            for name, coefficient in self.coefficients.items()
        )


def _ceil_div(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise SchedulerError("invalid ceiling division")
    return (numerator + denominator - 1) // denominator


def _work_expression(name: str, value: object) -> AffineCost:
    if type(value) is int:
        return AffineCost(_strict_int(name, value), {})
    return AffineCost.from_json(name, value)


@dataclass(frozen=True)
class ResourceLeaseProfile:
    lease_id: str
    resource_id: str
    slots: int
    start_offset_us: AffineCost
    duration_us: AffineCost
    duration_ucb_add_us: int

    @classmethod
    def from_json(cls, value: object) -> "ResourceLeaseProfile":
        row = _mapping("resource lease", value)
        return cls(
            lease_id=_text("resource lease id", row.get("lease_id")),
            resource_id=_text(
                "resource lease resource_id", row.get("resource_id")
            ),
            slots=_strict_int("resource lease slots", row.get("slots"), 1),
            start_offset_us=_work_expression(
                "resource lease start_offset_us", row.get("start_offset_us")
            ),
            duration_us=_work_expression(
                "resource lease duration_us", row.get("duration_us")
            ),
            duration_ucb_add_us=_strict_int(
                "resource lease duration_ucb_add_us",
                row.get("duration_ucb_add_us", 0),
            ),
        )

    def predict(self, request: Request) -> "LeaseDemand":
        offset = self.start_offset_us.predict(request)
        duration = self.duration_us.predict(request)
        if duration <= 0:
            raise SchedulerError("resource lease duration must be positive")
        return LeaseDemand(
            lease_id=self.lease_id,
            resource_id=self.resource_id,
            slots=self.slots,
            start_offset_us=offset,
            duration_us=duration,
            duration_upper_us=duration + self.duration_ucb_add_us,
        )


@dataclass(frozen=True)
class EnergyDomain:
    domain_id: str
    idle_power_mw: int

    @classmethod
    def from_json(cls, value: object) -> "EnergyDomain":
        row = _mapping("energy domain", value)
        return cls(
            domain_id=_text("energy domain id", row.get("domain_id")),
            idle_power_mw=_strict_int(
                "energy domain idle_power_mw", row.get("idle_power_mw")
            ),
        )


@dataclass(frozen=True)
class KernelEnergyProfile:
    kernel_id: str
    domain_id: str
    effective_ops_per_s: int
    effective_bytes_per_s: int
    launch_us: int
    active_power_mw: int

    @classmethod
    def from_json(cls, value: object) -> "KernelEnergyProfile":
        row = _mapping("kernel energy profile", value)
        return cls(
            kernel_id=_text("kernel id", row.get("kernel_id")),
            domain_id=_text("kernel domain id", row.get("domain_id")),
            effective_ops_per_s=_strict_int(
                "kernel effective_ops_per_s",
                row.get("effective_ops_per_s"),
                1,
            ),
            effective_bytes_per_s=_strict_int(
                "kernel effective_bytes_per_s",
                row.get("effective_bytes_per_s"),
                1,
            ),
            launch_us=_strict_int("kernel launch_us", row.get("launch_us")),
            active_power_mw=_strict_int(
                "kernel active_power_mw", row.get("active_power_mw"), 1
            ),
        )


@dataclass(frozen=True)
class KernelEnergyEstimate:
    active_us: int
    compute_us: int
    dynamic_uj: int
    memory_us: int


def estimate_kernel_energy(
    domain: EnergyDomain,
    kernel: KernelEnergyProfile,
    invocations: int,
    compute_ops: int,
    memory_bytes: int,
) -> KernelEnergyEstimate:
    """Estimate one profiled kernel using the operator_sum_v1 equation."""
    _strict_int("kernel invocations", invocations)
    _strict_int("kernel compute_ops", compute_ops)
    _strict_int("kernel memory_bytes", memory_bytes)
    if kernel.domain_id != domain.domain_id:
        raise SchedulerError("kernel and energy domain do not match")
    if kernel.active_power_mw < domain.idle_power_mw:
        raise SchedulerError("kernel active power is below domain idle power")
    if invocations == 0:
        if compute_ops or memory_bytes:
            raise SchedulerError("operator has work without an invocation")
        return KernelEnergyEstimate(0, 0, 0, 0)
    compute_us = _ceil_div(
        compute_ops * 1_000_000,
        kernel.effective_ops_per_s,
    )
    memory_us = _ceil_div(
        memory_bytes * 1_000_000,
        kernel.effective_bytes_per_s,
    )
    active_us = invocations * kernel.launch_us + max(compute_us, memory_us)
    dynamic_uj = _ceil_div(
        (kernel.active_power_mw - domain.idle_power_mw) * active_us,
        1000,
    )
    return KernelEnergyEstimate(
        active_us=active_us,
        compute_us=compute_us,
        dynamic_uj=dynamic_uj,
        memory_us=memory_us,
    )


@dataclass(frozen=True)
class OperatorWork:
    op_id: str
    kernel_id: str
    invocations: AffineCost
    compute_ops: AffineCost
    memory_bytes: AffineCost

    @classmethod
    def from_json(cls, value: object) -> "OperatorWork":
        row = _mapping("operator work", value)
        return cls(
            op_id=_text("operator id", row.get("op_id")),
            kernel_id=_text("operator kernel id", row.get("kernel_id")),
            invocations=_work_expression(
                "operator invocations", row.get("invocations")
            ),
            compute_ops=_work_expression(
                "operator compute_ops", row.get("compute_ops")
            ),
            memory_bytes=_work_expression(
                "operator memory_bytes", row.get("memory_bytes")
            ),
        )


@dataclass(frozen=True)
class OperatorEnergyCost:
    fixed_uj: int
    domains: Mapping[str, EnergyDomain]
    kernels: Mapping[str, KernelEnergyProfile]
    operators: tuple[OperatorWork, ...]

    @classmethod
    def from_json(cls, value: object) -> "OperatorEnergyCost":
        row = _mapping("operator energy", value)
        if row.get("kind") != "operator_sum_v1":
            raise SchedulerError("operator energy kind is unsupported")
        raw_domains = row.get("domains")
        if type(raw_domains) is not list or not raw_domains:
            raise SchedulerError("operator energy domains must not be empty")
        domains: dict[str, EnergyDomain] = {}
        for raw_domain in raw_domains:
            domain = EnergyDomain.from_json(raw_domain)
            if domain.domain_id in domains:
                raise SchedulerError("duplicate energy domain id")
            domains[domain.domain_id] = domain
        raw_kernels = row.get("kernels")
        if type(raw_kernels) is not list or not raw_kernels:
            raise SchedulerError("operator energy kernels must not be empty")
        kernels: dict[str, KernelEnergyProfile] = {}
        for raw_kernel in raw_kernels:
            kernel = KernelEnergyProfile.from_json(raw_kernel)
            if kernel.kernel_id in kernels:
                raise SchedulerError("duplicate kernel id")
            domain = domains.get(kernel.domain_id)
            if domain is None:
                raise SchedulerError("kernel references an unknown energy domain")
            if kernel.active_power_mw < domain.idle_power_mw:
                raise SchedulerError("kernel active power is below domain idle power")
            kernels[kernel.kernel_id] = kernel
        raw_operators = row.get("operators")
        if type(raw_operators) is not list or not raw_operators:
            raise SchedulerError("operator energy work must not be empty")
        operators: list[OperatorWork] = []
        op_ids: set[str] = set()
        for raw_operator in raw_operators:
            operator = OperatorWork.from_json(raw_operator)
            if operator.op_id in op_ids:
                raise SchedulerError("duplicate operator id")
            if operator.kernel_id not in kernels:
                raise SchedulerError("operator references an unknown kernel")
            op_ids.add(operator.op_id)
            operators.append(operator)
        return cls(
            fixed_uj=_strict_int(
                "operator energy fixed_uj", row.get("fixed_uj", 0)
            ),
            domains=domains,
            kernels=kernels,
            operators=tuple(operators),
        )

    def breakdown(self, request: Request, service_us: int) -> dict[str, Any]:
        _strict_int("operator energy service_us", service_us, 1)
        domain_rows: dict[str, dict[str, int]] = {
            domain_id: {
                "active_us": 0,
                "dynamic_uj": 0,
                "idle_uj": _ceil_div(
                    domain.idle_power_mw * service_us, 1000
                ),
            }
            for domain_id, domain in self.domains.items()
        }
        operator_rows: list[dict[str, int | str]] = []
        for operator in self.operators:
            invocations = operator.invocations.predict(request)
            compute_ops = operator.compute_ops.predict(request)
            memory_bytes = operator.memory_bytes.predict(request)
            kernel = self.kernels[operator.kernel_id]
            domain = self.domains[kernel.domain_id]
            estimate = estimate_kernel_energy(
                domain,
                kernel,
                invocations,
                compute_ops,
                memory_bytes,
            )
            domain_rows[kernel.domain_id]["active_us"] += estimate.active_us
            domain_rows[kernel.domain_id]["dynamic_uj"] += estimate.dynamic_uj
            operator_rows.append({
                "active_us": estimate.active_us,
                "compute_ops": compute_ops,
                "compute_us": estimate.compute_us,
                "dynamic_uj": estimate.dynamic_uj,
                "invocations": invocations,
                "memory_bytes": memory_bytes,
                "memory_us": estimate.memory_us,
                "op_id": operator.op_id,
            })
        for domain_id, domain_row in domain_rows.items():
            if domain_row["active_us"] > service_us:
                raise SchedulerError(
                    f"energy domain active time exceeds service time: {domain_id}"
                )
            domain_row["total_uj"] = (
                domain_row["idle_uj"] + domain_row["dynamic_uj"]
            )
        total_uj = self.fixed_uj + sum(
            row["total_uj"] for row in domain_rows.values()
        )
        return {
            "domains": domain_rows,
            "fixed_uj": self.fixed_uj,
            "kind": "operator_sum_v1",
            "operators": operator_rows,
            "total_uj": total_uj,
        }

    def predict(self, request: Request, service_us: int) -> int:
        return int(self.breakdown(request, service_us)["total_uj"])


@dataclass(frozen=True)
class LatencyVariant:
    selector_value: int
    label: str
    cost_us: AffineCost
    ucb_add_us: int
    sample_count: int
    measured: bool

    @classmethod
    def from_json(cls, value: object) -> "LatencyVariant":
        row = _mapping("latency variant", value)
        return cls(
            selector_value=_strict_int(
                "latency variant selector_value", row.get("selector_value")
            ),
            label=_text("latency variant label", row.get("label")),
            cost_us=AffineCost.from_json(
                "latency variant cost_us", row.get("cost_us")
            ),
            ucb_add_us=_strict_int(
                "latency variant ucb_add_us", row.get("ucb_add_us")
            ),
            sample_count=_strict_int(
                "latency variant sample_count", row.get("sample_count"), 1
            ),
            measured=_strict_bool(
                "latency variant measured", row.get("measured")
            ),
        )


@dataclass(frozen=True)
class LatencyProfile:
    cost_us: AffineCost | None
    ucb_add_us: int
    sample_count: int
    measured: bool
    selector_feature: str | None = None
    variants: Mapping[int, LatencyVariant] = field(default_factory=dict)

    @classmethod
    def from_json(cls, value: object) -> "LatencyProfile":
        row = _mapping("latency", value)
        if row.get("kind") == "conditioned_affine_features_v1":
            selector = _text(
                "latency selector_feature", row.get("selector_feature")
            )
            raw_variants = row.get("variants")
            if type(raw_variants) is not list or not raw_variants:
                raise SchedulerError("latency variants must be a non-empty list")
            variants = tuple(
                LatencyVariant.from_json(item) for item in raw_variants
            )
            values = [item.selector_value for item in variants]
            labels = [item.label for item in variants]
            if (
                len(values) != len(set(values))
                or len(labels) != len(set(labels))
            ):
                raise SchedulerError(
                    "latency variant selectors and labels must be unique"
                )
            return cls(
                cost_us=None,
                ucb_add_us=max(item.ucb_add_us for item in variants),
                sample_count=min(item.sample_count for item in variants),
                measured=all(item.measured for item in variants),
                selector_feature=selector,
                variants={item.selector_value: item for item in variants},
            )
        if row.get("kind") is not None:
            raise SchedulerError("latency.kind is unsupported")
        return cls(
            cost_us=AffineCost.from_json("latency.cost_us", row.get("cost_us")),
            ucb_add_us=_strict_int(
                "latency.ucb_add_us", row.get("ucb_add_us")
            ),
            sample_count=_strict_int(
                "latency.sample_count", row.get("sample_count"), 1
            ),
            measured=_strict_bool("latency.measured", row.get("measured")),
        )

    def variant(self, request: Request) -> LatencyVariant | None:
        if self.selector_feature is None:
            return None
        selector_value = request.feature(self.selector_feature)
        variant = self.variants.get(selector_value)
        if variant is None:
            raise SchedulerError(
                "request selects an unknown latency variant: "
                f"{self.selector_feature}={selector_value}"
            )
        return variant

    def measured_for(self, request: Request) -> bool:
        variant = self.variant(request)
        return self.measured if variant is None else variant.measured

    def predict_us(self, request: Request) -> int:
        variant = self.variant(request)
        cost = self.cost_us if variant is None else variant.cost_us
        assert cost is not None
        return max(1, cost.predict(request))

    def upper_us(self, request: Request) -> int:
        variant = self.variant(request)
        ucb = self.ucb_add_us if variant is None else variant.ucb_add_us
        return self.predict_us(request) + ucb


@dataclass(frozen=True)
class EnergyProfile:
    status: str
    cost_uj: AffineCost | OperatorEnergyCost | None
    lower_error_ppm: int
    upper_error_ppm: int
    boundary_id: str | None

    @classmethod
    def from_json(cls, value: object) -> "EnergyProfile":
        row = _mapping("energy", value)
        status = _text("energy.status", row.get("status"))
        if status not in ENERGY_STATUSES:
            raise SchedulerError("unknown energy status")
        raw_cost = row.get("cost_uj")
        if status == "unknown":
            if raw_cost is not None:
                raise SchedulerError("unknown energy cannot carry a cost model")
            cost = None
            boundary = None
        else:
            raw_cost_row = _mapping("energy.cost_uj", raw_cost)
            if raw_cost_row.get("kind") == "operator_sum_v1":
                cost = OperatorEnergyCost.from_json(raw_cost_row)
            else:
                cost = AffineCost.from_json("energy.cost_uj", raw_cost_row)
            boundary = _text("energy.boundary_id", row.get("boundary_id"))
        lower = _strict_int(
            "energy.lower_error_ppm", row.get("lower_error_ppm", 0)
        )
        upper = _strict_int(
            "energy.upper_error_ppm", row.get("upper_error_ppm", 0)
        )
        if lower > 1_000_000 or upper > 1_000_000:
            raise SchedulerError("energy error must not exceed 100 percent")
        return cls(status, cost, lower, upper, boundary)

    def breakdown(
        self, request: Request, service_us: int | None = None
    ) -> dict[str, Any] | None:
        if self.cost_uj is None:
            return None
        if isinstance(self.cost_uj, OperatorEnergyCost):
            if service_us is None:
                raise SchedulerError("operator energy requires service time")
            return self.cost_uj.breakdown(request, service_us)
        value = max(1, self.cost_uj.predict(request))
        return {"kind": "affine_v1", "total_uj": value}

    def predict_uj(
        self, request: Request, service_us: int | None = None
    ) -> int | None:
        value = self.breakdown(request, service_us)
        if value is None:
            return None
        return max(1, int(value["total_uj"]))

    def bounds_uj(
        self, request: Request, service_us: int | None = None
    ) -> tuple[int | None, int | None, int | None, dict[str, Any] | None]:
        breakdown = self.breakdown(request, service_us)
        if breakdown is None:
            return None, None, None, None
        value = max(1, int(breakdown["total_uj"]))
        lower = value * (1_000_000 - self.lower_error_ppm) // 1_000_000
        upper = (
            value * (1_000_000 + self.upper_error_ppm) + 999_999
        ) // 1_000_000
        return value, lower, upper, breakdown

    def lower_uj(
        self, request: Request, service_us: int | None = None
    ) -> int | None:
        value = self.predict_uj(request, service_us)
        if value is None:
            return None
        return value * (1_000_000 - self.lower_error_ppm) // 1_000_000

    def upper_uj(
        self, request: Request, service_us: int | None = None
    ) -> int | None:
        value = self.predict_uj(request, service_us)
        if value is None:
            return None
        return (
            value * (1_000_000 + self.upper_error_ppm) + 999_999
        ) // 1_000_000


@dataclass(frozen=True)
class OverlapProfile:
    status: str
    exposed_join_wait_ppm: int | None
    upper_error_ppm: int
    sample_count: int

    @classmethod
    def from_json(cls, value: object) -> "OverlapProfile":
        if value is None:
            return cls("unknown", None, 0, 0)
        row = _mapping("overlap", value)
        status = _text("overlap.status", row.get("status"))
        if status not in OVERLAP_STATUSES:
            raise SchedulerError("unknown overlap status")
        if status in {"unknown", "not_applicable"}:
            if any(
                row.get(name) is not None
                for name in (
                    "exposed_join_wait_ppm",
                    "upper_error_ppm",
                    "sample_count",
                )
            ):
                raise SchedulerError("unmeasured overlap cannot carry measurements")
            return cls(status, None, 0, 0)
        wait = _strict_int(
            "overlap.exposed_join_wait_ppm",
            row.get("exposed_join_wait_ppm"),
        )
        error = _strict_int(
            "overlap.upper_error_ppm", row.get("upper_error_ppm")
        )
        samples = _strict_int(
            "overlap.sample_count", row.get("sample_count"), 1
        )
        if wait > 1_000_000 or error > 1_000_000:
            raise SchedulerError("overlap fraction must not exceed 100 percent")
        return cls(status, wait, error, samples)

    def upper_join_wait_ppm(self) -> int | None:
        if self.exposed_join_wait_ppm is None:
            return None
        return min(1_000_000, self.exposed_join_wait_ppm + self.upper_error_ppm)


@dataclass(frozen=True)
class RouteProfile:
    route_id: str
    workload_id: str
    granularity: str
    baseline: bool
    resource_slots: Mapping[str, int]
    resource_leases: tuple[ResourceLeaseProfile, ...]
    latency: LatencyProfile
    energy: EnergyProfile
    overlap: OverlapProfile
    quality_class: str
    placement_verified: bool
    resident: bool
    runtime_contract: RouteRuntimeContract | None
    finish_before_feature: str | None
    server_busy_ppm: int
    server_memory_bytes: int
    evidence_ids: tuple[str, ...]

    @classmethod
    def from_json(cls, value: object) -> "RouteProfile":
        row = _mapping("route", value)
        granularity = _text("route.granularity", row.get("granularity"))
        if granularity not in GRANULARITIES:
            raise SchedulerError("unknown route granularity")
        raw_slots = _mapping("route.resource_slots", row.get("resource_slots"))
        slots = {
            _text("resource slot id", key): _strict_int(
                f"resource slot {key}", amount, 1
            )
            for key, amount in raw_slots.items()
        }
        if not slots:
            raise SchedulerError("route must use at least one resource")
        raw_leases = row.get("resource_leases", [])
        if type(raw_leases) is not list:
            raise SchedulerError("route.resource_leases must be a list")
        leases = tuple(ResourceLeaseProfile.from_json(item) for item in raw_leases)
        lease_ids = [lease.lease_id for lease in leases]
        if len(lease_ids) != len(set(lease_ids)):
            raise SchedulerError("duplicate resource lease id")
        quality = _text("route.quality_class", row.get("quality_class"))
        if quality not in QUALITY_RANK:
            raise SchedulerError("unknown route quality class")
        raw_evidence = row.get("evidence_ids")
        if type(raw_evidence) is not list or not raw_evidence:
            raise SchedulerError("route evidence_ids must be a non-empty list")
        evidence = tuple(_text("route evidence id", item) for item in raw_evidence)
        raw_runtime_contract = row.get("runtime_contract")
        raw_finish_before_feature = row.get("finish_before_feature")
        if raw_finish_before_feature is not None:
            raw_finish_before_feature = _text(
                "route.finish_before_feature", raw_finish_before_feature
            )
        try:
            runtime_contract = (
                None
                if raw_runtime_contract is None
                else RouteRuntimeContract.from_json(raw_runtime_contract)
            )
        except RuntimeGateError as exc:
            raise SchedulerError(str(exc)) from exc
        return cls(
            route_id=_text("route.route_id", row.get("route_id")),
            workload_id=_text("route.workload_id", row.get("workload_id")),
            granularity=granularity,
            baseline=_strict_bool("route.baseline", row.get("baseline")),
            resource_slots=slots,
            resource_leases=leases,
            latency=LatencyProfile.from_json(row.get("latency")),
            energy=EnergyProfile.from_json(row.get("energy")),
            overlap=OverlapProfile.from_json(row.get("overlap")),
            quality_class=quality,
            placement_verified=_strict_bool(
                "route.placement_verified", row.get("placement_verified")
            ),
            resident=_strict_bool("route.resident", row.get("resident")),
            runtime_contract=runtime_contract,
            finish_before_feature=raw_finish_before_feature,
            server_busy_ppm=_strict_int(
                "route.server_busy_ppm", row.get("server_busy_ppm")
            ),
            server_memory_bytes=_strict_int(
                "route.server_memory_bytes", row.get("server_memory_bytes")
            ),
            evidence_ids=evidence,
        )

    def lease_demands(
        self,
        request: Request,
        service_us: int,
        service_upper_us: int,
    ) -> tuple["LeaseDemand", ...]:
        if self.resource_leases:
            demands = tuple(lease.predict(request) for lease in self.resource_leases)
        else:
            demands = tuple(
                LeaseDemand(
                    lease_id=f"{resource_id}-full-route",
                    resource_id=resource_id,
                    slots=slots,
                    start_offset_us=0,
                    duration_us=service_us,
                    duration_upper_us=service_upper_us,
                )
                for resource_id, slots in sorted(self.resource_slots.items())
            )
        for demand in demands:
            if demand.start_offset_us + demand.duration_us > service_us:
                raise SchedulerError("resource lease exceeds route service time")
            if demand.start_offset_us + demand.duration_upper_us > service_upper_us:
                raise SchedulerError("resource lease exceeds route service upper bound")
        return demands


@dataclass(frozen=True)
class PolicyConfig:
    energy_saving_ppm: int = 50_000
    latency_limit_ppm: int = 1_050_000
    max_exposed_join_wait_ppm: int = 50_000
    offload_requires_baseline_queue: bool = False
    offload_min_finish_saving_us: int = 1

    def validate(self) -> None:
        _strict_int("energy_saving_ppm", self.energy_saving_ppm)
        _strict_int("latency_limit_ppm", self.latency_limit_ppm, 1)
        _strict_int(
            "max_exposed_join_wait_ppm", self.max_exposed_join_wait_ppm
        )
        _strict_bool(
            "offload_requires_baseline_queue",
            self.offload_requires_baseline_queue,
        )
        _strict_int(
            "offload_min_finish_saving_us",
            self.offload_min_finish_saving_us,
            1,
        )
        if self.energy_saving_ppm >= 1_000_000:
            raise SchedulerError("energy saving must be below 100 percent")
        if self.latency_limit_ppm < 1_000_000:
            raise SchedulerError("latency limit cannot be below the baseline")
        if self.max_exposed_join_wait_ppm > 1_000_000:
            raise SchedulerError("join-wait limit must not exceed 100 percent")


@dataclass(frozen=True)
class ProfileBundle:
    profile_id: str
    resources: Mapping[str, ResourceProfile]
    routes: tuple[RouteProfile, ...]
    trace_workload_map: Mapping[str, str]
    policy: PolicyConfig

    @classmethod
    def from_json(cls, value: object) -> "ProfileBundle":
        row = _mapping("profile", value)
        if row.get("schema") != PROFILE_SCHEMA:
            raise SchedulerError("profile schema mismatch")
        raw_resources = row.get("resources")
        if type(raw_resources) is not list or not raw_resources:
            raise SchedulerError("profile resources must be a non-empty list")
        resources: dict[str, ResourceProfile] = {}
        for raw in raw_resources:
            resource = ResourceProfile.from_json(raw)
            if resource.resource_id in resources:
                raise SchedulerError("duplicate resource id")
            resources[resource.resource_id] = resource
        raw_routes = row.get("routes")
        if type(raw_routes) is not list or not raw_routes:
            raise SchedulerError("profile routes must be a non-empty list")
        routes = tuple(RouteProfile.from_json(item) for item in raw_routes)
        route_ids: set[str] = set()
        baselines: dict[str, int] = {}
        for route in routes:
            if route.route_id in route_ids:
                raise SchedulerError("duplicate route id")
            if route.baseline and route.finish_before_feature is not None:
                raise SchedulerError(
                    "baseline route cannot have a finish window"
                )
            route_ids.add(route.route_id)
            baselines[route.workload_id] = baselines.get(route.workload_id, 0) + int(
                route.baseline
            )
            for resource_id, slots in route.resource_slots.items():
                resource = resources.get(resource_id)
                if resource is None:
                    raise SchedulerError("route references an unknown resource")
                if slots > resource.capacity:
                    raise SchedulerError("route requests too many resource slots")
            if route.resource_leases:
                lease_resources: set[str] = set()
                for lease in route.resource_leases:
                    declared_slots = route.resource_slots.get(lease.resource_id)
                    if declared_slots is None:
                        raise SchedulerError(
                            "resource lease references an undeclared resource"
                        )
                    if lease.slots > declared_slots:
                        raise SchedulerError(
                            "resource lease exceeds declared route slots"
                        )
                    lease_resources.add(lease.resource_id)
                if lease_resources != set(route.resource_slots):
                    raise SchedulerError(
                        "explicit leases must cover every declared route resource"
                    )
            if (
                route.runtime_contract is not None
                and set(route.runtime_contract.resources) != set(route.resource_slots)
            ):
                raise SchedulerError(
                    "runtime contract must cover every declared route resource"
                )
        if not baselines or any(count != 1 for count in baselines.values()):
            raise SchedulerError("each workload must have exactly one baseline")
        raw_map = _mapping("trace_workload_map", row.get("trace_workload_map"))
        trace_map = {
            _text("trace model id", key): _text("trace workload id", mapped)
            for key, mapped in raw_map.items()
        }
        if set(trace_map.values()) - set(baselines):
            raise SchedulerError("trace map references an unknown workload")
        raw_policy = _mapping("policy", row.get("policy", {}))
        policy = PolicyConfig(
            energy_saving_ppm=_strict_int(
                "policy.energy_saving_ppm",
                raw_policy.get("energy_saving_ppm", 50_000),
            ),
            latency_limit_ppm=_strict_int(
                "policy.latency_limit_ppm",
                raw_policy.get("latency_limit_ppm", 1_050_000),
                1,
            ),
            max_exposed_join_wait_ppm=_strict_int(
                "policy.max_exposed_join_wait_ppm",
                raw_policy.get("max_exposed_join_wait_ppm", 50_000),
            ),
            offload_requires_baseline_queue=_strict_bool(
                "policy.offload_requires_baseline_queue",
                raw_policy.get("offload_requires_baseline_queue", False),
            ),
            offload_min_finish_saving_us=_strict_int(
                "policy.offload_min_finish_saving_us",
                raw_policy.get("offload_min_finish_saving_us", 1),
                1,
            ),
        )
        policy.validate()
        return cls(
            profile_id=_text("profile_id", row.get("profile_id")),
            resources=resources,
            routes=routes,
            trace_workload_map=trace_map,
            policy=policy,
        )






@dataclass(frozen=True)
class MarginalSystemCostContext:
    context_id: str
    critical_path_end_us: int
    phase_power_mw: int
    gpu_idle_power_mw: int
    causal_tail_power_mw: int
    route_cpu_interference_ppm: Mapping[str, int]
    lower_error_ppm: int
    upper_error_ppm: int
    sample_count: int
    measured: bool

    def validate(self) -> None:
        _text("marginal cost context_id", self.context_id)
        _strict_int(
            "marginal cost critical_path_end_us",
            self.critical_path_end_us,
        )
        for name in (
            "phase_power_mw",
            "gpu_idle_power_mw",
            "causal_tail_power_mw",
        ):
            _strict_int(f"marginal cost {name}", getattr(self, name))
        if type(self.route_cpu_interference_ppm) is not dict:
            raise SchedulerError(
                "marginal route interference must be a dictionary"
            )
        for route_id, value in self.route_cpu_interference_ppm.items():
            _text("marginal route id", route_id)
            _strict_int("marginal route interference ppm", value)
        _strict_int("marginal lower_error_ppm", self.lower_error_ppm)
        _strict_int("marginal upper_error_ppm", self.upper_error_ppm)
        if (
            self.lower_error_ppm > 1_000_000
            or self.upper_error_ppm > 1_000_000
        ):
            raise SchedulerError(
                "marginal error must not exceed 100 percent"
            )
        _strict_int("marginal sample_count", self.sample_count, 1)
        _strict_bool("marginal measured", self.measured)

    def route_cost(
        self,
        route_id: str,
        service_us: int,
        service_upper_us: int,
        finish_us: int,
        finish_upper_us: int,
    ) -> dict[str, int | str | bool]:
        self.validate()
        route_id = _text("marginal cost route_id", route_id)
        if route_id not in self.route_cpu_interference_ppm:
            raise SchedulerError(
                f"marginal cost lacks route interference: {route_id}"
            )
        ppm = self.route_cpu_interference_ppm[route_id]
        try:
            result = calculate_marginal_system_cost(
                critical_path_end_us=self.critical_path_end_us,
                phase_power_mw=self.phase_power_mw,
                stranded_idle_power_mw=self.gpu_idle_power_mw,
                causal_tail_power_mw=self.causal_tail_power_mw,
                interference_ppm=ppm,
                lower_error_ppm=self.lower_error_ppm,
                upper_error_ppm=self.upper_error_ppm,
                service_us=service_us,
                service_upper_us=service_upper_us,
                finish_us=finish_us,
                finish_upper_us=finish_upper_us,
            )
        except RuntimeSystemCostError as exc:
            raise SchedulerError(str(exc)) from exc
        result["gpu_idle_uj"] = result.pop("stranded_idle_uj")
        result.update({
            "context_id": self.context_id,
            "measured": self.measured,
            "sample_count": self.sample_count,
        })
        return result




@dataclass(frozen=True)
class Candidate:
    route: RouteProfile
    service_us: int
    service_upper_us: int
    start_us: int
    finish_us: int
    finish_upper_us: int
    energy_uj: int | None
    energy_lower_uj: int | None
    energy_upper_uj: int | None
    energy_breakdown: Mapping[str, Any] | None
    server_busy_us: int
    lease_preview: LeasePreview
    runtime_gate: RuntimeGateReceipt | None
    system_finish_upper_us: int | None = None
    marginal_system_cost: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class Decision:
    request_id: str
    workload_id: str
    mode: str
    route_id: str
    granularity: str
    start_us: int
    finish_us: int
    finish_upper_us: int
    service_us: int
    queue_us: int
    queue_by_resource_us: Mapping[str, int]
    blocking_resources: tuple[str, ...]
    leases: tuple[LeaseRecord, ...]
    runtime_gate: RuntimeGateReceipt | None
    energy_uj: int | None
    energy_upper_uj: int | None
    energy_breakdown: Mapping[str, Any] | None
    server_busy_us: int
    reason: str
    rejected: tuple[tuple[str, str], ...]
    system_finish_upper_us: int | None = None
    marginal_system_cost: Mapping[str, Any] | None = None



class RoutePolicy:
    def __init__(
        self,
        profile: ProfileBundle,
        mode: str,
        ready_overrides: Mapping[str, bool] | None = None,
        runtime_snapshot: RuntimeSnapshot | None = None,
        timeline: ResourceTimeline | None = None,
    ) -> None:
        if mode not in POLICY_MODES:
            raise SchedulerError("unknown policy mode")
        self.profile = profile
        self.mode = mode
        if timeline is None:
            self.timeline = ResourceTimeline(profile.resources, ready_overrides)
        else:
            if ready_overrides is not None:
                raise SchedulerError(
                    "shared timeline cannot use readiness overrides"
                )
            timeline.require_compatible(profile.resources)
            self.timeline = timeline
        self.runtime_snapshot = runtime_snapshot
        self._routes: dict[str, list[RouteProfile]] = {}
        for route in profile.routes:
            self._routes.setdefault(route.workload_id, []).append(route)

    def update_runtime_snapshot(self, snapshot: RuntimeSnapshot) -> None:
        if (
            self.runtime_snapshot is not None
            and snapshot.generation <= self.runtime_snapshot.generation
        ):
            raise SchedulerError("runtime snapshot generation did not advance")
        self.runtime_snapshot = snapshot

    def _runtime_gate(
        self,
        request: Request,
        route: RouteProfile,
        runtime_now_us: int | None,
    ) -> RuntimeGateReceipt | None:
        if route.runtime_contract is None:
            return None
        try:
            return evaluate_runtime_gate(
                route.runtime_contract,
                request.semantics,
                self.runtime_snapshot,
                runtime_now_us,
            )
        except RuntimeGateError as exc:
            raise SchedulerError(str(exc)) from exc

    def _candidate(
        self,
        request: Request,
        route: RouteProfile,
        runtime_gate: RuntimeGateReceipt | None,
        earliest_start_us: int,
        marginal_system_context: MarginalSystemCostContext | None,
    ) -> Candidate:
        service = route.latency.predict_us(request)
        upper = route.latency.upper_us(request)
        demands = route.lease_demands(request, service, upper)
        lease_preview = self.timeline.preview_leases(
            demands,
            earliest_start_us,
            service,
            upper,
        )
        energy, energy_lower, energy_upper, energy_breakdown = (
            route.energy.bounds_uj(request, service)
        )
        marginal = None
        system_finish_upper_us = lease_preview.finish_upper_us
        if marginal_system_context is not None:
            marginal = marginal_system_context.route_cost(
                route.route_id,
                service,
                upper,
                lease_preview.finish_us,
                lease_preview.finish_upper_us,
            )
            if energy is not None:
                energy += int(marginal["total_uj"])
            if energy_lower is not None:
                energy_lower += int(marginal["lower_uj"])
            if energy_upper is not None:
                energy_upper += int(marginal["upper_uj"])
            energy_breakdown = {
                "direct": energy_breakdown,
                "kind": "marginal_system_v1",
                "marginal_system": marginal,
                "total_uj": energy,
            }
            system_finish_upper_us = (
                marginal_system_context.critical_path_end_us
                + int(marginal["critical_path_extension_upper_us"])
            )
        candidate = Candidate(
            route=route,
            service_us=service,
            service_upper_us=upper,
            start_us=lease_preview.start_us,
            finish_us=lease_preview.finish_us,
            finish_upper_us=lease_preview.finish_upper_us,
            energy_uj=energy,
            energy_lower_uj=energy_lower,
            energy_upper_uj=energy_upper,
            energy_breakdown=energy_breakdown,
            server_busy_us=(service * route.server_busy_ppm + 999_999) // 1_000_000,
            lease_preview=lease_preview,
            runtime_gate=runtime_gate,
            system_finish_upper_us=system_finish_upper_us,
            marginal_system_cost=marginal,
        )
        return candidate

    @staticmethod
    def _basic_gate(request: Request, route: RouteProfile) -> str | None:
        if not route.latency.measured_for(request):
            return "LATENCY_UNMEASURED"
        if not route.placement_verified:
            return "PLACEMENT_UNVERIFIED"
        if not route.resident:
            return "WEIGHTS_NOT_RESIDENT"
        if QUALITY_RANK[route.quality_class] < QUALITY_RANK[request.quality_requirement]:
            return "QUALITY_INSUFFICIENT"
        return None

    def _schedule_baseline(
        self,
        request: Request,
        runtime_now_us: int | None,
        runtime_route_admissions: Mapping[str, str] | None,
        marginal_system_context: MarginalSystemCostContext | None,
    ) -> tuple[int, list[RouteProfile], RouteProfile, Candidate]:
        request.validate()
        if (
            runtime_now_us is not None
            and (
                isinstance(runtime_now_us, bool)
                or not isinstance(runtime_now_us, int)
                or runtime_now_us < 0
            )
        ):
            raise SchedulerError("runtime_now_us must be a nonnegative integer")
        if request.semantics.cancelled:
            raise SchedulerError("request is cancelled before dispatch")
        if marginal_system_context is not None:
            if not isinstance(
                marginal_system_context, MarginalSystemCostContext
            ):
                raise SchedulerError("marginal system context is invalid")
            marginal_system_context.validate()
            if (
                self.mode in {"enforce", "adaptive"}
                and not marginal_system_context.measured
            ):
                raise SchedulerError("marginal system cost is unmeasured")
        earliest_start_us = max(
            request.arrival_us,
            runtime_now_us if runtime_now_us is not None else request.arrival_us,
        )
        routes = self._routes.get(request.workload_id)
        if not routes:
            raise SchedulerError("request has no route set")
        if runtime_route_admissions is not None:
            unknown_routes = set(runtime_route_admissions) - {
                route.route_id for route in routes
            }
            if unknown_routes:
                raise SchedulerError(
                    "runtime admission references an unknown route"
                )
        baseline_route = next(route for route in routes if route.baseline)
        if runtime_route_admissions is not None:
            admission = runtime_route_admissions.get(baseline_route.route_id)
            if admission != "ADMITTED":
                raise SchedulerError(
                    "baseline is unusable: "
                    + ("RUNTIME_COST_ABSENT" if admission is None else admission)
                )
        baseline_gate = self._basic_gate(request, baseline_route)
        if baseline_gate is not None:
            raise SchedulerError(f"baseline is unusable: {baseline_gate}")
        runtime_gate = self._runtime_gate(
            request, baseline_route, runtime_now_us
        )
        if runtime_gate is not None and not runtime_gate.admitted:
            raise SchedulerError(
                f"baseline is unusable: {runtime_gate.reason}"
            )
        baseline = self._candidate(
            request,
            baseline_route,
            runtime_gate,
            earliest_start_us,
            marginal_system_context,
        )
        return earliest_start_us, routes, baseline_route, baseline

    def _alternative_candidate(
        self,
        request: Request,
        route: RouteProfile,
        baseline: Candidate,
        baseline_route: RouteProfile,
        earliest_start_us: int,
        runtime_now_us: int | None,
        runtime_route_admissions: Mapping[str, str] | None,
        marginal_system_context: MarginalSystemCostContext | None,
    ) -> tuple[Candidate | None, str | None]:
        if runtime_route_admissions is not None:
            admission = runtime_route_admissions.get(route.route_id)
            if admission != "ADMITTED":
                return None, (
                    "RUNTIME_COST_ABSENT" if admission is None else admission
                )
        gate = self._basic_gate(request, route)
        if gate is not None:
            return None, gate
        runtime_gate = self._runtime_gate(request, route, runtime_now_us)
        if runtime_gate is not None and not runtime_gate.admitted:
            return None, runtime_gate.reason
        try:
            candidate = self._candidate(
                request,
                route,
                runtime_gate,
                earliest_start_us,
                marginal_system_context,
            )
        except SchedulerError as exc:
            return None, str(exc)
        if route.finish_before_feature is not None:
            try:
                finish_before_us = request.feature(route.finish_before_feature)
            except SchedulerError:
                return None, "FINISH_WINDOW_MISSING"
            if candidate.finish_upper_us > finish_before_us:
                return None, "FINISH_WINDOW_EXCEEDED"
        if (
            baseline.finish_upper_us <= request.deadline_us
            and candidate.finish_upper_us > request.deadline_us
        ):
            return None, "SLO_INFEASIBLE"
        if (
            baseline.finish_upper_us > request.deadline_us
            and candidate.finish_upper_us > baseline.finish_upper_us
        ):
            return None, "SLO_TARDINESS_REGRESSION"
        baseline_elapsed = baseline.finish_upper_us - earliest_start_us
        candidate_elapsed = candidate.finish_upper_us - earliest_start_us
        limit = (
            baseline_elapsed * self.profile.policy.latency_limit_ppm + 999_999
        ) // 1_000_000
        if candidate_elapsed > limit:
            return None, "LATENCY_LIMIT"
        reason = self._alternative_energy_rejection(
            route, baseline, baseline_route, candidate
        )
        return (candidate, None) if reason is None else (None, reason)

    def _alternative_energy_rejection(
        self,
        route: RouteProfile,
        baseline: Candidate,
        baseline_route: RouteProfile,
        candidate: Candidate,
    ) -> str | None:
        if self.mode not in {"enforce", "adaptive"}:
            return None
        if (
            baseline_route.energy.status != "measured"
            or route.energy.status != "measured"
            or baseline.energy_lower_uj is None
            or candidate.energy_upper_uj is None
        ):
            return "ENERGY_NOT_MEASURED"
        if baseline_route.energy.boundary_id != route.energy.boundary_id:
            return "ENERGY_BOUNDARY_MISMATCH"
        if self.mode == "enforce":
            threshold = (
                baseline.energy_lower_uj
                * (1_000_000 - self.profile.policy.energy_saving_ppm)
            ) // 1_000_000
            if candidate.energy_upper_uj > threshold:
                return "ENERGY_MARGIN"
        requires_overlap = (
            route.granularity != "task" and len(route.resource_slots) > 1
        )
        if requires_overlap:
            wait_upper = route.overlap.upper_join_wait_ppm()
            if route.overlap.status != "measured" or wait_upper is None:
                return "OVERLAP_NOT_MEASURED"
            if wait_upper > self.profile.policy.max_exposed_join_wait_ppm:
                return "OVERLAP_LIMIT"
        return None

    def _admitted_alternatives(
        self,
        request: Request,
        routes: Sequence[RouteProfile],
        baseline: Candidate,
        baseline_route: RouteProfile,
        earliest_start_us: int,
        runtime_now_us: int | None,
        runtime_route_admissions: Mapping[str, str] | None,
        marginal_system_context: MarginalSystemCostContext | None,
    ) -> tuple[list[Candidate], list[tuple[str, str]]]:
        admitted = []
        rejected = []
        for route in routes:
            if route.baseline:
                continue
            candidate, reason = self._alternative_candidate(
                request,
                route,
                baseline,
                baseline_route,
                earliest_start_us,
                runtime_now_us,
                runtime_route_admissions,
                marginal_system_context,
            )
            if candidate is None:
                assert reason is not None
                rejected.append((route.route_id, reason))
            else:
                admitted.append(candidate)
        return admitted, rejected

    def _select_adaptive_candidate(
        self,
        request: Request,
        baseline: Candidate,
        earliest_start_us: int,
        admitted: list[Candidate],
        rejected: list[tuple[str, str]],
    ) -> tuple[Candidate, str]:
        if not admitted:
            return baseline, "ADAPTIVE_NO_QUALIFIED_ALTERNATIVE"
        if (
            self.profile.policy.offload_requires_baseline_queue
            and baseline.start_us == earliest_start_us
        ):
            rejected.extend(
                (candidate.route.route_id, "BASELINE_NOT_QUEUED")
                for candidate in admitted
            )
            return baseline, "ADAPTIVE_BASELINE_AVAILABLE"
        if self.profile.policy.offload_requires_baseline_queue:
            minimum = self.profile.policy.offload_min_finish_saving_us
            slower = [
                candidate for candidate in admitted
                if candidate.finish_upper_us + minimum > baseline.finish_upper_us
            ]
            rejected.extend(
                (candidate.route.route_id, "NO_FINISH_SAVING")
                for candidate in slower
            )
            admitted = [
                candidate for candidate in admitted
                if candidate.finish_upper_us + minimum <= baseline.finish_upper_us
            ]
        if not admitted:
            return baseline, "ADAPTIVE_NO_SYSTEM_BENEFIT"
        feasible = [
            candidate for candidate in admitted
            if candidate.finish_upper_us <= request.deadline_us
        ]
        baseline_feasible = baseline.finish_upper_us <= request.deadline_us
        if baseline_feasible:
            threshold = (
                baseline.energy_lower_uj
                * (1_000_000 - self.profile.policy.energy_saving_ppm)
            ) // 1_000_000
            saving = [
                candidate for candidate in feasible
                if candidate.energy_upper_uj is not None
                and candidate.energy_upper_uj <= threshold
            ]
            if saving:
                return min(saving, key=lambda row: (
                    row.energy_upper_uj,
                    row.finish_upper_us,
                    row.route.route_id,
                )), "ADAPTIVE_ENERGY_SAVING"
            return baseline, "ADAPTIVE_BASELINE_FEASIBLE"
        if feasible:
            return min(feasible, key=lambda row: (
                row.energy_upper_uj,
                row.finish_upper_us,
                row.route.route_id,
            )), "ADAPTIVE_DEADLINE_RECOVERY"
        minimum = self.profile.policy.offload_min_finish_saving_us
        faster = [
            candidate for candidate in admitted
            if candidate.finish_upper_us + minimum <= baseline.finish_upper_us
        ]
        if faster:
            return min(faster, key=lambda row: (
                row.finish_upper_us,
                row.energy_upper_uj,
                row.route.route_id,
            )), "ADAPTIVE_TARDINESS_REDUCTION"
        return baseline, "ADAPTIVE_NO_SYSTEM_BENEFIT"

    def _select_scheduled_candidate(
        self,
        request: Request,
        baseline: Candidate,
        earliest_start_us: int,
        admitted: list[Candidate],
        rejected: list[tuple[str, str]],
    ) -> tuple[Candidate, str]:
        if self.mode == "enforce":
            ranked = [row for row in admitted if row.energy_upper_uj is not None]
            if ranked:
                return min(ranked, key=lambda row: (
                    row.energy_upper_uj, row.finish_us, row.route.route_id
                )), "VERIFIED_ENERGY_SAVING"
            return baseline, "FAIL_CLOSED_BASELINE"
        if self.mode == "adaptive":
            return self._select_adaptive_candidate(
                request, baseline, earliest_start_us, admitted, rejected
            )
        if self.mode == "capacity":
            selected = min([baseline, *admitted], key=lambda row: (
                row.server_busy_us,
                row.route.server_memory_bytes,
                row.finish_us,
                row.route.route_id,
            ))
            reason = (
                "MINIMUM_SERVER_CAPACITY_COST"
                if not selected.route.baseline else "CAPACITY_BASELINE"
            )
            return selected, reason
        selected = min([baseline, *admitted], key=lambda row: (
            row.finish_us, row.service_us, row.route.route_id
        ))
        reason = (
            "SHADOW_FASTEST_QUALIFIED"
            if not selected.route.baseline else "SHADOW_BASELINE"
        )
        return selected, reason

    def _atomic_runtime_selection(
        self,
        request: Request,
        selected: Candidate,
        reason: str,
        baseline_route: RouteProfile,
        earliest_start_us: int,
        runtime_now_us: int | None,
        marginal_system_context: MarginalSystemCostContext | None,
        rejected: list[tuple[str, str]],
    ) -> tuple[Candidate, str]:
        if selected.runtime_gate is None:
            return selected, reason
        final_gate = self._runtime_gate(
            request, selected.route, runtime_now_us
        )
        if final_gate is not None and final_gate.admitted:
            return selected, reason
        if selected.route.baseline:
            reason_code = (
                "RUNTIME_GATE_DISAPPEARED"
                if final_gate is None else final_gate.reason
            )
            raise SchedulerError(
                f"baseline failed atomic runtime gate: {reason_code}"
            )
        rejected.append((selected.route.route_id, "ATOMIC_RUNTIME_REVOKED"))
        fallback_gate = self._runtime_gate(
            request, baseline_route, runtime_now_us
        )
        if fallback_gate is not None and not fallback_gate.admitted:
            raise SchedulerError(
                f"fallback baseline is unusable: {fallback_gate.reason}"
            )
        return self._candidate(
            request,
            baseline_route,
            fallback_gate,
            earliest_start_us,
            marginal_system_context,
        ), "ATOMIC_RUNTIME_FALLBACK"

    def _commit_schedule_decision(
        self,
        request: Request,
        selected: Candidate,
        reason: str,
        rejected: Sequence[tuple[str, str]],
    ) -> Decision:
        leases = self.timeline.commit_leases(
            selected.lease_preview, request.request_id
        )
        return Decision(
            request_id=request.request_id,
            workload_id=request.workload_id,
            mode=self.mode,
            route_id=selected.route.route_id,
            granularity=selected.route.granularity,
            start_us=selected.start_us,
            finish_us=selected.finish_us,
            finish_upper_us=selected.finish_upper_us,
            service_us=selected.service_us,
            queue_us=selected.start_us - request.arrival_us,
            queue_by_resource_us=selected.lease_preview.queue_by_resource_us,
            blocking_resources=selected.lease_preview.blocking_resources,
            leases=leases,
            runtime_gate=selected.runtime_gate,
            energy_uj=selected.energy_uj,
            energy_upper_uj=selected.energy_upper_uj,
            energy_breakdown=selected.energy_breakdown,
            server_busy_us=selected.server_busy_us,
            reason=reason,
            rejected=tuple(sorted(rejected)),
            system_finish_upper_us=selected.system_finish_upper_us,
            marginal_system_cost=selected.marginal_system_cost,
        )

    def schedule(
        self,
        request: Request,
        runtime_now_us: int | None = None,
        runtime_route_admissions: Mapping[str, str] | None = None,
        marginal_system_context: MarginalSystemCostContext | None = None,
    ) -> Decision:
        (
            earliest_start_us,
            routes,
            baseline_route,
            baseline,
        ) = self._schedule_baseline(
            request,
            runtime_now_us,
            runtime_route_admissions,
            marginal_system_context,
        )
        if self.mode == "control":
            selected = baseline
            reason = "CONTROL_BASELINE"
            rejected: list[tuple[str, str]] = []
        else:
            admitted, rejected = self._admitted_alternatives(
                request,
                routes,
                baseline,
                baseline_route,
                earliest_start_us,
                runtime_now_us,
                runtime_route_admissions,
                marginal_system_context,
            )
            selected, reason = self._select_scheduled_candidate(
                request,
                baseline,
                earliest_start_us,
                admitted,
                rejected,
            )
        selected, reason = self._atomic_runtime_selection(
            request,
            selected,
            reason,
            baseline_route,
            earliest_start_us,
            runtime_now_us,
            marginal_system_context,
            rejected,
        )
        return self._commit_schedule_decision(
            request, selected, reason, rejected
        )


def decision_to_json(decision: Decision) -> dict[str, object]:
    return {
        "request_id": decision.request_id,
        "workload_id": decision.workload_id,
        "mode": decision.mode,
        "route_id": decision.route_id,
        "granularity": decision.granularity,
        "start_us": decision.start_us,
        "finish_us": decision.finish_us,
        "finish_upper_us": decision.finish_upper_us,
        "system_finish_upper_us": decision.system_finish_upper_us,
        "service_us": decision.service_us,
        "queue_us": decision.queue_us,
        "queue_by_resource_us": dict(decision.queue_by_resource_us),
        "blocking_resources": list(decision.blocking_resources),
        "leases": [
            {
                "token": lease.token,
                "owner_id": lease.owner_id,
                "lease_id": lease.lease_id,
                "resource_id": lease.resource_id,
                "lanes": list(lease.lanes),
                "start_us": lease.start_us,
                "predicted_end_us": lease.predicted_end_us,
                "reserved_until_us": lease.reserved_until_us,
            }
            for lease in decision.leases
        ],
        "runtime_gate": (
            None
            if decision.runtime_gate is None
            else {
                "admitted": decision.runtime_gate.admitted,
                "reason": decision.runtime_gate.reason,
                "snapshot_id": decision.runtime_gate.snapshot_id,
                "snapshot_generation": decision.runtime_gate.snapshot_generation,
                "epoch_key": decision.runtime_gate.epoch_key,
                "checked_resources": list(
                    decision.runtime_gate.checked_resources
                ),
            }
        ),
        "energy_uj": decision.energy_uj,
        "energy_upper_uj": decision.energy_upper_uj,
        "energy_breakdown": (
            None
            if decision.energy_breakdown is None
            else dict(decision.energy_breakdown)
        ),
        "marginal_system_cost": (
            None
            if decision.marginal_system_cost is None
            else dict(decision.marginal_system_cost)
        ),
        "server_busy_us": decision.server_busy_us,
        "reason": decision.reason,
        "rejected": [
            {"route_id": route_id, "reason": reason}
            for route_id, reason in decision.rejected
        ],
    }
