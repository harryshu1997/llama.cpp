"""Compatibility adapters for the existing S42 scheduler profiles."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, Sequence

from .types import (
    AccountingContract,
    AccountingKind,
    AccountingScope,
    ApplicabilityContract,
    CandidateSet,
    EnergyComponent,
    MemoryRequirement,
    MetricEstimate,
    ModelIdentity,
    OverlapEstimate,
    PhaseLease,
    PlacementGranularity,
    QualityClass,
    QualityContract,
    ResidencyRequirement,
    ResourceRequirement,
    RouteAlternative,
    RouteMaturity,
    SchedulingUnit,
    SchedulerContractError,
    UnitKind,
    make_work_set_hash,
)


LEGACY_QUALITY = {
    "unverified": QualityClass.UNVERIFIED,
    "approximate": QualityClass.SEMANTIC,
    "bounded_numeric": QualityClass.BOUNDED_NUMERIC,
    "exact": QualityClass.EXACT,
}


def _legacy_semantics(value: object) -> Mapping[str, bool | str]:
    if is_dataclass(value):
        raw = asdict(value)
    elif isinstance(value, Mapping):
        raw = dict(value)
    else:
        raise SchedulerContractError("legacy request semantics are unsupported")
    result: dict[str, bool | str] = {}
    for name, item in raw.items():
        if type(item) in {bool, str}:
            result[str(name)] = item
    return result


def adapt_legacy_request(
    request: Any,
    model: ModelIdentity,
    *,
    unit_kind: UnitKind = UnitKind.REQUEST,
    member_request_ids: Sequence[str] | None = None,
    unit_id: str | None = None,
    epoch_id: str | None = None,
) -> SchedulingUnit:
    """Bind a legacy Request to an explicit model and scheduling unit."""

    members = tuple(member_request_ids or (request.request_id,))
    quality = LEGACY_QUALITY.get(request.quality_requirement)
    if quality is None:
        raise SchedulerContractError("unknown legacy quality requirement")
    return SchedulingUnit(
        unit_id=unit_id or request.request_id,
        kind=unit_kind,
        member_request_ids=members,
        work_set_hash=make_work_set_hash(members),
        workload_id=request.workload_id,
        model=model,
        arrival_us=request.arrival_us,
        deadline_us=request.deadline_us,
        input_tokens=request.input_tokens,
        output_tokens=request.output_tokens,
        features=dict(request.features),
        quality_requirement=quality,
        semantics=_legacy_semantics(request.semantics),
        epoch_id=epoch_id,
    )


def _default_accounting(unit: SchedulingUnit, boundary_id: str) -> AccountingContract:
    if unit.kind != UnitKind.REQUEST:
        raise SchedulerContractError(
            "legacy non-request routes require an explicit accounting contract"
        )
    return AccountingContract(
        scope=AccountingScope.REQUEST,
        kind=AccountingKind.EXCLUSIVE_UNIT_TOTAL,
        boundary_id=boundary_id,
        work_set_hash=unit.work_set_hash,
    )


def _resource_role(kind: str) -> str:
    if kind in {"transport", "usb"}:
        return "transport"
    if kind in {"session", "rpc"}:
        return "session"
    return "compute"


def adapt_legacy_route(
    route: Any,
    request: Any,
    unit: SchedulingUnit,
    resources: Mapping[str, Any],
    profile_id: str,
    *,
    accounting: AccountingContract | None = None,
    applicability: ApplicabilityContract | None = None,
) -> RouteAlternative:
    """Materialize one evaluated canonical route for a concrete unit."""

    if route.workload_id != unit.workload_id:
        raise SchedulerContractError("legacy route workload does not match unit")
    request_quality = LEGACY_QUALITY.get(request.quality_requirement)
    if request_quality is None:
        raise SchedulerContractError("unknown legacy quality requirement")
    if (
        request.workload_id != unit.workload_id
        or request.arrival_us != unit.arrival_us
        or request.deadline_us != unit.deadline_us
        or request.input_tokens != unit.input_tokens
        or request.output_tokens != unit.output_tokens
        or dict(request.features) != dict(unit.features)
        or request_quality != unit.quality_requirement
    ):
        raise SchedulerContractError(
            "legacy request does not describe the canonical scheduling unit"
        )
    service_us = route.latency.predict_us(request)
    service_upper_us = route.latency.upper_us(request)
    energy_mean, energy_lower, energy_upper, _ = route.energy.bounds_uj(
        request, service_us
    )
    boundary_id = route.energy.boundary_id or "legacy-unknown-energy-boundary"
    route_accounting = accounting or _default_accounting(unit, boundary_id)
    energy: tuple[EnergyComponent, ...] = ()
    if energy_mean is not None:
        assert energy_upper is not None
        energy = (EnergyComponent(
            estimate_uj=MetricEstimate(
                mean=energy_mean,
                upper=energy_upper,
                sample_count=0,
                measured=route.energy.status == "measured",
                lower=energy_lower,
            ),
            accounting=route_accounting,
        ),)
    maturity = RouteMaturity.PREDICTED
    if route.latency.measured and route.energy.status == "measured":
        maturity = RouteMaturity.MEASURED
    resource_requirements: list[ResourceRequirement] = []
    for resource_id, slots in sorted(route.resource_slots.items()):
        try:
            resource = resources[resource_id]
        except KeyError as exc:
            raise SchedulerContractError(
                "legacy route references an unknown resource"
            ) from exc
        resource_requirements.append(ResourceRequirement(
            resource_id=resource_id,
            kind=resource.kind,
            role=_resource_role(resource.kind),
            slots=slots,
            identity=resource.identity,
        ))

    phase_leases: list[PhaseLease] = []
    for lease in route.lease_demands(request, service_us, service_upper_us):
        phase_leases.append(PhaseLease(
            lease_id=lease.lease_id,
            resource_id=lease.resource_id,
            slots=lease.slots,
            start_offset_us=lease.start_offset_us,
            duration=MetricEstimate(
                mean=lease.duration_us,
                upper=lease.duration_upper_us,
                sample_count=route.latency.sample_count,
                measured=route.latency.measured,
            ),
        ))

    memory: list[MemoryRequirement] = []
    if route.server_memory_bytes:
        memory_resource = next(
            (
                resources[item.resource_id]
                for item in resource_requirements
                if item.kind in {"gpu", "cuda"}
            ),
            None,
        )
        if memory_resource is None:
            memory_resource = next(
                (
                    resources[item.resource_id]
                    for item in resource_requirements
                    if item.kind in {"cpu", "server"}
                ),
                None,
            )
        if memory_resource is None:
            memory_resource = next(
                (
                    item
                    for _, item in sorted(resources.items())
                    if item.kind in {"cpu", "server"}
                ),
                None,
            )
        if memory_resource is None:
            raise SchedulerContractError(
                "legacy server memory has no declared server resource"
            )
        memory.append(MemoryRequirement(
            resource_id=memory_resource.resource_id,
            pool_id=(
                "device-memory"
                if memory_resource.kind in {"gpu", "cuda"}
                else "main-memory"
            ),
            bytes=route.server_memory_bytes,
            persistent=False,
        ))

    residency: list[ResidencyRequirement] = []
    if route.runtime_contract is not None:
        for resource_id, requirement in sorted(
            route.runtime_contract.resources.items()
        ):
            for residency_id in requirement.required_residency_ids:
                residency.append(ResidencyRequirement(
                    resource_id=resource_id,
                    residency_id=residency_id,
                    role="runtime",
                ))

    overlap_status = route.overlap.status
    if overlap_status == "diagnostic":
        overlap_status = "predicted"
    overlap = OverlapEstimate(
        status=overlap_status,
        exposed_join_wait_ppm=route.overlap.exposed_join_wait_ppm,
        upper_join_wait_ppm=route.overlap.upper_join_wait_ppm(),
        sample_count=route.overlap.sample_count,
    )
    quality_class = LEGACY_QUALITY.get(route.quality_class)
    if quality_class is None:
        raise SchedulerContractError("unknown legacy route quality class")

    try:
        granularity = PlacementGranularity(route.granularity)
    except ValueError as exc:
        raise SchedulerContractError("unknown legacy route granularity") from exc
    return RouteAlternative(
        route_id=route.route_id,
        workload_id=route.workload_id,
        baseline=route.baseline,
        placement_granularity=granularity,
        maturity=maturity,
        applicability=applicability or ApplicabilityContract.exact_for(unit),
        latency_us=MetricEstimate(
            mean=service_us,
            upper=service_upper_us,
            sample_count=route.latency.sample_count,
            measured=route.latency.measured,
        ),
        energy=energy,
        overlap=overlap,
        quality=QualityContract(
            quality_class=quality_class,
            finite_output_required=True,
            nonempty_output_required=True,
            validation_id=(
                None
                if quality_class == QualityClass.UNVERIFIED
                else route.evidence_ids[0]
            ),
        ),
        resources=tuple(resource_requirements),
        phase_leases=tuple(phase_leases),
        memory=tuple(memory),
        residency=tuple(residency),
        placement_verified=route.placement_verified,
        evidence_ids=route.evidence_ids,
        source_profile_id=profile_id,
    )


def adapt_legacy_profile(
    profile: Any,
    request: Any,
    unit: SchedulingUnit,
    *,
    accounting_by_route: Mapping[str, AccountingContract] | None = None,
    applicability_by_route: Mapping[str, ApplicabilityContract] | None = None,
) -> CandidateSet:
    accounting_rows = accounting_by_route or {}
    applicability_rows = applicability_by_route or {}
    routes = tuple(
        adapt_legacy_route(
            route,
            request,
            unit,
            profile.resources,
            profile.profile_id,
            accounting=accounting_rows.get(route.route_id),
            applicability=applicability_rows.get(route.route_id),
        )
        for route in profile.routes
        if route.workload_id == unit.workload_id
    )
    return CandidateSet(profile.profile_id, unit, routes)
