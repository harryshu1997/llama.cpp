"""Load epoch-bound cohort certificates into canonical scheduler routes."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, NamedTuple, Sequence

from .runtime_gates import RouteRuntimeContract
from .types import (
    AccountingContract,
    AccountingKind,
    AccountingScope,
    ApplicabilityContract,
    CandidateSet,
    EnergyComponent,
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
    UnitKind,
    make_work_set_hash,
)


__all__ = [
    "CertifiedCohortError",
    "CertifiedCohortProfile",
    "load_certified_cohort",
]


class CertifiedCohortError(ValueError):
    pass


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CertifiedCohortError(f"cannot read {path}: {exc}") from exc
    if type(value) is not dict:
        raise CertifiedCohortError(f"{path} must contain an object")
    return value


def _sha(value: object) -> str:
    if type(value) is not str:
        raise CertifiedCohortError("expected SHA-256 string")
    digest = value.removeprefix("sha256:")
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise CertifiedCohortError("invalid SHA-256 string")
    return "sha256:" + digest


def _resource_kind(resource_id: str) -> tuple[str, str]:
    if resource_id.startswith("cuda"):
        return "gpu", "compute"
    if resource_id.startswith("cpu"):
        return "cpu", "compute"
    if "usb" in resource_id:
        return "usb", "transport"
    if resource_id.startswith("op15-"):
        return "phone", "compute"
    return "resource", "compute"


@dataclass(frozen=True)
class CertifiedCohortProfile:
    candidates: CandidateSet
    runtime_contracts: Mapping[str, RouteRuntimeContract]
    dispatch_contracts: Mapping[str, Mapping[str, object]]
    epoch_key: str
    trace_sha256: str
    compiled_file_sha256: str | None


class _EpochBindings(NamedTuple):
    workload: dict[str, Any]
    models: dict[str, Any]
    trace_sha256: str
    members: tuple[str, ...]


@dataclass(frozen=True)
class _CompiledRoute:
    route_id: str
    runtime_contract: RouteRuntimeContract
    dispatch_contract: Mapping[str, object]
    route: RouteAlternative


def _verify_bundle_headers(
    compiled: Mapping[str, Any],
    epoch: Mapping[str, Any],
    contracts_file: Mapping[str, Any],
) -> str:
    if compiled.get("schema") != "s42-epoch-route-bundle-v1":
        raise CertifiedCohortError("compiled route schema mismatch")
    if epoch.get("schema") != "s42-runtime-epoch-v1":
        raise CertifiedCohortError("runtime epoch schema mismatch")
    if contracts_file.get("schema") != "s42-i3-runtime-gate-contracts-v1":
        raise CertifiedCohortError("runtime contract set schema mismatch")
    if compiled.get("status") != "PASS":
        raise CertifiedCohortError("compiled routes did not pass")
    epoch_key = _sha(compiled.get("epoch_key"))
    if _sha(contracts_file.get("epoch_key")) != epoch_key:
        raise CertifiedCohortError("runtime contract epoch differs")
    return epoch_key


def _epoch_bindings(
    epoch: Mapping[str, Any],
    member_request_ids: Sequence[str],
    quality_requirement: QualityClass,
) -> _EpochBindings:
    bindings = epoch.get("bindings")
    if type(bindings) is not dict:
        raise CertifiedCohortError("runtime epoch bindings are missing")
    workload = bindings.get("workload")
    models = bindings.get("models")
    if type(workload) is not dict or type(models) is not dict:
        raise CertifiedCohortError("workload or model binding is missing")
    trace_sha256 = _sha(workload.get("sha256"))
    members = tuple(member_request_ids)
    if len(members) != workload.get("requests"):
        raise CertifiedCohortError("cohort request count differs from epoch")
    if len(members) != len(set(members)):
        raise CertifiedCohortError("cohort request ids are not unique")
    if not isinstance(quality_requirement, QualityClass):
        raise CertifiedCohortError("quality requirement is invalid")
    return _EpochBindings(workload, models, trace_sha256, members)


def _cold_model(models: Mapping[str, Any]) -> ModelIdentity:
    cold = models.get("cold")
    if type(cold) is not dict:
        raise CertifiedCohortError("cold model binding is missing")
    return ModelIdentity(
        model_id="cold:" + cold.get("file_sha256"),
        model_hash=_sha(cold.get("file_sha256")),
        architecture=cold.get("architecture"),
        weight_format=cold.get("quantization"),
        weight_bytes=cold.get("file_bytes"),
    )


def _cohort_unit(
    *,
    raw_routes: list[Any],
    workload: Mapping[str, Any],
    workload_id: str,
    unit_id: str,
    members: tuple[str, ...],
    model: ModelIdentity,
    quality_requirement: QualityClass,
    epoch_key: str,
) -> SchedulingUnit:
    maximum_duration_us = max(
        round(route["metrics"]["duration_s"]["max_observed_bound"] * 1_000_000)
        for route in raw_routes
    )
    return SchedulingUnit(
        unit_id=unit_id,
        kind=UnitKind.COHORT,
        member_request_ids=members,
        work_set_hash=make_work_set_hash(members),
        workload_id=workload_id,
        model=model,
        arrival_us=0,
        deadline_us=maximum_duration_us + 1,
        input_tokens=workload.get("input_tokens"),
        output_tokens=workload.get("output_tokens"),
        features={
            "cold_requests": workload.get("cold_requests"),
            "hot_requests": workload.get("hot_requests"),
            "requests": workload.get("requests"),
        },
        quality_requirement=quality_requirement,
        semantics={
            "full_logits": True,
            "kv_owner": "llama_context",
            "sampler_location": "desktop",
        },
        epoch_id=epoch_key,
    )


def _join_wait_ppm(compiled: Mapping[str, Any]) -> int:
    comparison = compiled.get("cohort_comparison")
    if type(comparison) is not dict:
        raise CertifiedCohortError("cohort comparison is missing")
    return round(
        comparison.get("mean_exposed_join_wait_fraction") * 1_000_000
    )


def _route_metric_estimates(
    raw: Mapping[str, Any],
) -> tuple[MetricEstimate, MetricEstimate, Any]:
    metrics = raw.get("metrics")
    if type(metrics) is not dict:
        raise CertifiedCohortError("route metrics are missing")
    duration = metrics.get("duration_s")
    fleet = metrics.get("fleet_j")
    repetitions = metrics.get("repetitions")
    if type(duration) is not dict or type(fleet) is not dict:
        raise CertifiedCohortError("route metric bounds are missing")
    latency = MetricEstimate(
        mean=round(duration.get("mean") * 1_000_000),
        upper=round(duration.get("max_observed_bound") * 1_000_000),
        lower=round(duration.get("min") * 1_000_000),
        sample_count=repetitions,
        measured=True,
    )
    energy = MetricEstimate(
        mean=round(fleet.get("mean") * 1_000_000),
        upper=round(fleet.get("max_observed_bound") * 1_000_000),
        lower=round(fleet.get("min") * 1_000_000),
        sample_count=repetitions,
        measured=True,
    )
    return latency, energy, repetitions


def _route_resources(
    route_id: str,
    runtime_contract: RouteRuntimeContract,
    latency: MetricEstimate,
) -> tuple[
    tuple[ResourceRequirement, ...],
    tuple[PhaseLease, ...],
    tuple[ResidencyRequirement, ...],
]:
    resources: list[ResourceRequirement] = []
    leases: list[PhaseLease] = []
    residency: list[ResidencyRequirement] = []
    for resource_id, requirement in sorted(runtime_contract.resources.items()):
        kind, role = _resource_kind(resource_id)
        resources.append(ResourceRequirement(
            resource_id=resource_id,
            kind=kind,
            role=role,
            slots=1,
        ))
        leases.append(PhaseLease(
            lease_id=f"{route_id}:{resource_id}",
            resource_id=resource_id,
            slots=1,
            start_offset_us=0,
            duration=latency,
        ))
        residency.extend(
            ResidencyRequirement(resource_id, item, "runtime")
            for item in requirement.required_residency_ids
        )
    return tuple(resources), tuple(leases), tuple(residency)


def _route_quality_class(raw: Mapping[str, Any]) -> QualityClass:
    quality_raw = raw.get("quality")
    if type(quality_raw) is not dict:
        raise CertifiedCohortError("route quality is missing")
    quality_name = quality_raw.get("class")
    quality = {
        "exact": QualityClass.EXACT,
        "approximate": QualityClass.SEMANTIC,
    }.get(quality_name)
    if quality is None:
        raise CertifiedCohortError("route quality class is unsupported")
    return quality


def _compile_route(
    raw: object,
    *,
    raw_contracts: Mapping[str, Any],
    compiled: Mapping[str, Any],
    epoch_key: str,
    workload_id: str,
    applicability: ApplicabilityContract,
    accounting: AccountingContract,
    wait_ppm: int,
) -> _CompiledRoute:
    if type(raw) is not dict:
        raise CertifiedCohortError("compiled route must be an object")
    route_id = raw.get("route_id")
    contract_raw = raw_contracts.get(route_id)
    if type(contract_raw) is not dict:
        raise CertifiedCohortError(f"runtime contract is missing: {route_id}")
    runtime_contract = RouteRuntimeContract.from_json(contract_raw)
    if runtime_contract.epoch_key != epoch_key:
        raise CertifiedCohortError("route runtime epoch differs")

    dispatch = raw.get("dispatch_contract")
    if type(dispatch) is not dict:
        raise CertifiedCohortError("dispatch contract is missing")
    latency, energy, repetitions = _route_metric_estimates(raw)
    resources, leases, residency = _route_resources(
        route_id, runtime_contract, latency
    )
    quality = _route_quality_class(raw)
    granularity = PlacementGranularity(raw.get("scope"))
    route_overlap = (
        OverlapEstimate("not_applicable", None, None, repetitions)
        if granularity == PlacementGranularity.TASK
        else OverlapEstimate("measured", wait_ppm, wait_ppm, repetitions)
    )
    evidence = compiled.get("physical_evidence")
    if type(evidence) is not dict:
        raise CertifiedCohortError("physical evidence is missing")
    route = RouteAlternative(
        route_id=route_id,
        workload_id=workload_id,
        baseline=raw.get("role") == "control",
        placement_granularity=granularity,
        maturity=RouteMaturity.STABLE,
        applicability=applicability,
        latency_us=latency,
        energy=(EnergyComponent(energy, accounting),),
        overlap=route_overlap,
        quality=QualityContract(
            quality_class=quality,
            validation_id=_sha(evidence.get("aggregate_record_sha256")),
        ),
        resources=resources,
        phase_leases=leases,
        memory=(),
        residency=residency,
        placement_verified=True,
        evidence_ids=(
            _sha(evidence.get("aggregate_file_sha256")),
            _sha(evidence.get("aggregate_record_sha256")),
        ),
        source_profile_id=compiled.get("certificate_set_id"),
    )
    return _CompiledRoute(
        route_id=route_id,
        runtime_contract=runtime_contract,
        dispatch_contract=MappingProxyType(dict(dispatch)),
        route=route,
    )


def load_certified_cohort(
    *,
    compiled_path: Path,
    epoch_path: Path,
    runtime_contracts_path: Path,
    workload_id: str,
    unit_id: str,
    member_request_ids: Sequence[str],
    quality_requirement: QualityClass,
    boundary_id: str,
) -> CertifiedCohortProfile:
    compiled = _read_object(compiled_path)
    epoch = _read_object(epoch_path)
    contracts_file = _read_object(runtime_contracts_path)
    epoch_key = _verify_bundle_headers(compiled, epoch, contracts_file)
    bindings = _epoch_bindings(epoch, member_request_ids, quality_requirement)
    model = _cold_model(bindings.models)

    raw_routes = compiled.get("compiled_routes")
    raw_contracts = contracts_file.get("routes")
    if type(raw_routes) is not list or type(raw_contracts) is not dict:
        raise CertifiedCohortError("compiled routes or contracts are missing")
    if not raw_routes:
        raise CertifiedCohortError("compiled routes are empty")
    unit = _cohort_unit(
        raw_routes=raw_routes,
        workload=bindings.workload,
        workload_id=workload_id,
        unit_id=unit_id,
        members=bindings.members,
        model=model,
        quality_requirement=quality_requirement,
        epoch_key=epoch_key,
    )
    applicability = ApplicabilityContract.exact_for(unit)
    accounting = AccountingContract(
        scope=AccountingScope.COHORT,
        kind=AccountingKind.NON_ADDITIVE_COHORT_TOTAL,
        boundary_id=boundary_id,
        work_set_hash=unit.work_set_hash,
    )
    wait_ppm = _join_wait_ppm(compiled)

    runtime_contracts: dict[str, RouteRuntimeContract] = {}
    dispatch_contracts: dict[str, Mapping[str, object]] = {}
    routes: list[RouteAlternative] = []
    for raw in raw_routes:
        compiled_route = _compile_route(
            raw,
            raw_contracts=raw_contracts,
            compiled=compiled,
            epoch_key=epoch_key,
            workload_id=workload_id,
            applicability=applicability,
            accounting=accounting,
            wait_ppm=wait_ppm,
        )
        runtime_contracts[compiled_route.route_id] = compiled_route.runtime_contract
        dispatch_contracts[compiled_route.route_id] = compiled_route.dispatch_contract
        routes.append(compiled_route.route)

    candidates = CandidateSet(
        profile_id=compiled.get("certificate_set_id"),
        unit=unit,
        routes=tuple(routes),
    )
    return CertifiedCohortProfile(
        candidates=candidates,
        runtime_contracts=MappingProxyType(runtime_contracts),
        dispatch_contracts=MappingProxyType(dispatch_contracts),
        epoch_key=epoch_key,
        trace_sha256=bindings.trace_sha256,
        compiled_file_sha256=None,
    )
