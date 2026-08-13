"""Load epoch-bound cohort certificates into canonical scheduler routes."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

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

    cold = models.get("cold")
    if type(cold) is not dict:
        raise CertifiedCohortError("cold model binding is missing")
    model = ModelIdentity(
        model_id="cold:" + cold.get("file_sha256"),
        model_hash=_sha(cold.get("file_sha256")),
        architecture=cold.get("architecture"),
        weight_format=cold.get("quantization"),
        weight_bytes=cold.get("file_bytes"),
    )

    raw_routes = compiled.get("compiled_routes")
    raw_contracts = contracts_file.get("routes")
    if type(raw_routes) is not list or type(raw_contracts) is not dict:
        raise CertifiedCohortError("compiled routes or contracts are missing")
    if not raw_routes:
        raise CertifiedCohortError("compiled routes are empty")
    maximum_duration_us = max(
        round(route["metrics"]["duration_s"]["max_observed_bound"] * 1_000_000)
        for route in raw_routes
    )
    unit = SchedulingUnit(
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
    applicability = ApplicabilityContract.exact_for(unit)
    accounting = AccountingContract(
        scope=AccountingScope.COHORT,
        kind=AccountingKind.NON_ADDITIVE_COHORT_TOTAL,
        boundary_id=boundary_id,
        work_set_hash=unit.work_set_hash,
    )
    comparison = compiled.get("cohort_comparison")
    if type(comparison) is not dict:
        raise CertifiedCohortError("cohort comparison is missing")
    wait_ppm = round(
        comparison.get("mean_exposed_join_wait_fraction") * 1_000_000
    )

    runtime_contracts: dict[str, RouteRuntimeContract] = {}
    dispatch_contracts: dict[str, Mapping[str, object]] = {}
    routes: list[RouteAlternative] = []
    for raw in raw_routes:
        if type(raw) is not dict:
            raise CertifiedCohortError("compiled route must be an object")
        route_id = raw.get("route_id")
        contract_raw = raw_contracts.get(route_id)
        if type(contract_raw) is not dict:
            raise CertifiedCohortError(f"runtime contract is missing: {route_id}")
        runtime_contract = RouteRuntimeContract.from_json(contract_raw)
        if runtime_contract.epoch_key != epoch_key:
            raise CertifiedCohortError("route runtime epoch differs")
        runtime_contracts[route_id] = runtime_contract

        dispatch = raw.get("dispatch_contract")
        if type(dispatch) is not dict:
            raise CertifiedCohortError("dispatch contract is missing")
        dispatch_contracts[route_id] = MappingProxyType(dict(dispatch))
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
        granularity = PlacementGranularity(raw.get("scope"))
        route_overlap = (
            OverlapEstimate("not_applicable", None, None, repetitions)
            if granularity == PlacementGranularity.TASK
            else OverlapEstimate("measured", wait_ppm, wait_ppm, repetitions)
        )
        evidence = compiled.get("physical_evidence")
        if type(evidence) is not dict:
            raise CertifiedCohortError("physical evidence is missing")
        routes.append(RouteAlternative(
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
            resources=tuple(resources),
            phase_leases=tuple(leases),
            memory=(),
            residency=tuple(residency),
            placement_verified=True,
            evidence_ids=(
                _sha(evidence.get("aggregate_file_sha256")),
                _sha(evidence.get("aggregate_record_sha256")),
            ),
            source_profile_id=compiled.get("certificate_set_id"),
        ))

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
        trace_sha256=trace_sha256,
        compiled_file_sha256=None,
    )
