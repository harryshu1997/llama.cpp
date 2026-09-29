"""Convert automated candidate sets into runtime cost rows."""

from __future__ import annotations

from ..model_manifest import ModelManifest
from ..runtime_capabilities import (
    HeterogeneousRuntimeSnapshot,
    RuntimeCapabilityCatalog,
)
from ..runtime_cost import (
    RuntimeCostEstimateSet,
    RuntimeModelArtifact,
    RuntimeRouteCostEstimate,
)
from ..runtime_plan import AutomatedCandidateSet, AutomatedRouteCandidate
from ..types import canonical_sha256
from ..adaptive_decode_contracts import AdaptiveDecodeError
from ..adaptive_decode_planning import adaptive_probe_contracts
from .common import (
    RouteGenerationError,
)


def candidate_set_to_runtime_costs(
    candidate_set: AutomatedCandidateSet,
    request,
    manifest: ModelManifest,
    snapshot: HeterogeneousRuntimeSnapshot,
    catalog: RuntimeCapabilityCatalog,
    *,
    planning_profile_sha256: str | None = None,
    model_manifest_sha256: str | None = None,
    runtime_system_snapshot_sha256: str | None = None,
) -> RuntimeCostEstimateSet:
    estimates = []
    candidate_by_id = {
        row.candidate_id: row for row in candidate_set.candidates
    }
    if any(
        row.paired_baseline_route_id is not None
        and row.paired_baseline_route_id not in candidate_by_id
        for row in candidate_set.candidates
    ):
        raise RouteGenerationError(
            "paired baseline candidate is absent"
        )
    control = candidate_set.baseline
    try:
        adaptive_contracts = adaptive_probe_contracts(
            candidate_set,
            manifest,
            catalog,
            request.output_tokens,
        )
    except AdaptiveDecodeError:
        adaptive_contracts = {}

    def comparison(
        treatment: AutomatedRouteCandidate,
        baseline: AutomatedRouteCandidate,
    ) -> dict[str, object]:
        paired = treatment.residency_break_even or {}
        paired_delta_upper_uj = paired.get(
            "paired_energy_delta_upper_uj"
        )
        paired_conditional_upper_uj = None
        if (
            paired.get("paired_energy_evidence")
                in {"MEASURED", "ASSUMED_4P5W"}
            and paired.get("paired_energy_parent_route_id")
                == baseline.candidate_id
            and paired.get("paired_energy_parent_placement_sha256")
                == baseline.plan.desktop_placement_sha256
            and type(paired_delta_upper_uj) is int
            and baseline.cost.fleet_energy_lower_uj is not None
        ):
            value = (
                baseline.cost.fleet_energy_lower_uj
                + paired_delta_upper_uj
            )
            if value > 0:
                paired_conditional_upper_uj = value
        required_upper_uj = (
            None
            if baseline.cost.fleet_energy_lower_uj is None
            else baseline.cost.fleet_energy_lower_uj * (
                1_000_000 - catalog.minimum_energy_saving_ppm
            ) // 1_000_000
        )
        return {
            "baseline_energy_lower_uj": (
                baseline.cost.fleet_energy_lower_uj
            ),
            "baseline_executor_id": baseline.binding.executor_id,
            "baseline_route_id": baseline.candidate_id,
            "desktop_placement_sha256": (
                baseline.plan.desktop_placement_sha256
            ),
            "placement_matches": (
                baseline.plan.desktop_placement_sha256 is not None
                and baseline.plan.desktop_placement_sha256
                    == treatment.plan.desktop_placement_sha256
            ),
            "paired_energy_delta_upper_uj": paired_delta_upper_uj,
            "paired_treatment_conditional_upper_uj": (
                paired_conditional_upper_uj
            ),
            "required_treatment_upper_uj": required_upper_uj,
            "treatment_energy_upper_uj": (
                treatment.cost.fleet_energy_upper_uj
            ),
        }

    for candidate in candidate_set.candidates:
        route_profile = next((
            row for row in catalog.route_shape_profiles
            if row.selector_id == candidate.plan.route_profile_id
        ), None)
        additional: dict[str, int] = {}
        for demand in candidate.plan.memory_demands:
            additional[demand.resource_id] = (
                additional.get(demand.resource_id, 0) + demand.additional_bytes
            )
        details = {
            "adaptive_decode_contract": adaptive_contracts.get(
                candidate.candidate_id
            ),
            "assisted_operator_kind": candidate.assisted_operator_kind,
            "cost_breakdown": candidate.cost.to_json(),
            "device_ids": list(candidate.device_ids),
            "maturity": candidate.maturity,
            "model_placement_epoch": (
                candidate_set.search_metadata.get(
                    "model_placement_epoch"
                )
            ),
            "model_placement_epoch_fast_path": (
                candidate_set.search_metadata.get(
                    "model_placement_epoch_fast_path", False
                )
            ),
            "model_placement_epoch_invalidation_reason": (
                candidate_set.search_metadata.get(
                    "model_placement_epoch_invalidation_reason", "NONE"
                )
            ),
            "model_placement_resolution": (
                candidate_set.search_metadata.get(
                    "model_placement_resolution"
                )
            ),
            "latency_evidence": candidate.cost.latency_evidence,
            "energy_evidence": candidate.cost.energy_evidence,
            "phone_power_evidence_kind": (
                candidate.plan.adapter_parameters.get(
                    "phone_power_evidence_kind"
                )
            ),
            "phone_power_estimation_version": (
                candidate.plan.adapter_parameters.get(
                    "phone_power_estimation_version"
                )
            ),
            "desktop_control_comparison": comparison(
                candidate, control
            ),
            "desktop_placement_sha256": (
                candidate.plan.desktop_placement_sha256
            ),
            "marginal_system_cost": (
                None
                if candidate.marginal_system_cost is None
                else dict(candidate.marginal_system_cost)
            ),
            "operator_count": len(candidate.plan.operators),
            "operator_plan_sha256": candidate.plan.plan_sha256,
            "overlap_kind": candidate.plan.overlap_kind,
            "pareto_dominated": candidate.pareto_dominated,
            "primary_rejection_reason": (
                candidate.primary_rejection_reason
            ),
            "rejection_reasons": list(candidate.rejection_reasons),
            "residency_variant": candidate.residency_variant,
            "residency_break_even": (
                None
                if candidate.residency_break_even is None
                else dict(candidate.residency_break_even)
            ),
            "resource_ids": list(candidate.plan.resource_ids),
            "route_family": candidate.route_family,
            "route_template_audit_sha256": (
                candidate_set.search_metadata.get(
                    "route_template_audit_sha256"
                )
            ),
            "route_template_exact_request_reuse": (
                candidate_set.search_metadata.get(
                    "route_template_exact_request_reuse", False
                )
            ),
            "route_template_cross_shape_reuse": (
                candidate_set.search_metadata.get(
                    "route_template_cross_shape_reuse", False
                )
            ),
            "route_template_request_shape_bucket": (
                candidate_set.search_metadata.get(
                    "route_template_request_shape_bucket"
                )
            ),
            "route_template_source_shape_bucket": (
                candidate_set.search_metadata.get(
                    "route_template_source_shape_bucket"
                )
            ),
            "paired_baseline_comparison": (
                None
                if candidate.paired_baseline_route_id is None
                else comparison(
                    candidate,
                    candidate_by_id[
                        candidate.paired_baseline_route_id
                    ],
                )
            ),
            "paired_baseline_route_id": (
                candidate.paired_baseline_route_id
            ),
            "recovery_fallback": (
                candidate.candidate_id
                    == candidate_set.recovery_fallback_route_id
            ),
            "split_axis": candidate.split_axis,
            "split_fraction_ppm": candidate.split_fraction_ppm,
            "system_finish_upper_us": (
                candidate.system_finish_upper_us
            ),
            "transitions": [
                row.to_json() for row in candidate.plan.transitions
            ],
        }
        estimates.append(RuntimeRouteCostEstimate(
            route_id=candidate.candidate_id,
            executor_id=candidate.binding.executor_id,
            baseline=candidate.baseline,
            admitted=candidate.admitted,
            reason=(
                "ADMITTED" if candidate.admitted
                else candidate.primary_rejection_reason
            ),
            service_us=candidate.cost.service_us,
            service_upper_us=candidate.cost.service_upper_us,
            latency_profile_label=(
                "gguf-operator-dag"
                if route_profile is None
                else route_profile.selector_id
            ),
            latency_sample_count=(
                1
                if route_profile is None
                else route_profile.sample_count
            ),
            latency_measured=(
                candidate.cost.latency_evidence == "MEASURED"
            ),
            fleet_energy_uj=candidate.cost.fleet_energy_uj,
            fleet_energy_lower_uj=candidate.cost.fleet_energy_lower_uj,
            fleet_energy_upper_uj=candidate.cost.fleet_energy_upper_uj,
            additional_bytes=sum(additional.values()),
            memory_resource_id=candidate.binding.memory_resource_id,
            additional_bytes_by_resource=additional,
            memory_demands=candidate.plan.memory_demands,
            details=details,
            _owned_details=True,
        ))
    return RuntimeCostEstimateSet(
        request_id=request.request_id,
        workload_id=request.workload_id,
        model=RuntimeModelArtifact(
            manifest.model_id,
            manifest.artifact_sha256,
            manifest.artifact_bytes,
        ),
        snapshot=snapshot.memory,
        baseline_route_id=candidate_set.baseline_route_id,
        estimates=tuple(estimates),
        planning_profile_sha256=(
            canonical_sha256(catalog)
            if planning_profile_sha256 is None
            else planning_profile_sha256
        ),
        candidate_generation_sha256=candidate_set.generation_sha256,
        model_manifest_sha256=(
            canonical_sha256(manifest)
            if model_manifest_sha256 is None
            else model_manifest_sha256
        ),
        runtime_system_snapshot_sha256=(
            canonical_sha256(snapshot)
            if runtime_system_snapshot_sha256 is None
            else runtime_system_snapshot_sha256
        ),
    )
