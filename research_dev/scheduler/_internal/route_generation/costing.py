"""Per-candidate costing (`_one`), domination, rough cost and rough visit estimates.

Mixin of ``AutomatedRouteCompiler``; methods moved here verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from ..model_manifest import ModelManifest, ModelRequestWork
from ..placement import OperatorNode, TransferLink
from ..policy import SchedulerError
from ..runtime_capabilities import (
    HeterogeneousRuntimeSnapshot,
    RuntimeCompositeExecutorCapability,
    RuntimeExecutorCapability,
)
from ..capability_contracts.executors import whole_phone_launch_parameters
from ..runtime_cost import RuntimeExecutorBinding, RuntimeMemoryDemand
from ..runtime_plan import (
    AutomatedRouteCandidate,
    AutomatedRouteCost,
    RuntimeOperatorAssignment,
    RuntimeTransferCost,
)
from ..runtime_resources import transition_adjusted_memory_demands
from ..runtime_residency_cohorts import (
    RuntimeResidencyCohortHold,
    runtime_residency_component_identity_from_parts,
)
from ..runtime_search import RoughPlacementVisit, request_shape_bucket
from ..runtime_system_cost import RuntimeSystemCostError
from ..types import canonical_sha256
from .remote_resident import RouteRemoteResidentMixin
from .common import (
    RouteGenerationError,
    _DORMANT_PHONE_FFN_RUNTIME_PARAMETER,
    _MATURITY_RANK,
    _ceil_div,
    _minimum_maturity,
    _placement_rejection_reasons,
    _FfnResidentEnvelope,
    _Pattern,
)

__all__ = [
    'AutomatedRouteCandidate',
    'AutomatedRouteCost',
    'HeterogeneousRuntimeSnapshot',
    'ModelManifest',
    'ModelRequestWork',
    'OperatorNode',
    'RoughPlacementVisit',
    'RouteCostingMixin',
    'RouteGenerationError',
    'RouteRemoteResidentMixin',
    'RuntimeCompositeExecutorCapability',
    'RuntimeExecutorBinding',
    'RuntimeExecutorCapability',
    'RuntimeMemoryDemand',
    'RuntimeOperatorAssignment',
    'RuntimeResidencyCohortHold',
    'RuntimeSystemCostError',
    'RuntimeTransferCost',
    'SchedulerError',
    'TransferLink',
    '_DORMANT_PHONE_FFN_RUNTIME_PARAMETER',
    '_FfnResidentEnvelope',
    '_MATURITY_RANK',
    '_Pattern',
    '_ceil_div',
    '_minimum_maturity',
    '_placement_rejection_reasons',
    'canonical_sha256',
    'request_shape_bucket',
    'runtime_residency_component_identity_from_parts',
    'transition_adjusted_memory_demands',
    'whole_phone_launch_parameters',
]


@dataclass(frozen=True)
class _CandidateServiceEstimate:
    plan: object
    learned: object | None
    learned_decomposes_cold_energy: bool
    maturity: str
    component_service_us: int
    component_energy_uj: int | None
    transition_us: int
    transition_upper_us: int
    transition_energy_uj: int
    transition_energy_known: bool
    service_us: int
    physical_service_upper_us: int
    service_upper_us: int
    latency_evidence: str


@dataclass(frozen=True)
class _CandidateScheduleEstimate:
    reasons: list[str]
    residency_hysteresis_us: int
    start_us: int
    finish_us: int
    finish_upper_us: int
    queue_delay_us: int


@dataclass(frozen=True)
class _CandidateEnergyEstimate:
    energy_uj: int | None
    lower_uj: int | None
    upper_uj: int | None
    warm_energy_uj: int | None
    warm_lower_uj: int | None
    warm_upper_uj: int | None
    transition_energy_uj: int | None
    transition_lower_uj: int | None
    transition_upper_uj: int | None
    evidence: str
    queue_idle_energy_uj: int = 0
    queue_idle_power_mw: int = 0


@dataclass(frozen=True)
class _CandidateInputs:
    route_id: str
    coordinator: RuntimeCompositeExecutorCapability | None
    memory_demands: tuple[RuntimeMemoryDemand, ...]
    resident_envelope: _FfnResidentEnvelope | None
    transitions: tuple
    transition_executor_id: str
    route_profile: object | None
    use_assumed_phone_energy: bool
    maturity: str
    used_links: tuple[str, ...]
    resources: tuple[str, ...]
    interference_resources: tuple[str, ...]
    resource_slots: Mapping[str, int]
    protected_work: object | None
    system_profile: object | None
    system_cost_known: bool
    control_delay_us: int
    control_delay_upper_us: int
    adapter_parameters: Mapping[str, object]
    runtime_partition_error: str | None
    transport_error: str | None


@dataclass(frozen=True)
class _PreparedCandidate:
    inputs: _CandidateInputs
    plan: object
    transitions: tuple
    transition_estimates: tuple
    component_capability_sha256: str


def _physical_interference_resources(catalog, resources, used_links, coordinator, system_profile):
    """Resolve qualified FFN link aliases to their declared shared transport."""
    values = {key for key in resources if catalog.resources[key].kind not in {
        "coordinator", "memory-pool",
    }}
    if (coordinator is None or getattr(coordinator, "assisted_operator_kind", None) != "ffn"
            or system_profile is None):
        return tuple(sorted(values))
    phone_id = coordinator.adapter_parameters.get("phone_device_id")
    phone = catalog.executor_by_device.get(phone_id)
    groups = set(() if phone is None else (
        row.shared_transport_resource_ids for row in phone.phone_sessions
    ))
    if len(groups) != 1:
        return tuple(sorted(values))
    shared = set(next(iter(groups)))
    if (not shared or not shared.issubset(values)
            or coordinator.adapter_parameters.get("functionfs_resource_id") not in shared
            or any(catalog.resources[key].kind != "transport" for key in shared)):
        return tuple(sorted(values))
    for link in catalog.placement_profile.links:
        logical = "link:" + link.link_id
        if (link.link_id in used_links and logical in values
                and logical not in system_profile.interference_ppm_by_resource
                and phone_id in {link.source_device, link.target_device}
                and link.status == "measured" and link.transport_generation is not None
                and link.transport_profile_id is not None
                and link.qualification_identity_sha256 is not None):
            values.remove(logical)
            values.update(shared)
    return tuple(sorted(values))


class RouteCostingMixin:
    """Per-candidate costing (`_one`), domination, rough cost and rough visit estimates."""

    def _candidate_inputs(
        self,
        *,
        manifest: ModelManifest,
        work: ModelRequestWork,
        snapshot: HeterogeneousRuntimeSnapshot,
        profile,
        pattern: _Pattern,
        residency_variant: str,
        residency_states: Mapping[str, str],
        observed_at_us: int,
        placement,
    ) -> _CandidateInputs:
        route_id = f"auto:{pattern.route_key}:residency:{residency_variant}"
        coordinator = self._coordinator(pattern)
        memory_demands = self._memory_demands(
            manifest, work, pattern, snapshot, residency_states
        )
        (
            memory_demands,
            resident_envelope,
            resident_envelope_error,
            phone_shards_require_transition,
        ) = self._resident_phone_memory_demands(
            manifest, pattern, snapshot, coordinator, memory_demands
        )
        remote_resident = RouteRemoteResidentMixin._remote_resident_group(
            self, manifest, pattern, coordinator
        )
        if remote_resident is not None:
            memory_demands = self._remote_resident_memory_demands(
                manifest, remote_resident, memory_demands, snapshot
            )
        memory_demands = self._preallocated_request_memory_demands(
            manifest, work, pattern, snapshot, coordinator, memory_demands
        )
        memory_demands, transitions, transition_executor_id = (
            self._candidate_transitions(
                manifest,
                pattern,
                snapshot,
                coordinator,
                residency_states,
                memory_demands,
                resident_envelope,
                phone_shards_require_transition,
            )
        )
        (
            route_profile,
            phone_power_profiles,
            use_assumed_phone_energy,
            maturity,
        ) = self._candidate_route_profile(
            manifest,
            work,
            snapshot,
            pattern,
            residency_variant,
            coordinator,
            transitions,
            resident_envelope,
            placement,
        )
        resource_context = self._candidate_resource_context(
            work,
            snapshot,
            pattern,
            coordinator,
            transitions,
            placement,
            observed_at_us,
            profile=profile,
        )
        (
            used_links,
            resources,
            interference_resources,
            resource_slots,
            protected_work,
            system_profile,
            system_cost_known,
            control_delay_us,
            control_delay_upper_us,
        ) = resource_context
        adapter_parameters = self._base_candidate_adapter_parameters(
            manifest,
            pattern,
            snapshot,
            residency_variant,
            coordinator,
            phone_power_profiles,
        )
        runtime_partition_error = resident_envelope_error
        selected_geometry = self._selected_ffn_geometry(
            manifest, pattern, adapter_parameters, resident_envelope
        )
        if selected_geometry is not None:
            runtime_partition_error = self._apply_ffn_partition_parameters(
                pattern,
                coordinator,
                adapter_parameters,
                resident_envelope,
                manifest.feed_forward_length,
                selected_geometry[0],
                selected_geometry[1],
                runtime_partition_error,
            )
        transport_error = self._candidate_transport_parameters(
            profile, used_links, work, pattern, adapter_parameters
        )
        return _CandidateInputs(
            route_id=route_id,
            coordinator=coordinator,
            memory_demands=memory_demands,
            resident_envelope=resident_envelope,
            transitions=transitions,
            transition_executor_id=transition_executor_id,
            route_profile=route_profile,
            use_assumed_phone_energy=use_assumed_phone_energy,
            maturity=maturity,
            used_links=used_links,
            resources=resources,
            interference_resources=interference_resources,
            resource_slots=resource_slots,
            protected_work=protected_work,
            system_profile=system_profile,
            system_cost_known=system_cost_known,
            control_delay_us=control_delay_us,
            control_delay_upper_us=control_delay_upper_us,
            adapter_parameters=adapter_parameters,
            runtime_partition_error=runtime_partition_error,
            transport_error=transport_error,
        )

    def _prepare_candidate(
        self,
        *,
        manifest: ModelManifest,
        pattern: _Pattern,
        residency_variant: str,
        assignments: tuple[RuntimeOperatorAssignment, ...],
        inputs: _CandidateInputs,
        snapshot: HeterogeneousRuntimeSnapshot,
    ) -> _PreparedCandidate:
        plan, execution_contract, desktop_sha256, overlap_kind = (
            self._candidate_execution_plan(
                snapshot=snapshot,
                route_id=inputs.route_id,
                manifest=manifest,
                pattern=pattern,
                coordinator=inputs.coordinator,
                resident_envelope=inputs.resident_envelope,
                adapter_parameters=inputs.adapter_parameters,
                memory_demands=inputs.memory_demands,
                transition_executor_id=inputs.transition_executor_id,
                residency_variant=residency_variant,
                assignments=assignments,
                transitions=inputs.transitions,
                resources=inputs.resources,
                route_profile=inputs.route_profile,
                resource_slots=inputs.resource_slots,
            )
        )
        transitions, plan, estimates, component_sha256 = (
            self._candidate_measured_transitions(
                manifest=manifest,
                plan=plan,
                transitions=inputs.transitions,
                transition_executor_id=inputs.transition_executor_id,
                route_id=inputs.route_id,
                pattern=pattern,
                residency_variant=residency_variant,
                overlap_kind=overlap_kind,
                assignments=assignments,
                resources=inputs.resources,
                memory_demands=inputs.memory_demands,
                execution_contract=execution_contract,
                route_profile=inputs.route_profile,
                resource_slots=inputs.resource_slots,
                adapter_parameters=inputs.adapter_parameters,
                desktop_placement_sha256=desktop_sha256,
            )
        )
        return _PreparedCandidate(
            inputs=inputs,
            plan=plan,
            transitions=transitions,
            transition_estimates=estimates,
            component_capability_sha256=component_sha256,
        )

    def _finalize_candidate(
        self,
        *,
        request,
        manifest: ModelManifest,
        work: ModelRequestWork,
        profile,
        pattern: _Pattern,
        placement,
        inputs: _CandidateInputs,
        plan,
        transitions: tuple,
        maturity: str,
        service: _CandidateServiceEstimate,
        schedule: _CandidateScheduleEstimate,
        energy: _CandidateEnergyEstimate,
        marginal_system_cost,
        system_finish_upper_us: int | None,
    ) -> AutomatedRouteCandidate:
        transfer_us, transfer_costs = self._candidate_transfer_costs(placement)
        compute_us, memory_us, join_wait, memory_by_resource = (
            self._candidate_component_times(
                manifest=manifest,
                work=work,
                pattern=pattern,
                profile=profile,
                placement=placement,
                memory_demands=inputs.memory_demands,
            )
        )
        cost = self._candidate_cost_record(
            request=request,
            service=service,
            schedule=schedule,
            energy=energy,
            compute_us=compute_us,
            memory_us=memory_us,
            transfer_us=transfer_us,
            join_wait_us=join_wait,
            memory_by_resource=memory_by_resource,
            transfer_costs=transfer_costs,
            marginal_system_cost=marginal_system_cost,
        )
        reasons = tuple(sorted(set(schedule.reasons)))
        binding = self._candidate_binding(
            route_id=inputs.route_id,
            manifest=manifest,
            pattern=pattern,
            coordinator=inputs.coordinator,
            resources=inputs.resources,
            memory_demands=inputs.memory_demands,
            transitions=transitions,
            reasons=reasons,
            plan=plan,
        )
        return AutomatedRouteCandidate(
            candidate_id=inputs.route_id,
            plan=plan,
            binding=binding,
            cost=cost,
            maturity=maturity,
            admitted=not reasons,
            rejection_reasons=reasons,
            baseline=False,
            marginal_system_cost=marginal_system_cost,
            system_finish_upper_us=system_finish_upper_us,
        )

    def _one(
        self,
        request,
        manifest: ModelManifest,
        work: ModelRequestWork,
        snapshot: HeterogeneousRuntimeSnapshot,
        profile,
        pattern: _Pattern,
        residency_variant: str,
        residency_states: Mapping[str, str],
        observed_at_us: int,
        nodes: tuple[OperatorNode, ...],
        assignments: tuple[RuntimeOperatorAssignment, ...],
        placement,
        placement_error: str | None,
        residency_holds: Mapping[str, RuntimeResidencyCohortHold],
    ) -> AutomatedRouteCandidate:
        inputs = self._candidate_inputs(
            manifest=manifest,
            work=work,
            snapshot=snapshot,
            profile=profile,
            pattern=pattern,
            residency_variant=residency_variant,
            residency_states=residency_states,
            observed_at_us=observed_at_us,
            placement=placement,
        )
        prepared = self._prepare_candidate(
            snapshot=snapshot,
            manifest=manifest,
            pattern=pattern,
            residency_variant=residency_variant,
            assignments=assignments,
            inputs=inputs,
        )
        plan = prepared.plan
        transitions = prepared.transitions
        transition_estimates = prepared.transition_estimates
        route_capability_sha256 = self.route_capability_identity(
            plan, inputs.transition_executor_id
        )
        service = self._candidate_service_estimate(
            request=request,
            manifest=manifest,
            work=work,
            snapshot=snapshot,
            pattern=pattern,
            plan=plan,
            transitions=transitions,
            transition_estimates=transition_estimates,
            transition_executor_id=inputs.transition_executor_id,
            route_capability_sha256=route_capability_sha256,
            component_capability_sha256=(
                prepared.component_capability_sha256
            ),
            placement=placement,
            route_profile=inputs.route_profile,
            adapter_parameters=inputs.adapter_parameters,
            maturity=inputs.maturity,
            control_delay_upper_us=inputs.control_delay_upper_us,
        )
        plan = service.plan
        maturity = service.maturity
        schedule = self._candidate_schedule_estimate(
            request=request,
            manifest=manifest,
            pattern=pattern,
            snapshot=snapshot,
            observed_at_us=observed_at_us,
            route_id=inputs.route_id,
            residency_variant=residency_variant,
            residency_states=residency_states,
            transitions=transitions,
            transition_estimates=transition_estimates,
            used_links=inputs.used_links,
            maturity=maturity,
            memory_demands=inputs.memory_demands,
            coordinator=inputs.coordinator,
            transport_error=inputs.transport_error,
            runtime_partition_error=inputs.runtime_partition_error,
            placement_error=placement_error,
            protected_work=inputs.protected_work,
            system_cost_known=inputs.system_cost_known,
            residency_holds=residency_holds,
            resource_slots=inputs.resource_slots,
            service=service,
        )
        reasons = schedule.reasons
        energy = self._candidate_energy_estimate(
            profile=profile,
            work=work,
            snapshot=snapshot,
            transitions=transitions,
            transition_estimates=transition_estimates,
            service=service,
            schedule=schedule,
            placement=placement,
            route_profile=inputs.route_profile,
            use_assumed_phone_energy=inputs.use_assumed_phone_energy,
            control_delay_us=inputs.control_delay_us,
            control_delay_upper_us=inputs.control_delay_upper_us,
        )
        energy = self._validate_candidate_energy(
            energy, service, transitions, reasons
        )

        marginal_system_cost, system_finish_upper_us, energy = (
            self._candidate_marginal_system_cost(
                system_cost_known=inputs.system_cost_known,
                protected_work=inputs.protected_work,
                system_profile=inputs.system_profile,
                interference_resources=inputs.interference_resources,
                service=service,
                schedule=schedule,
                energy=energy,
            )
        )

        return self._finalize_candidate(
            request=request,
            manifest=manifest,
            work=work,
            profile=profile,
            pattern=pattern,
            placement=placement,
            inputs=inputs,
            plan=plan,
            transitions=transitions,
            maturity=maturity,
            service=service,
            schedule=schedule,
            energy=energy,
            marginal_system_cost=marginal_system_cost,
            system_finish_upper_us=system_finish_upper_us,
        )
