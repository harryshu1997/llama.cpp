"""Route costing: service, schedule, energy and marginal system cost estimates."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping
from ..model_manifest import ModelManifest, ModelRequestWork
from ..policy import SchedulerError
from ..runtime_capabilities import (
    HeterogeneousRuntimeSnapshot,
    RuntimeCompositeExecutorCapability,
)
from ..runtime_cost import RuntimeMemoryDemand
from ..runtime_plan import AutomatedRouteCost, RuntimeTransferCost
from ..runtime_residency_cohorts import RuntimeResidencyCohortHold
from ..runtime_system_cost import RuntimeSystemCostError
from .common import (
    RouteGenerationError,
    _ceil_div,
    _minimum_maturity,
    _placement_rejection_reasons,
    _Pattern,
)
from .costing import (
    _CandidateEnergyEstimate,
    _CandidateScheduleEstimate,
    _CandidateServiceEstimate,
)


class RouteEstimateMixin:
    """Route costing: service, schedule, energy and marginal system cost estimates."""

    def _candidate_service_estimate(
        self,
        *,
        request,
        manifest: ModelManifest,
        work: ModelRequestWork,
        snapshot: HeterogeneousRuntimeSnapshot,
        pattern: _Pattern,
        plan,
        transitions: tuple,
        transition_estimates: tuple,
        transition_executor_id: str,
        route_capability_sha256: str,
        component_capability_sha256: str,
        placement,
        route_profile,
        adapter_parameters: Mapping[str, object],
        maturity: str,
        control_delay_upper_us: int,
    ) -> _CandidateServiceEstimate:
        component_service_us = max(
            1, 1 if placement is None else placement.latency_us
        )
        component_energy_uj = (
            None if placement is None else placement.total_energy_uj
        )
        learned = self._observation_store.estimate(
            artifact_sha256=manifest.artifact_sha256,
            capability_generation_sha256=route_capability_sha256,
            plan=plan,
            input_tokens=work.input_tokens,
            output_tokens=work.output_tokens,
            quality_requirement=request.quality_requirement,
            cost_features=snapshot.cost_features,
            component_service_us=component_service_us,
            component_energy_uj=component_energy_uj,
            component_capability_generation_sha256=(
                component_capability_sha256
            ),
        )
        learned_decomposes_cold_energy = (
            not transitions
            or learned is not None
            and learned.energy_scope == "warm_execution"
        )
        if (
            learned is not None
            and learned_decomposes_cold_energy
            and learned.maturity in {"QUALIFIED", "QUARANTINED"}
        ):
            maturity = _minimum_maturity([
                learned.maturity,
                *(
                    self.catalog.executor_by_device[device_id].maturity
                    for device_id in pattern.device_ids
                ),
                *(row.maturity for row in transitions),
            ])
        elif (
            adapter_parameters.get("requires_measured_route_profile") == 1
            and (
                route_profile is None
                or route_profile.maturity != "QUALIFIED"
            )
        ):
            maturity = _minimum_maturity((maturity, "SHADOW"))
        if learned is not None and learned.latency_maturity == "QUALIFIED":
            plan = replace(plan, route_profile_id=learned.profile_id)
            self._bounded_cache_store(
                self._executor_id_by_plan_sha256,
                plan.plan_sha256,
                transition_executor_id,
                4_096,
            )
        transition_us = sum(row.latency_us for row in transitions)
        transition_upper_us = sum(
            estimate.latency_upper_us
            if estimate is not None
            and estimate.latency_upper_maturity == "QUALIFIED"
            else row.latency_us
            for row, estimate in zip(transitions, transition_estimates)
        )
        transition_energy_uj = sum(row.energy_uj for row in transitions)
        transition_energy_known = all(
            row.energy_maturity == "QUALIFIED" for row in transitions
        )
        if learned is not None and learned.latency_maturity == "QUALIFIED":
            service_us = learned.service_us + transition_us
            physical_service_upper_us = (
                learned.service_upper_us + transition_upper_us
            )
            latency_evidence = "MEASURED"
        elif route_profile is not None:
            route_service_us = route_profile.service_us(
                work.input_tokens,
                work.output_tokens,
                snapshot.cost_features,
            )
            service_us = route_service_us + transition_us
            physical_service_upper_us = (
                route_service_us
                + route_profile.service_upper_add_us
                + transition_upper_us
            )
            latency_evidence = (
                "MEASURED"
                if route_profile.maturity == "QUALIFIED"
                else "ASSUMED"
            )
        else:
            service_us = (
                1 if placement is None else placement.latency_us + transition_us
            )
            uncertainty_ppm = 50_000 if maturity == "QUALIFIED" else 500_000
            physical_service_upper_us = max(
                service_us,
                _ceil_div(
                    service_us * (1_000_000 + uncertainty_ppm),
                    1_000_000,
                ),
            )
            latency_evidence = (
                "CALIBRATED"
                if placement is not None and placement.measured
                else "ASSUMED"
            )
        return _CandidateServiceEstimate(
            plan=plan,
            learned=learned,
            learned_decomposes_cold_energy=learned_decomposes_cold_energy,
            maturity=maturity,
            component_service_us=component_service_us,
            component_energy_uj=component_energy_uj,
            transition_us=transition_us,
            transition_upper_us=transition_upper_us,
            transition_energy_uj=transition_energy_uj,
            transition_energy_known=transition_energy_known,
            service_us=service_us,
            physical_service_upper_us=physical_service_upper_us,
            service_upper_us=physical_service_upper_us + control_delay_upper_us,
            latency_evidence=latency_evidence,
        )

    def _candidate_schedule_estimate(
        self,
        *,
        request,
        manifest: ModelManifest,
        pattern: _Pattern,
        snapshot: HeterogeneousRuntimeSnapshot,
        observed_at_us: int,
        route_id: str,
        residency_variant: str,
        residency_states: Mapping[str, str],
        transitions: tuple,
        transition_estimates: tuple,
        used_links: tuple[str, ...],
        maturity: str,
        memory_demands: tuple[RuntimeMemoryDemand, ...],
        coordinator: RuntimeCompositeExecutorCapability | None,
        transport_error: str | None,
        runtime_partition_error: str | None,
        placement_error: str | None,
        protected_work,
        system_cost_known: bool,
        residency_holds: Mapping[str, RuntimeResidencyCohortHold],
        resource_slots: Mapping[str, int],
        service: _CandidateServiceEstimate,
    ) -> _CandidateScheduleEstimate:
        reasons = list(self._eligibility(
            manifest,
            pattern,
            snapshot,
            observed_at_us,
            residency_variant,
            residency_states,
            transitions,
            used_links,
            maturity,
            memory_demands,
            coordinator,
        ))
        if transport_error is not None:
            reasons.append("TRANSPORT_PROFILE_INCOMPLETE")
        if runtime_partition_error is not None:
            reasons.append("RUNTIME_PARTITION_CAPACITY")
        if placement_error is not None:
            reasons.extend(_placement_rejection_reasons(placement_error))
        if any(
            estimate is not None
            and (
                estimate.latency_upper_maturity == "QUARANTINED"
                or estimate.energy_maturity == "QUARANTINED"
            )
            for estimate in transition_estimates
        ):
            reasons.append("TRANSITION_PROFILE_QUARANTINED")
        if protected_work is not None and not system_cost_known:
            reasons.append("MARGINAL_SYSTEM_COST_UNKNOWN")
        residency_hold_until_us = max(
            (
                hold.hold_until_us
                for transition in transitions
                for eviction in transition.evictions
                for hold in (residency_holds.get(eviction.replacement_group),)
                if hold is not None
                and hold.artifact_sha256 == eviction.artifact_sha256
                and hold.model_id == eviction.model_id
            ),
            default=observed_at_us,
        )
        residency_hysteresis_us = max(
            (
                hold.replacement_cost_us
                for transition in transitions
                for eviction in transition.evictions
                for hold in (residency_holds.get(eviction.replacement_group),)
                if hold is not None
                and hold.artifact_sha256 == eviction.artifact_sha256
                and hold.model_id == eviction.model_id
            ),
            default=0,
        )
        preview = None
        blocking_reasons = (
            "EXECUTOR_OBSERVATION_ABSENT",
            "EXECUTOR_NOT_READY",
            "EXECUTOR_UNHEALTHY",
            "LINK_OBSERVATION_ABSENT",
            "LINK_NOT_READY",
            "PLACEMENT_INFEASIBLE",
        )
        if not any(reason in reasons for reason in blocking_reasons):
            try:
                preview = self._lease_preview(
                    route_id,
                    resource_slots,
                    max(observed_at_us, residency_hold_until_us),
                    service.service_us,
                    service.service_upper_us,
                    snapshot,
                    transitions=transitions,
                )
            except SchedulerError:
                reasons.append("RESOURCE_CALENDAR_INFEASIBLE")
        start_us = observed_at_us if preview is None else preview.start_us
        finish_us = start_us + service.service_us
        finish_upper_us = start_us + service.service_upper_us
        if finish_upper_us > request.deadline_us:
            reasons.append("SLO_UPPER_BOUND")
        return _CandidateScheduleEstimate(
            reasons=reasons,
            residency_hysteresis_us=residency_hysteresis_us,
            start_us=start_us,
            finish_us=finish_us,
            finish_upper_us=finish_upper_us,
            queue_delay_us=start_us - request.arrival_us,
        )

    def _candidate_energy_estimate(
        self,
        *,
        profile,
        work: ModelRequestWork,
        snapshot: HeterogeneousRuntimeSnapshot,
        transitions: tuple,
        transition_estimates: tuple,
        service: _CandidateServiceEstimate,
        schedule: _CandidateScheduleEstimate,
        placement,
        route_profile,
        use_assumed_phone_energy: bool,
        control_delay_us: int,
        control_delay_upper_us: int,
    ) -> _CandidateEnergyEstimate:
        queue_idle_power_mw = sum(
            profile.domains[domain_id].domain.idle_power_mw
            for domain_id in profile.idle_charge_domains
        )
        queue_idle_energy = sum(
            _ceil_div(
                profile.domains[domain_id].domain.idle_power_mw
                * (schedule.queue_delay_us + control_delay_us),
                1000,
            )
            for domain_id in profile.idle_charge_domains
        )
        queue_idle_energy_upper = queue_idle_energy + sum(
            _ceil_div(
                profile.domains[domain_id].domain.idle_power_mw
                * (control_delay_upper_us - control_delay_us),
                1000,
            )
            for domain_id in profile.idle_charge_domains
        )
        transition_point, transition_lower, transition_upper = (
            self._candidate_transition_energy_bounds(
                profile, transitions, transition_estimates, service
            )
        )
        learned = service.learned
        qualified_learned = (
            learned is not None
            and learned.energy_maturity == "QUALIFIED"
            and learned.energy_uj is not None
            and learned.energy_lower_uj is not None
            and learned.energy_upper_uj is not None
            and service.learned_decomposes_cold_energy
            and service.transition_energy_known
            and not use_assumed_phone_energy
        )
        qualified_profile = (
            route_profile is not None
            and route_profile.maturity == "QUALIFIED"
            and service.transition_energy_known
            and not use_assumed_phone_energy
        )
        calibrated_placement = (
            route_profile is None
            and placement is not None
            and placement.measured
            and service.transition_energy_known
            and not use_assumed_phone_energy
        )
        if placement is None:
            return _CandidateEnergyEstimate(
                None, None, None, None, None, None, None, None, None, "ABSENT"
            )
        if qualified_learned:
            point = learned.energy_uj
            lower = learned.energy_lower_uj
            upper = learned.energy_upper_uj
            evidence = "MEASURED"
        elif qualified_profile:
            point = route_profile.energy_uj(
                work.input_tokens, work.output_tokens, snapshot.cost_features
            )
            lower = (
                point * (1_000_000 - route_profile.energy_lower_error_ppm)
                // 1_000_000
            )
            upper = _ceil_div(
                point * (1_000_000 + route_profile.energy_upper_error_ppm),
                1_000_000,
            )
            evidence = "MEASURED"
        else:
            point = placement.total_energy_uj
            error_ppm = 100_000 if calibrated_placement else 500_000
            lower = point * (1_000_000 - error_ppm) // 1_000_000
            upper = _ceil_div(point * (1_000_000 + error_ppm), 1_000_000)
            evidence = "CALIBRATED" if calibrated_placement else "ASSUMED"
        warm = point + queue_idle_energy
        warm_lower = lower + queue_idle_energy
        warm_upper = upper + queue_idle_energy_upper
        return _CandidateEnergyEstimate(
            warm + transition_point,
            warm_lower + transition_lower,
            warm_upper + transition_upper,
            warm,
            warm_lower,
            warm_upper,
            transition_point,
            transition_lower,
            transition_upper,
            evidence,
            queue_idle_energy_uj=queue_idle_energy,
            queue_idle_power_mw=queue_idle_power_mw,
        )

    @staticmethod
    def _validate_candidate_energy(
        energy: _CandidateEnergyEstimate,
        service: _CandidateServiceEstimate,
        transitions: tuple,
        reasons: list[str],
    ) -> _CandidateEnergyEstimate:
        invalid_warm = (
            energy.warm_energy_uj is not None
            and (
                energy.warm_energy_uj <= 0
                or energy.warm_lower_uj is None
                or energy.warm_lower_uj <= 0
                or energy.warm_upper_uj is None
                or energy.warm_upper_uj <= 0
            )
        )
        learned = service.learned
        invalid_route_total = (
            learned is not None
            and transitions
            and learned.energy_scope == "route_total"
            and learned.energy_maturity == "QUALIFIED"
            and learned.energy_upper_uj is not None
            and energy.transition_upper_uj is not None
            and learned.energy_upper_uj <= energy.transition_upper_uj
        )
        if invalid_warm or invalid_route_total:
            reasons.append("COLD_WARM_ENERGY_DECOMPOSITION_INVALID")
        if invalid_warm:
            energy = _CandidateEnergyEstimate(
                None, None, None, None, None, None, None, None, None, "ABSENT"
            )
        if energy.energy_uj is None and not reasons:
            reasons.append("ENERGY_UNKNOWN")
        return energy

    @staticmethod
    def _candidate_marginal_system_cost(
        *,
        system_cost_known: bool,
        protected_work,
        system_profile,
        interference_resources: tuple[str, ...],
        service: _CandidateServiceEstimate,
        schedule: _CandidateScheduleEstimate,
        energy: _CandidateEnergyEstimate,
    ) -> tuple[Mapping[str, object] | None, int | None, _CandidateEnergyEstimate]:
        if not system_cost_known:
            return None, None, energy
        assert protected_work is not None
        assert system_profile is not None
        try:
            marginal_system_cost = system_profile.route_cost(
                interference_resources,
                protected_work,
                service_us=service.service_us,
                service_upper_us=service.physical_service_upper_us,
                finish_us=schedule.finish_us,
                finish_upper_us=schedule.finish_upper_us,
            )
        except RuntimeSystemCostError as exc:
            raise RouteGenerationError(str(exc)) from exc
        # Idle-domain energy charged while the protected work still runs is
        # spent whether this route starts now or waits; it cancels in the
        # start-now versus wait comparison.
        arrival_us = schedule.start_us - schedule.queue_delay_us
        overlap_us = max(0, min(
            schedule.start_us, protected_work.critical_path_end_us,
        ) - arrival_us)
        overlap_idle_uj = _ceil_div(energy.queue_idle_power_mw * overlap_us, 1000)
        marginal_system_cost.update({
            "route_energy_lower_uj": energy.lower_uj,
            "route_energy_uj": energy.energy_uj,
            "route_energy_upper_uj": energy.upper_uj,
            "protected_overlap_us": overlap_us,
            "protected_overlap_idle_uj": overlap_idle_uj,
            "route_energy_incremental_lower_uj": (
                None if energy.lower_uj is None
                else max(0, energy.lower_uj - overlap_idle_uj)
            ),
            "route_energy_incremental_upper_uj": (
                None if energy.upper_uj is None
                else max(0, energy.upper_uj - overlap_idle_uj)
            ),
        })
        system_finish_upper_us = int(
            marginal_system_cost["system_finish_upper_us"]
        )
        if (
            energy.energy_uj is None
            or energy.lower_uj is None
            or energy.upper_uj is None
        ):
            return marginal_system_cost, system_finish_upper_us, energy
        assert energy.warm_energy_uj is not None
        assert energy.warm_lower_uj is not None
        assert energy.warm_upper_uj is not None
        assert energy.transition_energy_uj is not None
        assert energy.transition_lower_uj is not None
        assert energy.transition_upper_uj is not None
        warm = energy.warm_energy_uj + int(marginal_system_cost["total_uj"])
        warm_lower = (
            energy.warm_lower_uj + int(marginal_system_cost["lower_uj"])
        )
        warm_upper = (
            energy.warm_upper_uj + int(marginal_system_cost["upper_uj"])
        )
        return (
            marginal_system_cost,
            system_finish_upper_us,
            replace(
                energy,
                energy_uj=warm + energy.transition_energy_uj,
                lower_uj=warm_lower + energy.transition_lower_uj,
                upper_uj=warm_upper + energy.transition_upper_uj,
                warm_energy_uj=warm,
                warm_lower_uj=warm_lower,
                warm_upper_uj=warm_upper,
            ),
        )

    @staticmethod
    def _candidate_transfer_costs(placement) -> tuple[int, tuple]:
        transfer_rows = []
        if placement is not None:
            for operator in placement.operator_decisions:
                transfer_rows.extend(operator.internal_transfers)
                if operator.transition is not None:
                    transfer_rows.append(operator.transition)
            if placement.final_transfer is not None:
                transfer_rows.append(placement.final_transfer)
        transfer_costs = tuple(
            RuntimeTransferCost(
                step_id=row.step_id,
                source_device=row.source_device,
                target_device=row.target_device,
                payload_bytes=row.payload_bytes,
                invocations=row.invocations,
                total_bytes=row.bytes,
                queue_depth=row.queue_depth,
                concurrent_streams=row.concurrent_streams,
                message_waves=row.message_waves,
                fixed_latency_us=row.fixed_latency_us,
                latency_us=row.latency_us,
                dynamic_energy_uj=row.dynamic_energy_uj,
                link_ids=row.link_ids,
            )
            for row in transfer_rows
        )
        return sum(row.latency_us for row in transfer_rows), transfer_costs

    def _candidate_component_times(
        self,
        *,
        manifest: ModelManifest,
        work: ModelRequestWork,
        pattern: _Pattern,
        profile,
        placement,
        memory_demands: tuple[RuntimeMemoryDemand, ...],
    ) -> tuple[int, int, int, dict[str, int]]:
        component_key = (
            manifest.artifact_sha256,
            work.input_tokens,
            work.output_tokens,
            pattern.route_key,
        )
        component_times = self._component_time_cache.get(component_key)
        if component_times is None:
            compute_us = 0
            memory_us = 0
            for operator_work in work.operators:
                primary, helper, fraction = pattern.assignments[
                    operator_work.operator_id
                ]
                devices = (primary,) if helper is None else (primary, helper)
                fractions = (
                    (1_000_000,)
                    if helper is None
                    else (1_000_000 - fraction, fraction)
                )
                for device_id, work_fraction in zip(devices, fractions):
                    capability = self.catalog.executor_by_device[device_id]
                    branch_ops = _ceil_div(
                        operator_work.compute_ops * work_fraction, 1_000_000
                    )
                    branch_memory = _ceil_div(
                        operator_work.memory_bytes * work_fraction, 1_000_000
                    )
                    kernel_profile_id, _ = capability.kernel_profile_for(
                        operator_work.kind,
                        input_tokens=work.input_tokens,
                        output_tokens=work.output_tokens,
                        compute_ops=branch_ops,
                        memory_bytes=branch_memory,
                    )
                    kernel = profile.kernels[kernel_profile_id].kernel
                    compute_us += branch_ops * 1_000_000 // kernel.effective_ops_per_s
                    memory_us += (
                        branch_memory * 1_000_000 // kernel.effective_bytes_per_s
                    )
            component_times = compute_us, memory_us
            if len(self._component_time_cache) >= 4_096:
                self._component_time_cache.pop(next(iter(self._component_time_cache)))
            self._component_time_cache[component_key] = component_times
        compute_us, memory_us = component_times
        join_wait_us = 0
        if pattern.route_family == "operator_split" and placement is not None:
            join_wait_us = max(0, compute_us - placement.latency_us)
        memory_by_resource: dict[str, int] = {}
        for demand in memory_demands:
            memory_by_resource[demand.resource_id] = (
                memory_by_resource.get(demand.resource_id, 0)
                + demand.required_bytes
            )
        return compute_us, memory_us, join_wait_us, memory_by_resource

    @staticmethod
    def _candidate_cost_record(
        *,
        request,
        service: _CandidateServiceEstimate,
        schedule: _CandidateScheduleEstimate,
        energy: _CandidateEnergyEstimate,
        compute_us: int,
        memory_us: int,
        transfer_us: int,
        join_wait_us: int,
        memory_by_resource: Mapping[str, int],
        transfer_costs: tuple,
        marginal_system_cost,
    ) -> AutomatedRouteCost:
        return AutomatedRouteCost(
            start_us=schedule.start_us,
            finish_us=schedule.finish_us,
            finish_upper_us=schedule.finish_upper_us,
            service_us=service.service_us,
            service_upper_us=service.service_upper_us,
            queue_delay_us=schedule.queue_delay_us,
            compute_us=compute_us,
            memory_us=memory_us,
            transfer_us=transfer_us,
            join_wait_us=join_wait_us,
            exposed_tail_us=max(0, schedule.finish_upper_us - request.deadline_us),
            load_us=service.transition_us,
            eviction_us=0,
            restore_us=0,
            switching_us=service.transition_us,
            interference_us=(
                0
                if marginal_system_cost is None
                else int(marginal_system_cost["interference_us"])
            ),
            fleet_energy_uj=energy.energy_uj,
            fleet_energy_lower_uj=energy.lower_uj,
            fleet_energy_upper_uj=energy.upper_uj,
            component_service_us=service.component_service_us,
            component_energy_uj=service.component_energy_uj,
            memory_by_resource_bytes=memory_by_resource,
            warm_execution_energy_uj=energy.warm_energy_uj,
            warm_execution_energy_lower_uj=energy.warm_lower_uj,
            warm_execution_energy_upper_uj=energy.warm_upper_uj,
            transition_energy_uj=energy.transition_energy_uj,
            transition_energy_lower_uj=energy.transition_lower_uj,
            transition_energy_upper_uj=energy.transition_upper_uj,
            residency_hysteresis_us=schedule.residency_hysteresis_us,
            transfer_costs=transfer_costs,
            latency_evidence=service.latency_evidence,
            energy_evidence=energy.evidence,
        )
