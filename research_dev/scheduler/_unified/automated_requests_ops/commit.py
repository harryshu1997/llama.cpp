"""AutomatedRequestMixin commit operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
import time
from types import MappingProxyType
from typing import Mapping, Sequence

from ..._internal.policy import Decision, LeasePreview, Request, SchedulerError
from ..._internal.adaptive_decode_planning import (
    adaptive_candidate_set_for_parent,
    adaptive_desktop_control_is_qualified,
)
from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.runtime_cost import RuntimeCostEstimateSet, RuntimeExecutorRegistry
from ..._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot
from ..._internal.route_generation import RuntimeRouteTemplateSet
from ..._internal.model_placement_controller import (
    ModelPlacementControllerError,
    RequestHelperEnvelopeBinding,
)
from ..._internal.runtime_plan import (
    AutomatedCandidateSet,
    AutomatedRouteCandidate,
    RuntimeTransitionReceipt,
)
from ..._internal.runtime_residency_cohorts import (
    RuntimeModelPlacementEpoch,
    runtime_residency_component_identity,
)
from ..._internal.runtime_controller import RuntimeRequestTicket
from ..._internal.runtime_admission import RuntimeRequestObservation


def _commit_decode_cohort_leases(
    controller,
    request: Request,
    selected: AutomatedRouteCandidate,
    preview: LeasePreview,
    observed_at_us: int,
) -> tuple[tuple, object | None, LeasePreview]:
    cohort_admission = controller._runtime_decode_cohorts.admit(
        request.request_id,
        selected.plan,
        selected.binding,
        quality_requirement=request.quality_requirement,
        observed_at_us=observed_at_us,
        service_upper_us=selected.cost.service_upper_us,
        output_tokens=request.output_tokens,
    )
    decode_cohort = None
    if cohort_admission is None:
        leases = controller.timeline.commit_leases(preview, request.request_id)
    elif cohort_admission.leader:
        leases = controller.timeline.commit_leases(
            preview, cohort_admission.binding.cohort_id
        )
        decode_cohort = controller._runtime_decode_cohorts.bind_leases(
            cohort_admission.binding.cohort_id, leases
        )
    else:
        leases = cohort_admission.shared_leases
        required_until_us = cohort_admission.required_reserved_until_us
        if required_until_us > max(row.reserved_until_us for row in leases):
            execution_tokens = tuple(
                row.token for row in leases
                if not row.lease_id.startswith(selected.plan.route_id + ":prepare:")
            )
            try:
                controller.timeline.extend_many(
                    execution_tokens, required_until_us
                )
            except SchedulerError:
                controller._runtime_decode_cohorts.withdraw(request.request_id)
                cohort_admission = None
                leases = controller.timeline.commit_leases(
                    preview, request.request_id
                )
            else:
                leases = tuple(
                    replace(row, reserved_until_us=required_until_us)
                    if row.token in execution_tokens else row
                    for row in leases
                )
                decode_cohort = (
                    controller._runtime_decode_cohorts.update_shared_leases(
                        cohort_admission.binding.cohort_id, leases
                    )
                )
                controller._runtime_controller.extend_decode_cohort(
                    decode_cohort,
                    leases,
                    pending_member_request_id=request.request_id,
                )
        else:
            decode_cohort = cohort_admission.binding
    if cohort_admission is not None and not cohort_admission.leader:
        shared_start_us = max(
            observed_at_us, min(row.start_us for row in leases)
        )
        preview = LeasePreview(
            start_us=shared_start_us,
            finish_us=shared_start_us + selected.cost.service_us,
            finish_upper_us=(
                shared_start_us + selected.cost.service_upper_us
            ),
            plans=(),
            queue_by_resource_us=MappingProxyType({
                resource_id: max(0, shared_start_us - observed_at_us)
                for resource_id in sorted(selected.plan.resource_ids)
            }),
            blocking_resources=(),
        )
    return leases, decode_cohort, preview


def _reset_previous_phone_layout_transition(
    controller, previous_ticket_id: str | None, observed_at_us: int
) -> None:
    if previous_ticket_id is None:
        return
    preparing_layout = (
        controller._model_placement_controller.preparing_phone_layout()
    )
    try:
        if (
            preparing_layout is not None
            and preparing_layout.transition_ticket_id == previous_ticket_id
            and preparing_layout.projection_token_sha256 is not None
        ):
            controller._model_placement_controller.reset_phone_layout_transition(
                previous_ticket_id,
                generation=preparing_layout.generation,
                projection_token_sha256=(
                    preparing_layout.projection_token_sha256
                ),
                observed_at_us=observed_at_us,
                reason="REQUEST_REPLAN",
            )
    except ModelPlacementControllerError as exc:
        raise UnifiedScheduleError(str(exc)) from exc


def _selected_phone_layout_generation(
    controller,
    selected: AutomatedRouteCandidate,
    selection_mode: str,
    placement_epoch: RuntimeModelPlacementEpoch | None,
    bound_placement_epoch: RuntimeModelPlacementEpoch | None,
) -> int | None:
    contract = selected.plan.execution_contract
    if contract.remote_resident_ffn is not None:
        placement = controller._model_placement_controller
        ready = placement.ready_phone_layout()
        for owner in contract.remote_resident_ffn.sessions:
            state = placement.phone_session_state(owner.session_id)
            shard = None if ready is None else next((
                row for row in ready.layout.shards if row.session_id == owner.session_id
            ), None)
            if (
                ready is None or state is None or state.state != "READY"
                or shard is None or shard.layer_mask != owner.layer_mask
                or shard.maximum_columns != selected.plan.adapter_parameters.get("ffn_resident_columns")
                or state.resident_artifact_sha256
                    != contract.remote_resident_ffn.parent_artifact_sha256
                or state.endpoint != owner.endpoint
                or state.shard_geometry_sha256 != owner.resident_geometry_sha256
                or state.operator_plan_sha256 != owner.operator_plan_sha256
                or state.session_generation != owner.session_generation
                or owner.session_generation < 1
                or state.resident_bytes != owner.resident_bytes
            ):
                raise UnifiedScheduleError(
                    "selected remote-resident owner differs from READY authority: "
                    + owner.session_id
                )
        return ready.generation
    if not selected.plan.execution_contract.phone_shards:
        return None
    phone_layout = controller._model_placement_controller.planning_phone_layout()
    selected_geometry = selected.plan.adapter_parameters.get(
        "phone_shard_set_geometry_sha256"
    )
    effective_epoch = (
        placement_epoch
        if bound_placement_epoch is None else bound_placement_epoch
    )
    epoch_layout_generation = (
        None
        if effective_epoch is None
        else effective_epoch.phone_layout_generation
    )
    invalid = (
        phone_layout is None
        or selected_geometry != phone_layout.layout.geometry_sha256
        or (
            selection_mode != "calibration"
            and epoch_layout_generation != phone_layout.generation
        )
        or (
            selection_mode == "calibration"
            and epoch_layout_generation is not None
            and epoch_layout_generation != phone_layout.generation
        )
    )
    if invalid:
        raise UnifiedScheduleError(
            "selected phone route lacks its authoritative layout: "
            + "selected=" + str(selected_geometry)
            + " authoritative=" + (
                "absent"
                if phone_layout is None
                else phone_layout.layout.geometry_sha256
            )
            + " epoch=" + (
                "absent"
                if effective_epoch is None
                else str(epoch_layout_generation)
            )
        )
    return phone_layout.generation


def _automated_decision(
    controller,
    *,
    request: Request,
    selected: AutomatedRouteCandidate,
    preview: LeasePreview,
    leases: tuple,
    reason: str,
    rejected: tuple[tuple[str, str], ...],
    observed_at_us: int,
    prediction_finish_upper_us: int,
) -> Decision:
    cost = selected.cost
    return Decision(
        request_id=request.request_id,
        workload_id=request.workload_id,
        mode=controller.mode,
        route_id=selected.candidate_id,
        granularity={
            "whole_model": "task",
            "layer_placement": "layer",
            "operator_offload": "operator",
            "operator_split": "operator",
        }[selected.route_family],
        start_us=preview.start_us,
        finish_us=preview.finish_us,
        finish_upper_us=prediction_finish_upper_us,
        service_us=cost.service_us,
        queue_us=preview.start_us - observed_at_us,
        queue_by_resource_us=preview.queue_by_resource_us,
        blocking_resources=preview.blocking_resources,
        leases=leases,
        runtime_gate=None,
        energy_uj=cost.fleet_energy_uj,
        energy_upper_uj=cost.fleet_energy_upper_uj,
        energy_breakdown={
            "kind": "gguf_operator_dag_v1",
            "marginal_system_cost": (
                None
                if selected.marginal_system_cost is None
                else dict(selected.marginal_system_cost)
            ),
            "route": cost.to_json(),
        },
        server_busy_us=cost.service_us,
        reason=reason,
        rejected=rejected,
        system_finish_upper_us=(
            prediction_finish_upper_us
            if selected.system_finish_upper_us is None
            else selected.system_finish_upper_us
        ),
        marginal_system_cost=selected.marginal_system_cost,
    )


def _automated_executor_bindings(
    candidate_set: AutomatedCandidateSet,
    selected: AutomatedRouteCandidate,
    estimates: RuntimeCostEstimateSet,
    selection_mode: str,
) -> tuple[RuntimeCostEstimateSet, tuple]:
    bindings = tuple(
        selected.binding
        if row.candidate_id == selected.candidate_id else row.binding
        for row in candidate_set.candidates
    )
    original_binding = next(
        row.binding for row in candidate_set.candidates
        if row.candidate_id == selected.candidate_id
    )
    if selected.binding == original_binding:
        return estimates, bindings
    if selection_mode not in {"adaptive-decode", "calibration"}:
        raise UnifiedScheduleError(
            "selected binding differs from generated candidate"
        )
    reason = (
        "ADAPTIVE_ENVELOPE_ADMITTED"
        if selection_mode == "adaptive-decode"
        else "CALIBRATION_ADMITTED"
    )
    detail_key = (
        "adaptive_original_rejection_reasons"
        if selection_mode == "adaptive-decode"
        else "calibration_original_rejection_reasons"
    )
    estimates = replace(
        estimates,
        estimates=tuple(
            replace(
                row,
                admitted=True,
                reason=reason,
                details={
                    **dict(row.details),
                    detail_key: list(row.details["rejection_reasons"]),
                },
            )
            if row.route_id == selected.candidate_id else row
            for row in estimates.estimates
        ),
    )
    return estimates, bindings


def _admit_automated_ticket(
    controller,
    *,
    request: Request,
    snapshot: HeterogeneousRuntimeSnapshot,
    estimates: RuntimeCostEstimateSet,
    decision: Decision,
    selected: AutomatedRouteCandidate,
    bindings: tuple,
    observed_at_us: int,
    memory: tuple,
    previous_ticket_id: str | None,
    failure_reason: str | None,
    previous_transition_receipts: Sequence[RuntimeTransitionReceipt],
    selection_mode: str,
    decode_cohort,
    residency_projection_token: str | None,
    phone_layout_generation: int | None,
) -> RuntimeRequestTicket:
    return controller._runtime_controller.admit(
        request=request,
        model=estimates.model,
        runtime_observation=RuntimeRequestObservation(
            observed_at_us,
            snapshot.cost_features,
            {
                resource_id: capacity.occupied_bytes
                for resource_id, capacity in snapshot.memory.capacities.items()
            },
        ),
        estimates=estimates,
        decision=decision,
        binding=selected.binding,
        executor_bindings=bindings,
        online_receipt=None,
        admitted_at_us=observed_at_us,
        memory_reservations=memory,
        execution_plan=selected.plan,
        planning_profile_sha256=estimates.planning_profile_sha256,
        previous_ticket_id=previous_ticket_id,
        failure_reason=failure_reason,
        previous_transition_receipts=previous_transition_receipts,
        selection_mode=selection_mode,
        decode_cohort=decode_cohort,
        residency_order_barrier=(
            controller._runtime_residency_order_barrier(selected.plan)
        ),
        residency_projection_token=residency_projection_token,
        phone_layout_generation=phone_layout_generation,
        residency_hysteresis_key=controller._runtime_residency_hysteresis_key(
            selected.plan, estimates.model.artifact_sha256
        ),
    )


def _dispatched_helper_envelope_binding(
    controller, selected: AutomatedRouteCandidate
) -> RequestHelperEnvelopeBinding | None:
    if selected.plan.helper_envelope is None:
        return None
    helper = selected.plan.helper_envelope
    helper_contract = helper.helper_plan.execution_contract
    try:
        helper_layout = controller._model_placement_controller.phone_layout(
            helper.phone_layout_generation
        )
    except ModelPlacementControllerError:
        helper_layout = None
    session_ids = tuple(
        row.session_id for row in helper_contract.phone_shards
    )
    if (
        helper_layout is None
        or helper_layout.layout.geometry_sha256
        != helper.phone_layout_geometry_sha256
        or not controller._model_placement_controller.phone_layout_sessions_are_usable(
            helper.phone_layout_generation,
            helper.phone_layout_geometry_sha256,
            session_ids,
        )
    ):
        return None
    return RequestHelperEnvelopeBinding(
        route_id=helper.route_id,
        operator_plan_sha256=helper.operator_plan_sha256,
        desktop_parent_route_id=helper.desktop_parent_route_id,
        desktop_placement_sha256=helper.desktop_placement_sha256,
        phone_layout_generation=helper.phone_layout_generation,
        phone_layout_geometry_sha256=helper.phone_layout_geometry_sha256,
        activation_dtype=helper.activation_dtype,
        assisted_layer_mask=helper.resident_layer_mask,
        maximum_columns=helper.resident_columns,
        allowed_fractions_ppm=(
            helper_contract.allowed_adaptive_fractions_ppm
        ),
        phone_session_ids=session_ids,
        resource_ids=helper.helper_plan.resource_ids,
    )


def _refresh_request_helper_opportunities(
    controller,
    request: Request,
    candidate_set: AutomatedCandidateSet,
    selected: AutomatedRouteCandidate,
    estimates: RuntimeCostEstimateSet,
    helper_envelope_binding: RequestHelperEnvelopeBinding | None,
) -> None:
    if not (
        selected.plan.helper_envelope is not None
        or controller._has_dormant_phone_ffn_runtime(selected.plan)
    ):
        controller._request_helper_opportunities.pop(request.request_id, None)
        controller._late_request_helper_contexts.pop(request.request_id, None)
        return
    helper_candidates = adaptive_candidate_set_for_parent(candidate_set, selected)
    if helper_envelope_binding is not None and adaptive_desktop_control_is_qualified(
        helper_candidates, controller._runtime_capabilities
    ):
        controller._request_helper_opportunities.pop(request.request_id, None)
        controller._late_request_helper_contexts.pop(request.request_id, None)
        return
    generated = controller._compact_helper_opportunities(
        helper_candidates, request
    )
    current_layouts = tuple(
        row
        for row in (
            controller._model_placement_controller.target_phone_layout(),
            controller._model_placement_controller.ready_phone_layout(),
        )
        if row is not None
    )
    layout_by_generation = {row.generation: row for row in current_layouts}
    retained = []
    for opportunity in controller._request_helper_opportunities.get(
        request.request_id, ()
    ):
        layout = layout_by_generation.get(
            opportunity.phone_layout_generation
        )
        if not controller._helper_opportunity_matches_layout(
            opportunity, layout, estimates, selected
        ):
            continue
        retained.append(replace(
            opportunity,
            desktop_parent_route_id=selected.candidate_id,
        ))
    identities = set()
    opportunities = []
    for opportunity in (*retained, *generated):
        identity = (
            opportunity.phone_layout_generation,
            opportunity.phone_layout_geometry_sha256,
            opportunity.operator_plan_sha256,
        )
        if identity in identities:
            continue
        identities.add(identity)
        opportunities.append(opportunity)
    if opportunities:
        controller._request_helper_opportunities[request.request_id] = tuple(
            opportunities
        )
    else:
        controller._request_helper_opportunities.pop(request.request_id, None)
    controller._late_request_helper_contexts.pop(request.request_id, None)


def _helper_opportunity_matches_layout(
    opportunity,
    layout,
    estimates: RuntimeCostEstimateSet,
    selected: AutomatedRouteCandidate,
) -> bool:
    if (
        layout is None
        or layout.layout.geometry_sha256
        != opportunity.phone_layout_geometry_sha256
        or not layout.covers_artifact(estimates.model.artifact_sha256)
        or opportunity.desktop_parent_placement_sha256
        != selected.plan.desktop_placement_sha256
        or opportunity.helper_operator_plan.baseline_executor_id
        != selected.binding.executor_id
        or opportunity.helper_binding.artifact_sha256
        != estimates.model.artifact_sha256
    ):
        return False
    expected_shards = [
        {
            "artifact_sha256": row.artifact_sha256,
            "endpoint": row.endpoint,
            "layer_mask": row.layer_mask,
            "maximum_columns": row.maximum_columns,
            "operator_plan_sha256": row.operator_plan_sha256,
            "resident_bytes": row.resident_bytes,
            "resident_geometry_sha256": row.resident_geometry_sha256,
            "session_id": row.session_id,
        }
        for row in layout.layout.shards
        if row.artifact_sha256 == estimates.model.artifact_sha256
    ]
    actual_shards = [
        row.to_json()
        for row in opportunity.helper_operator_plan.execution_contract.phone_shards
    ]
    return actual_shards == expected_shards


def _record_automated_residency_arrival(
    controller,
    *,
    event_kind: str,
    request: Request,
    candidate_set: AutomatedCandidateSet,
    selected_component,
    estimates: RuntimeCostEstimateSet,
) -> None:
    if event_kind != "DECISION":
        return
    planning_components = tuple({
        component.identity_sha256: component
        for row in candidate_set.candidates
        if any(
            demand.lifetime == "resident"
            for demand in row.plan.memory_demands
        )
        for component in (
            runtime_residency_component_identity(
                estimates.model.artifact_sha256,
                row.plan,
                row.binding,
            ),
        )
    }.values())
    if planning_components:
        controller._runtime_residency_cohorts.record_planning_arrival(
            request.request_id,
            planning_components,
            request.arrival_us,
        )
    controller._runtime_residency_cohorts.record_arrival(
        request.request_id,
        selected_component,
        request.arrival_us,
        estimates.model.artifact_sha256,
    )


def _publish_automated_placement_epoch(
    controller,
    candidate_set: AutomatedCandidateSet,
    placement_epoch: RuntimeModelPlacementEpoch | None,
    route_templates: RuntimeRouteTemplateSet | None,
) -> None:
    if (placement_epoch is None) != (route_templates is None):
        raise UnifiedScheduleError(
            "model placement publication is incomplete"
        )
    placement_resolution = candidate_set.search_metadata.get(
        "model_placement_resolution"
    )
    invalidated_epoch_sha256 = (
        None
        if not isinstance(placement_resolution, Mapping)
        else placement_resolution.get("invalidated_epoch_sha256")
    )
    if invalidated_epoch_sha256 is not None:
        controller._runtime_residency_cohorts.invalidate_model_placement_epoch_sha256(
            invalidated_epoch_sha256
        )
    if placement_epoch is not None:
        controller._publish_model_placement_epoch(
            placement_epoch, route_templates
        )


def _finish_automated_commit_timing(
    controller,
    *,
    ticket: RuntimeRequestTicket,
    event_kind: str,
    observed_at_us: int,
    commit_started_ns: int,
    reservation_finished_ns: int,
) -> None:
    admission_finished_ns = time.perf_counter_ns()
    controller._append_runtime_log(
        event_kind, ticket, observed_at_us, ticket.dispatch_state
    )
    finished_ns = time.perf_counter_ns()
    controller._last_runtime_commit_timing = MappingProxyType({
        "admission_us": (
            admission_finished_ns - reservation_finished_ns
        ) // 1000,
        "journal_commit_us": (
            finished_ns - admission_finished_ns
        ) // 1000,
        "reservation_us": (
            reservation_finished_ns - commit_started_ns
        ) // 1000,
        "total_us": (finished_ns - commit_started_ns) // 1000,
    })


def _commit_automated_attempt(
    controller,
    *,
    request: Request,
    snapshot: HeterogeneousRuntimeSnapshot,
    candidate_set: AutomatedCandidateSet,
    estimates: RuntimeCostEstimateSet,
    selected: AutomatedRouteCandidate,
    preview: object,
    rejected: tuple[tuple[str, str], ...],
    reason: str,
    observed_at_us: int,
    event_kind: str,
    previous_ticket_id: str | None = None,
    failure_reason: str | None = None,
    previous_transition_receipts: Sequence[
        RuntimeTransitionReceipt
    ] = (),
    selection_mode: str = "energy-aware",
    placement_epoch: RuntimeModelPlacementEpoch | None = None,
    route_templates: RuntimeRouteTemplateSet | None = None,
    bound_placement_epoch: RuntimeModelPlacementEpoch | None = None,
) -> RuntimeRequestTicket:
    commit_started_ns = time.perf_counter_ns()
    if not isinstance(preview, LeasePreview):
        raise UnifiedScheduleError(
            "automated lease preview is invalid"
        )
    leases, decode_cohort, preview = controller._commit_decode_cohort_leases(
        request, selected, preview, observed_at_us
    )
    prediction_finish_upper_us = (
        controller._automated_prediction_finish_upper_us(selected, preview)
    )
    controller._reset_previous_phone_layout_transition(
        previous_ticket_id, observed_at_us
    )
    residency_projection_token = (
        controller._phone_projection_token_for_plan(selected.plan)
    )
    phone_layout_generation = controller._selected_phone_layout_generation(
        selected,
        selection_mode,
        placement_epoch,
        bound_placement_epoch,
    )
    memory = controller._runtime_memory.reserve(
        request.request_id,
        selected.plan.memory_demands,
        snapshot.memory,
        start_us=preview.start_us,
        reserved_until_us=preview.finish_upper_us,
        transitions=selected.plan.transitions,
        residency=snapshot.residency,
        exclusive_resource_by_device=(
            controller._runtime_exclusive_memory_resources()
        ),
    )
    reservation_finished_ns = time.perf_counter_ns()
    decision = controller._automated_decision(
        request=request,
        selected=selected,
        preview=preview,
        leases=leases,
        reason=reason,
        rejected=rejected,
        observed_at_us=observed_at_us,
        prediction_finish_upper_us=prediction_finish_upper_us,
    )
    estimates, bindings = controller._automated_executor_bindings(
        candidate_set, selected, estimates, selection_mode
    )
    controller._runtime_executor_registry = RuntimeExecutorRegistry.from_bindings(
        bindings
    )
    ticket = controller._admit_automated_ticket(
        request=request,
        snapshot=snapshot,
        estimates=estimates,
        decision=decision,
        selected=selected,
        bindings=bindings,
        observed_at_us=observed_at_us,
        memory=memory,
        previous_ticket_id=previous_ticket_id,
        failure_reason=failure_reason,
        previous_transition_receipts=previous_transition_receipts,
        selection_mode=selection_mode,
        decode_cohort=decode_cohort,
        residency_projection_token=residency_projection_token,
        phone_layout_generation=phone_layout_generation,
    )
    controller._bind_phone_residency_transition(
        ticket, observed_at_us, snapshot
    )
    selected_component = runtime_residency_component_identity(
        estimates.model.artifact_sha256,
        selected.plan,
        selected.binding,
    )
    if previous_ticket_id is not None:
        controller._model_placement_controller.release_request(
            request.request_id
        )
    helper_envelope_binding = controller._dispatched_helper_envelope_binding(
        selected
    )
    controller._model_placement_controller.bind_dispatched_request(
        request.request_id,
        estimates.model.artifact_sha256,
        selected.candidate_id,
        selected_component.identity_sha256,
        selected.plan.execution_contract.initial_split_fraction_ppm,
        desktop_placement_sha256=selected.plan.desktop_placement_sha256,
        kv_cache_owner_id=ticket.ticket_id,
        sequence_identity=ticket.ticket_id,
        helper_envelope=helper_envelope_binding,
        output_tokens=request.output_tokens,
    )
    controller._refresh_request_helper_opportunities(
        request,
        candidate_set,
        selected,
        estimates,
        helper_envelope_binding,
    )
    controller._record_automated_residency_arrival(
        event_kind=event_kind,
        request=request,
        candidate_set=candidate_set,
        selected_component=selected_component,
        estimates=estimates,
    )
    controller._publish_automated_placement_epoch(
        candidate_set, placement_epoch, route_templates
    )
    controller._finish_automated_commit_timing(
        ticket=ticket,
        event_kind=event_kind,
        observed_at_us=observed_at_us,
        commit_started_ns=commit_started_ns,
        reservation_finished_ns=reservation_finished_ns,
    )
    return ticket


def _detach_decode_cohort_for_replan(
    controller, request_id: str
) -> None:
    release = controller._runtime_decode_cohorts.prepare_replan(request_id)
    if release is None:
        return
    if release.transferred_leases:
        controller.timeline.reassign_owner(
            release.previous_binding.shared_lease_tokens,
            expected_owner_id=release.previous_lease_owner_id,
            owner_id=request_id,
        )
    controller._runtime_controller.detach_decode_cohort_for_replan(
        request_id,
        release.previous_binding,
        release.transferred_leases,
    )
    if release.remaining_binding is not None:
        for member_id in (
            release.remaining_binding.member_request_ids
        ):
            controller._runtime_controller.bind_decode_cohort(
                member_id, release.remaining_binding
            )
