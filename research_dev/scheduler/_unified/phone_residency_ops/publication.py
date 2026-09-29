"""PhoneResidencyMixin publication operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import Mapping

from ..._internal.policy import Request
from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.model_manifest import ModelManifest
from ..._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot
from ..._internal.model_placement_controller import ModelPlacementControllerError
from ..._internal.runtime_plan import (
    AutomatedCandidateSet,
    AutomatedRouteCandidate,
    RuntimeExecutionPlan,
)
from ..._internal.runtime_resources import RuntimeResidencyProjectionToken, RuntimeResourceError
from ..._internal.runtime_residency_projection import (
    RuntimeResidencyProjectionError,
    projection_token_matches_plan,
    runtime_residency_projection_token,
)
from ..._internal.runtime_controller import RuntimeRequestTicket
from ..common import _ceil_div


def _current_phone_projection_token(
    controller,
) -> RuntimeResidencyProjectionToken | None:
    state = (
        controller._model_placement_controller.preparing_phone_layout()
    )
    if state is None or state.transition_ticket_id is None:
        return None
    ticket = next((
        row for row in controller._runtime_controller.current_tickets()
        if row.ticket_id == state.transition_ticket_id
    ), None)
    if ticket is None:
        return None
    try:
        token = runtime_residency_projection_token(
            ticket, state.generation
        )
    except (RuntimeResidencyProjectionError, RuntimeResourceError):
        return None
    if (
        token.token_sha256 != state.projection_token_sha256
        or token.target_geometry_sha256
            != state.layout.geometry_sha256
    ):
        return None
    return token


def _phone_projection_token_for_plan(
    controller, plan: RuntimeExecutionPlan
) -> RuntimeResidencyProjectionToken | None:
    token = controller._current_phone_projection_token()
    if token is None or not projection_token_matches_plan(token, plan):
        return None
    return token


def _bind_phone_residency_transition(
    controller,
    ticket: RuntimeRequestTicket,
    observed_at_us: int,
    snapshot: HeterogeneousRuntimeSnapshot,
) -> None:
    state = controller._model_placement_controller.planning_phone_layout()
    plan = ticket.execution_plan
    if state is None or plan is None:
        return
    geometry = plan.adapter_parameters.get(
        "phone_shard_set_geometry_sha256"
    )
    if geometry != state.layout.geometry_sha256:
        return
    if state.state == "READY":
        return
    if state.state == "PREPARING":
        token = ticket.residency_projection_token
        if (
            token is None
            or token.token_sha256
                != state.projection_token_sha256
            or not projection_token_matches_plan(token, plan)
        ):
            raise UnifiedScheduleError(
                "phone route does not bind the active layout transition"
            )
        return
    phone_transitions = tuple(
        row for row in plan.transitions if row.phone_shards
    )
    if not phone_transitions:
        verification = controller._observed_phone_layout_verification(
            state, ticket, snapshot
        )
        if verification is None:
            raise UnifiedScheduleError(
                "proposed phone layout lacks its physical transition"
            )
        workspace_bytes, verification_sha256 = verification
        try:
            ready = (
                controller._model_placement_controller
                .verify_observed_phone_layout(
                    state.generation,
                    workspace_bytes=workspace_bytes,
                    verified_at_us=observed_at_us,
                    verification_sha256=verification_sha256,
                )
            )
        except ModelPlacementControllerError as exc:
            raise UnifiedScheduleError(str(exc)) from exc
        if ready is None:
            return
        for compiler in (
            controller._automated_route_compiler,
            controller._runtime_epoch_route_compiler,
        ):
            if compiler is not None:
                compiler.set_phone_residency_layout(ready.layout)
        return
    token = runtime_residency_projection_token(
        ticket, state.generation
    )
    workspace_bytes = sum(
        demand.required_bytes
        for demand in plan.memory_demands
        if demand.kind == "workspace"
        and demand.device_id
            == plan.execution_contract.phone_device_id
    )
    try:
        controller._validate_phone_session_replacement_authorization(
            state,
            None
            if plan.helper_envelope is None else
            plan.helper_envelope.replacement_authorization,
        )
        controller._prevalidate_target_layout_helper_envelopes(
            state,
            request_id=ticket.request.request_id,
            observed_at_us=observed_at_us,
        )
        controller._model_placement_controller.begin_phone_layout_transition(
            state.generation,
            ticket_id=ticket.ticket_id,
            transition_ids=tuple(
                row.transition_id for row in phone_transitions
            ),
            ready_at_us=token.ready_at_us,
            projection_token_sha256=token.token_sha256,
            workspace_bytes=workspace_bytes,
            observed_at_us=observed_at_us,
        )
    except ModelPlacementControllerError as exc:
        raise UnifiedScheduleError(str(exc)) from exc


def _fail_phone_residency_transition(
    controller,
    ticket_id: str,
    failed_at_us: int,
    reason: str,
) -> None:
    state = (
        controller._model_placement_controller.preparing_phone_layout()
    )
    if (
        state is None
        or state.transition_ticket_id != ticket_id
        or state.projection_token_sha256 is None
    ):
        return
    try:
        controller._model_placement_controller.fail_phone_layout_transition(
            ticket_id,
            generation=state.generation,
            projection_token_sha256=(
                state.projection_token_sha256
            ),
            failed_at_us=failed_at_us,
            reason=reason,
        )
    except ModelPlacementControllerError as exc:
        raise UnifiedScheduleError(str(exc)) from exc
    state = controller._model_placement_controller.planning_phone_layout()
    layout = None if state is None else state.layout
    for compiler in (
        controller._automated_route_compiler,
        controller._runtime_epoch_route_compiler,
    ):
        if compiler is not None:
            compiler.set_phone_residency_layout(layout)


def _phone_residency_portfolio_authorization(
    controller,
    candidate: AutomatedRouteCandidate,
    manifest: ModelManifest,
) -> Mapping[str, object] | None:
    """Return the queue authorization for one selected cold layout."""

    layout_state = (
        controller._model_placement_controller.planning_phone_layout()
    )
    if layout_state is None:
        return None
    layout = layout_state.layout
    geometry = candidate.plan.adapter_parameters.get(
        "phone_shard_set_geometry_sha256"
    )
    if geometry != layout.geometry_sha256 or not any(
        shard.artifact_sha256 == manifest.artifact_sha256
        for shard in layout.shards
    ):
        return None
    if layout_state.state not in {"PROPOSED", "PREPARING", "READY"}:
        return None
    queue_benefit = layout_state.queue_benefit_uj
    transition_cost = layout_state.transition_cost_uj
    switching_margin = layout_state.switching_margin_uj
    queued_work = dict(layout_state.queue_work_by_artifact)
    if (
        type(queue_benefit) is not int
        or type(transition_cost) is not int
        or type(switching_margin) is not int
        or type(queued_work.get(manifest.artifact_sha256)) is not int
        or queue_benefit - transition_cost <= switching_margin
    ):
        return None
    return MappingProxyType({
        "artifact_queued_work": queued_work[
            manifest.artifact_sha256
        ],
        "artifact_queue_benefit_uj": (
            layout.queue_benefit_by_artifact[
                manifest.artifact_sha256
            ]
        ),
        "layout_geometry_sha256": layout.geometry_sha256,
        "layout_generation": layout_state.generation,
        "layout_selection_reason": layout_state.selection_reason,
        "layout_state": layout_state.state,
        "objective_uj": transition_cost - queue_benefit,
        "queue_benefit_uj": queue_benefit,
        "switching_margin_uj": switching_margin,
        "transition_cost_uj": transition_cost,
    })


def _phone_service_admission_reason(controller, candidate, manifest, snapshot, observed_at_us=None):
    parameters = candidate.plan.adapter_parameters
    device_id = parameters.get("gpu_device_id")
    if parameters.get("execution_adapter") != "android-llama-server-v1":
        return None
    catalog = controller._runtime_capabilities
    helper = catalog.executor_by_device.get(device_id)
    capped = device_id in controller._phone_htp_memory_caps
    if helper is None or not (capped or helper.phone_sessions):
        return None
    peak = parameters.get("whole_model_peak_memory_bytes")
    if type(peak) is not int or peak < manifest.tensor_bytes:
        return "PHONE_SERVICE_PEAK_MEMORY_UNKNOWN"
    if snapshot is None or device_id not in snapshot.telemetry_observations:
        return "PHONE_SERVICE_MEMORY_OBSERVATION_UNAVAILABLE"
    observed_at_us = snapshot.captured_at_us if observed_at_us is None else observed_at_us
    if snapshot.telemetry_unavailable_reason(device_id, observed_at_us) is not None:
        return "PHONE_SERVICE_MEMORY_OBSERVATION_UNAVAILABLE"
    budget = controller._phone_memory_budget(device_id, helper.phone_sessions, snapshot)
    if budget.reservation_error is not None:
        return budget.reservation_error
    if budget.current is not None and budget.current.resident_bytes > budget.phone_wide_limit:
        return "PHONE_SERVICE_MEMORY_REBALANCE_PENDING"
    ready = controller._model_placement_controller.ready_phone_layout()
    retained_bytes = sum(row.resident_bytes for row in snapshot.residency
                         if row.device_id == device_id and row.executor_id != helper.executor_id
                         and row.state in {"hot", "warm"})
    if retained_bytes and ready is None:
        return "PHONE_SERVICE_RESIDENCY_OBSERVATION_UNAVAILABLE"
    retained_bytes = max(retained_bytes, budget.accepted_resident_bytes)
    workspace = max(budget.htp_workspace_bytes, 0 if ready is None else ready.workspace_bytes)
    peaks = dict(controller._persistent_phone_service_reserve_by_artifact(
        device_id, snapshot, include_observed=True,
    ))
    peaks[manifest.artifact_sha256] = max(peak, peaks.get(manifest.artifact_sha256, 0))
    pool = catalog.placement_profile.memory_pools[helper.memory_resource_id]
    reserve = max(pool.reserved_bytes, 0 if budget.live_capacity is None else budget.live_capacity.reserve_bytes)
    limit = max(0, min(catalog.placement_profile.devices[device_id].allocation_limit_bytes,
                       pool.capacity_bytes) - reserve)
    if retained_bytes + workspace + sum(peaks.values()) > limit:
        return "PHONE_SERVICE_MEMORY_REBALANCE_PENDING"
    missing = controller._persistent_phone_service_reserve_by_artifact(device_id, snapshot).get(
        manifest.artifact_sha256, peak,
    )
    if helper.adapter_parameters.get("persistent_residency") == 1 and any(
        row.device_id == device_id and row.executor_id == helper.executor_id
        and row.artifact_sha256 == manifest.artifact_sha256
        and row.state in {"hot", "warm"} for row in snapshot.residency
    ):
        missing = controller._persistent_phone_service_reserve_by_artifact(device_id, snapshot).get(
            manifest.artifact_sha256, 0,
        )
    if budget.live_capacity is None or missing > budget.live_capacity.available_bytes:
        return "MEMORY_CAPACITY"
    return None


def _apply_phone_residency_portfolio_authorization(
    controller,
    candidate_set: AutomatedCandidateSet,
    manifest: ModelManifest,
    request: Request,
    *, snapshot: HeterogeneousRuntimeSnapshot | None = None,
    observed_at_us: int | None = None,
) -> AutomatedCandidateSet:
    """Replace request-local cold rejection with queue evidence."""

    changed = False
    rows = []
    candidate_by_id = {
        row.candidate_id: row for row in candidate_set.candidates
    }
    for candidate in candidate_set.candidates:
        reason = _phone_service_admission_reason(
            controller, candidate, manifest, snapshot, observed_at_us,
        )
        if reason is not None:
            reasons = tuple(sorted({*candidate.rejection_reasons, reason}))
            rows.append(replace(
                candidate, admitted=False, rejection_reasons=reasons,
                binding=replace(candidate.binding, ready=False, eligibility_reasons=reasons),
            ))
            changed = True
            continue
        if "COLD_RESIDENCY_BREAK_EVEN" not in (
            candidate.rejection_reasons
        ):
            rows.append(candidate)
            continue
        authorization = controller._phone_residency_portfolio_authorization(
            candidate, manifest
        )
        if authorization is None:
            rows.append(candidate)
            continue
        parent = candidate_by_id.get(
            candidate.paired_baseline_route_id
        )
        parent_lower = (
            None
            if parent is None
            else parent.cost.fleet_energy_lower_uj
        )
        artifact_work = authorization["artifact_queued_work"]
        artifact_benefit = authorization[
            "artifact_queue_benefit_uj"
        ]
        queue_benefit = authorization["queue_benefit_uj"]
        transition_cost = authorization["transition_cost_uj"]
        switching_margin = authorization["switching_margin_uj"]
        request_work = min(
            artifact_work, max(1, request.output_tokens)
        )
        request_benefit = (
            artifact_benefit * request_work // artifact_work
        )
        transition_share = _ceil_div(
            transition_cost * request_benefit, queue_benefit
        )
        margin_share = _ceil_div(
            switching_margin * request_benefit, queue_benefit
        )
        net_benefit = max(
            0,
            request_benefit - transition_share,
        )
        if parent_lower is None or net_benefit <= 0:
            rows.append(candidate)
            continue
        portfolio_upper = max(1, parent_lower - net_benefit)
        authorization = MappingProxyType({
            **dict(authorization),
            "hysteresis_share_uj": margin_share,
            "net_benefit_share_uj": net_benefit,
            "request_queue_benefit_share_uj": request_benefit,
            "request_work": request_work,
            "transition_share_uj": transition_share,
        })
        reasons = tuple(
            reason for reason in candidate.rejection_reasons
            if reason != "COLD_RESIDENCY_BREAK_EVEN"
        )
        break_even = {
            **dict(candidate.residency_break_even or {}),
            "passed": True,
            "phone_residency_portfolio_authorization": dict(
                authorization
            ),
            "portfolio_effective_paired_energy_upper_uj": (
                portfolio_upper
            ),
        }
        rows.append(replace(
            candidate,
            admitted=not reasons,
            binding=replace(
                candidate.binding,
                ready=not reasons,
                eligibility_reasons=reasons,
            ),
            rejection_reasons=reasons,
            residency_break_even=break_even,
        ))
        changed = True
    if not changed:
        return candidate_set
    return replace(candidate_set, candidates=tuple(rows))
