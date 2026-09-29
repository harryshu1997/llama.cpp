"""HelperPreparationMixin start operations on its existing owner."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Sequence

from ..._internal.policy import LeaseDemand
from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot
from ..._internal.model_placement_controller import (
    ModelPlacementControllerError,
    ModelPhoneResidencyLayout,
)
from ..._internal.runtime_plan import RuntimeHelperExecutionEnvelope, RuntimeTransitionPlan
from ..._internal.runtime_resources import RuntimeResourceError
from ..._internal.runtime_controller import RuntimeRequestTicket
from ..common import _StalePhoneSessionAssignment
from ..helper_preparation_checks import (
    _helper_preparation_lease_demands,
    _helper_preparation_projection_sha256,
)


def begin_request_helper_preparation(
    controller,
    request_id: str,
    *,
    observed_at_us: int,
    snapshot: HeterogeneousRuntimeSnapshot,
    expected_phone_layout_generation: int | None = None,
    expected_phone_layout_geometry_sha256: str | None = None,
    expected_operator_plan_sha256: str | None = None,
) -> Mapping[str, object]:
    """Reserve and bind one exact asynchronous helper transition."""

    if not isinstance(snapshot, HeterogeneousRuntimeSnapshot):
        raise UnifiedScheduleError(
            "request helper preparation snapshot is invalid"
        )
    snapshot.validate_at(observed_at_us)
    ticket = controller.runtime_ticket(request_id)
    if ticket.dispatch_state in {
        "CANCELLED", "COMPLETED", "FAILED"
    }:
        return MappingProxyType({"status": "INCOMPATIBLE"})
    resolved = controller._preparation_helper_envelope(
        ticket,
        request_id,
        expected_phone_layout_generation=(
            expected_phone_layout_generation
        ),
        expected_phone_layout_geometry_sha256=(
            expected_phone_layout_geometry_sha256
        ),
        expected_operator_plan_sha256=expected_operator_plan_sha256,
    )
    if isinstance(resolved, str):
        return MappingProxyType({"status": resolved})
    helper = resolved
    preparation_ticket_id = helper.preparation_ticket_id(
        ticket.ticket_id
    )
    existing = controller._request_helper_preparations.get(
        preparation_ticket_id
    )
    if existing is not None:
        return MappingProxyType({
            **controller._helper_preparation_json(existing),
            "status": (
                "READY" if existing.state == "READY" else "FOLLOWER"
            ),
        })
    state, early = controller._proposed_phone_layout_for_preparation(
        ticket, helper
    )
    if state is None:
        return early
    stale = controller._reject_stale_phone_layout_proposal(
        request_id, ticket, state, snapshot, observed_at_us
    )
    if stale is not None:
        return stale
    deferred = controller._phone_telemetry_deferral(
        snapshot, observed_at_us, request_id
    )
    if deferred is not None:
        return deferred
    deferred = controller._defer_phone_layout_revalidation(
        request_id, state, observed_at_us,
        sum(row.latency_us for row in helper.preparation_transitions))
    if deferred is not None:
        return deferred
    blockers = (
        controller._model_placement_controller
        .phone_layout_transition_blockers(state.generation)
    )
    if blockers:
        return controller._defer_preparation_until_release(
            request_id, state, blockers, observed_at_us
        ) or controller._defer_preparation_for_blockers(
            request_id, state, blockers, observed_at_us
        )
    verified = controller._verify_proposed_layout_ready(
        request_id,
        ticket,
        helper,
        state,
        snapshot=snapshot,
        observed_at_us=observed_at_us,
        preparation_ticket_id=preparation_ticket_id,
    )
    if verified is not None:
        return verified
    transitions = helper.preparation_transitions
    if not transitions:
        return MappingProxyType({"status": "INCOMPATIBLE"})
    transition_ids = tuple(
        row.transition_id for row in transitions
    )
    duration_us = max(1, sum(row.latency_us for row in transitions))
    ready_at_us = observed_at_us + duration_us
    yielding_resource_ids = (
        controller._copy_on_write_preparation_yielding_resources(
            controller._model_placement_controller.ready_phone_layout(),
            state,
        )
    )
    demands = _helper_preparation_lease_demands(
        transitions,
        preparation_ticket_id=preparation_ticket_id,
        yielding_resource_ids=yielding_resource_ids,
        duration_us=duration_us,
    )
    memory_owner_id = "helper-preparation:" + preparation_ticket_id
    try:
        with controller._transaction(errors=Exception, convert=False):
            return controller._begin_request_helper_preparation_transaction(
                request_id=request_id,
                ticket=ticket,
                helper=helper,
                state=state,
                snapshot=snapshot,
                preparation_ticket_id=preparation_ticket_id,
                transitions=transitions,
                transition_ids=transition_ids,
                yielding_resource_ids=yielding_resource_ids,
                demands=demands,
                duration_us=duration_us,
                ready_at_us=ready_at_us,
                memory_owner_id=memory_owner_id,
                observed_at_us=observed_at_us,
            )
    except Exception as exc:
        if isinstance(exc, _StalePhoneSessionAssignment):
            return controller._defer_preparation_for_stale_session(
                request_id, ticket, state, snapshot, observed_at_us
            )
        if (
            isinstance(exc, RuntimeResourceError)
            and str(exc).startswith(
                "memory capacity is insufficient:"
            )
        ):
            return controller._defer_preparation_for_memory_capacity(
                request_id, state, exc, observed_at_us
            )
        if isinstance(
            exc,
            (UnifiedScheduleError, RuntimeResourceError,
             ModelPlacementControllerError),
        ):
            raise UnifiedScheduleError(str(exc)) from exc
        raise


def _begin_request_helper_preparation_transaction(
    controller,
    *,
    request_id: str,
    ticket: RuntimeRequestTicket,
    helper: RuntimeHelperExecutionEnvelope,
    state: ModelPhoneResidencyLayout,
    snapshot: HeterogeneousRuntimeSnapshot,
    preparation_ticket_id: str,
    transitions: Sequence[RuntimeTransitionPlan],
    transition_ids: tuple[str, ...],
    yielding_resource_ids: tuple[str, ...],
    demands: Sequence[LeaseDemand],
    duration_us: int,
    ready_at_us: int,
    memory_owner_id: str,
    observed_at_us: int,
) -> Mapping[str, object]:
    if demands:
        preview = controller.timeline.preview_leases(
            demands, observed_at_us, duration_us, duration_us
        )
        if preview.start_us != observed_at_us:
            return MappingProxyType({
                "next_start_us": preview.start_us,
                "status": "DEFERRED",
            })
        leases = controller.timeline.commit_leases(preview, memory_owner_id)
    else:
        leases = ()
    phone_device_id = (
        helper.helper_plan.execution_contract.phone_device_id
    )
    memory_demands = controller._reserve_helper_preparation_memory(
        helper,
        state,
        transitions,
        snapshot,
        phone_device_id=phone_device_id,
        memory_owner_id=memory_owner_id,
        observed_at_us=observed_at_us,
    )
    projection_sha256 = _helper_preparation_projection_sha256(
        state,
        helper,
        memory_demands,
        preparation_ticket_id=preparation_ticket_id,
        ready_at_us=ready_at_us,
        transition_ids=transition_ids,
        yielding_resource_ids=yielding_resource_ids,
    )
    workspace_bytes = sum(
        row.required_bytes for row in memory_demands
        if row.kind == "workspace"
    )
    controller._prevalidate_target_layout_helper_envelopes(
        state,
        request_id=request_id,
        observed_at_us=observed_at_us,
    )
    controller._model_placement_controller.begin_phone_layout_transition(
        state.generation,
        ticket_id=preparation_ticket_id,
        transition_ids=transition_ids,
        ready_at_us=ready_at_us,
        projection_token_sha256=projection_sha256,
        workspace_bytes=workspace_bytes,
        observed_at_us=observed_at_us,
    )
    preparation = controller._new_request_helper_preparation(
        request_id,
        ticket,
        helper,
        state,
        preparation_ticket_id=preparation_ticket_id,
        projection_sha256=projection_sha256,
        transition_ids=transition_ids,
        lease_tokens=tuple(row.token for row in leases),
        yielding_resource_ids=yielding_resource_ids,
        memory_owner_id=memory_owner_id,
        observed_at_us=observed_at_us,
        ready_at_us=ready_at_us,
        phone_safety_state=controller._preparation_phone_safety_state(
            phone_device_id, snapshot
        ),
    )
    controller._request_helper_preparations[preparation_ticket_id] = preparation
    controller._model_placement_controller.record_request_helper_event(
        request_id,
        "PREPARATION_STARTED",
        observed_at_us,
        controller._helper_preparation_json(preparation),
    )
    return MappingProxyType({
        **controller._helper_preparation_json(preparation),
        "status": "OWNER",
    })
