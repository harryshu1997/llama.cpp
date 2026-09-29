"""HelperPreparationMixin attachment operations on its existing owner."""

from __future__ import annotations

from typing import Mapping

from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.model_placement_controller import (
    ModelPlacementControllerError,
    ModelPhoneResidencyLayout,
)
from ..._internal.runtime_plan import HelperOpportunity, RuntimeHelperExecutionEnvelope
from ..._internal.runtime_controller import RuntimeRequestTicket
from ..common import _RequestHelperAttachAttempt
from .common import _HelperAttachLeases, renew_shared_helper_leases, shared_helper_attachments


def _try_attach_ready_request_helper(
    controller,
    ticket: RuntimeRequestTicket,
    *,
    slot_id: int,
    token_index: int,
    at_us: int,
    fraction_ppm: int,
    ready_helper: tuple | None = None,
) -> _RequestHelperAttachAttempt:
    ready = (
        controller._ready_request_helper(ticket)
        if ready_helper is None else ready_helper
    )
    if ready is None:
        return _RequestHelperAttachAttempt(
            False, True, "PHONE_HELPER_NOT_READY"
        )
    helper, layout, component = ready
    binding = controller._model_placement_controller.request_binding(
        ticket.request.request_id
    )
    if binding is None:
        return _RequestHelperAttachAttempt(
            False, False, "REQUEST_BINDING_ABSENT"
        )
    if not binding.get("helper_attachment"):
        opportunity = controller._adaptive_decode.helper_attachment_opportunity(
            ticket.request.request_id, token_index=token_index, at_us=at_us,
        )
        if opportunity != "ELIGIBLE":
            controller._record_ready_helper_event_once(
                ticket, layout, "REJECTED", at_us, helper=helper,
                accepted=False, reason=opportunity,
            )
            return _RequestHelperAttachAttempt(False, True, opportunity)
    base = binding.get("base")
    if (
        not isinstance(base, Mapping)
        or base.get("artifact_sha256")
            != ticket.model.artifact_sha256
        or base.get("route_id") != ticket.decision.route_id
        or base.get("desktop_placement_sha256")
            != helper.desktop_placement_sha256
        or base.get("kv_cache_owner_id") != ticket.ticket_id
        or base.get("sequence_identity") != ticket.ticket_id
    ):
        return _RequestHelperAttachAttempt(
            False, False, "BASE_BINDING_MISMATCH"
        )
    try:
        controller._model_placement_controller.bind_request_server_slot(
            ticket.request.request_id,
            sequence_identity=ticket.ticket_id,
            server_slot_id=slot_id,
        )
    except ModelPlacementControllerError:
        return _RequestHelperAttachAttempt(
            False, False, "SERVER_SLOT_MISMATCH"
        )
    leases = controller._attach_helper_window_leases(
        ticket,
        helper,
        layout,
        current_attachment=binding.get("helper_attachment"),
        token_index=token_index,
        at_us=at_us,
        fraction_ppm=fraction_ppm,
    )
    if isinstance(leases, _RequestHelperAttachAttempt):
        return leases
    lease_tokens, newly_reserved_tokens, finish_us = leases
    envelope = controller._request_helper_envelope_binding(helper, layout)
    try:
        if controller._model_placement_controller\
                .request_helper_envelope_is_additive(
                    ticket.request.request_id, envelope
                ):
            controller._model_placement_controller\
                .expand_request_helper_envelope(
                    ticket.request.request_id,
                    envelope,
                    resident_component_identity_sha256=(
                        component.identity_sha256
                    ),
                    start_token_index=token_index,
                    fraction_ppm=fraction_ppm,
                    lease_tokens=lease_tokens,
                    lease_reserved_until_us=(
                        int(finish_us) if lease_tokens else None
                    ),
                    observed_at_us=at_us,
                )
        else:
            controller._model_placement_controller.attach_request_helper(
                ticket.request.request_id,
                phone_layout_generation=layout.generation,
                phone_layout_geometry_sha256=(
                    layout.layout.geometry_sha256
                ),
                resident_component_identity_sha256=(
                    component.identity_sha256
                ),
                operator_plan_sha256=helper.operator_plan_sha256,
                start_token_index=token_index,
                fraction_ppm=fraction_ppm,
                lease_tokens=lease_tokens,
                lease_reserved_until_us=(
                    int(finish_us) if lease_tokens else None
                ),
                observed_at_us=at_us,
            )
        renewal = controller._runtime_renewals.get(
            ticket.request.request_id
        )
        if renewal is not None:
            renewal.wake()
    except ModelPlacementControllerError:
        for token in newly_reserved_tokens:
            try:
                controller.release(token, at_us)
            except UnifiedScheduleError:
                pass
        return _RequestHelperAttachAttempt(
            False, False, "HELPER_ATTACHMENT_MISMATCH"
        )
    return _RequestHelperAttachAttempt(True, False, "ATTACHED")


def _attach_helper_window_leases(
    controller,
    ticket: RuntimeRequestTicket,
    helper: RuntimeHelperExecutionEnvelope,
    layout: ModelPhoneResidencyLayout,
    *,
    current_attachment: object,
    token_index: int,
    at_us: int,
    fraction_ppm: int,
) -> _HelperAttachLeases | _RequestHelperAttachAttempt:
    """Reserve, extend, or release the helper window leases."""

    lease_tokens: tuple[str, ...] = ()
    newly_reserved_tokens: tuple[str, ...] = ()
    current_lease_horizon_us: int | None = None
    if isinstance(current_attachment, Mapping):
        lease_tokens = tuple(
            str(value)
            for value in current_attachment.get("lease_tokens", ())
        )
        raw_horizon = current_attachment.get(
            "lease_reserved_until_us"
        )
        if type(raw_horizon) is int:
            current_lease_horizon_us = raw_horizon
    finish_us = (
        None
        if fraction_ppm == 0 else
        controller._helper_window_lease_horizon(
            ticket.request.request_id,
            requested_fraction_ppm=fraction_ppm,
            at_us=at_us,
        )
    )
    if fraction_ppm > 0 and not lease_tokens:
        shared = shared_helper_attachments(controller, ticket, helper)
        capacity = min(4, helper.helper_plan.execution_contract.maximum_batch_size,
                       int(ticket.execution_plan.adapter_parameters.get("parallel", 1)))
        if shared and len(shared) >= capacity:
            return _RequestHelperAttachAttempt(False, True, "HELPER_BATCH_FULL")
        if shared:
            token_sets = {tuple(row["lease_tokens"]) for _, row in shared}
            if len(token_sets) != 1:
                return _RequestHelperAttachAttempt(False, False, "HELPER_SHARED_LEASES_DIFFER")
            lease_tokens = next(iter(token_sets))
            current_lease_horizon_us = max(row["lease_reserved_until_us"] for _, row in shared)
            controller._model_placement_controller.record_request_helper_event(
                ticket.request.request_id, "SERVER_HELPER_LEASES_SHARED", at_us,
                {"member_request_ids": [request_id for request_id, _ in shared],
                 "lease_tokens": list(lease_tokens), "phone_layout_generation": layout.generation},
            )
    if fraction_ppm > 0 and not lease_tokens:
        base_resources = set(ticket.execution_plan.resource_ids)
        resources = tuple(sorted(
            set(helper.helper_plan.resource_ids) - base_resources
        ))
        if not resources:
            return _RequestHelperAttachAttempt(
                False, False, "HELPER_RESOURCE_PLAN_EMPTY"
            )
        owner_id, _bid = controller._select_helper_window_owner(
            ticket.request.request_id,
            resources,
            requested_fraction_ppm=fraction_ppm,
            observed_at_us=at_us,
        )
        if owner_id != ticket.request.request_id:
            return _RequestHelperAttachAttempt(
                False, True, "HELPER_WINDOW_NOT_SELECTED"
            )
        try:
            leases = controller.reserve_external_resources(
                resources,
                (
                    "request-helper:"
                    + ticket.ticket_id
                    + ":"
                    + str(layout.generation)
                    + ":window:"
                    + str(token_index)
                ),
                at_us,
                int(finish_us),
            )
        except UnifiedScheduleError:
            return _RequestHelperAttachAttempt(
                False, True, "HELPER_RESOURCES_BUSY"
            )
        lease_tokens = tuple(row.token for row in leases)
        newly_reserved_tokens = lease_tokens
    elif fraction_ppm > 0 and lease_tokens:
        finish_us = max(
            int(finish_us),
            0 if current_lease_horizon_us is None
            else current_lease_horizon_us,
        )
        try:
            with controller._transaction(convert=False):
                for token in lease_tokens:
                    controller.extend_lease(token, int(finish_us))
                renew_shared_helper_leases(controller, lease_tokens, int(finish_us), at_us)
        except UnifiedScheduleError:
            return _RequestHelperAttachAttempt(
                False, True, "HELPER_LEASE_EXTENSION_CONFLICT"
            )
    elif fraction_ppm == 0 and lease_tokens:
        controller._release_request_helper_leases(
            ticket.request.request_id, at_us
        )
        lease_tokens = ()
    return _HelperAttachLeases(
        lease_tokens, newly_reserved_tokens, finish_us
    )


def _attach_ready_request_helper(
    controller,
    ticket: RuntimeRequestTicket,
    *,
    slot_id: int,
    token_index: int,
    at_us: int,
    fraction_ppm: int,
    ready_helper: tuple | None = None,
) -> bool:
    return controller._try_attach_ready_request_helper(
        ticket,
        slot_id=slot_id,
        token_index=token_index,
        at_us=at_us,
        fraction_ppm=fraction_ppm,
        ready_helper=ready_helper,
    ).attached


def _request_helper_opportunity_for_ticket(
    controller,
    ticket: RuntimeRequestTicket,
) -> HelperOpportunity | None:
    plan = ticket.execution_plan
    if plan is None or plan.desktop_placement_sha256 is None:
        return None
    route_ids = {
        row.route_id for row in ticket.cost_estimates.estimates
    }
    return next((
        opportunity
        for opportunity in controller._request_helper_opportunities.get(
            ticket.request.request_id, ()
        )
        if (
            opportunity.route_id in route_ids
            and opportunity.desktop_parent_route_id
                == ticket.decision.route_id
            and opportunity.desktop_parent_placement_sha256
                == plan.desktop_placement_sha256
            and opportunity.helper_operator_plan.baseline_executor_id
                == ticket.binding.executor_id
            and opportunity.helper_binding.artifact_sha256
                == ticket.model.artifact_sha256
        )
    ), None)
