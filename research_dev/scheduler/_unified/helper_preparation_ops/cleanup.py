"""HelperPreparationMixin cleanup operations on its existing owner."""

from __future__ import annotations

from typing import Mapping

from ..automated_requests_ops.event_replanning import note_resource_release


def _release_request_helper_leases(
    controller,
    request_id: str,
    at_us: int,
) -> None:
    binding = controller._model_placement_controller.request_binding(
        request_id
    )
    if binding is None:
        return
    raw_attachment = binding.get("helper_attachment")
    if not isinstance(raw_attachment, Mapping):
        return
    lease_tokens = tuple(raw_attachment.get("lease_tokens", ()))
    if not lease_tokens:
        return
    retained = set()
    for ticket in controller._runtime_controller.current_tickets(("ACQUIRED",)):
        if ticket.request.request_id == request_id:
            continue
        other = controller._model_placement_controller.request_binding(ticket.request.request_id)
        attachment = None if other is None else other.get("helper_attachment")
        if isinstance(attachment, Mapping):
            retained.update(attachment.get("lease_tokens", ()))
    with controller._transaction(convert=False):
        for token in lease_tokens:
            if token not in retained:
                controller.release(str(token), at_us)
        controller._model_placement_controller.update_request_fraction(
            request_id,
            0,
            observed_at_us=at_us,
        )
        renewal = controller._runtime_renewals.get(request_id)
        if renewal is not None:
            renewal.wake()
    note_resource_release(controller, request_id, at_us, "HELPER_LEASES_RELEASED")


def _close_request_helper_runtime(
    controller,
    request_id: str,
    at_us: int,
    outcome: str,
) -> None:
    """Release request-scoped helper state without evicting residency."""

    for _preparation_ticket_id, preparation in tuple(
        controller._request_helper_preparations.items()
    ):
        if (
            preparation.request_id != request_id
            or preparation.state != "TRANSITIONING"
        ):
            continue
        controller._model_placement_controller.record_request_helper_event(
            request_id,
            "PREPARATION_CONTINUES_AFTER_REQUEST",
            at_us,
            {
                **controller._helper_preparation_json(preparation),
                "outcome": outcome,
            },
        )
    controller._release_request_helper_leases(request_id, at_us)
    binding = controller._model_placement_controller.request_binding(
        request_id
    )
    if (
        binding is not None
        and isinstance(binding.get("helper_attachment"), Mapping)
        and int(binding.get("fraction_ppm", 0)) != 0
    ):
        controller._model_placement_controller.update_request_fraction(
            request_id,
            0,
            observed_at_us=at_us,
        )
    binding = controller._model_placement_controller.request_binding(
        request_id
    )
    attachment = (
        None if binding is None else binding.get("helper_attachment")
    )
    base = None if binding is None else binding.get("base")
    if (
        isinstance(attachment, Mapping)
        and attachment.get("fallback_outcome") is None
    ):
        identities = attachment.get("phone_session_identities", ())
        controller._model_placement_controller.record_request_helper_event(
            request_id,
            "DETACHED",
            at_us,
            {
                "accepted": True,
                "operator_plan_sha256": attachment.get(
                    "operator_plan_sha256"
                ),
                "outcome": outcome,
                "phone_layout_generation": attachment.get(
                    "phone_layout_generation"
                ),
                "phone_layout_geometry_sha256": attachment.get(
                    "phone_layout_geometry_sha256"
                ),
                "request_ticket_id": (
                    None if not isinstance(base, Mapping) else
                    base.get("sequence_identity")
                ),
                "selected_fraction_ppm": 0,
                "session_generation_by_id": {
                    str(row["session_id"]): int(
                        row["session_generation"]
                    )
                    for row in identities
                    if isinstance(row, Mapping)
                },
                "template_layout_generation": attachment.get(
                    "phone_layout_generation"
                ),
            },
        )
    controller._request_helper_opportunities.pop(request_id, None)
    controller._request_helper_preparation_envelopes = {
        key: value
        for key, value in (
            controller._request_helper_preparation_envelopes.items()
        )
        if key[0] != request_id
    }
    controller._late_request_helper_contexts.pop(request_id, None)
    controller._request_helper_envelope_history.pop(request_id, None)
    note_resource_release(controller, request_id, at_us, outcome)
