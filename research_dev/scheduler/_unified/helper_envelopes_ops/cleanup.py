"""HelperEnvelopeMixin cleanup operations on its existing owner."""

from __future__ import annotations

from ..._internal.lifecycle import UnifiedScheduleError
from ..common import _RequestHelperPreparation


def _helper_preparation_json(
    preparation: _RequestHelperPreparation,
) -> dict[str, object]:
    return {
        "memory_owner_id": preparation.memory_owner_id,
        "operator_plan_sha256": preparation.operator_plan_sha256,
        "phone_layout_generation": (
            preparation.phone_layout_generation
        ),
        "phone_layout_geometry_sha256": (
            preparation.phone_layout_geometry_sha256
        ),
        "phone_safety_state": (
            None
            if preparation.phone_safety_state is None
            else preparation.phone_safety_state.to_json()
        ),
        "projection_token_sha256": (
            preparation.projection_token_sha256
        ),
        "replacement_authorization": (
            None
            if preparation.helper_envelope.replacement_authorization
                is None
            else preparation.helper_envelope
                .replacement_authorization.to_json()
        ),
        "preparation_ticket_id": preparation.preparation_ticket_id,
        "ready_at_us": preparation.ready_at_us,
        "request_id": preparation.request_id,
        "request_ticket_id": preparation.request_ticket_id,
        "resource_lease_tokens": list(
            preparation.resource_lease_tokens
        ),
        "yielding_resource_ids": list(
            preparation.yielding_resource_ids
        ),
        "started_at_us": preparation.started_at_us,
        "state": preparation.state,
        "transition_ids": list(preparation.transition_ids),
        "transition_receipts": [
            row.to_json() for row in preparation.transition_receipts
        ],
        "verification_sha256": preparation.verification_sha256,
    }


def _release_request_helper_preparation(
    controller,
    preparation: _RequestHelperPreparation,
    at_us: int,
) -> None:
    for token in preparation.resource_lease_tokens:
        try:
            controller.release(token, at_us)
        except UnifiedScheduleError as exc:
            if "unknown lease token" not in str(exc):
                raise
    controller._runtime_memory.release_owner(
        preparation.memory_owner_id
    )


def _cancel_request_helper_preparation(
    controller,
    preparation: _RequestHelperPreparation,
    at_us: int,
) -> None:
    controller.cancel(preparation.memory_owner_id, at_us)
    controller._runtime_memory.release_owner(
        preparation.memory_owner_id
    )
