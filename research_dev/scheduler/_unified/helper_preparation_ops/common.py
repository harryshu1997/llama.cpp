"""Shared records and helpers; no independent controller state."""

from __future__ import annotations

from typing import NamedTuple


class _HelperAttachLeases(NamedTuple):
    """Helper window leases settled before one attach attempt."""

    lease_tokens: tuple[str, ...]
    newly_reserved_tokens: tuple[str, ...]
    finish_us: int | None


def shared_helper_attachments(controller, ticket, helper):
    """Live attachments on the same server and immutable phone layout."""
    if not controller._adaptive_decode_config.server_policy_coherence:
        return ()
    matches = []
    for other in controller._runtime_controller.current_tickets(("ACQUIRED",)):
        if (other.request.request_id == ticket.request.request_id
                or other.model.artifact_sha256 != ticket.model.artifact_sha256
                or other.binding.endpoint != ticket.binding.endpoint):
            continue
        binding = controller._model_placement_controller.request_binding(other.request.request_id)
        if binding is None:
            continue
        attachment = binding.get("helper_attachment")
        envelope = binding.get("helper_envelope")
        if (attachment and envelope and attachment.get("lease_tokens")
                and binding["base"]["desktop_placement_sha256"] == helper.desktop_placement_sha256
                and attachment["phone_layout_generation"] == helper.phone_layout_generation
                and attachment["phone_layout_geometry_sha256"] == helper.phone_layout_geometry_sha256
                and envelope["assisted_layer_mask"] == helper.resident_layer_mask
                and envelope["maximum_columns"] == helper.resident_columns
                and set(envelope["resource_ids"]) == set(helper.helper_plan.resource_ids)):
            matches.append((other.request.request_id, attachment))
    return tuple(matches)


def renew_shared_helper_leases(controller, tokens, reserved_until_us, observed_at_us):
    for ticket in controller._runtime_controller.current_tickets(("ACQUIRED",)):
        request_id = ticket.request.request_id
        binding = controller._model_placement_controller.request_binding(request_id)
        attachment = None if binding is None else binding.get("helper_attachment")
        if (attachment and binding.get("fraction_ppm", 0) > 0
                and tuple(attachment.get("lease_tokens", ())) == tuple(tokens)):
            controller._model_placement_controller.renew_request_helper_leases(
                request_id, lease_tokens=tokens, reserved_until_us=reserved_until_us,
                observed_at_us=observed_at_us)
