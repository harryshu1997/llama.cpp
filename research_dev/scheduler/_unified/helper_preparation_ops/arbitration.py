"""HelperPreparationMixin arbitration operations on its existing owner."""

from __future__ import annotations

from typing import Mapping, Sequence

from ..._internal.adaptive_decode_contracts import AdaptiveDecodeError
from .common import shared_helper_attachments


def _helper_window_bid(
    controller,
    request_id: str,
    *,
    requested_fraction_ppm: int | None = None,
) -> Mapping[str, object] | None:
    try:
        return controller._adaptive_decode.helper_window_bid(
            request_id,
            requested_fraction_ppm=requested_fraction_ppm,
        )
    except AdaptiveDecodeError:
        return None


def _bounded_learning_exploration_bid(
    controller,
    request_id: str,
    bid: Mapping[str, object] | None,
) -> bool:
    return bool(
        bid is not None
        and bid.get("learning_exploration_eligible") is True
    )


def _select_helper_window_owner(
    controller,
    request_id: str,
    required_resource_ids: Sequence[str],
    *,
    requested_fraction_ppm: int,
    observed_at_us: int,
) -> tuple[str | None, Mapping[str, object] | None]:
    """Rank one shared HTP window without reserving future work."""

    resources = frozenset(required_resource_ids)
    bids = []
    learning_bids = []
    for ticket in controller._runtime_controller.current_tickets(
        ("ACQUIRED",)
    ):
        candidate_id = ticket.request.request_id
        binding = controller._model_placement_controller.request_binding(
            candidate_id
        )
        if binding is None:
            continue
        envelope = binding.get("helper_envelope")
        if not isinstance(envelope, Mapping):
            continue
        layout_generation = envelope.get(
            "phone_layout_generation"
        )
        layout_geometry = envelope.get(
            "phone_layout_geometry_sha256"
        )
        if (
            type(layout_generation) is not int
            or type(layout_geometry) is not str
            or not controller._model_placement_controller
                .request_helper_layout_is_usable(
                    candidate_id,
                    layout_generation,
                    layout_geometry,
                )
        ):
            continue
        candidate_resources = frozenset(
            str(value)
            for value in envelope.get("resource_ids", ())
        )
        if not resources.intersection(candidate_resources):
            continue
        rebind = (
            controller._model_placement_controller
            .request_helper_rebind_state(candidate_id)
        )
        if (
            rebind is not None
            and not tuple(rebind.get(
                "target_allowed_session_ids", ()
            ))
        ):
            continue
        bid = controller._helper_window_bid(
            candidate_id,
            requested_fraction_ppm=(
                requested_fraction_ppm
                if candidate_id == request_id else None
            ),
        )
        profitable = bool(
            bid is not None
            and bool(bid["latency_eligible"])
            and int(bid["gain_uj"]) >= 0
        )
        learning_exploration = (
            not profitable
            and (controller._bounded_learning_exploration_bid(candidate_id, bid)
                 or bid is not None and bid.get("server_policy_eligible") is True)
        )
        if not profitable and not learning_exploration:
            continue
        assert bid is not None
        target = learning_bids if learning_exploration else bids
        target.append((
            0
            if (
                bid["evidence"] != "CONSERVATIVE_MEASURED"
                and int(bid["phone_window_count"]) == 0
            ) else 1,
            -int(bid["gain_uj"]) if profitable else 0,
            ticket.request.arrival_us,
            candidate_id,
            bid,
        ))
    winner = (
        min(bids) if bids else
        min(learning_bids) if learning_bids else None
    )
    winner_id = None if winner is None else winner[3]
    winner_bid = None if winner is None else winner[4]
    selection_kind = (
        None if winner is None else
        "PROFITABLE" if bids else "BOUNDED_LEARNING_EXPLORATION"
    )
    requested_bid = controller._helper_window_bid(
        request_id,
        requested_fraction_ppm=requested_fraction_ppm,
    )
    if winner_id is not None and winner_id != request_id:
        ticket = controller.runtime_execution_ticket(request_id)
        helper = controller._request_helper_envelope(ticket)
        if helper is not None and any(
            member_id == winner_id for member_id, _ in shared_helper_attachments(controller, ticket, helper)
        ):
            winner_id = request_id
            selection_kind = "SHARED_SERVER_POLICY"
    controller._model_placement_controller.record_request_helper_event(
        request_id,
        (
            "WINDOW_LEASE_SELECTED"
            if winner_id == request_id else
            "WINDOW_LEASE_DEFERRED"
        ),
        observed_at_us,
        {
            "requested_bid": (
                None if requested_bid is None else dict(requested_bid)
            ),
            "required_resource_ids": sorted(resources),
            "selection_kind": selection_kind,
            "winner_bid": (
                None if winner_bid is None else dict(winner_bid)
            ),
            "winner_request_id": winner_id,
        },
    )
    return winner_id, requested_bid


def _helper_window_lease_horizon(
    controller,
    request_id: str,
    *,
    requested_fraction_ppm: int,
    at_us: int,
) -> int:
    bid = controller._helper_window_bid(
        request_id,
        requested_fraction_ppm=requested_fraction_ppm,
    )
    if bid is None:
        duration_us = max(
            controller._adaptive_decode_config.measurement_resolution_us,
            controller._adaptive_decode_config.transition_cost_us,
        )
    else:
        duration_us = max(
            controller._adaptive_decode_config.measurement_resolution_us,
            int(bid["window_tokens"])
                * int(bid["latency_per_token_us"]),
        )
    return (
        at_us
        + duration_us
        + controller._adaptive_decode_config.transition_cost_us
        + 250_000
    )
