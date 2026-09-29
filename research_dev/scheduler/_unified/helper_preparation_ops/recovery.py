"""HelperPreparationMixin recovery operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot
from ..._internal.model_placement_controller import ModelPlacementControllerError
from ..._internal.runtime_resources import RuntimeResourceError
from ..._internal.adaptive_decode_contracts import AdaptiveDecodeError
from ..common import _RECOVERABLE_ERRORS, _RequestHelperPreparation, _text
from ..helper_preparation_checks import _rebind_blocker_quiescence


def fail_request_helper_preparation(
    controller,
    request_id: str,
    preparation_ticket_id: str,
    *,
    failed_at_us: int,
    reason: str,
    unavailable_session_ids: Sequence[str] = (),
    restored_session_generations: Mapping[str, int] | None = None,
    snapshot: HeterogeneousRuntimeSnapshot | None = None,
) -> None:
    preparation = controller._request_helper_preparations.get(
        preparation_ticket_id
    )
    if preparation is None or preparation.state != "TRANSITIONING":
        return
    unavailable = tuple(sorted(set(
        _text("unavailable phone session", value)
        for value in unavailable_session_ids
    )))
    try:
        with controller._transaction(errors=Exception, convert=False):
            controller._fail_request_helper_preparation_transaction(
                request_id=request_id,
                preparation_ticket_id=preparation_ticket_id,
                preparation=preparation,
                failed_at_us=failed_at_us,
                reason=reason,
                unavailable=unavailable,
                restored_session_generations=(
                    restored_session_generations
                ),
            )
    except Exception as exc:
        if isinstance(
            exc,
            (UnifiedScheduleError, RuntimeResourceError,
             ModelPlacementControllerError),
        ):
            raise UnifiedScheduleError(str(exc)) from exc
        raise
    if snapshot is not None:
        controller._reevaluate_failed_layout_portfolio(
            request_id,
            failed_at_us,
            snapshot,
        )


def _fail_request_helper_preparation_transaction(
    controller,
    *,
    request_id: str,
    preparation_ticket_id: str,
    preparation: _RequestHelperPreparation,
    failed_at_us: int,
    reason: str,
    unavailable: tuple[str, ...],
    restored_session_generations: Mapping[str, int] | None,
) -> None:
    affected_rebinds = controller._rebinds_targeting_generation(
        preparation.phone_layout_generation
    )
    controller._cancel_request_helper_preparation(preparation, failed_at_us)
    controller._model_placement_controller.fail_phone_layout_transition(
        preparation_ticket_id,
        generation=preparation.phone_layout_generation,
        projection_token_sha256=preparation.projection_token_sha256,
        failed_at_us=failed_at_us,
        reason=reason,
        unavailable_session_ids=unavailable,
        restored_session_generations=restored_session_generations,
    )
    ready = controller._model_placement_controller.ready_phone_layout()
    for compiler in (
        controller._automated_route_compiler,
        controller._runtime_epoch_route_compiler,
    ):
        if compiler is not None:
            compiler.set_phone_residency_layout(
                None if ready is None else ready.layout
            )
    failed = replace(preparation, state="FAILED")
    controller._request_helper_preparations[preparation_ticket_id] = failed
    controller._model_placement_controller.record_request_helper_event(
        request_id,
        "PREPARATION_FAILED",
        failed_at_us,
        {
            **controller._helper_preparation_json(failed),
            "reason": reason,
        },
    )
    try:
        controller._adaptive_decode.helper_unavailable(request_id)
    except AdaptiveDecodeError:
        pass
    for affected_id, affected_rebind in affected_rebinds:
        controller._settle_rebind_after_transition_failure(
            affected_id,
            affected_rebind,
            preparation_ticket_id=preparation_ticket_id,
            failed_at_us=failed_at_us,
            reason=reason,
            unavailable=unavailable,
        )


def _reevaluate_failed_layout_portfolio(
    controller,
    request_id: str,
    failed_at_us: int,
    snapshot: HeterogeneousRuntimeSnapshot,
) -> None:
    """Replan from the restored physical authority after a load failure."""

    try:
        if not isinstance(snapshot, HeterogeneousRuntimeSnapshot):
            raise UnifiedScheduleError(
                "request helper failure snapshot is invalid"
            )
        observed_at_us = max(failed_at_us, snapshot.captured_at_us)
        snapshot.validate_at(observed_at_us)
        ticket = controller.runtime_ticket(request_id)
        if ticket.dispatch_state in {"CANCELLED", "COMPLETED", "FAILED"}:
            return
        with controller._transaction(convert=False):
            controller._update_phone_residency_portfolio(
                ticket.request,
                controller.runtime_model_manifest(ticket.model.model_id),
                observed_at_us,
                snapshot,
            )
    except _RECOVERABLE_ERRORS as exc:
        controller._model_placement_controller.record_request_helper_event(
            request_id,
            "FAILED_LAYOUT_REEVALUATION_FAILED",
            failed_at_us,
            {"reason": str(exc)},
        )


def _rebinds_targeting_generation(
    controller,
    generation: int,
) -> tuple[tuple[str, Mapping[str, object]], ...]:
    return tuple(
        (row.request.request_id, rebind)
        for row in controller._runtime_controller.current_tickets()
        if (
            (rebind := controller._model_placement_controller
                .request_helper_rebind_state(
                    row.request.request_id
                )) is not None
            and rebind.get("target_generation") == generation
        )
    )


def _settle_rebind_after_transition_failure(
    controller,
    affected_id: str,
    affected_rebind: Mapping[str, object],
    *,
    preparation_ticket_id: str,
    failed_at_us: int,
    reason: str,
    unavailable: tuple[str, ...],
) -> None:
    """Retain, cancel, or detach one rebind blocked by a failed transition."""

    affected_binding = (
        controller._model_placement_controller.request_binding(
            affected_id
        )
    )
    active_masked, fully_quiesced, retained_session_ids = (
        _rebind_blocker_quiescence(affected_binding, affected_rebind)
    )
    if not (active_masked or fully_quiesced):
        raise UnifiedScheduleError(
            "failed helper transition has a live blocker"
        )
    if (
        affected_rebind.get("state") == "QUIESCED"
        and not unavailable
        and controller._model_placement_controller
            .phone_layout_sessions_are_usable(
                int(affected_rebind["source_generation"]),
                str(affected_rebind["source_geometry_sha256"]),
                retained_session_ids,
            )
    ):
        # The source layout is physically restored, so the
        # quiesced helper keeps its rebind and a retried
        # proposal resumes it instead of re-requesting 0%.
        controller._model_placement_controller\
            .record_request_helper_event(
                affected_id,
                "REBIND_RETAINED_AFTER_ROLLBACK",
                failed_at_us,
                {
                    **dict(affected_rebind),
                    "failed_preparation_ticket_id": (
                        preparation_ticket_id
                    ),
                    "reason": reason,
                },
            )
        return
    retained_sessions_usable = bool(
        active_masked
        and not set(retained_session_ids) & set(unavailable)
        and controller._model_placement_controller
            .request_helper_layout_is_usable(
                affected_id,
                int(affected_rebind["source_generation"]),
                str(affected_rebind[
                    "source_geometry_sha256"
                ]),
            )
    )
    controller._model_placement_controller.cancel_request_helper_rebind(
        affected_id,
        observed_at_us=failed_at_us,
        reason="PHONE_LAYOUT_TRANSITION_FAILED",
    )
    if retained_sessions_usable:
        controller._model_placement_controller\
            .record_request_helper_event(
                affected_id,
                "HELPER_RETAINED_AFTER_SESSION_FAILURE",
                failed_at_us,
                {
                    **dict(affected_rebind),
                    "failed_preparation_ticket_id": (
                        preparation_ticket_id
                    ),
                    "reason": reason,
                    "unavailable_session_ids": list(unavailable),
                },
            )
        return
    controller._model_placement_controller.detach_request_helper(
        affected_id,
        fallback_outcome="PHONE_LAYOUT_TRANSITION_FAILED",
        observed_at_us=failed_at_us,
    )
    controller._request_helper_opportunities.pop(
        affected_id, None
    )
    controller._late_request_helper_contexts.pop(
        affected_id, None
    )
    try:
        controller._adaptive_decode.helper_unavailable(
            affected_id
        )
    except AdaptiveDecodeError:
        pass
