"""HelperPreparationMixin completion operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import Mapping, Sequence

from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot
from ..._internal.model_placement_controller import (
    ModelPlacementControllerError,
    ModelPhoneResidencyLayout,
)
from ..._internal.runtime_plan import RuntimeTransitionReceipt
from ..._internal.runtime_resources import RuntimeResourceError
from ..common import _RECOVERABLE_ERRORS, _RequestHelperPreparation
from ..helper_preparation_checks import _check_preparation_completion_receipts


def complete_request_helper_preparation(
    controller,
    request_id: str,
    preparation_ticket_id: str,
    receipts: Sequence[RuntimeTransitionReceipt],
    *,
    snapshot: HeterogeneousRuntimeSnapshot,
) -> Mapping[str, object]:
    preparation = controller._request_helper_preparations.get(
        preparation_ticket_id
    )
    rows = tuple(receipts)
    _check_preparation_completion_receipts(
        preparation,
        rows,
        request_id=request_id,
        preparation_ticket_id=preparation_ticket_id,
    )
    finished_at_us = max(row.finished_us for row in rows)
    if not isinstance(snapshot, HeterogeneousRuntimeSnapshot):
        raise UnifiedScheduleError(
            "request helper completion snapshot is invalid"
        )
    verification_observed_at_us = max(
        finished_at_us, snapshot.captured_at_us
    )
    snapshot.validate_at(verification_observed_at_us)
    helper = preparation.helper_envelope
    if (
        helper.artifact_sha256 != preparation.model.artifact_sha256
        or helper.operator_plan_sha256
            != preparation.operator_plan_sha256
        or helper.phone_layout_generation
            != preparation.phone_layout_generation
        or helper.phone_layout_geometry_sha256
            != preparation.phone_layout_geometry_sha256
    ):
        raise UnifiedScheduleError(
            "request helper preparation envelope differs"
        )
    state = (
        controller._model_placement_controller
        .preparing_phone_layout()
    )
    if (
        state is None
        or state.generation != preparation.phone_layout_generation
        or state.transition_ticket_id != preparation_ticket_id
        or state.layout.geometry_sha256
            != preparation.phone_layout_geometry_sha256
        or state.projection_token_sha256
            != preparation.projection_token_sha256
    ):
        return controller._ignore_stale_preparation_completion(
            preparation,
            request_id=request_id,
            preparation_ticket_id=preparation_ticket_id,
            finished_at_us=finished_at_us,
        )
    verification = (
        None
        if helper is None or state is None
        else controller._phone_layout_snapshot_verification(
            state,
            model_id=preparation.model.model_id,
            artifact_sha256=preparation.model.artifact_sha256,
            plan=helper.helper_plan,
            binding=helper.helper_binding,
            base_executor_id=preparation.base_executor_id,
            snapshot=snapshot,
            require_base_executor_ready=False,
        )
    )
    if verification is None:
        raise UnifiedScheduleError(
            "request helper preparation is not physically ready"
        )
    _workspace_bytes, verification_sha256 = verification
    try:
        with controller._transaction(errors=Exception, convert=False):
            return controller._complete_request_helper_preparation_transaction(
                request_id=request_id,
                preparation_ticket_id=preparation_ticket_id,
                preparation=preparation,
                rows=rows,
                snapshot=snapshot,
                finished_at_us=finished_at_us,
                verification_observed_at_us=(
                    verification_observed_at_us
                ),
                verification_sha256=verification_sha256,
            )
    except Exception as exc:
        if isinstance(
            exc,
            (UnifiedScheduleError, RuntimeResourceError,
             ModelPlacementControllerError),
        ):
            raise UnifiedScheduleError(str(exc)) from exc
        raise


def _complete_request_helper_preparation_transaction(
    controller,
    *,
    request_id: str,
    preparation_ticket_id: str,
    preparation: _RequestHelperPreparation,
    rows: tuple[RuntimeTransitionReceipt, ...],
    snapshot: HeterogeneousRuntimeSnapshot,
    finished_at_us: int,
    verification_observed_at_us: int,
    verification_sha256: str,
) -> Mapping[str, object]:
    if finished_at_us > preparation.ready_at_us:
        for token in preparation.resource_lease_tokens:
            controller.extend_lease(token, finished_at_us)
        preparation = replace(
            preparation, ready_at_us=finished_at_us
        )
    ready = (
        controller._model_placement_controller
        .complete_phone_layout_transition(
            generation=preparation.phone_layout_generation,
            ticket_id=preparation_ticket_id,
            transition_ids=preparation.transition_ids,
            geometry_sha256=(
                preparation.phone_layout_geometry_sha256
            ),
            projection_token_sha256=(
                preparation.projection_token_sha256
            ),
            finished_at_us=finished_at_us,
        )
    )
    if ready is None:
        stale = controller._mark_request_helper_preparation_stale(
            preparation, preparation_ticket_id, finished_at_us
        )
        return MappingProxyType(controller._helper_preparation_json(stale))
    for compiler in (
        controller._automated_route_compiler,
        controller._runtime_epoch_route_compiler,
    ):
        if compiler is not None:
            compiler.set_phone_residency_layout(ready.layout)
    controller._release_request_helper_preparation(
        preparation, finished_at_us
    )
    rematerialized = controller._rematerialize_committed_layout_helpers(
        request_id,
        ready,
        snapshot,
        verification_observed_at_us,
        preparation.phone_safety_state,
    )
    completed = replace(
        preparation,
        state="READY",
        transition_receipts=rows,
        verification_sha256=verification_sha256,
    )
    controller._request_helper_preparations[preparation_ticket_id] = completed
    controller._model_placement_controller.record_request_helper_event(
        request_id,
        "PREPARATION_READY",
        verification_observed_at_us,
        {
            **controller._helper_preparation_json(completed),
            "rematerialized_request_ids": list(rematerialized),
        },
    )
    controller._reevaluate_ready_layout_portfolio(
        request_id, snapshot, verification_observed_at_us
    )
    return MappingProxyType(controller._helper_preparation_json(completed))


def _mark_request_helper_preparation_stale(
    controller,
    preparation: _RequestHelperPreparation,
    preparation_ticket_id: str,
    finished_at_us: int,
) -> _RequestHelperPreparation:
    controller._release_request_helper_preparation(
        preparation, finished_at_us
    )
    stale = replace(preparation, state="STALE")
    controller._request_helper_preparations[
        preparation_ticket_id
    ] = stale
    return stale


def _ignore_stale_preparation_completion(
    controller,
    preparation: _RequestHelperPreparation,
    *,
    request_id: str,
    preparation_ticket_id: str,
    finished_at_us: int,
) -> Mapping[str, object]:
    """Release a preparation whose layout transition is no longer live."""

    try:
        with controller._transaction(errors=Exception, convert=False):
            stale = controller._mark_request_helper_preparation_stale(
                preparation, preparation_ticket_id, finished_at_us
            )
            (
                controller._model_placement_controller
                .record_request_helper_event(
                    request_id,
                    "STALE_PREPARATION_COMPLETION_IGNORED",
                    finished_at_us,
                    controller._helper_preparation_json(stale),
                )
            )
            return MappingProxyType(
                controller._helper_preparation_json(stale)
            )
    except Exception as exc:
        if isinstance(
            exc,
            (UnifiedScheduleError, RuntimeResourceError,
             ModelPlacementControllerError),
        ):
            raise UnifiedScheduleError(str(exc)) from exc
        raise


def _rematerialize_committed_layout_helpers(
    controller,
    request_id: str,
    ready: ModelPhoneResidencyLayout,
    snapshot: HeterogeneousRuntimeSnapshot,
    verification_observed_at_us: int,
    phone_safety_state,
) -> tuple[str, ...]:
    # The physical load is applied and verified; the layout is now
    # committed. Re-binding other requests' helpers is a post-commit
    # step that must never roll the commit back.
    try:
        with controller._transaction(convert=False):
            return controller._rematerialize_ready_layout_helpers(
                ready,
                snapshot,
                verification_observed_at_us,
                phone_safety_state,
            )
    except _RECOVERABLE_ERRORS as exc:
        controller._model_placement_controller.record_request_helper_event(
            request_id,
            "READY_LAYOUT_REMATERIALIZATION_FAILED",
            verification_observed_at_us,
            {
                "phone_layout_generation": ready.generation,
                "reason": str(exc),
            },
        )
        return ()


def _reevaluate_ready_layout_portfolio(
    controller,
    request_id: str,
    snapshot: HeterogeneousRuntimeSnapshot,
    verification_observed_at_us: int,
) -> None:
    try:
        tickets = tuple(
            row for row in controller._runtime_controller.current_tickets()
            if row.dispatch_state not in {"CANCELLED", "COMPLETED", "FAILED"}
        )
        if not tickets:
            return
        ticket = next(
            (row for row in tickets if row.request.request_id == request_id),
            tickets[0],
        )
        with controller._transaction(convert=False):
            controller._update_phone_residency_portfolio(
                ticket.request,
                controller.runtime_model_manifest(ticket.model.model_id),
                verification_observed_at_us,
                snapshot,
            )
    except _RECOVERABLE_ERRORS as exc:
        controller._model_placement_controller.record_request_helper_event(
            request_id,
            "READY_LAYOUT_REEVALUATION_FAILED",
            verification_observed_at_us,
            {"reason": str(exc)},
        )


def check_request_helper_preparation(
    controller,
    preparation_ticket_id: str,
    *,
    observed_at_us: int,
    guard_us: int = 250_000,
    quantum_us: int = 2_000_000,
) -> None:
    preparation = controller._request_helper_preparations.get(
        preparation_ticket_id
    )
    if preparation is None or preparation.state != "TRANSITIONING":
        raise UnifiedScheduleError(
            "request helper preparation is not active"
        )
    if observed_at_us + guard_us < preparation.ready_at_us:
        return
    with controller._transaction():
        extended_until_us = max(
            preparation.ready_at_us + quantum_us,
            observed_at_us + quantum_us,
        )
        for token in preparation.resource_lease_tokens:
            controller.extend_lease(token, extended_until_us)
        controller._request_helper_preparations[
            preparation_ticket_id
        ] = replace(
            preparation, ready_at_us=extended_until_us
        )
