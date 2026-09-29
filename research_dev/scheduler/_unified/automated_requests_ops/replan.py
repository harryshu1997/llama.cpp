"""AutomatedRequestMixin replan operations on its existing owner."""

from __future__ import annotations

import time

from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot, RuntimeCapabilityError
from ..._internal.runtime_resources import RuntimeResourceError, transition_adjusted_memory_demands
from ..._internal.runtime_controller import (
    RuntimeControllerError,
    RuntimeReplanRetryRequired,
    RuntimeRequestTicket,
)
from ..common import _text
from .affinity import AFFINITY_DISPLACEMENT_REASON
from .continuous_join import BARRIER_DISPLACEMENT_REASON
from .common import _AutomatedReplanPreparation, _AutomatedSubmitContext


def replan_automated_request(
    controller,
    request_id: str,
    *,
    observed_at_us: int,
    reason: str,
    snapshot: HeterogeneousRuntimeSnapshot,
    expected_ticket_id: str | None = None,
    expected_queue_generation: int | None = None,
) -> RuntimeRequestTicket:
    """Replace one attempt, compacting later conflicting reservations."""

    if reason not in {"capacity_released_early", "preparation_phase_completed"}:
        return controller._replan_automated_request_once(
            request_id,
            observed_at_us=observed_at_us,
            reason=reason,
            snapshot=snapshot,
            expected_ticket_id=expected_ticket_id,
            expected_queue_generation=expected_queue_generation,
        )
    root = controller.runtime_ticket(request_id)
    try:
        live_followers = (
            controller._runtime_controller.priority_compaction_followers(
                request_id, expected_queue_generation
            )
        )
    except RuntimeControllerError:
        return controller._replan_automated_request_once(
            request_id,
            observed_at_us=observed_at_us,
            reason=reason,
            snapshot=snapshot,
            expected_ticket_id=expected_ticket_id,
            expected_queue_generation=expected_queue_generation,
        )
    def rollback_if(exc: BaseException) -> bool:
        return not (
            isinstance(exc, UnifiedScheduleError)
            and controller.runtime_ticket(request_id).dispatch_state == "FAILED"
        )

    try:
        with controller._transaction(
            errors=Exception,
            convert=False,
            rollback_if=rollback_if,
        ):
            if not live_followers:
                return controller._replan_priority_compaction_without_followers(
                    request_id,
                    observed_at_us=observed_at_us,
                    reason=reason,
                    snapshot=snapshot,
                    expected_ticket_id=expected_ticket_id,
                    expected_queue_generation=expected_queue_generation,
                )
            return controller._replan_priority_compaction_with_followers(
                request_id,
                observed_at_us=observed_at_us,
                reason=reason,
                snapshot=snapshot,
                expected_ticket_id=expected_ticket_id,
                expected_queue_generation=expected_queue_generation,
                root=root,
            )
    except Exception as exc:
        if not rollback_if(exc):
            raise
        if isinstance(exc, RuntimeReplanRetryRequired):
            raise
        raise RuntimeReplanRetryRequired(
            request_id,
            root.ticket_id,
            "PRIORITY_COMPACTION_ROLLED_BACK",
        ) from exc


def _replan_priority_compaction_without_followers(
    controller,
    request_id: str,
    *,
    observed_at_us: int,
    reason: str,
    snapshot: HeterogeneousRuntimeSnapshot,
    expected_ticket_id: str | None,
    expected_queue_generation: int | None,
) -> RuntimeRequestTicket:
    with controller._runtime_controller.defer_dispatch_wake():
        replan_required_before = frozenset(
            controller._runtime_controller.replan_required_requests()
        )
        replacement = controller._replan_automated_request_once(
            request_id,
            observed_at_us=observed_at_us,
            reason=reason,
            snapshot=snapshot,
            expected_ticket_id=expected_ticket_id,
            expected_queue_generation=expected_queue_generation,
        )
        newly_released = tuple(
            follower_id
            for follower_id in (
                controller._runtime_controller.replan_required_requests()
            )
            if follower_id not in replan_required_before
            and follower_id != request_id
        )
        for follower_id in newly_released:
            follower = controller.runtime_ticket(follower_id)
            follower_observed_at_us = max(
                observed_at_us, follower.request.arrival_us
            )
            if follower.dispatch_state == "QUEUED":
                follower = (
                    controller._runtime_controller
                    .promote_priority_compaction_follower(
                        follower_id,
                        follower.ticket_id,
                        follower_observed_at_us,
                        controller.cancel,
                        controller._runtime_memory.release_owner,
                        controller._runtime_memory.owner_tokens,
                    )
                )
            elif follower.dispatch_state != "REPLAN_REQUIRED":
                raise UnifiedScheduleError(
                    "priority compaction follower state changed"
                )
            elif (
                controller._runtime_controller.dispatch_policy
                .work_conserving_admission
            ):
                follower = controller._runtime_controller.refresh_replan_receipt(
                    follower_id, follower.ticket_id, follower_observed_at_us
                )
            assert follower.dispatch_receipt is not None
            controller._replan_automated_request_once(
                follower_id,
                observed_at_us=follower_observed_at_us,
                reason="priority_compaction_follower",
                snapshot=snapshot,
                expected_ticket_id=follower.ticket_id,
                expected_queue_generation=(
                    follower.dispatch_receipt.queue_generation
                ),
                priority_compaction_active=True,
            )
        return replacement


def _replan_priority_compaction_with_followers(
    controller,
    request_id: str,
    *,
    observed_at_us: int,
    reason: str,
    snapshot: HeterogeneousRuntimeSnapshot,
    expected_ticket_id: str | None,
    expected_queue_generation: int | None,
    root: RuntimeRequestTicket,
) -> RuntimeRequestTicket:
    with controller._runtime_controller.defer_dispatch_wake():
        followers = (
            controller._runtime_controller.prepare_priority_compaction_followers(
                request_id,
                observed_at_us,
                controller.cancel,
                controller._runtime_memory.release_owner,
                controller._runtime_memory.owner_tokens,
                expected_queue_generation=expected_queue_generation,
            )
        )
        replacement = controller._replan_automated_request_once(
            request_id,
            observed_at_us=observed_at_us,
            reason=reason,
            snapshot=snapshot,
            expected_ticket_id=expected_ticket_id,
            expected_queue_generation=expected_queue_generation,
            priority_compaction_active=True,
        )
        if replacement.dispatch_state == "REPLAN_REQUIRED":
            # Projection deferral leaves followers behind the same root.
            return replacement
        released_frontier = (
            controller._runtime_controller.release_replanned_capacity_frontier(
                request_id,
                root.decision,
                observed_at_us,
                resource_capacities={
                    resource_id: resource.capacity
                    for resource_id, resource in (
                        controller._runtime_capabilities.resources.items()
                    )
                },
                cancel_owner=controller.cancel,
                release_memory=controller._runtime_memory.release_owner,
                memory_owner_tokens=controller._runtime_memory.owner_tokens,
            )
        )
        ticket_id_by_request = dict(followers)
        for follower_id in released_frontier:
            ticket_id_by_request.setdefault(
                follower_id,
                controller.runtime_ticket(follower_id).ticket_id,
            )
        follower_order = controller._runtime_controller.priority_compaction_order(
            tuple(ticket_id_by_request)
        )
        for follower_id in follower_order:
            follower_ticket_id = ticket_id_by_request[follower_id]
            follower_observed_at_us = max(
                observed_at_us,
                controller.runtime_ticket(follower_id).request.arrival_us,
            )
            follower = controller.runtime_ticket(follower_id)
            if follower.dispatch_state == "QUEUED":
                follower = (
                    controller._runtime_controller
                    .promote_priority_compaction_follower(
                        follower_id,
                        follower_ticket_id,
                        follower_observed_at_us,
                        controller.cancel,
                        controller._runtime_memory.release_owner,
                        controller._runtime_memory.owner_tokens,
                    )
                )
            elif (
                follower.dispatch_state != "REPLAN_REQUIRED"
                or follower.ticket_id != follower_ticket_id
            ):
                raise UnifiedScheduleError(
                    "priority compaction follower state changed"
                )
            elif (
                controller._runtime_controller.dispatch_policy
                .work_conserving_admission
            ):
                follower = controller._runtime_controller.refresh_replan_receipt(
                    follower_id, follower_ticket_id, follower_observed_at_us
                )
            assert follower.dispatch_receipt is not None
            controller._replan_automated_request_once(
                follower_id,
                observed_at_us=follower_observed_at_us,
                reason="priority_compaction_follower",
                snapshot=snapshot,
                expected_ticket_id=follower_ticket_id,
                expected_queue_generation=(
                    follower.dispatch_receipt.queue_generation
                ),
                priority_compaction_active=True,
            )
        return replacement


def _validated_replan_ticket(
    controller,
    request_id: str,
    snapshot: HeterogeneousRuntimeSnapshot,
    observed_at_us: int,
    expected_ticket_id: str | None,
    expected_queue_generation: int | None,
) -> tuple[RuntimeRequestTicket, int | None, bool]:
    current = controller.runtime_ticket(request_id)
    if expected_ticket_id is not None:
        expected_ticket_id = _text(
            "runtime expected ticket id", expected_ticket_id
        )
        prefix = request_id + ":attempt:"
        suffix = expected_ticket_id.removeprefix(prefix)
        if not expected_ticket_id.startswith(prefix) or not suffix.isdigit():
            raise UnifiedScheduleError(
                "runtime expected ticket identity differs"
            )
        expected_attempt = int(suffix)
        if current.ticket_id != expected_ticket_id:
            if current.attempt_index <= expected_attempt:
                raise UnifiedScheduleError(
                    "runtime expected ticket was not superseded"
                )
            return current, expected_queue_generation, True
    if current.dispatch_state in {"CANCELLED", "COMPLETED", "FAILED"}:
        raise UnifiedScheduleError(
            "runtime terminal ticket cannot be replanned"
        )
    if current.dispatch_state != "REPLAN_REQUIRED":
        if (
            expected_ticket_id is not None
            and current.ticket_id == expected_ticket_id
            and current.dispatch_state == "QUEUED"
        ):
            raise RuntimeReplanRetryRequired(
                request_id, current.ticket_id, "REPLAN_WAKE_ROLLED_BACK"
            )
        raise UnifiedScheduleError("runtime request does not need replan")
    if current.execution_plan is None:
        raise UnifiedScheduleError(
            "runtime ticket was not created by automated scheduling"
        )
    if expected_queue_generation is None:
        expected_queue_generation = (
            None
            if current.dispatch_receipt is None
            else current.dispatch_receipt.queue_generation
        )
    elif (
        type(expected_queue_generation) is not int
        or expected_queue_generation < 0
    ):
        raise UnifiedScheduleError(
            "runtime expected queue generation is invalid"
        )
    try:
        snapshot.validate_at(observed_at_us)
    except RuntimeCapabilityError as exc:
        if str(exc) != "system snapshot is stale":
            raise UnifiedScheduleError(str(exc)) from exc
        raise RuntimeReplanRetryRequired(
            request_id, current.ticket_id, "SYSTEM_SNAPSHOT_STALE"
        ) from exc
    return current, expected_queue_generation, False


def _prepare_automated_replan(
    controller,
    *,
    current: RuntimeRequestTicket,
    snapshot: HeterogeneousRuntimeSnapshot,
    manifest,
    observed_at_us: int,
    reason: str,
    expected_queue_generation: int | None,
) -> _AutomatedReplanPreparation:
    controller._model_placement_controller.notify(
        manifest.artifact_sha256,
        "REQUEST_REPLAN_REQUIRED",
        observed_at_us,
    )
    projection_request_ids = (
        controller._runtime_controller.replan_projection_request_ids(
            current.request.request_id, expected_queue_generation
        )
    )
    projection_before_us = max(
        (current.decision.start_us,)
        + tuple(
            max(lease.reserved_until_us for lease in ticket.decision.leases)
            for ticket in controller._runtime_controller.current_tickets()
            if ticket.request.request_id in projection_request_ids
            and ticket.decision.leases
        )
    )
    timings = {
        key: 0
        for key in (
            "projection", "generation", "conversion", "admission",
            "selection", "preview", "epoch_lookup", "epoch_validation",
            "epoch_preprojection",
        )
    }
    started_ns = time.perf_counter_ns()
    observed_snapshot = snapshot
    snapshot = controller._automated_snapshot_for_request(
        current.request,
        snapshot,
        exclude_request_id=current.request.request_id,
        project_before_us=projection_before_us,
        project_request_ids=projection_request_ids,
    )
    timings["projection"] += time.perf_counter_ns() - started_ns
    started_ns = time.perf_counter_ns()
    placement_epoch, route_templates, invalidation_reason = (
        controller._published_epoch_for_request(
            current.request,
            manifest,
            observed_at_us,
            current.selection_mode,
        )
    )
    current_epoch = placement_epoch
    timings["epoch_lookup"] += time.perf_counter_ns() - started_ns
    demand_snapshot, placement_action = controller._evaluate_model_placement(
        current.request,
        manifest,
        snapshot,
        observed_at_us,
        placement_epoch,
        current.selection_mode,
    )
    if placement_action.kind in {"RECOMPUTE_NOW", "FALLBACK"}:
        if placement_epoch is not None:
            invalidation_reason = "+".join(
                placement_action.trigger_reasons
            )
            controller._record_epoch_invalidation(invalidation_reason)
        placement_epoch = None
        route_templates = None
    residency_holds = controller._runtime_residency_cohorts.holds(
        controller._runtime_capabilities,
        snapshot,
        controller._runtime_controller.current_tickets(),
        observed_at_us,
    )
    planned = set(
        controller._runtime_residency_cohorts.planned_component_identity_sha256s(
            current.request.request_id
        )
    )
    hot_components = frozenset(
        planned
        & {
            hold.component_identity_sha256
            for hold in residency_holds.values()
        }
    )
    transition_changed = controller._replan_transition_projection_changed(
        current, snapshot
    )
    prioritize = bool(current.execution_plan.transitions and hot_components)
    controller._detach_decode_cohort_for_replan(current.request.request_id)
    previous = controller._runtime_controller.prepare_replan(
        current.request.request_id,
        observed_at_us,
        controller.cancel,
        controller._runtime_memory.release_owner,
        invalidate_dependents=(
            reason not in {
                "capacity_released_early",
                "lease_coverage_expired_before_dispatch",
                "predecessor_replan",
                "priority_compaction_follower",
                # Work queued behind a displaced attempt was displaced with
                # it; the later-sequence work ahead of it must not be deferred.
                AFFINITY_DISPLACEMENT_REASON,
                BARRIER_DISPLACEMENT_REASON,
            }
            and (prioritize or transition_changed)
        ),
        expected_ticket_id=current.ticket_id,
        expected_queue_generation=expected_queue_generation,
    )
    context = _AutomatedSubmitContext(
        source_snapshot=snapshot,
        snapshot=snapshot,
        placement_epoch=placement_epoch,
        current_epoch_for_publication=current_epoch,
        route_templates=route_templates,
        demand_snapshot=demand_snapshot,
        placement_action=placement_action,
        epoch_invalidation_reason=invalidation_reason,
        live_not_before_by_resource={},
        timings_ns=timings,
    )
    return _AutomatedReplanPreparation(
        context,
        previous,
        residency_holds,
        prioritize,
        hot_components,
        observed_snapshot=observed_snapshot,
        projection_request_ids=tuple(projection_request_ids),
    )


def _replan_transition_projection_changed(
    controller,
    current: RuntimeRequestTicket,
    snapshot: HeterogeneousRuntimeSnapshot,
) -> bool:
    if not current.execution_plan.transitions:
        return False
    try:
        transition_adjusted_memory_demands(
            current.execution_plan.memory_demands,
            transitions=current.execution_plan.transitions,
            residency=snapshot.residency,
            exclusive_resource_by_device=(
                controller._runtime_exclusive_memory_resources()
            ),
        )
    except RuntimeResourceError as exc:
        if not str(exc).startswith("transition eviction is stale:"):
            raise
        return True
    return False
