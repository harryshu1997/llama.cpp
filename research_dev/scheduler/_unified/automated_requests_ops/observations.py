"""AutomatedRequestMixin observations operations on its existing owner."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Sequence

from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot
from ..._internal.model_placement_controller import ModelPlacementControllerError
from ..._internal.runtime_plan import RuntimeTransitionReceipt
from ..._internal.runtime_residency_projection import (
    RuntimeResidencyProjectionError,
    project_scheduler_residency,
    transition_target_is_observed,
)
from ..._internal.runtime_controller import RuntimeRequestTicket
from .affinity import model_affinity_replan_displacement
from .join_publication import wake_server_live_joiners


def _transitions_are_realized(
    snapshot: HeterogeneousRuntimeSnapshot,
    ticket: RuntimeRequestTicket,
) -> bool:
    """Whether the live state already is the target of every transition.

    The target executor is published and holds this model on every prepared
    device, and every residency the plan would evict is already gone.
    """
    plan = ticket.execution_plan
    if plan is None or not plan.transitions:
        return False
    participants = {
        row.device_id: row.executor_id for row in ticket.binding.participants
    }
    for transition in plan.transitions:
        if transition.phone_shards:
            return False
        executor_id = transition.executor_id or participants.get(
            transition.device_id, ticket.binding.executor_id
        )
        executor = snapshot.executors.get(executor_id)
        if executor is None or not executor.ready:
            return False
        for device_id in transition.prepares_device_ids:
            residency = snapshot.residency_for(
                ticket.model.model_id, ticket.model.artifact_sha256, device_id
            )
            if (
                residency is None
                or residency.state != transition.target_state
                or residency.executor_id != executor_id
            ):
                return False
        for eviction in transition.evictions:
            current = snapshot.residency_for(
                eviction.model_id, eviction.artifact_sha256, eviction.device_id
            )
            if (
                current is not None
                and current.state in {"hot", "warm"}
                and current.generation == eviction.generation
                and current.executor_id == eviction.executor_id
            ):
                return False
    return True


def observe_automated_runtime_snapshot(
    controller,
    snapshot: HeterogeneousRuntimeSnapshot,
    *,
    observed_at_us: int,
) -> tuple[str, ...]:
    """Wake queued attempts whose selected residency transition is obsolete."""
    if not isinstance(snapshot, HeterogeneousRuntimeSnapshot):
        raise UnifiedScheduleError("runtime system snapshot is invalid")
    if type(observed_at_us) is not int or observed_at_us < 0:
        raise UnifiedScheduleError("automated observation time is invalid")
    with controller._transaction():
        snapshot.validate_at(observed_at_us)
        changed = []
        exclusive_devices = {
            device_id
            for device_id, capability in (
                controller._runtime_capabilities.executor_by_device.items()
            )
            if capability.exclusive_residency_resource_id is not None
        }
        maximum_invalidations = len(
            controller._runtime_controller.current_tickets()
        )
        publication_wakes = (
            controller._runtime_controller.dispatch_policy
            .work_conserving_admission
        )
        affinity_wakes = (
            controller._runtime_controller.dispatch_policy.model_affinity
        )
        for _ in range(maximum_invalidations + 1):
            ticket_by_request = {
                row.request.request_id: row
                for row in controller._runtime_controller.current_tickets()
            }
            projection_request_ids = (
                controller._runtime_controller.projection_request_ids()
            )
            causal_predecessors = (
                controller._runtime_controller
                .projection_causal_predecessors()
            )
            authoritative = tuple(
                ticket_by_request[request_id]
                for request_id in projection_request_ids
                if request_id in ticket_by_request
            )
            queued = tuple(
                row for row in authoritative
                if row.dispatch_state == "QUEUED"
            )
            queued_by_request = {
                row.request.request_id: row for row in queued
            }
            invalidation = None
            publication = False
            for ticket in queued:
                plan = ticket.execution_plan
                if plan is None:
                    continue
                if (
                    publication_wakes
                    and all(
                        predecessor_id in changed
                        or (
                            ticket_by_request.get(predecessor_id) is not None
                            and ticket_by_request[predecessor_id]
                                .dispatch_state == "ACQUIRED"
                        )
                        for predecessor_id in causal_predecessors.get(
                            ticket.request.request_id, ()
                        )
                    )
                    and _transitions_are_realized(snapshot, ticket)
                ) or (
                    # It still waits on another model's residency change that
                    # its replan may displace.
                    affinity_wakes
                    and _transitions_are_realized(snapshot, ticket)
                    and model_affinity_replan_displacement(
                        controller,
                        snapshot=snapshot,
                        current=ticket,
                        observed_at_us=observed_at_us,
                        reason="residency_observation_changed",
                        record_refusal=False,
                    ) is not None
                ):
                    # Running work already produced this attempt's residency:
                    # replan it now to join that executor.
                    invalidation = (
                        ticket.request.request_id,
                        "residency_observation_changed",
                    )
                    publication = True
                    break
                projected = snapshot
                if any(
                    device_id in exclusive_devices
                    for transition in plan.transitions
                    for device_id in transition.prepares_device_ids
                ):
                    try:
                        projected = project_scheduler_residency(
                            snapshot,
                            controller._runtime_capabilities,
                            authoritative,
                            controller._runtime_manifests,
                            stop_before_ticket_id=ticket.ticket_id,
                            causal_predecessors=causal_predecessors,
                        )
                    except RuntimeResidencyProjectionError as exc:
                        stale = queued_by_request.get(exc.request_id)
                        if stale is not None:
                            invalidation = (
                                stale.request.request_id,
                                "residency_projection_invalid",
                            )
                            break
                        continue
                transition_is_obsolete = bool(plan.transitions) and all(
                    transition_target_is_observed(
                        projected, ticket, transition
                    )
                    for transition in plan.transitions
                )
                prepared_device_ids = {
                    device_id
                    for transition in plan.transitions
                    for device_id in transition.prepares_device_ids
                }
                resident_weights_are_stale = any(
                    demand.kind == "model_weights"
                    and demand.resident_bytes > 0
                    and device_id not in prepared_device_ids
                    and projected.residency_for(
                        ticket.model.model_id,
                        ticket.model.artifact_sha256,
                        device_id,
                    ) is None
                    for demand in plan.memory_demands
                    for device_id in plan.device_ids
                    if demand.demand_id == "weights:" + device_id
                )
                if transition_is_obsolete or resident_weights_are_stale:
                    invalidation = (
                        ticket.request.request_id,
                        "residency_observation_changed",
                    )
                    break
            if invalidation is None:
                break
            request_id, reason = invalidation
            if publication:
                # Its predecessors run or replan now as well, so it does not
                # wait behind them.
                invalidated = controller._runtime_controller.replan_queued_now(
                    (request_id,),
                    reason,
                    observed_at_us,
                    cancel_owner=controller.cancel,
                    release_memory=controller._runtime_memory.release_owner,
                )
                controller._runtime_controller.record_dispatch_policy_event(
                    "publication_replans", len(invalidated)
                )
            else:
                invalidated = (
                    controller._runtime_controller.invalidate_queued_attempts(
                        (request_id,),
                        reason,
                        observed_at_us,
                        cancel_owner=controller.cancel,
                        release_memory=controller._runtime_memory.release_owner,
                    )
                )
            if not invalidated:
                break
            changed.extend(invalidated)
        else:
            raise UnifiedScheduleError(
                "residency observation invalidation did not converge"
            )
        # A published server lets the work that arrived during its load join it.
        changed.extend(
            wake_server_live_joiners(controller, snapshot, observed_at_us)
        )
        return tuple(changed)


def runtime_memory_state(controller) -> Mapping[str, object]:
    return MappingProxyType(controller._runtime_memory.snapshot())


def record_automated_transition_receipts(
    controller,
    request_id: str,
    receipts: Sequence[RuntimeTransitionReceipt],
) -> RuntimeRequestTicket:
    """Bind physical transition receipts to the selected attempt."""
    with controller._transaction():
        rows = tuple(receipts)
        if any(row.status != "COMPLETED" for row in rows):
            raise UnifiedScheduleError(
                "failed transitions require fail_automated_request"
            )
        previous = controller.runtime_ticket(request_id).completed_prepare_lease_tokens
        ticket = controller._runtime_controller.record_transition_receipts(
            request_id, rows
        )
        ticket = controller._runtime_controller.finish_prepare_phase(
            request_id, retime_leases=controller.timeline.retime_many,
            release_memory=controller._runtime_memory.release_owner,
            previously_completed_tokens=previous,
        )
        phone_layout = (
            controller._model_placement_controller
            .preparing_phone_layout()
        )
        if (
            phone_layout is not None
            and phone_layout.transition_ticket_id == ticket.ticket_id
            and set(phone_layout.transition_ids).issubset(row.transition_id for row in rows)
        ):
            phone_receipts = tuple(
                row for row in rows
                if row.transition_id in phone_layout.transition_ids
            )
            if (
                len(phone_receipts)
                    != len(phone_layout.transition_ids)
                or any(row.status != "COMPLETED" for row in phone_receipts)
            ):
                raise UnifiedScheduleError(
                    "phone layout transition receipt is incomplete"
                )
            try:
                ready = (
                    controller._model_placement_controller
                    .complete_phone_layout_transition(
                        generation=phone_layout.generation,
                        ticket_id=ticket.ticket_id,
                        transition_ids=tuple(
                            row.transition_id
                            for row in phone_receipts
                        ),
                        geometry_sha256=(
                            phone_layout.layout.geometry_sha256
                        ),
                        projection_token_sha256=str(
                            phone_layout.projection_token_sha256
                        ),
                        finished_at_us=max(
                            row.finished_us for row in phone_receipts
                        ),
                    )
                )
            except ModelPlacementControllerError as exc:
                raise UnifiedScheduleError(str(exc)) from exc
            if ready is not None:
                for compiler in (
                    controller._automated_route_compiler,
                    controller._runtime_epoch_route_compiler,
                ):
                    if compiler is not None:
                        compiler.set_phone_residency_layout(ready.layout)
        if ticket.transition_status == "COMPLETED" and not controller._runtime_decision_log.has_acquired(
            ticket.ticket_id
        ):
            controller._model_placement_controller.mark_request_acquired(
                request_id
            )
            controller._append_runtime_log(
                "ACQUIRED",
                ticket,
                max(row.finished_us for row in rows),
                ticket.dispatch_state,
            )
        return ticket


def runtime_execution_ticket(
    controller, request_id: str
) -> RuntimeRequestTicket:
    """Return the exact physical contract only when it is executable."""
    ticket = controller.runtime_ticket(request_id)
    if ticket.dispatch_state != "ACQUIRED":
        raise UnifiedScheduleError(
            "runtime execution resources are not acquired"
        )
    if ticket.transition_status not in {"COMPLETED", "NOT_REQUIRED"}:
        raise UnifiedScheduleError(
            "runtime execution transitions are not complete"
        )
    return ticket
