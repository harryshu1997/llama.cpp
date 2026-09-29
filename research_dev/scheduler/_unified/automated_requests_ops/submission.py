"""AutomatedRequestMixin submission operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
import time
from types import MappingProxyType
from typing import Mapping

from ..._internal.policy import Request
from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot, RuntimeCapabilityError
from ..._internal.runtime_residency_projection import RuntimeResidencyProjectionError
from ..._internal.runtime_controller import (
    RuntimeControllerError,
    RuntimeReplanRetryRequired,
    RuntimeRequestTicket,
)
from ..._internal.types import canonical_sha256
from ..._internal.route_generation import candidate_set_to_runtime_costs


def submit_startup_parent_preload(
    controller, request, model_id, snapshot, *, executor_id,
    desktop_placement_sha256, observed_at_us,
):
    """Use the normal reservations and physical calibration checks for startup."""
    request.validate()
    if request.input_tokens != 1 or request.output_tokens != 2:
        raise UnifiedScheduleError("startup verification requires one input and two output tokens")
    if any(row.dispatch_state not in {"COMPLETED", "FAILED", "CANCELLED"}
           for row in controller._runtime_controller.current_tickets()):
        raise UnifiedScheduleError("startup preload requires an idle request controller")
    manifest = controller.runtime_model_manifest(model_id)
    with controller._transaction():
        candidates = controller._automated_compiler().generate(
            request, manifest, snapshot, observed_at_us=observed_at_us,
            desktop_parent=(executor_id, desktop_placement_sha256),
        )
        matching = [row for row in candidates.candidates
                    if row.binding.executor_id == executor_id
                    and row.plan.desktop_placement_sha256 == desktop_placement_sha256
                    and row.plan.execution_contract.phone_device_id is None
                    and row.plan.helper_envelope is None]
        if len(matching) != 1:
            raise UnifiedScheduleError("startup exact desktop parent is absent or ambiguous")
        selected = matching[0]
        if not controller._calibration_physically_qualified(selected):
            raise UnifiedScheduleError("startup parent physical evidence is incomplete")
        if controller._candidate_quarantine_reason(selected) is not None:
            raise UnifiedScheduleError("startup parent is quarantined")
        # Startup is a bounded physical verification, not an energy selection.
        allowed = {"ROUTE_NOT_QUALIFIED", "ENERGY_UNKNOWN", "COLD_RESIDENCY_BREAK_EVEN", "SLO_UPPER_BOUND"}
        unsafe = (set(selected.rejection_reasons)
                  | set(selected.binding.eligibility_reasons)) - allowed
        if unsafe:
            raise UnifiedScheduleError("startup parent is unsafe: " + ",".join(sorted(unsafe)))
        if any(row.evictions for row in selected.plan.transitions):
            raise UnifiedScheduleError("startup parent would evict existing residency")
        selected = replace(selected, admitted=True, rejection_reasons=(),
                           binding=replace(selected.binding, ready=True, eligibility_reasons=()))
        preview = controller._preview_automated_resources(selected, observed_at_us=observed_at_us)
        if preview.start_us != observed_at_us:
            raise UnifiedScheduleError("startup parent resources are busy")
        controller._preview_automated_memory(
            selected.plan, snapshot, start_us=preview.start_us,
            reserved_until_us=preview.finish_upper_us,
        )
        estimates = candidate_set_to_runtime_costs(
            candidates, request, manifest, snapshot, controller._runtime_capabilities,
            planning_profile_sha256=controller._runtime_capability_generation_sha256,
            model_manifest_sha256=controller._runtime_manifest_generation_sha256.get(model_id),
        )
        return controller._commit_automated_attempt(
            request=request, snapshot=snapshot, candidate_set=candidates,
            estimates=estimates, selected=selected, preview=preview,
            rejected=tuple((row.candidate_id, "STARTUP_EXACT_PARENT_ONLY")
                           for row in candidates.candidates if row.candidate_id != selected.candidate_id),
            reason="STARTUP_PARENT_PRELOAD", observed_at_us=observed_at_us,
            event_kind="DECISION", selection_mode="calibration",
        )


def submit_automated_request(
    controller,
    request: Request,
    model_id: str,
    snapshot: HeterogeneousRuntimeSnapshot,
    *,
    observed_at_us: int | None = None,
    selection_mode: str = "energy-aware",
) -> RuntimeRequestTicket:
    """Repair causal predecessors, then admit one observed arrival."""
    observed = (
        request.arrival_us
        if observed_at_us is None else observed_at_us
    )
    maximum_repairs = max(
        4,
        len(controller._runtime_controller.current_tickets()) * 3 + 3,
    )
    causal_not_before_by_resource: dict[str, int] = {}
    repair_signatures = set()
    for _ in range(maximum_repairs):
        try:
            return controller._submit_automated_request_once(
                request,
                model_id,
                snapshot,
                observed_at_us=observed_at_us,
                selection_mode=selection_mode,
                causal_not_before_by_resource=(
                    causal_not_before_by_resource
                ),
            )
        except UnifiedScheduleError as exc:
            cause = exc.__cause__
            if not isinstance(cause, RuntimeResidencyProjectionError):
                raise
            if cause.request_id is None or cause.ticket_id is None:
                raise
            repaired = controller._repair_stale_projection_chain(
                cause,
                snapshot,
                observed,
            )
            for resource_id, not_before_us in repaired.items():
                causal_not_before_by_resource[resource_id] = max(
                    causal_not_before_by_resource.get(resource_id, 0),
                    not_before_us,
                )
            try:
                repaired_ticket = controller._runtime_controller.ticket(
                    cause.request_id
                )
            except RuntimeControllerError as ticket_error:
                raise UnifiedScheduleError(str(ticket_error)) from (
                    ticket_error
                )
            repaired_plan_sha256 = (
                None
                if repaired_ticket.execution_plan is None
                else repaired_ticket.execution_plan.plan_sha256
            )
            diagnostic = {
                "barrier_by_resource": dict(sorted(repaired.items())),
                "plan_sha256": repaired_plan_sha256,
                "projection_error": str(cause),
                "request_id": cause.request_id,
                "runtime_snapshot_sha256": canonical_sha256(
                    snapshot.to_json()
                ),
                "ticket_id": cause.ticket_id,
            }
            repair_signature = canonical_sha256({
                key: value for key, value in diagnostic.items()
                if key != "ticket_id"
            })
            if repair_signature in repair_signatures:
                raise UnifiedScheduleError(
                    "runtime stale projection repair made no progress: "
                    + repair_signature
                    + " request_id=" + cause.request_id
                    + " ticket_id=" + cause.ticket_id
                    + " plan_sha256=" + str(repaired_plan_sha256)
                    + " runtime_snapshot_sha256="
                    + diagnostic["runtime_snapshot_sha256"]
                    + " projection_error=" + str(cause)
                    + " barrier_sha256=" + canonical_sha256(
                        diagnostic["barrier_by_resource"]
                    )
                ) from cause
            repair_signatures.add(repair_signature)
    raise UnifiedScheduleError(
        "runtime stale projection repair did not converge"
    )


def _repair_stale_projection_chain(
    controller,
    error: RuntimeResidencyProjectionError,
    snapshot: HeterogeneousRuntimeSnapshot,
    observed_at_us: int,
) -> Mapping[str, int]:
    """Defer stale projected work until its causal state is current."""
    if error.request_id is None or error.ticket_id is None:
        raise UnifiedScheduleError(str(error)) from error
    try:
        stale = controller._runtime_controller.ticket(error.request_id)
    except RuntimeControllerError as exc:
        raise UnifiedScheduleError(str(exc)) from exc
    if stale.ticket_id != error.ticket_id:
        return MappingProxyType({})
    if stale.execution_plan is None:
        raise UnifiedScheduleError(
            "stale residency predecessor has no execution plan"
        ) from error
    exclusive_by_device = controller._runtime_exclusive_memory_resources()
    causal_resources = {
        exclusive_by_device[device_id]
        for transition in stale.execution_plan.transitions
        for device_id in transition.prepares_device_ids
        if device_id in exclusive_by_device
    }

    def causal_barrier(
        predecessor: RuntimeRequestTicket = stale,
    ) -> Mapping[str, int]:
        barrier: dict[str, int] = {}
        for lease in predecessor.decision.leases:
            if lease.resource_id not in causal_resources:
                continue
            barrier[lease.resource_id] = max(
                barrier.get(lease.resource_id, 0),
                predecessor.final_reserved_until_us.get(
                    lease.token, lease.reserved_until_us
                ),
            )
        for ticket in controller._runtime_controller.current_tickets():
            if ticket.dispatch_state in {
                "CANCELLED", "COMPLETED", "FAILED"
            } or ticket.lease_status == "CANCELLED":
                continue
            plan = ticket.execution_plan
            if plan is None:
                continue
            for resource_id in causal_resources & set(
                plan.resource_ids
            ):
                barrier[resource_id] = max(
                    barrier.get(resource_id, 0),
                    max(
                        lease.reserved_until_us
                        for lease in ticket.decision.leases
                        if lease.resource_id == resource_id
                    ),
                )
        return MappingProxyType(dict(sorted(barrier.items())))

    if stale.dispatch_state == "ACQUIRED":
        return causal_barrier()
    if stale.dispatch_state in {"CANCELLED", "COMPLETED", "FAILED"}:
        return MappingProxyType({})
    if stale.dispatch_state == "REPLANNING":
        return causal_barrier()
    if stale.dispatch_state not in {"QUEUED", "REPLAN_REQUIRED"}:
        raise UnifiedScheduleError(
            "stale residency predecessor state is invalid"
        ) from error

    if stale.dispatch_state == "QUEUED":
        try:
            controller._runtime_controller.invalidate_queued_attempts(
                (stale.request.request_id,),
                "residency_projection_invalid",
                observed_at_us,
                cancel_owner=controller.cancel,
                release_memory=controller._runtime_memory.release_owner,
            )
        except RuntimeControllerError as exc:
            raise UnifiedScheduleError(str(exc)) from exc
    if stale.request.request_id not in (
        controller._runtime_controller.replan_required_requests()
    ):
        return causal_barrier()
    if stale.dispatch_state == "REPLAN_REQUIRED":
        wake = stale
    else:
        epoch_ns = max(
            0, time.monotonic_ns() - observed_at_us * 1_000
        )
        try:
            wake = controller._runtime_controller.wait(
                stale.request.request_id, epoch_ns
            )
        except RuntimeControllerError as exc:
            raise UnifiedScheduleError(str(exc)) from exc
    if (
        wake.dispatch_state != "REPLAN_REQUIRED"
        or wake.dispatch_receipt is None
    ):
        raise UnifiedScheduleError(
            "stale residency predecessor did not request replan"
        ) from error
    scheduling_observed_at_us = max(
        observed_at_us, snapshot.captured_at_us
    )
    try:
        snapshot.validate_at(scheduling_observed_at_us)
    except RuntimeCapabilityError as exc:
        if str(exc) != "system snapshot is stale":
            raise UnifiedScheduleError(str(exc)) from exc
        raise RuntimeReplanRetryRequired(
            stale.request.request_id,
            stale.ticket_id,
            "SYSTEM_SNAPSHOT_STALE",
        ) from exc
    try:
        replacement = controller.replan_automated_request(
            stale.request.request_id,
            observed_at_us=scheduling_observed_at_us,
            reason=wake.dispatch_receipt.wake_reason,
            snapshot=snapshot,
            expected_ticket_id=stale.ticket_id,
            expected_queue_generation=(
                wake.dispatch_receipt.queue_generation
            ),
        )
    except RuntimeReplanRetryRequired:
        raise
    return causal_barrier(replacement)
