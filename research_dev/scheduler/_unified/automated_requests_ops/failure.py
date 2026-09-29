"""AutomatedRequestMixin failure operations on its existing owner."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Sequence

from ..._internal.policy import SchedulerError
from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot, RuntimeCapabilityError
from ..._internal.route_generation import RouteGenerationError, candidate_set_to_runtime_costs
from ..._internal.runtime_plan import RuntimeTransitionReceipt
from ..._internal.runtime_resources import RuntimeResourceError
from ..._internal.runtime_residency_projection import RuntimeResidencyProjectionError
from ..._internal.runtime_controller import (
    RuntimeFailureRecovery,
    RuntimeReplanRetryRequired,
    RuntimeRequestTicket,
)
from ..._internal.runtime_execution import (
    ELASTIC_FAILURE_PHASES,
    RuntimeExecutionFailure,
)


def _fail_runtime_ticket(
    controller,
    *,
    current: RuntimeRequestTicket,
    manifest,
    failure: RuntimeExecutionFailure,
    failed_at_us: int,
    reason: str,
    transition_receipts: Sequence[RuntimeTransitionReceipt],
) -> tuple:
    request_id = current.request.request_id
    controller._model_placement_controller.notify(
        manifest.artifact_sha256, "REQUEST_FAILED", failed_at_us
    )
    if transition_receipts:
        transition_ticket = controller._runtime_controller.record_transition_receipts(
            request_id, transition_receipts
        )
        if transition_ticket.transition_status != "FAILED":
            raise UnifiedScheduleError(
                "transition failure lacks a failed receipt"
            )
    controller._fail_phone_residency_transition(
        current.ticket_id, failed_at_us, reason
    )
    release_shared, cohort_owner_id = (
        controller._runtime_decode_cohorts.mark_terminal(request_id)
    )
    if cohort_owner_id is not None and not release_shared:
        controller._handoff_decode_cohort_singleton(cohort_owner_id)
    lease_owner_id = (
        None
        if cohort_owner_id is None
        else controller._runtime_decode_cohorts.lease_owner_id(cohort_owner_id)
    )

    def cancel_owner(owner_id: str, at_us: int) -> tuple[str, ...]:
        if owner_id == request_id and cohort_owner_id is not None:
            if not release_shared:
                return ()
            assert lease_owner_id is not None
            return controller.cancel(lease_owner_id, at_us)
        return controller.cancel(owner_id, at_us)

    result = controller._runtime_controller.fail(
        request_id,
        failed_at_us,
        reason,
        cancel_owner=cancel_owner,
        release_memory=controller._runtime_memory.release_owner,
        failed_resource_ids=failure.failed_resource_ids,
    )
    controller._close_request_helper_runtime(
        request_id, failed_at_us, "REQUEST_FAILED"
    )
    return result


def _terminal_failure_recovery(
    controller,
    *,
    current: RuntimeRequestTicket,
    manifest,
    failed,
    cancelled: tuple[str, ...],
    quarantine_action,
    failed_at_us: int,
    reason: str,
    notify_reason: str,
    recover_adaptive: bool,
) -> RuntimeFailureRecovery:
    request_id = current.request.request_id
    if recover_adaptive and (
        current.execution_plan.execution_contract.execution_mode
        == "adaptive-split"
    ):
        controller._adaptive_decode.recover_for_restart(request_id, reason)
    controller._runtime_residency_cohorts.record_terminal(request_id)
    controller._model_placement_controller.release_request(request_id)
    controller._model_placement_controller.notify(
        manifest.artifact_sha256, notify_reason, failed_at_us
    )
    controller._append_runtime_log(
        "FAILED", failed, failed_at_us, failed.dispatch_state
    )
    return RuntimeFailureRecovery(
        failed_ticket_id=failed.ticket_id,
        reason=reason,
        cancelled_tokens=cancelled,
        quarantine_action=quarantine_action,
        fallback=None,
    )


def _record_recovery_unavailable(controller, request_id: str, error: BaseException) -> None:
    """Keep why an elastic failure found no recovery (``recovery_unavailable_reason``)."""
    reasons = getattr(controller, "_elastic_recovery_unavailable", None)
    if reasons is None:
        reasons = {}
        controller._elastic_recovery_unavailable = reasons
    detail = " ".join(str(error).split())[:512]
    reasons[request_id] = type(error).__name__ + ": " + (detail or "unspecified")


def _record_elastic_attempt(controller, request_id: str, ticket_id: str) -> None:
    """Remember an attempt that failed with helper_lost / server_exited (elastic phones)."""
    attempts = getattr(controller, "_elastic_failed_attempts", None)
    if attempts is None:
        attempts = {}
        controller._elastic_failed_attempts = attempts
    attempts.setdefault(request_id, set()).add(ticket_id)


def elastic_failed_attempts(controller, request_id: str) -> frozenset[str]:
    """Tickets of a request whose attempts failed with helper_lost / server_exited."""
    return frozenset(getattr(controller, "_elastic_failed_attempts", {}).get(request_id, ()))


def recovery_unavailable_reason(controller, request_id: str) -> str | None:
    """Why the last helper_lost / server_exited failure of a request found no recovery."""
    return getattr(controller, "_elastic_recovery_unavailable", {}).get(request_id)


def _require_recovery_without_failed_devices(
    selected, failure: RuntimeExecutionFailure
) -> None:
    """A helper_lost recovery never runs on the device it just lost.

    The elastic-phones device quarantine (slice 2) excludes the device from
    every later candidate; this check holds even before that quarantine
    exists, and is a no-op for every failure without lost devices.
    """
    lost = set(failure.failed_device_ids)
    if lost and lost & set(selected.plan.device_ids):
        raise UnifiedScheduleError(
            "recovery route uses the lost helper device"
        )


def _select_elastic_desktop_recovery(
    controller,
    candidate_set,
    *,
    failed_route_id: str,
    runtime_rejections,
    selection_mode: str,
    exited_executor_id: str | None,
    masked_executor_id: str | None = None,
) -> tuple:
    """helper_lost / server_exited of a non-adaptive ticket: the paired desktop baseline first.

    The device (or the server process) failed, not the desktop route: the exact desktop control
    route re-executes the request from its prompt, through a load when the rig reaped or retired
    its server (the reaped route is cold, so it differs from the failed hot route). The qualified
    recovery fallback (a CPU whole-model route) remains the last resort, as for every other
    failure. Both refusals are kept in the error. When the failed attempt's server has exited
    (``exited_executor_id``), the failed route may be the target: a co-tenant's recovery may
    already have relaunched it, and that hot residency is a new server. When the live server
    masked the lost helper out (``masked_executor_id``, elastic phones S2a), the failed route is
    the target on that same server, without a reload.
    """
    try:
        selected, rejected, reason = controller._select_adaptive_desktop_recovery(
            candidate_set,
            failed_route_id=failed_route_id,
            runtime_rejections=runtime_rejections,
            **({} if exited_executor_id is None else {"exited_executor_id": exited_executor_id}),
            **({} if masked_executor_id is None else {"masked_executor_id": masked_executor_id}),
        )
        return selected, rejected, reason, "desktop-baseline"
    except UnifiedScheduleError as baseline_error:
        try:
            selected, rejected, reason = controller._select_automated_recovery_fallback(
                candidate_set,
                failed_route_id=failed_route_id,
                runtime_rejections=runtime_rejections,
            )
        except UnifiedScheduleError as fallback_error:
            raise UnifiedScheduleError(
                str(baseline_error) + "; " + str(fallback_error)
            ) from fallback_error
        return selected, rejected, reason, selection_mode


def _prepare_automated_failure_fallback(
    controller,
    *,
    current: RuntimeRequestTicket,
    manifest,
    snapshot: HeterogeneousRuntimeSnapshot,
    failed,
    failed_at_us: int,
    elastic: bool = False,
    exited_executor_id: str | None = None,
    masked_executor_id: str | None = None,
) -> tuple:
    snapshot = controller._automated_snapshot_for_request(
        current.request, snapshot, project_before_us=failed_at_us
    )
    candidate_set = controller._generate_automated_candidate_set(
        current.request,
        manifest,
        snapshot,
        failed_at_us,
        use_residency_holds=False,
    )
    estimates = candidate_set_to_runtime_costs(
        candidate_set,
        current.request,
        manifest,
        snapshot,
        controller._runtime_capabilities,
        planning_profile_sha256=controller._runtime_capability_generation_sha256,
        model_manifest_sha256=(
            controller._runtime_manifest_generation_sha256.get(manifest.model_id)
        ),
    )
    memory_rejections = controller._runtime_memory_rejections(
        candidate_set, snapshot
    )
    if current.execution_plan.execution_contract.execution_mode == "adaptive-split":
        selected, rejected, selection_reason = (
            controller._select_adaptive_desktop_recovery(
                candidate_set,
                failed_route_id=failed.decision.route_id,
                runtime_rejections=memory_rejections,
            )
        )
        fallback_selection_mode = "desktop-baseline"
    elif elastic:
        selected, rejected, selection_reason, fallback_selection_mode = (
            _select_elastic_desktop_recovery(
                controller,
                candidate_set,
                failed_route_id=failed.decision.route_id,
                runtime_rejections=memory_rejections,
                selection_mode=current.selection_mode,
                exited_executor_id=exited_executor_id,
                masked_executor_id=masked_executor_id,
            )
        )
    else:
        selected, rejected, selection_reason = (
            controller._select_automated_recovery_fallback(
                candidate_set,
                failed_route_id=failed.decision.route_id,
                runtime_rejections=memory_rejections,
            )
        )
        fallback_selection_mode = current.selection_mode
    preview = controller._preview_automated_resources(
        selected, observed_at_us=failed_at_us
    )
    controller._preview_automated_memory(
        selected.plan,
        snapshot,
        start_us=preview.start_us,
        reserved_until_us=preview.finish_upper_us,
    )
    return (
        snapshot,
        candidate_set,
        estimates,
        selected,
        preview,
        rejected,
        selection_reason,
        fallback_selection_mode,
    )


def fail_automated_request(
    controller,
    request_id: str,
    *,
    failed_at_us: int,
    reason: str,
    snapshot: HeterogeneousRuntimeSnapshot,
    physical_failure: RuntimeExecutionFailure | None = None,
    transition_receipts: Sequence[RuntimeTransitionReceipt] = (),
) -> RuntimeFailureRecovery:
    """Record a physical failure and make a new scheduler decision."""
    current = controller.runtime_ticket(request_id)
    if current.execution_plan is None:
        raise UnifiedScheduleError(
            "runtime ticket was not created by automated scheduling"
        )
    manifest = controller.runtime_model_manifest(current.model.model_id)
    if (
        manifest.artifact_sha256 != current.model.artifact_sha256
        or manifest.artifact_bytes != current.model.artifact_bytes
    ):
        raise UnifiedScheduleError("runtime model identity differs")
    failure = (
        RuntimeExecutionFailure("unspecified", True, False)
        if physical_failure is None else physical_failure
    )
    if not isinstance(failure, RuntimeExecutionFailure):
        raise UnifiedScheduleError(
            "runtime physical failure receipt is invalid"
        )
    elastic = failure.phase in ELASTIC_FAILURE_PHASES
    # The recovery of an elastic failure is planned when its snapshot was sampled: the adapter
    # samples it after the failure (and after the rig retired or reaped the failed server), so it
    # is judged at its own capture time, never at the earlier failure time. A snapshot that does
    # not cover that time is refused before the ticket changes: the adapter samples a new one.
    fallback_at_us = failed_at_us
    if elastic:
        if not isinstance(snapshot, HeterogeneousRuntimeSnapshot):
            raise UnifiedScheduleError("runtime system snapshot is invalid")
        fallback_at_us = max(failed_at_us, snapshot.captured_at_us)
        try:
            snapshot.validate_at(fallback_at_us)
        except RuntimeCapabilityError as exc:
            if str(exc) != "system snapshot is stale":
                raise UnifiedScheduleError(str(exc)) from exc
            raise RuntimeReplanRetryRequired(
                request_id, current.ticket_id, "SYSTEM_SNAPSHOT_STALE"
            ) from exc
    # An elastic recovery keeps the failed attempt's queue place (its arrival
    # sequence and the attempts that waited on it) when another model's work
    # waited on it, so the failure cannot switch the executor to that model.
    runtime = controller._runtime_controller
    retained_order = (
        runtime.recovery_dispatch_order(request_id) if elastic else None
    )
    with (
        nullcontext() if retained_order is None
        else runtime.defer_dispatch_wake()
    ), controller._transaction():
        failed, cancelled, quarantine_action = controller._fail_runtime_ticket(
            current=current,
            manifest=manifest,
            failure=failure,
            failed_at_us=failed_at_us,
            reason=reason,
            transition_receipts=transition_receipts,
        )
        if elastic:
            # The failed attempt's adaptive session (a desktop parent with a dormant FFN runtime
            # or a helper envelope opens one too) must not outlive it; adaptive-split sessions
            # are closed below as before. Any later attempt may replace what it left behind.
            _record_elastic_attempt(controller, request_id, failed.ticket_id)
            if (
                current.execution_plan.execution_contract.execution_mode
                != "adaptive-split"
            ):
                controller._adaptive_decode.recover_attempt_for_restart(
                    request_id, failed.ticket_id, reason
                )
        if not failure.fallback_allowed:
            return controller._terminal_failure_recovery(
                current=current,
                manifest=manifest,
                failed=failed,
                cancelled=cancelled,
                quarantine_action=quarantine_action,
                failed_at_us=failed_at_us,
                reason=reason,
                notify_reason="REQUEST_FAILED",
                recover_adaptive=True,
            )
        if retained_order is not None:
            # Attempts that waited on the failed one hold reservations after
            # its old end; they replan behind the recovery.
            runtime.invalidate_queued_attempts(
                retained_order[1],
                "predecessor_execution_failed",
                failed_at_us,
                cancel_owner=controller.cancel,
                release_memory=controller._runtime_memory.release_owner,
            )
        try:
            fallback_values = controller._prepare_automated_failure_fallback(
                current=current,
                manifest=manifest,
                snapshot=snapshot,
                failed=failed,
                failed_at_us=fallback_at_us,
                **({"elastic": True, "exited_executor_id": failure.exited_executor_id}
                   if elastic else {}),
                **({"masked_executor_id": failure.masked_executor_id}
                   if elastic and failure.masked_executor_id is not None else {}),
            )
            _require_recovery_without_failed_devices(
                fallback_values[3], failure
            )
        except (
            RouteGenerationError,
            RuntimeCapabilityError,
            RuntimeResourceError,
            RuntimeResidencyProjectionError,
            SchedulerError,
            UnifiedScheduleError,
        ) as unavailable:
            if elastic:
                # A lost helper or server that cannot be recovered ends the request with the
                # scheduler's own reason (the adapter raises it; the run stays fail-fast).
                _record_recovery_unavailable(controller, request_id, unavailable)
            return controller._terminal_failure_recovery(
                current=current,
                manifest=manifest,
                failed=failed,
                cancelled=cancelled,
                quarantine_action=quarantine_action,
                failed_at_us=failed_at_us,
                reason=reason,
                notify_reason="FALLBACK_UNAVAILABLE",
                recover_adaptive=False,
            )
        (
            snapshot,
            candidate_set,
            estimates,
            selected,
            preview,
            rejected,
            selection_reason,
            fallback_selection_mode,
        ) = fallback_values
        with (
            nullcontext() if retained_order is None
            else runtime.retained_dispatch_order(request_id, retained_order)
        ):
            fallback = controller._commit_automated_attempt(
                request=current.request,
                snapshot=snapshot,
                candidate_set=candidate_set,
                estimates=estimates,
                selected=selected,
                preview=preview,
                rejected=rejected,
                reason=selection_reason,
                observed_at_us=fallback_at_us,
                event_kind="FALLBACK",
                previous_ticket_id=failed.ticket_id,
                failure_reason=reason,
                previous_transition_receipts=failed.transition_receipts,
                selection_mode=fallback_selection_mode,
            )
        if (
            current.execution_plan.execution_contract.execution_mode
            == "adaptive-split"
        ):
            controller._adaptive_decode.recover_for_restart(request_id, reason)
        return RuntimeFailureRecovery(
            failed_ticket_id=failed.ticket_id,
            reason=reason,
            cancelled_tokens=cancelled,
            quarantine_action=quarantine_action,
            fallback=fallback,
        )
