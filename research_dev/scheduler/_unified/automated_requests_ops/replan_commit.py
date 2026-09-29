"""AutomatedRequestMixin replan commit operations on its existing owner."""

from __future__ import annotations

from contextlib import nullcontext
import time
from types import MappingProxyType
from typing import Callable, ContextManager, Mapping

from ..._internal.policy import LeasePreview, SchedulerError
from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.runtime_cost import RuntimeCostEstimateSet
from ..._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot, RuntimeCapabilityError
from ..._internal.route_generation import (
    DesktopControlUnavailableError,
    RuntimeRouteTemplateSet,
    candidate_set_to_runtime_costs,
)
from ..._internal.model_placement_controller import ModelPlacementControllerError
from ..._internal.runtime_plan import AutomatedCandidateSet, AutomatedRouteCandidate
from ..._internal.runtime_residency_projection import RuntimeResidencyProjectionError
from ..._internal.runtime_residency_cohorts import RuntimeModelPlacementEpoch
from ..._internal.runtime_controller import (
    RuntimeControllerError,
    RuntimeReplanRetryRequired,
    RuntimeRequestTicket,
)
from ..._internal.runtime_dispatch_policy import plan_changes_residency
from .affinity import (
    AFFINITY_DISPLACEMENT_REASON,
    _NoAffinityGain,
    model_affinity_replan_displacement,
)
from .common import _AutomatedReplanPreparation
from .continuous_join import (
    BARRIER_DISPLACEMENT_REASON,
    _NoJoinGain,
    continuous_join_replan_bypass,
    replan_bypass_refusal,
)


def _commit_replanned_attempt(
    controller,
    *,
    preparation: _AutomatedReplanPreparation,
    candidate_set: AutomatedCandidateSet,
    estimates: RuntimeCostEstimateSet,
    selected: AutomatedRouteCandidate,
    preview: LeasePreview,
    rejected: tuple[tuple[str, str], ...],
    selection_reason: str,
    observed_at_us: int,
    reason: str,
    publish_epoch: RuntimeModelPlacementEpoch | None,
    publish_templates: RuntimeRouteTemplateSet | None,
    priority_compaction_active: bool,
) -> RuntimeRequestTicket:
    context = preparation.context
    current = preparation.previous
    with controller._runtime_controller.defer_dispatch_wake():
        ticket = controller._commit_automated_attempt(
            request=current.request,
            snapshot=context.snapshot,
            candidate_set=candidate_set,
            estimates=estimates,
            selected=selected,
            preview=preview,
            rejected=rejected,
            reason=selection_reason,
            observed_at_us=observed_at_us,
            event_kind="REPLAN",
            previous_ticket_id=current.ticket_id,
            failure_reason=reason,
            selection_mode=current.selection_mode,
            placement_epoch=publish_epoch,
            route_templates=publish_templates,
            bound_placement_epoch=(
                publish_epoch or context.placement_epoch
            ),
        )
        if (
            not priority_compaction_active
            and ticket.decision.start_us <= observed_at_us
        ):
            controller._runtime_controller.release_replanned_capacity_frontier(
                current.request.request_id,
                current.decision,
                observed_at_us,
                resource_capacities={
                    resource_id: resource.capacity
                    for resource_id, resource
                    in controller._runtime_capabilities.resources.items()
                },
                cancel_owner=controller.cancel,
                release_memory=controller._runtime_memory.release_owner,
            )
    return ticket


def _record_replan_timing(
    controller,
    *,
    ticket: RuntimeRequestTicket,
    preparation: _AutomatedReplanPreparation,
    compiler,
    checkpoint_us: int,
    replan_started_ns: int,
) -> None:
    timings = preparation.context.timings_ns
    controller._runtime_decision_timings.append(MappingProxyType({
        "attempt_index": ticket.attempt_index,
        "candidate_generation": dict(compiler.last_generation_timing()),
        "candidate_generation_us": timings["generation"] // 1000,
        "candidate_serialization_us": timings["conversion"] // 1000,
        "checkpoint_us": checkpoint_us,
        "commit": dict(controller._last_runtime_commit_timing),
        "eligibility_admission_us": timings["admission"] // 1000,
        "epoch_lookup_us": timings["epoch_lookup"] // 1000,
        "epoch_validation_us": timings["epoch_validation"] // 1000,
        "event_kind": "REPLAN",
        "model_placement_epoch_fast_path": (
            preparation.context.placement_epoch is not None
        ),
        "model_placement_epoch_invalidation_reason": (
            preparation.context.epoch_invalidation_reason
        ),
        "request_id": ticket.request.request_id,
        "resident_component_priority": (
            preparation.prioritize_resident_component
        ),
        "resident_component_priority_sha256s": tuple(sorted(
            preparation.hot_compatible_components
        )),
        "reservation_preview_us": timings["preview"] // 1000,
        "residency_fixed_point_iterations": 1,
        "selection_us": timings["selection"] // 1000,
        "snapshot_projection_us": timings["projection"] // 1000,
        "ticket_id": ticket.ticket_id,
        "total_us": (time.perf_counter_ns() - replan_started_ns) // 1000,
    }))


def _clear_rejected_replan_baseline(
    controller,
    *,
    preparation: _AutomatedReplanPreparation,
    current: RuntimeRequestTicket,
    candidate_set: AutomatedCandidateSet,
    memory_rejections: Mapping[str, str],
    observed_at_us: int,
    priority_compaction_active: bool,
) -> bool:
    """Make a memory-rejected baseline plannable before selection fails.

    Causal dependents cannot dispatch before this attempt, so the ones
    reserved before its rejected memory window ends are deferred. Otherwise
    residency is re-projected at the baseline's earliest start, as arrivals
    do. Returns True when the candidates must be regenerated.
    """
    baseline = candidate_set.baseline
    if memory_rejections.get(baseline.candidate_id) in {
        None, "RESOURCE_CALENDAR_CURRENT",
    }:
        return False
    runtime = controller._runtime_controller
    request_id = current.request.request_id
    try:
        rejected_window = controller._preview_automated_resources(baseline)
    except SchedulerError:
        return False
    if not priority_compaction_active and runtime.defer_causal_dependents_before(
        request_id,
        rejected_window.finish_upper_us,
        controller.cancel,
        controller._runtime_memory.release_owner,
    ):
        return True
    context = preparation.context
    not_before = dict(context.live_not_before_by_resource)
    for resource_id, barrier_us in (
        controller._runtime_plan_not_before_by_resource(baseline.plan).items()
    ):
        not_before[resource_id] = max(
            not_before.get(resource_id, 0), barrier_us
        )
    started_ns = time.perf_counter_ns()
    try:
        preview = controller._preview_automated_resources(
            baseline,
            observed_at_us=controller._causal_candidate_observed_at(
                baseline, observed_at_us, not_before
            ),
        )
    except SchedulerError:
        return False
    # The attempt cannot start before the predecessors it waits for end.
    predecessors = frozenset(preparation.projection_request_ids)
    start_us = max((
        preview.start_us,
        *(
            lease.reserved_until_us
            for ticket in runtime.current_tickets()
            if ticket.request.request_id in predecessors
            for lease in ticket.decision.leases
        ),
    ))
    # A dependent's transitions must never be projected ahead of it.
    if runtime.queued_causal_dependents(request_id, start_us):
        return False
    projected = controller._automated_snapshot_for_request(
        current.request,
        preparation.observed_snapshot,
        exclude_request_id=request_id,
        project_before_us=start_us,
    )
    context.timings_ns["projection"] += time.perf_counter_ns() - started_ns
    if (
        projected.residency == context.snapshot.residency
        and projected.memory == context.snapshot.memory
    ):
        return False
    for resource_id in baseline.plan.resource_ids:
        context.live_not_before_by_resource[resource_id] = max(
            context.live_not_before_by_resource.get(resource_id, 0),
            start_us,
        )
    context.snapshot = projected
    return True


def _execute_automated_replan(
    controller,
    *,
    preparation: _AutomatedReplanPreparation,
    current: RuntimeRequestTicket,
    manifest,
    compiler,
    observed_at_us: int,
    reason: str,
    priority_compaction_active: bool,
    checkpoint_us: int,
    replan_started_ns: int,
    require_resident_plan: bool = False,
    commit_scope: Callable[
        [AutomatedRouteCandidate, LeasePreview], ContextManager[None]
    ] | None = None,
) -> RuntimeRequestTicket:
    """Select, preview and commit the replacement attempt.

    ``commit_scope(selected, preview)``, when given, may refuse the selection
    (by raising) and returns the context the commit runs in.
    """
    context = preparation.context
    # One budget covers dependent deferrals (each retires a queued
    # dependent) and baseline residency re-projections.
    for _ in range(len(controller._runtime_controller.current_tickets()) + 2):
        candidate_set = controller._replan_candidate_set(
            preparation=preparation,
            current=current,
            manifest=manifest,
            compiler=compiler,
            observed_at_us=observed_at_us,
        )
        started_ns = time.perf_counter_ns()
        memory_rejections = controller._runtime_memory_rejections(
            candidate_set,
            context.snapshot,
            projection_token=controller._current_phone_projection_token(),
        )
        context.timings_ns["admission"] += time.perf_counter_ns() - started_ns
        if _clear_rejected_replan_baseline(
            controller,
            preparation=preparation,
            current=current,
            candidate_set=candidate_set,
            memory_rejections=memory_rejections,
            observed_at_us=observed_at_us,
            priority_compaction_active=priority_compaction_active,
        ):
            continue
        started_ns = time.perf_counter_ns()
        (
            candidate_set,
            selected,
            rejected,
            selection_reason,
            publish_epoch,
            publish_templates,
        ) = controller._replan_candidate_selection(
            preparation=preparation,
            candidate_set=candidate_set,
            current=current,
            manifest=manifest,
            compiler=compiler,
            observed_at_us=observed_at_us,
            memory_rejections=memory_rejections,
        )
        context.timings_ns["selection"] += time.perf_counter_ns() - started_ns
        started_ns = time.perf_counter_ns()
        estimates = candidate_set_to_runtime_costs(
            candidate_set,
            current.request,
            manifest,
            context.snapshot,
            controller._runtime_capabilities,
            planning_profile_sha256=controller._runtime_capability_generation_sha256,
            model_manifest_sha256=(
                controller._runtime_manifest_generation_sha256.get(manifest.model_id)
            ),
        )
        context.timings_ns["conversion"] += time.perf_counter_ns() - started_ns
        started_ns = time.perf_counter_ns()
        plan_not_before = dict(context.live_not_before_by_resource)
        for resource_id, barrier_us in (
            controller._runtime_plan_not_before_by_resource(selected.plan).items()
        ):
            plan_not_before[resource_id] = max(
                plan_not_before.get(resource_id, 0), barrier_us
            )
        preview = controller._preview_automated_resources(
            selected,
            observed_at_us=controller._causal_candidate_observed_at(
                selected, observed_at_us, plan_not_before
            ),
        )
        # Dependents cannot dispatch before this attempt, so their earlier
        # reservations must not push it later; defer them and replan.
        if priority_compaction_active or not (
            controller._runtime_controller.defer_causal_dependents_before(
                current.request.request_id,
                preview.start_us,
                controller.cancel,
                controller._runtime_memory.release_owner,
            )
        ):
            break
    else:
        raise UnifiedScheduleError(
            "runtime replan residency planning did not converge"
        )
    if require_resident_plan and plan_changes_residency(
        selected.plan.transitions,
        controller._runtime_exclusive_memory_resources(),
    ):
        raise _NoAffinityGain("replanned attempt changes residency")
    controller._preview_automated_memory(
        selected.plan,
        context.snapshot,
        start_us=preview.start_us,
        reserved_until_us=preview.finish_upper_us,
    )
    context.timings_ns["preview"] += time.perf_counter_ns() - started_ns
    with (
        nullcontext() if commit_scope is None
        else commit_scope(selected, preview)
    ):
        ticket = controller._commit_replanned_attempt(
            preparation=preparation,
            candidate_set=candidate_set,
            estimates=estimates,
            selected=selected,
            preview=preview,
            rejected=rejected,
            selection_reason=selection_reason,
            observed_at_us=observed_at_us,
            reason=reason,
            publish_epoch=publish_epoch,
            publish_templates=publish_templates,
            priority_compaction_active=priority_compaction_active,
        )
    controller._record_replan_timing(
        ticket=ticket,
        preparation=preparation,
        compiler=compiler,
        checkpoint_us=checkpoint_us,
        replan_started_ns=replan_started_ns,
    )
    return ticket


def _handle_automated_replan_failure(
    controller,
    *,
    error: BaseException,
    request_id: str,
    current: RuntimeRequestTicket,
    observed_at_us: int,
) -> RuntimeRequestTicket:
    if isinstance(error, RuntimeReplanRetryRequired):
        raise error
    if (
        isinstance(error, RuntimeCapabilityError)
        and str(error) == "system snapshot is stale"
    ):
        raise RuntimeReplanRetryRequired(
            request_id, current.ticket_id, "SYSTEM_SNAPSHOT_STALE"
        ) from error
    if (
        isinstance(error, DesktopControlUnavailableError)
        and current.selection_mode == "desktop-baseline"
    ):
        raise RuntimeReplanRetryRequired(
            request_id,
            current.ticket_id,
            "DESKTOP_CONTROL_LIVE_STATE_INCOMPLETE",
        ) from error
    if (
        isinstance(error, RuntimeResidencyProjectionError)
        and error.request_id is not None
        and error.ticket_id is not None
        and error.request_id != request_id
    ):
        try:
            with controller._transaction(convert=False):
                return controller._defer_automated_replan_for_projection(
                    request_id, error, observed_at_us
                )
        except (RuntimeControllerError, ModelPlacementControllerError):
            pass
    failure_reason = (
        "scheduler_replan_failed:"
        + type(error).__name__
        + ":"
        + str(error)
    )
    if not failure_reason.isascii():
        failure_reason = failure_reason.encode(
            "ascii", "backslashreplace"
        ).decode("ascii")
    try:
        with controller._transaction(errors=Exception, convert=False):
            controller._publish_automated_replan_failure(
                request_id, observed_at_us, failure_reason
            )
    except Exception as terminal_error:
        raise UnifiedScheduleError(
            "runtime replan rollback was restored but terminal "
                "failure publication failed: " + str(terminal_error)
        ) from terminal_error
    if isinstance(error, UnifiedScheduleError):
        raise error
    raise UnifiedScheduleError(str(error)) from error


def _replan_automated_request_once(
    controller,
    request_id: str,
    *,
    observed_at_us: int,
    reason: str,
    snapshot: HeterogeneousRuntimeSnapshot,
    expected_ticket_id: str | None = None,
    expected_queue_generation: int | None = None,
    priority_compaction_active: bool = False,
) -> RuntimeRequestTicket:
    """Replace one scheduler-marked queued attempt atomically."""
    replan_started_ns = time.perf_counter_ns()
    current, expected_queue_generation, superseded = (
        controller._validated_replan_ticket(
            request_id,
            snapshot,
            observed_at_us,
            expected_ticket_id,
            expected_queue_generation,
        )
    )
    if superseded:
        return current
    manifest = controller.runtime_model_manifest(current.model.model_id)
    compiler = controller._automated_compiler()
    ticket = _replan_with_model_affinity(
        controller,
        current=current,
        snapshot=snapshot,
        manifest=manifest,
        compiler=compiler,
        observed_at_us=observed_at_us,
        reason=reason,
        expected_queue_generation=expected_queue_generation,
        priority_compaction_active=priority_compaction_active,
        replan_started_ns=replan_started_ns,
    )
    if ticket is None:
        ticket = _replan_with_barrier_bypass(
            controller,
            current=current,
            snapshot=snapshot,
            manifest=manifest,
            compiler=compiler,
            observed_at_us=observed_at_us,
            reason=reason,
            expected_queue_generation=expected_queue_generation,
            priority_compaction_active=priority_compaction_active,
            replan_started_ns=replan_started_ns,
        )
    if ticket is not None:
        return ticket
    checkpoint_started_ns = time.perf_counter_ns()
    try:
        with controller._transaction(errors=Exception, convert=False):
            checkpoint_us = (
                time.perf_counter_ns() - checkpoint_started_ns
            ) // 1000
            preparation = controller._prepare_automated_replan(
                current=current,
                snapshot=snapshot,
                manifest=manifest,
                observed_at_us=observed_at_us,
                reason=reason,
                expected_queue_generation=expected_queue_generation,
            )
            return controller._execute_automated_replan(
                preparation=preparation,
                current=current,
                manifest=manifest,
                compiler=compiler,
                observed_at_us=observed_at_us,
                reason=reason,
                priority_compaction_active=priority_compaction_active,
                checkpoint_us=checkpoint_us,
                replan_started_ns=replan_started_ns,
            )
    except Exception as exc:
        return controller._handle_automated_replan_failure(
            error=exc,
            request_id=request_id,
            current=current,
            observed_at_us=observed_at_us,
        )


def _replan_with_model_affinity(
    controller,
    *,
    current: RuntimeRequestTicket,
    snapshot: HeterogeneousRuntimeSnapshot,
    manifest,
    compiler,
    observed_at_us: int,
    reason: str,
    expected_queue_generation: int | None,
    priority_compaction_active: bool,
    replan_started_ns: int,
) -> RuntimeRequestTicket | None:
    """Replan ahead of the displaced work, or roll back to None.

    Any error rolls the attempt back; the ordinary replan then runs on the
    unchanged state and handles that error as before.
    """
    displacement = model_affinity_replan_displacement(
        controller,
        snapshot=snapshot,
        current=current,
        observed_at_us=observed_at_us,
        reason=reason,
    )
    if displacement is None:
        return None
    displaced, note = displacement
    runtime = controller._runtime_controller
    checkpoint_started_ns = time.perf_counter_ns()
    try:
        with runtime.defer_dispatch_wake(), controller._transaction(
            errors=Exception, convert=False
        ):
            checkpoint_us = (
                time.perf_counter_ns() - checkpoint_started_ns
            ) // 1000
            runtime.replan_queued_now(
                displaced,
                AFFINITY_DISPLACEMENT_REASON,
                observed_at_us,
                cancel_owner=controller.cancel,
                release_memory=controller._runtime_memory.release_owner,
            )
            with runtime.dispatch_precedence(
                current.request.request_id, displaced, note
            ):
                preparation = controller._prepare_automated_replan(
                    current=current,
                    snapshot=snapshot,
                    manifest=manifest,
                    observed_at_us=observed_at_us,
                    reason=reason,
                    expected_queue_generation=expected_queue_generation,
                )
                return controller._execute_automated_replan(
                    preparation=preparation,
                    current=current,
                    manifest=manifest,
                    compiler=compiler,
                    observed_at_us=observed_at_us,
                    reason=reason,
                    priority_compaction_active=priority_compaction_active,
                    checkpoint_us=checkpoint_us,
                    replan_started_ns=replan_started_ns,
                    require_resident_plan=True,
                )
    except Exception:
        runtime.record_dispatch_policy_event("affinity_refusals")
        return None


def _bypass_commit_scope(
    controller, current: RuntimeRequestTicket, displaced, note
) -> Callable[[AutomatedRouteCandidate, LeasePreview], ContextManager[None]]:
    """Refuse a replanned joiner that gains nothing, else commit it ahead of ``displaced``."""
    runtime = controller._runtime_controller
    bound_us = runtime.dispatch_policy.max_barrier_extension_s * 1_000_000

    def scope(selected, preview):
        refusal, extension_us = replan_bypass_refusal(
            selected, preview, note,
            controller._runtime_exclusive_memory_resources(), bound_us,
        )
        if refusal is not None:
            raise _NoJoinGain(refusal)
        return runtime.dispatch_precedence(current.request.request_id, displaced, {
            **note,
            "extension_us": extension_us,
            "finish_upper_us": preview.finish_upper_us,
            "max_barrier_extension_us": bound_us,
            "reserved_start_us": preview.start_us,
        })

    return scope


def _replan_with_barrier_bypass(
    controller,
    *,
    current: RuntimeRequestTicket,
    snapshot: HeterogeneousRuntimeSnapshot,
    manifest,
    compiler,
    observed_at_us: int,
    reason: str,
    expected_queue_generation: int | None,
    priority_compaction_active: bool,
    replan_started_ns: int,
) -> RuntimeRequestTicket | None:
    """Replan a joiner ahead of the queued changes within the bound, or roll back to None.

    Under ``continuous_join`` only (``continuous_join_replan_bypass``). The
    changes are cancelled and the joiner is re-planned as a joiner (the join
    selection rejects plans preparing an exclusive residency device); it is
    kept only when ``replan_bypass_refusal`` finds no reason against it. Any
    refusal or error rolls back, is recorded, and the ordinary replan runs on
    the unchanged state.
    """
    bypass = continuous_join_replan_bypass(
        controller,
        snapshot=snapshot,
        current=current,
        observed_at_us=observed_at_us,
        reason=reason,
    )
    if bypass is None:
        return None
    displaced, note = bypass
    runtime = controller._runtime_controller
    request_id = current.request.request_id
    checkpoint_started_ns = time.perf_counter_ns()
    try:
        with runtime.defer_dispatch_wake(), controller._transaction(
            errors=Exception, convert=False
        ):
            checkpoint_us = (time.perf_counter_ns() - checkpoint_started_ns) // 1000
            runtime.replan_queued_now(
                displaced,
                BARRIER_DISPLACEMENT_REASON,
                observed_at_us,
                cancel_owner=controller.cancel,
                release_memory=controller._runtime_memory.release_owner,
            )
            with runtime.continuous_join_resolution(request_id):
                preparation = controller._prepare_automated_replan(
                    current=current,
                    snapshot=snapshot,
                    manifest=manifest,
                    observed_at_us=observed_at_us,
                    reason=reason,
                    expected_queue_generation=expected_queue_generation,
                )
                return controller._execute_automated_replan(
                    preparation=preparation,
                    current=current,
                    manifest=manifest,
                    compiler=compiler,
                    observed_at_us=observed_at_us,
                    reason=reason,
                    priority_compaction_active=priority_compaction_active,
                    checkpoint_us=checkpoint_us,
                    replan_started_ns=replan_started_ns,
                    commit_scope=_bypass_commit_scope(
                        controller, current, displaced, note
                    ),
                )
    except Exception as exc:
        runtime.record_dispatch_policy_event("continuous_join_refusals")
        runtime.record_dispatch_policy_refusal(
            "CONTINUOUS_JOIN_REFUSED",
            request_id,
            str(exc) if isinstance(exc, _NoJoinGain)
            else f"bypass replan failed: {type(exc).__name__}: {exc}",
            observed_at_us,
        )
        return None


def _defer_automated_replan_for_projection(
    controller,
    request_id: str,
    error: RuntimeResidencyProjectionError,
    observed_at_us: int,
) -> RuntimeRequestTicket:
    assert error.request_id is not None
    assert error.ticket_id is not None
    with controller._runtime_controller.defer_dispatch_wake():
        predecessor = controller._runtime_controller.ticket(error.request_id)
        if predecessor.dispatch_state == "ACQUIRED":
            defer_active = (
                controller._runtime_controller
                .defer_replan_until_active_completion
            )
            defer_active(
                request_id,
                error.request_id,
                error.ticket_id,
                "residency_projection_invalid",
                observed_at_us,
                controller.cancel,
                controller._runtime_memory.release_owner,
            )
        else:
            controller._runtime_controller.defer_replan_behind(
                request_id,
                error.request_id,
                error.ticket_id,
                "residency_projection_invalid",
                observed_at_us,
                controller.cancel,
                controller._runtime_memory.release_owner,
            )
            controller._model_placement_controller.notify(
                predecessor.model.artifact_sha256,
                "REQUEST_REPLAN_REQUIRED",
                observed_at_us,
            )
        return controller._runtime_controller.ticket(request_id)


def _publish_automated_replan_failure(
    controller,
    request_id: str,
    observed_at_us: int,
    failure_reason: str,
) -> None:
    failed, _ = controller._runtime_controller.fail_replan(
        request_id,
        observed_at_us,
        failure_reason,
        cancel_owner=controller.cancel,
        release_memory=controller._runtime_memory.release_owner,
    )
    controller._fail_phone_residency_transition(
        failed.ticket_id,
        observed_at_us,
        failure_reason,
    )
    controller._runtime_residency_cohorts.record_terminal(request_id)
    controller._model_placement_controller.release_request(request_id)
    controller._model_placement_controller.notify(
        failed.model.artifact_sha256,
        "REQUEST_REPLAN_FAILED",
        observed_at_us,
    )
    controller._append_runtime_log(
        "FAILED",
        failed,
        observed_at_us,
        failed.dispatch_state,
    )
