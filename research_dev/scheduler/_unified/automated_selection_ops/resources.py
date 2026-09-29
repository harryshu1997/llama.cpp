"""AutomatedSelectionMixin resources operations on its existing owner."""

from __future__ import annotations

from ..._internal.runtime_resources import runtime_phase_lease_demands

from types import MappingProxyType
from typing import Mapping

from ..._internal.policy import LeasePreview, Request, SchedulerError
from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot
from ..._internal.runtime_plan import (
    AutomatedCandidateSet,
    AutomatedRouteCandidate,
    RuntimeExecutionPlan,
)
from ..._internal.runtime_resources import RuntimeResidencyProjectionToken, RuntimeResourceError
from ..._internal.runtime_dispatch_policy import plan_changes_residency
from ..._internal.runtime_residency_projection import (
    RuntimeResidencyProjectionError,
    project_scheduler_residency_with_token,
    projection_token_matches_plan,
)


def _runtime_memory_rejection(error: RuntimeResourceError) -> str:
    message = str(error)
    prefixes = (
        (
            "memory capacity is insufficient: ",
            "MEMORY_CAPACITY_CURRENT:",
        ),
        (
            "memory resource is absent: ",
            "MEMORY_RESOURCE_ABSENT_CURRENT:",
        ),
        (
            "exclusive memory replacement overlaps: ",
            "MEMORY_REPLACEMENT_CONFLICT_CURRENT:",
        ),
        (
            "transition eviction is stale: ",
            "TRANSITION_EVICTION_STALE_CURRENT:",
        ),
    )
    for prefix, code in prefixes:
        if message.startswith(prefix):
            resource_id = message.removeprefix(prefix)
            if resource_id and resource_id.isascii():
                return code + resource_id
    raise error


def _runtime_memory_rejections(
    controller,
    candidate_set: AutomatedCandidateSet,
    snapshot: HeterogeneousRuntimeSnapshot,
    *,
    exclude_owner_id: str | None = None,
    request: Request | None = None,
    source_snapshot: HeterogeneousRuntimeSnapshot | None = None,
    observed_at_us: int | None = None,
    not_before_by_resource: Mapping[str, int] = MappingProxyType({}),
    projection_token: RuntimeResidencyProjectionToken | None = None,
) -> Mapping[str, str]:
    exact_projection = request is not None or source_snapshot is not None
    if exact_projection and (
        request is None
        or source_snapshot is None
        or observed_at_us is None
    ):
        raise UnifiedScheduleError(
            "exact memory projection context is incomplete"
        )
    if projection_token is not None and not isinstance(
        projection_token, RuntimeResidencyProjectionToken
    ):
        raise UnifiedScheduleError(
            "residency projection token is invalid"
        )
    rejected = {}
    projected_snapshot_by_start_us = {}
    transitioning_layout = (
        controller._model_placement_controller.preparing_phone_layout()
    )
    placement_candidates = controller._model_placement_candidate_set(
        candidate_set
    )
    for candidate in placement_candidates.candidates:
        candidate_geometry = candidate.plan.adapter_parameters.get(
            "phone_shard_set_geometry_sha256"
        )
        targets_transitioning_layout = bool(
            transitioning_layout is not None
            and candidate_geometry
                == transitioning_layout.layout.geometry_sha256
        )
        if not candidate.admitted and not targets_transitioning_layout:
            continue
        try:
            candidate_not_before = dict(not_before_by_resource)
            token_matches = bool(
                projection_token is not None
                and projection_token_matches_plan(
                    projection_token, candidate.plan
                )
            )
            if (
                targets_transitioning_layout
                and not token_matches
            ):
                rejected[candidate.candidate_id] = (
                    "RESIDENCY_PROJECTION_CURRENT"
                )
                continue
            if token_matches:
                assert projection_token is not None
                for resource_id in candidate.plan.resource_ids:
                    candidate_not_before[resource_id] = max(
                        candidate_not_before.get(resource_id, 0),
                        projection_token.ready_at_us,
                    )
            if exact_projection:
                for resource_id, barrier_us in (
                    controller._exclusive_transition_barriers(
                        candidate.plan
                    ).items()
                ):
                    candidate_not_before[resource_id] = max(
                        candidate_not_before.get(resource_id, 0),
                        barrier_us,
                    )
            preview = controller._preview_automated_resources(
                candidate,
                observed_at_us=(
                    None
                    if not exact_projection
                    else controller._causal_candidate_observed_at(
                        candidate,
                        observed_at_us,
                        candidate_not_before,
                    )
                ),
            )
            memory_snapshot = snapshot
            if exact_projection and token_matches:
                assert projection_token is not None
                memory_snapshot = (
                    projected_snapshot_by_start_us.get(
                        preview.start_us
                    )
                )
                if memory_snapshot is None:
                    memory_snapshot = (
                        project_scheduler_residency_with_token(
                            source_snapshot,
                            controller._runtime_capabilities,
                            controller._runtime_controller.current_tickets(),
                            controller._runtime_manifests,
                            projection_token,
                            candidate_start_us=preview.start_us,
                        )
                    )
                    projected_snapshot_by_start_us[
                        preview.start_us
                    ] = memory_snapshot
            controller._preview_automated_memory(
                candidate.plan,
                memory_snapshot,
                start_us=preview.start_us,
                reserved_until_us=preview.finish_upper_us,
                exclude_owner_id=exclude_owner_id,
            )
        except RuntimeResidencyProjectionError:
            rejected[candidate.candidate_id] = (
                "RESIDENCY_PROJECTION_CURRENT"
            )
            continue
        except (RuntimeResourceError, SchedulerError) as exc:
            if isinstance(exc, SchedulerError):
                rejected[candidate.candidate_id] = (
                    "RESOURCE_CALENDAR_CURRENT"
                )
                continue
            rejected[candidate.candidate_id] = (
                controller._runtime_memory_rejection(exc)
            )
    return MappingProxyType(dict(sorted(rejected.items())))


def _preview_automated_resources(
    controller,
    candidate: AutomatedRouteCandidate,
    *,
    observed_at_us: int | None = None,
) -> LeasePreview:
    if not isinstance(candidate, AutomatedRouteCandidate):
        raise UnifiedScheduleError(
            "automated resource preview candidate is invalid"
        )
    arrival_us = candidate.cost.start_us
    if observed_at_us is not None:
        arrival_us = max(arrival_us, observed_at_us)
    demands = runtime_phase_lease_demands(
        candidate.candidate_id, candidate.plan.resource_slots,
        candidate.plan.transitions, candidate.cost.service_us, candidate.cost.service_us,
    )
    return controller.timeline.preview_leases(
        demands,
        arrival_us,
        candidate.cost.service_us,
        candidate.cost.service_us,
    )


def _automated_prediction_finish_upper_us(
    candidate: AutomatedRouteCandidate,
    preview: LeasePreview,
) -> int:
    return preview.start_us + candidate.cost.service_upper_us


def _causal_candidate_observed_at(
    candidate: AutomatedRouteCandidate,
    observed_at_us: int,
    not_before_by_resource: Mapping[str, int],
) -> int:
    return max((
        observed_at_us,
        *(
            not_before_by_resource.get(resource_id, observed_at_us)
            for resource_id in candidate.plan.resource_ids
        ),
    ))


def _exclusive_transition_barriers(
    controller, plan: RuntimeExecutionPlan
) -> Mapping[str, int]:
    exclusive_by_device = controller._runtime_exclusive_memory_resources()
    resource_ids = {
        exclusive_by_device[device_id]
        for transition in plan.transitions
        if transition.source_state != transition.target_state
        for device_id in transition.prepares_device_ids
        if device_id in exclusive_by_device
    }
    if not resource_ids:
        return MappingProxyType({})
    authoritative = frozenset(
        controller._runtime_controller.projection_request_ids()
    )
    barriers = {}
    for ticket in controller._runtime_controller.current_tickets():
        if (
            ticket.request.request_id not in authoritative
            or ticket.lease_status == "CANCELLED"
        ):
            continue
        for lease in ticket.decision.leases:
            if lease.resource_id not in resource_ids:
                continue
            barriers[lease.resource_id] = max(
                barriers.get(lease.resource_id, 0),
                ticket.final_reserved_until_us[lease.token],
            )
    return MappingProxyType(dict(sorted(barriers.items())))


def _runtime_plan_not_before_by_resource(
    controller, plan: RuntimeExecutionPlan
) -> Mapping[str, int]:
    barriers = dict(controller._exclusive_transition_barriers(plan))
    token = controller._phone_projection_token_for_plan(plan)
    if token is not None:
        for resource_id in plan.resource_ids:
            barriers[resource_id] = max(
                barriers.get(resource_id, 0), token.ready_at_us
            )
    return MappingProxyType(dict(sorted(barriers.items())))


def _runtime_exclusive_memory_resources(controller) -> Mapping[str | tuple[str, str], str]:
    if controller._runtime_capabilities is None:
        raise UnifiedScheduleError("runtime capabilities are absent")
    return controller._runtime_capabilities.exclusive_residency_resources


def _runtime_residency_order_barrier(
    controller, plan: RuntimeExecutionPlan
) -> bool:
    exclusive_by_device = controller._runtime_exclusive_memory_resources()
    if controller._runtime_controller.dispatch_policy.work_conserving_admission:
        # Only a residency change needs the arrival-order barrier; work on
        # the resident model is ordered by its reserved lanes.
        return plan_changes_residency(plan.transitions, exclusive_by_device)
    exclusive_resources = frozenset(exclusive_by_device.values())
    return bool(
        exclusive_resources.intersection(plan.resource_ids)
    )


def _runtime_residency_hysteresis_key(
    controller, plan: RuntimeExecutionPlan, artifact_sha256: str
) -> str | None:
    """The model a plan holds the exclusive residency resource for, or None.

    Set only under ``dispatch_policy.residency_hysteresis_s`` for plans that
    use or change an exclusive residency resource, so the queue can time the
    hold from that resource's last release.
    """
    if not controller._runtime_controller.dispatch_policy.residency_hysteresis_s:
        return None
    exclusive_by_device = controller._runtime_exclusive_memory_resources()
    if plan_changes_residency(plan.transitions, exclusive_by_device) or (
        frozenset(exclusive_by_device.values()).intersection(plan.resource_ids)
    ):
        return artifact_sha256
    return None


def _preview_automated_memory(
    controller,
    plan: RuntimeExecutionPlan,
    snapshot: HeterogeneousRuntimeSnapshot,
    *,
    start_us: int,
    reserved_until_us: int,
    exclude_owner_id: str | None = None,
    enforce_live_capacity: bool = True,
) -> Mapping[str, int]:
    return controller._runtime_memory.preview(
        plan.memory_demands,
        snapshot.memory,
        start_us=start_us,
        reserved_until_us=reserved_until_us,
        transitions=plan.transitions,
        residency=snapshot.residency,
        exclusive_resource_by_device=(
            controller._runtime_exclusive_memory_resources()
        ),
        exclude_owner_id=exclude_owner_id,
        enforce_live_capacity=enforce_live_capacity,
    )
