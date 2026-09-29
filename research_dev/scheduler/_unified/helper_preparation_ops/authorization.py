"""HelperPreparationMixin authorization operations on its existing owner."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Sequence

from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.runtime_cost import RuntimeMemoryDemand
from ..._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot
from ..._internal.route_generation import RouteGenerationError
from ..._internal.model_placement_controller import (
    ModelPlacementControllerError,
    ModelPhoneResidencyLayout,
)
from ..._internal.runtime_plan import RuntimeHelperExecutionEnvelope, RuntimeTransitionPlan
from ..._internal.runtime_resources import RuntimeResourceError
from ..._internal.runtime_residency_projection import RuntimeResidencyProjectionError
from ..._internal.runtime_residency_cohorts import RuntimeResidencyCohortError
from ..._internal.runtime_controller import RuntimeRequestTicket
from ..common import _RequestHelperPreparation


def _preparation_helper_envelope(
    controller,
    ticket: RuntimeRequestTicket,
    request_id: str,
    *,
    expected_phone_layout_generation: int | None,
    expected_phone_layout_geometry_sha256: str | None,
    expected_operator_plan_sha256: str | None,
) -> RuntimeHelperExecutionEnvelope | str:
    """The helper envelope to prepare, or the status when there is none."""

    plan = ticket.execution_plan
    helper = None if plan is None else plan.helper_envelope
    if expected_phone_layout_generation is not None:
        dynamic = controller._request_helper_preparation_envelopes.get((
            request_id,
            ticket.ticket_id,
            expected_phone_layout_generation,
        ))
        if dynamic is not None:
            helper = dynamic
    if helper is None:
        return "NOT_REQUIRED"
    if (
        expected_phone_layout_generation is not None
        and helper.phone_layout_generation
            != expected_phone_layout_generation
    ) or (
        expected_phone_layout_geometry_sha256 is not None
        and helper.phone_layout_geometry_sha256
            != expected_phone_layout_geometry_sha256
    ) or (
        expected_operator_plan_sha256 is not None
        and helper.operator_plan_sha256
            != expected_operator_plan_sha256
    ):
        return "INCOMPATIBLE"
    return helper


def _proposed_phone_layout_for_preparation(
    controller,
    ticket: RuntimeRequestTicket,
    helper: RuntimeHelperExecutionEnvelope,
) -> tuple[ModelPhoneResidencyLayout | None, Mapping[str, object] | None]:
    """The PROPOSED layout to prepare, or the early result to return."""

    try:
        state = controller._model_placement_controller.phone_layout(
            helper.phone_layout_generation
        )
    except ModelPlacementControllerError:
        return None, MappingProxyType({"status": "INCOMPATIBLE"})
    if (
        state.layout.geometry_sha256
            != helper.phone_layout_geometry_sha256
        or not state.covers_artifact(ticket.model.artifact_sha256)
    ):
        return None, MappingProxyType({"status": "INCOMPATIBLE"})
    helper_session_ids = tuple(
        row.session_id
        for row in helper.helper_plan.execution_contract.phone_shards
    )
    if (
        controller._model_placement_controller
            .phone_layout_sessions_are_usable(
                helper.phone_layout_generation,
                helper.phone_layout_geometry_sha256,
                helper_session_ids,
            )
    ):
        return None, MappingProxyType({
            "phone_layout_generation": state.generation,
            "status": "READY",
        })
    if state.state == "PREPARING":
        return None, MappingProxyType({
            "phone_layout_generation": state.generation,
            "preparation_ticket_id": state.transition_ticket_id,
            "status": "FOLLOWER",
        })
    if state.state != "PROPOSED":
        return None, MappingProxyType({"status": "INCOMPATIBLE"})
    return state, None


def _defer_preparation_for_blockers(
    controller,
    request_id: str,
    state: ModelPhoneResidencyLayout,
    blockers: Sequence[str],
    observed_at_us: int,
) -> Mapping[str, object]:
    """Drain the helper attachments that block one layout transition."""

    with controller._transaction(convert=False):
        for blocker_id in blockers:
            rebind = controller._model_placement_controller\
                .request_helper_rebind(
                blocker_id,
                state.generation,
                observed_at_us=observed_at_us,
            )
            if (
                rebind.state == "QUIESCED"
                or rebind.drain_policy_sha256 is not None
            ):
                continue
            if rebind.target_allowed_session_ids:
                drain = controller._adaptive_decode\
                    .request_helper_session_drain(
                        blocker_id,
                        retained_layer_mask=(
                            rebind.retained_layer_mask
                        ),
                    )
                policy_sha256 = str(drain["policy_hash"])
                controller._model_placement_controller\
                    .bind_request_helper_rebind_drain_policy(
                        blocker_id,
                        state.generation,
                        policy_sha256,
                        observed_at_us=observed_at_us,
                    )
                if bool(drain["already_applied"]):
                    controller._model_placement_controller\
                        .mark_request_helper_rebind_quiesced(
                            blocker_id,
                            state.generation,
                            observed_at_us=observed_at_us,
                            allowed_session_ids=(
                                rebind.target_allowed_session_ids
                            ),
                            drain_policy_sha256=policy_sha256,
                        )
            else:
                controller._adaptive_decode.helper_unavailable(blocker_id)
        controller._model_placement_controller.record_request_helper_event(
            request_id,
            "PREPARATION_DEFERRED",
            observed_at_us,
            {
                "blocking_request_ids": list(blockers),
                "phone_layout_generation": state.generation,
                "reason": "DRAINING_HELPER_ATTACHMENTS",
            },
        )
    return MappingProxyType({
        "blocking_request_ids": blockers,
        "phone_layout_generation": state.generation,
        "reason": "DRAINING_HELPER_ATTACHMENTS",
        "status": "DEFERRED",
    })


def _verify_proposed_layout_ready(
    controller,
    request_id: str,
    ticket: RuntimeRequestTicket,
    helper: RuntimeHelperExecutionEnvelope,
    state: ModelPhoneResidencyLayout,
    *,
    snapshot: HeterogeneousRuntimeSnapshot,
    observed_at_us: int,
    preparation_ticket_id: str,
) -> Mapping[str, object] | None:
    """Promote a PROPOSED layout the snapshot already proves READY."""

    verification = controller._phone_layout_snapshot_verification(
        state,
        model_id=ticket.model.model_id,
        artifact_sha256=ticket.model.artifact_sha256,
        plan=helper.helper_plan,
        binding=helper.helper_binding,
        base_executor_id=ticket.binding.executor_id,
        snapshot=snapshot,
        require_base_executor_ready=False,
    )
    if verification is None:
        return None
    workspace_bytes, verification_sha256 = verification
    try:
        ready = (
            controller._model_placement_controller
            .verify_observed_phone_layout(
                state.generation,
                workspace_bytes=workspace_bytes,
                verified_at_us=observed_at_us,
                verification_sha256=verification_sha256,
            )
        )
    except ModelPlacementControllerError as exc:
        raise UnifiedScheduleError(str(exc)) from exc
    if ready is None:
        return MappingProxyType({"status": "INCOMPATIBLE"})
    for compiler in (
        controller._automated_route_compiler,
        controller._runtime_epoch_route_compiler,
    ):
        if compiler is not None:
            compiler.set_phone_residency_layout(ready.layout)
    rematerialized = controller._rematerialize_ready_layout_helpers(
        ready, snapshot, observed_at_us
    )
    controller._model_placement_controller.record_request_helper_event(
        request_id,
        "PREPARATION_READY",
        observed_at_us,
        {
            "phone_layout_generation": ready.generation,
            "phone_layout_geometry_sha256": (
                ready.layout.geometry_sha256
            ),
            "preparation_ticket_id": preparation_ticket_id,
            "state": "READY",
            "verification_sha256": verification_sha256,
            "rematerialized_request_ids": list(rematerialized),
        },
    )
    return MappingProxyType({
        "phone_layout_generation": ready.generation,
        "status": "READY",
        "verification_sha256": verification_sha256,
    })


def _reserve_helper_preparation_memory(
    controller,
    helper: RuntimeHelperExecutionEnvelope,
    state: ModelPhoneResidencyLayout,
    transitions: Sequence[RuntimeTransitionPlan],
    snapshot: HeterogeneousRuntimeSnapshot,
    *,
    phone_device_id: str,
    memory_owner_id: str,
    observed_at_us: int,
) -> tuple[RuntimeMemoryDemand, ...]:
    """Reserve the phone memory of one preparation; returns its demands."""

    memory_demands = tuple(
        row for row in helper.helper_plan.memory_demands
        if row.device_id == phone_device_id
    )
    if not memory_demands:
        raise UnifiedScheduleError(
            "request helper preparation memory plan is absent"
        )
    assert controller._runtime_capabilities is not None
    exclusive = controller._runtime_capabilities.exclusive_residency_resources
    memory_demands = controller._partial_phone_replacement_memory_demands(
        memory_demands,
        phone_device_id=phone_device_id,
        source=(
            controller._model_placement_controller.ready_phone_layout()
        ),
        target=state,
        transitions=transitions,
        snapshot=snapshot,
    )
    controller._validate_phone_session_replacement_authorization(
        state, helper.replacement_authorization
    )
    try:
        controller._runtime_memory.reserve(
            memory_owner_id,
            memory_demands,
            snapshot.memory,
            start_us=observed_at_us,
            transitions=transitions,
            residency=snapshot.residency,
            exclusive_resource_by_device=exclusive,
        )
    except RuntimeResourceError as exc:
        if not str(exc).startswith(
            "memory capacity is insufficient: "
        ):
            raise
        capacities = {
            row.resource_id: row.to_json()
            for row in snapshot.memory.capacities.values()
            if row.resource_id in {
                demand.resource_id for demand in memory_demands
            }
        }
        raise RuntimeResourceError(
            str(exc)
            + "; capacities="
            + repr(capacities)
            + "; demands="
            + repr(tuple(
                demand.to_json() for demand in memory_demands
            ))
        ) from exc
    return memory_demands


def _new_request_helper_preparation(
    request_id: str,
    ticket: RuntimeRequestTicket,
    helper: RuntimeHelperExecutionEnvelope,
    state: ModelPhoneResidencyLayout,
    *,
    preparation_ticket_id: str,
    projection_sha256: str,
    transition_ids: tuple[str, ...],
    lease_tokens: tuple[str, ...],
    yielding_resource_ids: tuple[str, ...],
    memory_owner_id: str,
    observed_at_us: int,
    ready_at_us: int,
    phone_safety_state,
) -> _RequestHelperPreparation:
    return _RequestHelperPreparation(
        request_id=request_id,
        request_ticket_id=ticket.ticket_id,
        model=ticket.model,
        base_executor_id=ticket.binding.executor_id,
        preparation_ticket_id=preparation_ticket_id,
        phone_layout_generation=state.generation,
        phone_layout_geometry_sha256=(
            state.layout.geometry_sha256
        ),
        projection_token_sha256=projection_sha256,
        operator_plan_sha256=helper.operator_plan_sha256,
        helper_envelope=helper,
        transition_ids=transition_ids,
        resource_lease_tokens=lease_tokens,
        yielding_resource_ids=yielding_resource_ids,
        memory_owner_id=memory_owner_id,
        started_at_us=observed_at_us,
        ready_at_us=ready_at_us,
        state="TRANSITIONING",
        phone_safety_state=phone_safety_state,
    )


def _preparation_phone_safety_state(
    controller,
    phone_device_id: str | None,
    snapshot: HeterogeneousRuntimeSnapshot,
):
    if (
        phone_device_id is None
        or controller._runtime_capabilities is None
        or phone_device_id not in (
            controller._runtime_capabilities.executor_by_device
        )
    ):
        return None
    return snapshot.executors.get(
        controller._runtime_capabilities.executor_by_device[
            phone_device_id
        ].executor_id
    )


def _defer_preparation_for_stale_session(
    controller,
    request_id: str,
    ticket: RuntimeRequestTicket,
    state: ModelPhoneResidencyLayout,
    snapshot: HeterogeneousRuntimeSnapshot,
    observed_at_us: int,
) -> Mapping[str, object]:
    """Reject the stale proposal and request a fresh phone plan."""

    ready = (
        controller._model_placement_controller
        .reject_phone_layout_proposal(
            state.generation,
            observed_at_us=observed_at_us,
            reason="STALE_PHONE_SESSION_ASSIGNMENT",
        )
    )
    controller._request_helper_preparation_envelopes = {
        key: value
        for key, value in (
            controller._request_helper_preparation_envelopes.items()
        )
        if key[2] != state.generation
    }
    for compiler in (
        controller._automated_route_compiler,
        controller._runtime_epoch_route_compiler,
    ):
        if compiler is not None:
            compiler.set_phone_residency_layout(
                None if ready is None else ready.layout
            )
    replan_status = "REQUESTED"
    try:
        controller._update_phone_residency_portfolio(
            ticket.request,
            controller.runtime_model_manifest(
                ticket.model.model_id
            ),
            observed_at_us,
            snapshot,
        )
    except (
        ModelPlacementControllerError,
        RouteGenerationError,
        RuntimeResidencyCohortError,
        RuntimeResidencyProjectionError,
        UnifiedScheduleError,
        ValueError,
    ) as replan_error:
        replan_status = "FAILED:" + str(replan_error)
    controller._model_placement_controller.record_request_helper_event(
        request_id,
        "PREPARATION_DEFERRED",
        observed_at_us,
        {
            "phone_layout_generation": state.generation,
            "phone_layout_geometry_sha256": (
                state.layout.geometry_sha256
            ),
            "reason": "STALE_PHONE_SESSION_ASSIGNMENT",
            "replan_status": replan_status,
        },
    )
    return MappingProxyType({
        "phone_layout_generation": state.generation,
        "reason": "STALE_PHONE_SESSION_ASSIGNMENT",
        "replan_status": replan_status,
        "status": "DEFERRED",
    })


def _reject_stale_phone_layout_proposal(
    controller,
    request_id: str,
    ticket: RuntimeRequestTicket,
    state: ModelPhoneResidencyLayout,
    snapshot: HeterogeneousRuntimeSnapshot,
    observed_at_us: int,
) -> Mapping[str, object] | None:
    """Drop a PROPOSED layout whose new shards no longer serve arrived demand.

    A proposal outlives the demand that produced it when its owner nears
    completion or finishes before the load starts. Loading it then occupies a
    session for work nobody will decode; the current queue decides instead.
    """
    changed = set(state.layout.changed_session_ids)
    artifacts = {
        shard.artifact_sha256 for shard in state.layout.shards
        if shard.session_id in changed
    }
    if not artifacts:
        return None
    manifest = controller.runtime_model_manifest(ticket.model.model_id)
    demand = controller._phone_queue_demand(ticket.request, manifest)
    minimum = int(controller._adaptive_envelope_minimum_remaining_tokens)
    live = {
        artifact: (
            demand.active_remaining_tokens_by_artifact.get(artifact, 0)
            + demand.queued_output_tokens_by_artifact.get(artifact, 0)
        )
        for artifact in sorted(artifacts)
    }
    if any(tokens >= minimum for tokens in live.values()):
        return None
    target = controller._model_placement_controller.target_phone_layout()
    if target is None or target.generation != state.generation:
        return None
    controller._model_placement_controller.reject_phone_layout_proposal(
        state.generation,
        observed_at_us=observed_at_us,
        reason="PHONE_RESIDENCY_PROPOSAL_STALE",
    )
    replan_status = "REQUESTED"
    try:
        controller._update_phone_residency_portfolio(
            ticket.request, manifest, observed_at_us, snapshot,
        )
    except (
        ModelPlacementControllerError,
        RouteGenerationError,
        RuntimeResidencyCohortError,
        RuntimeResidencyProjectionError,
        UnifiedScheduleError,
        ValueError,
    ) as replan_error:
        replan_status = "FAILED:" + str(replan_error)
    payload = {
        "phone_layout_generation": state.generation,
        "phone_layout_geometry_sha256": state.layout.geometry_sha256,
        "reason": "PHONE_RESIDENCY_PROPOSAL_STALE",
        "live_decode_tokens_by_artifact": live,
        "minimum_remaining_tokens": minimum,
        "replan_status": replan_status,
    }
    controller._model_placement_controller.record_request_helper_event(
        request_id, "REJECTED", observed_at_us, payload,
    )
    return MappingProxyType({**payload, "status": "REJECTED"})


def _defer_preparation_for_memory_capacity(
    controller,
    request_id: str,
    state: ModelPhoneResidencyLayout,
    exc: RuntimeResourceError,
    observed_at_us: int,
) -> Mapping[str, object]:
    controller._model_placement_controller.record_request_helper_event(
        request_id,
        "PREPARATION_DEFERRED",
        observed_at_us,
        {
            "phone_layout_generation": state.generation,
            "phone_layout_geometry_sha256": (
                state.layout.geometry_sha256
            ),
            "reason": "LIVE_MEMORY_CAPACITY_NOT_READY",
            "resource_error": str(exc),
        },
    )
    return MappingProxyType({
        "phone_layout_generation": state.generation,
        "reason": "LIVE_MEMORY_CAPACITY_NOT_READY",
        "status": "DEFERRED",
    })
