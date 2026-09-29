"""HelperEnvelopeMixin refresh operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace

from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot, RuntimeExecutorState
from ..._internal.route_generation import RouteGenerationError
from ..._internal.model_placement_controller import (
    ModelPlacementControllerError,
    ModelPhoneResidencyLayout,
)
from ..._internal.runtime_plan import RuntimePlanError
from ..._internal.runtime_residency_projection import RuntimeResidencyProjectionError
from ..._internal.runtime_residency_cohorts import RuntimeResidencyCohortError
from ..._internal.runtime_controller import RuntimeRequestTicket
from ..._internal.adaptive_decode_contracts import AdaptiveDecodeError
from ..._internal.types import canonical_sha256
from ..common import _RECOVERABLE_ERRORS, _phone_shard_structure


def retained_execution_layer_mask(old, helper, layout) -> int:
    """Evidence reuse requires the exact retained data plane, not a layout number."""
    if old is None:
        return 0
    if any(getattr(old, name) != getattr(helper, name) for name in (
        "artifact_sha256", "desktop_parent_route_id", "desktop_placement_sha256",
        "activation_dtype",
    )):
        return 0
    if old.helper_plan.baseline_executor_id != helper.helper_plan.baseline_executor_id:
        return 0
    if any(getattr(old.helper_binding, name) != getattr(helper.helper_binding, name)
           for name in ("executor_id", "backend", "endpoint", "operator_plan_protocol")):
        return 0
    def participants(binding):
        return tuple((row.executor_id, row.device_id, row.endpoint, row.backend, row.resource_ids)
                     for row in binding.participants)
    if participants(old.helper_binding) != participants(helper.helper_binding):
        return 0
    before = old.helper_plan.execution_contract
    after = helper.helper_plan.execution_contract
    if replace(before, phone_shards=()) != replace(after, phone_shards=()):
        return 0
    old_shards = {row.session_id: row for row in before.phone_shards}
    ready = {row.session_id: row for row in layout.layout.shards}
    mask = 0
    for shard in after.phone_shards:
        source = old_shards.get(shard.session_id)
        actual = ready.get(shard.session_id)
        if (source is not None and actual is not None
                and source == shard and shard.session_generation > 0
                and _phone_shard_structure(shard) == _phone_shard_structure(actual)
                and shard.session_generation == layout.layout.session_generation_by_id.get(
                    shard.session_id)):
            mask |= shard.layer_mask
    return mask


def _ready_helper_configuration(
    controller, ticket: RuntimeRequestTicket, layout: ModelPhoneResidencyLayout,
) -> str:
    return canonical_sha256({
        "ticket_id": ticket.ticket_id,
        "parent_plan_sha256": ticket.execution_plan.plan_sha256,
        "capabilities": controller._runtime_capability_generation_sha256,
        "geometry_sha256": layout.layout.geometry_sha256,
        "session_generations": dict(layout.layout.session_generation_by_id),
        "learning": controller._runtime_placement_learning_generation_sha256(
            ticket.model.artifact_sha256
        ),
    })


def _ready_helper_parent_rejection_unchanged(
    controller, ticket: RuntimeRequestTicket, layout: ModelPhoneResidencyLayout,
) -> bool:
    configurations = tuple(
        row["configuration_sha256"]
        for row in controller._model_placement_controller.request_helper_events(
            ticket.request.request_id
        )
        if row.get("kind") == "HELPER_REMATERIALIZATION_FAILED"
        and "configuration_sha256" in row
    )
    return bool(configurations) and (
        controller._ready_helper_configuration(ticket, layout) in configurations
    )


def _rematerialize_ready_layout_helpers(
    controller,
    layout: ModelPhoneResidencyLayout,
    snapshot: HeterogeneousRuntimeSnapshot,
    observed_at_us: int,
    phone_safety_state: RuntimeExecutorState | None = None,
) -> tuple[str, ...]:
    """Bind exact helper plans once after one layout becomes READY."""

    if layout.state != "READY" or not isinstance(
        snapshot, HeterogeneousRuntimeSnapshot
    ):
        raise UnifiedScheduleError(
            "ready helper rematerialization input is invalid"
        )
    if phone_safety_state is None:
        phone_safety_state = controller._ready_layout_phone_safety_state(
            layout
        )
    covered = {row.artifact_sha256 for row in layout.layout.shards}
    updated = []
    for ticket in controller._runtime_controller.current_tickets():
        request_id = ticket.request.request_id
        if (
            ticket.dispatch_state != "ACQUIRED"
            or ticket.transition_status not in {"COMPLETED", "NOT_REQUIRED"}
            or not controller._model_placement_controller.request_is_acquired(
                request_id
            )
            or ticket.model.artifact_sha256 not in covered
            or ticket.execution_plan is None
            or ticket.execution_plan.desktop_placement_sha256 is None
        ):
            continue
        remaining_opportunity_tokens = (
            controller._model_placement_controller
                .remaining_request_decode_tokens(
                    request_id, ticket.request.output_tokens
                )
        )
        config = getattr(controller, "_adaptive_decode_config", None)
        minimum_remaining = (
            1 if config is not None and config.server_policy_coherence
            else controller._adaptive_envelope_minimum_remaining_tokens
        )
        # READY helpers can serve a shared policy without a request-local probe.
        if remaining_opportunity_tokens < minimum_remaining:
            controller._record_ready_helper_event_once(
                ticket,
                layout,
                "REJECTED",
                observed_at_us,
                accepted=False,
                reason="INSUFFICIENT_REMAINING_OPPORTUNITY",
                remaining_opportunity_tokens=(
                    remaining_opportunity_tokens
                ),
            )
            continue
        controller._record_ready_helper_event_once(
            ticket,
            layout,
            "ELIGIBLE",
            observed_at_us,
            accepted=True,
            remaining_opportunity_tokens=remaining_opportunity_tokens,
        )
        rebind = (
            controller._model_placement_controller
                .request_helper_rebind_state(request_id)
        )
        if (
            rebind is not None
            and rebind.get("target_generation") != layout.generation
        ):
            controller._model_placement_controller.record_request_helper_event(
                request_id,
                "HELPER_REMATERIALIZATION_DEFERRED",
                observed_at_us,
                {
                    "phone_layout_generation": layout.generation,
                    "reason": "REBIND_AWAITS_RETRY_TARGET",
                    "rebind_target_generation": rebind.get(
                        "target_generation"
                    ),
                },
            )
            continue
        if controller._retain_usable_ready_layout_helper(
            ticket, layout, observed_at_us
        ) or controller._retain_materialized_ready_layout_helper(
            ticket, layout, observed_at_us
        ):
            updated.append(request_id)
            continue
        if controller._ready_helper_parent_rejection_unchanged(ticket, layout):
            continue
        try:
            controller._rematerialize_ready_layout_helper(
                ticket,
                layout,
                snapshot,
                observed_at_us,
                phone_safety_state,
            )
        except (
            AdaptiveDecodeError,
            ModelPlacementControllerError,
            RouteGenerationError,
            RuntimePlanError,
            RuntimeResidencyProjectionError,
            RuntimeResidencyCohortError,
            UnifiedScheduleError,
        ) as exc:
            controller._handle_ready_helper_materialization_error(
                ticket, layout, exc, observed_at_us
            )
            continue
        updated.append(request_id)
    return tuple(updated)


def _ready_layout_phone_safety_state(
    controller,
    layout: ModelPhoneResidencyLayout,
) -> RuntimeExecutorState | None:
    exact_preparations = tuple(
        preparation
        for preparation in controller._request_helper_preparations.values()
        if (
            preparation.state == "READY"
            and preparation.phone_layout_generation
                == layout.generation
            and preparation.phone_layout_geometry_sha256
                == layout.layout.geometry_sha256
            and preparation.verification_sha256 is not None
            and preparation.phone_safety_state is not None
        )
    )
    exact_offline_stages = tuple(
        stage
        for plan in getattr(
            controller, "_offline_phone_residency_plans", {}
        ).values()
        for stage in plan.stages
        if (
            stage.state == "READY"
            and stage.layout.layout.geometry_sha256
                == layout.layout.geometry_sha256
            and stage.layout.layout.session_identities
                == layout.layout.session_identities
            and stage.phone_safety_state is not None
        )
    )
    exact = (*exact_preparations, *exact_offline_stages)
    if not exact:
        return None
    return max(
        exact,
        key=lambda row: (
            row.ready_at_us,
            row.started_at_us,
            row.preparation_ticket_id,
        ),
    ).phone_safety_state


def runtime_ready_helper_refresh_needed(
    controller,
    request_id: str,
    *,
    expected_ticket_id: str,
) -> bool:
    ticket = controller.runtime_ticket(request_id)
    if (
        ticket.ticket_id != expected_ticket_id
        or ticket.dispatch_state != "ACQUIRED"
    ):
        return False
    ready = controller._model_placement_controller.ready_phone_layout()
    context = controller._late_request_helper_contexts.get(request_id)
    if context is not None and (
        controller._model_placement_controller.request_helper_layout_is_usable(
            request_id,
            context.helper.phone_layout_generation,
            context.helper.phone_layout_geometry_sha256,
        )
    ):
        return False
    return bool(
        ready is not None
        and ready.covers_artifact(ticket.model.artifact_sha256)
        and not controller._ready_helper_parent_rejection_unchanged(ticket, ready)
    )


def refresh_ready_request_helper(
    controller,
    request_id: str,
    *,
    expected_ticket_id: str,
    observed_at_us: int,
    snapshot: HeterogeneousRuntimeSnapshot,
) -> bool:
    """Materialize the current READY helper for one acquired request."""

    ticket = controller.runtime_ticket(request_id)
    if (
        ticket.ticket_id != expected_ticket_id
        or ticket.dispatch_state != "ACQUIRED"
        or ticket.transition_status not in {
            "COMPLETED", "NOT_REQUIRED"
        }
    ):
        return False
    context = controller._late_request_helper_contexts.get(request_id)
    if (
        context is not None
        and controller._model_placement_controller
            .request_helper_layout_is_usable(
                request_id,
                context.helper.phone_layout_generation,
                context.helper.phone_layout_geometry_sha256,
            )
    ):
        return True
    ready = controller._model_placement_controller.ready_phone_layout()
    if (
        ready is None
        or not ready.covers_artifact(ticket.model.artifact_sha256)
    ):
        return False
    try:
        with controller._transaction(convert=False):
            updated = controller._rematerialize_ready_layout_helpers(
                ready, snapshot, observed_at_us
            )
            return request_id in updated
    except _RECOVERABLE_ERRORS as exc:
        controller._model_placement_controller.record_request_helper_event(
            request_id,
            "READY_HELPER_REFRESH_REJECTED",
            observed_at_us,
            {
                "phone_layout_generation": ready.generation,
                "phone_layout_geometry_sha256": (
                    ready.layout.geometry_sha256
                ),
                "reason": str(exc),
            },
        )
        return False
