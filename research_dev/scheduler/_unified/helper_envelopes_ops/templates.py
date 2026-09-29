"""HelperEnvelopeMixin templates operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace

from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.model_placement_controller import ModelPhoneResidencyLayout
from ..._internal.phone_shards import PhoneFfnResidencyLayout
from ..._internal.runtime_plan import RuntimeHelperExecutionEnvelope
from ..._internal.runtime_controller import RuntimeRequestTicket
from ..common import _StalePhoneSessionAssignment, _phone_shard_structure


def _reusable_phone_helper_template(
    helper: RuntimeHelperExecutionEnvelope,
) -> RuntimeHelperExecutionEnvelope:
    """Remove preparation-transaction state from a reusable helper."""

    if (
        not helper.preparation_changed_session_ids
        and helper.replacement_authorization is None
    ):
        return helper
    return replace(
        helper,
        preparation_changed_session_ids=(),
        replacement_authorization=None,
    )


def _ready_helper_event_values(
    ticket: RuntimeRequestTicket,
    layout: ModelPhoneResidencyLayout,
    *,
    helper: RuntimeHelperExecutionEnvelope | None = None,
    selected_fraction_ppm: int = 0,
    accepted: bool,
    reason: str | None = None,
    remaining_opportunity_tokens: int | None = None,
) -> dict[str, object]:
    session_ids = {
        row.session_id
        for row in layout.layout.shards
        if row.artifact_sha256 == ticket.model.artifact_sha256
    }
    values: dict[str, object] = {
        "accepted": accepted,
        "artifact_sha256": ticket.model.artifact_sha256,
        "desktop_parent_placement_sha256": (
            None
            if ticket.execution_plan is None else
            ticket.execution_plan.desktop_placement_sha256
        ),
        "phone_layout_generation": layout.generation,
        "phone_layout_geometry_sha256": (
            layout.layout.geometry_sha256
        ),
        "request_ticket_id": ticket.ticket_id,
        "selected_fraction_ppm": selected_fraction_ppm,
        "session_generation_by_id": {
            session_id: layout.layout.session_generation_by_id[session_id]
            for session_id in sorted(session_ids)
        },
        "template_layout_generation": layout.generation,
    }
    if helper is not None:
        values["operator_plan_sha256"] = helper.operator_plan_sha256
    if reason is not None:
        values["reason"] = reason
    if remaining_opportunity_tokens is not None:
        values["remaining_opportunity_tokens"] = (
            remaining_opportunity_tokens
        )
    return values


def _record_ready_helper_event_once(
    controller,
    ticket: RuntimeRequestTicket,
    layout: ModelPhoneResidencyLayout,
    kind: str,
    observed_at_us: int,
    *,
    helper: RuntimeHelperExecutionEnvelope | None = None,
    selected_fraction_ppm: int = 0,
    accepted: bool,
    reason: str | None = None,
    remaining_opportunity_tokens: int | None = None,
) -> None:
    values = controller._ready_helper_event_values(
        ticket,
        layout,
        helper=helper,
        selected_fraction_ppm=selected_fraction_ppm,
        accepted=accepted,
        reason=reason,
        remaining_opportunity_tokens=remaining_opportunity_tokens,
    )
    identity = (
        kind,
        values["phone_layout_generation"],
        values["phone_layout_geometry_sha256"],
        values["session_generation_by_id"],
        reason,
    )
    for event in controller._model_placement_controller.request_helper_events(
        ticket.request.request_id
    ):
        if (
            event.get("kind"),
            event.get("phone_layout_generation"),
            event.get("phone_layout_geometry_sha256"),
            event.get("session_generation_by_id"),
            event.get("reason"),
        ) == identity:
            return
    controller._model_placement_controller.record_request_helper_event(
        ticket.request.request_id,
        kind,
        observed_at_us,
        values,
    )


def _authoritative_ready_helper_template(
    controller,
    *,
    artifact_sha256: str,
    desktop_parent_route_id: str,
    desktop_placement_sha256: str,
    baseline_executor_id: str,
    allow_parent_route_rebind: bool = False,
) -> RuntimeHelperExecutionEnvelope | None:
    """Return the exact system-owned helper for the READY session view."""

    ready = controller._model_placement_controller.ready_phone_layout()
    if (
        ready is None
        or ready.state != "READY"
        or not ready.covers_artifact(artifact_sha256)
    ):
        return None
    expected_shards = tuple(
        _phone_shard_structure(row)
        for row in ready.layout.shards
        if row.artifact_sha256 == artifact_sha256
    )

    def exact(envelope: RuntimeHelperExecutionEnvelope) -> bool:
        shards = tuple(
            envelope.helper_plan.execution_contract.phone_shards
        )
        return bool(
            envelope.artifact_sha256 == artifact_sha256
            and (
                allow_parent_route_rebind
                or envelope.desktop_parent_route_id
                    == desktop_parent_route_id
            )
            and envelope.desktop_placement_sha256
                == desktop_placement_sha256
            and envelope.helper_plan.baseline_executor_id
                == baseline_executor_id
            and envelope.phone_layout_geometry_sha256
                == ready.layout.geometry_sha256
            and tuple(_phone_shard_structure(row) for row in shards)
                == expected_shards
            and all(
                row.session_generation
                    == ready.layout.session_generation_by_id.get(
                        row.session_id
                    )
                for row in shards
            )
        )

    candidates = []
    for plan in getattr(controller, "_offline_phone_residency_plans", {}).values():
        for stage in plan.stages:
            if (
                stage.state == "READY"
                and stage.layout.layout.geometry_sha256
                    == ready.layout.geometry_sha256
                and stage.layout.layout.session_identities
                    == ready.layout.session_identities
                and exact(stage.helper_envelope)
            ):
                candidates.append(
                    controller._reusable_phone_helper_template(
                        stage.helper_envelope
                    )
                )
    candidates.extend(
        controller._reusable_phone_helper_template(envelope)
        for envelope in getattr(
            controller, "_phone_helper_endpoint_templates", {}
        ).values()
        if exact(envelope)
    )
    by_plan = {}
    for envelope in candidates:
        previous = by_plan.get(envelope.operator_plan_sha256)
        if (
            previous is not None
            and previous != envelope
            and not (
                allow_parent_route_rebind
                and replace(
                    previous,
                    desktop_parent_route_id=(
                        envelope.desktop_parent_route_id
                    ),
                ) == envelope
            )
        ):
            raise UnifiedScheduleError(
                "READY helper template identity differs"
            )
        by_plan[envelope.operator_plan_sha256] = envelope
    if not by_plan:
        return None
    parameter_sets = [
        dict(envelope.helper_plan.adapter_parameters)
        for envelope in by_plan.values()
    ]
    if any(row != parameter_sets[0] for row in parameter_sets[1:]):
        raise UnifiedScheduleError(
            "READY helper template runtime is ambiguous"
        )
    return max(
        by_plan.values(),
        key=lambda row: (
            row.phone_layout_generation,
            row.operator_plan_sha256,
        ),
    )


def _phone_helper_authorization_layout(
    controller,
    layout: ModelPhoneResidencyLayout,
) -> PhoneFfnResidencyLayout:
    """Bind a delayed proposal to its exact authoritative source."""

    target = layout.layout
    changed = tuple(target.changed_session_ids)
    if len(changed) != 1 or target.replacement_source_identities:
        return target
    placement = controller._model_placement_controller
    source = placement.ready_phone_layout()
    if source is None or source.generation == layout.generation:
        return target
    if source.state != "READY":
        # begin_phone_layout_transition drains the READY source for exactly
        # this target's load; its shards and generations are still the exact
        # pre-transition map, so binding the source is bounded by that load.
        # Any other non-READY source (another transition, a restore) stays a
        # rejection, now naming the state that blocks it.
        preparing = placement.preparing_phone_layout()
        if not (
            source.state == "DRAINING"
            and layout.state == "PREPARING"
            and preparing is not None
            and preparing.generation == layout.generation
        ):
            raise _StalePhoneSessionAssignment(
                "phone helper replacement source is not ready: "
                f"generation {source.generation} is {source.state} while "
                f"target generation {layout.generation} is {layout.state}"
            )
    selected = changed[0]
    source_by_session = {
        row.session_id: row for row in source.layout.shards
    }
    target_by_session = {
        row.session_id: row for row in target.shards
    }
    if selected not in source_by_session:
        return target
    if (
        selected not in target_by_session
        or set(source_by_session) != set(target_by_session)
        or target.session_generation_by_id.get(selected)
            != source.layout.session_generation_by_id.get(selected, 0) + 1
        or any(
            _phone_shard_structure(source_by_session[session_id])
                != _phone_shard_structure(target_by_session[session_id])
            or source.layout.session_generation_by_id.get(session_id)
                != target.session_generation_by_id.get(session_id)
            for session_id in source_by_session
            if session_id != selected
        )
    ):
        raise _StalePhoneSessionAssignment(
            "phone helper replacement source differs from proposal"
        )
    return replace(
        target,
        replacement_source_identities=(
            source.layout.session_identity(selected),
        ),
        replacement_source_resident_bytes_by_session={
            selected: source_by_session[selected].resident_bytes,
        },
    )
