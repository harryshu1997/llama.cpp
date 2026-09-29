"""ModelPlacementController planning operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

from ..phone_shards import PhoneFfnResidencyLayout, progressive_ffn_residency_layouts
from ..model_placement_contracts.common import (
    ModelPlacementControllerError,
    _text,
    _integer,
    _sha256,
)
from ..model_placement_contracts.layout import ModelPhoneResidencyLayout


def _sessions_unchanged(
    source: ModelPhoneResidencyLayout,
    target: ModelPhoneResidencyLayout,
    session_ids: Sequence[str],
) -> bool:
    source_by_session = {
        row.session_id: row for row in source.layout.shards
    }
    target_by_session = {
        row.session_id: row for row in target.layout.shards
    }
    return bool(session_ids) and all(
        session_id in source_by_session
        and session_id in target_by_session
        and source_by_session[session_id].artifact_sha256
            == target_by_session[session_id].artifact_sha256
        and source_by_session[session_id].resident_geometry_sha256
            == target_by_session[session_id]
                .resident_geometry_sha256
        and source_by_session[session_id].operator_plan_sha256
            == target_by_session[session_id].operator_plan_sha256
        for session_id in session_ids
    )


def request_helper_layout_is_usable(
    controller,
    request_id: str,
    generation: int,
    geometry_sha256: str,
) -> bool:
    """Return whether an acquired helper's exact sessions remain ready."""

    request_id = _text("model placement request", request_id)
    generation = _integer("phone layout generation", generation, 1)
    geometry = _sha256(
        "phone layout request geometry", geometry_sha256
    )
    binding = controller._request_bindings.get(request_id)
    if binding is None or request_id not in controller._acquired_request_ids:
        return False
    attachment = binding.helper_attachment
    envelope = binding.helper_envelope
    bound_generation = (
        None
        if attachment is None and envelope is None
        else envelope.phone_layout_generation
        if attachment is None
        else attachment.phone_layout_generation
    )
    bound_geometry = (
        None
        if attachment is None and envelope is None
        else envelope.phone_layout_geometry_sha256
        if attachment is None
        else attachment.phone_layout_geometry_sha256
    )
    identities = (
        ()
        if attachment is None and envelope is None
        else envelope.phone_session_identities
        if attachment is None
        else attachment.phone_session_identities
    )
    if attachment is not None:
        allowed = set(attachment.allowed_session_ids)
        identities = tuple(
            row for row in identities
            if row.session_id in allowed
        )
    if (
        bound_generation != generation
        or bound_geometry != geometry
        or not identities
    ):
        return False
    return controller._session_identities_are_ready(identities)


def _collect_drained_phone_layouts(controller) -> None:
    referenced = {
        generation
        for binding in controller._request_bindings.values()
        for generation in (
            None
            if binding.helper_attachment is None
            else binding.helper_attachment.phone_layout_generation,
            None
            if binding.helper_envelope is None
            else binding.helper_envelope.phone_layout_generation,
        )
        if generation is not None
    }
    protected = {
        value for value in (
            controller._ready_phone_layout_generation,
            controller._target_phone_layout_generation,
        )
        if value is not None
    }
    for generation, layout in tuple(controller._phone_layouts.items()):
        if (
            layout.state == "DRAINING"
            and generation not in referenced
            and generation not in protected
        ):
            controller._phone_layouts.pop(generation, None)


def _restore_ready_phone_layout(controller) -> ModelPhoneResidencyLayout | None:
    ready = controller.ready_phone_layout()
    if ready is not None and ready.state == "DRAINING":
        ready = replace(ready, state="READY")
        controller._phone_layouts[ready.generation] = ready
    return ready


def reject_phone_layout_proposal(
    controller,
    generation: int,
    *,
    observed_at_us: int,
    reason: str,
) -> ModelPhoneResidencyLayout | None:
    """Discard one stale proposal without changing physical residency."""

    generation = _integer(
        "phone layout rejected generation", generation, 1
    )
    _integer("phone layout rejection time", observed_at_us)
    reason = _text("phone layout rejection reason", reason)
    target = controller._phone_layouts.get(generation)
    if (
        target is None
        or controller._target_phone_layout_generation != generation
        or target.state != "PROPOSED"
    ):
        raise ModelPlacementControllerError(
            "phone layout rejection proposal is stale"
        )
    del controller._phone_layouts[generation]
    controller._target_phone_layout_generation = None
    controller._phone_preload_layouts = ()
    ready = controller._restore_ready_phone_layout()
    controller._record_phone_layout_event(
        "PROPOSAL_REJECTED",
        observed_at_us,
        {
            "phone_layout_generation": generation,
            "phone_layout_geometry_sha256": (
                target.layout.geometry_sha256
            ),
            "reason": reason,
        },
    )
    return ready


def _carried_replacement_sources(
    controller,
    proposed: PhoneFfnResidencyLayout,
    layout: PhoneFfnResidencyLayout,
) -> PhoneFfnResidencyLayout:
    """Keep a PROPOSED target's stamped replacement authority on update.

    A same-geometry re-proposal arrives unstamped.  Dropping the source
    identities made every helper re-derive the source from the READY layout,
    which begin_phone_layout_transition marks DRAINING for the whole load,
    so each waiting request polled "phone helper replacement source is not
    ready" until the target itself was READY.  The sources are carried only
    while they still name the exact, READY physical sessions, and only when
    the owner opted in (resident-model re-provisioning, whose follow-the-
    desktop swaps are re-proposed at every boundary).  Without the opt-in
    the recorded proposal JSON is unchanged and the helper side alone
    (templates.py accepts the source a load drains for itself) ends the
    rejection loop.
    """

    if (
        not getattr(controller, "carry_replacement_sources_on_update", False)
        or layout.replacement_source_identities
        or not proposed.replacement_source_identities
        or layout.changed_session_ids != proposed.changed_session_ids
        or dict(layout.session_generation_by_id)
            != dict(proposed.session_generation_by_id)
    ):
        return layout
    for identity in proposed.replacement_source_identities:
        state = controller._phone_session_states.get(identity.session_id)
        if (
            state is None
            or state.state != "READY"
            or not state.matches(identity)
        ):
            return layout
    return replace(
        layout,
        replacement_source_identities=(
            proposed.replacement_source_identities
        ),
        replacement_source_resident_bytes_by_session=dict(
            proposed.replacement_source_resident_bytes_by_session
        ),
    )


def propose_phone_layout(
    controller,
    layout: PhoneFfnResidencyLayout,
    *,
    workspace_bytes: int,
    shared_compute_resource_id: str,
    shared_transport_resource_ids: Sequence[str],
    observed_at_us: int,
    selection_reason: str | None = None,
    queue_work_by_artifact: Mapping[str, int] | None = None,
    queue_benefit_uj: int | None = None,
    transition_cost_uj: int | None = None,
    switching_margin_uj: int | None = None,
    minimum_residency_us: int | None = None,
    force: bool = False,
    progressive: bool = False,
) -> ModelPhoneResidencyLayout:
    """Publish one structural proposal without claiming residency."""

    if not isinstance(layout, PhoneFfnResidencyLayout):
        raise ModelPlacementControllerError(
            "phone layout proposal is invalid"
        )
    _integer("phone layout proposal workspace", workspace_bytes)
    _integer("phone layout proposal time", observed_at_us)
    if type(force) is not bool:
        raise ModelPlacementControllerError(
            "phone layout proposal force flag is invalid"
        )
    if type(progressive) is not bool:
        raise ModelPlacementControllerError(
            "phone layout progressive flag is invalid"
        )
    if (
        progressive
        and controller.ready_phone_layout() is None
        and controller.target_phone_layout() is None
        and len(layout.shards) > 1
    ):
        controller._phone_preload_layouts = progressive_ffn_residency_layouts(
            layout
        )
        layout = controller._phone_preload_layouts[0]
        if queue_benefit_uj is not None:
            queue_benefit_uj = layout.queue_benefit
        if transition_cost_uj is not None:
            transition_cost_uj = layout.transition_cost
    residency_interval_us = (
        controller.policy.phone_minimum_residency_us
        if minimum_residency_us is None
        else _integer(
            "phone layout minimum residency interval",
            minimum_residency_us,
        )
    )
    selection_values = {
        "selection_reason": selection_reason,
        "queue_work_by_artifact": tuple(sorted(
            (queue_work_by_artifact or {}).items()
        )),
        "queue_benefit_uj": queue_benefit_uj,
        "transition_cost_uj": transition_cost_uj,
        "switching_margin_uj": switching_margin_uj,
    }
    target = controller.target_phone_layout()
    if target is not None:
        if target.layout.geometry_sha256 == layout.geometry_sha256:
            if target.state == "PROPOSED":
                layout = _carried_replacement_sources(
                    controller,
                    target.layout,
                    layout.with_session_generations(
                        target.layout.session_generation_by_id
                    ),
                )
                target = replace(
                    target,
                    layout=layout,
                    session_identities=layout.session_identities,
                    minimum_residency_interval_us=(
                        residency_interval_us
                    ),
                    minimum_resident_until_us=(
                        target.proposed_at_us + residency_interval_us
                    ),
                    resident_component_identity_sha256="",
                    **selection_values,
                )
                controller._phone_layouts[target.generation] = target
                controller._record_phone_layout_event(
                    "PROPOSAL_UPDATED",
                    observed_at_us,
                    target.to_json(),
                )
            return target
        if target.state == "PREPARING":
            if force:
                controller._phone_preload_layouts = ()
                controller._restore_phone_session_replacement(
                    target.generation,
                    observed_at_us=observed_at_us,
                    reason="PHONE_SESSION_AVAILABILITY_CHANGED",
                )
                controller._phone_layouts.pop(target.generation, None)
                controller._target_phone_layout_generation = None
                controller._restore_ready_phone_layout()
                controller._record_phone_layout_event(
                    "TRANSITION_INVALIDATED",
                    observed_at_us,
                    {
                        "active_generation": target.generation,
                        "active_geometry_sha256": (
                            target.layout.geometry_sha256
                        ),
                        "proposed_geometry_sha256": (
                            layout.geometry_sha256
                        ),
                        "reason": "PHONE_SESSION_AVAILABILITY_CHANGED",
                    },
                )
                target = None
            else:
                controller._record_phone_layout_event(
                    "PROPOSAL_DEFERRED",
                    observed_at_us,
                    {
                        "active_generation": target.generation,
                        "active_geometry_sha256": (
                            target.layout.geometry_sha256
                        ),
                        "proposed_geometry_sha256": layout.geometry_sha256,
                        "reason": "PHONE_LAYOUT_TRANSITION_INFLIGHT",
                    },
                )
                return target
        if target is not None:
            del controller._phone_layouts[target.generation]
            controller._target_phone_layout_generation = None
    ready = controller._restore_ready_phone_layout()
    if (
        ready is not None
        and ready.layout.geometry_sha256 == layout.geometry_sha256
    ):
        return ready
    blocked_sessions = tuple(sorted(
        session_id
        for session_id in layout.changed_session_ids
        if (
            (state := controller._phone_session_states.get(session_id))
                is not None
            and state.state == "READY"
            and observed_at_us < state.minimum_resident_until_us
        )
    ))
    if blocked_sessions and not force:
        controller._record_phone_layout_event(
            "PROPOSAL_DEFERRED",
            observed_at_us,
            {
                "active_generation": ready.generation,
                "active_geometry_sha256": (
                    ready.layout.geometry_sha256
                ),
                "proposed_geometry_sha256": layout.geometry_sha256,
                "blocked_session_ids": list(blocked_sessions),
                "reason": "PHONE_SESSION_MINIMUM_RESIDENCY",
            },
        )
        return ready
    layout = controller._stamp_phone_layout(layout)
    if controller._pending_phone_layout_geometry_sha256 \
            == layout.geometry_sha256:
        controller._pending_phone_layout_geometry_sha256 = None
        controller._pending_phone_layout_snapshot_sha256 = None
        controller._pending_phone_layout_snapshot_count = 0
        controller._pending_phone_layout_sampled_at_us = None
    controller._phone_layout_generation += 1
    proposed = ModelPhoneResidencyLayout(
        generation=controller._phone_layout_generation,
        state="PROPOSED",
        layout=layout,
        workspace_bytes=workspace_bytes,
        shared_compute_resource_id=shared_compute_resource_id,
        shared_transport_resource_ids=tuple(
            shared_transport_resource_ids
        ),
        proposed_at_us=observed_at_us,
        minimum_resident_until_us=(
            observed_at_us + residency_interval_us
        ),
        minimum_residency_interval_us=residency_interval_us,
        session_identities=layout.session_identities,
        **selection_values,
    )
    controller._phone_layouts[proposed.generation] = proposed
    controller._target_phone_layout_generation = proposed.generation
    controller._record_phone_layout_event(
        "PROPOSED",
        observed_at_us,
        proposed.to_json(),
    )
    return proposed


def phone_preload_inflight(controller) -> bool:
    """An already selected cold superset is still being published."""

    return bool(controller._phone_preload_layouts)


def _advance_phone_preload(
    controller, ready: ModelPhoneResidencyLayout, observed_at_us: int
) -> None:
    if not controller._phone_preload_layouts:
        return
    if (
        ready.layout.geometry_sha256
        != controller._phone_preload_layouts[0].geometry_sha256
    ):
        raise ModelPlacementControllerError(
            "phone preload completion differs from selected stage"
        )
    controller._phone_preload_layouts = controller._phone_preload_layouts[1:]
    if not controller._phone_preload_layouts:
        return
    selected = controller._phone_preload_layouts[0]
    controller.propose_phone_layout(
        selected,
        workspace_bytes=ready.workspace_bytes,
        shared_compute_resource_id=ready.shared_compute_resource_id,
        shared_transport_resource_ids=ready.shared_transport_resource_ids,
        observed_at_us=observed_at_us,
        selection_reason=ready.selection_reason,
        queue_work_by_artifact=dict(ready.queue_work_by_artifact),
        queue_benefit_uj=(
            None if ready.queue_benefit_uj is None
            else selected.queue_benefit
        ),
        transition_cost_uj=(
            None if ready.transition_cost_uj is None
            else selected.transition_cost
        ),
        switching_margin_uj=ready.switching_margin_uj,
        minimum_residency_us=ready.minimum_residency_interval_us,
    )


def _restore_phone_session_replacement(
    controller,
    generation: int,
    *,
    observed_at_us: int,
    reason: str,
    restored_session_generations: Mapping[str, int] | None = None,
) -> None:
    restored_generations = dict(restored_session_generations or {})
    sources = controller._phone_session_replacement_sources.pop(
        generation, None
    )
    target = controller._phone_layouts.get(generation)
    if sources is None or target is None:
        return
    if set(restored_generations) - set(
        target.layout.changed_session_ids
    ) or any(
        type(value) is not int or value < 1
        for value in restored_generations.values()
    ):
        raise ModelPlacementControllerError(
            "restored phone session generations are invalid"
        )
    for session_id in target.layout.changed_session_ids:
        source = sources.get(session_id)
        if source is None:
            controller._phone_session_states.pop(session_id, None)
            continue
        if source.state == "EMPTY":
            if session_id in restored_generations:
                raise ModelPlacementControllerError(
                    "empty phone session cannot carry a restored epoch"
                )
            controller._phone_session_states[session_id] = source
            controller._record_phone_layout_event(
                "SESSION_RESTORED_EMPTY",
                observed_at_us,
                {
                    "layout_generation": generation,
                    "reason": reason,
                    "session": source.to_json(),
                },
            )
            continue
        restored = replace(
            source,
            state="READY",
            session_generation=restored_generations.get(
                session_id, source.session_generation
            ),
        )
        controller._phone_session_states[session_id] = restored
        controller._phone_session_generation_by_id[session_id] = (
            restored.session_generation
        )
        controller._record_phone_layout_event(
            "SESSION_RESTORED",
            observed_at_us,
            {
                "layout_generation": generation,
                "reason": reason,
                "session": restored.to_json(),
            },
        )
    if restored_generations:
        ready = controller.ready_phone_layout()
        if ready is None:
            raise ModelPlacementControllerError(
                "restored phone layout authority is absent"
            )
        generation_by_id = dict(
            ready.layout.session_generation_by_id
        )
        for session_id, physical_generation in (
            restored_generations.items()
        ):
            if session_id not in generation_by_id:
                raise ModelPlacementControllerError(
                    "restored phone session is absent from source layout"
                )
            generation_by_id[session_id] = physical_generation
        # Compensation retires the old forward transition's source epoch.
        restored_layout = replace(
            ready.layout.with_session_generations(generation_by_id),
            replacement_source_identities=(),
            replacement_source_resident_bytes_by_session={},
        )
        controller._phone_layouts[ready.generation] = replace(
            ready,
            layout=restored_layout,
            session_identities=restored_layout.session_identities,
            resident_component_identity_sha256="",
        )
    controller._sync_phone_session_references()
