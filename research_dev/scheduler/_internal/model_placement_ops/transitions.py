"""ModelPlacementController transitions operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

from ..phone_shards import PhoneFfnResidencyLayout
from ..types import canonical_sha256
from ..model_placement_contracts.common import (
    ModelPlacementControllerError,
    _text,
    _integer,
    _sha256,
    _identities,
)
from ..model_placement_contracts.layout import ModelPhoneResidencyLayout


def begin_phone_layout_transition(
    controller,
    generation: int,
    *,
    ticket_id: str,
    transition_ids: Sequence[str],
    ready_at_us: int,
    projection_token_sha256: str,
    workspace_bytes: int,
    observed_at_us: int,
) -> ModelPhoneResidencyLayout:
    generation = _integer(
        "phone layout transition generation", generation, 1
    )
    target = controller._phone_layouts.get(generation)
    if (
        target is None
        or generation != controller._target_phone_layout_generation
        or target.state != "PROPOSED"
    ):
        raise ModelPlacementControllerError(
            "phone layout transition proposal is stale"
        )
    blockers = controller.phone_layout_transition_blockers(generation)
    if blockers:
        raise ModelPlacementControllerError(
            "phone layout transition has admitted blockers: "
            + ",".join(blockers)
        )
    controller._sync_phone_session_references()
    target_by_session = {
        row.session_id: row for row in target.layout.shards
    }
    target_identity_by_session = {
        row.session_id: row for row in target.session_identities
    }
    replacement_sources = {
        session_id: state
        for session_id in target.layout.changed_session_ids
        if (
            state := controller._phone_session_states.get(session_id)
        ) is not None
    }
    if any(
        state.state not in {"EMPTY", "READY"}
        or (
            state.state == "READY"
            and state.active_helper_references
        )
        for state in replacement_sources.values()
    ):
        raise ModelPlacementControllerError(
            "phone session replacement source is not drained"
        )
    controller._phone_session_replacement_sources[generation] = (
        replacement_sources
    )
    for session_id in target.layout.changed_session_ids:
        source = replacement_sources.get(session_id)
        if source is not None and source.state == "READY":
            draining = replace(source, state="DRAINING")
            controller._phone_session_states[session_id] = draining
            controller._record_phone_layout_event(
                "SESSION_DRAINING",
                observed_at_us,
                {
                    "layout_generation": generation,
                    "session": draining.to_json(),
                },
            )
        shard = target_by_session.get(session_id)
        identity = target_identity_by_session.get(session_id)
        if shard is None or identity is None:
            continue
        loading = controller._session_state_from_shard(
            shard,
            identity,
            state="LOADING",
            minimum_resident_until_us=(
                target.minimum_resident_until_us
            ),
            replacement_cost_uj=(
                target.layout.transition_cost_by_session.get(
                    session_id, 0
                )
            ),
        )
        controller._phone_session_states[session_id] = loading
        controller._record_phone_layout_event(
            "SESSION_LOADING",
            observed_at_us,
            {
                "layout_generation": generation,
                "session": loading.to_json(),
            },
        )
    preparing = replace(
        target,
        state="PREPARING",
        workspace_bytes=_integer(
            "phone layout transition workspace", workspace_bytes
        ),
        resident_component_identity_sha256="",
        transition_ticket_id=_text(
            "phone layout transition ticket", ticket_id
        ),
        transition_ids=tuple(transition_ids),
        ready_at_us=_integer(
            "phone layout transition ready time", ready_at_us
        ),
        projection_token_sha256=_sha256(
            "phone layout projection token",
            projection_token_sha256,
        ),
    )
    ready = controller.ready_phone_layout()
    draining = None
    if ready is not None and ready.generation != generation:
        if ready.state != "READY":
            raise ModelPlacementControllerError(
                "phone layout replacement source is not ready"
            )
        ready_session_ids = {
            row.session_id for row in ready.layout.shards
        }
        if ready_session_ids.intersection(
            target.layout.changed_session_ids
        ):
            draining = replace(ready, state="DRAINING")
    if draining is not None:
        controller._phone_layouts[draining.generation] = draining
    controller._phone_layouts[generation] = preparing
    controller._record_phone_layout_event(
        "PREPARING",
        observed_at_us,
        preparing.to_json(),
    )
    return preparing


def complete_phone_layout_transition(
    controller,
    *,
    generation: int,
    ticket_id: str,
    transition_ids: Sequence[str],
    geometry_sha256: str,
    projection_token_sha256: str,
    finished_at_us: int,
) -> ModelPhoneResidencyLayout | None:
    generation = _integer(
        "phone layout completion generation", generation, 1
    )
    token_sha256 = _sha256(
        "phone layout completion projection token",
        projection_token_sha256,
    )
    geometry_sha256 = _sha256(
        "phone layout completion geometry", geometry_sha256
    )
    transition_ids = tuple(sorted(
        _text("phone layout completion transition", value)
        for value in transition_ids
    ))
    target = controller.target_phone_layout()
    if target is None or target.generation != generation:
        controller._record_phone_layout_event(
            "STALE_PREPARATION_COMPLETION_IGNORED",
            finished_at_us,
            {
                "generation": generation,
                "geometry_sha256": geometry_sha256,
                "projection_token_sha256": token_sha256,
                "ticket_id": _text(
                    "phone layout completion ticket", ticket_id
                ),
                "transition_ids": list(transition_ids),
            },
        )
        return None
    if (
        target.state != "PREPARING"
        or target.transition_ticket_id != ticket_id
        or target.transition_ids != transition_ids
        or target.layout.geometry_sha256 != geometry_sha256
        or target.projection_token_sha256 != token_sha256
    ):
        raise ModelPlacementControllerError(
            "phone layout completion differs from transition"
        )
    target_by_session = {
        row.session_id: row for row in target.layout.shards
    }
    target_identity_by_session = {
        row.session_id: row for row in target.session_identities
    }
    sources = controller._phone_session_replacement_sources.get(generation)
    if sources is None:
        raise ModelPlacementControllerError(
            "phone session replacement authority is absent"
        )
    for session_id in target.layout.changed_session_ids:
        shard = target_by_session.get(session_id)
        identity = target_identity_by_session.get(session_id)
        state = controller._phone_session_states.get(session_id)
        if shard is None:
            if state is None or state.state != "DRAINING":
                raise ModelPlacementControllerError(
                    "removed phone session is not draining"
                )
            continue
        if (
            identity is None
            or state is None
            or state.state != "LOADING"
            or not state.matches(identity)
        ):
            raise ModelPlacementControllerError(
                "phone session completion differs from loading state"
            )
        verified = replace(state, state="VERIFIED")
        controller._phone_session_states[session_id] = verified
        controller._record_phone_layout_event(
            "SESSION_VERIFIED",
            finished_at_us,
            {
                "layout_generation": generation,
                "session": verified.to_json(),
            },
        )
    ready = replace(
        target,
        state="READY",
        ready_at_us=_integer(
            "phone layout completion time", finished_at_us
        ),
        minimum_resident_until_us=(
            finished_at_us
            + target.minimum_residency_interval_us
        ),
    )
    for session_id in target.layout.changed_session_ids:
        shard = target_by_session.get(session_id)
        identity = target_identity_by_session.get(session_id)
        if shard is None or identity is None:
            controller._phone_session_states.pop(session_id, None)
            continue
        session_ready = controller._session_state_from_shard(
            shard,
            identity,
            state="READY",
            minimum_resident_until_us=(
                ready.minimum_resident_until_us
            ),
            replacement_cost_uj=(
                target.layout.transition_cost_by_session.get(
                    session_id, 0
                )
            ),
        )
        controller._phone_session_states[session_id] = session_ready
        controller._phone_session_generation_by_id[session_id] = (
            identity.session_generation
        )
        controller._record_phone_layout_event(
            "SESSION_READY",
            finished_at_us,
            {
                "layout_generation": generation,
                "session": session_ready.to_json(),
            },
        )
    controller._phone_session_replacement_sources.pop(generation, None)
    previous_ready = controller.ready_phone_layout()
    if (
        previous_ready is not None
        and previous_ready.generation != ready.generation
    ):
        previous_ready = replace(previous_ready, state="DRAINING")
        controller._phone_layouts[previous_ready.generation] = previous_ready
    controller._phone_layouts[ready.generation] = ready
    controller._ready_phone_layout_generation = ready.generation
    controller._target_phone_layout_generation = None
    controller._sync_phone_session_references()
    controller._collect_drained_phone_layouts()
    controller._record_phone_layout_event(
        "READY", finished_at_us, ready.to_json()
    )
    controller._advance_phone_preload(ready, finished_at_us)
    return ready


def verify_observed_phone_layout(
    controller,
    generation: int,
    *,
    workspace_bytes: int,
    verified_at_us: int,
    verification_sha256: str,
) -> ModelPhoneResidencyLayout | None:
    """Adopt an already-hot layout proven by a runtime snapshot."""

    generation = _integer(
        "observed phone layout generation", generation, 1
    )
    target = controller._phone_layouts.get(generation)
    if (
        target is None
        or generation != controller._target_phone_layout_generation
    ):
        controller._record_phone_layout_event(
            "STALE_OBSERVED_READINESS_IGNORED",
            verified_at_us,
            {
                "generation": generation,
                "verification_sha256": _sha256(
                    "observed phone layout verification",
                    verification_sha256,
                ),
            },
        )
        return None
    if target.state != "PROPOSED":
        raise ModelPlacementControllerError(
            "observed phone layout proposal is not pending"
        )
    blockers = controller.phone_layout_transition_blockers(generation)
    if blockers:
        raise ModelPlacementControllerError(
            "observed phone layout has admitted blockers: "
            + ",".join(blockers)
        )
    ready = replace(
        target,
        state="READY",
        workspace_bytes=_integer(
            "observed phone layout workspace", workspace_bytes
        ),
        resident_component_identity_sha256="",
        ready_at_us=_integer(
            "observed phone layout verification time", verified_at_us
        ),
        minimum_resident_until_us=(
            verified_at_us
            + target.minimum_residency_interval_us
        ),
        verification_sha256=_sha256(
            "observed phone layout verification",
            verification_sha256,
        ),
    )
    target_by_session = {
        row.session_id: row for row in target.layout.shards
    }
    identity_by_session = {
        row.session_id: row for row in target.session_identities
    }
    for session_id in target.layout.changed_session_ids:
        shard = target_by_session.get(session_id)
        identity = identity_by_session.get(session_id)
        if shard is None or identity is None:
            controller._phone_session_states.pop(session_id, None)
            continue
        verified = controller._session_state_from_shard(
            shard,
            identity,
            state="VERIFIED",
            minimum_resident_until_us=(
                ready.minimum_resident_until_us
            ),
            replacement_cost_uj=(
                target.layout.transition_cost_by_session.get(
                    session_id, 0
                )
            ),
        )
        controller._phone_session_states[session_id] = verified
        controller._record_phone_layout_event(
            "SESSION_VERIFIED_FROM_OBSERVATION",
            verified_at_us,
            {
                "layout_generation": generation,
                "session": verified.to_json(),
            },
        )
        session_ready = controller._session_state_from_shard(
            shard,
            identity,
            state="READY",
            minimum_resident_until_us=(
                ready.minimum_resident_until_us
            ),
            replacement_cost_uj=(
                target.layout.transition_cost_by_session.get(
                    session_id, 0
                )
            ),
        )
        controller._phone_session_states[session_id] = session_ready
        controller._phone_session_generation_by_id[session_id] = (
            identity.session_generation
        )
        controller._record_phone_layout_event(
            "SESSION_READY_FROM_OBSERVATION",
            verified_at_us,
            {
                "layout_generation": generation,
                "session": session_ready.to_json(),
            },
        )
    previous_ready = controller.ready_phone_layout()
    if previous_ready is not None:
        previous_ready = replace(previous_ready, state="DRAINING")
        controller._phone_layouts[previous_ready.generation] = previous_ready
    controller._phone_layouts[ready.generation] = ready
    controller._ready_phone_layout_generation = ready.generation
    controller._target_phone_layout_generation = None
    controller._sync_phone_session_references()
    controller._collect_drained_phone_layouts()
    controller._record_phone_layout_event(
        "READY_FROM_OBSERVATION", verified_at_us, ready.to_json()
    )
    controller._advance_phone_preload(ready, verified_at_us)
    return ready


def adopt_verified_phone_layout(
    controller,
    layout: PhoneFfnResidencyLayout,
    *,
    workspace_bytes: int,
    shared_compute_resource_id: str,
    shared_transport_resource_ids: Sequence[str],
    verified_at_us: int,
    verification_sha256: str,
    selection_reason: str | None = None,
) -> ModelPhoneResidencyLayout:
    """Hydrate exact physical session epochs after a scheduler restart."""

    if not isinstance(layout, PhoneFfnResidencyLayout):
        raise ModelPlacementControllerError(
            "observed phone adoption layout is invalid"
        )
    generations = dict(layout.session_generation_by_id)
    if (
        set(generations) != {row.session_id for row in layout.shards}
        or any(value < 1 for value in generations.values())
    ):
        raise ModelPlacementControllerError(
            "observed phone adoption epochs are incomplete"
        )
    current = controller.ready_phone_layout()
    if current is not None:
        if current.layout == layout:
            return current
        raise ModelPlacementControllerError(
            "observed phone adoption conflicts with ready residency"
        )
    if controller.target_phone_layout() is not None:
        raise ModelPlacementControllerError(
            "observed phone adoption conflicts with a transition"
        )
    verified_at_us = _integer(
        "observed phone adoption time", verified_at_us
    )
    verification_sha256 = _sha256(
        "observed phone adoption verification", verification_sha256
    )
    for shard in layout.shards:
        existing = controller._phone_session_states.get(shard.session_id)
        if (
            existing is not None
            and existing.state != "EMPTY"
            and (
                not controller._session_shard_matches(existing, shard)
                or existing.session_generation
                    != generations[shard.session_id]
            )
        ):
            raise ModelPlacementControllerError(
                "observed phone adoption session conflicts"
            )
    controller._phone_layout_generation += 1
    selection_values = {}
    if selection_reason is not None:
        selection_values = {
            "selection_reason": selection_reason,
            "queue_work_by_artifact": tuple(sorted(
                layout.queued_work_by_artifact.items()
            )),
            "queue_benefit_uj": layout.queue_benefit,
            "transition_cost_uj": 0,
            "switching_margin_uj": 0,
        }
    ready = ModelPhoneResidencyLayout(
        generation=controller._phone_layout_generation,
        state="READY",
        layout=layout,
        workspace_bytes=_integer(
            "observed phone adoption workspace", workspace_bytes
        ),
        shared_compute_resource_id=shared_compute_resource_id,
        shared_transport_resource_ids=tuple(
            shared_transport_resource_ids
        ),
        proposed_at_us=verified_at_us,
        minimum_resident_until_us=verified_at_us,
        minimum_residency_interval_us=0,
        session_identities=layout.session_identities,
        ready_at_us=verified_at_us,
        verification_sha256=verification_sha256,
        **selection_values,
    )
    for shard in layout.shards:
        identity = layout.session_identity(shard.session_id)
        verified = controller._session_state_from_shard(
            shard,
            identity,
            state="VERIFIED",
            minimum_resident_until_us=verified_at_us,
            replacement_cost_uj=0,
        )
        controller._phone_session_states[shard.session_id] = verified
        controller._record_phone_layout_event(
            "SESSION_VERIFIED_FROM_OBSERVATION",
            verified_at_us,
            {
                "layout_generation": ready.generation,
                "session": verified.to_json(),
            },
        )
        session_ready = replace(verified, state="READY")
        controller._phone_session_states[shard.session_id] = session_ready
        controller._phone_session_generation_by_id[shard.session_id] = (
            identity.session_generation
        )
        controller._record_phone_layout_event(
            "SESSION_READY_FROM_OBSERVATION",
            verified_at_us,
            {
                "layout_generation": ready.generation,
                "session": session_ready.to_json(),
            },
        )
    controller._phone_layouts[ready.generation] = ready
    controller._ready_phone_layout_generation = ready.generation
    controller._target_phone_layout_generation = None
    controller._record_phone_layout_event(
        "READY_FROM_OBSERVATION", verified_at_us, ready.to_json()
    )
    return ready


def fail_phone_layout_transition(
    controller,
    ticket_id: str,
    *,
    generation: int,
    projection_token_sha256: str,
    failed_at_us: int,
    reason: str,
    unavailable_session_ids: Sequence[str] = (),
    restored_session_generations: Mapping[str, int] | None = None,
) -> bool:
    generation = _integer(
        "phone layout failure generation", generation, 1
    )
    ticket_id = _text("phone layout failure ticket", ticket_id)
    token_sha256 = _sha256(
        "phone layout failure projection token",
        projection_token_sha256,
    )
    failure_reason = _text("phone layout failure reason", reason)
    unavailable = _identities(
        "phone layout unavailable session", unavailable_session_ids
    )
    restored_generations = dict(
        restored_session_generations or {}
    )
    target = controller.target_phone_layout()
    if (
        target is None
        or target.generation != generation
        or target.state != "PREPARING"
        or target.transition_ticket_id != ticket_id
        or target.projection_token_sha256 != token_sha256
    ):
        controller._record_phone_layout_event(
            "STALE_PREPARATION_FAILURE_IGNORED",
            failed_at_us,
            {
                "generation": generation,
                "projection_token_sha256": token_sha256,
                "reason": _text(
                    "phone layout failure reason", reason
                ),
                "ticket_id": ticket_id,
            },
        )
        return False
    if (
        set(restored_generations) - set(
            target.layout.changed_session_ids
        )
        or set(restored_generations) & set(unavailable)
        or any(
            type(value) is not int
            or value
                != target.layout.session_generation_by_id.get(
                    session_id, -2
                ) + 1
            for session_id, value in restored_generations.items()
        )
    ):
        raise ModelPlacementControllerError(
            "restored phone session epoch differs from transition"
        )
    for session_id in target.layout.changed_session_ids:
        state = controller._phone_session_states.get(session_id)
        if state is None:
            continue
        failed = replace(
            state,
            state="FAILED",
            active_helper_references=(),
        )
        controller._phone_session_states[session_id] = failed
        controller._record_phone_layout_event(
            "SESSION_FAILED",
            failed_at_us,
            {
                "layout_generation": generation,
                "reason": failure_reason,
                "session": failed.to_json(),
            },
        )
    controller._restore_phone_session_replacement(
        generation,
        observed_at_us=failed_at_us,
        reason=failure_reason,
        restored_session_generations=restored_generations,
    )
    controller._phone_preload_layouts = ()
    if set(unavailable) - set(target.layout.changed_session_ids):
        raise ModelPlacementControllerError(
            "phone layout unavailable sessions differ from the transition"
        )
    for session_id in unavailable:
        state = controller._phone_session_states.get(session_id)
        if state is None:
            continue
        # Physical restoration was not proven: the session must never
        # be published as READY again until it is reloaded.
        lost = replace(
            state,
            state="UNAVAILABLE",
            active_helper_references=(),
        )
        controller._phone_session_states[session_id] = lost
        controller._record_phone_layout_event(
            "SESSION_UNAVAILABLE",
            failed_at_us,
            {
                "layout_generation": generation,
                "reason": failure_reason,
                "session": lost.to_json(),
            },
        )
    controller._phone_layouts.pop(target.generation, None)
    controller._target_phone_layout_generation = None
    controller._restore_ready_phone_layout()
    controller._sync_phone_session_references()
    previous_failures = sum(
        row.get("kind") == "TRANSITION_FAILED"
        and row.get("failed_geometry_sha256")
            == target.layout.geometry_sha256
        for row in controller._phone_layout_events
    )
    controller._pending_phone_layout_sampled_at_us = None
    if not unavailable and previous_failures == 0:
        controller._pending_phone_layout_geometry_sha256 = (
            target.layout.geometry_sha256
        )
        controller._pending_phone_layout_snapshot_sha256 = canonical_sha256({
            "failed_generation": target.generation,
            "restored_session_generations": dict(sorted(
                restored_generations.items()
            )),
            "schema": "phone-layout-restored-retry-snapshot-v1",
        })
        controller._pending_phone_layout_snapshot_count = max(
            1,
            controller.policy.phone_layout_confirmation_snapshots - 1,
        )
    else:
        controller._pending_phone_layout_geometry_sha256 = None
        controller._pending_phone_layout_snapshot_sha256 = None
        controller._pending_phone_layout_snapshot_count = 0
    controller._record_phone_layout_event(
        "TRANSITION_FAILED",
        failed_at_us,
        {
            "failed_generation": target.generation,
            "failed_geometry_sha256": (
                target.layout.geometry_sha256
            ),
            "reason": failure_reason,
            "ticket_id": ticket_id,
            "unavailable_session_ids": list(unavailable),
        },
    )
    return True


def reset_phone_layout_transition(
    controller,
    ticket_id: str,
    *,
    generation: int,
    projection_token_sha256: str,
    observed_at_us: int,
    reason: str,
) -> ModelPhoneResidencyLayout | None:
    """Return an obsolete queued transition to its proposal state."""

    generation = _integer(
        "phone layout reset generation", generation, 1
    )
    ticket_id = _text("phone layout reset ticket", ticket_id)
    token_sha256 = _sha256(
        "phone layout reset projection token",
        projection_token_sha256,
    )
    target = controller.target_phone_layout()
    if (
        target is None
        or target.generation != generation
        or target.state != "PREPARING"
        or target.transition_ticket_id != ticket_id
        or target.projection_token_sha256 != token_sha256
    ):
        controller._record_phone_layout_event(
            "STALE_PREPARATION_RESET_IGNORED",
            observed_at_us,
            {
                "generation": generation,
                "projection_token_sha256": token_sha256,
                "reason": _text(
                    "phone layout reset reason", reason
                ),
                "ticket_id": ticket_id,
            },
        )
        return None
    reset_reason = _text("phone layout reset reason", reason)
    controller._restore_phone_session_replacement(
        generation,
        observed_at_us=observed_at_us,
        reason=reset_reason,
    )
    proposed = replace(
        target,
        state="PROPOSED",
        transition_ticket_id=None,
        transition_ids=(),
        ready_at_us=None,
        projection_token_sha256=None,
    )
    controller._phone_layouts[proposed.generation] = proposed
    controller._restore_ready_phone_layout()
    controller._record_phone_layout_event(
        "TRANSITION_RESET",
        observed_at_us,
        {
            "generation": proposed.generation,
            "geometry_sha256": proposed.layout.geometry_sha256,
            "reason": reset_reason,
            "ticket_id": ticket_id,
        },
    )
    return proposed
