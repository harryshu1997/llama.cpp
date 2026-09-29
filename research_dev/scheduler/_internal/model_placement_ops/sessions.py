"""ModelPlacementController sessions operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

from ..phone_shards import PhoneFfnResidencyLayout, PhoneFfnSessionIdentity, PhoneFfnShardPlacement
from ..model_placement_contracts.common import (
    ModelPlacementControllerError,
    _text,
    _integer,
    _sha256,
    _identities,
)
from ..model_placement_contracts.layout import PhoneSessionResidencyState
from ..model_placement_contracts.requests import RequestHelperEnvelopeBinding


def _session_shard_matches(
    state: PhoneSessionResidencyState,
    shard: PhoneFfnShardPlacement,
) -> bool:
    return (
        state.session_id == shard.session_id
        and state.resident_artifact_sha256 == shard.artifact_sha256
        and state.shard_geometry_sha256
            == shard.resident_geometry_sha256
        and state.operator_plan_sha256
            == shard.operator_plan_sha256
    )


def _stamp_phone_layout(
    controller, layout: PhoneFfnResidencyLayout
) -> PhoneFfnResidencyLayout:
    generation_by_id = {}
    changed = []
    replacement_sources = []
    replacement_source_bytes = {}
    for shard in layout.shards:
        current = controller._phone_session_states.get(shard.session_id)
        if (
            current is not None
            and current.state == "READY"
            and controller._session_shard_matches(current, shard)
        ):
            generation_by_id[shard.session_id] = (
                current.session_generation
            )
            continue
        changed.append(shard.session_id)
        if (
            current is not None
            and current.resident_artifact_sha256 is not None
        ):
            replacement_sources.append(current.identity)
            replacement_source_bytes[shard.session_id] = (
                current.resident_bytes
            )
        generation_by_id[shard.session_id] = (
            controller._phone_session_generation_by_id.get(
                shard.session_id, 0
            ) + 1
        )
    changed.extend(
        session_id for session_id in controller._phone_session_states
        if session_id not in generation_by_id
        and controller._phone_session_states[session_id]
            .resident_artifact_sha256 is not None
    )
    for session_id, current in controller._phone_session_states.items():
        if (
            session_id not in generation_by_id
            and current.resident_artifact_sha256 is not None
            and current.identity not in replacement_sources
        ):
            replacement_sources.append(current.identity)
            replacement_source_bytes[session_id] = (
                current.resident_bytes
            )
    if tuple(sorted(set(changed))) != layout.changed_session_ids:
        raise ModelPlacementControllerError(
            "phone layout changed sessions differ from authority"
        )
    return replace(
        layout.with_session_generations(generation_by_id),
        replacement_source_identities=tuple(replacement_sources),
        replacement_source_resident_bytes_by_session=(
            replacement_source_bytes
        ),
    )


def _session_state_from_shard(
    shard: PhoneFfnShardPlacement,
    identity: PhoneFfnSessionIdentity,
    *,
    state: str,
    minimum_resident_until_us: int,
    replacement_cost_uj: int,
    active_helper_references: Sequence[str] = (),
) -> PhoneSessionResidencyState:
    return PhoneSessionResidencyState(
        session_id=shard.session_id,
        resident_artifact_sha256=shard.artifact_sha256,
        shard_geometry_sha256=shard.resident_geometry_sha256,
        operator_plan_sha256=shard.operator_plan_sha256,
        session_generation=identity.session_generation,
        state=state,
        active_helper_references=tuple(active_helper_references),
        minimum_resident_until_us=minimum_resident_until_us,
        replacement_cost_uj=replacement_cost_uj,
        resident_bytes=shard.resident_bytes,
        endpoint=shard.endpoint,
    )


def _sync_phone_session_references(controller) -> None:
    references: dict[str, set[str]] = {
        session_id: set() for session_id in controller._phone_session_states
    }
    for request_id, binding in controller._request_bindings.items():
        attachment = binding.helper_attachment
        if (
            request_id not in controller._acquired_request_ids
            or attachment is None
            or attachment.fraction_ppm <= 0
            or attachment.fallback_outcome is not None
        ):
            continue
        allowed = set(attachment.allowed_session_ids)
        for identity in attachment.phone_session_identities:
            if identity.session_id not in allowed:
                continue
            state = controller._phone_session_states.get(identity.session_id)
            if state is not None and state.matches(identity):
                references[identity.session_id].add(request_id)
    for session_id, state in tuple(controller._phone_session_states.items()):
        current = tuple(sorted(references[session_id]))
        if state.active_helper_references != current:
            controller._phone_session_states[session_id] = replace(
                state, active_helper_references=current
            )


def phone_session_state(
    controller, session_id: str
) -> PhoneSessionResidencyState | None:
    session_id = _text("phone session state identity", session_id)
    controller._sync_phone_session_references()
    return controller._phone_session_states.get(session_id)


def phone_session_states(
    controller,
) -> tuple[PhoneSessionResidencyState, ...]:
    controller._sync_phone_session_references()
    return tuple(
        controller._phone_session_states[session_id]
        for session_id in sorted(controller._phone_session_states)
    )


def register_empty_phone_sessions(
    controller,
    session_endpoints: Mapping[str, str],
    *,
    observed_at_us: int,
) -> tuple[PhoneSessionResidencyState, ...]:
    """Register discovered residency arenas before their first load."""

    _integer("phone session discovery time", observed_at_us)
    endpoints = {
        _text("phone session discovery identity", session_id): _text(
            "phone session discovery endpoint", endpoint
        )
        for session_id, endpoint in session_endpoints.items()
    }
    if not endpoints:
        raise ModelPlacementControllerError(
            "phone session discovery is empty"
        )
    for session_id, endpoint in sorted(endpoints.items()):
        current = controller._phone_session_states.get(session_id)
        if current is not None:
            if current.endpoint != endpoint:
                raise ModelPlacementControllerError(
                    "phone session discovery endpoint changed"
                )
            continue
        empty = PhoneSessionResidencyState(
            session_id=session_id,
            resident_artifact_sha256=None,
            shard_geometry_sha256=None,
            operator_plan_sha256=None,
            session_generation=0,
            state="EMPTY",
            active_helper_references=(),
            minimum_resident_until_us=0,
            replacement_cost_uj=0,
            resident_bytes=0,
            endpoint=endpoint,
        )
        controller._phone_session_states[session_id] = empty
        controller._phone_session_generation_by_id.setdefault(session_id, 0)
        controller._record_phone_layout_event(
            "SESSION_EMPTY",
            observed_at_us,
            {"session": empty.to_json()},
        )
    return controller.phone_session_states()


def _normalize_helper_envelope(
    controller, envelope: RequestHelperEnvelopeBinding
) -> RequestHelperEnvelopeBinding:
    layout = controller.phone_layout(envelope.phone_layout_generation)
    identities = tuple(
        layout.layout.session_identity(session_id)
        for session_id in envelope.phone_session_ids
    )
    if (
        envelope.phone_session_identities
        and envelope.phone_session_identities != identities
    ):
        raise ModelPlacementControllerError(
            "request helper session identity differs from layout"
        )
    return (
        envelope
        if envelope.phone_session_identities
        else replace(
            envelope, phone_session_identities=identities
        )
    )


def _session_identities_are_ready(
    controller, identities: Sequence[PhoneFfnSessionIdentity]
) -> bool:
    rows = tuple(identities)
    return bool(rows) and all(
        (state := controller._phone_session_states.get(identity.session_id))
            is not None
        and state.state == "READY"
        and state.matches(identity)
        for identity in rows
    )


def phone_layout_sessions_are_usable(
    controller,
    generation: int,
    geometry_sha256: str,
    session_ids: Sequence[str],
) -> bool:
    """Return whether one audited layout subset is still exactly ready."""

    generation = _integer("phone layout generation", generation, 1)
    geometry = _sha256("phone layout geometry", geometry_sha256)
    sessions = _identities("phone layout session", session_ids)
    try:
        layout = controller.phone_layout(generation)
        identities = tuple(
            layout.layout.session_identity(session_id)
            for session_id in sessions
        )
    except (KeyError, ModelPlacementControllerError):
        return False
    return bool(
        layout.layout.geometry_sha256 == geometry
        and controller._session_identities_are_ready(identities)
    )
